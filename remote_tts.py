
import requests
import torch
import numpy as np
import logging
import base64
import os
from typing import Optional, Generator, Tuple

# Configuration
STTS_SERVER_URL = "http://localhost:4000"  # Default URL for STTS Server

logger = logging.getLogger(__name__)

class RemoteChatterboxTTS:
    """
    A client-side wrapper that mimics ChatterboxTTS but delegates generation 
    to a remote STTS Server via HTTP streaming.
    """
    
    def __init__(self, device: str = "cpu", server_url: str = STTS_SERVER_URL):
        self.device = device
        self.server_url = server_url.rstrip("/")
        self.sr = 24000  # Default sample rate for Chatterbox
        self._reference_audio_b64: Optional[str] = None
        self._reference_exaggeration: Optional[float] = None
        self._predefined_voice_id: Optional[str] = None  # e.g. "Olivia.wav"
        self._reference_audio_filename: Optional[str] = None  # e.g. "uploaded.wav" (in STTS reference_audio dir)
        self.conds = None  # Compatibility flag: non-None after prepare_conditionals
        logger.info(f"Initialized RemoteChatterboxTTS pointing to {self.server_url}")

    @classmethod
    def from_pretrained(cls, device: str = "cpu", server_url: str = STTS_SERVER_URL) -> 'RemoteChatterboxTTS':
        return cls(device=device, server_url=server_url)

    def set_predefined_voice(self, voice_id: str):
        """Set a predefined voice from the STTS server's voices directory.
        
        Args:
            voice_id: Filename of the predefined voice (e.g., "Olivia.wav")
        """
        self._predefined_voice_id = voice_id
        self._reference_audio_filename = None
        self._reference_audio_b64 = None
        self.conds = True
        logger.info(f"Set predefined voice: {voice_id}")

    def set_clone_voice(self, reference_filename: str):
        """Set a clone voice from the STTS server's reference_audio directory.
        
        Args:
            reference_filename: Filename of the uploaded reference audio (e.g., "user_voice.wav")
        """
        self._reference_audio_filename = reference_filename
        self._predefined_voice_id = None
        self._reference_audio_b64 = None
        self.conds = True
        logger.info(f"Set clone voice: {reference_filename}")

    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5):
        """
        Store reference audio for subsequent generate() calls.
        The audio is read and base64-encoded so it can be sent to the remote server.
        Also uploads the file to the STTS server's reference_audio directory for streaming.
        """
        if not os.path.isfile(wav_fpath):
            raise FileNotFoundError(f"Reference audio file not found: {wav_fpath}")
        with open(wav_fpath, "rb") as f:
            audio_bytes = f.read()
        self._reference_audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        self._reference_exaggeration = exaggeration
        self._predefined_voice_id = None  # Clear predefined voice — reference audio takes priority
        self.conds = True  # Mark conditionals as prepared
        
        # Upload reference audio to STTS server for streaming endpoint use
        ref_filename = os.path.basename(wav_fpath)
        try:
            upload_url = f"{self.server_url}/upload_reference"
            with open(wav_fpath, "rb") as f:
                resp = requests.post(upload_url, files={"files": (ref_filename, f, "audio/wav")})
            if resp.status_code == 200:
                self._reference_audio_filename = ref_filename
                logger.info(f"Uploaded reference audio to STTS server: {ref_filename}")
            else:
                logger.warning(f"Failed to upload reference audio to {upload_url}: {resp.status_code} {resp.text}")
                self._reference_audio_filename = None
        except Exception as e:
            logger.warning(f"Could not upload reference audio to STTS server: {e}")
            self._reference_audio_filename = None
        
        logger.info(f"Prepared conditionals from {wav_fpath} ({len(audio_bytes)} bytes)")

    def generate(self, text: str, **kwargs) -> torch.Tensor:
        """
        Synchronous generation (non-streaming).
        Calls the PCM TTS endpoint.
        """
        url = f"{self.server_url}/pcm-tts/generate"
        
        payload = {
            "text": text,
            "sample_rate": self.sr,
            "temperature": kwargs.get("temperature"),
            "exaggeration": kwargs.get("exaggeration"),
            "cfg_weight": kwargs.get("cfg_weight"),
            "seed": kwargs.get("seed"),
        }

        # Include reference audio if prepare_conditionals was called
        if self._reference_audio_b64:
            payload["reference_audio_base64"] = self._reference_audio_b64
        
        try:
            response = requests.post(url, json=payload, stream=False)
            response.raise_for_status()
            
            # PCM 16-bit little-endian
            pcm_bytes = response.content
            audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            return torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)
            
        except Exception as e:
            logger.error(f"Remote TTS generation failed: {e}")
            raise

    def generate_stream(
        self,
        text: str,
        chunk_size: int = 25,
        **kwargs
    ) -> Generator[Tuple[torch.Tensor, dict], None, None]:
        """
        Streaming generation.
        Calls the Streaming TTS endpoint.
        Yields (audio_chunk_tensor, metrics_dict).
        """
        # Handle audio_prompt_path kwarg: if caller passes a file path,
        # prepare conditionals (upload to STTS) before streaming
        audio_prompt_path = kwargs.pop("audio_prompt_path", None)
        if audio_prompt_path and os.path.isfile(audio_prompt_path):
            self.prepare_conditionals(audio_prompt_path, exaggeration=kwargs.get("exaggeration", 0.5))
        
        url = f"{self.server_url}/tts/stream"
        
        payload = {
            "text": text,
            "chunk_size": chunk_size,
            "temperature": kwargs.get("temperature"),
            "exaggeration": kwargs.get("exaggeration"),
            "cfg_weight": kwargs.get("cfg_weight"),
            "seed": kwargs.get("seed"),
        }
        
        # Include voice identification for the STTS server
        if self._predefined_voice_id:
            payload["predefined_voice_id"] = self._predefined_voice_id
        elif self._reference_audio_filename:
            payload["reference_audio_filename"] = self._reference_audio_filename
        
        voice_info = payload.get("predefined_voice_id") or payload.get("reference_audio_filename") or "DEFAULT (no voice set)"
        logger.info(f"TTS stream request: voice={voice_info}, text='{text[:40]}...'")
        
        try:
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                
                # Check header for encoding
                encoding = response.headers.get("X-Encoding", "float32")
                
                # Assuming float32 stream
                dtype = np.float32
                item_size = 4  # bytes
                
                chunk_buffer = bytearray()
                
                # We need to yield chunks as they arrive.
                # However, the server yields chunks of variable size.
                # We just read from the stream.
                
                # The response.iter_content(chunk_size=None) yields whatever the server sends.
                for raw_chunk in response.iter_content(chunk_size=None):
                    if not raw_chunk:
                        continue
                        
                    # Convert raw bytes to numpy float32
                    # Note: We might receive partial float bytes if chunking is weird, 
                    # but typically iter_content with proper flushing SHOULD align.
                    # To be safe, we can buffer bytes.
                    
                    chunk_buffer.extend(raw_chunk)
                    
                    # Process complete floats
                    num_floats = len(chunk_buffer) // item_size
                    if num_floats > 0:
                        bytes_to_process = chunk_buffer[:num_floats * item_size]
                        del chunk_buffer[:num_floats * item_size]
                        
                        audio_np = np.frombuffer(bytes_to_process, dtype=dtype)
                        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0) # (1, T)
                        
                        # Yield in expected format (tensor, metrics)
                        # Metrics are dummy here as we don't get them from server yet
                        yield audio_tensor, {}
                        
        except Exception as e:
            logger.error(f"Remote TTS streaming failed: {e}")
            # Don't raise, just stop yielding? Or raise?
            # chatterbox generator usually doesn't raise, but prints error.
            print(f"RemoteChatterboxTTS Error: {e}")

