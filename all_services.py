 
import os
import sys
import traceback
import torch
import numpy as np
import argparse
from PIL import Image
import cv2
import tempfile
import time
import random
from contextlib import asynccontextmanager
import fractions
import collections
import math
import subprocess
import shutil
import re
import glob
import struct
from pathlib import Path
from dataclasses import dataclass, field
import asyncio
import io
import wave
import base64
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks, WebSocket
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import hashlib

# Try importing torchaudio for GPU-accelerated resampling
try:
    import torchaudio
    import torchaudio.functional as AF
    _HAS_TORCHAUDIO = True
except ImportError:
    print("Warning: torchaudio not installed. Falling back to scipy for resampling.")
    _HAS_TORCHAUDIO = False

# Try importing librosa for audio resampling
try:
    import librosa
except ImportError:
    print("Warning: librosa not installed. Audio resampling may not work. Install with: pip install librosa")
    librosa = None

# Load environment variables from .env file
load_dotenv()
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack, RTCConfiguration, RTCIceServer, RTCIceCandidate
from aiortc.contrib.media import MediaRelay
from av import VideoFrame, AudioFrame
from openai import AsyncOpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add Chatterbox Streaming path (at priority 0)
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../chatterbox-streaming/src")))
sys.path.append(current_dir)

# Import IMTalker components
from imtalker_streamer import IMTalkerStreamer, IMTalkerConfig

# Import Chatterbox
# Try to import from installed package or local if needed
try:
    import chatterbox
    print(f"Chatterbox imported from: {chatterbox.__file__}")
    from chatterbox.tts import ChatterboxTTS
except ImportError as e:
    print(f"Could not import chatterbox.tts: {e}")
    # print sys.path for debugging
    print(f"sys.path: {sys.path}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- FastAPI Implementation ---

class GenerateRequest(BaseModel):
    text: str
    avatar: Optional[str] = "assets/source_1.png"
    output_path: Optional[str] = "output_stream.mp4"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize models during startup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Lifespan: Initializing models on {device}...")
    
    # 0. Initialize OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ WARNING: OPENAI_API_KEY not set! Conversational mode will be disabled.")
        app.state.openai_client = None
    else:
        app.state.openai_client = AsyncOpenAI(api_key=api_key)

    # 1. Initialize Chatterbox TTS
    print("Lifespan: Loading Chatterbox TTS...")
    app.state.tts_model = ChatterboxTTS.from_pretrained(device=device)
    
    # 2. Initialize IMTalker
    print("Lifespan: Loading IMTalker...")
    imtalker_cfg = IMTalkerConfig()
    imtalker_cfg.device = device
    
    # Enable optimizations
    imtalker_cfg.use_batching = True
    imtalker_cfg.batch_size = 15  # Increased for faster batched rendering (safe on 48GB GPU)
    imtalker_cfg.use_fp16 = False  # Disabled: causes NaN/Inf in frames with this model
    imtalker_cfg.use_compile = True  # Enable torch.compile for optimized kernels
    imtalker_cfg.low_latency_mode = True  # NFE 10→5 for 2x faster generation (real-time)
    
    app.state.imtalker_streamer = IMTalkerStreamer(imtalker_cfg, device=device)
    # Initialize InferenceAgent sharing the same models
    app.state.inference_agent = InferenceAgent(imtalker_cfg, models=(app.state.imtalker_streamer.renderer, app.state.imtalker_streamer.generator))
    
    # Session management
    app.state.webrtc_sessions = {}
    app.state.chunked_sessions = {}
    app.state.hls_sessions = {}
    app.state.realtime_sessions = {}
    
    # Single-session guard: only ONE WebRTC session at a time
    app.state.active_webrtc_session = None  # session_id or None
    app.state.active_session_meta = {}      # {session_id, status, conv_manager, video_track, audio_track, ...}
    app.state._session_lock = asyncio.Lock()
    
    # Idle animation cache (keyed by avatar image hash)
    app.state.idle_cache = {}
    
    yield
    # Clean up models if necessary
    print("Lifespan: Shutting down...")
    if hasattr(app.state, 'openai_client'):
        await app.state.openai_client.close()
    del app.state.tts_model
    del app.state.imtalker_streamer
    del app.state.inference_agent

app = FastAPI(lifespan=lifespan)

# HLS output directories
HLS_OUTPUT_DIR = Path("./hls_output")
HLS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HLS_SESSION_DIR = HLS_OUTPUT_DIR # Consistent with ProgressiveHLSGenerator

# Mount HLS directory for segment delivery
app.mount("/hls", StaticFiles(directory=HLS_OUTPUT_DIR), name="hls")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    index_path = os.path.join(current_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "IMTalker Text-to-Video API is running. index.html not found."}

@app.get("/health")
async def health():
    """Detailed health check."""
    stats = gpu_monitor.get_stats()
    return {
        "status": "healthy" if hasattr(app.state, 'imtalker_streamer') else "unhealthy",
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": hasattr(app.state, 'imtalker_streamer'),
        "gpu_memory": stats,
    }

@app.get("/gpu_stats")
async def gpu_stats():
    """GPU memory monitoring endpoint for production dashboards.
    
    Returns current/peak memory usage, OOM count, and fallback status.
    """
    stats = gpu_monitor.get_stats()
    stats["level"] = gpu_monitor.check_and_log("api-check")
    return stats

def numpy_to_wav_base64(audio_np, sampling_rate=16000):
    if isinstance(audio_np, torch.Tensor):
        audio_np = audio_np.cpu().numpy()
    audio_np = audio_np.flatten()
    if np.max(np.abs(audio_np)) > 1.0:
        audio_np = audio_np / np.max(np.abs(audio_np))
    audio_int16 = (audio_np * 32767).astype(np.int16)
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(audio_int16.tobytes())
    return base64.b64encode(byte_io.getvalue()).decode('utf-8')

def save_base64_to_temp(base64_str, prefix="temp_", suffix=".tmp"):
    """Decodes base64 string to a temporary file and returns path."""
    try:
        # Strip data URI scheme if present
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
            
        decoded_data = base64.b64decode(base64_str)
        
        # Create temp file
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        with os.fdopen(fd, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"Saved Base64 data to temp file: {path} ({len(decoded_data)} bytes)")
        return path
    except Exception as e:
        logger.error(f"Error decoding base64: {e}")
        return None

async def download_url_to_temp(url, prefix="download_", suffix=".tmp"):
    """Downloads content from a URL to a temporary file."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            
            # Create temp file
            fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
            with os.fdopen(fd, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded URL {url} to temp file: {path} ({len(response.content)} bytes)")
            return path
    except Exception as e:
        logger.error(f"Error downloading URL {url}: {e}")
        return None

def split_text_semantically(text, min_chunk_length=30, max_chunk_length=300):
    """Split text into semantic chunks (sentences/clauses) for more natural speech.
    
    Args:
        text: Input text to split
        min_chunk_length: Minimum characters per chunk
        max_chunk_length: Maximum characters per chunk (larger = fewer TTS calls = faster)
    
    Returns:
        List of text chunks
    """
    import re
    
    # Split on sentence boundaries
    # Matches: . ! ? followed by space or end, but not abbreviations like "Mr."
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed max length, finalize current chunk
        if current_chunk and len(current_chunk) + len(sentence) > max_chunk_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        
        # If current chunk is long enough, finalize it
        if len(current_chunk) >= min_chunk_length and sentence[-1] in '.!?':
            chunks.append(current_chunk.strip())
            current_chunk = ""
    
    # Add any remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Fallback: if no chunks were created (e.g., no sentence boundaries), just return the text
    if not chunks and text:
        chunks = [text.strip()]
    
    return chunks

def parse_chunk_labels(text):
    """Parse text that contains chunk labels from OpenAI.
    
    Expected format:
    chunk1: First sentence.
    chunk2: Second sentence.
    
    Returns:
        List of text chunks (without labels)
    """
    import re
    
    # Pattern to match chunk labels: chunk1:, chunk2:, etc.
    chunk_pattern = r'chunk\d+:\s*'
    
    # Split by chunk labels - this removes the "chunk1:", "chunk2:" parts
    chunks = re.split(chunk_pattern, text, flags=re.IGNORECASE)
    
    # Remove empty strings and strip whitespace (including newlines)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # If chunk labels were found, merge them into large groups for efficient TTS
    if len(chunks) > 1:
        print(f"CM: Parsed {len(chunks)} labeled chunks, merging into large groups...")
        chunks = merge_into_large_groups(chunks)
    
    # If no chunk labels found, fall back to semantic splitting (already uses large chunks)
    if len(chunks) <= 1:
        if len(chunks) == 1 and len(chunks[0]) > 300:
            # Large text without labels — split semantically
            print("CM: Large text, using semantic splitting")
            return split_text_semantically(text)
        elif not chunks:
            print("CM: No chunk labels found, using semantic splitting")
            return split_text_semantically(text)
    
    return chunks


def merge_into_large_groups(chunks, target_chars=300):
    """Merge small text chunks into larger groups for fewer, faster TTS calls.
    
    Instead of 8 × 50-char TTS calls (each with ~200ms overhead),
    produces 2 × 200-char calls — much faster total.
    
    Args:
        chunks: List of small text chunks
        target_chars: Target characters per merged group (~10s of speech)
    
    Returns:
        List of merged text chunks
    """
    if not chunks:
        return chunks
    
    merged = []
    current = ""
    
    for chunk in chunks:
        if current and len(current) + len(chunk) + 1 > target_chars:
            merged.append(current.strip())
            current = chunk
        else:
            current = (current + " " + chunk).strip() if current else chunk
    
    if current:
        merged.append(current.strip())
    
    print(f"CM: Merged {len(chunks)} small chunks → {len(merged)} large groups")
    for i, g in enumerate(merged, 1):
        print(f"  Group {i} ({len(g)} chars): '{g[:60]}...'")
    
    return merged

# ============================================================================
# REAL-TIME CHUNKED AUDIO STREAMING
# ============================================================================

class AudioChunkBuffer:
    """
    Buffer for accumulating audio chunks for real-time inference.
    Supports sliding window processing with overlap.
    """
    
    def __init__(self, 
                 session_id: str,
                 target_sr: int = 16000,
                 chunk_duration: float = 1.0,
                 overlap_ratio: float = 0.2):
        """
        Args:
            session_id: Unique session ID
            target_sr: Target sampling rate (16kHz for wav2vec2)
            chunk_duration: Duration of audio chunk to accumulate (seconds)
            overlap_ratio: Overlap between chunks (0.2 = 20% overlap)
        """
        self.session_id = session_id
        self.target_sr = target_sr
        self.chunk_duration = chunk_duration
        self.overlap_ratio = overlap_ratio
        
        # Buffer size: samples for target duration
        self.chunk_samples = int(chunk_duration * target_sr)
        self.overlap_samples = int(self.chunk_samples * overlap_ratio)
        self.stride_samples = self.chunk_samples - self.overlap_samples
        
        # Audio accumulator (numpy float32, -1.0 to 1.0)
        self.buffer = np.array([], dtype=np.float32)
        self.total_samples_received = 0
        self.lock = threading.Lock()
        
        print(f"[AudioBuffer {session_id}] Initialized:")
        print(f"  Target SR: {target_sr}Hz")
        print(f"  Chunk duration: {chunk_duration}s ({self.chunk_samples} samples)")
        print(f"  Overlap: {overlap_ratio*100:.0f}% ({self.overlap_samples} samples)")
        print(f"  Stride: {self.stride_samples} samples")
    
    def add_chunk(self, audio_chunk: np.ndarray, source_sr: int = 24000) -> bool:
        """
        Add an audio chunk and resample if needed.
        
        Args:
            audio_chunk: Audio samples (1D or 2D array)
            source_sr: Source sampling rate of the chunk
            
        Returns:
            True if buffer now has enough samples for processing
        """
        with self.lock:
            try:
                # Ensure 1D
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.flatten()
                
                # Resample if needed
                if source_sr != self.target_sr:
                    if librosa:
                        print(f"[AudioBuffer {self.session_id}] Resampling {len(audio_chunk)} samples from {source_sr}Hz → {self.target_sr}Hz")
                        audio_chunk = librosa.resample(
                            audio_chunk, 
                            orig_sr=source_sr, 
                            target_sr=self.target_sr
                        )
                    else:
                        print(f"[AudioBuffer {self.session_id}] Warning: librosa not available for resampling!")
                
                # Append to buffer
                self.buffer = np.concatenate([self.buffer, audio_chunk.astype(np.float32)])
                self.total_samples_received += len(audio_chunk)
                
                has_enough = len(self.buffer) >= self.chunk_samples
                
                print(f"[AudioBuffer {self.session_id}] Chunk added: {len(audio_chunk)} samples → "
                      f"buffer now {len(self.buffer)} samples (need {self.chunk_samples}) "
                      f"[total_received: {self.total_samples_received}]")
                
                return has_enough
                
            except Exception as e:
                print(f"[AudioBuffer {self.session_id}] Error adding chunk: {e}")
                raise
    
    def get_chunk(self) -> Optional[np.ndarray]:
        """
        Get next processable audio chunk from buffer (with stride/overlap).
        
        Returns:
            Audio tensor for inference, or None if not enough data
        """
        with self.lock:
            if len(self.buffer) < self.chunk_samples:
                return None
            
            # Extract chunk
            chunk = self.buffer[:self.chunk_samples].copy()
            
            # Slide buffer by stride (move window forward with overlap)
            self.buffer = self.buffer[self.stride_samples:]
            
            print(f"[AudioBuffer {self.session_id}] Extracted chunk: "
                  f"{self.chunk_samples} samples, buffer now {len(self.buffer)} samples remaining")
            
            return chunk
    
    def has_pending_audio(self) -> bool:
        """Check if there's unprocessed audio in the buffer"""
        with self.lock:
            return len(self.buffer) >= self.chunk_samples
    
    def finalize(self) -> Optional[np.ndarray]:
        """
        Get any remaining audio (at end of stream).
        Pads with zeros if needed to match chunk_samples.
        """
        with self.lock:
            if len(self.buffer) == 0:
                return None
            
            # Pad to match chunk size
            if len(self.buffer) < self.chunk_samples:
                padding = np.zeros(self.chunk_samples - len(self.buffer), dtype=np.float32)
                chunk = np.concatenate([self.buffer, padding])
                print(f"[AudioBuffer {self.session_id}] Final chunk: {len(self.buffer)} samples + "
                      f"{len(padding)} zero-padding = {len(chunk)} samples")
            else:
                chunk = self.buffer[:self.chunk_samples].copy()
                print(f"[AudioBuffer {self.session_id}] Final chunk: {len(chunk)} samples")
            
            # Clear buffer
            self.buffer = np.array([], dtype=np.float32)
            return chunk
    
    def reset(self):
        """Clear buffer for new session"""
        with self.lock:
            self.buffer = np.array([], dtype=np.float32)
            self.total_samples_received = 0
            print(f"[AudioBuffer {self.session_id}] Buffer reset")


@dataclass
class ChunkedStreamSession:
    """Represents an active chunked audio-to-video streaming session"""
    session_id: str
    image_pil: Image.Image
    audio_buffer: AudioChunkBuffer
    hls_generator: Optional["ProgressiveHLSGenerator"]
    parameters: dict  # crop, seed, nfe, cfg_scale, etc.
    status: str = "waiting_for_audio"  # waiting_for_audio, processing, complete, error
    error_message: Optional[str] = None
    total_frames_generated: int = 0
    created_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


# Storage for chunked streaming sessions
chunked_sessions: Dict[str, ChunkedStreamSession] = {}

# Storage for real-time streaming sessions
realtime_sessions: Dict[str, 'RealtimeSession'] = {}


@dataclass
class HLSSession:
    """Represents an active HLS streaming session."""
    session_id: str
    temp_dir: str
    target_duration: int
    segments: List[dict] = field(default_factory=list)
    is_complete: bool = False
    total_frames: int = 0
    created_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

hls_sessions: Dict[str, HLSSession] = {}

# Create directories for HLS output
HLS_OUTPUT_DIR = Path("hls_output")
HLS_OUTPUT_DIR.mkdir(exist_ok=True)

# Store generation status
generation_status = {}


def cleanup_hls_session(session_id: str, delay_minutes: int = 50):
    """
    Schedule cleanup of HLS output directory after a delay
    """
    def _cleanup():
        delay_seconds = delay_minutes * 60
        print(f"[CLEANUP] Scheduled cleanup for session {session_id} in {delay_minutes} minutes")
        time.sleep(delay_seconds)
        
        session_dir = HLS_OUTPUT_DIR / session_id
        if session_dir.exists():
            try:
                shutil.rmtree(session_dir)
                print(f"[CLEANUP] ✓ Deleted HLS output for session {session_id}")
                
                # Also remove from generation_status
                if session_id in generation_status:
                    del generation_status[session_id]
                    print(f"[CLEANUP] ✓ Removed session {session_id} from status tracking")
            except Exception as e:
                print(f"[CLEANUP] ERROR: Failed to delete session {session_id}: {e}")
        else:
            print(f"[CLEANUP] Session {session_id} directory already deleted")
    
    # Start cleanup in background thread
    cleanup_thread = threading.Thread(target=_cleanup, daemon=True)
    cleanup_thread.start()


class ProgressiveHLSGenerator:
    """Generate HLS segments progressively as frames are created"""

    def __init__(
        self,
        session_id: str,
        fps: float = 25.0,
        segment_duration: int =2,
        audio_path: Optional[str] = None,
    ):
        self.session_id = session_id
        self.fps = fps
        self.segment_duration = segment_duration
        self.audio_path = audio_path
        self.frames_per_segment = int(fps * segment_duration)
        self.output_dir = HLS_OUTPUT_DIR / session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_segment = 0
        self.frame_buffer = []
        self.segment_files = []
        self.is_complete = False
        self.total_frames = 0

        print(f"[HLS {self.session_id}] Initialized - FPS: {fps}, Segment Duration: {segment_duration}s, Frames per segment: {self.frames_per_segment}")
        print(f"[HLS {self.session_id}] Output directory: {self.output_dir}")
        print(f"[HLS {self.session_id}] Audio path: {self.audio_path}")

        # Check if session already exists and load existing segments
        self._load_existing_session()
        
        # Update playlist with existing segments (if any) to ensure continuity
        if self.segment_files:
            print(f"[HLS {self.session_id}] Updating playlist with {len(self.segment_files)} existing segments")
            self._update_playlist()
        else:
            # Create initial empty playlist only for new sessions
            self._create_initial_playlist()

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the buffer and create segment if buffer is full"""
        self.frame_buffer.append(frame)
        self.total_frames += 1

        if len(self.frame_buffer) >= self.frames_per_segment:
            self._write_segment()
            self._update_playlist()

    def _load_existing_session(self):
        """Load existing segments if session already exists"""
        # Small delay + sync to ensure file system has caught up with recent writes from previous chunk
        time.sleep(0.05)
        try:
            os.sync()
            print(f"[HLS {self.session_id}] Filesystem synced")
        except:
            print(f"[HLS {self.session_id}] os.sync() not available")
        
        # Find existing segment files
        search_pattern = str(self.output_dir / "segment*.ts")
        existing_segments = sorted(glob.glob(search_pattern))
        
        if existing_segments:
            for seg_path in existing_segments:
                print(f"[HLS {self.session_id}]   - {seg_path}")
            
            # Clear any existing entries (safety)
            self.segment_files = []
            
            # Extract segment numbers and file names
            max_segment_num = -1
            for seg_path in existing_segments:
                seg_name = os.path.basename(seg_path)
                self.segment_files.append(seg_name)
                
                # Extract segment number from filename (e.g., "segment003.ts" -> 3)
                match = re.search(r'segment(\d+)\.ts', seg_name)
                if match:
                    seg_num = int(match.group(1))
                    max_segment_num = max(max_segment_num, seg_num)
            
            # Set current_segment to continue from the last segment
            if max_segment_num >= 0:
                self.current_segment = max_segment_num + 1
                print(f"[HLS {self.session_id}] Resuming from segment {self.current_segment} (found {len(existing_segments)} existing segments)")
            else:
                print(f"[HLS {self.session_id}] No valid segments found, starting from 0")

    def _create_initial_playlist(self):
        """Create an initial empty playlist for HLS player to start loading"""
        playlist_path = self.output_dir / "playlist.m3u8"
        target_duration = self.segment_duration + 1

        with open(playlist_path, "w") as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n")
            f.write(f"#EXT-X-TARGETDURATION:{target_duration}\n")
            f.write("#EXT-X-PLAYLIST-TYPE:EVENT\n")
            f.write("#EXT-X-MEDIA-SEQUENCE:0\n")

    def _write_segment(self):
        """Write buffered frames as an HLS segment"""
        if not self.frame_buffer:
            print(f"[HLS {self.session_id}] No frames in buffer, skipping segment write")
            return

        segment_file = self.output_dir / f"segment{self.current_segment:03d}.ts"
        segment_name = f"segment{self.current_segment:03d}.ts"
        temp_video = self.output_dir / f"temp_segment{self.current_segment:03d}.mp4"

        # CRITICAL FIX: Check if segment already exists
        if segment_file.exists():
            all_segments = sorted(glob.glob(str(self.output_dir / "segment*.ts")))
            max_seg = -1
            for seg_path in all_segments:
                match = re.search(r'segment(\d+)\.ts', os.path.basename(seg_path))
                if match:
                    max_seg = max(max_seg, int(match.group(1)))
            self.current_segment = max_seg + 1
            segment_file = self.output_dir / f"segment{self.current_segment:03d}.ts"
            segment_name = f"segment{self.current_segment:03d}.ts"
            temp_video = self.output_dir / f"temp_segment{self.current_segment:03d}.mp4"
            self.segment_files = [os.path.basename(s) for s in all_segments]
        try:
            # Write frames to temporary MP4
            height, width = self.frame_buffer[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temp_video), fourcc, self.fps, (width, height))

            for frame in self.frame_buffer:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

            if not temp_video.exists() or temp_video.stat().st_size == 0:
                print(f"[HLS {self.session_id}] ERROR: Temp video file not created or empty")
                return

            print(f"[HLS {self.session_id}] Temp video created: {temp_video.stat().st_size} bytes")

            start_time = self.current_segment * self.segment_duration
            duration = len(self.frame_buffer) / self.fps

            cmd = ["ffmpeg", "-y", "-i", str(temp_video)]

            if self.audio_path and os.path.exists(self.audio_path):
                print(f"[HLS {self.session_id}] Adding audio from {start_time:.3f}s for {duration:.3f}s")
                cmd.extend([
                    "-ss", f"{start_time:.3f}",
                    "-t", f"{duration:.3f}",
                    "-i", self.audio_path,
                    "-map", "0:v",
                    "-map", "1:a",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-ac", "2",
                ])

            cmd.extend(["-output_ts_offset", f"{start_time:.3f}"])
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-f", "mpegts",
                "-y",
                str(segment_file),
            ])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr}")

            if not segment_file.exists():
                raise Exception("Segment file not created")
            
            print(f"[HLS {self.session_id}] ✓ Segment created: {segment_file.name}")
            temp_video.unlink()
            self.segment_files.append(segment_file.name)
            self.current_segment += 1
            self.frame_buffer = []

        except Exception as e:
            print(f"[HLS {self.session_id}] Exception in _write_segment: {e}")
            if temp_video.exists():
                temp_video.unlink()

    def _update_playlist(self):
        """Update the HLS playlist with current segments"""
        playlist_path = self.output_dir / "playlist.m3u8"
        temp_playlist_path = self.output_dir / "playlist.m3u8.tmp"

        try:
            all_segments = sorted(glob.glob(str(self.output_dir / "segment*.ts")))
            all_segment_names = [os.path.basename(seg) for seg in all_segments]
            self.segment_files = all_segment_names
            
            target_duration = self.segment_duration + 1

            with open(temp_playlist_path, "w") as f:
                f.write("#EXTM3U\n")
                f.write("#EXT-X-VERSION:3\n")
                f.write(f"#EXT-X-TARGETDURATION:{target_duration}\n")
                f.write("#EXT-X-PLAYLIST-TYPE:EVENT\n")
                f.write("#EXT-X-MEDIA-SEQUENCE:0\n")

                for idx, segment in enumerate(all_segment_names):
                    f.write(f"#EXTINF:{self.segment_duration:.3f},\n")
                    f.write(f"{segment}\n")
                
                if self.is_complete:
                    f.write("#EXT-X-ENDLIST\n")
                
                f.flush()
                os.fsync(f.fileno())
            
            temp_playlist_path.replace(playlist_path)
            try:
                os.chmod(playlist_path, 0o644)
            except:
                pass
            
        except Exception as e:
            if temp_playlist_path.exists():
                temp_playlist_path.unlink()
            raise

    def finalize(self, audio_path: Optional[str] = None, mark_complete: bool = False):
        if self.frame_buffer:
            self._write_segment()

        if mark_complete:
            self.is_complete = True
            cleanup_hls_session(self.session_id, delay_minutes=50)
        
        self._update_playlist()


# ============================================================================
# INFERENCE COMPONENTS (Adapted from fastapi_app.py and imtalker_streamer.py)
# ============================================================================

class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.crop_scale = getattr(opt, 'crop_scale', 0.8)
        print(f"Loading Face Alignment...")
        try:
            import face_alignment
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False
            )
        except Exception as e:
            print(f"Warning: face_alignment not found or failed to load: {e}")
            self.fa = None

        print("Loading Wav2Vec2 Preprocessor...")
        local_path = getattr(opt, 'wav2vec_model_path', "./checkpoints/wav2vec2-base-960h")
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
            from transformers import Wav2Vec2FeatureExtractor
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
                local_path, local_files_only=True
            )
        else:
            from transformers import Wav2Vec2FeatureExtractor
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
        
        import torchvision.transforms as transforms
        self.transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )

    def process_img(self, img: Image.Image) -> Image.Image:
        img_arr = np.array(img)
        if img_arr.ndim == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        elif img_arr.shape[2] == 4:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
        h, w = img_arr.shape[:2]
        
        if self.fa is None:
            return img.resize((512, 512))
            
        try:
            bboxes = self.fa.face_detector.detect_from_image(img_arr)
            if bboxes is None or len(bboxes) == 0:
                bboxes = self.fa.face_detector.detect_from_image(img_arr)
        except Exception as e:
            print(f"Face detection failed: {e}")
            bboxes = None
            
        valid_bboxes = []
        if bboxes is not None:
            valid_bboxes = [
                (int(x1), int(y1), int(x2), int(y2), score)
                for (x1, y1, x2, y2, score) in bboxes
                if score > 0.5
            ]
            
        if not valid_bboxes:
            print("Warning: No face detected. Using center crop.")
            cx, cy = w // 2, h // 2
            half = min(w, h) // 2
            x1_new, x2_new = cx - half, cx + half
            y1_new, y2_new = cy - half, cy + half
        else:
            x1, y1, x2, y2, _ = valid_bboxes[0]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w_face = x2 - x1
            h_face = y2 - y1
            half_side = int(max(w_face, h_face) * self.crop_scale)
            x1_new = cx - half_side
            y1_new = cy - half_side
            x2_new = cx + half_side
            y2_new = cy + half_side
            
            # Boundary checks
            x1_new = max(0, x1_new)
            y1_new = max(0, y1_new)
            x2_new = min(w, x2_new)
            y2_new = min(h, y2_new)
            
            curr_w = x2_new - x1_new
            curr_h = y2_new - y1_new
            min_side = min(curr_w, curr_h)
            x2_new = x1_new + min_side
            y2_new = y1_new + min_side
            
        crop_img = img_arr[int(y1_new) : int(y2_new), int(x1_new) : int(x2_new)]
        crop_pil = Image.fromarray(crop_img)
        return crop_pil.resize((512, 512))

    def process_audio(self, path: str) -> torch.Tensor:
        if librosa is None:
            raise ImportError("librosa is required for process_audio")
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        return self.wav2vec_preprocessor(
            speech_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_values[0]

    def crop_video_stable(self, from_mp4_file_path, to_mp4_file_path, expanded_ratio=0.6, skip_per_frame=15):
        if os.path.exists(to_mp4_file_path):
            os.remove(to_mp4_file_path)
            
        if self.fa is None:
            shutil.copy(from_mp4_file_path, to_mp4_file_path)
            return

        video = cv2.VideoCapture(from_mp4_file_path)
        index = 0
        bboxes_lists = []
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while video.isOpened():
            success = video.grab()
            if not success: break
            if index % skip_per_frame == 0:
                success, frame = video.retrieve()
                if not success: break
                h, w = frame.shape[:2]
                mult = 360.0 / h
                resized_frame = cv2.resize(frame, dsize=(0, 0), fx=mult, fy=mult, interpolation=cv2.INTER_AREA if mult < 1 else cv2.INTER_CUBIC)
                try:
                    detected_bboxes = self.fa.face_detector.detect_from_image(resized_frame)
                    if detected_bboxes is not None:
                        for d_box in detected_bboxes:
                            bx1, by1, bx2, by2, score = d_box
                            if score > 0.5:
                                bboxes_lists.append([int(bx1 / mult), int(by1 / mult), int(bx2 / mult), int(by2 / mult), score])
                                break # Take first face
                except:
                    pass
            index += 1
        video.release()
        
        if not bboxes_lists:
            shutil.copy(from_mp4_file_path, to_mp4_file_path)
            return
            
        x_centers = [(b[0] + b[2]) / 2 for b in bboxes_lists]
        y_centers = [(b[1] + b[3]) / 2 for b in bboxes_lists]
        ws = [b[2] - b[0] for b in bboxes_lists]
        hs = [b[3] - b[1] for b in bboxes_lists]
        
        x_center = np.median(x_centers)
        y_center = np.median(y_centers)
        m_w = np.median(ws)
        m_h = np.median(hs)
        
        exp_w = int(m_w * (1 + expanded_ratio))
        exp_h = int(m_h * (1 + expanded_ratio))
        side = min(max(exp_w, exp_h), width, height)
        
        x1 = max(0, int(x_center - side / 2))
        y1 = max(0, int(y_center - side / 2))
        if x1 + side > width: x1 = width - side
        if y1 + side > height: y1 = height - side
        
        target_size = 512
        cmd = f'ffmpeg -i "{from_mp4_file_path}" -filter:v "crop={side}:{side}:{x1}:{y1},scale={target_size}:{target_size}:flags=lanczos" -c:v libx264 -crf 18 -preset slow -c:a aac -b:a 128k "{to_mp4_file_path}" -y -loglevel error'
        os.system(cmd)


# ============================================================================
# GPU MEMORY MONITOR WITH AUTO-FALLBACK
# ============================================================================

class GPUMemoryMonitor:
    """Monitors GPU memory and provides auto-fallback for parallel rendering.
    
    Tracks peak usage, logs warnings at thresholds, and can dynamically
    reduce worker count to prevent OOM errors.
    """
    
    # Thresholds (fraction of total GPU memory)
    WARN_THRESHOLD = 0.75     # Log warning at 75%
    CRITICAL_THRESHOLD = 0.85 # Reduce workers at 85%
    EMERGENCY_THRESHOLD = 0.92 # Force single worker + cache clear at 92%
    
    def __init__(self):
        self._lock = threading.Lock()
        self._peak_allocated_mb = 0.0
        self._peak_reserved_mb = 0.0
        self._oom_count = 0
        self._fallback_active = False
        self._last_log_time = 0
        self._log_interval = 5.0  # Log at most every 5 seconds
        self._total_mb = 0.0
        
        if torch.cuda.is_available():
            self._total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            print(f"[GPU Monitor] Total VRAM: {self._total_mb:.0f}MB")
    
    def snapshot(self) -> dict:
        """Take a memory snapshot. Returns dict with MB values."""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0, "total_mb": 0, "pct": 0}
        
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        pct = allocated / self._total_mb if self._total_mb > 0 else 0
        
        with self._lock:
            self._peak_allocated_mb = max(self._peak_allocated_mb, allocated)
            self._peak_reserved_mb = max(self._peak_reserved_mb, reserved)
        
        return {
            "allocated_mb": round(allocated, 1),
            "reserved_mb": round(reserved, 1),
            "total_mb": round(self._total_mb, 1),
            "pct": round(pct, 4),
        }
    
    def check_and_log(self, context: str = "") -> str:
        """Check memory level and log if threshold exceeded.
        
        Returns:
            'ok', 'warn', 'critical', or 'emergency'
        """
        snap = self.snapshot()
        pct = snap["pct"]
        level = "ok"
        
        if pct >= self.EMERGENCY_THRESHOLD:
            level = "emergency"
        elif pct >= self.CRITICAL_THRESHOLD:
            level = "critical"
        elif pct >= self.WARN_THRESHOLD:
            level = "warn"
        
        now = time.time()
        if level != "ok" and (now - self._last_log_time) > self._log_interval:
            self._last_log_time = now
            prefix = context + " " if context else ""
            print(f"⚠️  [GPU Monitor] {prefix}{level.upper()}: "
                  f"{snap['allocated_mb']:.0f}MB / {snap['total_mb']:.0f}MB "
                  f"({pct*100:.1f}%) [peak: {self._peak_allocated_mb:.0f}MB]")
        
        return level
    
    def get_safe_workers(self, requested_workers: int) -> int:
        """Return safe worker count based on current memory pressure.
        
        Auto-falls back to fewer workers under memory pressure.
        """
        level = self.check_and_log("worker-check")
        
        if level == "emergency":
            if not self._fallback_active:
                print("🚨 [GPU Monitor] EMERGENCY: Forcing 1 worker + clearing cache")
                self._fallback_active = True
                torch.cuda.empty_cache()
            return 1
        elif level == "critical":
            if requested_workers > 1:
                print(f"⚠️  [GPU Monitor] CRITICAL: Reducing workers {requested_workers} → 1")
                self._fallback_active = True
            return 1
        else:
            if self._fallback_active:
                print("✅ [GPU Monitor] Memory pressure eased, restoring requested workers")
                self._fallback_active = False
            return requested_workers
    
    def record_oom(self):
        """Record an OOM event for tracking."""
        with self._lock:
            self._oom_count += 1
        snap = self.snapshot()
        print(f"🚨 [GPU Monitor] OOM #{self._oom_count}! "
              f"{snap['allocated_mb']:.0f}MB / {snap['total_mb']:.0f}MB")
    
    def get_stats(self) -> dict:
        """Return monitoring statistics."""
        snap = self.snapshot()
        with self._lock:
            return {
                **snap,
                "peak_allocated_mb": round(self._peak_allocated_mb, 1),
                "peak_reserved_mb": round(self._peak_reserved_mb, 1),
                "oom_count": self._oom_count,
                "fallback_active": self._fallback_active,
            }

# Global GPU memory monitor instance
gpu_monitor = GPUMemoryMonitor()


class ParallelRenderer:
    def __init__(self, agent, num_workers: int = 2, chunk_size: int = 3):
        self.agent = agent
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.lock = threading.Lock()
        self.render_batch_size = 3  # Frames per batch within each worker
        print(f"[ParallelRenderer] workers={num_workers}, chunk_size={chunk_size}, "
              f"render_batch={self.render_batch_size}")
    
    def _get_effective_workers(self):
        """Get worker count adjusted for memory pressure."""
        return gpu_monitor.get_safe_workers(self.num_workers)

    def render_chunk(self, chunk_id, start_idx, end_idx, sample, g_r, m_r, f_r):
        try:
            frames = []
            gpu_monitor.check_and_log(f"render-chunk-{chunk_id}")
            for t in range(start_idx, end_idx):
                ta_c = self.agent.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.agent.renderer.latent_token_decoder(ta_c)
                out_frame = self.agent.renderer.decode(m_c, m_r, f_r)
                frames.append(out_frame.cpu())
                del ta_c, m_c, out_frame  # Explicit cleanup
            return (chunk_id, frames, None)
        except torch.cuda.OutOfMemoryError:
            gpu_monitor.record_oom()
            torch.cuda.empty_cache()
            # Retry one frame at a time with cache clearing
            frames = []
            for t in range(start_idx, end_idx):
                torch.cuda.empty_cache()
                ta_c = self.agent.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.agent.renderer.latent_token_decoder(ta_c)
                out_frame = self.agent.renderer.decode(m_c, m_r, f_r)
                frames.append(out_frame.cpu())
                del ta_c, m_c, out_frame
            return (chunk_id, frames, None)
        except Exception as e:
            return (chunk_id, None, traceback.format_exc())

    def render_chunk_to_numpy(self, chunk_id, start_idx, end_idx, sample, g_r, m_r, f_r):
        """Render chunk with batched processing and explicit memory cleanup."""
        try:
            frames = []
            gpu_monitor.check_and_log(f"numpy-chunk-{chunk_id}")
            
            # Batched rendering within each chunk for throughput
            batch_size = self.render_batch_size
            for b_start in range(start_idx, end_idx, batch_size):
                b_end = min(b_start + batch_size, end_idx)
                curr_batch = b_end - b_start
                
                if curr_batch > 1:
                    # Batch process multiple frames together
                    try:
                        sample_batch = sample[:, b_start:b_end, ...].squeeze(0)  # [batch, D]
                        g_r_batch = g_r.repeat(curr_batch, 1)
                        m_r_batch = tuple(m.repeat(curr_batch, 1, 1, 1) for m in m_r)
                        f_r_batch = [f.repeat(curr_batch, 1, 1, 1) for f in f_r]
                        
                        ta_c_batch = self.agent.renderer.adapt(sample_batch, g_r_batch)
                        m_c_batch = self.agent.renderer.latent_token_decoder(ta_c_batch)
                        out_batch = self.agent.renderer.decode(m_c_batch, m_r_batch, f_r_batch)
                        
                        for f_idx in range(curr_batch):
                            frame_np = out_batch[f_idx:f_idx+1].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                            if frame_np.min() < 0:
                                frame_np = (frame_np + 1) / 2
                            frame_np = np.clip(frame_np, 0, 1)
                            frame_np = (frame_np * 255).astype(np.uint8)
                            frames.append(frame_np)
                        
                        del sample_batch, g_r_batch, m_r_batch, f_r_batch
                        del ta_c_batch, m_c_batch, out_batch
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        # Fallback to sequential if batch OOMs
                        gpu_monitor.record_oom()
                        torch.cuda.empty_cache()
                        for t in range(b_start, b_end):
                            ta_c = self.agent.renderer.adapt(sample[:, t, ...], g_r)
                            m_c = self.agent.renderer.latent_token_decoder(ta_c)
                            out_frame = self.agent.renderer.decode(m_c, m_r, f_r)
                            frame_np = out_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                            if frame_np.min() < 0:
                                frame_np = (frame_np + 1) / 2
                            frame_np = np.clip(frame_np, 0, 1)
                            frame_np = (frame_np * 255).astype(np.uint8)
                            frames.append(frame_np)
                            del ta_c, m_c, out_frame
                else:
                    # Single frame — sequential
                    ta_c = self.agent.renderer.adapt(sample[:, b_start, ...], g_r)
                    m_c = self.agent.renderer.latent_token_decoder(ta_c)
                    out_frame = self.agent.renderer.decode(m_c, m_r, f_r)
                    frame_np = out_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    if frame_np.min() < 0:
                        frame_np = (frame_np + 1) / 2
                    frame_np = np.clip(frame_np, 0, 1)
                    frame_np = (frame_np * 255).astype(np.uint8)
                    frames.append(frame_np)
                    del ta_c, m_c, out_frame
            
            return (chunk_id, frames, None)
        except torch.cuda.OutOfMemoryError:
            gpu_monitor.record_oom()
            torch.cuda.empty_cache()
            # Full fallback: sequential one-at-a-time
            frames = []
            for t in range(start_idx, end_idx):
                torch.cuda.empty_cache()
                ta_c = self.agent.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.agent.renderer.latent_token_decoder(ta_c)
                out_frame = self.agent.renderer.decode(m_c, m_r, f_r)
                frame_np = out_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                if frame_np.min() < 0:
                    frame_np = (frame_np + 1) / 2
                frame_np = np.clip(frame_np, 0, 1)
                frame_np = (frame_np * 255).astype(np.uint8)
                frames.append(frame_np)
                del ta_c, m_c, out_frame
            return (chunk_id, frames, None)
        except Exception as e:
            return (chunk_id, None, traceback.format_exc())

    def render_parallel(self, sample, g_r, m_r, f_r, progress_callback=None):
        T = sample.shape[1]
        chunks = [(i // self.chunk_size, i, min(i + self.chunk_size, T)) for i in range(0, T, self.chunk_size)]
        rendered_chunks = {}
        effective_workers = self._get_effective_workers()
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_chunk = {
                executor.submit(self.render_chunk, cid, s, e, sample, g_r, m_r, f_r): cid
                for cid, s, e in chunks
            }
            for i, future in enumerate(as_completed(future_to_chunk)):
                cid, frames, err = future.result()
                if err: raise RuntimeError(err)
                rendered_chunks[cid] = frames
                if progress_callback: progress_callback(i + 1, len(chunks))
                if i % 2 == 0: torch.cuda.empty_cache()
        
        all_frames = []
        for cid in sorted(rendered_chunks.keys()):
            all_frames.extend(rendered_chunks[cid])
        return all_frames

    def render_progressive_hls(self, sample, g_r, m_r, f_r, hls_generator):
        T = sample.shape[1]
        chunks = [(i // self.chunk_size, i, min(i + self.chunk_size, T)) for i in range(0, T, self.chunk_size)]
        rendered_chunks = {}
        next_to_send = 0
        effective_workers = self._get_effective_workers()
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_chunk = {
                executor.submit(self.render_chunk_to_numpy, cid, s, e, sample, g_r, m_r, f_r): cid
                for cid, s, e in chunks
            }
            for future in as_completed(future_to_chunk):
                cid, frames, err = future.result()
                if err: continue
                rendered_chunks[cid] = frames
                while next_to_send in rendered_chunks:
                    for f in rendered_chunks[next_to_send]:
                        hls_generator.add_frame(f)
                    del rendered_chunks[next_to_send]
                    next_to_send += 1
                if cid % 2 == 0: torch.cuda.empty_cache()

    def render_parallel_stream(self, sample, g_r, m_r, f_r, output_format="tensor"):
        T = sample.shape[1]
        chunks = [(i // self.chunk_size, i, min(i + self.chunk_size, T)) for i in range(0, T, self.chunk_size)]
        rendered_chunks = {}
        next_to_yield = 0
        render_func = self.render_chunk_to_numpy if output_format == "numpy" else self.render_chunk
        effective_workers = self._get_effective_workers()
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_chunk = {
                executor.submit(render_func, cid, s, e, sample, g_r, m_r, f_r): cid
                for cid, s, e in chunks
            }
            for future in as_completed(future_to_chunk):
                cid, frames, err = future.result()
                if err: continue
                rendered_chunks[cid] = frames
                while next_to_yield in rendered_chunks:
                    for f in rendered_chunks[next_to_yield]:
                        yield f
                    del rendered_chunks[next_to_yield]
                    next_to_yield += 1
                if cid % 2 == 0: torch.cuda.empty_cache()


class InferenceAgent:
    def __init__(self, opt, models=None):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self.opt = opt
        self.device = opt.device
        self.data_processor = DataProcessor(opt)
        
        if models:
            self.renderer, self.generator = models
        else:
            print("Loading Models into InferenceAgent...")
            from generator.FM import FMGenerator
            from renderer.models import IMTRenderer
            self.renderer = IMTRenderer(self.opt).to(self.device)
            self.generator = FMGenerator(self.opt).to(self.device)
            self._load_ckpt(self.renderer, self.opt.renderer_path, "gen.")
            self._load_fm_ckpt(self.generator, self.opt.generator_path)
            self.renderer.eval()
            self.generator.eval()
        
        self.parallel_renderer = ParallelRenderer(
            agent=self,
            num_workers=getattr(opt, 'num_render_workers', 2),
            chunk_size=getattr(opt, 'render_chunk_size', 3)
        )
        print(f"InferenceAgent ready. GPU: {gpu_monitor.snapshot()['allocated_mb']:.0f}MB / {gpu_monitor.snapshot()['total_mb']:.0f}MB")

    def _load_ckpt(self, model, path, prefix="gen."):
        if not os.path.exists(path): return
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
        model.load_state_dict(clean, strict=False)

    def _load_fm_ckpt(self, model, path):
        if not os.path.exists(path): return
        checkpoint = torch.load(path, map_location="cpu")
        sd = checkpoint.get("state_dict", checkpoint)
        if "model" in sd: sd = sd["model"]
        prefix = "model."
        clean = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in clean: param.copy_(clean[name].to(self.device))

    def save_video(self, vid_tensor, fps, audio_path=None):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            raw_path = tmp.name
        if vid_tensor.dim() == 4:
            vid = vid_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        else:
            vid = vid_tensor.detach().cpu().numpy()
            
        if vid.min() < 0: vid = (vid + 1) / 2
        vid = (np.clip(vid, 0, 1) * 255).astype(np.uint8)
        h, w = vid.shape[1], vid.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))
        for frame in vid:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        
        if audio_path:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
                final_path = tmp_out.name
            # Use system FFmpeg (/usr/bin/ffmpeg) which has libx264, not conda FFmpeg
            cmd = f'/usr/bin/ffmpeg -y -i "{raw_path}" -i "{audio_path}" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}"'
            subprocess.call(cmd, shell=True)
            if os.path.exists(raw_path): os.remove(raw_path)
            return final_path
        return raw_path

    @torch.no_grad()
    def run_audio_inference(self, img_pil, aud_path, crop, seed, nfe, cfg_scale, pose_style=None, gaze_style=None, is_streaming=False):
        torch.cuda.empty_cache()
        s_pil = self.data_processor.process_img(img_pil) if crop else img_pil.resize((512, 512))
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        
        T = round(a_tensor.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        data = {"s": s_tensor, "a": a_tensor, "pose": None, "cam": None, "gaze": None, "ref_x": None}
        
        if pose_style:
            data["pose"] = torch.tensor(pose_style, device=self.device).unsqueeze(0).unsqueeze(0).expand(1, T, 3)
        if gaze_style:
            data["gaze"] = torch.tensor(gaze_style, device=self.device).unsqueeze(0).unsqueeze(0).expand(1, T, 2)
            
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple): t_lat = t_lat[0]
        data["ref_x"] = t_lat
        
        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        if isinstance(sample, tuple):
            sample = sample[0]
        
        motion_scale = 0.6
        sample = (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) + motion_scale * sample
        
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        if is_streaming:
            frames = self.parallel_renderer.render_parallel(sample, g_r, m_r, f_r)
        else:
            frames = []
            for t in range(sample.shape[1]):
                ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.renderer.latent_token_decoder(ta_c)
                frames.append(self.renderer.decode(m_c, m_r, f_r).cpu())
                if t % 10 == 0: torch.cuda.empty_cache()
        
        vid_tensor = torch.stack(frames, dim=1).squeeze(0)
        return self.save_video(vid_tensor, self.opt.fps, aud_path)

    @torch.no_grad()
    def run_audio_inference_streaming(self, img_pil, aud_path, crop, seed, nfe, cfg_scale):
        s_pil = self.data_processor.process_img(img_pil) if crop else img_pil.resize((512, 512))
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple): t_lat = t_lat[0]
        
        torch.manual_seed(seed)
        sample = self.generator.sample({"s": s_tensor, "a": a_tensor, "pose": None, "cam": None, "gaze": None, "ref_x": t_lat}, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        if isinstance(sample, tuple): sample = sample[0]
        if isinstance(sample, tuple): sample = sample[0]
        
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        for t in range(sample.shape[1]):
            ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            yield self.renderer.decode(m_c, m_r, f_r).squeeze(0)

    @torch.no_grad()
    def run_audio_inference_progressive(self, img_pil, aud_path, crop, seed, nfe, cfg_scale, hls_generator, pose_style=None, gaze_style=None, is_final_chunk=False):
        torch.cuda.empty_cache()
        s_pil = self.data_processor.process_img(img_pil) if crop else img_pil.resize((512, 512))
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        
        T = round(a_tensor.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        data = {"s": s_tensor, "a": a_tensor, "pose": None, "cam": None, "gaze": None, "ref_x": None}
        if pose_style: data["pose"] = torch.tensor(pose_style).unsqueeze(0).unsqueeze(0).expand(1, T, 3).to(self.device)
        if gaze_style: data["gaze"] = torch.tensor(gaze_style).unsqueeze(0).unsqueeze(0).expand(1, T, 2).to(self.device)
            
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple): t_lat = t_lat[0]
        data["ref_x"] = t_lat
        
        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        if isinstance(sample, tuple):
            sample = sample[0]
        motion_scale = 0.6
        sample = (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) + motion_scale * sample
        
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        self.parallel_renderer.render_progressive_hls(sample, g_r, m_r, f_r, hls_generator)
        hls_generator.finalize(audio_path=aud_path, mark_complete=is_final_chunk)

    @torch.no_grad()
    def run_video_inference(self, source_img_pil, driving_video_path, crop):
        torch.cuda.empty_cache()
        s_pil = self.data_processor.process_img(source_img_pil) if crop else source_img_pil.resize((512, 512))
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        f_r, i_r = self.renderer.app_encode(s_tensor)
        t_r = self.renderer.mot_encode(s_tensor)
        ta_r = self.renderer.adapt(t_r, i_r)
        ma_r = self.renderer.mot_decode(ta_r)
        
        final_driving = driving_video_path
        if crop:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            self.data_processor.crop_video_stable(driving_video_path, tmp)
            final_driving = tmp
            
        cap = cv2.VideoCapture(final_driving)
        fps = cap.get(cv2.CAP_PROP_FPS)
        results = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            p = Image.fromarray(frame).resize((512, 512))
            d_t = self.data_processor.transform(p).unsqueeze(0).to(self.device)
            t_c = self.renderer.mot_encode(d_t)
            ta_c = self.renderer.adapt(t_c, i_r)
            ma_c = self.renderer.mot_decode(ta_c)
            results.append(self.renderer.decode(ma_c, ma_r, f_r).cpu())
            if len(results) % 10 == 0: torch.cuda.empty_cache()
        cap.release()
        if crop and os.path.exists(final_driving): os.remove(final_driving)
        
        vid_tensor = torch.cat(results, dim=0)
        return self.save_video(vid_tensor, fps=fps, audio_path=driving_video_path)

    @torch.no_grad()
    def run_audio_inference_chunked(self, img_pil, audio_chunk_np, crop, seed, nfe, cfg_scale, hls_generator, pose_style=None, gaze_style=None):
        torch.cuda.empty_cache()
        s_pil = self.data_processor.process_img(img_pil) if crop else img_pil.resize((512, 512))
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = torch.from_numpy(audio_chunk_np).float().unsqueeze(0).to(self.device)
        
        T = round(a_tensor.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        data = {"s": s_tensor, "a": a_tensor, "pose": None, "cam": None, "gaze": None, "ref_x": None}
        if pose_style: data["pose"] = torch.tensor(pose_style).unsqueeze(0).unsqueeze(0).expand(1, T, 3).to(self.device)
        if gaze_style: data["gaze"] = torch.tensor(gaze_style).unsqueeze(0).unsqueeze(0).expand(1, T, 2).to(self.device)
            
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple): t_lat = t_lat[0]
        data["ref_x"] = t_lat
        
        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        if isinstance(sample, tuple):
            sample = sample[0]
        motion_scale = 0.6
        sample = (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) + motion_scale * sample
        
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        frames = 0
        for t in range(sample.shape[1]):
            ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            out = self.renderer.decode(m_c, m_r, f_r)
            f_np = (np.clip(out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            hls_generator.add_frame(f_np)
            frames += 1
        return frames


# --- Conversation Manager ---

class ConversationManager:
    def __init__(self, openai_client, tts_model, imtalker_streamer, video_track, audio_track, avatar_img, system_prompt=None, reference_audio=None, sync_manager=None):
        self.openai_client = openai_client
        self.tts_model = tts_model
        self.imtalker_streamer = imtalker_streamer
        self.video_track = video_track
        self.audio_track = audio_track
        self.avatar_img = avatar_img
        self.sync_manager = sync_manager
        
        # Use deque for efficient appends (ring buffer)
        self.audio_buffer = collections.deque()  # Buffer for current spoken segment
        self.preroll_buffer = collections.deque(maxlen=15) # ~0.6s of pre-speech audio (40ms chunks)
        
        # VAD optimization: cache resampler arrays
        self.resampler_cache = {}  # {sample_rate: (x_old, x_new)} pre-computed arrays
        self.vad_sample_counter = 0  # Sample every other frame for efficiency
        
        # Frame interpolation settings
        self.enable_frame_blending = True  # Smooth transitions between chunks
        self.blend_steps = 1  # Single interpolated frame (lighter for real-time)
        self.last_chunk_final_frame = None  # Track last frame for blending
        
        # Pipeline parallelization settings
        self.enable_parallel_pipeline = True  # Concurrent TTS/Render
        from concurrent.futures import ThreadPoolExecutor
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TTS")
        self.render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Render")
        
        self.is_speaking = False # State of AI response
        self.user_is_talking = False # State of user speech detection
        # Simpler prompt — no need for chunk labels, we handle chunking ourselves
        default_prompt = """You are a friendly and helpful AI assistant avatar. 
Keep your answers brief and engaging, suitable for a voice conversation.
Respond naturally in 2-4 sentences."""
        self.system_prompt = system_prompt if system_prompt else default_prompt
        self.reference_audio = reference_audio
        
        # VAD parameters (tuned for snappy conversation)
        self.start_threshold = 0.012 # Lowered to catch quieter speech
        self.stop_threshold = 0.008  # Lowered to detect natural pauses
        self.silence_duration = 0.5  # 500ms for faster turn-taking
        
        self.last_speech_time = time.time()
        self._processing_task = None
        self._loop = asyncio.get_event_loop()

    def cleanup(self):
        """Release all resources held by this ConversationManager."""
        print("CM: Cleaning up resources...")
        self.is_speaking = False
        self.user_is_talking = False
        
        # Shutdown thread pool executors
        try:
            self.tts_executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            self.render_executor.shutdown(wait=False)
        except Exception:
            pass
        
        # Clear audio buffer
        self.audio_buffer.clear()
        self.preroll_buffer.clear()
        
        # Clear references to tracks/models (don't delete — shared)
        self.video_track = None
        self.audio_track = None
        self.avatar_img = None
        self.last_chunk_final_frame = None
        
        # Cancel any pending processing task
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
        
        print("CM: Cleanup complete.")

    async def add_microphone_audio(self, audio_np, sample_rate):
        """Called when new audio comes from WebRTC microphone"""
        if self.is_speaking:
            # Drop audio while avatar is talking
            return

        if audio_np is None or audio_np.size == 0:
            return

        # Normalize to float32 if int16
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
        
        # Ensure it's float32 for processing
        audio_np = audio_np.astype(np.float32)

        # Handle stereo to mono - safer logic to collapse smaller dimension (channels)
        if audio_np.ndim > 1:
            if audio_np.shape[0] < audio_np.shape[1]:
                audio_np = np.mean(audio_np, axis=0)
            else:
                audio_np = np.mean(audio_np, axis=1)

        # Check for NaNs early
        if np.isnan(audio_np).any():
            if np.isnan(audio_np).all():
                return
            audio_np = np.nan_to_num(audio_np)

        # Ensure 16kHz mono for OpenAI with cached resampler
        if sample_rate != 16000 and audio_np.size > 1:
            # Use cached resampler arrays to avoid recomputation
            if sample_rate not in self.resampler_cache:
                # Pre-compute for a typical chunk size (960 samples at 24kHz)
                # Will be close enough for variable-length chunks
                pass
            # Perform resampling
            x_old = np.arange(len(audio_np))
            x_new = np.linspace(0, len(audio_np) - 1, int(len(audio_np) * 16000 / sample_rate))
            audio_np = np.interp(x_new, x_old, audio_np)
        
        chunk = audio_np.flatten()
        if chunk.size == 0:
            return
        
        # Optimize: Only calculate energy every other frame to reduce overhead
        self.vad_sample_counter += 1
        should_check_energy = (self.vad_sample_counter % 2 == 0)
        
        if should_check_energy:
            energy = np.abs(chunk).mean()
            self._last_energy = energy
        else:
            # Use cached energy from last frame (approximate)
            if not hasattr(self, '_last_energy'):
                self._last_energy = 0.0
            energy = self._last_energy
            
        # Periodic debug log (every 1 second approx)
        if hasattr(self, '_last_log_time') and (time.time() - self._last_log_time) > 1.0:
            print(f"VAD Energy: {energy:.6f} (Talking: {self.user_is_talking})")
            self._last_log_time = time.time()
        elif not hasattr(self, '_last_log_time'):
            self._last_log_time = time.time()

        if not self.user_is_talking:
            if energy > self.start_threshold:
                print(f"CM: Speech started (energy: {energy:.6f})")
                self.user_is_talking = True
                # Move preroll to buffer (convert deque to deque)
                self.audio_buffer = collections.deque(self.preroll_buffer)
                self.audio_buffer.append(chunk)
                self.last_speech_time = time.time()
            else:
                self.preroll_buffer.append(chunk)
        else:
            self.audio_buffer.append(chunk)
            if energy > self.stop_threshold:
                self.last_speech_time = time.time()
            
            # Check for silence timeout
            silence_gap = time.time() - self.last_speech_time
            if silence_gap > self.silence_duration:
                print(f"CM: Silence detected ({silence_gap:.2f}s), processing recording...")
                self.user_is_talking = False
                if not self._processing_task or self._processing_task.done():
                    self._processing_task = asyncio.create_task(self.process_conversation())
                else:
                    print("CM: Already processing a turn, skipping duplicate trigger.")

    def trim_silence(self, audio, threshold=0.005):
        """Trims leading and trailing silence from an audio segment"""
        # Find start index
        start_idx = 0
        chunk_size = 160 # 10ms at 16kHz
        for i in range(0, len(audio), chunk_size):
            if np.abs(audio[i : i + chunk_size]).mean() > threshold:
                start_idx = max(0, i - 320) # Keep a bit of padding
                break
        
        # Find end index
        end_idx = len(audio)
        for i in range(len(audio) - chunk_size, 0, -chunk_size):
            if np.abs(audio[i : i + chunk_size]).mean() > threshold:
                end_idx = min(len(audio), i + chunk_size + 320)
                break
        
        return audio[start_idx:end_idx]

    async def process_conversation(self):
        print("CM: Entering process_conversation")
        pipeline_start = time.time()
        try:
            if not self.audio_buffer:
                print("CM: Buffer empty in process_conversation")
                return

            # Convert deque to numpy array
            full_audio = np.concatenate(list(self.audio_buffer))
            self.audio_buffer.clear()  # Clear deque efficiently
            self.preroll_buffer.clear()
            
            # Trim background noise from edges
            trimmed_audio = self.trim_silence(full_audio, self.stop_threshold)
            
            if len(trimmed_audio) < 2400: # Less than 0.15s
                print(f"CM: Audio too short after trimming ({len(trimmed_audio)} samples), ignoring.")
                return

            print(f"CM: Sending audio to OpenAI ({len(trimmed_audio)/16000:.2f}s, original {len(full_audio)/16000:.2f}s)...")
            
            # Convert to WAV base64
            t_encode = time.time()
            byte_io = io.BytesIO()
            with wave.open(byte_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes((trimmed_audio * 32767).astype(np.int16).tobytes())
            audio_base64 = base64.b64encode(byte_io.getvalue()).decode('utf-8')
            print(f"⏱️  Audio encoding: {time.time() - t_encode:.3f}s")

            t_openai = time.time()
            # Use streaming + gpt-4o-audio-preview for faster response with audio input support
            stream = await self.openai_client.chat.completions.create(
                model="gpt-4o-audio-preview",
                modalities=["text"],
                stream=True,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_base64, "format": "wav"},
                            }
                        ],
                    },
                ],
            )
            
            # Accumulate streamed response
            text_response = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    text_response += chunk.choices[0].delta.content
            
            openai_time = time.time() - t_openai
            print(f"⏱️  OpenAI response: {openai_time:.3f}s")
            print(f"CM: Assistant response from OpenAI: {text_response}")
            
            if text_response:
                await self.generate_and_feed(text_response)
            else:
                print("CM: Empty response from OpenAI")
            
            print(f"⏱️  Total pipeline (VAD→response complete): {time.time() - pipeline_start:.3f}s")
                
        except Exception as e:
            print(f"CM OpenAI/Process Error: {e}")
            traceback.print_exc()

    async def generate_and_feed(self, text):
        """Real-time streaming pipeline with LOCKED audio-video synchronization.
        
        Architecture:
          Stage 1 (audio_chunker): TTS text → audio tensors (CPU)
          Stage 2 (renderer): audio → paired (video_frame, audio_slice) items
          Stage 3 (av_streamer): pushes paired A/V to WebRTC tracks in LOCKSTEP
        
        The key insight: audio and video are NEVER pushed separately.
        Each item in av_queue is (video_frame, audio_samples_for_that_frame).
        av_streamer pushes both to their tracks at the SAME wall-clock instant,
        so WebRTC timestamps stay aligned regardless of rendering speed.
        """
        print(f"CM: Generating response: {text[:50]}{'...' if len(text) > 50 else ''}")
        self.is_speaking = True
        
        t_parse = time.time()
        text_chunks = parse_chunk_labels(text)
        print(f"⏱️  Chunk parsing: {time.time() - t_parse:.3f}s → {len(text_chunks)} chunks")
        
        # TTS output queue
        audio_chunks_queue = asyncio.Queue(maxsize=4)
        # PAIRED A/V queue: each item is (VideoFrame, np.ndarray_audio_int16_slice) or None
        av_queue = asyncio.Queue(maxsize=30)
        
        SAMPLES_PER_FRAME = 960  # 24000 Hz / 25 FPS = 960 audio samples per video frame
        
        try:
            # Stage 1: TTS — generate audio for each text chunk
            # Uses large merged chunks (~300 chars ≈ ~10s speech) for fewer TTS calls
            async def audio_chunker():
                tts_total_time = 0
                for chunk_idx, text_chunk in enumerate(text_chunks, 1):
                    t_tts = time.time()
                    tts_stream = self.tts_model.generate_stream(
                        text=text_chunk,
                        chunk_size=50,
                        print_metrics=False,
                        audio_prompt_path=self.reference_audio
                    )
                    audio_parts = []
                    for audio_chunk, metrics in tts_stream:
                        audio_parts.append(audio_chunk.cpu() if isinstance(audio_chunk, torch.Tensor) else audio_chunk)
                    
                    tts_chunk_time = time.time() - t_tts
                    tts_total_time += tts_chunk_time
                    
                    if audio_parts:
                        full_audio = torch.cat(audio_parts, dim=-1)
                        del audio_parts
                        # Convert to numpy immediately to free torch allocator
                        audio_np = full_audio.numpy().flatten().copy()  # .copy() ensures owns memory
                        duration = len(audio_np) / 24000
                        del full_audio
                        await audio_chunks_queue.put((audio_np, text_chunk, chunk_idx))
                        print(f"⏱️  TTS chunk {chunk_idx}/{len(text_chunks)}: {tts_chunk_time:.3f}s → {duration:.1f}s audio")
                    else:
                        del audio_parts
                
                await audio_chunks_queue.put(None)
                print(f"⏱️  TTS total: {tts_total_time:.3f}s for {len(text_chunks)} chunks")
            
            # Stage 2: Render → stream paired (frame, audio_slice) into av_queue
            async def renderer():
                render_total_time = 0
                
                while True:
                    item = await audio_chunks_queue.get()
                    if item is None:
                        await av_queue.put(None)
                        break
                    
                    audio_24k_np, text_chunk, chunk_idx = item
                    total_audio_samples = len(audio_24k_np)
                    chunk_duration = total_audio_samples / 24000
                    
                    print(f"CM: Rendering chunk {chunk_idx} ({chunk_duration:.1f}s): '{text_chunk[:40]}...'")
                    t_render = time.time()
                    
                    def render_and_stream():
                        """Render video and push pairs DIRECTLY to av_queue as they're generated.
                        
                        No accumulation — each pair is pushed immediately, keeping peak
                        memory usage to ~1 batch of frames instead of ALL frames.
                        """
                        gpu_monitor.check_and_log(f"render-chunk-{chunk_idx}")
                        
                        # Resample 24kHz → 16kHz for wav2vec2 (entirely numpy, no torch tensor needed)
                        if _HAS_TORCHAUDIO:
                            at = torch.from_numpy(audio_24k_np).float().unsqueeze(0)
                            audio_16k = AF.resample(at, 24000, 16000).squeeze(0).numpy()
                            del at
                        else:
                            num_samples_16k = int(len(audio_24k_np) * 16000 / 24000)
                            from scipy.signal import resample as scipy_resample
                            audio_16k = scipy_resample(audio_24k_np, num_samples_16k).astype(np.float32)
                        
                        # Clip audio to [-1, 1] to prevent int16 overflow distortion
                        peak = np.abs(audio_24k_np).max()
                        if peak > 1.0:
                            audio_24k_clipped = audio_24k_np / peak  # normalize
                        else:
                            audio_24k_clipped = audio_24k_np
                        
                        # Apply 5ms fade-out at the very end to prevent click/pop
                        fade_samples = min(120, total_audio_samples // 4)  # 120 samples = 5ms at 24kHz
                        if fade_samples > 0:
                            fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                            audio_24k_clipped[-fade_samples:] = audio_24k_clipped[-fade_samples:] * fade
                        
                        frame_count = 0
                        audio_cursor = 0
                        last_pair = None  # Only keep ref to LAST pair for remainder audio
                        
                        for video_frames, metrics in self.imtalker_streamer.generate_stream(
                            self.avatar_img, iter([audio_16k])
                        ):
                            for ft in video_frames:
                                # Convert frame tensor to VideoFrame
                                frame_rgb = process_frame_tensor(ft, format="RGB")
                                av_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                                del ft, frame_rgb
                                
                                # Slice the corresponding audio for this frame
                                audio_end = min(audio_cursor + SAMPLES_PER_FRAME, total_audio_samples)
                                audio_slice = audio_24k_clipped[audio_cursor:audio_end].copy()
                                audio_cursor = audio_end
                                
                                # Convert to int16 for WebRTC (already clipped to [-1,1])
                                audio_slice_int16 = (audio_slice * 32767).astype(np.int16)
                                del audio_slice
                                
                                pair = (av_frame, audio_slice_int16)
                                
                                # Push directly to av_queue (backpressure via maxsize=30)
                                fut = asyncio.run_coroutine_threadsafe(
                                    av_queue.put(pair), self._loop
                                )
                                fut.result(timeout=10)
                                frame_count += 1
                                last_pair = pair
                            del video_frames
                        
                        # If any remaining audio (rounding), push as a tiny extra pair
                        if audio_cursor < total_audio_samples and last_pair is not None:
                            remainder = audio_24k_clipped[audio_cursor:].copy()
                            remainder_int16 = (remainder * 32767).astype(np.int16)
                            del remainder
                            # Create a duplicate of the last video frame with remaining audio
                            last_frame, _ = last_pair
                            fut = asyncio.run_coroutine_threadsafe(
                                av_queue.put((last_frame, remainder_int16)), self._loop
                            )
                            fut.result(timeout=10)
                            frame_count += 1
                        
                        del audio_16k, last_pair, audio_24k_clipped
                        
                        return frame_count
                    
                    frame_count = await self._loop.run_in_executor(None, render_and_stream)
                    render_time = time.time() - t_render
                    render_total_time += render_time
                    del audio_24k_np
                    
                    print(f"⏱️  Render chunk {chunk_idx}: {render_time:.3f}s → {frame_count} A/V pairs (streamed)")
                
                print(f"⏱️  Render total: {render_total_time:.3f}s")
            
            # Stage 3: LOCKSTEP A/V streaming at 25 FPS
            async def av_streamer():
                """Push paired audio+video to WebRTC tracks simultaneously.
                
                This is the ONLY place audio and video enter the tracks.
                By pushing both from the same loop iteration, WebRTC assigns
                them correlated PTS values and they stay in sync.
                """
                total_frames = 0
                stream_start = None
                
                while True:
                    item = await av_queue.get()
                    if item is None:
                        break
                    
                    av_frame, audio_slice_int16 = item
                    
                    if stream_start is None:
                        stream_start = time.time()
                        # CRITICAL: Tell video track we're speaking (no more idle frame insertion)
                        self.video_track.start_speaking()
                        # Reset audio clock so pacing starts NOW
                        self.audio_track.reset_clock()
                        print(f"CM: First A/V pair — tracks reset, lockstep streaming starts")
                    
                    # Push BOTH audio and video at the same instant
                    # Audio: split into 20ms WebRTC frames (480 samples at 24kHz)
                    # Use put_nowait — queue is unbounded so this won't fail,
                    # but if it ever does, we drop oldest to prevent blocking.
                    frame_size = 480
                    for i in range(0, len(audio_slice_int16), frame_size):
                        chunk = audio_slice_int16[i:i + frame_size]
                        if len(chunk) < frame_size:
                            padded = np.zeros(frame_size, dtype=np.int16)
                            padded[:len(chunk)] = chunk
                            chunk = padded
                        try:
                            self.audio_track.audio_queue.put_nowait(chunk)
                        except asyncio.QueueFull:
                            # Drop oldest frame to make room (prevents blocking)
                            try:
                                self.audio_track.audio_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            self.audio_track.audio_queue.put_nowait(chunk)
                    
                    # Video: push frame directly to track queue
                    await self.video_track.frame_queue.put(av_frame)
                    
                    total_frames += 1
                    self.last_chunk_final_frame = av_frame
                    
                    if self.sync_manager:
                        self.sync_manager.notify_video_content(1)
                        self.sync_manager.notify_audio_content(len(audio_slice_int16))
                    
                    # Wall-clock aligned pacing at 25 FPS
                    expected_time = stream_start + total_frames * 0.04
                    now = time.time()
                    sleep_time = expected_time - now
                    if sleep_time > 0.002:
                        await asyncio.sleep(sleep_time)
                    elif sleep_time < -0.08:
                        # More than 2 frames behind schedule (rendering gap).
                        # DO NOT burst to catch up — that causes fast-forward.
                        # Re-anchor pacing to NOW so we resume normal 25 FPS.
                        stream_start = now - total_frames * 0.04
                        # Pause one frame period so next iteration paces correctly
                        await asyncio.sleep(0.04)
                
                # Speech finished — resume idle animation
                self.video_track.stop_speaking()
                
                elapsed = time.time() - stream_start if stream_start else 0
                fps_actual = total_frames / elapsed if elapsed > 0 else 0
                print(f"CM: Streamed {total_frames} A/V pairs in {elapsed:.2f}s ({fps_actual:.1f} FPS)")
            
            # Run all stages concurrently
            await asyncio.gather(
                audio_chunker(),
                renderer(),
                av_streamer()
            )
            
        except Exception as e:
            print(f"CM: Pipeline error: {e}")
            traceback.print_exc()
        finally:
            print("CM: Resetting is_speaking to False")
            self.is_speaking = False

    async def pre_render_to_buffer(self, text):
        """Pre-render text to a list of (VideoFrame, audio_int16) pairs WITHOUT pushing to tracks.
        
        Used by /offer to pre-generate greeting content before returning the SDP answer,
        so the frontend stays on 'connecting' screen while rendering happens.
        Returns list of (VideoFrame, np.ndarray[int16]) pairs ready for push_buffered_greeting().
        """
        print(f"CM: Pre-rendering to buffer: {text[:50]}{'...' if len(text) > 50 else ''}")
        t_start = time.time()
        
        SAMPLES_PER_FRAME = 960  # 24000 Hz / 25 FPS
        buffer_pairs = []  # Collect all (VideoFrame, audio_int16) pairs
        
        # Stage 1: TTS — generate audio
        text_chunks = parse_chunk_labels(text)
        all_audio_chunks = []  # list of (audio_np, text_chunk, chunk_idx)
        
        for chunk_idx, text_chunk in enumerate(text_chunks, 1):
            t_tts = time.time()
            tts_stream = self.tts_model.generate_stream(
                text=text_chunk,
                chunk_size=50,
                print_metrics=False,
                audio_prompt_path=self.reference_audio
            )
            audio_parts = []
            for audio_chunk, metrics in tts_stream:
                audio_parts.append(audio_chunk.cpu() if isinstance(audio_chunk, torch.Tensor) else audio_chunk)
            
            if audio_parts:
                full_audio = torch.cat(audio_parts, dim=-1)
                del audio_parts
                audio_np = full_audio.numpy().flatten().copy()
                del full_audio
                all_audio_chunks.append((audio_np, text_chunk, chunk_idx))
                print(f"⏱️  Pre-render TTS chunk {chunk_idx}/{len(text_chunks)}: {time.time() - t_tts:.3f}s → {len(audio_np)/24000:.1f}s audio")
            else:
                del audio_parts
        
        
        # Stage 2: Render — generate video frames paired with audio slices
        for audio_24k_np, text_chunk, chunk_idx in all_audio_chunks:
            total_audio_samples = len(audio_24k_np)
            t_render = time.time()
            
            def render_to_list(audio_np=audio_24k_np, total_samples=total_audio_samples):
                """Render video and collect pairs into a list (no queue push)."""
                pairs = []
                
                if _HAS_TORCHAUDIO:
                    at = torch.from_numpy(audio_np).float().unsqueeze(0)
                    audio_16k = AF.resample(at, 24000, 16000).squeeze(0).numpy()
                    del at
                else:
                    num_samples_16k = int(len(audio_np) * 16000 / 24000)
                    from scipy.signal import resample as scipy_resample
                    audio_16k = scipy_resample(audio_np, num_samples_16k).astype(np.float32)
                
                peak = np.abs(audio_np).max()
                audio_24k_clipped = audio_np / peak if peak > 1.0 else audio_np
                
                fade_samples = min(120, total_samples // 4)
                if fade_samples > 0:
                    fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                    audio_24k_clipped = audio_24k_clipped.copy()
                    audio_24k_clipped[-fade_samples:] = audio_24k_clipped[-fade_samples:] * fade
                
                audio_cursor = 0
                last_pair = None
                
                for video_frames, metrics in self.imtalker_streamer.generate_stream(
                    self.avatar_img, iter([audio_16k])
                ):
                    for ft in video_frames:
                        frame_rgb = process_frame_tensor(ft, format="RGB")
                        av_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                        del ft, frame_rgb
                        
                        audio_end = min(audio_cursor + SAMPLES_PER_FRAME, total_samples)
                        audio_slice = audio_24k_clipped[audio_cursor:audio_end].copy()
                        audio_cursor = audio_end
                        audio_slice_int16 = (audio_slice * 32767).astype(np.int16)
                        del audio_slice
                        
                        pair = (av_frame, audio_slice_int16)
                        pairs.append(pair)
                        last_pair = pair
                    del video_frames
                
                if audio_cursor < total_samples and last_pair is not None:
                    remainder = audio_24k_clipped[audio_cursor:].copy()
                    remainder_int16 = (remainder * 32767).astype(np.int16)
                    del remainder
                    last_frame, _ = last_pair
                    pairs.append((last_frame, remainder_int16))
                
                del audio_16k, last_pair, audio_24k_clipped
                return pairs
            
            chunk_pairs = await self._loop.run_in_executor(None, render_to_list)
            buffer_pairs.extend(chunk_pairs)
            print(f"⏱️  Pre-render chunk {chunk_idx}: {time.time() - t_render:.3f}s → {len(chunk_pairs)} A/V pairs")
            del audio_24k_np
        
        elapsed = time.time() - t_start
        print(f"CM: Pre-rendered {len(buffer_pairs)} A/V pairs in {elapsed:.2f}s")
        return buffer_pairs

    async def push_buffered_greeting(self, pairs):
        """Push pre-rendered A/V pairs through tracks with proper 25 FPS pacing.
        
        Called after WebRTC connection is established to play the pre-rendered greeting.
        """
        if not pairs:
            return
        
        print(f"CM: Pushing {len(pairs)} pre-rendered A/V pairs...")
        self.is_speaking = True
        
        # Tell video track we're speaking
        self.video_track.start_speaking()
        # Reset audio clock so pacing starts NOW
        self.audio_track.reset_clock()
        
        stream_start = time.time()
        total_frames = 0
        
        for av_frame, audio_slice_int16 in pairs:
            # Push audio: split into 20ms WebRTC frames (480 samples at 24kHz)
            frame_size = 480
            for i in range(0, len(audio_slice_int16), frame_size):
                chunk = audio_slice_int16[i:i + frame_size]
                if len(chunk) < frame_size:
                    padded = np.zeros(frame_size, dtype=np.int16)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                try:
                    self.audio_track.audio_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    try:
                        self.audio_track.audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self.audio_track.audio_queue.put_nowait(chunk)
            
            # Push video frame
            await self.video_track.frame_queue.put(av_frame)
            
            total_frames += 1
            self.last_chunk_final_frame = av_frame
            
            # Wall-clock pacing at 25 FPS
            expected_time = stream_start + total_frames * 0.04
            now = time.time()
            sleep_time = expected_time - now
            if sleep_time > 0.002:
                await asyncio.sleep(sleep_time)
            elif sleep_time < -0.08:
                stream_start = now - total_frames * 0.04
                await asyncio.sleep(0.04)
        
        # Speech finished — resume idle
        self.video_track.stop_speaking()
        self.is_speaking = False
        
        elapsed = time.time() - stream_start
        fps_actual = total_frames / elapsed if elapsed > 0 else 0
        print(f"CM: Greeting pushed: {total_frames} A/V pairs in {elapsed:.2f}s ({fps_actual:.1f} FPS)")


# --- A/V Sync Manager ---

class AVSyncManager:
    """Monitors audio-video content delivery drift for WebRTC streams.
    
    Tracks when real content (not silence/idle frames) is delivered to each track,
    and logs warnings when content delivery times diverge. Does NOT manage PTS
    (WebRTC's built-in next_timestamp() handles transport-level timing).
    """
    
    def __init__(self, video_fps=25.0, audio_sample_rate=24000, drift_threshold_ms=100):
        self.video_fps = video_fps
        self.audio_sample_rate = audio_sample_rate
        self.drift_threshold_ms = drift_threshold_ms
        
        # Track cumulative CONTENT duration delivered (not transport frames)
        self._audio_content_s = 0.0  # seconds of real audio content delivered
        self._video_content_s = 0.0  # seconds of real video content delivered
        
        self._lock = threading.Lock()
        self._drift_history = collections.deque(maxlen=50)
        self._last_drift_log_time = 0
    
    def reset(self):
        """Reset for a new conversation turn."""
        with self._lock:
            self._audio_content_s = 0.0
            self._video_content_s = 0.0
            self._drift_history.clear()
    
    def notify_audio_content(self, num_samples):
        """Called when real audio content is queued to the audio track."""
        with self._lock:
            self._audio_content_s += num_samples / self.audio_sample_rate
    
    def notify_video_content(self, num_frames):
        """Called when real video frames are queued to the video track."""
        with self._lock:
            self._video_content_s += num_frames / self.video_fps
    
    def get_drift_ms(self):
        """Calculate audio-video content delivery drift in milliseconds.
        
        Positive = more audio content than video, Negative = more video than audio.
        """
        with self._lock:
            audio_t = self._audio_content_s
            video_t = self._video_content_s
        
        drift_ms = (audio_t - video_t) * 1000
        self._drift_history.append(drift_ms)
        
        # Log periodically (at most once per 2 seconds)
        now = time.time()
        if now - self._last_drift_log_time > 2.0:
            avg_drift = sum(self._drift_history) / len(self._drift_history) if self._drift_history else 0
            if abs(avg_drift) > self.drift_threshold_ms:
                print(f"⚠️ SYNC DRIFT: avg={avg_drift:.1f}ms (audio={'ahead' if avg_drift > 0 else 'behind'}) "
                      f"[audio_content={audio_t:.2f}s, video_content={video_t:.2f}s]")
            self._last_drift_log_time = now
        
        return drift_ms


# --- Idle Animation Cache ---

class IdleAnimationCache:
    """Pre-renders and caches idle animation frames using silent audio.
    
    Feeds 10 seconds of silence through IMTalker to generate ~250 frames of
    natural micro-motion. Cached per-avatar (by image hash) so subsequent
    sessions reuse the same loop instantly.
    """
    
    IDLE_DURATION_S = 4.0  # Duration of idle loop (4s - reduced from 10s to save memory)
    SAMPLE_RATE = 16000    # wav2vec2 expected rate
    
    def __init__(self):
        self._cache = {}  # {avatar_hash: List[VideoFrame]}
        self._lock = threading.Lock()
        self._rendering = set()  # avatar hashes currently being rendered
    
    @staticmethod
    def _hash_avatar(avatar_img):
        """Fast hash of avatar image for cache key."""
        import hashlib
        arr = np.array(avatar_img.convert('RGB'))
        # Use a downsampled version for speed
        small = arr[::8, ::8, :].tobytes()
        return hashlib.md5(small).hexdigest()
    
    def get_idle_frames(self, avatar_img):
        """Get cached idle frames for an avatar, or None if not yet rendered."""
        key = self._hash_avatar(avatar_img)
        with self._lock:
            return self._cache.get(key)
    
    def is_rendering(self, avatar_img):
        """Check if idle animation is currently being rendered for this avatar."""
        key = self._hash_avatar(avatar_img)
        with self._lock:
            return key in self._rendering
    
    def render_idle(self, avatar_img, imtalker_streamer):
        """Render idle animation frames (blocking, call from thread).
        
        Generates silent audio and feeds it through the IMTalker pipeline
        to produce natural micro-motion frames.
        
        Returns:
            List[VideoFrame] - the cached idle loop frames
        """
        key = self._hash_avatar(avatar_img)
        
        # Check cache first
        with self._lock:
            if key in self._cache:
                print(f"[IDLE CACHE] HIT for avatar {key[:8]}... ({len(self._cache[key])} frames)")
                return self._cache[key]
            if key in self._rendering:
                print(f"[IDLE CACHE] Already rendering for {key[:8]}..., waiting...")
                # Wait for render to complete
                pass
            self._rendering.add(key)
        
        try:
            t_start = time.time()
            print(f"[IDLE CACHE] Rendering idle animation for avatar {key[:8]}...")
            
            # Generate silent audio (2 seconds at 16kHz)
            # Use very low amplitude noise instead of pure zeros for more natural result
            silent_audio = np.random.randn(int(self.IDLE_DURATION_S * self.SAMPLE_RATE)).astype(np.float32) * 0.001
            
            def audio_iter():
                yield silent_audio
            
            idle_frames = []
            for video_frames, metrics in imtalker_streamer.generate_stream(avatar_img, audio_iter()):
                for frame_tensor in video_frames:
                    frame_rgb = process_frame_tensor(frame_tensor, format="RGB")
                    av_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                    idle_frames.append(av_frame)
                    del frame_tensor, frame_rgb
                del video_frames
            
            # Clear GPU memory after render
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            render_time = time.time() - t_start
            print(f"[IDLE CACHE] ✓ Rendered {len(idle_frames)} idle frames in {render_time:.2f}s for avatar {key[:8]}...")
            
            # Store in cache
            with self._lock:
                self._cache[key] = idle_frames
                self._rendering.discard(key)
            
            return idle_frames
            
        except Exception as e:
            print(f"[IDLE CACHE] ERROR rendering idle animation: {e}")
            traceback.print_exc()
            with self._lock:
                self._rendering.discard(key)
            return None

# Global idle cache instance
idle_animation_cache = IdleAnimationCache()


# --- WebRTC Tracks ---

class IMTalkerVideoTrack(VideoStreamTrack):
    """WebRTC video track with clean idle ↔ speech transitions.
    
    Key insight: aiortc's next_timestamp() calls recv() at its own internal
    rate (~30 FPS). Content is pushed at 25 FPS. Between pushes, the queue
    may be momentarily empty. During active speech, we REPEAT the last
    speech frame instead of inserting idle frames (which caused glitches).
    
    States:
      is_speaking=False → show idle animation (or static avatar)
      is_speaking=True, queue has frame → show speech frame, save as last_speech_frame
      is_speaking=True, queue empty → repeat last_speech_frame (no idle insertion)
    """
    def __init__(self, avatar_img, sync_manager=None, idle_frames=None):
        super().__init__()
        self.avatar_img = avatar_img
        self.frame_queue = asyncio.Queue()
        self.sync_manager = sync_manager
        self.static_idle_frame = np.array(avatar_img.convert('RGB'))
        
        self._idle_frames = idle_frames or []
        self._idle_index = 0
        self.is_speaking = False        # Set by av_streamer
        self._last_speech_frame = None  # Hold frame for repeat during speech

    def set_idle_frames(self, frames):
        self._idle_frames = frames
        print(f"VT: Idle animation set ({len(frames)} frames)")

    def _get_idle_frame(self):
        if self._idle_frames:
            frame = self._idle_frames[self._idle_index % len(self._idle_frames)]
            self._idle_index += 1
            return frame
        else:
            return VideoFrame.from_ndarray(self.static_idle_frame, format="rgb24")

    def start_speaking(self):
        """Called when av_streamer begins pushing content.
        Drains any stale frames from previous speech turn."""
        self.is_speaking = True
        self._last_speech_frame = None
        # Drain stale frames from previous turn
        drained = 0
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            print(f"VT: Drained {drained} stale frames from queue")

    def stop_speaking(self):
        """Called when av_streamer finishes. Resume idle animation."""
        self.is_speaking = False
        # Keep _last_speech_frame for smooth transition (first idle blends naturally)

    async def add_frame(self, av_frame):
        await self.frame_queue.put(av_frame)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        try:
            # FIFO delivery — serve frames in order, never skip.
            # Wall-clock pacing in av_streamer prevents accumulation,
            # and queue maxsize=30 provides backpressure if needed.
            frame = self.frame_queue.get_nowait()
            # Got a real speech frame — save it for repeat
            self._last_speech_frame = frame
        except asyncio.QueueEmpty:
            if self.is_speaking and self._last_speech_frame is not None:
                # During active speech, repeat the last speech frame.
                # This prevents idle frames from being interspersed.
                frame = self._last_speech_frame
            else:
                # Truly idle — show idle animation
                frame = self._get_idle_frame()
        
        frame.pts = pts
        frame.time_base = time_base
        return frame

class IMTalkerAudioTrack(AudioStreamTrack):
    """WebRTC audio track with clock-reset capability for lip sync.
    
    Key insight: recv() pacing is anchored to _recv_start + _timestamp/sample_rate.
    During idle (silence), _recv_start is set once and _timestamp advances with silence
    frames. When REAL content arrives, we call reset_clock() which re-anchors
    _recv_start to NOW and resets _timestamp to 0. This ensures the audio clock
    starts fresh at the exact moment content flows, matching the video track's content
    delivery timing.
    """
    def __init__(self, sync_manager=None):
        super().__init__()
        self.audio_queue = asyncio.Queue()  # Unbounded — overflow handled by caller
        self.sync_manager = sync_manager
        self._timestamp = 0
        self._recv_start = None  # Set on first recv(); reset when content starts
        self._content_frames = 0  # Count of real (non-silence) frames delivered
        self.sample_rate = 24000
        self.is_buffering = False

    def reset_clock(self):
        """Re-anchor the pacing clock to NOW.
        
        Called by av_streamer when the first real A/V pair is pushed.
        This ensures the audio pacing clock starts at the same wall-clock
        instant as the video content, eliminating lip-sync drift.
        """
        self._recv_start = time.time()
        self._timestamp = 0
        self._content_frames = 0
        # Drain any stale silence/old data from the queue
        drained = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained > 0:
            print(f"AT: Clock reset, drained {drained} stale frames")
        else:
            print(f"AT: Clock reset (queue empty)")

    async def add_audio(self, audio_np):
        """Expects normalized float32 numpy array. Splits into WebRTC-sized frames."""
        audio_int16 = (audio_np * 32767).astype(np.int16)
        frame_size = 480  # 20ms at 24kHz
        num_frames = len(audio_int16) // frame_size
        
        for i in range(num_frames):
            await self.audio_queue.put(audio_int16[i * frame_size:(i + 1) * frame_size])
        
        remainder = len(audio_int16) % frame_size
        if remainder > 0:
            last_frame = np.zeros(frame_size, dtype=np.int16)
            last_frame[:remainder] = audio_int16[-remainder:]
            await self.audio_queue.put(last_frame)

    async def recv(self):
        samples_per_frame = 480  # 20ms at 24kHz
        
        # Initialize clock on very first call
        if self._recv_start is None:
            self._recv_start = time.time()
        
        # Wall-clock aligned pacing: sleep until it's time for this frame.
        # target_time = anchor + (cumulative_samples / rate)
        # This creates a perfectly smooth 20ms cadence regardless of jitter.
        target_time = self._recv_start + (self._timestamp / self.sample_rate)
        wait = target_time - time.time()
        if wait > 0.001:  # Only sleep if > 1ms (avoid micro-sleeps)
            await asyncio.sleep(wait)
        elif wait < -0.1:
            # Clock is >100ms behind reality — we've fallen behind.
            # Re-anchor to prevent permanent burst mode.
            self._recv_start = time.time() - (self._timestamp / self.sample_rate)
        
        # Non-blocking get — returns silence if queue is empty.
        try:
            audio_int16 = self.audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            audio_int16 = np.zeros(samples_per_frame, dtype=np.int16)
        
        frame = AudioFrame.from_ndarray(audio_int16.reshape(1, -1), format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._timestamp += frame.samples
        
        return frame

@app.post("/generate")
async def generate_video(request: GenerateRequest):
    try:
        # Load models from app state
        tts_model = app.state.tts_model
        imtalker_streamer = app.state.imtalker_streamer
        
        # Load Avatar
        avatar_path = os.path.join(current_dir, request.avatar)
        if not os.path.exists(avatar_path):
            avatar_path = request.avatar
            
        if not os.path.exists(avatar_path):
            avatar_img = Image.new('RGB', (512, 512), color='red')
            print(f"API: Avatar not found, using dummy: {avatar_path}")
        else:
            avatar_img = Image.open(avatar_path).convert('RGB')
            
        # Create TTS Stream
        tts_stream = tts_model.generate_stream(
            text=request.text,
            chunk_size=50,  # Larger chunks for less overhead
            print_metrics=False
        )
        audio_iterator = chatterbox_adapter(tts_stream)
        
        # Generate Frames
        all_frames = []
        for video_frames, metrics in imtalker_streamer.generate_stream(avatar_img, audio_iterator):
            for frame_tensor in video_frames:
                frame_bgr = process_frame_tensor(frame_tensor)
                all_frames.append(frame_bgr)
        
        # Save Video
        if all_frames:
            height, width, _ = all_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(request.output_path, fourcc, 25.0, (width, height))
            for frame in all_frames:
                out.write(frame)
            out.release()
            return {"status": "success", "output_path": request.output_path, "frames": len(all_frames)}
        else:
            raise HTTPException(status_code=500, detail="No frames generated")
            
    except Exception as e:
        print(f"API Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/generate_stream")
async def generate_video_stream(request: GenerateRequest):
    try:
        # Load models from app state
        tts_model = app.state.tts_model
        imtalker_streamer = app.state.imtalker_streamer
        
        # Load Avatar
        avatar_path = os.path.join(current_dir, request.avatar)
        if not os.path.exists(avatar_path):
            avatar_path = request.avatar
            
        if not os.path.exists(avatar_path):
            avatar_img = Image.new('RGB', (512, 512), color='red')
            print(f"API Stream: Avatar not found, using dummy: {avatar_path}")
        else:
            avatar_img = Image.open(avatar_path).convert('RGB')
            
        # Create TTS Stream
        tts_stream = tts_model.generate_stream(
            text=request.text,
            chunk_size=50,  # Larger chunks for less overhead
            print_metrics=False
        )
        audio_iterator = chatterbox_adapter(tts_stream)
        
        async def frame_generator():
            # Use a queue to overlap rendering and transmission
            queue = asyncio.Queue(maxsize=10)
            
            # To sync audio/video, we need to send the audio chunk alongside the frames
            # Chatterbox yields (audio, metrics), imtalker yields (frames, metrics)
            current_audio = [None]
            def wrapped_audio_iterator():
                for chunk in audio_iterator:
                    current_audio[0] = chunk
                    yield chunk

            # Producer task: runs the renderer and puts payloads in the queue
            async def producer():
                try:
                    # Run IMTalker in a thread pool since it's CPU/GPU intensive
                    loop = asyncio.get_event_loop()
                    def run_rendering():
                        for video_frames, metrics in imtalker_streamer.generate_stream(avatar_img, wrapped_audio_iterator()):
                            # Prepare frames as base64
                            b64_frames = []
                            for frame_tensor in video_frames:
                                frame_bgr = process_frame_tensor(frame_tensor)
                                ret, buffer = cv2.imencode('.jpg', frame_bgr)
                                if ret:
                                    b64_frames.append(base64.b64encode(buffer).decode('utf-8'))
                            
                            # Prepare audio as base64 WAV
                            audio_b64 = ""
                            if current_audio[0] is not None:
                                audio_b64 = numpy_to_wav_base64(current_audio[0])
                            
                            payload = {
                                "audio": audio_b64,
                                "frames": b64_frames,
                                "metrics": metrics
                            }
                            # Put payload into queue (block if full)
                            # Since we are in a thread, we use run_coroutine_threadsafe or similar?
                            # Actually, we can just yield symbols from imtalker_streamer and process them in the producer asyncly.
                            # But imtalker_streamer.generate_stream is a blocking generator.
                            return (payload, True) # This is not ideal for multiple yields.
                    
                    # Better producer logic:
                    def render_loop():
                        for video_frames, metrics in imtalker_streamer.generate_stream(avatar_img, wrapped_audio_iterator()):
                            b64_frames = []
                            for frame_tensor in video_frames:
                                frame_bgr = process_frame_tensor(frame_tensor)
                                ret, buffer = cv2.imencode('.jpg', frame_bgr)
                                if ret:
                                    b64_frames.append(base64.b64encode(buffer).decode('utf-8'))
                            
                            audio_b64 = ""
                            if current_audio[0] is not None:
                                audio_b64 = numpy_to_wav_base64(current_audio[0])
                            
                            payload = {
                                "audio": audio_b64,
                                "frames": b64_frames,
                                "metrics": metrics
                            }
                            # Send payload back to async world
                            asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
                        
                        # Signal end of stream
                        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

                    await loop.run_in_executor(None, render_loop)
                except Exception as e:
                    print(f"Producer Error: {e}")
                    traceback.print_exc()
                    await queue.put(None)

            # Start the producer
            producer_task = asyncio.create_task(producer())

            # Consumer: yield from the queue
            try:
                while True:
                    payload = await queue.get()
                    if payload is None:
                        break
                    yield json.dumps(payload) + "\n"
            finally:
                if not producer_task.done():
                    producer_task.cancel()

        return StreamingResponse(frame_generator(), media_type="application/x-ndjson")
            
    except Exception as e:
        print(f"API Stream Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_stream_get")
async def generate_video_stream_get(text: str, avatar: str = "assets/source_1.png"):
    # This is a GET wrapper for easier integration with <img> tags
    try:
        # Load models from app state
        tts_model = app.state.tts_model
        imtalker_streamer = app.state.imtalker_streamer
        
        # Load Avatar
        avatar_path = os.path.join(current_dir, avatar)
        if not os.path.exists(avatar_path):
            avatar_path = avatar
            
        if not os.path.exists(avatar_path):
            avatar_img = Image.new('RGB', (512, 512), color='red')
            print(f"API Stream (GET): Avatar not found, using dummy: {avatar_path}")
        else:
            avatar_img = Image.open(avatar_path).convert('RGB')
            
        # Create TTS Stream
        tts_stream = tts_model.generate_stream(
            text=text,
            chunk_size=50,  # Larger chunks for less overhead
            print_metrics=False
        )
        audio_iterator = chatterbox_adapter(tts_stream)
        
        async def frame_generator():
            queue = asyncio.Queue(maxsize=10)
            current_audio = [None]
            def wrapped_audio_iterator():
                for chunk in audio_iterator:
                    current_audio[0] = chunk
                    yield chunk

            async def producer():
                try:
                    loop = asyncio.get_event_loop()
                    def render_loop():
                        for video_frames, metrics in imtalker_streamer.generate_stream(avatar_img, wrapped_audio_iterator()):
                            b64_frames = []
                            for frame_tensor in video_frames:
                                frame_bgr = process_frame_tensor(frame_tensor)
                                ret, buffer = cv2.imencode('.jpg', frame_bgr)
                                if ret:
                                    b64_frames.append(base64.b64encode(buffer).decode('utf-8'))
                            
                            audio_b64 = ""
                            if current_audio[0] is not None:
                                audio_b64 = numpy_to_wav_base64(current_audio[0])
                            
                            payload = {
                                "audio": audio_b64,
                                "frames": b64_frames,
                                "metrics": metrics
                            }
                            asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
                        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

                    await loop.run_in_executor(None, render_loop)
                except Exception as e:
                    print(f"Producer Gen Error: {e}")
                    await queue.put(None)

            producer_task = asyncio.create_task(producer())

            try:
                while True:
                    payload = await queue.get()
                    if payload is None:
                        break
                    yield json.dumps(payload) + "\n"
            finally:
                if not producer_task.done():
                    producer_task.cancel()

        return StreamingResponse(frame_generator(), media_type="application/x-ndjson")
            
    except Exception as e:
        print(f"API Stream (GET) Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def _cleanup_webrtc_session(session_id: str, reason: str = "unknown"):
    """Full cleanup of a WebRTC session: close PC, cleanup ConvManager, free GPU, clear state."""
    logger.info(f"🧹 Cleaning up session {session_id} (reason: {reason})")
    
    meta = app.state.active_session_meta.get(session_id)
    
    # 1. Cleanup ConversationManager (executors, buffers)
    if meta and meta.get("conv_manager"):
        try:
            meta["conv_manager"].cleanup()
        except Exception as e:
            logger.error(f"ConvManager cleanup error: {e}")
    
    # 2. Drain track queues to free memory
    if meta:
        for track_key in ("video_track", "audio_track"):
            track = meta.get(track_key)
            if track and hasattr(track, 'frame_queue'):
                while not track.frame_queue.empty():
                    try: track.frame_queue.get_nowait()
                    except: break
            if track and hasattr(track, 'audio_queue'):
                while not track.audio_queue.empty():
                    try: track.audio_queue.get_nowait()
                    except: break
    
    # 3. Close PeerConnection
    pc = app.state.webrtc_sessions.get(session_id)
    if pc:
        try:
            await pc.close()
        except Exception as e:
            logger.error(f"PC close error: {e}")
    
    # 4. Remove from all session stores
    app.state.webrtc_sessions.pop(session_id, None)
    app.state.active_session_meta.pop(session_id, None)
    
    # 5. Clear the active session slot
    if app.state.active_webrtc_session == session_id:
        app.state.active_webrtc_session = None
    
    # 6. Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()
    
    logger.info(f"✅ Session {session_id} fully cleaned up.")


@app.post("/offer")
async def webrtc_offer(request: Request):
    try:
        t_start = time.time()
        data = await request.json()
        logger.info(f"WebRTC Offer Received with keys: {list(data.keys())}")
        
        sdp = data.get("sdp")
        sdp_type = data.get("type", "offer")
        avatar = data.get("avatar", "assets/source_1.png")
        conv_mode = data.get("conv_mode", False)
        text = data.get("initial_text", data.get("text", "Hello, I am ready to talk."))
        greeting = data.get("greeting", data.get("initial_greeting", "Hi there! How are you doing today?"))
        reference_audio = data.get("reference_audio", None)
        system_prompt = data.get("system_prompt", None)
        crop = data.get("crop", True)
        force = data.get("force", False)  # Force-disconnect existing session
        
        logger.info(f"Params: conv_mode={conv_mode}, crop={crop}, greeting='{greeting[:40] if greeting else None}'")
        
        if not sdp:
            raise HTTPException(status_code=400, detail="SDP is required")

        # ─── Single-session guard: only ONE WebRTC session at a time ───
        async with app.state._session_lock:
            existing = app.state.active_webrtc_session
            if existing and existing in app.state.webrtc_sessions:
                if force:
                    logger.warning(f"⚠️ Force-disconnecting existing session {existing}")
                    await _cleanup_webrtc_session(existing, reason="force-replaced")
                else:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Another session is already active: {existing}. "
                               f"Send force=true to disconnect it, or POST /disconnect first."
                    )
        
        if not sdp:
            raise HTTPException(status_code=400, detail="SDP is required")

        # Initialize models
        tts_model = app.state.tts_model
        imtalker_streamer = app.state.imtalker_streamer
        openai_client = app.state.openai_client
        
        # Process Avatar (Path, URL or Base64)
        avatar_path = None
        t_avatar = time.time()
        if avatar:
            if avatar.startswith("http://") or avatar.startswith("https://"):
                logger.info(f"Downloading avatar from URL: {avatar}")
                avatar_path = await download_url_to_temp(avatar, prefix="avatar_url_", suffix=".png")
            elif avatar.startswith("data:image") or len(avatar) > 200: # Simple heuristic for Base64
                logger.info("Decoding Base64 avatar...")
                avatar_path = save_base64_to_temp(avatar, prefix="avatar_", suffix=".png")
            else:
                avatar_path = os.path.join(current_dir, avatar)
                if not os.path.exists(avatar_path):
                    avatar_path = avatar
        
        logger.info(f"Avatar processing time: {(time.time() - t_avatar)*1000:.2f}ms")

        if avatar_path and os.path.exists(avatar_path):
             logger.info(f"Using avatar from: {avatar_path}")
             avatar_img_raw = Image.open(avatar_path).convert('RGB')
             # Apply crop if requested
             if crop and hasattr(app.state, 'inference_agent'):
                 logger.info("Applying face crop to avatar...")
                 avatar_img = app.state.inference_agent.data_processor.process_img(avatar_img_raw)
             else:
                 avatar_img = avatar_img_raw.resize((512, 512))
        else:
             logger.warning(f"Avatar not found or invalid: {avatar[:50]}...")
             avatar_img = Image.new('RGB', (512, 512), color='red')

        # Process Reference Audio (Path, URL or Base64)
        t_audio = time.time()
        if reference_audio:
            if reference_audio.startswith("http://") or reference_audio.startswith("https://"):
                logger.info(f"Downloading reference audio from URL: {reference_audio}")
                reference_audio_path = await download_url_to_temp(reference_audio, prefix="refstart_url_", suffix=".wav")
                reference_audio = reference_audio_path
            elif len(reference_audio) > 200 or reference_audio.startswith("data:audio"): # Base64 heuristic
                logger.info("Decoding Base64 reference audio...")
                reference_audio_path = save_base64_to_temp(reference_audio, prefix="refstart_", suffix=".wav")
                reference_audio = reference_audio_path # Update variable to point to file
            else:
                 # It's a path
                 if not os.path.exists(reference_audio):
                     logger.warning(f"Reference audio path not found: {reference_audio}")
                     reference_audio = None
        logger.info(f"Reference audio processing time: {(time.time() - t_audio)*1000:.2f}ms")
        
        if reference_audio:
            logger.info(f"Using reference audio from: {reference_audio}")

        # Create PeerConnection with STUN
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            )
        )
        
        # Create shared A/V sync manager
        sync_manager = AVSyncManager(video_fps=25.0, audio_sample_rate=24000)
        
        # Get or render idle animation frames for instant realistic idle
        idle_frames = idle_animation_cache.get_idle_frames(avatar_img)
        
        # Create Tracks with shared sync and idle animation
        video_track = IMTalkerVideoTrack(avatar_img, sync_manager=sync_manager, idle_frames=idle_frames)
        audio_track = IMTalkerAudioTrack(sync_manager=sync_manager)
        
        pc.addTrack(video_track)
        pc.addTrack(audio_track)

        # Session ID + register as active
        session_id = str(uuid.uuid4())
        app.state.webrtc_sessions[session_id] = pc
        app.state.active_webrtc_session = session_id
        app.state.active_session_meta[session_id] = {
            "session_id": session_id,
            "status": "connecting",       # connecting → greeting → ready → disconnected
            "greeting_done": False,
            "conv_mode": conv_mode,
            "video_track": video_track,
            "audio_track": audio_track,
            "conv_manager": None,
            "created_at": time.time(),
        }
        
        # Kick off idle animation render in background if not cached yet
        if idle_frames is None and not idle_animation_cache.is_rendering(avatar_img):
            async def render_idle_bg():
                try:
                    frames = await asyncio.get_event_loop().run_in_executor(
                        None, idle_animation_cache.render_idle, avatar_img, imtalker_streamer
                    )
                    if frames:
                        video_track.set_idle_frames(frames)
                        logger.info(f"Idle animation ready for session {session_id} ({len(frames)} frames)")
                except Exception as e:
                    logger.error(f"Background idle render failed: {e}")
            asyncio.create_task(render_idle_bg())

        # Conversation Manager (optional)
        conv_manager = None
        if conv_mode:
            if not openai_client:
                raise HTTPException(status_code=400, detail="OpenAI API key not configured, conversational mode unavailable.")
            conv_manager = ConversationManager(
                openai_client, tts_model, imtalker_streamer, 
                video_track, audio_track, avatar_img,
                system_prompt=system_prompt,
                reference_audio=reference_audio,
                sync_manager=sync_manager
            )
            app.state.active_session_meta[session_id]["conv_manager"] = conv_manager

        @pc.on("track")
        def on_track(track):
            logger.info(f"📡 Track received: {track.kind}")
            if track.kind == "audio" and conv_manager:
                async def handle_mic():
                    print("🎤 Starting microphone consumer...")
                    try:
                        while True:
                            frame = await track.recv()
                            # WebRTC audio for OpenAI
                            pcm = frame.to_ndarray()
                            await conv_manager.add_microphone_audio(pcm, frame.sample_rate)
                    except Exception as e:
                        print(f"Mic Track Error: {e}")
                
                asyncio.create_task(handle_mic())

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            logger.info(f"Connection state for {session_id}: {state}")
            meta = app.state.active_session_meta.get(session_id, {})
            
            if state == "connected":
                meta["status"] = "greeting"
                
                # Push the PRE-RENDERED greeting (rendered during /offer processing)
                pairs = meta.get("pre_rendered_pairs")
                greeting_cm = meta.get("greeting_cm") or conv_manager
                
                if pairs and greeting_cm:
                    logger.info(f"🎙️ Pushing pre-rendered greeting ({len(pairs)} A/V pairs)...")
                    try:
                        await greeting_cm.push_buffered_greeting(pairs)
                        logger.info("✅ Greeting delivered (pre-rendered).")
                    except Exception as e:
                        logger.error(f"Greeting push failed: {e}")
                    
                    # Free the buffer
                    meta.pop("pre_rendered_pairs", None)
                    
                    # Cleanup temp CM if one-shot
                    temp_cm = meta.pop("greeting_cm", None)
                    if not conv_mode and temp_cm and temp_cm is not conv_manager:
                        temp_cm.cleanup()
                else:
                    logger.warning("No pre-rendered greeting available")
                
                meta["status"] = "ready"
                meta["greeting_done"] = True
                logger.info(f"📡 Session {session_id} is READY (greeting delivered).")

            elif state in ("failed", "closed", "disconnected"):
                await _cleanup_webrtc_session(session_id, reason=state)

        # ─── Pre-render greeting BEFORE returning SDP answer ───
        # This keeps the frontend on 'connecting' screen while TTS + render runs.
        # When WebRTC connects, the pre-rendered greeting plays INSTANTLY.
        pre_rendered_pairs = None
        greet_text = greeting or text
        if greet_text:
            t_prerender = time.time()
            logger.info(f"🎙️ Pre-rendering greeting: '{greet_text[:50]}...'")
            target_cm = conv_manager
            if not target_cm:
                # One-shot: create temp ConversationManager for rendering
                target_cm = ConversationManager(
                    openai_client, tts_model, imtalker_streamer, 
                    video_track, audio_track, avatar_img,
                    system_prompt=system_prompt,
                    reference_audio=reference_audio,
                    sync_manager=sync_manager
                )
            try:
                pre_rendered_pairs = await target_cm.pre_render_to_buffer(greet_text)
                logger.info(f"✅ Greeting pre-rendered: {len(pre_rendered_pairs)} pairs in {time.time() - t_prerender:.2f}s")
            except Exception as e:
                logger.error(f"Greeting pre-render failed: {e}")
                traceback.print_exc()
            
            # Store in session meta for the connected handler
            app.state.active_session_meta[session_id]["pre_rendered_pairs"] = pre_rendered_pairs
            app.state.active_session_meta[session_id]["greeting_cm"] = target_cm

        # Handle offer
        offer = RTCSessionDescription(sdp=sdp, type="offer") # default to offer
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        logger.info(f"Offer processing total time: {(time.time() - t_start)*1000:.2f}ms")

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id,
            "status": "connecting",
            "greeting": greeting if greeting else text,
            "conv_mode": conv_mode
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"WebRTC Offer Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ice")
async def webrtc_ice(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        candidate = data.get("candidate")
        
        if not session_id or session_id not in app.state.webrtc_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
            
        pc = app.state.webrtc_sessions[session_id]
        if candidate:
            print(f"ICE candidate received for {session_id}")
            # candidate looks like: {candidate: "...", sdpMid: "...", sdpMLineIndex: 0}
            cand = RTCIceCandidate(
                candidate=candidate["candidate"],
                sdpMid=candidate["sdpMid"],
                sdpMLineIndex=candidate["sdpMLineIndex"]
            )
            await pc.addIceCandidate(cand)
            
        return {"status": "ok"}
    except Exception as e:
        print(f"ICE Error: {e}")
        return {"status": "error", "detail": str(e)}


@app.post("/disconnect")
async def webrtc_disconnect(request: Request):
    """Explicitly disconnect a WebRTC session, freeing all resources.
    
    Body: {"session_id": "..."} or {} to disconnect the active session.
    """
    try:
        data = await request.json()
        session_id = data.get("session_id", app.state.active_webrtc_session)
        
        if not session_id:
            return {"status": "ok", "detail": "No active session to disconnect."}
        
        if session_id not in app.state.webrtc_sessions:
            # Already gone — just clear state
            app.state.active_session_meta.pop(session_id, None)
            if app.state.active_webrtc_session == session_id:
                app.state.active_webrtc_session = None
            return {"status": "ok", "detail": f"Session {session_id} already cleaned up."}
        
        await _cleanup_webrtc_session(session_id, reason="client-disconnect")
        
        return {"status": "ok", "session_id": session_id, "detail": "Session disconnected and cleaned up."}
    except Exception as e:
        logger.error(f"Disconnect error: {e}")
        return {"status": "error", "detail": str(e)}


@app.get("/session/status")
async def session_status(session_id: Optional[str] = None):
    """Check session status. Returns 'connecting', 'greeting', 'ready', or 'no_session'.
    
    The client should poll this after /offer until status='ready' (greeting finished).
    """
    sid = session_id or app.state.active_webrtc_session
    
    if not sid or sid not in app.state.active_session_meta:
        return {
            "status": "no_session",
            "active_session": None,
            "detail": "No active WebRTC session."
        }
    
    meta = app.state.active_session_meta[sid]
    return {
        "session_id": sid,
        "status": meta.get("status", "unknown"),
        "greeting_done": meta.get("greeting_done", False),
        "conv_mode": meta.get("conv_mode", False),
        "uptime_s": round(time.time() - meta.get("created_at", time.time()), 1),
    }

# --- Original CLI Logic (Modified to use models if available) ---

def chatterbox_adapter(tts_stream):
    """
    Adapts Chatterbox stream (audio_chunk, metrics) to audio_chunk iterator.
    """
    for audio_chunk, metrics in tts_stream:
        # audio_chunk is [1, T] tensor
        yield audio_chunk

def save_video(frames, fps, output_path):
    if not frames:
        print("No frames to save.")
        return
        
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # IMTalker returns RGB [3, H, W] tensors or [C,H,W] numpy?
        # imtalker_streamer yields list of tensor [1, 3, H, W]
        # Let's inspect what we get.
        # imtalker_streamer implementation:
        # out_frame = self.renderer.decode(...) -> [1, 3, H, W]
        # yields frames_for_chunk = [out_frame1, out_frame2]
        
        # We need to process the frames
        pass 
    out.release()
    
def process_frame_tensor(frame_tensor, format="BGR"):
    # frame_tensor: [1, 3, H, W] or [3, H, W]
    if frame_tensor.dim() == 4:
        frame_tensor = frame_tensor.squeeze(0)
    
    # [3, H, W] -> [H, W, 3] -> [0, 255] uint8
    # Fast path: model outputs [0, 1] range with FP32 (no NaN/range check needed)
    frame_np = (frame_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    if format == "RGB":
        return frame_np
    
    # RGB to BGR for OpenCV
    return cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description="Text to Video Streaming with Chatterbox and IMTalker")
    parser.add_argument("--text", type=str, default="Hello, this is a streaming test.", help="Text to speak")
    parser.add_argument("--avatar", type=str, default="assets/source_1.png", help="Path to avatar image")
    parser.add_argument("--output", type=str, default="output_stream.mp4", help="Output video path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print(f"Initializing on {args.device}...")
    
    # 1. Initialize Chatterbox TTS
    print("Loading Chatterbox TTS...")
    tts_model = ChatterboxTTS.from_pretrained(device=args.device)
    
    # 2. Initialize IMTalker
    print("Loading IMTalker...")
    imtalker_cfg = IMTalkerConfig()
    imtalker_cfg.device = args.device
    imtalker_streamer = IMTalkerStreamer(imtalker_cfg, device=args.device)
    
    # Load Avatar
    avatar_path = os.path.join(current_dir, args.avatar)
    if not os.path.exists(avatar_path):
        # try relative to script execution
        avatar_path = args.avatar
        
    if not os.path.exists(avatar_path):
        print(f"Avatar not found: {avatar_path}")
        # dummy
        print("Using dummy red avatar")
        avatar_img = Image.new('RGB', (512, 512), color='red')
    else:
        print(f"Using avatar: {avatar_path}")
        avatar_img = Image.open(avatar_path).convert('RGB')
        
    print(f"\nProcessing Text: '{args.text}'")
    
    # 3. Create TTS Stream Iterator
    # Chatterbox stream yields (audio_chunk, metrics)
    tts_stream = tts_model.generate_stream(
        text=args.text,
        chunk_size=20, # tokens
        print_metrics=False
    )
    
    audio_iterator = chatterbox_adapter(tts_stream)
    
    # 4. Stream Video
    print("Starting Video Streaming...")
    
    all_frames = []
    total_latency_start = time.time()
    
    try:
        chunk_idx = 0
        for video_frames, metrics in imtalker_streamer.generate_stream(avatar_img, audio_iterator):
            chunk_idx += 1
            print(f"Chunk {chunk_idx}: Generated {len(video_frames)} frames. "
                  f"RTF: {metrics['rtf']:.2f}")
            
            # Process frames for saving
            for frame_tensor in video_frames:
                frame_bgr = process_frame_tensor(frame_tensor)
                all_frames.append(frame_bgr)
                
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()
        
    total_time = time.time() - total_latency_start
    print(f"\nStreaming Finished in {total_time:.2f}s")
    print(f"Total Video Frames: {len(all_frames)}")
    
    # 5. Save Video
    if all_frames:
        print(f"Saving video to {args.output}...")
        height, width, _ = all_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, 25.0, (width, height))
        for frame in all_frames:
            out.write(frame)
        out.release()
        print("Done.")
    else:
        print("No video generated.")

# ============================================================================
# API ENDPOINTS (HLS, CHUNKED, GENERAL INFERENCE)
# ============================================================================

@app.post("/crop")
async def crop_image(
    file: UploadFile = File(..., description="Image file to crop"),
    crop_scale: float = Form(0.8, description="Crop scale factor (0.5=tight, 0.8=default, 1.2=loose)", ge=0.3, le=2.0),
    output_size: Optional[int] = Form(None, description="Output size in pixels (square)", ge=64, le=2048),
    format: str = Form("jpeg", description="Output format (jpeg, png, webp)")
):
    """
    Crop image centered on detected face with configurable padding.
    """
    if not hasattr(app.state, 'inference_agent'):
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate format
    format = format.lower()
    if format not in ["jpeg", "jpg", "png", "webp"]:
        raise HTTPException(status_code=400, detail="Invalid format. Use jpeg, png, or webp")
    
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Use InferenceAgent's DataProcessor
        dp = app.state.inference_agent.data_processor
        
        # Temporarily set crop_scale on the processor
        original_crop_scale = dp.crop_scale
        dp.crop_scale = crop_scale
        
        try:
            # Process image (crop around face)
            cropped_img = dp.process_img(img)
            
            # Resize if output_size specified
            if output_size:
                cropped_img = cropped_img.resize((output_size, output_size), Image.LANCZOS)
        finally:
            # Restore original crop_scale
            dp.crop_scale = original_crop_scale
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        if format in ["jpeg", "jpg"]:
            cropped_img.save(img_byte_arr, format='JPEG', quality=95)
            media_type = "image/jpeg"
        elif format == "png":
            cropped_img.save(img_byte_arr, format='PNG')
            media_type = "image/png"
        elif format == "webp":
            cropped_img.save(img_byte_arr, format='WEBP', quality=95)
            media_type = "image/webp"
        else:
            # Fallback (should be caught by validation)
            cropped_img.save(img_byte_arr, format='JPEG', quality=95)
            media_type = "image/jpeg"
            
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type=media_type)
        
    except Exception as e:
        logger.error(f"Crop operation failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Crop operation failed: {str(e)}")

@app.post("/api/audio-driven/stream")
async def audio_driven_stream_hls(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    crop: bool = Form(True),
    seed: int = Form(42),
    nfe: int = Form(10),
    cfg_scale: float = Form(3.0),
    session_id: Optional[str] = Form(None)
):
    """Start or resume a progressive HLS streaming session"""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    # Check if session already exists
    if session_id in app.state.hls_sessions:
        session = app.state.hls_sessions[session_id]
        if not session.is_complete:
            return JSONResponse({
                "session_id": session_id,
                "status": "already_running",
                "playlist_url": f"/hls/{session_id}/playlist.m3u8"
            })
    
    # Save files
    temp_dir = Path(tempfile.mkdtemp())
    img_path = temp_dir / "source.png"
    aud_path = temp_dir / "audio.wav"
    
    with img_path.open("wb") as f:
        f.write(await image.read())
    with aud_path.open("wb") as f:
        f.write(await audio.read())
        
    img_pil = Image.open(img_path).convert("RGB")
    
    # Initialize HLS session
    hls_gen = ProgressiveHLSGenerator(session_id)
    app.state.hls_sessions[session_id] = HLSSession(
        session_id=session_id,
        temp_dir=str(temp_dir),
        target_duration=0,
        is_complete=False
    )
    
    # Start inference in background
    background_tasks.add_task(
        app.state.inference_agent.run_audio_inference_progressive,
        img_pil=img_pil,
        aud_path=str(aud_path),
        crop=crop,
        seed=seed,
        nfe=nfe,
        cfg_scale=cfg_scale,
        hls_generator=hls_gen,
        is_final_chunk=True
    )
    
    return {
        "session_id": session_id,
        "status": "started",
        "playlist_url": f"/hls/{session_id}/playlist.m3u8"
    }

@app.get("/api/status/{session_id}")
async def get_session_status(session_id: str):
    """Check status of an HLS generation session"""
    if session_id not in app.state.hls_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = app.state.hls_sessions[session_id]
    hls_dir = HLS_OUTPUT_DIR / session_id
    playlist_ready = (hls_dir / "playlist.m3u8").exists()
    
    return {
        "session_id": session_id,
        "is_complete": session.is_complete,
        "playlist_ready": playlist_ready,
        "playlist_url": f"/hls/{session_id}/playlist.m3u8" if playlist_ready else None
    }

@app.post("/api/session/cleanup")
async def cleanup_sessions(session_id: Optional[str] = None):
    """Cleanup temporary files for one or all sessions"""
    if session_id:
        sessions_to_clean = [session_id] if session_id in app.state.hls_sessions else []
    else:
        sessions_to_clean = list(app.state.hls_sessions.keys())
        
    cleaned = []
    for sid in sessions_to_clean:
        session = app.state.hls_sessions[sid]
        if os.path.exists(session.temp_dir):
            shutil.rmtree(session.temp_dir)
        # We keep HLS files for now but could clean them too
        cleaned.append(sid)
        del app.state.hls_sessions[sid]
        
    return {"status": "success", "cleaned_sessions": cleaned}

@app.get("/api/hls/{session_id}/playlist.m3u8")
async def get_playlist(session_id: str):
    """
    Get HLS playlist file for a session.
    Waits for the first segment to be ready before returning the playlist.
    """
    playlist_path = HLS_OUTPUT_DIR / session_id / "playlist.m3u8"
    segment_path = HLS_OUTPUT_DIR / session_id / "segment000.ts"

    if not playlist_path.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    # Wait up to 60 seconds for first segment and valid playlist
    max_wait_seconds = 60
    check_interval = 0.1
    elapsed = 0

    while elapsed < max_wait_seconds:
        if not segment_path.exists():
            await asyncio.sleep(check_interval)
            elapsed += check_interval
            continue

        try:
            with open(playlist_path, 'r') as f:
                content = f.read()
                if '#EXTINF' in content:
                    return FileResponse(
                        playlist_path,
                        media_type="application/vnd.apple.mpegurl",
                        headers={"Cache-Control": "no-cache"},
                    )
        except Exception:
            pass

        await asyncio.sleep(check_interval)
        elapsed += check_interval

    raise HTTPException(status_code=202, detail="Playlist still waiting for first segment")

@app.get("/api/hls/{session_id}/segment{segment_num}.ts")
async def get_segment(session_id: str, segment_num: str, request: Request):
    segment_path = HLS_OUTPUT_DIR / session_id / f"segment{segment_num}.ts"
    if not segment_path.exists():
        raise HTTPException(status_code=404, detail="Segment not found")

    file_size = os.path.getsize(segment_path)
    range_header = request.headers.get("range")

    if not range_header:
        raise HTTPException(status_code=416, detail="Range header required")

    try:
        start, end = range_header.replace("bytes=", "").split("-")
        start = int(start)
        end = int(end) if end else file_size - 1
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid range format")

    def iterfile():
        with open(segment_path, "rb") as f:
            f.seek(start)
            yield f.read(end - start + 1)

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Cache-Control": "no-cache",
        "Content-Type": "video/mp2t",
    }
    return StreamingResponse(iterfile(), status_code=206, headers=headers)

@app.post("/api/audio-driven/chunked/init")
async def chunked_stream_init(
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    crop: bool = Form(True),
    seed: int = Form(42),
    nfe: int = Form(10),
    cfg_scale: float = Form(3.0),
    chunk_duration: float = Form(2.0),
    pose_style: Optional[str] = Form(None),
    gaze_style: Optional[str] = Form(None),
):
    sid = session_id or str(uuid.uuid4())
    img_data = await image.read()
    img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
    
    buffer = AudioChunkBuffer(session_id=sid, chunk_duration=chunk_duration)
    hls_gen = ProgressiveHLSGenerator(session_id=sid, segment_duration=int(chunk_duration))
    
    pose_arr = json.loads(pose_style) if pose_style else None
    gaze_arr = json.loads(gaze_style) if gaze_style else None
    
    session = ChunkedStreamSession(
        session_id=sid,
        image_pil=img_pil,
        audio_buffer=buffer,
        hls_generator=hls_gen,
        parameters={
            "crop": crop, "seed": seed, "nfe": nfe, "cfg_scale": cfg_scale,
            "pose_style": pose_arr, "gaze_style": gaze_arr
        }
    )
    app.state.chunked_sessions[sid] = session
    return {
        "status": "success",
        "session_id": sid,
        "hls_url": f"/api/hls/{sid}/playlist.m3u8"
    }

@app.post("/api/audio-driven/chunked/feed")
async def audio_driven_chunked_feed(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    audio_chunk: UploadFile = File(...),
    is_final: bool = Form(False),
):
    """Feed an audio chunk to a real-time streaming session"""
    # 1. Get or create session
    if session_id not in app.state.chunked_sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call init first.")
    session = app.state.chunked_sessions[session_id]
        
    # 2. Add audio chunk to buffer
    audio_bytes = await audio_chunk.read()
    # Assume 16-bit PCM if not specified, but let's try to detect if it's a wav file
    if audio_chunk.filename.endswith(".wav"):
        with io.BytesIO(audio_bytes) as b:
            import wave
            with wave.open(b, 'rb') as w:
                sr = w.getframerate()
                # Read all frames and convert to numpy
                frames = w.readframes(w.getnframes())
                audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        # Generic float32 or int16
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        sr = 16000 # Default
        
    has_enough = session.audio_buffer.add_chunk(audio_np, source_sr=sr)
    
    # 3. Trigger inference if enough audio or final
    if has_enough or is_final:
        # Check if already processing
        with session.lock:
            if session.status == "processing":
                # Already running, audio added to buffer
                return {"session_id": session_id, "status": "audio_buffered"}
            
            session.status = "processing"
            
        async def process_chunks():
            try:
                while True:
                    chunk = session.audio_buffer.get_chunk()
                    if chunk is None:
                        if is_final:
                            final_chunk = session.audio_buffer.finalize()
                            if final_chunk is not None:
                                await asyncio.to_thread(
                                    hls_generator=session.hls_generator,
                                    pose_style=session.parameters.get("pose_style"),
                                    gaze_style=session.parameters.get("gaze_style")
                                )
                            session.hls_generator.finalize(mark_complete=True)
                            session.status = "complete"
                        break
                    
                    # Run inference on chunk
                    await asyncio.to_thread(
                        app.state.inference_agent.run_audio_inference_chunked,
                        img_pil=session.image_pil,
                        audio_chunk_np=chunk,
                        crop=session.parameters["crop"],
                        seed=session.parameters["seed"],
                        nfe=session.parameters["nfe"],
                        cfg_scale=session.parameters["cfg_scale"],
                        hls_generator=session.hls_generator,
                        pose_style=session.parameters.get("pose_style"),
                        gaze_style=session.parameters.get("gaze_style")
                    )
            except Exception as e:
                session.status = "error"
                session.error_message = str(e)
                traceback.print_exc()
            finally:
                if session.status == "processing":
                    session.status = "waiting_for_audio"
        
        background_tasks.add_task(process_chunks)
        
    return {
        "session_id": session_id,
        "status": session.status,
        "playlist_url": f"/hls/{session_id}/playlist.m3u8"
    }

@app.get("/api/chunked/status/{session_id}")
async def get_chunked_status(session_id: str):
    if session_id not in app.state.chunked_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = app.state.chunked_sessions[session_id]
    return {
        "session_id": session_id,
        "status": session.status,
        "error_message": session.error_message,
        "total_frames": session.total_frames_generated,
        "created_at": session.created_at
    }

@app.post("/api/audio-driven")
async def api_audio_driven(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    crop: bool = Form(True),
    seed: int = Form(42),
    nfe: int = Form(10),
    cfg_scale: float = Form(2.0),
    crop_scale: float = Form(0.8),
    pose_style: Optional[str] = Form(None),
    gaze_style: Optional[str] = Form(None)
):
    temp_dir = Path(tempfile.mkdtemp())
    try:
        img_path = temp_dir / "source.png"
        aud_path = temp_dir / "audio.wav"
        with img_path.open("wb") as f: f.write(await image.read())
        with aud_path.open("wb") as f: f.write(await audio.read())
        
        img_pil = Image.open(img_path).convert("RGB")
        
        dp = app.state.inference_agent.data_processor
        original_crop_scale = dp.crop_scale
        dp.crop_scale = crop_scale
        
        pose_arr = json.loads(pose_style) if pose_style else None
        gaze_arr = json.loads(gaze_style) if gaze_style else None
        
        try:
            out_path = app.state.inference_agent.run_audio_inference(
                img_pil, str(aud_path), crop, seed, nfe, cfg_scale,
                pose_style=pose_arr, gaze_style=gaze_arr
            )
        finally:
            dp.crop_scale = original_crop_scale
            
        return FileResponse(out_path, media_type="video/mp4")
    finally:
        shutil.rmtree(temp_dir)

@app.post("/api/video-driven")
async def api_video_driven(
    source_image: UploadFile = File(...),
    driving_video: UploadFile = File(...),
    crop: bool = Form(True),
    crop_scale: float = Form(0.8) # Added to match VideoService
):
    temp_dir = Path(tempfile.mkdtemp())
    try:
        img_path = temp_dir / "source.png"
        vid_path = temp_dir / "driving.mp4"
        with img_path.open("wb") as f: f.write(await source_image.read())
        with vid_path.open("wb") as f: f.write(await driving_video.read())
        
        img_pil = Image.open(img_path).convert("RGB")
        
        # Apply crop_scale to DataProcessor
        dp = app.state.inference_agent.data_processor
        original_crop_scale = dp.crop_scale
        dp.crop_scale = crop_scale
        
        try:
            out_path = app.state.inference_agent.run_video_inference(
                img_pil, str(vid_path), crop
            )
        finally:
            dp.crop_scale = original_crop_scale
            
        return FileResponse(out_path, media_type="video/mp4")
    finally:
        shutil.rmtree(temp_dir)

@app.get("/api/hls/{session_id}/download")
async def download_full_video(session_id: str):
    """Download complete video by merging HLS segments"""
    session_dir = HLS_OUTPUT_DIR / session_id
    playlist_path = session_dir / "playlist.m3u8"
    
    if not session_dir.exists() or not playlist_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    output_video = session_dir / "complete_video.mp4"
    if output_video.exists():
        return FileResponse(output_video, media_type="video/mp4", filename=f"{session_id}.mp4")
    
    try:
        cmd = [
            "ffmpeg", "-i", str(playlist_path),
            "-c", "copy", "-bsf:a", "aac_adtstoasc", "-y",
            str(output_video)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg merge failed: {result.stderr}")
        
        return FileResponse(output_video, media_type="video/mp4", filename=f"{session_id}.mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/realtime-sse")
async def realtime_sse(
    session_id: str,
    image_base64: Optional[str] = None,
    audio_path: Optional[str] = None,
    crop: bool = True,
    seed: int = 42,
    nfe: int = 10,
    cfg_scale: float = 3.0
):
    """Real-time streaming via Server-Sent Events (SSE)"""
    if not image_base64 or not audio_path:
        raise HTTPException(status_code=400, detail="Missing image or audio")
    
    img_data = base64.b64decode(image_base64)
    img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
    
    async def event_generator():
        # This is a simplified version, in reality we'd use the parallel renderer stream
        # However, run_audio_inference_streaming is also available
        for frame_tensor in app.state.inference_agent.run_audio_inference_streaming(
            img_pil, audio_path, crop, seed, nfe, cfg_scale
        ):
            # Convert to JPG
            frame_np = (np.clip(frame_tensor.permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', frame_bgr)
            frame_base64 = base64.b64encode(buffer).decode()
            yield f"data: {frame_base64}\n\n"
            await asyncio.sleep(0.01) # Small sleep to avoid blocking
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/realtime-mjpeg")
async def realtime_mjpeg(
    session_id: str,
    audio_path: Optional[str] = None,
    crop: bool = True,
    seed: int = 42,
    nfe: int = 10,
    cfg_scale: float = 3.0
):
    """Real-time streaming via MJPEG (standard browser <img> tag support)"""
    # Assuming image is already set in session or something
    # For simplicity, MJPEG often used with persistent sessions
    if session_id not in app.state.chunked_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = app.state.chunked_sessions[session_id]
    
    async def mjpeg_generator():
        # Dummy example, real MJPEG would stream from a queue or renderer
        while True:
            # Yield boundaries
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'JPEG_DATA' + b'\r\n'
            await asyncio.sleep(0.04)
            
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/realtime/{session_id}")
async def realtime_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time commands or audio chunks via WS
            await websocket.send_text(f"Processed command for {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.get("/api/outputs")
async def list_outputs():
    """List all generated video files"""
    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_files = glob.glob(str(output_dir / "*.mp4"))
    results = []
    for f in video_files:
        path = Path(f)
        results.append({
            "filename": path.name,
            "size": path.stat().st_size,
            "created_at": path.stat().st_mtime
        })
    return sorted(results, key=lambda x: x["created_at"], reverse=True)

@app.get("/api/outputs/{filename}")
async def get_output_file(filename: str):
    """Retrieve a specific generated video file"""
    file_path = Path("./outputs") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4")

@app.post("/api/tts/generate")
async def api_tts_generate(
    text: str = Form(...),
    reference_audio: Optional[UploadFile] = File(None),
    exaggeration: float = Form(0.5),
    temperature: float = Form(0.8),
    seed: int = Form(0),
    diffusion_steps: int = Form(10),
    min_p: float = Form(0.05),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.2)
):
    """
    Generate speech from text with optional voice cloning.
    Splits long text into segments to avoid CUDA positional encoding overflow.
    """
    try:
        # 1. Prepare conditionals if reference audio is provided
        if reference_audio:
            temp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                content = await reference_audio.read()
                temp_ref.write(content)
                temp_ref.close()
                
                await asyncio.to_thread(
                    app.state.tts_model.prepare_conditionals,
                    wav_fpath=temp_ref.name,
                    exaggeration=exaggeration
                )
            finally:
                if os.path.exists(temp_ref.name):
                    os.remove(temp_ref.name)
        
        # 2. Split long text into safe segments to avoid positional encoding overflow
        # Chatterbox's positional encoding table has a fixed size; long texts cause
        # CUDA indexSelectLargeIndex assertion failures.
        MAX_SEGMENT_CHARS = 200  # ~8s of speech, safe for positional encoding
        text_segments = split_text_semantically(text, min_chunk_length=20, max_chunk_length=MAX_SEGMENT_CHARS)
        logger.info(f"TTS API: Generating {len(text_segments)} segments from {len(text)} chars")
        
        # 3. Generate audio for each segment and concatenate
        all_audio = []
        
        def generate_segments():
            for i, segment in enumerate(text_segments):
                logger.info(f"TTS API: Segment {i+1}/{len(text_segments)} ({len(segment)} chars): '{segment[:50]}...'")
                audio_tensor = app.state.tts_model.generate(
                    text=segment,
                    exaggeration=exaggeration,
                    temperature=temperature
                )
                audio_np = audio_tensor.squeeze(0).cpu().numpy()
                all_audio.append(audio_np)
                del audio_tensor
                # Free GPU memory between segments to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return np.concatenate(all_audio, axis=-1)
        
        audio_np = await asyncio.to_thread(generate_segments)
        
        # 4. Convert to WAV bytes
        sr = getattr(app.state.tts_model, 'sr', 24000)
        
        byte_io = io.BytesIO()
        with wave.open(byte_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sr)
            
            # Normalize and convert to int16
            peak = np.max(np.abs(audio_np))
            if peak > 1.0:
                audio_np = audio_np / peak
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        del audio_np, all_audio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        byte_io.seek(0)
        return StreamingResponse(byte_io, media_type="audio/wav")
        
    except Exception as e:
        logger.error(f"TTS Generation failed: {str(e)}")
        traceback.print_exc()
        # Clean up GPU memory on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Check if we should run the CLI test or the server
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Remove --test from sys.argv so argparse doesn't complain
        sys.argv.pop(1)
        main()
    else:
        print("Starting IMTalker Unified Server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
