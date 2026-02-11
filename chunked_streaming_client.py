"""
Real-Time Chunked Audio-to-Video Streaming Client for IMTalker FastAPI Server

This client demonstrates how to use the new chunked streaming endpoints to create
real-time video from continuously fed audio chunks (e.g., from WebRTC, microphone, etc.).

Usage:
    python chunked_streaming_client.py --image face.jpg --audio audio.wav
"""

import requests
import argparse
import time
import json
from pathlib import Path
import numpy as np
import librosa


class ChunkedStreamingClient:
    """Client for real-time chunked audio-to-video streaming"""
    
    def __init__(self, server_url: str = "http://localhost:9999"):
        self.server_url = server_url
        self.session_id = None
        
    def init_session(self, 
                     image_path: str,
                     crop: bool = True,
                     seed: int = 42,
                     nfe: int = 10,
                     cfg_scale: float = 2.0,
                     chunk_duration: float = 1.0,
                     audio_sample_rate: int = 24000) -> str:
        """
        Initialize a new chunked streaming session.
        
        Args:
            image_path: Path to source face image
            crop: Whether to auto-crop face
            seed: Random seed
            nfe: Number of function evaluations (5-50)
            cfg_scale: Classifier-free guidance scale (1.0-5.0)
            chunk_duration: Duration of each audio chunk to process (seconds)
            audio_sample_rate: Sample rate of incoming audio (Hz)
            
        Returns:
            session_id: Use this for feeding audio chunks
        """
        print(f"[Client] Initializing session with image: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'crop': crop,
                'seed': seed,
                'nfe': nfe,
                'cfg_scale': cfg_scale,
                'chunk_duration': chunk_duration,
                'audio_sample_rate': audio_sample_rate,
            }
            
            response = requests.post(
                f"{self.server_url}/api/audio-driven/chunked/init",
                files=files,
                data=data
            )
        
        if response.status_code != 200:
            raise Exception(f"Init failed: {response.status_code} - {response.text}")
        
        result = response.json()
        self.session_id = result['session_id']
        
        print(f"[Client] ✓ Session initialized: {self.session_id}")
        print(f"[Client] HLS URL: {result['hls_url']}")
        print(f"[Client] Stream audio chunks to feed frames in real-time")
        
        return self.session_id
    
    def feed_audio_chunk(self, audio_data: bytes, is_final: bool = False) -> dict:
        """
        Feed an audio chunk to the session.
        
        Args:
            audio_data: Audio chunk as bytes (WAV, MP3, etc.)
            is_final: Set to True for the last chunk
            
        Returns:
            Status dict with generation progress
        """
        if not self.session_id:
            raise ValueError("Session not initialized. Call init_session() first.")
        
        files = {'audio_chunk': ('chunk.wav', audio_data, 'audio/wav')}
        data = {
            'session_id': self.session_id,
            'is_final_chunk': is_final,
        }
        
        response = requests.post(
            f"{self.server_url}/api/audio-driven/chunked/feed",
            files=files,
            data=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Feed failed: {response.status_code} - {response.text}")
        
        result = response.json()
        return result
    
    def get_status(self) -> dict:
        """Get current session status"""
        if not self.session_id:
            raise ValueError("Session not initialized")
        
        response = requests.get(
            f"{self.server_url}/api/audio-driven/chunked/status/{self.session_id}"
        )
        
        if response.status_code != 200:
            raise Exception(f"Status query failed: {response.status_code}")
        
        return response.json()
    
    def cleanup(self):
        """Cleanup session and remove HLS files"""
        if not self.session_id:
            return
        
        response = requests.delete(
            f"{self.server_url}/api/audio-driven/chunked/session/{self.session_id}"
        )
        
        if response.status_code == 200:
            print(f"[Client] ✓ Session {self.session_id} cleaned up")
        else:
            print(f"[Client] Warning: Cleanup failed: {response.text}")


def example_full_audio_file(server_url: str, image_path: str, audio_path: str):
    """
    Example: Stream a complete audio file as chunks (useful for testing).
    
    This shows how to break a full audio file into chunks and stream them
    one at a time, simulating real-time audio input.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Streaming a complete audio file as chunks")
    print("="*70)
    
    client = ChunkedStreamingClient(server_url)
    
    # Initialize session
    session_id = client.init_session(
        image_path=image_path,
        chunk_duration=1.0,  # 1 second chunks
        audio_sample_rate=24000,
    )
    
    print(f"\n[Example] Loading audio file: {audio_path}")
    audio_data, sr = librosa.load(audio_path, sr=24000)
    print(f"[Example] Audio: {len(audio_data)} samples at {sr}Hz = {len(audio_data)/sr:.2f}s")
    
    # Break into 1-second chunks and send
    chunk_samples = int(sr * 1.0)  # 1 second at 24kHz = 24000 samples
    total_chunks = (len(audio_data) + chunk_samples - 1) // chunk_samples
    
    print(f"[Example] Streaming {total_chunks} chunks of {chunk_samples} samples each\n")
    
    for i in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[i:i+chunk_samples]
        is_final = (i + chunk_samples) >= len(audio_data)
        
        # Convert to WAV bytes
        import io
        import soundfile as sf
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, chunk, sr, format='WAV')
        wav_bytes = wav_buffer.getvalue()
        
        # Send chunk
        print(f"[Example] Chunk {i//chunk_samples + 1}/{total_chunks} "
              f"({len(chunk)} samples, final={is_final})")
        
        result = client.feed_audio_chunk(wav_bytes, is_final=is_final)
        
        print(f"  Status: {result['session_status']}")
        print(f"  Frames: {result['frames_generated']}")
        print(f"  Buffer: {result['buffer_samples']} samples\n")
        
        # Small delay to simulate streaming
        time.sleep(0.1)
    
    # Check final status
    print("\n[Example] Streaming complete. Final status:")
    status = client.get_status()
    print(f"  Total frames: {status['total_frames_generated']}")
    print(f"  Status: {status['status']}")
    
    print(f"\n[Example] ✓ View stream at: http://localhost:9999/hls/{session_id}/playlist.m3u8")


def example_realtime_microphone(server_url: str, image_path: str):
    """
    Example: Stream from microphone in real-time.
    
    Requires: pip install sounddevice
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("Error: sounddevice not installed. Install with: pip install sounddevice")
        return
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Real-time microphone streaming")
    print("="*70)
    
    client = ChunkedStreamingClient(server_url)
    
    # Initialize session
    session_id = client.init_session(
        image_path=image_path,
        chunk_duration=0.5,  # 500ms chunks for low latency
        audio_sample_rate=16000,  # Microphone default
    )
    
    print(f"\n[Example] Recording from microphone for 10 seconds...")
    print(f"[Example] Stream available at: http://localhost:9999/hls/{session_id}/playlist.m3u8")
    
    sr = 16000
    chunk_samples = int(sr * 0.5)  # 500ms chunks
    duration_seconds = 10
    total_chunks = int(duration_seconds / 0.5)
    
    try:
        for i in range(total_chunks):
            # Record chunk from microphone
            audio_chunk = sd.rec(chunk_samples, samplerate=sr, channels=1, dtype='float32')
            sd.wait()
            audio_chunk = audio_chunk.flatten()
            
            is_final = (i == total_chunks - 1)
            
            # Convert to WAV bytes
            import io
            import soundfile as sf
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_chunk, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()
            
            # Send chunk
            print(f"[Example] Chunk {i+1}/{total_chunks} (final={is_final})")
            result = client.feed_audio_chunk(wav_bytes, is_final=is_final)
            print(f"  Frames: {result['frames_generated']}")
            
    except KeyboardInterrupt:
        print("\n[Example] Recording interrupted by user")
    
    # Check final status
    status = client.get_status()
    print(f"\n[Example] Final: {status['total_frames_generated']} frames generated")
    
    print(f"\n[Example] Stream saved at: http://localhost:9999/hls/{session_id}/playlist.m3u8")


def example_webrtc_integration(server_url: str, image_path: str):
    """
    Example: Integration with WebRTC for bidirectional streaming.
    
    This shows how to feed WebRTC audio into the chunked streaming endpoint.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: WebRTC Integration Pattern")
    print("="*70)
    
    print("""
In your WebRTC server (e.g., with aiortc):

    import asyncio
    from aiohttp import web
    from chunked_streaming_client import ChunkedStreamingClient

    client = ChunkedStreamingClient("http://localhost:9999")

    async def handle_webrtc_offer(request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        
        # Initialize streaming session
        session_id = client.init_session(
            image_path="face.jpg",
            chunk_duration=0.5,
            audio_sample_rate=16000
        )
        
        # Create peer connection
        pc = RTCPeerConnection()
        pc.add_ice_candidate_callback = handle_ice
        
        @pc.on("datachannel")
        def on_datachannel(channel):
            # Handle incoming data/control
            pass
        
        @pc.on("track")
        async def on_track(track):
            # Receive audio from client
            while True:
                frame = await track.recv()
                
                # Convert to WAV bytes and send to streaming client
                audio_bytes = frame.to_ndarray()  # Gets audio samples
                # (convert to WAV format)
                
                client.feed_audio_chunk(audio_bytes, is_final=False)
            
            # Create video track from HLS stream
            @pc.on("track")
            async def send_video():
                # Stream frames from HLS to WebRTC video track
                # (implement frame feeding from HLS segments)
                pass
        
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id
        })

    app = web.Application()
    app.router.add_post("/offer", handle_webrtc_offer)
    web.run_app(app, port=8080)
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunked Streaming Client")
    parser.add_argument("--server", type=str, default="http://localhost:9999",
                        help="FastAPI server URL")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to source face image")
    parser.add_argument("--audio", type=str,
                        help="Path to audio file (for Example 1)")
    parser.add_argument("--example", type=int, default=1, choices=[1, 2, 3],
                        help="Which example to run")
    
    args = parser.parse_args()
    
    if args.example == 1:
        if not args.audio:
            print("Error: --audio required for example 1")
            exit(1)
        example_full_audio_file(args.server, args.image, args.audio)
    
    elif args.example == 2:
        example_realtime_microphone(args.server, args.image)
    
    elif args.example == 3:
        example_webrtc_integration(args.server, args.image)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)
