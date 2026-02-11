
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64
import json
import io
import wave
import asyncio
import threading
import queue
import uuid
import fractions
import collections
from dotenv import load_dotenv

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
    imtalker_cfg.batch_size = 15  # Increased from 10 for better throughput
    imtalker_cfg.use_fp16 = False  # Disabled: causes NaN/Inf in frames with this model
    imtalker_cfg.use_compile = True  # Enable torch.compile for optimized kernels
    
    app.state.imtalker_streamer = IMTalkerStreamer(imtalker_cfg, device=device)
    app.state.webrtc_sessions = {}
    
    yield
    # Clean up models if necessary
    print("Lifespan: Shutting down...")
    if hasattr(app.state, 'openai_client'):
        await app.state.openai_client.close()
    del app.state.tts_model
    del app.state.imtalker_streamer

app = FastAPI(lifespan=lifespan)

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

# --- Conversation Manager ---

class ConversationManager:
    def __init__(self, openai_client, tts_model, imtalker_streamer, video_track, audio_track, avatar_img):
        self.openai_client = openai_client
        self.tts_model = tts_model
        self.imtalker_streamer = imtalker_streamer
        self.video_track = video_track
        self.audio_track = audio_track
        self.avatar_img = avatar_img
        
        # Use deque for efficient appends (ring buffer)
        self.audio_buffer = collections.deque()  # Buffer for current spoken segment
        self.preroll_buffer = collections.deque(maxlen=15) # ~0.6s of pre-speech audio (40ms chunks)
        
        # VAD optimization: cache resampler arrays
        self.resampler_cache = {}  # {sample_rate: (x_old, x_new)} pre-computed arrays
        self.vad_sample_counter = 0  # Sample every other frame for efficiency
        
        # Frame interpolation settings
        self.enable_frame_blending = True  # Smooth transitions between chunks
        self.blend_steps = 2  # Number of interpolated frames (alpha: 0.33, 0.66)
        self.last_chunk_final_frame = None  # Track last frame for blending
        
        # Pipeline parallelization settings
        self.enable_parallel_pipeline = True  # Concurrent TTS/Render
        from concurrent.futures import ThreadPoolExecutor
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TTS")
        self.render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Render")
        
        self.is_speaking = False # State of AI response
        self.user_is_talking = False # State of user speech detection
        
        self.system_prompt = """You are a friendly and helpful AI assistant avatar. 
Keep your answers brief and engaging, suitable for a voice conversation."""
        
        # VAD parameters
        self.start_threshold = 0.018 # Slightly higher to avoid static
        self.stop_threshold = 0.012  # Higher to ignore background hum
        self.silence_duration = 0.7 # Shorter for snappier feel
        
        self.last_speech_time = time.time()
        self._processing_task = None
        self._loop = asyncio.get_event_loop()

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
            byte_io = io.BytesIO()
            with wave.open(byte_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes((trimmed_audio * 32767).astype(np.int16).tobytes())
            audio_base64 = base64.b64encode(byte_io.getvalue()).decode('utf-8')

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-audio-preview",
                modalities=["text"],
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
            
            text_response = response.choices[0].message.content
            print(f"CM: Assistant response from OpenAI: {text_response}")
            
            if text_response:
                await self.generate_and_feed(text_response)
            else:
                print("CM: Empty response from OpenAI")
                
        except Exception as e:
            print(f"CM OpenAI/Process Error: {e}")
            traceback.print_exc()

    async def generate_and_feed(self, text):
        print(f"CM: Generating response: {text[:50]}{'...' if len(text) > 50 else ''}")
        self.is_speaking = True
        
        # Pipeline queues
        audio_chunks_queue = asyncio.Queue(maxsize=3)
        video_chunks_queue = asyncio.Queue(maxsize=2)
        
        try:
            # Stage 1: Audio Chunking
            async def audio_chunker():
                print("CM: Stage 1 - Audio chunking started")
                tts_stream = self.tts_model.generate_stream(
                    text=text,
                    chunk_size=20,  # Reduced from 50 for smaller chunks
                    print_metrics=False
                )
                
                buffer = []
                buffer_samples = 0
                target_samples = 2 * 24000  # 2 seconds at 24kHz (reduced from 5s for lower latency)
                
                for audio_chunk, metrics in tts_stream:
                    # chunk is [1, T] tensor
                    # DON'T feed audio immediately - store it with video frames
                    
                    buffer.append(audio_chunk)
                    buffer_samples += audio_chunk.shape[-1]
                    
                    # When we have 2 seconds, push to rendering
                    if buffer_samples >= target_samples:
                        chunk = torch.cat(buffer, dim=-1)
                        # Monitor queue depth for backpressure
                        queue_depth = audio_chunks_queue.qsize()
                        if queue_depth >= 2:
                            print(f"CM: Stage 1 - ⚠️  Audio queue backpressure ({queue_depth}/3)")
                        await audio_chunks_queue.put(chunk)
                        print(f"CM: Stage 1 - Pushed {buffer_samples/24000:.1f}s audio chunk (queue: {queue_depth+1}/3)")
                        buffer = []
                        buffer_samples = 0
                
                # Push remaining audio
                if buffer:
                    chunk = torch.cat(buffer, dim=-1)
                    await audio_chunks_queue.put(chunk)
                    print(f"CM: Stage 1 - Pushed final {buffer_samples/24000:.1f}s audio chunk")
                
                # Signal end of audio
                await audio_chunks_queue.put(None)
                print("CM: Stage 1 - Audio chunking finished")
            
            # Stage 2: Video Rendering
            async def video_renderer():
                print("CM: Stage 2 - Video renderer started")
                chunk_idx = 0
                
                while True:
                    audio_chunk = await audio_chunks_queue.get()
                    if audio_chunk is None:
                        await video_chunks_queue.put(None)
                        break
                    
                    chunk_idx += 1
                    chunk_duration = audio_chunk.shape[-1] / 24000  # Chatterbox uses 24kHz
                    print(f"CM: Stage 2 - Rendering chunk {chunk_idx} ({chunk_duration:.1f}s, {audio_chunk.shape[-1]} samples)")
                    
                    # Render in thread pool
                    def render():
                        frames = []
                        # Create a simple audio iterator for this chunk
                        def audio_iter():
                            yield audio_chunk
                        
                        for video_frames, metrics in self.imtalker_streamer.generate_stream(self.avatar_img, audio_iter()):
                            for frame_tensor in video_frames:
                                frame_rgb = process_frame_tensor(frame_tensor, format="RGB")
                                av_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                                frames.append(av_frame)
                        
                        # Clear CUDA cache after rendering to prevent OOM
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        return frames, metrics
                    
                    frames, metrics = await self._loop.run_in_executor(None, render)
                    # Package audio WITH video frames for synchronized playback
                    queue_depth = video_chunks_queue.qsize()
                    if queue_depth >= 1:
                        print(f"CM: Stage 2 - ⚠️  Video queue backpressure ({queue_depth}/2)")
                    await video_chunks_queue.put((frames, metrics, audio_chunk))
                    print(f"CM: Stage 2 - Chunk {chunk_idx} rendered: {len(frames)} frames, RTF: {metrics['rtf']:.2f} (queue: {queue_depth+1}/2)")
                
                print("CM: Stage 2 - Video renderer finished")
            
            # Stage 3: Video Streaming (Synchronized A/V with Frame Blending)
            async def video_streamer():
                print("CM: Stage 3 - Video streamer started")
                total_frames = 0
                chunk_count = 0
                
                # Helper function for frame blending
                def blend_frames(frame1, frame2, alpha):
                    """Alpha blend between two VideoFrame objects"""
                    arr1 = frame1.to_ndarray(format="rgb24")
                    arr2 = frame2.to_ndarray(format="rgb24")
                    blended = (arr1 * (1 - alpha) + arr2 * alpha).astype(np.uint8)
                    return VideoFrame.from_ndarray(blended, format="rgb24")
                
                while True:
                    item = await video_chunks_queue.get()
                    if item is None:
                        break
                    
                    frames, metrics, audio_chunk = item
                    chunk_count += 1
                    print(f"CM: Stage 3 - Streaming chunk {chunk_count}: {len(frames)} frames")
                    
                    # Feed audio to track first (it buffers internally)
                    audio_np = audio_chunk.cpu().numpy().flatten()
                    asyncio.run_coroutine_threadsafe(
                        self.audio_track.add_audio(audio_np), 
                        self._loop
                    )
                    
                    # Insert blended transition frames if enabled and we have a previous chunk
                    if self.enable_frame_blending and self.last_chunk_final_frame is not None and len(frames) > 0:
                        first_frame = frames[0]
                        # Generate interpolated frames (alpha: 0.33, 0.66)
                        for i in range(1, self.blend_steps + 1):
                            alpha = i / (self.blend_steps + 1)  # 0.33, 0.66 for blend_steps=2
                            blended = blend_frames(self.last_chunk_final_frame, first_frame, alpha)
                            asyncio.run_coroutine_threadsafe(self.video_track.add_frame(blended), self._loop)
                            await asyncio.sleep(0.04)  # 25 FPS
                            total_frames += 1
                        print(f"CM: Stage 3 - Inserted {self.blend_steps} blended transition frames")
                    
                    # Stream the actual chunk frames
                    for av_frame in frames:
                        asyncio.run_coroutine_threadsafe(self.video_track.add_frame(av_frame), self._loop)
                        # Pace at 25 FPS
                        await asyncio.sleep(0.04)
                    
                    total_frames += len(frames)
                    
                    # Save last frame for next chunk blending
                    if len(frames) > 0:
                        self.last_chunk_final_frame = frames[-1]
                
                print(f"CM: Stage 3 - Video streaming finished ({total_frames} total frames)")
            
            # Run all stages concurrently
            await asyncio.gather(
                audio_chunker(),
                video_renderer(),
                video_streamer()
            )
            
        except Exception as e:
            print(f"CM: Pipeline error: {e}")
            traceback.print_exc()
        finally:
            print("CM: Resetting is_speaking to False")
            self.is_speaking = False


# --- WebRTC Tracks ---

class IMTalkerVideoTrack(VideoStreamTrack):
    def __init__(self, avatar_img):
        super().__init__()
        self.avatar_img = avatar_img
        self.frame_queue = asyncio.Queue()
        # Convert avatar to RGB for idle frames
        self.idle_frame = np.array(avatar_img.convert('RGB'))
        
        # Smoothness settings
        self.prebuffer_size = 6  # Reduced from 20 to minimize startup latency (240ms vs 800ms)
        self.is_buffering = True

    async def add_frame(self, av_frame):
        await self.frame_queue.put(av_frame)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        # If we are buffering, wait until we have enough frames
        if self.is_buffering:
            if self.frame_queue.qsize() < self.prebuffer_size:
                # Return idle frame while buffering
                frame = VideoFrame.from_ndarray(self.idle_frame, format="rgb24")
                frame.pts = pts
                frame.time_base = time_base
                return frame
            else:
                self.is_buffering = False
                print(f"VT: Pre-buffer filled ({self.frame_queue.qsize()} frames), starting playback.")

        try:
            # Wait up to 50ms for a frame (2 frames at 25 FPS)
            # This tolerates brief gaps between chunks while maintaining smooth FPS
            frame = await asyncio.wait_for(self.frame_queue.get(), timeout=0.05)
        except (asyncio.QueueEmpty, asyncio.TimeoutError):
            # If nothing after 2 seconds, show idle frame but DON'T re-buffer
            # This keeps playback smooth and avoids the 5+ second delays
            frame = VideoFrame.from_ndarray(self.idle_frame, format="rgb24")
        
        frame.pts = pts
        frame.time_base = time_base
        return frame

class IMTalkerAudioTrack(AudioStreamTrack):
    # This track handles audio generated by Chatterbox (24kHz)
    def __init__(self):
        super().__init__()
        self.audio_queue = asyncio.Queue()
        self._timestamp = 0
        self.sample_rate = 24000  # Chatterbox TTS sample rate

    async def add_audio(self, audio_np):
        """Expects normalized float32 numpy array. Splits into WebRTC-sized frames."""
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # Check if audio has actual data
        max_val = np.abs(audio_int16).max()
        mean_val = np.abs(audio_int16).mean()
        
        # WebRTC expects small frames (20ms at 24kHz = 480 samples)
        frame_size = 480
        num_frames = len(audio_int16) // frame_size
        
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio_int16[start:end]
            await self.audio_queue.put(frame)
        
        # Handle remaining samples
        remainder = len(audio_int16) % frame_size
        if remainder > 0:
            # Pad with zeros to make a complete frame
            last_frame = np.zeros(frame_size, dtype=np.int16)
            last_frame[:remainder] = audio_int16[-remainder:]
            await self.audio_queue.put(last_frame)
        
        print(f"AT: Split {len(audio_int16)} samples into {num_frames + (1 if remainder > 0 else 0)} frames (queue: {self.audio_queue.qsize()}) - max: {max_val}, mean: {mean_val:.1f}")

    async def recv(self):
        try:
            # Wait up to 100ms for audio data
            audio_int16 = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
            is_silence = False
        except (asyncio.QueueEmpty, asyncio.TimeoutError):
            # Return silence (20ms at 24kHz = 480 samples)
            audio_int16 = np.zeros(480, dtype=np.int16)
            is_silence = True
        
        # Log every few frames for debugging
        if not is_silence:
            if self.audio_queue.qsize() == 0 or self._timestamp % (self.sample_rate * 2) < 1000:
                print(f"AT: Sending audio frame ({len(audio_int16)} samples, queue: {self.audio_queue.qsize()}, pts: {self._timestamp})")
        
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
            chunk_size=10,  # Reduced for lower latency
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
            chunk_size=10,  # Reduced for lower latency
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
            chunk_size=10,  # Reduced for lower latency
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

@app.post("/offer")
async def webrtc_offer(request: Request):
    try:
        data = await request.json()
        sdp = data.get("sdp")
        sdp_type = data.get("type", "offer")
        avatar = data.get("avatar", "assets/source_1.png")
        conv_mode = data.get("conv_mode", False)
        text = data.get("text", "Hello, I am ready to talk.")
        
        if not sdp:
            raise HTTPException(status_code=400, detail="SDP is required")

        # Initialize models
        tts_model = app.state.tts_model
        imtalker_streamer = app.state.imtalker_streamer
        openai_client = app.state.openai_client
        
        # Load Avatar
        avatar_path = os.path.join(current_dir, avatar)
        if not os.path.exists(avatar_path):
            avatar_path = avatar
            
        if not os.path.exists(avatar_path):
            avatar_img = Image.new('RGB', (512, 512), color='red')
        else:
            avatar_img = Image.open(avatar_path).convert('RGB')

        # Create PeerConnection with STUN
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            )
        )
        
        # Create Tracks
        video_track = IMTalkerVideoTrack(avatar_img)
        audio_track = IMTalkerAudioTrack()
        
        pc.addTrack(video_track)
        pc.addTrack(audio_track)

        # Session ID
        session_id = str(uuid.uuid4())
        app.state.webrtc_sessions[session_id] = pc

        # Conversation Manager (optional)
        conv_manager = None
        if conv_mode:
            if not openai_client:
                raise HTTPException(status_code=400, detail="OpenAI API key not configured, conversational mode unavailable.")
            conv_manager = ConversationManager(
                openai_client, tts_model, imtalker_streamer, 
                video_track, audio_track, avatar_img
            )

        @pc.on("track")
        def on_track(track):
            print(f"📡 Track received: {track.kind}")
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
            print(f"Connection state for {session_id} is {pc.connectionState}")
            if pc.connectionState == "connected":
                # If not in conversational mode, we might want to start a one-shot generation
                if not conv_mode:
                    # Original logic for one-shot text-to-video
                    tts_stream = tts_model.generate_stream(text=text, chunk_size=20, print_metrics=False)
                    audio_iterator = chatterbox_adapter(tts_stream)
                    
                    def run_one_shot():
                        for video_frames, metrics in imtalker_streamer.generate_stream(avatar_img, audio_iterator):
                            for frame_tensor in video_frames:
                                frame_bgr = process_frame_tensor(frame_tensor)
                                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                av_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                                asyncio.run_coroutine_threadsafe(video_track.add_frame(av_frame), asyncio.get_event_loop())
                    
                        # Feed audio as well
                        # Re-running TTS stream to feed audio track properly (or adapt chatterbox_adapter)
                        # Actually the tracks are now refillable.
                    
                    # For one-shot, we can use a simpler distributor like before but feeding into tracks
                    # but let's keep it consistent.
                    async def feed_one_shot():
                        # We need to feed both. The new tracks expect add_frame/add_audio.
                        # Let's reuse generating_and_feed logic but without OpenAI
                        temp_conv = ConversationManager(openai_client, tts_model, imtalker_streamer, video_track, audio_track, avatar_img)
                        await temp_conv.generate_and_feed(text)
                    
                    asyncio.create_task(feed_one_shot())

            elif pc.connectionState == "failed" or pc.connectionState == "closed":
                await pc.close()
                if session_id in app.state.webrtc_sessions:
                    del app.state.webrtc_sessions[session_id]

        # Handle offer
        offer = RTCSessionDescription(sdp=sdp, type="offer") # default to offer
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id
        }
        
    except Exception as e:
        print(f"WebRTC Offer Error: {e}")
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
    
    # [3, H, W] -> [H, W, 3]
    frame_np = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
    
    # Post-process: [-1, 1] or [0, 1] -> [0, 255]
    # Check range
    if frame_np.min() < 0:
        frame_np = (frame_np + 1) / 2
    
    frame_np = np.nan_to_num(frame_np)
    frame_np = np.clip(frame_np, 0, 1)
    frame_np = (frame_np * 255).astype(np.uint8)
    
    
    if format == "RGB":
        return frame_np
    
    # RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    return frame_bgr

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

if __name__ == "__main__":
    main()
