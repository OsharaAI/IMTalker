"""
FastAPI wrapper for IMTalker video rendering inference server.
Provides REST API endpoints for audio-driven and video-driven talking face generation.
"""

import os
import sys
import tempfile
import shutil
import io
import subprocess
import glob
import base64
from pathlib import Path
from typing import Optional, Generator
import uuid

import torch
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import AppConfig, InferenceAgent, ensure_checkpoints

# Initialize FastAPI app
app = FastAPI(
    title="IMTalker API",
    description="Audio-driven and Video-driven Talking Face Generation API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
cfg = None
agent = None
output_dir = "./outputs"

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global cfg, agent
    
    print("Starting FastAPI server...")
    print("Ensuring checkpoints are downloaded...")
    ensure_checkpoints()
    
    print("Initializing Configuration...")
    cfg = AppConfig()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if os.path.exists(cfg.renderer_path) and os.path.exists(cfg.generator_path):
            print("Loading models...")
            agent = InferenceAgent(cfg)
            print("Models loaded successfully!")
        else:
            print("Error: Checkpoints not found.")
            raise FileNotFoundError("Model checkpoints not found")
    except Exception as e:
        print(f"Initialization Error: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "IMTalker API is running",
        "device": cfg.device if cfg else "unknown",
        "model_loaded": agent is not None
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if agent is not None else "unhealthy",
        "device": cfg.device if cfg else "unknown",
        "cuda_available": torch.cuda.is_available(),
        "models": {
            "renderer": os.path.exists(cfg.renderer_path) if cfg else False,
            "generator": os.path.exists(cfg.generator_path) if cfg else False,
        }
    }


def create_fmp4_chunk(frames: list, audio_path: str, start_frame: int, fps: float = 25.0, temp_dir: str = "/tmp") -> bytes:
    """
    Create a fragmented MP4 chunk from frames and corresponding audio segment.
    
    Args:
        frames: List of numpy arrays (H, W, C) in BGR format
        audio_path: Path to full audio file (can be None for video-only)
        start_frame: Starting frame number (for audio sync)
        fps: Frames per second
        temp_dir: Temporary directory for intermediate files
        
    Returns:
        bytes of the fMP4 chunk
    """
    if not frames:
        return b''
    
    chunk_id = uuid.uuid4().hex[:8]
    temp_video = os.path.join(temp_dir, f"chunk_{chunk_id}_video.mp4")
    temp_audio = os.path.join(temp_dir, f"chunk_{chunk_id}_audio.wav")
    temp_output = os.path.join(temp_dir, f"chunk_{chunk_id}_output.mp4")
    
    try:
        # Calculate timing
        num_frames = len(frames)
        start_time = start_frame / fps
        duration = num_frames / fps
        
        # Write frames to temporary video file
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))
        
        for frame in frames:
            writer.write(frame)
        writer.release()
        
        # Check if we have audio to include
        has_audio = audio_path is not None and os.path.exists(audio_path)
        
        if has_audio:
            # Extract corresponding audio segment
            extract_audio_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                temp_audio
            ]
            result = subprocess.run(extract_audio_cmd, capture_output=True)
            has_audio = result.returncode == 0 and os.path.exists(temp_audio)
        
        if has_audio:
            # Create fragmented MP4 with video + audio
            mux_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', temp_video,
                '-i', temp_audio,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
                '-f', 'mp4',
                temp_output
            ]
        else:
            # Create fragmented MP4 with video only
            mux_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', temp_video,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-pix_fmt', 'yuv420p',
                '-an',  # No audio
                '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
                '-f', 'mp4',
                temp_output
            ]
        
        subprocess.run(mux_cmd, check=True, capture_output=True)
        
        # Read the output
        with open(temp_output, 'rb') as f:
            chunk_data = f.read()
        
        return chunk_data
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return b''
    except Exception as e:
        print(f"Error creating fMP4 chunk: {e}")
        return b''
    finally:
        # Cleanup temp files
        for f in [temp_video, temp_audio, temp_output]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass


def create_hls_stream(frames: list, audio_path: str, fps: float = 25.0, hls_time: int = 4, temp_dir: str = "/tmp") -> tuple:
    """
    Create HLS stream (.m3u8 playlist + .ts segments) from frames and audio.
    
    Args:
        frames: List of numpy arrays (H, W, C) in BGR format
        audio_path: Path to audio file (can be None for video-only)
        fps: Frames per second
        hls_time: Duration of each HLS segment in seconds (default: 4)
        temp_dir: Temporary directory for output files
        
    Returns:
        Tuple of (hls_dir, playlist_path) where:
            - hls_dir: Directory containing all HLS files
            - playlist_path: Path to .m3u8 playlist file
    """
    if not frames:
        return None, None
    
    # Create unique HLS output directory
    hls_id = uuid.uuid4().hex[:8]
    hls_dir = os.path.join(temp_dir, f"hls_{hls_id}")
    os.makedirs(hls_dir, exist_ok=True)
    
    temp_video = os.path.join(hls_dir, "input_video.mp4")
    playlist_path = os.path.join(hls_dir, "stream.m3u8")
    
    try:
        # Write frames to temporary video file
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))
        
        for frame in frames:
            writer.write(frame)
        writer.release()
        
        # Check if we have audio
        has_audio = audio_path is not None and os.path.exists(audio_path)
        
        if has_audio:
            # Create HLS with video + audio
            hls_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', temp_video,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-f', 'hls',
                '-hls_time', str(hls_time),
                '-hls_list_size', '0',  # Keep all segments in playlist
                '-hls_segment_filename', os.path.join(hls_dir, 'segment_%03d.ts'),
                playlist_path
            ]
        else:
            # Create HLS with video only
            hls_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', temp_video,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-pix_fmt', 'yuv420p',
                '-an',
                '-f', 'hls',
                '-hls_time', str(hls_time),
                '-hls_list_size', '0',
                '-hls_segment_filename', os.path.join(hls_dir, 'segment_%03d.ts'),
                playlist_path
            ]
        
        subprocess.run(hls_cmd, check=True, capture_output=True)
        
        # Clean up temp input video
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        return hls_dir, playlist_path
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg HLS error: {e.stderr.decode() if e.stderr else str(e)}")
        return None, None
    except Exception as e:
        print(f"Error creating HLS stream: {e}")
        return None, None


@app.post("/api/v1/audio-driven/stream")
async def audio_driven_inference_stream(
    image: UploadFile = File(..., description="Source image file (PNG, JPG)"),
    audio: UploadFile = File(..., description="Driving audio file (WAV, MP3)"),
    crop: bool = Form(True, description="Auto crop face from image"),
    seed: int = Form(42, description="Random seed for generation"),
    nfe: int = Form(10, description="Number of function evaluations (steps), range: 5-50"),
    cfg_scale: float = Form(2.0, description="Classifier-free guidance scale, range: 1.0-5.0"),
    format: str = Form("fmp4", description="Stream format: 'fmp4' (with audio), 'hls' (HLS playlist), 'mjpeg' (video only), or 'raw'"),
    chunk_duration: float = Form(6.4, description="Chunk/segment duration in seconds (1.0-10.0, default: 6.4 = 160 frames)")
):
    """
    Stream talking face video with audio in various formats.
    
    Supported Formats:
        - 'fmp4': Fragmented MP4 chunks streamed in real-time (video + audio)
        - 'hls': HLS playlist with .ts segments (video + audio) - returns JSON with base64 segments
        - 'mjpeg': Motion JPEG stream (video only, no audio)
        - 'raw': Raw length-prefixed JPEG frames
    
    Args:
        image: Source image containing a face
        audio: Audio file to drive the lip movements
        crop: Whether to automatically crop and detect face
        seed: Random seed for reproducibility
        nfe: Number of sampling steps (higher = better quality but slower)
        cfg_scale: Guidance scale for generation quality
        format: Output format ('fmp4', 'hls', 'mjpeg', 'raw')
        chunk_duration: Duration of each chunk/segment in seconds (default: 2.0)
        
    Returns:
        - fmp4: StreamingResponse with video/mp4 chunks
        - hls: JSONResponse with playlist and base64-encoded .ts segments
        - mjpeg: StreamingResponse with multipart JPEG frames
        - raw: StreamingResponse with length-prefixed JPEG data
    """
    print(f"Received streaming audio-driven inference request: crop={crop}, seed={seed}, nfe={nfe}, cfg_scale={cfg_scale}, format={format}, chunk_duration={chunk_duration}")
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")
    
    # Validate parameters
    if nfe < 5 or nfe > 50:
        raise HTTPException(status_code=400, detail="nfe must be between 5 and 50")
    if cfg_scale < 1.0 or cfg_scale > 5.0:
        raise HTTPException(status_code=400, detail="cfg_scale must be between 1.0 and 5.0")
    if chunk_duration < 1.0 or chunk_duration > 10.0:
        chunk_duration = 3.0  # Default to 3 seconds
    
    fps = 25.0
    frames_per_chunk = int(chunk_duration * fps)  # e.g., 3.0 * 25 = 75 frames
    
    temp_img_path = None
    temp_audio_path = None
    
    try:
        # Save uploaded image
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(image.filename)[1], delete=False) as tmp:
            temp_img_path = tmp.name
            shutil.copyfileobj(image.file, tmp)
        
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio.filename)[1], delete=False) as tmp:
            temp_audio_path = tmp.name
            shutil.copyfileobj(audio.file, tmp)
        
        # Load and process image
        img_pil = Image.open(temp_img_path).convert('RGB')
        
        def fmp4_chunk_generator():
            """Generate fragmented MP4 chunks with video + audio."""
            try:
                frame_stream = agent.run_audio_inference_streaming(
                    img_pil, temp_audio_path, crop, seed, nfe, cfg_scale
                )
                
                frame_buffer = []  # Store BGR numpy frames
                total_frame_count = 0
                chunk_start_frame = 0
                
                for frame_tensor in frame_stream:
                    # Convert tensor (C, H, W) to numpy (H, W, C)
                    frame_np = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
                    
                    # Normalize from [-1, 1] or [0, 1] to [0, 255]
                    if frame_np.min() < 0:
                        frame_np = (frame_np + 1) / 2
                    frame_np = np.clip(frame_np, 0, 1)
                    frame_np = (frame_np * 255).astype(np.uint8)
                    
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    frame_buffer.append(frame_bgr)
                    total_frame_count += 1
                    
                    # When we have enough frames, create and send fMP4 chunk
                    if len(frame_buffer) >= frames_per_chunk:
                        print(f"Creating fMP4 chunk: frames {chunk_start_frame}-{chunk_start_frame + len(frame_buffer) - 1}")
                        
                        chunk_data = create_fmp4_chunk(
                            frames=frame_buffer,
                            audio_path=temp_audio_path,
                            start_frame=chunk_start_frame,
                            fps=fps,
                            temp_dir=tempfile.gettempdir()
                        )
                        
                        if chunk_data:
                            yield chunk_data
                        
                        chunk_start_frame += len(frame_buffer)
                        frame_buffer = []
                
                # Send remaining frames as final chunk
                if frame_buffer:
                    print(f"Creating final fMP4 chunk: frames {chunk_start_frame}-{chunk_start_frame + len(frame_buffer) - 1}")
                    
                    chunk_data = create_fmp4_chunk(
                        frames=frame_buffer,
                        audio_path=temp_audio_path,
                        start_frame=chunk_start_frame,
                        fps=fps,
                        temp_dir=tempfile.gettempdir()
                    )
                    
                    if chunk_data:
                        yield chunk_data
                
                print(f"Streaming complete: {total_frame_count} total frames")
                        
            finally:
                # Clean up temp files
                if temp_img_path and os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
        
        def mjpeg_generator():
            """Generate MJPEG stream (video only, no audio)."""
            try:
                frame_stream = agent.run_audio_inference_streaming(
                    img_pil, temp_audio_path, crop, seed, nfe, cfg_scale
                )
                
                frame_buffer = []
                for frame_tensor in frame_stream:
                    frame_np = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
                    if frame_np.min() < 0:
                        frame_np = (frame_np + 1) / 2
                    frame_np = np.clip(frame_np, 0, 1)
                    frame_np = (frame_np * 255).astype(np.uint8)
                    
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                    frame_bytes = buffer.tobytes()
                    frame_buffer.append(frame_bytes)
                    
                    # Batch frames together
                    if len(frame_buffer) >= frames_per_chunk:
                        batch_data = b''
                        for frame in frame_buffer:
                            batch_data += (b'--frame\r\n'
                                         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        yield batch_data
                        frame_buffer = []
                
                # Send remaining
                if frame_buffer:
                    batch_data = b''
                    for frame in frame_buffer:
                        batch_data += (b'--frame\r\n'
                                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    yield batch_data
                        
            finally:
                if temp_img_path and os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
        
        # Return streaming response based on format
        if format == "fmp4":
            return StreamingResponse(
                fmp4_chunk_generator(),
                media_type="video/mp4",
                headers={
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked",
                    "X-Generation-Params": f"crop={crop},seed={seed},nfe={nfe},cfg_scale={cfg_scale},chunk_duration={chunk_duration}"
                }
            )
        elif format == "hls":
            # Stream HLS segments progressively as frames are generated
            def hls_segment_generator():
                """Generate HLS segments progressively and stream them."""
                try:
                    frame_stream = agent.run_audio_inference_streaming(
                        img_pil, temp_audio_path, crop, seed, nfe, cfg_scale
                    )
                    
                    frame_buffer = []
                    segment_index = 0
                    total_frames = 0
                    playlist_lines = [
                        "#EXTM3U",
                        "#EXT-X-VERSION:3",
                        "#EXT-X-TARGETDURATION:" + str(int(chunk_duration) + 1),
                        "#EXT-X-MEDIA-SEQUENCE:0"
                    ]
                    
                    temp_hls_dir = os.path.join(tempfile.gettempdir(), f"hls_{uuid.uuid4().hex[:8]}")
                    os.makedirs(temp_hls_dir, exist_ok=True)
                    
                    for frame_tensor in frame_stream:
                        # Convert frame
                        frame_np = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
                        if frame_np.min() < 0:
                            frame_np = (frame_np + 1) / 2
                        frame_np = np.clip(frame_np, 0, 1)
                        frame_np = (frame_np * 255).astype(np.uint8)
                        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                        frame_buffer.append(frame_bgr)
                        total_frames += 1
                        
                        # Create segment when buffer is full
                        if len(frame_buffer) >= frames_per_chunk:
                            segment_name = f"segment_{segment_index:03d}.ts"
                            segment_path = os.path.join(temp_hls_dir, segment_name)
                            
                            # Create temporary video from frames
                            temp_video = os.path.join(temp_hls_dir, f"temp_{segment_index}.mp4")
                            h, w = frame_buffer[0].shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            writer = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))
                            for frame in frame_buffer:
                                writer.write(frame)
                            writer.release()
                            
                            # Calculate audio segment timing - use frames_per_chunk for consistent timing
                            start_time = (segment_index * frames_per_chunk) / fps
                            duration = len(frame_buffer) / fps
                            
                            # Extract audio segment
                            temp_audio_seg = os.path.join(temp_hls_dir, f"audio_{segment_index}.wav")
                            has_audio = temp_audio_path and os.path.exists(temp_audio_path)
                            
                            if has_audio:
                                try:
                                    result = subprocess.run([
                                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                                        '-i', temp_audio_path,
                                        '-ss', str(start_time), '-t', str(duration),
                                        '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                                        temp_audio_seg
                                    ], capture_output=True, text=True)
                                    
                                    if result.returncode != 0:
                                        print(f"Warning: Audio extraction failed for segment {segment_index}: {result.stderr}")
                                        has_audio = False
                                    elif not os.path.exists(temp_audio_seg) or os.path.getsize(temp_audio_seg) < 1000:
                                        print(f"Warning: Audio segment {segment_index} is empty or too small")
                                        has_audio = False
                                except Exception as e:
                                    print(f"Error extracting audio for segment {segment_index}: {e}")
                                    has_audio = False
                            
                            # Create HLS segment (.ts file)
                            if has_audio and os.path.exists(temp_audio_seg):
                                ffmpeg_cmd = [
                                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                                    '-i', temp_video, '-i', temp_audio_seg,
                                    '-c:v', 'libx264', '-preset', 'ultrafast',
                                    '-c:a', 'aac', '-b:a', '128k',
                                    '-f', 'mpegts',
                                    segment_path
                                ]
                            else:
                                ffmpeg_cmd = [
                                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                                    '-i', temp_video,
                                    '-c:v', 'libx264', '-preset', 'ultrafast', '-an',
                                    '-f', 'mpegts',
                                    segment_path
                                ]
                            
                            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                            
                            # Read segment and send as JSON
                            with open(segment_path, 'rb') as f:
                                segment_data = base64.b64encode(f.read()).decode('utf-8')
                            
                            # Add to playlist
                            playlist_lines.append(f"#EXTINF:{duration:.3f},")
                            playlist_lines.append(segment_name)
                            
                            # Create partial playlist (for progressive playback)
                            partial_playlist = "\n".join(playlist_lines)
                            
                            # Yield segment with partial playlist
                            import json
                            yield json.dumps({
                                "type": "segment",
                                "index": segment_index,
                                "name": segment_name,
                                "duration": duration,
                                "data": segment_data,
                                "playlist": partial_playlist  # Include current playlist state
                            }) + "\n"
                            
                            print(f"Streamed HLS segment {segment_index}: {len(frame_buffer)} frames")
                            
                            # Cleanup temp files for this segment
                            for f in [temp_video, temp_audio_seg]:
                                if os.path.exists(f):
                                    os.remove(f)
                            
                            segment_index += 1
                            frame_buffer = []
                    
                    # Process remaining frames
                    if frame_buffer:
                        segment_name = f"segment_{segment_index:03d}.ts"
                        segment_path = os.path.join(temp_hls_dir, segment_name)
                        
                        temp_video = os.path.join(temp_hls_dir, f"temp_{segment_index}.mp4")
                        h, w = frame_buffer[0].shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))
                        for frame in frame_buffer:
                            writer.write(frame)
                        writer.release()
                        
                        start_time = (segment_index * frames_per_chunk) / fps
                        duration = len(frame_buffer) / fps
                        
                        temp_audio_seg = os.path.join(temp_hls_dir, f"audio_{segment_index}.wav")
                        has_audio = temp_audio_path and os.path.exists(temp_audio_path)
                        
                        if has_audio:
                            try:
                                result = subprocess.run([
                                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                                    '-i', temp_audio_path,
                                    '-ss', str(start_time), '-t', str(duration),
                                    '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                                    temp_audio_seg
                                ], capture_output=True, text=True)
                                
                                if result.returncode != 0:
                                    print(f"Warning: Audio extraction failed for final segment {segment_index}: {result.stderr}")
                                    has_audio = False
                                elif not os.path.exists(temp_audio_seg) or os.path.getsize(temp_audio_seg) < 1000:
                                    print(f"Warning: Final audio segment {segment_index} is empty or too small")
                                    has_audio = False
                            except Exception as e:
                                print(f"Error extracting audio for final segment {segment_index}: {e}")
                                has_audio = False
                        
                        if has_audio and os.path.exists(temp_audio_seg):
                            ffmpeg_cmd = [
                                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                                '-i', temp_video, '-i', temp_audio_seg,
                                '-c:v', 'libx264', '-preset', 'ultrafast',
                                '-c:a', 'aac', '-b:a', '128k',
                                '-f', 'mpegts', segment_path
                            ]
                        else:
                            ffmpeg_cmd = [
                                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                                '-i', temp_video,
                                '-c:v', 'libx264', '-preset', 'ultrafast', '-an',
                                '-f', 'mpegts', segment_path
                            ]
                        
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                        
                        with open(segment_path, 'rb') as f:
                            segment_data = base64.b64encode(f.read()).decode('utf-8')
                        
                        playlist_lines.append(f"#EXTINF:{duration:.3f},")
                        playlist_lines.append(segment_name)
                        
                        # Create partial playlist for this segment
                        partial_playlist = "\n".join(playlist_lines)
                        
                        import json
                        yield json.dumps({
                            "type": "segment",
                            "index": segment_index,
                            "name": segment_name,
                            "duration": duration,
                            "data": segment_data,
                            "playlist": partial_playlist  # Include current playlist state
                        }) + "\n"
                        
                        for f in [temp_video, temp_audio_seg]:
                            if os.path.exists(f):
                                os.remove(f)
                        
                        segment_index += 1
                    
                    # Send final playlist
                    playlist_lines.append("#EXT-X-ENDLIST")
                    playlist_content = "\n".join(playlist_lines)
                    
                    import json
                    yield json.dumps({
                        "type": "playlist",
                        "content": playlist_content,
                        "total_segments": segment_index,
                        "total_frames": total_frames
                    }) + "\n"
                    
                    print(f"HLS streaming complete: {segment_index} segments, {total_frames} frames")
                    
                finally:
                    # Cleanup
                    if os.path.exists(temp_hls_dir):
                        shutil.rmtree(temp_hls_dir)
                    if temp_img_path and os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
            
            return StreamingResponse(
                hls_segment_generator(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Content-Type": "hls-segments",
                    "X-Generation-Params": f"crop={crop},seed={seed},nfe={nfe},cfg_scale={cfg_scale},segment_duration={chunk_duration}"
                }
            )
        elif format == "mjpeg":
            return StreamingResponse(
                mjpeg_generator(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Generation-Params": f"crop={crop},seed={seed},nfe={nfe},cfg_scale={cfg_scale}"
                }
            )
        else:
            # Raw format - send length-prefixed JPEG frames
            def raw_generator():
                try:
                    frame_stream = agent.run_audio_inference_streaming(
                        img_pil, temp_audio_path, crop, seed, nfe, cfg_scale
                    )
                    for frame_tensor in frame_stream:
                        frame_np = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
                        if frame_np.min() < 0:
                            frame_np = (frame_np + 1) / 2
                        frame_np = np.clip(frame_np, 0, 1)
                        frame_np = (frame_np * 255).astype(np.uint8)
                        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                        frame_bytes = buffer.tobytes()
                        yield len(frame_bytes).to_bytes(4, 'big') + frame_bytes
                finally:
                    if temp_img_path and os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
            
            return StreamingResponse(
                raw_generator(),
                media_type="application/octet-stream",
                headers={
                    "X-Generation-Params": f"crop={crop},seed={seed},nfe={nfe},cfg_scale={cfg_scale}"
                }
            )
        
    except Exception as e:
        # Clean up on error
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        print(f"Error during streaming audio-driven inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/api/v1/audio-driven")
async def audio_driven_inference(
    image: UploadFile = File(..., description="Source image file (PNG, JPG)"),
    audio: UploadFile = File(..., description="Driving audio file (WAV, MP3)"),
    crop: bool = Form(True, description="Auto crop face from image"),
    seed: int = Form(42, description="Random seed for generation"),
    nfe: int = Form(10, description="Number of function evaluations (steps), range: 5-50"),
    cfg_scale: float = Form(2.0, description="Classifier-free guidance scale, range: 1.0-5.0")
):
    """
    Generate talking face video from a source image and driving audio.
    
    Args:
        image: Source image containing a face
        audio: Audio file to drive the lip movements
        crop: Whether to automatically crop and detect face
        seed: Random seed for reproducibility
        nfe: Number of sampling steps (higher = better quality but slower)
        cfg_scale: Guidance scale for generation quality
        
    Returns:
        Video file with the generated talking face
    """
    print("Received audio-driven inference request params ", {
        "crop": crop,
        "seed": seed,
        "nfe": nfe,
        "cfg_scale": cfg_scale
    })
    if agent is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")
    
    # Validate parameters
    if nfe < 5 or nfe > 50:
        raise HTTPException(status_code=400, detail="nfe must be between 5 and 50")
    if cfg_scale < 1.0 or cfg_scale > 5.0:
        raise HTTPException(status_code=400, detail="cfg_scale must be between 1.0 and 5.0")
    
    # Create temporary files for input
    temp_img_path = None
    temp_audio_path = None
    output_path = None
    
    try:
        # Save uploaded image
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(image.filename)[1], delete=False) as tmp:
            temp_img_path = tmp.name
            shutil.copyfileobj(image.file, tmp)
        
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio.filename)[1], delete=False) as tmp:
            temp_audio_path = tmp.name
            shutil.copyfileobj(audio.file, tmp)
        
        # Load and process image
        img_pil = Image.open(temp_img_path).convert('RGB')
        
        # Run inference
        print(f"Running audio-driven inference: crop={crop}, seed={seed}, nfe={nfe}, cfg_scale={cfg_scale}")
        output_video_path = agent.run_audio_inference(
            img_pil, 
            temp_audio_path, 
            crop, 
            seed, 
            nfe, 
            cfg_scale
        )
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        output_filename = f"audio_driven_{unique_id}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Move output to persistent location
        shutil.move(output_video_path, output_path)
        
        print(f"Video generated successfully: {output_path}")
        
        # Return the video file
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=output_filename,
            headers={
                "X-Output-Path": output_path,
                "X-Generation-Params": f"crop={crop},seed={seed},nfe={nfe},cfg_scale={cfg_scale}"
            }
        )
        
    except Exception as e:
        print(f"Error during audio-driven inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        
    finally:
        # Clean up temporary input files
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@app.post("/api/v1/video-driven/stream")
async def video_driven_inference_stream(
    image: UploadFile = File(..., description="Source image file (PNG, JPG)"),
    video: UploadFile = File(..., description="Driving video file (MP4, AVI)"),
    crop: bool = Form(False, description="Auto crop face from both source and driving video"),
    format: str = Form("fmp4", description="Stream format: 'fmp4' (with audio), 'mjpeg' (video only), or 'raw'"),
    chunk_duration: float = Form(3.0, description="Chunk duration in seconds (1.0-10.0, default: 3.0)")
):
    """
    Stream video with audio as fragmented MP4 chunks (motion transferred from driving video).
    
    Args:
        image: Source image containing a face
        video: Driving video with facial movements to transfer
        crop: Whether to automatically crop and detect faces
        format: 'fmp4' for fragmented MP4 with audio, 'mjpeg' for video only, 'raw' for raw frames
        chunk_duration: Duration of each chunk in seconds (default: 3.0)
        
    Returns:
        Stream of fragmented MP4 chunks (video + audio) or MJPEG frames
    """
    print(f"Received streaming video-driven inference request: crop={crop}, format={format}, chunk_duration={chunk_duration}")
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")
    
    # Validate parameters
    if chunk_duration < 1.0 or chunk_duration > 10.0:
        chunk_duration = 3.0  # Default to 3 seconds
    
    fps = 25.0
    frames_per_chunk = int(chunk_duration * fps)
    
    temp_img_path = None
    temp_video_path = None
    temp_audio_path = None
    
    try:
        # Save uploaded image
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(image.filename)[1], delete=False) as tmp:
            temp_img_path = tmp.name
            shutil.copyfileobj(image.file, tmp)
        
        # Save uploaded video
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(video.filename)[1], delete=False) as tmp:
            temp_video_path = tmp.name
            shutil.copyfileobj(video.file, tmp)
        
        # Extract audio from driving video for fMP4 streaming
        temp_audio_path = None
        if format == "fmp4":
            temp_audio_path = tempfile.mktemp(suffix='.wav')
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', temp_video_path,
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    temp_audio_path
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print("Warning: Could not extract audio from driving video, streaming video only")
                temp_audio_path = None
        
        # Load and process image
        img_pil = Image.open(temp_img_path).convert('RGB')
        
        def fmp4_chunk_generator():
            """Generate fragmented MP4 chunks with video + audio from driving video."""
            try:
                frame_stream = agent.run_video_inference_streaming(
                    img_pil, temp_video_path, crop
                )
                
                frame_buffer = []
                total_frame_count = 0
                chunk_start_frame = 0
                
                for frame_tensor in frame_stream:
                    frame_np = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
                    if frame_np.min() < 0:
                        frame_np = (frame_np + 1) / 2
                    frame_np = np.clip(frame_np, 0, 1)
                    frame_np = (frame_np * 255).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    frame_buffer.append(frame_bgr)
                    total_frame_count += 1
                    
                    if len(frame_buffer) >= frames_per_chunk:
                        print(f"Creating fMP4 chunk: frames {chunk_start_frame}-{chunk_start_frame + len(frame_buffer) - 1}")
                        
                        if temp_audio_path and os.path.exists(temp_audio_path):
                            chunk_data = create_fmp4_chunk(
                                frames=frame_buffer,
                                audio_path=temp_audio_path,
                                start_frame=chunk_start_frame,
                                fps=fps,
                                temp_dir=tempfile.gettempdir()
                            )
                        else:
                            # No audio - create video-only fMP4
                            chunk_data = create_fmp4_chunk(
                                frames=frame_buffer,
                                audio_path=None,
                                start_frame=chunk_start_frame,
                                fps=fps,
                                temp_dir=tempfile.gettempdir()
                            )
                        
                        if chunk_data:
                            yield chunk_data
                        
                        chunk_start_frame += len(frame_buffer)
                        frame_buffer = []
                
                # Send remaining frames
                if frame_buffer:
                    print(f"Creating final fMP4 chunk: frames {chunk_start_frame}-{chunk_start_frame + len(frame_buffer) - 1}")
                    
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        chunk_data = create_fmp4_chunk(
                            frames=frame_buffer,
                            audio_path=temp_audio_path,
                            start_frame=chunk_start_frame,
                            fps=fps,
                            temp_dir=tempfile.gettempdir()
                        )
                    else:
                        chunk_data = create_fmp4_chunk(
                            frames=frame_buffer,
                            audio_path=None,
                            start_frame=chunk_start_frame,
                            fps=fps,
                            temp_dir=tempfile.gettempdir()
                        )
                    
                    if chunk_data:
                        yield chunk_data
                
                print(f"Streaming complete: {total_frame_count} total frames")
                        
            finally:
                if temp_img_path and os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                if temp_video_path and os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
        
        def mjpeg_generator():
            """Generate MJPEG stream (video only)."""
            try:
                frame_stream = agent.run_video_inference_streaming(
                    img_pil, temp_video_path, crop
                )
                
                frame_buffer = []
                for frame_tensor in frame_stream:
                    frame_np = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
                    if frame_np.min() < 0:
                        frame_np = (frame_np + 1) / 2
                    frame_np = np.clip(frame_np, 0, 1)
                    frame_np = (frame_np * 255).astype(np.uint8)
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                    frame_bytes = buffer.tobytes()
                    frame_buffer.append(frame_bytes)
                    
                    if len(frame_buffer) >= frames_per_chunk:
                        batch_data = b''
                        for frame in frame_buffer:
                            batch_data += (b'--frame\r\n'
                                         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        yield batch_data
                        frame_buffer = []
                
                if frame_buffer:
                    batch_data = b''
                    for frame in frame_buffer:
                        batch_data += (b'--frame\r\n'
                                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    yield batch_data
                        
            finally:
                if temp_img_path and os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
                if temp_video_path and os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        
        # Return streaming response based on format
        if format == "fmp4":
            return StreamingResponse(
                fmp4_chunk_generator(),
                media_type="video/mp4",
                headers={
                    "Cache-Control": "no-cache",
                    "Transfer-Encoding": "chunked",
                    "X-Generation-Params": f"crop={crop},chunk_duration={chunk_duration}"
                }
            )
        elif format == "mjpeg":
            return StreamingResponse(
                mjpeg_generator(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Generation-Params": f"crop={crop}"
                }
            )
        else:
            # Raw format
            def raw_generator():
                try:
                    frame_stream = agent.run_video_inference_streaming(
                        img_pil, temp_video_path, crop
                    )
                    for frame_tensor in frame_stream:
                        frame_np = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
                        if frame_np.min() < 0:
                            frame_np = (frame_np + 1) / 2
                        frame_np = np.clip(frame_np, 0, 1)
                        frame_np = (frame_np * 255).astype(np.uint8)
                        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                        frame_bytes = buffer.tobytes()
                        yield len(frame_bytes).to_bytes(4, 'big') + frame_bytes
                finally:
                    if temp_img_path and os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                    if temp_video_path and os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
            
            return StreamingResponse(
                raw_generator(),
                media_type="application/octet-stream",
                headers={
                    "X-Generation-Params": f"crop={crop}"
                }
            )
        
    except Exception as e:
        # Clean up on error
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        print(f"Error during streaming video-driven inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/api/v1/video-driven")
async def video_driven_inference(
    image: UploadFile = File(..., description="Source image file (PNG, JPG)"),
    video: UploadFile = File(..., description="Driving video file (MP4, AVI)"),
    crop: bool = Form(False, description="Auto crop face from both source and driving video")
):
    """
    Generate video by transferring motion from a driving video to a source image.
    
    Args:
        image: Source image containing a face
        video: Driving video with facial movements to transfer
        crop: Whether to automatically crop and detect faces
        
    Returns:
        Video file with the motion transferred to the source face
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")
    
    temp_img_path = None
    temp_video_path = None
    output_path = None
    
    try:
        # Save uploaded image
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(image.filename)[1], delete=False) as tmp:
            temp_img_path = tmp.name
            shutil.copyfileobj(image.file, tmp)
        
        # Save uploaded video
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(video.filename)[1], delete=False) as tmp:
            temp_video_path = tmp.name
            shutil.copyfileobj(video.file, tmp)
        
        # Load and process image
        img_pil = Image.open(temp_img_path).convert('RGB')
        
        # Run inference
        print(f"Running video-driven inference: crop={crop}")
        output_video_path = agent.run_video_inference(
            img_pil,
            temp_video_path,
            crop
        )
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        output_filename = f"video_driven_{unique_id}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Move output to persistent location
        shutil.move(output_video_path, output_path)
        
        print(f"Video generated successfully: {output_path}")
        
        # Return the video file
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=output_filename,
            headers={
                "X-Output-Path": output_path,
                "X-Generation-Params": f"crop={crop}"
            }
        )
        
    except Exception as e:
        print(f"Error during video-driven inference: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        
    finally:
        # Clean up temporary input files
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)


@app.get("/api/v1/outputs/{filename}")
async def get_output(filename: str):
    """
    Retrieve a previously generated output video.
    
    Args:
        filename: Name of the output file
        
    Returns:
        Video file
    """
    file_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename
    )


@app.delete("/api/v1/outputs/{filename}")
async def delete_output(filename: str):
    """
    Delete a previously generated output video.
    
    Args:
        filename: Name of the output file to delete
        
    Returns:
        Success message
    """
    file_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        return {"status": "success", "message": f"File {filename} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@app.get("/api/v1/outputs")
async def list_outputs():
    """
    List all generated output videos.
    
    Returns:
        List of output filenames with metadata
    """
    try:
        files = []
        for filename in os.listdir(output_dir):
            if filename.endswith('.mp4'):
                file_path = os.path.join(output_dir, filename)
                stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime
                })
        
        return {"count": len(files), "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IMTalker FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9999, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "fastapi_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
