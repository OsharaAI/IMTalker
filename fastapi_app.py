"""
FastAPI wrapper for IMTalker video rendering inference server.
Provides REST API endpoints for audio-driven and video-driven talking face generation.
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
import cv2
import torch
import torchvision
import librosa
import face_alignment
import json
import math
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm
import random
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import asyncio
import threading
from pathlib import Path
import uuid
import asyncio
from typing import Optional
import threading
import time
import queue


# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import AppConfig, ensure_checkpoints, DataProcessor

# Try importing local modules
try:
    from generator.FM import FMGenerator
    from renderer.models import IMTRenderer
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure 'generator' and 'renderer' folders are in the same directory.")


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

# HLS session storage for live streaming
import threading
from dataclasses import dataclass, field
from typing import Dict, List

class InferenceAgent:
    def __init__(self, opt):
        # Clear cache only if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.opt = opt
        self.device = opt.device
        self.data_processor = DataProcessor(opt)
        print("Loading Models...")
        self.renderer = IMTRenderer(self.opt).to(self.device)
        self.generator = FMGenerator(self.opt).to(self.device)
        if not os.path.exists(self.opt.renderer_path) or not os.path.exists(
            self.opt.generator_path
        ):
            raise FileNotFoundError(
                "Checkpoints not found even after download attempt."
            )
        self._load_ckpt(self.renderer, self.opt.renderer_path, "gen.")
        self._load_fm_ckpt(self.generator, self.opt.generator_path)
        self.renderer.eval()
        self.generator.eval()
        print("Models loaded successfully.")

    def _load_ckpt(self, model, path, prefix="gen."):
        if not os.path.exists(path):
            print(f"Warning: Checkpoint {path} not found.")
            return
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_state_dict = {
            k.replace(prefix, ""): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        model.load_state_dict(clean_state_dict, strict=False)

    def _load_fm_ckpt(self, model, path):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        prefix = "model."
        clean_dict = {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in clean_dict:
                    param.copy_(clean_dict[name].to(self.device))

    def save_video(self, vid_tensor, fps, audio_path=None):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            raw_path = tmp.name
        if vid_tensor.dim() == 4:
            vid = vid_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        if vid.min() < 0:
            vid = (vid + 1) / 2
        vid = np.clip(vid, 0, 1)
        vid = (vid * 255).astype(np.uint8)
        height, width = vid.shape[1], vid.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
        for frame in vid:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        if audio_path:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
                final_path = tmp_out.name
            cmd = f'ffmpeg -y -i "{raw_path}" -i "{audio_path}" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}"'
            subprocess.call(cmd, shell=True)
            if os.path.exists(raw_path):
                os.remove(raw_path)
            return final_path
        else:
            return raw_path

    @torch.no_grad()
    def run_audio_inference(
        self,
        img_pil,
        aud_path,
        crop,
        seed,
        nfe,
        cfg_scale,
        pose_style=None,
        gaze_style=None,
    ):
        # Clear memory before starting
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        s_pil = (
            self.data_processor.process_img(img_pil)
            if crop
            else img_pil.resize((self.opt.input_size, self.opt.input_size))
        )
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = (
            self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        )
        data = {
            "s": s_tensor,
            "a": a_tensor,
            "pose": None,
            "cam": None,
            "gaze": None,
            "ref_x": None,
        }

        # Handle Advanced Controls (Pose & Gaze)
        # Calculate target frame count T based on audio length
        T = math.ceil(a_tensor.shape[-1] * self.opt.fps / self.opt.sampling_rate)

        if pose_style is not None:
            # pose_style is [pitch, yaw, roll] -> Create (1, T, 3) tensor
            # Using simple expansion for static pose offset
            p_tensor = (
                torch.tensor(pose_style, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, T, 3)
            )
            data["pose"] = p_tensor

        if gaze_style is not None:
            # gaze_style is [x, y] -> Create (1, T, 2) tensor
            g_tensor = (
                torch.tensor(gaze_style, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, T, 2)
            )
            data["gaze"] = g_tensor
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        data["ref_x"] = t_lat
        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)

        # Stabilization: blend generated motion with reference motion to reduce large background/head movements
        # This addresses the user's request for "steady and calm" motion.
        motion_scale = 0.6  # More stabilization for steadier result (was 0.75)
        sample = (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) + motion_scale * sample

        d_hat = []
        T = sample.shape[1]
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        for t in range(T):
            ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            out_frame = self.renderer.decode(m_c, m_r, f_r)
            # Move to CPU immediately to save memory
            d_hat.append(out_frame.cpu())
            # Clear cache periodically
            if t % 10 == 0 and self.device == "mps":
                torch.mps.empty_cache()
        vid_tensor = torch.stack(d_hat, dim=1).squeeze(0)

        # Final cleanup
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        return self.save_video(vid_tensor, self.opt.fps, aud_path)

    @torch.no_grad()
    def run_audio_inference_streaming(self, img_pil, aud_path, crop, seed, nfe, cfg_scale):
        """Stream frames as they are generated for audio-driven inference."""
        s_pil = self.data_processor.process_img(img_pil) if crop else img_pil.resize((self.opt.input_size, self.opt.input_size))
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        data = {'s': s_tensor, 'a': a_tensor, 'pose': None, 'cam': None, 'gaze': None, 'ref_x': None}
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple): t_lat = t_lat[0]
        data['ref_x'] = t_lat
        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        T = sample.shape[1]
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        # Yield frames one by one
        for t in range(T):
            ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            out_frame = self.renderer.decode(m_c, m_r, f_r)
            yield out_frame.squeeze(0)  # Yield single frame tensor (C, H, W)

    @torch.no_grad()
    def run_audio_inference_progressive(
        self,
        img_pil,
        aud_path,
        crop,
        seed,
        nfe,
        cfg_scale,
        hls_generator: "ProgressiveHLSGenerator",
        pose_style=None,
        gaze_style=None,
        is_final_chunk=False,
    ):
        """Run audio inference with progressive HLS generation"""
        # Clear memory before starting
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        s_pil = (
            self.data_processor.process_img(img_pil)
            if crop
            else img_pil.resize((self.opt.input_size, self.opt.input_size))
        )
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = (
            self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        )
        data = {
            "s": s_tensor,
            "a": a_tensor,
            "pose": None,
            "cam": None,
            "gaze": None,
            "ref_x": None,
        }
        
        # Handle Advanced Controls (Pose & Gaze)
        T_est = math.ceil(a_tensor.shape[-1] * self.opt.fps / self.opt.sampling_rate)

        if pose_style is not None:
            p_tensor = (
                torch.tensor(pose_style, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, T_est, 3)
            )
            data["pose"] = p_tensor

        if gaze_style is not None:
            g_tensor = (
                torch.tensor(gaze_style, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, T_est, 2)
            )
            data["gaze"] = g_tensor

        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        data["ref_x"] = t_lat
        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)

        # Stabilization: blend generated motion with reference motion to reduce large background/head movements
        motion_scale = 0.6  # More stabilization (was 0.75)
        sample = (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) + motion_scale * sample


        T = sample.shape[1]
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)

        # Generate frames progressively
        for t in range(T):
            ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            out_frame = self.renderer.decode(m_c, m_r, f_r)

            # Convert frame to numpy for HLS generator
            frame_np = out_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            if frame_np.min() < 0:
                frame_np = (frame_np + 1) / 2
            frame_np = np.clip(frame_np, 0, 1)
            frame_np = (frame_np * 255).astype(np.uint8)

            # Add frame to HLS generator (will create segments automatically)
            hls_generator.add_frame(frame_np)

            # Clear cache periodically
            if t % 10 == 0 and self.device == "mps":
                torch.mps.empty_cache()

        # Finalize HLS stream - mark complete only if this is the final chunk
        hls_generator.finalize(audio_path=aud_path, mark_complete=is_final_chunk)

        # Final cleanup
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def run_video_inference(self, source_img_pil, driving_video_path, crop):
        # Clear memory before starting
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        s_pil = (
            self.data_processor.process_img(source_img_pil)
            if crop
            else source_img_pil.resize((self.opt.input_size, self.opt.input_size))
        )
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        f_r, i_r = self.renderer.app_encode(s_tensor)
        t_r = self.renderer.mot_encode(s_tensor)
        ta_r = self.renderer.adapt(t_r, i_r)
        ma_r = self.renderer.mot_decode(ta_r)
        final_driving_path = driving_video_path
        temp_crop_video = None
        if crop:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                temp_crop_video = tmp.name
            self.data_processor.crop_video_stable(driving_video_path, temp_crop_video)
            final_driving_path = temp_crop_video
        cap = cv2.VideoCapture(final_driving_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        vid_results = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame).resize(
                (self.opt.input_size, self.opt.input_size)
            )
            d_tensor = (
                self.data_processor.transform(frame_pil).unsqueeze(0).to(self.device)
            )
            t_c = self.renderer.mot_encode(d_tensor)
            ta_c = self.renderer.adapt(t_c, i_r)
            ma_c = self.renderer.mot_decode(ta_c)
            out = self.renderer.decode(ma_c, ma_r, f_r)
            vid_results.append(out.cpu())

            # Clear cache periodically
            frame_count += 1
            if frame_count % 10 == 0 and self.device == "mps":
                torch.mps.empty_cache()

        cap.release()
        if temp_crop_video and os.path.exists(temp_crop_video):
            os.remove(temp_crop_video)
        if not vid_results:
            raise Exception("Driving video reading failed.")
        vid_tensor = torch.cat(vid_results, dim=0)

        # Final cleanup
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        return self.save_video(vid_tensor, fps=fps, audio_path=driving_video_path)



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


# Create directories for HLS output
HLS_OUTPUT_DIR = Path("hls_output")
HLS_OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files for serving HLS segments and playlists (Use S3 or CDN later)
app.mount("/hls", StaticFiles(directory=str(HLS_OUTPUT_DIR)), name="hls")

# Store generation status
generation_status = {}


def cleanup_hls_session(session_id: str, delay_minutes: int = 5):
    """
    Schedule cleanup of HLS output directory after a delay
    
    Args:
        session_id: The session ID to clean up
        delay_minutes: Minutes to wait before cleanup (default 5)
    """
    def _cleanup():
        import time
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
        # This prevents the playlist from starting empty when resuming
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

        if self.total_frames % 25 == 0:  # Log every second of video
            print(f"[HLS {self.session_id}] Progress: {self.total_frames} frames, buffer: {len(self.frame_buffer)}/{self.frames_per_segment}")

        if len(self.frame_buffer) >= self.frames_per_segment:
            print(f"[HLS {self.session_id}] Buffer full ({len(self.frame_buffer)} frames), writing segment...")
            self._write_segment()
            self._update_playlist()

    def _load_existing_session(self):
        """Load existing segments if session already exists"""
        import glob

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
            for seg_path in existing_segments:
                seg_name = os.path.basename(seg_path)
                self.segment_files.append(seg_name)
                
                # Extract segment number from filename (e.g., "segment003.ts" -> 3)
                import re
                match = re.search(r'segment(\d+)\.ts', seg_name)
                if match:
                    seg_num = int(match.group(1))
                    if seg_num >= self.current_segment:
                        self.current_segment = seg_num + 1
        

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
            # No segments yet, player will poll for updates

    def _write_segment(self):
        """Write buffered frames as an HLS segment"""
        if not self.frame_buffer:
            print(f"[HLS {self.session_id}] No frames in buffer, skipping segment write")
            return

        segment_file = self.output_dir / f"segment{self.current_segment:03d}.ts"
        segment_name = f"segment{self.current_segment:03d}.ts"
        temp_video = self.output_dir / f"temp_segment{self.current_segment:03d}.mp4"

        # CRITICAL FIX: Check if segment already exists (prevent overwriting from concurrent chunks)
        if segment_file.exists():
            # Find all actual segments on disk
            import glob
            all_segments = sorted(glob.glob(str(self.output_dir / "segment*.ts")))

            import re
            max_seg = -1
            for seg_path in all_segments:
                match = re.search(r'segment(\d+)\.ts', os.path.basename(seg_path))
                if match:
                    max_seg = max(max_seg, int(match.group(1)))
            self.current_segment = max_seg + 1
            segment_file = self.output_dir / f"segment{self.current_segment:03d}.ts"
            segment_name = f"segment{self.current_segment:03d}.ts"
            temp_video = self.output_dir / f"temp_segment{self.current_segment:03d}.mp4"
            # Also reload all segments into our list
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

            # Calculate start time for this segment (used for audio slice AND timestamp offset)
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
            else:
                print(f"[HLS {self.session_id}] No audio provided for segment")

            # Apply timestamp offset to ensure HLS continuity
            cmd.extend(["-output_ts_offset", f"{start_time:.3f}"])

            cmd.extend([
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-f", "mpegts",
                "-y",
                str(segment_file),
            ])

            print(f"[HLS {self.session_id}] Running FFmpeg: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[HLS {self.session_id}] FFmpeg ERROR: {result.stderr}")
                raise Exception(f"FFmpeg failed: {result.stderr}")

            # Verify segment was created
            if not segment_file.exists():
                print(f"[HLS {self.session_id}] ERROR: Segment file not created!")
                raise Exception("Segment file not created")
            
            segment_size = segment_file.stat().st_size
            print(f"[HLS {self.session_id}] ✓ Segment created: {segment_file.name} ({segment_size} bytes)")

            # Clean up temp file
            temp_video.unlink()

            self.segment_files.append(segment_file.name)
            self.current_segment += 1
            self.frame_buffer = []

        except Exception as e:
            print(f"[HLS {self.session_id}] Exception in _write_segment: {e}")
            import traceback
            traceback.print_exc()
            # Clean up on error
            if temp_video.exists():
                temp_video.unlink()

    def _update_playlist(self):
        """Update the HLS playlist with current segments"""
        playlist_path = self.output_dir / "playlist.m3u8"

        # Calculate target duration (max segment duration)
        target_duration = self.segment_duration + 1

        with open(playlist_path, "w") as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n")
            f.write(f"#EXT-X-TARGETDURATION:{target_duration}\n")
            f.write("#EXT-X-PLAYLIST-TYPE:EVENT\n")
            f.write("#EXT-X-MEDIA-SEQUENCE:0\n")

            # Add all segments
            for idx, segment in enumerate(self.segment_files):
                f.write(f"#EXTINF:{self.segment_duration:.3f},\n")
                f.write(f"{segment}\n")
                print(f"[HLS {self.session_id}]   [{idx}] {segment}")

            # Add END tag only if generation is complete
            if self.is_complete:
                f.write("#EXT-X-ENDLIST\n")
                print(f"[HLS {self.session_id}] ✓✓✓ Added #EXT-X-ENDLIST tag ✓✓✓")
            else:
                print(f"[HLS {self.session_id}] No ENDLIST (stream open for more chunks)")
            
            # Flush to disk immediately
            f.flush()
            os.fsync(f.fileno())
        
        print(f"[HLS {self.session_id}] ========== PLAYLIST UPDATED ==========\n")

    def finalize(self, audio_path: Optional[str] = None, mark_complete: bool = False):
        
        # Write any remaining frames
        if self.frame_buffer:
            print(f"[HLS {self.session_id}] Writing final segment with {len(self.frame_buffer)} remaining frames")
            self._write_segment()

        # Only mark as complete if explicitly requested (when no more chunks are coming)
        if mark_complete:
            self.is_complete = True
            print(f"[HLS {self.session_id}] Stream marked as COMPLETE")
            
            # Schedule cleanup after 5 minutes
            cleanup_hls_session(self.session_id, delay_minutes=5)
        
        self._update_playlist()
        print(f"[HLS {self.session_id}] Total frames generated: {self.total_frames}")


def convert_video_to_hls(
    video_path: str, output_dir: Path, segment_duration: int = 2
) -> str:
    """
    Convert MP4 video to HLS format with .m3u8 playlist and .ts segments

    Args:
        video_path: Path to input MP4 video
        output_dir: Directory to store HLS files
        segment_duration: Duration of each segment in seconds

    Returns:
        Path to the .m3u8 playlist file
    """
    playlist_name = "playlist.m3u8"
    playlist_path = output_dir / playlist_name

    # FFmpeg command to convert to HLS
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-codec:",
        "copy",
        "-start_number",
        "0",
        "-hls_time",
        str(segment_duration),
        "-hls_list_size",
        "0",
        "-hls_segment_filename",
        str(output_dir / "segment%03d.ts"),
        "-f",
        "hls",
        str(playlist_path),
        "-y",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return str(playlist_path)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg conversion failed: {e.stderr.decode()}")


@app.post("/api/audio-driven/stream")
async def audio_driven_inference_stream(
    background_tasks: BackgroundTasks,
    unique_id: str = Form(..., description="Unique session ID"),
    image: UploadFile = File(..., description="Source image file (PNG, JPG)"),
    audio: UploadFile = File(..., description="Driving audio file (WAV, MP3)"),
    crop: bool = Form(True, description="Auto crop face from image"),
    seed: int = Form(42, description="Random seed for generation"),
    nfe: int = Form(10, description="Number of function evaluations (steps), range: 5-50"),
    cfg_scale: float = Form(2.0, description="Classifier-free guidance scale, range: 1.0-5.0"),
    format: str = Form("fmp4", description="Stream format: 'fmp4' (with audio), 'hls' (HLS playlist), 'mjpeg' (video only), or 'raw'"),
    segment_duration: float = Form(2, description="Chunk/segment duration in seconds (1.0-10.0, default: 6.4 = 160 frames)"),
    fps: float = Form(25.0, description="Frames per second for output video"),
    pose_style: Optional[str] = Form(None, description="Optional pose style as JSON array [pitch, yaw, roll]"),
    gaze_style: Optional[str] = Form(None, description="Optional gaze style as JSON array [x, y]"),
    is_final_chunk: bool = Form(False, description="Set to True for the last chunk to finalize the HLS stream"),
):
    """
    Generate talking face video from image and audio with progressive HLS streaming
    Returns immediately with HLS URL - video starts streaming as soon as first segments are ready
    
    Args:
        image: Source image containing a face
        audio: Audio file to drive the lip movements
        crop: Whether to automatically crop and detect face
        seed: Random seed for reproducibility
        nfe: Number of sampling steps (higher = better quality but slower)
        cfg_scale: Guidance scale for generation quality
        format: Output format ('fmp4', 'hls', 'mjpeg', 'raw')
        segment_duration: Duration of each segment in seconds

    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")

    # Create unique session ID
    session_id = unique_id or str(uuid.uuid4())
    session_dir = HLS_OUTPUT_DIR / session_id
    
    # Check if session already exists
    session_exists = session_dir.exists()
    if session_exists:
        # Check if there's already an HLS playlist
        playlist_path = session_dir / "playlist.m3u8"
    else:
        print(f"[{session_id}] New session, creating directory")
    
    session_dir.mkdir(exist_ok=True)
    
    try:
        # Detect image extension - fallback to .png if not found
        image_ext = os.path.splitext(image.filename)[1] if image.filename else ""
        if not image_ext or image_ext.lower() not in ['.png', '.jpg', '.jpeg']:
            image_ext = ".png"
        
        # Detect audio extension - fallback to .wav if not found
        audio_ext = os.path.splitext(audio.filename)[1] if audio.filename else ""
        if not audio_ext or audio_ext.lower() not in ['.wav', '.mp3', '.m4a']:
            audio_ext = ".wav"
        
        print(f"[{session_id}] Detected extensions: image={image_ext}, audio={audio_ext}, image.filename={image.filename}, audio.filename={audio.filename}")
        
        # Save uploaded image
        with tempfile.NamedTemporaryFile(suffix=image_ext, delete=False) as tmp:
            img_path = tmp.name
            shutil.copyfileobj(image.file, tmp)
        
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(suffix=audio_ext, delete=False) as tmp:
            aud_path = tmp.name
            shutil.copyfileobj(audio.file, tmp)
        
        print(f"[{session_id}] Saved files: img={img_path}, aud={aud_path}")
        
        # Load and process image
        img_pil = Image.open(img_path).convert('RGB')

        # Initialize progressive HLS generator FIRST (creates initial playlist)
        # Pass audio path so segments can include audio
        hls_generator = ProgressiveHLSGenerator(
            session_id, fps=fps, segment_duration=segment_duration, audio_path=aud_path
        )

        # Verify playlist was created
        playlist_path = session_dir / "playlist.m3u8"
        if not playlist_path.exists():
            raise Exception("Failed to create initial playlist")

        # Store generation status
        generation_status[session_id] = {
            "status": "generating",
            "progress": 0,
            "total_frames": 0,
        }

        # Parse JSON styles
        pose_arr = json.loads(pose_style) if pose_style else None
        gaze_arr = json.loads(gaze_style) if gaze_style else None

        def generate_video():
            try:
                print(f"[{session_id}] Starting background generation...")

                # Load and process image inside background task
                img_pil = Image.open(img_path).convert("RGB")
                print(f"[{session_id}] Image loaded successfully")

                # Run progressive inference
                print(f"[{session_id}] Starting progressive inference with is_final_chunk={is_final_chunk}...")
                agent.run_audio_inference_progressive(
                    img_pil,
                    aud_path,
                    crop,
                    seed,
                    nfe,
                    cfg_scale,
                    hls_generator,
                    pose_style=pose_arr,
                    gaze_style=gaze_arr,
                    is_final_chunk=is_final_chunk,
                )
                print(f"[{session_id}] Progressive inference completed")

                # Update status
                generation_status[session_id]["status"] = "complete"
                generation_status[session_id]["progress"] = 100

                # Clean up temporary files
                os.remove(img_path)
                os.remove(aud_path)
                print(f"[{session_id}] Cleanup complete")

            except Exception as e:
                generation_status[session_id]["status"] = "error"
                generation_status[session_id]["error"] = str(e)
                print(f"[{session_id}] Generation error: {e}")
                import traceback

                traceback.print_exc()

        # Start generation in background
        background_tasks.add_task(generate_video)

        # Return immediately with HLS URL
        hls_url = f"/hls/{session_id}/playlist.m3u8"

        return {
            "status": "success",
            "session_id": session_id,
            "hls_url": hls_url,
            "message": "Video generation started. Stream will begin as soon as first segments are ready.",
            "streaming": "progressive",
        }

    except Exception as e:
        # Clean up on error
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hls/{session_id}/playlist.m3u8")
async def get_playlist(session_id: str):
    """
    Get HLS playlist file for a session
    """
    playlist_path = HLS_OUTPUT_DIR / session_id / "playlist.m3u8"

    if not playlist_path.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    return FileResponse(
        playlist_path,
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/hls/{session_id}/segment{segment_num}.ts")
async def get_segment(session_id: str, segment_num: str):
    """
    Get HLS video segment for a session
    """
    segment_path = HLS_OUTPUT_DIR / session_id / f"segment{segment_num}.ts"

    if not segment_path.exists():
        raise HTTPException(status_code=404, detail="Segment not found")

    return FileResponse(
        segment_path,
        media_type="video/mp2t",
        headers={"Cache-Control": "public, max-age=31536000"},
    )


@app.delete("/api/hls/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete HLS files for a session to free up space
    """
    session_dir = HLS_OUTPUT_DIR / session_id

    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        shutil.rmtree(session_dir)
        return {"status": "success", "message": f"Session {session_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/audio-driven")
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


@app.post("/api/video-driven")
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


@app.get("/api/outputs/{filename}")
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


@app.delete("/api/outputs/{filename}")
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


@app.get("/api/outputs")
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
