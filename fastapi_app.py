"""
FastAPI wrapper for IMTalker video rendering inference server.
Provides REST API endpoints for audio-driven and video-driven talking face generation.
"""

import os
import sys
import io
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
import base64
import gradio as gr
import glob
import re
import io
import traceback
import struct
from PIL import Image
import torchvision.transforms as transforms
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm
import random
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request, WebSocket
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
from parallel_renderer import ParallelRenderer
from realtime_streaming import RealtimeSession

# Try importing local modules
try:
    from generator.FM import FMGenerator
    from renderer.models import IMTRenderer
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure 'generator' and 'renderer' folders are in the same directory.")

try: 
    from trt_handler import TRTInferenceHandler
except ImportError:
    print("TRT Handler not found/importable.")
    TRTInferenceHandler = None


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
trt_handler = None
output_dir = "./outputs"

# HLS session storage for live streaming
import threading
from dataclasses import dataclass, field
from typing import Dict, List

# Import librosa for audio resampling
try:
    import librosa
except ImportError:
    print("Warning: librosa not installed. Audio resampling may not work. Install with: pip install librosa")
    librosa = None


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
                    print(f"[AudioBuffer {self.session_id}] Resampling {len(audio_chunk)} samples from {source_sr}Hz → {self.target_sr}Hz")
                    audio_chunk = librosa.resample(
                        audio_chunk, 
                        orig_sr=source_sr, 
                        target_sr=self.target_sr
                    )
                
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
realtime_sessions: Dict[str, RealtimeSession] = {}


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
        
        # Initialize parallel renderer for performance optimization
        self.parallel_renderer = ParallelRenderer(
            agent=self,
            num_workers=opt.num_render_workers,
            chunk_size=opt.render_chunk_size
        )
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
        is_streaming=False,
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

        T = sample.shape[1]
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)

        if is_streaming:
            # PARALLEL RENDERING - Up to 2.5x faster!
            # print(f"Rendering {T} frames using parallel renderer ({self.opt.num_render_workers} workers)...")
            # d_hat = self.parallel_renderer.render_parallel(
            #     sample,
            #     g_r,
            #     m_r,
            #     f_r,
            #     progress_callback=lambda curr,
            #     total: print(f"  Progress: {curr}/{total} chunks")
            # )
            d_hat = []
            for t in range(T):
                ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.renderer.latent_token_decoder(ta_c)
                out_frame = self.renderer.decode(m_c, m_r, f_r)
                # Move to CPU immediately to save memory
                d_hat.append(out_frame.cpu())
                # Clear cache periodically
                if t % 10 == 0 and self.device == "mps":
                    torch.mps.empty_cache()
        else:
            # STANDARD RENDERING - Single-threaded
            print(f"Rendering {T} frames...")
            d_hat = []
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

        # PROGRESSIVE PARALLEL HLS RENDERING - Streams as it renders!
        print(f"Progressive HLS: Rendering {T} frames with {self.opt.num_render_workers} workers...")
        # self.parallel_renderer.render_progressive_hls(
        #     sample, g_r, m_r, f_r,
        #     hls_generator
        # )

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

    @torch.no_grad()
    def run_audio_inference_chunked(
        self,
        img_pil,
        audio_chunk_np: np.ndarray,
        crop,
        seed,
        nfe,
        cfg_scale,
        hls_generator: "ProgressiveHLSGenerator",
        pose_style=None,
        gaze_style=None,
    ):
        """
        Run audio inference with a single audio chunk (real-time streaming mode).
        Processes one audio chunk at a time and streams frames immediately.
        
        Args:
            img_pil: PIL Image of source face
            audio_chunk_np: Single audio chunk (numpy array, already resampled to 16kHz)
            crop: Whether to crop face
            seed: Random seed
            nfe: Number of function evaluations
            cfg_scale: Classifier-free guidance scale
            hls_generator: HLS generator for writing frames
            pose_style: Optional pose parameters
            gaze_style: Optional gaze parameters
        """
        # Clear memory
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        try:
            # Image preprocessing (only once per session, done by caller)
            s_pil = (
                self.data_processor.process_img(img_pil)
                if crop
                else img_pil.resize((self.opt.input_size, self.opt.input_size))
            )
            s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
            
            # Convert numpy chunk to torch tensor (16kHz audio)
            # audio_chunk_np is shape (16000,) or similar - wav2vec expects this
            audio_feat = torch.from_numpy(audio_chunk_np).float().unsqueeze(0).to(self.device)
            
            # Process with feature extractor (Wav2Vec2)
            a_tensor = audio_feat  # Already in correct format
            
            data = {
                "s": s_tensor,
                "a": a_tensor,
                "pose": None,
                "cam": None,
                "gaze": None,
                "ref_x": None,
            }

            # Calculate target frame count T based on audio chunk length
            T = math.ceil(a_tensor.shape[-1] * self.opt.fps / self.opt.sampling_rate)

            if pose_style is not None:
                p_tensor = (
                    torch.tensor(pose_style, device=self.device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(1, T, 3)
                )
                data["pose"] = p_tensor

            if gaze_style is not None:
                g_tensor = (
                    torch.tensor(gaze_style, device=self.device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(1, T, 2)
                )
                data["gaze"] = g_tensor

            # Feature extraction
            f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
            t_lat = self.renderer.latent_token_encoder(s_tensor)
            if isinstance(t_lat, tuple):
                t_lat = t_lat[0]
            data["ref_x"] = t_lat

            # Generate motion
            torch.manual_seed(seed)
            sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)

            # Stabilization
            motion_scale = 0.6
            sample = (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) + motion_scale * sample

            T = sample.shape[1]
            ta_r = self.renderer.adapt(t_lat, g_r)
            m_r = self.renderer.latent_token_decoder(ta_r)

            # Render frames progressively
            frames_rendered = 0
            for t in range(T):
                ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.renderer.latent_token_decoder(ta_c)
                out_frame = self.renderer.decode(m_c, m_r, f_r)

                # Convert to numpy
                frame_np = out_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                if frame_np.min() < 0:
                    frame_np = (frame_np + 1) / 2
                frame_np = np.clip(frame_np, 0, 1)
                frame_np = (frame_np * 255).astype(np.uint8)

                # Add to HLS generator
                hls_generator.add_frame(frame_np)
                frames_rendered += 1

                # Clear cache periodically
                if t % 10 == 0 and self.device == "mps":
                    torch.mps.empty_cache()

            print(f"[ChunkedInference] Rendered {frames_rendered} frames from audio chunk")
            return frames_rendered

        except Exception as e:
            print(f"[ChunkedInference] Error during inference: {e}")
            traceback.print_exc()
            raise
        finally:
            # Final cleanup
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

    @torch.no_grad()
    def preprocess_for_realtime(self, img_pil, crop=True):
        """Pre-calculate static features for real-time inference (run once per session)"""
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        # Image processing
        s_pil = (
            self.data_processor.process_img(img_pil)
            if crop
            else img_pil.resize((self.opt.input_size, self.opt.input_size))
        )
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        
        # Feature extraction (static for the session)
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
            
        return {
            "s_tensor": s_tensor,
            "f_r": f_r,
            "g_r": g_r,
            "t_lat": t_lat,
            "ref_x": t_lat
        }

    @torch.no_grad()
    def warmup_realtime(self):
        """Optional warmup to ensure CUDA context is ready"""
        pass

    @torch.no_grad()
    def run_realtime_step(self, preprocessed, audio_chunk, is_silence=False):
        """
        Run inference for a single frame.
        
        Args:
            preprocessed: Dict from preprocess_for_realtime
            audio_chunk: Float32 numpy array (1 frame worth of audio)
            is_silence: Bool indicating if this is silence (optimization hint)
        """
        # audio_chunk needs to be compatible with wav2vec2 feature extractor?
        # DataProcessor.process_audio usually loads from file.
        # Here we have raw numpy.
        
        a_tensor = torch.from_numpy(audio_chunk).float().to(self.device)
        if a_tensor.dim() == 1:
            a_tensor = a_tensor.unsqueeze(0) # (1, L)
            
        data = {
            "s": preprocessed["s_tensor"],
            "a": a_tensor,
            "pose": None,
            "cam": None,
            "gaze": None,
            "ref_x": preprocessed["ref_x"]
        }
        
        # Sampling
        nfe = 10 
        cfg_scale = 2.0
        seed = random.randint(0, 1000000)
        
        torch.manual_seed(seed)
        
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        
        # Stabilization
        motion_scale = 0.6
        t_lat = preprocessed["t_lat"]
        sample = (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) + motion_scale * sample
        
        # Decode (First frame only)
        g_r = preprocessed["g_r"]
        f_r = preprocessed["f_r"]
        m_r = self.renderer.latent_token_decoder(self.renderer.adapt(t_lat, g_r))
        
        ta_c = self.renderer.adapt(sample[:, 0, ...], g_r)
        m_c = self.renderer.latent_token_decoder(ta_c)
        out_frame = self.renderer.decode(m_c, m_r, f_r)
        
        # To JPEG
        frame = out_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        if frame.min() < 0:
            frame = (frame + 1) / 2
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        
        # RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Compress to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return jpeg.tobytes()


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

            # Initialize TRT Handler
            if TRTInferenceHandler is not None:
                global trt_handler
                try:
                    trt_handler = TRTInferenceHandler(agent)
                except Exception as e:
                    print(f"Failed to initialize TRT Handler: {e}")
                    trt_handler = None

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
            "trt_available": trt_handler.is_available() if trt_handler else False,
        }
    }


@app.post("/crop")
async def crop_image(
    file: UploadFile = File(..., description="Image file to crop"),
    crop_scale: float = Form(0.8, description="Crop scale factor (0.5=tight, 0.8=default, 1.2=loose)", ge=0.3, le=2.0),
    output_size: Optional[int] = Form(None, description="Output size in pixels (square)", ge=64, le=2048),
    format: str = Form("jpeg", description="Output format (jpeg, png, webp)")
):
    """
    Crop image centered on detected face with configurable padding.
    
    This endpoint uses face detection to intelligently crop images around detected faces.
    If no face is detected, it falls back to center crop.
    
    Args:
        file: Image file (JPEG, PNG, WebP)
        crop_scale: Scale factor for crop padding (0.5=tight, 0.8=default, 1.2=loose)
        output_size: Optional output size in pixels (square)
        format: Output image format (jpeg, png, webp)
        
    Returns:
        Cropped image file
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate format
    format = format.lower()
    if format not in ["jpeg", "jpg", "png", "webp"]:
        raise HTTPException(status_code=400, detail="Invalid format. Use jpeg, png, or webp")
    
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Use the existing DataProcessor to crop the image
        # Temporarily set crop_scale on the processor
        original_crop_scale = agent.data_processor.crop_scale
        agent.data_processor.crop_scale = crop_scale
        
        try:
            # Process image (crop around face)
            cropped_img = agent.data_processor.process_img(img)
            
            # Resize if output_size specified
            if output_size:
                cropped_img = cropped_img.resize((output_size, output_size), Image.LANCZOS)
        finally:
            # Restore original crop_scale
            agent.data_processor.crop_scale = original_crop_scale
        
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
        
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type=media_type)
        
    except Exception as e:
        print(f"Crop error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Crop failed: {str(e)}")


# Create directories for HLS output
HLS_OUTPUT_DIR = Path("hls_output")
HLS_OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files for serving HLS segments and playlists (Use S3 or CDN later)
app.mount("/hls", StaticFiles(directory=str(HLS_OUTPUT_DIR)), name="hls")

# Store generation status
generation_status = {}


def cleanup_hls_session(session_id: str, delay_minutes: int = 50):
    """
    Schedule cleanup of HLS output directory after a delay
    
    Args:
        session_id: The session ID to clean up
        delay_minutes: Minutes to wait before cleanup (default 50)
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

        if len(self.frame_buffer) >= self.frames_per_segment:
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
            max_segment_num = -1
            for seg_path in existing_segments:
                seg_name = os.path.basename(seg_path)
                self.segment_files.append(seg_name)
                
                # Extract segment number from filename (e.g., "segment003.ts" -> 3)
                import re
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
        temp_playlist_path = self.output_dir / "playlist.m3u8.tmp"

        try:
            # CRITICAL: Always re-scan directory for ALL segments to handle multi-chunk scenarios
            # This ensures we don't lose segments from previous chunks
            all_segments = sorted(glob.glob(str(self.output_dir / "segment*.ts")))
            all_segment_names = [os.path.basename(seg) for seg in all_segments]
            
            # Update our internal list to match disk state
            self.segment_files = all_segment_names
            
            print(f"[HLS {self.session_id}] Updating playlist with {len(all_segment_names)} total segments")

            # Calculate target duration (max segment duration)
            target_duration = self.segment_duration + 1

            # Write to temporary file first to avoid partial reads
            with open(temp_playlist_path, "w") as f:
                f.write("#EXTM3U\n")
                f.write("#EXT-X-VERSION:3\n")
                f.write(f"#EXT-X-TARGETDURATION:{target_duration}\n")
                f.write("#EXT-X-PLAYLIST-TYPE:EVENT\n")
                f.write("#EXT-X-MEDIA-SEQUENCE:0\n")

                # Add all segments from disk
                for idx, segment in enumerate(all_segment_names):
                    f.write(f"#EXTINF:{self.segment_duration:.3f},\n")
                    f.write(f"{segment}\n")
                    if idx < 5 or idx >= len(all_segment_names) - 2:  # Log first 5 and last 2
                        print(f"[HLS {self.session_id}]   [{idx}] {segment}")
                
                if len(all_segment_names) > 7:
                    print(f"[HLS {self.session_id}]   ... ({len(all_segment_names) - 7} more segments)")

                # Add END tag only if generation is complete
                if self.is_complete:
                    f.write("#EXT-X-ENDLIST\n")
                    print(f"[HLS {self.session_id}] ✓✓✓ Added #EXT-X-ENDLIST tag ✓✓✓")
                else:
                    print(f"[HLS {self.session_id}] No ENDLIST (stream open for more chunks)")
                
                # Flush to disk immediately
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename to replace the old playlist
            temp_playlist_path.replace(playlist_path)
            
            # Set proper permissions
            try:
                os.chmod(playlist_path, 0o644)
            except Exception as e:
                print(f"[HLS {self.session_id}] Warning: Could not set playlist permissions: {e}")
            
        except Exception as e:
            print(f"[HLS {self.session_id}] Error updating playlist: {e}")
            import traceback
            traceback.print_exc()
            # Clean up temp file if it exists
            if temp_playlist_path.exists():
                temp_playlist_path.unlink()
            raise
        
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
            
            # Schedule cleanup after 50 minutes
            cleanup_hls_session(self.session_id, delay_minutes=50)
        
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
    # cmd = [
    #     "ffmpeg",
    #     "-i",
    #     video_path,
    #     "-codec:",
    #     "copy",
    #     "-start_number",
    #     "0",
    #     "-hls_time",
    #     str(segment_duration),
    #     "-hls_list_size",
    #     "0",
    #     "-hls_segment_filename",
    #     str(output_dir / "segment%03d.ts"),
    #     "-f",
    #     "hls",
    #     str(playlist_path),
    #     "-y",
    # ]
    cmd = [
        "ffmpeg",
        "-re",
        "-i", video_path,

        # Video
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-keyint_min", "48",
        "-sc_threshold", "0",

        # Audio
        "-c:a", "aac",
        "-b:a", "128k",

        # HLS
        "-hls_time", str(segment_duration),
        "-hls_list_size", "6",
        "-hls_flags", "delete_segments+independent_segments",
        "-hls_segment_filename", str(output_dir / "segment_%03d.ts"),
        "-f", "hls",
        str(output_dir / "playlist.m3u8"),

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
    crop_scale: float = Form(0.8, description="Crop scale factor"),
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
        crop_scale: Crop scale factor
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")

    # Create unique session ID
    session_id = unique_id or str(uuid.uuid4())
    session_dir = HLS_OUTPUT_DIR / session_id
    
    # Check if session already exists
    session_exists = session_dir.exists()
    if session_exists:
        print(f"[{session_id}] Resuming existing session")
    else:
        print(f"[{session_id}] New session, creating directory")
        session_dir.mkdir(parents=True, exist_ok=True)
        # Set proper permissions for the directory
        try:
            os.chmod(session_dir, 0o755)
        except Exception as e:
            print(f"[{session_id}] Warning: Could not set directory permissions: {e}")
    
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
        
        # Load and process image (only once, reuse for subsequent chunks)
        img_pil = Image.open(img_path).convert('RGB')

        # CRITICAL FIX: Only create initial playlist for NEW sessions
        # For existing sessions, the generator will resume without recreating the playlist
        if not session_exists:
            # Initialize progressive HLS generator FIRST (creates initial playlist)
            # Pass audio path so segments can include audio
            hls_generator = ProgressiveHLSGenerator(
                session_id, fps=fps, segment_duration=segment_duration, audio_path=aud_path
            )

            # Verify playlist was created
            playlist_path = session_dir / "playlist.m3u8"
            if not playlist_path.exists():
                raise Exception("Failed to create initial playlist")
            
            print(f"[{session_id}] Initial playlist created for new session")
        else:
            print(f"[{session_id}] Resuming existing session - will create generator in background task")

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

                # Temporarily set crop_scale for this request
                original_crop_scale = agent.data_processor.crop_scale
                agent.data_processor.crop_scale = crop_scale

                # Create HLS generator inside background task (safe for concurrent chunks)
                # This will properly resume existing sessions
                hls_generator = ProgressiveHLSGenerator(
                    session_id, fps=fps, segment_duration=segment_duration, audio_path=aud_path
                )
                print(f"[{session_id}] HLS generator initialized/resumed")

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

                # Restore original crop_scale
                agent.data_processor.crop_scale = original_crop_scale

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


# ============================================================================
# REAL-TIME CHUNKED AUDIO-DRIVEN INFERENCE
# ============================================================================

@app.post("/api/audio-driven/chunked/init")
async def chunked_stream_init(
    image: UploadFile = File(..., description="Source image file (PNG, JPG)"),
    session_id: Optional[str] = Form(None, description="Optional session ID (auto-generated if not provided)"),
    crop: bool = Form(True, description="Auto crop face from image"),
    seed: int = Form(42, description="Random seed for generation"),
    nfe: int = Form(10, description="Number of function evaluations (steps), range: 5-50"),
    cfg_scale: float = Form(2.0, description="Classifier-free guidance scale, range: 1.0-5.0"),
    chunk_duration: float = Form(1.0, description="Duration of each audio chunk to process (seconds)"),
    audio_sample_rate: int = Form(24000, description="Sample rate of incoming audio chunks (Hz)"),
    pose_style: Optional[str] = Form(None, description="Optional pose style as JSON array [pitch, yaw, roll]"),
    gaze_style: Optional[str] = Form(None, description="Optional gaze style as JSON array [x, y]"),
):
    """
    Initialize a real-time chunked audio-driven inference session.
    Call this ONCE at the start, then send audio chunks via /api/audio-driven/chunked/feed.
    
    Returns:
        session_id: Use this to feed audio chunks and get the HLS stream
        hls_url: URL to HLS playlist for real-time video display
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")
    
    # Validate parameters
    if nfe < 5 or nfe > 50:
        raise HTTPException(status_code=400, detail="nfe must be between 5 and 50")
    if cfg_scale < 1.0 or cfg_scale > 5.0:
        raise HTTPException(status_code=400, detail="cfg_scale must be between 1.0 and 5.0")
    if chunk_duration < 0.1 or chunk_duration > 10.0:
        raise HTTPException(status_code=400, detail="chunk_duration must be between 0.1 and 10.0 seconds")
    
    try:
        # Generate session ID if not provided
        sid = session_id or str(uuid.uuid4())
        
        # Create HLS output directory
        session_dir = HLS_OUTPUT_DIR / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[ChunkedInit {sid}] Initializing chunked stream session")
        
        # Load and validate image
        img_data = await image.read()
        img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
        print(f"[ChunkedInit {sid}] Image loaded: {img_pil.size}")
        
        # Initialize audio buffer
        audio_buffer = AudioChunkBuffer(
            session_id=sid,
            target_sr=16000,  # wav2vec2 requires 16kHz
            chunk_duration=chunk_duration,
            overlap_ratio=0.2
        )
        
        # Initialize HLS generator
        hls_generator = ProgressiveHLSGenerator(
            session_id=sid,
            fps=cfg.fps if cfg else 25.0,
            segment_duration=2,
            audio_path=None  # Audio will be added per-chunk
        )
        
        # Parse pose/gaze styles
        pose_arr = json.loads(pose_style) if pose_style else None
        gaze_arr = json.loads(gaze_style) if gaze_style else None
        
        # Create session object
        session = ChunkedStreamSession(
            session_id=sid,
            image_pil=img_pil,
            audio_buffer=audio_buffer,
            hls_generator=hls_generator,
            parameters={
                "crop": crop,
                "seed": seed,
                "nfe": nfe,
                "cfg_scale": cfg_scale,
                "audio_sample_rate": audio_sample_rate,
                "pose_style": pose_arr,
                "gaze_style": gaze_arr,
            }
        )
        
        # Store session
        chunked_sessions[sid] = session
        
        print(f"[ChunkedInit {sid}] ✓ Session initialized. Ready for audio chunks.")
        
        return {
            "status": "success",
            "session_id": sid,
            "hls_url": f"/hls/{sid}/playlist.m3u8",
            "message": "Session ready. Send audio chunks to /api/audio-driven/chunked/feed",
            "chunk_duration": chunk_duration,
            "audio_sample_rate": audio_sample_rate,
        }
        
    except Exception as e:
        print(f"[ChunkedInit] Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/audio-driven/chunked/feed")
async def chunked_stream_feed(
    background_tasks: BackgroundTasks,
    session_id: str = Form(..., description="Session ID from init"),
    audio_chunk: UploadFile = File(..., description="Audio chunk (WAV, MP3, etc.)"),
    is_final_chunk: bool = Form(False, description="Set to True for the last chunk"),
):
    """
    Feed an audio chunk to an active chunked streaming session.
    Processes audio immediately and streams generated frames via HLS.
    
    Call this repeatedly with audio chunks until is_final_chunk=True.
    """
    if session_id not in chunked_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found. Call /api/audio-driven/chunked/init first.")
    
    session = chunked_sessions[session_id]
    
    try:
        print(f"[ChunkedFeed {session_id}] Received audio chunk (final={is_final_chunk})")
        
        # Load audio chunk
        audio_data = await audio_chunk.read()
        
        # Use librosa to load audio and get sample rate
        if librosa is None:
            raise HTTPException(status_code=500, detail="librosa not installed. Cannot process audio chunks.")
        
        audio_chunk_np, sr = librosa.load(io.BytesIO(audio_data), sr=None)
        print(f"[ChunkedFeed {session_id}] Loaded audio: {len(audio_chunk_np)} samples at {sr}Hz")
        
        # Add to buffer (auto-resamples to 16kHz)
        has_enough = session.audio_buffer.add_chunk(audio_chunk_np, source_sr=sr)
        
        def process_audio_chunks():
            """Background task to process audio chunks and generate video frames"""
            try:
                session.status = "processing"
                
                # Process all available chunks in the buffer
                while session.audio_buffer.has_pending_audio():
                    audio_chunk_16k = session.audio_buffer.get_chunk()
                    if audio_chunk_16k is None:
                        break
                    
                    print(f"[ChunkedFeed {session_id}] Processing audio chunk: {len(audio_chunk_16k)} samples")
                    
                    # Run inference on this chunk
                    frames = agent.run_audio_inference_chunked(
                        img_pil=session.image_pil,
                        audio_chunk_np=audio_chunk_16k,
                        crop=session.parameters["crop"],
                        seed=session.parameters["seed"],
                        nfe=session.parameters["nfe"],
                        cfg_scale=session.parameters["cfg_scale"],
                        hls_generator=session.hls_generator,
                        pose_style=session.parameters["pose_style"],
                        gaze_style=session.parameters["gaze_style"],
                    )
                    
                    session.total_frames_generated += frames
                    print(f"[ChunkedFeed {session_id}] Total frames generated so far: {session.total_frames_generated}")
                
                # If this is the final chunk, process any remaining audio
                if is_final_chunk:
                    final_chunk = session.audio_buffer.finalize()
                    if final_chunk is not None:
                        print(f"[ChunkedFeed {session_id}] Processing final audio chunk")
                        frames = agent.run_audio_inference_chunked(
                            img_pil=session.image_pil,
                            audio_chunk_np=final_chunk,
                            crop=session.parameters["crop"],
                            seed=session.parameters["seed"],
                            nfe=session.parameters["nfe"],
                            cfg_scale=session.parameters["cfg_scale"],
                            hls_generator=session.hls_generator,
                            pose_style=session.parameters["pose_style"],
                            gaze_style=session.parameters["gaze_style"],
                        )
                        session.total_frames_generated += frames
                    
                    # Mark HLS stream as complete
                    session.hls_generator.finalize(mark_complete=True)
                    session.status = "complete"
                    print(f"[ChunkedFeed {session_id}] ✓ Stream finalized. Total frames: {session.total_frames_generated}")
                    
                    # Schedule cleanup
                    cleanup_hls_session(session_id, delay_minutes=50)
                else:
                    session.status = "waiting_for_audio"
                    
            except Exception as e:
                session.status = "error"
                session.error_message = str(e)
                print(f"[ChunkedFeed {session_id}] Error processing audio: {e}")
                traceback.print_exc()
        
        # Start processing in background if we have enough audio buffered
        if has_enough or is_final_chunk:
            background_tasks.add_task(process_audio_chunks)
        
        return {
            "status": "success",
            "session_id": session_id,
            "buffer_samples": session.audio_buffer.total_samples_received,
            "frames_generated": session.total_frames_generated,
            "session_status": session.status,
            "message": "Audio chunk received and queued for processing"
        }
        
    except Exception as e:
        session.status = "error"
        session.error_message = str(e)
        print(f"[ChunkedFeed {session_id}] Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audio-driven/chunked/status/{session_id}")
async def chunked_stream_status(session_id: str):
    """Get the status of an active chunked streaming session"""
    if session_id not in chunked_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = chunked_sessions[session_id]
    
    return {
        "session_id": session_id,
        "status": session.status,
        "total_frames_generated": session.total_frames_generated,
        "buffer_samples": session.audio_buffer.total_samples_received,
        "error": session.error_message,
        "created_at": session.created_at,
        "uptime_seconds": time.time() - session.created_at,
    }


@app.delete("/api/audio-driven/chunked/session/{session_id}")
async def chunked_stream_cleanup(session_id: str):
    """Manually cleanup a chunked streaming session"""
    if session_id not in chunked_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    try:
        # Remove session
        del chunked_sessions[session_id]
        
        # Remove HLS files
        session_dir = HLS_OUTPUT_DIR / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
        
        return {
            "status": "success",
            "message": f"Session {session_id} cleaned up"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hls/{session_id}/playlist.m3u8")
async def get_playlist(session_id: str):
    """
    Get HLS playlist file for a session.
    
    Waits for the first segment to be ready before returning the playlist.
    This ensures the playlist contains at least one #EXTINF entry and segment_000.ts exists.
    """
    playlist_path = HLS_OUTPUT_DIR / session_id / "playlist.m3u8"
    segment_path = HLS_OUTPUT_DIR / session_id / "segment000.ts"

    if not playlist_path.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    # Wait up to 60 seconds for first segment and valid playlist
    max_wait_seconds = 60
    check_interval = 0.1  # Check every 100ms
    elapsed = 0

    while elapsed < max_wait_seconds:
        # Check if first segment exists
        if not segment_path.exists():
            await asyncio.sleep(check_interval)
            elapsed += check_interval
            continue

        # Check if playlist contains at least one #EXTINF entry
        try:
            with open(playlist_path, 'r') as f:
                content = f.read()
                if '#EXTINF' in content:
                    # Playlist is ready!
                    return FileResponse(
                        playlist_path,
                        media_type="application/vnd.apple.mpegurl",
                        headers={"Cache-Control": "no-cache"},
                    )
        except Exception:
            pass

        await asyncio.sleep(check_interval)
        elapsed += check_interval

    # Timeout - still waiting for first segment
    raise HTTPException(
        status_code=202,
        detail=f"Playlist not ready yet. Waiting for first segment to be generated (segment_000.ts must exist and playlist must contain #EXTINF)"
    )

@app.get("/api/hls/{session_id}/segment{segment_num}.ts")
async def get_segment(
    session_id: str,
    segment_num: str,
    request: Request
):
    segment_path = HLS_OUTPUT_DIR / session_id / f"segment{segment_num}.ts"

    if not segment_path.exists():
        raise HTTPException(status_code=404, detail="Segment not found")

    file_size = os.path.getsize(segment_path)
    range_header = request.headers.get("range")

    if not range_header:
        # hls.js ALWAYS sends Range — if missing, reject
        raise HTTPException(status_code=416, detail="Range header required")

    start, end = range_header.replace("bytes=", "").split("-")
    start = int(start)
    end = int(end) if end else file_size - 1

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

    return StreamingResponse(
        iterfile(),
        status_code=206,
        headers=headers,
    )

@app.get("/api/hls/{session_id}/download")
async def download_full_video(session_id: str):
    """
    Download the complete video by merging all HLS segments into a single MP4 file.
    
    Args:
        session_id: The HLS session ID
        
    Returns:
        Complete MP4 video file
    """
    session_dir = HLS_OUTPUT_DIR / session_id
    playlist_path = session_dir / "playlist.m3u8"
    
    if not session_dir.exists() or not playlist_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Output merged video path
    output_video = session_dir / "complete_video.mp4"
    
    # If already merged, return it
    if output_video.exists():
        return FileResponse(
            output_video,
            media_type="video/mp4",
            filename=f"{session_id}.mp4",
            headers={"Content-Disposition": f"attachment; filename={session_id}.mp4"}
        )
    
    try:
        # Use FFmpeg to merge HLS segments into MP4
        # This reads the playlist and concatenates all segments
        cmd = [
            "ffmpeg",
            "-i", str(playlist_path),
            "-c", "copy",  # Copy without re-encoding for speed
            "-bsf:a", "aac_adtstoasc",  # Fix AAC stream
            "-y",
            str(output_video)
        ]
        
        print(f"[DOWNLOAD] Merging HLS segments for session {session_id}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[DOWNLOAD] FFmpeg error: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to merge video segments: {result.stderr}"
            )
        
        if not output_video.exists():
            raise HTTPException(status_code=500, detail="Failed to create merged video")
        
        print(f"[DOWNLOAD] Video merged successfully: {output_video.stat().st_size} bytes")
        
        return FileResponse(
            output_video,
            media_type="video/mp4",
            filename=f"{session_id}.mp4",
            headers={"Content-Disposition": f"attachment; filename={session_id}.mp4"}
        )
        
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg merge failed: {str(e)}")
    except Exception as e:
        print(f"[DOWNLOAD] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
    crop_scale: float = Form(0.8, description="Crop padding scale (0.5=tight, 0.8=default, 1.2=loose)"),
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
        "cfg_scale": cfg_scale,
        "crop_scale": crop_scale
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
        
        # Temporarily set crop_scale for this request
        original_crop_scale = agent.data_processor.crop_scale
        agent.data_processor.crop_scale = crop_scale
        
        # Run inference
        print(f"Running audio-driven inference: crop={crop}, crop_scale={crop_scale}, seed={seed}, nfe={nfe}, cfg_scale={cfg_scale}")
        output_video_path = agent.run_audio_inference(
            img_pil, 
            temp_audio_path, 
            crop, 
            seed, 
            nfe, 
            cfg_scale
        )
        
        # Restore original crop_scale
        agent.data_processor.crop_scale = original_crop_scale
        
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


# ============================================================================
# REAL-TIME STREAMING ENDPOINTS (Frame-by-Frame, Lowest Latency)
# ============================================================================

@app.post("/api/realtime-sse")
async def realtime_streaming_sse(
    image: UploadFile = File(..., description="Source image file (PNG, JPG)"),
    audio: UploadFile = File(..., description="Driving audio file (WAV, MP3)"),
    crop: bool = Form(True, description="Auto crop face from image"),
    seed: int = Form(42, description="Random seed for generation"),
    nfe: int = Form(10, description="Number of function evaluations (steps), range: 5-50"),
    cfg_scale: float = Form(2.0, description="Classifier-free guidance scale, range: 1.0-5.0"),
    pose_style: Optional[str] = Form(None, description="Optional pose style as JSON array [pitch, yaw, roll]"),
    gaze_style: Optional[str] = Form(None, description="Optional gaze style as JSON array [x, y]"),
):
    """
    REAL-TIME frame-by-frame streaming via Server-Sent Events (SSE).
    Frames are yielded immediately as they render in parallel - LOWEST LATENCY!
    
    Latency: ~40ms per frame (vs 2-4 seconds for HLS)
    
    Client usage (JavaScript):
    ```javascript
    const eventSource = new EventSource('/api/realtime-sse');
    const img = document.getElementById('video-frame');
    
    eventSource.onmessage = (event) => {
        img.src = 'data:image/jpeg;base64,' + event.data;
    };
    
    eventSource.addEventListener('complete', (e) => {
        console.log('Total frames:', e.data);
        eventSource.close();
    });
    
    eventSource.addEventListener('error', (e) => {
        console.error('Stream error');
    });
    ```
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")
    
    # Validate parameters
    if nfe < 5 or nfe > 50:
        raise HTTPException(status_code=400, detail="nfe must be between 5 and 50")
    if cfg_scale < 1.0 or cfg_scale > 5.0:
        raise HTTPException(status_code=400, detail="cfg_scale must be between 1.0 and 5.0")
    
    try:
        # Save uploaded files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = tmp.name
            shutil.copyfileobj(image.file, tmp)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            aud_path = tmp.name
            shutil.copyfileobj(audio.file, tmp)
        
        # Load image
        img_pil = Image.open(img_path).convert('RGB')
        
        async def generate():
            try:
                print("[REALTIME-SSE] Starting real-time frame streaming...")
                
                # Prepare inference
                s_pil = (
                    agent.data_processor.process_img(img_pil) if crop
                    else img_pil.resize((agent.opt.input_size, agent.opt.input_size))
                )
                s_tensor = agent.data_processor.transform(s_pil).unsqueeze(0).to(agent.device)
                a_tensor = agent.data_processor.process_audio(aud_path).unsqueeze(0).to(agent.device)
                
                # Calculate target frame count
                T_est = math.ceil(a_tensor.shape[-1] * agent.opt.fps / agent.opt.sampling_rate)
                
                data = {
                    "s": s_tensor,
                    "a": a_tensor,
                    "pose": None,
                    "cam": None,
                    "gaze": None,
                    "ref_x": None,
                }
                
                # Handle pose/gaze styles
                pose_arr = json.loads(pose_style) if pose_style else None
                gaze_arr = json.loads(gaze_style) if gaze_style else None
                
                if pose_arr is not None:
                    data["pose"] = (
                        torch.tensor(pose_arr, device=agent.device)
                        .unsqueeze(0).unsqueeze(0).expand(1, T_est, 3)
                    )
                
                if gaze_arr is not None:
                    data["gaze"] = (
                        torch.tensor(gaze_arr, device=agent.device)
                        .unsqueeze(0).unsqueeze(0).expand(1, T_est, 2)
                    )
                
                # Generate motion tokens
                f_r, g_r = agent.renderer.dense_feature_encoder(s_tensor)
                t_lat = agent.renderer.latent_token_encoder(s_tensor)
                if isinstance(t_lat, tuple):
                    t_lat = t_lat[0]
                data["ref_x"] = t_lat
                
                torch.manual_seed(seed)
                sample = agent.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
                
                # Stabilization
                motion_scale = 0.6
                sample = (
                    (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) 
                    + motion_scale * sample
                )
                
                # Prepare rendering
                ta_r = agent.renderer.adapt(t_lat, g_r)
                m_r = agent.renderer.latent_token_decoder(ta_r)
                
                # REAL-TIME PARALLEL STREAMING - Frames yielded immediately!
                print(f"[REALTIME-SSE] Rendering {sample.shape[1]} frames with parallel streaming...")
                frame_count = 0
                start_time = time.time()
                
                for frame_np in agent.parallel_renderer.render_parallel_stream(
                    sample, g_r, m_r, f_r,
                    output_format="numpy"
                ):
                    # Convert to JPEG
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR), 
                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Yield as SSE event (frames sent IMMEDIATELY as they render!)
                    yield f"data: {frame_base64}\n\n"
                    
                    frame_count += 1
                    if frame_count % 25 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"[REALTIME-SSE] Streamed {frame_count} frames ({fps:.1f} fps)")
                    
                    # Small delay to control frame rate
                    await asyncio.sleep(1.0 / 30.0)  # ~30 fps deivery
                
                # Send completion event
                total_time = time.time() - start_time
                yield f"event: complete\ndata: {{\"frames\": {frame_count}, \"time\": {total_time:.2f}}}\n\n"
                print(f"[REALTIME-SSE] ✓ Streaming complete: {frame_count} frames in {total_time:.2f}s")
                
            except Exception as e:
                print(f"[REALTIME-SSE] Error: {e}")
                import traceback
                traceback.print_exc()
                yield f"event: error\ndata: {str(e)}\n\n"
            finally:
                # Cleanup
                if os.path.exists(img_path):
                    os.remove(img_path)
                if os.path.exists(aud_path):
                    os.remove(aud_path)
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/realtime-mjpeg")
async def realtime_streaming_mjpeg(
    image: UploadFile = File(..., description="Source image file (PNG, JPG)"),
    audio: UploadFile = File(..., description="Driving audio file (WAV, MP3)"),
    crop: bool = Form(True, description="Auto crop face from image"),
    seed: int = Form(42, description="Random seed for generation"),
    nfe: int = Form(10, description="Number of function evaluations (steps), range: 5-50"),
    cfg_scale: float = Form(2.0, description="Classifier-free guidance scale, range: 1.0-5.0"),
    pose_style: Optional[str] = Form(None, description="Optional pose style as JSON array [pitch, yaw, roll]"),
    gaze_style: Optional[str] = Form(None, description="Optional gaze style as JSON array [x, y]"),
):
    """
    REAL-TIME MJPEG streaming - display directly in <img> tag or video player.
    Frames rendered in parallel and streamed immediately with minimal latency.
    
    Client usage (HTML):
    ```html
    <img src="/api/realtime-mjpeg?crop=true&seed=42" style="width: 512px; height: 512px;" />
    ```
    
    Or with form data:
    ```html
    <form id="stream-form" action="/api/realtime-mjpeg" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" />
        <input type="file" name="audio" accept="audio/*" />
        <button type="submit">Stream</button>
    </form>
    <img id="video" />
    <script>
        document.getElementById('stream-form').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const response = await fetch('/api/realtime-mjpeg', {
                method: 'POST',
                body: formData
            });
            document.getElementById('video').src = URL.createObjectURL(await response.blob());
        };
    </script>
    ```
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")
    
    # Validate parameters
    if nfe < 5 or nfe > 50:
        raise HTTPException(status_code=400, detail="nfe must be between 5 and 50")
    if cfg_scale < 1.0 or cfg_scale > 5.0:
        raise HTTPException(status_code=400, detail="cfg_scale must be between 1.0 and 5.0")
    
    try:
        # Save uploaded files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = tmp.name
            shutil.copyfileobj(image.file, tmp)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            aud_path = tmp.name
            shutil.copyfileobj(audio.file, tmp)
        
        img_pil = Image.open(img_path).convert('RGB')
        
        async def generate():
            try:
                print("[REALTIME-MJPEG] Starting MJPEG streaming...")
                
                # Prepare inference (same as SSE)
                s_pil = (
                    agent.data_processor.process_img(img_pil) if crop
                    else img_pil.resize((agent.opt.input_size, agent.opt.input_size))
                )
                s_tensor = agent.data_processor.transform(s_pil).unsqueeze(0).to(agent.device)
                a_tensor = agent.data_processor.process_audio(aud_path).unsqueeze(0).to(agent.device)
                
                T_est = math.ceil(a_tensor.shape[-1] * agent.opt.fps / agent.opt.sampling_rate)
                
                data = {
                    "s": s_tensor, "a": a_tensor,
                    "pose": None, "cam": None, "gaze": None, "ref_x": None
                }
                
                # Handle pose/gaze styles
                pose_arr = json.loads(pose_style) if pose_style else None
                gaze_arr = json.loads(gaze_style) if gaze_style else None
                
                if pose_arr is not None:
                    data["pose"] = (
                        torch.tensor(pose_arr, device=agent.device)
                        .unsqueeze(0).unsqueeze(0).expand(1, T_est, 3)
                    )
                
                if gaze_arr is not None:
                    data["gaze"] = (
                        torch.tensor(gaze_arr, device=agent.device)
                        .unsqueeze(0).unsqueeze(0).expand(1, T_est, 2)
                    )
                
                # Generate motion
                f_r, g_r = agent.renderer.dense_feature_encoder(s_tensor)
                t_lat = agent.renderer.latent_token_encoder(s_tensor)
                if isinstance(t_lat, tuple):
                    t_lat = t_lat[0]
                data["ref_x"] = t_lat
                
                torch.manual_seed(seed)
                sample = agent.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
                
                motion_scale = 0.6
                sample = (
                    (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) 
                    + motion_scale * sample
                )
                
                ta_r = agent.renderer.adapt(t_lat, g_r)
                m_r = agent.renderer.latent_token_decoder(ta_r)
                
                # MJPEG STREAMING - Each frame as separate JPEG
                print(f"[REALTIME-MJPEG] Rendering {sample.shape[1]} frames...")
                frame_count = 0
                start_time = time.time()
                
                for frame_np in agent.parallel_renderer.render_parallel_stream(
                    sample, g_r, m_r, f_r,
                    output_format="numpy"
                ):
                    # Encode as JPEG
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    # Yield in MJPEG format
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                    )
                    
                    frame_count += 1
                    if frame_count % 25 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"[REALTIME-MJPEG] Streamed {frame_count} frames ({fps:.1f} fps)")
                    
                    await asyncio.sleep(1.0 / 30.0)  # ~30 fps
                
                total_time = time.time() - start_time
                print(f"[REALTIME-MJPEG] ✓ Complete: {frame_count} frames in {total_time:.2f}s")
                
            except Exception as e:
                print(f"[REALTIME-MJPEG] Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if os.path.exists(img_path):
                    os.remove(img_path)
                if os.path.exists(aud_path):
                    os.remove(aud_path)
        
        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/realtime/init")
async def init_realtime_session(
    image: UploadFile = File(...), 
    crop: bool = Form(True)
):
    """
    Initialize a real-time low-latency streaming session.
    Upload image once, then connect via WebSocket.
    """
    try:
        if agent is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
            
        session_id = str(uuid.uuid4())
        
        # Load image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = tmp.name
            shutil.copyfileobj(image.file, tmp)
            
        img_pil = Image.open(img_path).convert('RGB')
        
        # Create session but don't start loop yet
        # We need to wire output callback during WS connection
        session = RealtimeSession(
            session_id=session_id,
            inference_agent=agent,
            img_pil=img_pil,
            crop=crop,
            output_callback=None
        )
        realtime_sessions[session_id] = session
        
        # Cleanup temp file
        if os.path.exists(img_path):
            os.remove(img_path)
            
        return {"session_id": session_id, "websocket_url": f"/ws/realtime/{session_id}"}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/realtime/{session_id}")
async def realtime_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in realtime_sessions:
        await websocket.close(code=4004, reason="Session not found")
        return
        
    session = realtime_sessions[session_id]
    loop = asyncio.get_event_loop()
    
    # Bridge callback to send to WS from thread
    def bridge_callback(type, payload, timestamp):
        try:
            # Protocol: [1 byte type] [8 bytes TS] [4 bytes len] [payload]
            # type: 0x01 audio, 0x02 video
            type_byte = 0x01 if type == "audio" else 0x02
            header = struct.pack(">BQ I", type_byte, timestamp, len(payload))
            packet = header + payload
            
            # Fire and forget into asyncio loop
            asyncio.run_coroutine_threadsafe(websocket.send_bytes(packet), loop)
        except Exception as e:
            print(f"Callback Error: {e}")

    session.output_callback = bridge_callback
    session.start()
    
    print(f"[WS] Session {session_id} connected")
    
    try:
        while True:
            # Protocol: [1 byte type] [8 bytes TS] [4 bytes len] [payload]
            # Client sends Audio Only usually, or maybe control
            # Let's assume client sends: [1 byte type] ...
            
            # Read header
            header = await websocket.receive_bytes()
             
            # Minimal header check
            if len(header) < 13:
                 # Maybe it's a small control packet? Or fragmented?
                 # For now assume full packet arrives or at least header
                 if len(header) == 0: break
                 continue
                 
            msg_type = header[0]
            # timestamp = struct.unpack(">Q", header[1:9])[0] # Unused for now
            input_len = struct.unpack(">I", header[9:13])[0]
            
            # If we received more than header, splitting might be needed if using receive_bytes()
            # But receive_bytes returns one "frame".
            # If the client sends binary frame correctly, 'header' variable contains the whole frame
            # Let's assume the WebSocket frame IS the packet.
            
            payload = header[13:]
            if len(payload) < input_len:
                # This shouldn't happen if client sends frame as one WS message
                # But if it does, we might need to read more.
                pass
            
            if msg_type == 0x01: # Audio
                # Payload is PCM 16-bit 16kHz
                audio_np = np.frombuffer(payload, dtype=np.int16)
                session.push_audio(audio_np)
                
    except Exception as e:
        print(f"[WS] Error or Disconnect: {e}")
        # traceback.print_exc()
    finally:
        print(f"[WS] Session {session_id} closing")
        session.stop()
        if session_id in realtime_sessions:
            del realtime_sessions[session_id]
        try:
            await websocket.close()
        except:
            pass
            

@app.post("/api/trt-inference")
async def schedule_trt_inference(
    image: UploadFile = File(..., description="Source image (PNG/JPG)"),
    audio: UploadFile = File(..., description="Driving audio (WAV/MP3)"),
    crop: bool = Form(True, description="Auto crop face"),
    seed: int = Form(42, description="Random seed"),
    nfe: int = Form(10, description="Steps"),
    cfg_scale: float = Form(2.0, description="CFG Scale"),
):
    """
    Run inference using the optimized TensorRT hybrid pipeline.
    
    This endpoint utilizes TensorRT engines for the Source Encoder and Audio Renderer,
    while running the Motion Generator in PyTorch. This offers significantly faster
    inference speeds (up to 2x) compared to the standard PyTorch pipeline.
    
    Requires:
    - TensorRT installed
    - Built TRT engines in ./trt_engines/
    """
    if trt_handler is None or not trt_handler.is_available():
        raise HTTPException(
            status_code=503, 
            detail="TensorRT Inference not available. Check server logs to see if engines were loaded correctly."
        )
        
    img_path = None
    aud_path = None
    
    try:
        # Save temporary files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_path = tmp.name
            shutil.copyfileobj(image.file, tmp)
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            aud_path = tmp.name
            shutil.copyfileobj(audio.file, tmp)
            
        img_pil = Image.open(img_path).convert('RGB')
        
        print(f"[TRT API] Starting inference for {img_path} + {aud_path}")
        
        # Run in threadpool to avoid blocking event loop
        # Note: trt_handler.run_inference returns path to saved video
        video_path = await asyncio.to_thread(
            trt_handler.run_inference, 
            img_pil, 
            aud_path, 
            crop, 
            seed, 
            nfe, 
            cfg_scale
        )
        
        print(f"[TRT API] Success! Video saved to {video_path}")
        
        # Return video file
        return FileResponse(
            video_path, 
            media_type="video/mp4", 
            filename="output.mp4"
        )
        
    except Exception as e:
        print(f"[TRT API] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup input files
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        if aud_path and os.path.exists(aud_path):
            os.remove(aud_path)


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
