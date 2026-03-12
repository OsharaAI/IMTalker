import os
import sys
import traceback
import gc
import torch
import numpy as np
from scipy import signal
import argparse
from PIL import Image
import cv2
import tempfile
import time
import random
from datetime import datetime, timezone
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
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    WebSocket,
)
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
    print(
        "Warning: librosa not installed. Audio resampling may not work. Install with: pip install librosa"
    )
    librosa = None

# Load environment variables from .env file
load_dotenv()
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    AudioStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCIceCandidate,
)
from aiortc.mediastreams import MediaStreamError
from aiortc.contrib.media import MediaRelay
from av import VideoFrame, AudioFrame
from openai import AsyncOpenAI
import websockets
import websockets.exceptions
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add sherpa-TTS path for Nepali support
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../sherpa-TTS")))
sys.path.append(current_dir)

# Import IMTalker components
from imtalker_streamer import IMTalkerStreamer, IMTalkerConfig
from remote_tts import RemoteChatterboxTTS

# Import Sherpa TTS for Nepali language support
try:
    from sherpa_tts_wrapper import SherpaTTS, detect_nepali, is_sherpa_available

    SHERPA_AVAILABLE = is_sherpa_available()
except ImportError as e:
    print(f"Warning: Could not import sherpa_tts_wrapper: {e}")
    SHERPA_AVAILABLE = False
    detect_nepali = lambda text: False

# Try importing TensorRT handler for accelerated inference
try:
    from trt_handler import TRTInferenceHandler

    print("TRTInferenceHandler imported successfully.")
except ImportError as e:
    print(f"TRT Handler not found/importable: {e}")
    TRTInferenceHandler = None

# ─── Configurable Constants ───
MAX_CONCURRENT_WEBRTC_SESSIONS = (
    3  # Max simultaneous WebRTC sessions (increase if GPU can handle it)
)

# Oshara API base URL for session auth & close
OSHARA_API_BASE = "https://api.oshara.ai/api"

# Load avatar configuration (avatar→voice mapping)
AVATAR_CONFIG = {}
AVATAR_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "avatar_config.json"
)
try:
    with open(AVATAR_CONFIG_PATH, "r") as f:
        AVATAR_CONFIG = json.load(f)
    print(
        f"Loaded avatar config with {len(AVATAR_CONFIG.get('avatars', {}))} avatars from {AVATAR_CONFIG_PATH}"
    )
except Exception as e:
    print(f"Warning: Could not load avatar_config.json: {e}")
    AVATAR_CONFIG = {
        "avatars": {},
        "default_voice": "Olivia.wav",
        "default_voice_type": "predefined",
    }


def resolve_avatar_voice(avatar_path: str, tts_model) -> dict:
    """Resolve avatar to its configured voice and apply it to the TTS model.

    Args:
        avatar_path: The avatar image path (e.g., "assets/source_1.png")
        tts_model: The RemoteChatterboxTTS instance

    Returns:
        dict with 'voice', 'voice_type', 'greeting' keys (or empty dict)
    """
    avatars = AVATAR_CONFIG.get("avatars", {})
    config = avatars.get(avatar_path, {})

    voice = config.get("voice") or AVATAR_CONFIG.get("default_voice")
    voice_type = config.get("voice_type") or AVATAR_CONFIG.get(
        "default_voice_type", "predefined"
    )

    if voice and hasattr(tts_model, "set_predefined_voice"):
        if voice_type == "predefined":
            tts_model.set_predefined_voice(voice)
            logger.info(f"Avatar '{avatar_path}' → predefined voice '{voice}'")
        elif voice_type == "clone":
            tts_model.set_clone_voice(voice)
            logger.info(f"Avatar '{avatar_path}' → clone voice '{voice}'")

    return config


# # Import Chatterbox
# # Try to import from installed package or local if needed
# try:
#     import chatterbox

#     print(f"Chatterbox imported from: {chatterbox.__file__}")
#     from chatterbox.tts import ChatterboxTTS
# except ImportError as e:
#     print(f"Could not import chatterbox.tts: {e}")
#     # print sys.path for debugging
#     print(f"sys.path: {sys.path}")
#     import traceback

#     traceback.print_exc()
#     sys.exit(1)

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
        print(
            "⚠️ WARNING: OPENAI_API_KEY not set! Conversational mode will be disabled."
        )
        app.state.openai_client = None
    else:
        app.state.openai_client = AsyncOpenAI(api_key=api_key)

    # 1. Initialize Chatterbox TTS (Remote)
    print("Lifespan: Loading RemoteChatterboxTTS...")
    # Point to STTS-Server (assuming running on port 5000)
    app.state.tts_model = RemoteChatterboxTTS.from_pretrained(
        device=device, server_url="http://localhost:4000"
    )

    # 1b. Warmup STTS server (pre-cache voice conditioning + compile T3 model)
    # This eliminates ~2-5s latency on the first TTS request
    stts_warmup_success = False
    try:
        import httpx

        print("Lifespan: Warming up STTS server TTS model...")
        warmup_resp = httpx.post(
            "http://localhost:4000/tts/warmup",
            json={},
            timeout=120.0,  # warmup can take up to ~60s on first compile
        )
        if warmup_resp.status_code == 200:
            warmup_data = warmup_resp.json()
            print(f"✅ STTS server warmup complete: {warmup_data}")
            stts_warmup_success = True
        else:
            print(
                f"❌ STTS warmup returned {warmup_resp.status_code}: {warmup_resp.text[:200]}"
            )
            print(
                "⚠️ WARNING: First TTS request will be slow (~2-5s compilation overhead)"
            )
    except Exception as e_warmup:
        print(f"❌ STTS warmup failed: {e_warmup}")
        print(
            "⚠️ WARNING: STTS server may not be ready. First TTS request will be slow."
        )

    # 2. Initialize IMTalker
    print("Lifespan: Loading IMTalker...")
    imtalker_cfg = IMTalkerConfig()
    imtalker_cfg.device = device

    # Enable optimizations (tuned for g6e.2xlarge / L40S 48GB)
    imtalker_cfg.use_batching = True
    imtalker_cfg.batch_size = (
        20  # L40S has 48GB VRAM — can handle larger batches (TTS is remote)
    )
    imtalker_cfg.use_fp16 = False  # Disabled: causes NaN/Inf in frames with this model
    imtalker_cfg.use_compile = True  # Enable torch.compile for optimized kernels
    imtalker_cfg.low_latency_mode = (
        True  # NFE 10→5 for 2x faster generation (real-time)
    )

    app.state.imtalker_streamer = IMTalkerStreamer(imtalker_cfg, device=device)
    # Initialize InferenceAgent sharing the same models
    app.state.inference_agent = InferenceAgent(
        imtalker_cfg,
        models=(
            app.state.imtalker_streamer.renderer,
            app.state.imtalker_streamer.generator,
        ),
    )

    # 3. Initialize TensorRT Handler (optional, for accelerated inference)
    app.state.trt_handler = None
    if TRTInferenceHandler is not None:
        try:
            trt_engine_dir = os.path.join(current_dir, "trt_engines")
            trt_handler = TRTInferenceHandler(
                app.state.inference_agent, engine_dir=trt_engine_dir
            )
            if trt_handler.is_available():
                app.state.trt_handler = trt_handler
                # Also attach to IMTalkerStreamer for WebRTC rendering acceleration
                app.state.imtalker_streamer.trt_handler = trt_handler
                print(
                    f"✅ TensorRT Handler initialized with engines from {trt_engine_dir}"
                )
            else:
                print(
                    f"⚠️ TRT engines not found in {trt_engine_dir}. Running in PyTorch-only mode."
                )
        except Exception as e:
            print(f"⚠️ Failed to initialize TRT Handler: {e}")
            app.state.trt_handler = None
    else:
        print("TRTInferenceHandler not available. Running in PyTorch-only mode.")

    # 4. Warmup IMTalker (pre-compile models + cache default avatar embeddings)
    # This eliminates 3-10s torch.compile delay on first generation
    print("Lifespan: Warming up IMTalker pipeline...")
    try:
        # Load a default avatar for warmup if available
        warmup_avatar = None
        default_avatar_path = os.path.join(current_dir, "assets/source_1.png")
        if os.path.exists(default_avatar_path):
            from PIL import Image

            warmup_avatar = Image.open(default_avatar_path).convert("RGB")
            print(f"  Using default avatar: {default_avatar_path}")

        warmup_timings = app.state.imtalker_streamer.warmup(avatar_img=warmup_avatar)
        if warmup_timings.get("status") == "ok":
            print(f"✅ IMTalker warmup complete in {warmup_timings['total_ms']:.0f}ms")
            print(f"   - Avatar embeddings: CACHED")
            print(f"   - torch.compile: DONE")
            print(f"   - Test frames: {warmup_timings.get('frames_generated', 0)}")
        else:
            print(
                f"⚠️ IMTalker warmup completed with errors: {warmup_timings.get('pipeline_error', 'unknown')}"
            )
    except Exception as e:
        print(f"⚠️ IMTalker warmup failed: {e}")
        print("   First generation will be slower (~3-10s compilation overhead)")

    # 5. Pre-warm configured avatars (optional, for instant first-use)
    # NOTE: IMTalkerStreamer can only cache ONE avatar at a time (per-instance limitation)
    # Multiple preload entries will result in only the LAST avatar being cached
    # For best results, set preload=true only for your most commonly used avatar
    print("Lifespan: Pre-warming configured avatars...")
    prewarmed_count = 0
    try:
        for avatar_path, avatar_config in AVATAR_CONFIG.get("avatars", {}).items():
            if avatar_config.get("preload", False):
                full_path = os.path.join(current_dir, avatar_path)
                if os.path.exists(full_path):
                    from PIL import Image

                    avatar_img = Image.open(full_path).convert("RGB")
                    # Just process through the streamer to cache embeddings
                    # (no need to generate frames, just cache the avatar)
                    _ = app.state.imtalker_streamer._get_avatar_cache_key(avatar_img)
                    # Run a quick embedding pass to populate cache
                    s_tensor = app.state.imtalker_streamer.process_img(avatar_img)
                    with torch.no_grad():
                        f_r, app_feat = (
                            app.state.imtalker_streamer.renderer.dense_feature_encoder(
                                s_tensor
                            )
                        )
                        t_lat = (
                            app.state.imtalker_streamer.renderer.latent_token_encoder(
                                s_tensor
                            )
                        )
                        if isinstance(t_lat, tuple):
                            t_lat = t_lat[0]
                        ta_r = app.state.imtalker_streamer.renderer.adapt(
                            t_lat, app_feat
                        )
                        m_r = app.state.imtalker_streamer.renderer.latent_token_decoder(
                            ta_r
                        )
                    # Cache manually (overwrites previous cache - only one avatar cached)
                    cache_key = app.state.imtalker_streamer._get_avatar_cache_key(
                        avatar_img
                    )
                    app.state.imtalker_streamer._avatar_cache_key = cache_key
                    app.state.imtalker_streamer._cached_s_tensor = s_tensor
                    app.state.imtalker_streamer._cached_f_r = f_r
                    app.state.imtalker_streamer._cached_app_feat = app_feat
                    app.state.imtalker_streamer._cached_t_lat = t_lat
                    app.state.imtalker_streamer._cached_ta_r = ta_r
                    app.state.imtalker_streamer._cached_m_r = m_r
                    prewarmed_count += 1
                    print(f"  ✓ Pre-warmed: {avatar_path}")
    except Exception as e:
        print(f"⚠️ Avatar pre-warming error: {e}")

    if prewarmed_count > 0:
        print(f"✅ Pre-warmed {prewarmed_count} configured avatar(s) - LAST one cached")
    else:
        print(
            "ℹ️ No avatars configured for pre-warming (add 'preload': true to avatar_config.json)"
        )

    # 6. Pre-render idle animations for preloaded avatars (eliminates 5-15s delay on first use)
    print("Lifespan: Pre-rendering idle animations for configured avatars...")
    prerendered_idle_count = 0
    try:
        # Initialize idle animation cache early
        from PIL import Image

        for avatar_path, avatar_config in AVATAR_CONFIG.get("avatars", {}).items():
            if avatar_config.get("preload", False):
                full_path = os.path.join(current_dir, avatar_path)
                if os.path.exists(full_path):
                    try:
                        avatar_img = Image.open(full_path).convert("RGB")
                        # Render idle animation in main thread (blocking but only during startup)
                        idle_frames = idle_animation_cache.render_idle(
                            avatar_img, app.state.imtalker_streamer
                        )
                        if idle_frames:
                            prerendered_idle_count += 1
                            print(
                                f"  ✓ Pre-rendered idle: {avatar_path} ({len(idle_frames)} frames)"
                            )
                    except Exception as e:
                        print(f"  ⚠️ Failed to pre-render idle for {avatar_path}: {e}")
    except Exception as e:
        print(f"⚠️ Idle animation pre-rendering error: {e}")

    if prerendered_idle_count > 0:
        print(f"✅ Pre-rendered idle animations for {prerendered_idle_count} avatar(s)")
    else:
        print(
            "ℹ️ No idle animations pre-rendered (set 'preload': true in avatar_config.json)"
        )

    # Session management
    app.state.webrtc_sessions = {}
    app.state.chunked_sessions = {}
    app.state.hls_sessions = {}
    app.state.realtime_sessions = {}

    # Session guard: limit concurrent WebRTC sessions (configured by MAX_CONCURRENT_WEBRTC_SESSIONS)
    app.state.active_webrtc_session = None  # legacy: last created session_id
    app.state.active_session_meta = (
        {}
    )  # {session_id: {status, conv_manager, video_track, audio_track, ...}}
    app.state._session_lock = asyncio.Lock()

    # Idle animation cache (keyed by avatar image hash)
    app.state.idle_cache = {}

    # TTS generation lock: only ONE TTS generation at a time to prevent CUDA OOM
    app.state.tts_lock = asyncio.Lock()

    # Initialize Sherpa TTS for Nepali language support
    if SHERPA_AVAILABLE:
        try:
            from sherpa_tts_wrapper import get_sherpa_tts

            app.state.sherpa_tts = get_sherpa_tts()
            print("✅ Sherpa TTS initialized for Nepali language support")
        except Exception as e:
            print(f"⚠️ Failed to initialize Sherpa TTS: {e}")
            app.state.sherpa_tts = None
    else:
        app.state.sherpa_tts = None
        print("ℹ️ Sherpa TTS not available (Nepali language not supported)")

    yield
    # Clean up models if necessary
    print("Lifespan: Shutting down...")
    if hasattr(app.state, "openai_client"):
        await app.state.openai_client.close()
    del app.state.tts_model
    del app.state.imtalker_streamer
    del app.state.inference_agent


app = FastAPI(lifespan=lifespan)

# HLS output directories
HLS_OUTPUT_DIR = Path("./hls_output")
HLS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HLS_SESSION_DIR = HLS_OUTPUT_DIR  # Consistent with ProgressiveHLSGenerator

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


@app.get("/api/avatars")
async def get_avatars():
    """Return available avatars and their voice configurations."""
    return AVATAR_CONFIG


@app.post("/api/avatar/voice")
async def upload_avatar_voice(
    avatar: str = Form(..., description="Avatar key (e.g., 'assets/source_1.png')"),
    voice_file: UploadFile = File(..., description="Voice .wav file for this avatar"),
    greeting: Optional[str] = Form(
        None, description="Custom greeting text for this avatar"
    ),
):
    """Upload a custom voice for a specific avatar.

    The voice file is forwarded to the STTS server's reference_audio directory
    and the avatar config is updated to use it in clone mode.
    """
    global AVATAR_CONFIG

    if not voice_file.filename or not voice_file.filename.lower().endswith(
        (".wav", ".mp3")
    ):
        raise HTTPException(
            status_code=400, detail="Only .wav and .mp3 files are accepted."
        )

    # Sanitize filename: prefix with avatar key to avoid collisions
    safe_avatar = avatar.replace("/", "_").replace(".", "_")
    ext = os.path.splitext(voice_file.filename)[1]
    ref_filename = f"avatar_{safe_avatar}{ext}"

    # Forward file to STTS server's upload endpoint
    stts_url = (
        app.state.tts_model.server_url
        if hasattr(app.state, "tts_model")
        else "http://localhost:4000"
    )
    try:
        file_content = await voice_file.read()
        resp = requests.post(
            f"{stts_url}/upload_reference",
            files={"files": (ref_filename, file_content, "audio/wav")},
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=502,
                detail=f"STTS server rejected upload: {resp.status_code} {resp.text[:200]}",
            )
        logger.info(f"Uploaded voice '{ref_filename}' to STTS server")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach STTS server: {e}")

    # Update avatar config
    avatars = AVATAR_CONFIG.setdefault("avatars", {})
    entry = avatars.setdefault(avatar, {"name": avatar})
    entry["voice"] = ref_filename
    entry["voice_type"] = "clone"
    if greeting:
        entry["greeting"] = greeting

    # Persist to disk
    try:
        with open(AVATAR_CONFIG_PATH, "w") as f:
            json.dump(AVATAR_CONFIG, f, indent=4)
        logger.info(f"Saved avatar config: {avatar} → clone voice '{ref_filename}'")
    except Exception as e:
        logger.error(f"Failed to save avatar config: {e}")

    return {
        "message": f"Voice for '{avatar}' updated successfully.",
        "avatar": avatar,
        "voice": ref_filename,
        "voice_type": "clone",
    }


@app.delete("/api/avatar/voice")
async def remove_avatar_voice(
    avatar: str = Form(..., description="Avatar key (e.g., 'assets/source_1.png')"),
):
    """Remove a custom voice from an avatar, reverting to the default predefined voice."""
    global AVATAR_CONFIG

    avatars = AVATAR_CONFIG.get("avatars", {})
    if avatar not in avatars:
        raise HTTPException(
            status_code=404, detail=f"Avatar '{avatar}' not found in config."
        )

    entry = avatars[avatar]
    default_voice = AVATAR_CONFIG.get("default_voice", "Olivia.wav")
    entry["voice"] = default_voice
    entry["voice_type"] = "predefined"

    try:
        with open(AVATAR_CONFIG_PATH, "w") as f:
            json.dump(AVATAR_CONFIG, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save avatar config: {e}")

    return {
        "message": f"Voice for '{avatar}' reverted to default.",
        "avatar": avatar,
        "voice": default_voice,
        "voice_type": "predefined",
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    stats = gpu_monitor.get_stats()
    trt_handler = getattr(app.state, "trt_handler", None)
    return {
        "status": "healthy" if hasattr(app.state, "imtalker_streamer") else "unhealthy",
        "device": (
            str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"
        ),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": hasattr(app.state, "imtalker_streamer"),
        "trt_available": trt_handler.is_available() if trt_handler else False,
        "trt_engines": {
            "source_encoder": (
                trt_handler.source_encoder is not None if trt_handler else False
            ),
            "audio_renderer": (
                trt_handler.audio_renderer is not None if trt_handler else False
            ),
        },
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


@app.post("/warmup")
async def warmup_imtalker():
    """
    Warm up the IMTalker pipeline to eliminate first-request latency.

    Pre-compiles models and caches default avatar embeddings.
    Call this after server startup (or after model reload) to ensure
    instant response on first real generation.

    Returns timing info and warmup status.
    """
    if not hasattr(app.state, "imtalker_streamer"):
        raise HTTPException(
            status_code=503,
            detail="IMTalker not loaded. Server may still be starting up.",
        )

    try:
        # Load default avatar if available
        warmup_avatar = None
        default_avatar_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets/source_1.png"
        )
        if os.path.exists(default_avatar_path):
            from PIL import Image

            warmup_avatar = Image.open(default_avatar_path).convert("RGB")

        timings = await asyncio.get_event_loop().run_in_executor(
            None, app.state.imtalker_streamer.warmup, warmup_avatar
        )

        return {
            "status": "ok",
            "timings": timings,
            "message": "IMTalker warmup complete. Models compiled and avatar cached.",
        }
    except Exception as e:
        logger.error(f"Warmup failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


def numpy_to_wav_base64(audio_np, sampling_rate=16000):
    if isinstance(audio_np, torch.Tensor):
        audio_np = audio_np.cpu().numpy()
    audio_np = audio_np.flatten()
    if np.max(np.abs(audio_np)) > 1.0:
        audio_np = audio_np / np.max(np.abs(audio_np))
    audio_int16 = (audio_np * 32767).astype(np.int16)
    byte_io = io.BytesIO()
    with wave.open(byte_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(audio_int16.tobytes())
    return base64.b64encode(byte_io.getvalue()).decode("utf-8")


def save_base64_to_temp(base64_str, prefix="temp_", suffix=".tmp"):
    """Decodes base64 string to a temporary file and returns path."""
    try:
        # Strip data URI scheme if present
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]

        decoded_data = base64.b64decode(base64_str)

        # Create temp file
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(decoded_data)
        logger.info(
            f"Saved Base64 data to temp file: {path} ({len(decoded_data)} bytes)"
        )
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
            with os.fdopen(fd, "wb") as f:
                f.write(response.content)

            logger.info(
                f"Downloaded URL {url} to temp file: {path} ({len(response.content)} bytes)"
            )
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
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
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
        if len(current_chunk) >= min_chunk_length and sentence[-1] in ".!?":
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
    chunk_pattern = r"chunk\d+:\s*"

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

    def __init__(
        self,
        session_id: str,
        target_sr: int = 16000,
        chunk_duration: float = 1.0,
        overlap_ratio: float = 0.2,
    ):
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
                        print(
                            f"[AudioBuffer {self.session_id}] Resampling {len(audio_chunk)} samples from {source_sr}Hz → {self.target_sr}Hz"
                        )
                        audio_chunk = librosa.resample(
                            audio_chunk, orig_sr=source_sr, target_sr=self.target_sr
                        )
                    else:
                        print(
                            f"[AudioBuffer {self.session_id}] Warning: librosa not available for resampling!"
                        )

                # Append to buffer
                self.buffer = np.concatenate(
                    [self.buffer, audio_chunk.astype(np.float32)]
                )
                self.total_samples_received += len(audio_chunk)

                has_enough = len(self.buffer) >= self.chunk_samples

                print(
                    f"[AudioBuffer {self.session_id}] Chunk added: {len(audio_chunk)} samples → "
                    f"buffer now {len(self.buffer)} samples (need {self.chunk_samples}) "
                    f"[total_received: {self.total_samples_received}]"
                )

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
            chunk = self.buffer[: self.chunk_samples].copy()

            # Slide buffer by stride (move window forward with overlap)
            self.buffer = self.buffer[self.stride_samples :]

            print(
                f"[AudioBuffer {self.session_id}] Extracted chunk: "
                f"{self.chunk_samples} samples, buffer now {len(self.buffer)} samples remaining"
            )

            return chunk

    def has_pending_audio(self) -> bool:
        """Check if there's unprocessed audio in the buffer"""
        with self.lock:
            return len(self.buffer) >= self.chunk_samples

    def finalize(self) -> Optional[np.ndarray]:
        """
        Get any remaining audio (at end of stream).
        Applies fade-out and pads to match chunk_samples.
        """
        with self.lock:
            if len(self.buffer) == 0:
                return None

            if len(self.buffer) < self.chunk_samples:
                # Apply fade-out on actual audio before padding to avoid
                # abrupt silence boundary that sounds like a click/pause
                actual = self.buffer.copy()
                fade_len = min(len(actual), int(0.02 * 16000))  # ~20ms fade
                if fade_len > 1:
                    actual[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                padding = np.zeros(
                    self.chunk_samples - len(actual), dtype=np.float32
                )
                chunk = np.concatenate([actual, padding])
                print(
                    f"[AudioBuffer {self.session_id}] Final chunk: {len(self.buffer)} samples + "
                    f"{len(padding)} zero-padding (faded) = {len(chunk)} samples"
                )
            else:
                chunk = self.buffer[: self.chunk_samples].copy()
                print(
                    f"[AudioBuffer {self.session_id}] Final chunk: {len(chunk)} samples"
                )

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
realtime_sessions: Dict[str, "RealtimeSession"] = {}


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
        print(
            f"[CLEANUP] Scheduled cleanup for session {session_id} in {delay_minutes} minutes"
        )
        time.sleep(delay_seconds)

        session_dir = HLS_OUTPUT_DIR / session_id
        if session_dir.exists():
            try:
                shutil.rmtree(session_dir)
                print(f"[CLEANUP] ✓ Deleted HLS output for session {session_id}")

                # Also remove from generation_status
                if session_id in generation_status:
                    del generation_status[session_id]
                    print(
                        f"[CLEANUP] ✓ Removed session {session_id} from status tracking"
                    )
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
        segment_duration: int = 2,
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

        print(
            f"[HLS {self.session_id}] Initialized - FPS: {fps}, Segment Duration: {segment_duration}s, Frames per segment: {self.frames_per_segment}"
        )
        print(f"[HLS {self.session_id}] Output directory: {self.output_dir}")
        print(f"[HLS {self.session_id}] Audio path: {self.audio_path}")

        # Check if session already exists and load existing segments
        self._load_existing_session()

        # Update playlist with existing segments (if any) to ensure continuity
        if self.segment_files:
            print(
                f"[HLS {self.session_id}] Updating playlist with {len(self.segment_files)} existing segments"
            )
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
                match = re.search(r"segment(\d+)\.ts", seg_name)
                if match:
                    seg_num = int(match.group(1))
                    max_segment_num = max(max_segment_num, seg_num)

            # Set current_segment to continue from the last segment
            if max_segment_num >= 0:
                self.current_segment = max_segment_num + 1
                print(
                    f"[HLS {self.session_id}] Resuming from segment {self.current_segment} (found {len(existing_segments)} existing segments)"
                )
            else:
                print(
                    f"[HLS {self.session_id}] No valid segments found, starting from 0"
                )

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
            print(
                f"[HLS {self.session_id}] No frames in buffer, skipping segment write"
            )
            return

        segment_file = self.output_dir / f"segment{self.current_segment:03d}.ts"
        segment_name = f"segment{self.current_segment:03d}.ts"
        temp_video = self.output_dir / f"temp_segment{self.current_segment:03d}.mp4"

        # CRITICAL FIX: Check if segment already exists
        if segment_file.exists():
            all_segments = sorted(glob.glob(str(self.output_dir / "segment*.ts")))
            max_seg = -1
            for seg_path in all_segments:
                match = re.search(r"segment(\d+)\.ts", os.path.basename(seg_path))
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
                print(
                    f"[HLS {self.session_id}] ERROR: Temp video file not created or empty"
                )
                return

            print(
                f"[HLS {self.session_id}] Temp video created: {temp_video.stat().st_size} bytes"
            )

            start_time = self.current_segment * self.segment_duration
            duration = len(self.frame_buffer) / self.fps

            cmd = ["ffmpeg", "-y", "-i", str(temp_video)]

            if self.audio_path and os.path.exists(self.audio_path):
                print(
                    f"[HLS {self.session_id}] Adding audio from {start_time:.3f}s for {duration:.3f}s"
                )
                cmd.extend(
                    [
                        "-ss",
                        f"{start_time:.3f}",
                        "-t",
                        f"{duration:.3f}",
                        "-i",
                        self.audio_path,
                        "-map",
                        "0:v",
                        "-map",
                        "1:a",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        "-ac",
                        "2",
                    ]
                )

            cmd.extend(["-output_ts_offset", f"{start_time:.3f}"])
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-f",
                    "mpegts",
                    "-y",
                    str(segment_file),
                ]
            )

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
        self.crop_scale = getattr(opt, "crop_scale", 0.8)
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
        local_path = getattr(
            opt, "wav2vec_model_path", "./checkpoints/wav2vec2-base-960h"
        )
        if os.path.exists(local_path) and os.path.exists(
            os.path.join(local_path, "config.json")
        ):
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

    def get_crop_coords(self, img_arr: np.ndarray) -> tuple:
        """Return (x1, y1, x2, y2) crop coords from a numpy RGB image."""
        h, w = img_arr.shape[:2]

        if self.fa is None:
            cx, cy = w // 2, h // 2
            half = min(w, h) // 2
            return (cx - half, cy - half, cx + half, cy + half)

        try:
            bboxes = self.fa.face_detector.detect_from_image(img_arr)
            if bboxes is None or len(bboxes) == 0:
                bboxes = self.fa.face_detector.detect_from_image(img_arr)
        except Exception as e:
            print(f"Face detection failed in get_crop_coords: {e}")
            bboxes = None

        valid_bboxes = []
        if bboxes is not None:
            valid_bboxes = [
                (int(x1), int(y1), int(x2), int(y2), score)
                for (x1, y1, x2, y2, score) in bboxes
                if score > 0.5
            ]

        if not valid_bboxes:
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

        return (int(x1_new), int(y1_new), int(x2_new), int(y2_new))

    def paste_back_tensor(self, face_tensor, full_img_tensor, coords):
        """Paste face_tensor [1,3,H,W] back into full_img_tensor [1,3,H,W] at coords."""
        import torch.nn.functional as F

        x1, y1, x2, y2 = coords
        _, _, fh, fw = full_img_tensor.shape

        # Normalize face to [0,1] if needed
        face = face_tensor.clone()
        if face.min() < 0:
            face = (face + 1) / 2
        face = face.clamp(0, 1)

        # Resize face to target region size
        region_h, region_w = y2 - y1, x2 - x1
        face_resized = F.interpolate(
            face, size=(region_h, region_w), mode="bilinear", align_corners=False
        )

        # Paste into full image copy
        result = full_img_tensor.clone()
        if result.min() < 0:
            result = (result + 1) / 2
        result = result.clamp(0, 1)
        result[:, :, y1:y2, x1:x2] = face_resized
        return result

    def process_audio(self, path: str) -> torch.Tensor:
        if librosa is None:
            raise ImportError("librosa is required for process_audio")
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        return self.wav2vec_preprocessor(
            speech_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_values[0]

    def crop_video_stable(
        self,
        from_mp4_file_path,
        to_mp4_file_path,
        expanded_ratio=0.6,
        skip_per_frame=15,
    ):
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
            if not success:
                break
            if index % skip_per_frame == 0:
                success, frame = video.retrieve()
                if not success:
                    break
                h, w = frame.shape[:2]
                mult = 360.0 / h
                resized_frame = cv2.resize(
                    frame,
                    dsize=(0, 0),
                    fx=mult,
                    fy=mult,
                    interpolation=cv2.INTER_AREA if mult < 1 else cv2.INTER_CUBIC,
                )
                try:
                    detected_bboxes = self.fa.face_detector.detect_from_image(
                        resized_frame
                    )
                    if detected_bboxes is not None:
                        for d_box in detected_bboxes:
                            bx1, by1, bx2, by2, score = d_box
                            if score > 0.5:
                                bboxes_lists.append(
                                    [
                                        int(bx1 / mult),
                                        int(by1 / mult),
                                        int(bx2 / mult),
                                        int(by2 / mult),
                                        score,
                                    ]
                                )
                                break  # Take first face
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
        if x1 + side > width:
            x1 = width - side
        if y1 + side > height:
            y1 = height - side

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
    WARN_THRESHOLD = 0.75  # Log warning at 75%
    CRITICAL_THRESHOLD = 0.85  # Reduce workers at 85%
    EMERGENCY_THRESHOLD = 0.92  # Force single worker + cache clear at 92%

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
            self._total_mb = torch.cuda.get_device_properties(0).total_memory / (
                1024**2
            )
            print(f"[GPU Monitor] Total VRAM: {self._total_mb:.0f}MB")

    def snapshot(self) -> dict:
        """Take a memory snapshot. Returns dict with MB values."""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0, "total_mb": 0, "pct": 0}

        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        reserved = torch.cuda.memory_reserved(0) / (1024**2)
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
            print(
                f"⚠️  [GPU Monitor] {prefix}{level.upper()}: "
                f"{snap['allocated_mb']:.0f}MB / {snap['total_mb']:.0f}MB "
                f"({pct*100:.1f}%) [peak: {self._peak_allocated_mb:.0f}MB]"
            )

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
                print(
                    f"⚠️  [GPU Monitor] CRITICAL: Reducing workers {requested_workers} → 1"
                )
                self._fallback_active = True
            return 1
        else:
            if self._fallback_active:
                print(
                    "✅ [GPU Monitor] Memory pressure eased, restoring requested workers"
                )
                self._fallback_active = False
            return requested_workers

    def record_oom(self):
        """Record an OOM event for tracking."""
        with self._lock:
            self._oom_count += 1
        snap = self.snapshot()
        print(
            f"🚨 [GPU Monitor] OOM #{self._oom_count}! "
            f"{snap['allocated_mb']:.0f}MB / {snap['total_mb']:.0f}MB"
        )

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
        print(
            f"[ParallelRenderer] workers={num_workers}, chunk_size={chunk_size}, "
            f"render_batch={self.render_batch_size}"
        )

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

    def render_chunk_to_numpy(
        self, chunk_id, start_idx, end_idx, sample, g_r, m_r, f_r
    ):
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
                        sample_batch = sample[:, b_start:b_end, ...].squeeze(
                            0
                        )  # [batch, D]
                        g_r_batch = g_r.repeat(curr_batch, 1)
                        m_r_batch = tuple(m.repeat(curr_batch, 1, 1, 1) for m in m_r)
                        f_r_batch = [f.repeat(curr_batch, 1, 1, 1) for f in f_r]

                        ta_c_batch = self.agent.renderer.adapt(sample_batch, g_r_batch)
                        m_c_batch = self.agent.renderer.latent_token_decoder(ta_c_batch)
                        out_batch = self.agent.renderer.decode(
                            m_c_batch, m_r_batch, f_r_batch
                        )

                        for f_idx in range(curr_batch):
                            frame_np = (
                                out_batch[f_idx : f_idx + 1]
                                .squeeze(0)
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                            )
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
                            frame_np = (
                                out_frame.squeeze(0)
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                            )
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
                    frame_np = (
                        out_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    )
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
        chunks = [
            (i // self.chunk_size, i, min(i + self.chunk_size, T))
            for i in range(0, T, self.chunk_size)
        ]
        rendered_chunks = {}
        effective_workers = self._get_effective_workers()

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    self.render_chunk, cid, s, e, sample, g_r, m_r, f_r
                ): cid
                for cid, s, e in chunks
            }
            for i, future in enumerate(as_completed(future_to_chunk)):
                cid, frames, err = future.result()
                if err:
                    raise RuntimeError(err)
                rendered_chunks[cid] = frames
                if progress_callback:
                    progress_callback(i + 1, len(chunks))
                if i % 2 == 0:
                    torch.cuda.empty_cache()

        all_frames = []
        for cid in sorted(rendered_chunks.keys()):
            all_frames.extend(rendered_chunks[cid])
        return all_frames

    def render_progressive_hls(
        self, sample, g_r, m_r, f_r, hls_generator, full_img_tensor=None, coords=None
    ):
        T = sample.shape[1]
        chunks = [
            (i // self.chunk_size, i, min(i + self.chunk_size, T))
            for i in range(0, T, self.chunk_size)
        ]
        rendered_chunks = {}
        next_to_send = 0
        effective_workers = self._get_effective_workers()

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    self.render_chunk_to_numpy, cid, s, e, sample, g_r, m_r, f_r
                ): cid
                for cid, s, e in chunks
            }
            for future in as_completed(future_to_chunk):
                cid, frames, err = future.result()
                if err:
                    continue
                rendered_chunks[cid] = frames
                while next_to_send in rendered_chunks:
                    for f in rendered_chunks[next_to_send]:
                        if full_img_tensor is not None and coords is not None:
                            f_t = (
                                torch.from_numpy(f)
                                .permute(2, 0, 1)
                                .unsqueeze(0)
                                .float()
                                / 255.0
                            )
                            f_t = f_t.to(full_img_tensor.device)
                            pasted = self.agent.data_processor.paste_back_tensor(
                                f_t, full_img_tensor, coords
                            )
                            f = (
                                np.clip(
                                    pasted.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                                    0,
                                    1,
                                )
                                * 255
                            ).astype(np.uint8)
                        hls_generator.add_frame(f)
                    del rendered_chunks[next_to_send]
                    next_to_send += 1
                if cid % 2 == 0:
                    torch.cuda.empty_cache()

    def render_parallel_stream(self, sample, g_r, m_r, f_r, output_format="tensor"):
        T = sample.shape[1]
        chunks = [
            (i // self.chunk_size, i, min(i + self.chunk_size, T))
            for i in range(0, T, self.chunk_size)
        ]
        rendered_chunks = {}
        next_to_yield = 0
        render_func = (
            self.render_chunk_to_numpy
            if output_format == "numpy"
            else self.render_chunk
        )
        effective_workers = self._get_effective_workers()

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_chunk = {
                executor.submit(render_func, cid, s, e, sample, g_r, m_r, f_r): cid
                for cid, s, e in chunks
            }
            for future in as_completed(future_to_chunk):
                cid, frames, err = future.result()
                if err:
                    continue
                rendered_chunks[cid] = frames
                while next_to_yield in rendered_chunks:
                    for f in rendered_chunks[next_to_yield]:
                        yield f
                    del rendered_chunks[next_to_yield]
                    next_to_yield += 1
                if cid % 2 == 0:
                    torch.cuda.empty_cache()


class InferenceAgent:
    def __init__(self, opt, models=None):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
            num_workers=getattr(opt, "num_render_workers", 2),
            chunk_size=getattr(opt, "render_chunk_size", 3),
        )
        print(
            f"InferenceAgent ready. GPU: {gpu_monitor.snapshot()['allocated_mb']:.0f}MB / {gpu_monitor.snapshot()['total_mb']:.0f}MB"
        )

    def _load_ckpt(self, model, path, prefix="gen."):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean = {
            k.replace(prefix, ""): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        model.load_state_dict(clean, strict=False)

    def _load_fm_ckpt(self, model, path):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location="cpu")
        sd = checkpoint.get("state_dict", checkpoint)
        if "model" in sd:
            sd = sd["model"]
        prefix = "model."
        clean = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in clean:
                    param.copy_(clean[name].to(self.device))

    def prepare_source_image(self, img_pil: Image.Image, crop: bool):
        """
        Prepare source image for inference.

        If crop=True:  face-detect and crop (existing behaviour via process_img).
        If crop=False: Crop-Infer-Paste mode -- detect face crop coords, crop the face
                       region, resize it for the model, AND return the full-image tensor
                       + coords needed for paste-back.

        Returns:
            s_pil    (PIL Image at input_size): ready for self.data_processor.transform()
            full_img_tensor (Tensor [1,3,H,W] on self.device, or None if crop=True)
            coords   (x1,y1,x2,y2 tuple, or None if crop=True)
        """
        if crop:
            s_pil = self.data_processor.process_img(img_pil)
            return s_pil, None, None

        import torchvision.transforms as transforms

        # Crop-Infer-Paste Logic
        img_arr = np.array(img_pil)
        if img_arr.ndim == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        elif img_arr.shape[2] == 4:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)

        x1, y1, x2, y2 = self.data_processor.get_crop_coords(img_arr)
        coords = (x1, y1, x2, y2)

        crop_img = img_arr[y1:y2, x1:x2]
        s_pil = Image.fromarray(crop_img).resize(
            (self.opt.input_size, self.opt.input_size)
        )
        full_img_tensor = transforms.ToTensor()(img_pil).to(self.device).unsqueeze(0)
        return s_pil, full_img_tensor, coords

    def save_video(self, vid_tensor, fps, audio_path=None):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            raw_path = tmp.name
        if vid_tensor.dim() == 4:
            vid = vid_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        else:
            vid = vid_tensor.detach().cpu().numpy()

        if vid.min() < 0:
            vid = (vid + 1) / 2
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
            # Pre-process audio: apply fade-in/out to prevent click/pop artifacts
            processed_audio_path = audio_path
            try:
                import soundfile as sf

                aud_data, aud_sr = sf.read(audio_path, dtype="float32")
                aud_data = apply_audio_chunk_fades(aud_data, sr=aud_sr, fade_ms=10.0)
                processed_audio_path = audio_path + ".faded.wav"
                sf.write(processed_audio_path, aud_data, aud_sr)
            except Exception:
                pass  # Fall back to original if processing fails
            # Use system FFmpeg (/usr/bin/ffmpeg) which has libx264, not conda FFmpeg
            cmd = f'/usr/bin/ffmpeg -y -i "{raw_path}" -i "{processed_audio_path}" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}"'
            subprocess.call(cmd, shell=True)
            if os.path.exists(raw_path):
                os.remove(raw_path)
            if processed_audio_path != audio_path and os.path.exists(
                processed_audio_path
            ):
                os.remove(processed_audio_path)
            return final_path
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
        torch.cuda.empty_cache()
        s_pil, full_img_tensor, coords = self.prepare_source_image(img_pil, crop)
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = (
            self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        )

        T = round(a_tensor.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        data = {
            "s": s_tensor,
            "a": a_tensor,
            "pose": None,
            "cam": None,
            "gaze": None,
            "ref_x": None,
        }

        if pose_style:
            data["pose"] = (
                torch.tensor(pose_style, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, T, 3)
            )
        if gaze_style:
            data["gaze"] = (
                torch.tensor(gaze_style, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, T, 2)
            )

        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        data["ref_x"] = t_lat

        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        if isinstance(sample, tuple):
            sample = sample[0]

        motion_scale = 0.6
        sample = (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(
            1, sample.shape[1], 1
        ) + motion_scale * sample

        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)

        if is_streaming:
            raw_frames = self.parallel_renderer.render_parallel(sample, g_r, m_r, f_r)
            frames = []
            for f in raw_frames:
                if full_img_tensor is not None:
                    f = self.data_processor.paste_back_tensor(
                        f.to(self.device), full_img_tensor, coords
                    ).cpu()
                frames.append(f)
        else:
            frames = []
            for t in range(sample.shape[1]):
                ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.renderer.latent_token_decoder(ta_c)
                out = self.renderer.decode(m_c, m_r, f_r)
                if full_img_tensor is not None:
                    out = self.data_processor.paste_back_tensor(
                        out, full_img_tensor, coords
                    )
                frames.append(out.cpu())
                if t % 10 == 0:
                    torch.cuda.empty_cache()

        vid_tensor = torch.stack(frames, dim=1).squeeze(0)
        return self.save_video(vid_tensor, self.opt.fps, aud_path)

    @torch.no_grad()
    def run_audio_inference_streaming(
        self, img_pil, aud_path, crop, seed, nfe, cfg_scale
    ):
        s_pil, full_img_tensor, coords = self.prepare_source_image(img_pil, crop)
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = (
            self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        )

        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]

        torch.manual_seed(seed)
        sample = self.generator.sample(
            {
                "s": s_tensor,
                "a": a_tensor,
                "pose": None,
                "cam": None,
                "gaze": None,
                "ref_x": t_lat,
            },
            a_cfg_scale=cfg_scale,
            nfe=nfe,
            seed=seed,
        )
        if isinstance(sample, tuple):
            sample = sample[0]
        if isinstance(sample, tuple):
            sample = sample[0]

        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)

        for t in range(sample.shape[1]):
            ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            out = self.renderer.decode(m_c, m_r, f_r)
            if full_img_tensor is not None:
                out = self.data_processor.paste_back_tensor(
                    out, full_img_tensor, coords
                )
            yield out.squeeze(0)

    @torch.no_grad()
    def run_audio_inference_progressive(
        self,
        img_pil,
        aud_path,
        crop,
        seed,
        nfe,
        cfg_scale,
        hls_generator,
        pose_style=None,
        gaze_style=None,
        is_final_chunk=False,
    ):
        torch.cuda.empty_cache()
        s_pil, full_img_tensor, coords = self.prepare_source_image(img_pil, crop)
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = (
            self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        )

        T = round(a_tensor.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        data = {
            "s": s_tensor,
            "a": a_tensor,
            "pose": None,
            "cam": None,
            "gaze": None,
            "ref_x": None,
        }
        if pose_style:
            data["pose"] = (
                torch.tensor(pose_style)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, T, 3)
                .to(self.device)
            )
        if gaze_style:
            data["gaze"] = (
                torch.tensor(gaze_style)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, T, 2)
                .to(self.device)
            )

        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        data["ref_x"] = t_lat

        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        if isinstance(sample, tuple):
            sample = sample[0]
        motion_scale = 0.6
        sample = (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(
            1, sample.shape[1], 1
        ) + motion_scale * sample

        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)

        self.parallel_renderer.render_progressive_hls(
            sample, g_r, m_r, f_r, hls_generator, full_img_tensor, coords
        )
        hls_generator.finalize(audio_path=aud_path, mark_complete=is_final_chunk)

    @torch.no_grad()
    def run_video_inference(self, source_img_pil, driving_video_path, crop):
        torch.cuda.empty_cache()
        s_pil = (
            self.data_processor.process_img(source_img_pil)
            if crop
            else source_img_pil.resize((512, 512))
        )
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
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            p = Image.fromarray(frame).resize((512, 512))
            d_t = self.data_processor.transform(p).unsqueeze(0).to(self.device)
            t_c = self.renderer.mot_encode(d_t)
            ta_c = self.renderer.adapt(t_c, i_r)
            ma_c = self.renderer.mot_decode(ta_c)
            results.append(self.renderer.decode(ma_c, ma_r, f_r).cpu())
            if len(results) % 10 == 0:
                torch.cuda.empty_cache()
        cap.release()
        if crop and os.path.exists(final_driving):
            os.remove(final_driving)

        vid_tensor = torch.cat(results, dim=0)
        return self.save_video(vid_tensor, fps=fps, audio_path=driving_video_path)

    @torch.no_grad()
    def run_audio_inference_chunked(
        self,
        img_pil,
        audio_chunk_np,
        crop,
        seed,
        nfe,
        cfg_scale,
        hls_generator,
        pose_style=None,
        gaze_style=None,
    ):
        torch.cuda.empty_cache()
        s_pil, full_img_tensor, coords = self.prepare_source_image(img_pil, crop)
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = torch.from_numpy(audio_chunk_np).float().unsqueeze(0).to(self.device)

        T = round(a_tensor.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        data = {
            "s": s_tensor,
            "a": a_tensor,
            "pose": None,
            "cam": None,
            "gaze": None,
            "ref_x": None,
        }
        if pose_style:
            data["pose"] = (
                torch.tensor(pose_style)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, T, 3)
                .to(self.device)
            )
        if gaze_style:
            data["gaze"] = (
                torch.tensor(gaze_style)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, T, 2)
                .to(self.device)
            )

        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        data["ref_x"] = t_lat

        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        if isinstance(sample, tuple):
            sample = sample[0]
        motion_scale = 0.6
        sample = (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(
            1, sample.shape[1], 1
        ) + motion_scale * sample

        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)

        frames = 0
        for t in range(sample.shape[1]):
            ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            out = self.renderer.decode(m_c, m_r, f_r)
            if full_img_tensor is not None:
                out = self.data_processor.paste_back_tensor(
                    out, full_img_tensor, coords
                )
            f_np = (
                np.clip(out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), 0, 1)
                * 255
            ).astype(np.uint8)
            hls_generator.add_frame(f_np)
            frames += 1
        return frames

    @torch.no_grad()
    def run_trt_inference(
        self, trt_handler, img_pil, aud_path, crop, seed, nfe, cfg_scale
    ):
        """
        Run inference using TensorRT-accelerated source encoder and audio renderer.
        Motion generation remains in PyTorch. Falls back to PyTorch if TRT fails.

        Args:
            trt_handler: TRTInferenceHandler instance
            img_pil: PIL Image of the source face
            aud_path: Path to driving audio file
            crop: Whether to auto-crop the face
            seed: Random seed for generation
            nfe: Number of function evaluations (steps)
            cfg_scale: Classifier-free guidance scale

        Returns:
            Path to the output video file
        """
        torch.cuda.empty_cache()
        print(f"[TRT InferenceAgent] Starting TensorRT-accelerated inference...")
        t_start = time.time()

        try:
            video_path = trt_handler.run_inference(
                img_pil, aud_path, crop, seed, nfe, cfg_scale
            )
            t_total = time.time() - t_start
            print(f"[TRT InferenceAgent] Completed in {t_total:.2f}s")
            return video_path
        except Exception as e:
            print(
                f"[TRT InferenceAgent] TRT inference failed: {e}, falling back to PyTorch"
            )
            traceback.print_exc()
            return self.run_audio_inference(
                img_pil, aud_path, crop, seed, nfe, cfg_scale
            )

    @torch.no_grad()
    def run_trt_inference_streaming(
        self, trt_handler, img_pil, aud_path, crop, seed, nfe, cfg_scale
    ):
        """
        Run TRT-accelerated inference and yield frames one-by-one for streaming.
        Uses TRT for source encoding and per-frame rendering, PyTorch for motion generation.
        Falls back to PyTorch streaming if TRT fails.

        Yields:
            torch.Tensor frames [1, 3, H, W]
        """
        torch.cuda.empty_cache()
        print(f"[TRT Streaming] Starting TensorRT-accelerated streaming inference...")

        try:
            # 1. Preprocessing
            s_pil, full_img_tensor, coords = self.prepare_source_image(img_pil, crop)
            s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)

            # 2. Source Encoding (TensorRT)
            enc_outputs = trt_handler.source_encoder.infer([s_tensor])
            i_r = enc_outputs["i_r"]
            t_r = enc_outputs["t_r"]
            f_r = [enc_outputs[f"f_r_{i}"] for i in range(6)]
            ma_r = [enc_outputs[f"ma_r_{i}"] for i in range(4)]

            # 3. Audio Processing & Motion Generation (PyTorch)
            a_tensor = (
                self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
            )
            data = {
                "s": s_tensor,
                "a": a_tensor,
                "pose": None,
                "cam": None,
                "gaze": None,
                "ref_x": t_r,
            }

            torch.manual_seed(seed)
            sample, _ = self.generator.sample(
                data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed
            )

            motion_scale = 0.6
            sample = (1.0 - motion_scale) * t_r.unsqueeze(1).repeat(
                1, sample.shape[1], 1
            ) + motion_scale * sample

            # 4. Rendering (TensorRT) — frame by frame
            T = sample.shape[1]
            print(f"[TRT Streaming] Rendering {T} frames via TensorRT...")

            for t in range(T):
                motion_code = sample[:, t, ...]
                trt_inputs = {"motion_code": motion_code, "i_r": i_r}
                for idx in range(6):
                    trt_inputs[f"f_r_{idx}"] = f_r[idx]
                for idx in range(4):
                    trt_inputs[f"ma_r_{idx}"] = ma_r[idx]

                out_dict = trt_handler.audio_renderer.infer(trt_inputs)
                out_frame = out_dict["output_image"]

                if full_img_tensor is not None:
                    out_frame = self.data_processor.paste_back_tensor(
                        out_frame, full_img_tensor, coords
                    )

                yield out_frame.unsqueeze(0).cpu()

                if t % 20 == 0:
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"[TRT Streaming] TRT streaming failed: {e}, falling back to PyTorch")
            traceback.print_exc()
            yield from self.run_audio_inference_streaming(
                img_pil, aud_path, crop, seed, nfe, cfg_scale
            )


# --- OpenAI Realtime API Session ---


class OpenAIRealtimeSession:
    """Manages a WebSocket connection to OpenAI's Realtime API.

    Handles:
    - WebSocket lifecycle (connect / disconnect / reconnect)
    - Session configuration (server VAD, text-only output, system prompt)
    - Streaming PCM audio to the Realtime API via input_audio_buffer.append
    - Receiving server events: speech detection, transcription, text responses
    - Conversation history is managed entirely server-side by the Realtime API

    The session is configured for TEXT-ONLY output (no audio output from OpenAI).
    We use our own TTS (ChatterboxTTS) + IMTalker for speech synthesis and
    avatar rendering, which gives us lip-synced video.
    """

    REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"
    TARGET_SAMPLE_RATE = 24000  # OpenAI Realtime API expects 24kHz PCM16

    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        on_response_text=None,
        on_response_text_delta=None,
        on_speech_started=None,
        on_speech_stopped=None,
        on_user_transcript=None,
        model: str = None,
        vad_threshold: float = 0.5,
        vad_prefix_padding_ms: int = 300,
        vad_silence_duration_ms: int = 700,
        temperature: float = 0.7,
        max_output_tokens: int = 150,
        character_slug: Optional[str] = None,
        kb_bearer_token: Optional[str] = None,
    ):
        """
        Args:
            api_key: OpenAI API key
            system_prompt: System instructions for the model
            on_response_text: async callback(full_text: str) — fired when complete response is ready
            on_response_text_delta: async callback(delta: str) — fired for each text token
            on_speech_started: async callback() — fired when server VAD detects user speech
            on_speech_stopped: async callback() — fired when server VAD detects end of speech
            on_user_transcript: async callback(text: str) — fired when user audio is transcribed
            model: Optional model override (default: gpt-4o-mini-realtime-preview-2024-12-17)
            vad_threshold: Voice activity detection threshold 0.0-1.0 (default: 0.5)
            vad_prefix_padding_ms: Audio before speech to include (default: 300ms)
            vad_silence_duration_ms: Silence duration to mark end of speech (default: 700ms)
            temperature: Model temperature 0.0-1.0 (default: 0.7)
            max_output_tokens: Maximum tokens per response (default: 150)
            character_slug: Oshara character slug for KB RAG (enables tool calling when set)
            kb_bearer_token: Bearer token for Oshara KB API requests
        """
        self._api_key = api_key
        self._system_prompt = system_prompt
        self._model = model
        self._vad_threshold = vad_threshold
        self._vad_prefix_padding_ms = vad_prefix_padding_ms
        self._vad_silence_duration_ms = vad_silence_duration_ms
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._ws = None
        self._listen_task = None
        self._connected = False
        self._loop = None

        # Knowledge Base RAG via function calling
        self._character_slug = character_slug
        self._kb_bearer_token = kb_bearer_token
        self._kb_api_url = f"{OSHARA_API_BASE}/knowledge-base/query/"
        self._kb_enabled = bool(character_slug)

        # Callbacks
        self.on_response_text = on_response_text
        self.on_response_text_delta = on_response_text_delta
        self.on_speech_started = on_speech_started
        self.on_speech_stopped = on_speech_stopped
        self.on_user_transcript = on_user_transcript

        # Audio resampler cache
        self._resample_cache = {}

        # Mic audio accumulation buffer: accumulate ~100ms of audio before
        # resampling+sending. This eliminates FFT edge artifacts from resampling
        # tiny 20ms WebRTC frames independently.
        self._mic_buffer = np.array([], dtype=np.float32)
        self._mic_buffer_sr = None
        self._MIC_BUFFER_FLUSH_SAMPLES = 960 * 5  # 5 WebRTC frames (~100ms at 48kHz)

        # High-pass filter state (removes sub-80Hz mic rumble that confuses VAD)
        # Maintained across calls to avoid transients at chunk boundaries.
        self._hp_filter_initialized = False
        self._hp_sos = None
        self._hp_zi = None

        # Track current response accumulation
        self._current_response_text = ""
        self._current_response_id = None

        # Track ongoing transcriptions (keyed by item_id)
        self._transcription_buffers = {}  # {item_id: accumulated_text}

        # Track in-progress function calls (keyed by item_id)
        self._pending_function_calls = (
            {}
        )  # {item_id: {"call_id": str, "name": str, "arguments": str}}

        # Reconnection state
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._should_run = False

    async def connect(self):
        """Establish WebSocket connection to OpenAI Realtime API and configure the session."""
        self._should_run = True
        self._loop = asyncio.get_event_loop()

        try:
            # Use custom model if specified, otherwise use class default
            url = self.REALTIME_URL
            if self._model:
                url = f"wss://api.openai.com/v1/realtime?model={self._model}"

            self._ws = await websockets.connect(
                url,
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
                max_size=2**24,  # 16MB max message size
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            )
            self._connected = True
            self._reconnect_attempts = 0
            print("🔌 OpenAI Realtime: WebSocket connected")

            # Wait for session.created event
            init_msg = await asyncio.wait_for(self._ws.recv(), timeout=10)
            init_event = json.loads(init_msg)
            if init_event.get("type") == "session.created":
                print(
                    f"🔌 OpenAI Realtime: Session created (id={init_event.get('session', {}).get('id', 'unknown')})"
                )

            # Configure session: text-only output, server VAD, input transcription
            await self._send_session_update()

            # Start background listener
            self._listen_task = asyncio.create_task(self._listen_loop())

        except Exception as e:
            print(f"🔌 OpenAI Realtime: Connection failed: {e}")
            traceback.print_exc()
            self._connected = False
            raise

    async def _send_session_update(self):
        """Configure the Realtime session for our use case."""
        session_body = {
            "modalities": ["text"],  # Text-only output (we use our own TTS)
            "instructions": self._system_prompt,
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "whisper-1",  # Enable input transcription for logging
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": self._vad_threshold,
                "prefix_padding_ms": self._vad_prefix_padding_ms,
                "silence_duration_ms": self._vad_silence_duration_ms,
            },
            "temperature": self._temperature,
            "max_response_output_tokens": self._max_output_tokens,
        }

        if self._kb_enabled:
            session_body["tools"] = [
                {
                    "type": "function",
                    "name": "query_knowledge_base",
                    "description": (
                        "Search the character's knowledge base for relevant information. "
                        "Call this whenever the user asks about topics that may be covered "
                        "in the knowledge base documents, such as specific facts, policies, "
                        "procedures, or detailed information the character should know."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant information",
                            }
                        },
                        "required": ["query"],
                    },
                }
            ]
            session_body["tool_choice"] = "auto"

        session_config = {"type": "session.update", "session": session_body}
        await self._ws.send(json.dumps(session_config))
        kb_status = (
            f", KB RAG enabled (slug={self._character_slug})"
            if self._kb_enabled
            else ""
        )
        print(
            f"🔌 OpenAI Realtime: Session configured (text output, server_vad threshold={self._vad_threshold}, whisper transcription{kb_status})"
        )

    # ── Knowledge Base RAG via Function Calling ──────────────────────────────

    async def _execute_function_call(self, call_id: str, name: str, arguments_str: str):
        """Dispatch a model-requested function call and return the result."""
        print(f"🔧 [ToolCall] Executing function '{name}' | call_id={call_id} | args={arguments_str[:120]}")
        try:
            args = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            print(f"🔧 [ToolCall] WARNING: Could not parse arguments JSON: {arguments_str[:120]}")
            args = {}

        if name == "query_knowledge_base":
            query = args.get("query", "")
            print(f"🔧 [ToolCall] → query_knowledge_base(query='{query}')")
            output = await self._call_kb_api(query)
        else:
            print(f"🔧 [ToolCall] WARNING: Unknown function '{name}' — returning error")
            output = json.dumps({"error": f"Unknown function: {name}"})

        print(f"🔧 [ToolCall] Result preview: {output[:200]}")
        await self._submit_function_output(call_id, output)

    async def _call_kb_api(self, query: str) -> str:
        """POST to the Oshara KB query endpoint and return formatted chunks."""
        if not self._kb_enabled:
            print(f"🔍 [KB] Skipped — KB not enabled (character_slug not set)")
            return "Knowledge base is not configured for this character."
        if not query:
            print(f"🔍 [KB] Skipped — empty query")
            return "No query provided."

        print(f"🔍 [KB] Querying '{self._kb_api_url}' | character_slug='{self._character_slug}' | query='{query}'")
        headers = {"Content-Type": "application/json"}
        if self._kb_bearer_token:
            headers["Authorization"] = f"Bearer {self._kb_bearer_token}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self._kb_api_url,
                    json={
                        "character_slug": self._character_slug,
                        "query_text": query,
                        "top_k": 3,
                    },
                    headers=headers,
                )
                print(f"🔍 [KB] HTTP {resp.status_code} from KB API")
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            print(f"🔍 [KB] Received {len(results)} result(s) for query: '{query[:60]}'")

            if not results:
                print(f"🔍 [KB] No results returned — model will answer without KB context")
                return "No relevant information found in the knowledge base."

            parts = []
            for i, r in enumerate(results):
                doc = r.get('document_filename', 'document')
                text = r.get('text', '')
                score = r.get('score', r.get('similarity', 'n/a'))
                print(f"🔍 [KB] Chunk {i+1}: [{doc}] score={score} | {text[:120]}")
                parts.append(f"[{doc}]: {text}")

            formatted = "\n\n".join(parts)
            print(f"🔍 [KB] Total context sent to model: {len(formatted)} chars")
            return formatted

        except httpx.HTTPStatusError as e:
            print(f"🔍 [KB] HTTP error {e.response.status_code}: {e.response.text[:200]}")
            return f"Knowledge base lookup failed: HTTP {e.response.status_code}"
        except Exception as e:
            print(f"🔍 [KB] Unexpected error: {e}")
            return f"Knowledge base lookup failed: {e}"

    async def _submit_function_output(self, call_id: str, output: str):
        """Send function_call_output back to the Realtime API and trigger a new response."""
        if not self._connected or self._ws is None:
            print(f"🔧 [ToolCall] Cannot submit output — WebSocket not connected (call_id={call_id})")
            return
        try:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": output,
                        },
                    }
                )
            )
            # Ask the model to continue and incorporate the retrieved context
            await self._ws.send(json.dumps({"type": "response.create"}))
            print(f"🔧 [ToolCall] Output submitted to Realtime API, triggered response.create (call_id={call_id})")
        except Exception as e:
            print(f"🔧 [ToolCall] Error submitting function output: {e}")

    async def send_audio(self, audio_np: np.ndarray, sample_rate: int):
        """Send PCM audio to the Realtime API input buffer.

        Accumulates ~100ms of audio before resampling to 24kHz PCM16 and sending.
        This eliminates FFT edge artifacts from resampling tiny 20ms WebRTC frames.
        Uses polyphase resampling, high-pass filter, and AGC for maximum clarity.

        Args:
            audio_np: Audio samples as numpy array (any dtype)
            sample_rate: Source sample rate (e.g. 48000 from WebRTC)
        """
        if not self._connected or self._ws is None:
            return

        try:
            # Normalize to float32
            if audio_np.dtype == np.int16:
                audio_f32 = audio_np.astype(np.float32) / 32768.0
            elif audio_np.dtype == np.int32:
                audio_f32 = audio_np.astype(np.float32) / 2147483648.0
            else:
                audio_f32 = audio_np.astype(np.float32)

            # Mono
            if audio_f32.ndim > 1:
                if audio_f32.shape[0] < audio_f32.shape[1]:
                    audio_f32 = np.mean(audio_f32, axis=0)
                else:
                    audio_f32 = np.mean(audio_f32, axis=1)
            audio_f32 = audio_f32.flatten()

            if audio_f32.size == 0:
                return

            # NaN check
            if np.isnan(audio_f32).any():
                if np.isnan(audio_f32).all():
                    return
                audio_f32 = np.nan_to_num(audio_f32)

            # Accumulate audio in buffer (avoids resampling tiny 20ms chunks)
            if self._mic_buffer_sr is not None and self._mic_buffer_sr != sample_rate:
                # Sample rate changed — flush old buffer first
                await self._flush_mic_buffer()
            self._mic_buffer_sr = sample_rate
            self._mic_buffer = np.concatenate([self._mic_buffer, audio_f32])

            # When enough accumulated (~100ms), process and send
            if len(self._mic_buffer) >= self._MIC_BUFFER_FLUSH_SAMPLES:
                await self._flush_mic_buffer()

        except websockets.exceptions.ConnectionClosed:
            print("🔌 OpenAI Realtime: Connection closed while sending audio")
            self._connected = False
            asyncio.create_task(self._try_reconnect())
        except Exception as e:
            # Don't spam logs for transient send errors
            if (
                not hasattr(self, "_last_send_err_time")
                or time.time() - self._last_send_err_time > 5
            ):
                print(f"🔌 OpenAI Realtime: Send audio error: {e}")
                self._last_send_err_time = time.time()

    async def _flush_mic_buffer(self):
        """Process accumulated mic audio: high-pass filter, resample, AGC, and send to OpenAI."""
        if len(self._mic_buffer) == 0 or not self._connected or self._ws is None:
            return

        audio = self._mic_buffer.copy()
        sr = self._mic_buffer_sr or 48000
        self._mic_buffer = np.array([], dtype=np.float32)

        try:
            # 1. High-pass filter at 80Hz — removes mic rumble, HVAC, breathing noise
            #    that confuses OpenAI's VAD. Filter state is maintained across calls
            #    to avoid transients at chunk boundaries.
            if not self._hp_filter_initialized or self._hp_sos is None:
                self._hp_sos = signal.butter(2, 80, btype='highpass', fs=sr, output='sos')
                self._hp_zi = signal.sosfilt_zi(self._hp_sos) * 0.0
                self._hp_filter_initialized = True
            audio, self._hp_zi = signal.sosfilt(self._hp_sos, audio, zi=self._hp_zi)
            audio = audio.astype(np.float32)

            # 2. Resample to 24kHz using polyphase filter (NOT FFT-based signal.resample)
            #    Polyphase avoids spectral ringing on chunk boundaries that degrades
            #    Whisper transcription accuracy.
            if sr != self.TARGET_SAMPLE_RATE:
                g = math.gcd(int(sr), int(self.TARGET_SAMPLE_RATE))
                up = int(self.TARGET_SAMPLE_RATE) // g
                down = int(sr) // g
                audio = signal.resample_poly(audio, up, down).astype(np.float32)

            # 3. AGC (Automatic Gain Control) — boost quiet microphones so VAD
            #    and Whisper can reliably detect and transcribe speech.
            peak = np.max(np.abs(audio))
            if 0.0 < peak < 0.1:
                # Audio is very quiet — boost to ~50% of full scale
                gain = min(0.5 / peak, 10.0)  # cap at 10x to avoid amplifying noise
                audio = audio * gain

            # Debug: Log audio stats periodically
            if not hasattr(self, "_resample_log_count"):
                self._resample_log_count = 0
            if self._resample_log_count % 20 == 0:  # Log every ~2s
                rms = np.sqrt(np.mean(audio**2))
                print(
                    f"🎤 Mic audio: {sr}Hz→24kHz, RMS={rms:.4f}, peak={np.max(np.abs(audio)):.4f}, samples={len(audio)}"
                )
            self._resample_log_count += 1

            # 4. Convert to PCM16 bytes and send
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            audio_b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")

            event = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
            await self._ws.send(json.dumps(event))

        except websockets.exceptions.ConnectionClosed:
            print("🔌 OpenAI Realtime: Connection closed while sending audio")
            self._connected = False
            asyncio.create_task(self._try_reconnect())
        except Exception as e:
            if (
                not hasattr(self, "_last_send_err_time")
                or time.time() - self._last_send_err_time > 5
            ):
                print(f"🔌 OpenAI Realtime: Flush mic buffer error: {e}")
                self._last_send_err_time = time.time()

    async def _listen_loop(self):
        """Background task that processes server events from the Realtime API."""
        print("🔌 OpenAI Realtime: Listener started")
        try:
            async for message in self._ws:
                try:
                    event = json.loads(message)
                    await self._handle_event(event)
                except json.JSONDecodeError:
                    print(f"🔌 OpenAI Realtime: Non-JSON message received")
                except Exception as e:
                    print(f"🔌 OpenAI Realtime: Event handler error: {e}")
                    traceback.print_exc()
        except websockets.exceptions.ConnectionClosed as e:
            print(f"🔌 OpenAI Realtime: Connection closed: {e}")
        except asyncio.CancelledError:
            print("🔌 OpenAI Realtime: Listener cancelled")
        except Exception as e:
            print(f"🔌 OpenAI Realtime: Listener error: {e}")
            traceback.print_exc()
        finally:
            self._connected = False
            if self._should_run:
                asyncio.create_task(self._try_reconnect())

    async def _handle_event(self, event: dict):
        """Route server events to appropriate handlers."""
        event_type = event.get("type", "")

        # --- Session events ---
        if event_type == "session.updated":
            print(f"🔌 OpenAI Realtime: Session updated successfully")

        # --- VAD events ---
        elif event_type == "input_audio_buffer.speech_started":
            print(
                f"🎤 OpenAI VAD: Speech started (audio_start_ms={event.get('audio_start_ms')})"
            )
            if self.on_speech_started:
                await self.on_speech_started()

        elif event_type == "input_audio_buffer.speech_stopped":
            print(
                f"🎤 OpenAI VAD: Speech stopped (audio_end_ms={event.get('audio_end_ms')})"
            )
            if self.on_speech_stopped:
                await self.on_speech_stopped()

        elif event_type == "input_audio_buffer.committed":
            print(f"🎤 OpenAI VAD: Audio committed (item_id={event.get('item_id')})")

        # --- Transcription events ---
        elif event_type == "conversation.item.input_audio_transcription.delta":
            # Accumulate streaming transcription deltas
            item_id = event.get("item_id")
            delta = event.get("delta", "")
            content_index = event.get("content_index", 0)

            if item_id:
                if item_id not in self._transcription_buffers:
                    self._transcription_buffers[item_id] = ""
                self._transcription_buffers[item_id] += delta
                # Log progress periodically (every ~50 chars to avoid spam)
                current_text = self._transcription_buffers[item_id]
                if len(current_text) % 50 < len(delta):
                    print(
                        f"🎤 Transcribing... '{current_text[:60]}...' ({len(current_text)} chars)"
                    )

        elif event_type == "conversation.item.input_audio_transcription.completed":
            item_id = event.get("item_id")
            transcript = event.get("transcript", "").strip()

            # Use accumulated buffer if available, otherwise use the transcript field
            if item_id and item_id in self._transcription_buffers:
                transcript = self._transcription_buffers[item_id].strip()
                # Clean up buffer
                del self._transcription_buffers[item_id]

            if transcript:
                print(f"🎤 User said: '{transcript}'")
                if self.on_user_transcript:
                    await self.on_user_transcript(transcript)
            else:
                print(f"🎤 Transcription completed but empty (item_id={item_id})")

        elif event_type == "conversation.item.input_audio_transcription.failed":
            item_id = event.get("item_id")
            error = event.get("error", {})
            print(f"🎤 Transcription failed: {error.get('message', 'unknown')}")
            # Clean up buffer if exists
            if item_id and item_id in self._transcription_buffers:
                del self._transcription_buffers[item_id]

        # --- Response events ---
        elif event_type == "response.created":
            resp = event.get("response", {})
            self._current_response_id = resp.get("id")
            self._current_response_text = ""

        elif event_type == "response.text.delta":
            delta_text = event.get("delta", "")
            self._current_response_text += delta_text
            if self.on_response_text_delta:
                await self.on_response_text_delta(delta_text)

        elif event_type == "response.text.done":
            full_text = event.get("text", "") or self._current_response_text
            # This fires per content part — accumulate

        elif event_type == "response.output_item.done":
            # Content part is complete; log it
            item = event.get("item", {})
            if item.get("type") == "message" and item.get("role") == "assistant":
                content_parts = item.get("content", [])
                for part in content_parts:
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            self._current_response_text = text

        elif event_type == "response.done":
            resp = event.get("response", {})
            status = resp.get("status", "unknown")

            # Extract full text from response output
            full_text = ""
            for output_item in resp.get("output", []):
                if output_item.get("type") == "message":
                    for content in output_item.get("content", []):
                        if content.get("type") == "text":
                            full_text += content.get("text", "")

            if not full_text:
                full_text = self._current_response_text

            if status in ("completed", "incomplete") and full_text.strip():
                if status == "incomplete":
                    reason = resp.get("status_details", {}).get("reason", "max_tokens")
                    print(f"⚠️  AI response truncated (incomplete, reason={reason}) — using partial text ({len(full_text)} chars)")
                else:
                    print(f"🤖 AI response: '{full_text.strip()[:80]}...'")
                if self.on_response_text:
                    await self.on_response_text(full_text.strip())
            elif status == "incomplete" and not full_text.strip():
                # Truncated before any text was produced (e.g. tool call hit token limit mid-arguments)
                reason = resp.get("status_details", {}).get("reason", "max_tokens")
                print(f"⚠️  AI response incomplete with no text (reason={reason}) — flushing accumulator to unblock pipeline")
                if self.on_response_text:
                    await self.on_response_text("")
            elif status == "cancelled":
                print(f"🤖 AI response cancelled")
            elif status == "failed":
                error = resp.get("status_details", {}).get("error", {})
                print(f"🤖 AI response failed: {error}")
            elif status == "completed" and not full_text.strip():
                # Completed with no text — a tool call turn; no flush needed here
                print(f"🤖 AI response completed (tool call turn — no text output)")

            self._current_response_text = ""
            self._current_response_id = None

            # Log token usage
            usage = resp.get("usage", {})
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
                print(
                    f"📊 Tokens: {input_tokens} in, {output_tokens} out, {total_tokens} total"
                )
            else:
                # Debug: log when usage is missing
                print(f"⚠️  No usage data in response (status={status})")

        # --- Error events ---
        elif event_type == "error":
            error = event.get("error", {})
            error_type = error.get("type", "unknown")
            error_code = error.get("code", "")
            error_message = error.get("message", "no message")

            # Handle specific error types
            if error_type == "invalid_request_error":
                print(f"❌ OpenAI Realtime: Invalid request - {error_message}")
                # Could be bad session config, invalid audio format, etc.
            elif error_type == "server_error":
                print(f"❌ OpenAI Realtime: Server error - {error_message}")
                # Might want to reconnect automatically
                if self._should_run:
                    asyncio.create_task(self._try_reconnect())
            elif error_code == "rate_limit_exceeded":
                print(f"❌ OpenAI Realtime: Rate limit exceeded - {error_message}")
                # The API will send rate_limits.updated events to inform about limits
            elif error_code == "session_expired":
                print(f"❌ OpenAI Realtime: Session expired - {error_message}")
                # Reconnect to create a new session
                if self._should_run:
                    asyncio.create_task(self._try_reconnect())
            else:
                print(f"❌ OpenAI Realtime error: {error_type}: {error_message}")

        # --- Rate limits ---
        elif event_type == "rate_limits.updated":
            pass  # Silently ignore rate limit updates

        # --- Conversation item events (informational) ---
        elif event_type == "response.output_item.added":
            # Track function_call items as they are opened
            item = event.get("item", {})
            if item.get("type") == "function_call":
                item_id = item.get("id", "")
                call_id = item.get("call_id", "")
                name = item.get("name", "")
                self._pending_function_calls[item_id] = {
                    "call_id": call_id,
                    "name": name,
                    "arguments": "",
                }
                print(
                    f"� [ToolCall] Model requested tool: '{name}' | call_id={call_id} | item_id={item_id} | KB enabled={self._kb_enabled}"
                )

        # --- Function call argument streaming ---
        elif event_type == "response.function_call_arguments.delta":
            item_id = event.get("item_id", "")
            delta = event.get("delta", "")
            if item_id in self._pending_function_calls:
                self._pending_function_calls[item_id]["arguments"] += delta

        elif event_type == "response.function_call_arguments.done":
            item_id = event.get("item_id", "")
            final_args = event.get("arguments", "")
            pending = self._pending_function_calls.pop(item_id, None)
            if pending:
                call_id = pending["call_id"] or event.get("call_id", "")
                name = pending["name"] or event.get("name", "")
                arguments = final_args or pending["arguments"]
                print(f"� [ToolCall] Arguments ready for '{name}': {arguments[:120]} (call_id={call_id})")
                asyncio.create_task(
                    self._execute_function_call(call_id, name, arguments)
                )
            else:
                print(f"🔧 [ToolCall] WARNING: Got arguments.done for unknown item_id={item_id}")

        elif event_type in (
            "conversation.item.created",
            "conversation.item.added",
            "conversation.created",
            "response.content_part.added",
            "response.content_part.done",
        ):
            pass  # Normal lifecycle events, no action needed

        else:
            # Log unknown event types for debugging
            if not event_type.startswith("response."):
                print(f"🔌 OpenAI Realtime: Unhandled event: {event_type}")

    async def cancel_response(self):
        """Cancel any in-progress response from the model."""
        if not self._connected or self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"type": "response.cancel"}))
            print("🔌 OpenAI Realtime: Response cancelled")
        except Exception as e:
            print(f"🔌 OpenAI Realtime: Cancel error: {e}")

    async def clear_audio_buffer(self):
        """Clear the input audio buffer on the server."""
        if not self._connected or self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        except Exception as e:
            print(f"🔌 OpenAI Realtime: Clear buffer error: {e}")

    async def create_response(self, modalities=None):
        """Manually trigger a response from the model.

        Useful for:
        - Push-to-talk mode (commit audio buffer then trigger response)
        - Text-only conversations
        - When turn_detection is disabled

        Args:
            modalities: List of output modalities (e.g., ["text"], ["audio"], ["text", "audio"])
                       If None, uses session defaults (text-only in our case)
        """
        if not self._connected or self._ws is None:
            return
        try:
            event = {"type": "response.create"}
            if modalities:
                event["response"] = {"modalities": modalities}
            await self._ws.send(json.dumps(event))
            print("🔌 OpenAI Realtime: Response triggered manually")
        except Exception as e:
            print(f"🔌 OpenAI Realtime: Create response error: {e}")

    async def commit_audio_buffer(self):
        """Manually commit the audio buffer to create a conversation item.

        Useful in push-to-talk mode: user finishes speaking, you commit
        the buffer to transcribe it, then call create_response().
        """
        if not self._connected or self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            print("🔌 OpenAI Realtime: Audio buffer committed")
        except Exception as e:
            print(f"🔌 OpenAI Realtime: Commit buffer error: {e}")

    async def add_conversation_item(
        self, role: str, content: str, item_type: str = "message"
    ):
        """Manually add a conversation item (useful for injecting context).

        Args:
            role: "user", "assistant", or "system"
            content: Text content of the message
            item_type: "message" or "function_call" or "function_call_output"

        Example use cases:
        - Pre-load conversation history when reconnecting
        - Inject system messages mid-conversation
        - Add assistant messages for few-shot prompting
        """
        if not self._connected or self._ws is None:
            return
        try:
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": item_type,
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                },
            }
            await self._ws.send(json.dumps(event))
            print(f"🔌 OpenAI Realtime: Added {role} message to conversation")
        except Exception as e:
            print(f"🔌 OpenAI Realtime: Add conversation item error: {e}")

    async def update_instructions(self, instructions: str):
        """Update the system prompt mid-session."""
        if not self._connected or self._ws is None:
            return
        try:
            event = {
                "type": "session.update",
                "session": {
                    "instructions": instructions,
                },
            }
            await self._ws.send(json.dumps(event))
            self._system_prompt = instructions
            print("🔌 OpenAI Realtime: Instructions updated")
        except Exception as e:
            print(f"🔌 OpenAI Realtime: Update instructions error: {e}")

    async def _try_reconnect(self):
        """Attempt to reconnect to the Realtime API with exponential backoff."""
        if not self._should_run:
            return

        self._reconnect_attempts += 1
        if self._reconnect_attempts > self._max_reconnect_attempts:
            print(
                f"🔌 OpenAI Realtime: Max reconnection attempts ({self._max_reconnect_attempts}) reached, giving up"
            )
            return

        delay = min(2**self._reconnect_attempts, 30)  # Exponential backoff, max 30s
        print(
            f"🔌 OpenAI Realtime: Reconnecting in {delay}s (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})"
        )
        await asyncio.sleep(delay)

        try:
            await self.connect()
            print("🔌 OpenAI Realtime: Reconnected successfully")
        except Exception as e:
            print(f"🔌 OpenAI Realtime: Reconnection failed: {e}")

    async def disconnect(self):
        """Cleanly disconnect from the Realtime API."""
        self._should_run = False

        # Flush any remaining accumulated mic audio before closing
        try:
            await self._flush_mic_buffer()
        except Exception:
            pass

        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listen_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._connected = False
        # Reset filter state for next session
        self._hp_filter_initialized = False
        self._hp_zi = None
        self._mic_buffer = np.array([], dtype=np.float32)
        print("🔌 OpenAI Realtime: Disconnected")

    @property
    def connected(self):
        return self._connected


# --- Streaming Sentence Accumulator ---


class StreamingSentenceAccumulator:
    """Accumulates streaming tokens from OpenAI Realtime API and emits
    complete sentences as soon as sentence boundaries are detected.

    This allows TTS to start on the first sentence while the LLM is still
    generating subsequent sentences, saving ~500-1500ms on first-frame latency.
    """

    # Regex pattern for sentence boundaries: period, !, ? followed by space or end
    _SENTENCE_END = re.compile(r"[.!?](?:\s|$)")

    def __init__(self, sentence_queue: asyncio.Queue):
        self.buffer = ""
        self.sentence_index = 0
        self.sentence_queue = sentence_queue
        self._pipeline_started = False
        self._response_text = ""  # Full response accumulation for logging

    async def add_token(self, token: str):
        """Called per-token from OpenAI response.text.delta."""
        self.buffer += token
        self._response_text += token

        # Check for sentence boundary in buffer
        match = self._SENTENCE_END.search(self.buffer)
        if match:
            end_pos = match.end()
            sentence = self.buffer[:end_pos].strip()
            self.buffer = self.buffer[end_pos:].lstrip()
            if sentence:
                await self.sentence_queue.put(sentence)
                self.sentence_index += 1

    async def flush(self):
        """Called on response.done — emit any remaining buffered text."""
        remaining = self.buffer.strip()
        if remaining:
            await self.sentence_queue.put(remaining)
            self.sentence_index += 1
            self.buffer = ""
        await self.sentence_queue.put(None)  # Sentinel: end of response

    def reset(self):
        """Reset for next response."""
        self.buffer = ""
        self.sentence_index = 0
        self._pipeline_started = False
        self._response_text = ""

    @property
    def full_text(self):
        return self._response_text


# --- Conversation Manager ---


class ConversationManager:
    def __init__(
        self,
        openai_client,
        tts_model,
        imtalker_streamer,
        video_track,
        audio_track,
        avatar_img,
        system_prompt=None,
        reference_audio=None,
        sync_manager=None,
        enable_realtime=True,
        webrtc_session_id=None,
        full_img_rgb=None,
        crop_coords=None,
        character_slug=None,
        kb_bearer_token=None,
    ):
        self.openai_client = openai_client
        self.tts_model = tts_model
        self.imtalker_streamer = imtalker_streamer
        self.video_track = video_track
        self.audio_track = audio_track
        self.avatar_img = avatar_img
        self.sync_manager = sync_manager
        self.webrtc_session_id = webrtc_session_id  # For Oshara log lookup

        # Crop-Infer-Paste: when set, rendered 512x512 face frames are composited
        # back into the full-resolution original image before sending to WebRTC.
        self._full_img_rgb = full_img_rgb  # np.ndarray (H, W, 3) uint8, or None
        self._crop_coords = crop_coords  # (x1, y1, x2, y2), or None
        self._paste_back_enabled = full_img_rgb is not None and crop_coords is not None
        if self._paste_back_enabled:
            print(
                f"CM: Crop-Infer-Paste enabled — output will be {full_img_rgb.shape[1]}x{full_img_rgb.shape[0]}"
            )

        # Frame interpolation settings
        self.enable_frame_blending = True  # Smooth transitions between chunks
        self.blend_steps = 1  # Single interpolated frame (lighter for real-time)
        self.last_chunk_final_frame = None  # Track last frame for blending

        # Pipeline parallelization settings
        self.enable_parallel_pipeline = True  # Concurrent TTS/Render
        from concurrent.futures import ThreadPoolExecutor

        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TTS")
        self.render_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="Render"
        )

        self.is_speaking = False  # State of AI response
        self.user_is_talking = False  # State of user speech detection
        # Ultra-brief prompt — optimized for <2s response time
        default_prompt = """You are a concise AI avatar. Reply in 1-2 short sentences MAX. Be direct, warm, and conversational. Never use lists or long explanations."""
        self.system_prompt = system_prompt if system_prompt else default_prompt
        self.reference_audio = reference_audio

        # Conversation history is now managed by OpenAI Realtime API server-side.
        # We keep a local mirror only for logging/debugging.
        self.conversation_history = []
        self.max_history_turns = 6

        self._processing_task = None
        self._loop = asyncio.get_event_loop()

        # ── Barge-in speech confirmation ──
        # When OpenAI VAD fires during AI speech, we verify with local audio
        # energy to avoid false barge-ins from ambient noise / brief sounds.
        self._barge_in_pending = False
        self._barge_in_energy_samples = []  # recent mic energy readings
        self._barge_in_start_time = 0.0
        self._BARGE_IN_ENERGY_THRESHOLD = 0.015  # RMS energy for real speech
        self._BARGE_IN_CONFIRM_WINDOW = 0.35  # seconds of sustained speech needed
        self._BARGE_IN_MIN_FRAMES = 5  # minimum frames above threshold

        # ── Pipeline timing tracking (for Oshara logs) ──
        self._last_user_msg_time = None
        self._current_pipeline_start = None
        self._current_llm_latency = None
        self._pipeline_tts_time = None
        self._pipeline_render_time = None
        self._pipeline_stream_time = None
        self._pipeline_total_time = None
        self._pipeline_total_frames = None

        # ── Streaming sentence accumulator (Optimization: token-level pipelining) ──
        # Instead of waiting for the full LLM response, we detect sentence
        # boundaries in the token stream and start TTS/render immediately.
        self._streaming_sentence_queue = asyncio.Queue(maxsize=6)
        self._sentence_accumulator = StreamingSentenceAccumulator(
            self._streaming_sentence_queue
        )

        # ── Thinking transition state ──
        self._thinking_active = False

        # ── OpenAI Realtime API Session ──
        # Replaces: local VAD + Whisper STT + ChatCompletions LLM
        # The Realtime API handles VAD, transcription, conversation context,
        # and generates text responses. We feed those into our TTS→render pipeline.
        # OPTIMIZED: Uses on_response_text_delta for per-token streaming instead
        # of waiting for the full response (saves ~500-1500ms on first frame).
        self.realtime_session = None
        self._realtime_connected = False
        if enable_realtime:
            api_key = os.getenv("OPENAI_API_KEY", "")
            self.realtime_session = OpenAIRealtimeSession(
                api_key=api_key,
                system_prompt=self.system_prompt,
                on_response_text=self._on_streaming_response_done,
                on_response_text_delta=self._on_streaming_token,
                on_speech_started=self._on_realtime_speech_started,
                on_speech_stopped=self._on_realtime_speech_stopped,
                on_user_transcript=self._on_realtime_user_transcript,
                # VAD tuning: lower threshold for better sensitivity, longer silence
                # tolerance so natural pauses don't trigger end-of-turn, more prefix
                # padding to capture the start of speech reliably.
                vad_threshold=0.3,  # Lowered from 0.5 default — detects softer speech
                vad_prefix_padding_ms=500,  # Raised from 300ms — captures more speech onset
                vad_silence_duration_ms=550,  # Balanced: 400 was too aggressive (cut off mid-sentence), 700 too slow
                max_output_tokens=500,  # Raised: 100 caused silent truncation (status=incomplete) on detailed topics
                temperature=0.9,  # Increased from 0.7 default (-50-100ms LLM time)
                # Knowledge Base RAG via function calling
                character_slug=character_slug,
                kb_bearer_token=kb_bearer_token,
            )

    async def connect_realtime(self):
        """Connect the OpenAI Realtime session. Call after __init__ from an async context."""
        if not self.realtime_session:
            print("CM: No Realtime session configured, skipping connect")
            return
        try:
            await self.realtime_session.connect()
            self._realtime_connected = True
            print("CM: OpenAI Realtime session connected")
        except Exception as e:
            print(f"CM: Failed to connect Realtime session: {e}")
            self._realtime_connected = False

    def paste_back_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Paste a rendered 512x512 face frame back into the full-resolution image.

        If Crop-Infer-Paste is not enabled, returns the frame unchanged.

        Args:
            frame_rgb: np.ndarray (H, W, 3) uint8 — rendered face frame (typically 512x512)
        Returns:
            np.ndarray (H, W, 3) uint8 — full-res image with face composited, or original frame
        """
        if not self._paste_back_enabled:
            return frame_rgb

        x1, y1, x2, y2 = self._crop_coords
        region_h, region_w = y2 - y1, x2 - x1

        # Resize rendered face to the crop region size
        face_resized = cv2.resize(
            frame_rgb, (region_w, region_h), interpolation=cv2.INTER_LINEAR
        )

        # Composite into a copy of the full image
        result = self._full_img_rgb.copy()
        result[y1:y2, x1:x2] = face_resized
        return result

    async def cleanup_async(self):
        """Async cleanup — disconnects Realtime session."""
        if self.realtime_session:
            await self.realtime_session.disconnect()
            self._realtime_connected = False

    def cleanup(self):
        """Release all resources held by this ConversationManager."""
        print("CM: Cleaning up resources...")
        self.is_speaking = False
        self.user_is_talking = False

        # Disconnect Realtime session (schedule if event loop is running)
        if self.realtime_session and self._realtime_connected:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.cleanup_async())
                else:
                    loop.run_until_complete(self.cleanup_async())
            except Exception:
                pass

        # Shutdown thread pool executors
        try:
            self.tts_executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            self.render_executor.shutdown(wait=False)
        except Exception:
            pass

        # Clear references to tracks/models (don't delete — shared)
        self.video_track = None
        self.audio_track = None
        self.avatar_img = None
        self.last_chunk_final_frame = None
        self._thinking_active = False

        # Drain streaming sentence queue
        while not self._streaming_sentence_queue.empty():
            try:
                self._streaming_sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Cancel any pending processing task
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()

        print("CM: Cleanup complete.")

    # ── OpenAI Realtime API Callbacks (Streaming Pipeline) ──

    async def _on_streaming_token(self, delta: str):
        """Called per-token from OpenAI response.text.delta.

        Feeds tokens into the StreamingSentenceAccumulator which detects
        sentence boundaries and pushes complete sentences to the pipeline.
        The pipeline starts on the FIRST sentence — not the full response.
        """
        # If this is the very first token of a new response, create a fresh queue
        if self._sentence_accumulator._response_text == "":
            # Fresh queue for this response (old pipeline drains the old one)
            self._streaming_sentence_queue = asyncio.Queue(maxsize=6)
            self._sentence_accumulator.sentence_queue = self._streaming_sentence_queue

        await self._sentence_accumulator.add_token(delta)

        # Start the streaming pipeline as soon as the first sentence is ready
        if (
            not self._sentence_accumulator._pipeline_started
            and self._sentence_accumulator.sentence_index > 0
        ):
            self._sentence_accumulator._pipeline_started = True

            # Stop thinking animation if active
            self._thinking_active = False

            # Measure LLM latency to first sentence
            llm_latency = None
            if hasattr(self, "_last_user_msg_time") and self._last_user_msg_time:
                llm_latency = round(time.time() - self._last_user_msg_time, 3)
            self._current_pipeline_start = time.time()
            self._current_llm_latency = llm_latency
            print(
                f"CM: First sentence detected at {llm_latency}s — starting streaming pipeline"
            )

            # Cancel any existing pipeline
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except (asyncio.CancelledError, Exception):
                    pass

            # Start concurrent pipeline consuming from _streaming_sentence_queue
            self._processing_task = asyncio.create_task(self._run_streaming_pipeline())

    async def _on_streaming_response_done(self, full_text: str):
        """Called when OpenAI response.done fires — flush any remaining text.

        If the accumulator never fired (very short response), fall back to
        the original full-text pipeline.
        """
        full_text = (
            full_text.strip()
            if full_text
            else self._sentence_accumulator.full_text.strip()
        )
        print(f"CM: Realtime response complete: '{full_text[:80]}...'")

        # Mirror to local history for logging
        if full_text:
            self.conversation_history.append(
                {"role": "assistant", "content": full_text}
            )

        # Record to Oshara conversation logs
        llm_latency = self._current_llm_latency
        if (
            llm_latency is None
            and hasattr(self, "_last_user_msg_time")
            and self._last_user_msg_time
        ):
            llm_latency = round(time.time() - self._last_user_msg_time, 3)
            self._current_llm_latency = llm_latency
        _record_oshara_log(
            app,
            response=full_text,
            additional_data={
                "llm_response_time_s": llm_latency,
            },
            session_id=self.webrtc_session_id,
        )
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[
                -(self.max_history_turns * 2) :
            ]

        if not self._sentence_accumulator._pipeline_started:
            # Very short response (no sentence boundary detected during streaming)
            # Fall back: put entire text + sentinel into queue and start pipeline
            self._sentence_accumulator._pipeline_started = True
            self._thinking_active = False

            if not self._current_pipeline_start:
                self._current_pipeline_start = time.time()

            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except (asyncio.CancelledError, Exception):
                    pass

            if full_text:
                await self._streaming_sentence_queue.put(full_text)
            await self._streaming_sentence_queue.put(None)

            self._processing_task = asyncio.create_task(self._run_streaming_pipeline())
        else:
            # Normal path: flush remaining buffered text and send sentinel
            await self._sentence_accumulator.flush()

        # Reset accumulator for next response
        self._sentence_accumulator.reset()

    async def _on_realtime_speech_started(self):
        """Called when OpenAI server VAD detects user started speaking.

        If the AI is NOT speaking, just set user_is_talking.
        If the AI IS speaking, start a barge-in confirmation window:
        we collect mic audio energy for ~350ms and only actually interrupt
        if sustained speech energy is detected (filters clicks, coughs, noise).
        """
        print("CM: User speech started (OpenAI VAD)")
        self.user_is_talking = True

        if self.is_speaking:
            # Don't barge-in immediately — start confirmation window
            self._barge_in_pending = True
            self._barge_in_energy_samples = []
            self._barge_in_start_time = time.time()
            print("CM: Barge-in pending — confirming sustained speech...")

    async def _check_barge_in_audio(self, audio_np, sample_rate):
        """Check if incoming mic audio confirms a real barge-in (sustained speech).

        Called from add_microphone_audio when _barge_in_pending is True.
        Collects energy readings over a short window, and only fires barge-in
        if enough frames exceed the speech energy threshold.
        """
        # Calculate RMS energy of this audio chunk
        if audio_np.dtype == np.int16:
            chunk = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            chunk = audio_np.astype(np.float32) / 2147483648.0
        else:
            chunk = audio_np.astype(np.float32)

        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=0 if chunk.shape[0] < chunk.shape[1] else 1)

        rms = np.sqrt(np.mean(chunk**2))
        self._barge_in_energy_samples.append(rms)

        elapsed = time.time() - self._barge_in_start_time

        # Count how many frames have speech-level energy
        speech_frames = sum(
            1
            for e in self._barge_in_energy_samples
            if e > self._BARGE_IN_ENERGY_THRESHOLD
        )
        total_frames = len(self._barge_in_energy_samples)

        # Log periodically
        if total_frames % 5 == 0:
            avg_rms = np.mean(self._barge_in_energy_samples[-5:])
            print(
                f"CM: Barge-in check: {speech_frames}/{total_frames} speech frames, avg RMS={avg_rms:.4f}, elapsed={elapsed:.2f}s"
            )

        # Decision: enough sustained speech → confirm barge-in
        if speech_frames >= self._BARGE_IN_MIN_FRAMES and elapsed >= 0.15:
            print(
                f"🛑 BARGE-IN CONFIRMED: {speech_frames}/{total_frames} frames above threshold after {elapsed:.2f}s"
            )
            self._barge_in_pending = False
            self._barge_in_energy_samples = []
            await self._execute_barge_in()
            return

        # Timeout: if we've waited long enough without confirmation, cancel the pending barge-in
        if elapsed > self._BARGE_IN_CONFIRM_WINDOW:
            avg_rms = (
                np.mean(self._barge_in_energy_samples)
                if self._barge_in_energy_samples
                else 0
            )
            print(
                f"CM: Barge-in dismissed — only {speech_frames}/{total_frames} speech frames (avg RMS={avg_rms:.4f})"
            )
            self._barge_in_pending = False
            self._barge_in_energy_samples = []

    async def _execute_barge_in(self):
        """Actually perform the barge-in: cancel pipeline, flush tracks."""
        print("🛑 BARGE-IN: User interrupted — cancelling speech pipeline...")
        # Cancel any in-progress response from Realtime API
        await self.realtime_session.cancel_response()

        # Stop thinking animation
        self._thinking_active = False

        # Reset sentence accumulator to prevent stale tokens from old response
        self._sentence_accumulator.reset()

        # Drain streaming sentence queue
        while not self._streaming_sentence_queue.empty():
            try:
                self._streaming_sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Cancel our TTS/render pipeline
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except (asyncio.CancelledError, Exception):
                pass
            self._processing_task = None

        # Flush WebRTC track queues
        if self.video_track:
            drained = 0
            while not self.video_track.frame_queue.empty():
                try:
                    self.video_track.frame_queue.get_nowait()
                    drained += 1
                except asyncio.QueueEmpty:
                    break
            self.video_track.stop_speaking()
            if drained:
                print(f"   Drained {drained} video frames")

        if self.audio_track:
            drained = 0
            while not self.audio_track.audio_queue.empty():
                try:
                    self.audio_track.audio_queue.get_nowait()
                    drained += 1
                except asyncio.QueueEmpty:
                    break
            self.audio_track.stop_expecting()
            if drained:
                print(f"   Drained {drained} audio frames")

        self.is_speaking = False
        print("🛑 BARGE-IN: Pipeline cancelled, tracks flushed — listening to user")

    async def _on_realtime_speech_stopped(self):
        """Called when OpenAI server VAD detects user stopped speaking.

        Starts a subtle 'thinking' transition on the idle animation to
        fill the visual gap while LLM generates the response (~500-1500ms).
        This makes the avatar feel responsive even before speech starts.
        """
        print("CM: User speech stopped (OpenAI VAD)")
        self.user_is_talking = False

        # Start thinking transition if not already speaking
        if (
            not self.is_speaking
            and self.video_track
            and hasattr(self.video_track, "_idle_frames")
            and self.video_track._idle_frames
        ):
            self._thinking_active = True
            asyncio.create_task(self._play_thinking_transition())

    async def _play_thinking_transition(self):
        """Push sped-up idle frames to give a subtle 'thinking' animation.

        Runs until the streaming pipeline starts (first sentence detected).
        Uses existing idle frames at 1.5x speed for a subtle head nod effect.
        """
        try:
            think_idx = self.video_track._idle_index
            while self._thinking_active and not self.is_speaking:
                # Advance idle animation at 1.5x speed (skip every 3rd frame)
                think_idx += 2
                if self.video_track._idle_frames:
                    frame = self.video_track._idle_frames[
                        think_idx % len(self.video_track._idle_frames)
                    ]
                    self.video_track._idle_index = think_idx
                await asyncio.sleep(0.027)  # ~37 FPS = 1.5x normal 25 FPS
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            self._thinking_active = False

    async def _on_realtime_user_transcript(self, text: str):
        """Called when the user's speech is transcribed."""
        print(f"CM: User transcript: '{text}'")
        # Mirror to local history for logging
        self.conversation_history.append({"role": "user", "content": text})

        # ── Record to Oshara conversation logs ──
        # Track when user message arrived so we can measure LLM response latency
        self._last_user_msg_time = time.time()
        _record_oshara_log(
            app,
            message=text,
            session_id=self.webrtc_session_id,
        )
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[
                -(self.max_history_turns * 2) :
            ]

    async def add_microphone_audio(self, audio_np, sample_rate):
        """Called when new audio comes from WebRTC microphone.

        Instead of doing local VAD/STT, we forward all audio to the OpenAI
        Realtime API, which handles VAD, transcription, and response generation.
        """
        if audio_np is None or audio_np.size == 0:
            return

        if not self._realtime_connected:
            return

        # If barge-in is pending, check this audio chunk for sustained speech
        if self._barge_in_pending:
            await self._check_barge_in_audio(audio_np, sample_rate)

        # Forward audio directly to the Realtime API
        await self.realtime_session.send_audio(audio_np, sample_rate)

    async def _run_streaming_pipeline(self):
        """Streaming pipeline with SYNCED audio and video delivery.

        Audio and video are pushed in lockstep: each rendered video frame
        is paired with its corresponding audio samples, ensuring tight
        lip sync. TTS still runs ahead via pipeline parallelism.

        Architecture (all stages run concurrently via asyncio.gather):
          sentence_queue (fed live by accumulator as LLM tokens stream in)
              ↓
          Stage 1 (tts_worker): sentence → streaming TTS → audio_data_queue
              ↓
          Stage 2 (parallel_av_handler): audio_data_queue →
              └─ prepares WebRTC audio frames + forwards to render_queue
              ↓
          Stage 3 (video_renderer): render_queue → IMTalker →
              ├─ video_track.frame_queue (per frame)
              └─ audio_track.audio_queue (2 audio frames per video frame)

        Key optimizations:
        - TTS starts on FIRST sentence while LLM still generates
        - Sub-sentence streaming: render starts at 1.5s of audio
        - Audio + video pushed in lockstep for tight lip sync
        """
        pipeline_start = self._current_pipeline_start or time.time()
        print("CM: Starting PARALLEL A/V streaming pipeline")

        pipeline_timings = {
            "pipeline_start_utc": datetime.now(timezone.utc).isoformat(),
            "llm_response_time_s": self._current_llm_latency,
            "response_text": "",
            "tts_total_time_s": None,
            "tts_per_sentence": [],
            "render_total_time_s": None,
            "render_per_sentence": [],
            "total_audio_duration_s": 0.0,
            "first_byte_latency_s": None,
            "streaming_time_s": None,
            "total_pipeline_time_s": None,
            "total_frames": 0,
            "status": "started",
            "pipeline_mode": "streaming_token_level",
        }

        try:
            self.is_speaking = True
            self._thinking_active = False  # Stop thinking animation

            # Inter-stage queues for parallel A/V pipeline
            audio_data_queue = asyncio.Queue(maxsize=8)  # TTS → parallel_av_handler
            render_queue = (
                asyncio.Queue()
            )  # parallel_av_handler → video_renderer (unbounded: never block audio delivery)

            SAMPLES_PER_FRAME = 960  # 24000 Hz / 25 FPS
            MIN_RENDER_DURATION_S = (
                3.0  # Start rendering with 3.0s of audio (sub-sentence streaming)
            )

            # ── Stage 1: TTS Worker — progressive push: sub-chunks go to queue AS they're ready ──
            async def tts_worker():
                tts_total_time = 0
                chunk_idx = 0

                while True:
                    sentence = await self._streaming_sentence_queue.get()
                    if sentence is None:
                        await audio_data_queue.put(None)
                        break

                    chunk_idx += 1
                    t_tts = time.time()
                    pipeline_timings["response_text"] += (
                        " " if pipeline_timings["response_text"] else ""
                    ) + sentence

                    _cur_idx = chunk_idx  # capture for closure
                    _cur_sentence = sentence
                    _loop = self._loop
                    _queue = audio_data_queue
                    _ref_audio = self.reference_audio
                    _tts_model = self.tts_model

                    def do_tts_progressive():
                        """Run TTS and push sub-chunks directly to the async queue
                        via run_coroutine_threadsafe so downstream stages can
                        start processing (audio delivery + rendering) while TTS
                        is still generating the rest of the sentence."""
                        total_samples = 0
                        sub_chunk_count = 0

                        # Check if text is Nepali and use Sherpa TTS if available
                        if (
                            SHERPA_AVAILABLE
                            and detect_nepali(_cur_sentence)
                            and hasattr(app.state, "sherpa_tts")
                            and app.state.sherpa_tts is not None
                        ):
                            try:
                                audio_np = app.state.sherpa_tts.generate(
                                    _cur_sentence, sid=5, speed=1.0
                                )
                                sherpa_sr = app.state.sherpa_tts.sample_rate
                                if sherpa_sr != 24000:
                                    import librosa

                                    audio_np = librosa.resample(
                                        audio_np, orig_sr=sherpa_sr, target_sr=24000
                                    )
                                if audio_np is not None:
                                    chunk = audio_np.flatten().copy()
                                    fut = asyncio.run_coroutine_threadsafe(
                                        _queue.put(
                                            (chunk, _cur_sentence, _cur_idx, True)
                                        ),
                                        _loop,
                                    )
                                    fut.result(timeout=30)
                                    return {
                                        "total_samples": len(chunk),
                                        "sub_chunks": 1,
                                    }
                            except Exception as e:
                                print(f"⚠️  Sherpa TTS failed, falling back: {e}")

                        # STTS streaming with progressive accumulation
                        tts_stream = _tts_model.generate_stream(
                            text=_cur_sentence,
                            chunk_size=100,
                            print_metrics=False,
                            audio_prompt_path=_ref_audio,
                        )

                        accumulated = []
                        accumulated_samples = 0
                        sent_first = False

                        for audio_chunk, metrics in tts_stream:
                            chunk_np = (
                                audio_chunk.cpu().numpy().flatten()
                                if isinstance(audio_chunk, torch.Tensor)
                                else audio_chunk.flatten()
                            )
                            accumulated.append(chunk_np)
                            accumulated_samples += len(chunk_np)
                            current_duration = accumulated_samples / 24000

                            if (
                                not sent_first
                                and current_duration >= MIN_RENDER_DURATION_S
                            ):
                                audio_np = np.concatenate(accumulated)
                                fut = asyncio.run_coroutine_threadsafe(
                                    _queue.put(
                                        (
                                            audio_np.copy(),
                                            _cur_sentence,
                                            _cur_idx,
                                            False,
                                        )
                                    ),
                                    _loop,
                                )
                                fut.result(timeout=30)
                                total_samples += len(audio_np)
                                sub_chunk_count += 1
                                accumulated = []
                                accumulated_samples = 0
                                sent_first = True
                            elif sent_first and current_duration >= 2.0:
                                audio_np = np.concatenate(accumulated)
                                fut = asyncio.run_coroutine_threadsafe(
                                    _queue.put(
                                        (
                                            audio_np.copy(),
                                            _cur_sentence,
                                            _cur_idx,
                                            False,
                                        )
                                    ),
                                    _loop,
                                )
                                fut.result(timeout=30)
                                total_samples += len(audio_np)
                                sub_chunk_count += 1
                                accumulated = []
                                accumulated_samples = 0

                        if accumulated:
                            audio_np = np.concatenate(accumulated)
                            audio_np = trim_trailing_tts_audio(audio_np, sr=24000)
                            fut = asyncio.run_coroutine_threadsafe(
                                _queue.put(
                                    (audio_np.copy(), _cur_sentence, _cur_idx, True)
                                ),
                                _loop,
                            )
                            fut.result(timeout=30)
                            total_samples += len(audio_np)
                            sub_chunk_count += 1

                        return {
                            "total_samples": total_samples,
                            "sub_chunks": sub_chunk_count,
                        }

                    result = await self._loop.run_in_executor(
                        self.tts_executor, do_tts_progressive
                    )

                    tts_time = time.time() - t_tts
                    tts_total_time += tts_time

                    total_samples = result.get("total_samples", 0) if result else 0
                    sub_chunks = result.get("sub_chunks", 0) if result else 0
                    if total_samples > 0:
                        duration = total_samples / 24000
                        print(
                            f"⏱️  TTS sentence {chunk_idx}: {tts_time:.3f}s → {duration:.1f}s audio ({sub_chunks} sub-chunks): '{sentence[:40]}...'"
                        )
                        pipeline_timings["tts_per_sentence"].append(
                            {
                                "sentence": sentence,
                                "tts_time_s": round(tts_time, 3),
                                "audio_duration_s": round(duration, 3),
                                "sub_chunks": sub_chunks,
                            }
                        )
                        pipeline_timings["total_audio_duration_s"] += duration
                    else:
                        print(f"⏱️  TTS sentence {chunk_idx}: empty audio, skipping")

                print(f"⏱️  TTS total: {tts_total_time:.3f}s")
                pipeline_timings["tts_total_time_s"] = round(tts_total_time, 3)

            # ── Stage 2: A/V Sync Handler ──
            # Prepares audio for browser output and forwards BOTH prepared audio frames
            # + raw audio to the renderer, which pushes them in lockstep for lip sync.
            _audio_crossfader = AudioChunkCrossfader(sr=24000, crossfade_ms=15.0)
            AUDIO_FRAMES_PER_VIDEO = (
                SAMPLES_PER_FRAME // 480
            )  # 960/480 = 2 WebRTC audio frames per video frame
            _speaking_start_time = [None]  # Shared: set by av_handler, read by renderer

            async def parallel_av_handler():
                audio_started = False

                while True:
                    item = await audio_data_queue.get()
                    if item is None:
                        await render_queue.put(None)
                        break

                    audio_24k_np, sentence, chunk_idx, is_final = item

                    if not audio_started:
                        # First audio chunk — set clock anchor for both tracks
                        clock = time.time()
                        self.video_track.start_speaking(clock_anchor=clock)
                        self.audio_track.reset_clock(clock_anchor=clock)
                        audio_started = True
                        _speaking_start_time[0] = clock
                        latency = clock - pipeline_start
                        pipeline_timings["first_byte_latency_s"] = round(latency, 3)
                        print(
                            f"⏱️  FIRST BYTE: {latency:.3f}s from pipeline start (synced A/V mode)"
                        )

                    # Prepare audio for browser (crossfade, clip, split into WebRTC frames)
                    peak = np.abs(audio_24k_np).max()
                    audio_for_speaker = (
                        audio_24k_np / peak if peak > 1.0 else audio_24k_np.copy()
                    )
                    audio_for_speaker = _audio_crossfader.process(audio_for_speaker)
                    audio_int16 = (audio_for_speaker * 32767).astype(np.int16)
                    audio_webrtc_frames = _split_audio_to_webrtc(audio_int16)

                    if self.sync_manager:
                        self.sync_manager.notify_audio_content(len(audio_int16))

                    # Forward BOTH prepared audio AND raw audio to renderer for lockstep push
                    await render_queue.put(
                        (
                            audio_24k_np,
                            audio_webrtc_frames,
                            sentence,
                            chunk_idx,
                            is_final,
                        )
                    )

                print(f"⏱️  A/V sync handler: all chunks forwarded to renderer")

            # ── Stage 3: Video Renderer (pushes audio + video in lockstep for lip sync) ──
            _prev_chunk_last_rgb = [None]
            BLEND_FRAMES = 2
            _cumul_audio = [0]  # Cumulative 24kHz samples — eliminates per-chunk rounding drift
            _cumul_video = [0]  # Cumulative video frames pushed

            async def video_renderer():
                render_total_time = 0
                total_frames = 0

                while True:
                    item = await render_queue.get()
                    if item is None:
                        break

                    audio_24k_np, audio_webrtc_frames, sentence, chunk_idx, is_final = (
                        item
                    )
                    total_audio_samples = len(audio_24k_np)

                    print(
                        f"CM: Rendering chunk {chunk_idx}: '{sentence[:30]}...' ({total_audio_samples/24000:.1f}s)"
                    )
                    t_render = time.time()

                    def render_and_push_av():
                        """Render video frames and push audio + video in lockstep for lip sync."""
                        gpu_monitor.check_and_log(f"render-sync-{chunk_idx}")

                        # Resample 24kHz → 16kHz for wav2vec2
                        if _HAS_TORCHAUDIO:
                            at = torch.from_numpy(audio_24k_np).float().unsqueeze(0)
                            audio_16k = AF.resample(at, 24000, 16000).squeeze(0).numpy()
                            del at
                        else:
                            num_samples_16k = int(len(audio_24k_np) * 16000 / 24000)
                            from scipy.signal import resample as scipy_resample

                            audio_16k = scipy_resample(
                                audio_24k_np, num_samples_16k
                            ).astype(np.float32)

                        # Cumulative frame count eliminates per-chunk rounding drift
                        _cumul_audio[0] += total_audio_samples
                        expected_frames = _cumul_audio[0] // SAMPLES_PER_FRAME - _cumul_video[0]

                        frame_count = 0
                        audio_frame_idx = 0
                        last_rgb = None

                        for (
                            video_frames,
                            metrics,
                        ) in self.imtalker_streamer.generate_stream(
                            self.avatar_img, iter([audio_16k])
                        ):
                            for ft in video_frames:
                                if frame_count >= expected_frames:
                                    del ft
                                    continue

                                frame_rgb = process_frame_tensor(ft, format="RGB")

                                if (
                                    _prev_chunk_last_rgb[0] is not None
                                    and frame_count < BLEND_FRAMES
                                ):
                                    alpha = (frame_count + 1) / (BLEND_FRAMES + 1)
                                    prev = _prev_chunk_last_rgb[0].astype(np.float32)
                                    curr = frame_rgb.astype(np.float32)
                                    frame_rgb = (
                                        (1 - alpha) * prev + alpha * curr
                                    ).astype(np.uint8)

                                last_rgb = frame_rgb.copy()
                                output_rgb = self.paste_back_frame(frame_rgb)
                                av_frame = VideoFrame.from_ndarray(
                                    output_rgb, format="rgb24"
                                )
                                del ft, frame_rgb, output_rgb

                                # Push video frame to video track
                                fut = asyncio.run_coroutine_threadsafe(
                                    self.video_track.frame_queue.put(av_frame),
                                    self._loop,
                                )
                                fut.result(timeout=10)

                                # Push corresponding audio frames in lockstep
                                for _ in range(AUDIO_FRAMES_PER_VIDEO):
                                    if audio_frame_idx < len(audio_webrtc_frames):
                                        afut = asyncio.run_coroutine_threadsafe(
                                            self.audio_track.audio_queue.put(
                                                audio_webrtc_frames[audio_frame_idx]
                                            ),
                                            self._loop,
                                        )
                                        afut.result(timeout=10)
                                        audio_frame_idx += 1

                                frame_count += 1
                            del video_frames

                        # Pad remaining frames if needed
                        if frame_count > 0 and frame_count < expected_frames:
                            last_frame = VideoFrame.from_ndarray(
                                (
                                    self.paste_back_frame(last_rgb)
                                    if last_rgb is not None
                                    else np.zeros((512, 512, 3), dtype=np.uint8)
                                ),
                                format="rgb24",
                            )
                            while frame_count < expected_frames:
                                fut = asyncio.run_coroutine_threadsafe(
                                    self.video_track.frame_queue.put(last_frame),
                                    self._loop,
                                )
                                fut.result(timeout=10)
                                for _ in range(AUDIO_FRAMES_PER_VIDEO):
                                    if audio_frame_idx < len(audio_webrtc_frames):
                                        afut = asyncio.run_coroutine_threadsafe(
                                            self.audio_track.audio_queue.put(
                                                audio_webrtc_frames[audio_frame_idx]
                                            ),
                                            self._loop,
                                        )
                                        afut.result(timeout=10)
                                        audio_frame_idx += 1
                                frame_count += 1

                        # Push any remaining audio frames
                        while audio_frame_idx < len(audio_webrtc_frames):
                            afut = asyncio.run_coroutine_threadsafe(
                                self.audio_track.audio_queue.put(
                                    audio_webrtc_frames[audio_frame_idx]
                                ),
                                self._loop,
                            )
                            afut.result(timeout=10)
                            audio_frame_idx += 1

                        del audio_16k
                        _prev_chunk_last_rgb[0] = last_rgb

                        if self.sync_manager:
                            self.sync_manager.notify_video_content(frame_count)

                        _cumul_video[0] += frame_count
                        return frame_count

                    frame_count = await self._loop.run_in_executor(
                        None, render_and_push_av
                    )
                    render_time = time.time() - t_render
                    render_total_time += render_time
                    total_frames += frame_count
                    del audio_24k_np

                    self.last_chunk_final_frame = None  # Updated via frame_queue

                    pipeline_timings["render_per_sentence"].append(
                        {
                            "sentence": sentence,
                            "render_time_s": round(render_time, 3),
                            "frame_count": frame_count,
                        }
                    )
                    print(
                        f"⏱️  Render chunk {chunk_idx}: {render_time:.3f}s → {frame_count} video frames"
                    )

                # All A/V frames delivered — signal completion
                self.audio_track.stop_expecting()
                self.video_track.mark_speech_ending()

                # Wait for video playback to finish (frames drain from queue)
                if total_frames > 0 and _speaking_start_time[0] is not None:
                    total_duration = total_frames * 0.04  # 40ms per frame at 25 FPS
                    elapsed_speaking = time.time() - _speaking_start_time[0]
                    remaining = total_duration - elapsed_speaking + 0.08
                    if remaining > 0:
                        await asyncio.sleep(remaining)

                self.video_track.stop_speaking()

                print(
                    f"⏱️  Render total: {render_total_time:.3f}s → {total_frames} frames"
                )
                pipeline_timings["render_total_time_s"] = round(render_total_time, 3)
                pipeline_timings["total_frames"] = total_frames

                total_pipeline_time = time.time() - pipeline_start
                pipeline_timings["streaming_time_s"] = round(total_pipeline_time, 3)
                pipeline_timings["total_pipeline_time_s"] = round(
                    total_pipeline_time, 3
                )
                print(f"⏱️  TOTAL PIPELINE (synced A/V): {total_pipeline_time:.3f}s")

            # Run all stages concurrently — TTS runs ahead, but A/V pushed in lockstep
            await asyncio.gather(tts_worker(), parallel_av_handler(), video_renderer())
            pipeline_timings["status"] = "completed"

        except asyncio.CancelledError:
            print("CM: Streaming pipeline cancelled (barge-in)")
            pipeline_timings["status"] = "cancelled"
            pipeline_timings["total_pipeline_time_s"] = round(
                time.time() - pipeline_start, 3
            )
        except Exception as e:
            print(f"CM Streaming Pipeline Error: {e}")
            traceback.print_exc()
            pipeline_timings["status"] = "error"
            pipeline_timings["error"] = str(e)
            pipeline_timings["total_pipeline_time_s"] = round(
                time.time() - pipeline_start, 3
            )
        finally:
            self.is_speaking = False
            pipeline_timings["total_audio_duration_s"] = round(
                pipeline_timings.get("total_audio_duration_s", 0), 3
            )
            logger.info(
                f"Streaming pipeline timings: {json.dumps(pipeline_timings, default=str)[:800]}"
            )
            _update_last_oshara_log_timings(
                app, pipeline_timings, session_id=self.webrtc_session_id
            )
            # Drain any leftover items in the sentence queue (e.g., after cancellation)
            while not self._streaming_sentence_queue.empty():
                try:
                    self._streaming_sentence_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def process_response_text(self, response_text):
        """TTS → synced A/V → WebRTC pipeline for a text response.

        Audio and video are pushed in lockstep: each rendered video frame
        is paired with its corresponding audio samples for lip sync.
        TTS still runs ahead via pipeline parallelism.

        Architecture (all stages run concurrently):
          Stage 1 (sentence_feeder): Split text into sentences
          Stage 2 (tts_worker): sentence → audio numpy → audio_data_queue
          Stage 3 (parallel_av_handler): audio_data_queue →
              └─ prepares WebRTC audio frames + forwards to render_queue
          Stage 4 (video_renderer): render_queue → IMTalker →
              ├─ video_track (per frame)
              └─ audio_track (2 audio frames per video frame)
        """
        print(f"CM: Processing response text (parallel A/V): '{response_text[:80]}...'")
        pipeline_start = time.time()

        # Shared timing dict — populated by each pipeline stage
        pipeline_timings = {
            "pipeline_start_utc": datetime.now(timezone.utc).isoformat(),
            "llm_response_time_s": getattr(self, "_current_llm_latency", None),
            "response_text": response_text,
            "tts_total_time_s": None,
            "tts_per_sentence": [],  # [{sentence, tts_time_s, audio_duration_s}]
            "render_total_time_s": None,
            "render_per_sentence": [],  # [{sentence, render_time_s, frame_count}]
            "total_audio_duration_s": 0.0,
            "first_byte_latency_s": None,
            "streaming_time_s": None,
            "total_pipeline_time_s": None,
            "total_frames": 0,
            "status": "started",
        }

        try:
            self.is_speaking = True

            # Queues for parallel A/V pipeline
            sentence_queue = asyncio.Queue(maxsize=6)  # Sentences → TTS
            audio_data_queue = asyncio.Queue(maxsize=8)  # TTS → parallel_av_handler
            render_queue = (
                asyncio.Queue()
            )  # parallel_av_handler → video_renderer (unbounded: never block audio delivery)

            SAMPLES_PER_FRAME = 960  # 24000 Hz / 25 FPS

            # ── Stage 1: Sentence Splitter ──
            async def sentence_feeder():
                sentence_end_pattern = re.compile(r"[.!?](?:\s|$)")
                buffer = response_text
                sentence_count = 0

                while buffer.strip():
                    match = sentence_end_pattern.search(buffer)
                    if match:
                        end_pos = match.end()
                        sentence = buffer[:end_pos].strip()
                        if sentence:
                            sentence_count += 1
                            await sentence_queue.put(sentence)
                        buffer = buffer[end_pos:].lstrip()
                    else:
                        remaining = buffer.strip()
                        if remaining:
                            sentence_count += 1
                            await sentence_queue.put(remaining)
                        break

                await sentence_queue.put(None)
                print(
                    f"⏱️  Sentence split: {sentence_count} sentences from response text"
                )

            # ── Stage 2: TTS Worker ──
            async def tts_worker():
                tts_total_time = 0
                chunk_idx = 0

                while True:
                    sentence = await sentence_queue.get()
                    if sentence is None:
                        await audio_data_queue.put(None)
                        break

                    chunk_idx += 1
                    t_tts = time.time()

                    def do_tts(text):
                        if (
                            SHERPA_AVAILABLE
                            and detect_nepali(text)
                            and hasattr(app.state, "sherpa_tts")
                            and app.state.sherpa_tts is not None
                        ):
                            print(
                                f"🇳🇵 Using Sherpa TTS for Nepali text: '{text[:50]}...'"
                            )
                            try:
                                audio_np = app.state.sherpa_tts.generate(
                                    text, sid=5, speed=1.0
                                )
                                sherpa_sr = app.state.sherpa_tts.sample_rate
                                if sherpa_sr != 24000:
                                    import librosa

                                    audio_np = librosa.resample(
                                        audio_np, orig_sr=sherpa_sr, target_sr=24000
                                    )
                                return (
                                    audio_np.flatten().copy()
                                    if audio_np is not None
                                    else None
                                )
                            except Exception as e:
                                print(
                                    f"⚠️  Sherpa TTS failed for Nepali, falling back to STTS: {e}"
                                )

                        tts_stream = self.tts_model.generate_stream(
                            text=text,
                            chunk_size=15,
                            print_metrics=False,
                            audio_prompt_path=self.reference_audio,
                        )
                        parts = []
                        for audio_chunk, metrics in tts_stream:
                            parts.append(
                                audio_chunk.cpu()
                                if isinstance(audio_chunk, torch.Tensor)
                                else audio_chunk
                            )
                        if parts:
                            audio_full = torch.cat(parts, dim=-1).numpy().flatten().copy()
                            return trim_trailing_tts_audio(audio_full, sr=24000)
                        return None

                    audio_24k_np = await self._loop.run_in_executor(
                        self.tts_executor, do_tts, sentence
                    )

                    tts_time = time.time() - t_tts
                    tts_total_time += tts_time

                    if audio_24k_np is not None and len(audio_24k_np) > 0:
                        duration = len(audio_24k_np) / 24000
                        print(
                            f"⏱️  TTS sentence {chunk_idx}: {tts_time:.3f}s → {duration:.1f}s audio: '{sentence[:40]}...'"
                        )
                        pipeline_timings["tts_per_sentence"].append(
                            {
                                "sentence": sentence,
                                "tts_time_s": round(tts_time, 3),
                                "audio_duration_s": round(duration, 3),
                            }
                        )
                        pipeline_timings["total_audio_duration_s"] += duration
                        await audio_data_queue.put((audio_24k_np, sentence, chunk_idx))
                    else:
                        print(f"⏱️  TTS sentence {chunk_idx}: empty audio, skipping")

                print(f"⏱️  TTS total: {tts_total_time:.3f}s")
                pipeline_timings["tts_total_time_s"] = round(tts_total_time, 3)

            # ── Stage 3: A/V Sync Handler ──
            # Prepares audio for browser output and forwards BOTH prepared audio frames
            # + raw audio to the renderer, which pushes them in lockstep for lip sync.
            _audio_crossfader = AudioChunkCrossfader(sr=24000, crossfade_ms=15.0)
            AUDIO_FRAMES_PER_VIDEO = SAMPLES_PER_FRAME // 480  # 960/480 = 2
            _speaking_start_time = [None]  # Shared: set by av_handler, read by renderer

            async def parallel_av_handler():
                audio_started = False

                while True:
                    item = await audio_data_queue.get()
                    if item is None:
                        await render_queue.put(None)
                        break

                    audio_24k_np, sentence, chunk_idx = item

                    if not audio_started:
                        clock = time.time()
                        self.video_track.start_speaking(clock_anchor=clock)
                        self.audio_track.reset_clock(clock_anchor=clock)
                        audio_started = True
                        _speaking_start_time[0] = clock
                        latency = clock - pipeline_start
                        pipeline_timings["first_byte_latency_s"] = round(latency, 3)
                        print(
                            f"⏱️  FIRST BYTE: {latency:.3f}s from pipeline start (synced A/V mode)"
                        )

                    # Prepare audio for browser (crossfade, clip, split into WebRTC frames)
                    peak = np.abs(audio_24k_np).max()
                    audio_for_speaker = (
                        audio_24k_np / peak if peak > 1.0 else audio_24k_np.copy()
                    )
                    audio_for_speaker = _audio_crossfader.process(audio_for_speaker)
                    audio_int16 = (audio_for_speaker * 32767).astype(np.int16)
                    audio_webrtc_frames = _split_audio_to_webrtc(audio_int16)

                    if self.sync_manager:
                        self.sync_manager.notify_audio_content(len(audio_int16))

                    # Forward BOTH prepared audio AND raw audio to renderer for lockstep push
                    await render_queue.put(
                        (audio_24k_np, audio_webrtc_frames, sentence, chunk_idx)
                    )

                print(f"⏱️  A/V sync handler: all chunks forwarded to renderer")

            # ── Stage 4: Video Renderer (pushes audio + video in lockstep for lip sync) ──
            _prev_chunk_last_rgb = [None]
            BLEND_FRAMES = 2
            _cumul_audio = [0]  # Cumulative 24kHz samples — eliminates per-chunk rounding drift
            _cumul_video = [0]  # Cumulative video frames pushed

            async def video_renderer():
                render_total_time = 0
                total_frames = 0

                while True:
                    item = await render_queue.get()
                    if item is None:
                        break

                    audio_24k_np, audio_webrtc_frames, sentence, chunk_idx = item
                    total_audio_samples = len(audio_24k_np)

                    print(f"CM: Rendering sentence {chunk_idx}: '{sentence[:40]}...'")
                    t_render = time.time()

                    def render_and_push_av():
                        gpu_monitor.check_and_log(f"render-sync-{chunk_idx}")

                        if _HAS_TORCHAUDIO:
                            at = torch.from_numpy(audio_24k_np).float().unsqueeze(0)
                            audio_16k = AF.resample(at, 24000, 16000).squeeze(0).numpy()
                            del at
                        else:
                            num_samples_16k = int(len(audio_24k_np) * 16000 / 24000)
                            from scipy.signal import resample as scipy_resample

                            audio_16k = scipy_resample(
                                audio_24k_np, num_samples_16k
                            ).astype(np.float32)

                        # Cumulative frame count eliminates per-chunk rounding drift
                        _cumul_audio[0] += total_audio_samples
                        expected_frames = _cumul_audio[0] // SAMPLES_PER_FRAME - _cumul_video[0]

                        frame_count = 0
                        audio_frame_idx = 0
                        last_rgb = None

                        for (
                            video_frames,
                            metrics,
                        ) in self.imtalker_streamer.generate_stream(
                            self.avatar_img, iter([audio_16k])
                        ):
                            for ft in video_frames:
                                if frame_count >= expected_frames:
                                    del ft
                                    continue

                                frame_rgb = process_frame_tensor(ft, format="RGB")

                                if (
                                    _prev_chunk_last_rgb[0] is not None
                                    and frame_count < BLEND_FRAMES
                                ):
                                    alpha = (frame_count + 1) / (BLEND_FRAMES + 1)
                                    prev = _prev_chunk_last_rgb[0].astype(np.float32)
                                    curr = frame_rgb.astype(np.float32)
                                    frame_rgb = (
                                        (1 - alpha) * prev + alpha * curr
                                    ).astype(np.uint8)

                                last_rgb = frame_rgb.copy()
                                output_rgb = self.paste_back_frame(frame_rgb)
                                av_frame = VideoFrame.from_ndarray(
                                    output_rgb, format="rgb24"
                                )
                                del ft, frame_rgb, output_rgb

                                # Push video frame to video track
                                fut = asyncio.run_coroutine_threadsafe(
                                    self.video_track.frame_queue.put(av_frame),
                                    self._loop,
                                )
                                fut.result(timeout=10)

                                # Push corresponding audio frames in lockstep
                                for _ in range(AUDIO_FRAMES_PER_VIDEO):
                                    if audio_frame_idx < len(audio_webrtc_frames):
                                        afut = asyncio.run_coroutine_threadsafe(
                                            self.audio_track.audio_queue.put(
                                                audio_webrtc_frames[audio_frame_idx]
                                            ),
                                            self._loop,
                                        )
                                        afut.result(timeout=10)
                                        audio_frame_idx += 1

                                frame_count += 1
                            del video_frames

                        if frame_count > 0 and frame_count < expected_frames:
                            last_frame = VideoFrame.from_ndarray(
                                (
                                    self.paste_back_frame(last_rgb)
                                    if last_rgb is not None
                                    else np.zeros((512, 512, 3), dtype=np.uint8)
                                ),
                                format="rgb24",
                            )
                            while frame_count < expected_frames:
                                fut = asyncio.run_coroutine_threadsafe(
                                    self.video_track.frame_queue.put(last_frame),
                                    self._loop,
                                )
                                fut.result(timeout=10)
                                for _ in range(AUDIO_FRAMES_PER_VIDEO):
                                    if audio_frame_idx < len(audio_webrtc_frames):
                                        afut = asyncio.run_coroutine_threadsafe(
                                            self.audio_track.audio_queue.put(
                                                audio_webrtc_frames[audio_frame_idx]
                                            ),
                                            self._loop,
                                        )
                                        afut.result(timeout=10)
                                        audio_frame_idx += 1
                                frame_count += 1

                        # Push any remaining audio frames
                        while audio_frame_idx < len(audio_webrtc_frames):
                            afut = asyncio.run_coroutine_threadsafe(
                                self.audio_track.audio_queue.put(
                                    audio_webrtc_frames[audio_frame_idx]
                                ),
                                self._loop,
                            )
                            afut.result(timeout=10)
                            audio_frame_idx += 1

                        del audio_16k
                        _prev_chunk_last_rgb[0] = last_rgb

                        if self.sync_manager:
                            self.sync_manager.notify_video_content(frame_count)

                        _cumul_video[0] += frame_count
                        return frame_count

                    frame_count = await self._loop.run_in_executor(
                        None, render_and_push_av
                    )
                    render_time = time.time() - t_render
                    render_total_time += render_time
                    total_frames += frame_count
                    del audio_24k_np

                    pipeline_timings["render_per_sentence"].append(
                        {
                            "sentence": sentence,
                            "render_time_s": round(render_time, 3),
                            "frame_count": frame_count,
                        }
                    )
                    print(
                        f"⏱️  Render sentence {chunk_idx}: {render_time:.3f}s → {frame_count} video frames"
                    )

                # All A/V frames delivered — signal completion
                self.audio_track.stop_expecting()
                self.video_track.mark_speech_ending()

                if total_frames > 0 and _speaking_start_time[0] is not None:
                    total_duration = total_frames * 0.04
                    elapsed_speaking = time.time() - _speaking_start_time[0]
                    remaining = total_duration - elapsed_speaking + 0.08
                    if remaining > 0:
                        await asyncio.sleep(remaining)

                self.video_track.stop_speaking()

                print(
                    f"⏱️  Render total: {render_total_time:.3f}s → {total_frames} frames"
                )
                pipeline_timings["render_total_time_s"] = round(render_total_time, 3)
                pipeline_timings["total_frames"] = total_frames

                total_pipeline_time = time.time() - pipeline_start
                pipeline_timings["streaming_time_s"] = round(total_pipeline_time, 3)
                pipeline_timings["total_pipeline_time_s"] = round(
                    total_pipeline_time, 3
                )
                print(f"⏱️  TOTAL PIPELINE (synced A/V): {total_pipeline_time:.3f}s")

            # Run ALL stages concurrently — TTS runs ahead, but A/V pushed in lockstep
            await asyncio.gather(
                sentence_feeder(), tts_worker(), parallel_av_handler(), video_renderer()
            )

            pipeline_timings["status"] = "completed"

        except asyncio.CancelledError:
            print("CM: Pipeline cancelled (barge-in)")
            pipeline_timings["status"] = "cancelled"
            pipeline_timings["total_pipeline_time_s"] = round(
                time.time() - pipeline_start, 3
            )
        except Exception as e:
            print(f"CM Pipeline Error: {e}")
            traceback.print_exc()
            pipeline_timings["status"] = "error"
            pipeline_timings["error"] = str(e)
            pipeline_timings["total_pipeline_time_s"] = round(
                time.time() - pipeline_start, 3
            )
        finally:
            self.is_speaking = False
            # Round the audio duration
            pipeline_timings["total_audio_duration_s"] = round(
                pipeline_timings.get("total_audio_duration_s", 0), 3
            )
            # ── Update last Oshara log entry with ALL step timings ──
            logger.info(
                f"Pipeline timings for Oshara log: {json.dumps(pipeline_timings, default=str)[:800]}"
            )
            _update_last_oshara_log_timings(
                app, pipeline_timings, session_id=self.webrtc_session_id
            )

    async def generate_and_feed(self, text):
        """Real-time streaming pipeline with SYNCED audio and video delivery.

        Architecture:
          Stage 1 (audio_chunker): TTS text → audio tensors → audio_data_queue
          Stage 2 (parallel_av_handler): audio_data_queue →
              └→ prepares WebRTC audio frames + forwards to render_queue
          Stage 3 (video_renderer): render_queue → IMTalker →
              ├→ video_track (per frame)
              └→ audio_track (2 audio frames per video frame, lockstep)

        Audio + video are pushed in lockstep for tight lip sync.
        TTS runs ahead via pipeline parallelism.
        """
        print(f"CM: Generating response: {text[:50]}{'...' if len(text) > 50 else ''}")
        self.is_speaking = True
        pipeline_start = time.time()

        t_parse = time.time()
        text_chunks = parse_chunk_labels(text)
        print(
            f"⏱️  Chunk parsing: {time.time() - t_parse:.3f}s → {len(text_chunks)} chunks"
        )

        # Shared timing dict — populated by each pipeline stage
        pipeline_timings = {
            "pipeline_start_utc": datetime.now(timezone.utc).isoformat(),
            "llm_response_time_s": getattr(self, "_current_llm_latency", None),
            "response_text": text,
            "tts_total_time_s": None,
            "tts_per_chunk": [],  # [{chunk_text, tts_time_s, audio_duration_s}]
            "render_total_time_s": None,
            "render_per_chunk": [],  # [{chunk_text, render_time_s, frame_count}]
            "total_audio_duration_s": 0.0,
            "first_byte_latency_s": None,
            "streaming_time_s": None,
            "total_pipeline_time_s": None,
            "total_frames": 0,
            "status": "started",
        }

        # TTS output → parallel handler
        audio_data_queue = asyncio.Queue(maxsize=8)
        # parallel handler → video renderer (unbounded: never block audio delivery)
        render_queue = asyncio.Queue()

        SAMPLES_PER_FRAME = 960  # 24000 Hz / 25 FPS = 960 audio samples per video frame

        try:
            # Stage 1: TTS — generate audio for each text chunk (in executor to not block event loop)
            async def audio_chunker():
                tts_total_time = 0
                for chunk_idx, text_chunk in enumerate(text_chunks, 1):
                    t_tts = time.time()

                    _text = text_chunk
                    _ref_audio = self.reference_audio
                    _tts_model = self.tts_model

                    def do_tts():
                        if (
                            SHERPA_AVAILABLE
                            and detect_nepali(_text)
                            and hasattr(app.state, "sherpa_tts")
                            and app.state.sherpa_tts is not None
                        ):
                            try:
                                audio_np = app.state.sherpa_tts.generate(
                                    _text, sid=5, speed=1.0
                                )
                                sherpa_sr = app.state.sherpa_tts.sample_rate
                                if sherpa_sr != 24000:
                                    import librosa

                                    audio_np = librosa.resample(
                                        audio_np, orig_sr=sherpa_sr, target_sr=24000
                                    )
                                return (
                                    audio_np.flatten().copy()
                                    if audio_np is not None
                                    else None
                                )
                            except Exception as e:
                                print(f"⚠️  Sherpa TTS failed, falling back: {e}")

                        tts_stream = _tts_model.generate_stream(
                            text=_text,
                            chunk_size=15,
                            print_metrics=False,
                            audio_prompt_path=_ref_audio,
                        )
                        audio_parts = []
                        for audio_chunk, metrics in tts_stream:
                            audio_parts.append(
                                audio_chunk.cpu()
                                if isinstance(audio_chunk, torch.Tensor)
                                else audio_chunk
                            )
                        if audio_parts:
                            full_audio = torch.cat(audio_parts, dim=-1)
                            audio_np = full_audio.numpy().flatten().copy()
                            del full_audio, audio_parts
                            return trim_trailing_tts_audio(audio_np, sr=24000)
                        del audio_parts
                        return None

                    audio_np = await self._loop.run_in_executor(
                        self.tts_executor, do_tts
                    )

                    tts_chunk_time = time.time() - t_tts
                    tts_total_time += tts_chunk_time

                    if audio_np is not None and len(audio_np) > 0:
                        duration = len(audio_np) / 24000
                        await audio_data_queue.put((audio_np, text_chunk, chunk_idx))
                        print(
                            f"⏱️  TTS chunk {chunk_idx}/{len(text_chunks)}: {tts_chunk_time:.3f}s → {duration:.1f}s audio"
                        )
                        pipeline_timings["tts_per_chunk"].append(
                            {
                                "chunk_text": text_chunk,
                                "tts_time_s": round(tts_chunk_time, 3),
                                "audio_duration_s": round(duration, 3),
                            }
                        )
                        pipeline_timings["total_audio_duration_s"] += duration

                await audio_data_queue.put(None)
                print(
                    f"⏱️  TTS total: {tts_total_time:.3f}s for {len(text_chunks)} chunks"
                )
                pipeline_timings["tts_total_time_s"] = round(tts_total_time, 3)

            # Stage 2: A/V Sync Handler — prepares audio, forwards to renderer for lockstep push
            _audio_crossfader_feed = AudioChunkCrossfader(sr=24000, crossfade_ms=15.0)
            AUDIO_FRAMES_PER_VIDEO_FEED = SAMPLES_PER_FRAME // 480  # 960/480 = 2
            _speaking_start_time = [None]  # Shared: set by av_handler, read by renderer

            async def parallel_av_handler():
                audio_started = False

                while True:
                    item = await audio_data_queue.get()
                    if item is None:
                        await render_queue.put(None)
                        break

                    audio_24k_np, text_chunk, chunk_idx = item

                    if not audio_started:
                        clock = time.time()
                        self.video_track.start_speaking(clock_anchor=clock)
                        self.audio_track.reset_clock(clock_anchor=clock)
                        audio_started = True
                        _speaking_start_time[0] = clock
                        latency = clock - pipeline_start
                        pipeline_timings["first_byte_latency_s"] = round(latency, 3)
                        print(
                            f"⏱️  FIRST BYTE: {latency:.3f}s from pipeline start (synced A/V mode)"
                        )

                    # Prepare audio for browser (crossfade, clip, split into WebRTC frames)
                    peak = np.abs(audio_24k_np).max()
                    audio_for_speaker = (
                        audio_24k_np / peak if peak > 1.0 else audio_24k_np.copy()
                    )
                    audio_for_speaker = _audio_crossfader_feed.process(
                        audio_for_speaker
                    )
                    audio_int16 = (audio_for_speaker * 32767).astype(np.int16)
                    audio_webrtc_frames = _split_audio_to_webrtc(audio_int16)

                    if self.sync_manager:
                        self.sync_manager.notify_audio_content(len(audio_int16))

                    # Forward BOTH prepared audio AND raw audio to renderer for lockstep push
                    await render_queue.put(
                        (audio_24k_np, audio_webrtc_frames, text_chunk, chunk_idx)
                    )

                print(f"⏱️  A/V sync handler: all chunks forwarded to renderer")

            # Stage 3: Video Renderer (pushes audio + video in lockstep for lip sync)
            _prev_chunk_last_rgb_feed = [None]
            BLEND_FRAMES_FEED = 2
            _cumul_audio = [0]  # Cumulative 24kHz samples — eliminates per-chunk rounding drift
            _cumul_video = [0]  # Cumulative video frames pushed

            async def video_renderer():
                render_total_time = 0
                total_frames = 0

                while True:
                    item = await render_queue.get()
                    if item is None:
                        break

                    audio_24k_np, audio_webrtc_frames, text_chunk, chunk_idx = item
                    total_audio_samples = len(audio_24k_np)
                    chunk_duration = total_audio_samples / 24000

                    print(
                        f"CM: Rendering chunk {chunk_idx} ({chunk_duration:.1f}s): '{text_chunk[:40]}...'"
                    )
                    t_render = time.time()

                    def render_and_push_av():
                        gpu_monitor.check_and_log(f"render-sync-feed-{chunk_idx}")

                        if _HAS_TORCHAUDIO:
                            at = torch.from_numpy(audio_24k_np).float().unsqueeze(0)
                            audio_16k = AF.resample(at, 24000, 16000).squeeze(0).numpy()
                            del at
                        else:
                            num_samples_16k = int(len(audio_24k_np) * 16000 / 24000)
                            from scipy.signal import resample as scipy_resample

                            audio_16k = scipy_resample(
                                audio_24k_np, num_samples_16k
                            ).astype(np.float32)

                        # Cumulative frame count eliminates per-chunk rounding drift
                        _cumul_audio[0] += total_audio_samples
                        expected_frames = _cumul_audio[0] // SAMPLES_PER_FRAME - _cumul_video[0]

                        frame_count = 0
                        audio_frame_idx = 0
                        last_rgb = None

                        for (
                            video_frames,
                            metrics,
                        ) in self.imtalker_streamer.generate_stream(
                            self.avatar_img, iter([audio_16k])
                        ):
                            for ft in video_frames:
                                if frame_count >= expected_frames:
                                    del ft
                                    continue

                                frame_rgb = process_frame_tensor(ft, format="RGB")

                                if (
                                    _prev_chunk_last_rgb_feed[0] is not None
                                    and frame_count < BLEND_FRAMES_FEED
                                ):
                                    alpha = (frame_count + 1) / (BLEND_FRAMES_FEED + 1)
                                    prev = _prev_chunk_last_rgb_feed[0].astype(
                                        np.float32
                                    )
                                    curr = frame_rgb.astype(np.float32)
                                    frame_rgb = (
                                        (1 - alpha) * prev + alpha * curr
                                    ).astype(np.uint8)

                                last_rgb = frame_rgb.copy()
                                output_rgb = self.paste_back_frame(frame_rgb)
                                av_frame = VideoFrame.from_ndarray(
                                    output_rgb, format="rgb24"
                                )
                                del ft, frame_rgb, output_rgb

                                # Push video frame to video track
                                fut = asyncio.run_coroutine_threadsafe(
                                    self.video_track.frame_queue.put(av_frame),
                                    self._loop,
                                )
                                fut.result(timeout=10)

                                # Push corresponding audio frames in lockstep
                                for _ in range(AUDIO_FRAMES_PER_VIDEO_FEED):
                                    if audio_frame_idx < len(audio_webrtc_frames):
                                        afut = asyncio.run_coroutine_threadsafe(
                                            self.audio_track.audio_queue.put(
                                                audio_webrtc_frames[audio_frame_idx]
                                            ),
                                            self._loop,
                                        )
                                        afut.result(timeout=10)
                                        audio_frame_idx += 1

                                frame_count += 1
                            del video_frames

                        if frame_count > 0 and frame_count < expected_frames:
                            last_frame = VideoFrame.from_ndarray(
                                (
                                    self.paste_back_frame(last_rgb)
                                    if last_rgb is not None
                                    else np.zeros((512, 512, 3), dtype=np.uint8)
                                ),
                                format="rgb24",
                            )
                            while frame_count < expected_frames:
                                fut = asyncio.run_coroutine_threadsafe(
                                    self.video_track.frame_queue.put(last_frame),
                                    self._loop,
                                )
                                fut.result(timeout=10)
                                for _ in range(AUDIO_FRAMES_PER_VIDEO_FEED):
                                    if audio_frame_idx < len(audio_webrtc_frames):
                                        afut = asyncio.run_coroutine_threadsafe(
                                            self.audio_track.audio_queue.put(
                                                audio_webrtc_frames[audio_frame_idx]
                                            ),
                                            self._loop,
                                        )
                                        afut.result(timeout=10)
                                        audio_frame_idx += 1
                                frame_count += 1

                        # Push any remaining audio frames
                        while audio_frame_idx < len(audio_webrtc_frames):
                            afut = asyncio.run_coroutine_threadsafe(
                                self.audio_track.audio_queue.put(
                                    audio_webrtc_frames[audio_frame_idx]
                                ),
                                self._loop,
                            )
                            afut.result(timeout=10)
                            audio_frame_idx += 1

                        del audio_16k
                        _prev_chunk_last_rgb_feed[0] = last_rgb

                        if self.sync_manager:
                            self.sync_manager.notify_video_content(frame_count)

                        _cumul_video[0] += frame_count
                        return frame_count

                    frame_count = await self._loop.run_in_executor(
                        None, render_and_push_av
                    )
                    render_time = time.time() - t_render
                    render_total_time += render_time
                    total_frames += frame_count
                    del audio_24k_np

                    print(
                        f"⏱️  Render chunk {chunk_idx}: {render_time:.3f}s → {frame_count} video frames"
                    )
                    pipeline_timings["render_per_chunk"].append(
                        {
                            "chunk_text": text_chunk,
                            "render_time_s": round(render_time, 3),
                            "frame_count": frame_count,
                        }
                    )

                # All A/V frames delivered — signal completion
                self.audio_track.stop_expecting()
                self.video_track.mark_speech_ending()

                if total_frames > 0 and _speaking_start_time[0] is not None:
                    total_duration = total_frames * 0.04
                    elapsed_speaking = time.time() - _speaking_start_time[0]
                    remaining = total_duration - elapsed_speaking + 0.08
                    if remaining > 0:
                        await asyncio.sleep(remaining)

                self.video_track.stop_speaking()

                print(
                    f"⏱️  Render total: {render_total_time:.3f}s → {total_frames} frames"
                )
                pipeline_timings["render_total_time_s"] = round(render_total_time, 3)
                pipeline_timings["streaming_time_s"] = round(
                    time.time() - pipeline_start, 3
                )
                pipeline_timings["total_frames"] = total_frames

            # Run all stages concurrently — TTS runs ahead, but A/V pushed in lockstep
            await asyncio.gather(
                audio_chunker(), parallel_av_handler(), video_renderer()
            )

            pipeline_timings["status"] = "completed"
            pipeline_timings["total_pipeline_time_s"] = round(
                time.time() - pipeline_start, 3
            )

        except Exception as e:
            print(f"CM: Pipeline error: {e}")
            traceback.print_exc()
            pipeline_timings["status"] = "error"
            pipeline_timings["error"] = str(e)
            pipeline_timings["total_pipeline_time_s"] = round(
                time.time() - pipeline_start, 3
            )
        finally:
            print("CM: Resetting is_speaking to False")
            self.is_speaking = False
            pipeline_timings["total_audio_duration_s"] = round(
                pipeline_timings.get("total_audio_duration_s", 0), 3
            )
            # ── Update last Oshara log entry with ALL step timings ──
            logger.info(
                f"Pipeline timings for Oshara log: {json.dumps(pipeline_timings, default=str)[:800]}"
            )
            _update_last_oshara_log_timings(
                app, pipeline_timings, session_id=self.webrtc_session_id
            )

    async def pre_render_to_buffer(self, text):
        """Pre-render text to a list of (VideoFrame, audio_int16) pairs WITHOUT pushing to tracks.

        Used by /offer to pre-generate greeting content before returning the SDP answer,
        so the frontend stays on 'connecting' screen while rendering happens.
        Returns list of (VideoFrame, np.ndarray[int16]) pairs ready for push_buffered_greeting().
        """
        print(
            f"CM: Pre-rendering to buffer: {text[:50]}{'...' if len(text) > 50 else ''}"
        )
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
                chunk_size=15,  # Smaller chunks for faster first byte
                print_metrics=False,
                audio_prompt_path=self.reference_audio,
            )
            audio_parts = []
            for audio_chunk, metrics in tts_stream:
                audio_parts.append(
                    audio_chunk.cpu()
                    if isinstance(audio_chunk, torch.Tensor)
                    else audio_chunk
                )

            if audio_parts:
                full_audio = torch.cat(audio_parts, dim=-1)
                del audio_parts
                audio_np = full_audio.numpy().flatten().copy()
                del full_audio
                audio_np = trim_trailing_tts_audio(audio_np, sr=24000)
                all_audio_chunks.append((audio_np, text_chunk, chunk_idx))
                print(
                    f"⏱️  Pre-render TTS chunk {chunk_idx}/{len(text_chunks)}: {time.time() - t_tts:.3f}s → {len(audio_np)/24000:.1f}s audio"
                )
            else:
                del audio_parts

        # Stage 2: Render — generate video frames paired with audio slices
        _audio_crossfader_greet = AudioChunkCrossfader(sr=24000, crossfade_ms=15.0)
        for audio_24k_np, text_chunk, chunk_idx in all_audio_chunks:
            total_audio_samples = len(audio_24k_np)
            t_render = time.time()

            def render_to_list(
                audio_np=audio_24k_np, total_samples=total_audio_samples
            ):
                """Render video and collect pairs into a list (no queue push)."""
                pairs = []

                if _HAS_TORCHAUDIO:
                    at = torch.from_numpy(audio_np).float().unsqueeze(0)
                    audio_16k = AF.resample(at, 24000, 16000).squeeze(0).numpy()
                    del at
                else:
                    num_samples_16k = int(len(audio_np) * 16000 / 24000)
                    from scipy.signal import resample as scipy_resample

                    audio_16k = scipy_resample(audio_np, num_samples_16k).astype(
                        np.float32
                    )

                peak = np.abs(audio_np).max()
                audio_24k_clipped = audio_np / peak if peak > 1.0 else audio_np.copy()
                # Inter-chunk crossfade + DC removal to prevent clicks at boundaries
                audio_24k_clipped = _audio_crossfader_greet.process(audio_24k_clipped)

                # ── Lip Sync: force exact frame-to-audio alignment ──
                expected_frames = (
                    total_samples + SAMPLES_PER_FRAME - 1
                ) // SAMPLES_PER_FRAME

                audio_cursor = 0
                frame_count = 0
                last_av_frame = None

                for video_frames, metrics in self.imtalker_streamer.generate_stream(
                    self.avatar_img, iter([audio_16k])
                ):
                    for ft in video_frames:
                        if frame_count >= expected_frames:
                            del ft
                            continue  # Discard excess frames

                        frame_rgb = process_frame_tensor(ft, format="RGB")
                        # Crop-Infer-Paste: composite face into full image
                        output_rgb = self.paste_back_frame(frame_rgb)
                        av_frame = VideoFrame.from_ndarray(output_rgb, format="rgb24")
                        last_av_frame = av_frame
                        del ft, frame_rgb, output_rgb

                        # Exactly SAMPLES_PER_FRAME per frame (zero-pad last)
                        audio_slice = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
                        avail = min(SAMPLES_PER_FRAME, total_samples - audio_cursor)
                        if avail > 0:
                            audio_slice[:avail] = audio_24k_clipped[
                                audio_cursor : audio_cursor + avail
                            ]
                        audio_cursor += SAMPLES_PER_FRAME
                        audio_slice_int16 = (audio_slice * 32767).astype(np.int16)

                        pairs.append((av_frame, audio_slice_int16))
                        frame_count += 1
                    del video_frames

                # Pad with duplicate last frame if IMTalker produced fewer than expected
                while frame_count < expected_frames and last_av_frame is not None:
                    audio_slice = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
                    avail = min(SAMPLES_PER_FRAME, total_samples - audio_cursor)
                    if avail > 0:
                        audio_slice[:avail] = audio_24k_clipped[
                            audio_cursor : audio_cursor + avail
                        ]
                    audio_cursor += SAMPLES_PER_FRAME
                    audio_slice_int16 = (audio_slice * 32767).astype(np.int16)
                    pairs.append((last_av_frame, audio_slice_int16))
                    frame_count += 1

                if frame_count != expected_frames:
                    print(
                        f"⚠️  SYNC: pre-render expected {expected_frames} frames, got {frame_count}"
                    )

                del audio_16k, audio_24k_clipped
                return pairs

            chunk_pairs = await self._loop.run_in_executor(None, render_to_list)
            buffer_pairs.extend(chunk_pairs)
            print(
                f"⏱️  Pre-render chunk {chunk_idx}: {time.time() - t_render:.3f}s → {len(chunk_pairs)} A/V pairs"
            )
            del audio_24k_np

        elapsed = time.time() - t_start
        print(f"CM: Pre-rendered {len(buffer_pairs)} A/V pairs in {elapsed:.2f}s")
        return buffer_pairs

    async def push_buffered_greeting(self, pairs):
        """Push pre-rendered A/V pairs through tracks with proper 25 FPS pacing.

        Called after WebRTC connection is established to play the pre-rendered greeting.
        Pushes video and audio to separate track queues with shared clock for sync.
        """
        if not pairs:
            return

        print(f"CM: Pushing {len(pairs)} pre-rendered A/V pairs...")
        self.is_speaking = True

        # Signal tracks that speech is starting with shared clock
        clock = time.time()
        self.video_track.start_speaking(clock_anchor=clock)
        self.audio_track.reset_clock(clock_anchor=clock)

        stream_start = clock
        total_frames = 0

        # Push ALL pairs — video frames and audio chunks to SEPARATE queues.
        for av_frame, audio_slice_int16 in pairs:
            audio_chunks = _split_audio_to_webrtc(audio_slice_int16)
            await self.video_track.frame_queue.put(av_frame)
            for chunk in audio_chunks:
                await self.audio_track.audio_queue.put(chunk)
            total_frames += 1
            self.last_chunk_final_frame = av_frame

        # Signal that no more frames are coming
        self.video_track.mark_speech_ending()

        # Wait for all content to be consumed by recv() before stopping speech.
        total_duration = total_frames * 0.04
        elapsed_so_far = time.time() - stream_start
        remaining = total_duration - elapsed_so_far
        if remaining > 0:
            await asyncio.sleep(remaining)

        # Add small buffer to ensure last audio frames are consumed
        await asyncio.sleep(0.06)

        # Speech finished — resume idle
        self.audio_track.stop_expecting()
        self.video_track.stop_speaking()
        self.is_speaking = False

        elapsed = time.time() - stream_start
        fps_actual = total_frames / elapsed if elapsed > 0 else 0
        print(
            f"CM: Greeting pushed: {total_frames} A/V pairs in {elapsed:.2f}s ({fps_actual:.1f} FPS)"
        )


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

        # SHARED wall-clock anchor — both audio and video tracks use this
        # to ensure they pace from the EXACT same origin. Neither track
        # re-anchors independently, eliminating clock divergence.
        self._anchor = None  # Set via set_anchor() when speech begins

        # Track cumulative CONTENT duration delivered (not transport frames)
        self._audio_content_s = 0.0  # seconds of real audio content delivered
        self._video_content_s = 0.0  # seconds of real video content delivered

        self._lock = threading.Lock()
        self._drift_history = collections.deque(maxlen=50)
        self._last_drift_log_time = 0

    @property
    def anchor(self):
        """The shared wall-clock anchor for A/V pacing."""
        return self._anchor

    def set_anchor(self, t=None):
        """Set the shared pacing anchor. Called once when speech starts.
        Both audio and video recv() use this to compute their target delivery times."""
        self._anchor = t if t is not None else time.time()

    def clear_anchor(self):
        """Clear the shared anchor when speech ends."""
        self._anchor = None

    def reset(self):
        """Reset for a new conversation turn."""
        with self._lock:
            self._audio_content_s = 0.0
            self._video_content_s = 0.0
            self._drift_history.clear()
        self._anchor = None

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
            avg_drift = (
                sum(self._drift_history) / len(self._drift_history)
                if self._drift_history
                else 0
            )
            if abs(avg_drift) > self.drift_threshold_ms:
                print(
                    f"⚠️ SYNC DRIFT: avg={avg_drift:.1f}ms (audio={'ahead' if avg_drift > 0 else 'behind'}) "
                    f"[audio_content={audio_t:.2f}s, video_content={video_t:.2f}s]"
                )
            self._last_drift_log_time = now

        return drift_ms


# --- Idle Animation Cache ---


class IdleAnimationCache:
    """Pre-renders and caches idle animation frames using natural ambient audio.

    Generates 8 seconds of breathing-like ambient audio through IMTalker to produce
    ~200 frames of natural micro-motion (subtle head sway, blinking, breathing).
    The loop uses ping-pong playback (forward then reverse) to eliminate visible
    loop seams, and includes crossfade blending at the boundary.

    Cached per-avatar (by image hash) so subsequent sessions reuse instantly.
    """

    IDLE_DURATION_S = 8.0  # 8s of content → 16s ping-pong loop before repeat
    SAMPLE_RATE = 16000  # wav2vec2 expected rate
    CROSSFADE_FRAMES = 8  # Frames to crossfade at loop boundary (320ms)

    def __init__(self):
        self._cache = {}  # {avatar_hash: List[VideoFrame]}
        self._lock = threading.Lock()
        self._rendering = set()  # avatar hashes currently being rendered

    @staticmethod
    def _hash_avatar(avatar_img):
        """Fast hash of avatar image for cache key."""
        import hashlib

        arr = np.array(avatar_img.convert("RGB"))
        # Use a downsampled version for speed
        small = arr[::8, ::8, :].tobytes()
        return hashlib.md5(small).hexdigest()

    @staticmethod
    def _generate_breathing_audio(duration_s, sample_rate=16000):
        """Generate natural breathing-like ambient audio for realistic idle motion.

        Creates a signal that mimics quiet breathing rhythm (~12-15 breaths/min)
        with subtle amplitude modulation. This produces much more natural
        micro-motions than pure noise or silence.
        """
        num_samples = int(duration_s * sample_rate)
        t = np.linspace(0, duration_s, num_samples, dtype=np.float32)

        # Breathing envelope: ~0.25 Hz (15 breaths/min) with harmonics
        breath_rate = 0.25  # Hz
        breath_envelope = (
            0.4 * np.sin(2 * np.pi * breath_rate * t)  # Primary breathing cycle
            + 0.15
            * np.sin(
                2 * np.pi * breath_rate * 2 * t + 0.3
            )  # 2nd harmonic (inhale/exhale asymmetry)
            + 0.05 * np.sin(2 * np.pi * 0.07 * t)  # Very slow head sway
        ).astype(np.float32)

        # Low-frequency noise for natural variation
        noise = np.random.randn(num_samples).astype(np.float32)
        # Simple low-pass via exponential moving average (~50Hz cutoff)
        filtered = np.zeros_like(noise)
        alpha = 0.02  # Lower = smoother
        filtered[0] = noise[0]
        for i in range(1, num_samples):
            filtered[i] = alpha * noise[i] + (1 - alpha) * filtered[i - 1]

        # Combine: breathing envelope modulates filtered noise
        audio = breath_envelope * 0.003 + filtered * 0.002

        # Normalize to safe range
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.005  # Very quiet — just enough for micro-motion

        return audio

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
        """Render idle animation frames with ping-pong looping (blocking, call from thread).

        Generates breathing-like audio and feeds it through the IMTalker pipeline
        to produce natural micro-motion frames. Creates a forward+reverse (ping-pong)
        loop with crossfade blending at the boundary for seamless looping.

        Returns:
            List[VideoFrame] - the cached idle loop frames (forward + reverse)
        """
        key = self._hash_avatar(avatar_img)

        # Check cache first
        with self._lock:
            if key in self._cache:
                print(
                    f"[IDLE CACHE] HIT for avatar {key[:8]}... ({len(self._cache[key])} frames)"
                )
                return self._cache[key]
            if key in self._rendering:
                print(f"[IDLE CACHE] Already rendering for {key[:8]}..., waiting...")
                pass
            self._rendering.add(key)

        try:
            t_start = time.time()
            print(
                f"[IDLE CACHE] Rendering natural idle animation for avatar {key[:8]}..."
            )

            # Generate breathing-like ambient audio for natural micro-motion
            breathing_audio = self._generate_breathing_audio(
                self.IDLE_DURATION_S, self.SAMPLE_RATE
            )

            def audio_iter():
                yield breathing_audio

            raw_frames = []  # List of numpy RGB arrays
            for video_frames, metrics in imtalker_streamer.generate_stream(
                avatar_img, audio_iter()
            ):
                for frame_tensor in video_frames:
                    frame_rgb = process_frame_tensor(frame_tensor, format="RGB")
                    raw_frames.append(frame_rgb)
                    del frame_tensor
                del video_frames

            # Clear GPU memory after render
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not raw_frames:
                print(f"[IDLE CACHE] WARNING: No frames rendered for {key[:8]}")
                with self._lock:
                    self._rendering.discard(key)
                return None

            # Create ping-pong loop: forward + reverse with crossfade at boundary
            n = len(raw_frames)
            cf = min(self.CROSSFADE_FRAMES, n // 4)  # Don't crossfade more than 25%

            # Forward pass (all frames)
            forward = list(raw_frames)
            # Reverse pass (skip first and last to avoid duplicates at boundaries)
            reverse = list(raw_frames[-2:0:-1]) if n > 2 else []

            # Apply crossfade blending at the forward→reverse boundary
            if cf > 0 and len(reverse) >= cf:
                for i in range(cf):
                    alpha = i / cf  # 0.0 (all forward-end) → 1.0 (all reverse-start)
                    fwd_frame = forward[-(cf - i)].astype(np.float32)
                    rev_frame = reverse[i].astype(np.float32)
                    blended = ((1 - alpha) * fwd_frame + alpha * rev_frame).astype(
                        np.uint8
                    )
                    forward[-(cf - i)] = blended
                    reverse[i] = blended

            # Combine into final loop
            all_frames_np = forward + reverse

            # Convert to VideoFrame objects
            idle_frames = [
                VideoFrame.from_ndarray(f, format="rgb24") for f in all_frames_np
            ]
            del raw_frames, forward, reverse, all_frames_np

            render_time = time.time() - t_start
            loop_duration = len(idle_frames) / 25.0
            print(
                f"[IDLE CACHE] ✓ Rendered {len(idle_frames)} idle frames in {render_time:.2f}s "
                f"({loop_duration:.1f}s ping-pong loop) for avatar {key[:8]}"
            )

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


def _split_audio_to_webrtc(audio_int16, frame_size=480):
    """Split int16 audio array into 20ms WebRTC frames (480 samples at 24kHz).

    Returns a list of np.ndarray chunks, each exactly frame_size samples.
    The last chunk applies a fade-out instead of harsh zero-padding.
    """
    chunks = []
    for i in range(0, len(audio_int16), frame_size):
        chunk = audio_int16[i : i + frame_size]
        if len(chunk) < frame_size:
            padded = np.zeros(frame_size, dtype=np.int16)
            padded[: len(chunk)] = chunk
            # Apply fade-out on the actual audio portion to avoid click at boundary
            fade_len = min(len(chunk), 48)  # ~1ms fade at 24kHz
            if fade_len > 1:
                fade = np.linspace(1.0, 0.0, fade_len)
                padded[len(chunk) - fade_len:len(chunk)] = (
                    padded[len(chunk) - fade_len:len(chunk)].astype(np.float32) * fade
                ).astype(np.int16)
            chunk = padded
        chunks.append(chunk)
    return chunks


class IMTalkerVideoTrack(VideoStreamTrack):
    """WebRTC video track with 25 FPS pacing for lip sync.

    ARCHITECTURE: This track reads video frames from frame_queue. Audio is
    pushed to the audio track SEPARATELY by av_streamer at the same time,
    so both tracks receive their content simultaneously from the same source.
    Both tracks pace off the same shared wall-clock anchor for tight sync.

    PTS: Overrides aiortc's 30 FPS next_timestamp() with 25 FPS (PTS += 3600
    at 90kHz clock). Includes reset_pacing() for phase alignment with audio.

    States:
      is_speaking=False → show idle animation (or static avatar)
      is_speaking=True, queue has frame → show speech frame
      is_speaking=True, queue empty, _speech_ending=False → repeat last_speech_frame (more coming)
      is_speaking=True, queue empty, _speech_ending=True → crossfade to idle
    """

    VIDEO_CLOCK_RATE = 90000
    VIDEO_FPS = 25.0
    FRAME_DURATION = 1.0 / VIDEO_FPS  # 0.04s = 40ms per frame
    PTS_PER_FRAME = int(VIDEO_CLOCK_RATE / VIDEO_FPS)  # 3600

    def __init__(
        self,
        avatar_img,
        sync_manager=None,
        idle_frames=None,
        full_img_rgb=None,
        crop_coords=None,
    ):
        super().__init__()
        self.avatar_img = avatar_img
        self.frame_queue = asyncio.Queue()  # VideoFrame objects from av_streamer
        self.sync_manager = sync_manager

        # Crop-Infer-Paste for idle frames
        self._full_img_rgb = full_img_rgb  # np.ndarray (H, W, 3) uint8, or None
        self._crop_coords = crop_coords  # (x1, y1, x2, y2), or None
        self._paste_back_idle = full_img_rgb is not None and crop_coords is not None

        if self._paste_back_idle:
            # Static idle = full image (already has the face in it)
            self.static_idle_frame = full_img_rgb.copy()
        else:
            self.static_idle_frame = np.array(avatar_img.convert("RGB"))

        self._idle_frames = []
        if idle_frames:
            # set_idle_frames applies paste-back if needed
            self.set_idle_frames(idle_frames)
        self._idle_index = 0
        self.is_speaking = False  # Set by av_streamer
        self._speech_ending = False  # True once all pairs pushed, queue draining
        self._last_speech_frame = None  # Hold frame for repeat during speech

        # Smooth transitions: crossfade between speech↔idle
        self._transition_frames = []  # Pre-computed crossfade frames
        self._transition_index = 0
        self._in_transition = False  # True during speech→idle crossfade
        self.TRANSITION_STEPS = 6  # ~240ms crossfade (6 frames at 25 FPS)

        # PTS counter — always advances monotonically at 25 FPS rate
        self._pts_counter = 0
        self._pts_start_wall = None  # Wall-clock time of first recv()
        self._rush_next_frame = False  # Skip sleep on first speech frame

    def set_idle_frames(self, frames):
        if self._paste_back_idle:
            # Paste-back each idle frame into the full image
            x1, y1, x2, y2 = self._crop_coords
            region_h, region_w = y2 - y1, x2 - x1
            pasted_frames = []
            for vf in frames:
                face_rgb = vf.to_ndarray(format="rgb24")
                face_resized = cv2.resize(
                    face_rgb, (region_w, region_h), interpolation=cv2.INTER_LINEAR
                )
                result = self._full_img_rgb.copy()
                result[y1:y2, x1:x2] = face_resized
                pasted_frames.append(VideoFrame.from_ndarray(result, format="rgb24"))
            self._idle_frames = pasted_frames
            print(
                f"VT: Idle animation set ({len(pasted_frames)} frames, paste-back applied)"
            )
        else:
            self._idle_frames = frames
            print(f"VT: Idle animation set ({len(frames)} frames)")

    def _get_idle_frame(self):
        """Get next idle animation frame with seamless looping."""
        if self._idle_frames:
            frame = self._idle_frames[self._idle_index % len(self._idle_frames)]
            self._idle_index += 1
            return frame
        else:
            return VideoFrame.from_ndarray(self.static_idle_frame, format="rgb24")

    def _compute_crossfade(self, from_frame, to_frame_fn, steps):
        """Pre-compute crossfade frames from a source frame to idle animation.

        Args:
            from_frame: The last speech VideoFrame
            to_frame_fn: Callable that returns the next idle frame
            steps: Number of crossfade steps
        Returns:
            List of blended VideoFrame objects
        """
        try:
            src = from_frame.to_ndarray(format="rgb24").astype(np.float32)
            blended_frames = []
            for i in range(1, steps + 1):
                alpha = i / steps  # 0→1 (speech→idle)
                dst_frame = to_frame_fn()
                dst = dst_frame.to_ndarray(format="rgb24").astype(np.float32)
                blended = ((1 - alpha) * src + alpha * dst).astype(np.uint8)
                blended_frames.append(VideoFrame.from_ndarray(blended, format="rgb24"))
            return blended_frames
        except Exception:
            return []

    def reset_pacing(self, clock_anchor=None):
        """Re-anchor wall-clock pacing, preserving PTS monotonicity.

        Args:
            clock_anchor: Optional wall-clock time to use as anchor.
                          If None, uses current time.
        """
        if self._pts_start_wall is not None:
            anchor = clock_anchor if clock_anchor is not None else time.time()
            # Shift wall-clock anchor so next next_timestamp() targets the anchor time.
            self._pts_start_wall = (
                anchor
                - (self._pts_counter + self.PTS_PER_FRAME) / self.VIDEO_CLOCK_RATE
            )

    def mark_speech_ending(self):
        """Signal that all A/V pairs have been pushed — no more coming.

        When queue empties after this, recv() will initiate a smooth crossfade
        from the last speech frame to idle animation instead of an abrupt cut.
        """
        self._speech_ending = True

    def start_speaking(self, clock_anchor=None):
        """Called when av_streamer begins pushing content.
        Re-anchors pacing, clears transitions, and drains stale data.

        Args:
            clock_anchor: Optional wall-clock time for pacing anchor.
                          Pass the SAME value to audio_track.reset_clock()
                          for tight A/V sync.
        """
        self.is_speaking = True
        self._speech_ending = False
        self._last_speech_frame = None
        self._in_transition = False
        self._transition_frames = []
        self._transition_index = 0
        self._rush_next_frame = True  # Deliver first speech frame ASAP
        self.reset_pacing(clock_anchor=clock_anchor)
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
        self._speech_ending = False
        self._in_transition = False
        self._transition_frames = []
        self._transition_index = 0
        if self.sync_manager:
            self.sync_manager.clear_anchor()

    async def next_timestamp(self):
        """Override aiortc's 30 FPS next_timestamp with 25 FPS PTS and pacing.

        aiortc default: PTS += 3000 (90kHz/30), paces at 33.3ms
        Our override:   PTS += 3600 (90kHz/25), paces at 40.0ms

        This ensures the RTP timestamps tell the browser to play video at 25 FPS,
        matching the audio frame rate (960 samples at 24kHz = 40ms per video frame).
        """
        if self.readyState != "live":
            raise MediaStreamError

        if self._pts_start_wall is not None:
            self._pts_counter += self.PTS_PER_FRAME  # 3600 for 25 FPS
            if self._rush_next_frame:
                # First speech frame — deliver ASAP without sleeping.
                # This prevents audio-leads-video at speech start when
                # video's recv() was mid-sleep for the next idle frame.
                # Browser uses PTS (not delivery time) for playback.
                self._rush_next_frame = False
            else:
                target = self._pts_start_wall + (
                    self._pts_counter / self.VIDEO_CLOCK_RATE
                )
                wait = target - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
        else:
            self._pts_start_wall = time.time()
            self._pts_counter = 0

        return self._pts_counter, fractions.Fraction(1, self.VIDEO_CLOCK_RATE)

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        if self._in_transition:
            # Playing crossfade transition from speech → idle
            if self._transition_index < len(self._transition_frames):
                frame = self._transition_frames[self._transition_index]
                self._transition_index += 1
            else:
                # Transition complete — resume normal idle
                self._in_transition = False
                self._transition_frames = []
                self._transition_index = 0
                frame = self._get_idle_frame()
        elif self.is_speaking:
            # Pop a video frame from the queue.
            # Audio is pushed separately by av_streamer to the audio track,
            # so both tracks stay in sync via shared wall-clock anchor.
            try:
                frame = self.frame_queue.get_nowait()
                self._last_speech_frame = frame
            except asyncio.QueueEmpty:
                if self._speech_ending:
                    # All speech content consumed — start smooth crossfade to idle
                    if self._last_speech_frame and self._idle_frames:
                        self._transition_frames = self._compute_crossfade(
                            self._last_speech_frame,
                            self._get_idle_frame,
                            self.TRANSITION_STEPS,
                        )
                        if self._transition_frames:
                            self._in_transition = True
                            self._transition_index = 0
                            frame = self._transition_frames[0]
                            self._transition_index = 1
                        else:
                            frame = self._get_idle_frame()
                    else:
                        frame = self._get_idle_frame()
                else:
                    # Queue temporarily empty, more content coming — hold last frame.
                    frame = self._last_speech_frame or self._get_idle_frame()
        else:
            frame = self._get_idle_frame()

        frame.pts = pts
        frame.time_base = time_base
        return frame


class IMTalkerAudioTrack(AudioStreamTrack):
    """WebRTC audio track with proper PTS and pacing for lip sync.

    PTS and pacing are DECOUPLED:
    - _pts: monotonically increasing, never reset. Used for RTP timestamps.
      A backward PTS jump would confuse the browser's jitter buffer.
    - _pacing_start / _pacing_count: resettable pacing clock. Controls the
      wall-clock delivery rate (20ms per frame at 24kHz).

    On reset_clock(): pacing restarts from NOW, but PTS continues advancing.
    This ensures smooth RTP timestamp progression while allowing the pacing
    to re-sync with the video track at each speech start.
    """

    def __init__(self, sync_manager=None):
        super().__init__()
        self.audio_queue = asyncio.Queue()  # Unbounded — overflow handled by caller
        self.sync_manager = sync_manager
        self._pts = 0  # Monotonic PTS counter (never reset)
        self._pacing_start = None  # Wall-clock anchor for pacing
        self._pacing_count = 0  # Frames since pacing start
        self._content_frames = 0  # Count of real (non-silence) frames delivered
        self.sample_rate = 24000
        self.samples_per_frame = 480  # 20ms at 24kHz
        self.is_buffering = False
        self._expecting_content = (
            False  # True during speech — enables jitter-tolerant wait
        )

    def reset_clock(self, clock_anchor=None):
        """Re-anchor the pacing clock.

        Called by av_streamer when the first real A/V pair is pushed.
        Resets pacing so audio delivery starts from this instant,
        but does NOT reset PTS (to avoid RTP timestamp discontinuity).

        Args:
            clock_anchor: Optional wall-clock time to use as anchor.
                          If None, uses time.time(). Pass the SAME value
                          to video_track.start_speaking() for tight A/V sync.
        """
        self._pacing_start = clock_anchor if clock_anchor is not None else time.time()
        self._pacing_count = 0
        self._content_frames = 0
        self._expecting_content = True
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

    def stop_expecting(self):
        """Signal that no more audio content is expected (speech ended)."""
        self._expecting_content = False

    async def add_audio(self, audio_np):
        """Expects normalized float32 numpy array. Splits into WebRTC-sized frames."""
        # Apply fade-in/out to prevent clicks at audio boundaries
        audio_np = apply_audio_chunk_fades(audio_np, sr=self.sample_rate, fade_ms=10.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        frame_size = self.samples_per_frame
        num_frames = len(audio_int16) // frame_size

        for i in range(num_frames):
            await self.audio_queue.put(
                audio_int16[i * frame_size : (i + 1) * frame_size]
            )

        remainder = len(audio_int16) % frame_size
        if remainder > 0:
            last_frame = np.zeros(frame_size, dtype=np.int16)
            last_frame[:remainder] = audio_int16[-remainder:]
            await self.audio_queue.put(last_frame)

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError

        # Initialize pacing clock on very first call
        if self._pacing_start is None:
            self._pacing_start = time.time()

        # Wall-clock pacing: deliver one frame every 20ms
        target_time = self._pacing_start + self._pacing_count * (
            self.samples_per_frame / self.sample_rate
        )
        wait = target_time - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        self._pacing_count += 1

        # Get audio with jitter-tolerant waiting during speech.
        # During speech, audio is pushed by av_streamer. A short wait absorbs
        # scheduling jitter and prevents silence gaps (scratchy audio).
        # During idle, return silence immediately (no waiting).
        try:
            audio_int16 = self.audio_queue.get_nowait()
            self._content_frames += 1
        except asyncio.QueueEmpty:
            if self._expecting_content:
                try:
                    audio_int16 = await asyncio.wait_for(
                        self.audio_queue.get(), timeout=0.008
                    )
                    self._content_frames += 1
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    audio_int16 = np.zeros(self.samples_per_frame, dtype=np.int16)
            else:
                audio_int16 = np.zeros(self.samples_per_frame, dtype=np.int16)

        frame = AudioFrame.from_ndarray(
            audio_int16.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._pts += frame.samples

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
            avatar_img = Image.new("RGB", (512, 512), color="red")
            print(f"API: Avatar not found, using dummy: {avatar_path}")
        else:
            avatar_img = Image.open(avatar_path).convert("RGB")

        # Create TTS Stream
        tts_stream = tts_model.generate_stream(
            text=request.text,
            chunk_size=25,  # Balanced chunks to avoid GPU OOM
            print_metrics=False,
        )
        audio_iterator = chatterbox_adapter(tts_stream)

        # Generate Frames
        all_frames = []
        for video_frames, metrics in imtalker_streamer.generate_stream(
            avatar_img, audio_iterator
        ):
            for frame_tensor in video_frames:
                frame_bgr = process_frame_tensor(frame_tensor)
                all_frames.append(frame_bgr)

        # Save Video
        if all_frames:
            height, width, _ = all_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(request.output_path, fourcc, 25.0, (width, height))
            for frame in all_frames:
                out.write(frame)
            out.release()
            return {
                "status": "success",
                "output_path": request.output_path,
                "frames": len(all_frames),
            }
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
            avatar_img = Image.new("RGB", (512, 512), color="red")
            print(f"API Stream: Avatar not found, using dummy: {avatar_path}")
        else:
            avatar_img = Image.open(avatar_path).convert("RGB")

        # Create TTS Stream
        tts_stream = tts_model.generate_stream(
            text=request.text,
            chunk_size=25,  # Balanced chunks to avoid GPU OOM
            print_metrics=False,
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
                        for video_frames, metrics in imtalker_streamer.generate_stream(
                            avatar_img, wrapped_audio_iterator()
                        ):
                            # Prepare frames as base64
                            b64_frames = []
                            for frame_tensor in video_frames:
                                frame_bgr = process_frame_tensor(frame_tensor)
                                ret, buffer = cv2.imencode(".jpg", frame_bgr)
                                if ret:
                                    b64_frames.append(
                                        base64.b64encode(buffer).decode("utf-8")
                                    )

                            # Prepare audio as base64 WAV
                            audio_b64 = ""
                            if current_audio[0] is not None:
                                audio_b64 = numpy_to_wav_base64(current_audio[0])

                            payload = {
                                "audio": audio_b64,
                                "frames": b64_frames,
                                "metrics": metrics,
                            }
                            # Put payload into queue (block if full)
                            # Since we are in a thread, we use run_coroutine_threadsafe or similar?
                            # Actually, we can just yield symbols from imtalker_streamer and process them in the producer asyncly.
                            # But imtalker_streamer.generate_stream is a blocking generator.
                            return (
                                payload,
                                True,
                            )  # This is not ideal for multiple yields.

                    # Better producer logic:
                    def render_loop():
                        for video_frames, metrics in imtalker_streamer.generate_stream(
                            avatar_img, wrapped_audio_iterator()
                        ):
                            b64_frames = []
                            for frame_tensor in video_frames:
                                frame_bgr = process_frame_tensor(frame_tensor)
                                ret, buffer = cv2.imencode(".jpg", frame_bgr)
                                if ret:
                                    b64_frames.append(
                                        base64.b64encode(buffer).decode("utf-8")
                                    )

                            audio_b64 = ""
                            if current_audio[0] is not None:
                                audio_b64 = numpy_to_wav_base64(current_audio[0])

                            payload = {
                                "audio": audio_b64,
                                "frames": b64_frames,
                                "metrics": metrics,
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
            avatar_img = Image.new("RGB", (512, 512), color="red")
            print(f"API Stream (GET): Avatar not found, using dummy: {avatar_path}")
        else:
            avatar_img = Image.open(avatar_path).convert("RGB")

        # Create TTS Stream
        tts_stream = tts_model.generate_stream(
            text=text,
            chunk_size=25,  # Balanced chunks to avoid GPU OOM
            print_metrics=False,
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
                        for video_frames, metrics in imtalker_streamer.generate_stream(
                            avatar_img, wrapped_audio_iterator()
                        ):
                            b64_frames = []
                            for frame_tensor in video_frames:
                                frame_bgr = process_frame_tensor(frame_tensor)
                                ret, buffer = cv2.imencode(".jpg", frame_bgr)
                                if ret:
                                    b64_frames.append(
                                        base64.b64encode(buffer).decode("utf-8")
                                    )

                            audio_b64 = ""
                            if current_audio[0] is not None:
                                audio_b64 = numpy_to_wav_base64(current_audio[0])

                            payload = {
                                "audio": audio_b64,
                                "frames": b64_frames,
                                "metrics": metrics,
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


# ============================================================================
# OSHARA SESSION HELPERS
# ============================================================================


def _record_oshara_log(
    app_instance,
    message: str = "",
    response: str = "",
    additional_data: dict = None,
    session_id: str = None,
):
    """Append a conversation log entry to a WebRTC session's Oshara log list.

    Args:
        app_instance: The FastAPI app instance
        message: User message text
        response: AI response text
        additional_data: Extra data dict (timings, etc.)
        session_id: Explicit WebRTC session ID (preferred). Falls back to active_webrtc_session.
    """
    sid = session_id or getattr(app_instance.state, "active_webrtc_session", None)
    if not sid:
        logger.warning("Oshara log: no session_id available, skipping")
        return
    meta = app_instance.state.active_session_meta.get(sid)
    if not meta:
        logger.warning(
            f"Oshara log: session {sid} not found in active_session_meta, skipping"
        )
        return
    if not meta.get("oshara_session_id"):
        logger.warning(f"Oshara log: session {sid} has no oshara_session_id, skipping")
        return

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "response": response,
        "additional_data": additional_data or {},
    }
    meta.setdefault("conversation_logs", []).append(log_entry)
    total_logs = len(meta["conversation_logs"])
    logger.info(
        f"Oshara log recorded for session {sid} (total={total_logs}): msg='{message[:40] if message else ''}', resp='{response[:40] if response else ''}'"
    )
    logger.info(f"Oshara log entry: {json.dumps(log_entry, default=str)[:300]}")


def _update_last_oshara_log_timings(
    app_instance, timings: dict, session_id: str = None
):
    """Update the most recent Oshara log entry's additional_data with step timings.

    Called at the end of a pipeline to attach TTS/render/streaming durations
    to the response log entry that was created when the AI response arrived.

    Args:
        app_instance: The FastAPI app instance
        timings: Dict of timing values to merge into additional_data
        session_id: Explicit WebRTC session ID (preferred). Falls back to active_webrtc_session.
    """
    sid = session_id or getattr(app_instance.state, "active_webrtc_session", None)
    if not sid:
        return
    meta = app_instance.state.active_session_meta.get(sid)
    if not meta or not meta.get("oshara_session_id"):
        return

    logs = meta.get("conversation_logs", [])
    if not logs:
        return

    # Find the last log entry that has a response (AI response log)
    found = False
    for entry in reversed(logs):
        if entry.get("response"):
            entry.setdefault("additional_data", {}).update(timings)
            found = True
            logger.info(
                f"Oshara log timings updated for session {sid} — keys: {list(timings.keys())}"
            )
            logger.info(
                f"Oshara updated log entry additional_data: {json.dumps(entry.get('additional_data', {}), default=str)[:600]}"
            )
            break
    if not found:
        logger.warning(
            f"Oshara log timings: no response entry found in {len(logs)} log(s) for session {sid} — timings NOT saved"
        )


async def _close_oshara_session(meta: dict):
    """Send conversation logs to Oshara close API and mark session as closed."""
    oshara_sid = meta.get("oshara_session_id")
    bearer_token = meta.get("oshara_bearer_token")
    if not oshara_sid or not bearer_token:
        return

    raw_logs = meta.get("conversation_logs", [])
    logger.info(f"Closing Oshara session {oshara_sid} with {len(raw_logs)} log entries")
    if raw_logs:
        for i, entry in enumerate(raw_logs):
            logger.info(
                f"  Log[{i}]: msg='{(entry.get('message','') or '')[:60]}', resp='{(entry.get('response','') or '')[:60]}', data={entry.get('additional_data',{})}"
            )
    else:
        logger.warning(f"Oshara session {oshara_sid}: NO conversation logs collected!")

    # Sanitize logs: round-trip through JSON to convert any non-serializable types
    # (numpy floats/ints, datetime objects, etc.) to plain JSON-safe values.
    try:
        logs = json.loads(json.dumps(raw_logs, default=str))
    except Exception as ser_err:
        logger.error(f"Oshara log serialization error: {ser_err}; sending raw logs")
        logs = raw_logs

    payload = {"logs": logs}
    logger.info(
        f"Oshara close request URL: {OSHARA_API_BASE}/ai-conversation-session/{oshara_sid}/close/"
    )
    logger.info(
        f"Oshara close request payload: {json.dumps(payload, default=str)[:1000]}"
    )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{OSHARA_API_BASE}/ai-conversation-session/{oshara_sid}/close/",
                json=payload,
                headers={"Authorization": f"Bearer {bearer_token}"},
            )
            logger.info(f"Oshara close response status: {resp.status_code}")
            logger.info(f"Oshara close response body: {resp.text[:500]}")
            if resp.status_code >= 400:
                logger.error(f"Oshara close API error {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Oshara session {oshara_sid} closed successfully: {result}")
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Failed to close Oshara session {oshara_sid}: HTTP {e.response.status_code} — {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to close Oshara session {oshara_sid}: {e}")
        import traceback

        logger.error(traceback.format_exc())


async def _cleanup_webrtc_session(session_id: str, reason: str = "unknown"):
    """Full cleanup of a WebRTC session: close PC, cleanup ConvManager, free GPU, clear state."""
    logger.info(f"🧹 Cleaning up session {session_id} (reason: {reason})")

    meta = app.state.active_session_meta.get(session_id)

    # 0. Close Oshara session and send conversation logs
    if meta:
        await _close_oshara_session(meta)

    # 1. Cleanup ConversationManager (executors, buffers, Realtime session)
    if meta and meta.get("conv_manager"):
        try:
            await meta["conv_manager"].cleanup_async()
        except Exception as e:
            logger.error(f"ConvManager cleanup error: {e}")

    # 2. Drain track queues to free memory
    if meta:
        for track_key in ("video_track", "audio_track"):
            track = meta.get(track_key)
            if track and hasattr(track, "frame_queue"):
                while not track.frame_queue.empty():
                    try:
                        track.frame_queue.get_nowait()
                    except:
                        break
            if track and hasattr(track, "audio_queue"):
                while not track.audio_queue.empty():
                    try:
                        track.audio_queue.get_nowait()
                    except:
                        break

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
    import gc

    gc.collect()

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
        greeting = data.get(
            "initial_text", data.get("text", "Hello, I am ready to talk.")
        )
        reference_audio = data.get("reference_audio", None)
        system_prompt = data.get("system_prompt", None)
        crop = data.get("crop", True)
        force = data.get("force", False)  # Force-disconnect existing session
        character_slug = data.get("character_slug", None)  # Oshara KB character slug

        logger.info(
            f"Params: conv_mode={conv_mode}, crop={crop}, greeting='{greeting[:40] if greeting else None}' character_slug={character_slug}, reference_audio={'provided' if reference_audio else 'none'}"
        )

        if not sdp:
            raise HTTPException(status_code=400, detail="SDP is required")

        # ─── Oshara session auth ───
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid Authorization header. Bearer token required.",
            )
        bearer_token = auth_header.split(" ", 1)[1]

        oshara_session_id = None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                oshara_resp = await client.post(
                    f"{OSHARA_API_BASE}/ai-conversation-session/auth/",
                    json={"metadata": {}},
                    headers={"Authorization": f"Bearer {bearer_token}"},
                )
                oshara_resp.raise_for_status()
                oshara_data = oshara_resp.json()
                logger.info(f"Oshara auth response: {oshara_data}")

                # Response is nested: {"success": ..., "data": {"allow_to_connect": ..., "session_id": ...}}
                oshara_payload = oshara_data.get("data", oshara_data)

                if not oshara_payload.get("allow_to_connect"):
                    raise HTTPException(
                        status_code=403,
                        detail=oshara_payload.get(
                            "message", "Connection not allowed by Oshara API."
                        ),
                    )
                oshara_session_id = oshara_payload.get("session_id")
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Oshara auth HTTP error: {e.response.status_code} {e.response.text}"
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Oshara auth failed: {e.response.text}",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Oshara auth error: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Could not reach Oshara auth API: {e}",
            )

        # ─── Session limit guard: enforce MAX_CONCURRENT_WEBRTC_SESSIONS ───
        async with app.state._session_lock:
            active_count = len(app.state.webrtc_sessions)
            if active_count >= MAX_CONCURRENT_WEBRTC_SESSIONS:
                if force:
                    # Force-disconnect the oldest session(s) to make room
                    oldest_id = next(iter(app.state.webrtc_sessions))
                    logger.warning(
                        f"⚠️ Force-disconnecting session {oldest_id} (at limit {MAX_CONCURRENT_WEBRTC_SESSIONS})"
                    )
                    await _cleanup_webrtc_session(oldest_id, reason="force-replaced")
                else:
                    active_ids = list(app.state.webrtc_sessions.keys())
                    raise HTTPException(
                        status_code=409,
                        detail=f"Session limit reached ({active_count}/{MAX_CONCURRENT_WEBRTC_SESSIONS}). "
                        f"Active: {active_ids}. Send force=true to disconnect oldest, or POST /disconnect first.",
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
                avatar_path = await download_url_to_temp(
                    avatar, prefix="avatar_url_", suffix=".png"
                )
            elif (
                avatar.startswith("data:image") or len(avatar) > 200
            ):  # Simple heuristic for Base64
                logger.info("Decoding Base64 avatar...")
                avatar_path = save_base64_to_temp(
                    avatar, prefix="avatar_", suffix=".png"
                )
            else:
                avatar_path = os.path.join(current_dir, avatar)
                if not os.path.exists(avatar_path):
                    avatar_path = avatar

        logger.info(f"Avatar processing time: {(time.time() - t_avatar)*1000:.2f}ms")

        # Crop-Infer-Paste state: when crop=False, we detect the face, crop it
        # for the model, but keep the full image + coords for paste-back after render.
        full_img_rgb_for_paste = None  # np.ndarray (H, W, 3) of full image, or None
        crop_coords_for_paste = None  # (x1, y1, x2, y2) tuple, or None

        if avatar_path and os.path.exists(avatar_path):
            logger.info(f"Using avatar from: {avatar_path}")
            avatar_img_raw = Image.open(avatar_path).convert("RGB")
            if crop and hasattr(app.state, "inference_agent"):
                # crop=True: face-detect + crop to 512x512 (no paste-back)
                logger.info("Applying face crop to avatar...")
                avatar_img = app.state.inference_agent.data_processor.process_img(
                    avatar_img_raw
                )
            elif not crop and hasattr(app.state, "inference_agent"):
                # crop=False: Crop-Infer-Paste mode
                # Detect face, crop for model input, keep full image for paste-back
                logger.info("Crop-Infer-Paste mode: detecting face for paste-back...")
                dp = app.state.inference_agent.data_processor
                img_arr = np.array(avatar_img_raw)
                if img_arr.ndim == 2:
                    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
                elif img_arr.shape[2] == 4:
                    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
                crop_coords_for_paste = dp.get_crop_coords(img_arr)
                x1, y1, x2, y2 = crop_coords_for_paste
                cropped_face = img_arr[y1:y2, x1:x2]
                avatar_img = Image.fromarray(cropped_face).resize((512, 512))
                full_img_rgb_for_paste = (
                    img_arr  # Keep full-res original for paste-back
                )
                logger.info(
                    f"Crop-Infer-Paste: face region ({x1},{y1})-({x2},{y2}), full image {img_arr.shape[1]}x{img_arr.shape[0]}"
                )
            else:
                avatar_img = avatar_img_raw.resize((512, 512))
        else:
            logger.warning(f"Avatar not found or invalid: {avatar[:50]}...")
            avatar_img = Image.new("RGB", (512, 512), color="red")

        # Process Reference Audio (Path, URL or Base64)
        t_audio = time.time()
        # reference_audio can be: local path, URL, base64, predefined voice name, or STTS reference filename
        reference_audio_mode = None  # "file", "predefined", "clone", or None
        if reference_audio:
            if reference_audio.startswith("http://") or reference_audio.startswith(
                "https://"
            ):
                logger.info(f"Downloading reference audio from URL: {reference_audio}")
                reference_audio_path = await download_url_to_temp(
                    reference_audio, prefix="refstart_url_", suffix=".wav"
                )
                if reference_audio_path and os.path.isfile(reference_audio_path):
                    reference_audio = reference_audio_path
                    reference_audio_mode = "file"
                else:
                    logger.error(
                        f"Failed to download reference audio from URL: {reference_audio}"
                    )
                    reference_audio = None
            elif len(reference_audio) > 200 or reference_audio.startswith(
                "data:audio"
            ):  # Base64 heuristic
                logger.info("Decoding Base64 reference audio...")
                reference_audio_path = save_base64_to_temp(
                    reference_audio, prefix="refstart_", suffix=".wav"
                )
                reference_audio = (
                    reference_audio_path  # Update variable to point to file
                )
                reference_audio_mode = "file"
            elif os.path.exists(reference_audio):
                # Full local file path
                reference_audio_mode = "file"
            else:
                # Not a local path — check if it's a known STTS voice name.
                # Check predefined voices directory first, then reference_audio dir.
                stts_voices_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "STTS-Server",
                    "voices",
                )
                stts_ref_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "STTS-Server",
                    "reference_audio",
                )

                if os.path.isfile(os.path.join(stts_voices_dir, reference_audio)):
                    logger.info(
                        f"reference_audio '{reference_audio}' matched STTS predefined voice"
                    )
                    reference_audio_mode = "predefined"
                elif os.path.isfile(os.path.join(stts_ref_dir, reference_audio)):
                    logger.info(
                        f"reference_audio '{reference_audio}' matched STTS reference audio"
                    )
                    reference_audio_mode = "clone"
                else:
                    logger.warning(
                        f"Reference audio not found locally or on STTS server: {reference_audio}"
                    )
                    reference_audio = None
        logger.info(
            f"Reference audio processing time: {(time.time() - t_audio)*1000:.2f}ms"
        )

        if reference_audio:
            logger.info(
                f"Using reference audio: {reference_audio} (mode={reference_audio_mode})"
            )

        # Resolve voice: explicit reference_audio takes priority over avatar config
        avatar_voice_config = {}
        if not reference_audio:
            # No explicit voice — fall back to avatar→voice mapping from config
            avatar_voice_config = resolve_avatar_voice(avatar, tts_model)
            if avatar_voice_config.get("voice"):
                logger.info(
                    f"Auto-resolved avatar voice: {avatar_voice_config['voice']} (type: {avatar_voice_config.get('voice_type', 'predefined')})"
                )
            # Use avatar-specific greeting if available and no custom greeting was specified
            if (
                avatar_voice_config.get("greeting")
                and greeting == "Hi there! How are you doing today?"
            ):
                greeting = avatar_voice_config["greeting"]
                logger.info(f"Using avatar-specific greeting: '{greeting[:50]}'")
        elif reference_audio_mode == "predefined":
            # Voice filename that exists in STTS predefined voices dir
            tts_model.set_predefined_voice(reference_audio)
            logger.info(f"Set TTS to predefined voice: {reference_audio}")
        elif reference_audio_mode == "clone":
            # Voice filename that exists in STTS reference_audio dir
            tts_model.set_clone_voice(reference_audio)
            logger.info(f"Set TTS to clone voice: {reference_audio}")
        elif reference_audio_mode == "file":
            # Local file — upload to STTS server and set as clone voice
            if hasattr(tts_model, "prepare_conditionals") and os.path.isfile(
                reference_audio
            ):
                try:
                    tts_model.prepare_conditionals(reference_audio)
                    logger.info(f"Prepared TTS conditionals from reference audio file")
                except Exception as e:
                    logger.warning(f"Failed to prepare TTS conditionals: {e}")

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
        video_track = IMTalkerVideoTrack(
            avatar_img,
            sync_manager=sync_manager,
            idle_frames=idle_frames,
            full_img_rgb=full_img_rgb_for_paste,
            crop_coords=crop_coords_for_paste,
        )
        audio_track = IMTalkerAudioTrack(sync_manager=sync_manager)
        # Audio is now pushed directly by av_streamer to both tracks separately
        # (no longer video→audio linked — eliminates timing dependency)

        pc.addTrack(video_track)
        pc.addTrack(audio_track)

        # Session ID + register as active
        session_id = str(uuid.uuid4())
        app.state.webrtc_sessions[session_id] = pc
        app.state.active_webrtc_session = session_id
        app.state.active_session_meta[session_id] = {
            "session_id": session_id,
            "status": "connecting",  # connecting → greeting → ready → disconnected
            "greeting_done": False,
            "conv_mode": conv_mode,
            "video_track": video_track,
            "audio_track": audio_track,
            "conv_manager": None,
            "created_at": time.time(),
            # Oshara session tracking
            "oshara_session_id": oshara_session_id,
            "oshara_bearer_token": bearer_token,
            "conversation_logs": [],  # [{timestamp, message, response, additional_data}]
        }

        # Kick off idle animation render in background if not cached yet
        if idle_frames is None and not idle_animation_cache.is_rendering(avatar_img):

            async def render_idle_bg():
                try:
                    frames = await asyncio.get_event_loop().run_in_executor(
                        None,
                        idle_animation_cache.render_idle,
                        avatar_img,
                        imtalker_streamer,
                    )
                    if frames:
                        video_track.set_idle_frames(frames)
                        logger.info(
                            f"Idle animation ready for session {session_id} ({len(frames)} frames)"
                        )
                except Exception as e:
                    logger.error(f"Background idle render failed: {e}")

            asyncio.create_task(render_idle_bg())

        # Conversation Manager (optional)
        conv_manager = None
        if conv_mode:
            if not openai_client:
                raise HTTPException(
                    status_code=400,
                    detail="OpenAI API key not configured, conversational mode unavailable.",
                )
            conv_manager = ConversationManager(
                openai_client,
                tts_model,
                imtalker_streamer,
                video_track,
                audio_track,
                avatar_img,
                system_prompt=system_prompt,
                reference_audio=reference_audio,
                sync_manager=sync_manager,
                webrtc_session_id=session_id,
                full_img_rgb=full_img_rgb_for_paste,
                crop_coords=crop_coords_for_paste,
                character_slug=character_slug,
                kb_bearer_token=bearer_token,
            )
            # Connect to OpenAI Realtime API (WebSocket for VAD + STT + LLM)
            await conv_manager.connect_realtime()
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
                            await conv_manager.add_microphone_audio(
                                pcm, frame.sample_rate
                            )
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

                # OPTIMIZED: Show idle animation immediately while greeting renders in background
                # This eliminates the 2-8s perceived "connecting" delay
                # Idle animation is automatically shown by video track when not speaking
                logger.info(
                    f"🎬 Connection established - idle animation visible while greeting renders"
                )

                # Wait for greeting to be ready (if rendering in background)
                # Poll with faster interval for responsiveness
                greeting_wait_start = time.time()
                max_wait = 10.0
                check_interval = 0.05  # Check every 50ms (faster than original 100ms)

                while True:
                    pairs = meta.get("pre_rendered_pairs")
                    greeting_error = meta.get("greeting_error")

                    if pairs:
                        # Greeting is ready!
                        break
                    elif greeting_error:
                        # Greeting render failed
                        logger.error(f"Greeting render failed: {greeting_error}")
                        break
                    elif (time.time() - greeting_wait_start) > max_wait:
                        # Timeout waiting for greeting
                        logger.warning(f"Greeting render timeout after {max_wait}s")
                        break
                    else:
                        # Still rendering - user sees idle animation (instant visual feedback)
                        await asyncio.sleep(check_interval)

                wait_time = time.time() - greeting_wait_start
                if wait_time > 0.5:
                    logger.info(
                        f"Greeting ready after {wait_time:.2f}s (user experienced: instant idle → greeting)"
                    )

                # Push the greeting (if available)
                greeting_cm = meta.get("greeting_cm") or conv_manager

                if pairs and greeting_cm:
                    logger.info(f"🎙️ Pushing greeting ({len(pairs)} A/V pairs)...")
                    try:
                        await greeting_cm.push_buffered_greeting(pairs)
                        render_time = meta.get("greeting_render_time", 0)
                        logger.info(
                            f"✅ Greeting delivered (rendered in {render_time:.2f}s, UX: instant idle → greeting)"
                        )
                    except Exception as e:
                        logger.error(f"Greeting push failed: {e}")

                    # Free the buffer
                    meta.pop("pre_rendered_pairs", None)

                    # Cleanup temp CM if one-shot
                    temp_cm = meta.pop("greeting_cm", None)
                    if not conv_mode and temp_cm and temp_cm is not conv_manager:
                        temp_cm.cleanup()
                else:
                    logger.warning(
                        "No greeting available (render may have failed or timed out)"
                    )

                meta["status"] = "ready"
                meta["greeting_done"] = True
                logger.info(f"📡 Session {session_id} is READY (greeting delivered).")

            elif state in ("failed", "closed", "disconnected"):
                await _cleanup_webrtc_session(session_id, reason=state)

        # ─── Render greeting in BACKGROUND (deferred) ───
        # Return SDP answer immediately, render greeting asynchronously.
        # This eliminates the 2-5s perceived "connecting" delay.
        # Greeting will be pushed when WebRTC connection establishes.
        greet_text = greeting or text
        if greet_text:

            async def render_greeting_bg():
                """Background task to render greeting while WebRTC connects."""
                try:
                    t_prerender = time.time()
                    logger.info(
                        f"🎙️ Background: Rendering greeting '{greet_text[:50]}...'"
                    )

                    target_cm = conv_manager
                    if not target_cm:
                        # One-shot: create temp ConversationManager for rendering (no Realtime needed)
                        target_cm = ConversationManager(
                            openai_client,
                            tts_model,
                            imtalker_streamer,
                            video_track,
                            audio_track,
                            avatar_img,
                            system_prompt=system_prompt,
                            reference_audio=reference_audio,
                            sync_manager=sync_manager,
                            enable_realtime=False,
                            webrtc_session_id=session_id,
                            full_img_rgb=full_img_rgb_for_paste,
                            crop_coords=crop_coords_for_paste,
                        )

                    pre_rendered_pairs = await target_cm.pre_render_to_buffer(
                        greet_text
                    )
                    render_time = time.time() - t_prerender
                    logger.info(
                        f"✅ Greeting rendered in background: {len(pre_rendered_pairs)} pairs in {render_time:.2f}s"
                    )

                    # Store in session meta for the connected handler
                    meta = app.state.active_session_meta.get(session_id)
                    if meta:
                        meta["pre_rendered_pairs"] = pre_rendered_pairs
                        meta["greeting_cm"] = target_cm
                        meta["greeting_render_time"] = render_time
                        logger.info(
                            f"Session {session_id}: Greeting ready for playback"
                        )
                    else:
                        logger.warning(
                            f"Session {session_id} not found after greeting render (disconnected?)"
                        )
                        if not conv_mode and target_cm:
                            target_cm.cleanup()

                except Exception as e:
                    logger.error(f"Background greeting render failed: {e}")
                    traceback.print_exc()
                    # Store error so connection handler knows greeting failed
                    meta = app.state.active_session_meta.get(session_id)
                    if meta:
                        meta["greeting_error"] = str(e)

            # Start greeting render in background (non-blocking)
            asyncio.create_task(render_greeting_bg())
            logger.info(f"✅ SDP answer ready. Greeting rendering in background...")

        # Handle offer
        offer = RTCSessionDescription(sdp=sdp, type="offer")  # default to offer
        await pc.setRemoteDescription(offer)

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        logger.info(
            f"Offer processing total time: {(time.time() - t_start)*1000:.2f}ms"
        )

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id,
            "oshara_session_id": oshara_session_id,
            "status": "connecting",
            "greeting": greeting if greeting else text,
            "conv_mode": conv_mode,
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
                sdpMLineIndex=candidate["sdpMLineIndex"],
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
            return {
                "status": "ok",
                "detail": f"Session {session_id} already cleaned up.",
            }

        await _cleanup_webrtc_session(session_id, reason="client-disconnect")

        return {
            "status": "ok",
            "session_id": session_id,
            "detail": "Session disconnected and cleaned up.",
        }
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
            "detail": "No active WebRTC session.",
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
    Accumulates all chunks and trims trailing silence/noise before yielding.
    """
    all_chunks = []
    for audio_chunk, metrics in tts_stream:
        # audio_chunk is [1, T] tensor
        if isinstance(audio_chunk, torch.Tensor):
            all_chunks.append(audio_chunk.cpu())
        else:
            all_chunks.append(audio_chunk)

    if all_chunks:
        full_audio = torch.cat(all_chunks, dim=-1)
        audio_np = full_audio.numpy().flatten()
        trimmed = trim_trailing_tts_audio(audio_np, sr=16000)
        yield torch.from_numpy(trimmed).float().unsqueeze(0)


def save_video(frames, fps, output_path):
    if not frames:
        print("No frames to save.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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


def trim_trailing_tts_audio(audio_np: np.ndarray, sr: int = 24000,
                             silence_threshold: float = 0.015,
                             min_trailing_silence_ms: int = 120,
                             keep_tail_ms: int = 40) -> np.ndarray:
    """Trim trailing silence and hallucinated noise from TTS output.

    Chatterbox often generates extra tokens after the text is finished,
    producing trailing silence followed by hallucinated/garbage audio.
    This detects the end of real speech and trims everything after it.

    Strategy: scan backwards. If we find a silence gap >= min_trailing_silence_ms,
    check if what follows is short (< 500ms). If so, it's likely hallucinated —
    cut at the silence gap. Otherwise, just trim trailing silence.
    """
    if audio_np is None or len(audio_np) == 0:
        return audio_np

    window_ms = 25
    window_samples = int(window_ms / 1000.0 * sr)
    if window_samples < 1:
        return audio_np

    n = len(audio_np)
    num_windows = n // window_samples
    if num_windows < 3:
        return audio_np

    # Compute RMS energy per window
    rms_values = np.array([
        np.sqrt(np.mean(audio_np[i * window_samples:(i + 1) * window_samples] ** 2))
        for i in range(num_windows)
    ])

    is_voiced = rms_values >= silence_threshold
    min_silence_windows = max(1, int(min_trailing_silence_ms / window_ms))

    # --- Strategy 1: Find silence gap followed by short hallucinated burst ---
    max_hallucination_windows = int(500 / window_ms)  # 500ms max hallucination
    i = num_windows - 1

    # Skip any trailing silence
    while i >= 0 and not is_voiced[i]:
        i -= 1

    # Check if there's a short voiced segment at the end (hallucination)
    halluc_end = i
    while i >= 0 and is_voiced[i]:
        i -= 1
    halluc_start = i + 1
    halluc_len = halluc_end - halluc_start + 1

    if halluc_len > 0 and halluc_len <= max_hallucination_windows and halluc_start > 0:
        # Check if there's a silence gap before this burst
        gap_end = halluc_start - 1
        gap_start = gap_end
        while gap_start >= 0 and not is_voiced[gap_start]:
            gap_start -= 1
        gap_start += 1
        gap_len = gap_end - gap_start + 1

        if gap_len >= min_silence_windows:
            # Found: [real speech] [silence gap] [short hallucinated burst]
            keep_tail_samples = int(keep_tail_ms / 1000.0 * sr)
            cut_point = gap_start * window_samples + keep_tail_samples
            cut_point = min(cut_point, n)

            if cut_point >= n * 0.3:  # Safety: don't cut more than 70%
                trimmed = audio_np[:cut_point].copy()
                fade_samples = min(int(0.02 * sr), len(trimmed) // 4)
                if fade_samples > 0:
                    trimmed[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                trimmed_ms = (n - cut_point) / sr * 1000
                if trimmed_ms > 10:
                    print(f"  ✂️  Trimmed {trimmed_ms:.0f}ms trailing silence+hallucination from TTS")
                return trimmed

    # --- Strategy 2: Simple trailing silence trim ---
    silence_count = 0
    last_voiced_window = num_windows - 1
    for j in range(num_windows - 1, -1, -1):
        if not is_voiced[j]:
            silence_count += 1
        else:
            last_voiced_window = j
            break

    if silence_count < min_silence_windows:
        return audio_np

    keep_tail_samples = int(keep_tail_ms / 1000.0 * sr)
    cut_point = (last_voiced_window + 1) * window_samples + keep_tail_samples
    cut_point = min(cut_point, n)

    if cut_point < n * 0.5:
        return audio_np

    trimmed = audio_np[:cut_point].copy()

    fade_samples = min(int(0.02 * sr), len(trimmed) // 4)
    if fade_samples > 0:
        trimmed[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    trimmed_ms = (n - cut_point) / sr * 1000
    if trimmed_ms > 10:
        print(f"  ✂️  Trimmed {trimmed_ms:.0f}ms trailing silence/noise from TTS output")

    return trimmed


def apply_audio_chunk_fades(
    audio_np: np.ndarray, sr: int = 24000, fade_ms: float = 10.0
) -> np.ndarray:
    """Apply DC-offset removal + fade-in/fade-out to an audio chunk.

    Args:
        audio_np: float32 numpy audio array (1D mono or 2D multi-channel)
        sr: sample rate
        fade_ms: fade duration in milliseconds
    Returns:
        Audio with DC removed and fades applied (copy)
    """
    if audio_np.size == 0:
        return audio_np
    audio = audio_np.copy().astype(np.float32)
    # Remove DC offset — prevents step-discontinuity clicks between chunks
    if audio.ndim == 1:
        audio -= np.mean(audio)
    else:
        audio -= np.mean(audio, axis=0, keepdims=True)
    # Number of samples along the time axis
    n_samples = audio.shape[0]
    fade_samples = int(fade_ms / 1000.0 * sr)
    fade_samples = min(fade_samples, n_samples // 4)  # Don't fade more than 25%
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        # Handle both 1D (mono) and 2D (multi-channel) arrays
        if audio.ndim == 2:
            fade_in = fade_in[:, np.newaxis]
            fade_out = fade_out[:, np.newaxis]
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    return audio


class AudioChunkCrossfader:
    """Stateful crossfader that blends consecutive audio chunks at their boundaries.

    Instead of independently fading each chunk (which creates amplitude dips
    at boundaries), this keeps the tail of each chunk and crossfades it with
    the head of the next chunk.  Only the very first chunk gets a fade-in and
    the very last chunk gets a fade-out; intermediate boundaries are seamless.
    """

    def __init__(self, sr: int = 24000, crossfade_ms: float = 15.0):
        self.sr = sr
        self.crossfade_samples = int(crossfade_ms / 1000.0 * sr)
        self._prev_tail = None  # float32 tail AFTER fade-out (for continuity)
        self._prev_tail_raw = None  # float32 tail BEFORE fade-out (for crossfade)
        self._is_first = True

    def process(self, audio_np: np.ndarray) -> np.ndarray:
        """Process one chunk.  Returns audio with DC removed + crossfade applied.

        Call once per sentence/text chunk in order.
        """
        if audio_np.size == 0:
            return audio_np

        audio = audio_np.copy().astype(np.float32)
        # Remove DC offset
        audio -= np.mean(audio)

        xf = min(self.crossfade_samples, len(audio) // 4)

        if self._prev_tail_raw is not None and xf > 0:
            # Inter-chunk crossfade: blend previous chunk's ending with this chunk's beginning.
            # `_prev_tail_raw` was saved BEFORE fade-out, so it's at real amplitude.
            # We use it to create a smooth bridge: old audio fades out while new fades in.
            tail = self._prev_tail_raw
            blend = min(xf, len(tail), len(audio))
            if blend > 0:
                fo = np.linspace(1.0, 0.0, blend, dtype=np.float32)
                fi = np.linspace(0.0, 1.0, blend, dtype=np.float32)
                audio[:blend] = tail[-blend:] * fo + audio[:blend] * fi
        elif self._is_first and xf > 0:
            # Very first chunk — fade-in from silence
            audio[:xf] *= np.linspace(0.0, 1.0, xf, dtype=np.float32)
            self._is_first = False

        # Save tail BEFORE fade-out (raw) for crossfade with next chunk
        tail_len = min(xf, len(audio))
        if tail_len > 0:
            self._prev_tail_raw = audio[-tail_len:].copy()
            # Apply gentler fade-out (half-cosine) to rendered audio.
            # Linear fade to zero creates audible dips when chunks arrive
            # in quick succession. Cosine preserves more energy in the middle.
            t = np.linspace(0, np.pi / 2, tail_len, dtype=np.float32)
            audio[-tail_len:] *= np.cos(t)

        return audio


def process_frame_tensor(frame_tensor, format="BGR"):
    # frame_tensor: [1, 1, 3, H, W] or [1, 3, H, W] or [3, H, W]
    while frame_tensor.dim() > 3:
        frame_tensor = frame_tensor.squeeze(0)

    # [3, H, W] -> [H, W, 3] -> [0, 255] uint8
    # Fast path: model outputs [0, 1] range with FP32 (no NaN/range check needed)
    frame_np = (
        (frame_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )

    if format == "RGB":
        return frame_np

    # RGB to BGR for OpenCV
    return cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(
        description="Text to Video Streaming with Chatterbox and IMTalker"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a streaming test.",
        help="Text to speak",
    )
    parser.add_argument(
        "--avatar", type=str, default="assets/source_1.png", help="Path to avatar image"
    )
    parser.add_argument(
        "--output", type=str, default="output_stream.mp4", help="Output video path"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

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
        avatar_img = Image.new("RGB", (512, 512), color="red")
    else:
        print(f"Using avatar: {avatar_path}")
        avatar_img = Image.open(avatar_path).convert("RGB")

    print(f"\nProcessing Text: '{args.text}'")

    # 3. Create TTS Stream Iterator
    # Chatterbox stream yields (audio_chunk, metrics)
    tts_stream = tts_model.generate_stream(
        text=args.text, chunk_size=20, print_metrics=False  # tokens
    )

    audio_iterator = chatterbox_adapter(tts_stream)

    # 4. Stream Video
    print("Starting Video Streaming...")

    all_frames = []
    total_latency_start = time.time()

    try:
        chunk_idx = 0
        for video_frames, metrics in imtalker_streamer.generate_stream(
            avatar_img, audio_iterator
        ):
            chunk_idx += 1
            print(
                f"Chunk {chunk_idx}: Generated {len(video_frames)} frames. "
                f"RTF: {metrics['rtf']:.2f}"
            )

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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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
    crop_scale: float = Form(
        0.8,
        description="Crop scale factor (0.5=tight, 0.8=default, 1.2=loose)",
        ge=0.3,
        le=2.0,
    ),
    output_size: Optional[int] = Form(
        None, description="Output size in pixels (square)", ge=64, le=2048
    ),
    format: str = Form("jpeg", description="Output format (jpeg, png, webp)"),
):
    """
    Crop image centered on detected face with configurable padding.
    """
    if not hasattr(app.state, "inference_agent"):
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Validate format
    format = format.lower()
    if format not in ["jpeg", "jpg", "png", "webp"]:
        raise HTTPException(
            status_code=400, detail="Invalid format. Use jpeg, png, or webp"
        )

    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

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
                cropped_img = cropped_img.resize(
                    (output_size, output_size), Image.LANCZOS
                )
        finally:
            # Restore original crop_scale
            dp.crop_scale = original_crop_scale

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        if format in ["jpeg", "jpg"]:
            cropped_img.save(img_byte_arr, format="JPEG", quality=95)
            media_type = "image/jpeg"
        elif format == "png":
            cropped_img.save(img_byte_arr, format="PNG")
            media_type = "image/png"
        elif format == "webp":
            cropped_img.save(img_byte_arr, format="WEBP", quality=95)
            media_type = "image/webp"
        else:
            # Fallback (should be caught by validation)
            cropped_img.save(img_byte_arr, format="JPEG", quality=95)
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
    session_id: Optional[str] = Form(None),
    voice_gender: str = Form(
        "male", description="Voice gender: 'male' or 'female' (for Nepali TTS)"
    ),
):
    """Start or resume a progressive HLS streaming session"""
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Check if session already exists
    if session_id in app.state.hls_sessions:
        session = app.state.hls_sessions[session_id]
        if not session.is_complete:
            return JSONResponse(
                {
                    "session_id": session_id,
                    "status": "already_running",
                    "playlist_url": f"/hls/{session_id}/playlist.m3u8",
                }
            )

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
        is_complete=False,
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
        is_final_chunk=True,
    )

    return {
        "session_id": session_id,
        "status": "started",
        "playlist_url": f"/hls/{session_id}/playlist.m3u8",
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
        "playlist_url": f"/hls/{session_id}/playlist.m3u8" if playlist_ready else None,
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
            with open(playlist_path, "r") as f:
                content = f.read()
                if "#EXTINF" in content:
                    return FileResponse(
                        playlist_path,
                        media_type="application/vnd.apple.mpegurl",
                        headers={"Cache-Control": "no-cache"},
                    )
        except Exception:
            pass

        await asyncio.sleep(check_interval)
        elapsed += check_interval

    raise HTTPException(
        status_code=202, detail="Playlist still waiting for first segment"
    )


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
    voice_gender: str = Form(
        "male", description="Voice gender: 'male' or 'female' (for Nepali TTS)"
    ),
):
    sid = session_id or str(uuid.uuid4())
    img_data = await image.read()
    img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")

    buffer = AudioChunkBuffer(session_id=sid, chunk_duration=chunk_duration)
    hls_gen = ProgressiveHLSGenerator(
        session_id=sid, segment_duration=int(chunk_duration)
    )

    pose_arr = json.loads(pose_style) if pose_style else None
    gaze_arr = json.loads(gaze_style) if gaze_style else None

    session = ChunkedStreamSession(
        session_id=sid,
        image_pil=img_pil,
        audio_buffer=buffer,
        hls_generator=hls_gen,
        parameters={
            "crop": crop,
            "seed": seed,
            "nfe": nfe,
            "cfg_scale": cfg_scale,
            "pose_style": pose_arr,
            "gaze_style": gaze_arr,
        },
    )
    app.state.chunked_sessions[sid] = session
    return {
        "status": "success",
        "session_id": sid,
        "hls_url": f"/api/hls/{sid}/playlist.m3u8",
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
        raise HTTPException(
            status_code=404, detail="Session not found. Call init first."
        )
    session = app.state.chunked_sessions[session_id]

    # 2. Add audio chunk to buffer
    audio_bytes = await audio_chunk.read()
    # Assume 16-bit PCM if not specified, but let's try to detect if it's a wav file
    if audio_chunk.filename.endswith(".wav"):
        with io.BytesIO(audio_bytes) as b:
            import wave

            with wave.open(b, "rb") as w:
                sr = w.getframerate()
                # Read all frames and convert to numpy
                frames = w.readframes(w.getnframes())
                audio_np = (
                    np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                )
    else:
        # Generic float32 or int16
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        sr = 16000  # Default

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
                                    gaze_style=session.parameters.get("gaze_style"),
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
                        gaze_style=session.parameters.get("gaze_style"),
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
        "playlist_url": f"/hls/{session_id}/playlist.m3u8",
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
        "created_at": session.created_at,
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
    gaze_style: Optional[str] = Form(None),
    use_trt: bool = Form(False, description="Use TensorRT acceleration if available"),
    voice_gender: str = Form(
        "male", description="Voice gender: 'male' or 'female' (for Nepali TTS)"
    ),
):
    temp_dir = Path(tempfile.mkdtemp())
    try:
        img_path = temp_dir / "source.png"
        aud_path = temp_dir / "audio.wav"
        with img_path.open("wb") as f:
            f.write(await image.read())
        with aud_path.open("wb") as f:
            f.write(await audio.read())

        img_pil = Image.open(img_path).convert("RGB")

        dp = app.state.inference_agent.data_processor
        original_crop_scale = dp.crop_scale
        dp.crop_scale = crop_scale

        pose_arr = json.loads(pose_style) if pose_style else None
        gaze_arr = json.loads(gaze_style) if gaze_style else None

        try:
            # Use TRT if requested and available
            trt_handler = getattr(app.state, "trt_handler", None)
            if use_trt and trt_handler and trt_handler.is_available():
                logger.info("[/api/audio-driven] Using TensorRT-accelerated inference")
                out_path = await asyncio.get_event_loop().run_in_executor(
                    None,
                    app.state.inference_agent.run_trt_inference,
                    trt_handler,
                    img_pil,
                    str(aud_path),
                    crop,
                    seed,
                    nfe,
                    cfg_scale,
                )
            else:
                if use_trt:
                    logger.warning(
                        "[/api/audio-driven] TRT requested but not available, falling back to PyTorch"
                    )
                out_path = app.state.inference_agent.run_audio_inference(
                    img_pil,
                    str(aud_path),
                    crop,
                    seed,
                    nfe,
                    cfg_scale,
                    pose_style=pose_arr,
                    gaze_style=gaze_arr,
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
    crop_scale: float = Form(0.8),  # Added to match VideoService
):
    temp_dir = Path(tempfile.mkdtemp())
    try:
        img_path = temp_dir / "source.png"
        vid_path = temp_dir / "driving.mp4"
        with img_path.open("wb") as f:
            f.write(await source_image.read())
        with vid_path.open("wb") as f:
            f.write(await driving_video.read())

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


@app.post("/api/trt-inference")
async def trt_inference(
    image: UploadFile = File(..., description="Source image (PNG/JPG)"),
    audio: UploadFile = File(..., description="Driving audio (WAV/MP3)"),
    crop: bool = Form(True, description="Auto crop face"),
    seed: int = Form(42, description="Random seed"),
    nfe: int = Form(10, description="Steps"),
    cfg_scale: float = Form(2.0, description="CFG Scale"),
):
    """
    Run inference using the optimized TensorRT hybrid pipeline.

    Uses TensorRT engines for Source Encoder and Audio Renderer,
    while running the Motion Generator in PyTorch. Offers significantly faster
    inference speeds (up to 2x) compared to the standard PyTorch pipeline.

    Requires:
    - TensorRT installed
    - Built TRT engines in ./trt_engines/
    """
    trt_handler = getattr(app.state, "trt_handler", None)
    if trt_handler is None or not trt_handler.is_available():
        raise HTTPException(
            status_code=503,
            detail="TensorRT Inference not available. Check server logs to see if engines were loaded correctly.",
        )

    temp_dir = Path(tempfile.mkdtemp())
    try:
        img_path = temp_dir / "source.png"
        aud_path = temp_dir / "audio.wav"
        with img_path.open("wb") as f:
            f.write(await image.read())
        with aud_path.open("wb") as f:
            f.write(await audio.read())

        img_pil = Image.open(img_path).convert("RGB")

        logger.info(f"[TRT API] Starting TensorRT inference")

        # Run in threadpool to avoid blocking event loop
        video_path = await asyncio.get_event_loop().run_in_executor(
            None,
            trt_handler.run_inference,
            img_pil,
            str(aud_path),
            crop,
            seed,
            nfe,
            cfg_scale,
        )

        logger.info(f"[TRT API] Success! Video saved to {video_path}")
        return FileResponse(video_path, media_type="video/mp4", filename="output.mp4")

    except Exception as e:
        logger.error(f"[TRT API] Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
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
        return FileResponse(
            output_video, media_type="video/mp4", filename=f"{session_id}.mp4"
        )

    try:
        cmd = [
            "ffmpeg",
            "-i",
            str(playlist_path),
            "-c",
            "copy",
            "-bsf:a",
            "aac_adtstoasc",
            "-y",
            str(output_video),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(
                status_code=500, detail=f"FFmpeg merge failed: {result.stderr}"
            )

        return FileResponse(
            output_video, media_type="video/mp4", filename=f"{session_id}.mp4"
        )
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
    cfg_scale: float = 3.0,
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
            frame_np = (
                np.clip(frame_tensor.permute(1, 2, 0).cpu().numpy(), 0, 1) * 255
            ).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", frame_bgr)
            frame_base64 = base64.b64encode(buffer).decode()
            yield f"data: {frame_base64}\n\n"
            await asyncio.sleep(0.01)  # Small sleep to avoid blocking

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/realtime-mjpeg")
async def realtime_mjpeg(
    session_id: str,
    audio_path: Optional[str] = None,
    crop: bool = True,
    seed: int = 42,
    nfe: int = 10,
    cfg_scale: float = 3.0,
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
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"JPEG_DATA" + b"\r\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(
        mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


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
        results.append(
            {
                "filename": path.name,
                "size": path.stat().st_size,
                "created_at": path.stat().st_mtime,
            }
        )
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
    repetition_penalty: float = Form(1.2),
    voice_gender: str = Form(
        "male", description="Voice gender: 'male' or 'female' (for Nepali TTS)"
    ),
):
    """
    Generate speech from text with optional voice cloning.
    Splits long text into segments to avoid CUDA positional encoding overflow.
    Uses a lock to prevent concurrent GPU access and aggressive memory cleanup.

    Automatically detects Nepali language and uses Sherpa TTS for better quality.
    """
    # Check if text is Nepali and use Sherpa TTS if available
    if SHERPA_AVAILABLE and detect_nepali(text):
        logger.info(
            f"Detected Nepali text, using Sherpa TTS ({voice_gender} voice): '{text[:50]}...'"
        )
        try:
            # Generate audio using Sherpa TTS with gender selection
            from sherpa_tts_wrapper import get_sherpa_tts
            import soundfile as sf

            sherpa_tts = get_sherpa_tts(voice=voice_gender)
            audio_samples = sherpa_tts.generate(text, speed=1.0)
            sample_rate = sherpa_tts.sample_rate

            # Convert to WAV bytes
            byte_io = io.BytesIO()
            with wave.open(byte_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)

                # Normalize and convert to int16
                audio_np = np.array(audio_samples, dtype=np.float32)
                peak = np.max(np.abs(audio_np))
                if peak > 1.0:
                    audio_np = audio_np / peak
                audio_int16 = (audio_np * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

            byte_io.seek(0)
            logger.info(
                f"Sherpa TTS generated {len(audio_samples)} samples at {sample_rate}Hz"
            )
            return StreamingResponse(byte_io, media_type="audio/wav")
        except Exception as e:
            logger.error(f"Sherpa TTS failed, falling back to STTS: {e}")
            # Fall through to use STTS server

    # Use STTS server for non-Nepali languages or as fallback
    # Acquire the TTS lock — only one generation at a time to prevent CUDA OOM
    async with app.state.tts_lock:
        try:
            # 0. Pre-flight: free any stale GPU memory before we start
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
                logger.info(
                    f"TTS API: Free GPU memory before generation: {free_mem:.2f} GB"
                )

            # 1. Prepare conditionals if reference audio is provided
            if reference_audio:
                temp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                try:
                    content = await reference_audio.read()
                    temp_ref.write(content)
                    temp_ref.close()

                    def _prepare_conds():
                        with torch.inference_mode():
                            app.state.tts_model.prepare_conditionals(
                                wav_fpath=temp_ref.name, exaggeration=exaggeration
                            )

                    await asyncio.to_thread(_prepare_conds)
                finally:
                    if os.path.exists(temp_ref.name):
                        os.remove(temp_ref.name)

            # 2. Split long text into safe segments to avoid positional encoding overflow
            # Chatterbox's positional encoding table has a fixed size; long texts cause
            # CUDA indexSelectLargeIndex assertion failures.
            # Reduced from 200 → 150 chars to lower peak GPU memory per segment.
            MAX_SEGMENT_CHARS = 150
            text_segments = split_text_semantically(
                text, min_chunk_length=20, max_chunk_length=MAX_SEGMENT_CHARS
            )
            logger.info(
                f"TTS API: Generating {len(text_segments)} segments from {len(text)} chars"
            )

            # 3. Generate audio for each segment and concatenate
            sr = getattr(app.state.tts_model, "sr", 24000)

            def generate_segments():
                """Run all TTS segments sequentially with aggressive memory cleanup."""
                audio_pieces = []
                for i, segment in enumerate(text_segments):
                    logger.info(
                        f"TTS API: Segment {i+1}/{len(text_segments)} ({len(segment)} chars): '{segment[:50]}...'"
                    )

                    # Synchronize before generation to catch any async CUDA errors early
                    # This surfaces any latent async errors from a previous call right here
                    # rather than letting them appear inside the model as misleading stacktraces.
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError as sync_err:
                            logger.error(
                                f"TTS API: CUDA context corrupted before segment {i+1}: {sync_err}"
                            )
                            logger.info(
                                "TTS API: Attempting CUDA context recovery via device reset..."
                            )
                            torch.cuda.empty_cache()
                            gc.collect()
                            # Reset the device to clear the corrupted context
                            torch.cuda.reset_peak_memory_stats()
                            # Re-synchronize to verify recovery
                            torch.cuda.synchronize()
                            logger.info("TTS API: CUDA context recovered")

                    # Generate with retry: if the model produces empty/invalid tokens
                    # (e.g. all tokens >= SPEECH_VOCAB_SIZE), retry with lower temperature
                    max_retries = 2
                    audio_np = None
                    for attempt in range(max_retries):
                        try:
                            with torch.inference_mode():
                                gen_temp = (
                                    temperature
                                    if attempt == 0
                                    else max(0.3, temperature * 0.5)
                                )
                                audio_tensor = app.state.tts_model.generate(
                                    text=segment,
                                    exaggeration=exaggeration,
                                    temperature=gen_temp,
                                )
                                # Immediately move to CPU numpy and discard GPU tensor
                                audio_np = audio_tensor.squeeze(0).cpu().numpy()
                                del audio_tensor
                            break  # success
                        except RuntimeError as gen_err:
                            err_msg = str(gen_err).lower()
                            is_cuda_error = (
                                "cuda" in err_msg or "device-side assert" in err_msg
                            )
                            logger.error(
                                f"TTS API: Segment {i+1} attempt {attempt+1} failed: {gen_err}"
                            )
                            # Clean up after failed attempt
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # Clear the T3 patched_model that may hold corrupted state
                            tts = app.state.tts_model
                            if hasattr(tts, "t3") and hasattr(tts.t3, "patched_model"):
                                del tts.t3.patched_model
                                tts.t3.compiled = False
                            if attempt == max_retries - 1:
                                raise  # exhausted retries

                    if audio_np is not None and len(audio_np) > 0:
                        audio_pieces.append(audio_np)
                    else:
                        logger.warning(
                            f"TTS API: Segment {i+1} produced empty audio, skipping"
                        )

                    # Cleanup between segments
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if not audio_pieces:
                    raise RuntimeError("All TTS segments produced empty audio")

                return (
                    np.concatenate(audio_pieces, axis=-1)
                    if len(audio_pieces) > 1
                    else audio_pieces[0]
                )

            audio_np = await asyncio.to_thread(generate_segments)

            # 4. Convert to WAV bytes
            byte_io = io.BytesIO()
            with wave.open(byte_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sr)

                # Normalize and convert to int16
                peak = np.max(np.abs(audio_np))
                if peak > 1.0:
                    audio_np = audio_np / peak
                audio_int16 = (audio_np * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

            # Final cleanup — release all intermediate data and TTS internal caches
            del audio_np, audio_int16

            # Clear the T3 patched_model and its KV cache that persist on GPU
            # between calls. Setting compiled=False forces a fresh (small) rebuild
            # next call instead of keeping the stale attention-cache-holding wrapper.
            tts = app.state.tts_model
            if hasattr(tts, "t3") and hasattr(tts.t3, "patched_model"):
                del tts.t3.patched_model
                tts.t3.compiled = False

            # Clear the s3gen LRU resampler cache (holds GPU tensors)
            try:
                from chatterbox.models.s3gen.s3gen import get_resampler

                get_resampler.cache_clear()
            except Exception:
                pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_after = torch.cuda.mem_get_info(0)[0] / 1024**3
                logger.info(
                    f"TTS API: Free GPU memory after cleanup: {free_after:.2f} GB"
                )

            byte_io.seek(0)
            return StreamingResponse(byte_io, media_type="audio/wav")

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"TTS Generation CUDA OOM: {str(e)}")
            traceback.print_exc()
            # Emergency cleanup: nuke T3 patched_model + resampler cache + CUDA cache
            tts = app.state.tts_model
            if hasattr(tts, "t3") and hasattr(tts.t3, "patched_model"):
                del tts.t3.patched_model
                tts.t3.compiled = False
            try:
                from chatterbox.models.s3gen.s3gen import get_resampler

                get_resampler.cache_clear()
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            raise HTTPException(
                status_code=503,
                detail="GPU out of memory. Try shorter text or retry in a moment.",
            )
        except Exception as e:
            logger.error(f"TTS Generation failed: {str(e)}")
            traceback.print_exc()
            # Clean up T3 cache + GPU memory on failure
            tts = app.state.tts_model
            if hasattr(tts, "t3") and hasattr(tts.t3, "patched_model"):
                del tts.t3.patched_model
                tts.t3.compiled = False
            try:
                from chatterbox.models.s3gen.s3gen import get_resampler

                get_resampler.cache_clear()
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(
                status_code=500, detail=f"TTS Generation failed: {str(e)}"
            )


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
