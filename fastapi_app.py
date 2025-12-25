"""
FastAPI wrapper for IMTalker video rendering inference server.
Provides REST API endpoints for audio-driven and video-driven talking face generation.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import uuid

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
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


@app.post("/api/v1/audio-driven")
async def audio_driven_inference(
    image: UploadFile = File(..., description="Source image file (PNG, JPG)"),
    audio: UploadFile = File(..., description="Driving audio file (WAV, MP3)"),
    crop: bool = Form(False, description="Auto crop face from image"),
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
