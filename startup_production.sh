#!/bin/bash
set -e

# Production Startup Script for IMTalker with TensorRT

ONNX_DIR="./onnx_models"
ENGINE_DIR="./trt_engines"

# 1. Ensure Checkpoints are Present
echo "[INFO] Verifying model checkpoints..."
python3 -c "from app import ensure_checkpoints; ensure_checkpoints()"

# 2. Check if TensorRT Engines need to be built
if [ ! -f "$ENGINE_DIR/source_encoder.trt" ] || [ ! -f "$ENGINE_DIR/audio_renderer.trt" ]; then
    echo "[INFO] TensorRT engines missing. Starting build pipeline..."
    
    # Create output directories
    mkdir -p $ONNX_DIR
    mkdir -p $ENGINE_DIR

    # Export to ONNX
    echo "[INFO] 1/2 Exporting PyTorch models to ONNX..."
    python3 utils/export_onnx.py \
        --renderer_path ./checkpoints/renderer.ckpt \
        --output_dir $ONNX_DIR \
        --input_size 512

    # Build TensorRT Engines
    echo "[INFO] 2/2 Building TensorRT Engines (FP16 enabled)..."
    python3 utils/build_engine.py \
        --onnx_dir $ONNX_DIR \
        --engine_dir $ENGINE_DIR \
        --fp16 1

    echo "[SUCCESS] TensorRT Engines built successfully!"
    
    # Optional: Clean up ONNX files to save space
    # rm -rf $ONNX_DIR
else
    echo "[INFO] TensorRT engines found. Skipping build."
fi

# 3. Start FastAPI Server
echo "[INFO] Starting FastAPI server..."
# Using --workers 1 to avoid excessive VRAM usage per worker
uvicorn fastapi_app:app --host 0.0.0.0 --port 9999 --workers 1
