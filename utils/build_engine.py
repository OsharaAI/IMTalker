import tensorrt as trt
import os
import argparse
import sys

def build_engine(onnx_file_path, engine_file_path, fp16=True):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # Create network definition
    # explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    parser = trt.OnnxParser(network, logger)
    
    config = builder.create_builder_config()
    
    # Enable FP16 if requested
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 Enabled.")
        else:
            print("FP16 not supported on this platform, using FP32.")

    # Parse ONNX
    print(f"Parsing ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Build Engine
    print("Building TensorRT Engine... this may take a few minutes...")
    try:
        # For TensorRT 8.6+ / 10.x
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            print("ERROR: Failed to build serialized network.")
            return None
            
        with open(engine_file_path, "wb") as f:
            f.write(plan)
            
        print(f"Engine saved to {engine_file_path}")
        return True
    except Exception as e:
        print(f"Build failed with exception: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str, default="./onnx_models")
    parser.add_argument("--engine_dir", type=str, default="./trt_engines")
    parser.add_argument("--fp16", type=int, default=1, help="Enable FP16 (1=True, 0=False)")
    args = parser.parse_args()
    
    os.makedirs(args.engine_dir, exist_ok=True)
    
    # Build Source Encoder
    onnx_path = os.path.join(args.onnx_dir, "source_encoder.onnx")
    engine_path = os.path.join(args.engine_dir, "source_encoder.trt")
    if os.path.exists(onnx_path):
        build_engine(onnx_path, engine_path, fp16=bool(args.fp16))
    else:
        print(f"File not found: {onnx_path}")
        
    # Build Audio Renderer
    onnx_path = os.path.join(args.onnx_dir, "audio_renderer.onnx")
    engine_path = os.path.join(args.engine_dir, "audio_renderer.trt")
    if os.path.exists(onnx_path):
        build_engine(onnx_path, engine_path, fp16=bool(args.fp16))
    else:
        print(f"File not found: {onnx_path}")
