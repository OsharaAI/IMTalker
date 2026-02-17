
import argparse
import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import psutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import AppConfig, InferenceAgent
    from trt_handler import TRTInferenceHandler
except ImportError:
    try:
        from IMTalker.app import AppConfig, InferenceAgent
        from IMTalker.trt_handler import TRTInferenceHandler
    except ImportError:
        print("Error: Could not import app or trt_handler. Make sure you are in the IMTalker directory.")
        sys.exit(1)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024 ** 3)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
    else:
        vram_gb = 0
        vram_reserved_gb = 0
    return ram_gb, vram_gb, vram_reserved_gb

def main():
    parser = argparse.ArgumentParser(description="IMTalker TensorRT Inference with Metrics")
    parser.add_argument("--source_path", type=str, required=True, help="Path to source image")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to driving audio")
    parser.add_argument("--save_path", type=str, default="results", help="Directory to save output")
    parser.add_argument("--output_name", type=str, default="output_trt", help="Output filename (without extension)")
    parser.add_argument("--crop", action="store_true", help="Auto crop face")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nfe", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="Classifier-free guidance scale")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--fps", type=int, default=25, help="Output FPS")
    parser.add_argument("--engine_dir", type=str, default="trt_engines", help="Path to TensorRT engines")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print("=" * 50)
    print("IMTalker TensorRT Inference")
    print("=" * 50)
    
    # 1. Initialization
    t_init_start = time.time()
    cfg = AppConfig()
    cfg.device = args.device
    cfg.fps = args.fps
    
    print("Loading PyTorch models (for Motion Generator)...")
    agent = InferenceAgent(cfg)
    agent.opt.fps = args.fps
    
    print(f"Loading TensorRT engines from {args.engine_dir}...")
    trt_handler = TRTInferenceHandler(agent, engine_dir=args.engine_dir)
    
    if not trt_handler.is_available():
        print("Error: TensorRT engines not found. Please build them first using utils/build_engine.py")
        sys.exit(1)
        
    t_init_end = time.time()
    print(f"Initialization Time: {t_init_end - t_init_start:.4f}s")
    
    # 2. Data Loading
    print(f"Loading inputs: {args.source_path} + {args.audio_path}")
    try:
        img_pil = Image.open(args.source_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    ram, vram, vram_res = get_memory_usage()
    print(f"Memory (Start): RAM: {ram:.2f}GB | VRAM: {vram:.2f}GB (Reserved: {vram_res:.2f}GB)")
    
    # 3. Inference
    print("Starting inference...")
    t_inf_start = time.time()


    with torch.inference_mode():
        # Re-implementing logic with metrics:
        t_prep_start = time.time()
        if args.crop:
            img = agent.data_processor.process_img(img_pil)
        else:
            img = img_pil.resize((agent.opt.input_size, agent.opt.input_size))
                
        s_tensor = agent.data_processor.transform(img).unsqueeze(0).to(agent.device)
        t_prep_end = time.time()
        
        # Source Encoder
        print("Running Source Encoder (TRT)...")
        t_enc_start = time.time()
        # Access TRT models from handler
        # Note: handler loads them in __init__
        enc_outputs = trt_handler.source_encoder.infer([s_tensor])
        
        i_r = enc_outputs['i_r']
        t_r = enc_outputs['t_r']
        f_r = [enc_outputs[f'f_r_{i}'] for i in range(6)]
        ma_r = [enc_outputs[f'ma_r_{i}'] for i in range(4)]
        t_enc_end = time.time()
        
        # Audio & Motion
        print("Generating Motion (PyTorch)...")
        t_gen_start = time.time()
        a_tensor = agent.data_processor.process_audio(args.audio_path).unsqueeze(0).to(agent.device)
        
        data = {
            's': s_tensor,
            'a': a_tensor,
            'pose': None,
            'cam': None,
            'gaze': None,
            'ref_x': t_r 
        }
        
        torch.manual_seed(args.seed)
        sample, _ = agent.generator.sample(data, a_cfg_scale=args.cfg_scale, nfe=args.nfe, seed=args.seed)
        
        # Stabilization
        motion_scale = 0.6
        sample = (1.0 - motion_scale) * t_r.unsqueeze(1).repeat(1, sample.shape[1], 1) + motion_scale * sample
        t_gen_end = time.time()
        
        # Rendering
        T = sample.shape[1]
        print(f"Rendering {T} frames (TRT)...")
        t_render_start = time.time()
        
        frames = []
        pbar = tqdm(total=T, unit="frame")
        
        for t in range(T):
            motion_code = sample[:, t, ...] # (1, 32)
            
            trt_inputs = {
                'motion_code': motion_code,
                'i_r': i_r
            }
            for i in range(6): trt_inputs[f'f_r_{i}'] = f_r[i]
            for i in range(4): trt_inputs[f'ma_r_{i}'] = ma_r[i]
            
            out_dict = trt_handler.audio_renderer.infer(trt_inputs)
            out_frame = out_dict['output_image']
            
            frames.append(out_frame.cpu())
            pbar.update(1)
            
        pbar.close()
        t_render_end = time.time()
        t_inf_end = time.time()
    
    # Saving
    print("Muxing video...")
    vid_tensor = torch.stack(frames, dim=1).squeeze(0)
    final_path = os.path.join(args.save_path, f"{args.output_name}.mp4")
    
    temp_video_path = agent.save_video(vid_tensor, args.fps, args.audio_path)
    if os.path.exists(temp_video_path):
        import shutil
        shutil.move(temp_video_path, final_path)
        print(f"Saved to {final_path}")
    
    # Metrics
    prep_time = t_prep_end - t_prep_start
    enc_time = t_enc_end - t_enc_start
    gen_time = t_gen_end - t_gen_start
    render_time = t_render_end - t_render_start
    total_time = t_inf_end - t_inf_start
    
    render_fps = T / render_time if render_time > 0 else 0
    overall_fps = T / total_time if total_time > 0 else 0
    
    ram_end, vram_end, vram_res_end = get_memory_usage()
    
    print("-" * 50)
    print("METRICS REPORT (TRT)")
    print("-" * 50)
    print(f"Total Frames:      {T}")
    print(f"Preprocessing:     {prep_time:.4f}s")
    print(f"Source Encoder:    {enc_time:.4f}s")
    print(f"Motion Generation: {gen_time:.4f}s")
    print(f"Rendering:         {render_time:.4f}s")
    print(f"Total Inference:   {total_time:.4f}s")
    print(f"Rendering FPS:     {render_fps:.2f} fps")
    print(f"Overall FPS:       {overall_fps:.2f} fps")
    print("-" * 50)
    print(f"Memory (End): RAM: {ram_end:.2f}GB | VRAM: {vram_end:.2f}GB")
    print("-" * 50)

if __name__ == "__main__":
    main()
