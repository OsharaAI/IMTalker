
import argparse
import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import psutil

# Add parent directory to path to allow importing app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import AppConfig, InferenceAgent
except ImportError:
    # If running from root, app might be available directly
    try:
        from IMTalker.app import AppConfig, InferenceAgent
    except ImportError:
        print("Error: Could not import app. Make sure you are in the IMTalker directory.")
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
    parser = argparse.ArgumentParser(description="IMTalker Audio Inference with Metrics")
    parser.add_argument("--source_path", type=str, required=True, help="Path to source image")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to driving audio")
    parser.add_argument("--save_path", type=str, default="results", help="Directory to save output")
    parser.add_argument("--output_name", type=str, default="output", help="Output filename (without extension)")
    parser.add_argument("--crop", action="store_true", help="Auto crop face")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nfe", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="Classifier-free guidance scale")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--fps", type=int, default=25, help="Output FPS")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print("=" * 50)
    print("IMTalker Audio Inference (PyTorch)")
    print("=" * 50)
    
    # 1. Initialization
    t_init_start = time.time()
    cfg = AppConfig()
    cfg.device = args.device
    cfg.fps = args.fps # Update FPS in config if needed, though agent might use its own
    
    print("Loading models...")
    agent = InferenceAgent(cfg)
    # Ensure agent respects args.fps if it uses it internally
    agent.opt.fps = args.fps 
    
    t_init_end = time.time()
    print(f"Model Load Time: {t_init_end - t_init_start:.4f}s")
    
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
        # 3.1 Preprocess
        t_prep_start = time.time()
        if args.crop:
            img = agent.data_processor.process_img(img_pil)
        else:
            img = img_pil.resize((agent.opt.input_size, agent.opt.input_size))
            
        s_tensor = agent.data_processor.transform(img).unsqueeze(0).to(agent.device)
        a_tensor = agent.data_processor.process_audio(args.audio_path).unsqueeze(0).to(agent.device)
        t_prep_end = time.time()
        
        # 3.2 Motion Generation
        t_gen_start = time.time()
        f_r, g_r = agent.renderer.dense_feature_encoder(s_tensor)
        t_lat = agent.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple): t_lat = t_lat[0]
        
        data = {
            's': s_tensor,
            'a': a_tensor,
            'pose': None,
            'cam': None,
            'gaze': None,
            'ref_x': t_lat
        }
        
        torch.manual_seed(args.seed)
        
        # Calculate Expected Frames based on audio
        print("Generating Motion...")
        sample, _ = agent.generator.sample(data, a_cfg_scale=args.cfg_scale, nfe=args.nfe, seed=args.seed)
        
        # Stabilization
        motion_scale = 0.6
        sample = (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) + motion_scale * sample
        t_gen_end = time.time()
        
        # 3.3 Rendering
        T = sample.shape[1]
        print(f"Rendering {T} frames...")
        
        t_render_start = time.time()
        
        # Decode reference motion once
        ta_r = agent.renderer.adapt(t_lat, g_r)
        m_r = agent.renderer.latent_token_decoder(ta_r)
        
        frames = []
        
        pbar = tqdm(total=T, unit="frame")
        
        for t in range(T):
            t_frame_start = time.time()
            
            ta_c = agent.renderer.adapt(sample[:, t, ...], g_r)
            m_c = agent.renderer.latent_token_decoder(ta_c)
            out_frame = agent.renderer.decode(m_c, m_r, f_r)
            
            frames.append(out_frame.cpu())
            
            pbar.update(1)
            
        pbar.close()
        t_render_end = time.time()
        t_inf_end = time.time()
    
    # 4. Save Video
    print("Muxing video...")
    vid_tensor = torch.stack(frames, dim=1).squeeze(0)
    
    final_path = os.path.join(args.save_path, f"{args.output_name}.mp4")
    
    temp_video_path = agent.save_video(vid_tensor, args.fps, args.audio_path)
    if os.path.exists(temp_video_path):
        import shutil
        shutil.move(temp_video_path, final_path)
        print(f"Saved to {final_path}")
    else:
        print("Error saving video.")

    # 5. Metrics Report
    prep_time = t_prep_end - t_prep_start
    gen_time = t_gen_end - t_gen_start
    render_time = t_render_end - t_render_start
    total_time = t_inf_end - t_inf_start
    
    render_fps = T / render_time if render_time > 0 else 0
    overall_fps = T / total_time if total_time > 0 else 0
    
    ram_end, vram_end, vram_res_end = get_memory_usage()
    
    print("-" * 50)
    print("METRICS REPORT")
    print("-" * 50)
    print(f"Total Frames:      {T}")
    print(f"Preprocessing:     {prep_time:.4f}s")
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
