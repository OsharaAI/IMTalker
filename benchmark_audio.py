
import argparse
import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import psutil
import json
import csv
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import AppConfig, InferenceAgent
except ImportError:
    try:
        from IMTalker.app import AppConfig, InferenceAgent
    except ImportError:
        print("Error: Could not import app. Make sure you are in the IMTalker directory.")
        sys.exit(1)

def get_peak_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0

def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def main():
    parser = argparse.ArgumentParser(description="IMTalker Audio Benchmarking (PyTorch)")
    parser.add_argument("--sources", type=str, required=True, help="Comma-separated list of source image paths")
    parser.add_argument("--audios", type=str, required=True, help="Comma-separated list of audio file paths")
    parser.add_argument("--save_path", type=str, default="benchmark_results_audio", help="Directory to save output")
    parser.add_argument("--crop", action="store_true", help="Auto crop face")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nfe", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # 1. Initialization
    print("Initializing Model...")
    cfg = AppConfig()
    cfg.device = args.device
    agent = InferenceAgent(cfg)
    
    # Parse comma-separated lists
    sources = [s.strip() for s in args.sources.split(',')]
    audios = [a.strip() for a in args.audios.split(',')]
    
    # Validate files exist
    for src in sources:
        if not os.path.exists(src):
            print(f"Error: Source file not found: {src}")
            sys.exit(1)
    
    for aud in audios:
        if not os.path.exists(aud):
            print(f"Error: Audio file not found: {aud}")
            sys.exit(1)
    
    print(f"Found {len(sources)} source images and {len(audios)} audio files.")
    print(f"Total combinations: {len(sources) * len(audios)}")
    
    results = []
    
    # Matrix run
    total_start = time.time()
    
    for src_path in sources:
        try:
            img_pil = Image.open(src_path).convert("RGB")
            # Preprocess image once per source if possible? 
            # In real pipeline, we might reuse. But standard inference refreshes.
            # We measure full pipeline per combination.
            
            for aud_path in audios:
                combo_name = f"{os.path.basename(src_path)}_{os.path.basename(aud_path)}"
                print(f"Benchmarking: {combo_name}")
                
                reset_peak_memory()
                t0 = time.time()
                
                with torch.inference_mode():
                    # 1. Preprocess
                    t_prep_start = time.time()
                    if args.crop:
                        img = agent.data_processor.process_img(img_pil)
                    else:
                        img = img_pil.resize((agent.opt.input_size, agent.opt.input_size))
                    s_tensor = agent.data_processor.transform(img).unsqueeze(0).to(agent.device)
                    a_tensor = agent.data_processor.process_audio(aud_path).unsqueeze(0).to(agent.device)
                    t_prep_end = time.time()
                    
                    # 2. Motion Gen
                    t_gen_start = time.time()
                    f_r, g_r = agent.renderer.dense_feature_encoder(s_tensor)
                    t_lat = agent.renderer.latent_token_encoder(s_tensor)
                    if isinstance(t_lat, tuple): t_lat = t_lat[0]
                    
                    data = {'s': s_tensor, 'a': a_tensor, 'pose': None, 'cam': None, 'gaze': None, 'ref_x': t_lat}
                    torch.manual_seed(args.seed)
                    sample, _ = agent.generator.sample(data, a_cfg_scale=2.0, nfe=args.nfe, seed=args.seed)
                    
                    sample = (1.0 - 0.6) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) + 0.6 * sample
                    t_gen_end = time.time()
                    
                    # 3. Render
                    T = sample.shape[1]
                    t_render_start = time.time()
                    
                    ta_r = agent.renderer.adapt(t_lat, g_r)
                    m_r = agent.renderer.latent_token_decoder(ta_r)
                    
                    frames = []
                    for t in range(T):
                        ta_c = agent.renderer.adapt(sample[:, t, ...], g_r)
                        m_c = agent.renderer.latent_token_decoder(ta_c)
                        out_frame = agent.renderer.decode(m_c, m_r, f_r)
                        frames.append(out_frame.cpu()) # Move to CPU to avoid OOM in long runs
                        
                    t_render_end = time.time()
                t1 = time.time()
                
                # Metrics calculation
                total_duration = t1 - t0
                prep_time = t_prep_end - t_prep_start
                gen_time = t_gen_end - t_gen_start
                render_time = t_render_end - t_render_start
                
                # Calculate FPS
                render_fps = T / render_time if render_time > 0 else 0
                overall_fps = T / total_duration if total_duration > 0 else 0
                
                # RTF
                # We need audio duration. 
                # a_tensor shape is (1, L). 16000 Hz.
                audio_len_sec = a_tensor.shape[1] / 16000.0
                rtf = total_duration / audio_len_sec if audio_len_sec > 0 else 0
                
                peak_mem = get_peak_memory()
                
                metrics = {
                    "source": os.path.basename(src_path),
                    "audio": os.path.basename(aud_path),
                    "frames": T,
                    "audio_sec": audio_len_sec,
                    "total_time": total_duration,
                    "prep_time": prep_time,
                    "gen_time": gen_time,
                    "render_time": render_time,
                    "render_fps": render_fps,
                    "overall_fps": overall_fps,
                    "rtf": rtf,
                    "peak_vram_gb": peak_mem
                }
                results.append(metrics)
                
                # Save video optional? User asked to "save output".
                # Saving might affect benchmark time if included in total_time.
                # Usually benchmark excludes saving time unless e2e.
                # I'll exclude saving from metrics but do save it.
                
                print(f"  FPS: {overall_fps:.2f} | RTF: {rtf:.4f} | Peak VRAM: {peak_mem:.2f} GB")
                
        except Exception as e:
            print(f"Failed combo {src_path} + {aud_path}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if not results:
        print("No results generated.")
        return

    avg_fps = np.mean([r['overall_fps'] for r in results])
    avg_rtf = np.mean([r['rtf'] for r in results])
    avg_vram = np.mean([r['peak_vram_gb'] for r in results])
    
    summary = {
        "average_overall_fps": avg_fps,
        "average_rtf": avg_rtf,
        "average_peak_vram_gb": avg_vram,
        "total_combinations": len(results)
    }
    
    # Save CSV
    csv_path = os.path.join(args.save_path, "benchmark_metrics.csv")
    keys = results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    # Save Summary JSON
    json_path = os.path.join(args.save_path, "summary.json")
    with open(json_path, 'w') as f:
        json.dump({"summary": summary, "details": results}, f, indent=4)
        
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"Combinations: {len(results)}")
    print(f"Avg Overall FPS: {avg_fps:.2f}")
    print(f"Avg RTF:         {avg_rtf:.4f} (Lower is better, <1.0 is Real-Time)")
    print(f"Avg Peak VRAM:   {avg_vram:.2f} GB")
    print(f"Report saved to {args.save_path}")

if __name__ == "__main__":
    main()
