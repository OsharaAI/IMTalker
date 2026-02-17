
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

# Add parent directory
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

def get_peak_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0

def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def main():
    parser = argparse.ArgumentParser(description="IMTalker TRT Benchmarking")
    parser.add_argument("--sources", type=str, required=True, help="Comma-separated list of source image paths")
    parser.add_argument("--audios", type=str, required=True, help="Comma-separated list of audio file paths")
    parser.add_argument("--save_path", type=str, default="benchmark_results_trt", help="Directory to save output")
    parser.add_argument("--crop", action="store_true", help="Auto crop face")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nfe", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--engine_dir", type=str, default="trt_engines", help="TensorRT engine directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # 1. Initialization
    print("Initializing Model...")
    cfg = AppConfig()
    cfg.device = args.device
    agent = InferenceAgent(cfg)
    
    # Load TRT Handler
    from trt_handler import TRTInferenceHandler
    trt_handler = TRTInferenceHandler(agent, engine_dir=args.engine_dir)
    
    if not trt_handler.is_available():
        print("Error: TensorRT engines not found. Please build them first.")
        sys.exit(1)
    
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
    
    for src_path in sources:
        try:
            img_pil = Image.open(src_path).convert("RGB")
            
            for aud_path in audios:
                combo_name = f"{os.path.basename(src_path)}_{os.path.basename(aud_path)}"
                print(f"Benchmarking (TRT): {combo_name}")
                
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
                    t_prep_end = time.time()
                    
                    # 2. Source Encoder (TRT)
                    t_enc_start = time.time()
                    enc_outputs = trt_handler.source_encoder.infer([s_tensor])
                    i_r = enc_outputs['i_r']
                    t_r = enc_outputs['t_r']
                    f_r = [enc_outputs[f'f_r_{i}'] for i in range(6)]
                    ma_r = [enc_outputs[f'ma_r_{i}'] for i in range(4)]
                    t_enc_end = time.time()
                    
                    # 3. Audio & Motion (PyTorch)
                    t_gen_start = time.time()
                    a_tensor = agent.data_processor.process_audio(aud_path).unsqueeze(0).to(agent.device)
                    
                    data = {'s': s_tensor, 'a': a_tensor, 'pose': None, 'cam': None, 'gaze': None, 'ref_x': t_r}
                    torch.manual_seed(args.seed)
                    sample, _ = agent.generator.sample(data, a_cfg_scale=2.0, nfe=args.nfe, seed=args.seed)
                    sample = (1.0 - 0.6) * t_r.unsqueeze(1).repeat(1, sample.shape[1], 1) + 0.6 * sample
                    t_gen_end = time.time()
                    
                    # 4. Rendering (TRT)
                    T = sample.shape[1]
                    t_render_start = time.time()
                    
                    frames = []
                    for t in range(T):
                        motion_code = sample[:, t, ...]
                        trt_inputs = {'motion_code': motion_code, 'i_r': i_r}
                        for i in range(6): trt_inputs[f'f_r_{i}'] = f_r[i]
                        for i in range(4): trt_inputs[f'ma_r_{i}'] = ma_r[i]
                        
                        out_dict = trt_handler.audio_renderer.infer(trt_inputs)
                        out_frame = out_dict['output_image']
                        frames.append(out_frame.cpu())
                        
                    t_render_end = time.time()
                t1 = time.time()
                
                # Metrics
                total_duration = t1 - t0
                prep_time = t_prep_end - t_prep_start
                enc_time = t_enc_end - t_enc_start
                gen_time = t_gen_end - t_gen_start
                render_time = t_render_end - t_render_start
                
                render_fps = T / render_time if render_time > 0 else 0
                overall_fps = T / total_duration if total_duration > 0 else 0
                
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
                    "enc_time": enc_time,
                    "gen_time": gen_time,
                    "render_time": render_time,
                    "render_fps": render_fps,
                    "overall_fps": overall_fps,
                    "rtf": rtf,
                    "peak_vram_gb": peak_mem
                }
                results.append(metrics)
                
                print(f"  FPS: {overall_fps:.2f} | RTF: {rtf:.4f} | Peak VRAM: {peak_mem:.2f} GB")
                
                # Save Video
                vid_tensor = torch.stack(frames, dim=1).squeeze(0)
                try:
                    agent.save_video(vid_tensor, 25, aud_path) 
                except: pass

        except Exception as e:
            print(f"Failed combo {src_path} + {aud_path}: {e}")
            import traceback
            traceback.print_exc()
            
    # Summary
    if not results:
        print("No results.")
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
    
    csv_path = os.path.join(args.save_path, "benchmark_metrics_trt.csv")
    keys = results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    json_path = os.path.join(args.save_path, "summary_trt.json")
    with open(json_path, 'w') as f:
        json.dump({"summary": summary, "details": results}, f, indent=4)
        
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY (TRT)")
    print("="*50)
    print(f"Combinations: {len(results)}")
    print(f"Avg Overall FPS: {avg_fps:.2f}")
    print(f"Avg RTF:         {avg_rtf:.4f}")
    print(f"Avg Peak VRAM:   {avg_vram:.2f} GB")
    print(f"Report saved to {args.save_path}")

if __name__ == "__main__":
    main()
