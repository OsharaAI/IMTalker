import os
import sys
import time
import json
import shutil
from pathlib import Path

import psutil
import torch
from PIL import Image
import librosa


def _project_root() -> Path:
    # scripts/ -> project root
    return Path(__file__).resolve().parent.parent


def _setup_sys_path():
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _import_app():
    """
    Import app.py and reuse its already-defined cfg/InferenceAgent if possible.
    We avoid starting Gradio; we just want the model + functions.
    """
    _setup_sys_path()
    import importlib

    spec = importlib.util.spec_from_file_location("app", _project_root() / "app.py")
    app = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(app)  # type: ignore
    return app


def _measure_gpu_memory():
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    return device, before


def _finalize_gpu_memory(device, before_allocated):
    if device is None:
        return None, None
    after = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)
    used_during_run = max(0, after - before_allocated)
    return {
        "allocated_before_bytes": int(before_allocated),
        "allocated_after_bytes": int(after),
        "peak_bytes": int(peak),
        "delta_bytes": int(used_during_run),
    }


def run_benchmarks():
    """
    Run audio-driven inference for all 5 audio files in assets with a single
    source image (chandler.jpg) and report:
    - wall time per run
    - CPU usage (user/system seconds)
    - GPU VRAM usage (bytes) if CUDA is available
    - audio duration (seconds)
    - real-time factor (RTF = wall_time / audio_duration)

    Results are saved under ./tmp as JSON.
    """
    app = _import_app()

    if getattr(app, "agent", None) is None:
        raise RuntimeError("app.agent is not initialized; ensure checkpoints are available before running benchmarks.")

    agent = app.agent
    cfg = app.cfg

    root = _project_root()
    assets_dir = root / "assets"

    # Use chandler.jpg as the common source image and benchmark all 5 audios.
    # image_path = assets_dir / "source_1.png"
    image_path = assets_dir / "chandler.jpg"
    audio_files = sorted(assets_dir.glob("audio_*.wav"))

    combinations = []
    for audio_path in audio_files:
        combinations.append(
            {
                "name": audio_path.stem,
                "image": image_path,
                "audio": audio_path,
                "crop": True,
                "seed": 25,
                "nfe": 30,
                "cfg_scale": 2.0,
                "repeats": 1,  # one run per audio file
            }
        )

    process = psutil.Process(os.getpid())
    overall_results = []
    tmp_dir = _project_root() / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    for combo in combinations:
        combo_results = {
            "name": combo["name"],
            "image": str(combo["image"]),
            "audio": str(combo["audio"]),
            "crop": combo["crop"],
            "seed": combo["seed"],
            "nfe": combo["nfe"],
            "cfg_scale": combo["cfg_scale"],
            "runs": [],
        }

        if not combo["image"].exists():
            print(f"Skipping {combo['name']} – image not found: {combo['image']}")
            overall_results.append(combo_results)
            continue
        if not combo["audio"].exists():
            print(f"Skipping {combo['name']} – audio not found: {combo['audio']}")
            overall_results.append(combo_results)
            continue

        print(f"\n=== Benchmarking: {combo['name']} ===")

        for run_idx in range(combo["repeats"]):
            print(f"  Run {run_idx + 1}/{combo['repeats']} ...")

            # Prepare inputs
            img_pil = Image.open(combo["image"]).convert("RGB")
            audio_path = str(combo["audio"])
            # Audio duration in seconds for RTF
            audio_duration = librosa.get_duration(path=audio_path)

            # Reset GPU stats and capture baseline
            gpu_device, gpu_before = _measure_gpu_memory()

            # CPU measurement: percent over interval is more meaningful,
            # but to keep inference untampered we sample before/after.
            cpu_before = process.cpu_times()
            wall_start = time.time()

            video_path = agent.run_audio_inference(
                img_pil=img_pil,
                aud_path=audio_path,
                crop=combo["crop"],
                seed=int(combo["seed"]),
                nfe=int(combo["nfe"]),
                cfg_scale=float(combo["cfg_scale"]),
            )

            wall_end = time.time()
            cpu_after = process.cpu_times()

            gpu_stats = _finalize_gpu_memory(gpu_device, gpu_before) if gpu_device is not None else None

            # Move/copy output video into ./tmp with descriptive name:
            # {audio_filename}_{image_filename}.mp4
            audio_stem = combo["audio"].stem
            image_stem = combo["image"].stem
            dest_video_name = f"{audio_stem}_{image_stem}.mp4"
            dest_video_path = tmp_dir / dest_video_name
            try:
                shutil.copyfile(video_path, dest_video_path)
                saved_video_path = str(dest_video_path)
            except Exception:
                # Fallback to original path if copy fails
                saved_video_path = str(video_path)

            user_cpu = cpu_after.user - cpu_before.user
            system_cpu = cpu_after.system - cpu_before.system

            run_result = {
                "run_index": run_idx,
                "wall_time_sec": wall_end - wall_start,
                "audio_duration_sec": audio_duration,
                "real_time_factor": (wall_end - wall_start) / audio_duration if audio_duration > 0 else None,
                "cpu_user_sec": user_cpu,
                "cpu_system_sec": system_cpu,
                "output_video": saved_video_path,
                "gpu": gpu_stats,
            }
            combo_results["runs"].append(run_result)

            print(f"    Wall time: {run_result['wall_time_sec']:.2f}s "
                  f"(user {user_cpu:.2f}s, sys {system_cpu:.2f}s)")
            if gpu_stats is not None:
                print(
                    f"    GPU delta: {gpu_stats['delta_bytes'] / (1024 ** 2):.2f} MiB, "
                    f"peak: {gpu_stats['peak_bytes'] / (1024 ** 2):.2f} MiB"
                )

        overall_results.append(combo_results)

    # Save under ./tmp as requested
    reports_dir = _project_root() / "tmp"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "benchmark_report.json"
    with report_path.open("w") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\nBenchmark report saved to: {report_path}")


if __name__ == "__main__":
    run_benchmarks()


