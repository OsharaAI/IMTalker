
import os
import sys
import time
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor
import threading
import queue

# Add current directory to path so we can import generator/renderer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generator.FM import FMGenerator
from renderer.models import IMTRenderer

class IMTalkerConfig:
    def __init__(self):
        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"Running on device: {self.device}") 
        # (moved print to Streamer init or keep silent)

        self.seed = 42
        self.fix_noise_seed = False
        self.renderer_path = "./checkpoints/renderer.ckpt"
        self.generator_path = "./checkpoints/generator.ckpt"
        self.wav2vec_model_path = "./checkpoints/wav2vec2-base-960h"
        self.input_size = 512
        self.input_nc = 3
        self.fps = 25.0
        self.rank = "cuda"
        self.sampling_rate = 16000
        self.audio_marcing = 2
        self.wav2vec_sec = 2.0
        self.attention_window = 5
        self.only_last_features = True
        self.audio_dropout_prob = 0.1
        self.style_dim = 512
        self.dim_a = 512
        self.dim_h = 512
        self.dim_e = 7
        self.dim_motion = 32
        self.dim_c = 32
        self.dim_w = 32
        self.fmt_depth = 8
        self.num_heads = 8
        self.mlp_ratio = 4.0
        self.no_learned_pe = False
        self.num_prev_frames = 10
        self.max_grad_norm = 1.0
        self.ode_atol = 1e-5
        self.ode_rtol = 1e-5
        self.nfe = 10
        self.torchdiffeq_ode_method = "euler"
        self.a_cfg_scale = 3.0
        self.swin_res_threshold = 128
        self.window_size = 8

        self.crop = True
        self.crop_scale = 0.8

        # Optimization flags
        self.use_batching = False
        self.batch_size = 10
        self.use_fp16 = False  # Disabled by default: causes NaN/Inf with IMTalker
        self.use_compile = True # Set to True if torch >= 2.0 and performance is critical
        
        # Low-latency mode options
        self.low_latency_mode = False
        self.nfe_low_latency = 5 # Reduced NFE for 2x generator speedup (vs default 10)


class IMTalkerStreamer:
    def __init__(self, config_class, device=None):
        self.opt = config_class
        if device:
            self.opt.device = device
        else:
            self.opt.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"IMTalkerStreamer initializing on {self.opt.device}...")
        
        # Load Models
        self.renderer = IMTRenderer(self.opt).to(self.opt.device)
        self.generator = FMGenerator(self.opt).to(self.opt.device)
        
        self._load_ckpt(self.renderer, self.opt.renderer_path, "gen.")
        self._load_fm_ckpt(self.generator, self.opt.generator_path)
        
        self.renderer.eval()
        self.generator.eval()
        
        # Audio Processor
        print(f"Loading Wav2Vec2 from {self.opt.wav2vec_model_path}")
        if os.path.exists(self.opt.wav2vec_model_path) and os.path.exists(
            os.path.join(self.opt.wav2vec_model_path, "config.json")
        ):
             self.wav2vec = Wav2Vec2FeatureExtractor.from_pretrained(
                self.opt.wav2vec_model_path, local_files_only=True
            )
        else:
            self.wav2vec = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            
        self.transform = None # Defined during process_img
        
        # Optional: torch.compile
        if self.opt.use_compile and hasattr(torch, 'compile'):
            print("Compiling models for faster inference...")
            try:
                self.renderer = torch.compile(self.renderer)
                self.generator = torch.compile(self.generator)
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Proceeding without compilation.")
        
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
                    param.copy_(clean_dict[name].to(self.opt.device))

    def process_img(self, img_pil, crop=True):
        # Similar to DataProcessor.process_img but simplified or reused if possible
        # For now, implementing basic resize/crop logic
        import torchvision.transforms as transforms
        import face_alignment
        
        if self.transform is None:
             self.transform = transforms.Compose(
                [transforms.Resize((self.opt.input_size, self.opt.input_size)), transforms.ToTensor()]
            )

        img_arr = np.array(img_pil)
        if crop:
             # Only init face alignment if needed and not already done, to save time/memory if not cropping
             # But usually for good results we need cropping.
             # Note: FaceAlignment is heavy to reload every time. 
             # Ideally should be persistent or passed in. 
             # For this streamer, we assume img_pil is already a good portrait or we do a one-time crop.
             # Let's assume the user passes a ready-to-use image or we do simple center crop if no face detection.
             pass 

        # Simple resize for now to match input size
        # in a real scenario we'd use the full detection logic from app.py
        s_pil = img_pil.resize((self.opt.input_size, self.opt.input_size))
        s_tensor = self.transform(s_pil).unsqueeze(0).to(self.opt.device)
        return s_tensor

    @torch.no_grad()
    def generate_stream(self, avatar_image, audio_chunk_iterator, seed=42, nfe=10, cfg_scale=3.0):
        """
        avatar_image: PIL Image
        audio_chunk_iterator: iterator yielding numpy arrays of audio (waveform)
        """
        
        # 1. Prepare Avatar
        s_tensor = self.process_img(avatar_image)
        
        # Use autocast for FP16 inference (if enabled)
        autocast_context = torch.amp.autocast('cuda', enabled=self.opt.use_fp16)
        
        with autocast_context:
            # Pre-compute avatar embeddings
            f_r, app_feat = self.renderer.dense_feature_encoder(s_tensor)
            t_lat = self.renderer.latent_token_encoder(s_tensor)
            if isinstance(t_lat, tuple): t_lat = t_lat[0]
            
            # dense_feature_encoder returns (features, identity_embedding)
            # app_feat is the identity embedding [1, 512]
            # f_r is the list of multi-scale feature maps
            
            ta_r = self.renderer.adapt(t_lat, app_feat)
            m_r = self.renderer.latent_token_decoder(ta_r)
        
        torch.manual_seed(seed)
        
        total_gen_time = 0
        total_frames = 0
        start_stream_time = time.time()
        
        sampling_rate = self.opt.sampling_rate # 16000
        
        # Maintain temporal context across chunks for smooth motion
        prev_context = None
        
        for i, audio_chunk in enumerate(audio_chunk_iterator):
            if isinstance(audio_chunk, torch.Tensor):
                audio_chunk = audio_chunk.cpu().numpy()
            
            audio_chunk = audio_chunk.flatten()
            chunk_start_time = time.time()
            
            # Process Audio
            input_values = self.wav2vec(
                audio_chunk, 
                sampling_rate=sampling_rate, 
                return_tensors="pt"
            ).input_values.to(self.opt.device) 
            
            # Prepare data dict 
            data = {
                "s": s_tensor,
                "a": input_values, # This is the chunk audio features
                "pose": None,
                "cam": None,
                "gaze": None,
                "ref_x": t_lat,
            }
            
            # Use low-latency NFE if enabled
            effective_nfe = self.opt.nfe_low_latency if self.opt.low_latency_mode else nfe
            
            with autocast_context:
                # Generate Motion for this chunk
                sample, prev_context = self.generator.sample(
                    data, 
                    a_cfg_scale=cfg_scale, 
                    nfe=effective_nfe,  # Use low-latency NFE if enabled
                    seed=seed,
                    prev_context=prev_context
                )
                # sample shape: [B, T, D]
                
                T = sample.shape[1]
                frames_for_chunk = []
                
                # Render Frames in Batches
                if self.opt.use_batching:
                    batch_size = self.opt.batch_size
                    for b_start in range(0, T, batch_size):
                        b_end = min(b_start + batch_size, T)
                        current_batch_size = b_end - b_start
                        
                        # Prepare batched inputs for renderer
                        sample_batch = sample[:, b_start:b_end, ...] # [1, batch, D]
                        sample_batch = sample_batch.squeeze(0) # [batch, D]
                        
                        # Expand reference features for batch
                        app_feat_batch = app_feat.repeat(current_batch_size, 1)
                        m_r_batch = tuple(m.repeat(current_batch_size, 1, 1, 1) for m in m_r)
                        f_r_batch = [f.repeat(current_batch_size, 1, 1, 1) for f in f_r]
                        
                        # Render batch
                        ta_c_batch = self.renderer.adapt(sample_batch, app_feat_batch)
                        m_c_batch = self.renderer.latent_token_decoder(ta_c_batch)
                        out_frames_batch = self.renderer.decode(m_c_batch, m_r_batch, f_r_batch)
                        
                        # out_frames_batch is [batch, 3, H, W]
                        # Split and yield or store? we yield list for this chunk.
                        for f_idx in range(current_batch_size):
                            frames_for_chunk.append(out_frames_batch[f_idx:f_idx+1])
                else:
                    # Sequential rendering (original slow way)
                    for t in range(T):
                        ta_c = self.renderer.adapt(sample[:, t, ...], app_feat)
                        m_c = self.renderer.latent_token_decoder(ta_c)
                        out_frame = self.renderer.decode(m_c, m_r, f_r)
                        frames_for_chunk.append(out_frame)
                
            chunk_gen_time = time.time() - chunk_start_time
            total_gen_time += chunk_gen_time
            total_frames += T
            
            # FP16 Validation: Check for NaN/Inf in frames
            if self.opt.use_fp16 and len(frames_for_chunk) > 0:
                sample_frame = frames_for_chunk[0]  # Check first frame
                if torch.isnan(sample_frame).any() or torch.isinf(sample_frame).any():
                    print(f"⚠️  WARNING: NaN/Inf detected in rendered frames (chunk {i}). FP16 may be unstable.")
                    print(f"    Consider setting use_fp16=False or checking model inputs.")
            
            # Calculate metrics
            chunk_video_duration = T / self.opt.fps
            rtf = chunk_gen_time / chunk_video_duration if chunk_video_duration > 0 else 0
            
            # Yield results
            metrics = {
                "chunk_id": i,
                "generation_time": chunk_gen_time,
                "video_duration": chunk_video_duration,
                "rtf": rtf,
                "num_frames": T
            }
            
            yield frames_for_chunk, metrics

        # Final Summary Metrics
        rtf_total = total_gen_time / (total_frames / self.opt.fps) if total_frames > 0 else 0
        print("\n--- Streaming Finished ---")
        print(f"Total Generation Time: {total_gen_time:.4f}s")
        print(f"Total Video Duration: {total_frames / self.opt.fps:.4f}s")
        print(f"Overall RTF: {rtf_total:.4f}")
        print(f"Total Frames Yielded: {total_frames}")

