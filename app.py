import os
import sys
import tempfile
import subprocess
import numpy as np
import cv2
import torch
import torchvision
import librosa
import face_alignment
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm
import random
from huggingface_hub import hf_hub_download

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try importing local modules
try:
    from generator.FM import FMGenerator
    from renderer.models import IMTRenderer
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure 'generator' and 'renderer' folders are in the same directory.")


# ==========================================
# Automatic Model Download Logic
# ==========================================
def ensure_checkpoints():
    print("Checking model checkpoints...")

    REPO_ID = "cbsjtu01/IMTalker"
    REPO_TYPE = "model"

    files_to_download = [
        "config.yaml",
        "renderer.ckpt",
        "generator.ckpt",
        "wav2vec2-base-960h/config.json",
        "wav2vec2-base-960h/pytorch_model.bin",
        "wav2vec2-base-960h/preprocessor_config.json",
        "wav2vec2-base-960h/feature_extractor_config.json",
    ]

    TARGET_DIR = "checkpoints"
    os.makedirs(TARGET_DIR, exist_ok=True)

    for remote_filename in files_to_download:
        local_file_path = os.path.join(TARGET_DIR, remote_filename)

        # Check if file exists and size is valid (> 1KB)
        if (
            not os.path.exists(local_file_path)
            or os.path.getsize(local_file_path) < 1024
        ):
            print(f"Downloading {remote_filename} to {TARGET_DIR}...")
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=remote_filename,
                    repo_type=REPO_TYPE,
                    local_dir=TARGET_DIR,
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                print(f"Failed to download {remote_filename}: {e}")
                pass
        else:
            pass



class AppConfig:
    def __init__(self):
        ensure_checkpoints()
        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on device: {self.device}")

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

        # Parallel rendering configuration (for fastapi_app.py)
        # Auto-configure based on device and available VRAM
        self.num_render_workers = self._get_optimal_workers()
        self.render_chunk_size = self._get_optimal_chunk_size()

        self.ref_path = None
        self.pose_path = None
        self.gaze_path = None
        self.aud_path = None
        self.crop = True
        self.crop_scale = 0.8  # Face crop padding scale (0.5=tight, 0.8=default, 1.2=loose)
        self.source_path = None
        self.driving_path = None

    def _get_optimal_workers(self):
        """Auto-detect optimal number of parallel rendering workers"""
        if torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_gb > 16:
                    return 4  # High VRAM
                elif vram_gb > 8:
                    return 3  # Medium VRAM (recommended)
                else:
                    return 2  # Low VRAM
            except:
                return 3  # Default
        elif torch.backends.mps.is_available():
            return 2  # MPS has memory constraints
        else:
            import os

            return max(1, os.cpu_count() // 2 if os.cpu_count() else 2)  # CPU

    def _get_optimal_chunk_size(self):
        """Auto-detect optimal chunk size for parallel rendering"""
        if torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_gb > 16:
                    return 10  # Large chunks for high VRAM
                elif vram_gb > 8:
                    return 5  # Medium chunks (recommended)
                else:
                    return 3  # Small chunks for low VRAM
            except:
                return 5  # Default
        elif torch.backends.mps.is_available():
            return 3  # Small chunks for MPS
        else:
            return 10  # Larger chunks for CPU (less overhead)


class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        # Crop scale: controls how much padding around the face (0.5=tight, 0.8=default, 1.2=loose)
        self.crop_scale = getattr(opt, 'crop_scale', 0.8)
        print(f"Loading Face Alignment...")
        # Load FaceAlignment on CPU to save VRAM for the generator
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False
        )

        print("Loading Wav2Vec2...")
        local_path = opt.wav2vec_model_path
        if os.path.exists(local_path) and os.path.exists(
            os.path.join(local_path, "config.json")
        ):
            print(f"Loading local wav2vec from {local_path}")
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
                local_path, local_files_only=True
            )
        else:
            print(
                "Local wav2vec model not found, downloading from 'facebook/wav2vec2-base-960h'..."
            )
            self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
        self.transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )
        self.blend_mask = None

    def get_crop_coords(self, img_arr):
        """
        Calculate face crop coordinates.
        Returns: (x1, y1, x2, y2)
        """
        h, w = img_arr.shape[:2]
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
            
            # Padding checks
            if x1_new < 0:
                x2_new += 0 - x1_new
                x1_new = 0
            if y1_new < 0:
                y2_new += 0 - y1_new
                y1_new = 0
            if x2_new > w:
                x1_new -= x2_new - w
                x2_new = w
            if y2_new > h:
                y1_new -= y2_new - h
                y2_new = h
                
            x1_new = max(0, x1_new)
            y1_new = max(0, y1_new)
            x2_new = min(w, x2_new)
            y2_new = min(h, y2_new)
            
            curr_w = x2_new - x1_new
            curr_h = y2_new - y1_new
            min_side = min(curr_w, curr_h)
            x2_new = x1_new + min_side
            y2_new = y1_new + min_side
            
        return int(x1_new), int(y1_new), int(x2_new), int(y2_new)

    def process_img(self, img: Image.Image) -> Image.Image:
        img_arr = np.array(img)
        if img_arr.ndim == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        elif img_arr.shape[2] == 4:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
            
        x1, y1, x2, y2 = self.get_crop_coords(img_arr)
        crop_img = img_arr[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop_img)
        return crop_pil.resize((self.opt.input_size, self.opt.input_size))

    def paste_back(self, face_img: torch.Tensor, full_img: Image.Image, coords):
        """
        Paste 512x512 face tensor back into full image PIL.
        """
        x1, y1, x2, y2 = coords
        w_crop = x2 - x1
        h_crop = y2 - y1
        
        # Convert tensor to PIL
        if isinstance(face_img, torch.Tensor):
            if face_img.dim() == 4:
                face_img = face_img.squeeze(0)
            face_img = face_img.permute(1, 2, 0).cpu().numpy()
            face_img = np.clip(face_img * 255, 0, 255).astype(np.uint8)
            face_pil = Image.fromarray(face_img)
        else:
            face_pil = face_img
            
        # Resize back to crop size
        face_resized = face_pil.resize((w_crop, h_crop), Image.LANCZOS)
        
        # Paste
        full_img_copy = full_img.copy()
        full_img_copy.paste(face_resized, (x1, y1))
        
        return full_img_copy

    def paste_back_tensor(self, face_tensor: torch.Tensor, full_img_tensor: torch.Tensor, coords):
        """
        Paste 512x512 face tensor back into full image tensor on GPU with soft blending.
        Args:
            face_tensor: (1, 3, 512, 512) or (3, 512, 512)
            full_img_tensor: (1, 3, H, W) or (3, H, W)
            coords: (x1, y1, x2, y2)
        Returns:
            Combined tensor (same shape as full_img_tensor)
        """
        if face_tensor.dim() == 4:
            face_tensor = face_tensor.squeeze(0)
        
        was_4d = False
        if full_img_tensor.dim() == 4:
            full_img_tensor = full_img_tensor.squeeze(0)
            was_4d = True
            
        x1, y1, x2, y2 = coords
        w_crop = x2 - x1
        h_crop = y2 - y1
        
        # Resize face tensor to target crop size using interpolation
        # F.interpolate expects 4D input (N, C, H, W)
        face_resized = torch.nn.functional.interpolate(
            face_tensor.unsqueeze(0), 
            size=(h_crop, w_crop), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Clone full image to avoid modifying original
        combined = full_img_tensor.clone()
        
        # Paste
        # Check boundaries to be safe
        h_full, w_full = combined.shape[1], combined.shape[2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_full, x1 + face_resized.shape[2])
        y2 = min(h_full, y1 + face_resized.shape[1])
        
        # Adjust face_resized if boundaries clipped (rare but possible)
        h_paste = y2 - y1
        w_paste = x2 - x1
        
        # Generate Blend Mask (lazy init)
        if self.blend_mask is None:
            # Create a 512x512 mask with soft edges (simulating crop feathering)
            # Distance from edge
            try:
                # Meshgrid order varies by version, indexing='ij' is standard for matrix
                Y, X = torch.meshgrid(torch.arange(512), torch.arange(512), indexing='ij')
            except TypeError:
                # Fallback for older torch
                Y, X = torch.meshgrid(torch.arange(512), torch.arange(512))
                
            dist_x = torch.min(X, 511 - X)
            dist_y = torch.min(Y, 511 - Y)
            dist = torch.min(dist_x, dist_y).float()
            
            # Feathering margin (pixels in 512 domain)
            margin = 32.0 
            mask = torch.clamp(dist / margin, 0.0, 1.0)
            self.blend_mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, 512, 512)
            
        if self.blend_mask.device != face_tensor.device:
            self.blend_mask = self.blend_mask.to(face_tensor.device)
            
        # Resize mask to crop size
        mask_resized = torch.nn.functional.interpolate(
            self.blend_mask,
            size=(h_crop, w_crop),
            mode='bilinear',
            align_corners=False
        ).squeeze(0) # (1, h_crop, w_crop)
        
        # Get regions
        full_region = combined[:, y1:y2, x1:x2]
        face_region = face_resized[:, :h_paste, :w_paste]
        mask_region = mask_resized[:, :h_paste, :w_paste]
        
        # Blend: face * mask + full * (1 - mask)
        blended = face_region * mask_region + full_region * (1.0 - mask_region)
        
        combined[:, y1:y2, x1:x2] = blended
        
        if was_4d:
            combined = combined.unsqueeze(0)
        
        return combined

    def resize_with_pad(self, img: Image.Image) -> Image.Image:
        """
        Resize image to target size (input_size x input_size) while maintaining aspect ratio.
        Pads with black borders.
        """
        target_size = self.opt.input_size
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
        
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize using PIL for better quality
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Create black background
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        
        # Paste centered
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        
        return new_img

    def process_audio(self, path: str) -> torch.Tensor:
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
        video = cv2.VideoCapture(from_mp4_file_path)
        index = 0
        bboxes_lists = []
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Analyzing video for stable cropping: {from_mp4_file_path}")
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
                except:
                    detected_bboxes = None
                current_frame_bboxes = []
                if detected_bboxes is not None:
                    for d_box in detected_bboxes:
                        bx1, by1, bx2, by2, score = d_box
                        if score > 0.5:
                            current_frame_bboxes.append(
                                [
                                    int(bx1 / mult),
                                    int(by1 / mult),
                                    int(bx2 / mult),
                                    int(by2 / mult),
                                    score,
                                ]
                            )
                if len(current_frame_bboxes) > 0:
                    max_bboxes = max(
                        current_frame_bboxes, key=lambda bbox: bbox[2] - bbox[0]
                    )
                    bboxes_lists.append(max_bboxes)
            index += 1
        video.release()
        x_center_lists, y_center_lists, width_lists, height_lists = [], [], [], []
        for bbox in bboxes_lists:
            x1, y1, x2, y2 = bbox[:4]
            x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
            x_center_lists.append(x_center)
            y_center_lists.append(y_center)
            width_lists.append(x2 - x1)
            height_lists.append(y2 - y1)
        if not (x_center_lists and y_center_lists and width_lists and height_lists):
            import shutil

            shutil.copy(from_mp4_file_path, to_mp4_file_path)
            return
        x_center = sorted(x_center_lists)[len(x_center_lists) // 2]
        y_center = sorted(y_center_lists)[len(y_center_lists) // 2]
        median_width = sorted(width_lists)[len(width_lists) // 2]
        median_height = sorted(height_lists)[len(height_lists) // 2]
        expanded_width = int(median_width * (1 + expanded_ratio))
        expanded_height = int(median_height * (1 + expanded_ratio))
        fixed_cropped_width = min(max(expanded_width, expanded_height), width, height)
        x1, y1 = int(x_center - fixed_cropped_width / 2), int(
            y_center - fixed_cropped_width / 2
        )
        x1 = max(0, x1)
        y1 = max(0, y1)
        if x1 + fixed_cropped_width > width:
            x1 = width - fixed_cropped_width
        if y1 + fixed_cropped_width > height:
            y1 = height - fixed_cropped_width
        target_size = self.opt.input_size

        cmd = f'ffmpeg -i "{from_mp4_file_path}" -filter:v "crop={fixed_cropped_width}:{fixed_cropped_width}:{x1}:{y1},scale={target_size}:{target_size}:flags=lanczos" -c:v libx264 -crf 18 -preset slow -c:a aac -b:a 128k "{to_mp4_file_path}" -y -loglevel error'
        if os.system(cmd) != 0:
            print("FFmpeg command failed. Copying original.")
            import shutil

            shutil.copy(from_mp4_file_path, to_mp4_file_path)


class InferenceAgent:
    def __init__(self, opt):
        torch.cuda.empty_cache()
        self.opt = opt
        self.device = opt.device
        self.data_processor = DataProcessor(opt)
        print("Loading Models...")
        self.renderer = IMTRenderer(self.opt).to(self.device)
        self.generator = FMGenerator(self.opt).to(self.device)
        if not os.path.exists(self.opt.renderer_path) or not os.path.exists(
            self.opt.generator_path
        ):
            raise FileNotFoundError(
                "Checkpoints not found even after download attempt."
            )
        self._load_ckpt(self.renderer, self.opt.renderer_path, "gen.")
        self._load_fm_ckpt(self.generator, self.opt.generator_path)
        self.renderer.eval()
        self.generator.eval()
        print("Models loaded successfully.")

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
                    param.copy_(clean_dict[name].to(self.device))

    def save_video(self, vid_tensor, fps, audio_path=None):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            raw_path = tmp.name
        if vid_tensor.dim() == 4:
            vid = vid_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        if vid.min() < 0:
            vid = (vid + 1) / 2
        vid = np.clip(vid, 0, 1)
        vid = (vid * 255).astype(np.uint8)
        height, width = vid.shape[1], vid.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
        for frame in vid:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        if audio_path:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
                final_path = tmp_out.name
            cmd = f'ffmpeg -y -i "{raw_path}" -i "{audio_path}" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{final_path}"'
            subprocess.call(cmd, shell=True)
            if os.path.exists(raw_path):
                os.remove(raw_path)
            return final_path
        else:
            return raw_path

    @torch.no_grad()
    def run_audio_inference_streaming(
        self, img_pil, aud_path, crop, seed, nfe, cfg_scale
    ):
        """Stream frames as they are generated for audio-driven inference."""
        s_pil = (
            self.data_processor.process_img(img_pil)
            if crop
            else img_pil.resize((self.opt.input_size, self.opt.input_size))
        )
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = (
            self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        )
        data = {
            "s": s_tensor,
            "a": a_tensor,
            "pose": None,
            "cam": None,
            "gaze": None,
            "ref_x": None,
        }
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        data["ref_x"] = t_lat
        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        T = sample.shape[1]
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)

        # Yield frames one by one
        for t in range(T):
            ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            out_frame = self.renderer.decode(m_c, m_r, f_r)
            yield out_frame.squeeze(0)  # Yield single frame tensor (C, H, W)

    @torch.no_grad()
    def run_audio_inference(self, img_pil, aud_path, crop, seed, nfe, cfg_scale):
        """Legacy batch method - collects all frames and saves video."""
        s_pil = (
            self.data_processor.process_img(img_pil)
            if crop
            else img_pil.resize((self.opt.input_size, self.opt.input_size))
        )
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = (
            self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        )
        data = {
            "s": s_tensor,
            "a": a_tensor,
            "pose": None,
            "cam": None,
            "gaze": None,
            "ref_x": None,
        }
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        data["ref_x"] = t_lat
        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        d_hat = []
        T = sample.shape[1]
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        for t in range(T):
            ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            out_frame = self.renderer.decode(m_c, m_r, f_r)
            d_hat.append(out_frame)
        vid_tensor = torch.stack(d_hat, dim=1).squeeze(0)
        return self.save_video(vid_tensor, self.opt.fps, aud_path)

    @torch.no_grad()
    def run_video_inference_streaming(self, source_img_pil, driving_video_path, crop):
        """Stream frames as they are generated for video-driven inference."""
        s_pil = (
            self.data_processor.process_img(source_img_pil)
            if crop
            else source_img_pil.resize((self.opt.input_size, self.opt.input_size))
        )
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        f_r, i_r = self.renderer.app_encode(s_tensor)
        t_r = self.renderer.mot_encode(s_tensor)
        ta_r = self.renderer.adapt(t_r, i_r)
        ma_r = self.renderer.mot_decode(ta_r)
        final_driving_path = driving_video_path
        temp_crop_video = None
        if crop:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                temp_crop_video = tmp.name
            self.data_processor.crop_video_stable(driving_video_path, temp_crop_video)
            final_driving_path = temp_crop_video
        cap = cv2.VideoCapture(final_driving_path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame).resize(
                    (self.opt.input_size, self.opt.input_size)
                )
                d_tensor = (
                    self.data_processor.transform(frame_pil)
                    .unsqueeze(0)
                    .to(self.device)
                )
                t_c = self.renderer.mot_encode(d_tensor)
                ta_c = self.renderer.adapt(t_c, i_r)
                ma_c = self.renderer.mot_decode(ta_c)
                out = self.renderer.decode(ma_c, ma_r, f_r)
                yield out.squeeze(0).cpu()  # Yield single frame (C, H, W)
        finally:
            cap.release()
            if temp_crop_video and os.path.exists(temp_crop_video):
                os.remove(temp_crop_video)

    @torch.no_grad()
    def run_video_inference(self, source_img_pil, driving_video_path, crop):
        """Legacy batch method - collects all frames and saves video."""
        s_pil = (
            self.data_processor.process_img(source_img_pil)
            if crop
            else source_img_pil.resize((self.opt.input_size, self.opt.input_size))
        )
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        f_r, i_r = self.renderer.app_encode(s_tensor)
        t_r = self.renderer.mot_encode(s_tensor)
        ta_r = self.renderer.adapt(t_r, i_r)
        ma_r = self.renderer.mot_decode(ta_r)
        final_driving_path = driving_video_path
        temp_crop_video = None
        if crop:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                temp_crop_video = tmp.name
            self.data_processor.crop_video_stable(driving_video_path, temp_crop_video)
            final_driving_path = temp_crop_video
        cap = cv2.VideoCapture(final_driving_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        vid_results = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame).resize(
                (self.opt.input_size, self.opt.input_size)
            )
            d_tensor = (
                self.data_processor.transform(frame_pil).unsqueeze(0).to(self.device)
            )
            t_c = self.renderer.mot_encode(d_tensor)
            ta_c = self.renderer.adapt(t_c, i_r)
            ma_c = self.renderer.mot_decode(ta_c)
            out = self.renderer.decode(ma_c, ma_r, f_r)
            vid_results.append(out.cpu())
        cap.release()
        if temp_crop_video and os.path.exists(temp_crop_video):
            os.remove(temp_crop_video)
        if not vid_results:
            raise Exception("Driving video reading failed.")
        vid_tensor = torch.cat(vid_results, dim=0)
        return self.save_video(vid_tensor, fps=fps, audio_path=driving_video_path)


agent = None

if __name__ == "__main__":
    print("Initializing Configuration...")
    cfg = AppConfig()

    try:
        if os.path.exists(cfg.renderer_path) and os.path.exists(cfg.generator_path):
            agent = InferenceAgent(cfg)
        else:
            print(
                "Error: Checkpoints not found. They should have been downloaded automatically."
            )
            # Try again if download just happened
            if os.path.exists(cfg.renderer_path):
                agent = InferenceAgent(cfg)
    except Exception as e:
        print(f"Initialization Error: {e}")
        import traceback
    
        traceback.print_exc()
    
    
    def fn_audio_driven(image, audio, crop, seed, nfe, cfg_scale, progress=gr.Progress()):
        if agent is None:
            raise gr.Error("Models not loaded properly. Check logs.")
        if image is None or audio is None:
            raise gr.Error("Missing image or audio.")
    
        img_pil = Image.fromarray(image).convert("RGB")
        try:
            return agent.run_audio_inference(
                img_pil, audio, crop, int(seed), int(nfe), float(cfg_scale)
            )
        except Exception as e:
            import traceback
    
            traceback.print_exc()
            raise gr.Error(f"Error: {e}")
    
    
    def fn_video_driven(source_image, driving_video, crop, progress=gr.Progress()):
        if agent is None:
            raise gr.Error("Models not loaded properly. Check logs.")
        if source_image is None or driving_video is None:
            raise gr.Error("Missing inputs.")
    
        img_pil = Image.fromarray(source_image).convert("RGB")
        try:
            return agent.run_video_inference(img_pil, driving_video, crop)
        except Exception as e:
            import traceback
    
            traceback.print_exc()
            raise gr.Error(f"Error: {e}")
    
    
    # Gradio Interface
    with gr.Blocks(title="IMTalker Demo") as demo:
        gr.Markdown("# 🗣️ IMTalker: Efficient Audio-driven Talking Face Generation")
    
        with gr.Accordion("💡 Best Practices (Click to read)", open=False):
            gr.Markdown(
                """
            To obtain the highest quality generation results, we recommend following these guidelines:
    
            1.  **Input Image Composition**: 
                Please ensure the input image features the person's head as the primary subject. Since our model is explicitly trained on facial data, it does not support full-body video generation. 
                * The inference pipeline automatically **crops the input image** to focus on the face by default.
                * **Note on Resolution**: The model generates video at a fixed resolution of **512×512**. Using extremely high-resolution inputs will result in downscaling, so prioritize facial clarity over raw image dimensions.
    
            2.  **Audio Selection**: 
                Our model was trained primarily on **English datasets**. Consequently, we recommend using **English audio** inputs to achieve the best lip-synchronization performance and naturalness.
    
            3.  **Background Quality**: 
                We strongly recommend using source images with **solid colored** or **blurred (bokeh)** backgrounds. Complex or highly detailed backgrounds may lead to visual artifacts or jitter in the generated video.
            """
            )
    
        with gr.Tabs():
            # ==========================
            # Tab 1: Audio Driven
            # ==========================
            with gr.TabItem("Audio Driven"):
                with gr.Row():
                    with gr.Column():
                        a_img = gr.Image(
                            label="Source Image", type="numpy", height=512, width=512
                        )
    
                        gr.Examples(
                            examples=[
                                ["assets/source_1.png"],
                                ["assets/source_2.png"],
                                ["assets/source_3.jpg"],
                                ["assets/source_4.png"],
                                ["assets/source_5.png"],
                                ["assets/source_6.png"],
                            ],
                            inputs=[a_img],
                            label="Example Images",
                            cache_examples=False,
                        )
    
                        a_aud = gr.Audio(label="Driving Audio", type="filepath")
    
                        gr.Examples(
                            examples=[
                                ["assets/audio_1.wav"],
                                ["assets/audio_2.wav"],
                                ["assets/audio_3.wav"],
                                ["assets/audio_4.wav"],
                                ["assets/audio_5.wav"],
                            ],
                            inputs=[a_aud],
                            label="Example Audios",
                            cache_examples=False,
                        )
    
                        with gr.Accordion("Settings", open=True):
                            a_crop = gr.Checkbox(label="Auto Crop Face", value=False)
                            a_seed = gr.Number(label="Seed", value=42)
                            a_nfe = gr.Slider(5, 50, value=10, step=1, label="Steps (NFE)")
                            a_cfg = gr.Slider(1.0, 5.0, value=2.0, label="CFG Scale")
    
                        a_btn = gr.Button("Generate (Audio Driven)", variant="primary")
    
                    with gr.Column():
                        a_out = gr.Video(label="Result", height=512, width=512)
    
                a_btn.click(
                    fn_audio_driven, [a_img, a_aud, a_crop, a_seed, a_nfe, a_cfg], a_out
                )
    
            # ==========================
            # Tab 2: Video Driven
            # ==========================
            with gr.TabItem("Video Driven"):
                with gr.Row():
                    with gr.Column():
                        v_img = gr.Image(
                            label="Source Image", type="numpy", height=512, width=512
                        )
    
                        gr.Examples(
                            examples=[
                                ["assets/source_7.png"],
                                ["assets/source_8.png"],
                                ["assets/source_9.png"],
                                ["assets/source_10.png"],
                                ["assets/source_11.png"],
                            ],
                            inputs=[v_img],
                            label="Example Images",
                            cache_examples=False,
                        )
    
                        v_vid = gr.Video(
                            label="Driving Video", sources=["upload"], height=512, width=512
                        )
    
                        gr.Examples(
                            examples=[
                                ["assets/driving_1.mp4"],
                                ["assets/driving_2.mp4"],
                                ["assets/driving_3.mp4"],
                                ["assets/driving_4.mp4"],
                                ["assets/driving_5.mp4"],
                            ],
                            inputs=[v_vid],
                            label="Example Videos",
                            cache_examples=False,
                        )
    
                        v_crop = gr.Checkbox(
                            label="Auto Crop (Both Source & Driving)", value=False
                        )
                        v_btn = gr.Button("Generate (Video Driven)", variant="primary")
    
                    with gr.Column():
                        v_out = gr.Video(label="Result", height=512, width=512)
    
                v_btn.click(fn_video_driven, [v_img, v_vid, v_crop], v_out)
    
    if __name__ == "__main__":
        demo.launch(share=False)
