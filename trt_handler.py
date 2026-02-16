
import os
import time
import numpy as np
import torch
import cv2

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None

from pathlib import Path
from tqdm import tqdm

def trt_dtype_to_torch(trt_dtype):
    if trt_dtype == trt.float32: return torch.float32
    if trt_dtype == trt.float16: return torch.float16
    if trt_dtype == trt.int32: return torch.int32
    if trt_dtype == trt.int8: return torch.int8
    if trt_dtype == trt.bool: return torch.bool
    return torch.float32

class TensorRTModel:
    def __init__(self, engine_path):
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not installed")
            
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        self.input_names = []
        self.output_info = [] # list of (name, shape, dtype)

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                shape = self.engine.get_tensor_shape(name)
                dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
                self.output_info.append((name, shape, dtype))

    def infer(self, inputs):
        """
        Execute inference.
        inputs: List of torch.Tensor (must be on GPU)
        Returns: List of torch.Tensor (on GPU)
        """
        if len(inputs) != len(self.input_names):
             raise ValueError(f"Expected {len(self.input_names)} inputs, got {len(inputs)}")
             
        # Bind inputs
        for name, tensor in zip(self.input_names, inputs):
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
            
        # Allocate and bind outputs
        outputs = []
        for name, shape, dtype in self.output_info:
            out_tensor = torch.empty(tuple(shape), device='cuda', dtype=dtype)
            self.context.set_tensor_address(name, int(out_tensor.data_ptr()))
            outputs.append(out_tensor)
            
        # Execute
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        
        return outputs

class TRTInferenceHandler:
    def __init__(self, agent, engine_dir="./trt_engines"):
        self.agent = agent
        self.engine_dir = engine_dir
        self.source_encoder = None
        self.audio_renderer = None
        
        if not TRT_AVAILABLE:
            print("[TRT] TensorRT not found. TRT inference disabled.")
            return

        # Try to load engines
        try:
            enc_path = os.path.join(engine_dir, "source_encoder.trt")
            ren_path = os.path.join(engine_dir, "audio_renderer.trt")
            
            if os.path.exists(enc_path) and os.path.exists(ren_path):
                print(f"[TRT] Loading engines from {engine_dir}...")
                self.source_encoder = TensorRTModel(enc_path)
                self.audio_renderer = TensorRTModel(ren_path)
                print("[TRT] Engines loaded successfully.")
            else:
                print(f"[TRT] Engines not found at {engine_dir}. TRT inference disabled.")
        except Exception as e:
            print(f"[TRT] Error loading engines: {e}")
            self.source_encoder = None
            self.audio_renderer = None

    def is_available(self):
        return self.source_encoder is not None and self.audio_renderer is not None

    def run_inference(self, img_pil, aud_path, crop, seed, nfe, cfg_scale):
        if not self.is_available():
            raise RuntimeError("TensorRT engines not loaded")

        device = self.agent.device
        processor = self.agent.data_processor
        generator = self.agent.generator
        opt = self.agent.opt

        # 1. Process Inputs
        s_pil = processor.process_img(img_pil) if crop else img_pil.resize((opt.input_size, opt.input_size))
        
        # Transform Source -> Tensor (CPU) -> Tensor (GPU)
        source_tensor = processor.transform(s_pil).unsqueeze(0).to(device)
        # TRT expects float32 usually, make sure
        source_tensor = source_tensor.float()
        
        # Audio
        audio_tensor = processor.process_audio(aud_path).unsqueeze(0).to(device)
        
        # 2. Run Source Encoder (TRT)
        # Outputs: i_r, t_r, f_r_0..5, ma_r_0..3
        enc_outputs = self.source_encoder.infer([source_tensor])
        
        # t_r is at index 1
        t_r_tensor = enc_outputs[1]
        
        # 3. Run FMGenerator (PyTorch)
        # Ensure t_r shape
        if t_r_tensor.dim() == 1:
             t_r_tensor = t_r_tensor.unsqueeze(0)
        elif t_r_tensor.dim() == 3:
             t_r_tensor = t_r_tensor.squeeze(1)
        
        data = {
            's': source_tensor,
            'a': audio_tensor,
            'pose': None, 
            'cam': None,
            'gaze': None,
            'ref_x': t_r_tensor
        }
        
        # Calculate T
        T_est = int(audio_tensor.shape[-1] * opt.fps / opt.sampling_rate)
        
        torch.manual_seed(seed)
        
        # Update Generator FPS
        if hasattr(generator, 'fps'): generator.fps = opt.fps
        if hasattr(generator, 'audio_encoder') and hasattr(generator.audio_encoder, 'fps'):
             generator.audio_encoder.fps = opt.fps
             
        with torch.no_grad():
             sample = generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        
        T = sample.shape[1]
        print(f"[TRT] Motion generated: {T} frames")
        
        # 4. Render Loop (TRT)
        # renderer_static_inputs: [i_r] + enc_outputs[2:]
        renderer_static_inputs = [enc_outputs[0]] + enc_outputs[2:]
        
        d_hat = []
        
        for t in range(T):
            motion_code = sample[:, t, :].float() # [1, 32]
            
            # Inputs
            renderer_inputs = [motion_code] + renderer_static_inputs
            
            # Infer
            out_list = self.audio_renderer.infer(renderer_inputs)
            out_tensor = out_list[0] # (1, 3, 512, 512)
            
            d_hat.append(out_tensor)
            
        # Stack: [1, T, 3, H, W] after stacking list of [1, 3, H, W]
        vid_tensor = torch.stack(d_hat, dim=1).squeeze(0) # [T, 3, H, W]
        
        # Hand off to agent to save
        return self.agent.save_video(vid_tensor, opt.fps, aud_path)
