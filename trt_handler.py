import tensorrt as trt
import torch
import numpy as np
from collections import OrderedDict
import os
import cv2
from PIL import Image
import torchvision.transforms as transforms

class TensorRTModel:
    """
    Generic TensorRT Model Wrapper for PyTorch.
    Handles engine loading, context creation, and inference.
    """

    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream()

        # Parse inputs and outputs
        self.inputs = []
        self.outputs = []
        self.bindings = []

        # Keep references to input tensors to prevent GC during async execution
        self.input_refs = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            # dtype = self.engine.get_tensor_dtype(name) # Only available in recent TRT
            # shape = self.engine.get_tensor_shape(name)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            if is_input:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

    def _load_engine(self):
        with open(self.engine_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def infer(self, input_tensors):
        """
        Run inference.
        Args:
            input_tensors: List of PyTorch tensors in the order of engine inputs,
                           OR Dictionary {name: tensor}.
        Returns:
            Dictionary {name: tensor} of outputs.
        """
        self.bindings = []
        self.input_refs = []  # Clear previous refs

        # Handle dict input
        if isinstance(input_tensors, dict):
            # Sort inputs according to engine expectation
            sorted_inputs = []
            for name in self.inputs:
                if name in input_tensors:
                    sorted_inputs.append(input_tensors[name])
                else:
                    raise ValueError(f"Missing input tensor: {name}")
            input_tensors = sorted_inputs

        if len(input_tensors) != len(self.inputs):
            raise ValueError(
                f"Expected {len(self.inputs)} inputs, got {len(input_tensors)}"
            )

        # Bind inputs
        for i, tensor in enumerate(input_tensors):
            # Ensure tensor is contiguous and on device
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            tensor = tensor.contiguous()
            self.input_refs.append(tensor)  # Keep ref

            name = self.inputs[i]
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Bind outputs
        output_tensors = {}
        for name in self.outputs:
            shape = self.context.get_tensor_shape(name)
            # Handle dynamic shapes if needed (usually -1) - for now assume static or inferred
            # If shape has -1, we might need to deduce it?
            # In update, set_input_shape usually resolves output shapes.

            dtype = torch.float32  # Default, check engine if possible.
            # For now assuming float32. Improved: check engine dtype if needed.
            # Most IMTalker models are float32/float16. PyTorch can allocate usually.

            tensor = torch.empty(tuple(shape), device="cuda", dtype=dtype)
            self.context.set_tensor_address(name, tensor.data_ptr())
            output_tensors[name] = tensor

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        return output_tensors


class TRTInferenceHandler:
    """
    High-level handler for IMTalker TensorRT inference.
    Manages SourceEncoder and AudioRenderer engines.
    """

    def __init__(self, agent, engine_dir="trt_engines"):
        self.agent = agent
        self.device = agent.device
        self.source_encoder = None
        self.audio_renderer = None

        if not os.path.isabs(engine_dir):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            engine_dir = os.path.join(base_dir, engine_dir)

        source_path = os.path.join(engine_dir, "source_encoder.trt")
        audio_path = os.path.join(engine_dir, "audio_renderer.trt")

        if os.path.exists(source_path):
            print(f"Loading SourceEncoder from {source_path}...")
            self.source_encoder = TensorRTModel(source_path)

        if os.path.exists(audio_path):
            print(f"Loading AudioRenderer from {audio_path}...")
            self.audio_renderer = TensorRTModel(audio_path)

    def is_available(self):
        return self.source_encoder is not None and self.audio_renderer is not None

    def run_inference(self, img_pil, audio_path, crop, seed, nfe, cfg_scale):
        # 1. Preprocessing (PyTorch)
        if crop:
            img = self.agent.data_processor.process_img(img_pil)
            full_img_tensor = None
            coords = None
        else:
            # Crop-Infer-Paste Logic
            img_arr = np.array(img_pil)
            if img_arr.ndim == 2: img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
            elif img_arr.shape[2] == 4: img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
            
            x1, y1, x2, y2 = self.agent.data_processor.get_crop_coords(img_arr)
            coords = (x1, y1, x2, y2)
            
            crop_img = img_arr[y1:y2, x1:x2]
            img = Image.fromarray(crop_img).resize((self.agent.opt.input_size, self.agent.opt.input_size))
            
            # Prepare full image tensor on GPU for paste-back
            full_img_tensor = transforms.ToTensor()(img_pil).to(self.device).unsqueeze(0)
            
        s_tensor = self.agent.data_processor.transform(img).unsqueeze(0).to(self.device)

        # 2. Source Encoding (TensorRT)
        # Inputs: source_image
        # Outputs: i_r, t_r, f_r_0..5, ma_r_0..3
        print("Running Source Encoder (TRT)...")
        enc_outputs = self.source_encoder.infer([s_tensor])

        # Extract outputs by name
        # Names are defined in export_onnx.py: i_r, t_r, f_r_0...

        i_r = enc_outputs["i_r"]
        t_r = enc_outputs["t_r"]

        f_r = [enc_outputs[f"f_r_{i}"] for i in range(6)]
        ma_r = [enc_outputs[f"ma_r_{i}"] for i in range(4)]

        # 3. Audio Processing (PyTorch)
        # We need Motion Generator inputs.
        # Data preparation
        data = {
            "s": s_tensor,
            "a": None,  # Filled later
            "pose": None,
            "cam": None,
            "gaze": None,
            "ref_x": t_r,  # t_lat is ref_x
        }

        # Process Audio
        a_tensor = (
            self.agent.data_processor.process_audio(audio_path)
            .unsqueeze(0)
            .to(self.device)
        )
        data["a"] = a_tensor

        # 4. Motion Generation (PyTorch)
        # This part is complex and uses the Generator model which is not exported yet effectively.
        # We use the existing PyTorch Generator for this step.
        print("Generating Motion (PyTorch)...")
        if seed is not None:
            torch.manual_seed(seed)

        # NOTE: generator.sample returns (sample, context)
        sample, _ = self.agent.generator.sample(
            data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed
        )

        # Stabilization (Standard logic)
        motion_scale = 0.6
        sample = (1.0 - motion_scale) * t_r.unsqueeze(1).repeat(
            1, sample.shape[1], 1
        ) + motion_scale * sample

        # 5. Rendering (TensorRT)
        # Loop mainly due to memory or structure, but AudioRenderer takes ONE frame motion code?
        # No, AudioRenderer in ONNX export (AudioRenderer class) takes:
        # (motion_code, i_r, f_r..., ma_r...) -> output_image

        # We need to run this for every frame t in sample.
        T = sample.shape[1]
        print(f"Rendering {T} frames (TRT)...")

        import time
        from tqdm import tqdm

        frames = []
        t_render_start = time.time()

        for t in tqdm(range(T), desc="Rendering", unit="frame"):
            motion_code = sample[:, t, ...]  # (1, 32)

            # Inputs: motion_code, i_r, f_r_0..5, ma_r_0..3
            # We must pass them as list or dict.
            trt_inputs = {"motion_code": motion_code, "i_r": i_r}
            for i in range(6):
                trt_inputs[f"f_r_{i}"] = f_r[i]
            for i in range(4):
                trt_inputs[f"ma_r_{i}"] = ma_r[i]

            out_dict = self.audio_renderer.infer(trt_inputs)
            out_frame = out_dict['output_image']
            
            if full_img_tensor is not None:
                # Need to handle dimensions carefully. TRT output is typical (1, 3, 512, 512).
                # paste_back_tensor expects (1, 3, 512, 512) for face and (1, 3, H, W) for full.
                # It handles it.
                out_frame = self.agent.data_processor.paste_back_tensor(out_frame, full_img_tensor, coords)
            
            frames.append(out_frame.cpu())

        t_render_end = time.time()
        render_time = t_render_end - t_render_start
        render_fps = T / render_time if render_time > 0 else 0
        print(
            f"Rendering completed: {render_fps:.2f} fps ({render_time:.2f}s for {T} frames)"
        )

        # 6. Post-processing & Saving (PyTorch)
        vid_tensor = torch.stack(frames, dim=1).squeeze(0)  # (T, 3, H, W)

        # Use existing save_video
        return self.agent.save_video(vid_tensor, self.agent.opt.fps, audio_path)
