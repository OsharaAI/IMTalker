import argparse
import os
import torch
import torch.nn as nn
from renderer.models import IMTRenderer

class IMTRendererWrapper(nn.Module):
    """
    Wrapper to export IMTRenderer as a single ONNX model.
    Inputs:
        source_image: (1, 3, 512, 512)
        driving_motion: (1, 32)
    Outputs:
        output_image: (1, 3, 512, 512)
    """
    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer

    def forward(self, source_image, driving_motion):
        # 1. Encode Source
        f_r, i_r = self.renderer.app_encode(source_image)
        t_r = self.renderer.mot_encode(source_image)
        
        # 2. Adapt Source Motion
        ta_r = self.renderer.adapt(t_r, i_r)
        
        # 3. Decode Source Motion (Global Reference)
        ma_r = self.renderer.mot_decode(ta_r)
        
        # 4. Adapt Driving Motion (we assume driving_motion is already encoded t_c)
        # Wait, the Generator usually takes 'source_image' and 'driving_image' or 'driving_motion'.
        # In inference_batch.py: 
        #   t_c = self.gen.mot_encode(batch_tensor) -> but batch_tensor is driving_frame
        #   ta_c = self.gen.adapt(t_c, i_r_batch)
        #   ma_c = self.gen.mot_decode(ta_c)
        #   out = self.gen.decode(ma_c, ma_r_batch, f_r_batch)
        
        # To make a monolithic export useful, we should probably input:
        #   source_image (static, can be pre-computed but let's keep it simple first)
        #   driving_image (1, 3, 512, 512)
        
        # Let's export the Full Pipeline: Source + Driving -> Output
        # Inputs: source (1,3,H,W), driving (1,3,H,W)
        pass

    # Actually, for efficiency, we want to pre-compute Source Features once.
    # So we should export 2 models:
    # Model A: Source Encoder (Source -> f_r, i_r, ma_r) [Run Once]
    # Model B: Driving Generator (Driving Image + f_r, i_r, ma_r -> Output) [Run Per Frame]

class SourceEncoder(nn.Module):
    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer
    
    def forward(self, source_image):
        f_r, i_r = self.renderer.app_encode(source_image)
        t_r = self.renderer.mot_encode(source_image)
        ta_r = self.renderer.adapt(t_r, i_r)
        ma_r = self.renderer.mot_decode(ta_r)
        
        # f_r is a list of tensors. ma_r is tuple/list of tensors. i_r is tensor.
        # t_r is needed for FMGenerator in the inference pipeline.
        return i_r, t_r, f_r[0], f_r[1], f_r[2], f_r[3], f_r[4], f_r[5], ma_r[0], ma_r[1], ma_r[2], ma_r[3]

class AudioRenderer(nn.Module):
    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer
        
    def forward(self, motion_code, i_r, f_r_0, f_r_1, f_r_2, f_r_3, f_r_4, f_r_5, ma_r_0, ma_r_1, ma_r_2, ma_r_3):
        # Reconstruct lists
        f_r = [f_r_0, f_r_1, f_r_2, f_r_3, f_r_4, f_r_5]
        ma_r = [ma_r_0, ma_r_1, ma_r_2, ma_r_3]
        
        # 1. Adapt Motion (Input is already motion code t_c)
        ta_c = self.renderer.adapt(motion_code, i_r)
        
        # 2. Decode Motion
        ma_c = self.renderer.mot_decode(ta_c)
        
        # 3. Synthesize
        out = self.renderer.decode(ma_c, ma_r, f_r)
        return out

def export(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Renderer
    renderer = IMTRenderer(args).to(device)
    checkpoint = torch.load(args.renderer_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    clean_state_dict = {k.replace("gen.", ""): v for k, v in state_dict.items() if k.startswith("gen.")}
    renderer.load_state_dict(clean_state_dict, strict=False)
    renderer.eval()

    # Create directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dummy Inputs
    dummy_source = torch.randn(1, 3, args.input_size, args.input_size).to(device)
    dummy_motion = torch.randn(1, 32).to(device) # Motion code is 32-dim
    
    # 1. Export Source Encoder
    print("Exporting Source Encoder...")
    source_enc = SourceEncoder(renderer)
    
    # Run once to initialize any buffers
    source_enc(dummy_source)
    
    input_names = ["source_image"]
    output_names = ["i_r", "t_r"] + [f"f_r_{i}" for i in range(6)] + [f"ma_r_{i}" for i in range(4)]
    
    torch.onnx.export(
        source_enc,
        (dummy_source,),
        os.path.join(args.output_dir, "source_encoder.onnx"),
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
        do_constant_folding=False
    )
    
    # 2. Export Audio Renderer
    print("Exporting Audio Renderer...")
    audio_renderer = AudioRenderer(renderer)
    
    # Get dummy outputs from source encoder to use as input
    with torch.no_grad():
        out_tuple = source_enc(dummy_source)
    
    # AudioRenderer takes (motion_code, i_r, f_r..., ma_r...)
    # out_tuple is (i_r, t_r, f_r..., ma_r...)
    # We need to skip t_r (index 1)
    dummy_inputs = (dummy_motion, out_tuple[0]) + out_tuple[2:]
    
    input_names = ["motion_code", "i_r"] + [f"f_r_{i}" for i in range(6)] + [f"ma_r_{i}" for i in range(4)]
    output_names = ["output_image"]
    
    torch.onnx.export(
        audio_renderer,
        dummy_inputs,
        os.path.join(args.output_dir, "audio_renderer.onnx"),
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
        do_constant_folding=False
    )
    
    print("Export completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer_path", type=str, default="./checkpoints/renderer.ckpt")
    parser.add_argument("--output_dir", type=str, default="./onnx_models")
    parser.add_argument("--input_size", type=int, default=512)
    # Model args needed for initialization
    parser.add_argument('--swin_res_threshold', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--window_size', type=int, default=8)
    
    args = parser.parse_args()
    export(args)
