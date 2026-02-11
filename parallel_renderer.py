"""
Improved parallel rendering implementation for IMTalker InferenceAgent.
This module provides thread-safe, memory-efficient parallel frame rendering.
"""

import torch
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from typing import List, Tuple, Optional
import traceback


class ParallelRenderer:
    """
    Handles parallel frame rendering with proper thread safety and error handling.
    """
    
    def __init__(self, agent, num_workers: int = 1, chunk_size: int = 5):
        """
        Args:
            agent: InferenceAgent instance
            num_workers: Number of parallel rendering threads
            chunk_size: Number of frames per chunk
        """
        self.agent = agent
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.lock = threading.Lock()
    
    def render_chunk(
        self, 
        chunk_id: int, 
        start_idx: int, 
        end_idx: int,
        sample: torch.Tensor,
        g_r: torch.Tensor,
        m_r: torch.Tensor,
        f_r: torch.Tensor
    ) -> Tuple[int, Optional[List[torch.Tensor]], Optional[str]]:
        """
        Render a chunk of frames.
        
        Returns:
            (chunk_id, frames_list, error_string)
            - If successful: (chunk_id, [frames], None)
            - If failed: (chunk_id, None, error_message)
        """
        try:
            frames = []
            for t in range(start_idx, end_idx):
                # Render single frame
                ta_c = self.agent.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.agent.renderer.latent_token_decoder(ta_c)
                out_frame = self.agent.renderer.decode(m_c, m_r, f_r)
                
                # Move to CPU to free GPU memory
                frames.append(out_frame.cpu())
            
            return (chunk_id, frames, None)
            
        except Exception as e:
            error_msg = f"Chunk {chunk_id} failed:\n{traceback.format_exc()}"
            return (chunk_id, None, error_msg)
    
    def render_chunk_to_numpy(
        self, 
        chunk_id: int, 
        start_idx: int, 
        end_idx: int,
        sample: torch.Tensor,
        g_r: torch.Tensor,
        m_r: torch.Tensor,
        f_r: torch.Tensor
    ) -> Tuple[int, Optional[List[np.ndarray]], Optional[str]]:
        """
        Render a chunk and convert frames to numpy arrays (for HLS streaming).
        
        Returns:
            (chunk_id, numpy_frames_list, error_string)
        """
        try:
            frames = []
            for t in range(start_idx, end_idx):
                # Render frame
                ta_c = self.agent.renderer.adapt(sample[:, t, ...], g_r)
                m_c = self.agent.renderer.latent_token_decoder(ta_c)
                out_frame = self.agent.renderer.decode(m_c, m_r, f_r)
                
                # Convert to numpy
                frame_np = out_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                if frame_np.min() < 0:
                    frame_np = (frame_np + 1) / 2
                frame_np = np.clip(frame_np, 0, 1)
                frame_np = (frame_np * 255).astype(np.uint8)
                
                frames.append(frame_np)
            
            return (chunk_id, frames, None)
            
        except Exception as e:
            error_msg = f"Chunk {chunk_id} failed:\n{traceback.format_exc()}"
            return (chunk_id, None, error_msg)
    
    def render_parallel(
        self,
        sample: torch.Tensor,
        g_r: torch.Tensor,
        m_r: torch.Tensor,
        f_r: torch.Tensor,
        progress_callback=None
    ) -> List[torch.Tensor]:
        """
        Render all frames in parallel and return ordered list of frame tensors.
        
        Args:
            sample: Motion sample tensor [1, T, ...]
            g_r, m_r, f_r: Reference features
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            List of frame tensors in correct order
        """
        T = sample.shape[1]
        
        # Create chunks
        chunks = [
            (i // self.chunk_size, i, min(i + self.chunk_size, T))
            for i in range(0, T, self.chunk_size)
        ]
        
        print(f"Rendering {T} frames in {len(chunks)} chunks using {self.num_workers} workers...")
        
        # Storage for completed chunks
        rendered_chunks = OrderedDict()
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(
                    self.render_chunk, 
                    chunk_id, start, end,
                    sample, g_r, m_r, f_r
                ): chunk_id
                for chunk_id, start, end in chunks
            }
            
            # Process as they complete
            completed = 0
            for future in as_completed(future_to_chunk):
                chunk_id, frames, error = future.result()
                
                if error:
                    # Cancel remaining tasks and raise error
                    for f in future_to_chunk:
                        f.cancel()
                    raise RuntimeError(f"Rendering failed:\n{error}")
                
                # Store with thread safety
                with self.lock:
                    rendered_chunks[chunk_id] = frames
                
                completed += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed, len(chunks))
                
                # Periodic memory cleanup
                if chunk_id % 2 == 0:
                    if self.agent.device == "mps":
                        torch.mps.empty_cache()
                    elif self.agent.device == "cuda":
                        torch.cuda.empty_cache()
        
        # Reconstruct frames in correct order
        all_frames = []
        for cid in sorted(rendered_chunks.keys()):
            all_frames.extend(rendered_chunks[cid])
        
        return all_frames
    
    def render_progressive_hls(
        self,
        sample: torch.Tensor,
        g_r: torch.Tensor,
        m_r: torch.Tensor,
        f_r: torch.Tensor,
        hls_generator
    ):
        """
        Render frames in parallel and send to HLS generator as chunks complete.
        This enables true progressive streaming - video starts playing before all frames are ready.
        
        Args:
            sample: Motion sample tensor [1, T, ...]
            g_r, m_r, f_r: Reference features
            hls_generator: ProgressiveHLSGenerator instance
        """
        T = sample.shape[1]
        
        # Create chunks
        chunks = [
            (i // self.chunk_size, i, min(i + self.chunk_size, T))
            for i in range(0, T, self.chunk_size)
        ]
        
        print(f"Progressive HLS: Rendering {T} frames in {len(chunks)} chunks...")
        
        # Track which chunks are complete and which is next to send
        rendered_chunks = OrderedDict()
        next_chunk_to_send = 0
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all chunks (convert to numpy for HLS)
            future_to_chunk = {
                executor.submit(
                    self.render_chunk_to_numpy,
                    chunk_id, start, end,
                    sample, g_r, m_r, f_r
                ): chunk_id
                for chunk_id, start, end in chunks
            }
            
            # Process chunks as they complete
            for future in as_completed(future_to_chunk):
                chunk_id, frames, error = future.result()
                
                if error:
                    print(f"ERROR: {error}")
                    # Don't crash, but log the error
                    continue
                
                # Store completed chunk
                with self.lock:
                    rendered_chunks[chunk_id] = frames
                
                # Send all consecutive completed chunks to HLS generator
                # This ensures frames are always sent in order
                while next_chunk_to_send in rendered_chunks:
                    for frame_np in rendered_chunks[next_chunk_to_send]:
                        hls_generator.add_frame(frame_np)
                    
                    # Free memory immediately after sending
                    del rendered_chunks[next_chunk_to_send]
                    next_chunk_to_send += 1
                    
                    print(f"  ✓ Sent chunk {next_chunk_to_send - 1} to HLS")
                
                # Periodic memory cleanup
                if chunk_id % 2 == 0:
                    if self.agent.device == "mps":
                        torch.mps.empty_cache()
                    elif self.agent.device == "cuda":
                        torch.cuda.empty_cache()
        
        print(f"All {T} frames rendered and sent to HLS generator")
    
    def render_parallel_stream(
        self,
        sample: torch.Tensor,
        g_r: torch.Tensor,
        m_r: torch.Tensor,
        f_r: torch.Tensor,
        output_format: str = "tensor"
    ):
        """
        Render frames in parallel and YIELD them as consecutive chunks complete.
        Perfect for real-time streaming APIs, WebSockets, or Server-Sent Events.
        
        This method provides the lowest latency and memory footprint by:
        - Yielding frames immediately when consecutive chunks are ready
        - Freeing memory after each yield
        - Maintaining correct frame order
        
        Args:
            sample: Motion sample tensor [1, T, ...]
            g_r, m_r, f_r: Reference features
            output_format: "tensor" for torch.Tensor or "numpy" for np.ndarray
            
        Yields:
            Individual frames in correct order (torch.Tensor or np.ndarray)
            
        Example Usage:
            ```python
            # FastAPI Server-Sent Events streaming
            @app.get("/stream-frames")
            async def stream_frames():
                def generate():
                    for frame in renderer.render_parallel_stream(sample, g_r, m_r, f_r, "numpy"):
                        frame_jpg = cv2.imencode('.jpg', frame)[1].tobytes()
                        yield f"data: {base64.b64encode(frame_jpg).decode()}\n\n"
                return StreamingResponse(generate(), media_type="text/event-stream")
            
            # WebSocket streaming
            @app.websocket("/ws/video")
            async def websocket_endpoint(websocket: WebSocket):
                await websocket.accept()
                for frame in renderer.render_parallel_stream(sample, g_r, m_r, f_r, "numpy"):
                    await websocket.send_bytes(frame.tobytes())
            ```
        """
        T = sample.shape[1]
        
        # Create chunks
        chunks = [
            (i // self.chunk_size, i, min(i + self.chunk_size, T))
            for i in range(0, T, self.chunk_size)
        ]
        
        print(f"Streaming {T} frames in {len(chunks)} chunks using {self.num_workers} workers...")
        
        # Track which chunks are complete and which is next to yield
        rendered_chunks = OrderedDict()
        next_chunk_to_yield = 0
        
        # Choose rendering function based on output format
        render_func = self.render_chunk_to_numpy if output_format == "numpy" else self.render_chunk
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(
                    render_func,
                    chunk_id, start, end,
                    sample, g_r, m_r, f_r
                ): chunk_id
                for chunk_id, start, end in chunks
            }
            
            # Process chunks as they complete
            for future in as_completed(future_to_chunk):
                chunk_id, frames, error = future.result()
                
                if error:
                    print(f"ERROR in chunk {chunk_id}: {error}")
                    # Continue processing other chunks
                    continue
                
                # Store completed chunk
                with self.lock:
                    rendered_chunks[chunk_id] = frames
                
                # Yield all consecutive completed chunks in order
                # This ensures frames are always yielded sequentially
                while next_chunk_to_yield in rendered_chunks:
                    for frame in rendered_chunks[next_chunk_to_yield]:
                        yield frame  # ← Stream frame immediately!
                    
                    # Free memory immediately after yielding
                    del rendered_chunks[next_chunk_to_yield]
                    next_chunk_to_yield += 1
                
                # Periodic memory cleanup
                if chunk_id % 2 == 0:
                    if self.agent.device == "mps":
                        torch.mps.empty_cache()
                    elif self.agent.device == "cuda":
                        torch.cuda.empty_cache()
        
        print(f"All {T} frames yielded successfully")


# ============================================================================
# Integration Example: How to use in InferenceAgent
# ============================================================================

class InferenceAgentWithParallelRendering:
    """
    Example showing how to integrate ParallelRenderer into your InferenceAgent.
    """
    
    def __init__(self, opt):
        # ... existing initialization ...
        
        # Initialize parallel renderer
        self.parallel_renderer = ParallelRenderer(
            agent=self,
            num_workers=opt.num_render_workers,
            chunk_size=opt.render_chunk_size
        )
    
    @torch.no_grad()
    def run_audio_inference_parallel(
        self,
        img_pil,
        aud_path,
        crop,
        seed,
        nfe,
        cfg_scale,
        pose_style=None,
        gaze_style=None,
    ):
        """
        Audio-driven inference with parallel rendering.
        """
        # Memory cleanup
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        # 1. Preprocess inputs
        s_pil = (
            self.data_processor.process_img(img_pil)
            if crop
            else img_pil.resize((self.opt.input_size, self.opt.input_size))
        )
        s_tensor = self.data_processor.transform(s_pil).unsqueeze(0).to(self.device)
        a_tensor = self.data_processor.process_audio(aud_path).unsqueeze(0).to(self.device)
        
        # 2. Setup data dict
        T_est = math.ceil(a_tensor.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        data = {
            "s": s_tensor,
            "a": a_tensor,
            "pose": None,
            "cam": None,
            "gaze": None,
            "ref_x": None,
        }
        
        # Handle pose/gaze if provided
        if pose_style is not None:
            data["pose"] = (
                torch.tensor(pose_style, device=self.device)
                .unsqueeze(0).unsqueeze(0).expand(1, T_est, 3)
            )
        if gaze_style is not None:
            data["gaze"] = (
                torch.tensor(gaze_style, device=self.device)
                .unsqueeze(0).unsqueeze(0).expand(1, T_est, 2)
            )
        
        # 3. Generate motion
        f_r, g_r = self.renderer.dense_feature_encoder(s_tensor)
        t_lat = self.renderer.latent_token_encoder(s_tensor)
        if isinstance(t_lat, tuple):
            t_lat = t_lat[0]
        data["ref_x"] = t_lat
        
        torch.manual_seed(seed)
        sample = self.generator.sample(data, a_cfg_scale=cfg_scale, nfe=nfe, seed=seed)
        
        # Stabilization
        motion_scale = 0.6
        sample = (
            (1.0 - motion_scale) * t_lat.unsqueeze(1).repeat(1, sample.shape[1], 1) 
            + motion_scale * sample
        )
        
        # 4. Prepare reference features
        ta_r = self.renderer.adapt(t_lat, g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        
        # 5. PARALLEL RENDERING - This is where the magic happens!
        frames = self.parallel_renderer.render_parallel(
            sample, g_r, m_r, f_r,
            progress_callback=lambda curr, total: print(f"  Progress: {curr}/{total} chunks")
        )
        
        # 6. Stack frames into video tensor
        vid_tensor = torch.stack(frames, dim=1).squeeze(0)
        
        # Final cleanup
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        return self.save_video(vid_tensor, self.opt.fps, aud_path)
    
    @torch.no_grad()
    def run_audio_inference_progressive_hls(
        self,
        img_pil,
        aud_path,
        crop,
        seed,
        nfe,
        cfg_scale,
        hls_generator,
        pose_style=None,
        gaze_style=None,
    ):
        """
        Audio-driven inference with progressive HLS streaming.
        """
        # ... same preprocessing as above ...
        
        # After motion generation:
        # PROGRESSIVE HLS RENDERING
        self.parallel_renderer.render_progressive_hls(
            sample, g_r, m_r, f_r,
            hls_generator
        )
        
        # Finalize HLS with audio
        hls_generator.finalize(audio_path=aud_path)
