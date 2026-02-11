import threading
import time
import queue
import numpy as np
import torch
import asyncio
import struct
import logging
from typing import Optional, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealtimeStreaming")

class RealtimeSession:
    def __init__(self, 
                 session_id: str, 
                 inference_agent: Any, 
                 img_pil: Any, 
                 crop: bool = True,
                 target_fps: int = 25,
                 audio_sample_rate: int = 16000,
                 output_callback: callable = None):
        
        self.session_id = session_id
        self.agent = inference_agent
        self.target_fps = target_fps
        self.sample_rate = audio_sample_rate
        self.frame_duration = 1.0 / target_fps
        self.samples_per_frame = int(self.sample_rate * self.frame_duration)
        self.output_callback = output_callback
        
        # Audio Buffer (Thread safe)
        self.audio_queue = queue.Queue()
        self.current_audio_buffer = np.array([], dtype=np.float32)
        
        # State
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Preprocess image once
        logger.info(f"[{session_id}] Preprocessing image...")
        self.processed_data = self.agent.preprocess_for_realtime(img_pil, crop)
        logger.info(f"[{session_id}] Ready.")

    def push_audio(self, audio_chunk: np.ndarray):
        """Add audio data to the buffer. Expects float32 or int16 numpy array."""
        if not self.running:
            return
            
        with self.lock:
            # Normalize to float32 -1..1
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            
            # Simple queueing - in a real production system, might want a circular buffer to avoid fragmentation
            # But for Python queue of arrays is okay usually
            self.audio_queue.put(audio_chunk)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.thread.start()
        logger.info(f"[{self.session_id}] Streaming loop started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info(f"[{self.session_id}] Streaming loop stopped.")

    def _get_next_audio_frame(self) -> Tuple[np.ndarray, bool]:
        """
        Get exactly samples_per_frame audio.
        If buffer is empty or partial, pad with zeros (silence).
        Returns (audio_frame, is_silence_filled)
        """
        needed = self.samples_per_frame
        out_audio = np.array([], dtype=np.float32)
        
        # Try to fill from current buffer remnants
        if len(self.current_audio_buffer) > 0:
            take = min(len(self.current_audio_buffer), needed)
            out_audio = self.current_audio_buffer[:take]
            self.current_audio_buffer = self.current_audio_buffer[take:]
            needed -= take
            
        # If still need more, fetch from queue
        while needed > 0:
            try:
                # Non-blocking get with very short timeout? 
                # Actually we can't block long because we need to maintain FPS.
                # If queue is empty, we MUST generate silence immediately.
                chunk = self.audio_queue.get_nowait()
                
                take = min(len(chunk), needed)
                out_audio = np.concatenate([out_audio, chunk[:take]])
                
                if len(chunk) > take:
                    # Save the rest
                    self.current_audio_buffer = chunk[take:]
                
                needed -= take
            except queue.Empty:
                break
        
        is_silence = False
        if needed > 0:
            # Fill the rest with silence
            padding = np.zeros(needed, dtype=np.float32) # Silence
            if len(out_audio) > 0:
                out_audio = np.concatenate([out_audio, padding])
            else:
                out_audio = padding
            is_silence = True
            
        return out_audio, is_silence

    def _streaming_loop(self):
        """
        Main heartbeat loop.
        """
        next_tick = time.time()
        
        # Initial context warmup
        self.agent.warmup_realtime()
        
        while self.running:
            now = time.time()
            # Wait for next tick
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Only warn if significantly behind (> 10ms)
                if sleep_time < -0.01:
                    logger.debug(f"[{self.session_id}] Behind schedule by {-sleep_time*1000:.1f}ms")
            
            # Update next tick target
            next_tick += self.frame_duration
            # If we fell way behind, reset the clock to avoid burst catch-up
            if time.time() > next_tick + 0.1:
                next_tick = time.time() + self.frame_duration
            
            # 1. Get Audio
            audio_frame, is_silence = self._get_next_audio_frame()
            
            # 2. Inference
            start_infer = time.time()
            video_frame_jpeg = self.agent.run_realtime_step(
                self.processed_data, 
                audio_frame, 
                is_silence
            )
            infer_dur = time.time() - start_infer
            
            # 3. Output
            if self.output_callback:
                timestamp_us = int(time.time() * 1000000)
                
                # Protocol: 
                # Video Message
                self.output_callback(
                    type="video",
                    payload=video_frame_jpeg,
                    timestamp=timestamp_us
                )
                
                # Audio Echo (The stream we actually used)
                # Convert back to PCM 16-bit for transport efficiency usually, 
                # or keep float if requested. Let's do PCM16.
                audio_pcm16 = (audio_frame * 32767).astype(np.int16).tobytes()
                self.output_callback(
                    type="audio",
                    payload=audio_pcm16,
                    timestamp=timestamp_us
                )
