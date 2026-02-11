
import os
import sys
import torch
import numpy as np
import librosa
from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from imtalker_streamer import IMTalkerStreamer
from app import AppConfig # re-use config from app.py

def create_audio_chunk_iterator(audio_path, chunk_duration_sec=0.2, sampling_rate=16000):
    """
    Yields audio chunks from a file.
    """
    print(f"Loading audio from {audio_path}...")
    speech_array, _ = librosa.load(audio_path, sr=sampling_rate)
    
    chunk_size = int(sampling_rate * chunk_duration_sec)
    total_len = len(speech_array)
    
    print(f"Total audio length: {total_len/sampling_rate:.2f}s")
    print(f"Chunk duration: {chunk_duration_sec}s ({chunk_size} samples)")
    
    for i in range(0, total_len, chunk_size):
        yield speech_array[i : i + chunk_size]

def main():
    # Setup Config
    cfg = AppConfig()
    
    # Initialize Streamer
    streamer = IMTalkerStreamer(cfg)
    
    # Inputs
    # Adjust paths as needed based on where this script is run
    base_dir = os.path.dirname(os.path.abspath(__file__))
    avatar_path = os.path.join(base_dir, "assets", "source_1.png")
    
    # Use a dummy audio or existing one
    # I'll look for an audio file in assets or root
    audio_candidates = [
        os.path.join(base_dir, "assets", "audio_1.wav"),
        "test-1.wav",
        "../test-1.wav"
    ]
    
    audio_path = None
    for p in audio_candidates:
        if os.path.exists(p):
            audio_path = p
            break
            
    if not audio_path:
        print("No audio file found. Generating dummy audio.")
        # Generate 2 seconds of silence/noise
        sr = 16000
        dummy_audio = np.random.uniform(-0.1, 0.1, sr * 2).astype(np.float32)
        import soundfile as sf
        audio_path = "dummy_test_audio.wav"
        sf.write(audio_path, dummy_audio, sr)
    
    if not os.path.exists(avatar_path):
        print(f"Avatar not found at {avatar_path}, using dummy.")
        img = Image.new('RGB', (512, 512), color = 'red')
    else:
        print(f"Using avatar: {avatar_path}")
        img = Image.open(avatar_path)
    
    print(f"Using audio: {audio_path}")
    
    # Run Streaming
    print("\nStarting Streaming Test...")
    audio_iterator = create_audio_chunk_iterator(audio_path, chunk_duration_sec=0.5)
    
    total_chunks = 0
    
    try:
        for frames, metrics in streamer.generate_stream(img, audio_iterator):
            total_chunks += 1
            print(f"Received chunk {metrics['chunk_id']}: {metrics['num_frames']} frames, "
                  f"Gen Time: {metrics['generation_time']:.4f}s, "
                  f"RTF: {metrics['rtf']:.4f}")
            
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    print("Test finished.")

if __name__ == "__main__":
    main()
