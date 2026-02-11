import requests
import json
import time

def test_streaming():
    url = "http://localhost:8000/generate_stream"
    payload = {
        "text": "This is a test of the optimized streaming pipeline. We want to see if the frames and audio are delivered without long gaps between segments.",
        "avatar": "assets/source_1.png"
    }
    
    print(f"Connecting to {url}...")
    start_time = time.time()
    try:
        with requests.post(url, json=payload, stream=True) as r:
            r.raise_for_status()
            print("Connected. Reading stream...")
            
            chunk_idx = 0
            for line in r.iter_lines():
                if line:
                    chunk_idx += 1
                    data = json.loads(line)
                    metrics = data.get("metrics", {})
                    frames_count = len(data.get("frames", []))
                    audio_len = len(data.get("audio", ""))
                    
                    elapsed = time.time() - start_time
                    print(f"Chunk {chunk_idx}: Received {frames_count} frames, audio len {audio_len}. "
                          f"Elapsed: {elapsed:.2f}s, RTF: {metrics.get('rtf', 'N/A')}")
                    
                    # We don't need to process many chunks for timing verification
                    if chunk_idx >= 5:
                        break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_streaming()
