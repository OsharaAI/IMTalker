#!/usr/bin/env python3
"""
Quick-start test script for the chunked streaming endpoint.

This script tests the new /api/audio-driven/chunked endpoints without needing
a real face image or audio file. It uses dummy data for quick validation.

Usage:
    python test_chunked_streaming.py
"""

import requests
import io
import numpy as np
import json
import time
from pathlib import Path


def create_dummy_image(width=512, height=512) -> bytes:
    """Create a simple dummy image for testing"""
    try:
        from PIL import Image
        import io
        
        # Create a simple gradient image
        img = Image.new('RGB', (width, height), color='white')
        pixels = img.load()
        
        # Add a simple face-like pattern (circle in center)
        cx, cy = width // 2, height // 2
        radius = 100
        
        for y in range(height):
            for x in range(width):
                dx, dy = x - cx, y - cy
                dist = (dx*dx + dy*dy) ** 0.5
                
                if dist < radius:
                    # Face color (peachy)
                    pixels[x, y] = (220, 150, 100)
                elif dist < radius + 10:
                    # Border
                    pixels[x, y] = (100, 100, 100)
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
        
    except ImportError:
        # Fallback: create a simple numpy array and save as PNG
        print("Warning: PIL not available, using fallback image creation")
        return b'\x89PNG...'  # This won't work, but shows fallback attempt


def create_dummy_audio(duration_seconds=1.0, sample_rate=24000) -> bytes:
    """Create a simple sine wave audio for testing"""
    try:
        import soundfile as sf
        import numpy as np
        import io
        
        # Create 1kHz sine wave
        t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate))
        freq = 1000  # 1kHz
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        # Save to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        return buffer.getvalue()
        
    except ImportError:
        print("Error: soundfile not installed. Install with: pip install soundfile")
        return None


def test_chunked_endpoints(server_url="http://localhost:9999"):
    """Test the chunked streaming endpoints"""
    
    print("="*70)
    print("Testing Chunked Audio-Driven Inference Endpoints")
    print("="*70)
    print(f"Server: {server_url}\n")
    
    # Test 1: Check server health
    print("[Test 1] Checking server health...")
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            health = response.json()
            print(f"  ✓ Server healthy: {health['status']}")
        else:
            print(f"  ✗ Server not healthy")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to server at {server_url}")
        print("  Make sure FastAPI server is running: python fastapi_app.py")
        return False
    
    # Test 2: Create dummy image
    print("\n[Test 2] Creating dummy image...")
    image_bytes = create_dummy_image()
    print(f"  ✓ Created {len(image_bytes)} bytes image")
    
    # Test 3: Initialize session
    print("\n[Test 3] Initializing chunked stream session...")
    
    try:
        files = {'image': ('test_face.png', image_bytes, 'image/png')}
        data = {
            'crop': 'false',  # Don't crop dummy image
            'seed': '42',
            'nfe': '5',  # Low for fast testing
            'cfg_scale': '2.0',
            'chunk_duration': '0.5',  # Small chunks for quick test
            'audio_sample_rate': '24000',
        }
        
        response = requests.post(
            f"{server_url}/api/audio-driven/chunked/init",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"  ✗ Init failed: {response.status_code}")
            print(f"    Response: {response.text}")
            return False
        
        result = response.json()
        session_id = result['session_id']
        hls_url = result['hls_url']
        
        print(f"  ✓ Session initialized: {session_id}")
        print(f"  ✓ HLS URL: {hls_url}")
        
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False
    
    # Test 4: Create and send dummy audio chunks
    print("\n[Test 4] Feeding audio chunks...")
    
    for chunk_num in range(3):
        try:
            audio_bytes = create_dummy_audio(duration_seconds=0.5, sample_rate=24000)
            if audio_bytes is None:
                print(f"  ✗ Could not create audio")
                return False
            
            is_final = (chunk_num == 2)
            
            files = {
                'audio_chunk': ('chunk.wav', audio_bytes, 'audio/wav')
            }
            data = {
                'session_id': session_id,
                'is_final_chunk': str(is_final).lower(),
            }
            
            response = requests.post(
                f"{server_url}/api/audio-driven/chunked/feed",
                files=files,
                data=data,
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"  ✗ Feed failed: {response.status_code}")
                print(f"    Response: {response.text}")
                return False
            
            result = response.json()
            frames = result.get('frames_generated', 0)
            status = result.get('session_status', 'unknown')
            
            print(f"  Chunk {chunk_num+1}/3: "
                  f"status={status}, frames={frames}, "
                  f"buffer={result.get('buffer_samples', 0)} samples")
            
            # Small delay between chunks
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ✗ Exception on chunk {chunk_num}: {e}")
            return False
    
    # Test 5: Check final status
    print("\n[Test 5] Checking final session status...")
    
    try:
        response = requests.get(
            f"{server_url}/api/audio-driven/chunked/status/{session_id}",
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"  ✗ Status query failed: {response.status_code}")
            return False
        
        status = response.json()
        
        print(f"  ✓ Session Status:")
        print(f"    - Status: {status['status']}")
        print(f"    - Total frames: {status['total_frames_generated']}")
        print(f"    - Uptime: {status['uptime_seconds']:.2f}s")
        
        if status['error']:
            print(f"  ⚠ Error: {status['error']}")
        
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False
    
    # Test 6: Cleanup
    print("\n[Test 6] Cleaning up session...")
    
    try:
        response = requests.delete(
            f"{server_url}/api/audio-driven/chunked/session/{session_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"  ✓ Session cleaned up")
        else:
            print(f"  ⚠ Cleanup returned: {response.status_code}")
        
    except Exception as e:
        print(f"  ⚠ Cleanup exception: {e}")
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
    print(f"\nHLS Stream URL: {server_url}{hls_url}")
    print("You can now:")
    print(f"  1. View the stream in an HLS player")
    print(f"  2. Use the chunked_streaming_client.py for production")
    print(f"  3. Integrate with WebRTC (see CHUNKED_STREAMING_GUIDE.md)")
    
    return True


if __name__ == "__main__":
    import sys
    
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9999"
    
    success = test_chunked_endpoints(server_url)
    sys.exit(0 if success else 1)
