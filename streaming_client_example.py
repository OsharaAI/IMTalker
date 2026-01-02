"""
Example client for consuming streaming video from IMTalker FastAPI server.
Demonstrates how to receive and display/save streaming MJPEG frames.
"""

import requests
import cv2
import numpy as np
import argparse
from pathlib import Path


def stream_audio_driven_mjpeg(
    server_url: str,
    image_path: str,
    audio_path: str,
    output_video_path: str = None,
    crop: bool = True,
    seed: int = 42,
    nfe: int = 10,
    cfg_scale: float = 2.0,
    display: bool = True
):
    """
    Stream audio-driven video generation and optionally save to file.
    
    Args:
        server_url: Base URL of the FastAPI server (e.g., "http://localhost:9999")
        image_path: Path to source image
        audio_path: Path to driving audio
        output_video_path: Optional path to save output video
        crop: Whether to crop face
        seed: Random seed
        nfe: Number of function evaluations
        cfg_scale: Guidance scale
        display: Whether to display frames in real-time
    """
    url = f"{server_url}/api/v1/audio-driven/stream"
    
    # Prepare multipart form data
    files = {
        'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg'),
        'audio': ('audio.wav', open(audio_path, 'rb'), 'audio/wav')
    }
    
    data = {
        'crop': str(crop).lower(),
        'seed': seed,
        'nfe': nfe,
        'cfg_scale': cfg_scale,
        'format': 'mjpeg'
    }
    
    print(f"Connecting to {url}...")
    print(f"Parameters: {data}")
    
    # Make streaming request
    response = requests.post(url, files=files, data=data, stream=True)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return
    
    print("Streaming started...")
    print(f"Content-Type: {response.headers.get('content-type')}")
    
    # Initialize video writer if saving
    video_writer = None
    frame_count = 0
    
    try:
        # Parse MJPEG stream
        bytes_data = b''
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            
            # Find JPEG boundaries
            a = bytes_data.find(b'\xff\xd8')  # JPEG start
            b = bytes_data.find(b'\xff\xd9')  # JPEG end
            
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                # Decode JPEG frame
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    frame_count += 1
                    print(f"Received frame {frame_count}: {frame.shape}", end='\r')
                    
                    # Initialize video writer on first frame
                    if output_video_path and video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (w, h))
                        print(f"\nSaving video to {output_video_path}")
                    
                    # Write frame to video
                    if video_writer:
                        video_writer.write(frame)
                    
                    # Display frame
                    if display:
                        cv2.imshow('Streaming Video', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nStopped by user")
                            break
    
    finally:
        # Clean up
        if video_writer:
            video_writer.release()
            print(f"\n\nVideo saved: {output_video_path}")
        
        if display:
            cv2.destroyAllWindows()
        
        print(f"\nTotal frames received: {frame_count}")


def stream_video_driven_mjpeg(
    server_url: str,
    image_path: str,
    video_path: str,
    output_video_path: str = None,
    crop: bool = False,
    display: bool = True
):
    """
    Stream video-driven video generation and optionally save to file.
    
    Args:
        server_url: Base URL of the FastAPI server
        image_path: Path to source image
        video_path: Path to driving video
        output_video_path: Optional path to save output video
        crop: Whether to crop face
        display: Whether to display frames in real-time
    """
    url = f"{server_url}/api/v1/video-driven/stream"
    
    # Prepare multipart form data
    files = {
        'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg'),
        'video': ('video.mp4', open(video_path, 'rb'), 'video/mp4')
    }
    
    data = {
        'crop': str(crop).lower(),
        'format': 'mjpeg'
    }
    
    print(f"Connecting to {url}...")
    print(f"Parameters: {data}")
    
    # Make streaming request
    response = requests.post(url, files=files, data=data, stream=True)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return
    
    print("Streaming started...")
    
    # Initialize video writer if saving
    video_writer = None
    frame_count = 0
    
    try:
        # Parse MJPEG stream
        bytes_data = b''
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            
            # Find JPEG boundaries
            a = bytes_data.find(b'\xff\xd8')  # JPEG start
            b = bytes_data.find(b'\xff\xd9')  # JPEG end
            
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                # Decode JPEG frame
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    frame_count += 1
                    print(f"Received frame {frame_count}: {frame.shape}", end='\r')
                    
                    # Initialize video writer on first frame
                    if output_video_path and video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (w, h))
                        print(f"\nSaving video to {output_video_path}")
                    
                    # Write frame to video
                    if video_writer:
                        video_writer.write(frame)
                    
                    # Display frame
                    if display:
                        cv2.imshow('Streaming Video', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nStopped by user")
                            break
    
    finally:
        # Clean up
        if video_writer:
            video_writer.release()
            print(f"\n\nVideo saved: {output_video_path}")
        
        if display:
            cv2.destroyAllWindows()
        
        print(f"\nTotal frames received: {frame_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMTalker Streaming Client Example")
    parser.add_argument("--server", type=str, default="http://localhost:9999", help="Server URL")
    parser.add_argument("--mode", type=str, choices=["audio", "video"], required=True, help="Generation mode")
    parser.add_argument("--image", type=str, required=True, help="Path to source image")
    parser.add_argument("--audio", type=str, help="Path to driving audio (for audio mode)")
    parser.add_argument("--video", type=str, help="Path to driving video (for video mode)")
    parser.add_argument("--output", type=str, help="Path to save output video (optional)")
    parser.add_argument("--crop", action="store_true", help="Crop face from source")
    parser.add_argument("--no-display", action="store_true", help="Don't display video in real-time")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (audio mode)")
    parser.add_argument("--nfe", type=int, default=10, help="Number of steps (audio mode)")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Guidance scale (audio mode)")
    
    args = parser.parse_args()
    
    if args.mode == "audio":
        if not args.audio:
            print("Error: --audio required for audio mode")
            exit(1)
        
        stream_audio_driven_mjpeg(
            server_url=args.server,
            image_path=args.image,
            audio_path=args.audio,
            output_video_path=args.output,
            crop=args.crop,
            seed=args.seed,
            nfe=args.nfe,
            cfg_scale=args.cfg_scale,
            display=not args.no_display
        )
    
    elif args.mode == "video":
        if not args.video:
            print("Error: --video required for video mode")
            exit(1)
        
        stream_video_driven_mjpeg(
            server_url=args.server,
            image_path=args.image,
            video_path=args.video,
            output_video_path=args.output,
            crop=args.crop,
            display=not args.no_display
        )
