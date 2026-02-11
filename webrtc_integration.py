"""
WebRTC Integration Example for Real-Time Chunked Audio-Driven Video Streaming

This example shows how to integrate the chunked audio streaming with WebRTC
to create a bidirectional real-time video conferencing system.

The flow:
  1. Client uploads a face image
  2. Server initializes chunked streaming session
  3. Client's microphone audio → sent to chunked stream endpoint
  4. Server generates video frames from audio chunks
  5. Generated frames → sent back to client via WebRTC video track
  6. Client displays video in real-time

Requirements:
    pip install aiohttp aiortc python-av

Usage:
    python webrtc_integration.py --port 8080
"""

import asyncio
import json
import logging
from pathlib import Path
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import av

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the client
import sys
sys.path.append(str(Path(__file__).parent))
from chunked_streaming_client import ChunkedStreamingClient


# Global state for WebRTC sessions
webrtc_sessions = {}


class AudioPCMProcessor:
    """Convert WebRTC audio frames to WAV format for chunked streaming"""
    
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.frame_buffer = []
    
    def process_audio_frame(self, frame):
        """Convert audio frame to numpy array"""
        try:
            # Extract audio data
            sound = frame.to_ndarray()
            return sound
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            return None
    
    def to_wav_bytes(self, audio_data):
        """Convert numpy audio to WAV bytes"""
        try:
            import soundfile as sf
            import io
            
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, self.sample_rate, format='WAV')
            return buffer.getvalue()
        except ImportError:
            logger.error("soundfile not installed. Install with: pip install soundfile")
            return None


class VideoFrameSender:
    """Send video frames from HLS stream to WebRTC peer"""
    
    def __init__(self, session_id, server_url="http://localhost:9999"):
        self.session_id = session_id
        self.server_url = server_url
        self.hls_dir = Path("hls_output") / session_id
    
    async def send_frames(self, track):
        """
        Fetch frames from HLS segments and send via WebRTC
        This is a simplified example - in production you'd use an HLS player
        """
        try:
            import cv2
            from PIL import Image
            import glob
            
            last_segment_num = -1
            frame_interval = 1.0 / 25.0  # 25 FPS
            
            while True:
                # Find latest segment
                segments = sorted(glob.glob(str(self.hls_dir / "segment*.ts")))
                
                for segment_path in segments:
                    # Extract segment number
                    import re
                    match = re.search(r'segment(\d+)\.ts', segment_path)
                    if match:
                        seg_num = int(match.group(1))
                        
                        if seg_num > last_segment_num:
                            # New segment - extract frames
                            logger.info(f"Processing segment {seg_num}")
                            
                            # Use ffmpeg to extract frames from segment
                            # (simplified - in production use ffmpeg-python)
                            
                            last_segment_num = seg_num
                
                await asyncio.sleep(frame_interval)
                
        except Exception as e:
            logger.error(f"Error sending frames: {e}")


async def handle_webrtc_offer(request):
    """
    Handle WebRTC offer from client.
    
    Expected POST data:
    {
        "sdp": "v=0...",
        "type": "offer",
        "image": <file data>
    }
    """
    try:
        # Parse multipart form data
        data = await request.post()
        
        # Get image file
        image_file = data.get('image')
        if not image_file:
            return web.json_response(
                {"error": "image file required"},
                status=400
            )
        
        # Get SDP offer
        sdp = data.get('sdp')
        sdp_type = data.get('type', 'offer')
        
        if not sdp:
            return web.json_response(
                {"error": "sdp required"},
                status=400
            )
        
        logger.info("Received WebRTC offer from client")
        
        # Initialize chunked streaming session
        client = ChunkedStreamingClient("http://localhost:9999")
        
        # Save image to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image_file.file.read())
            image_path = tmp.name
        
        logger.info(f"Saved image to {image_path}")
        
        # Initialize streaming
        session_id = client.init_session(
            image_path=image_path,
            crop=True,
            nfe=10,
            cfg_scale=2.0,
            chunk_duration=0.5,  # 500ms for lower latency
            audio_sample_rate=16000
        )
        
        logger.info(f"Initialized streaming session: {session_id}")
        
        # Create RTCPeerConnection
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            )
        )
        
        # Store in sessions
        webrtc_sessions[session_id] = {
            'pc': pc,
            'client': client,
            'audio_processor': AudioPCMProcessor(),
            'video_sender': VideoFrameSender(session_id)
        }
        
        @pc.on("track")
        async def on_track(track):
            logger.info(f"Received track: {track.kind}")
            
            if track.kind == "audio":
                # Receive audio from client
                logger.info("Receiving audio from client")
                
                audio_processor = webrtc_sessions[session_id]['audio_processor']
                
                try:
                    while True:
                        frame = await track.recv()
                        
                        # Process audio frame
                        audio_data = audio_processor.process_audio_frame(frame)
                        if audio_data is None:
                            continue
                        
                        # Convert to WAV
                        wav_bytes = audio_processor.to_wav_bytes(audio_data)
                        if wav_bytes is None:
                            continue
                        
                        # Feed to chunked streaming
                        result = client.feed_audio_chunk(wav_bytes, is_final=False)
                        logger.info(f"Frames generated: {result.get('frames_generated', 0)}")
                
                except Exception as e:
                    logger.error(f"Error receiving audio: {e}")
            
            elif track.kind == "video":
                # Send video to client
                logger.info("Sending video frames to client")
                
                video_sender = webrtc_sessions[session_id]['video_sender']
                await video_sender.send_frames(track)
        
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"Connection state: {pc.connectionState}")
        
        # Process offer
        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        logger.info(f"Sending WebRTC answer")
        
        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id,
            "hls_url": f"http://localhost:9999/hls/{session_id}/playlist.m3u8"
        })
    
    except Exception as e:
        logger.error(f"Error handling offer: {e}", exc_info=True)
        return web.json_response(
            {"error": str(e)},
            status=500
        )


async def handle_ice_candidate(request):
    """Handle ICE candidate from client"""
    try:
        data = await request.json()
        session_id = data.get('session_id')
        
        if session_id not in webrtc_sessions:
            return web.json_response(
                {"error": "session not found"},
                status=404
            )
        
        pc = webrtc_sessions[session_id]['pc']
        
        # Add ICE candidate
        if data.get('candidate'):
            candidate_dict = {
                'candidate': data['candidate'],
                'sdpMLineIndex': data.get('sdpMLineIndex'),
                'sdpMid': data.get('sdpMid'),
            }
            await pc.addIceCandidate(
                RTCIceCandidate(**candidate_dict)
            )
        
        return web.json_response({"status": "ok"})
    
    except Exception as e:
        logger.error(f"Error handling ICE candidate: {e}")
        return web.json_response(
            {"error": str(e)},
            status=500
        )


async def create_app():
    """Create FastAPI application"""
    app = web.Application()
    
    # Routes
    app.router.add_post("/offer", handle_webrtc_offer)
    app.router.add_post("/ice", handle_ice_candidate)
    app.router.add_static("/static", "static", name="static")
    
    return app


# HTML Client for testing
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>IMTalker WebRTC - Real-Time Video Streaming</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #333; }
        .row { display: flex; gap: 20px; margin: 20px 0; }
        .column { flex: 1; }
        video, canvas, img { max-width: 100%; border: 2px solid #ddd; border-radius: 4px; }
        button { padding: 10px 20px; font-size: 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #45a049; }
        button:disabled { background: #cccccc; cursor: not-allowed; }
        #status { padding: 10px; margin: 10px 0; border-radius: 4px; background: #e3f2fd; color: #1976d2; }
        input[type="file"] { padding: 10px; }
        .info { background: #f9f9f9; padding: 10px; border-left: 4px solid #2196F3; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 IMTalker WebRTC - Real-Time Video Streaming</h1>
        
        <div id="status">Initializing...</div>
        
        <div class="row">
            <div class="column">
                <h3>📸 Your Face</h3>
                <input type="file" id="imageInput" accept="image/*" />
                <canvas id="canvas" width="512" height="512"></canvas>
            </div>
            
            <div class="column">
                <h3>🎬 Generated Video</h3>
                <video id="video" autoplay muted></video>
                <div id="stats"></div>
            </div>
        </div>
        
        <div class="row">
            <button id="startBtn" onclick="startStreaming()">Start Streaming</button>
            <button id="stopBtn" onclick="stopStreaming()" disabled>Stop</button>
        </div>
        
        <div class="info">
            <h4>Instructions:</h4>
            <ol>
                <li>Upload your face image</li>
                <li>Click "Start Streaming"</li>
                <li>Grant microphone permission</li>
                <li>Speak naturally - video will be generated in real-time</li>
            </ol>
        </div>
    </div>
    
    <script>
        class IMTalkerWebRTC {
            constructor() {
                this.pc = null;
                this.sessionId = null;
                this.imageFile = null;
            }
            
            async start() {
                const imageFile = document.getElementById('imageInput').files[0];
                if (!imageFile) {
                    alert('Please select an image first');
                    return;
                }
                
                this.imageFile = imageFile;
                
                // Create peer connection
                this.pc = new RTCPeerConnection({
                    iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }]
                });
                
                // Get microphone
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const audioTrack = stream.getAudioTracks()[0];
                this.pc.addTrack(audioTrack);
                
                // Create offer
                const offer = await this.pc.createOffer();
                await this.pc.setLocalDescription(offer);
                
                // Send to server with image
                const formData = new FormData();
                formData.append('sdp', this.pc.localDescription.sdp);
                formData.append('type', this.pc.localDescription.type);
                formData.append('image', imageFile);
                
                updateStatus('Initializing server...');
                
                const response = await fetch('/offer', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                this.sessionId = data.session_id;
                
                // Set remote description
                await this.pc.setRemoteDescription(
                    new RTCSessionDescription({ sdp: data.sdp, type: 'answer' })
                );
                
                updateStatus(`Connected! Session: ${this.sessionId.substring(0, 8)}...`);
                
                // Display video
                this.pc.ontrack = (event) => {
                    console.log('Received track:', event.track.kind);
                    if (event.track.kind === 'video') {
                        document.getElementById('video').srcObject = event.streams[0];
                    }
                };
            }
            
            async stop() {
                if (this.pc) {
                    this.pc.close();
                }
                updateStatus('Disconnected');
            }
        }
        
        const imtalker = new IMTalkerWebRTC();
        
        function startStreaming() {
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            imtalker.start().catch(e => {
                console.error('Error:', e);
                updateStatus('Error: ' + e.message);
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            });
        }
        
        function stopStreaming() {
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            imtalker.stop();
        }
        
        function updateStatus(msg) {
            document.getElementById('status').textContent = msg;
        }
        
        // Display selected image
        document.getElementById('imageInput').addEventListener('change', (e) => {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.getElementById('canvas');
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        });
    </script>
</body>
</html>
"""


async def main():
    """Run WebRTC server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="IMTalker WebRTC Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--imtalker-url", type=str, default="http://localhost:9999",
                        help="IMTalker FastAPI server URL")
    
    args = parser.parse_args()
    
    # Create static files directory
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    # Write HTML client
    with open(static_dir / "index.html", "w") as f:
        f.write(HTML_CLIENT)
    
    # Create app
    app = await create_app()
    
    # Add index route
    async def handle_index(request):
        return web.FileResponse("static/index.html")
    
    app.router.add_get("/", handle_index)
    
    # Run server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()
    
    logger.info(f"WebRTC server started on http://0.0.0.0:{args.port}")
    logger.info(f"IMTalker server: {args.imtalker_url}")
    logger.info(f"Open http://localhost:{args.port} in your browser")
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
