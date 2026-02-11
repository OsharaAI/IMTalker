# Real-Time Chunked Audio-Driven Video Streaming

## Overview

The FastAPI server now supports **real-time chunked audio-driven inference**, allowing you to:

- ✅ **Feed audio continuously** in small chunks (not requiring full audio upfront)
- ✅ **Generate video frames in real-time** as audio arrives
- ✅ **Stream frames via HLS** to WebRTC clients with low latency (~2-3 seconds)
- ✅ **Support overlapping audio windows** for smooth motion transitions
- ✅ **Scale with multiple concurrent sessions** without blocking

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WebRTC Client                                │
│  (Browser, Microphone, or Custom Audio Source)                      │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ Audio chunks (PCM, WAV, MP3)
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    ChunkedStreamingClient                           │
│  - Manages session lifecycle                                        │
│  - Buffers audio chunks                                             │
│  - Sends to FastAPI server                                          │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
          HTTP POST /api/audio-driven/chunked/*
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Server                                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ /api/audio-driven/chunked/init                               │  │
│  │ - Create session                                              │  │
│  │ - Initialize HLS stream                                       │  │
│  │ - Load source image                                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ /api/audio-driven/chunked/feed                               │  │
│  │ - Receive audio chunks                                        │  │
│  │ - Buffer with overlap                                         │  │
│  │ - Run inference                                               │  │
│  │ - Stream frames to HLS                                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │         Audio Chunk Buffer (AudioChunkBuffer)                │  │
│  │  [||||...frame1...|||||...frame2...|||||]                    │  │
│  │   ↑ Input         ↑ Sliding Window  ↑ Output to Inference   │  │
│  │   Overlapping audio for smooth motion                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │     InferenceAgent (run_audio_inference_chunked)             │  │
│  │  - Process single audio chunk                                │  │
│  │ - Generate motion tokens                                      │  │
│  │ - Render frames                                               │  │
│  │ - Stream to HLS                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │      ProgressiveHLSGenerator                                 │  │
│  │  [segment0.ts] [segment1.ts] [segment2.ts] ...              │  │
│  │  playlist.m3u8 continuously updated                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ HLS segments + playlist
                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    WebRTC Client                                     │
│  - HLS player receives /hls/{session_id}/playlist.m3u8             │
│  - Plays video frames as segments arrive                           │
│  - Low latency: ~2-3 seconds end-to-end                            │
└─────────────────────────────────────────────────────────────────────┘
```

## API Endpoints

### 1. Initialize Session: `POST /api/audio-driven/chunked/init`

Create a new streaming session and start the HLS output.

**Request:**
```bash
curl -X POST http://localhost:9999/api/audio-driven/chunked/init \
  -F "image=@face.jpg" \
  -F "crop=true" \
  -F "seed=42" \
  -F "nfe=10" \
  -F "cfg_scale=2.0" \
  -F "chunk_duration=1.0" \
  -F "audio_sample_rate=24000"
```

**Parameters:**
- `image` (file, required): Source face image (PNG, JPG)
- `crop` (bool, default=true): Auto-crop face from image
- `seed` (int, default=42): Random seed for reproducibility
- `nfe` (int, default=10): Number of function evaluations (5-50, higher=better quality)
- `cfg_scale` (float, default=2.0): Classifier-free guidance (1.0-5.0)
- `chunk_duration` (float, default=1.0): Duration of each audio chunk (0.1-10.0 seconds)
- `audio_sample_rate` (int, default=24000): Sample rate of incoming audio
- `pose_style` (JSON, optional): Pose control [pitch, yaw, roll]
- `gaze_style` (JSON, optional): Gaze control [x, y]

**Response:**
```json
{
  "status": "success",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "hls_url": "/hls/550e8400-e29b-41d4-a716-446655440000/playlist.m3u8",
  "message": "Session ready. Send audio chunks to /api/audio-driven/chunked/feed",
  "chunk_duration": 1.0,
  "audio_sample_rate": 24000
}
```

### 2. Feed Audio Chunk: `POST /api/audio-driven/chunked/feed`

Send an audio chunk to the session. Triggers inference and frame generation.

**Request:**
```bash
curl -X POST http://localhost:9999/api/audio-driven/chunked/feed \
  -F "session_id=550e8400-e29b-41d4-a716-446655440000" \
  -F "audio_chunk=@chunk.wav" \
  -F "is_final_chunk=false"
```

**Parameters:**
- `session_id` (string, required): Session ID from init
- `audio_chunk` (file, required): Audio chunk (WAV, MP3, etc.)
- `is_final_chunk` (bool, default=false): Set to true for the last chunk

**Response:**
```json
{
  "status": "success",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "buffer_samples": 24000,
  "frames_generated": 125,
  "session_status": "processing",
  "message": "Audio chunk received and queued for processing"
}
```

### 3. Get Status: `GET /api/audio-driven/chunked/status/{session_id}`

Monitor session progress.

**Request:**
```bash
curl http://localhost:9999/api/audio-driven/chunked/status/550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "total_frames_generated": 375,
  "buffer_samples": 24000,
  "error": null,
  "created_at": 1704067200.123,
  "uptime_seconds": 45.678
}
```

### 4. Cleanup Session: `DELETE /api/audio-driven/chunked/session/{session_id}`

Manually clean up session and HLS files.

**Request:**
```bash
curl -X DELETE http://localhost:9999/api/audio-driven/chunked/session/550e8400-e29b-41d4-a716-446655440000
```

## Usage Examples

### Python Client (Recommended)

```python
from chunked_streaming_client import ChunkedStreamingClient

# Initialize client
client = ChunkedStreamingClient("http://localhost:9999")

# Create session
session_id = client.init_session(
    image_path="face.jpg",
    chunk_duration=1.0,
    audio_sample_rate=24000
)

# Stream audio chunks
import librosa

audio_data, sr = librosa.load("speech.wav", sr=24000)
chunk_samples = int(sr * 1.0)  # 1 second

for i in range(0, len(audio_data), chunk_samples):
    chunk = audio_data[i:i+chunk_samples]
    is_final = (i + chunk_samples) >= len(audio_data)
    
    # Convert to WAV bytes and send
    import soundfile as sf
    import io
    
    buffer = io.BytesIO()
    sf.write(buffer, chunk, sr, format='WAV')
    wav_bytes = buffer.getvalue()
    
    result = client.feed_audio_chunk(wav_bytes, is_final=is_final)
    print(f"Frames: {result['frames_generated']}")

# Get final status
status = client.get_status()
print(f"Total frames: {status['total_frames_generated']}")

# View stream at: http://localhost:9999/hls/{session_id}/playlist.m3u8
```

### JavaScript/TypeScript (Browser/WebRTC)

```javascript
class ChunkedStreamingClient {
  constructor(serverUrl = "http://localhost:9999") {
    this.serverUrl = serverUrl;
    this.sessionId = null;
  }

  async initSession(imageFile, options = {}) {
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("crop", options.crop ?? true);
    formData.append("seed", options.seed ?? 42);
    formData.append("nfe", options.nfe ?? 10);
    formData.append("cfg_scale", options.cfg_scale ?? 2.0);
    formData.append("chunk_duration", options.chunk_duration ?? 1.0);
    formData.append("audio_sample_rate", options.audio_sample_rate ?? 24000);

    const response = await fetch(
      `${this.serverUrl}/api/audio-driven/chunked/init`,
      { method: "POST", body: formData }
    );

    const data = await response.json();
    this.sessionId = data.session_id;
    return data;
  }

  async feedAudioChunk(audioBlob, isFinal = false) {
    const formData = new FormData();
    formData.append("session_id", this.sessionId);
    formData.append("audio_chunk", audioBlob);
    formData.append("is_final_chunk", isFinal);

    const response = await fetch(
      `${this.serverUrl}/api/audio-driven/chunked/feed`,
      { method: "POST", body: formData }
    );

    return await response.json();
  }

  async getStatus() {
    const response = await fetch(
      `${this.serverUrl}/api/audio-driven/chunked/status/${this.sessionId}`
    );
    return await response.json();
  }
}

// Usage in WebRTC context
const client = new ChunkedStreamingClient();

// Initialize with face image
const imageFile = document.getElementById("faceImage").files[0];
const sessionInfo = await client.initSession(imageFile);

console.log(`HLS URL: ${sessionInfo.hls_url}`);

// In your WebRTC audio track handler:
audioTrack.addEventListener("data", async (audioData) => {
  // audioData is raw PCM or WAV
  const isLastChunk = /* ... determine if last ... */;
  await client.feedAudioChunk(audioData, isLastChunk);
});

// Display video
const video = document.createElement("video");
video.src = `http://localhost:9999${sessionInfo.hls_url}`;
video.play();
```

### Integration with WebRTC (Full Example)

```python
from fastapi import WebSocket
from aiohttp import web
import asyncio
from chunked_streaming_client import ChunkedStreamingClient

client = ChunkedStreamingClient("http://localhost:9999")

# Initialize session with user's face image
async def handle_webrtc_offer(request):
    offer_data = await request.json()
    
    # User uploads their face image
    image_file = request.files.get("image")
    
    # Start streaming session
    session_id = client.init_session(
        image_path=str(image_file.filename),
        chunk_duration=0.5,  # 500ms for lower latency
        audio_sample_rate=16000
    )
    
    # Set up RTCPeerConnection to receive audio
    pc = RTCPeerConnection()
    
    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            # Receive audio and feed to chunked stream
            while True:
                frame = await track.recv()
                audio_data = frame.to_ndarray()
                
                # Convert to WAV bytes
                wav_bytes = convert_to_wav(audio_data, sr=16000)
                
                # Feed to streaming endpoint
                client.feed_audio_chunk(wav_bytes, is_final=False)
    
    # Set up video track to send HLS frames
    @pc.on("track")
    async def send_video():
        # Fetch HLS segments and send as video frames
        # (implement HLS frame feeding)
        pass
    
    await pc.setRemoteDescription(offer_data)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "session_id": session_id,
        "hls_url": f"/hls/{session_id}/playlist.m3u8"
    })
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Latency (Init→First Frame)** | ~30-60 seconds | Model loading + first audio processing |
| **Per-Chunk Latency** | ~2-5 seconds | Depends on chunk_duration and nfe |
| **HLS Streaming Latency** | ~2-3 seconds | Client buffer + segment generation |
| **Total End-to-End Latency** | ~5-8 seconds | From audio input to display |
| **Max Concurrent Sessions** | Limited by VRAM | Typically 2-4 sessions on 24GB GPU |
| **Chunk Processing Time** | ~0.5-2s per second of audio | Depends on nfe and GPU |
| **Memory per Session** | ~1-2GB | Image buffer + inference state |

## Advanced Configuration

### Tuning for Lower Latency

```python
client.init_session(
    image_path="face.jpg",
    chunk_duration=0.5,    # ↓ Smaller chunks = faster streaming
    nfe=5,                 # ↓ Fewer steps = faster generation (lower quality)
    cfg_scale=1.5,         # ↓ Lower guidance = faster (less stylized)
)
```

### Tuning for Higher Quality

```python
client.init_session(
    image_path="face.jpg",
    chunk_duration=2.0,    # ↑ Larger chunks = better continuity
    nfe=15,                # ↑ More steps = better quality
    cfg_scale=3.0,         # ↑ Higher guidance = more stylized
)
```

### Overlapping Audio Windows

The `AudioChunkBuffer` uses 20% overlap by default:
- Chunk duration: 1.0 second
- Stride: 0.8 seconds (80% forward)
- Overlap: 0.2 seconds (20% backward) for smooth motion

This ensures smooth transitions between chunks.

## Troubleshooting

### "Session not found"
- Make sure you're using the correct `session_id` from `init_session`
- Sessions expire after 50 minutes of inactivity

### "Buffer not ready / not enough audio samples"
- The buffer needs enough audio to generate frames
- For `chunk_duration=1.0`, you need ~1-2 seconds of audio before generation starts
- Reduce `chunk_duration` for faster feedback

### Frames are disjointed or jerky
- Increase `chunk_duration` for smoother transitions
- Increase `nfe` for better motion generation
- Check network latency between client and server

### Server out of memory
- Reduce number of concurrent sessions
- Reduce `chunk_duration` (smaller = less memory per session)
- Increase GPU VRAM or use smaller GPU with lower batch sizes

## Comparison: Chunked vs. Static Endpoints

| Feature | `/api/audio-driven` | `/api/audio-driven/stream` | `/api/audio-driven/chunked` |
|---------|---|---|---|
| **Input** | Full audio file | Full audio file | Audio chunks (streaming) |
| **Latency** | ~10-30s to first frame | ~2-4s to first frame | ~5-8s end-to-end, ~2-3s per chunk |
| **Use Case** | Batch processing | Progressive HLS | Real-time WebRTC |
| **Memory** | High (full audio) | Medium (segments) | Low (buffering) |
| **Best For** | API automation | Live streaming preview | WebRTC, microphone input |

## Next Steps

1. **Test with provided examples**: Run `chunked_streaming_client.py`
2. **Integrate with WebRTC**: Use JavaScript client in browser
3. **Monitor performance**: Check session status during streaming
4. **Optimize parameters**: Tune chunk_duration, nfe, cfg_scale for your use case
5. **Deploy**: Use Docker or cloud deployment for production

## Further Reading

- [HLS Streaming Protocol](https://tools.ietf.org/html/draft-pantos-http-live-streaming-23)
- [WebRTC Real-time Communication](https://webrtc.org/)
- [Librosa Audio Processing](https://librosa.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
