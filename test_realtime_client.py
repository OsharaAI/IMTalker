import asyncio
import websockets
import aiohttp
import struct
import argparse
import sys
import numpy as np
import time
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestClient")

async def test_streaming(host, port, image_path, audio_file=None):
    base_url = f"http://{host}:{port}"
    logger.info(f"Connecting to {base_url}")
    
    # 1. Init Session
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('image', open(image_path, 'rb'), filename='face.jpg')
        data.add_field('crop', 'true')
        
        async with session.post(f"{base_url}/api/realtime/init", data=data) as resp:
            if resp.status != 200:
                print(f"Init failed: {await resp.text()}")
                return
            
            res_json = await resp.json()
            session_id = res_json['session_id']
            ws_url = res_json['websocket_url']
            logger.info(f"Session created: {session_id}")
            
    # 2. Connect WebSocket
    full_ws_url = f"ws://{host}:{port}{ws_url}"
    logger.info(f"Connecting to WS: {full_ws_url}")
    
    async with websockets.connect(full_ws_url) as ws:
        logger.info("WebSocket open.")
        
        # Start receive task
        async def receive_loop():
            try:
                frames = 0
                start_time = None
                while True:
                    msg = await ws.recv()
                    if not start_time: start_time = time.time()
                    
                    if len(msg) < 13: continue
                    
                    # Parse header
                    msg_type = msg[0]
                    timestamp_us = struct.unpack(">Q", msg[1:9])[0]
                    payload_len = struct.unpack(">I", msg[9:13])[0]
                    
                    payload = msg[13:]
                    # logger.info(f"Received Type={msg_type} TS={timestamp_us} Len={payload_len}")
                    
                    frames += 1
                    if frames % 25 == 0:
                         fps = frames / (time.time() - start_time)
                         logger.info(f"Received {frames} frames. FPS: {fps:.2f}")
            except Exception as e:
                logger.error(f"Receive error: {e}")

        asyncio.create_task(receive_loop())
        
        # 3. Send Audio
        logger.info("Sending audio...")
        
        # Generate dummy sine wave audio if no file
        sample_rate = 16000
        # 10 seconds of audio
        t = np.linspace(0, 10, 10 * sample_rate, endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        # Convert to int16 PCM
        audio_pcm = (audio_data * 32767).astype(np.int16)
        
        # Send in chunks of 40ms (640 samples)
        chunk_size = int(sample_rate * 0.04)
        
        start_send = time.time()
        for i in range(0, len(audio_pcm), chunk_size):
            chunk = audio_pcm[i:i+chunk_size]
            if len(chunk) < chunk_size: break
            
            # Construct packet
            # [1 byte type] [8 bytes TS] [4 bytes len] [payload]
            ts_us = int(time.time() * 1000000)
            header = struct.pack(">BQ I", 0x01, ts_us, len(chunk.tobytes()))
            packet = header + chunk.tobytes()
            
            await ws.send(packet)
            
            # Throttle to real-time (approx)
            await asyncio.sleep(0.04) # 40ms
            
        logger.info("Finished sending audio. Waiting a bit...")
        await asyncio.sleep(2.0)
        
        # 4. Tesst Silence
        logger.info("Testing silence (sending nothing)...")
        await asyncio.sleep(5.0)
        
        logger.info("Closing.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=9999)
    parser.add_argument("image_path")
    args = parser.parse_args()
    
    asyncio.run(test_streaming(args.host, args.port, args.image_path))
