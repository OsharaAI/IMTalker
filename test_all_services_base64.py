
import sys
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Add IMTalker to path
sys.path.append("/home/ubuntu/video-rendering-inference-server/IMTalker")

# Mock dependencies
sys.modules['openai'] = MagicMock()
sys.modules['aiortc'] = MagicMock()
sys.modules['aiortc.contrib'] = MagicMock()
sys.modules['aiortc.contrib.media'] = MagicMock()
sys.modules['av'] = MagicMock()
sys.modules['imtalker_streamer'] = MagicMock()
sys.modules['chatterbox'] = MagicMock()
sys.modules['chatterbox'].__file__ = "/mock/path/to/chatterbox/__init__.py"
sys.modules['chatterbox.tts'] = MagicMock()
sys.modules['httpx'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['fastapi.responses'] = MagicMock()
sys.modules['fastapi.staticfiles'] = MagicMock()
sys.modules['fastapi.middleware.cors'] = MagicMock()
sys.modules['pydantic'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

# Now import the module under test
# We need to handle potential side effects of import
try:
    from all_services import ConversationManager, save_base64_to_temp, download_url_to_temp
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

async def test_base64_decoding():
    print("Testing save_base64_to_temp...")
    dummy_data = b"Hello world"
    import base64
    b64_str = base64.b64encode(dummy_data).decode('utf-8')
    
    path = save_base64_to_temp(b64_str)
    assert path is not None
    assert os.path.exists(path)
    
    with open(path, 'rb') as f:
        content = f.read()
    assert content == dummy_data
    
    os.remove(path)
    print("Base64 decoding passed.")

async def test_url_download():
    print("Testing download_url_to_temp...")
    import httpx
    
    # Mock httpx.AsyncClient
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = b"Image data"
    mock_response.raise_for_status = MagicMock()
    
    # Configure context manager mock
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = mock_response
    
    httpx.AsyncClient = MagicMock(return_value=mock_client)
    
    url = "http://example.com/image.png"
    path = await download_url_to_temp(url)
    
    assert path is not None
    assert os.path.exists(path)
    
    with open(path, 'rb') as f:
        content = f.read()
    assert content == b"Image data"
    
    os.remove(path)
    print("URL download passed.")

async def test_conversation_manager():
    print("Testing ConversationManager initialization...")
    
    # Mocks
    openai_client = MagicMock()
    tts_model = MagicMock()
    imtalker_streamer = MagicMock()
    video_track = MagicMock()
    audio_track = MagicMock()
    avatar_img = MagicMock()
    
    # Defines expected values
    system_prompt = "Custom system prompt"
    reference_audio = "/path/to/ref.wav"
    
    # Instantiate
    cm = ConversationManager(
        openai_client, tts_model, imtalker_streamer,
        video_track, audio_track, avatar_img,
        system_prompt=system_prompt,
        reference_audio=reference_audio
    )
    
    # Verify attributes
    assert cm.system_prompt == system_prompt, f"Expected system_prompt '{system_prompt}', got '{cm.system_prompt}'"
    assert cm.reference_audio == reference_audio, f"Expected reference_audio '{reference_audio}', got '{cm.reference_audio}'"
    print("Initialization passed.")
    
    # Test generate_and_feed uses reference_audio
    print("Testing generate_and_feed...")
    text = "Hello world"
    
    # Mock tts_stream generator
    tts_model.generate_stream.return_value = [] # Empty iterator
    
    try:
        await cm.generate_and_feed(text)
    except Exception as e:
        print(f"Pipeline error (expected with mocks): {e}")

    # check call args
    tts_model.generate_stream.assert_called()
    call_kwargs = tts_model.generate_stream.call_args.kwargs
    
    assert call_kwargs.get('text') == text
    assert call_kwargs.get('audio_prompt_path') == reference_audio
    print("generate_and_feed passed parameters correctly.")

if __name__ == "__main__":
    asyncio.run(test_base64_decoding())
    asyncio.run(test_url_download())
    asyncio.run(test_conversation_manager())
