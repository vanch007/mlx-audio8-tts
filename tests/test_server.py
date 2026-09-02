from fastapi.testclient import TestClient
from mlx_audio8_tts import server
from unittest.mock import MagicMock
import numpy as np


def test_server_endpoints():
    # Mock Audio8TTS
    mock_tts = MagicMock()
    mock_tts.sample_rate = 44100
    mock_audio = np.zeros((44100,), dtype=np.float32)
    mock_tts.generate.side_effect = lambda *args, **kwargs: iter([mock_audio])

    server.tts_instance = mock_tts
    client = TestClient(server.app)

    # 1. Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["sample_rate"] == 44100

    # 2. Speech synthesis (WAV)
    resp = client.post("/v1/audio/speech", json={"input": "Hello world"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert len(resp.content) > 44  # WAV header + pcm data

    # 3. Speech synthesis (PCM)
    resp_pcm = client.post("/v1/audio/speech", json={"input": "Hello world", "response_format": "pcm"})
    assert resp_pcm.status_code == 200
    assert resp_pcm.headers["content-type"] == "audio/pcm"
    assert len(resp_pcm.content) == 44100 * 2  # 16-bit PCM bytes
