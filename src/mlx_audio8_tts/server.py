import io
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from . import Audio8TTS, load

app = FastAPI(title="MLX Audio8 TTS Server")
tts_instance: Optional[Audio8TTS] = None


class SpeechRequest(BaseModel):
    input: str
    model: Optional[str] = "Audio8-TTS"
    voice: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    response_format: Optional[str] = "wav"
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 1024


@app.get("/health")
def health():
    if tts_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model": "Audio8-TTS",
        "sample_rate": tts_instance.sample_rate,
    }


@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest):
    if tts_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        audio = next(
            tts_instance.generate(
                text=req.input,
                ref_audio=req.ref_audio,
                ref_text=req.ref_text,
                max_new_tokens=req.max_tokens or 1024,
                temperature=req.temperature or 0.8,
                top_p=req.top_p or 0.95,
                top_k=req.top_k or 50,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.response_format == "pcm":
        # 16-bit PCM
        pcm_data = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        return Response(content=pcm_data, media_type="audio/pcm")
    else:
        buffer = io.BytesIO()
        sf.write(buffer, audio, tts_instance.sample_rate, format="WAV")
        return Response(content=buffer.getvalue(), media_type="audio/wav")


def run_server(model_path: str, host: str = "127.0.0.1", port: int = 8000):
    global tts_instance
    print(f"Loading model from {model_path}...")
    tts_instance = load(model_path)
    print(f"Serving on http://{host}:{port}...")
    uvicorn.run(app, host=host, port=port)
