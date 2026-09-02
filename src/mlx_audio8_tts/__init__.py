import json
from pathlib import Path
from typing import Generator, Optional, Union

import mlx.core as mx
import numpy as np
import soundfile as sf

from .codec import ArkttsCodec
from .config import ArkttsConfig
from .model import ArkttsModel
from .processor import ArkttsProcessor

__version__ = "0.1.0"


class Audio8TTS:
    def __init__(self, model: ArkttsModel, processor: ArkttsProcessor, config: ArkttsConfig):
        self.model = model
        self.processor = processor
        self.config = config
        self.sample_rate = config.codec_sample_rate

    def generate(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path, np.ndarray]] = None,
        ref_text: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        stream: bool = False,
        streaming_interval: int = 2,
    ) -> Generator[np.ndarray, None, None]:
        ref_codes = None
        if ref_audio is not None:
            if isinstance(ref_audio, np.ndarray) and ref_audio.ndim == 2 and ref_audio.shape[0] == 10:
                ref_codes = ref_audio
            else:
                audio_np = self.processor.load_audio(ref_audio, target_sample_rate=self.sample_rate)
                audio_mx = mx.array(audio_np)[None, :]
                codes_mx = self.model.codec.encode(audio_mx)
                ref_codes = np.array(codes_mx[0], copy=False)

        prompt = self.processor.prepare_prompt(
            text=text,
            reference_text=ref_text,
            reference_codes=ref_codes,
        )

        for audio_chunk in self.model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=stream,
            streaming_interval=streaming_interval,
        ):
            yield np.array(audio_chunk, copy=False).squeeze()


def write_audio(path: Union[str, Path], audio: Union[mx.array, np.ndarray], sample_rate: int = 44100):
    if isinstance(audio, mx.array):
        audio = np.array(audio, copy=False)
    if audio.ndim == 2:
        audio = audio.squeeze(0)
    sf.write(str(path), audio, sample_rate)


def load(model_path: Union[str, Path], dtype=mx.bfloat16) -> Audio8TTS:
    model_dir = Path(model_path)
    if not model_dir.exists():
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(repo_id=str(model_path)))

    config_file = model_dir / "config.json"
    with open(config_file, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    config = ArkttsConfig.from_dict(cfg_dict)

    model = ArkttsModel(config)

    # Load LM weights from model.safetensors
    model_safetensors = model_dir / "model.safetensors"
    if model_safetensors.exists():
        weights = mx.load(str(model_safetensors))
        clean_weights = {}
        for k, v in weights.items():
            new_k = k[6:] if k.startswith("model.") else k
            clean_weights[new_k] = v.astype(dtype)
        model.load_weights(list(clean_weights.items()), strict=False)

    # Load Codec weights from codec.safetensors
    codec_safetensors = model_dir / config.codec_filename
    if codec_safetensors.exists():
        c_weights = mx.load(str(codec_safetensors))
        clean_c_weights = {}
        for k, v in c_weights.items():
            new_k = k[6:] if k.startswith("codec.") else k
            clean_c_weights[new_k] = v.astype(mx.float32)
        model.codec.load_weights(list(clean_c_weights.items()), strict=False)

    processor = ArkttsProcessor.from_pretrained(model_dir)
    return Audio8TTS(model=model, processor=processor, config=config)


__all__ = [
    "Audio8TTS",
    "ArkttsModel",
    "ArkttsConfig",
    "ArkttsProcessor",
    "load",
    "write_audio",
]
