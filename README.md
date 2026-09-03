# MLX Audio8 TTS (ArkTTS)

Standalone Apple Silicon port of [Audio8_TTS](https://github.com/Audio8-AI/Audio8_TTS) (ArkTTS).

Features full feature parity with upstream:
- **DualAR Transformer**: 24-layer slow AR for semantic planning + 4-layer fast AR for depth codebook decoding.
- **44.1 kHz Hi-Fi Neural Codec**: 10 codebooks (1 semantic + 9 residual) operating at 44.1 kHz.
- **8-bit Quantized & BF16 Models**: Native affine 8-bit quantization with `sensitive-bf16` policy (RTF < 1.0 on M3 Max).
- **Zero-shot Voice Cloning**: prompt-conditioned voice transfer from short reference audio (cross-lingual supported).
- **11 Supported Languages**: Cantonese (粤语), Chinese (中文), English, French, German, Italian, Japanese, Korean, Polish, Spanish, Dutch.
- **Non-streaming & Streaming**: real-time audio chunk generation (TTFA ~0.39s).
- **Pure MLX Implementation**: zero runtime dependency on `mlx_audio`.

## Installation

```bash
git clone https://github.com/vanch007/mlx-audio8-tts.git
cd mlx-audio8-tts
uv venv .venv
source .venv/bin/activate
pip install -e '.[dev,server]'
```

## Model Artifacts

| Precision | Policy | Size | Memory | M3 Max RTF |
|---|---|---:|---:|---:|
| **8-bit** (Recommended) | `sensitive-bf16` | **827 MB** | ~1.4 GB | **0.64 – 0.82** (Real-time) |
| **BF16** (Baseline) | Full precision | 1,146 MB | ~2.1 GB | 1.24 – 1.48 |

## Quick Start

```bash
# Generate speech (defaults to 8-bit model if present)
mlx-audio8-tts generate \
  --text "你好，欢迎使用 Audio8 TTS 的 MLX 原生版本。" \
  --output output.wav

# Zero-shot voice cloning
mlx-audio8-tts generate \
  --text "今日天气真系好，等阵一齐去饮茶啦。" \
  --ref-audio reference.wav \
  --ref-text "这是参考音频的文字内容。" \
  --output cantonese_clone.wav

# Streaming mode (low latency)
mlx-audio8-tts generate \
  --text "流式生成测试，首包音频快速输出。" \
  --stream \
  --output stream_output.wav
```

## Model Conversion (Quantization)

Convert any BF16 checkpoint into an affine 8-bit quantized model:

```bash
mlx-audio8-tts convert \
  --source models/Audio8-TTS-Preview-0.6b-bf16 \
  --output models/Audio8-TTS-Preview-0.6b-8bit \
  --bits 8 \
  --policy sensitive-bf16
```

## Python API

```python
from mlx_audio8_tts import load, write_audio

# Automatically detects and loads 8-bit or BF16 models
tts = load("models/Audio8-TTS-Preview-0.6b-8bit")

# Generate audio
audio = next(tts.generate("欢迎体验 MLX 原生的高保真语音合成。"))
write_audio("output.wav", audio, tts.sample_rate)
```

## OpenAI-Compatible HTTP Server

```bash
mlx-audio8-tts serve --model models/Audio8-TTS-Preview-0.6b-8bit --port 8000
```

Endpoints:
- `GET /health`: Health check and model details
- `POST /v1/audio/speech`: OpenAI-compatible speech synthesis endpoint
