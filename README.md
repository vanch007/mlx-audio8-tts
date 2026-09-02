# MLX Audio8 TTS (ArkTTS)

Standalone Apple Silicon port of [Audio8_TTS](https://github.com/Audio8-AI/Audio8_TTS) (ArkTTS).

Features full feature parity with upstream:
- **DualAR Transformer**: 24-layer slow AR for semantic planning + 4-layer fast AR for depth codebook decoding.
- **44.1 kHz Hi-Fi Neural Codec**: 10 codebooks (1 semantic + 9 residual) operating at 44.1 kHz.
- **Zero-shot Voice Cloning**: prompt-conditioned voice transfer from short reference audio.
- **11 Supported Languages**: Cantonese (粤语), Chinese (中文), English, French, German, Italian, Japanese, Korean, Polish, Spanish, Dutch.
- **Non-streaming & Streaming**: real-time audio chunk generation.
- **Pure MLX Implementation**: zero runtime dependency on `mlx_audio`.

## Installation

```bash
git clone https://github.com/vanch007/mlx-audio8-tts.git
cd mlx-audio8-tts
uv venv .venv
source .venv/bin/activate
pip install -e '.[dev,server]'
```

## Quick Start

```bash
# Generate speech
mlx-audio8-tts generate \
  --text "你好，欢迎使用 Audio8 TTS 的 MLX 原生版本。" \
  --output output.wav

# Zero-shot voice cloning
mlx-audio8-tts generate \
  --text "今日天气真系好，等阵一齐去饮茶啦。" \
  --ref-audio reference.wav \
  --ref-text "这是参考音频的文字内容。" \
  --output cantonese_clone.wav
```
