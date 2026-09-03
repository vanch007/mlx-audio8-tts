# MLX Audio8 TTS (ArkTTS)

Standalone Apple Silicon MLX port of
[Audio8-AI/Audio8_TTS](https://github.com/Audio8-AI/Audio8_TTS), based on the
0.6B Preview checkpoint. The runtime does not depend on PyTorch or
`mlx_audio`.

## Models

| Variant | Hugging Face | LM weights | Complete download | M3 Max RTF |
|---|---|---:|---:|---:|
| 8-bit `sensitive-bf16` | [Audio8-TTS-MLX-8bit](https://huggingface.co/vanch007/Audio8-TTS-MLX-8bit) | 827 MiB | 2.08 GiB | 0.64–0.82 |
| BF16 baseline | [Audio8-TTS-MLX-BF16](https://huggingface.co/vanch007/Audio8-TTS-MLX-BF16) | 1,147 MiB | 2.39 GiB | 1.24–1.48 |

The complete download includes the shared 1.26 GiB, 44.1 kHz neural codec and
tokenizer files. The 8-bit size reduction applies to the 0.6B language model;
the codec deliberately remains unquantized to protect audio quality.

## Implemented capabilities

- 24-layer Slow AR plus 4-layer Fast AR DualAR inference.
- 44.1 kHz, 10-codebook neural codec encoder and decoder.
- Speech generation with or without a reference voice.
- Zero-shot and cross-lingual voice cloning with reference transcript.
- Streaming and non-streaming generation.
- Cantonese, Chinese, Dutch, English, French, German, Italian, Japanese,
  Korean, Polish, and Spanish.
- Temperature, top-p, top-k, and maximum-token controls.
- Native affine 8-bit conversion with `sensitive-bf16` and `full` policies.
- Python API, command-line interface, and OpenAI-compatible HTTP endpoint.

For best quality, keep each input under 150 characters and use a clean, dry
reference recording with an exact transcript.

## Requirements and installation

- Apple Silicon Mac
- macOS 14 or newer
- Python 3.10 or newer

```bash
git clone https://github.com/vanch007/mlx-audio8-tts.git
cd mlx-audio8-tts
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[server]'
```

The first command-line run downloads the recommended 8-bit model from
Hugging Face automatically.

## Command line

Generate speech:

```bash
mlx-audio8-tts generate \
  --text "你好，欢迎使用 Audio8 TTS 的 MLX 原生版本。" \
  --output output.wav
```

Zero-shot voice cloning:

```bash
mlx-audio8-tts generate \
  --text "今日天气真系好，等阵一齐去饮茶啦。" \
  --ref-audio reference.wav \
  --ref-text "这是参考音频的准确文字内容。" \
  --output cantonese_clone.wav
```

Streaming generation:

```bash
mlx-audio8-tts generate \
  --text "流式生成测试，首包音频快速输出。" \
  --stream \
  --output stream_output.wav
```

Use BF16 explicitly:

```bash
mlx-audio8-tts generate \
  --model vanch007/Audio8-TTS-MLX-BF16 \
  --text "BF16 baseline test." \
  --output bf16.wav
```

## Python API

```python
from mlx_audio8_tts import load, write_audio

tts = load("vanch007/Audio8-TTS-MLX-8bit")
audio = next(tts.generate("欢迎体验 MLX 原生语音合成。"))
write_audio("output.wav", audio, tts.sample_rate)
```

Voice cloning accepts a local audio path or a NumPy waveform:

```python
audio = next(
    tts.generate(
        "This is a new sentence in the reference voice.",
        ref_audio="reference.wav",
        ref_text="Exact transcript of the reference recording.",
        temperature=0.8,
        top_p=0.95,
        top_k=50,
    )
)
```

## HTTP server

```bash
mlx-audio8-tts serve --port 8000
```

```bash
curl -sS http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from MLX Audio8 TTS.","response_format":"wav"}' \
  -o output.wav
```

Endpoints:

- `GET /health`
- `POST /v1/audio/speech` (`wav` or signed 16-bit `pcm`)

## Convert and audit models

```bash
mlx-audio8-tts convert \
  --source models/Audio8-TTS-Preview-0.6b-bf16 \
  --output models/Audio8-TTS-Preview-0.6b-8bit \
  --bits 8 \
  --group-size 64 \
  --policy sensitive-bf16

mlx-audio8-tts audit --model vanch007/Audio8-TTS-MLX-8bit
```

The recommended policy quantizes 120 Slow AR linear layers while preserving
the embeddings, Fast AR depth decoder, and codec at higher precision.

## Verification

The local test suite covers model generation, streaming, codec decoding,
processor prompts, quantization, HTTP serving, published defaults, and strict
runtime isolation from `mlx_audio`.

```bash
pip install -e '.[dev,server]'
pytest -q
```

Real-device evidence is stored under [`reports/evaluation`](reports/evaluation).
The checked-in BF16 matrix contains 11 languages, voice cloning, and streaming.
RTF excludes model download time; lower is better.

## License and attribution

Apache-2.0. This is an independent MLX port. Model architecture and base
checkpoint are from [Audio8-AI/Audio8_TTS](https://github.com/Audio8-AI/Audio8_TTS)
and [Audio8/Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b).
