import argparse

import numpy as np

from . import load, write_audio
from .quantization import convert_and_save

DEFAULT_MODEL = "vanch007/Audio8-TTS-MLX-8bit"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-audio8-tts",
        description="Standalone Audio8 TTS (ArkTTS) CLI on Apple Silicon with MLX",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen_parser = subparsers.add_parser("generate", help="Synthesize speech from text")
    gen_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Local path or Hugging Face repo ID (default: {DEFAULT_MODEL})",
    )
    gen_parser.add_argument("--text", required=True, help="Text to synthesize")
    gen_parser.add_argument(
        "--ref-audio", default=None, help="Reference WAV file for voice cloning"
    )
    gen_parser.add_argument("--ref-text", default=None, help="Exact transcript of reference audio")
    gen_parser.add_argument("--output", default="output.wav", help="Output WAV file path")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    gen_parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling")
    gen_parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    gen_parser.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens")
    gen_parser.add_argument("--stream", action="store_true", help="Stream audio chunks")

    # convert / quantize
    conv_parser = subparsers.add_parser("convert", help="Quantize model checkpoint")
    conv_parser.add_argument("--source", required=True, help="Source model directory")
    conv_parser.add_argument("--output", required=True, help="Output quantized model directory")
    conv_parser.add_argument(
        "--bits", type=int, default=8, choices=(4, 8), help="Quantization bits (4 or 8)"
    )
    conv_parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    conv_parser.add_argument(
        "--policy",
        default="sensitive-bf16",
        choices=("sensitive-bf16", "full"),
        help="Quantization policy",
    )

    # audit
    audit_parser = subparsers.add_parser("audit", help="Audit model weights and structure")
    audit_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Local path or Hugging Face repo ID (default: {DEFAULT_MODEL})",
    )

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible HTTP server")
    serve_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Local path or Hugging Face repo ID (default: {DEFAULT_MODEL})",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "generate":
        tts = load(args.model)
        if args.stream:
            chunks = list(
                tts.generate(
                    text=args.text,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    stream=True,
                )
            )
            audio = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
        else:
            audio = next(
                tts.generate(
                    text=args.text,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
            )
        write_audio(args.output, audio, tts.sample_rate)
        print(f"Audio saved to {args.output} ({len(audio) / tts.sample_rate:.2f}s)")

    elif args.command == "convert":
        meta = convert_and_save(
            source_dir=args.source,
            output_dir=args.output,
            bits=args.bits,
            group_size=args.group_size,
            policy=args.policy,
        )
        print(f"Conversion complete to {args.output}")
        print(f"- Quantized modules: {meta['quantized_count']}")
        print(
            f"- Size reduction: {meta['source_size_mb']} MB -> {meta['quantized_size_mb']} MB ({meta['reduction_pct']}%)"
        )

    elif args.command == "audit":
        tts = load(args.model)
        print(f"Audit PASSED for {args.model}")
        print(f"- LM Layers: {len(tts.model.layers)}")
        print(f"- Fast Layers: {len(tts.model.fast_layers)}")
        print(f"- Codec Sample Rate: {tts.sample_rate} Hz")
        if hasattr(tts.model.layers[0].attention.wqkv, "scales"):
            print("- Model Status: 8-bit Quantized (Affine)")
        else:
            print("- Model Status: BF16 Full Precision")

    elif args.command == "serve":
        from .server import run_server

        run_server(model_path=args.model, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
