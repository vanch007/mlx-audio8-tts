"""Reproducible steady-state benchmark for the public 8-bit checkpoint."""

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

from mlx_audio8_tts import load, write_audio

CASES = (
    ("en", "English", "Welcome to the Audio8 multilingual text-to-speech model on Apple Silicon."),
    ("zh", "Chinese", "欢迎使用 Audio8 多语言文本转语音模型，在苹果芯片上进行原生推理。"),
    ("yue", "Cantonese", "今日天气真系几好，等阵我哋一齐去饮茶啦。"),
)


def _hardware_name() -> str:
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in result.stdout.splitlines():
            if "Chip:" in line:
                return line.split("Chip:", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/Audio8-TTS-Preview-0.6b-8bit",
        help="Local model directory or Hugging Face repository ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/evaluation/8bit-release"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=150)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    load_started = time.perf_counter()
    tts = load(args.model)
    load_elapsed = time.perf_counter() - load_started

    # Compile and populate caches before collecting steady-state timings.
    mx.random.seed(20260903)
    next(tts.generate("Audio8 benchmark warm-up.", max_new_tokens=24))

    results = []
    for index, (language, language_name, text) in enumerate(CASES):
        mx.random.seed(20260903 + index)
        started = time.perf_counter()
        audio = next(
            tts.generate(
                text,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
            )
        )
        elapsed = time.perf_counter() - started
        duration = len(audio) / tts.sample_rate
        rtf = elapsed / duration
        audio_path = audio_dir / f"{language}.wav"
        write_audio(audio_path, audio, tts.sample_rate)
        results.append(
            {
                "language": language,
                "language_name": language_name,
                "text": text,
                "duration_s": round(duration, 3),
                "elapsed_s": round(elapsed, 3),
                "rtf": round(rtf, 3),
                "audio_path": str(audio_path),
                "status": "pass" if len(audio) > 0 else "fail",
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "hardware": _hardware_name(),
        "sample_rate": tts.sample_rate,
        "max_new_tokens": args.max_new_tokens,
        "model_load_time_s": round(load_elapsed, 3),
        "measurement": "single seeded steady-state run per language after warm-up",
        "results": results,
        "all_passed": all(result["status"] == "pass" for result in results),
    }
    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Audio8 TTS MLX 8-bit Release Benchmark",
        "",
        f"- Date (UTC): {report['generated_at']}",
        f"- Hardware: {report['hardware']}",
        f"- Model: `{report['model']}`",
        f"- Sample rate: {report['sample_rate']} Hz",
        f"- Load time: {report['model_load_time_s']} s (excluded from RTF)",
        f"- Method: {report['measurement']}",
        "",
        "| Language | Audio (s) | Elapsed (s) | RTF | Status |",
        "|---|---:|---:|---:|:---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['language_name']} | {result['duration_s']} | "
            f"{result['elapsed_s']} | {result['rtf']} | {result['status']} |"
        )
    lines.extend(
        [
            "",
            "RTF is wall-clock generation time divided by generated audio duration; lower is better.",
            "The WAV files are retained locally for listening checks and ignored by Git.",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
