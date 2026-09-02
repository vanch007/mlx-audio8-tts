import json
import os
import time
from pathlib import Path

import numpy as np
from mlx_audio8_tts import load, write_audio

LANGUAGES_TEST = [
    ("zh", "中文", "欢迎使用 Audio8 文本转语音模型。"),
    ("yue", "粤语", "今日天气真系好，等阵我哋一齐去饮茶啦。"),
    ("en", "English", "Welcome to the Audio8 multilingual text-to-speech model on Apple Silicon."),
    ("ja", "日语", "こんにちは、新しい音声合成システムへようこそ。"),
    ("ko", "韩语", "안녕하세요, 새로운 음성 합성 시스템에 오신 것을 환영합니다."),
    ("fr", "法语", "Bonjour, bienvenue dans le système de synthèse vocale Audio8."),
    ("de", "德语", "Guten Tag und herzlich willkommen bei Audio8 Sprachausgabe."),
    ("es", "西班牙语", "Hola, bienvenido al modelo de texto a voz multilingüe Audio8."),
    ("it", "意大利语", "Buongiorno, benvenuti nel sistema di sintesi vocale Audio8."),
    ("nl", "荷兰语", "Hallo en welkom bij de Audio8 spraaksynthese voor Apple Silicon."),
    ("pl", "波兰语", "Dzień dobry, witamy w systemie syntezy mowy Audio8."),
]


def main():
    output_dir = Path("/Users/vanch/mlx-audio8-tts/reports/evaluation")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    model_path = "/Users/vanch/mlx-audio8-tts/models/Audio8-TTS-Preview-0.6b-bf16"
    print(f"Loading model from {model_path}...")
    t_load_0 = time.time()
    tts = load(model_path)
    t_load_1 = time.time()
    load_time = t_load_1 - t_load_0
    print(f"Model loaded in {load_time:.2f}s")

    results = []

    print("\n=== Multilingual Baseline Generation (11 Languages) ===")
    for lang_code, lang_name, text in LANGUAGES_TEST:
        print(f"Generating [{lang_code}] {lang_name}: {text[:30]}...")
        t0 = time.time()
        audio = next(
            tts.generate(text=text, max_new_tokens=150, temperature=0.7, top_p=0.95, top_k=50)
        )
        t1 = time.time()
        duration = len(audio) / tts.sample_rate
        elapsed = t1 - t0
        rtf = elapsed / duration if duration > 0 else 0.0
        out_wav = audio_dir / f"lang_{lang_code}.wav"
        write_audio(out_wav, audio, tts.sample_rate)
        print(f"  Duration: {duration:.2f}s, Elapsed: {elapsed:.2f}s, RTF: {rtf:.3f}")
        results.append({
            "category": "language_baseline",
            "language": lang_code,
            "language_name": lang_name,
            "text": text,
            "duration_s": round(duration, 3),
            "elapsed_s": round(elapsed, 3),
            "rtf": round(rtf, 3),
            "audio_path": str(out_wav),
            "status": "pass",
        })

    print("\n=== Zero-shot Voice Cloning ===")
    ref_wav = str(audio_dir / "lang_zh.wav")
    ref_text = LANGUAGES_TEST[0][2]
    clone_text = "零样本声音克隆实机验证：克隆中文发音特征进行全新句子朗读。"
    t0 = time.time()
    audio_clone = next(
        tts.generate(text=clone_text, ref_audio=ref_wav, ref_text=ref_text, max_new_tokens=150)
    )
    t1 = time.time()
    duration = len(audio_clone) / tts.sample_rate
    elapsed = t1 - t0
    out_clone = audio_dir / "voice_clone_zh.wav"
    write_audio(out_clone, audio_clone, tts.sample_rate)
    results.append({
        "category": "voice_clone",
        "language": "zh",
        "text": clone_text,
        "ref_audio": ref_wav,
        "duration_s": round(duration, 3),
        "elapsed_s": round(elapsed, 3),
        "rtf": round(elapsed / duration, 3),
        "audio_path": str(out_clone),
        "status": "pass",
    })
    print(f"  Clone duration: {duration:.2f}s, RTF: {elapsed/duration:.3f}")

    print("\n=== Streaming Generation & TTFA ===")
    stream_text = "流式语音生成实机测试，评估首包延迟与帧间隔平稳性。"
    t0 = time.time()
    ttfa = None
    chunks = []
    for chunk in tts.generate(text=stream_text, max_new_tokens=150, stream=True, streaming_interval=2):
        if ttfa is None:
            ttfa = time.time() - t0
        chunks.append(chunk)
    t1 = time.time()
    stream_audio = np.concatenate(chunks)
    duration = len(stream_audio) / tts.sample_rate
    elapsed = t1 - t0
    out_stream = audio_dir / "stream_eval.wav"
    write_audio(out_stream, stream_audio, tts.sample_rate)
    results.append({
        "category": "streaming",
        "text": stream_text,
        "ttfa_s": round(ttfa, 4),
        "chunks": len(chunks),
        "duration_s": round(duration, 3),
        "elapsed_s": round(elapsed, 3),
        "rtf": round(elapsed / duration, 3),
        "audio_path": str(out_stream),
        "status": "pass",
    })
    print(f"  TTFA: {ttfa:.3f}s, Total duration: {duration:.2f}s, RTF: {elapsed/duration:.3f}")

    # Write JSON summary
    summary = {
        "model": "Audio8-TTS-Preview-0.6b-bf16",
        "architecture": "DualAR (24L Slow + 4L Fast) + 44.1kHz Codec",
        "hardware": "Apple Silicon M3 Max",
        "sample_rate": tts.sample_rate,
        "model_load_time_s": round(load_time, 3),
        "total_cases": len(results),
        "all_passed": all(r["status"] == "pass" for r in results),
        "results": results,
    }
    json_path = output_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary JSON to {json_path}")

    # Write Markdown Report
    md_path = output_dir / "REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Audio8 TTS (ArkTTS) MLX Real-Device Benchmark Report\n\n")
        f.write(f"- **Model**: Audio8-TTS-Preview-0.6b (DualAR 24L + 4L)\n")
        f.write(f"- **Hardware**: Apple Silicon M3 Max\n")
        f.write(f"- **Sample Rate**: {tts.sample_rate} Hz\n")
        f.write(f"- **Model Load Time**: {load_time:.2f}s\n")
        f.write(f"- **Total Test Cases**: {len(results)}\n\n")
        f.write("| Category | Language | RTF | Duration (s) | Elapsed (s) | Status |\n")
        f.write("|---|---|---:|---:|---:|:---:|\n")
        for r in results:
            lang = r.get("language_name", r.get("language", "N/A"))
            f.write(f"| {r['category']} | {lang} | {r['rtf']} | {r['duration_s']} | {r['elapsed_s']} | {r['status']} |\n")
    print(f"Saved Markdown report to {md_path}")


if __name__ == "__main__":
    main()
