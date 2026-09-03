# Audio8 TTS MLX 8-bit Release Benchmark

- Date (UTC): 2026-09-03T03:06:23.585762+00:00
- Hardware: Apple M3 Max
- Model: `models/Audio8-TTS-Preview-0.6b-8bit`
- Sample rate: 44100 Hz
- Load time: 0.387 s (excluded from RTF)
- Method: single seeded steady-state run per language after warm-up

| Language | Audio (s) | Elapsed (s) | RTF | Status |
|---|---:|---:|---:|:---:|
| English | 6.966 | 6.85 | 0.983 | pass |
| Chinese | 6.966 | 6.422 | 0.922 | pass |
| Cantonese | 6.966 | 5.527 | 0.793 | pass |

RTF is wall-clock generation time divided by generated audio duration; lower is better.
The WAV files are retained locally for listening checks and ignored by Git.
