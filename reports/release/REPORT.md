# Public Release Verification

Verified on 2026-09-03 from the Apple Silicon development machine.

## Published artifacts

| Artifact | Public URL | Release revision |
|---|---|---|
| Source implementation | https://github.com/vanch007/mlx-audio8-tts | `2fc7c09` |
| 8-bit model | https://huggingface.co/vanch007/Audio8-TTS-MLX-8bit | `54f83f3fd966db81a98df41b4c9825395689024d` |
| BF16 model | https://huggingface.co/vanch007/Audio8-TTS-MLX-BF16 | `9e6603126f4a2ebc2306477c87f9903785497591` |

## Verification status

- `pass`: source repository is public and its default branch is `main`.
- `pass`: 9/9 Python tests.
- `pass`: the reproducible post-warm-up 8-bit benchmark completed for English,
  Chinese, and Cantonese at RTF 0.983, 0.922, and 0.793 respectively; see
  `reports/evaluation/8bit-release/`.
- `pass`: local structure audit identifies the 8-bit model as affine 8-bit and
  the baseline as BF16; both have 24 Slow AR layers, 4 Fast AR layers, and a
  44.1 kHz codec.
- `pass`: each Hugging Face repository is public and contains the eight
  required release files plus Hugging Face's generated `.gitattributes`.
- `pass`: every remote release file matches its local source by exact byte
  size; all LFS objects also match by SHA-256.
- `pass`: the GitHub README links to both model repositories, and both model
  cards link back to the GitHub source repository.

The full remote checkpoints were not downloaded again after publication:
remote size and LFS SHA-256 equality prove that they are byte-identical to the
locally audited checkpoints without consuming another 4.47 GiB of disk space.
