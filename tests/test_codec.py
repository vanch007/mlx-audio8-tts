import mlx.core as mx
from mlx_audio8_tts.codec import ArkttsCodec
from mlx_audio8_tts.config import ArkttsConfig


def test_codec_decode_shape():
    config = ArkttsConfig()
    codec = ArkttsCodec(config)
    # 10 codebooks, 4 frames
    codes = mx.zeros((1, 10, 4), dtype=mx.int32)
    audio = codec.decode(codes)
    assert audio.ndim == 2
    assert audio.shape[0] == 1
    # 4 frames * 2048 samples per frame = 8192 samples
    assert audio.shape[1] == 4 * 2048, f"Expected {4 * 2048}, got {audio.shape[1]}"
