import mlx.core as mx
from mlx_audio8_tts.config import ArkttsConfig
from mlx_audio8_tts.model import ArkttsModel


def test_model_initialization_and_forward():
    # Create lightweight mini config for fast test
    config = ArkttsConfig(
        dim=64,
        n_layer=2,
        n_head=2,
        n_local_heads=1,
        head_dim=32,
        intermediate_size=128,
        vocab_size=1000,
        max_seq_len=256,
        fast_dim=64,
        fast_head_dim=32,
        fast_intermediate_size=128,
        fast_n_head=2,
        fast_n_local_heads=1,
        n_fast_layer=1,
        num_codebooks=10,
        codebook_size=100,
        semantic_begin_id=500,
        semantic_end_id=600,
        pad_token_id=0,
        eos_token_id=1,
        codec_post_n_layer=1,
        codec_post_n_head=2,
        codec_post_n_local_heads=1,
        codec_post_intermediate_size=64,
    )
    model = ArkttsModel(config)

    # Test prompt shape: (1, 11, 8)
    prompt = mx.zeros((1, 11, 8), dtype=mx.int64)
    prompt[0, 0, :] = 50  # text tokens

    # Test non-streaming generate
    chunks = list(model.generate(prompt, max_new_tokens=3, stream=False))
    assert len(chunks) == 1
    audio = chunks[0]
    assert audio.ndim == 2
    assert audio.shape[0] == 1
    print("Generated audio shape (non-stream):", audio.shape)

    # Test streaming generate
    stream_chunks = list(model.generate(prompt, max_new_tokens=4, stream=True, streaming_interval=2))
    assert len(stream_chunks) > 0
    for chunk in stream_chunks:
        assert chunk.ndim == 2
        assert chunk.shape[0] == 1
    print(f"Generated {len(stream_chunks)} stream chunks")
