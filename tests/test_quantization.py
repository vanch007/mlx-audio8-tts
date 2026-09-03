from mlx_audio8_tts.config import ArkttsConfig
from mlx_audio8_tts.model import ArkttsModel
from mlx_audio8_tts.quantization import (
    POLICIES,
    get_quantization_predicate,
    quantize_model,
    should_quantize_path,
    validate_policy,
)


def test_quantization_policies():
    assert "sensitive-bf16" in POLICIES
    assert "full" in POLICIES
    assert validate_policy("sensitive-bf16") == "sensitive-bf16"
    assert should_quantize_path("layers.0.attention.wqkv", "sensitive-bf16") is True
    assert should_quantize_path("fast_layers.0.attention.wqkv", "sensitive-bf16") is False
    assert should_quantize_path("embeddings", "sensitive-bf16") is False
    assert should_quantize_path("fast_layers.0.attention.wqkv", "full") is True


def test_model_quantize_execution():
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
    )
    model = ArkttsModel(config)
    meta = quantize_model(model, bits=8, group_size=64, policy="sensitive-bf16")
    assert meta["bits"] == 8
    assert meta["quantized_count"] > 0
    # Verify layer 0 wqkv is quantized
    assert hasattr(model.layers[0].attention.wqkv, "scales")
