from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ArkttsConfig:
    dim: int = 896
    n_layer: int = 24
    n_head: int = 14
    n_local_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 4864
    vocab_size: int = 155776
    max_seq_len: int = 2048
    rope_base: float = 1000000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    num_codebooks: int = 10
    codebook_size: int = 4096
    semantic_begin_id: int = 151678
    semantic_end_id: int = 155773
    pad_token_id: int = 151643
    eos_token_id: int = 151645
    tie_word_embeddings: bool = True

    # Fast depth layers
    fast_dim: int = 896
    fast_head_dim: int = 64
    fast_intermediate_size: int = 4864
    fast_n_head: int = 14
    fast_n_local_heads: int = 2
    n_fast_layer: int = 4
    norm_fastlayer_input: bool = True

    # Repetition-Aware Sampling (RAS)
    ras_temperature: float = 1.0
    ras_top_p: float = 0.9
    ras_window_size: int = 10

    # Codec configuration
    codec_sample_rate: int = 44100
    codec_frame_size: int = 2048
    codec_filename: str = "codec.safetensors"
    codec_post_n_layer: int = 8
    codec_post_n_head: int = 16
    codec_post_n_local_heads: int = 8
    codec_post_intermediate_size: int = 1216

    # Extra fields
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArkttsConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        extra = {k: v for k, v in data.items() if k not in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.extra_fields = extra
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "extra_fields"}
        d.update(self.extra_fields)
        return d
