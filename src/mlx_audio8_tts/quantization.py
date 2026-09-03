from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ArkttsConfig
from .model import ArkttsModel

POLICIES = ("sensitive-bf16", "full")


def validate_policy(policy: str) -> str:
    if policy not in POLICIES:
        raise ValueError(f"Quantization policy must be one of: {', '.join(POLICIES)}")
    return policy


def should_quantize_path(path: str, policy: str = "sensitive-bf16") -> bool:
    validate_policy(policy)
    if policy == "sensitive-bf16":
        # Quantize Slow AR Transformer backbone (layers.*), keep embeddings & depth decoder in BF16
        return path.startswith("layers.")
    elif policy == "full":
        # Quantize both Slow AR and Fast AR layers, keep embeddings and codec unquantized
        return path.startswith("layers.") or path.startswith("fast_layers.")
    return False


def get_quantization_predicate(
    group_size: int = 64,
    policy: str = "sensitive-bf16",
) -> Callable[[str, nn.Module], bool]:
    def predicate(path: str, module: nn.Module) -> bool:
        if not hasattr(module, "to_quantized") or not hasattr(module, "weight"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        return should_quantize_path(path, policy)

    return predicate


def quantize_model(
    model: ArkttsModel,
    bits: int = 8,
    group_size: int = 64,
    policy: str = "sensitive-bf16",
) -> Dict[str, Any]:
    validate_policy(policy)
    quantized_modules: List[str] = []
    excluded_modules: List[str] = []

    def tracker(path: str, module: nn.Module) -> bool:
        eligible = (
            hasattr(module, "to_quantized")
            and hasattr(module, "weight")
            and module.weight.shape[-1] % group_size == 0
        )
        if not eligible:
            return False
        selected = should_quantize_path(path, policy)
        if selected:
            quantized_modules.append(path)
        else:
            excluded_modules.append(path)
        return selected

    nn.quantize(model, group_size=group_size, bits=bits, mode="affine", class_predicate=tracker)
    return {
        "bits": bits,
        "group_size": group_size,
        "mode": "affine",
        "policy": policy,
        "quantized_count": len(quantized_modules),
        "quantized_modules": sorted(quantized_modules),
        "excluded_modules": sorted(excluded_modules),
    }


def convert_and_save(
    source_dir: str | Path,
    output_dir: str | Path,
    bits: int = 8,
    group_size: int = 64,
    policy: str = "sensitive-bf16",
) -> Dict[str, Any]:
    src_path = Path(source_dir)
    dst_path = Path(output_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    config_file = src_path / "config.json"
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    config = ArkttsConfig.from_dict(config_data)

    print(f"Loading source weights from {src_path}...")
    model = ArkttsModel(config)
    source_weights_file = src_path / "model.safetensors"
    raw_weights = mx.load(str(source_weights_file))
    clean_weights = {}
    for k, v in raw_weights.items():
        new_k = k[6:] if k.startswith("model.") else k
        clean_weights[new_k] = v.astype(mx.bfloat16)
    model.load_weights(list(clean_weights.items()), strict=False)

    print(f"Quantizing model to {bits}-bit ({policy})...")
    meta = quantize_model(model, bits=bits, group_size=group_size, policy=policy)

    # Flatten parameters with 'model.' prefix
    quantized_weights = dict(tree_flatten(model.parameters()))
    prefixed_weights = {}
    for k, v in quantized_weights.items():
        # Do not save codec weights in model.safetensors (it is saved in codec.safetensors)
        if k.startswith("codec."):
            continue
        prefixed_weights[f"model.{k}"] = v

    out_model_file = dst_path / "model.safetensors"
    print(f"Saving quantized weights to {out_model_file}...")
    mx.save_safetensors(str(out_model_file), prefixed_weights)

    # Update config.json with quantization details
    config_data["quantization"] = {
        "bits": bits,
        "group_size": group_size,
        "mode": "affine",
        "policy": policy,
    }
    with open(dst_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    # Copy supporting files (tokenizer, codec, metadata)
    for fname in (
        "codec.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        src_f = src_path / fname
        if src_f.exists():
            dst_f = dst_path / fname
            if not dst_f.exists():
                try:
                    # Attempt hard link first to save disk space
                    os.link(src_f, dst_f)
                except Exception:
                    shutil.copy(src_f, dst_f)

    src_size = source_weights_file.stat().st_size / (1024 * 1024)
    dst_size = out_model_file.stat().st_size / (1024 * 1024)
    reduction_pct = (1.0 - (dst_size / src_size)) * 100.0
    print(f"Size: {src_size:.2f} MB -> {dst_size:.2f} MB ({reduction_pct:.1f}% reduction)")

    meta["source_size_mb"] = round(src_size, 2)
    meta["quantized_size_mb"] = round(dst_size, 2)
    meta["reduction_pct"] = round(reduction_pct, 1)
    return meta
