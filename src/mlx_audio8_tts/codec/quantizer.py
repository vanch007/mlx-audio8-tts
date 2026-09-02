from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .layers import (
    ArkttsCausalConv1d,
    ArkttsCausalConvTranspose1d,
    ArkttsCodecWindowTransformer,
    ArkttsConvNeXtBlock,
    CodecTransformerConfig,
)


class ArkttsVectorQuantizer(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.in_proj = nn.Conv1d(input_dim, codebook_dim, kernel_size=1, bias=True)
        self.out_proj = nn.Conv1d(codebook_dim, input_dim, kernel_size=1, bias=True)

    def decode_code(self, indices: mx.array) -> mx.array:
        # indices: (B, T)
        latents = self.codebook(indices)
        return self.out_proj(latents)

    def decode_latents(self, latents: mx.array) -> Tuple[mx.array, mx.array]:
        # latents: (B, T, codebook_dim)
        batch, length, dim = latents.shape
        flattened = latents.reshape(batch * length, dim)
        flat_norm = mx.linalg.norm(flattened, axis=-1, keepdims=True)
        flattened = flattened / mx.maximum(flat_norm, 1e-9)
        cb = self.codebook.weight
        cb_norm = mx.linalg.norm(cb, axis=-1, keepdims=True)
        cb = cb / mx.maximum(cb_norm, 1e-9)
        sim = mx.matmul(flattened, cb.T)
        indices = mx.argmax(sim, axis=-1).reshape(batch, length)
        return self.decode_code(indices), indices


class ArkttsResidualQuantizer(nn.Module):
    def __init__(self, input_dim: int, n_codebooks: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.quantizers = [
            ArkttsVectorQuantizer(input_dim, codebook_size, codebook_dim)
            for _ in range(n_codebooks)
        ]

    def from_codes(self, codes: mx.array) -> mx.array:
        # codes: (B, n_codebooks, T)
        out = 0.0
        for i in range(codes.shape[1]):
            out = out + self.quantizers[i].decode_code(codes[:, i])
        return out


class ArkttsDownsampleQuantizer(nn.Module):
    def __init__(
        self,
        codec_post_n_layer: int = 8,
        codec_post_n_head: int = 16,
        codec_post_n_local_heads: int = 8,
        codec_post_intermediate_size: int = 1216,
    ):
        super().__init__()
        self.semantic_quantizer = ArkttsResidualQuantizer(1024, 1, 4096, 8)
        self.quantizer = ArkttsResidualQuantizer(1024, 9, 1024, 8)

        # downsample: 2 stages
        self.downsample = [
            [ArkttsCausalConv1d(1024, 1024, kernel_size=2, stride=2), ArkttsConvNeXtBlock(1024)],
            [ArkttsCausalConv1d(1024, 1024, kernel_size=2, stride=2), ArkttsConvNeXtBlock(1024)],
        ]

        # upsample: 2 stages
        self.upsample = [
            [ArkttsCausalConvTranspose1d(1024, 1024, kernel_size=2, stride=2), ArkttsConvNeXtBlock(1024)],
            [ArkttsCausalConvTranspose1d(1024, 1024, kernel_size=2, stride=2), ArkttsConvNeXtBlock(1024)],
        ]

        pre_transformer_config = CodecTransformerConfig(
            n_layer=8,
            n_head=16,
            dim=1024,
            intermediate_size=3072,
        )
        post_transformer_config = CodecTransformerConfig(
            n_layer=codec_post_n_layer,
            n_head=codec_post_n_head,
            n_local_heads=codec_post_n_local_heads,
            dim=1024,
            intermediate_size=codec_post_intermediate_size,
        )
        self.pre_module = ArkttsCodecWindowTransformer(pre_transformer_config, 1024, window_size=128)
        self.post_module = ArkttsCodecWindowTransformer(post_transformer_config, 1024, window_size=128)

    def encode(self, z: mx.array) -> mx.array:
        # z is (B, T, 1024)
        for stage in self.downsample:
            conv, block = stage
            z = block(conv(z))
        z = self.pre_module(z)
        sem_proj = self.semantic_quantizer.quantizers[0].in_proj(z)
        quantized_sem, sem_indices = self.semantic_quantizer.quantizers[0].decode_latents(sem_proj)
        residual = z - quantized_sem
        res_codes = []
        for q in self.quantizer.quantizers:
            q_proj = q.in_proj(residual)
            quantized, idx = q.decode_latents(q_proj)
            res_codes.append(idx)
            residual = residual - quantized
        # shape: (B, 10, T)
        all_codes = mx.concatenate([sem_indices[:, None, :], mx.stack(res_codes, axis=1)], axis=1)
        return all_codes

    def decode(self, indices: mx.array) -> mx.array:
        # indices: (B, 10, T)
        indices = mx.array(indices)
        sem_indices = mx.clip(indices[:, :1], 0, self.semantic_quantizer.codebook_size - 1)
        res_indices = mx.clip(indices[:, 1:], 0, self.quantizer.codebook_size - 1)
        semantic = self.semantic_quantizer.from_codes(sem_indices)
        residual = self.quantizer.from_codes(res_indices)
        h = self.post_module(semantic + residual)
        for stage in self.upsample:
            conv, block = stage
            h = block(conv(h))
        return h
