import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _rope(length: int, head_dim: int, base: float = 10000.0) -> mx.array:
    dim_indices = mx.arange(0, head_dim, 2, dtype=mx.float32)
    frequencies = 1.0 / (base ** (dim_indices / head_dim))
    t = mx.arange(length, dtype=mx.float32)
    phases = mx.outer(t, frequencies)
    cos = mx.cos(phases)
    sin = mx.sin(phases)
    return mx.stack([cos, sin], axis=-1)  # (length, head_dim // 2, 2)


def _apply_rope(x: mx.array, rope_values: mx.array) -> mx.array:
    # x is (B, num_heads, length, head_dim)
    batch, n_heads, length, head_dim = x.shape
    shaped = x.reshape(batch, n_heads, length, head_dim // 2, 2)
    # rope_values is (length, head_dim // 2, 2) -> broadcast to (1, 1, length, head_dim // 2, 2)
    cos = rope_values[None, None, :, :, 0]
    sin = rope_values[None, None, :, :, 1]
    x0 = shaped[..., 0]
    x1 = shaped[..., 1]
    out0 = x0 * cos - x1 * sin
    out1 = x1 * cos + x0 * sin
    return mx.stack([out0, out1], axis=-1).reshape(batch, n_heads, length, head_dim)


class ArkttsCodecRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return x * norm * self.weight


class ArkttsCodecLayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-2):
        super().__init__()
        self.gamma = mx.full((dim,), init_values)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class ArkttsSnake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # in MLX channels-last: alpha shape is (1, 1, channels)
        self.alpha = mx.ones((1, 1, channels))

    def __call__(self, x: mx.array) -> mx.array:
        # x is (B, T, C)
        return x + (1.0 / (self.alpha + 1e-9)) * (mx.sin(self.alpha * x) ** 2)


def _extra_padding(length: int, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    frames = (length - kernel_size + padding_total) / stride + 1
    ideal = (math.ceil(frames) - 1) * stride + kernel_size - padding_total
    return ideal - length


class ArkttsCausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=True,
        )
        self.stride = stride
        self.effective_kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.effective_kernel_size - self.stride

    def __call__(self, x: mx.array) -> mx.array:
        # x is (B, T, C)
        length = x.shape[1]
        right = _extra_padding(length, self.effective_kernel_size, self.stride, self.padding)
        if self.padding > 0 or right > 0:
            x = mx.pad(x, [(0, 0), (self.padding, right), (0, 0)])
        return self.conv(x)


class ArkttsCausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=True,
        )
        self.stride = stride
        self.kernel_size = kernel_size
        self.crop = kernel_size - stride

    def __call__(self, x: mx.array) -> mx.array:
        # x is (B, T, C)
        x = self.conv(x)
        if self.crop > 0:
            x = x[:, : x.shape[1] - self.crop, :]
        return x


class Sequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self._modules_list = list(modules)
        for i, m in enumerate(modules):
            setattr(self, str(i), m)

    def __call__(self, x: mx.array) -> mx.array:
        for m in self._modules_list:
            x = m(x)
        return x


class ArkttsResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = [
            ArkttsSnake1d(dim),
            ArkttsCausalConv1d(dim, dim, kernel_size=7, dilation=dilation),
            ArkttsSnake1d(dim),
            ArkttsCausalConv1d(dim, dim, kernel_size=1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        out = x
        for layer in self.block:
            out = layer(out)
        diff = residual.shape[1] - out.shape[1]
        if diff > 0:
            residual = residual[:, :-diff, :]
        return residual + out


class ArkttsConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = ArkttsCausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = mx.full((dim,), 1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        # x is (B, T, C)
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = nn.gelu(self.pwconv1(x))
        x = self.pwconv2(x)
        x = x * self.gamma
        return residual + x


@dataclass
class CodecTransformerConfig:
    n_layer: int
    n_head: int
    dim: int
    intermediate_size: int
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head


class ArkttsCodecAttention(nn.Module):
    def __init__(self, config: CodecTransformerConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim
        total = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)

    def __call__(self, x: mx.array, rope_values: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        batch, length, _ = x.shape
        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        qkv = self.wqkv(x)
        q = qkv[:, :, :q_size].reshape(batch, length, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = qkv[:, :, q_size : q_size + kv_size].reshape(batch, length, self.n_local_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = qkv[:, :, q_size + kv_size :].reshape(batch, length, self.n_local_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = _apply_rope(q, rope_values)
        k = _apply_rope(k, rope_values)

        if self.n_local_heads < self.n_head:
            repeats = self.n_head // self.n_local_heads
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, q_size)
        return self.wo(out)


class ArkttsCodecFeedForward(nn.Module):
    def __init__(self, config: CodecTransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class ArkttsCodecTransformerBlock(nn.Module):
    def __init__(self, config: CodecTransformerConfig):
        super().__init__()
        self.attention = ArkttsCodecAttention(config)
        self.feed_forward = ArkttsCodecFeedForward(config)
        self.attention_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.attention_layer_scale = ArkttsCodecLayerScale(config.dim)
        self.ffn_layer_scale = ArkttsCodecLayerScale(config.dim)

    def __call__(self, x: mx.array, rope_values: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.attention_layer_scale(self.attention(self.attention_norm(x), rope_values, mask))
        x = x + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(x)))
        return x


class ArkttsCodecWindowTransformer(nn.Module):
    def __init__(
        self,
        config: CodecTransformerConfig,
        input_dim: int,
        window_size: Optional[int] = 128,
        causal: bool = True,
    ):
        super().__init__()
        self.layers = [ArkttsCodecTransformerBlock(config) for _ in range(config.n_layer)]
        self.norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.window_size = window_size
        self.causal = causal
        self.head_dim = config.head_dim
        self.rope_base = config.rope_base
        self.input_proj = (
            nn.Linear(input_dim, config.dim) if input_dim != config.dim else None
        )
        self.output_proj = (
            nn.Linear(config.dim, input_dim) if input_dim != config.dim else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x is (B, T, C)
        if self.input_proj is not None:
            x = self.input_proj(x)
        length = x.shape[1]
        row = mx.arange(length)[:, None]
        column = mx.arange(length)[None, :]
        bool_mask = column <= row
        if self.window_size is not None:
            min_col = mx.clip(row - self.window_size + 1, a_min=0, a_max=length)
            bool_mask = bool_mask & (column >= min_col)
        # Convert boolean mask to additive attention mask: 0 for True, -1e9 for False
        attn_mask = mx.where(bool_mask[None, None, :, :], 0.0, -1e9)
        rope_values = _rope(length, self.head_dim, self.rope_base)
        for layer in self.layers:
            x = layer(x, rope_values, attn_mask)
        x = self.norm(x)
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x
