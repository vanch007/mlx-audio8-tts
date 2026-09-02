import mlx.core as mx
import mlx.nn as nn

from .layers import (
    ArkttsCausalConv1d,
    ArkttsCodecWindowTransformer,
    ArkttsResidualUnit,
    ArkttsSnake1d,
    CodecTransformerConfig,
)


class ArkttsEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int, transformer_layers: int):
        super().__init__()
        self.block = [
            ArkttsResidualUnit(dim // 2, 1),
            ArkttsResidualUnit(dim // 2, 3),
            ArkttsResidualUnit(dim // 2, 9),
            ArkttsSnake1d(dim // 2),
            ArkttsCausalConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride),
        ]
        if transformer_layers > 0:
            config = CodecTransformerConfig(
                n_layer=transformer_layers,
                n_head=dim // 64,
                dim=dim,
                intermediate_size=dim * 3,
            )
            self.block.append(ArkttsCodecWindowTransformer(config, dim, window_size=512))

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class ArkttsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 64
        self.block = [ArkttsCausalConv1d(1, dim, kernel_size=7)]
        for stride, transformer_layers in zip((2, 4, 8, 8), (0, 0, 0, 4)):
            dim *= 2
            self.block.append(ArkttsEncoderBlock(dim, stride, transformer_layers))
        self.block.extend([ArkttsSnake1d(dim), ArkttsCausalConv1d(dim, 1024, kernel_size=3)])

    def __call__(self, x: mx.array) -> mx.array:
        # x is (B, num_samples, 1)
        for layer in self.block:
            x = layer(x)
        return x
