from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .encoder import ArkttsEncoder
from .layers import (
    ArkttsCausalConv1d,
    ArkttsCausalConvTranspose1d,
    ArkttsResidualUnit,
    ArkttsSnake1d,
)
from .quantizer import ArkttsDownsampleQuantizer


class ArkttsDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = [
            ArkttsSnake1d(input_dim),
            ArkttsCausalConvTranspose1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride),
            ArkttsResidualUnit(output_dim, dilation=1),
            ArkttsResidualUnit(output_dim, dilation=3),
            ArkttsResidualUnit(output_dim, dilation=9),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class ArkttsDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 1536
        self.model = [ArkttsCausalConv1d(1024, channels, kernel_size=7)]
        for index, stride in enumerate((8, 8, 4, 2)):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            self.model.append(ArkttsDecoderBlock(input_dim, output_dim, stride))
        self.model.append(ArkttsSnake1d(output_dim))
        self.model.append(ArkttsCausalConv1d(output_dim, 1, kernel_size=7))

    def __call__(self, x: mx.array) -> mx.array:
        # x is (B, T, C=1024)
        for layer in self.model:
            x = layer(x)
        # final tanh
        x = mx.tanh(x)
        # Output is (B, num_samples, 1) -> squeeze to (B, num_samples)
        return x.squeeze(-1)


class ArkttsCodec(nn.Module):
    sample_rate: int = 44100
    frame_length: int = 2048
    hop_length: int = 512

    def __init__(self, config=None):
        super().__init__()
        post_layer = int(getattr(config, "codec_post_n_layer", 8)) if config else 8
        post_head = int(getattr(config, "codec_post_n_head", 16)) if config else 16
        post_local_heads = int(getattr(config, "codec_post_n_local_heads", 8)) if config else 8
        post_inter = int(getattr(config, "codec_post_intermediate_size", 1216)) if config else 1216

        self.encoder = ArkttsEncoder()
        self.quantizer = ArkttsDownsampleQuantizer(
            codec_post_n_layer=post_layer,
            codec_post_n_head=post_head,
            codec_post_n_local_heads=post_local_heads,
            codec_post_intermediate_size=post_inter,
        )
        self.decoder = ArkttsDecoder()

    def encode(self, audio: mx.array) -> mx.array:
        # audio: (B, num_samples) or (B, num_samples, 1)
        if audio.ndim == 2:
            audio = audio[:, :, None]
        encoded = self.encoder(audio)
        codes = self.quantizer.encode(encoded)
        return codes

    def decode(self, codes: mx.array) -> mx.array:
        # codes: (B, 10, num_frames)
        features = self.quantizer.decode(codes)
        audio = self.decoder(features)
        return audio
