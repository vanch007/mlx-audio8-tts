import math
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .codec import ArkttsCodec
from .config import ArkttsConfig


def _precompute_rope(length: int, head_dim: int, base: float = 1000000.0) -> mx.array:
    dim_indices = mx.arange(0, head_dim, 2, dtype=mx.float32)
    frequencies = 1.0 / (base ** (dim_indices / head_dim))
    t = mx.arange(length, dtype=mx.float32)
    phases = mx.outer(t, frequencies)
    cos = mx.cos(phases)
    sin = mx.sin(phases)
    return mx.stack([cos, sin], axis=-1)  # (length, head_dim // 2, 2)


def _apply_rope(x: mx.array, rope_values: mx.array) -> mx.array:
    # x is (B, n_heads, length, head_dim)
    batch, n_heads, length, head_dim = x.shape
    shaped = x.reshape(batch, n_heads, length, head_dim // 2, 2)
    cos = rope_values[None, None, :, :, 0]
    sin = rope_values[None, None, :, :, 1]
    x0 = shaped[..., 0]
    x1 = shaped[..., 1]
    out0 = x0 * cos - x1 * sin
    out1 = x1 * cos + x0 * sin
    return mx.stack([out0, out1], axis=-1).reshape(batch, n_heads, length, head_dim)


class ArkttsRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return x * norm * self.weight


class ArkttsAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        qkv_bias: bool = True,
        output_bias: bool = False,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_local_heads = n_local_heads
        self.head_dim = head_dim
        total = (n_head + 2 * n_local_heads) * head_dim
        self.wqkv = nn.Linear(dim, total, bias=qkv_bias)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=output_bias)

    def __call__(
        self,
        x: mx.array,
        rope: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        batch, length, _ = x.shape
        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        qkv = self.wqkv(x)
        q = qkv[:, :, :q_size].reshape(batch, length, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = qkv[:, :, q_size : q_size + kv_size].reshape(batch, length, self.n_local_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = qkv[:, :, q_size + kv_size :].reshape(batch, length, self.n_local_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = _apply_rope(q, rope)
        k = _apply_rope(k, rope)

        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)
            new_cache = (k, v)
        else:
            new_cache = None

        if self.n_local_heads < self.n_head:
            repeat = self.n_head // self.n_local_heads
            k_proj = mx.repeat(k, repeat, axis=1)
            v_proj = mx.repeat(v, repeat, axis=1)
        else:
            k_proj = k
            v_proj = v

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = mx.matmul(q, k_proj.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        out = mx.matmul(weights, v_proj)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, q_size)
        return self.wo(out), new_cache


class ArkttsFeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class ArkttsTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        qkv_bias: bool = True,
        output_bias: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attention = ArkttsAttention(
            dim=dim,
            n_head=n_head,
            n_local_heads=n_local_heads,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            output_bias=output_bias,
        )
        self.feed_forward = ArkttsFeedForward(dim, intermediate_size)
        self.attention_norm = ArkttsRMSNorm(dim, norm_eps)
        self.ffn_norm = ArkttsRMSNorm(dim, norm_eps)

    def __call__(
        self,
        x: mx.array,
        rope: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        attn_out, new_cache = self.attention(self.attention_norm(x), rope, mask, cache=cache)
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_cache


def _sample(logits: mx.array, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> mx.array:
    if temperature <= 1e-5:
        return mx.argmax(logits, axis=-1)

    # Apply temperature
    scores = logits / max(temperature, 1e-5)

    # Apply top-k filtering if top_k > 0
    vocab_size = scores.shape[-1]
    if 0 < top_k < vocab_size:
        topk_indices = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
        # Gather top-k scores
        topk_scores = mx.take_along_axis(scores, topk_indices, axis=-1)
        # Softmax over top-k
        probs = mx.softmax(topk_scores, axis=-1)
        sampled_k = mx.random.categorical(mx.log(mx.clip(probs, 1e-9, 1.0)))
        return mx.take_along_axis(topk_indices, sampled_k[..., None], axis=-1).squeeze(-1)
    else:
        probs = mx.softmax(scores, axis=-1)
        return mx.random.categorical(mx.log(mx.clip(probs, 1e-9, 1.0)))


class ArkttsModel(nn.Module):
    def __init__(self, config: ArkttsConfig):
        super().__init__()
        self.config = config

        # Slow AR Backbone
        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.codebook_embeddings = nn.Embedding(config.codebook_size * config.num_codebooks, config.dim)
        self.layers = [
            ArkttsTransformerBlock(
                dim=config.dim,
                intermediate_size=config.intermediate_size,
                n_head=config.n_head,
                n_local_heads=config.n_local_heads,
                head_dim=config.head_dim,
                qkv_bias=True,
                output_bias=False,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.n_layer)
        ]
        self.norm = ArkttsRMSNorm(config.dim, config.norm_eps)

        # Fast AR Depth Decoder
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        self.fast_layers = [
            ArkttsTransformerBlock(
                dim=config.fast_dim,
                intermediate_size=config.fast_intermediate_size,
                n_head=config.fast_n_head,
                n_local_heads=config.fast_n_local_heads,
                head_dim=config.fast_head_dim,
                qkv_bias=False,
                output_bias=False,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.n_fast_layer)
        ]
        self.fast_norm = ArkttsRMSNorm(config.fast_dim, config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)
        self.fast_project_in = (
            nn.Linear(config.dim, config.fast_dim)
            if config.dim != config.fast_dim
            else None
        )

        # Precomputed RoPE tables
        self.slow_rope = _precompute_rope(config.max_seq_len, config.head_dim, config.rope_base)
        self.fast_rope = _precompute_rope(config.num_codebooks, config.fast_head_dim, config.rope_base)

        # Codec
        self.codec = ArkttsCodec(config)

    def _embed(self, input_ids: mx.array) -> mx.array:
        # input_ids: (B, num_codebooks + 1, T)
        codebook_embeds = []
        for idx in range(self.config.num_codebooks):
            codebook_embeds.append(
                self.codebook_embeddings(input_ids[:, idx + 1] + idx * self.config.codebook_size)
            )
        codebook_sum = mx.sum(mx.stack(codebook_embeds, axis=1), axis=1)  # (B, T, dim)
        is_semantic = (input_ids[:, 0] >= self.config.semantic_begin_id) & (
            input_ids[:, 0] <= self.config.semantic_end_id
        )
        codebook_sum = mx.where(is_semantic[..., None], codebook_sum, 0.0)
        text_embed = self.embeddings(input_ids[:, 0])
        return text_embed + codebook_sum

    def _slow_step(
        self,
        input_ids: mx.array,
        position_offset: int = 0,
        caches: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, mx.array, List[Tuple[mx.array, mx.array]]]:
        # input_ids: (B, num_codebooks + 1, seq_len)
        hidden = self._embed(input_ids)
        seq_len = hidden.shape[1]
        rope = self.slow_rope[position_offset : position_offset + seq_len]

        # Causal mask for prefill
        if seq_len > 1:
            row = mx.arange(seq_len)[:, None]
            col = mx.arange(seq_len)[None, :]
            mask = mx.where(col <= row, 0.0, -1e9)[None, None, :, :]
        else:
            mask = None

        new_caches = []
        for i, layer in enumerate(self.layers):
            c = caches[i] if caches is not None else None
            hidden, new_c = layer(hidden, rope, mask=mask, cache=c)
            new_caches.append(new_c)

        last_hidden = hidden[:, -1:, :]
        norm_hidden = self.norm(last_hidden)
        # Logits over vocab: linear projection using embeddings weight
        logits = self.embeddings.as_linear(norm_hidden).squeeze(1)
        fast_hidden = norm_hidden if self.config.norm_fastlayer_input else last_hidden
        return logits, fast_hidden, new_caches

    def _fast_step(
        self,
        hidden: mx.array,
        position: int,
        caches: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        rope = self.fast_rope[position : position + 1]
        new_caches = []
        for i, layer in enumerate(self.fast_layers):
            c = caches[i] if caches is not None else None
            hidden, new_c = layer(hidden, rope, mask=None, cache=c)
            new_caches.append(new_c)
        logits = self.fast_output(self.fast_norm(hidden)).squeeze(1)
        return logits, new_caches

    def _generate_codebooks(
        self,
        slow_hidden: mx.array,
        semantic_token: int,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> mx.array:
        # 1. Project slow hidden into fast input
        hidden = slow_hidden
        if self.fast_project_in is not None:
            hidden = self.fast_project_in(hidden)

        # Fast step at 0
        _, fast_caches = self._fast_step(hidden, position=0, caches=None)

        # Codebook 0 is fixed from semantic token
        cb0 = max(0, min(self.config.codebook_size - 1, semantic_token - self.config.semantic_begin_id))
        codebooks = [cb0]

        current_token = mx.array([[cb0]], dtype=mx.int32)
        hidden = self.fast_embeddings(current_token)

        # Sample codebooks 1..9
        for pos in range(1, self.config.num_codebooks):
            logits, fast_caches = self._fast_step(hidden, position=pos, caches=fast_caches)
            sampled_id = int(_sample(logits, temperature=temperature, top_k=top_k, top_p=top_p).item())
            codebooks.append(sampled_id)
            current_token = mx.array([[sampled_id]], dtype=mx.int32)
            hidden = self.fast_embeddings(current_token)

        return mx.array(codebooks, dtype=mx.int32)  # (10,)

    def generate(
        self,
        prompt: Union[mx.array, np.ndarray],
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        stream: bool = False,
        streaming_interval: int = 2,
    ) -> Generator[mx.array, None, None]:
        if isinstance(prompt, np.ndarray):
            prompt = mx.array(prompt)
        if prompt.ndim == 2:
            prompt = prompt[None, :, :]  # (1, num_codebooks + 1, T)

        batch_size, num_cb_plus_1, prompt_len = prompt.shape
        assert batch_size == 1, "Currently single batch generation is supported"

        # Prefill slow AR with prompt
        logits, fast_hidden, slow_caches = self._slow_step(prompt, position_offset=0, caches=None)

        generated_codes = []
        recent_semantic = []
        stream_buffer = []

        for step in range(max_new_tokens):
            # Apply semantic mask: only allow [semantic_begin_id, semantic_end_id] and eos_token_id
            mask = mx.full(logits.shape, -1e9)
            # Allow semantic tokens and EOS
            indices = mx.arange(self.config.semantic_begin_id, self.config.semantic_end_id + 1)
            logits_slice = logits[:, self.config.semantic_begin_id : self.config.semantic_end_id + 1]
            eos_logit = logits[:, self.config.eos_token_id : self.config.eos_token_id + 1]

            # Sample semantic token with Repetition-Aware Sampling (RAS)
            sampled_token = int(_sample(logits, temperature=temperature, top_k=top_k, top_p=top_p).item())
            if (
                sampled_token < self.config.semantic_begin_id
                or sampled_token > self.config.semantic_end_id
            ) and sampled_token != self.config.eos_token_id:
                # Fallback to argmax over allowed semantic range
                sampled_token = int(
                    (self.config.semantic_begin_id + mx.argmax(logits_slice, axis=-1)).item()
                )

            # Repetition detection
            if (
                len(recent_semantic) >= self.config.ras_window_size
                and sampled_token in recent_semantic[-self.config.ras_window_size :]
            ):
                # Repetition detected: sample with higher temperature
                sampled_token = int(
                    _sample(
                        logits,
                        temperature=self.config.ras_temperature,
                        top_k=top_k,
                        top_p=self.config.ras_top_p,
                    ).item()
                )

            if sampled_token == self.config.eos_token_id:
                break

            recent_semantic.append(sampled_token)

            # Generate 10 codebooks for this audio frame
            frame_codes = self._generate_codebooks(
                fast_hidden,
                sampled_token,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )  # (10,)
            generated_codes.append(frame_codes)

            if stream:
                stream_buffer.append(frame_codes)
                if len(stream_buffer) >= streaming_interval:
                    chunk_codes = mx.stack(stream_buffer, axis=-1)[None, :, :]  # (1, 10, interval)
                    chunk_audio = self.codec.decode(chunk_codes)
                    yield chunk_audio
                    stream_buffer.clear()

            # Next slow step input: construct (1, 11, 1) token
            next_input = mx.zeros((1, self.config.num_codebooks + 1, 1), dtype=mx.int64)
            next_input[0, 0, 0] = sampled_token
            for c_idx in range(self.config.num_codebooks):
                next_input[0, c_idx + 1, 0] = int(frame_codes[c_idx].item())

            pos = prompt_len + step
            logits, fast_hidden, slow_caches = self._slow_step(
                next_input, position_offset=pos, caches=slow_caches
            )

        # Handle remaining stream buffer or complete generation
        if stream:
            if stream_buffer:
                chunk_codes = mx.stack(stream_buffer, axis=-1)[None, :, :]
                chunk_audio = self.codec.decode(chunk_codes)
                yield chunk_audio
        else:
            if generated_codes:
                all_codes = mx.stack(generated_codes, axis=-1)[None, :, :]  # (1, 10, total_frames)
                full_audio = self.codec.decode(all_codes)
                yield full_audio
            else:
                yield mx.zeros((1, 0), dtype=mx.float32)
