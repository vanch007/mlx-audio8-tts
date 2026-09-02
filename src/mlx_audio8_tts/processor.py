import math
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from transformers import AutoTokenizer


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ArkttsProcessor:
    def __init__(
        self,
        tokenizer: Any,
        num_codebooks: int = 10,
        semantic_begin_id: int = 151678,
        audio_sampling_rate: int = 44100,
    ):
        self.tokenizer = tokenizer
        self.num_codebooks = int(num_codebooks)
        self.semantic_begin_id = int(semantic_begin_id)
        self.audio_sampling_rate = int(audio_sampling_rate)

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "ArkttsProcessor":
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            use_fast=True,
            trust_remote_code=False,
        )
        return cls(tokenizer=tokenizer)

    def _encode(self, text: str) -> List[int]:
        if not text:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _format_reference_text(text: str) -> str:
        cleaned = clean_text(text)
        if re.search(r"<\|speaker:\d+\|>", cleaned):
            return cleaned
        return f"<|speaker:0|>{cleaned}"

    def _prompt_segments(
        self, text: str, reference_text: Optional[str] = None, has_reference: bool = False
    ) -> Tuple[List[int], List[int]]:
        target = clean_text(text)
        if not target:
            raise ValueError("Text must not be empty")

        if not has_reference:
            full_str = (
                "<|im_start|>system\n"
                "convert the provided text to speech"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{target}"
                "<|im_end|>\n"
                "<|im_start|>assistant\n<|voice|>"
            )
            return self._encode(full_str), []

        if not reference_text:
            raise ValueError("reference_text is required when reference audio is provided")

        prefix_str = (
            "<|im_start|>system\n"
            "convert the provided text to speech reference to the following:\n\nText:\n"
            f"{self._format_reference_text(reference_text)}"
            "\n\nSpeech:\n"
        )
        suffix_str = (
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{target}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n<|voice|>"
        )
        return self._encode(prefix_str), self._encode(suffix_str)

    def load_audio(
        self,
        audio_path: Union[str, Path, np.ndarray],
        target_sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        target_rate = target_sample_rate or self.audio_sampling_rate
        if isinstance(audio_path, (str, Path)):
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
            # convert to mono
            audio = data.mean(axis=1)
            if sr != target_rate:
                # resample if needed
                import scipy.signal
                num_samples = int(len(audio) * float(target_rate) / sr)
                audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
            return audio
        elif isinstance(audio_path, np.ndarray):
            if audio_path.ndim > 1:
                audio_path = audio_path.mean(axis=-1)
            return audio_path.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_path)}")

    def prepare_prompt(
        self,
        text: str,
        reference_text: Optional[str] = None,
        reference_codes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        has_ref = reference_codes is not None
        prefix_ids, suffix_ids = self._prompt_segments(
            text=text,
            reference_text=reference_text,
            has_reference=has_ref,
        )

        if not has_ref:
            total_len = len(prefix_ids)
            prompt = np.zeros((self.num_codebooks + 1, total_len), dtype=np.int64)
            prompt[0] = np.array(prefix_ids, dtype=np.int64)
            return prompt

        # reference_codes is shape (num_codebooks, ref_len)
        ref_len = reference_codes.shape[-1]
        sem_codes = reference_codes[0] + self.semantic_begin_id
        full_semantic = np.concatenate([prefix_ids, sem_codes, suffix_ids])
        total_len = len(full_semantic)

        prompt = np.zeros((self.num_codebooks + 1, total_len), dtype=np.int64)
        prompt[0] = full_semantic
        start_idx = len(prefix_ids)
        prompt[1:, start_idx : start_idx + ref_len] = reference_codes
        return prompt
