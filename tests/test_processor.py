from unittest.mock import MagicMock
import numpy as np
from mlx_audio8_tts.processor import ArkttsProcessor, clean_text


def test_clean_text():
    assert clean_text("  hello   world  \n ") == "hello world"
    assert clean_text("") == ""


def test_processor_prompt_preparation():
    mock_tokenizer = MagicMock()
    # Mock encode to return simple token lists based on length
    mock_tokenizer.encode.side_effect = lambda text, **kwargs: [1, 2, 3]

    proc = ArkttsProcessor(tokenizer=mock_tokenizer)

    # 1. Without reference
    prompt = proc.prepare_prompt(text="Hello world")
    assert prompt.ndim == 2
    assert prompt.shape[0] == 11  # num_codebooks + 1
    assert prompt.shape[1] > 0

    # 2. With reference codes
    ref_codes = np.ones((10, 5), dtype=np.int64)
    prompt_ref = proc.prepare_prompt(
        text="Hello world",
        reference_text="Sample reference",
        reference_codes=ref_codes,
    )
    assert prompt_ref.ndim == 2
    assert prompt_ref.shape[0] == 11
    assert prompt_ref.shape[1] > 5
