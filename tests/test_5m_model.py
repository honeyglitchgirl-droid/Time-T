"""Tests for Time-T 5M Language Model (TransformerLM-5M)."""
import numpy as np
import pytest
from pathlib import Path

from models.transformer_lm_5m import TimeTLanguageModel5M, create_model
from timet.tensor import tensor
import timet.checkpoint as checkpoint


def test_model_parameter_count():
    model = create_model()
    pcount = model.count_parameters()
    # 5.16 Million parameters
    assert 5_000_000 <= pcount <= 5_500_000
    assert pcount == 5_162_080


def test_model_forward_and_generate():
    model = create_model()
    prompt = "The "
    out = model.generate(prompt, max_new_tokens=10, temperature=0.5)
    assert isinstance(out, str)
    assert out.startswith(prompt)


def test_checkpoint_roundtrip(tmp_path):
    model = create_model()
    ckpt_path = tmp_path / "test_5m.ttck"
    checkpoint.save_bin(model.lm, ckpt_path)
    assert ckpt_path.is_file()

    model2 = create_model()
    checkpoint.load_into(model2.lm, ckpt_path)

    # Check weights match exactly
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        np.testing.assert_array_equal(p1.data, p2.data)
