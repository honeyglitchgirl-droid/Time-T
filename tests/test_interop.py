"""Interoperability and model export tests (Milestone 11, Master Prompt §20, §24, §25).

Tests:
- HuggingFace safetensors export and round-trip loading
- NumPy .npz export and round-trip loading
- Content-based sniffing in load() for .safetensors and .npz
- CLI export command (time-t export in.ttck -o out.safetensors)
- CLI export format validation and error diagnostics (E0800, E0801)
"""
import json
import numpy as np
import pytest

import timet.nn as nn
from timet.tensor import Tensor, tensor
from timet import checkpoint
from timet.cli import main as cli_main


def _sample_model():
    return nn.Sequential(
        nn.Linear(4, 8, seed=1),
        nn.LayerNorm(8),
        nn.Linear(8, 2, seed=2)
    )


def test_safetensors_roundtrip(tmp_path):
    m = _sample_model()
    path = tmp_path / "model.safetensors"
    checkpoint.save_safetensors(m, path)
    assert path.is_file()

    # Content-sniffed load via generic checkpoint.load()
    loaded = checkpoint.load(path)
    sd = checkpoint.state_dict(m)
    assert sorted(loaded.keys()) == sorted(sd.keys())
    for k in sd:
        assert np.array_equal(loaded[k].data, sd[k].data)
        assert loaded[k].shape == sd[k].shape


def test_npz_roundtrip(tmp_path):
    m = _sample_model()
    path = tmp_path / "model.npz"
    checkpoint.save_npz(m, path)
    assert path.is_file()

    loaded = checkpoint.load(path)
    sd = checkpoint.state_dict(m)
    assert sorted(loaded.keys()) == sorted(sd.keys())
    for k in sd:
        assert np.array_equal(loaded[k].data, sd[k].data)


def test_cli_export_to_safetensors_and_npz(tmp_path, capsys):
    m = _sample_model()
    bin_path = tmp_path / "model.ttck"
    checkpoint.save_bin(m, bin_path)

    out_safe = tmp_path / "exported.safetensors"
    ret = cli_main(["export", str(bin_path), "-o", str(out_safe), "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    assert res["status"] == "ok"
    assert res["format"] == "safetensors"
    assert out_safe.is_file()

    # Load and verify exported safetensors
    loaded = checkpoint.load_safetensors(out_safe)
    assert len(loaded) == len(checkpoint.state_dict(m))

    out_npz = tmp_path / "exported.npz"
    ret2 = cli_main(["export", str(bin_path), "-o", str(out_npz)])
    assert ret2 == 0
    assert out_npz.is_file()
    loaded_npz = checkpoint.load_npz(out_npz)
    assert len(loaded_npz) == len(checkpoint.state_dict(m))


def test_cli_export_errors(tmp_path, capsys):
    ret1 = cli_main(["export", str(tmp_path / "nonexistent.ttck"), "-o", str(tmp_path / "out.safetensors"), "--json"])
    assert ret1 == 1
    captured = capsys.readouterr()
    err1 = json.loads(captured.err)
    assert "E0800" in err1["diagnostic"]["code"]
