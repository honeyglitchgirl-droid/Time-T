"""Binary checkpoint (v2, DD-20): determinism, round-trip, manifest
integrity checks, JSON compat, and the resume-equivalence theorem."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import timet.nn as nn
from timet import checkpoint as ck
from timet import train as tr
from timet.optim import Adam


def _model():
    return nn.Sequential(nn.Linear(4, 6, seed=1), nn.Tanh(), nn.Linear(6, 2, seed=2))


def _data():
    x = np.random.default_rng(0).normal(0, 1, (12, 4)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    from timet.tensor import tensor
    return tensor(x), tensor(y)


def test_save_bin_is_byte_identical_across_runs(tmp_path):
    m = _model()
    p1 = ck.save_bin(m, tmp_path / "a.ttck")
    p2 = ck.save_bin(m, tmp_path / "b.ttck")
    assert p1.read_bytes() == p2.read_bytes()


def test_binary_is_much_smaller_than_json(tmp_path):
    """The honest reason v2 exists: raw f32 payload beats decimal text.
    Measured on a model large enough that text overhead dominates zip
    overhead (at tiny sizes both are just header noise)."""
    m = nn.Sequential(nn.Linear(64, 128, seed=1), nn.Tanh(), nn.Linear(128, 16, seed=2))
    pj = ck.save(m, tmp_path / "m.json")
    pb = ck.save_bin(m, tmp_path / "m.ttck")
    ratio = pb.stat().st_size / pj.stat().st_size
    assert ratio < 0.25, f"binary not smaller: {pj.stat().st_size} vs {pb.stat().st_size}"


def test_round_trip_values_and_shapes(tmp_path):
    m = _model()
    named_before = ck.state_dict(m)
    path = ck.save_bin(m, tmp_path / "m.ttck")
    loaded = ck.load(path)  # sniffed by content, not extension
    assert sorted(loaded) == sorted(named_before)
    for name, old in named_before.items():
        assert loaded[name].shape == old.shape
        assert np.array_equal(loaded[name].data, old.data.astype(np.float32))


def test_load_into_resumes_identically_from_binary(tmp_path):
    """The v1 resume theorem, re-proven for v2: train 20, save binary,
    restore elsewhere, train 30 more == uninterrupted train 50.

    Uses PLAIN SGD deliberately, exactly like the v1 theorem: checkpoints
    (both formats) store PARAMETER VALUES ONLY -- optimizer state is
    documented-not-checkpointed -- so the equivalence claim is
    'resume with a fresh optimizer of the same family', never
    'resume Adam's moments'. An Adam version of this 'theorem' would be
    false (verified while writing this test: fresh Adam state diverges
    from continuous Adam by up to 0.35 on this task) -- that finding is
    recorded in DD-20.
    """
    from timet.optim import SGD
    x, y = _data()
    ce = nn.CrossEntropyLoss()

    def train_some(model, epochs):
        tr.fit(model, ce, SGD(model.parameters(), lr=0.05), x, y, epochs=epochs)

    m1 = _model()
    train_some(m1, 20)
    ck.save_bin(m1, tmp_path / "r.ttck")
    m1b = _model()
    ck.load_into(m1b, tmp_path / "r.ttck")
    train_some(m1b, 30)

    m2 = _model()
    train_some(m2, 50)

    for name, t in ck.state_dict(m1b).items():
        assert np.array_equal(t.data, ck.state_dict(m2)[name].data), name


def test_extension_does_not_matter_for_load(tmp_path):
    m = _model()
    weird = tmp_path / "model.weird"
    ck.save_bin(m, weird)
    out = ck.load(weird)
    assert sorted(out) == sorted(ck.state_dict(m))


def test_json_checkpoints_still_load_after_binary_exists(tmp_path):
    m = _model()
    p = ck.save(m, tmp_path / "m.json")
    out = ck.load(p)
    assert sorted(out) == sorted(ck.state_dict(m))


def _tamper_manifest(path: Path, mutate):
    """Rewrite the zip with a mutated manifest (keeps entries valid)."""
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(path, "r") as zf, \
            zipfile.ZipFile(tmp, "w") as out:
        for info in zf.infolist():
            data = zf.read(info.filename)
            if info.filename == "manifest.json":
                data = json.dumps(mutate(json.loads(data.decode()))).encode()
            out.writestr(info.filename, data)
    tmp.replace(path)


def test_truncated_binary_is_clean_error(tmp_path):
    p = ck.save_bin(_model(), tmp_path / "m.ttck")
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])
    with pytest.raises(ck.CheckpointError) as ei:
        ck.load(p)
    assert "E0645" in str(ei.value)


def test_missing_tensor_entry_is_structural_error(tmp_path):
    p = ck.save_bin(_model(), tmp_path / "m.ttck")
    def drop_one(m):
        name = next(iter(m["tensors"]))
        m["tensors"] = {}
        return m
    _tamper_manifest(p, drop_one)
    with pytest.raises(ck.CheckpointError) as ei:
        ck.load_bin(p)
    assert "E0648" in str(ei.value) and "extra" in str(ei.value)


def test_wrong_version_is_rejected(tmp_path):
    p = ck.save_bin(_model(), tmp_path / "m.ttck")
    _tamper_manifest(p, lambda m: {**m, "version": 99})
    with pytest.raises(ck.CheckpointError) as ei:
        ck.load_bin(p)
    assert "E0648" in str(ei.value)


def test_wrong_format_key_is_rejected(tmp_path):
    p = ck.save_bin(_model(), tmp_path / "m.ttck")
    _tamper_manifest(p, lambda m: {**m, "format": "pickle", "version": 2})
    with pytest.raises(ck.CheckpointError) as ei:
        ck.load_bin(p)
    assert "E0647" in str(ei.value)


def test_shape_mismatch_inside_entry_is_detected(tmp_path):
    p = ck.save_bin(_model(), tmp_path / "m.ttck")
    def lie(m):
        name = next(iter(m["tensors"]))
        m["tensors"][name]["shape"] = [1, 1]  # lie about shape
        return m
    _tamper_manifest(p, lie)
    with pytest.raises(ck.CheckpointError) as ei:
        ck.load_bin(p)
    assert "E0648" in str(ei.value)
