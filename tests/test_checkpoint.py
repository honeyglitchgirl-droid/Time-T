import json

import numpy as np
import pytest

from timet.nn import Linear, ReLU, Sequential, MSELoss
from timet.tensor import Tensor
from timet import checkpoint
from timet.checkpoint import CheckpointError
from tests.numerics import assert_close


def _model():
    return Sequential(Linear(2, 4, seed=0), ReLU(), Linear(4, 1, seed=1))


def test_state_dict_names_are_stable_and_complete():
    sd = checkpoint.state_dict(_model())
    assert sorted(sd.keys()) == ["0.bias", "0.weight", "2.bias", "2.weight"]


def test_save_is_deterministic_byte_for_byte(tmp_path):
    m = _model()
    p1 = checkpoint.save(m, tmp_path / "a.json")
    p2 = checkpoint.save(_model(), tmp_path / "b.json")
    assert p1.read_bytes() == p2.read_bytes()


def test_save_load_roundtrip_restores_parameters(tmp_path):
    model = _model()
    x = Tensor(np.random.default_rng(0).normal(size=(5, 2)).astype(np.float32))
    before = model(x).data.copy()
    path = checkpoint.save(model, tmp_path / "ckpt.json")
    # corrupt the model, then restore
    for p in model.parameters():
        p.data = np.zeros_like(p.data)
    checkpoint.load_into(model, path)
    assert_close(model(x).data, before, msg="model output after checkpoint restore")


def test_load_into_rejects_shape_mismatch(tmp_path):
    path = checkpoint.save(_model(), tmp_path / "ckpt.json")
    payload = json.loads(path.read_text())
    # tamper consistently (valid file, wrong shape for this model)
    payload["tensors"]["0.weight"]["shape"] = [1, 8]
    payload["tensors"]["0.weight"]["data"] = [0.0] * 8
    path.write_text(json.dumps(payload))
    with pytest.raises(CheckpointError) as e:
        checkpoint.load_into(_model(), path)
    assert "shape mismatch" in str(e.value)


def test_load_rejects_internally_inconsistent_tensor(tmp_path):
    path = checkpoint.save(_model(), tmp_path / "ckpt.json")
    payload = json.loads(path.read_text())
    # shape claims 8 values but data only has 4 -> corrupt file
    payload["tensors"]["0.weight"]["data"] = payload["tensors"]["0.weight"]["data"][:4]
    path.write_text(json.dumps(payload))
    with pytest.raises(CheckpointError) as e:
        checkpoint.load(path)
    assert "corrupt" in str(e.value)


def test_load_into_strict_rejects_missing_keys(tmp_path):
    path = checkpoint.save(_model(), tmp_path / "ckpt.json")
    payload = json.loads(path.read_text())
    del payload["tensors"]["2.bias"]
    path.write_text(json.dumps(payload))
    with pytest.raises(CheckpointError) as e:
        checkpoint.load_into(_model(), path)
    assert "missing" in str(e.value)


def test_load_rejects_wrong_format_and_version(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "other", "version": 1, "tensors": {}}))
    with pytest.raises(CheckpointError):
        checkpoint.load(bad)
    bad.write_text(json.dumps({"format": "timet-checkpoint", "version": 999, "tensors": {}}))
    with pytest.raises(CheckpointError):
        checkpoint.load(bad)


def test_checkpoint_then_continue_training_matches_uninterrupted(tmp_path):
    """Save mid-training, restore into a fresh model, continue: the result
    must equal an identical uninterrupted run (real resume semantics)."""
    # 2-feature inputs matching _model()'s Linear(2, 4) first layer: y = x0 + 2*x1
    x = Tensor(np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=np.float32))
    y = Tensor(np.array([[2.0], [1.0], [4.0]], dtype=np.float32))
    ckpt = tmp_path / "mid.json"

    def train_some(model, epochs):
        from timet.optim import SGD
        loss_fn = MSELoss()
        opt = SGD(model.parameters(), lr=0.05)
        for _ in range(epochs):
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad()

    ref = _model()
    train_some(ref, 50)

    part = _model()
    train_some(part, 20)
    checkpoint.save(part, ckpt)
    resumed = _model()
    checkpoint.load_into(resumed, ckpt)
    train_some(resumed, 30)

    for a, b in zip(ref.parameters(), resumed.parameters()):
        assert_close(a.data, b.data, msg="resumed training diverged from uninterrupted run")
