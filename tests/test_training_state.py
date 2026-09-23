"""Training-state checkpoints (DD-21): the Adam-resume theorem that DD-20's
falsification demanded, scheduler continuity, loader RNG continuity, and
refusal-to-half-restore error paths."""

import numpy as np
import pytest

import timet.nn as nn
from timet import checkpoint as ck
from timet import train as tr
from timet.optim import Adam, AdamW, SGD
from timet.tensor import tensor


def _data():
    x = np.random.default_rng(0).normal(0, 1, (12, 4)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    return tensor(x), tensor(y)


def _model():
    return nn.Sequential(nn.Linear(4, 6, seed=1), nn.Tanh(), nn.Linear(6, 2, seed=2))


def test_adam_resume_is_byte_identical_to_uninterrupted(tmp_path):
    """THE theorem this tranche exists to make true (DD-20 falsified it for
    param-only checkpoints): train 20 -> save_state -> fresh model+opt ->
    load_state -> train 30 == train 50 with Adam, EXACTLY."""
    x, y = _data()
    ce = nn.CrossEntropyLoss()

    m1 = _model()
    o1 = Adam(m1.parameters(), lr=0.02)
    tr.fit(m1, ce, o1, x, y, epochs=20)
    ck.save_state(tmp_path / "s.ttck", model=m1, optimizer=o1, epoch=20)

    m2 = _model()
    o2 = Adam(m2.parameters(), lr=0.99)  # wrong lr on purpose: file wins
    info = ck.load_state(tmp_path / "s.ttck", model=m2, optimizer=o2)
    assert info["epoch"] == 20
    tr.fit(m2, ce, o2, x, y, epochs=30)

    m3 = _model()
    o3 = Adam(m3.parameters(), lr=0.02)
    tr.fit(m3, ce, o3, x, y, epochs=50)

    for name, t in ck.state_dict(m2).items():
        assert np.array_equal(t.data, ck.state_dict(m3)[name].data), name


def test_adamw_state_round_trips_weight_decay_and_moments(tmp_path):
    x, y = _data()
    m1 = _model()
    o1 = AdamW(m1.parameters(), lr=0.03, weight_decay=0.01)
    tr.fit(m1, nn.CrossEntropyLoss(), o1, x, y, epochs=3)
    ck.save_state(tmp_path / "s.ttck", model=m1, optimizer=o1)

    m2 = _model()
    o2 = AdamW(m2.parameters(), lr=0.03, weight_decay=0.99)
    ck.load_state(tmp_path / "s.ttck", model=m2, optimizer=o2)
    assert o2.weight_decay == 0.01
    assert len(o2._m) == len(o1._m)  # all moments present


def test_scheduler_state_continues_exactly(tmp_path):
    """Resumed StepLR must produce the SAME lr trajectory as an
    uninterrupted one (not restart at base, not skip)."""
    x, y = _data()
    m1 = _model()
    o1 = Adam(m1.parameters(), lr=0.1)
    s1 = tr.StepLR(o1, step_size=2, gamma=0.5)
    s1.step(); s1.step(); s1.step()        # 3 epochs in
    ck.save_state(tmp_path / "s.ttck", model=m1, optimizer=o1, scheduler=s1)

    m2 = _model()
    o2 = Adam(m2.parameters(), lr=0.1)
    s2 = tr.StepLR(o2, step_size=2, gamma=0.5)
    ck.load_state(tmp_path / "s.ttck", model=m2, optimizer=o2, scheduler=s2)
    s2.step()
    assert o2.lr == 0.1 * 0.5 ** 2         # epoch 4 -> floor(4/2)=2


def test_loader_rng_stream_continues_at_save_point(tmp_path):
    """A resumed loader's NEXT epoch order equals the uninterrupted
    loader's next epoch order -- shuffle streams are transportable."""
    from timet.train import TensorDataset, DataLoader
    x, y = _data()
    ds = TensorDataset(x, y)
    ld1 = DataLoader(ds, batch_size=4, shuffle=True, seed=9)
    e1 = [xb.data.copy() for xb, _ in ld1]          # uninterrupted epoch 1
    ck.save_state(tmp_path / "s.ttck", loaders={"train": ld1})
    e1b = [xb.data.copy() for xb, _ in ld1]         # uninterrupted epoch 2

    ld2 = DataLoader(ds, batch_size=4, shuffle=True, seed=12345)  # wrong seed
    ck.load_state(tmp_path / "s.ttck", loaders={"train": ld2})
    e2 = [xb.data.copy() for xb, _ in ld2]          # resumed epoch 2
    for got, want in zip(e2, e1b):
        assert np.array_equal(got, want)


def test_optimizer_state_present_but_no_optimizer_passed_errors(tmp_path):
    m = _model()
    o = Adam(m.parameters(), lr=0.1)
    ck.save_state(tmp_path / "s.ttck", model=m, optimizer=o)
    with pytest.raises(ck.CheckpointError) as ei:
        ck.load_state(tmp_path / "s.ttck", model=_model())  # optimizer missing
    assert "E0651" in str(ei.value)


def test_optimizer_type_mismatch_errors(tmp_path):
    m = _model()
    o = Adam(m.parameters(), lr=0.1)
    m_corner = _model()
    tr.fit(m, nn.CrossEntropyLoss(), o, *_data(), epochs=1)
    ck.save_state(tmp_path / "s.ttck", model=m, optimizer=o)
    with pytest.raises(ck.CheckpointError) as ei:
        ck.load_state(tmp_path / "s.ttck", model=_model(),
                      optimizer=SGD(_model().parameters(), lr=0.1))
    assert "E0652" in str(ei.value)


def test_shape_mismatch_between_state_and_model_errors(tmp_path):
    m = _model()
    o = Adam(m.parameters(), lr=0.1)
    ck.save_state(tmp_path / "s.ttck", model=m, optimizer=o)
    other = nn.Sequential(nn.Linear(4, 7, seed=1), nn.Tanh(), nn.Linear(7, 2, seed=2))
    with pytest.raises(ck.CheckpointError) as ei:
        ck.load_state(tmp_path / "s.ttck", model=other,
                      optimizer=Adam(other.parameters(), lr=0.1))
    assert "E0650" in str(ei.value)


def test_param_only_save_then_load_state_with_optimizer_errors(tmp_path):
    """A save_state WITHOUT optimizer + a load WITH optimizer is a refused
    silent mismatch (E0654), not a shrug."""
    m = _model()
    ck.save_state(tmp_path / "s.ttck", model=m)
    with pytest.raises(ck.CheckpointError) as ei:
        ck.load_state(tmp_path / "s.ttck", model=_model(),
                      optimizer=SGD(_model().parameters()))
    assert "E0654" in str(ei.value)


def test_unknown_loader_name_in_state_errors(tmp_path):
    from timet.train import TensorDataset, DataLoader
    x, y = _data()
    ld = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True, seed=1)
    ck.save_state(tmp_path / "s.ttck", loaders={"train": ld})
    with pytest.raises(ck.CheckpointError) as ei:
        ck.load_state(tmp_path / "s.ttck",
                      loaders={"vel": DataLoader(TensorDataset(x, y), batch_size=4)})
    assert "E0658" in str(ei.value)


def test_nonTraining_state_file_rejected(tmp_path):
    m = _model()
    ck.save_bin(m, tmp_path / "p.ttck")  # param-only binary, not training state
    with pytest.raises(ck.CheckpointError) as ei:
        ck.load_state(tmp_path / "p.ttck", model=_model())
    assert "E0649" in str(ei.value)


def test_state_saves_are_byte_identical(tmp_path):
    m = _model()
    o = Adam(m.parameters(), lr=0.1)
    tr.fit(m, nn.CrossEntropyLoss(), o, *_data(), epochs=2)
    a = ck.save_state(tmp_path / "a.ttck", model=m, optimizer=o, epoch=2)
    b = ck.save_state(tmp_path / "b.ttck", model=m, optimizer=o, epoch=2)
    assert a.read_bytes() == b.read_bytes()


def test_uninitialized_optimizer_state_round_trips(tmp_path):
    """Adam that never step()ed has no moments; resume must still work and
    step normally afterwards."""
    m = _model()
    o = Adam(m.parameters(), lr=0.05)
    ck.save_state(tmp_path / "s.ttck", model=m, optimizer=o)
    m2 = _model()
    o2 = Adam(m2.parameters(), lr=0.05)
    ck.load_state(tmp_path / "s.ttck", model=m2, optimizer=o2)
    x, y = _data()
    tr.fit(m2, nn.CrossEntropyLoss(), o2, x, y, epochs=1)
    assert len(o2._m) > 0
