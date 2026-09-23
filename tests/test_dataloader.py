"""DataLoader / TensorDataset tests: batching math, shuffle determinism,
edge-case honesty (partial + dropped batches)."""
import numpy as np
import pytest

from timet.tensor import tensor
from timet import train as tr


def _ds(n=10, d=3):
    x = tensor(np.arange(n * d, dtype=np.float32).reshape(n, d))
    y = tensor(np.arange(n) % 2)
    return tr.TensorDataset(x, y)


def test_tensor_dataset_holds_pairs_and_length():
    ds = _ds(10)
    assert len(ds) == 10
    xb, yb = ds[4]
    assert tuple(xb.shape) == (3,) and int(yb.item()) == 0  # 4 % 2 == 0


def test_dataset_rejects_mismatched_sample_counts():
    with pytest.raises(tr.TrainError) as ei:
        tr.TensorDataset(tensor(np.ones((5, 2), np.float32)),
                         tensor(np.ones((4,), np.float32)))
    assert "E0635" in str(ei.value)


def test_dataloader_batches_cover_every_sample_once():
    ds = _ds(10)
    seen = []
    for xb, yb in tr.DataLoader(ds, batch_size=4):
        seen.extend(int(xb.data[i, 0] // 3) for i in range(xb.shape[0]))
    assert sorted(seen) == list(range(10))
    assert len(tr.DataLoader(ds, batch_size=4)) == 3  # 4+4+2


def test_drop_last_drops_exactly_the_partial_batch():
    ds = _ds(10)
    batches = list(tr.DataLoader(ds, batch_size=4, drop_last=True))
    assert [xb.shape[0] for xb, _ in batches] == [4, 4]
    assert len(tr.DataLoader(ds, batch_size=4, drop_last=True)) == 2


def test_shuffle_is_seeded_and_deterministic_across_loaders():
    ds = _ds(12)
    def epoch_order(seed):
        ld = tr.DataLoader(ds, batch_size=5, shuffle=True, seed=seed)
        out = []
        for xb, _ in ld:
            out.extend(int(xb.data[i, 0] // 3) for i in range(xb.shape[0]))
        return out
    assert epoch_order(7) == epoch_order(7)          # same seed -> same order
    assert epoch_order(7) != epoch_order(8)          # different seed -> different order
    assert sorted(epoch_order(7)) == list(range(12)) # still a permutation


def test_reshuffle_between_epochs_is_deterministic():
    ds = _ds(12)
    ld1 = tr.DataLoader(ds, batch_size=5, shuffle=True, seed=3)
    ld2 = tr.DataLoader(ds, batch_size=5, shuffle=True, seed=3)
    e1 = [xb.data.copy() for xb, _ in ld1]   # epoch 1
    e1b = [xb.data.copy() for xb, _ in ld1]  # epoch 2 of same loader
    e2 = [xb.data.copy() for xb, _ in ld2]
    e2b = [xb.data.copy() for xb, _ in ld2]
    for a, b in zip(e1, e2):
        assert np.array_equal(a, b)
    for a, b in zip(e1b, e2b):
        assert np.array_equal(a, b)
    assert not np.array_equal(np.concatenate([b.reshape(-1) for b in e1]),
                              np.concatenate([b.reshape(-1) for b in e1b]))


def test_batch_size_validation():
    with pytest.raises(tr.TrainError) as ei:
        tr.DataLoader(_ds(), batch_size=0)
    assert "E0637" in str(ei.value)


def test_fit_loader_trains_on_symmetric_blocks():
    """20 "bright-first-half" rows vs 20 "bright-second-half", a single
    Linear must hit 100% train accuracy with mini-batches."""
    import timet.nn as nn
    from timet.nn import Linear, CrossEntropyLoss
    rng = np.random.default_rng(3)
    x = np.concatenate([np.concatenate([rng.normal(1, 0.2, (20, 5)),
                                        rng.normal(0, 0.2, (20, 5))], axis=1),
                        np.concatenate([rng.normal(0, 0.2, (20, 5)),
                                        rng.normal(1, 0.2, (20, 5))], axis=1)])
    y = np.array([1] * 20 + [0] * 20)
    # deterministic presentation order: alternate classes
    order = np.argsort(np.arange(40) * 2 % 41 + (np.arange(40) >= 20))
    ds = tr.TensorDataset(tensor(x[order]), tensor(y[order]))
    loader = tr.DataLoader(ds, batch_size=8, shuffle=True, seed=11)
    model = Linear(10, 2, seed=1)
    opt = __import__("timet.optim", fromlist=["Adam"]).Adam(model.parameters(), lr=0.05)
    hist = tr.fit_loader(model, CrossEntropyLoss(), opt, loader, epochs=40,
                         metrics={"acc": lambda m, xb, yb: tr.accuracy(m(xb), yb)})
    assert hist.losses[-1] < 0.05
    assert hist.metrics["acc"][-1] == 1.0
    assert len(hist.losses) == 40


def test_fit_loader_epoch_loss_is_mean_of_batches_not_last_batch():
    """First epoch's recorded loss must equal the MEAN of its batch losses
    (catches the silent 'record last batch' bug that misreports curves)."""
    import timet.nn as nn
    ds = _ds(8, 2)
    ld = tr.DataLoader(ds, batch_size=8 // 2, shuffle=False)
    model = nn.Linear(2, 2, seed=1)
    mce = nn.CrossEntropyLoss()
    opt = __import__("timet.optim", fromlist=["SGD"]).SGD(model.parameters(), lr=0.0)  # lr=0: params frozen
    hist = tr.fit_loader(model, mce, opt, ld, epochs=1)
    # compute expected with frozen params manually
    losses = []
    for xb, yb in ld:
        losses.append(float(mce(model(xb), yb).item()))
    assert abs(hist.losses[-1] - float(np.mean(losses))) < 1e-6


def test_fit_loader_rejects_empty_loader():
    import timet.nn as nn
    ds = _ds(3)
    ld = tr.DataLoader(ds, batch_size=5, drop_last=True)
    with pytest.raises(tr.TrainError) as ei:
        tr.fit_loader(nn.Linear(3, 2), nn.MSELoss(),
                      __import__("timet.optim", fromlist=["SGD"]).SGD(
                          nn.Linear(3, 2).parameters()), ld, epochs=1)
    assert "E0638" in str(ei.value)
