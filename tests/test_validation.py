"""Validation-split plumbing: val_loss recording, monitor selection,
eval-mode discipline, and early-stopping-on-val (Milestone 8, v0.7.0)."""
import numpy as np
import pytest

import timet.nn as nn
from timet.tensor import Tensor, tensor
from timet.optim import Adam, SGD
from timet import train as tr


def _split_data(n=20, d=6, seed=5, flip_val=False):
    """Two well-separated blobs; flip_val inverts the LABELS on the val
    half so a model that learns train MUST get val loss rising."""
    rng = np.random.default_rng(seed)
    xa = rng.normal([1.0] * d, 0.2, (n, d)).astype(np.float32)
    xb = rng.normal([-1.0] * d, 0.2, (n, d)).astype(np.float32)
    x = np.concatenate([xa, xb]); y = np.array([0] * n + [1] * n)
    xval, yval = x.copy(), y.copy()
    if flip_val:
        yval = 1 - yval
    return (tr.TensorDataset(tensor(x), tensor(y)),
            tr.TensorDataset(tensor(xval), tensor(yval)))


def _model(d=6):
    return nn.Linear(d, 2, seed=1)


def test_val_loss_is_recorded_and_num_epochs_wide():
    trn, val = _split_data()
    loader = tr.DataLoader(trn, batch_size=8, shuffle=True, seed=1)
    vloader = tr.DataLoader(val, batch_size=8)
    model = _model()
    hist = tr.fit_loader(model, nn.CrossEntropyLoss(),
                         Adam(model.parameters(), lr=0.05),
                         loader, epochs=3, val_loader=vloader)
    assert len(hist.val_losses) == 3
    assert all(np.isfinite(v) for v in hist.val_losses)
    assert hist.losses[-1] != hist.val_losses[-1]  # distinct series


def test_no_val_loader_keeps_previous_behavior():
    trn, _ = _split_data()
    model = _model()
    h = tr.fit_loader(model, nn.CrossEntropyLoss(),
                      Adam(model.parameters(), lr=0.05),
                      tr.DataLoader(trn, batch_size=8, seed=2), epochs=2)
    assert h.val_losses == []


def test_val_runs_with_dropout_silenced_and_mode_restored():
    """Dropout(p=0.9) model left in train(): during the val pass dropout
    must be OFF (eval) and afterwards the model must be back in train()."""
    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(6, 2, seed=1)
            self.drop = nn.Dropout(0.9)
            self.modes_seen = []
        def forward(self, x):
            self.modes_seen.append(self.training)
            return self.lin(self.drop(x))

    trn, val = _split_data()
    probe = Probe()
    probe.train()
    tr.fit_loader(probe, nn.CrossEntropyLoss(),
                  SGD(probe.parameters(), lr=0.0),   # frozen: batch count == mode-sample count
                  tr.DataLoader(trn, batch_size=10, shuffle=False), epochs=1,
                  val_loader=tr.DataLoader(val, batch_size=10))
    # 4 train batches + 4 val batches seen (40 samples, bs=10)
    assert len(probe.modes_seen) == 8
    assert all(m for m in probe.modes_seen[:4])          # train: dropout active
    assert not any(probe.modes_seen[4:])                 # val: eval mode
    assert probe.training is True                        # restored


def test_early_stopping_can_monitor_val_loss():
    """Labels flipped on val: model fits train (train loss < ...), val RISES;
    ES(monitor='val_loss', patience=1) must stop early; default ES would not."""
    trn, val = _split_data(flip_val=True)
    es = tr.EarlyStopping(patience=1)
    model = _model()
    hist = tr.fit_loader(model, nn.CrossEntropyLoss(),
                         Adam(model.parameters(), lr=0.1),
                         tr.DataLoader(trn, batch_size=8, shuffle=True, seed=3),
                         epochs=30, val_loader=tr.DataLoader(val, batch_size=8),
                         early_stopping=es, monitor="val_loss")
    assert len(hist.losses) < 30
    # and val loss indeed climbed by the stop point vs its minimum
    best = min(hist.val_losses)
    assert hist.val_losses[-1] >= best


def test_monitor_val_loss_without_val_loader_is_an_error():
    trn, _ = _split_data()
    with pytest.raises(tr.TrainError) as ei:
        tr.fit_loader(_model(), nn.CrossEntropyLoss(),
                      SGD(_model().parameters(), lr=0.01),
                      tr.DataLoader(trn, batch_size=8), epochs=1,
                      early_stopping=tr.EarlyStopping(patience=1),
                      monitor="val_loss")
    assert "E0643" in str(ei.value)


def test_unknown_monitor_name_is_an_error():
    trn, _ = _split_data()
    with pytest.raises(tr.TrainError) as ei:
        tr.fit_loader(_model(), nn.CrossEntropyLoss(),
                      SGD(_model().parameters(), lr=0.01),
                      tr.DataLoader(trn, batch_size=8), epochs=1,
                      early_stopping=tr.EarlyStopping(patience=1),
                      monitor="nope")
    assert "E0644" in str(ei.value)
