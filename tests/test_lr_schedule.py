"""LR schedule tests: exact formula math + e2e effect."""
import numpy as np
import pytest

from timet.tensor import Tensor, tensor
from timet.optim import SGD, Adam
from timet import train as tr


def _opt(lr=0.1):
    p = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    return SGD([p], lr=lr)


def test_step_lr_exact_sequence():
    opt = _opt(0.1)
    s = tr.StepLR(opt, step_size=2, gamma=0.5)
    assert opt.lr == 0.1          # before any step
    expected = [0.1, 0.05, 0.05, 0.025, 0.025]  # epochs 1..5
    for want in expected:
        s.step()
        assert abs(opt.lr - want) < 1e-12, f"epoch {s.last_epoch}: {opt.lr} != {want}"


def test_exponential_lr_exact():
    opt = _opt(0.2)
    s = tr.ExponentialLR(opt, gamma=0.9)
    for e in range(1, 6):
        s.step()
        assert abs(opt.lr - 0.2 * (0.9 ** e)) < 1e-12


def test_cosine_annealing_endpoints_and_midpoint():
    opt = _opt(0.4)
    s = tr.CosineAnnealingLR(opt, t_max=4)
    s.step(); s.step()                      # epoch 2 = midpoint
    assert abs(opt.lr - 0.2) < 1e-7         # cos(pi/2)=0 -> base*0.5
    s.step(); s.step()                      # epoch 4 -> 0
    assert abs(opt.lr - 0.0) < 1e-7
    s.step()                                # beyond t_max clamps at 0
    assert abs(opt.lr) < 1e-7


def test_scheduler_validation():
    with pytest.raises(tr.TrainError):
        tr.StepLR(_opt(), step_size=0)
    with pytest.raises(tr.TrainError):
        tr.CosineAnnealingLR(_opt(), t_max=0)
    with pytest.raises(tr.TrainError):
        tr.StepLR(object(), step_size=1)    # no .lr attribute -> E0639


def test_fit_loader_applies_scheduler_once_per_epoch_not_per_batch():
    """Pin the textbook stepping order: with 5 batches/epoch and 2 epochs
    plus StepLR(step_size=1, gamma=0.5), lr must be base*0.5 in epoch 2
    and base*0.25 after -- NOT base*0.5**10 (per-batch stepping)."""
    import timet.nn as nn
    x = tensor(np.random.default_rng(0).normal(0, 1, (10, 2)).astype(np.float32))
    y = tensor(np.array([0, 1] * 5))
    ds = tr.TensorDataset(x, y)
    ld = tr.DataLoader(ds, batch_size=2)     # 5 batches per epoch
    model = nn.Linear(2, 2, seed=1)
    opt = SGD(model.parameters(), lr=0.1)
    sched = tr.StepLR(opt, step_size=1, gamma=0.5)
    tr.fit_loader(model, nn.CrossEntropyLoss(), opt, ld, epochs=2, scheduler=sched)
    assert abs(opt.lr - 0.025) < 1e-12, f"lr={opt.lr}: scheduler stepped per-batch?"


def test_schedule_rescues_a_diverging_lr_on_quadratic_basin():
    """lr=1.5 on (p-1)^2 DIVERGES with constant lr (factor |1-2*lr|>1);
    the cosine-annealed run with the same base lr must stay bounded and end
    closer to the optimum -- this pins that scheduler.step() actually wires
    lr into the optimizer (and once per loop, not per epoch-batch)."""
    def run(scheduled):
        p = Tensor(np.array([5.0], dtype=np.float32), requires_grad=True)
        opt = SGD([p], lr=1.5)
        sched = tr.CosineAnnealingLR(opt, t_max=20) if scheduled else None
        for _ in range(20):
            loss = (p - 1.0) * (p - 1.0)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if sched:
                sched.step()
        return float(p.data[0])
    a = abs(run(False) - 1.0)
    b = abs(run(True) - 1.0)
    assert a > 1e4, f"constant lr=1.5 should diverge, got {a}"
    assert b < 1.0, f"annealed run should approach 1.0, got {b}"
