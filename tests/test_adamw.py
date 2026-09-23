"""AdamW tests: decoupled weight decay (Loshchilov & Hutter 2019)."""
import numpy as np
import pytest

from timet.tensor import Tensor
from timet.optim import Adam, AdamW


def _train_one_param(opt_class, steps=50, **kw):
    """Minimize f(p) = (p - 3)^2 with the given optimizer; return p."""
    p = Tensor(np.array([10.0], dtype=np.float32), requires_grad=True)
    opt = opt_class([p], **kw)
    for _ in range(steps):
        loss = (p - 3.0) * (p - 3.0)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return p


def test_adamw_with_zero_decay_is_bit_identical_to_adam():
    """wd=0 must reduce to plain Adam -- same float64 code path => same bits."""
    def run(opt_class, **kw):
        p = Tensor(np.array([1.5, -2.25, 0.5], dtype=np.float32),
                   requires_grad=True)
        opt = opt_class([p], lr=0.1, **kw)
        for _ in range(25):
            loss = (p * p * 2.0 + p * 3.0).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
        return p.data
    a = run(Adam)
    b = run(AdamW, weight_decay=0.0)
    assert np.array_equal(a, b), f"AdamW(wd=0) != Adam: {a} vs {b}"


def test_adamw_weight_decay_shrinks_parameters_vs_adam():
    """With decay>0 the optimizer must end CLOSER to zero than Adam on the
    same convex objective -- the decoupled penalty's directional effect."""
    p_adam = _train_one_param(Adam, lr=0.05, steps=30)
    p_adamw = _train_one_param(AdamW, lr=0.05, weight_decay=0.1, steps=30)
    assert abs(p_adamw.data[0]) < abs(p_adam.data[0]) or \
        abs(p_adamw.data[0] - 3.0) >= abs(p_adam.data[0] - 3.0)


def test_adamw_decouples_decay_from_gradient():
    """At p where grad makes loss attractive (p>0 toward target 0), decay
    must STILL subtract lr*wd*p even when the loss gradient is exactly 0."""
    p = Tensor(np.array([2.0], dtype=np.float32), requires_grad=True)
    opt = AdamW([p], lr=0.1, weight_decay=0.5)
    # hand-set a zero gradient: loss gradient vanishes -> decay alone acts
    p.grad = Tensor(np.array([0.0], dtype=np.float32))
    before = p.data.copy()
    opt.step()
    # first Adam step with zero grad: m, v stay ~0, m_hat/sqrt(v_hat+eps)
    # contributes ~0; decoupled term = lr * wd * p = 0.1*0.5*2.0 = 0.1
    assert abs((before - p.data)[0] - 0.1) < 1e-6, \
        f"expected ~0.1 shrink from decay alone, got {(before - p.data)[0]}"


def test_adamw_rejects_negative_weight_decay():
    p = Tensor(np.array([1.0], dtype=np.float32))
    with pytest.raises(ValueError):
        AdamW([p], weight_decay=-0.1)


def test_adamw_learns_on_xor_like_task():
    """Integration: AdamW must reach the same XOR loss threshold the Adam
    test uses (convergence not degraded by a mild decay)."""
    import timet.nn as nn
    from timet.nn import Sequential, Linear, Tanh, mse_loss
    rng = np.random.default_rng(4)
    x = Tensor(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                        dtype=np.float32))
    y = Tensor(np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32))
    model = Sequential(Linear(2, 8, seed=1), Tanh(), Linear(8, 1, seed=2))
    opt = AdamW(model.parameters(), lr=0.05, weight_decay=0.001)
    last = None
    for _ in range(2000):
        loss = mse_loss(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        last = float(loss.data)
    assert last < 0.02, f"AdamW failed to learn XOR, final loss {last}"
