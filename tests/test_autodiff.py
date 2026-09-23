import numpy as np
import pytest

from timet.tensor import Tensor, tensor
from timet.autodiff import no_grad
from tests.numerics import assert_close, finite_difference_grad


def test_master_prompt_example_grad():
    """Literal example from the master prompt section 10."""
    x = tensor([1.0, 2.0, 3.0], grad=True)
    y = (x * x).sum()
    y.backward()
    assert_close(x.grad.data, [2.0, 4.0, 6.0], msg="master-prompt example gradient")


def _check_grad(fn_tensor, fn_numpy, x_np, msg):
    x = Tensor(x_np.copy(), requires_grad=True)
    y = fn_tensor(x)
    y.backward()
    analytic = x.grad.data

    def f_scalar(flat_x):
        return fn_numpy(flat_x)

    numeric = finite_difference_grad(f_scalar, x_np.astype(np.float64))
    assert_close(analytic, numeric, msg=msg)


def test_grad_add():
    x_np = np.random.rand(4).astype(np.float32)

    def fwd(x):
        y = x + 2.0
        return y.sum()

    def fwd_np(x):
        return float(np.sum(x + 2.0))

    _check_grad(fwd, fwd_np, x_np, "add gradient")


def test_grad_mul():
    x_np = np.random.rand(4).astype(np.float32)

    def fwd(x):
        return (x * x * 3.0).sum()

    def fwd_np(x):
        return float(np.sum(x * x * 3.0))

    _check_grad(fwd, fwd_np, x_np, "mul gradient")


def test_grad_div():
    x_np = (np.random.rand(4) + 1.0).astype(np.float32)

    def fwd(x):
        return (10.0 / x).sum()

    def fwd_np(x):
        return float(np.sum(10.0 / x))

    _check_grad(fwd, fwd_np, x_np, "div gradient")


def test_grad_matmul():
    a_np = np.random.rand(3, 4).astype(np.float32)
    w_np = np.random.rand(4, 2).astype(np.float32)
    w = Tensor(w_np)

    def fwd(x):
        return x.matmul(w).sum()

    def fwd_np(x):
        return float(np.sum(x.reshape(3, 4) @ w_np))

    x = Tensor(a_np.copy(), requires_grad=True)
    y = fwd(x)
    y.backward()

    numeric = finite_difference_grad(lambda flat: fwd_np(flat), a_np.astype(np.float64))
    assert_close(x.grad.data, numeric, msg="matmul gradient")


def test_grad_relu():
    # Avoid x == 0 exactly: ReLU is non-differentiable there, so a finite
    # difference straddling the kink is not expected to match either
    # subgradient (0 or 1) that a reverse-mode implementation picks.
    x_np = np.array([-2.0, -0.5, 0.3, 0.5, 2.0], dtype=np.float32)

    def fwd(x):
        return x.relu().sum()

    def fwd_np(x):
        return float(np.sum(np.maximum(x, 0)))

    _check_grad(fwd, fwd_np, x_np, "relu gradient")


def test_grad_sigmoid():
    x_np = np.random.randn(5).astype(np.float32)

    def fwd(x):
        return x.sigmoid().sum()

    def fwd_np(x):
        return float(np.sum(1 / (1 + np.exp(-x))))

    _check_grad(fwd, fwd_np, x_np, "sigmoid gradient")


def test_grad_tanh():
    x_np = np.random.randn(5).astype(np.float32)

    def fwd(x):
        return x.tanh().sum()

    def fwd_np(x):
        return float(np.sum(np.tanh(x)))

    _check_grad(fwd, fwd_np, x_np, "tanh gradient")


def test_grad_exp_log_sqrt():
    x_np = (np.random.rand(5) + 0.5).astype(np.float32)

    def fwd(x):
        return (x.exp().log() + x.sqrt()).sum()

    def fwd_np(x):
        return float(np.sum(np.log(np.exp(x)) + np.sqrt(x)))

    _check_grad(fwd, fwd_np, x_np, "exp/log/sqrt gradient")


def test_grad_mean():
    x_np = np.random.rand(6).astype(np.float32)

    def fwd(x):
        return x.mean()

    def fwd_np(x):
        return float(np.mean(x))

    _check_grad(fwd, fwd_np, x_np, "mean gradient")


def test_grad_softmax_cross_entropy_like():
    x_np = np.random.randn(4).astype(np.float32)

    def fwd(x):
        return (x.softmax() * x.softmax()).sum()

    def fwd_np(x):
        shifted = x - x.max()
        e = np.exp(shifted)
        s = e / e.sum()
        return float(np.sum(s * s))

    _check_grad(fwd, fwd_np, x_np, "softmax gradient")


def test_no_grad_disables_tape():
    x = tensor([1.0, 2.0], grad=True)
    with no_grad():
        y = x * 2.0
    assert y.requires_grad is False
    assert y._node is None


def test_detach_stops_gradient():
    x = tensor([1.0, 2.0, 3.0], grad=True)
    y = x * 2.0
    z = y.detach()
    assert z.requires_grad is False
    w = (z * z).sum()
    with pytest.raises(Exception):
        w.backward()


def test_backward_requires_grad_enabled_tensor():
    x = tensor([1.0, 2.0, 3.0])  # grad not requested
    y = x.sum()
    with pytest.raises(Exception) as exc:
        y.backward()
    assert "does not require grad" in str(exc.value)


def test_gradient_accumulates_across_multiple_uses():
    x = tensor([1.0, 2.0], grad=True)
    y = (x + x).sum()  # dy/dx = 2 for each use, total should be 2 (not 4, not 1)
    y.backward()
    assert_close(x.grad.data, [2.0, 2.0], msg="shared-tensor gradient accumulation")
