import numpy as np
import pytest

from timet.tensor import Tensor
from timet.nn import LayerNorm, NNError
from tests.numerics import assert_close, finite_difference_grad


def naive_layernorm_1d(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    """Reference implementation of LayerNorm over the last dimension."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = x_hat * gamma + beta
    return out


def test_layernorm_forward_shape_and_numerical_output():
    ln = LayerNorm(normalized_shape=4, eps=1e-5, seed=42)
    # Give gamma and beta custom non-trivial values
    ln.weight.data = np.array([1.5, 0.5, 2.0, -1.0], dtype=np.float32)
    ln.bias.data = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)

    x_np = np.array([[1.0, 2.0, 3.0, 4.0],
                     [-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    x = Tensor(x_np)
    y = ln(x)
    assert y.shape == (2, 4)

    expected = naive_layernorm_1d(x_np, ln.weight.data, ln.bias.data, eps=1e-5)
    assert_close(y.data.astype(np.float64), expected.astype(np.float64), rtol=1e-4, atol=1e-4)


def test_layernorm_parameters():
    ln = LayerNorm(normalized_shape=8)
    params = ln.parameters()
    assert len(params) == 2
    assert params[0].shape == (8,)  # weight
    assert params[1].shape == (8,)  # bias
    assert np.all(params[0].data == 1.0)
    assert np.all(params[1].data == 0.0)


def test_layernorm_multidimensional_shape():
    ln = LayerNorm(normalized_shape=(4, 5))
    x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    y = ln(x)
    assert y.shape == (2, 3, 4, 5)
    assert ln.weight.shape == (4, 5)
    assert ln.bias.shape == (4, 5)


def test_layernorm_zero_mean_unit_var():
    ln = LayerNorm(normalized_shape=100)
    rng = np.random.default_rng(123)
    x = Tensor(rng.normal(loc=10.0, scale=5.0, size=(10, 100)).astype(np.float32))
    y = ln(x)
    # output should have mean ~0 and var ~1 across last dimension
    mean = y.data.mean(axis=-1)
    var = y.data.var(axis=-1)
    assert_close(mean, np.zeros(10), atol=1e-4)
    assert_close(var, np.ones(10), atol=1e-3)


def test_layernorm_gradient_finite_difference_all_inputs():
    # Shape (2, 3) normalized over last dimension 3
    x_np = np.array([[0.5, -1.2, 2.3],
                     [1.1, 0.4, -0.8]], dtype=np.float64)
    w_np = np.array([1.2, 0.8, -0.5], dtype=np.float64)
    b_np = np.array([0.1, -0.3, 0.4], dtype=np.float64)
    eps = 1e-5

    def scalar_loss(xd, wd, bd):
        mean = xd.mean(axis=-1, keepdims=True)
        var = ((xd - mean) ** 2).mean(axis=-1, keepdims=True)
        x_hat = (xd - mean) / np.sqrt(var + eps)
        out = x_hat * wd + bd
        # arbitrary linear upstream weight
        weights = np.array([[1.0, -0.5, 2.0], [0.3, 1.2, -1.0]])
        return float((out * weights).sum())

    # Analytical grad via autograd
    xt = Tensor(x_np.astype(np.float32), requires_grad=True)
    wt = Tensor(w_np.astype(np.float32), requires_grad=True)
    bt = Tensor(b_np.astype(np.float32), requires_grad=True)
    ln = LayerNorm(3, eps=eps)
    ln.weight = wt
    ln.bias = bt
    out = ln(xt)
    upstream = np.array([[1.0, -0.5, 2.0], [0.3, 1.2, -1.0]], dtype=np.float32)
    out.backward(Tensor(upstream))

    gx_ad = xt.grad.data.astype(np.float64)
    gw_ad = wt.grad.data.astype(np.float64)
    gb_ad = bt.grad.data.astype(np.float64)

    gx_fd = finite_difference_grad(lambda x_: scalar_loss(x_, w_np, b_np), x_np)
    gw_fd = finite_difference_grad(lambda w_: scalar_loss(x_np, w_, b_np), w_np)
    gb_fd = finite_difference_grad(lambda b_: scalar_loss(x_np, w_np, b_), b_np)

    assert_close(gx_ad, gx_fd, rtol=1e-3, atol=1e-3, msg="LayerNorm input gradient")
    assert_close(gw_ad, gw_fd, rtol=1e-3, atol=1e-3, msg="LayerNorm weight gradient")
    assert_close(gb_ad, gb_fd, rtol=1e-3, atol=1e-3, msg="LayerNorm bias gradient")


def test_layernorm_without_elementwise_affine():
    ln = LayerNorm(normalized_shape=4, elementwise_affine=False)
    assert len(ln.parameters()) == 0
    assert ln.weight is None
    assert ln.bias is None
    x_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    y = ln(Tensor(x_np))
    expected = (x_np - x_np.mean(axis=-1, keepdims=True)) / np.sqrt(x_np.var(axis=-1, keepdims=True) + 1e-5)
    assert_close(y.data, expected, rtol=1e-4, atol=1e-4)


def test_layernorm_shape_mismatch_raises_error():
    ln = LayerNorm(normalized_shape=4)
    with pytest.raises(NNError) as e:
        ln(Tensor(np.ones((2, 5), dtype=np.float32)))
    assert "E0614" in str(e.value)
    assert "normalized_shape" in str(e.value)


def test_layernorm_learns_affine_parameters():
    """Toy optimization: LayerNorm must adapt weight and bias to match target shift/scale."""
    from timet.optim import Adam
    from timet.nn import MSELoss

    ln = LayerNorm(4, seed=1)
    target_scale = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    target_bias = np.array([1.0, 0.5, -0.5, -2.0], dtype=np.float32)

    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(20, 4)).astype(np.float32)
    # Normalized x
    x_norm = (x_data - x_data.mean(axis=-1, keepdims=True)) / np.sqrt(x_data.var(axis=-1, keepdims=True) + 1e-5)
    y_target = Tensor(x_norm * target_scale + target_bias)
    x = Tensor(x_data)

    opt = Adam(ln.parameters(), lr=0.1)
    loss_fn = MSELoss()

    for _ in range(200):
        pred = ln(x)
        loss = loss_fn(pred, y_target)
        loss.backward()
        opt.step()
        opt.zero_grad()

    assert float(loss.data) < 0.01, f"LayerNorm failed to learn scale/shift: final loss={float(loss.data)}"
    assert_close(ln.weight.data, target_scale, atol=0.1)
    assert_close(ln.bias.data, target_bias, atol=0.1)


def test_layernorm_checkpoint_roundtrip(tmp_path):
    from timet.nn import LayerNorm, Sequential, Linear
    from timet import checkpoint
    m = Sequential(Linear(2, 4, seed=0), LayerNorm(4), Linear(4, 1, seed=1))
    p = checkpoint.save(m, tmp_path / "ln.json")
    loaded = checkpoint.load(p)
    assert "1.weight" in loaded and "1.bias" in loaded
    m2 = Sequential(Linear(2, 4, seed=9), LayerNorm(4), Linear(4, 1, seed=8))
    checkpoint.load_into(m2, p)
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert np.array_equal(p1.data, p2.data)

