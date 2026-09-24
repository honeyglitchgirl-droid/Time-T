"""Conv1D tests (Milestone 7 expansion).

Verifies 1D convolution / cross-correlation:
- forward matches 4-nested-loop naive reference implementation
- backward gradients w.r.t input, weight, and bias match central finite differences
- error handling for invalid shapes and parameters (code E0616)
- module determinism and parameter properties
- checkpoint serialization
- end-to-end learning test
"""
import numpy as np
import pytest

from timet.tensor import Tensor
from timet.nn import Conv1D, conv1d, NNError
from tests.numerics import assert_close, finite_difference_grad


def naive_conv1d(x, w, b, stride=1, padding=0):
    N, C, L = x.shape
    F, Cw, K = w.shape
    L_out = (L + 2 * padding - K) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    out = np.zeros((N, F, L_out), dtype=x.dtype)
    for n in range(N):
        for f in range(F):
            for l in range(L_out):
                acc = 0.0
                for c in range(C):
                    for k in range(K):
                        acc += xp[n, c, l * stride + k] * w[f, c, k]
                out[n, f, l] = acc + (b[f] if b is not None else 0.0)
    return out


def make_conv1d_inputs(idx=0, seed=42):
    rng = np.random.default_rng(seed + idx)
    x = Tensor(rng.normal(size=(2, 3, 11)).astype(np.float32), requires_grad=True)
    w = Tensor(rng.normal(size=(4, 3, 3)).astype(np.float32), requires_grad=True)
    b = Tensor(rng.normal(size=(4,)).astype(np.float32), requires_grad=True)
    return x, w, b


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1), (3, 2)])
def test_conv1d_forward_matches_naive_reference(stride, padding):
    x, w, b = make_conv1d_inputs(0)
    got = conv1d(x, w, b, stride=stride, padding=padding)
    want = naive_conv1d(x.data, w.data, b.data, stride, padding)
    assert_close(got.data.astype(np.float64), want, rtol=1e-5, atol=1e-5)


def test_conv1d_forward_without_bias():
    x, w, _ = make_conv1d_inputs(1)
    got = conv1d(x, w, None, stride=1, padding=0)
    want = naive_conv1d(x.data, w.data, None, 1, 0)
    assert_close(got.data.astype(np.float64), want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), (3, 2)])
def test_conv1d_backward_matches_finite_differences(stride, padding):
    x, w, b = make_conv1d_inputs(2)

    from timet.nn import _conv1d_forward

    def scalar_from(xd=None, wd=None, bd=None):
        out, _, _ = _conv1d_forward(
            xd if xd is not None else x.data.astype(np.float64),
            wd if wd is not None else w.data.astype(np.float64),
            bd if bd is not None else b.data.astype(np.float64),
            stride, padding)
        weights = np.arange(out.size, dtype=np.float64).reshape(out.shape)
        return float((out * weights).sum())

    xt = Tensor(x.data.copy(), requires_grad=True)
    wt = Tensor(w.data.copy(), requires_grad=True)
    bt = Tensor(b.data.copy(), requires_grad=True)
    out = conv1d(xt, wt, bt, stride=stride, padding=padding)
    upstream = np.arange(out.data.size, dtype=np.float32).reshape(out.data.shape)
    out.backward(Tensor(upstream))

    gx_ad = xt.grad.data.astype(np.float64)
    gw_ad = wt.grad.data.astype(np.float64)
    gb_ad = bt.grad.data.astype(np.float64)

    gx_fd = finite_difference_grad(lambda a: scalar_from(xd=a), x.data.astype(np.float64))
    gw_fd = finite_difference_grad(lambda a: scalar_from(wd=a), w.data.astype(np.float64))
    gb_fd = finite_difference_grad(lambda a: scalar_from(bd=a), b.data.astype(np.float64))

    assert_close(gx_ad, gx_fd, rtol=1e-3, atol=1e-3, msg=f"dx s={stride} p={padding}")
    assert_close(gw_ad, gw_fd, rtol=1e-3, atol=1e-3, msg=f"dw s={stride} p={padding}")
    assert_close(gb_ad, gb_fd, rtol=1e-3, atol=1e-3, msg=f"db s={stride} p={padding}")


def test_conv1d_layer_properties_and_determinism():
    c1 = Conv1D(in_channels=3, out_channels=5, kernel_size=3, stride=2, padding=1, seed=12)
    c2 = Conv1D(in_channels=3, out_channels=5, kernel_size=3, stride=2, padding=1, seed=12)
    assert np.array_equal(c1.weight.data, c2.weight.data)
    assert len(c1.parameters()) == 2
    assert c1.weight.shape == (5, 3, 3)
    assert c1.bias.shape == (5,)

    x = Tensor(np.random.randn(2, 3, 10).astype(np.float32))
    y = c1(x)
    assert y.shape == (2, 5, 5)  # (10 + 2 - 3) // 2 + 1 = 5


def test_conv1d_errors():
    with pytest.raises(NNError) as e1:
        Conv1D(2, 4, kernel_size=0)
    assert "E0616" in str(e1.value)

    c = Conv1D(2, 4, kernel_size=5)
    with pytest.raises(NNError) as e2:
        c(Tensor(np.ones((2, 2, 3), dtype=np.float32)))  # length 3 < kernel 5
    assert "exceeds input" in str(e2.value)


def test_conv1d_checkpoint_roundtrip(tmp_path):
    from timet import checkpoint
    c = Conv1D(2, 4, 3, seed=1)
    p = checkpoint.save_bin(c, tmp_path / "conv1d.ttck")
    c2 = Conv1D(2, 4, 3, seed=99)
    checkpoint.load_into(c2, p)
    for p1, p2 in zip(c.parameters(), c2.parameters()):
        assert np.array_equal(p1.data, p2.data)


def test_conv1d_learns_peak_detector():
    """End-to-end optimization test: learn to detect peaks in 1D sequence."""
    rng = np.random.default_rng(0)
    N = 10
    L = 12
    xs, ys = [], []
    for i in range(N):
        seq = rng.uniform(0.0, 0.2, size=(1, L)).astype(np.float32)
        label = i % 2
        if label == 1:
            seq[0, 5:7] += 1.0
        xs.append(seq)
        ys.append(label)
    x = Tensor(np.stack(xs))
    y = np.array(ys)

    conv = Conv1D(1, 1, kernel_size=3, padding=0, seed=3)
    from timet.nn import Flatten, Linear, cross_entropy_loss
    head_w = Tensor(rng.uniform(-0.1, 0.1, size=(2, 10)).astype(np.float32), requires_grad=True)
    head_b = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
    from timet.optim import Adam

    opt = Adam(conv.parameters() + [head_w, head_b], lr=0.05)
    for step in range(250):
        feat = conv(x) # (N, 1, 10)
        flat = feat.reshape((N, 10))
        logits = flat.matmul(head_w.transpose()) + head_b
        loss = cross_entropy_loss(logits, Tensor(y))
        loss.backward()
        opt.step()
        opt.zero_grad()

    assert float(loss.data) < 0.05
    preds = logits.data.argmax(axis=1)
    assert (preds == y).all()
