"""Conv2D tests (Milestone 7 remainder).

The load-bearing check is the gradient: hand-written col2im backward code
is exactly where conv implementations rot, so EVERY backward direction
(x, weight, bias) is finite-difference checked in multiple stride/padding
configurations -- per master prompt §29 ("gradient checks", not vibes).
"""
import numpy as np
import pytest

from timet.tensor import Tensor
from timet.nn import Conv2D, conv2d, NNError
from tests.numerics import assert_close, finite_difference_grad


def naive_conv2d(x, w, b, stride=1, padding=0):
    """Dead-simple reference implementation (independent of im2col code):
    6 nested loops computing cross-correlation directly."""
    N, C, H, W = x.shape
    F, Cw, KH, KW = w.shape
    ph = pw = padding
    H_out = (H + 2 * ph - KH) // stride + 1
    W_out = (W + 2 * pw - KW) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out = np.zeros((N, F, H_out, W_out))
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    acc = 0.0
                    for c in range(C):
                        for ky in range(KH):
                            for kx in range(KW):
                                acc += xp[n, c, i * stride + ky, j * stride + kx] * w[f, c, ky, kx]
                    out[n, f, i, j] = acc + (b[f] if b is not None else 0.0)
    return out


def make(idx, seed=42):
    rng = np.random.default_rng(seed + idx)
    x = Tensor(rng.normal(size=(2, 3, 7, 8)).astype(np.float32), requires_grad=True)
    w = Tensor(rng.normal(size=(5, 3, 3, 4)).astype(np.float32), requires_grad=True)
    b = Tensor(rng.normal(size=(5,)).astype(np.float32), requires_grad=True)
    return x, w, b


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 1), (3, 2)])
def test_forward_matches_naive_reference(stride, padding):
    x, w, b = make(0)
    got = conv2d(x, w, b, stride=stride, padding=padding)
    want = naive_conv2d(x.data, w.data, b.data, stride, padding)
    assert_close(got.data.astype(np.float64), want,
                 rtol=1e-5, atol=1e-5, msg=f"forward s={stride} p={padding}")


def test_forward_without_bias():
    x, w, _ = make(1)
    got = conv2d(x, w, None, stride=1, padding=0)
    want = naive_conv2d(x.data, w.data, None)
    assert_close(got.data.astype(np.float64), want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), (3, 2)])
def test_backward_matches_finite_differences_all_inputs(stride, padding):
    # small tensors to keep fd checks fast
    x, w, b = make(2)
    eps = 1e-4

    from timet.nn import _conv2d_forward

    def scalar_from(xd=None, wd=None, bd=None):
        # IMPORTANT: run the reference forward in float64. Going through
        # Tensor here would cast to float32, whose quantization noise in
        # the summed scalar (~O(1e-2) at these magnitudes) completely
        # swamps the h=1e-4 finite difference. This bit us in the first
        # version of this test (dx err 6.8 -- not a backward bug at all).
        out, _, _ = _conv2d_forward(
            xd if xd is not None else x.data.astype(np.float64),
            wd if wd is not None else w.data.astype(np.float64),
            bd if bd is not None else b.data.astype(np.float64),
            stride, stride, padding, padding)
        return float((out * np.arange(out.size).reshape(out.shape)).sum())

    upstream = None
    def ad_grads():
        xt = Tensor(x.data.copy(), requires_grad=True)
        wt = Tensor(w.data.copy(), requires_grad=True)
        bt = Tensor(b.data.copy(), requires_grad=True)
        out = conv2d(xt, wt, bt, stride=stride, padding=padding)
        upstream = np.arange(out.data.size, dtype=np.float64).reshape(out.data.shape)
        out.backward(Tensor(upstream))
        return xt.grad.data, wt.grad.data, bt.grad.data

    gx_ad, gw_ad, gb_ad = ad_grads()
    gx_fd = finite_difference_grad(lambda a: scalar_from(xd=a), x.data.astype(np.float64))
    gw_fd = finite_difference_grad(lambda a: scalar_from(wd=a), w.data.astype(np.float64))
    gb_fd = finite_difference_grad(lambda a: scalar_from(bd=a), b.data.astype(np.float64))
    assert_close(np.asarray(gx_ad, dtype=np.float64), gx_fd, rtol=1e-3, atol=1e-3,
                 msg=f"dx s={stride} p={padding}")
    assert_close(np.asarray(gw_ad, dtype=np.float64), gw_fd, rtol=1e-3, atol=1e-3,
                 msg=f"dw s={stride} p={padding}")
    assert_close(np.asarray(gb_ad, dtype=np.float64), gb_fd, rtol=1e-3, atol=1e-3,
                 msg=f"db s={stride} p={padding}")


def test_backward_without_bias_flows_to_weight_and_input():
    x, w, _ = make(3)
    out = conv2d(x, w).sum()
    out.backward()
    assert x.grad.data.shape == x.shape
    assert w.grad.data.shape == w.shape
    assert np.any(w.grad.data != 0)


def test_conv2d_layer_shapes_and_parameters():
    layer = Conv2D(3, 4, kernel_size=3, stride=2, padding=1, seed=7)
    x = Tensor(np.ones((2, 3, 8, 8), dtype=np.float32))
    y = layer(x)
    assert y.shape == (2, 4, 4, 4)  # (8+2-3)//2+1 = 4
    params = layer.parameters()
    assert len(params) == 2 and params[0].shape == (4, 3, 3, 3)
    assert params[1].shape == (4,)
    assert np.allclose(params[1].data, 0), "bias must be zero-initialized"


def test_conv2d_is_deterministic_given_seed():
    a = Conv2D(1, 2, 3, seed=11)
    b = Conv2D(1, 2, 3, seed=11)
    assert np.array_equal(a.weight.data, b.weight.data)


def test_shape_errors_are_clear():
    tiny = Tensor(np.ones((1, 1, 2, 2), dtype=np.float32))
    w5 = Tensor(np.ones((1, 1, 5, 5), dtype=np.float32))
    with pytest.raises(NNError) as e:
        conv2d(tiny, w5)
    assert "exceeds input" in str(e.value)
    ch_bad = Tensor(np.ones((1, 2, 8, 8), dtype=np.float32))
    with pytest.raises(NNError) as e2:
        conv2d(ch_bad, w5)
    assert "channel mismatch" in str(e2.value)


def test_conv_layer_learns_a_center_detector():
    """End-to-end (§29): one conv filter must learn 'is the 3x3 center
    bright?' on a tiny fixed dataset -- full-batch SGD, deterministic."""
    rng = np.random.default_rng(0)
    N = 8
    xs, ys = [], []
    for i in range(N):
        img = rng.uniform(0.0, 0.2, size=(1, 7, 7))
        label = i % 2
        if label == 1:
            img[0, 2:5, 2:5] += 1.0
        xs.append(img)
        ys.append(label)
    x = Tensor(np.stack(xs).astype(np.float32))
    y = np.array(ys)

    conv = Conv2D(1, 1, kernel_size=3, padding=0, seed=5)
    head_w = Tensor(rng.uniform(-0.1, 0.1, size=(2, 25)).astype(np.float32),
                    requires_grad=True)
    head_b = Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)
    params = conv.parameters() + [head_w, head_b]
    from timet.optim import Adam
    from timet.nn import cross_entropy_loss
    opt = Adam(params, lr=0.05)
    losses = []
    for step in range(300):
        feats = conv(x)                     # (N, 1, 5, 5)
        flat = feats.reshape((N, 25))
        logits = flat.matmul(head_w.transpose()) + head_b
        loss = cross_entropy_loss(logits, Tensor(np.array(ys)))
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.data))
    assert losses[-1] < 0.05, f"conv net did not learn; last losses: {losses[-3:]}"
    preds = logits.data.argmax(axis=1)
    assert (preds == y).all(), f"train accuracy not 100%: {preds} vs {y}"
