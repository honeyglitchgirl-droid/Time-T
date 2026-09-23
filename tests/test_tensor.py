import numpy as np
import pytest

from timet.tensor import Tensor, tensor, zeros, ones, TensorError
from tests.numerics import assert_close


def test_construction_and_dtype_default_f32():
    t = tensor([1.0, 2.0, 3.0])
    assert t.dtype == "f32"
    assert t.shape == (3,)


def test_zeros_ones():
    z = zeros([2, 3])
    o = ones([3])
    assert z.shape == (2, 3)
    assert np.all(z.data == 0)
    assert np.all(o.data == 1)


def test_add_matches_numpy():
    a_np = np.random.rand(5).astype(np.float32)
    b_np = np.random.rand(5).astype(np.float32)
    a, b = Tensor(a_np), Tensor(b_np)
    assert_close((a + b).data, a_np + b_np)


def test_mul_matches_numpy():
    a_np = np.random.rand(4, 3).astype(np.float32)
    b_np = np.random.rand(4, 3).astype(np.float32)
    a, b = Tensor(a_np), Tensor(b_np)
    assert_close((a * b).data, a_np * b_np)


def test_matmul_matches_numpy():
    a_np = np.random.rand(4, 3).astype(np.float32)
    b_np = np.random.rand(3, 5).astype(np.float32)
    a, b = Tensor(a_np), Tensor(b_np)
    assert_close(a.matmul(b).data, a_np @ b_np)


def test_matmul_shape_error_message():
    a = Tensor(np.zeros((32, 128), dtype=np.float32))
    b = Tensor(np.zeros((64, 128), dtype=np.float32))
    with pytest.raises(TensorError) as exc:
        a.matmul(b)
    err = exc.value
    assert err.code == "E0403"
    assert "32, 128" in err.actual
    assert "64, 128" in err.actual
    assert err.expected == "left[-1] == right[-2]"


def test_broadcasting_add():
    a = Tensor(np.ones((3, 1), dtype=np.float32))
    b = Tensor(np.ones((1, 4), dtype=np.float32))
    result = (a + b).data
    assert result.shape == (3, 4)
    assert np.all(result == 2)


def test_reshape_matches_numpy():
    a_np = np.arange(6).astype(np.float32)
    a = Tensor(a_np)
    assert_close(a.reshape([2, 3]).data, a_np.reshape(2, 3))


def test_reshape_error_message():
    a = Tensor(np.zeros(6, dtype=np.float32))
    with pytest.raises(TensorError) as exc:
        a.reshape([4, 4])
    assert exc.value.code == "E0404"


def test_transpose_matches_numpy():
    a_np = np.random.rand(2, 3).astype(np.float32)
    a = Tensor(a_np)
    assert_close(a.transpose().data, a_np.T)


def test_sum_mean_max_min_match_numpy():
    a_np = np.random.rand(4, 5).astype(np.float32)
    a = Tensor(a_np)
    assert_close(a.sum().data, a_np.sum())
    assert_close(a.mean().data, a_np.mean())
    assert_close(a.max().data, a_np.max())
    assert_close(a.min().data, a_np.min())


def test_sum_axis_matches_numpy():
    a_np = np.random.rand(4, 5).astype(np.float32)
    a = Tensor(a_np)
    assert_close(a.sum(axis=0).data, a_np.sum(axis=0))
    assert_close(a.sum(axis=1).data, a_np.sum(axis=1))


def test_exp_log_sqrt_match_numpy():
    a_np = np.random.rand(5).astype(np.float32) + 0.1
    a = Tensor(a_np)
    assert_close(a.exp().data, np.exp(a_np))
    assert_close(a.log().data, np.log(a_np))
    assert_close(a.sqrt().data, np.sqrt(a_np))


def test_relu_sigmoid_tanh_match_numpy():
    a_np = np.random.randn(5).astype(np.float32)
    a = Tensor(a_np)
    assert_close(a.relu().data, np.maximum(a_np, 0))
    assert_close(a.sigmoid().data, 1 / (1 + np.exp(-a_np)))
    assert_close(a.tanh().data, np.tanh(a_np))


def test_softmax_sums_to_one_and_matches_numpy():
    a_np = np.random.randn(3, 4).astype(np.float32)
    a = Tensor(a_np)
    result = a.softmax(axis=-1).data
    shifted = a_np - a_np.max(axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    assert_close(result, expected)
    assert_close(result.sum(axis=-1), np.ones(3))


def test_indexing():
    a = Tensor(np.arange(6).astype(np.float32).reshape(2, 3))
    assert_close(a[0].data, np.array([0.0, 1.0, 2.0]))


def test_negative_and_sub():
    a_np = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    a = Tensor(a_np)
    assert_close((-a).data, -a_np)


def test_repr_contains_values():
    t = tensor([1.0, 2.0])
    assert "1.0" in repr(t)


# ---- new in v0.2: argmax / clip / log_softmax / one_hot ----

def test_argmax_axis_none_and_axis():
    a = Tensor(np.array([[1.0, 5.0, 2.0], [4.0, 0.5, 3.0]], dtype=np.float32))
    assert int(a.argmax().data) == 1
    cols = a.argmax(axis=1)
    assert cols.dtype == "i64"
    assert list(cols.data) == [1, 0]


def test_clip_forward_values():
    a_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    a = Tensor(a_np)
    assert_close(a.clip(-1.0, 1.0).data, np.clip(a_np, -1.0, 1.0))


def test_log_softmax_matches_manual_logsumexp():
    a_np = np.random.randn(4, 5).astype(np.float32)
    a = Tensor(a_np)
    result = a.log_softmax(axis=-1).data
    m = a_np.max(axis=-1, keepdims=True)
    expected = (a_np - m) - np.log(np.exp(a_np - m).sum(axis=-1, keepdims=True))
    assert_close(result, expected)
    # exp(log_softmax) rows are valid probability distributions
    assert_close(np.exp(result).sum(axis=-1), np.ones(4))


def test_log_softmax_numerically_stable_with_large_logits():
    a = Tensor(np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32))
    out = a.log_softmax(axis=-1).data
    assert np.isfinite(out).all()


def test_one_hot_basic():
    from timet.tensor import one_hot
    oh = one_hot([0, 2, 1], 3)
    assert oh.shape == (3, 3)
    assert_close(oh.data, np.eye(3)[[0, 2, 1]])


def test_one_hot_exact_class_boundaries():
    from timet.tensor import one_hot
    oh = one_hot([0, 4], 5)
    assert_close(oh.data, np.eye(5)[[0, 4]])


def test_one_hot_rejects_out_of_range_and_nonpositive_classes():
    import pytest
    from timet.tensor import one_hot
    from timet.tensor import TensorError
    with pytest.raises(TensorError):
        one_hot([3], 3)
    with pytest.raises(TensorError):
        one_hot([-1], 3)
    with pytest.raises(TensorError):
        one_hot([0], 0)
