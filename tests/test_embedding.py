"""Embedding layer tests (Milestone 7 remainder).

Gradient of a gather op is a scatter-add; duplicate indices must
ACCUMULATE (a classic drop-on-the-floor bug), which is pinned explicitly.
"""
import numpy as np
import pytest

from timet.tensor import Tensor
from timet.nn import Embedding, NNError
from tests.numerics import assert_close, finite_difference_grad


def test_forward_gathers_rows_with_shape():
    emb = Embedding(6, 3, seed=1)
    out = emb([0, 3, 3, 5])
    assert out.shape == (4, 3)
    assert_close(out.data.astype(np.float64),
                 emb.weight.data[np.array([0, 3, 3, 5])].astype(np.float64))


def test_forward_preserves_index_shape():
    emb = Embedding(10, 2, seed=2)
    out = emb(np.array([[1, 2], [3, 4]]))
    assert out.shape == (2, 2, 2)
    assert_close(out.data[0, 1].astype(np.float64),
                 emb.weight.data[2].astype(np.float64))


def test_index_out_of_range_is_a_clear_error():
    emb = Embedding(4, 2, seed=3)
    with pytest.raises(NNError) as e:
        emb([0, 9])
    assert "out of range" in str(e.value)


def test_weight_grad_matches_finite_differences():
    emb = Embedding(5, 3, seed=4)
    w_np = np.array([[1.3, 0.2, -0.7],
                     [0.0, 0.5,  0.9],
                     [-1.1, 0.3, 0.4],
                     [0.8, -0.6, 0.1],
                     [0.4, 0.9, -1.0]], dtype=np.float64)
    emb.weight = Tensor(w_np.astype(np.float32), requires_grad=True)
    idx = np.array([0, 3, 3, 1, 4])

    w_repr = emb.weight
    w_repr.grad = None
    out = emb(idx)
    upstream = np.array([[1.0, -0.5, 0.25],
                         [0.3, 0.7, -1.2],
                         [-0.4, 0.9, 0.6],
                         [1.1, 0.0, -0.8],
                         [0.2, -0.3, 0.5]])
    out.backward(Tensor(upstream))
    ad = np.asarray(w_repr.grad.data, dtype=np.float64)

    def scalar(w64):
        return (w64[idx] * upstream).sum()
    fd = finite_difference_grad(scalar, w_np)
    assert_close(ad, fd, rtol=1e-3, atol=1e-3, msg="embedding weight grad")


def test_duplicate_indices_accumulate_gradients():
    emb = Embedding(4, 2, seed=6)
    emb.weight = Tensor(np.arange(8, dtype=np.float32).reshape(4, 2),
                        requires_grad=True)
    out = emb([1, 1, 2, 1])
    out.backward(Tensor(np.ones((4, 2), dtype=np.float32)))
    gw = np.asarray(emb.weight.grad.data, dtype=np.float64)
    # row 1 addressed 3 times -> upstream accumulates to 3; row 2 once -> 1
    assert_close(gw, np.array([[0, 0], [3, 3], [1, 1], [0, 0]],
                              dtype=np.float64))


def test_embedding_is_deterministic_given_seed():
    a = Embedding(8, 4, seed=9)
    b = Embedding(8, 4, seed=9)
    assert np.array_equal(a.weight.data, b.weight.data)


def test_embedding_trains_with_linear_head():
    """Toy vocab-classification task must actually learn (loss + accuracy),
    not merely run backward without crashing."""
    rng = np.random.default_rng(13)
    # 3 "words"; class = word id. Embedding rows + linear head must learn it.
    words = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1, 0])
    labels = words.copy()
    emb = Embedding(3, 6, seed=21)
    head_w = Tensor(rng.uniform(-0.1, 0.1, size=(3, 6)).astype(np.float32),
                    requires_grad=True)
    head_b = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)
    from timet.optim import Adam
    from timet.nn import cross_entropy_loss
    params = emb.parameters() + [head_w, head_b]
    opt = Adam(params, lr=0.1)
    last = None
    for step in range(400):
        e = emb(words)                              # (10, 6)
        logits = e.matmul(head_w.transpose()) + head_b
        loss = cross_entropy_loss(logits, Tensor(labels))
        loss.backward()
        opt.step()
        opt.zero_grad()
        last = float(loss.data)
    assert last < 0.01, f"embedding task did not learn: final loss {last}"
    preds = logits.data.argmax(axis=1)
    assert (preds == labels).all()
