import numpy as np

from timet.tensor import Tensor
from timet.nn import Linear, ReLU, Sequential, MSELoss
from timet.optim import SGD
from tests.numerics import assert_close


def test_linear_forward_shape():
    layer = Linear(4, 3, seed=1)
    x = Tensor(np.random.rand(2, 4).astype(np.float32))
    y = layer(x)
    assert y.shape == (2, 3)


def test_linear_parameters_require_grad():
    layer = Linear(4, 3, seed=1)
    for p in layer.parameters():
        assert p.requires_grad is True


def test_sequential_forward():
    model = Sequential(Linear(2, 4, seed=0), ReLU(), Linear(4, 1, seed=1))
    x = Tensor(np.random.rand(5, 2).astype(np.float32))
    y = model(x)
    assert y.shape == (5, 1)


def test_mse_loss_matches_manual_computation():
    pred = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    target = Tensor(np.array([1.5, 2.5, 2.5], dtype=np.float32))
    loss = MSELoss()(pred, target)
    expected = np.mean((np.array([1.0, 2.0, 3.0]) - np.array([1.5, 2.5, 2.5])) ** 2)
    assert_close(loss.data, expected)


def test_linear_regression_training_reduces_loss():
    """y = 2x + 1 -- a tiny linear regression that must actually converge."""
    rng = np.random.default_rng(0)
    x_np = rng.uniform(-5, 5, size=(64, 1)).astype(np.float32)
    y_np = (2 * x_np + 1).astype(np.float32)

    x = Tensor(x_np)
    y = Tensor(y_np)

    model = Linear(1, 1, seed=42)
    loss_fn = MSELoss()
    optim = SGD(model.parameters(), lr=0.05)

    losses = []
    for _ in range(200):
        pred = model(x)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss.data))

    assert losses[-1] < losses[0], "loss should decrease over training"
    assert losses[-1] < 0.5, f"final loss too high: {losses[-1]}"


def test_xor_mlp_training_reduces_loss():
    x = Tensor(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    y = Tensor(np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32))

    model = Sequential(Linear(2, 8, seed=7), ReLU(), Linear(8, 1, seed=13))
    loss_fn = MSELoss()
    optim = SGD(model.parameters(), lr=0.5)

    losses = []
    for _ in range(500):
        pred = model(x)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss.data))

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.1, f"XOR MLP failed to converge, final loss={losses[-1]}"


# ---- v0.2 additions: layers, losses, Adam ----

def test_tanh_softmax_flatten_modules():
    from timet.nn import Tanh, Softmax, Flatten
    x = Tensor(np.random.randn(2, 3, 4))
    assert Tanh()(x).shape == x.shape
    s = Softmax(axis=-1)(x)
    assert_close(s.data.sum(axis=-1), np.ones((2, 3)))
    f = Flatten()(x)
    assert f.shape == (2, 12)


def test_dropout_eval_is_identity_train_zeroes_some():
    from timet.nn import Dropout
    x = Tensor(np.ones((100,), dtype=np.float32))
    d = Dropout(p=0.5, seed=0)
    d.eval()
    assert_close(d(x).data, np.ones(100))
    d.train()
    out = d(x).data
    fraction_zero = float((out == 0).mean())
    assert 0.3 < fraction_zero < 0.7, f"expected ~50% dropped, got {fraction_zero}"
    # survivors are scaled by 1/(1-p) so expectation is preserved
    kept = out[out != 0]
    assert_close(kept, np.full(kept.shape, 2.0))


def test_dropout_deterministic_given_seed():
    from timet.nn import Dropout
    x = Tensor(np.ones((50,), dtype=np.float32))
    a = Dropout(p=0.5, seed=7)(x).data
    b = Dropout(p=0.5, seed=7)(x).data
    assert_close(a, b)


def test_sequential_train_eval_propagates():
    from timet.nn import Dropout
    model = Sequential(Linear(2, 2, seed=0), Dropout(0.5, seed=0), ReLU())
    model.eval()
    assert all(layer.training is False for layer in model.layers)
    model.train()
    assert all(layer.training is True for layer in model.layers)


def test_cross_entropy_matches_manual_numpy():
    from timet.nn import CrossEntropyLoss
    logits_np = np.array([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]], dtype=np.float32)
    targets = [0, 2]
    loss = CrossEntropyLoss()(Tensor(logits_np), targets)
    m = logits_np.max(axis=1, keepdims=True)
    logp = (logits_np - m) - np.log(np.exp(logits_np - m).sum(axis=1, keepdims=True))
    expected = -logp[np.arange(2), targets].mean()
    assert_close(loss.data, expected, msg="cross-entropy value")


def test_cross_entropy_logit_gradient_finite_difference():
    from timet.nn import CrossEntropyLoss
    from tests.numerics import finite_difference_grad
    logits_np = np.array([[0.4, -1.0, 2.0], [1.1, 0.3, -0.2]], dtype=np.float32)
    targets = [2, 0]

    def f(flat):
        m = flat.max(axis=1, keepdims=True)
        logp = (flat - m) - np.log(np.exp(flat - m).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(2), targets].mean())

    logits = Tensor(logits_np, requires_grad=True)
    loss = CrossEntropyLoss()(logits, targets)
    loss.backward()
    numeric = finite_difference_grad(f, logits_np.astype(np.float64))
    assert_close(logits.grad.data, numeric, msg="cross-entropy logit gradient")


def test_cross_entropy_rejects_bad_shapes():
    import pytest
    from timet.nn import CrossEntropyLoss, NNError
    ce = CrossEntropyLoss()
    with pytest.raises(NNError):
        ce(Tensor(np.zeros((3,), dtype=np.float32)), [0])
    with pytest.raises(NNError):
        ce(Tensor(np.zeros((2, 3), dtype=np.float32)), [0, 1, 2])


def test_bce_loss_value_and_gradient():
    from timet.nn import BCELoss
    from tests.numerics import finite_difference_grad
    pred_np = np.array([0.2, 0.6, 0.9], dtype=np.float32)
    target_np = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    loss = BCELoss()(Tensor(pred_np), Tensor(target_np))
    expected = -(target_np * np.log(pred_np) + (1 - target_np) * np.log(1 - pred_np)).mean()
    assert_close(loss.data, expected)

    def f(flat):
        return float(-(target_np * np.log(flat) + (1 - target_np) * np.log(1 - flat)).mean())

    pred = Tensor(pred_np, requires_grad=True)
    BCELoss()(pred, Tensor(target_np)).backward()
    numeric = finite_difference_grad(f, pred_np.astype(np.float64))
    assert_close(pred.grad.data, numeric, msg="BCE gradient")


def test_adam_converges_on_xor_faster_than_sgd_budget():
    from timet.optim import Adam
    from timet.nn import MSELoss
    rng = np.random.default_rng(3)
    x_np = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_np = np.array([[0], [1], [1], [0]], dtype=np.float32)
    x, y = Tensor(x_np), Tensor(y_np)
    model = Sequential(Linear(2, 8, seed=5), Tanh_(), Linear(8, 1, seed=6))
    opt = Adam(model.parameters(), lr=0.05)
    loss_fn = MSELoss()
    losses = []
    for _ in range(200):
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"
    assert losses[-1] < 0.05, f"Adam failed to fit XOR in 200 steps, final loss {losses[-1]}"


def Tanh_():
    from timet.nn import Tanh
    return Tanh()


def test_adam_step_skips_params_without_grad():
    from timet.optim import Adam
    p1 = Tensor([1.0, 2.0], requires_grad=True)
    p2 = Tensor([3.0], requires_grad=True)
    (p1 * p1).sum().backward()
    before = p2.data.copy()
    opt = Adam([p1, p2], lr=0.1)
    opt.step()  # must not crash on p2.grad is None
    assert_close(p2.data, before)
