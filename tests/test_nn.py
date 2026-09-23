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
