import numpy as np
import pytest

from timet.nn import Linear, Tanh, Sequential, MSELoss, CrossEntropyLoss
from timet.optim import SGD, Adam
from timet.tensor import Tensor
from timet.train import accuracy, fit, EarlyStopping, History, TrainError
from tests.numerics import assert_close


def test_accuracy_perfect_and_half():
    # rows: argmax is [0, 1]; targets [0, 0] -> 1/2 correct
    logits = Tensor(np.array([[3.0, 1.0], [0.5, 2.5]], dtype=np.float32))
    assert accuracy(logits, [0, 1]) == 1.0
    assert accuracy(logits, [0, 0]) == 0.5


def test_accuracy_rejects_bad_shapes():
    with pytest.raises(TrainError):
        accuracy(Tensor(np.zeros((4,), dtype=np.float32)), [0])
    with pytest.raises(TrainError):
        accuracy(Tensor(np.zeros((2, 3), dtype=np.float32)), [0])


def test_fit_trains_classifier_to_perfect_accuracy():
    """A real end-to-end classifier run: 2D points, class = (x + y > 0),
    MLP + Adam + cross-entropy, FULL batch. Must actually reach 100% train
    accuracy -- not assumed, asserted from the recorded history."""
    rng = np.random.default_rng(0)
    x_np = rng.uniform(-1, 1, size=(64, 2)).astype(np.float32)
    y_np = (x_np[:, 0] + x_np[:, 1] > 0).astype(np.int64)
    x = Tensor(x_np)
    model = Sequential(Linear(2, 8, seed=0), Tanh(), Linear(8, 2, seed=1))
    opt = Adam(model.parameters(), lr=0.05)
    history = fit(model, CrossEntropyLoss(), opt, x, y_np, epochs=60,
                  metrics={"acc": lambda m, xx, yy: accuracy(m(xx), yy)})
    assert history.losses[-1] < history.losses[0]
    assert history.metrics["acc"][-1] == 1.0, \
        f"classifier did not reach 100% train accuracy (last={history.metrics['acc'][-1]})"


def test_fit_rejects_zero_epochs():
    with pytest.raises(TrainError):
        fit(Linear(1, 1, seed=0), MSELoss(), SGD([]), Tensor([1.0]), [1], epochs=0)


def test_fit_logs_at_requested_intervals():
    lines = []
    x = Tensor(np.array([[0.0], [1.0]], dtype=np.float32))
    y = Tensor(np.array([[1.0], [3.0]], dtype=np.float32))
    model = Linear(1, 1, seed=0)
    fit(model, MSELoss(), SGD(model.parameters(), lr=0.05), x, y,
        epochs=10, log_every=5, log=lines.append)
    # epochs 0, 5 (interval hits) and 9 (final epoch) => 3 log lines
    assert len(lines) == 3 and all("loss=" in line for line in lines)


def test_early_stopping_triggers_and_records_best():
    es = EarlyStopping(patience=3, min_delta=0.01)
    assert es.step(1.0) is False     # first value becomes best
    assert es.step(0.99) is False    # improvement 0.01 is NOT > min_delta -> bad=1
    assert es.step(0.985) is False   # real improvement -> best=0.985, bad=0
    assert es.step(0.99) is False    # bad=1
    assert es.step(0.991) is False   # bad=2
    assert es.step(0.992) is True    # bad=3 == patience -> stop
    assert es.best == 0.985


def test_history_final_loss_property():
    h = History()
    assert h.final_loss is None
    h.losses.append(0.5)
    assert h.final_loss == 0.5
