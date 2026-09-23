"""Training utilities (master prompt section 23, Milestone 8).

What exists (all tested in tests/test_train.py):
  - accuracy()   classification accuracy from logits + integer targets
  - History      recorded per-epoch losses
  - fit()        a full-batch training loop: forward -> backward -> step
  - EarlyStopping

What does NOT exist yet (docs/ROADMAP.md Milestone 8): datasets/dataloaders
(mini-batches beyond full-batch), LR schedules, validation-split plumbing,
mixed precision, distributed anything. fit() below is FULL-BATCH ONLY and
says so, rather than pretending to be a general trainer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from timet.tensor import Tensor, tensor
from timet.diagnostics import Diagnostic


class TrainError(Diagnostic):
    pass


def accuracy(logits: Tensor, targets) -> float:
    """Fraction of samples whose argmax(logits) equals the target label."""
    if logits.ndim != 2:
        raise TrainError(code="E0630",
                         message=f"accuracy: expected logits of shape (N, C), got {logits.shape}",
                         stage="train")
    pred = logits.argmax(axis=1).data.astype(np.int64)
    true = targets.data.astype(np.int64) if isinstance(targets, Tensor) \
        else np.asarray(list(targets), dtype=np.int64)
    if true.shape != pred.shape:
        raise TrainError(code="E0631",
                         message=f"accuracy: targets shape {true.shape} != predictions shape {pred.shape}",
                         stage="train")
    if pred.size == 0:
        raise TrainError(code="E0632", message="accuracy: empty batch", stage="train")
    return float((pred == true).mean())


@dataclass
class History:
    losses: List[float] = field(default_factory=list)

    @property
    def final_loss(self) -> Optional[float]:
        return self.losses[-1] if self.losses else None


@dataclass
class EarlyStopping:
    """Stop when the monitored loss has not improved for `patience` epochs.

    state is intentionally explicit (best, bad_epochs) so behavior is
    inspectable and testable.
    """
    patience: int = 10
    min_delta: float = 0.0
    best: float = field(default=float("inf"))
    bad_epochs: int = 0

    def step(self, loss: float) -> bool:
        """Returns True when training should stop."""
        if loss < self.best - self.min_delta:
            self.best = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def fit(model, loss_fn, optimizer, x: Tensor, y, epochs: int,
        log_every: int = 0, log: Optional[Callable[[str], None]] = None,
        early_stopping: Optional[EarlyStopping] = None,
        metrics: Optional[dict] = None) -> History:
    """Full-batch gradient-descent training loop.

    Each epoch:
      1. forward pass on the ENTIRE batch (no mini-batching yet)
      2. loss.backward()
      3. optimizer.step(); optimizer.zero_grad()

    `metrics` may map name -> fn(model, x, y) -> float, evaluated each epoch
    and recorded in `history` under `metrics[name]`.
    The model's own train()/eval() mode is left to the caller -- fit() does
    not silently flip it.
    """
    if epochs <= 0:
        raise TrainError(code="E0633", message=f"fit: epochs must be positive, got {epochs}",
                         stage="train")
    history = History()
    metric_log = {name: [] for name in (metrics or {})}
    history.metrics = metric_log  # type: ignore[attr-defined]
    for epoch in range(epochs):
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_val = float(loss.item())
        history.losses.append(loss_val)
        for name, fn in (metrics or {}).items():
            metric_log[name].append(float(fn(model, x, y)))
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1) and log:
            extra = " ".join(f"{k}={v[-1]:.4f}" for k, v in metric_log.items() if v)
            log(f"epoch {epoch + 1}/{epochs} loss={loss_val:.6f} {extra}".rstrip())
        if early_stopping is not None and early_stopping.step(loss_val):
            if log:
                log(f"early stopping at epoch {epoch + 1} (best={early_stopping.best:.6f})")
            break
    return history
