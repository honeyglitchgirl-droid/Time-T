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


# --------------------------------------------------------------------------- #
# Datasets & DataLoader (Milestone 8: mini-batching)
# --------------------------------------------------------------------------- #
class TensorDataset:
    """Pairs of (x, y) Tensors aligned on a leading sample axis.

    Groups (dict-of-columns, on-disk formats, transforms) do not exist yet;
    this is the honest minimum for mini-batch training, with N as the only
    notion of 'sample' the runtime needs.
    """

    def __init__(self, x: Tensor, y: Tensor):
        if not isinstance(x, Tensor) or not isinstance(y, Tensor):
            raise TrainError(code="E0634",
                             message="TensorDataset: both x and y must be Tensors",
                             stage="train")
        if x.shape[0] != y.shape[0]:
            raise TrainError(code="E0635",
                             message=f"TensorDataset: sample counts differ "
                                     f"(x has {x.shape[0]}, y has {y.shape[0]})",
                             stage="train")
        self.x, self.y = x, y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DataLoader:
    """Iterable mini-batch view over a TensorDataset.

    shuffle=True draws a fresh permutation each epoch from a DEDICATED RNG
    seeded at construction (np.random.default_rng(seed)) -- iterating twice
    gives two different-but-deterministic orderings, and a loader with the
    same seed reproduces both exactly. drop_last=False yields a smaller
    final batch rather than silently down-padded or dropped data (the
    classic ways training quietly goes wrong).
    """

    def __init__(self, dataset: TensorDataset, batch_size: int,
                 shuffle: bool = False, seed: Optional[int] = None,
                 drop_last: bool = False):
        if not isinstance(dataset, TensorDataset):
            raise TrainError(code="E0636",
                             message="DataLoader: dataset must be a TensorDataset",
                             stage="train")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise TrainError(code="E0637",
                             message=f"DataLoader: batch_size must be a positive Int, got {batch_size!r}",
                             stage="train")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)
        self.seed = seed

    def __len__(self) -> int:
        n, b = len(self.dataset), self.batch_size
        full, rem = divmod(n, b)
        return full if (self.drop_last or rem == 0) else full + 1

    def __iter__(self):
        n = len(self.dataset)
        idx = self._rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            bidx = idx[start:start + self.batch_size]
            if self.drop_last and len(bidx) < self.batch_size:
                break
            yield self.dataset.x[bidx], self.dataset.y[bidx]


def fit_loader(model, loss_fn, optimizer, loader: DataLoader, epochs: int,
               log_every: int = 0, log: Optional[Callable[[str], None]] = None,
               early_stopping: Optional[EarlyStopping] = None,
               metrics: Optional[dict] = None,
               scheduler: "LRScheduler" = None) -> History:
    """Mini-batch training loop: one step per batch, one History entry per
    epoch (the MEAN batch loss of that epoch -- not the last batch's, which
    a noisy final batch would misrepresent).

    `metrics` map name -> fn(model, xb, yb) -> float, averaged per epoch.
    `scheduler.step()` (if given) runs ONCE PER EPOCH, after the epoch's
    optimizer steps -- the textbook order; stepping it per batch is a
    silent way to anneal 100x faster than intended.
    """
    if epochs <= 0:
        raise TrainError(code="E0633", message=f"fit_loader: epochs must be positive, got {epochs}",
                         stage="train")
    if len(loader) == 0:
        raise TrainError(code="E0638", message="fit_loader: loader produces zero batches "
                                               "(drop_last with batch_size > dataset size?)",
                         stage="train")
    history = History()
    metric_log = {name: [] for name in (metrics or {})}
    history.metrics = metric_log  # type: ignore[attr-defined]
    for epoch in range(epochs):
        losses = []
        for name in metric_log:
            metric_log[name].append(0.0)
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.item()))
            for name, fn in (metrics or {}).items():
                metric_log[name][-1] += float(fn(model, xb, yb))
        epoch_loss = float(np.mean(losses))
        history.losses.append(epoch_loss)
        for name in metric_log:
            metric_log[name][-1] /= len(losses)
        if scheduler is not None:
            scheduler.step()
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1) and log:
            extra = " ".join(f"{k}={v[-1]:.4f}" for k, v in metric_log.items() if v)
            log(f"epoch {epoch + 1}/{epochs} loss={epoch_loss:.6f} {extra}".rstrip())
        if early_stopping is not None and early_stopping.step(epoch_loss):
            if log:
                log(f"early stopping at epoch {epoch + 1} (best={early_stopping.best:.6f})")
            break
    return history


# --------------------------------------------------------------------------- #
# LR schedules (Milestone 8) -- crucial checkpoint: optimizer.lr is INT in
# test data more often than people think; keep exact-rational math where the
# formulas allow by computing from base_lr each step instead of compounding
# floats.
# --------------------------------------------------------------------------- #
class LRScheduler:
    """Base schedule: adjust optimizer.lr from its CONSTRUCTION-TIME base
    value once per .step() call. Call once per epoch (fit_loader does)."""

    def __init__(self, optimizer):
        if not hasattr(optimizer, "lr"):
            raise TrainError(code="E0639", message="LRScheduler: optimizer has no 'lr' attribute",
                             stage="train")
        self.optimizer = optimizer
        self.base_lr = float(optimizer.lr)
        self.last_epoch = 0

    def get_lr(self, epoch: int) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr(self.last_epoch)

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.lr)


class StepLR(LRScheduler):
    """lr = base * gamma ** floor(epoch / step_size)."""

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        if step_size <= 0:
            raise TrainError(code="E0640", message=f"StepLR: step_size must be positive, got {step_size}",
                             stage="train")
        self.step_size, self.gamma = int(step_size), float(gamma)
        super().__init__(optimizer)

    def get_lr(self, epoch: int) -> float:
        return self.base_lr * (self.gamma ** (epoch // self.step_size))


class ExponentialLR(LRScheduler):
    """lr = base * gamma ** epoch."""

    def __init__(self, optimizer, gamma: float = 0.95):
        self.gamma = float(gamma)
        super().__init__(optimizer)

    def get_lr(self, epoch: int) -> float:
        return self.base_lr * (self.gamma ** epoch)


class CosineAnnealingLR(LRScheduler):
    """lr = base * 0.5 * (1 + cos(pi * epoch / T_max)) -- decays to 0."""

    def __init__(self, optimizer, t_max: int):
        if t_max <= 0:
            raise TrainError(code="E0641", message=f"CosineAnnealingLR: t_max must be positive, got {t_max}",
                             stage="train")
        self.t_max = int(t_max)
        super().__init__(optimizer)

    def get_lr(self, epoch: int) -> float:
        return self.base_lr * 0.5 * (1.0 + np.cos(np.pi * min(epoch, self.t_max) / self.t_max))
