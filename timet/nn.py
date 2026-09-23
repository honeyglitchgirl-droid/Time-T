"""Neural-network layer library (master prompt sections 8, 22).

Implemented (all covered by tests in tests/test_nn.py):
  Layers:    Linear, ReLU, Sigmoid, Tanh, Softmax, Flatten, Dropout, Sequential
  Losses:    MSELoss, CrossEntropyLoss, BCELoss
  Plumbing:  Module.parameters(), Module.train()/eval() (recursive)

Still NOT implemented (see docs/ROADMAP.md, Milestone 7):
  Embedding, Conv1D/Conv2D, normalization layers, attention/transformer
  blocks, weight-initialization schemes beyond Kaiming-uniform for Linear.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from timet.tensor import Tensor, tensor, one_hot
from timet.diagnostics import Diagnostic


class NNError(Diagnostic):
    pass


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        return []

    def children(self) -> List["Module"]:
        """Immediate sub-modules. Overridden by containers (Sequential)."""
        return []

    def train(self, mode: bool = True) -> "Module":
        """Set training mode recursively (affects Dropout). Returns self."""
        self.training = bool(mode)
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError


class Linear(Module):
    """y = x @ W^T + b, Kaiming-uniform initialized (seeded, deterministic)."""

    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        bound = 1.0 / math.sqrt(in_features)
        w = rng.uniform(-bound, bound, size=(out_features, in_features)).astype(np.float32)
        b = rng.uniform(-bound, bound, size=(out_features,)).astype(np.float32)
        self.weight = Tensor(w, requires_grad=True)
        self.bias = Tensor(b, requires_grad=True)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        return x.matmul(self.weight.transpose()) + self.bias

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)


class Flatten(Module):
    """Flattens all dimensions except the leading (batch) dimension."""

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1:
            raise NNError(code="E0610", message="Flatten: expected at least 1 dimension",
                          stage="nn")
        return x.reshape((x.shape[0], -1)) if x.ndim > 1 else x


class Dropout(Module):
    """Inverted dropout: scales surviving activations by 1/(1-p) at train
    time, identity at eval time. Seeded => deterministic, so tests and
    examples are reproducible."""

    def __init__(self, p: float = 0.5, seed: int = 0):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise NNError(code="E0611", message=f"Dropout: p must be in [0, 1), got {p}",
                          stage="nn")
        self.p = float(p)
        self._rng = np.random.default_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (self._rng.random(x.shape) < keep).astype(np.float32) / keep
        return x * Tensor(mask)


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def children(self) -> List[Module]:
        return list(self.layers)


class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return (diff * diff).mean()


class CrossEntropyLoss(Module):
    """Mean negative log-likelihood over a batch of logits.

    pred:    Tensor of shape (N, C) -- raw logits (NOT probabilities).
    targets: length-N integer class labels (python list, NumPy array, or
             integer Tensor).

    Computed as:  mean( -sum( one_hot(target) * log_softmax(logits), axis=1 ) )
    Gradient flows through log_softmax's verified primitive-op backward
    rules; one_hot is a constant lookup. Finite-difference checked in
    tests/test_nn.py.
    """

    def forward(self, pred: Tensor, targets) -> Tensor:
        if pred.ndim != 2:
            raise NNError(
                code="E0612",
                message=f"CrossEntropyLoss: expected logits of shape (N, C), got {pred.shape}",
                stage="nn",
            )
        n, c = pred.shape
        targets_t = targets if isinstance(targets, Tensor) else tensor(list(targets))
        if targets_t.data.shape != (n,):
            raise NNError(
                code="E0613",
                message=f"CrossEntropyLoss: expected {n} target label(s), got shape {targets_t.data.shape}",
                stage="nn",
            )
        oh = one_hot(targets_t, c)
        logp = pred.log_softmax(axis=1)
        per_sample = -(oh * logp).sum(axis=1)
        return per_sample.mean()


class BCELoss(Module):
    """Binary cross-entropy on probabilities in (0, 1).

    pred:   predicted probabilities (typically the output of a sigmoid), any shape.
    target: same-shaped 0/1 targets.

    Computed entirely from autograd-tracked primitive ops:
        BCE = mean( -(t * log(pc) + (1 - t) * log(1 - pc)) ),
        pc = clip(pred, eps, 1 - eps)     (numerical stability)
    so gradients reuse clip/log/mean's finite-difference-checked backward
    rules. See docs/AUTOGRAD.md.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pc = pred.clip(self.eps, 1.0 - self.eps)
        one = tensor(1.0)
        return -(target * pc.log() + (one - target) * (one - pc).log()).mean()


# ---------------- functional conveniences ----------------
# Thin wrappers so call sites (including Time-T programs) can write
# nn.mse_loss(p, y) instead of nn.MSELoss()(p, y).

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return MSELoss()(pred, target)


def cross_entropy_loss(logits: Tensor, targets) -> Tensor:
    return CrossEntropyLoss()(logits, targets)


def binary_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    return BCELoss()(pred, target)
