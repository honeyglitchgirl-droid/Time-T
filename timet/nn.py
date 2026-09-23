"""Minimal neural-network layer library (master prompt sections 8, 22).

Only Linear, ReLU, Sequential, and MSELoss exist today -- intentionally, per
docs/ROADMAP.md Milestone 7 ("MINIMAL START"). Do not assume Embedding,
Conv1D/Conv2D, normalization, dropout, or attention exist; they raise
ImportError-style AttributeError because they are simply not defined.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from timet.tensor import Tensor, tensor


class Module:
    def parameters(self) -> List[Tensor]:
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError


class Linear(Module):
    """y = x @ W^T + b, Kaiming-uniform initialized."""

    def __init__(self, in_features: int, out_features: int, seed: int = 0):
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


class Sequential(Module):
    def __init__(self, *layers: Module):
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


class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return (diff * diff).mean()
