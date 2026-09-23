"""Minimal optimizer library (master prompt section 22).

Only SGD exists today. Adam/AdamW are listed as future work in
docs/ROADMAP.md Milestone 7 and intentionally not implemented yet.
"""
from __future__ import annotations

from typing import List

import numpy as np

from timet.tensor import Tensor


class SGD:
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.data = p.data - self.lr * p.grad.data

    def zero_grad(self):
        for p in self.params:
            p.grad = None
