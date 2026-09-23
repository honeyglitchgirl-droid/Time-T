"""Optimizer library (master prompt section 22).

Implemented: SGD, Adam. Both are covered by tests in tests/test_nn.py,
including a real convergence run (not just "step() doesn't crash").

Not implemented yet: AdamW, LR schedules, gradient clipping, momentum for
SGD. See docs/ROADMAP.md Milestone 7/8.
"""
from __future__ import annotations

from typing import List

import numpy as np

from timet.tensor import Tensor


class SGD:
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.data = p.data - self.lr * p.grad.data

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def get_state(self) -> dict:
        """Serializable optimizer state (DD-21). Plain SGD is stateless
        beyond its config; momentum variants override this."""
        return {"type": "SGD", "config": {"lr": self.lr}, "arrays": {}}

    def set_state(self, state: dict) -> None:
        if state.get("type") != "SGD":
            raise ValueError(f"SGD.set_state: state type {state.get('type')!r} != 'SGD'")
        self.lr = state["config"]["lr"]


class Adam:
    """Adam (Kingma & Ba, 2015) with bias correction.

    State (first/second moment estimates + timestep) is kept per parameter,
    keyed by object id, and created lazily on the first step() after a
    gradient exists. Updates happen under autodiff.no_grad semantics because
    they mutate `.data` directly, outside the tape.
    """

    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        if lr <= 0:
            raise ValueError(f"Adam: lr must be positive, got {lr}")
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._m = {}  # id(p) -> first moment np.ndarray
        self._v = {}  # id(p) -> second moment np.ndarray
        self._t = {}  # id(p) -> int timestep

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            k = id(p)
            g = np.asarray(p.grad.data, dtype=np.float64)
            if k not in self._m:
                self._m[k] = np.zeros_like(g)
                self._v[k] = np.zeros_like(g)
                self._t[k] = 0
            self._t[k] += 1
            t = self._t[k]
            self._m[k] = self.beta1 * self._m[k] + (1.0 - self.beta1) * g
            self._v[k] = self.beta2 * self._v[k] + (1.0 - self.beta2) * g * g
            m_hat = self._m[k] / (1.0 - self.beta1 ** t)
            v_hat = self._v[k] / (1.0 - self.beta2 ** t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = (p.data.astype(np.float64) - update).astype(p.data.dtype)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def get_state(self) -> dict:
        """Moment arrays keyed by PARAM POSITION (index in self.params at
        construction), never by id(): positions are stable under
        save/load, ids are not. The contract for a resumed optimizer is
        'constructed over model.parameters() in the same order' -- the
        same determinism load_into already relies on (DD-21)."""
        id_to_pos = {id(p): i for i, p in enumerate(self.params)}
        state = {
            "type": type(self).__name__,
            "config": {"lr": self.lr, "betas": [self.beta1, self.beta2], "eps": self.eps,
                       "t_max_steps_seen": None},
            "arrays": {
                "m": {str(id_to_pos[k]): v for k, v in self._m.items()},
                "v": {str(id_to_pos[k]): v for k, v in self._v.items()},
            },
            "t": {str(id_to_pos[k]): t for k, t in self._t.items()},
        }
        if type(self).__name__ == "AdamW":
            state["config"]["weight_decay"] = self.weight_decay
        return state

    def set_state(self, state: dict) -> None:
        if state.get("type") != type(self).__name__:
            raise ValueError(f"{type(self).__name__}.set_state: state type "
                             f"{state.get('type')!r} does not match")
        n_params = len(self.params)
        for coll in (state["arrays"]["m"], state["arrays"]["v"]):
            for pos, arr in coll.items():
                i = int(pos)
                if i >= n_params:
                    raise ValueError(f"state array at position {i} but optimizer "
                                     f"has {n_params} parameter(s)")
                if arr.shape != self.params[i].data.shape:
                    raise ValueError(f"state array shape {arr.shape} != parameter "
                                     f"{i} shape {self.params[i].data.shape}")
        self.lr = state["config"]["lr"]
        self.beta1, self.beta2 = state["config"]["betas"]
        self.eps = state["config"]["eps"]
        if type(self).__name__ == "AdamW" and "weight_decay" in state["config"]:
            self.weight_decay = state["config"]["weight_decay"]
        self._m = {id(self.params[int(k)]): v for k, v in state["arrays"]["m"].items()}
        self._v = {id(self.params[int(k)]): v for k, v in state["arrays"]["v"].items()}
        self._t = {id(self.params[int(k)]): int(t) for k, t in state.get("t", {}).items()}


class AdamW(Adam):
    """Adam with DECOUPLED weight decay (Loshchilov & Hutter, 2019).

    Unlike L2-in-the-gradient (which interacts badly with Adam's adaptive
    scaling), the penalty is applied directly to the parameter outside the
    adaptive update:

        p = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)

    With weight_decay=0 the update is exactly Adam's (bit-identical on
    the same float64 path -- pinned by a test). Default weight_decay=0.01
    matches common practice; lr and moment state semantics are Adam's.
    """

    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        if weight_decay < 0:
            raise ValueError(f"AdamW: weight_decay must be >= 0, got {weight_decay}")
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            k = id(p)
            g = np.asarray(p.grad.data, dtype=np.float64)
            if k not in self._m:
                self._m[k] = np.zeros_like(g)
                self._v[k] = np.zeros_like(g)
                self._t[k] = 0
            self._t[k] += 1
            t = self._t[k]
            self._m[k] = self.beta1 * self._m[k] + (1.0 - self.beta1) * g
            self._v[k] = self.beta2 * self._v[k] + (1.0 - self.beta2) * g * g
            m_hat = self._m[k] / (1.0 - self.beta1 ** t)
            v_hat = self._v[k] / (1.0 - self.beta2 ** t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps) \
                + self.lr * self.weight_decay * p.data.astype(np.float64)
            p.data = (p.data.astype(np.float64) - update).astype(p.data.dtype)
