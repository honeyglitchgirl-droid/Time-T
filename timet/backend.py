"""Backend abstraction (master prompt section 14).

Only one backend exists today: NumpyCpuBackend. The interface is designed so
a second backend (GPU/ARM64/etc.) is an additive change. See
docs/DESIGN_DECISIONS.md DD-5.
"""
from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    supports_f32: bool
    supports_f64: bool
    gpu: bool
    simd: str
    max_tensor_rank: int
    device_name: str

    def to_json(self) -> dict:
        return dict(
            name=self.name,
            supports_f32=self.supports_f32,
            supports_f64=self.supports_f64,
            gpu=self.gpu,
            simd=self.simd,
            max_tensor_rank=self.max_tensor_rank,
            device_name=self.device_name,
        )


class Backend(ABC):
    name: str

    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @abstractmethod
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def elementwise(self, op: str, *args: np.ndarray) -> np.ndarray: ...


_ELEMENTWISE_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "neg": lambda a: -a,
    "exp": lambda a: np.exp(a),
    "log": lambda a: np.log(a),
    "sqrt": lambda a: np.sqrt(a),
    "relu": lambda a: np.maximum(a, 0),
    "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a)),
    "tanh": lambda a: np.tanh(a),
}


class NumpyCpuBackend(Backend):
    name = "cpu-numpy"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            supports_f32=True,
            supports_f64=True,
            gpu=False,
            simd=np.show_config.__module__ and _detect_blas(),
            max_tensor_rank=32,
            device_name=platform.processor() or platform.machine() or "unknown-cpu",
        )

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def elementwise(self, op: str, *args: np.ndarray) -> np.ndarray:
        if op not in _ELEMENTWISE_OPS:
            raise ValueError(f"backend '{self.name}' does not implement elementwise op '{op}'")
        return _ELEMENTWISE_OPS[op](*args)


def _detect_blas() -> str:
    try:
        cfg = np.show_config(mode="dicts")
        if isinstance(cfg, dict):
            blas = cfg.get("Build Dependencies", {}).get("blas", {})
            name = blas.get("name")
            if name:
                return f"numpy+{name}"
    except Exception:
        pass
    return "numpy-generic"


_default_backend = NumpyCpuBackend()


def get_default_backend() -> Backend:
    return _default_backend
