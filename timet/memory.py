"""Lightweight real memory accounting for Time-T tensors (master prompt section 17).

This tracks actual bytes allocated/freed by Tensor objects backed by NumPy
arrays -- it is not a simulation. Peak/active counters update as tensors are
created and garbage collected.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class MemoryStats:
    allocated: int = 0     # bytes currently held by live tensors
    peak: int = 0          # highest 'allocated' ever observed
    active_tensors: int = 0
    total_allocations: int = 0
    parameters: int = 0    # bytes tagged as parameters (requires_grad params)
    gradients: int = 0     # bytes tagged as gradient buffers

    def to_json(self) -> dict:
        return {
            "allocated": self.allocated,
            "peak": self.peak,
            "active_tensors": self.active_tensors,
            "total_allocations": self.total_allocations,
            "parameters": self.parameters,
            "gradients": self.gradients,
        }


_lock = threading.Lock()
_stats = MemoryStats()


def stats() -> MemoryStats:
    return _stats


def reset():
    global _stats
    with _lock:
        _stats = MemoryStats()


def record_alloc(nbytes: int, is_grad: bool = False):
    with _lock:
        _stats.allocated += nbytes
        _stats.active_tensors += 1
        _stats.total_allocations += 1
        if is_grad:
            _stats.gradients += nbytes
        if _stats.allocated > _stats.peak:
            _stats.peak = _stats.allocated


def record_free(nbytes: int, is_grad: bool = False):
    with _lock:
        _stats.allocated = max(0, _stats.allocated - nbytes)
        _stats.active_tensors = max(0, _stats.active_tensors - 1)
        if is_grad:
            _stats.gradients = max(0, _stats.gradients - nbytes)
