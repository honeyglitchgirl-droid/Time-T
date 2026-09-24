"""Tests for Native JIT Accelerator Kernels (Production-Grade Acceleration)."""
import numpy as np
import pytest

from timet.tensor import Tensor, tensor
import timet.jit_kernels as jk
import timet.nn as nn


def test_native_jit_active_and_loaded():
    assert jk.has_fast_kernels() is True


def test_fast_relu_matches_numpy():
    x = np.random.randn(1000).astype(np.float32)
    ref = np.maximum(x, 0.0)
    fast = jk.fast_relu(x)
    np.testing.assert_allclose(fast, ref, atol=1e-6)


def test_fast_gelu_matches_reference():
    x = np.random.randn(2000).astype(np.float32)
    c = float(np.sqrt(2.0 / np.pi))
    ref = 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))
    fast = jk.fast_gelu(x)
    np.testing.assert_allclose(fast, ref, atol=1e-4)


def test_fast_layernorm_matches_reference():
    x = np.random.randn(8, 128).astype(np.float32)
    w = np.ones((128,), dtype=np.float32)
    b = np.zeros((128,), dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    ref = (x - mean) / np.sqrt(var + 1e-5)
    fast = jk.fast_layernorm(x, w, b, eps=1e-5)
    np.testing.assert_allclose(fast, ref, atol=1e-5)


def test_nn_gelu_backward_with_fast_kernels():
    x = Tensor([[0.5, -1.0, 2.0], [-0.2, 0.8, -1.5]], requires_grad=True)
    y = nn.gelu(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Analytical gradient verification
    xd = x.data
    c = float(np.sqrt(2.0 / np.pi))
    inner = c * (xd + 0.044715 * xd ** 3)
    t = np.tanh(inner)
    dt = 1.0 - t ** 2
    d_inner = c * (1.0 + 3.0 * 0.044715 * xd ** 2)
    expected_grad = 0.5 * (1.0 + t) + 0.5 * xd * dt * d_inner
    np.testing.assert_allclose(x.grad.data, expected_grad, atol=1e-4)

