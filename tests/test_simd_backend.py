"""Tests for Native Multi-Threaded SIMD OpenMP Backend."""
import numpy as np
import pytest

from timet.simd_backend import SimdCpuBackend, _LOADED_SIMD
from timet.backend import get_default_backend


def test_simd_backend_compiled_and_active():
    assert _LOADED_SIMD is not None
    backend = SimdCpuBackend()
    caps = backend.capabilities()
    assert caps.name == "cpu-simd-openmp"
    assert "AVX" in caps.simd or "OpenMP" in caps.simd


def test_simd_matmul_matches_numpy_reference():
    b = SimdCpuBackend()
    rng = np.random.default_rng(42)
    a = rng.standard_normal((64, 128), dtype=np.float32)
    c = rng.standard_normal((128, 32), dtype=np.float32)
    out_simd = b.matmul(a, c)
    out_ref = a @ c
    np.testing.assert_allclose(out_simd, out_ref, rtol=1e-4, atol=1e-4)


def test_simd_elementwise_add_and_relu():
    b = SimdCpuBackend()
    x = np.array([-2.0, -1.0, 0.0, 1.5, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    res_add = b.elementwise("add", x, y)
    np.testing.assert_allclose(res_add, x + y)

    res_relu = b.elementwise("relu", x)
    np.testing.assert_allclose(res_relu, np.maximum(x, 0.0))
