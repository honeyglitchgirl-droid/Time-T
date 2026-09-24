"""Tests for INT8 quantization and mobile packaging (Milestone 10, DD-27)."""
import numpy as np
import pytest

from timet.tensor import Tensor, tensor
import timet.nn as nn
from timet.mobile import (
    quantize_linear,
    dequantize_linear,
    QuantizedLinear,
    quantize_dynamic,
    package_mobile,
    load_mobile,
    QuantizationError,
)


def test_quantize_dequantize_linear_symmetric():
    data = np.array([[-1.0, 0.0, 0.5], [1.0, -0.5, 0.25]], dtype=np.float32)
    q, scale, zp = quantize_linear(data, symmetric=True)
    assert q.dtype == np.int8
    assert zp == 0
    deq = dequantize_linear(q, scale, zp)
    np.testing.assert_allclose(data, deq, atol=0.02)


def test_quantize_dequantize_linear_asymmetric():
    data = np.array([[0.0, 1.5, 3.0], [0.5, 2.0, 2.5]], dtype=np.float32)
    q, scale, zp = quantize_linear(data, symmetric=False)
    assert q.dtype == np.int8
    deq = dequantize_linear(q, scale, zp)
    np.testing.assert_allclose(data, deq, atol=0.02)


def test_quantized_linear_forward():
    lin = nn.Linear(4, 3, seed=42)
    q_lin = QuantizedLinear.from_float(lin)
    assert q_lin.weight_q.shape == (3, 4)
    assert q_lin.weight_q.dtype == np.int8

    x = tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 1.5, -2.0]])
    out_float = lin(x)
    out_q = q_lin(x)
    # Output shapes match
    assert out_q.shape == out_float.shape
    # Dynamic INT8 outputs are close to float32
    np.testing.assert_allclose(out_q.data, out_float.data, atol=0.2)



def test_quantize_dynamic_sequential():
    model = nn.Sequential(
        nn.Linear(6, 4, seed=1),
        nn.ReLU(),
        nn.Linear(4, 2, seed=2),
    )
    q_model = quantize_dynamic(model)
    assert isinstance(q_model.layers[0], QuantizedLinear)
    assert isinstance(q_model.layers[1], nn.ReLU)
    assert isinstance(q_model.layers[2], QuantizedLinear)

    x = tensor([[0.5, -0.2, 1.0, -1.0, 0.0, 0.8]])
    y_orig = model(x)
    y_q = q_model(x)
    assert y_q.shape == y_orig.shape
    np.testing.assert_allclose(y_q.data, y_orig.data, atol=0.25)


def test_package_and_load_mobile(tmp_path):
    model = nn.Sequential(
        nn.Linear(8, 4, seed=10),
        nn.GELU(),
        nn.Linear(4, 2, seed=20),
    )
    q_model = quantize_dynamic(model)
    pack_path = tmp_path / "model.ttm"
    package_mobile(q_model, pack_path)
    assert pack_path.is_file()

    loaded = load_mobile(pack_path)
    assert len(loaded.layers) == 3
    assert isinstance(loaded.layers[0], QuantizedLinear)
    assert isinstance(loaded.layers[1], nn.GELU)
    assert isinstance(loaded.layers[2], QuantizedLinear)

    x = tensor([[0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]])
    y1 = q_model(x)
    y2 = loaded(x)
    np.testing.assert_allclose(y1.data, y2.data, atol=1e-5)


def test_package_mobile_unsupported_layer(tmp_path):
    class CustomLayer(nn.Module):
        def forward(self, x):
            return x

    model = nn.Sequential(CustomLayer())
    with pytest.raises(QuantizationError) as exc_info:
        package_mobile(model, tmp_path / "bad.ttm")
    assert "E0901" in exc_info.value.code


def test_cli_package_subcommand(tmp_path):
    from timet.cli import main as cli_main
    from timet import checkpoint
    m = nn.Sequential(nn.Linear(4, 2, seed=1))
    ck_path = tmp_path / "model.ttck"
    checkpoint.save_bin(m, ck_path)

    pkg_path = tmp_path / "packaged.ttm"
    res = cli_main(["package", str(ck_path), "-o", str(pkg_path), "--json"])
    assert res == 0
    assert pkg_path.is_file()
    loaded = load_mobile(pkg_path)
    assert len(loaded.layers) >= 1

