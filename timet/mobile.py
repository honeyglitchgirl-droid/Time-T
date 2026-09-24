"""Mobile and ARM64-oriented lightweight inference and INT8 quantization (Milestone 10, DD-27).

Provides:
- Symmetric and asymmetric INT8 linear quantization and dequantization
- QuantizedTensor representation storing int8 data, scale, and zero_point
- QuantizedLinear layer executing integer dot products with float scale output
- quantize_dynamic(model): converts Linear layers in an nn.Module / Sequential to QuantizedLinear
- package_mobile(model, path): serializes a lightweight standalone mobile inference package
- load_mobile(path): loads a packaged mobile model for zero-overhead inference
"""
from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from timet.tensor import Tensor, tensor
from timet.diagnostics import Diagnostic
import timet.nn as nn


class QuantizationError(Diagnostic):
    pass


def quantize_linear(
    x: Union[Tensor, np.ndarray],
    scale: Optional[float] = None,
    zero_point: Optional[int] = None,
    symmetric: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """Quantize floating point array/tensor to INT8 [-128, 127].

    If scale/zero_point are None, compute dynamically from min/max.
    Returns (q_int8, scale, zero_point).
    """
    arr = np.asarray(x.data if isinstance(x, Tensor) else x, dtype=np.float32)
    if symmetric:
        max_val = float(np.max(np.abs(arr)))
        scale = max_val / 127.0 if max_val > 1e-8 else 1.0
        zero_point = 0
        q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    else:
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val - min_val > 1e-8:
            scale = (max_val - min_val) / 255.0
            zero_point = int(np.clip(np.round(-min_val / scale) - 128, -128, 127))
        else:
            scale = 1.0
            zero_point = 0
        q = np.clip(np.round(arr / scale) + zero_point, -128, 127).astype(np.int8)
    return q, float(scale), int(zero_point)


def dequantize_linear(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize INT8 array back to float32."""
    return ((q.astype(np.float32) - zero_point) * scale).astype(np.float32)


class QuantizedLinear(nn.Module):
    """Linear layer operating with INT8 quantized weights.
    Reduces weight memory footprint by 4x compared to float32.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_q = np.zeros((out_features, in_features), dtype=np.int8)
        self.scale = 1.0
        self.zero_point = 0
        self.bias = Tensor(np.zeros((out_features,), dtype=np.float32), requires_grad=False) if bias else None

    @classmethod
    def from_float(cls, float_linear: nn.Linear) -> "QuantizedLinear":
        """Convert a trained nn.Linear layer to QuantizedLinear."""
        q_lin = cls(
            float_linear.in_features,
            float_linear.out_features,
            bias=float_linear.bias is not None,
        )
        q_w, scale, zp = quantize_linear(float_linear.weight.data, symmetric=True)
        q_lin.weight_q = q_w
        q_lin.scale = scale
        q_lin.zero_point = zp
        if float_linear.bias is not None:
            q_lin.bias = Tensor(np.copy(float_linear.bias.data), requires_grad=False)
        return q_lin

    def forward(self, x: Tensor) -> Tensor:
        x_arr = np.asarray(x.data, dtype=np.float32)
        # Dynamic activation quantization for int32 dot products:
        x_q, x_scale, _ = quantize_linear(x_arr, symmetric=True)
        # Integer matrix multiplication with transposed weight (out_features, in_features)
        acc_int32 = np.matmul(x_q.astype(np.int32), self.weight_q.T.astype(np.int32))
        # Rescale to float32
        out_f32 = acc_int32.astype(np.float32) * (x_scale * self.scale)
        if self.bias is not None:
            out_f32 = out_f32 + self.bias.data
        return Tensor(out_f32, requires_grad=False)

    def parameters(self):
        # Parameters for inference / inspection
        p = []
        if self.bias is not None:
            p.append(self.bias)
        return p



def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Post-training dynamic INT8 quantization for neural network modules.
    Replaces all Linear modules with QuantizedLinear.
    """
    if isinstance(model, nn.Linear):
        return QuantizedLinear.from_float(model)
    if isinstance(model, nn.Sequential):
        new_layers = []
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                new_layers.append(QuantizedLinear.from_float(layer))
            else:
                new_layers.append(layer)
        return nn.Sequential(*new_layers)
    # Generic container
    for name, child in list(model.__dict__.items()):
        if isinstance(child, nn.Linear):
            setattr(model, name, QuantizedLinear.from_float(child))
        elif isinstance(child, nn.Module):
            quantize_dynamic(child)
    return model


MOBILE_FORMAT = "timet-mobile-package"
MOBILE_VERSION = 1


def package_mobile(model: nn.Module, path: Union[str, Path]) -> Path:
    """Export a lightweight mobile inference package (.ttm / .ttpack).

    Contains:
    - manifest.json describing layers, shapes, quantization scales
    - raw weight buffers (int8 quantized or float32 bias)
    """
    path = Path(path)
    layers_meta = []
    buffers = {}

    if not isinstance(model, nn.Sequential):
        raise QuantizationError(
            code="E0900",
            message="Mobile packaging currently supports Sequential architectures",
            stage="mobile",
        )

    for i, layer in enumerate(model.layers):
        if isinstance(layer, QuantizedLinear):
            w_key = f"layer_{i}_weight_q.bin"
            buffers[w_key] = layer.weight_q.tobytes()
            meta = {
                "type": "QuantizedLinear",
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "scale": float(layer.scale),
                "zero_point": int(layer.zero_point),
                "weight_buffer": w_key,
                "has_bias": layer.bias is not None,
            }
            if layer.bias is not None:
                b_key = f"layer_{i}_bias.bin"
                buffers[b_key] = np.asarray(layer.bias.data, dtype=np.float32).tobytes()
                meta["bias_buffer"] = b_key
            layers_meta.append(meta)
        elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax, nn.GELU, nn.Flatten)):
            layers_meta.append({"type": type(layer).__name__})
        else:
            raise QuantizationError(
                code="E0901",
                message=f"Layer type {type(layer).__name__} not supported in mobile package v1",
                stage="mobile",
            )

    manifest = {
        "format": MOBILE_FORMAT,
        "version": MOBILE_VERSION,
        "layers": layers_meta,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for name, data in buffers.items():
            zf.writestr(name, data)

    path.write_bytes(buf.getvalue())
    return path


def load_mobile(path: Union[str, Path]) -> nn.Sequential:
    """Load a mobile inference package into an executable nn.Sequential model."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            if manifest.get("format") != MOBILE_FORMAT:
                raise QuantizationError(
                    code="E0902",
                    message=f"Invalid mobile package format: {manifest.get('format')}",
                    stage="mobile",
                )
            layers = []
            for meta in manifest["layers"]:
                ltype = meta["type"]
                if ltype == "QuantizedLinear":
                    in_f = meta["in_features"]
                    out_f = meta["out_features"]
                    has_bias = meta["has_bias"]
                    ql = QuantizedLinear(in_f, out_f, bias=has_bias)
                    ql.scale = meta["scale"]
                    ql.zero_point = meta["zero_point"]
                    raw_w = zf.read(meta["weight_buffer"])
                    ql.weight_q = np.frombuffer(raw_w, dtype=np.int8).reshape(out_f, in_f)
                    if has_bias:
                        raw_b = zf.read(meta["bias_buffer"])
                        ql.bias = Tensor(np.frombuffer(raw_b, dtype=np.float32).reshape(out_f))
                    layers.append(ql)

                elif ltype == "ReLU":
                    layers.append(nn.ReLU())
                elif ltype == "Sigmoid":
                    layers.append(nn.Sigmoid())
                elif ltype == "Tanh":
                    layers.append(nn.Tanh())
                elif ltype == "Softmax":
                    layers.append(nn.Softmax())
                elif ltype == "GELU":
                    layers.append(nn.GELU())
                elif ltype == "Flatten":
                    layers.append(nn.Flatten())
            return nn.Sequential(*layers)
    except Exception as e:
        if isinstance(e, QuantizationError):
            raise
        raise QuantizationError(
            code="E0903",
            message=f"Failed to load mobile package: {e}",
            stage="mobile",
        )
