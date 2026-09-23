"""Checkpointing: save/load model parameters (master prompt section 24-lite).

Format: a single deterministic JSON file (versioned, human-inspectable,
diff-friendly). This is deliberately simple and honest about its limits:
floats are stored as decimal lists, so files are large and loading is slow
for big models. It is intended for the small models Time-T can actually
train today (see docs/ROADMAP.md). A binary format (e.g. safetensors-style)
is future work and will be documented as a Design Decision before it lands.

Determinism: keys are sorted, floats round-trip exactly via repr(), and the
same parameters always produce byte-identical files (verified by tests).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from timet.tensor import Tensor
from timet.diagnostics import Diagnostic

FORMAT = "timet-checkpoint"
VERSION = 1


class CheckpointError(Diagnostic):
    pass


def _name_parameters(target) -> Dict[str, Tensor]:
    """Return {name: tensor} for either an nn.Module or a list of Tensors.

    For modules we walk well-known containers (Sequential) and name
    parameters by path, e.g. "layers.0.weight". For plain lists we name by
    position ("param_0", ...). Names are stable for a given structure, which
    is what load() relies on.
    """
    named: Dict[str, Tensor] = {}

    def walk(obj, prefix: str):
        if isinstance(obj, Tensor):
            named[prefix] = obj
            return
        if hasattr(obj, "parameters") and callable(getattr(obj, "parameters")):
            params = obj.parameters()
            # containers first (so Sequential gets numbered sub-paths) ...
            children = obj.children() if hasattr(obj, "children") else []
            if children:
                for i, child in enumerate(children):
                    walk(child, f"{prefix}.{i}" if prefix else str(i))
                return
            # ... otherwise this is a leaf module: attach its tensors by attr name
            attrs = obj.__dict__
            tensor_attrs = [k for k, v in attrs.items() if isinstance(v, Tensor)]
            if tensor_attrs:
                for k in sorted(tensor_attrs):
                    named[f"{prefix}.{k}" if prefix else k] = attrs[k]
            else:
                for i, p in enumerate(params):
                    named[f"{prefix}.{i}" if prefix else str(i)] = p
            return
        raise CheckpointError(
            code="E0620",
            message=f"cannot checkpoint object of type {type(obj).__name__}",
            stage="checkpoint",
        )

    if isinstance(target, Tensor) or hasattr(target, "parameters"):
        walk(target, "")
        return dict(sorted(named.items()))
    if isinstance(target, (list, tuple)):
        for i, p in enumerate(target):
            if not isinstance(p, Tensor):
                raise CheckpointError(
                    code="E0621",
                    message="parameter lists must contain only Tensors",
                    stage="checkpoint",
                )
            named[f"param_{i}"] = p
        return named
    raise CheckpointError(
        code="E0622",
        message="save/load expects an nn.Module, a Tensor, or a list of Tensors",
        stage="checkpoint",
    )


def state_dict(target) -> Dict[str, Tensor]:
    """Public alias: {name: Tensor} snapshot view of a module's parameters."""
    return _name_parameters(target)


def save(target, path: Union[str, Path]) -> Path:
    """Serialize parameters of `target` to `path` (JSON). Returns the path."""
    named = _name_parameters(target)
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "tensors": {
            name: {
                "dtype": t.dtype,
                "shape": list(t.shape),
                "data": [float(x) for x in np.asarray(t.data, dtype=np.float64).ravel()],
            }
            for name, t in named.items()
        },
    }
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load(path: Union[str, Path]) -> Dict[str, Tensor]:
    """Load a checkpoint file into {name: Tensor} (no model required)."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise CheckpointError(code="E0623", message=f"cannot read checkpoint: {e}",
                              stage="checkpoint")
    if payload.get("format") != FORMAT:
        raise CheckpointError(code="E0624",
                              message=f"not a timet checkpoint (format={payload.get('format')!r})",
                              stage="checkpoint")
    if payload.get("version") != VERSION:
        raise CheckpointError(
            code="E0625",
            message=f"unsupported checkpoint version {payload.get('version')!r} (expected {VERSION})",
            stage="checkpoint",
        )
    out: Dict[str, Tensor] = {}
    for name, spec in payload.get("tensors", {}).items():
        raw = np.asarray(spec["data"], dtype=np.float64)
        try:
            arr = raw.reshape(spec["shape"])
        except ValueError:
            raise CheckpointError(
                code="E0628",
                message=f"corrupt checkpoint: tensor '{name}' has {raw.size} value(s) "
                        f"but declares shape {spec['shape']}",
                stage="checkpoint",
            )
        dtype = spec["dtype"] if spec["dtype"] in ("f32", "f64", "i32", "i64", "bool") else "f32"
        out[name] = Tensor(arr, dtype=dtype, requires_grad=True)
    return out


def load_into(target, path: Union[str, Path], strict: bool = True):
    """Load checkpoint values from `path` into `target`'s parameters in place.

    With strict=True every name must match exactly both ways and shapes must
    agree; any mismatch raises CheckpointError naming the offending key.
    """
    named = _name_parameters(target)
    loaded = load(path)
    if strict:
        missing = sorted(set(named) - set(loaded))
        extra = sorted(set(loaded) - set(named))
        if missing or extra:
            raise CheckpointError(
                code="E0626",
                message=f"checkpoint key mismatch: missing={missing}, unexpected={extra}",
                stage="checkpoint",
            )
    for name, t in named.items():
        if name not in loaded:
            continue
        src = loaded[name]
        if src.shape != t.shape:
            raise CheckpointError(
                code="E0627",
                message=f"shape mismatch for '{name}': checkpoint {src.shape} vs model {t.shape}",
                stage="checkpoint",
            )
        t.data = src.data.astype(t.data.dtype)
    return target
