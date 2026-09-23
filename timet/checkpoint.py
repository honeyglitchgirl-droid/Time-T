"""Checkpointing: save/load model parameters (master prompt section 24-lite).

Formats (two, deliberately):
  v1 ".json" -- a single deterministic JSON file (versioned,
      human-inspectable, diff-friendly). Floats are stored as decimal
      lists: files are large and loading is slow for big models.
  v2 ".ttck" -- a deterministic zip containing manifest.json plus raw
      little-endian f32 .npy payloads (DD-20). Chosen over the npz
      SHORTCUT of np.savez_compressed because that does not pin
      timestamps/headers: two npz saves of identical parameters can
      differ at the byte level. v2 fixes every zip timestamp to the DOS
      epoch (1980-01-01) and sorts entries, so byte-identity holds.
      HONEST LIMITS: parameter names must be unique (true for all
      current models); non-f32 arrays are CAST to f32 with the original
      dtype merely declared, not restored; dtype metadata beyond f32 is
      future work (see ROADMAP Milestone 8).

Determinism: keys are sorted; v1 floats round-trip exactly via repr();
same parameters -> byte-identical file under BOTH formats (tested).
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from timet.tensor import Tensor
from timet.diagnostics import Diagnostic

FORMAT = "timet-checkpoint"
VERSION = 1
BIN_FORMAT = "timet-checkpoint-bin"
BIN_VERSION = 2
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # fixed DOS-epoch stamp for byte-identity


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
    # format detection by CONTENT (zip magic), not extension:
    # mis-named files still load correctly.
    try:
        with open(path, "rb") as fh:
            if fh.read(2) == b"PK":
                return load_bin(path)
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



# --------------------------------------------------------------------------- #
# v2: deterministic binary format (DD-20): zip + manifest.json + raw npy,
# fixed zip timestamps + sorted entries so byte-identity is part of FORMAT.
# --------------------------------------------------------------------------- #
def save_bin(target, path: Union[str, Path]) -> Path:
    """Save parameters as a deterministic binary checkpoint (.ttck).

    Arrays are stored little-endian f32 (other dtypes are cast; the original
    dtype is DECLARED in the manifest but not restored -- honest limit).
    Same parameters -> byte-identical file across runs.
    """
    named = _name_parameters(target)
    arrays: Dict[str, np.ndarray] = {}
    declared: Dict[str, dict] = {}
    for name, ten in named.items():
        arr = np.asarray(ten.data)
        declared[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                          "stored_dtype": "float32"}
        arrays[name] = arr.astype(np.float32)
    manifest = {
        "format": BIN_FORMAT,
        "version": BIN_VERSION,
        "iteration_scheme": "name-sorted zip entries; raw little-endian f32 payloads",
        "tensors": declared,
        "limits": [
            "all values stored as little-endian f32 (original dtype declared, not restored)",
            "parameter names must be unique across the model",
        ],
    }
    mbytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        info = zipfile.ZipInfo("manifest.json", date_time=_ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0
        zf.writestr(info, mbytes)
        for name in sorted(arrays):
            raw = io.BytesIO()
            np.save(raw, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"tensors/{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0
            zf.writestr(info, raw.getvalue())
    path = Path(path)
    path.write_bytes(buf.getvalue())
    return path


def load_bin(path: Union[str, Path]) -> Dict[str, Tensor]:
    """Load a v2 binary checkpoint. Content-checked against its manifest:
    extra OR missing entries are structural errors (E0648), not warnings."""
    path = Path(path)
    try:
        zf = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as e:
        raise CheckpointError(code="E0645",
                              message=f"cannot read binary checkpoint: {e}",
                              stage="checkpoint")
    with zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise CheckpointError(code="E0646",
                                  message="binary checkpoint missing manifest.json",
                                  stage="checkpoint")
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != BIN_FORMAT:
            raise CheckpointError(code="E0647",
                                  message=f"unsupported binary checkpoint format "
                                          f"{manifest.get('format')!r}",
                                  stage="checkpoint")
        if manifest.get("version") != BIN_VERSION:
            raise CheckpointError(code="E0648",
                                  message=f"unsupported binary checkpoint version "
                                          f"{manifest.get('version')} (expected {BIN_VERSION})",
                                  stage="checkpoint")
        declared = manifest.get("tensors", {})
        expected = {"manifest.json"} | {f"tensors/{n}.npy" for n in declared}
        if names != expected:
            extra = names - expected
            missing = expected - names
            raise CheckpointError(code="E0648",
                                  message=f"binary checkpoint contents mismatch "
                                          f"(extra={sorted(extra)}, missing={sorted(missing)})",
                                  stage="checkpoint")
        out: Dict[str, Tensor] = {}
        for name in sorted(declared):
            arr = np.load(io.BytesIO(zf.read(f"tensors/{name}.npy")), allow_pickle=False)
            shape = declared[name].get("shape")
            if shape is not None and list(arr.shape) != list(shape):
                raise CheckpointError(code="E0648",
                                      message=f"tensor '{name}' shape {list(arr.shape)} "
                                              f"!= manifest {shape}",
                                      stage="checkpoint")
            out[name] = Tensor(arr, dtype="f32", requires_grad=True)
        return out
