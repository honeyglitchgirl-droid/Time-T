# Time-T — Tensor Reference

`timet.tensor.Tensor` wraps a NumPy `ndarray`. See `docs/ARCHITECTURE.md` §4
for the high-level model and `docs/DESIGN_DECISIONS.md` DD-1 for why NumPy is
the storage/kernel layer.

## Construction

| Time-T | Python API |
|---|---|
| `tensor([1.0, 2.0])` | `timet.tensor.tensor([1.0, 2.0])` |
| `tensor([1.0, 2.0], grad=true)` | `tensor([1.0, 2.0], grad=True)` |
| `zeros([2, 3])` | `timet.tensor.zeros([2, 3])` |
| `ones([3])` | `timet.tensor.ones([3])` |

Default dtype is `f32` (`np.float32`) unless an explicit `dtype` is given or
the source array is already an integer/bool NumPy array.

## Dtypes

`f32` (default), `f64`, `i32`, `i64`, `bool`. Mapped in `_DTYPE_MAP`.

## Shape semantics

Shape is **not** part of the static type (see `docs/LANGUAGE.md` §2) — it's
tracked dynamically on the `Tensor.shape` property and validated at each
operation. Broadcasting follows standard NumPy broadcasting rules
(trailing-dimension alignment, size-1 dimensions stretch).

## Operations and their gradient rules

Every op below has: (1) a differential test against the equivalent raw NumPy
computation (`tests/test_tensor.py`), and (2) for differentiable ops, a
finite-difference gradient check (`tests/test_autodiff.py`).

| Op | Forward | Backward rule |
|---|---|---|
| `add` | `a + b` | grad flows unchanged to both (broadcast-summed back) |
| `sub` | `a - b` | `+g` to `a`, `-g` to `b` |
| `mul` | `a * b` | `g*b` to `a`, `g*a` to `b` |
| `div` | `a / b` | `g/b` to `a`, `-g*a/b^2` to `b` |
| `neg` | `-a` | `-g` |
| `matmul` | `a @ b` | `g @ b^T` to `a`, `a^T @ g` to `b` |
| `sum` | reduce-sum | broadcast `g` back to input shape |
| `mean` | reduce-mean | broadcast `g/N` back to input shape |
| `max`/`min` | reduce-extreme | gradient split evenly across tied maxima |
| `exp` | `e^x` | `g * e^x` |
| `log` | `ln(x)` | `g / x` |
| `sqrt` | `sqrt(x)` | `g * 0.5 / sqrt(x)` |
| `relu` | `max(x, 0)` | `g` where `x>0` else `0` |
| `sigmoid` | `1/(1+e^-x)` | `g * y * (1-y)` |
| `tanh` | `tanh(x)` | `g * (1 - y^2)` |
| `softmax` | row-wise softmax | Jacobian-vector product `y*(g - sum(g*y))` |
| `log_softmax` | stable `log(softmax(x))` via max-shift | composed from the verified primitive rules (v0.2.0) |
| `clip` | clamp to `[min, max]` | `g` where strictly inside, else `0` (v0.2.0) |
| `argmax` | index of max (i64) | NOT differentiable — no gradient flows (v0.2.0) |
| `one_hot` | class-index → one-hot (f32) | NOT differentiable — constant lookup (v0.2.0) |
| `reshape` | reshape | reshape `g` back to input shape |
| `transpose`/`permute` | axis permutation | inverse permutation of `g` |
| `broadcast_to` | explicit broadcast | sum-reduce `g` back to input shape |
| `__getitem__` | basic slicing | scatter `g` into a zero tensor at the same index |

## Error messages

Shape errors follow the master prompt §27 format exactly — see
`TensorError` raises in `timet/tensor.py`, e.g. matmul:

```
TypeError [E0403] (tensor): matrix multiplication requires compatible inner dimensions
  expected: left[-1] == right[-2]
  actual:   left=[32, 128], right=[64, 128]
```

## What's not implemented

Non-contiguous/strided views, explicit device transfer (`device="cpu"` is
the only value), lazy/deferred execution (everything is eager), quantized
dtypes, sparse tensors, complex numbers, advanced (fancy/boolean-mask)
indexing beyond basic slicing.
