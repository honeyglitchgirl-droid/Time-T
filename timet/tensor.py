"""Time-T Tensor: NumPy-backed dense array with reverse-mode autodiff.

See docs/ARCHITECTURE.md section 4-5 and docs/DESIGN_DECISIONS.md DD-1, DD-4.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from timet import autodiff, memory
from timet.backend import get_default_backend
from timet.diagnostics import Diagnostic, SourceSpan

_DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "bool": np.bool_,
}


class TensorError(Diagnostic):
    pass


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Sum-reduce `grad` back down to `target_shape` after a broadcasted op."""
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(target_shape)


class Tensor:
    __slots__ = ("data", "requires_grad", "grad", "_node", "device", "_freed")

    def __init__(self, data, dtype: Optional[str] = None, requires_grad: bool = False,
                 device: str = "cpu", _node: Optional[autodiff.Node] = None):
        if isinstance(data, Tensor):
            arr = data.data
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            arr = np.array(data, dtype=_DTYPE_MAP[dtype] if dtype else None)
        if dtype is not None:
            arr = arr.astype(_DTYPE_MAP[dtype])
        elif arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype == np.int64 and arr.size and not isinstance(data, np.ndarray):
            pass  # keep native int64 for plain python int lists
        self.data = arr
        self.requires_grad = requires_grad and autodiff.is_grad_enabled()
        self.grad: Optional["Tensor"] = None
        self._node = _node
        self.device = device
        self._freed = False
        memory.record_alloc(self.nbytes)

    def __del__(self):
        if not self._freed:
            try:
                memory.record_free(self.nbytes)
            except Exception:
                pass
            self._freed = True

    # ---------------- basic properties ----------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> str:
        for name, np_ty in _DTYPE_MAP.items():
            if self.data.dtype == np_ty:
                return name
        return str(self.data.dtype)

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        grad_str = f", grad_fn={self._node.name}" if self._node else ""
        req = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data.tolist()}{req}{grad_str})"

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return bool(np.array_equal(self.data, other.data))
        return NotImplemented

    # ---------------- autodiff plumbing ----------------

    def _make_result(self, arr: np.ndarray, parents: Sequence["Tensor"],
                      backward_fn, name: str) -> "Tensor":
        needs_grad = autodiff.is_grad_enabled() and any(p.requires_grad for p in parents)
        node = None
        if needs_grad:
            node = autodiff.Node(inputs=list(parents), backward_fn=backward_fn,
                                  output_ref=None, name=name)
        out = Tensor(arr, requires_grad=needs_grad, _node=node)
        if node is not None:
            node.output_ref = out
        return out

    def backward(self, grad: Optional["Tensor"] = None):
        if not self.requires_grad and self._node is None:
            raise TensorError(
                code="E0400",
                message="called backward() on a tensor that does not require grad",
                stage="autodiff",
                note="create the tensor with grad=true (tensor(..., grad=true)) to enable gradients",
            )
        if grad is None:
            if self.data.size != 1:
                raise TensorError(
                    code="E0401",
                    message="backward() without an explicit gradient requires a scalar output",
                    stage="autodiff",
                    expected="shape ()",
                    actual=f"shape {self.shape}",
                )
            grad = Tensor(np.ones_like(self.data, dtype=np.float32))

        grads = {id(self): grad.data.astype(np.float32)}
        order = autodiff.topological_order(self)
        for t in reversed(order):
            if id(t) not in grads:
                continue
            g = grads[id(t)]
            if t.requires_grad and t._node is None:
                # leaf tensor: accumulate
                if t.grad is None:
                    t.grad = Tensor(np.zeros_like(t.data, dtype=np.float32))
                t.grad.data = t.grad.data + g
            node = t._node
            if node is None:
                continue
            input_grads = node.backward_fn(g)
            for parent, pg in zip(node.inputs, input_grads):
                if pg is None or not parent.requires_grad:
                    continue
                pg = _unbroadcast(np.asarray(pg, dtype=np.float32), parent.shape)
                if id(parent) in grads:
                    grads[id(parent)] = grads[id(parent)] + pg
                else:
                    grads[id(parent)] = pg

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self):
        self.grad = None

    def sub_(self, other) -> "Tensor":
        """In-place subtract, used for parameter updates. Does not touch the
        autodiff tape (mirrors typical optimizer-internal parameter update
        semantics) and preserves this tensor's identity/requires_grad."""
        other_arr = other.data if isinstance(other, Tensor) else np.asarray(other)
        self.data = (self.data - other_arr).astype(self.data.dtype)
        return self

    def add_(self, other) -> "Tensor":
        other_arr = other.data if isinstance(other, Tensor) else np.asarray(other)
        self.data = (self.data + other_arr).astype(self.data.dtype)
        return self

    # ---------------- elementwise ops ----------------

    def _binop(self, other, fwd, back_a, back_b, name) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        try:
            result = fwd(self.data, other_t.data)
        except ValueError as e:
            raise TensorError(
                code="E0402",
                message=f"shape mismatch in '{name}'",
                stage="tensor",
                expected="broadcastable shapes",
                actual=f"{self.shape} vs {other_t.shape}",
                note=str(e),
            )

        def backward_fn(g):
            ga = back_a(g, self.data, other_t.data) if back_a else None
            gb = back_b(g, self.data, other_t.data) if back_b else None
            return (ga, gb)

        return self._make_result(result, [self, other_t], backward_fn, name)

    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b,
                            lambda g, a, b: g, lambda g, a, b: g, "add")

    __radd__ = __add__

    def __sub__(self, other):
        return self._binop(other, lambda a, b: a - b,
                            lambda g, a, b: g, lambda g, a, b: -g, "sub")

    def __rsub__(self, other):
        return Tensor(other)._binop(self, lambda a, b: a - b,
                                     lambda g, a, b: g, lambda g, a, b: -g, "sub")

    def __mul__(self, other):
        return self._binop(other, lambda a, b: a * b,
                            lambda g, a, b: g * b, lambda g, a, b: g * a, "mul")

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binop(other, lambda a, b: a / b,
                            lambda g, a, b: g / b,
                            lambda g, a, b: -g * a / (b * b), "div")

    def __rtruediv__(self, other):
        return Tensor(other)._binop(self, lambda a, b: a / b,
                                     lambda g, a, b: g / b,
                                     lambda g, a, b: -g * a / (b * b), "div")

    def __neg__(self):
        def backward_fn(g):
            return (-g,)
        return self._make_result(-self.data, [self], backward_fn, "neg")

    def __matmul__(self, other):
        return self.matmul(other)

    def matmul(self, other) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.shape[-1] != other_t.data.shape[-2]:
            raise TensorError(
                code="E0403",
                message="matrix multiplication requires compatible inner dimensions",
                stage="tensor",
                expected="left[-1] == right[-2]",
                actual=f"left={list(self.shape)}, right={list(other_t.shape)}",
            )
        backend = get_default_backend()
        result = backend.matmul(self.data, other_t.data)

        def backward_fn(g):
            ga = g @ np.swapaxes(other_t.data, -1, -2)
            gb = np.swapaxes(self.data, -1, -2) @ g
            return (ga, gb)

        return self._make_result(result, [self, other_t], backward_fn, "matmul")

    # ---------------- unary math ----------------

    def _unop(self, fwd, back, name) -> "Tensor":
        result = fwd(self.data)

        def backward_fn(g):
            return (back(g, self.data, result),)

        return self._make_result(result, [self], backward_fn, name)

    def exp(self):
        return self._unop(np.exp, lambda g, x, y: g * y, "exp")

    def log(self):
        return self._unop(np.log, lambda g, x, y: g / x, "log")

    def sqrt(self):
        return self._unop(np.sqrt, lambda g, x, y: g * 0.5 / y, "sqrt")

    def relu(self):
        return self._unop(lambda x: np.maximum(x, 0),
                           lambda g, x, y: g * (x > 0), "relu")

    def sigmoid(self):
        def fwd(x):
            return 1.0 / (1.0 + np.exp(-x))
        return self._unop(fwd, lambda g, x, y: g * y * (1 - y), "sigmoid")

    def tanh(self):
        return self._unop(np.tanh, lambda g, x, y: g * (1 - y * y), "tanh")

    def clip(self, min_val: float, max_val: float) -> "Tensor":
        """Elementwise clamp to [min_val, max_val].

        Backward rule: gradient passes through where data is strictly inside
        the interval and is zero outside/at the boundary (same convention as
        PyTorch's clamp). Finite-difference tested away from boundaries.
        """
        def fwd(x):
            return np.clip(x, min_val, max_val)

        return self._unop(
            fwd,
            lambda g, x, y: g * ((x > min_val) & (x < max_val)).astype(np.float32),
            "clip",
        )

    def softmax(self, axis: int = -1):
        def fwd(x):
            shifted = x - np.max(x, axis=axis, keepdims=True)
            e = np.exp(shifted)
            return e / np.sum(e, axis=axis, keepdims=True)

        y = fwd(self.data)

        def backward_fn(g):
            s = y
            dot = np.sum(g * s, axis=axis, keepdims=True)
            return (s * (g - dot),)

        return self._make_result(y, [self], backward_fn, "softmax")

    # ---------------- reductions ----------------

    def log_softmax(self, axis: int = -1) -> "Tensor":
        """Numerically stable log-softmax, composed from primitive ops so the
        backward pass comes for free from their verified gradient rules:
            log_softmax(x) = (x - max(x)) - log(sum(exp(x - max(x))))
        See docs/TENSOR.md.
        """
        m = self.max(axis=axis, keepdims=True)
        shifted = self - m
        lse = shifted.exp().sum(axis=axis, keepdims=True).log()
        return shifted - lse

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        result = self.data.sum(axis=axis, keepdims=keepdims)
        in_shape = self.shape

        def backward_fn(g):
            g = np.asarray(g, dtype=np.float32)
            if axis is None:
                return (np.ones(in_shape, dtype=np.float32) * g,)
            if not keepdims:
                g = np.expand_dims(g, axis=axis if isinstance(axis, int) else tuple(axis))
            return (np.broadcast_to(g, in_shape).astype(np.float32).copy(),)

        return self._make_result(np.asarray(result), [self], backward_fn, "sum")

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        result = self.data.mean(axis=axis, keepdims=keepdims)
        in_shape = self.shape
        if axis is None:
            count = self.data.size
        elif isinstance(axis, int):
            count = self.data.shape[axis]
        else:
            count = 1
            for a in axis:
                count *= self.data.shape[a]

        def backward_fn(g):
            g = np.asarray(g, dtype=np.float32) / count
            if axis is None:
                return (np.ones(in_shape, dtype=np.float32) * g,)
            if not keepdims:
                g = np.expand_dims(g, axis=axis if isinstance(axis, int) else tuple(axis))
            return (np.broadcast_to(g, in_shape).astype(np.float32).copy(),)

        return self._make_result(np.asarray(result), [self], backward_fn, "mean")

    def argmax(self, axis: Optional[int] = None) -> "Tensor":
        """Index of the maximum value. Not differentiable (returns int indices).

        Gradient does not flow through argmax -- this mirrors PyTorch, and is
        documented in docs/AUTOGRAD.md.
        """
        idx = np.argmax(self.data, axis=axis)
        return Tensor(idx, dtype="i64")

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        return self._reduce_extreme(np.max, axis, keepdims, "max")

    def min(self, axis=None, keepdims: bool = False) -> "Tensor":
        return self._reduce_extreme(np.min, axis, keepdims, "min")

    def _reduce_extreme(self, fn, axis, keepdims, name) -> "Tensor":
        result = fn(self.data, axis=axis, keepdims=True)
        in_shape = self.shape
        mask = (self.data == result).astype(np.float32)
        counts = mask.sum(axis=axis, keepdims=True)
        out = result if keepdims else np.squeeze(result, axis=axis) if axis is not None else result.reshape(())

        def backward_fn(g):
            g = np.asarray(g, dtype=np.float32)
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis=axis if isinstance(axis, int) else tuple(axis))
            elif not keepdims and axis is None:
                g = g.reshape((1,) * len(in_shape))
            return ((mask / np.maximum(counts, 1)) * g,)

        return self._make_result(np.asarray(out), [self], backward_fn, name)

    # ---------------- shape ops ----------------

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        target = tuple(shape)
        try:
            result = self.data.reshape(target)
        except ValueError as e:
            raise TensorError(
                code="E0404",
                message="cannot reshape tensor",
                stage="tensor",
                expected=f"a shape compatible with {self.data.size} elements",
                actual=str(target),
                note=str(e),
            )
        in_shape = self.shape

        def backward_fn(g):
            return (np.asarray(g).reshape(in_shape),)

        return self._make_result(result, [self], backward_fn, "reshape")

    def transpose(self, axes: Optional[Sequence[int]] = None) -> "Tensor":
        result = np.transpose(self.data, axes)

        def backward_fn(g):
            if axes is None:
                return (np.transpose(g),)
            inv = np.argsort(axes)
            return (np.transpose(g, inv),)

        return self._make_result(result, [self], backward_fn, "transpose")

    def permute(self, axes: Sequence[int]) -> "Tensor":
        return self.transpose(axes)

    def broadcast_to(self, shape: Sequence[int]) -> "Tensor":
        target = tuple(shape)
        result = np.broadcast_to(self.data, target).copy()
        in_shape = self.shape

        def backward_fn(g):
            return (_unbroadcast(np.asarray(g, dtype=np.float32), in_shape),)

        return self._make_result(result, [self], backward_fn, "broadcast_to")

    def __getitem__(self, key):
        result = self.data[key]
        in_shape = self.shape

        def backward_fn(g):
            full = np.zeros(in_shape, dtype=np.float32)
            full[key] = g
            return (full,)

        return self._make_result(np.asarray(result), [self], backward_fn, "index")

    # ---------------- comparisons (non-differentiable) ----------------

    def __gt__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data > other_t.data, dtype="bool")

    def __lt__(self, other):
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(self.data < other_t.data, dtype="bool")


def tensor(data, dtype: Optional[str] = None, grad: bool = False) -> Tensor:
    return Tensor(data, dtype=dtype, requires_grad=grad)


def zeros(shape: Sequence[int], dtype: str = "f32", grad: bool = False) -> Tensor:
    return Tensor(np.zeros(tuple(shape)), dtype=dtype, requires_grad=grad)


def ones(shape: Sequence[int], dtype: str = "f32", grad: bool = False) -> Tensor:
    return Tensor(np.ones(tuple(shape)), dtype=dtype, requires_grad=grad)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a.matmul(b)


def sum(t: Tensor, axis=None, keepdims=False) -> Tensor:
    return t.sum(axis=axis, keepdims=keepdims)


def mean(t: Tensor, axis=None, keepdims=False) -> Tensor:
    return t.mean(axis=axis, keepdims=keepdims)


def relu(t: Tensor) -> Tensor:
    return t.relu()


def sigmoid(t: Tensor) -> Tensor:
    return t.sigmoid()


def tanh(t: Tensor) -> Tensor:
    return t.tanh()


def softmax(t: Tensor, axis: int = -1) -> Tensor:
    return t.softmax(axis=axis)


def log_softmax(t: Tensor, axis: int = -1) -> Tensor:
    return t.log_softmax(axis=axis)


def argmax(t: Tensor, axis: Optional[int] = None) -> Tensor:
    return t.argmax(axis=axis)


def one_hot(indices, num_classes: int) -> Tensor:
    """One-hot encode class indices. Non-differentiable lookup (dtype f32).

    `indices` may be a python list of ints, a NumPy array, or an integer
    Tensor. Result shape: indices.shape + (num_classes,).
    """
    if isinstance(indices, Tensor):
        idx = indices.data.astype(np.int64)
    else:
        idx = np.asarray(indices, dtype=np.int64)
    if num_classes <= 0:
        raise TensorError(
            code="E0600",
            message=f"one_hot: num_classes must be positive, got {num_classes}",
            stage="tensor",
        )
    if idx.size and (idx.min() < 0 or idx.max() >= num_classes):
        raise TensorError(
            code="E0601",
            message=f"one_hot: class index out of range [0, {num_classes})",
            stage="tensor",
        )
    out = np.zeros(idx.shape + (num_classes,), dtype=np.float32)
    if idx.size:
        np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
    return Tensor(out)


def exp(t: Tensor) -> Tensor:
    return t.exp()


def log(t: Tensor) -> Tensor:
    return t.log()


def sqrt(t: Tensor) -> Tensor:
    return t.sqrt()
