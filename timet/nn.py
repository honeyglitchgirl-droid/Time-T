"""Neural-network layer library (master prompt sections 8, 22).

Implemented (all covered by tests in tests/test_nn.py, tests/test_layernorm.py, tests/test_transformer.py):
  Layers:    Linear, ReLU, Sigmoid, Tanh, GELU, Softmax, Flatten, Conv2D, Embedding,
             LayerNorm, MultiheadAttention, TransformerBlock, Dropout, Sequential
  Losses:    MSELoss, CrossEntropyLoss, BCELoss
  Plumbing:  Module.parameters(), Module.train()/eval() (recursive)

Still NOT implemented (see docs/ROADMAP.md, Milestone 7):
  Conv1D, BatchNorm,
  weight-initialization schemes beyond Kaiming-uniform.

"""
from __future__ import annotations

import math
from typing import List, Sequence, Union

import numpy as np

from timet.tensor import Tensor, tensor, one_hot
from timet.diagnostics import Diagnostic


class NNError(Diagnostic):
    pass


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        return []

    def children(self) -> List["Module"]:
        """Immediate sub-modules. Overridden by containers (Sequential)."""
        return []

    def train(self, mode: bool = True) -> "Module":
        """Set training mode recursively (affects Dropout). Returns self."""
        self.training = bool(mode)
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError


class Linear(Module):
    """y = x @ W^T + b, Kaiming-uniform initialized (seeded, deterministic)."""

    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        bound = 1.0 / math.sqrt(in_features)
        w = rng.uniform(-bound, bound, size=(out_features, in_features)).astype(np.float32)
        b = rng.uniform(-bound, bound, size=(out_features,)).astype(np.float32)
        self.weight = Tensor(w, requires_grad=True)
        self.bias = Tensor(b, requires_grad=True)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        return x.matmul(self.weight.transpose()) + self.bias

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class GELU(Module):
    """Gaussian Error Linear Unit (Hendrycks & Gimpel 2016).

    Uses the standard fast approximation:
        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    def forward(self, x: Tensor) -> Tensor:
        return gelu(x)


class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)


class Flatten(Module):
    """Flattens all dimensions except the leading (batch) dimension."""

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 1:
            raise NNError(code="E0610", message="Flatten: expected at least 1 dimension",
                          stage="nn")
        return x.reshape((x.shape[0], -1)) if x.ndim > 1 else x


class Conv2D(Module):
    """2-D cross-correlation layer (PyTorch-style; NOT true convolution),
    over NCHW tensors: input (N, C_in, H, W) -> output (N, C_out, H', W')
    with H' = (H + 2*padding - KH) // stride + 1 (and same for W').

    Implementation: explicit im2col + matmul with a hand-written col2im
    backward. Naive loops -- clarity over speed at the sizes this
    interpreter targets; benchmarked honesty: no speed claim is made for
    conv workloads anywhere in the docs (docs/ROADMAP.md Milestone 7).

    Gradients: checked against central finite differences from ALL THREE
    inputs (x, weight, bias) in tests/test_conv2d.py -- including strided
    and padded configurations, where col2im bugs most often hide.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride: int = 1, padding: int = 0, seed: int = 0):
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        if not (kh >= 1 and kw >= 1):
            raise NNError(code="E0611", message="Conv2D: kernel size must be >= 1",
                          stage="nn")
        if stride < 1 or padding < 0:
            raise NNError(code="E0611", message="Conv2D: stride >= 1 and padding >= 0 required",
                          stage="nn")
        rng = np.random.default_rng(seed)
        fan_in = in_channels * kh * kw
        bound = 1.0 / math.sqrt(fan_in)
        w = rng.uniform(-bound, bound, size=(out_channels, in_channels, kh, kw)).astype(np.float32)
        b = np.zeros(out_channels, dtype=np.float32)
        self.weight = Tensor(w, requires_grad=True)
        self.bias = Tensor(b, requires_grad=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, self.bias,
                      stride=self.stride, padding=self.padding)

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]


def _conv2d_forward(xd, wd, bd, sh, sw, ph, pw):
    """Returns (out, cols, padded_shape). Input must be ndim-4 NCHW."""
    N, C, H, W = xd.shape
    F, Cw, KH, KW = wd.shape
    assert C == Cw
    H_out = (H + 2 * ph - KH) // sh + 1
    W_out = (W + 2 * pw - KW) // sw + 1
    xp = np.pad(xd, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    L = H_out * W_out
    cols = np.zeros((N, C * KH * KW, L), dtype=xd.dtype)
    for ky in range(KH):
        for kx in range(KW):
            # patch: (N, C, H_out, W_out). NOTE the index below is
            # CHANNEL-major so it lines up with wd.reshape(F, C*KH*KW)
            # (whose row layout is (c, ky, kx) in C-order).
            patch = xp[:, :, ky:ky + sh * H_out:sh, kx:kx + sw * W_out:sw]
            i0 = ky * KW + kx
            for c in range(C):
                cols[:, c * KH * KW + i0, :] = patch[:, c].reshape(N, L)
    w2d = wd.reshape(F, C * KH * KW)
    out = np.einsum("fc,ncl->nfl", w2d, cols)
    if bd is not None:
        out = out + bd.reshape(1, F, 1)
    return out.reshape(N, F, H_out, W_out), cols, xp.shape


def _conv2d_backward(g, cols, wd, xp_shape, C, H, W, sh, sw, ph, pw):
    """g: (N, F, H_out, W_out). Returns (dx, dw, db) matching parents."""
    N, F = g.shape[0], g.shape[1]
    KH = wd.shape[2]
    KW = wd.shape[3]
    H_out, W_out = g.shape[2], g.shape[3]
    L = H_out * W_out
    g2 = g.reshape(N, F, L)
    w2d = wd.reshape(F, C * KH * KW)
    # dw
    dw2d = np.einsum("nfl,ncl->fc", g2, cols)
    dw = dw2d.reshape(wd.shape)
    # db
    db = g.sum(axis=(0, 2, 3))
    # dx via col2im
    dcols = np.einsum("fc,nfl->ncl", w2d, g2)
    dxp = np.zeros(xp_shape, dtype=g.dtype)
    N_, C_ = dcols.shape[0], C
    for ky in range(KH):
        for kx in range(KW):
            i0 = ky * KW + kx
            for c in range(C):
                patch_grad = dcols[:, c * KH * KW + i0, :].reshape(N_, H_out, W_out)
                for h in range(H_out):
                    for w_ in range(W_out):
                        dxp[:, c, ky + h * sh, kx + w_ * sw] += patch_grad[:, h, w_]
    dx = dxp[:, :, ph:ph + H, pw:pw + W]
    return dx, dw, db


def conv2d(x: Tensor, weight: Tensor, bias: Tensor = None,
           stride: int = 1, padding: int = 0) -> Tensor:
    """Functional 2-D cross-correlation, tape-aware (see Conv2D docstring)."""
    if x.ndim != 4:
        raise NNError(code="E0611", message=f"conv2d: expected 4-D NCHW input, got shape {x.shape}",
                      stage="nn")
    if weight.ndim != 4:
        raise NNError(code="E0611", message=f"conv2d: expected 4-D (F, C, KH, KW) weight, got {weight.shape}",
                      stage="nn")
    if x.shape[1] != weight.shape[1]:
        raise NNError(code="E0611",
                      message=f"conv2d: channel mismatch: input C={x.shape[1]} vs weight C={weight.shape[1]}",
                      stage="nn")
    sh = sw = stride
    ph = pw = padding
    N, C, H, W = x.data.shape
    KH, KW = weight.data.shape[2], weight.data.shape[3]
    if (H + 2 * ph - KH) < 0 or (W + 2 * pw - KW) < 0:
        raise NNError(code="E0611",
                      message=f"conv2d: kernel {KH}x{KW} (padding {padding}) exceeds input {H}x{W}",
                      stage="nn")
    out, cols, xp_shape = _conv2d_forward(x.data, weight.data,
                                          bias.data if bias is not None else None,
                                          sh, sw, ph, pw)
    parents = [x, weight] + ([bias] if bias is not None else [])

    def backward_fn(g):
        dx, dw, db = _conv2d_backward(g, cols, weight.data, xp_shape, C, H, W,
                                      sh, sw, ph, pw)
        return (dx, dw, db) if bias is not None else (dx, dw)

    return x._make_result(out, parents, backward_fn, "conv2d")


class Embedding(Module):
    """Lookup table: rows of `weight`, gathered by integer indices.

    forward(indices) accepts any array-like of ints (arbitrary shape);
    output shape = indices.shape + (embedding_dim,). The gradient of the
    lookup is a scatter-add over the addressed rows (handled at full
    precision, then cast to the weight dtype, like every other op here).

    Gradient of the weight is finite-difference checked, incl. repeated
    indices (duplicate rows accumulate, gradient is NOT a copy-overwrite).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, seed: int = 0):
        super().__init__()
        if num_embeddings < 1 or embedding_dim < 1:
            raise NNError(code="E0612", message="Embedding: sizes must be >= 1", stage="nn")
        rng = np.random.default_rng(seed)
        w = rng.normal(0.0, 1.0, size=(num_embeddings, embedding_dim)).astype(np.float32)
        self.weight = Tensor(w, requires_grad=True)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, indices) -> Tensor:
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            raise NNError(code="E0612", message="Embedding: empty indices", stage="nn")
        if idx.min() < 0 or idx.max() >= self.num_embeddings:
            raise NNError(code="E0612",
                          message=f"Embedding: index out of range [0, {self.num_embeddings})",
                          stage="nn")
        out = self.weight.data[idx]
        w = self.weight
        n_rows, dim = w.data.shape

        def backward_fn(g):
            gw = np.zeros((n_rows, dim), dtype=np.float64)
            np.add.at(gw, idx.reshape(-1), g.reshape(-1, dim))
            return (gw.astype(w.data.dtype),)

        return w._make_result(out, [w], backward_fn, "embedding")

    def parameters(self) -> List[Tensor]:
        return [self.weight]


class LayerNorm(Module):
    """Layer Normalization (Ba, Kiros, Hinton 2016).

    Applies Layer Normalization over the last D dimensions defined by
    `normalized_shape`:
        y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

    `normalized_shape`: int or Sequence[int].
    `eps`: small value added to the denominator for numerical stability.
    `elementwise_affine`: if True, learnable scale (weight) initialized to 1
                          and bias initialized to 0 are applied.
    """

    def __init__(self, normalized_shape: Union[int, Sequence[int]], eps: float = 1e-5,
                 elementwise_affine: bool = True, seed: int = 0):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        if any(d <= 0 for d in self.normalized_shape):
            raise NNError(code="E0614", message="LayerNorm: dimensions in normalized_shape must be > 0",
                          stage="nn")
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            w = np.ones(self.normalized_shape, dtype=np.float32)
            b = np.zeros(self.normalized_shape, dtype=np.float32)
            self.weight = Tensor(w, requires_grad=True)
            self.bias = Tensor(b, requires_grad=True)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


    def parameters(self) -> List[Tensor]:
        if self.elementwise_affine:
            return [self.weight, self.bias]
        return []


class MultiheadAttention(Module):
    """Multi-Head Attention (Vaswani et al. 2017).

    Projects queries, keys, and values into `num_heads` subspace projections
    of dimension `head_dim = embed_dim // num_heads`, computes scaled dot-product
    attention, and projects the concatenated outputs back to `embed_dim`.

    Input shapes:
        query: (B, Sq, E)
        key:   (B, Sk, E) (defaults to query for self-attention)
        value: (B, Sk, E) (defaults to key)
        mask:  optional additive attention mask broadcastable to (B, H, Sq, Sk)
    Output shape:
        (B, Sq, E)
    """

    def __init__(self, embed_dim: int, num_heads: int, seed: int = 0):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise NNError(code="E0615",
                          message=f"MultiheadAttention: embed_dim and num_heads must be > 0",
                          stage="nn")
        if embed_dim % num_heads != 0:
            raise NNError(
                code="E0615",
                message=f"MultiheadAttention: embed_dim {embed_dim} must be divisible by num_heads {num_heads}",
                stage="nn"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim, seed=seed)
        self.k_proj = Linear(embed_dim, embed_dim, seed=seed + 1)
        self.v_proj = Linear(embed_dim, embed_dim, seed=seed + 2)
        self.out_proj = Linear(embed_dim, embed_dim, seed=seed + 3)

    def forward(self, query: Tensor, key: Optional[Tensor] = None,
                value: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        if query.ndim != 3:
            raise NNError(
                code="E0615",
                message=f"MultiheadAttention: query must be 3-D (B, S, E), got {query.shape}",
                stage="nn"
            )
        if key is None:
            key = query
        if value is None:
            value = key

        if key.ndim != 3 or value.ndim != 3:
            raise NNError(
                code="E0615",
                message="MultiheadAttention: key and value must be 3-D tensors",
                stage="nn"
            )
        if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
            raise NNError(
                code="E0615",
                message="MultiheadAttention: batch size mismatch among query, key, value",
                stage="nn"
            )
        if key.shape[1] != value.shape[1]:
            raise NNError(
                code="E0615",
                message="MultiheadAttention: key and value sequence length mismatch",
                stage="nn"
            )
        if query.shape[2] != self.embed_dim or key.shape[2] != self.embed_dim or value.shape[2] != self.embed_dim:
            raise NNError(
                code="E0615",
                message=f"MultiheadAttention: expected embed_dim {self.embed_dim}",
                stage="nn"
            )

        B, Sq, E = query.shape
        Sk = key.shape[1]
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(query).reshape((B, Sq, H, D)).transpose([0, 2, 1, 3])  # (B, H, Sq, D)
        k = self.k_proj(key).reshape((B, Sk, H, D)).transpose([0, 2, 1, 3])    # (B, H, Sk, D)
        v = self.v_proj(value).reshape((B, Sk, H, D)).transpose([0, 2, 1, 3])  # (B, H, Sk, D)

        scale = float(1.0 / math.sqrt(D))
        scores = (q.matmul(k.transpose([0, 1, 3, 2]))) * scale  # (B, H, Sq, Sk)
        if mask is not None:
            scores = scores + mask
        attn = scores.softmax(axis=-1)
        out = attn.matmul(v)  # (B, H, Sq, D)
        out = out.transpose([0, 2, 1, 3]).reshape((B, Sq, E))
        return self.out_proj(out)

    def parameters(self) -> List[Tensor]:
        return (self.q_proj.parameters() + self.k_proj.parameters() +
                self.v_proj.parameters() + self.out_proj.parameters())

    def children(self) -> List[Module]:
        return [self.q_proj, self.k_proj, self.v_proj, self.out_proj]


class TransformerBlock(Module):
    """Pre-LN Transformer Encoder Block (Vaswani et al. 2017).

    Structure:
        x = x + attn(norm1(x), mask=mask)
        x = x + mlp(norm2(x))
    where mlp is Linear -> GELU -> Linear.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, seed: int = 0):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads, seed=seed)
        self.norm2 = LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Sequential(
            Linear(embed_dim, hidden_dim, seed=seed + 10),
            GELU(),
            Linear(hidden_dim, embed_dim, seed=seed + 20),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x

    def parameters(self) -> List[Tensor]:
        return (self.norm1.parameters() + self.attn.parameters() +
                self.norm2.parameters() + self.mlp.parameters())

    def children(self) -> List[Module]:
        return [self.norm1, self.attn, self.norm2, self.mlp]


class Dropout(Module):
    """Inverted dropout: scales surviving activations by 1/(1-p) at train
    time, identity at eval time. Seeded => deterministic, so tests and
    examples are reproducible."""

    def __init__(self, p: float = 0.5, seed: int = 0):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise NNError(code="E0611", message=f"Dropout: p must be in [0, 1), got {p}",
                          stage="nn")
        self.p = float(p)
        self._rng = np.random.default_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (self._rng.random(x.shape) < keep).astype(np.float32) / keep
        return x * Tensor(mask)


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def children(self) -> List[Module]:
        return list(self.layers)


class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return (diff * diff).mean()


class CrossEntropyLoss(Module):
    """Mean negative log-likelihood over a batch of logits.

    pred:    Tensor of shape (N, C) -- raw logits (NOT probabilities).
    targets: length-N integer class labels (python list, NumPy array, or
             integer Tensor).

    Computed as:  mean( -sum( one_hot(target) * log_softmax(logits), axis=1 ) )
    Gradient flows through log_softmax's verified primitive-op backward
    rules; one_hot is a constant lookup. Finite-difference checked in
    tests/test_nn.py.
    """

    def forward(self, pred: Tensor, targets) -> Tensor:
        if pred.ndim != 2:
            raise NNError(
                code="E0612",
                message=f"CrossEntropyLoss: expected logits of shape (N, C), got {pred.shape}",
                stage="nn",
            )
        n, c = pred.shape
        targets_t = targets if isinstance(targets, Tensor) else tensor(list(targets))
        if targets_t.data.shape != (n,):
            raise NNError(
                code="E0613",
                message=f"CrossEntropyLoss: expected {n} target label(s), got shape {targets_t.data.shape}",
                stage="nn",
            )
        oh = one_hot(targets_t, c)
        logp = pred.log_softmax(axis=1)
        per_sample = -(oh * logp).sum(axis=1)
        return per_sample.mean()


class BCELoss(Module):
    """Binary cross-entropy on probabilities in (0, 1).

    pred:   predicted probabilities (typically the output of a sigmoid), any shape.
    target: same-shaped 0/1 targets.

    Computed entirely from autograd-tracked primitive ops:
        BCE = mean( -(t * log(pc) + (1 - t) * log(1 - pc)) ),
        pc = clip(pred, eps, 1 - eps)     (numerical stability)
    so gradients reuse clip/log/mean's finite-difference-checked backward
    rules. See docs/AUTOGRAD.md.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pc = pred.clip(self.eps, 1.0 - self.eps)
        one = tensor(1.0)
        return -(target * pc.log() + (one - target) * (one - pc).log()).mean()


# ---------------- functional conveniences ----------------
# Thin wrappers so call sites (including Time-T programs) can write
# nn.mse_loss(p, y) instead of nn.MSELoss()(p, y).

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return MSELoss()(pred, target)


def cross_entropy_loss(logits: Tensor, targets) -> Tensor:
    return CrossEntropyLoss()(logits, targets)


def binary_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    return BCELoss()(pred, target)


def gelu(x: Tensor) -> Tensor:
    """Gaussian Error Linear Unit functional op."""
    c = float(math.sqrt(2.0 / math.pi))
    inner = (x + (x * x * x) * 0.044715) * c
    return (x * 0.5) * (inner.tanh() + 1.0)


def layer_norm(x: Tensor, normalized_shape: Union[int, Sequence[int]],
               weight: Tensor = None, bias: Tensor = None, eps: float = 1e-5) -> Tensor:
    """Functional Layer Normalization over the trailing dimensions."""
    if isinstance(normalized_shape, int):
        norm_shape = (normalized_shape,)
    else:
        norm_shape = tuple(normalized_shape)
    norm_ndim = len(norm_shape)
    if x.ndim < norm_ndim or tuple(x.shape[-norm_ndim:]) != norm_shape:
        raise NNError(
            code="E0614",
            message=f"layer_norm: input tail shape {tuple(x.shape[-norm_ndim:])} "
                    f"does not match normalized_shape {norm_shape}",
            stage="nn"
        )
    axes = tuple(range(x.ndim - norm_ndim, x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    diff = x - mean
    var = (diff * diff).mean(axis=axes, keepdims=True)
    std = (var + eps).sqrt()
    x_hat = diff / std
    if weight is not None:
        x_hat = x_hat * weight
    if bias is not None:
        x_hat = x_hat + bias
    return x_hat

