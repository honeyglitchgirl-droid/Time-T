import math
import numpy as np
import pytest

from timet.tensor import Tensor, tensor
from timet.nn import MultiheadAttention, TransformerBlock, GELU, gelu, NNError
from tests.numerics import assert_close, finite_difference_grad


def test_gelu_forward_and_backward():
    c = float(math.sqrt(2.0 / math.pi))
    def gelu_ref(x):
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * (x ** 3))))

    x_np = np.linspace(-3.0, 3.0, 15).astype(np.float64)
    expected = gelu_ref(x_np)

    xt = Tensor(x_np.astype(np.float32), requires_grad=True)
    y = GELU()(xt)
    assert_close(y.data.astype(np.float64), expected, rtol=1e-4, atol=1e-4)

    # Numerical gradient check
    fd = finite_difference_grad(lambda a: float(gelu_ref(a).sum()), x_np)
    y.sum().backward()
    assert_close(xt.grad.data.astype(np.float64), fd, rtol=1e-3, atol=1e-3, msg="GELU gradient")


def test_mha_forward_shapes():
    mha = MultiheadAttention(embed_dim=16, num_heads=2, seed=1)
    q = Tensor(np.random.randn(3, 5, 16).astype(np.float32))
    out = mha(q)
    assert out.shape == (3, 5, 16)


def test_mha_cross_attention_shapes():
    mha = MultiheadAttention(embed_dim=12, num_heads=3, seed=2)
    q = Tensor(np.random.randn(2, 4, 12).astype(np.float32))
    k = Tensor(np.random.randn(2, 7, 12).astype(np.float32))
    v = Tensor(np.random.randn(2, 7, 12).astype(np.float32))
    out = mha(q, k, v)
    assert out.shape == (2, 4, 12)


def test_mha_masking():
    # Mask out some keys
    mha = MultiheadAttention(embed_dim=8, num_heads=2, seed=3)
    q = Tensor(np.random.randn(1, 3, 8).astype(np.float32))
    k = Tensor(np.random.randn(1, 3, 8).astype(np.float32))
    # large negative mask on key index 2
    mask_np = np.zeros((1, 2, 3, 3), dtype=np.float32)
    mask_np[:, :, :, 2] = -1e9
    mask = Tensor(mask_np)

    out = mha(q, k, k, mask=mask)
    assert out.shape == (1, 3, 8)


def test_mha_parameter_count_and_state_dict():
    mha = MultiheadAttention(embed_dim=16, num_heads=4, seed=4)
    # 4 linears (q, k, v, out), each has weight and bias -> 8 params
    params = mha.parameters()
    assert len(params) == 8
    assert all(p.requires_grad for p in params)
    assert len(mha.children()) == 4


def test_mha_gradient_finite_difference():
    embed_dim = 6
    num_heads = 2
    B, S = 2, 3

    mha = MultiheadAttention(embed_dim, num_heads, seed=5)
    q_np = np.random.randn(B, S, embed_dim).astype(np.float64)

    # Reference computation using mha parameters
    qw = mha.q_proj.weight.data.astype(np.float64)
    qb = mha.q_proj.bias.data.astype(np.float64)
    kw = mha.k_proj.weight.data.astype(np.float64)
    kb = mha.k_proj.bias.data.astype(np.float64)
    vw = mha.v_proj.weight.data.astype(np.float64)
    vb = mha.v_proj.bias.data.astype(np.float64)
    ow = mha.out_proj.weight.data.astype(np.float64)
    ob = mha.out_proj.bias.data.astype(np.float64)

    H = num_heads
    D = embed_dim // num_heads

    def ref_forward(x):
        q = (np.matmul(x, qw.T) + qb).reshape(B, S, H, D).transpose(0, 2, 1, 3)
        k = (np.matmul(x, kw.T) + kb).reshape(B, S, H, D).transpose(0, 2, 1, 3)
        v = (np.matmul(x, vw.T) + vb).reshape(B, S, H, D).transpose(0, 2, 1, 3)
        scale = 1.0 / np.sqrt(D)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        attn = e / e.sum(axis=-1, keepdims=True)
        out = np.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, S, embed_dim)
        res = np.matmul(out, ow.T) + ob
        return float(res.sum())

    fd = finite_difference_grad(ref_forward, q_np)

    q = Tensor(q_np.astype(np.float32), requires_grad=True)
    out = mha(q)
    out.sum().backward()

    assert_close(q.grad.data.astype(np.float64), fd, rtol=1e-3, atol=1e-3,
                 msg="MultiheadAttention input gradient")


def test_mha_invalid_dimensions_raise_errors():
    with pytest.raises(NNError) as e1:
        MultiheadAttention(embed_dim=15, num_heads=2)
    assert "E0615" in str(e1.value)

    mha = MultiheadAttention(8, 2)
    with pytest.raises(NNError) as e2:
        mha(Tensor(np.ones((2, 8), dtype=np.float32)))  # 2D instead of 3D
    assert "E0615" in str(e2.value)


def test_transformer_block_forward_and_params():
    block = TransformerBlock(embed_dim=16, num_heads=4, mlp_ratio=2.0, seed=10)
    assert len(block.parameters()) == 16  # norm1(2) + attn(8) + norm2(2) + mlp(4)
    assert len(block.children()) == 4

    x = Tensor(np.random.randn(2, 6, 16).astype(np.float32))
    out = block(x)
    assert out.shape == (2, 6, 16)


def test_transformer_block_checkpoint_roundtrip(tmp_path):
    from timet import checkpoint
    block = TransformerBlock(embed_dim=8, num_heads=2, seed=1)
    p = checkpoint.save_bin(block, tmp_path / "block.ttck")
    block2 = TransformerBlock(embed_dim=8, num_heads=2, seed=99)
    checkpoint.load_into(block2, p)
    for p1, p2 in zip(block.parameters(), block2.parameters()):
        assert np.array_equal(p1.data, p2.data)


def test_transformer_block_learns_target_mapping():
    from timet.optim import Adam
    from timet.nn import MSELoss

    block = TransformerBlock(embed_dim=8, num_heads=2, seed=7)
    opt = Adam(block.parameters(), lr=0.02)
    loss_fn = MSELoss()

    rng = np.random.default_rng(42)
    x_np = rng.normal(size=(2, 4, 8)).astype(np.float32)
    y_np = rng.normal(size=(2, 4, 8)).astype(np.float32)
    x = Tensor(x_np)
    y_target = Tensor(y_np)

    losses = []
    for _ in range(60):
        pred = block(x)
        loss = loss_fn(pred, y_target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.data))

    assert losses[-1] < losses[0] / 3, f"TransformerBlock failed to converge: {losses[0]} -> {losses[-1]}"
