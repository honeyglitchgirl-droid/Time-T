"""Stress test runner for Time-T v2.1.0 release verification."""
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import tempfile
import shutil
import numpy as np
from pathlib import Path

from timet.tensor import Tensor
import timet.nn as nn
import timet.optim as optim
import timet.checkpoint as checkpoint
from timet.jit_kernels import fast_gelu, fast_relu, fast_layernorm, has_fast_kernels
from timet.lexer import Lexer
from timet.parser import Parser
from timet.typechecker import TypeChecker
from timet.ir import lower_program
from timet.interpreter import Interpreter
from timet.ir_exec import IRExecutor


def test_repeated_tensor_lifecycle(iterations: int = 100):
    start = time.perf_counter()
    for i in range(iterations):
        a = Tensor(np.random.randn(32, 64).astype(np.float32), requires_grad=True)
        w = Tensor(np.random.randn(64, 32).astype(np.float32), requires_grad=True)
        c = (a @ w).relu().sum()
        c.backward()
        assert a.grad is not None and w.grad is not None
    dur = time.perf_counter() - start
    return {"name": "tensor_autograd_lifecycle", "iterations": iterations, "duration_s": dur, "status": "PASS"}


def test_repeated_model_train_cycle(iterations: int = 50):
    start = time.perf_counter()
    for i in range(iterations):
        m = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        opt = optim.AdamW(m.parameters(), lr=1e-3)
        x = Tensor(np.random.randn(8, 16).astype(np.float32))
        y = Tensor(np.random.randn(8, 8).astype(np.float32))
        out = m(x)
        loss = nn.mse_loss(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    dur = time.perf_counter() - start
    return {"name": "model_train_cycle", "iterations": iterations, "duration_s": dur, "status": "PASS"}


def test_repeated_jit_lifecycle(iterations: int = 30):
    start = time.perf_counter()
    assert has_fast_kernels()
    for i in range(iterations):
        x = np.random.randn(1024).astype(np.float32)
        out1 = fast_gelu(x)
        out2 = fast_relu(x)
        gamma = np.ones(1024, dtype=np.float32)
        beta = np.zeros(1024, dtype=np.float32)
        out3 = fast_layernorm(x, gamma, beta, 1e-5)
        assert out1 is not None and out2 is not None and out3 is not None
    dur = time.perf_counter() - start
    return {"name": "jit_kernel_lifecycle", "iterations": iterations, "duration_s": dur, "status": "PASS"}


def test_repeated_checkpoint_io(iterations: int = 50):
    start = time.perf_counter()
    tmpdir = tempfile.mkdtemp(prefix="timet_stress_ckpt_")
    try:
        p = Path(tmpdir) / "stress.ttck"
        for i in range(iterations):
            m = nn.Linear(8, 4)
            checkpoint.save(m, str(p))
            m2 = nn.Linear(8, 4)
            checkpoint.load_into(m2, str(p))
            assert np.allclose(m.weight.data, m2.weight.data)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    dur = time.perf_counter() - start
    return {"name": "checkpoint_io_cycle", "iterations": iterations, "duration_s": dur, "status": "PASS"}


def test_repeated_compiler_ir_exec(iterations: int = 40):
    start = time.perf_counter()
    src = "fn compute(x: Int, y: Int) -> Int { return x + y }\nfn main() { print(compute(10, 20)) }"
    for i in range(iterations):
        tokens = Lexer(src).tokenize()
        ast = Parser(tokens).parse_program()
        tc = TypeChecker()
        tc.check_program(ast)
        ir_prog = lower_program(ast)
        assert "compute" in [fn.name for fn in ir_prog.functions]
    dur = time.perf_counter() - start
    return {"name": "compiler_ir_exec_cycle", "iterations": iterations, "duration_s": dur, "status": "PASS"}


def run_all_stress_tests():
    print("=" * 60)
    print("Time-T v2.1.0 Long-Running Stability Stress Test")
    print("=" * 60)
    tests = [
        test_repeated_tensor_lifecycle,
        test_repeated_model_train_cycle,
        test_repeated_jit_lifecycle,
        test_repeated_checkpoint_io,
        test_repeated_compiler_ir_exec,
    ]
    results = []
    total_start = time.perf_counter()
    for t in tests:
        res = t()
        results.append(res)
        print(f"[{res['status']}] {res['name']:<30} {res['iterations']} iter in {res['duration_s']:.3f}s")
    total_dur = time.perf_counter() - total_start
    print("-" * 60)
    print(f"Stress test suite passed in {total_dur:.2f}s with 0 errors.")
    print("=" * 60)
    return results


if __name__ == "__main__":
    run_all_stress_tests()
