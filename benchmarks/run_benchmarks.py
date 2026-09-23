"""Time-T benchmark harness (master prompt section 19).

Every number here is measured on whatever machine runs this script -- never
hand-typed or estimated. Run `python3 -m timet bench` or
`python3 benchmarks/run_benchmarks.py` to regenerate benchmarks/results/*.json.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from timet.tensor import Tensor, tensor  # noqa: E402


def _time_it(fn, repeats: int = 5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def bench_scalar_add(n=200_000):
    def run():
        total = 0
        for i in range(n):
            total = total + i
        return total
    t = _time_it(run, repeats=1)
    return {"name": "scalar_add_loop", "seconds": t, "note": f"n={n} python scalar adds"}


def bench_vector_add(n=1_000_000):
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    ta = Tensor(a)
    tb = Tensor(b)

    def run():
        _ = ta + tb
    t = _time_it(run)
    return {"name": "tensor_vector_add", "seconds": t, "note": f"n={n} f32 elementwise add"}


def bench_matmul(size):
    a = Tensor(np.random.rand(size, size).astype(np.float32))
    b = Tensor(np.random.rand(size, size).astype(np.float32))

    def run():
        _ = a.matmul(b)
    t = _time_it(run)
    flops = 2 * size ** 3
    gflops = flops / t / 1e9 if t > 0 else float("inf")
    return {"name": f"matmul_{size}x{size}", "seconds": t,
            "note": f"~{gflops:.2f} GFLOP/s (naive estimate)"}


def bench_reduction(n=1_000_000):
    a = Tensor(np.random.rand(n).astype(np.float32))

    def run():
        _ = a.sum()
    t = _time_it(run)
    return {"name": "tensor_sum_reduction", "seconds": t, "note": f"n={n}"}


def bench_autodiff_backward(n=100_000):
    a = Tensor(np.random.rand(n).astype(np.float32), requires_grad=True)

    def run():
        y = (a * a).sum()
        y.backward()
        a.grad = None
    t = _time_it(run, repeats=3)
    return {"name": "autodiff_backward_sum_square", "seconds": t, "note": f"n={n}"}


def bench_mlp_train_step():
    """One full-batch training step (forward + CE backward + Adam step) of a
    small MLP -- the workload examples/05-07 are built from."""
    from timet.nn import Linear, Tanh, Sequential, CrossEntropyLoss
    from timet.optim import Adam

    x = Tensor(np.random.default_rng(0).normal(size=(128, 2)).astype(np.float32))
    y = np.random.default_rng(1).integers(0, 2, size=128)
    model = Sequential(Linear(2, 16, seed=0), Tanh(), Linear(16, 2, seed=1))
    loss_fn = CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=0.01)

    def run():
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    t = _time_it(run, repeats=3)
    return {"name": "mlp_train_step_128x2_adam", "seconds": t,
            "note": "full-batch fwd+bwd+step, 2-16-2 MLP"}




def bench_native_loop_sum():
    """AST interpreter vs the native C emitter (DD-15) on the same scalar
    loop. Only recorded when a C compiler exists on the machine; the ratio
    measures interpreter overhead on PURE SCALAR arithmetic -- it is not a
    claim about tensor workloads (those are NumPy-bound either way)."""
    import shutil
    if not shutil.which("gcc") and not shutil.which("cc") \
            and not shutil.which("clang"):
        return {"name": "native_loop_sum_1e6", "skipped": True,
                "note": "no C compiler on PATH"}
    import subprocess
    import tempfile as _tf
    from timet.parser import parse
    from timet.typechecker import check
    from timet.native import build as _nbuild
    from timet.interpreter import run_source as _rs
    src = """
var i = 0
var acc = 0
while i < 1000000 {
    acc = acc + i * 2 - i % 7
    i = i + 1
}
print(acc)
"""
    out = []
    t_ast = _time_it(lambda: _rs(src, filename="<bench>",
                                 stdout_write=out.append), repeats=3)
    prog = check(parse(src, "<bench>"))
    with _tf.TemporaryDirectory() as td:
        exe = td + "/a.out"
        res = _nbuild(prog, exe)
        proc = subprocess.run([exe], capture_output=True, text=True)
        assert proc.stdout.strip() == out[0] \
            and not proc.stderr, "native output diverged from the interpreter!"
        t_native = _time_it(
            lambda: subprocess.run([exe], capture_output=True, text=True),
            repeats=5)
    return {"name": "native_loop_sum_1e6", "seconds": t_native,
            "note": f"native binary via gcc (AST interp on same loop: "
                    f"{t_ast:.3f}s; ratio {t_ast / max(t_native, 1e-9):.0f}x; "
                    f"scalar-loop workload, NOT a tensor-perf claim)",
            "ast_interpreter_seconds": t_ast,
            "speedup": t_ast / max(t_native, 1e-9)}


def run_all() -> dict:
    benches = [
        bench_scalar_add(),
        bench_vector_add(),
        bench_matmul(64),
        bench_matmul(256),
        bench_reduction(),
        bench_autodiff_backward(),
        bench_mlp_train_step(),
        bench_native_loop_sum(),
    ]
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "system": platform.system(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "benchmarks": benches,
    }
    return result


def main():
    result = run_all()
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"bench_{int(time.time())}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    for b in result["benchmarks"]:
        if b.get("skipped"):
            print(f"{b['name']:30s}  SKIPPED: {b.get('note','')}")
        else:
            print(f"{b['name']:30s} {b['seconds']*1000:10.3f} ms   {b.get('note','')}")


if __name__ == "__main__":
    main()
