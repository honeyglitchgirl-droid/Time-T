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


def run_all() -> dict:
    benches = [
        bench_scalar_add(),
        bench_vector_add(),
        bench_matmul(64),
        bench_matmul(256),
        bench_reduction(),
        bench_autodiff_backward(),
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
        print(f"{b['name']:30s} {b['seconds']*1000:10.3f} ms   {b.get('note','')}")


if __name__ == "__main__":
    main()
