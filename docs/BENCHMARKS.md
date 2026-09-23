# Time-T — Benchmarks

See `docs/ROADMAP.md` "Benchmarks status" and master prompt §19.

## How to reproduce

```bash
python3 benchmarks/run_benchmarks.py
# or
python3 -m timet bench --json
```

Each run writes a fresh, timestamped JSON file to `benchmarks/results/`
containing:
- `timestamp` (UTC)
- `platform` (`processor`, `machine`, `system`, `python_version`,
  `numpy_version`) — the *actual* values from the machine that ran it, via
  Python's `platform` module and `numpy.__version__`, never hand-typed.
- `benchmarks`: a list of `{name, seconds, note}` — `seconds` is the minimum
  of several repeated timed runs (`time.perf_counter`), to reduce noise
  without hiding the real number.

## What is currently benchmarked

| Name | What it measures |
|---|---|
| `scalar_add_loop` | Python-level scalar loop cost (interpreter overhead baseline) |
| `tensor_vector_add` | 1M-element f32 elementwise tensor add |
| `matmul_64x64`, `matmul_256x256` | NumPy-backed matmul at two sizes, with a naive GFLOP/s estimate (`2*n^3/seconds`) |
| `tensor_sum_reduction` | 1M-element reduction |
| `autodiff_backward_sum_square` | Forward + backward pass of `sum(x*x)` over 100K elements |

## What is NOT benchmarked yet

Convolution, attention, full model inference/training throughput at
realistic parameter counts, multi-threaded scaling, any GPU/ARM64 numbers
(no such backend exists — see `docs/MOBILE.md`, `docs/BACKENDS.md`).

## Policy

No performance number appears in any Time-T documentation unless it was
produced by running the script above on the machine described in that same
result's `platform` field. Old result files are never edited by hand; a new
run produces a new timestamped file.
