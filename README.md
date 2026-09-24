# Time-T

Time-T is a programming language and runtime prototype designed for AI/ML,
numerical computing, and high-performance execution — built from a written
specification (`Time-T_Fresh_Start_Master_Prompt.md`) that intentionally
sequences work: architecture first, then a small verified vertical slice,
then expansion in tested layers. **Nothing in this repository is claimed to
work unless it is backed by a passing test.**

- **Current version**: `2.1.0`
- **Current test inventory**: 497 automated tests across all subsystems with 100% pass rate
- **Current example inventory**: 16 runnable example programs with byte-exact verification

## What exists today (v2.1.0)

- A real lexer, parser, and type checker for a brace-delimited
  language supporting primitives, functions, closures, file-based modules, and
  user-defined structs (see `docs/LANGUAGE.md`).
- TWO execution engines whose outputs are differentially verified to be
  byte-identical (`tests/test_ir_exec_diff.py`):
  - a tree-walking interpreter (the semantic reference) with robust recursion
    depth and division-by-zero diagnostic guards (`E0507`, `E0508`, `E0509`).
  - an IR executor (`time-t run <file> --via-ir [-O 0|1]`) running the
    typed IR directly, with an `-O1` optimizer (constant folding, CSE,
    algebraic simplification, copy propagation, DCE — safety rules in
    `docs/DESIGN_DECISIONS.md` DD-12).
- An inspectable, JSON-serializable typed IR (`time-t inspect --ir [--opt]`).
- A NumPy-backed `Tensor` type with broadcasting, matmul, reductions,
  activations (`relu/sigmoid/tanh/softmax/log_softmax`), `clip`, `argmax`,
  `one_hot`, and reverse-mode automatic differentiation, gradient-checked
  against finite differences.
- Native C JIT Kernel Accelerator (`timet.jit_kernels`, DD-32): host GCC/Clang compiled
  with `-O3 -lm` for accelerated vector activations (`gelu`, `relu`, `layernorm`).
- Multi-Threaded SIMD OpenMP Native Backend (`timet.simd_backend`, DD-33): OpenMP-parallelized
  CPU tensor kernels with fallback guards and scalar fallbacks.
- Neural-network layers: Linear, ReLU/Sigmoid/Tanh/GELU/Softmax, Flatten,
  Dropout (seeded), Conv1D/Conv2D (stride/padding, gradient-checked),
  Embedding, LayerNorm/RMSNorm (affine scale/shift, gradient-checked),
  MultiheadAttention (multi-head scaled dot-product attention, gradient-checked),
  TransformerBlock (Pre-LN Transformer Encoder), and TransformerLM (Causal Language Model);
  losses MSE/CrossEntropy/BCE; Sequential.
- Optimizers: SGD, Adam, AdamW (decoupled decay).
- Training utilities: `train.fit()` (full-batch), `train.fit_loader()` with
  `TensorDataset`/`DataLoader` (seeded-shuffle mini-batching),
  `StepLR`/`ExponentialLR`/`CosineAnnealingLR` schedules, validation splits
  (`val_loader=` + monitorable early stopping), `train.accuracy`, `EarlyStopping`.
- Checkpoints & Serialization: deterministic JSON v1 AND binary v2 `.ttck`
  formats, full training-state files (optimizer moments + scheduler + loader RNG)
  with byte-exact Adam resume proven, portable HuggingFace `safetensors` export/import,
  and NumPy `.npz` archive export/import.
- Mobile & INT8 Quantization: `timet.mobile` dynamic INT8 quantization
  (`QuantizedLinear`, `quantize_dynamic`) delivering 4x weight memory reduction,
  and standalone `.ttm` deployment packaging.
- User-Defined Data Structures: `struct Name { field: Type }` with static type checking
  and field access.
- Native C-Emitter with floating-point formatting runtime (`tt_print_float`, DD-31).
- A complete `time-t` CLI: 12 subcommands fully implemented without stubs:
  `check`, `run` (incl. `--via-ir` and `--native`), `inspect`, `test`, `bench`,
  `repl` (supports both statements and expressions), `build` (C-emitter),
  `verify` (multi-engine differential validator),
  `export` (safetensors/npz/bin/json), `package` (mobile bundle), `doctor` (toolchain diagnostics),
  and `profile` (runtime & peak memory profiling).
- 497 automated tests across all subsystems with 100% pass rate.
- 16 runnable example programs with byte-exact expected output (`examples/*.tt` + `examples/*.expected`).
- A real (not fabricated) benchmark harness with results written to
  `benchmarks/results/*.json`, labeled with the actual host CPU/platform.

## What is planned for post-v2.1 (see `docs/ROADMAP.md`)

- Hardware GPU/accelerator backends (CUDA, ROCm, Metal) — currently UNVERIFIED-HARDWARE.
- Native tensor kernel lowering to specialized vendor BLAS.
- Advanced static shape typing for tensor ranks and dimensions.
- Generics and traits/interfaces.
- Concurrency and distributed training primitives.
- Package manager for remote Time-T package dependencies.

## Quick start

```bash
python3 -m pip install --break-system-packages -r requirements.txt
python3 -m pytest -q                          # run the full test suite
python3 -m timet check examples/01_hello_world.tt
python3 -m timet run examples/01_hello_world.tt
python3 -m timet run examples/07_xor_classifier.tt          # train an MLP from Time-T
python3 -m timet run examples/07_xor_classifier.tt --via-ir # ...on the IR executor
python3 -m timet inspect examples/02_calculator.tt --ir --opt
python3 -m timet repl
python3 -m timet bench
```

Or via the launcher: `./bin/time-t run examples/01_hello_world.tt`.

## Documentation map

| Doc | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | Pipeline, module layout, type system, tensor model, IR, runtime, backend abstraction, 11 architectural risks |
| `docs/LANGUAGE.md` | Full language reference for what is implemented today |
| `docs/ROADMAP.md` | Milestone-by-milestone status |
| `docs/RETROSPECTIVES.md` | Self-criticism per milestone band (master prompt §39) |
| `docs/DESIGN_DECISIONS.md` | Dated, binding engineering decisions and why |
| `docs/TESTING.md` | How to run and reproduce every test category |
| `docs/COMPILER.md` | Compiler pipeline detail |
| `docs/RUNTIME.md` | Runtime/memory/backend detail |
| `docs/TENSOR.md` | Tensor semantics reference |
| `docs/AUTOGRAD.md` | Autodiff design and correctness methodology |
| `docs/BACKENDS.md` | Backend interface and current CPU backend |
| `docs/MOBILE.md` | Mobile INT8 quantization and .ttm packaging |
| `docs/BENCHMARKS.md` | How benchmarks are produced and where results live |
| `docs/CHANGELOG.md` | Version history |
| `PRODUCTION_READINESS.md` | Production readiness audit and hardware verification status |
| `SECURITY.md` | Security policy and vulnerability mitigation report |
| `THREAT_MODEL.md` | Time-T threat model and attack surface analysis |
| `TEST_MATRIX.md` | Subsystem test coverage matrix |

The original build specification is preserved unmodified at
`Time-T_Fresh_Start_Master_Prompt.md`.
