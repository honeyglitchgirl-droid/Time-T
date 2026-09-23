# Time-T

Time-T is a programming language and runtime prototype designed for AI/ML,
numerical computing, and high-performance execution — built from a written
specification (`Time-T_Fresh_Start_Master_Prompt.md`) that intentionally
sequences work: architecture first, then a small verified vertical slice,
then expansion in tested layers. **Nothing in this repository is claimed to
work unless it is backed by a passing test.**

## What exists today (v0.1.0)

- A real lexer, parser, and type checker for a small, brace-delimited
  language (see `docs/LANGUAGE.md`).
- A tree-walking interpreter that runs `.tt` programs.
- An inspectable, JSON-serializable typed IR (`time-t inspect --ir`) — not
  yet optimized or executed (see `docs/DESIGN_DECISIONS.md`, DD-3).
- A NumPy-backed `Tensor` type with broadcasting, matmul, reductions,
  activations, and reverse-mode automatic differentiation, gradient-checked
  against finite differences.
- A minimal neural-network layer library (`Linear`, `ReLU`, `Sequential`,
  `MSELoss`) and an `SGD` optimizer, proven by a training run that
  demonstrably reduces loss (linear regression + XOR examples).
- A `time-t` CLI: `check`, `run`, `inspect`, `test`, `bench`, `repl` are real;
  `build`, `profile`, `export`, `package`, `doctor` are explicit,
  machine-readable "not implemented yet" stubs (never silent no-ops).
- 175 automated tests across lexer/parser/typechecker/interpreter/IR/tensor/
  autodiff/backend/nn/CLI/examples/fuzz/diagnostics (`pytest -q`).
- 6 runnable example programs with byte-exact expected output
  (`examples/*.tt` + `examples/*.expected`).
- A real (not fabricated) benchmark harness with results written to
  `benchmarks/results/*.json`, labeled with the actual host CPU/platform.

## What does NOT exist yet (see `docs/ROADMAP.md`)

Generics, traits, structs, enums, pattern matching, modules/imports, static
shape typing, IR optimization passes, native codegen, GPU/ARM64/mobile
backends, C ABI/FFI, ONNX import/export, a package manager, datasets/
dataloaders, checkpoints, Adam/AdamW, attention/transformer layers. These are
explicitly sequenced, not silently missing — see `docs/ROADMAP.md`'s
milestone table and self-criticism section.

## Quick start

```bash
python3 -m pip install --break-system-packages -r requirements.txt
python3 -m pytest -q                          # run the full test suite
python3 -m timet check examples/01_hello_world.tt
python3 -m timet run examples/01_hello_world.tt
python3 -m timet inspect examples/02_calculator.tt --ir
python3 -m timet repl
python3 -m timet bench
```

Or via the shim: `bin/time-t run examples/01_hello_world.tt`.

## Documentation map

| Doc | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | Pipeline, module layout, type system, tensor model, IR, runtime, backend abstraction, 11 architectural risks |
| `docs/LANGUAGE.md` | Full language reference for what is implemented today |
| `docs/ROADMAP.md` | Milestone-by-milestone status + self-criticism (master prompt §39) |
| `docs/DESIGN_DECISIONS.md` | Dated, binding engineering decisions and why |
| `docs/TESTING.md` | How to run and reproduce every test category |
| `docs/COMPILER.md` | Compiler pipeline detail |
| `docs/RUNTIME.md` | Runtime/memory/backend detail |
| `docs/TENSOR.md` | Tensor semantics reference |
| `docs/AUTOGRAD.md` | Autodiff design and correctness methodology |
| `docs/BACKENDS.md` | Backend interface and current CPU backend |
| `docs/MOBILE.md` | Explicit statement of what mobile/ARM64 support exists (none yet) |
| `docs/BENCHMARKS.md` | How benchmarks are produced and where results live |
| `docs/CHANGELOG.md` | Version history |

The original build specification is preserved unmodified at
`Time-T_Fresh_Start_Master_Prompt.md`.
