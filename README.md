# Time-T

Time-T is a programming language and runtime prototype designed for AI/ML,
numerical computing, and high-performance execution — built from a written
specification (`Time-T_Fresh_Start_Master_Prompt.md`) that intentionally
sequences work: architecture first, then a small verified vertical slice,
then expansion in tested layers. **Nothing in this repository is claimed to
work unless it is backed by a passing test.**

## What exists today (v0.2.0)

- A real lexer, parser, and type checker for a small, brace-delimited
  language (see `docs/LANGUAGE.md`).
- TWO execution engines whose outputs are differentially verified to be
  byte-identical (`tests/test_ir_exec_diff.py`):
  - a tree-walking interpreter (the semantic reference), and
  - an IR executor (`time-t run <file> --via-ir [-O 0|1]`) running the
    typed IR directly, with an `-O1` optimizer (constant folding, CSE,
    algebraic simplification, copy propagation, DCE — safety rules in
    `docs/DESIGN_DECISIONS.md` DD-12).
- An inspectable, JSON-serializable typed IR (`time-t inspect --ir [--opt]`).
- A NumPy-backed `Tensor` type with broadcasting, matmul, reductions,
  activations (`relu/sigmoid/tanh/softmax/log_softmax`), `clip`, `argmax`,
  `one_hot`, and reverse-mode automatic differentiation, gradient-checked
  against finite differences.
- A neural-network library usable from BOTH Python and Time-T code:
  layers (`Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Flatten`,
  seeded `Dropout`), losses (`MSE`, `CrossEntropy`, `BCE`), optimizers
  (`SGD`, `Adam`) — proven by training runs that must reach loss/accuracy
  thresholds in tests, not just "loss went down".
- Training utilities: `train.fit()` (full-batch, honest about it),
  `train.accuracy`, `EarlyStopping`, and versioned deterministic JSON
  checkpoints (`timet/checkpoint.py`) with a tested resume guarantee.
- A `time-t` CLI: `check`, `run` (incl. `--via-ir`), `inspect`, `test`,
  `bench`, `repl` are real; `build`, `profile`, `export`, `package`,
  `doctor` are explicit, machine-readable "not implemented yet" stubs
  (never silent no-ops).
- 247 automated tests across lexer/parser/typechecker/interpreter/IR/
  IR-executor/optimizer/tensor/autodiff/backend/nn/train/checkpoint/CLI/
  examples/differential/fuzz/diagnostics (`pytest -q`).
- 7 runnable example programs with byte-exact expected output
  (`examples/*.tt` + `examples/*.expected`), incl. an Adam + cross-entropy
  XOR classifier written in Time-T.
- A real (not fabricated) benchmark harness with results written to
  `benchmarks/results/*.json`, labeled with the actual host CPU/platform.

## What does NOT exist yet (see `docs/ROADMAP.md`)

Generics, traits, structs, enums, pattern matching, a real module/import
system, static shape typing, CFG-form IR, inlining/fusion passes, native
codegen, GPU/ARM64/mobile backends, C ABI/FFI, ONNX import/export, a package
manager, datasets/dataloaders, mini-batching, LR schedules, binary
checkpoints, AdamW, convolutions/embeddings/attention, distributed training.
These are explicitly sequenced, not silently missing — see
`docs/ROADMAP.md`'s milestone table and `docs/RETROSPECTIVES.md`.

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

Or via the shim: `bin/time-t run examples/01_hello_world.tt`.

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
| `docs/MOBILE.md` | Explicit statement of what mobile/ARM64 support exists (none yet) |
| `docs/BENCHMARKS.md` | How benchmarks are produced and where results live |
| `docs/CHANGELOG.md` | Version history |

The original build specification is preserved unmodified at
`Time-T_Fresh_Start_Master_Prompt.md`.
