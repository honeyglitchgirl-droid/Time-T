# Changelog

All notable changes to Time-T are recorded here. Format loosely follows
Keep a Changelog; versioning follows master prompt §37 (targets, not
promises).

## [0.1.0] - Initial implementation (Milestones 0-5, partial 6-7)

### Added
- Architecture, language, roadmap, testing, and design-decision documents
  (`docs/`), written before substantial implementation per master prompt §4.
- Lexer, parser, and AST (`timet/lexer.py`, `timet/parser.py`,
  `timet/ast_nodes.py`) for a brace-delimited language with `let`/`var`,
  functions/closures, `if`/`while`, tensors, and `no_grad` blocks.
- Type checker (`timet/typechecker.py`) with scope-aware name resolution,
  structural type inference/checking, and master-prompt-§27-style
  diagnostics (`timet/diagnostics.py`).
- Typed, inspectable, JSON-serializable IR (`timet/ir.py`) and
  `time-t inspect --ir`.
- Tree-walking interpreter (`timet/interpreter.py`).
- NumPy-backed `Tensor` (`timet/tensor.py`) with broadcasting, matmul,
  reductions, activations, and basic indexing.
- Reverse-mode automatic differentiation (`timet/autodiff.py`,
  `Tensor.backward`), gradient-checked against finite differences for every
  op, plus the literal master-prompt worked example as a regression test.
- Backend abstraction (`timet/backend.py`) with one real implementation
  (`NumpyCpuBackend`) and a capability-query interface.
- Real memory accounting (`timet/memory.py`).
- Minimal neural-network layer library and SGD optimizer (`timet/nn.py`,
  `timet/optim.py`), proven by a converging linear-regression and XOR
  training run.
- `time-t` CLI (`timet/cli.py`): `check`, `run`, `inspect`, `test`, `bench`,
  `repl` implemented; `build`, `profile`, `export`, `package`, `doctor`
  explicitly marked not-implemented (never silent).
- 175 automated tests (`pytest -q`), covering lexer, parser, type checker,
  interpreter, IR, tensor, autodiff, backend, nn, CLI, examples, fuzzing
  (lexer/parser), and diagnostics.
- 6 runnable example programs with byte-exact expected output.
- Real, reproducible benchmark harness (`benchmarks/run_benchmarks.py`).

### Explicitly not included in this release
Generics, traits, structs, enums, pattern matching, modules/imports, static
shape typing, IR optimization passes, IR execution, native codegen, GPU/
ARM64/mobile backends, C ABI/FFI, ONNX, package manager, datasets/
dataloaders, checkpoints, Adam/AdamW, attention/transformer layers. See
`docs/ROADMAP.md` for the sequencing and self-criticism.
