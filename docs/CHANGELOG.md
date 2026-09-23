# Changelog

All notable changes to Time-T are recorded here. Format loosely follows
Keep a Changelog; versioning follows master prompt §37 (targets, not
promises).

## [0.2.0] - Milestone 6 complete (IR execution + optimization), Milestones 7-8 expanded

### Added
- **IR executor** (`timet/ir_exec.py`): the typed IR now RUNS. Structured
  control markers become a block tree; `var`s live in chained named-storage
  frames; while-conditions re-execute per iteration; user `fn main()` is
  called after top-level statements, exactly like the AST engine. CLI:
  `time-t run <file> --via-ir [-O 0|1]`. Deliberate, loudly-signalled
  limits: DD-9 (no lambdas/nested fns/if-expressions; eager `&&`/`||`).
- **IR optimizer** (`timet/optimize.py`): constant folding, type-aware
  algebraic simplification, segment-local CSE, segment-local copy
  propagation, DCE — with documented safety rules (DD-12: no `x*0`
  folding, `1/0` stays a runtime error, no cross-control-boundary CSE,
  stores invalidate loads) and per-pass statistics
  (`time-t inspect --ir --opt`).
- **Engine-level differential testing** (`tests/test_ir_exec_diff.py`):
  all 7 examples + 70 generated random programs run byte-identically on AST
  vs IR-O0 vs IR-O1.
- **Tensor ops**: `Tensor.argmax`, `Tensor.clip` (with backward rule),
  `Tensor.log_softmax` (numerically stable, composed of verified
  primitives), `one_hot`. All differentially/gradient tested.
- **nn**: `Tanh`, `Softmax`, `Flatten`, `Dropout` (inverted, seeded,
  train/eval-aware), `Module.train()/eval()` recursion;
  `CrossEntropyLoss` and `BCELoss` (both finite-difference checked);
  functional forms `mse_loss`/`cross_entropy_loss`/`binary_cross_entropy`.
- **optim**: `Adam` (bias-corrected), with a test that must reach a loss
  threshold on XOR (measured value, not assumed).
- **train** (`timet/train.py`): `fit()` full-batch training loop,
  `History`, `accuracy()`, `EarlyStopping`.
- **checkpoint** (`timet/checkpoint.py`): versioned, deterministic JSON
  parameter save/load (`save`, `load`, `load_into`, `state_dict`) with
  strict key/shape/corruption checking; resume-equality test
  (train/save/restore/train == uninterrupted) (DD-11).
- **Time-T programs can now use the NN library directly** via pre-bound
  `nn`/`optim`/`train` globals (DD-10), in both engines. New builtins
  `log_softmax`, `argmax`, `one_hot`.
- **Example 7**: `examples/07_xor_classifier.tt` — MLP + Tanh +
  cross-entropy + Adam trained from Time-T to 100% XOR accuracy; runs
  identically on all three engines.
- **New CLI flags**: `run --via-ir`, `run -O/--opt-level {0,1}`,
  `inspect --ir --opt`. `run --json` now reports which `engine` ran.
- New benchmark row (`mlp_train_step_128x2_adam`).
- New docs: `docs/RETROSPECTIVES.md` (§39 entries per milestone band);
  DD-9..DD-12 in DESIGN_DECISIONS.md.

### Fixed
- IR lowering bugs found by the new differential suite: while-condition was
  evaluated once instead of per-iteration (executor-side fix: cond now
  lives inside the loop region); CSE arg-rewrite not written back to
  non-CSE instructions (could reference eliminated temps); `for`-loop
  variable typing (now the iterated tensor's dtype, not `TUnknown`).

### Changed
- for-loop variables over `Tensor` iterables are typed as element tensors;
  `.item()` returns a scalar type (`Float`/`Int` by dtype), `.backward()`
  returns `Unit`.

### Test count
- 175 → **247 passing** (`pytest -q`): 11 IR-executor + 13 optimizer +
  9 engine-differential + 7 tensor + 3 autodiff + 10 nn + 7 train +
  8 checkpoint + 3 CLI + 1 example test added in this release.


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
