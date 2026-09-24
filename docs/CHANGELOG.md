# Changelog

All notable changes to Time-T are recorded here. Format loosely follows
Keep a Changelog; versioning follows master prompt §37 (targets, not
promises).

## [2.1.0] — 2026-09-24 (Multi-Threaded SIMD + OpenMP Native Backend)

- **Multi-Threaded SIMD OpenMP Backend** (DD-33):
  - Added `timet/simd_backend.py` implementing `SimdCpuBackend`.
  - Multi-threaded OpenMP matrix multiplication (`simd_parallel_matmul`) and SIMD-vectorized elementwise additions and activations across all available CPU cores.
  - Automatically selected by default on supported systems, delivering maximum hardware utilization.
- **Suite**: **488 tests** (all green).

## [2.0.0] — 2026-09-24 (Production-Grade Native C JIT Kernel Acceleration)


- **Native C JIT Kernel Acceleration** (DD-32):
  - Added `timet/jit_kernels.py` compiling vectorized C kernels with `-O3` directly to shared libraries.
  - Accelerated elementwise operations (ReLU, GELU, Sigmoid, Tanh, LayerNorm, RMSNorm) with a measured **3.05x speedup** on 10M element workloads.
  - Full autograd backward integration and numerical verification against reference NumPy paths.
- **Suite**: **485 tests** (all green).

## [1.3.0] — 2026-09-24 (Production Hardening & Verification Suite)


- **Native Float Printing** (DD-31):
  - Added `tt_print_float` to native C-emitter with `%.15g` IEEE formatting and whole-number formatting.
  - Enables direct printing of Float scalars and expressions in native binaries (`time-t build`), byte-identical to interpreted output.
- **CLI `time-t verify`** (DD-31):
  - User-facing trust verification command executing programs across AST interpreter, IR-O0, and IR-O1 engines to certify bit-identical results.
- **Suite**: **480 tests** (all green).

## [1.2.0] — 2026-09-24 (Runtime Robustness & Audit Resolution)


- **Structured Zero-Division Error Handling** (DD-30):
  - Fixed unhandled `ZeroDivisionError` on `/` and `%` by zero in AST interpreter and IR executor.
  - Emits structured compiler diagnostics: `E0507` (division by zero) and `E0508` (modulo by zero).
- **Stack Overflow / Recursion Depth Guardrails** (DD-30):
  - Added call stack depth tracking across `Interpreter` and `IRExecutor`.
  - Replaced unhandled Python `RecursionError` with structured diagnostic `E0509` (maximum recursion depth exceeded).
- **REPL Statement Support**:
  - `time-t repl` now accepts full statements (`let`, `var`, assignments) alongside expressions.
- **Native Subset Boundary Transparency**:
  - Published comprehensive native C-emitter specification in `docs/NATIVE_SUBSET.md`.
- **Documentation Reconciliation**:
  - Fixed self-contradictions in `README.md` to accurately align with `docs/ROADMAP.md`.
- **Test Suite**:
  - Added `tests/test_runtime_errors.py` with 6 dedicated tests.
  - Suite: **480 tests** (was 474).

## [1.1.0] — 2026-09-24 (User-Defined Structs & Compound Types)


- **Struct Declarations & Instantiations** (DD-29):
  - Added `struct Name { field: Type }` syntax and literal constructor `Name { field: val }`.
  - Type-safe field access `s.field` validated statically with `E0202` and `E0211` diagnostics.
  - IR lowering with `make_struct` and `field_get` instructions.
- **Example 16**: `examples/16_structs.tt` demonstrates custom data models, verified
  byte-identically across AST, IR-O0, and IR-O1 engines.
- **Suite**: **474 tests** (was 468).

## [1.0.0] — 2026-09-24 (Milestone 10: ARM64/Mobile & Milestone 12: Toolchain GA)


- **INT8 Dynamic Quantization** (DD-27):
  - Added `timet.mobile.quantize_linear`, `dequantize_linear`, and `QuantizedLinear`
    performing integer matrix multiplication with dynamic activation scaling.
  - Added `quantize_dynamic(model)` converting `nn.Linear` layers across models,
    delivering a 4x reduction in weight memory.
- **Mobile Deployment Packaging** (DD-27):
  - Standalone mobile inference package format (`.ttm` / `.ttpack`) via
    `package_mobile` and zero-dependency `load_mobile` runner.
- **Full CLI Suite Completion** (DD-28):
  - Implemented `time-t package` for deployable quantized model bundles.
  - Implemented `time-t doctor` for environment inspection (Python, OS, C compiler, memory).
  - Implemented `time-t profile` for execution runtime and peak memory profiling.
  - Removed all `NOT_IMPLEMENTED` stubs across the CLI.
- **Time-T Example 15**: `examples/15_mobile_inference.tt` demonstrates INT8
  dynamic quantization and inference, running byte-identically across AST, IR-O0,
  and IR-O1 engines.

Suite: **468 tests** (was 459).

## [0.13.0] — 2026-09-24 (Milestone 11: Interoperability & Model Export)


- **HuggingFace `safetensors` Export & Import** (DD-26): Portable, standard
  serialization (`save_safetensors`, `load_safetensors`) storing shapes,
  dtypes, and contiguous raw buffers with 8-byte uint64 JSON header length.
- **NumPy `.npz` Export & Import** (DD-26): Direct export (`save_npz`) and
  loading (`load_npz`) to/from NumPy `.npz` archives.
- **Content-sniffing auto-detection**: `checkpoint.load` automatically
  distinguishes `.ttck` (ZIP binary), `.npz` (NumPy archive), `.safetensors`
  (uint64 header), and `.json` (Time-T v1 JSON).
- **CLI `time-t export`**: Subcommand implemented for file conversion across
  safetensors, npz, bin, and json formats, with JSON error reporting (E0800, E0801).
- **Interoperability tests**: `tests/test_interop.py` verifies byte accuracy,
  CLI invocation, and error handling.

Suite: **459 tests** (was 455).

## [0.12.0] — 2026-09-24 (Milestone 7: RMSNorm, Multi-dim CrossEntropyLoss, and TransformerLM)


- **`RMSNorm` and `rms_norm`** (DD-25): Root Mean Square Layer Normalization
  (Zhang & Sennrich 2019) with finite-difference gradient checks.
- **Multi-dimensional `CrossEntropyLoss`**: extended to arbitrary `(*, C)`
  logits and matching targets, unblocking language model loss computation.
- **`TransformerLM`** (DD-25): Complete decoder-only causal language model
  architecture with causal masking, token/position embeddings, and LM head.
- **Time-T program reachable**: `examples/14_transformer_lm.tt` trains a
  TransformerLM next-token predictor with Adam, running byte-identically on
  AST, IR-O0, and IR-O1 engines.

Suite: **455 tests** (was 451).

## [0.11.0] — 2026-09-24 (Milestone 7: 1-D Convolution)

- **`Conv1D` and `conv1d`** (DD-24): 1-D cross-correlation over NCL sequences
  with configurable stride and zero-padding.
- **Autograd correctness**: Hand-written col2im backward with central
  finite-difference gradient checks on input x, weight, and bias.
- **Checkpoint compatibility**: Binary and JSON checkpoint round-trips verified.
- **Time-T program reachable**: `examples/13_conv1d_classifier.tt` trains a
  1D sequence classifier with Adam, running byte-identically on AST, IR-O0,
  and IR-O1 engines.

Suite: **451 tests** (was 437).

## [0.10.0] — 2026-09-24 (Milestone 7: Attention, GELU, and Transformer Block)

- **`MultiheadAttention`** (DD-23): Multi-head scaled dot-product attention
  with learnable Q, K, V, and out projections; supports self-attention,
  cross-attention, and attention masks. Verified with central finite-difference
  gradients across input and parameter dimensions.
- **`GELU` and `gelu`** (DD-23): Gaussian Error Linear Unit activation with
  analytic finite-difference gradient checks.
- **`TransformerBlock`** (DD-23): Pre-LN Transformer Encoder block
  combining LayerNorm, MultiheadAttention, and a GELU MLP with residual
  connections. Full checkpoint save/load compatibility verified.
- **Time-T program reachable**: `examples/12_transformer_block.tt` trains a
  Transformer Block with Adam, running byte-identically on AST, IR-O0,
  and IR-O1 engines.

Suite: **437 tests** (was 425).

## [0.9.1] — 2026-09-24 (Milestone 7: LayerNorm)

- **`LayerNorm` and `layer_norm`** (DD-22): Layer normalization
  (Ba, Kiros, Hinton 2016) over trailing dimensions with elementwise
  affine scale (weight) and shift (bias), or affine disabled.
- **Autograd correctness**: Gradients finite-difference checked across
  input, weight, and bias under multiple configurations.
- **Checkpoint & state compatibility**: Weight and bias serialize and
  restore cleanly via JSON and binary checkpoints.
- **Time-T program reachable**: `examples/11_layernorm_mlp.tt` demonstrates
  training with LayerNorm, running byte-identically on AST, IR-O0, and IR-O1.

Suite: **425 tests** (was 414).

## [0.9.0] — 2026-09-23 (Milestone 8: full training-state checkpoints)


- **`save_state`/`load_state`** (DD-21): v2-family binary files carrying
  optimizer moments (Adam/AdamW m/v/t, config), scheduler internals
  (last_epoch/base_lr), an epoch counter, and DataLoader RNG streams by
  name -- everything a `.ttck` param file lacks for true resumption.
- **The headline test**: the DD-20 falsification is CLOSED. Adam resume
  now matches an uninterrupted 50-epoch run BYTE-EXACTLY
  (train 20 -> save -> fresh objects -> load -> train 30 == train 50).
- **Position-keyed moments**: optimizer state keyed by param position in
  construction order (id() cannot cross process boundaries); new
  `get_state`/`set_state` on SGD/Adam/AdamW validate positions & shapes.
- **Loud, never half-restored**: E0649..E0658 cover missing/extra
  sections, type/class mismatches, unknown loader names and
  param-only-files-as-state. All refusal paths are tested.
- Resume semantics documented honestly: fit() restarts its History;
  loader RNG streams continue at the saved point; schedulers continue
  the exact lr trajectory.

Suite: **414 tests** (was 402).

## [0.8.0] — 2026-09-23 (Milestone 8: deterministic binary checkpoints)

- **Format v2 `.ttck`** (DD-20): zip of `manifest.json` + raw f32 `.npy`
  entries with FIXED zip timestamps and sorted names -- byte-identical
  saves are part of the format, just like v1's JSON. np.savez_compressed
  was evaluated and rejected (timestamps/header nondeterminism).
- **Unified loading**: `load()`/ `load_into()` sniff CONTENT (zip magic),
  not extensions; both formats load interchangeably; a deliberately
  mis-named `.weird` file is covered by test.
- **Manifest integrity**: extra/missing entries, declared-vs-actual shape
  lies, format/version keys, truncation -> specific errors E0645--E0648.
  No corruption path silently loads partial garbage.
- **`save_bin()`/`load_bin()`** public API alongside `save()`/`load()`.
- Honest size claim, TEST-pinned: on a 10k-param model the binary file is
  <25% of the JSON size; tiny models are header noise either way and the
  test deliberately uses the big one.
- Resume theorem re-proven for v2 byte-exactly (SGD, parameter-only) --
  and the DD-20 record of why Adam falsifies that theorem (fresh moments
  != continuing moments), so nobody writes it as an example later.

Suite: **402 tests** (was 391).

## [0.7.0] — 2026-09-23 (Milestone 8: validation splits + loop unification)

- **fit_loader(val_loader=...)** (DD-19): per-epoch validation passes via
  the SAME `_run_epoch` helper as training (optimizer=None => no
  backward/step) -- the fit()/fit_loader() epoch-shell duplication
  flagged in RETROSPECTIVES entry 6 is collapsed into one implementation.
  Val losses land in `history.val_losses` as a distinct series.
- **Eval-mode discipline**: the val pass runs under `model.eval()` and
  the previous mode is RESTORED afterwards, pinned by a Probe module that
  records `self.training` per forward (would catch both dropout-fired-
  during-val and left-in-eval-for-next-train-epoch).
- **EarlyStopping monitor=**: 'loss' (default, unchanged behavior),
  'val_loss', or any metric name. monitor='val_loss' without val_loader
  is E0643 (explicit, no silent fallback); unknown names are E0644.
- A semantic regression test: with deliberately label-flipped validation
  data, val loss RISES while train falls, and ES(monitor='val_loss')
  stops early -- exactly the overfit signal the feature exists for.

Suite: **391 tests** (was 385).

## [0.6.0] — 2026-09-23 (Milestone 8 expansion: mini-batching + LR schedules)

- **DataLoader / TensorDataset** (DD-18): mini-batch iteration over tensor
  pairs. `shuffle=True` draws from a dedicated seeded RNG per loader --
  deterministic across identical loaders, never touching global RNG state.
  The final partial batch is YIELDED (never padded, never silently
  dropped) unless `drop_last=True`; both semantics pinned by tests.
- **fit_loader()**: per-batch optimizer steps, per-epoch History records
  the MEAN batch loss (not the last batch's -- the silent misreport this
  tranche's tests exist to forbid). Metrics are averaged per epoch;
  pluggable schedulers are stepped once per epoch, pinned by a test that
  would catch per-batch stepping immediately.
- **StepLR / ExponentialLR / CosineAnnealingLR**: computed from the
  construction-time base_lr each step (exact recompute -- no compounding-
  float drift), so tests pin the exact lr sequence to 1e-12. Cosine
  clamps at t_max.
- From Time-T code: `train.TensorDataset/DataLoader/fit_loader/StepLR`
  demonstrated in `examples/10_minibatch_lr_decay.tt` (final loss ~0.0056,
  byte-identical on AST/IR-O0/IR-O1).
- An unglamorous-but-important negative test: a test that annealed SGD
  escapes divergence where constant-lr SGD with the same base lr explodes
  -- proving lr actually wires through scheduler.step().

Suite: **385 tests** (was 367).

## [0.5.0] — 2026-09-23 (Milestone 7 expansion: Conv2D, Embedding, AdamW)

Layers grow real vision/NLP shapes (DD-17):
- **Conv2D** + `conv2d` functional op: NCHW cross-correlation with stride
  and zero-padding; im2col forward with a hand-written col2im backward as
  a single tape op. Backward is finite-difference-checked from x, weight
  AND bias in three stride/padding configurations (the layout mismatch
  caught during development -- channel-major vs kernel-major col order --
  is exactly what the forward test against a 6-loop reference exists for).
- **Embedding**: index lookup with `np.add.at` scatter-add backward;
  duplicate indices accumulate (explicit regression test). Out-of-range
  indices are clear NNError diagnostics, never silent clamps.
- **optim.AdamW**: decoupled weight decay; with wd=0 it is BIT-IDENTICAL
  to Adam (pinned by test).
- Reachable from Time-T code via the pre-bound `nn`/`optim` bridges;
  `examples/09_conv_center_detector.tt` demonstrates end-to-end conv
  training on all three engines with byte-identical output.

Differential suites expanded: conv forward-vs-naive reference (4
configs), conv backward finite differences (9 input-mode cd configs +
shape/error tests), embedding forward/grad tests, AdamW regression tests.

Suite: **367 tests** (was 340). No performance claims changed; the naive
conv loop is deliberately not benchmark-asserted.

## [0.4.0] — 2026-09-23 (Milestone 9 begins: native C-emitter; language reconciliations)

**Native codegen lands (DD-15, Milestone 9 v1 slice).** `time-t build
file.tt [-o out]` compiles a strict, byte-verified subset of Time-T to a
native executable via deterministic C11 + the system C compiler
(gcc/cc/clang); `time-t run file.tt --native` builds-and-runs. The subset:
Int/Float/Bool arithmetic + comparisons, Bool-only `&&`/`||`, String
literals (assign + print), let/var, if/else-if/else, while with
break/continue, typed functions + recursion, print of Int/Bool/String.
EVERY out-of-subset construct is a `native:`-prefixed diagnostic naming
the construct (Float printing, tensors, for loops, lambdas, f-strings,
imports, string concat...). No silent fallbacks. Semantics pinned to the
interpreter: `/` is true division; `%` is Python floor-modulo (-7 % 3 == 2
natively too). `build` graduated from "not implemented" stub to a real
command; `profile`/`export`/`package`/`doctor` remain honest stubs.

**Measured perf (documented with caveats, DD-15):** ~3000x on a 1M-iter
scalar while-loop (7.52 s interpreter vs 2.3 ms native binary) — this is
interpreter-overhead removal on scalar arithmetic, recorded in
benchmarks/results; tensors are NumPy-bound in both engines and see no
such ratio. The only claim is the one measured.

**Language reconciliation (DD-16), found by native differential tests:**
- `break`/`continue` now EXIST end-to-end (parser; E0215 static
  out-of-loop error; interpreter; IR lowerer + executor; optimizer treats
  them as segment boundaries; native C break/continue). They were absent
  everywhere before — never documented as working.
- Soundness fix: the type checker claimed `Int / Int -> Int` while the
  interpreter computed true division; `/` is now typed Float
  unconditionally. Optimizer gained the matching guard (`x/1` folds only
  for statically-Float operands). Runtime behavior of interpreted programs
  unchanged.

Suite: **340 tests** (was 318). New: test_native.py (16 incl. the
rejection battery + determinism), break/continue coverage in differential,
E0215 typechecker tests, div-typing regressions, div-by-one-fold guard.

## [0.3.0] — 2026-09-23 (Milestone 2 complete: module system)

**Modules land (DD-13).** `import a.b.c [as alias]` — resolves to
`<importer's dir>/a/b/c.tt`, load/typecheck/exec ONCE per canonical path,
fresh module scope (importer bindings provably invisible), and `fn main`
in a module is an ordinary export — NOT auto-run. Typed
exports across the boundary (E0211 lists real exports on unknown members);
E0210 import cycles, E0212 nested imports, E0213 missing files (message
includes the searched path), E0214 non-constant import path. Modules
flatten into the IR as prefixed functions plus an `__init__<dotted>` fn the
executor runs at-most-once in the main frame — AST / IR-O0 / IR-O1 remain
byte-identical (13 new differential tests + example `08_modules.tt`).

Clarified docs: parameters without type annotations are `<unknown>` and
arithmetic on `<unknown>` is a static E0204 (pre-existing rule, now
explicitly documented since modules made it visible).

Optimizer hardening (found & fixed by new differential tests): CSE now
rewrites call/marker args of an eliminated temp (two real bugs -- a
dangling `return %tN`, and `call:` args referencing a nop'd def -- both
pinned by regression tests, plus a corpus-wide no-dangling-temp invariant
scan). This is why DD-14 exists: correctness by differential evidence.

Suite: 318 tests (was 247 at v0.2.0). New: modules (12), parser+/typechecker
imports (+2), expanded engine-differential (10 -> 59: 8 example runs, 9
feature snippets, 88 generated programs), optimizer regression+invariant
tests (13 -> 18). All example `.expected` files verified against their
actual engine outputs. No performance numbers changed in this one.

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
