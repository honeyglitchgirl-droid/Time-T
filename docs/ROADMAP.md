# Time-T — Roadmap

Versioning follows master prompt §37. Status reflects reality as of this
commit, verified by the test suite (`pytest -q`) — see TESTING.md for how to
reproduce every claim below.

## Milestone 0 — Architecture ✅ DONE
- ARCHITECTURE.md, LANGUAGE.md, ROADMAP.md, TESTING.md, DESIGN_DECISIONS.md
  written before substantial implementation, per §4.
- 11 architectural risks + mitigations recorded (ARCHITECTURE.md §11).

## Milestone 1 — Lexer + parser + minimal language ✅ DONE
- `timet/lexer.py`, `timet/parser.py`, `timet/ast_nodes.py`.
- Tests: `tests/test_lexer.py`, `tests/test_parser.py`.
- Fuzz: `tests/test_fuzz.py` (random byte streams + mutated valid programs
  must never crash the process — only raise `LexError`/`ParseError`).

## Milestone 2 — Types + functions + modules ✅ DONE (v0.3.0)
- DONE: primitive types, function declarations, closures, type checking,
  scoping, immutable/mutable bindings (`let`/`var`), `break`/`continue`
  with static out-of-loop rejection (E0215, v0.4.0), true-division typing
  (`/` always Float, soundness fix DD-16), diagnostics with
  location + expected/actual (`timet/diagnostics.py`); file modules
  (`import a.b.c [as m]`, typed exports, import-once, fresh scope —
  DD-13, `timet/modules.py`).
- NOT DONE (tracked, not silently skipped): generics, traits/interfaces,
  structs, enums, pattern matching, static shape types.
- Tests: `tests/test_typechecker.py`, `tests/test_modules.py`.

## Milestone 3 — Executable runtime ✅ DONE (interpreter)
- `timet/interpreter.py` tree-walking evaluator executes typed AST: `let`,
  `var`, `fn`, closures, `if`/`while`, arithmetic, comparisons, `print`.
- Tests: `tests/test_interpreter.py`, `tests/test_examples.py` (example
  programs run end-to-end and their stdout is checked byte-for-byte against
  `examples/*.expected`).

## Milestone 4 — Arrays / tensors ✅ DONE (CPU, NumPy-backed)
- `timet/tensor.py`: construction, dtype, shape, broadcasting, matmul,
  reshape/transpose/permute, reductions (sum/mean/max/min), elementwise
  math (exp/log/sqrt), activations (relu/sigmoid/tanh/softmax), basic
  indexing/slicing.
- Tests: `tests/test_tensor.py`, including differential tests against
  NumPy's own reference operations (§29) with documented tolerances
  (`tests/numerics.py`, DD-7).

## Milestone 5 — Autodiff ✅ DONE (reverse-mode, dynamic tape)
- `timet/autodiff.py`, `timet/tensor.py` backward rules for every
  implemented op.
- Tests: `tests/test_autodiff.py` — every backward rule is checked against
  finite differences (central difference, h=1e-4) and reports max
  abs/rel error on failure, never a bare assert (§18, §29).
- Worked example from the master prompt itself is a literal regression
  test: `x=[1,2,3]` grad, `y=sum(x*x)`, backward → `grad == [2,4,6]`.

## Milestone 6 — IR + optimization ✅ DONE (v0.2.0; hard limits documented)
- DONE (v0.1.0): `timet/ir.py` typed IR, lowering from typed AST, JSON
  serialization, `time-t inspect --ir`.
- DONE (v0.2.0):
  - The IR is **executable**: `timet/ir_exec.py` runs IR directly
    (structured block-tree walk; `var`s via named storage frames, `let`s as
    SSA temps). CLI: `time-t run <file> --via-ir [-O 0|1]`.
  - The IR is **optimizable**: `timet/optimize.py` implements constant
    folding, type-aware algebraic simplification, segment-local CSE and
    copy propagation, and DCE — all documented with their exact safety
    rules in DD-12 (e.g. `x*0` is never folded: NaN/Inf; `1/0` stays a
    runtime error).
  - Lowering extended: while-conditions live inside the loop region,
    for-loops, `no_grad` regions, kwargs, dynamic calls, and top-level
    statements (a synthetic `__main__` function).
  - **Differentially verified**: AST interpreter vs IR-O0 vs IR-O1 must
    produce byte-identical stdout on every example program AND on
    generated random programs (`tests/test_ir_exec_diff.py`). This gate has
    already caught real bugs (CSE leaving a `return` arg dangling;
    an eliminated temp still referenced by a call) -- both pinned as
    regression tests. gate: DD-14.
- DELIBERATE LIMITS (executor fails loudly, never guesses): lambdas,
  nested `fn` decls, if-expressions are not lowered; `&&`/`||` are eager in
  IR; no inlining/fusion/LICM (needs dataflow/CFG — DD-9, DD-12).
- Tests: `tests/test_ir.py`, `tests/test_ir_exec.py`,
  `tests/test_optimize.py`, `tests/test_ir_exec_diff.py`.

## Milestone 7 — Neural-network framework 🟡 PARTIAL (expanded v0.2.0)
- DONE: `timet/nn.py` — layers `Linear`, `ReLU`, `Sigmoid`, `Tanh`,
  `Softmax`, `Flatten`, `Dropout` (inverted, seeded = deterministic,
  train/eval aware), `Sequential`; losses `MSELoss`, `CrossEntropyLoss`
  (log-softmax + one-hot NLL, gradient finite-difference checked),
  `BCELoss`; `Module.train()/eval()` recursion; functional conveniences
  (`nn.cross_entropy_loss`, ...). `timet/optim.py` — `SGD`, `Adam` (with
  bias correction; Adam convergence is asserted by a test that must reach
  a loss threshold on XOR, with the measured value printed).
- DONE (v0.2.0): the library is reachable FROM TIME-T CODE via the
  pre-bound `nn`/`optim`/`train` globals (DD-10's documented bridge until a
  real module system exists) — `examples/07_xor_classifier.tt` trains a
  2-8-2 MLP+Adam+cross-entropy classifier to 100% XOR accuracy and runs
  byte-identically on all three execution engines.
- NOT DONE: Embedding, Conv1D/Conv2D, normalization layers,
  attention/transformer blocks, AdamW, weights/init schemes beyond
  Kaiming-uniform, mixed precision.

## Milestone 8 — Training 🟡 PARTIAL (started v0.2.0)
- DONE: `timet/train.py` — `accuracy()` metric, `History`,
  `fit()` (full-batch loop: forward → backward → step → zero_grad, logging
  + pluggable metrics) and `EarlyStopping`. `timet/checkpoint.py` —
  save/load/resume of parameters as deterministic, versioned JSON
  (DD-11); resume correctness is proven by the train-20/save/restore/
  train-30 == uninterrupted-train-50 test.
- NOT DONE: datasets/dataloader abstraction (fit is FULL-BATCH ONLY and
  says so), mini-batching, LR schedules, validation splits, mixed
  precision, training-run log format, binary checkpoint format (needed for
  larger models).

## Milestone 9 — Native optimization 🟡 PARTIAL (started v0.4.0)
- DONE (v0.4.0): a working native path — `timet/native.py` C-emitter over
  a strict, byte-verified subset (DD-15; scalars/control flow/typed
  functions/recursion/print of Int-Bool-String), CLI `build` (upgraded
  from stub to real) and `run --native`; gcc compile ~50 ms for small
  programs; `benchmarks/run_benchmarks.py` records the measured
  ~3000x scalar-loop interpreter-overhead removal (caveat-recorded; tensor
  workloads are NumPy-bound either way and see no such ratio).
- NOT DONE (tracked): tensors in the native backend, Float printing
  (shortest-repr formatting parity), `for` loops natively, f-strings and
  string concat natively, modules natively, whole-program optimization
  (inlining, const-prop across functions — needs CFG/dataflow, DD-12),
  loop optimizations, backend kernel selection, multi-objective
  optimization, LLVM-vs-C re-targeting decision (deliberately deferred,
  DD-15 explains why C-first keeps that door open).

## Milestone 10 — ARM64 / mobile 🔲 NOT STARTED
- No ARM64-specific work, no quantization, no mobile runtime. No performance
  claim about mobile/ARM64 exists anywhere in this repo (§15).

## Milestone 11 — Interoperability 🔲 NOT STARTED
- No C ABI, no FFI, no ONNX import/export, no Python-embedding API beyond
  "the whole compiler happens to be written in Python" (which is an
  implementation detail, not an interop feature).

## Milestone 12 — Stable release 🔲 NOT STARTED

---

## Benchmarks status (§19)
`benchmarks/run_benchmarks.py` measures, on **this sandbox's actual CPU**
(labelled in the output JSON with `platform.processor()`/`platform.uname()`),
wall-clock time and throughput for: scalar add loop, vector add, matmul at a
few sizes, a tensor reduction, and one autodiff backward pass. Raw results
are written to `benchmarks/results/*.json` with a timestamp and are
regenerated by running the script — no numbers in documentation are
hand-typed or estimated.

## Model-scale status (§16)
Only capability **A (can be represented)** and **C (can run inference)** have
been exercised, and only at the smallest scales (≤ a few thousand
parameters, e.g. the XOR MLP and linear-regression examples) — this is a
tree-walking Python interpreter; no claim is made about 1M+ parameter
training throughput because it has not been measured. This will be updated
only after real measurements are taken at larger scale.

## Self-Criticism
Milestone retrospectives (master prompt §39, all 10 questions) live in
docs/RETROSPECTIVES.md — one dated entry per milestone band, kept as history.
