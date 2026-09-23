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

## Milestone 2 — Types + functions + modules 🟡 PARTIAL
- DONE: primitive types, function declarations, closures, type checking,
  scoping, immutyou/mutable bindings (`let`/`var`), diagnostics with
  location + expected/actual (`timet/diagnostics.py`).
- NOT DONE (tracked, not silently skipped): modules/`import`, generics,
  traits/interfaces, structs, enums, pattern matching.
- Tests: `tests/test_typechecker.py`.

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

## Milestone 6 — IR + optimization 🟡 PARTIAL
- DONE: `timet/ir.py` typed IR, lowering from typed AST, JSON
  serialization, `time-t inspect --ir`.
- NOT DONE: no optimization passes run on the IR yet (no constant folding,
  CSE, algebraic simplification, inlining, fusion). The IR is not executed;
  the interpreter still runs directly off the typed AST (DD-3). This is
  intentionally sequenced after the language + autodiff foundation was
  verified, per §3/§40 ("do not generate the entire project blindly").
- Tests: `tests/test_ir.py` (lowering correctness + JSON round-trip +
  determinism).

## Milestone 7 — Neural-network framework 🟡 MINIMAL START
- DONE: `timet/nn.py` — `Linear`, `ReLU`, `Sequential`, `MSELoss`;
  `timet/optim.py` — `SGD`. A linear-regression example and an XOR-MLP
  example train and converge (see `examples/`, `tests/test_nn.py` — trains
  for N steps and asserts loss decreases and crosses a threshold, with the
  actual measured final loss printed, not assumed).
- NOT DONE: Embedding, Conv1D/Conv2D, normalization layers, dropout,
  attention/transformer blocks, Adam/AdamW, cross-entropy/BCE losses,
  datasets/dataloaders, checkpoints, LR schedules, mixed precision.

## Milestone 8 — Training 🔲 NOT STARTED (beyond the minimal loop above)
- No dataset/dataloader abstraction, no checkpoint/resume, no LR schedule,
  no mixed precision, no training-run logging format yet.

## Milestone 9 — Native optimization 🔲 NOT STARTED
- No native codegen. Depends on Milestone 6 optimization passes existing
  first, then a lowering target (LLVM, C, or direct machine code) chosen
  and documented as a new Design Decision before implementation begins.

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

## Self-Criticism (Milestones 0–7, answering master prompt §39)

1. **What works?** Lexer, parser, type checker, tree-walking interpreter,
   NumPy-backed tensors with broadcasting, reverse-mode autodiff with
   gradient-checked backward rules, a JSON-inspectable IR (unexecuted), a
   minimal NN layer (Linear/ReLU/Sequential/MSELoss) + SGD that provably
   trains (loss decreases, checked in tests), and a CLI (`check/run/test/
   inspect/repl/bench`) with `--json` output.
2. **What does not work / doesn't exist?** Modules, generics, traits, structs,
   enums, pattern matching, static shape typing, IR optimization/execution,
   native codegen, GPU/ARM64/mobile backends, C ABI/FFI, ONNX, package
   manager, datasets/dataloaders, checkpoints, Adam/AdamW, attention layers.
3. **What is untested?** REPL interactive edge cases beyond the smoke test;
   very large tensors (memory pressure behavior); concurrent/thread-safety
   (no concurrency primitives exist, so none is claimed); fuzzing currently
   only covers lexer/parser, not IR (de)serialization or tensor indexing
   edge cases (§30 asks for broader fuzzing — tracked, not done).
4. **What is slow?** Everything, relative to a native compiler — this is a
   Python tree-walking interpreter over NumPy. Matmul is only as fast as
   NumPy's linked BLAS; there is no operator fusion, no kernel selection.
   Real numbers are in `benchmarks/results/`.
5. **What consumes excessive memory?** The dynamic autodiff tape retains
   every intermediate tensor referenced by a `requires_grad=True` graph
   until `backward()` runs or the graph is dropped; there is no
   checkpointing/recomputation yet, so training loops with `no_grad()` used
   incorrectly will grow memory. `timet.memory.stats()` at least makes this
   visible now.
6. **What architectural debt exists?** IR is computed but unused by
   execution (DD-3); single backend so the `Backend` abstraction is
   unverified by a second real implementation; no shape typing means shape
   errors surface at runtime, not compile time.
7. **What assumptions may be wrong?** That a brace-delimited syntax (DD-2)
   is the right long-term surface syntax for "concise AI/ML code" is
   unverified against real user programs; that method-call tensor ops
   (`x.sum()`) plus free functions (`sum(x)`) both being supported is worth
   the duplication is also unverified and could be simplified later.
8. **What should be redesigned before continuing?** Before Milestone 6's
   optimizer work begins in earnest, the interpreter should be switched to
   execute the IR (not the AST) so there is only one source of execution
   truth — currently listed as the Milestone 6 entry condition.
9. **What should NOT be implemented yet?** GPU/ARM64 backends, package
   manager, FFI/C ABI — all correctly deferred per the master prompt's own
   sequencing (§14, §32, §20 all say "do not implement all of these
   immediately" / plan-only for now).
10. **What evidence supports current claims?** `pytest -q` output (attached
    below in TESTING.md instructions to reproduce), `benchmarks/results/*`
    raw JSON, and `examples/*.expected` byte-exact output comparisons — all
    reproducible by re-running the commands in TESTING.md, not asserted from
    memory.
