# Time-T — Design Decisions

This document records the concrete engineering decisions made while building
Time-T, and the reasoning behind them. It is updated as the project evolves.
Decisions here are binding until explicitly revisited and superseded by a new
entry (old entries are never deleted, only marked superseded).

---

## DD-1: Implementation host language for the compiler/runtime prototype

**Decision:** The Time-T compiler, type checker, IR, interpreter/runtime, and
tensor/autodiff engine are implemented in Python 3.11, with NumPy used as the
underlying dense-array math library for CPU tensor storage and kernels.

**Status:** Active (Milestones 0–5).

**Why:**
- Priority order from the master prompt is Correctness > Architecture >
  Simplicity > Testability > Performance > ... Python + pytest gives the
  fastest path to a correct, well-tested, inspectable vertical slice.
- NumPy is a mature, independently-tested numerical library. Using it as the
  storage/kernel layer for Time-T's own `Tensor` type lets us focus early
  effort on Time-T's *own* semantics (grad tracking, shape/dtype rules,
  broadcasting rules, the language surface) instead of re-implementing BLAS.
- A tree-walking interpreter over a typed AST is the simplest thing that can
  be made fully correct and fully tested before any optimizing compiler work
  begins (see master prompt §40, "do not start by generating thousands of
  lines of code").

**What this decision does NOT mean:**
- It does not mean Time-T "is" Python. Time-T has its own grammar, its own
  parser, its own type checker, and its own execution semantics. Python is an
  implementation detail of the *current* prototype, exactly the way early
  versions of many production languages were bootstrapped in a host language.
- It does not mean NumPy semantics are Time-T semantics. Time-T's `Tensor`
  wraps `ndarray` but defines its own dtype defaults, its own autodiff tape,
  and its own error messages.

**Revisit trigger:** Milestone 9 (native optimization) and Milestone 10
(ARM64/mobile) will require a native (non-interpreted) execution path. At
that point the typed IR (already emitted, see DD-3) becomes the input to a
native lowering pass. This is tracked in ROADMAP.md and is explicitly *not*
attempted yet.

---

## DD-2: Concrete, brace-delimited syntax (not indentation-sensitive)

**Decision:** Time-T v0.1 syntax uses `{ }` blocks and explicit keywords
(`fn`, `let`, `var`, `if`, `else`, `while`, `return`), not Python-style
significant whitespace.

**Why:**
- Indentation-sensitive grammars add real parser complexity and edge cases
  (mixed tabs/spaces, continuation lines, nested blocks) that are a poor use
  of effort during the foundational milestone, where the priority is a
  *correct* lexer/parser, not a maximally terse one.
- Brace syntax is still concise (per §5 philosophy: fewer lines, more
  expressive operations, clear semantics) — verbosity comes from missing
  language features (generics, pattern matching, modules), not from `{ }`.
- This keeps the door open to an alternate/optional indentation-based syntax
  as a *front-end* concern later (a second parser could target the same AST)
  without having designed the whole language around it prematurely.

**Revisit trigger:** If, after Milestone 7 (framework ergonomics), verbosity
is shown (with real examples) to be a genuine usability problem, propose a
syntax revision as a new dated design decision — do not silently change
syntax underneath existing examples/tests.

---

## DD-3: IR is generated and inspectable, but not yet executed

**Decision:** Time-T emits an explicit three-address-code style typed IR
(`timet/ir.py`, `TirProgram`) for every successfully type-checked program,
and it is dumped via `time-t inspect <file> --ir`. However, the interpreter
(`timet/interpreter.py`) executes the **typed AST directly**, not the IR.

**Why:**
- Master prompt §12 requires the IR to be "inspectable, serializable,
  testable" from early on — building it now, in parallel with the AST
  interpreter, means IR shape mistakes are caught immediately by IR unit
  tests instead of being discovered later when optimization passes are
  bolted onto a rushed representation.
- Building an IR-executing VM *and* an optimizer *and* getting it all
  correct in the same milestone as "does the language exist at all" would
  violate the explicit instruction not to generate the whole system in one
  blind pass (§3, §40).

**Honesty note:** This is real architectural debt, listed in the Milestone 1–5
self-criticism (see ROADMAP.md §Self-Criticism). No optimization pass
(constant folding, CSE, fusion, etc.) currently runs on the IR. The IR is
correct and tested, but currently "read-only" — a foundation, not a finished
optimizing pipeline.

**Revisit trigger:** Milestone 6 (IR + optimization) makes the interpreter
consume the IR (or a bytecode lowered from it) instead of the raw AST, and
adds the first real optimization pass (constant folding) with before/after
tests proving semantics are preserved.

---

## DD-4: Autodiff is a dynamic tape (Wengert list) via operator overloading

**Decision:** Automatic differentiation is implemented as a dynamic
computation graph recorded at runtime by `Tensor` operator overloads
(`timet/tensor.py`, `timet/autodiff.py`), not as a static-graph transform of
the IR.

**Why:**
- This is the well-understood "eager mode autograd" design (as used by
  PyTorch, autograd, micrograd). It is small, testable in isolation, and
  every op's backward rule can be gradient-checked independently against
  finite differences (§10, §18, §29).
- A static-graph/IR-level autodiff transform is a legitimate *future*
  architecture (useful for whole-program optimization of training loops) but
  is materially harder to get correct first, so it is deferred.

**Revisit trigger:** Milestone 6+ may add a graph-mode autodiff that
differentiates the IR directly, used for fusion/optimization; the dynamic
tape remains as the default/eager execution path either way (dual approach
is normal in production ML systems).

---

## DD-5: Only one backend exists today: `cpu-numpy`

**Decision:** `timet/backend.py` defines a `Backend` capability interface.
Exactly one implementation exists: `NumpyCpuBackend`. No GPU, ARM64-specific,
SIMD-specific, CUDA, Vulkan, Metal, OpenCL, or NPU backend exists.

**Why:** Master prompt §14 explicitly says "do not implement all of these
immediately" and asks for a capability-queryable interface so the
compiler/runtime can pick an execution path later. Building the interface
now, with one honest implementation, avoids both (a) hard-coding
CPU-only assumptions everywhere and (b) claiming hardware support that has
not been written or measured (§15: "do not claim mobile performance until it
is measured on real hardware").

---

## DD-6: Type checker performs local/structural inference, not Hindley–Milner

**Decision:** The type checker (`timet/typechecker.py`) infers literal types,
propagates types through expressions bottom-up, checks function signatures
and `let`/`var` annotations, and rejects mismatches — but does not do full
unification-based generic inference. Generics, traits/interfaces, enums,
pattern matching, structs, modules, and a package system are **not
implemented** in this milestone.

**Why:** §6 explicitly says "do not implement every feature immediately;
create a dependency-aware roadmap." Building a correct, well-tested
monomorphic type checker first is the honest, achievable step; a generic
system without a working base language would be unverifiable.

**Tracked in:** ROADMAP.md, Milestone 2 (partially done: functions + basic
types are implemented; generics/traits/structs/enums/pattern
matching/modules are explicitly deferred to later, listed milestones).

---

## DD-7: Numeric tolerance policy

**Decision:** All gradient/numerical differential tests use:
- `rtol = 1e-3`, `atol = 1e-4` for finite-difference gradient checks against
  float32 tensors (finite differences are inherently noisy in float32; the
  tolerance is chosen to be tight enough to catch real bugs, loose enough to
  avoid false failures from float32 rounding).
- Exact equality is required for integer arithmetic.

Every gradient-check test reports max absolute error and max relative error
on failure (never just "assert equal"), per §18.

---

## DD-8: CLI commands that are not implemented say so explicitly

**Decision:** `time-t build`, `time-t profile`, `time-t export`,
`time-t package`, and `time-t doctor` are present in the CLI's `--help` but,
when invoked, print a clear "not implemented yet" message to stderr, a
machine-readable `{"status": "not_implemented"}` JSON in `--json` mode, and
exit non-zero. They never silently succeed or print fabricated output.

**Why:** §26 and §38 ("do not mark incomplete systems as complete") — an
AI agent or human scripting against the CLI must be able to trust that a
non-error exit code means the operation actually happened.
