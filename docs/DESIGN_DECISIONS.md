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

---

## DD-9: IR is executed via a structured block-tree interpreter (supersedes part of DD-3)

**Status update (v0.2.0):** DD-3's "the IR is not executed" is no longer
true. The IR is now directly executable (`timet/ir_exec.py`) and optimized
(`timet/optimize.py`); everything else in DD-3 (AST interpreter remains the
semantic reference; no backend IR lowering yet) still holds.

**Decision:** IR control flow uses STRUCTURED markers
(`if_begin`/`else`/`if_end`, `while_begin`/`while_check`/`while_end`,
`for_begin`/`for_end`, `nograd_begin`/`nograd_end`) which the executor
parses into a block tree. Loop conditions are lowered INSIDE the loop
region, so they re-execute every iteration by construction (a flat
"evaluate-cond-once" lowering would be an infinite loop). Mutable `var`s
lower to named storage (`var_def`/`store`/`load`) in chained frames
(mirroring the interpreter's Environment chain); immutable `let`s stay
SSA temps.

**Deliberate limits (executor raises `IrExecError`, never guesses):**
lambdas, nested `fn` declarations, and if-EXPRESSIONS are not lowered;
`&&`/`||` are eager (not short-circuiting) in IR; a `var` declared in a
nested block shares the flat frame storage rather than being scoped out.
The differential suite (`tests/test_ir_exec_diff.py`) pins AST vs IR-O0 vs
IR-O1 output equality on every example plus generated random programs, so
any future extension of the lowered subset is verified, not assumed.

**Why structured markers instead of basic blocks + jumps:** the IR consumer
list so far (inspector, executor, local optimizations) is better served by
the simplest representation that is still executable. A CFG form is future
work (DD candidate for Milestone 9 native codegen).

---

## DD-10: Host-library exposure (`nn` / `optim` / `train`) as pre-bound global values

**Decision:** Time-T programs reach the neural-net library through three
pre-bound global names (`nn`, `optim`, `train`) — plain Python module
objects injected into the global scope by both engines. `nn.Linear(2, 8)`
is a field access + a call of a host callable; no new syntax and no module
system was invented for this. Type checking treats these as `TUnknown`
(they are typed dynamically), which is documented in docs/LANGUAGE.md.

**Why:** Milestone 2's real module system (import, namespacing) is not
built yet, and faking one would violate §38. Pre-bound values give real,
testable NN capability from Time-T code TODAY (see
`examples/07_xor_classifier.tt`, which trains with Adam + cross-entropy and
runs byte-identically on AST, IR-O0, and IR-O1 engines) without pretending
to be a user-extensible import mechanism.

**Consequences:** user code cannot currently define its own modules; that
remains roadmap work. Static typing across the `nn.*` boundary is
explicitly deferred (TUnknown flows through, no false confidence).

---

## DD-11: Checkpoints are versioned, deterministic JSON (not a binary format)

**Decision:** `timet/checkpoint.py` saves parameters as a single JSON file:
`{"format": "timet-checkpoint", "version": 1, "tensors": {name: {dtype,
shape, data}}}` with sorted keys and round-trip-exact floats. Byte-identical
files for identical models (tested). `load_into` verifies format, version,
key sets (strict mode), and shapes, and reports the offending key name.

**Why:** the models Time-T can actually train today are tiny (examples are
2-8-2 MLPs). JSON is inspectable, diffable, dependency-free, and honest.
For large models this format is wrong (file size, parse cost) — the
ROADMAP lists a binary tensor format as future work, to be designed as its
own DD before implementation (§3: architecture before code).

---

## DD-12: The O1 optimizer uses only local, order-independent-safe passes

**Decision:** `timet/optimize.py` implements constant folding, restricted
algebraic simplification, segment-local CSE, segment-local copy
propagation, and one-backward-sweep DCE. CSE/copy-propagation NEVER cross
structured-control markers (a value computed in one branch must not be
assumed available in another), loads are invalidated by intervening stores,
and algebraic rules are type-aware:

- `x + 0` / `x - 0` folded for Int only (for Float it changes `-0.0` to
  `+0.0`);
- `x * 1` folded for Int and Float (exact IEEE identity, NaN-safe);
- `x / 1` folded for Float-typed ops only (Python `7 / 1` is a Float; an
  Int-div result copy would change the value's type);
- `x * 0` is NEVER folded (NaN/Inf inputs);
- `1 / 0` is never folded to a compile-time error — it stays a runtime
  error, exactly as in the interpreter.

**Why:** these five passes are individually verifiable and compose into a
pipeline whose total effect is checked END-TO-END by the differential suite
(IR-O0 vs IR-O1 byte-identical output on all examples + random programs).
Anything interprocedural or requiring dataflow analysis (inlining, LICM,
fusion) is explicitly out of scope until a CFG form exists (DD-9).

---

## DD-13: Modules are files with typed exports; IR flattens them with mangled names

**Decision (v0.3.0):** `import a.b.c` resolves to the file
`<importer's dir>/a/b/c.tt`; the bound name is `c` (or the `as` alias).
A module is parsed, type-checked, and executed exactly once per canonical
path (Python-style import-once) in a FRESH global scope containing only the
engine's builtins/modules — it never sees the importer's variables
(enforced by test). Import cycles raise E0210 naming the cycle; imports
are top-level only (E0212). ALL top-level `fn`/`let`/`var` bindings are
exported; there are no visibility modifiers, no packages/`__init__`, no
`from x import y` (all deferred, all clearly errored if attempted... except
`from`, which parses as neither — it raises a normal parse error).

**Static typing across the boundary is REAL, not TUnknown:** the type
checker recursively checks the module file and records each export's type
(`TModule(dotted, exports)`); `mathlib.add(1, 2)` is checked against the
exported `TFunction` signature, and `mathlib.nope` is a compile-time E0211
listing the actual exports. This deliberately does better than DD-10's
TUnknown bridge — `nn`/`optim`/`train` pre-bound globals remain TUnknown
until they are re-packaged as Time-T modules (tracked, not yet done).

**IR representation:** modules are FLATTENED into the single-program IR —
module functions are prefixed with the dotted module name
(`mathlib.add`), module top-level statements become a synthetic
`__init__<dotted>` function whose every binding lives in prefixed named
storage, and `import x` lowers to `call:__init__x` at the statement's
source position. The IR executor special-cases `__init__*`: at-most-once
execution, directly in the main frame (so prefixed bindings are visible
program-wide). Byte-identical output with the AST engine is pinned by
`tests/test_modules.py` and the example-08 differential test on all three
engines.

**Why file-based with no search path:** the smallest design that covers
real program organization without inventing a package manager (ROADMAP
Milestone 12 territory). Resolution is deterministic and local:
relative-to-importer only.

**Known limits:** higher-order use of module functions in IR (passing
`mathlib.add` as a value) raises the same clear error as other
higher-order use (DD-9); module-level mutable state is process-global
(same as Python, but worth stating); error spans for errors INSIDE an
imported module point at the module file via the importer-path chain but
the diagnostic message does not yet print the full import stack.

---

## DD-14: Optimizer/executor correctness is established by differential testing, not by proof

**Decision (v0.3.0):** every nontrivial program shipped in the repo (8
examples + 13 hand-written feature snippets + 40 seeded random straight-line
programs, growing) MUST run byte-identically on the AST interpreter, the IR
executor at -O0, and the IR executor at -O1 (`tests/test_ir_exec_diff.py`).
Optimization passes are additionally unit-tested for their documented
safety rules (e.g. `1/0` is NOT folded, CSE never crosses markers or
stores — both pinned by regression tests from actual bugs found).

**Why:** the optimizer passes are small and local, but correctness bugs
(In1: CSE not rewriting call args of eliminated temps; In2: an elimination
that leaves `return %t8` dangling because a control marker swallowed the
last def) were found only by running the engines side by side. Differential
testing is the honest way to gain confidence without a full formal
verification, and it scales to each new pass: new passes must keep the
differential suite green or they do not land.

**Known limits:** the two random generators cover arithmetic + if/while
at fixed shapes; they do not generate nested closures, module imports, or
tensor-shuffle programs (each covered only by snippet/example tests).
`f`-strings and `if`-expressions are
deliberately not in the IR executor (executor raises clearly, tests assert
the error is raised, not a wrong result).

---

## DD-15: Native codegen is a C-emitter over a strict, loudly-bounded subset

**Decision (v0.4.0):** Milestone 9 begins with a C11 emitter
(`timet/native.py`): typed AST → deterministic C source → the system C
compiler (`gcc`/`cc`/`clang`). The v1 subset is intentionally SMALL and
every out-of-subset construct is rejected with a `NativeError` naming the
construct and its span — there is NO silent fallback and NO partial
lowering: a program either compiles natively as a whole or builds fail.

In the subset (all byte-verified against the AST interpreter in
`tests/test_native.py`): Int (int64 — see divergence note), Float (double),
Bool, String literals (assign + print only), `+ - * / %` and comparisons,
`&&`/`||` on Bool operands, `!`, unary `-`, `let`/`var` with block scoping,
`if`/`else if`/`else`, `while` (+`break`/`continue`), typed functions with
recursion, `print` of Int/Bool/String.

Deliberately NOT in v1 (each with its own error message, tests pin them):
printing Float values (Python's shortest-repr float formatting has no
trivial C equivalent; floats COMPUTE fully — only printing is excluded),
tensors and all tensor builtins, all builtins except `print`, `for` loops,
lambdas/closures-as-values, f-strings, imports/modules, string
concatenation/comparison, mixed Int/Float `%`.

Semantics pinned to the interpreter: `/` is ALWAYS true division (double),
`%` is Python floor-modulo (emitted as a `tt_mod` helper — `-7 % 3 == 2`
in both engines, unlike C's `-1`).

**Known divergences (documented, tested at the boundary of the defined
contract, not fixed):** native Int is int64 with two's-complement wrap —
Time-T's Python Int is arbitrary precision; programs exceeding 2^63
diverge. This is stated in LANGUAGE.md and DD-15, not hidden.

**Why C emission instead of LLVM/direct machine code:** the master prompt
(§12, Milestone 9) says to start with CPU-focused optimizations, not to
pick a heavyweight IR first. A C emitter reuses the host compiler's 40
years of register allocation/instruction selection at zero dependency
cost, keeps the generated code READABLE (auditable: `time-t build`
prints C-line counts; source lands next to the binary in debug scenarios),
and leaves the LLVM decision genuinely open — the IR optimizer pipeline
(DD-12/DD-14) is the actual substrate for future whole-program passes,
and nothing in this decision blocks re-targeting them.

**Performance honesty:** the measured ~3000x scalar-loop speedup
(benchmarks/results) is interpreter-overhead removal on PURE SCALAR
arithmetic; tensor workloads are NumPy-bound in both worlds and see no
such ratio. No claim beyond that exists anywhere in the repo.

---

## DD-16: `break`/`continue` are statements; `/` is true division at the TYPE level

**Decision (v0.4.0):** two language-semantics fixes, both found by the
native backend's differential testing (the exact scenario DD-14 exists for):

1. `break`/`continue` now exist — parser, type checker (out-of-loop use is
   E0215), interpreter, IR lowerer + executor (control-flow exceptions
   within loop regions), optimizer (they are segment boundaries), and the
   native emitter (C `break`/`continue`). Before v0.4.0 they were not in
   the language at all (docs never claimed them; the gap went unnoticed
   until tests needed them).
2. The type checker previously inferred `Int / Int -> Int` while the
   interpreter computed true division (`7 / 2 == 3.5`) — a latent
   soundness bug invisible to the interpreted engines but fatal to typed
   codegen (a native `int64_t q = ...` would silently truncate). Now `/`
   is typed `Float` unconditionally. The optimizer grew a matching guard:
   `x / 1` folds to `x` only when x is static Float (an Int operand would
   change the value's type); param-typed operands are conservatively not
   folded because parameter types aren't representable in the current IR
   (tracked as a DD-12/DD-15 follow-up).

---

## DD-17: Conv/Embedding as custom tape ops; AdamW as decoupled decay

**Decisions (v0.5.0):**
1. `conv2d`/`Conv2D` is explicit im2col+matmul with a hand-written col2im
   backward, integrated into the tape as ONE custom node (not decomposed
   into fine-grained tracked ops). Rationale: gradients through a patch-
   extraction composition would mean teaching `index`/slice ops full
   gradient semantics first -- a much bigger surface for the same result.
   Correctness is established by central finite differences from ALL
   THREE inputs in three stride/padding configurations, plus a forward
   check against a 6-nested-loop reference (tests/test_conv2d.py).
   HONEST SCOPE: single conv op only (NCHW); no groups, dilation,
   transposed conv, or 1-D/3-D variants; loops are clarity-first (no
   speed claim; benchmark table untouched).
2. `Embedding` backward is `np.add.at` scatter-add -- duplicate indices
   ACCUMULATE (pinned by a dedicated test; a copy-overwrite backward is
   the classic silent bug here).
3. `AdamW` implements the Loshchilov & Hutter decoupled penalty
   (`p -= lr*wd*p` OUTSIDE the adaptive term), not L2-in-grad. With
   `weight_decay=0` it is BIT-IDENTICAL to `Adam` (pinned by test) -- same
   float64 path modulo an exact `-0` term.

**Process lesson recorded for future contributors (cost a debugging
cycle):** finite-difference checks of float32 pipelines must not route
the perturbed scalar through float32 tensors; summing ~250 float32
outputs quantizes the scalar at ~1e-2, which swamps an h=1e-4 difference
quotient and manufactures a fake "backward bug". tests/contest_conv2d.py
documents the pattern: perturb in float64, evaluate the forward in
float64 (`_conv2d_forward` directly), and keep the autodiff comparison
at the repo's standard 1e-3 tolerance.


---

## DD-18: DataLoader + fit_loader + LR schedules (Milestone 8)

**Decisions (v0.6.0):**
1. `TensorDataset` is the only dataset abstraction: (x, y) Tensors aligned
   on axis 0. No transforms, no dict-of-columns, no disk formats yet --
   the honest minimum, rather than a fake-general Dataset protocol.
2. `DataLoader` NEVER pads and NEVER silently drops: the final partial
   batch is YIELDED unless `drop_last=True` (both pinned by test). Silent
   padding/dropping are the classic ways training curves quietly lie.
3. `shuffle=True` uses a dedicated `np.random.default_rng(seed)` created
   at construction: iterating twice yields different-but-deterministic
   permutations, and two loaders with the same seed walk the same stream
   (pinned by test). The global RNG is untouched -- training cannot
   perturb library-level randomness elsewhere.
4. `fit_loader`'s epoch History entry is the MEAN of the epoch's batch
   losses (not the last batch's, which misreports curves) -- pinned by test
   with frozen params.
5. LR schedulers recompute `lr` from the CONSTRUCTION-TIME `base_lr` each
   step (`base * gamma ** floor(...)`); compounding floats (`lr *= gamma`)
   would make the schedule path-dependent and untestable as "exact
   sequence" -- our StepLR/ExponentialLR tests pin exact values to 1e-12.
6. `fit_loader(scheduler=...)` steps the scheduler ONCE PER EPOCH after
   that epoch's optimizer steps (textbook order). Per-batch stepping is
   a silent 100x-too-fast anneal; a test pins the exact final lr after
   2 epochs x 5 batches.
7. Schedulers work on any optimizer with an `.lr` attribute (duck-typed on
   purpose: SGD/Adam/AdamW all qualify). CosineAnnealing clamps at t_max.


---

## DD-19: Validation splits inside fit_loader (Milestone 8)

**Decisions (v0.7.0):**
1. `fit(val_loader=...)` computes a val pass per epoch via the same
   `_run_epoch` helper as training (optimizer=None -> no backward/step):
   ONE loop implementation for both phases, which is the only way
   "validation matches training" can stay a reviewable claim. The old
   fit()/fit_loader() duplication flagged in Entry 6 is now collapsed.
2. Eval-mode discipline is enforced for the val pass: `model.eval()`
   before, `model.train(<previous mode>)` after -- pinned by a probe-Module
   test. Silently leaving Dropout active through validation (or leaving
   the model in eval for the next train epoch) are both real-world bugs
   people ship; the test exists because this tranche nearly let the
   second one through during writing.
3. `EarlyStopping` gains NO new state; a `monitor=` name ("loss",
   "val_loss", or a metric key) picks the series at the fit() call site.
   Monitoring 'val_loss' without a val_loader is an explicit E0643 --
   never silently falling back to train loss, which is the failure mode
   nobody notices for weeks.
4. Val loss means simple mean over val batches (same approximation as
   train, DD-18 item); series live in `history.val_losses`, distinct from
   train  `history.losses`, so curves can't be confused downstream.


---

## DD-20: Binary checkpoint format v2 (.ttck)

**Decisions (v0.8.0):**
1. `.ttck` is a deterministic zip: `manifest.json` (sorted keys) + raw
   little-endian f32 `.npy` payloads per parameter, zip timestamps FIXED
   to the DOS epoch and entries name-sorted, so byte-identity across runs
   is part of the format (same guarantee v1 makes for JSON).
   np.savez_compressed was REJECTED as the format: it inherits zip
   timestamps and header nondeterminism, and two saves of identical
   parameters can differ at the byte level.
2. Content-sniffed loading (zip magic), never extension-sniffed --
   mis-named files still load (tested with a `.weird` extension).
3. The manifest is the single source of structure: unknown/extra/missing
   zip entries are E0648 structural errors, declared-vs-actual shape
   mismatches are E0648, wrong format/version keys are E0647/E0648,
   truncation is E0645. Corruption never silently becomes "a few params
   at random values".
4. All values stored f32; original dtype is DECLARED in the manifest but
   not restored (honest limit, listed IN the manifest itself). Today's
   entire parameter space is f32, so this loses nothing real.
5. **Finding recorded while testing (matters for every future claim):**
   the v1 "resume theorem" holds ONLY for parameter-stateless optimizers
   (SGD). Re-proving it for v2 with Adam FAILS -- fresh Adam moments take
   a different trajectory than continuous Adam (0.35 max param deviation
   observed on the canonical task). Optimizer-state checkpointing is now
   explicitly and prominently NOT DONE in both formats, and the v2
   theorem is pinned with SGD byte-exactly.


---

## DD-21: Training-state checkpoints (closing the DD-20 falsification)

**Decisions (v0.9.0):**
1. `save_state`/`load_state` write a v2-family binary file with a
   `training` section: optimizer moments (m/v/t), scheduler internals
   (last_epoch, base_lr), epoch counter, and DataLoader RNG states by
   name. The Adam-resume theorem that DD-20 FALSIFIED for param-only
   checkpoints now holds BYTE-EXACTLY and is the headline test.
2. Optimizer moments are keyed by PARAM POSITION (index in the
   optimizer's construction list), never by object id (ids don't survive
   load; positions do -- the same order-stability load_into already
   depends on). Optimizers gained `get_state()`/`set_state()`; SGD's
   state is config-only, Adam/AdamW carry arrays + timestep, AdamW adds
   weight_decay to config.
3. HALF-RESTORED RESUMES ARE REFUSED: file-has-optimizer-but-none-
   passed (E0651), type mismatch (E0652), set_state position/shape
   errors (E0653), optimizer-passed-but-file-has-none (E0654), scheduler
   analogues (E0655/E0656), unknown loader names (E0658), and param-only
   files fed to load_state (E0649) all ERROR. A quiet half-resume is the
   bug nobody notices for a week; Time-T chooses the loud path.
4. History/logs are NOT part of training state -- a resumed fit()
   restarts its History by design (stated in the file's own limits
   section). Best-epoch weights remain checkpoint.py's job.
---

## DD-27: Mobile INT8 Quantization and Lightweight Package Deployment (Milestone 10, Master Prompt §15, §24)

**Decision (v1.0.0):**
1. Dynamic INT8 linear quantization (`timet.mobile`):
   - Symmetric and asymmetric dynamic quantization mapping float32 $\rightarrow$ int8 $[-128, 127]$.
   - `QuantizedLinear` layer executing integer matrix multiplication (int32 accumulator)
     with dynamically scaled float32 output. Reduces parameter footprint by 4x.
   - `quantize_dynamic(model)` converts `nn.Linear` layers across `nn.Sequential` and custom containers.
2. Mobile standalone package format (`.ttm`):
   - Zip container bundling a JSON manifest with raw binary parameter buffers.
   - Zero-overhead loading via `load_mobile(path)` returning executable inference pipelines.
3. Language accessibility:
   - Exposed to Time-T programs via pre-bound `mobile` module.
   - Verified in `examples/15_mobile_inference.tt` with byte-identical output across AST, IR-O0, and IR-O1 engines.

---

## DD-31: Native Float Printing Support and User-Facing Multi-Engine Verification (v1.3.0, Master Prompt §25, §29)

**Decision (v1.3.0):**
1. Native Float Printing:
   - Implemented `tt_print_float` in the native C emitter prelude using `snprintf(..., "%.15g")` and shortest-repr formatting with `.0` suffix enforcement for whole floating-point numbers.
   - Matches Python's shortest-repr output byte-for-byte.
   - Removed the limitation in `native.py` and `tests/test_native.py` that prohibited printing `Float` expressions.
2. CLI `time-t verify`:
   - Added user-facing CLI command `time-t verify <file.tt> [--json]`.
   - Executes the program synchronously through the AST interpreter, IR-O0, and IR-O1 engines.
   - Certifies bit-identical stdout output across all three engines, reporting diagnostic diffs if any engine diverges.

---

## DD-30: Robust Runtime Error Guardrails (Audit Resolution, Master Prompt §27, §31)


**Decision (v1.2.0):**
1. Division and Modulo by Zero:
   - Integers and floating point division (`/`) by zero raise `RuntimeErr` with code `E0507` across the AST interpreter and `IrExecError` in the IR executor.
   - Modulo (`%`) by zero raises structured diagnostic `E0508`.
   - Replaces unhandled Python `ZeroDivisionError` with compiler-standard diagnostic reporting.
2. Recursion Limit Safety:
   - Both `Interpreter` and `IRExecutor` enforce configurable recursion safety depth limits (default 1000 frames).
   - Exceeding call depth raises diagnostic `E0509` ("maximum recursion depth exceeded") with span and location, eliminating unhandled `RecursionError` crashes.
3. Interactive REPL Enhancement:
   - `time-t repl` accepts both top-level statements (`let`, `var`, assignments) and expressions, maintaining persistent bindings throughout the session.
4. Native Build Boundary Transparency:
   - Documented the exact C-emitter supported language boundary in `docs/NATIVE_SUBSET.md`.
5. Self-Contradiction Resolution in Documentation:
   - Reconciled `README.md` to reflect actual implemented features and align with `docs/ROADMAP.md`.

---

## DD-29: User-Defined Struct Declarations and Compound Types (Milestone 2 completion, Master Prompt §6, §7)


**Decision (v1.1.0):**
1. Syntax:
   - `struct Name { field1: Type, field2: Type }`
   - Instantiation: `let s = Name { field1: val1, field2: val2 }`
   - Member access: `s.field1`
2. Static Type Checking:
   - Type representation: `TStruct(name, fields)`.
   - Name hoisting ensures mutual references and forward use in function parameters and return types.
   - Strict field name and type validation emits diagnostics `E0202` (type mismatch) and `E0211` (unknown field).
3. Runtime & Intermediate Representation:
   - Evaluated as `StructInstance` in AST interpreter.
   - Lowers to `make_struct` instruction in IR, preserving field order and labels.
   - Field reads compile to `field_get` in IR.
4. Differential Verification:
   - Verified across AST interpreter, IR-O0, and IR-O1 in `tests/test_structs.py` and `examples/16_structs.tt`.

---

## DD-28: Complete Language Toolchain and CLI General Availability (Milestone 12, Master Prompt §25, §37)


**Decision (v1.0.0):**
1. All 11 foundational CLI commands specified in master prompt §25 are fully implemented:
   - `check`: static type-checking and name resolution
   - `run`: AST, IR-O0, IR-O1, and native C-emitter execution
   - `inspect`: typed IR, memory accounting, and backend capability dumps
   - `test`: pytest suite runner with pattern matching
   - `bench`: reproducible benchmark suite
   - `repl`: interactive evaluation environment
   - `build`: native C-emitter compilation
   - `export`: model export to safetensors, npz, bin, and json
   - `package`: deployable mobile model packaging with optional quantization
   - `doctor`: system environment diagnostics (OS, compiler, backend, python, memory)
   - `profile`: execution runtime and peak memory profiling
2. Version graduation: Time-T reaches v1.0.0 milestone. All milestones 0 through 12 have completed vertical slices with accompanying tests.

---

## DD-26: Portable Model Export and Format Interoperability (Milestone 11, Master Prompt §20, §24, §25)


**Decision (v0.13.0):**
1. HuggingFace Safetensors export and import:
   - `save_safetensors` writes valid, portable safetensors binary format
     (8-byte little-endian header length `N`, `N`-byte UTF-8 JSON header containing
     tensor shapes, dtypes, and byte offsets, followed by contiguous raw buffer bytes).
   - `load_safetensors` parses safetensors headers and extracts tensors directly.
2. NumPy `.npz` archive export and import:
   - `save_npz` and `load_npz` allow direct interchange with the scientific Python ecosystem.
3. Content-sniffed format auto-detection in `checkpoint.load(path)`:
   - Auto-detects `.ttck` (Zip with `manifest.json`), `.npz` (Zip of `.npy` entries),
     `.safetensors` (8-byte unsigned integer header length prefix), and `.json` (Time-T v1 JSON format).
4. CLI `time-t export`:
   - Subcommand `time-t export <input_file> -o <output_file> [-f format] [--json]`
     supports conversion between all supported checkpoint formats (`safetensors`, `npz`, `bin`, `json`).
   - Emits structured diagnostic errors `E0800` (file error) and `E0801` (conversion failure).
5. Interoperability suite `tests/test_interop.py` verifies byte accuracy and round-trip preservation.


**Decisions (v0.12.0):**
1. `RMSNorm` (Zhang & Sennrich 2019) layer and `rms_norm` functional operator:
   - $y = \frac{x}{\sqrt{\text{mean}(x^2, -1) + \epsilon}} \cdot \text{weight}$
   - Finite-difference gradient checks on both input and weights.
2. `CrossEntropyLoss` extended to multi-dimensional tensors `(*, C)` with
   integer targets matching `(*)` (e.g. sequence generation logits `(B, S, V)`
   with target labels `(B, S)`).
3. `Embedding` enhanced to accept both raw Python sequences and `Tensor`
   indices directly with automatic dtype normalization.
4. `TransformerLM`:
   - Complete decoder-only autoregressive language model:
     $\text{Token Embedding} + \text{Positional Embedding} \rightarrow \text{Transformer Blocks} \rightarrow \text{Norm} \rightarrow \text{LM Head}$.
   - Causal upper-triangular masking ensures strictly autoregressive attention.
   - Verified by next-token sequence memorization reaching 100% accuracy.
5. Checkpoint compatibility: fully serializable and loadable via `.ttck` and
   JSON formats.
6. Reachable from Time-T code via `nn.RMSNorm`, `nn.TransformerLM`, `nn.rms_norm`.
   Demonstrated in `examples/14_transformer_lm.tt` running byte-identically
   across AST interpreter, IR-O0, and IR-O1 engines.


**Decisions (v0.11.0):**
1. `Conv1D` layer and `conv1d` functional operator:
   - Input format: 3-D NCL `(N, C_in, L)`.
   - Weights format: 3-D `(F, C_in, K)`.
   - Bias format: 1-D `(F,)`.
   - Output length: `(L + 2*padding - K) // stride + 1`.
2. Implemented with explicit 1D im2col forward and col2im backward tape
   operations.
3. Correctness verified by central finite-difference gradient checks
   against input `x`, `weight`, and `bias` across multiple stride and padding
   configurations in `tests/test_conv1d.py` with tight tolerances
   (`rtol=1e-3, atol=1e-3`), and forward matches a 4-nested-loop naive reference.
4. Checkpoint compatibility verified for `.ttck` binary formats.
5. Reachable from Time-T code via `nn.Conv1D` and `nn.conv1d`.
   Demonstrated in `examples/13_conv1d_classifier.tt` running byte-identically
   across AST interpreter, IR-O0, and IR-O1 engines.


**Decisions (v0.10.0):**
1. `MultiheadAttention` (Vaswani et al. 2017) layer:
   - Projects queries, keys, and values into `num_heads` heads of dimension
     `head_dim = embed_dim // num_heads`.
   - Scaled dot-product attention computed via autograd-tracked 4D tensor
     matmul, transpose, and softmax: `scores = (q @ k.T) / sqrt(d_k) + mask`.
   - Supports self-attention (key=None, value=None defaults to query),
     cross-attention, and optional attention mask broadcasting.
   - Input shape (B, S, E); Output shape (B, S, E).
2. `GELU` (Hendrycks & Gimpel 2016) activation module and `gelu` functional
   op implemented using the standard fast approximation:
   `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`. Verified by
   central finite differences.
3. `TransformerBlock`:
   - Modern Pre-LN architecture:
     `x = x + attn(norm1(x), mask=mask)`
     `x = x + mlp(norm2(x))`
     where `mlp = Linear -> GELU -> Linear`.
   - Fully compatible with `checkpoint.save`, `checkpoint.save_bin`,
     and training state resumption.
4. Correctness verified by central finite-difference gradient checking on MHA
   and GELU, dimension validation error handling (`E0615`), and end-to-end
   optimization convergence.
5. Reachable from Time-T code via `nn.MultiheadAttention`, `nn.GELU`,
   and `nn.TransformerBlock`. Demonstrated in `examples/12_transformer_block.tt`
   with byte-exact output across AST interpreter, IR-O0, and IR-O1.


**Decisions (v0.9.1):**
1. `LayerNorm` (Ba, Kiros, Hinton 2016) layer and `layer_norm` functional
   operator operate over arbitrary trailing dimensions `normalized_shape`.
2. Computes `(x - mean) / sqrt(var + eps) * gamma + beta` directly via
   autograd-tracked primitive operations on `Tensor`. As a result, the backward
   pass flows through the existing verified primitive-op backward rules
   (`mean`, `sub`, `mul`, `div`, `sqrt`).
3. Correctness verified by central finite-difference gradient checks
   against input `x`, `weight` (gamma), and `bias` (beta) in
   `tests/test_layernorm.py` with tolerances (`rtol=1e-3, atol=1e-3`), as
   well as forward verification against an independent reference.
4. `elementwise_affine=True` by default, initializing `weight` to 1 and `bias`
   to 0. When set to False, no parameters are tracked or returned.
5. Incompatible shapes raise diagnostic `NNError` code `E0614`.
6. Verified through state dict and checkpoint save/load roundtrips, and
   exposed to Time-T programs via the pre-bound `nn` module with a new
   runnable example (`examples/11_layernorm_mlp.tt`) executing byte-identically
   across AST, IR-O0, and IR-O1 engines.

