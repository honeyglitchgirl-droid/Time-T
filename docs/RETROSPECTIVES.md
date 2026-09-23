# Time-T — Milestone Retrospectives (master prompt §39)

At the end of every major milestone the project answers the 10
self-criticism questions from §39, in writing, without deleting prior
answers. Entries are dated by the version that closed the milestone band.
Older entries are preserved verbatim — including statements that later
became outdated — so the record shows what was believed when.

---

# Entry 1 — v0.1.0, closing Milestones 0–5 (+ partial 6–7)

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

(Notes from later: items 2/6/8 above were resolved in v0.2.0 — IR is now
executed and optimized; see Entry 2. Item 8's advice was followed:

the optimizer work began only after the IR itself became executable.)

---

# Entry 2 — v0.2.0, closing Milestone 6 (+ expanding 7–8)

1. **What works?** Everything from v0.1.0, plus: the IR is directly
   executable (`timet/ir_exec.py`) — all 7 example programs, including the
   autodiff training loops with `no_grad`/`.backward()`, run through IR
   with byte-identical output to the AST interpreter; an O1 optimizer
   (constant folding, algebraic simplification, CSE, copy-prop, DCE) whose
   output is *also* byte-identically verified; `nn` grew Tanh/Softmax/
   Flatten/Dropout(seeded)/CrossEntropy/BCE and `optim` grew Adam, all
   finite-difference-checked where differentiable; `train.fit()`,
   `accuracy`, `EarlyStopping`; JSON checkpoints with a proven
   save/restore/continue-training equality test; and the library is
   callable from Time-T syntax (`examples/07_xor_classifier.tt` reaches
   100% XOR train accuracy on all three engines).
2. **What does not work / doesn't exist?** Modules/import (DD-10 is a
   bridge, not a module system), generics/traits/structs/enums/pattern
   matching, static shape typing, CFG-form IR, inlining/fusion/LICM,
   native codegen, GPU/ARM64/mobile backends, FFI/C ABI/ONNX, package
   manager, datasets/dataloaders, LR schedules, mini-batching,
   binary checkpoints, AdamW, Conv/Embedding/attention layers.
3. **What is untested?** IR execution of extremely deep recursion (executor
   recursion limit is Python's); optimizer passes against pathological
   instruction orderings beyond the randomized straight-line programs in
   the differential suite (a structured-IR fuzzer is still missing);
   checkpoint files with hostile/hand-edited content beyond the corruption
   cases tested; `fit()` on genuinely large batches (it is full-batch
   only, and memory behavior there is uncharacterized).
4. **What is slow?** The IR executor is now the SLOWEST engine (block-tree
   rebuilding per call, marker parsing per invocation) — acceptable because
   its job is verification, not speed; numbers: matrix ops remain
   NumPy-bound, interpreter overhead dominates small workloads (real
   timings in `benchmarks/results/bench_1790172750.json`, including the new
   `mlp_train_step_128x2_adam` row).
5. **What consumes excessive memory?** Unchanged from Entry 1 (dynamic tape
   retention) PLUS: checkpoints store float64-expanded JSON — fine at
   current model scale, wrong at 1M+ (documented in DD-11); the IR
   executor rebuilds block trees per function call —fine at current
   program scale.
6. **What architectural debt exists?** Builtins are duplicated between the
   interpreter's `_builtin` mapping and the IR executor's `_BUILTINS`
   (parity is enforced only by differential tests, not shared code);
   `TUnknown` flows through the `nn.*` boundary so static checking of
   NN-typed code is weak (DD-10); the executor's flat var storage can't
   model shadowed `var`s in nested blocks (documented, avoided by tests).
7. **What assumptions may be wrong?** That structured-marker IR is a good
   substrate for Milestone 9's native codegen — a CFG form may be needed
   and would obsolete `_build_tree`; that pre-bound `nn`/`optim`/`train`
   globals (DD-10) won't conflict with a future real module system's
   names — a migration path (aliases, deprecation) will be needed.
8. **What should be redesigned before continuing?** Before any LARGE
   tensor ops arrive (conv), the Tensor class needs a stride/view design
   review (currently every op materializes a new array — fine for the
   verified small scale, a known cost); before Milestone 9, the IR needs
   its CFG form decision (DD-9 flags this).
9. **What should NOT be implemented yet?** Native codegen and any second
   backend (the Backend abstraction is still verified by only ONE real
   implementation; adding a fake GPU stub would violate §38); mini-batching
   infrastructure (needs the stride/view review first); anything requiring
   async/concurrency — no concurrency model exists or is claimed.
10. **What evidence supports current claims?** `pytest -q` = 247 passing
    (was 175 in v0.1.0), including: `tests/test_ir_exec_diff.py` (7 examples +
    70 generated programs, three engines, byte-exact),
    `tests/test_optimize.py` (13 pass-level tests incl. the no-fold-1/0 and
    no-CSE-across-store rules), `tests/test_nn.py` (16 tests incl.
    finite-difference checks of CrossEntropy/BCE and an
    Adam-reaches-threshold XOR test), `tests/test_checkpoint.py` (8 tests
    incl. save/restore/continue equality), `tests/test_train.py` (7 tests
    incl. a fit()-to-100%-accuracy integration test); and the CLI
    end-to-end checks (`--via-ir` vs AST `run` outputs compared in
    `tests/test_cli.py`).

---

# Entry 3 — v0.3.0, closing the Milestone 2 remainder (modules; DD-13)

1. **What works?** The module system: `import a.b.c [as alias]` resolves
   relative-to-importer; imports are load-once (top-level prints run
   exactly once, proven when one module is imported through two paths);
   modules type-check in a fresh scope (importer's `let`s provably do NOT
   leak in — compile error tested); exports are typed for real
   (`mathlib.add(1,2)` is static-checked against `Int, Int -> Int`;
   unknown members are E0211 *listing the actual exports*); cycles are
   E0210, missing files E0213 with the looked-for path. Everything runs
   byte-identically on AST, IR-O0 and IR-O1 (modules flatten into the IR
   as prefixed functions + once-only `__init__<mod>` — DD-13).
2. **What does not work / doesn't exist?** Packages with `__init__`,
   `from x import y`, wildcard imports, relative `./`-imports, visibility
   modifiers, module-level metadata/docstrings; higher-order use of module
   functions in IR (clear DD-9 error, not silent failure); loading modules
   from a search path plus standard-library modules written in Time-T
   (`nn`/`optim`/`train` are still DD-10 pre-bound globals, NOT `import`-able
   .tt files — a deliberate migration question, not an oversight).
3. **What is untested?** Module resolution across symlinked/aliased
   directory layouts (canonical-path cache would treat symlink target and
   link as one module — desired — but no test pins it); import behavior
   from REPL in a non-project cwd (CWD-relative — documented, fuzz-ish
   edge not pinned); module combined with `--via-ir` + `-O1` on programs
   with heavy module-state mutation (basic mutation IS tested: the counter
   bump test).
4. **What is slow?** Nothing NEW on named axes; module init adds one
   synthetic-call indirection per import point in the IR — drown-out in
   interpreter overhead. `benchmarks/` untouched: no perf claim changed.
5. **What consumes excessive memory?** Unchanged. Module ASTs and
   ModuleValue environments stay alive for the whole run (normal import
   caching; no eviction exists — fine at current program scale, would need
   revisiting for REPL sessions importing thousands of files).
6. **What architectural debt exists?** The built-in `nn`/`optim`/`train`
   globals should eventually be Time-T source modules under a stdlib path
   so that imports and builtins share ONE mechanism — the current split
   (TModule-for-files + TModule-for-prebound-globals) is deliberate
   plumbing, not a final design; the type checker re-parses nothing but
   re-CHECKS module bodies when importers change mid-chain — cheap at
   current scale because of the per-path cache flags; `ir.py`'s
   `_Lowerer` is growing parameters (prefix/storage_prefix/module maps) —
   a small config object is due if one more axis lands.
7. **What assumptions may be wrong?** That import-once *module-level
   mutable state* (Python-style) is the right semantics for an ML language
   — PyTorch's own module-state pitfalls suggest it deserves a design
   note if/when tensors-in-module-state appear at scale; that flattening
   modules into single-program IR stays viable when native codegen (M9)
   arrives — a linkage/unit model may be needed then (DD-13 leaves the
   door open; flattening keeps O0 verification exact).
8. **What should be redesigned before continuing?** Nothing blocking. If
   the NEXT milestone remainder (M6b: CFG-form IR + inlining) begins,
   re-visit the `__init__<mod>` call markers — CFG form wants them as real
   call sites with interprocedural rules, not string-name conventions.
9. **What should NOT be implemented yet?** Package management, stdlib
   written in Time-T (the modules machinery is fresh — dogfood it via
   examples first), module visibility/encapsulation (needs the
   struct/enum data-model decisions it will probably interact with),
   search-path resolution (invites dependency-hell without a package
   manager to govern it).
10. **What evidence supports current claims?** `pytest -q` = 318 passing:
    `tests/test_modules.py` (13 tests incl. the fresh-scope-negative test,
    once-only-init test via `print` emission, cycle/missing/nested/unknown-export
    error codes, and a 3-engine differential for every behavioral test);
    example `08_modules.tt` (with `examples/modules/linalg.tt` +
    `consts.tt`) runs through AST/IR-O0/IR-O1 byte-identically and through
    the CLI (`examples` differential suite auto-picks it up); CLI
    diagnostics for E0213 verified in `tests/test_cli.py` (now 14).


---

# Entry 4 — v0.4.0, opening Milestone 9 (native C-emitter) and reconciling language semantics

1. **What works?** `time-t build` compiles a byte-verified subset of
   Time-T to native executables through a C emitter + system cc; every
   out-of-subset construct fails loudly with a span-tagged diagnostic.
   `break`/`continue` now exist in all four execution paths (AST, IR-O0,
   IR-O1, native). The `Int/Int -> Int` soundness bug is fixed; both
   divergent subsystems (checker's division typing and optimizer's
   div-by-one fold) have regression tests.
2. **What does not work / doesn't exist?** Native: tensors, Float
   printing, for loops, f-strings, string concat, modules, closures as
   values. Milestone 9's real substance — whole-program optimization,
   inlining, LICM/fusion, kernel selection, multi-objective optimization —
   is untouched (needs CFG-form IR per DD-9/DD-12). ARM64/mobile still
   zero.
3. **What is untested?** Native Int overflow behavior AT the 2^63 boundary
   (documented divergence, no test pins the exact wrap value — the
   contract is "stay within int64", asserted by differential tests,
   but a wrap-at-boundary case isn't one of them); `build` against clang
   (only gcc exercised — harness searches gcc first); emitted-C warnings
   (-Wall output is not checked, only compilation success).
4. **What is slow?** Nothing new in the INTERPRETER; the native path adds
   a ~50 ms gcc invocation per build (baseline for tiny programs; it is
   the honest price of host-compiler optimization). Benchmarks refreshed:
   bench_1790177898.json is the current truth.
5. **What consumes excessive memory?** Unchanged (tape retention). The
   native path spawns gcc via tempfiles and cleans them; nothing retained.
6. **What architectural debt exists?** The native emitter re-walks the
   typed AST directly rather than consuming the IR — a second codegen
   path that will want re-targeting to the IR once CFG form exists
   (recorded, not hidden; the AST path was chosen deliberately for the
   vertical slice per "minimal first"); `timet/native.py` duplicates a
   small amount of semantic knowledge (floor-mod helper, division rule)
   in C text — acceptable while the subset is scalars-only; param types
   are NOT in the IR, which limits the optimizer (conservative div-by-one
   fold) and the native path alike.
7. **What assumptions may be wrong?** That byte-equality for Float
   printing is achievable cheaply later (shortest-repr in C is a real
   port of Grisu/Ryu — it may force ND wrapper or C++ <charconv>); that
   int64-as-Int is acceptable for an ML language (PyTorch uses int64 by
   default — probably fine, but arbitrary-precision interpreter Int vs
   int64 native IS a stated contract wrinkle users could trip on); that
   gcc-on-PATH is a reasonable build prerequisite (doctor/build already
   report it cleanly when absent — tests skip).
8. **What should be redesigned before continuing?** Milestone 9's
   REMAINDER should NOT grow the AST-directed emitter indefinitely: the
   next native increment should first land param types in the IR (cheap,
   unblocks BOTH optimizer folding and emitter type-lookup), then decide
   whether native lowers from IR instead of AST for v2. That decision and
   the CFG-form decision are the same door.
9. **What should NOT be implemented yet?** LLVM (per DD-15 — premature);
   tensor codegen natively (needs a kernel/BLAS decision first, then a
   TENSOR.md-level design note); multi-objective optimization (needs the
   baseline single-objective passes and REAL benchmarks to trade);
   packaging the native binary as an installer (M12 concerns).
10. **What evidence supports current claims?** `pytest -q` = 340 passing:
    tests/test_native.py (16: byte-equality across the subset battery,
    examples 01/02 compiled, deterministic C emission, 9 rejection
    tests naming each construct, temp-compile determinism);
    benchmarks/results/bench_1790177898.json (measured 3247x scalar-loop
    ratio with the caveat in the row itself); the break/continue and
    division-typing regressions in tests/test_optimize.py +
    tests/test_typechecker.py + differential suite (now 60 cases).
