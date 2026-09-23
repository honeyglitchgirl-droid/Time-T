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


---

# Entry 5 — v0.5.0, expanding Milestone 7 (Conv2D, Embedding, AdamW; DD-17)

1. **What works?** Conv2D (stride + zero-padding over NCHW, engine-agnostic
   since the tape handles it), Embedding with scatter-add gradients,
   AdamW with bit-identical-to-Adam behavior when decay is off. All
   trainable from Time-T code, byte-identical on AST/IR-O0/IR-O1
   (example 09 demonstrates conv training end to end).
2. **What does not work / doesn't exist?** All other conv shapes: 1-D/3-D
   /grouped/dilated/transposed; conv1x1-as-matmul fast paths; padding
   modes beyond zeros ("same"/"reflect"); normalization layers,
   attention blocks, LR schedules, mini-batching, binary checkpoints,
   Conv in the NATIVE backend (out of the DD-15 v1 subset, of course).
3. **What is untested?** Conv on kernels larger than ~4x5 (im2col cols
   matrix blowup is uncharacterized); Embedding with vocab > a few
   hundred (np.add.at contention is real at scale but our scale is
   small); AdamW float64-to-float32 cast behavior over very long
   trainings (tests run 2000 steps max).
4. **What is slow?** `_conv2d_backward`'s inner Python loops over all
   spatial/kernel positions (O(KH*KW*C*H_out*W_out) Python iterations)
   -- small tests take <0.5 s, a 28x28 MNIST image with 16 filters would
   take minutes. This is recorded as a known slow point, no claim made.
5. **What consumes excessive memory?** im2col duplicates the input by
   KH*KW before the matmul (standard tradeoff, same as PyTorch's unfold
   path), on top of the existing tape retention; both documented.
6. **What architectural debt exists?** The conv packing layout knowledge
   (channel-major cols ordering) is duplicated in `forward` and
   `backward` -- a single `_col_layout()` helper would prevent the
   mismatch this tranche spent a debug cycle on (recorded in DD-17);
   `Tensor._make_result` is becoming the blessed custom-op entry point
   WITHOUT a documented contract for third-party-ish ops -- it deserves a
   section in AUTOGRAD.md once a second custom op exists (conv being the
   first); the fd-test float64 pattern is tribal knowledge in two test
   files, not yet a shared helper.
7. **What assumptions may be wrong?** That einsum availability/behaviour
   is stable across numpy versions the way the solver assumes (pinning
   only in requirements is "numpy>=1.24"); that NCHW-only is the right
   default (PyTorch's default, but MLPerf-style workloads often prefer
   channels-last); that Linear-25 to 2 for example 09 generalizes -- it
   does not, it is a 2-image toy and docs say so.
8. **What should be redesigned before continuing?** Before Conv1D/3D
   arrive, collapse the col layout into ONE helper with forward/backward
   symmetry TEST (1-line change, large maintenance win). Before attention:
   dropout/train-eval semantics are simple but the eval-mode dropout-off
   path has no conv-adjacent test; add one when BatchNorm exists.
9. **What should NOT be implemented yet?** Kernel-selection machinery
   (im2col vs FFT vs Winograd -- premature at proof-of-training sizes,
   exactly what Milestone 9's "do not implement all immediately" warns),
   tensor cores / mixed precision, and NN layers in the native emitter
   (needs a BLAS-story decision in the native backend first: how do
   generated .c files call matmul? None of that exists).
10. **What evidence supports current claims?** `pytest -q` = 367 passing:
    tests/test_conv2d.py (13: forward-vs-6-loop-reference in 4 configs,
    fd-checked gradients from all three inputs in 3 configs, determinism,
    shape errors, and a conv-learns-center-detector integration test),
    tests/test_embedding.py (7: row/shape gathers, out-of-range error,
    fd weight grad, duplicate-index accumulation, learnability),
    tests/test_adamw.py (5: bit-identical-at-wd0, directional-decay
    effect, zero-gradient decay isolation, validation, XOR convergence),
    plus example-09 differentially byte-identical across all engines.


---

# Entry 6 — v0.6.0, expanding Milestone 8 (DataLoader, fit_loader, LR schedules; DD-18)

1. **What works?** Seeded determinism: same-seed loaders reproduce the
   exact batch stream across epochs and processes. fit_loader reaches
   100% train accuracy and ~0.005 loss on the semantic two-block task;
   all three LR schedules match their formulas to 1e-12; the whole loop
   runs from Time-T code identically on all three engines.
2. **What does not work / doesn't exist?** Validation-split plumbing
   (metrics are computed on the SAME data being trained on -- honest but
   incomplete), eval()-mode discipline inside fit* (caller responsibility,
   documented not enforced), workers/prefetch, generator-based datasets,
   weighted sampling, per-batch scheduler modes, warmup, checkpointing
   of optimizer+RNG state (resume test covers params only, a shuffled
   loader mid-epoch is NOT resumed bit-exactly -- nothing claimed it).
3. **What is untested?** Shuffle uniformity statistics (np.random is
   trusted, only determinism is pinned), batch_size > n with
   drop_last=False (behaves as one full batch; not explicitly pinned),
   DataLoader reuse across fit_loader calls mid-RNG-stream (documented
   behavior: continues the permutation stream; no test pins which batch
   composition results), schedules on top of Adam/AdamW interactions
   (SGD only in e2e).
4. **What is slow?** fit_loader assembles batches with advanced indexing
   (`x[bidx]`) creating a fresh copy per batch -- fine at our scales,
   would matter only with large datasets, which trains in .tt cannot
   load anyway (no file IO in the language beyond the stdlib).
5. **What consumes excessive memory?** Same batch-copy note; TensorDataset
   holds full dataset in RAM by construction (documented).
6. **What architectural debt exists?** fit and fit_loader duplicate the
   epoch shell (metrics loop, logging, early stopping) -- a THIRD entry
   (val splits) should trigger factoring into one loop with a
   batch-source strategy. LRScheduler duck-types on `optimizer.lr`
   without an interface definition -- acceptable while we have one
   optimizer family, needs a protocol if distributed/Per-Param lrs
   arrive. History remains a bare dataclass with a tacked-on metrics
   dict (recorded in entry 4).
7. **What assumptions may be wrong?** That "epoch" boundaries matter to
   users as the step unit for schedulers (PyTorch convention, fine); that
   default-rng(seed) is the right determinism contract for data pipes
   (some frameworks re-seed per epoch instead -- ours continues the
   stream; DD-18 records the choice); that mean-of-batch-losses is what
   users expect in History (alternatives: sample-weighted mean -- ours
   weights PARTIAL last batches equally with full ones. Honest-now,
   flagged in DD-18 and worth revisiting when val splits exist).
8. **What should be redesigned before continuing?** Before validation
   splits land, unify fit/fit_loader around "iterate batches, call
   step_hook" so the val loop is the same loop with train=False and no
   optimizer; that kills the per-batch/per-epoch ambiguity once instead
   of per feature.
9. **What should NOT be implemented yet?** Data pipelines with
   workers/prefetch, mixed precision, GradScaler, distributed sampling,
   weighted samplers, and serialization formats for datasets -- all
   premature at toy scale (master prompt: primitives before polish,
   never another MNIST).
10. **What evidence supports current claims?** `pytest -q` = 385 passing:
    tests/test_dataloader.py (10: batching/coverage math, seeded-shuffle
    determinism, partial/drop_last honesty, empty-loader error,
    epoch-mean pinning, full e2e to 100% acc), tests/test_lr_schedule.py
    (6: exact lr sequences for all three schedules to 1e-12, per-epoch
    stepping pin, validation errors, annealed-rescues-divergence e2e),
    example 10 byte-identical across AST/IR-O0/IR-O1.


---

# Entry 7 — v0.7.0, Milestone 8 (validation splits, loop unification; DD-19)

1. **What works?** val_loader plumbing with recorded val_losses, eval-mode
   discipline (probe-pinned), monitorable early stopping incl. a semantic
   overfit-detection test on deliberately flipped val labels, and one
   `_run_epoch` shared by train/val.
2. **What does not work / doesn't exist?** No train/eval distinction for
   fit() (full-batch), no `best-epoch model restore` (ES just stops; the
   caller keeps final params -- PyTorch Lightning-style restore-on-stop
   is unimplemented and unclaimed), no per-sample val weighting, no
   k-fold, no stratification.
3. **What is untested?** val_loader with shuffle=True (allowed but
   pointless; not pinned), monitor=<metric-name> early-stopping (code
   path identified, negative-path tested, positive-path not), nested
   Sequential-mode propagation through eval()/train() on models with
   inner submodules beyond Dropout.
4. **What is slow?** Val pass doubles epoch cost at full val size --
   standard, documented; the Probe test froze the optimizer (lr=0) so
   its cost accounting stays honest.
5. **What consumes excessive memory?** Val forward builds a tape that is
   discarded (no no_grad engine yet); at our scales trivial, becomes the
   first thing to revisit if TESTING.md's memory tests ever cover loops.
6. **What architectural debt exists?** History accretes fields by
   attr-attachment (metrics, val_losses) -- a typed shape change now
   (before M8 binary logging exists) would be cheap, later it won't be.
   EarlyStopping remains breadcrumb-minimal; checkpoint-on-best needs
   checkpoint.py integration -- deferred until the binary format exists
   so we do not write it twice.
7. **What assumptions may be wrong?** That one val pass per epoch is the
   right granularity (some workflows val per-n-steps); that eval-mode
   restore is always desirable (a caller might WANT to keep eval on --
   but silent mutation beat optional behavior for v1); that flipped-label
   val is a representative overfit proxy (it is deliberately adversarial
   and that is fine for a signal test).
8. **What should be redesigned before continuing?** Serialize History:
   once the "training-run log format" roadmap item lands, val_losses
   must be first-class there, which argues for doing the History typed
   reshape IN that tranche, not before.
9. **What should NOT be implemented yet?** No-grad tape suppression
   engine (worth engineering only when val sets are big), best-epoch
   weights restore (needs binary checkpoints first), TQDM-style progress
   (noise in byte-identical differential outputs), val schedulers
   (ReduceLROnPlateau) -- a schedule keyed on a loss is real but its
   path-through-History is exactly the History-redesign work, so it
   waits for that.
10. **What evidence supports current claims?** `pytest -q` = 391 passing:
    tests/test_validation.py (6: series recording/distinctness, default
    unchanged, dropout-silenced-and-restored probe, semantic flipped-val
    ES stop, E0643, E0644), plus the whole pre-existing fit/fit_loader
    corpus passing UNCHANGED after the loop unification -- the strongest
    available evidence byte-behavior is preserved.


---

# Entry 8 — v0.8.0, Milestone 8 (binary checkpoints; DD-20 + the Adam falsification)

1. **What works?** v2 saves/loads round-trip values and shapes exactly;
   saves are byte-identical across runs; content-sniffing loads both
   formats; the manifest integrity suite rejects every corruption class
   we could construct; the SGD resume theorem holds BYTE-EXACTLY through
   v2 serialization.
2. **What does not work / doesn't exist?** Optimizer-state checkpointing
   (Adam m/v, SGD momentum buffers, scheduler epoch), RNG-stream
   checkpointing (a resumed run cannot continue a shuffled loader's exact
   stream), dtype-faithful storage (f64 params silently become f32 --
   declared in the manifest so at least detectable; the loader does NOT
   yet honor the declared dtype and that is now an open gap between
   write-path honesty and read-path honesty).
3. **What is untested?** zip bomb / pathological compression ratios (we
   trust the stdlib; worth a fuzz seed), very large SINGLE tensors
   (>2GB, not reachable at our scales), manifest with non-ASCII names
   (our namer uses dotted ASCII paths exclusively).
4. **What is slow?** Nothing measured; ZIP_DEFLATED at level 9 on tensor
   float noise is expectedly modest -- a benchmark row for checkpoint
   save/load would be honest to add when the Benchmarks table next
   updates. No claim made today.
5. **What consumes excessive memory?** save_bin builds the whole zip in
   memory (io.BytesIO) before writing -- fine for MBs, wrong for the
   future "larger models" case that motivated this very tranche. Streaming
   zip writes are marked here as the known fix when sizes demand it.
6. **What architectural debt exists?** The matched error-code scheme
   (E0645..E0648 reused for three distinct mismatch kinds with the same
   code) compresses diagnostics at the message level -- acceptable now,
   but if a registry of diagnosis codes ever exists these should split.
   load_bin trusts `np.load(..., allow_pickle=False)` for payload safety;
   documented, worth a periodic re-audit when adding formats.
7. **What assumptions may be wrong?** That .npy-in-zip is "portable
   enough" -- it is universal across numpy but asks non-Python readers to
   implement .npy; a future C interop layer will need its OWN tiny .npy
   parser or a raw-bin variant (noted for Milestone 11). That f32 is
   always enough for Time-T params (today true). That fixed-epoch zip
   timestamps never leak into user-visible diffs (they do--that is the
   POINT for byte identity).
8. **What should be redesigned before continuing?** The declared-dtype
   read-path gap (#2 above) should close before any non-f32 parameter
   exists; likewise optimizer-state serialization design (moment arrays
   keyed how? Adam state is keyed by id() today -- a serialization
   redesign must come with key-by-parameter-NAME, which is itself a
   Decision).
9. **What should NOT be implemented yet?** safetensors interop, GC-
   addressable sparsity, sharded multi-file checkpoints, encryption/
   signing of checkpoints, and versioning N>2 formats -- one good binary
   format plus the JSON debug format is the correct footing.
10. **What evidence supports current claims?** `pytest -q` = 402 passing:
    tests/test_checkpoint_bin.py (11: byte-identity, sub-25% size on a
    10k-param model, round-trip values/shapes, byte-exact SGD resume
    theorem, extension-independent loading, JSON compat intact, and 6
    corruption-rejection tests), plus the untouched v1 suite and 391 prior
    tests -- all green after the sniffing change to load().


---

# Entry 9 — v0.9.0, Milestone 8 (training-state checkpoints; DD-21)

1. **What works?** Full-state resume that is byte-exact under Adam AND
   SGD, scheduler trajectory continuation, loader stream continuation, a
   complete loud-refusal path set, and byte-identical state saves (all
   tested).
2. **What does not work / doesn't exist?** Epoch-bounded resumption
   inside fit() itself (you resume by CALLING fit for 30 more epochs --
   the 'epochs' argument is relative, not absolute; documented, likely
   permanent), History/log carry-over, multiple model sections in one
   state file (teacher+student, generator+discriminator pairs crash the
   single-section design -- noted for GAN-style features), and state
   save/load for models whose parameter ORDER changed between versions
   (position-keyed names make reorder = wrong values -- but load_into/
   _name_parameters NAME keying catches it for params; the OPTIMIZER
   path is position-keyed and would mis-pair moments on a re-ordered
   constructor. This is a real latent hazard, recorded here and in the
   manifest limits -- mitigations are checked only per-shapes within
   equal lengths).
3. **What is untested?** Huge moment arrays (state files of GB scale --
   same BytesIO note as DD-20), restore-into-a-DIFFERENT but
   shape-identical model (permitted by design; semantic risk above),
   AdamW resumed with eps drift between versions (config compatibility
   is taken on trust across file versions -- no version-migration
   machinery exists).
4. **What is slow?** Nothing new; the save path serializes f64 moments
   uncompressed-inside-DEFLATE (arrays come back smaller than the JSON
   of v1 would have been; no benchmark row yet -- Entry 8's pending note
   bundles this).
5. **What consumes excessive memory?** Same BytesIO whole-file pattern as
   v2 params (Entry 8 item 5) -- now with 2x f64 moment payload, so the
   streaming-write note doubles in relevance; still bounded by our
   actual model sizes.
6. **What architectural debt exists?** Two naming worlds now coexist in
   one file: params are NAME-keyed, optimizer arrays are POSITION-keyed
   (paths like optstate/m.3.npy). A principled unification (param NAME
   -> moment) requires the optimizer to know names -- it currently
   receives a bare param list, so names would be a parallel argument;
   deferred but the seam is clear. optim.py's hand-grown state protocol
   lacks a formal interface (duck-typed again, fine while only three
   optimizer classes exist).
7. **What assumptions may be wrong?** That bit_generator.state round-
   tripping is stable across numpy versions (documented public API,
   but a version pin in requirements is the real guard); that
   scheduler.base_lr restoral is sufficient (true for our schedulers
   since they recompute-from-base; a scheduler with internal counters
   beyond last_epoch would silently fail -- no such scheduler exists);
   that users construct resume optimizers over EXACTLY model.parameters()
   in order (the only supported way documented; the E0653 shape guard
   catches common reorderings blindly but not all).
8. **What should be redesigned before continuing?** A statefile VERSION
   MIGRATION story: today 'version: 2' is hard-rejected for anything
   else, correct but brittle; the first format tweak is also the first
   time we must answer 'how do old files load'. Decide then, not now.
9. **What should NOT be implemented yet?** Distributed-shard state,
   GC-friendly streaming writes at GB scale, name-keyed optimizer
   moments (needs the parallel-names design above), state encryption,
   and LR-restart conveniences (warm restarts are a NEW scheduler, not
   resume machinery).
10. **What evidence supports current claims?** `pytest -q` = 414 passing:
    tests/test_training_state.py (12: byte-exact Adam theorem, AdamW
    config+moments round trip, exact scheduler continuation, loader
    stream continuation via wrong-seed-then-restore, five refusal paths
    E0649-E0658, byte-identical saves, no-moment-yet resume), plus the
    entire 402-test suite unharmed.
