# Time-T — Runtime

## Execution

`timet/interpreter.py`'s `Interpreter` tree-walks the typed AST. Program
execution:

1. Every top-level statement runs in order (function declarations are bound
   as `Function` closures over the current `Environment`).
2. If a top-level `fn main()` exists, it is invoked automatically after the
   top-level statements run (mirroring the C/Rust convention referenced
   implicitly by the master prompt's example programs, e.g. §9's `model MLP`
   snippet and every `examples/*.tt` file).

`Environment` is a simple parent-linked scope chain; closures capture their
defining `Environment` object directly (Python's normal reference semantics
give correct closure-over-mutable-variable behavior for `var` bindings).

### Second engine: IR executor (v0.2.0)

`timet/ir_exec.py` executes the typed IR directly
(`time-t run <file> --via-ir [-O 0|1]`). It shares the program semantics of
the AST engine (byte-identical output is enforced by the differential test
suite) but is slower and intentionally supports a smaller subset (DD-9).
Host-library access works identically: `nn` / `optim` / `train` are
pre-bound values in the main frame (DD-10), reachable from any function
through the frame chain.

## Neural-network library (`timet/nn.py`, `timet/optim.py`)

Modules: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Flatten`,
`Dropout` (inverted, seeded → deterministic; honors `train()/eval()`),
`Sequential`. Losses: `MSELoss`, `CrossEntropyLoss` (log_softmax + one-hot
NLL), `BCELoss` (with `Tensor.clip` for stability), plus functional forms
(`nn.mse_loss`, `nn.cross_entropy_loss`, `nn.binary_cross_entropy`).
Optimizers: `SGD`, `Adam` (bias-corrected; moment state keyed by parameter
identity). Gradients of every loss flow through the finite-difference-
checked primitive backward rules — no hand-written gradient code exists in
nn.py except via composition of verified rules.

Usable from BOTH the Python API and Time-T source (`examples/07`):
`let model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))`.

## Training utilities (`timet/train.py`)

`fit(model, loss_fn, optimizer, x, y, epochs, ...)` — FULL-BATCH training
loop (no mini-batching yet; it says so rather than faking generality) with
optional logging, per-epoch pluggable metrics, and `EarlyStopping`
(patience + min_delta, explicit inspectable state). `accuracy(logits,
targets)` — argmax-based classification accuracy used by the tests and
example 07.

## Serialization / checkpoints (`timet/checkpoint.py`)

Implemented in v0.2.0 (DD-11): parameters save to a versioned,
deterministic JSON file and load back with strict key/shape/format/version
checks (`save`, `load`, `load_into`, `state_dict`). Resume semantics are
proven by test: train-20 → save → restore-into-fresh-model → train-30
produces parameters exactly equal to an uninterrupted 50-epoch run.
Limits: float64-expanded JSON is for the small models of today; a binary
format is future work.

## Memory (`timet/memory.py`)

See `docs/ARCHITECTURE.md` §7. `Tensor.__init__`/`__del__` call
`memory.record_alloc`/`record_free` with the tensor's real `nbytes`
(`data.nbytes` from the underlying NumPy array). `timet.memory.stats()`
returns a `MemoryStats` snapshot: `allocated, peak, active_tensors,
total_allocations, parameters, gradients`. Exposed via
`time-t inspect <file> --mem --json`.

Known limitation (documented, not hidden): Python's reference counting/GC
timing means `record_free` may run later than the "last use" point a
programmer would expect (e.g. when a value is captured by a closure or the
autodiff tape) — so `active_tensors`/`allocated` are upper bounds on true
liveness at any exact instant, not an exact live-set. This is called out here
so it isn't misread as a precise leak detector.

## Randomness

`timet/nn.py`'s `Linear` uses `numpy.random.default_rng(seed)` — an explicit
seed argument, defaulting to `0`, so weight initialization is reproducible.
There is currently no language-level `random()`/RNG builtin exposed to `.tt`
programs; this is future work.

## Determinism

Given a fixed seed, tensor operations here are deterministic (NumPy's
default CPU BLAS operations are deterministic for the shapes exercised by
the test suite). No multi-threaded reduction or GPU nondeterminism exists in
this backend, so "deterministic training" (master prompt §23) is
automatically satisfied by the current single-backend implementation --
this is a property of scope, not yet a deliberately engineered guarantee for
a future multi-backend world.

## Threading / concurrency

Not implemented. No concurrency primitives exist in the language or runtime.
