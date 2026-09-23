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

## Serialization / checkpoints

Not implemented. The only serialization format that exists is the IR's own
JSON form (`TirProgram.to_json`/`from_json`), used purely for inspection and
tooling, not for saving/loading trained models. No model checkpoint format
exists yet (`docs/ROADMAP.md` Milestone 8).

## Threading / concurrency

Not implemented. No concurrency primitives exist in the language or runtime.
