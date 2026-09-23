# Time-T — Architecture

Status: Milestones 0–5 (see ROADMAP.md). This document describes what is
**actually implemented**, clearly separated from what is **planned**.

## 1. Compiler / Runtime Pipeline (as built today)

```
 .tt source text
      │
      ▼
 ┌─────────┐   timet/lexer.py
 │  Lexer  │   hand-written, single-pass, tracks line/col for diagnostics
 └────┬────┘
      │ Token stream
      ▼
 ┌─────────┐   timet/parser.py
 │ Parser  │   recursive-descent, Pratt-style expression parsing
 └────┬────┘
      │ AST (timet/ast_nodes.py)
      ▼
 ┌──────────────┐   timet/typechecker.py
 │ Name Resol.  │   single pass: builds scopes, resolves identifiers,
 │ + Type Check │   infers/checks types, decorates AST nodes with `.ty`
 └────┬─────────┘
      │ Typed AST
      ├─────────────────────────────┐
      ▼                              ▼
 ┌───────────────┐            ┓ timet/ir.py
 │ Interpreter   │            ┃ IR lowering (typed AST → TirProgram)
 │ (tree-walking)│            ┃ inspectable / serializable (JSON)
 │ timet/interp- │            ┃ NOT executed yet (see DESIGN_DECISIONS DD-3)
 │ reter.py      │            ┛
 └────┬──────────┘
      │ calls into
      ▼
 ┌───────────────────────┐
 │ Runtime (timet/runtime)│  Tensor, autodiff tape, backend dispatch,
 │                        │  memory stats, RNG
 └────────────────────────┘
```

Both the interpreter and the IR lowering run on every successfully
type-checked program. This lets IR shape/dtype propagation be tested against
the same programs the interpreter executes, so the two representations can be
cross-checked (`tests/test_ir.py`).

## 2. Module Layout

```
timet/
  __init__.py
  lexer.py          tokenizer, Token, TokenKind, LexError
  ast_nodes.py       AST node dataclasses
  parser.py          recursive-descent parser, ParseError
  diagnostics.py     structured Diagnostic objects (human + JSON rendering)
  types.py           TimeT type system (TInt, TFloat, TBool, TString,
                     TTensor, TFunction, TUnit, TOption, ...)
  typechecker.py     scope/env, type inference & checking, TypeError diag
  modules.py         ModuleLoader: file resolution, parse caching, cycle
                     detection; ModuleValue: the runtime module object (DD-13)
  ir.py              typed IR (TirProgram, TirFunction, TirInstr, ...),
                     lowering pass incl. MODULE FLATTENING (mangled names +
                     once-only __init__<dotted> fns; DD-13), JSON round-trip
  interpreter.py     tree-walking evaluator, Environment, control flow
  tensor.py          Tensor class (wraps numpy.ndarray), autodiff tape hooks,
                     broadcasting, core ops (add/sub/mul/div/matmul/reshape/
                     transpose/sum/mean/max/min/exp/log/sqrt/softmax/...)
  autodiff.py        Tape, Node, backward(), no_grad(), detach()
  backend.py         Backend ABC + capability descriptor + NumpyCpuBackend
  memory.py          allocation/lifetime counters exposed to CLI/diagnostics
  nn.py              Linear, ReLU, Sequential, MSELoss (minimal NN layer)
  optim.py           SGD optimizer
  cli.py             `time-t` command-line entry point
  __main__.py        `python -m timet` entry point

tests/               pytest suite (lexer/parser/typechecker/ir/tensor/
                     autodiff/backend/nn/cli/fuzz)
examples/            *.tt example programs + matching *.expected files
benchmarks/          benchmark scripts + benchmarks/results/*.json (raw,
                     reproducible, timestamped, hardware-labeled)
docs/                this documentation set
bin/time-t           thin shell shim invoking `python3 -m timet`
```

## 3. Type System (implemented subset)

Concrete types implemented today: `Int` (Python int, arbitrary precision
surfaced as 64-bit semantics), `Float` (float64), `Bool`, `String`, `Unit`,
`Tensor[dtype]` (dtype ∈ {f32, f64, i32, i64, bool}, rank tracked but not
required at compile time — see DD below), and function types
`(T1, T2) -> R`.

Shape typing is **not** mandatory (per §7: "do not make shape typing
mandatory if it creates unnecessary complexity") — shapes are tracked
dynamically on `Tensor` values and checked at runtime with clear error
messages (matmul inner-dimension mismatch, broadcast mismatch, etc.), exactly
in the style shown in the master prompt's §27 example. Static shape checking
of the `Tensor<f32, [batch, hidden]>` form is listed as future work in
ROADMAP.md.

## 4. Tensor Model

`Tensor` wraps a NumPy `ndarray` (see DD-1) and adds:
- `requires_grad: bool`
- `grad: Optional[Tensor]`
- a reference to the originating `autodiff.Node` when part of an active tape
- `device: str` (`"cpu"` only today; field exists for forward compatibility)

Supported ops (all tested, all differentiable where mathematically defined):
`add, sub, mul, div, neg, matmul, reshape, transpose, permute, sum, mean,
max, min, exp, log, sqrt, relu, sigmoid, tanh, softmax, broadcast_to,
__getitem__ (basic slicing)`.

## 5. Automatic Differentiation

Reverse-mode, dynamic tape (Wengert list), see DESIGN_DECISIONS DD-4.
`Tensor.backward()` walks the tape in reverse topological order accumulating
`.grad`. `autodiff.no_grad()` is a context manager that disables tape
recording. `Tensor.detach()` returns a value-sharing tensor outside the tape.
Every backward rule has a matching finite-difference gradient-check test
(`tests/test_autodiff.py`), reporting max abs/rel error per DD-7.

## 6. IR

`timet/ir.py` defines a flat, typed, three-address-style instruction list per
function (`TirInstr(op, args, result, ty)`), plus a `TirProgram` container.
It is:
- **Inspectable**: `time-t inspect file.tt --ir` pretty-prints it.
- **Serializable**: `TirProgram.to_json()` / `from_json()` round-trip tested.
- **Deterministic**: instruction numbering and temp naming are stable given
  identical input (tested by hashing repeated lowerings).

It is not yet optimized or executed (DD-3) — this is explicitly listed as
current architectural debt, not hidden.

## 7. Runtime & Memory

`timet/memory.py` maintains a lightweight allocation counter (bytes
allocated/freed per `Tensor` create/free, active tensor count, peak bytes)
updated by `Tensor.__init__`/`__del__` hooks. `time-t inspect --mem` and the
`timet.memory.stats()` API expose: `allocated, peak, active_tensors,
parameters, gradients` — matching the vocabulary requested in §17. This is a
real counter over real NumPy allocations, not a simulated number.

## 8. Backend Abstraction

`timet/backend.py`:
```python
class Backend(ABC):
    name: str
    def capabilities(self) -> BackendCapabilities: ...
    def matmul(self, a, b): ...
    def elementwise(self, op, *args): ...
    ...
```
`BackendCapabilities` reports `{supports_f64, supports_f32, simd, gpu,
max_tensor_rank, device_name}`. Only `NumpyCpuBackend` exists; it reports
`gpu=False`, `simd="numpy-blas"` (whatever NumPy's linked BLAS provides on
this machine — inspected, not assumed).

## 9. CLI

`time-t` (see `timet/cli.py`) subcommands implemented for real:
`check`, `run`, `test`, `inspect`, `repl`, `bench`.
Subcommands present but explicitly marked not-implemented (DD-8):
`build`, `profile`, `export`, `package`, `doctor`.
All commands accept `--json` for machine-readable output (§26).

## 10. Package/Module System, Mobile, GPU, Native Codegen

Not implemented. Explicitly deferred; see ROADMAP.md Milestones 6, 9, 10, 11.
No claims are made about ARM64/mobile/GPU performance anywhere in this repo
because nothing has been measured on that hardware (§15, §16, §19).

## 11. Architectural Risks & Mitigations

1. **Tree-walking interpreter is slow.**
   Mitigation: IR already exists in parallel so a bytecode VM or native
   lowering can consume it later without redesigning the front end.
2. **Python/NumPy as host stack limits raw throughput vs. a native compiler.**
   Mitigation: benchmarks are labeled with the host stack; DD-1 documents the
   revisit trigger (Milestone 9) explicitly.
3. **Dynamic tape autodiff retains full graph in memory (no fusion yet).**
   Mitigation: `no_grad()`/`detach()` exist now; memory stats expose
   active/peak so regressions are visible early.
4. **No static shape checking yet — shape errors are runtime-only.**
   Mitigation: error messages meet the §27 quality bar today; static shape
   inference is roadmap Milestone 6+.
5. **Single backend means no real portability test exists yet.**
   Mitigation: `Backend` ABC is designed now so a second backend is an
   additive change, not a rewrite; capability queries are exercised by tests
   using a `FakeBackend` double.
6. **IR is unused at runtime (dead weight risk / bit-rot risk).**
   Mitigation: IR has its own test suite independent of the interpreter, and
   lowering runs on every `check`/`run` invocation, not just on demand.
7. **No generics/traits/enums/pattern matching yet limits real-world
   expressiveness for ML code.**
   Mitigation: explicitly sequenced in ROADMAP; nn.py APIs are kept minimal
   so they don't accidentally assume unimplemented type-system features.
8. **Numerical tolerance choices (DD-7) could mask real regressions if too
   loose, or cause flaky CI if too tight.**
   Mitigation: tolerances are written down and centralized in
   `tests/numerics.py`, not copy-pasted per test, so they can be tuned once.
9. **Fuzzing coverage is currently limited to lexer/parser (§30 lists more
   surfaces: serialization, model loading, tensor indexing).**
   Mitigation: tracked as explicit backlog item in ROADMAP Milestone 6/8; not
   claimed as done.
10. **No package manager means no dependency isolation story yet (§31).**
    Mitigation: current surface area (single-file `.tt` programs, no
    third-party Time-T packages) has no dependency attack surface to secure
    yet; this will be revisited before any package system ships.
11. **Security of FFI/C ABI boundary is entirely unimplemented.**
    Mitigation: no FFI exists yet, so no unsafe boundary exists yet; §20/§31
    work is explicitly future (Milestone 11).
