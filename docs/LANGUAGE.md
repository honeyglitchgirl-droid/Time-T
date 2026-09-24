# Time-T — Language Reference (v0.1, current implementation)

This describes exactly what `timet/lexer.py` + `timet/parser.py` +
`timet/typechecker.py` accept today. Anything not listed here is not yet
implemented — see ROADMAP.md.

## 1. Lexical Structure

- Line comments: `// ...`
- Block comments: `/* ... */` (non-nesting)
- Identifiers: `[A-Za-z_][A-Za-z0-9_]*`
- Integer literals: `123`, `0x1F`, `0b1010`, with optional `_` separators (`1_000_000`)
- Float literals: `1.5`, `1.5e10`, `1_000.5`
- Bool literals: `true`, `false`
- String literals: `"double quoted"`, with `\n \t \" \\` escapes
- Unit: `()`

## 2. Types

Primitive: `Int` (arbitrary-precision in interpreter/IR engines; 64-bit two's-complement `int64_t` in native C compilation), `Float` (IEEE 754 64-bit double), `Bool`, `String`, `Unit`
Tensor: `Tensor[f32]`, `Tensor[f64]`, `Tensor[i32]`, `Tensor[i64]`,
`Tensor[bool]` — rank/shape are **not** part of the static type (checked at
runtime); this mirrors §7's explicit permission to keep shape typing
optional.
Function type (inferred, not user-written yet): `(T1, T2) -> R`

### Integer Width & Arithmetic Semantics
Time-T follows **Option A (Arbitrary Precision with Native 64-bit Mapping)**:
- In the reference AST Interpreter and IR Executor, scalar `Int` values provide unbounded arbitrary-precision integer arithmetic without overflow truncation.
- In the Native C11 Emitter (`time-t build`), `Int` is mapped to `int64_t` (two's-complement 64-bit signed integer) for maximum execution speed, with modulo operations implementing Python-semantic floor modulo (`tt_mod`).
- In Tensor buffers (`Tensor[i32]`, `Tensor[i64]`), explicit fixed-width contiguous arrays are enforced for memory efficiency and hardware alignment.

## 3. Bindings

```
let x = 5            // immutable
let y: Float = 2.0   // immutable, annotated
var z = 0             // mutable
z = z + 1             // assignment only allowed to `var`
```
Assigning to a `let` binding is a compile-time error
(`E0301 cannot assign to immutable binding 'x'`).

## 4. Functions

```
fn add(a: Int, b: Int) -> Int {
    return a + b
}

fn greet(name: String) {       // return type Unit inferred if omitted
    print(name)
}
```
Functions are values; closures over the enclosing scope are supported:
```
fn make_adder(n: Int) -> (Int) -> Int {
    fn adder(x: Int) -> Int {
        return x + n
    }
    return adder
}
```

## 5. Control Flow

```
if x > 0 {
    print("positive")
} else if x < 0 {
    print("negative")
} else {
    print("zero")
}

var i = 0
while i < 10 {
    print(i)
    i = i + 1
}
```
`break` and `continue` (v0.4.0) work in `while` and `for` loops; using one
outside a loop is a compile-time E0215:
```
while true {
    i = i + 1
    if i % 2 == 0 { continue }
    if i > 9 { break }
}
```
`if` is also an expression when both branches produce a value:
```
let sign = if x >= 0 { 1 } else { -1 }
```

## 6. Operators

Arithmetic: `+ - * / %`.
- IMPORTANT: `/` is TRUE division — `7 / 2 == 3.5` — and is typed `Float`
  even for Int operands (fixed in v0.4.0: it was mis-typed Int before; no
  program's RUNTIME behavior changed, only the type checker's soundness).
- `%` is Python floor-modulo (`-7 % 3 == 2`) on Int; any Float operand
  promotions behave like Python (`Float` results).
- `+ - *` are Int-preserving on Int/Int; any Float operand → Float;
  `Tensor` operands → elementwise with broadcasting.
Comparison: `== != < <= > >=`
Logical: `&& || !`
Matrix multiply: `@` (Tensor only)
Unary: `-x`, `!b`

## 7. Tensors (literal + call-style construction)

```
let x = tensor([1.0, 2.0, 3.0])                 // Tensor[f32], shape (3,)
let m = tensor([[1.0, 2.0], [3.0, 4.0]])        // shape (2,2)
let g = tensor([1.0, 2.0, 3.0], grad=true)      // requires_grad tensor
let z = zeros([2, 3])
let o = ones([3])
```
Method-call style tensor ops:
```
x.sum() x.mean() x.reshape([3,1]) x.transpose() m.matmul(m)
x.exp() x.log() x.sqrt() x.relu() x.sigmoid() x.tanh() x.softmax()
x.log_softmax() x.clip(-1.0, 1.0) x.argmax() x.item() x.detach()
```
and free functions mirroring them: `sum(x) mean(x) matmul(a,b) relu(x) ...`

## 8. Automatic Differentiation Surface

```
let x = tensor([1.0, 2.0, 3.0], grad=true)
let y = sum(x * x)
y.backward()
print(x.grad)          // [2.0, 4.0, 6.0]
```
`no_grad { ... }` block disables tape recording inside it:
```
no_grad {
    let y = x * 2.0
}
```
`x.detach()` returns a tensor sharing storage but detached from the tape.

## 9. Built-in functions available today

`print(x)`, `len(x)` (tensors/strings), `tensor(...)`, `zeros(...)`,
`ones(...)`, `sum/mean/max/min/exp/log/sqrt/relu/sigmoid/tanh/softmax/
log_softmax/argmax/one_hot/matmul`,
`assert(cond, msg)` — raises a runtime `AssertionError`-style diagnostic if
`cond` is false.

## 9a. Pre-bound library modules (v0.2.0, DD-10)

Three names are bound in every program's global scope (before any user
code runs), in BOTH execution engines:

- `nn` — layers/losses: `nn.Sequential`, `nn.Linear`, `nn.ReLU`,
  `nn.Sigmoid`, `nn.Tanh`, `nn.GELU`, `nn.Softmax`, `nn.Flatten`, `nn.Dropout`,
  `nn.Conv1D`, `nn.Conv2D`, `nn.Embedding`, `nn.LayerNorm`, `nn.RMSNorm`,
  `nn.MultiheadAttention`, `nn.TransformerBlock`, `nn.TransformerLM`;
  `nn.mse_loss`, `nn.cross_entropy_loss`, `nn.binary_cross_entropy`,
  `nn.gelu`, `nn.layer_norm`, `nn.rms_norm`, `nn.conv1d`, `nn.conv2d`
  (+ class forms `nn.MSELoss()` etc.)
- `optim` — `optim.SGD(params, lr=...)`, `optim.Adam(params, lr=...)`
- `train` — `train.accuracy(logits, targets)`

Typing note (honest): these are typed `TUnknown` at the boundary — the type
checker does not yet statically check `nn.*` member types; errors raise at
runtime with a stage tag. This is a documented bridge until the real module
system (Milestone 2 remainder) exists — it is NOT a module system.

Worked example: `examples/07_xor_classifier.tt` (trains an MLP with Adam to
100% XOR accuracy; identical output under AST, IR-O0, IR-O1 engines).

## 9b. Modules (v0.3.0, DD-13)

```
// file: modules/linalg.tt
fn dot(a: Tensor[f32], b: Tensor[f32]) -> Float { ... }
let SCALE = 2.0

// file: main.tt
import modules.linalg          // resolves modules/linalg.tt next to main.tt
import modules.consts as consts // aliased import

fn main() {
    print(linalg.dot([1.0, 2.0], [3.0, 4.0]))
    print(consts.SCALE)
}
```

Rules (all enforced by tests in `tests/test_modules.py`):
- Resolution is relative to the IMPORTING file's directory (or the current
  working directory for REPL/STDIN input); no search paths.
- Top-level imports only (E0212 inside blocks/functions).
- A module is loaded + type-checked + executed ONCE per path; its top-level
  statements (incl. any `print`s) run at import time, exactly once.
- A module gets a FRESH scope: it cannot see the importer's variables
  (compile error if it references them). `fn main()` inside a module is an
  ordinary function, not auto-invoked.
- ALL top-level `fn`/`let`/`var` bindings are exported; member types are
  checked statically (`module.fn(...)`, `module.value`); an unknown member
  is a compile-time E0211 listing the real exports.
- Import cycles are a compile-time E0210 naming the cycle.
- Typed parameters matter: function parameters WITHOUT annotations are
  typed `<unknown>`, and arithmetic on `<unknown>` is a compile-time error
  (E0204) — annotate parameters you do math with (this rule predates
  modules and is unchanged).

Not yet: `from x import y`, packages with `__init__`, wildcard imports,
relative `./` imports, visibility modifiers, module-level doc metadata.

## 9c. Native compilation (v0.4.0, DD-15) — strict subset

`time-t build file.tt` compiles a program to a native executable via a C
emitter + system C compiler; `time-t run file.tt --native` builds-and-runs.
The v1 subset is deliberately small, and EVERY out-of-subset construct is
rejected with a `native:`-prefixed diagnostic naming the construct (never
a silent fallback, never a partially-compiled program).

Supported (byte-verified against the interpreter in tests): Int/Float/Bool
arithmetic and comparisons, `&&`/`||` on Bool operands, String literals
(assignment + print), `let`/`var`, `if`/`else if`/`else`, `while` with
`break`/`continue`, typed functions (annotations REQUIRED for params and
return types) including direct recursion, `print` of Int/Bool/String.

Not supported in v1: printing Float values (shortest-repr formatting
parity is a separate work item), tensors and all builtins except `print`,
`for` loops, lambdas as values, f-strings, string concatenation, modules.

Semantic caveat (documented divergence): native Int is **int64** —
Time-T interpreter Int is arbitrary precision (Python). Values beyond
2^63 wrap mod 2^64 natively. Differential tests stay inside the
well-defined contract.

## 10. Explicitly NOT implemented yet (see ROADMAP.md)

structs, enums, pattern matching, generics, traits/interfaces, modules /
`import`, error-handling (`Result`/`Option` sugar), iterators, user-defined
collections beyond `Tensor`, compile-time evaluation (`const fn`), concurrency
primitives, package manifests. These are sequenced in ROADMAP.md and are not
silently half-supported — the parser raises a clear `ParseError` naming the
unsupported construct if you try to use them.

## 11. Example: XOR-style tiny program (this runs today)

```
fn main() {
    let x = tensor([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
    let y = tensor([[0.0],[1.0],[1.0],[0.0]])
    print(x)
    print(y)
}
```
See `examples/` for programs that actually execute, each paired with an
`.expected` output file checked by `tests/test_examples.py`.
