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

Primitive: `Int`, `Float`, `Bool`, `String`, `Unit`
Tensor: `Tensor[f32]`, `Tensor[f64]`, `Tensor[i32]`, `Tensor[i64]`,
`Tensor[bool]` — rank/shape are **not** part of the static type (checked at
runtime); this mirrors §7's explicit permission to keep shape typing
optional.
Function type (inferred, not user-written yet): `(T1, T2) -> R`

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
`if` is also an expression when both branches produce a value:
```
let sign = if x >= 0 { 1 } else { -1 }
```

## 6. Operators

Arithmetic: `+ - * / %` (int/int → int; any float operand → float; `Tensor`
operands → elementwise, with broadcasting)
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
`ones(...)`, `sum/mean/max/min/exp/log/sqrt/relu/sigmoid/tanh/softmax/matmul`,
`assert(cond, msg)` — raises a runtime `AssertionError`-style diagnostic if
`cond` is false.

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
