# Time-T Native Compilation Subset Specification (Milestone 9, DD-15)

The `time-t build <file.tt>` command compiles a strict, byte-verified subset of Time-T to standalone C11 source and invokes the system C compiler (`gcc`, `cc`, or `clang`) with `-O2`.

## Fully Supported in Native Build

1. **Primitive Data Types**:
   - `Int` (compiled to C `int64_t`, 64-bit two's-complement arithmetic).
   - `Float` (compiled to C `double`, IEEE-754 64-bit floating point).
   - `Bool` (compiled to C `int`, `1` for `true`, `0` for `false`).
   - `String` literals (compiled to `const char *`, supports assignment and print).

2. **Operators**:
   - Binary arithmetic: `+`, `-`, `*`.
   - Division: `/` (always performs double division, identical to Time-T interpreter semantics).
   - Modulo: `%` (implements Python-style floor modulo on `int64_t` via `tt_mod`, matching interpreter).
   - Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=` on numeric types.
   - Boolean logic: `&&`, `||` with eager evaluation on Bool operands.
   - Unary operators: `-` (negation), `!` (logical NOT).

3. **Control Flow**:
   - `if`, `else if`, `else` branch blocks.
   - `while` loops.
   - `break` and `continue` statements.

4. **Variables and Scoping**:
   - `let` (immutable bindings with block scoping).
   - `var` (mutable bindings with block scoping).
   - Assignment statements (`x = expr`).

5. **Functions & Procedures**:
   - Statically declared functions with typed parameters (`Int`, `Float`, `Bool`, `String`) and annotated return types.
   - Full recursion (e.g. recursive Fibonacci, factorial).
   - `print(x)` of `Int`, `Bool`, and `String` values.

## Explicitly Out of Scope (Emits Diagnostic `E0900` at Build Time)

The native C-emitter strictly enforces its boundaries and will **never** silently fall back or produce undefined behavior. Attempting to compile the following will fail at build time with a structured diagnostic naming the unsupported construct:
- Printing `Float` values directly (requires exact shortest-representation float formatting; compute with floats and print comparisons or cast).
- Tensor instantiations and neural network layers (`tensor()`, `matmul()`, `nn.Linear`, etc.).
- Higher-order functions, lambdas, and closures as first-class values.
- Dynamic `for ... in` loops.
- File module imports (`import`).
- String concatenation or string manipulation.
