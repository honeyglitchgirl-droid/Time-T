# Time-T — Compiler

See `docs/ARCHITECTURE.md` section 1 for the pipeline diagram. This document
gives per-stage detail for the compiler front end.

## Lexer (`timet/lexer.py`)

Hand-written, single-pass, no external lexer-generator dependency. Produces
`Token(kind, text, line, col, value)`. Every error raised is a `LexError`
(subclass of `timet.diagnostics.Diagnostic`) carrying a `SourceSpan`.

Recognizes: integers (decimal/hex/binary, `_` separators), floats
(with exponents), booleans, double-quoted strings with `\n \t \" \\ \r \0`
escapes, identifiers, the fixed keyword set in `KEYWORDS`, and the operator
set in `OPERATORS` (checked longest-match-first so `->` is never split into
`-` `>`).

## Parser (`timet/parser.py`)

Recursive descent for statements/declarations; Pratt (operator-precedence)
parsing for expressions, with the precedence table in `PRECEDENCE`. Producing
node types from `timet/ast_nodes.py`. Parse errors are `ParseError`
diagnostics with an `expected`/`actual` pair, per master prompt §27.

Constructs the parser explicitly rejects today (raising a clear, dedicated
`ParseError`, not a generic failure): `struct`, `enum`, `match`, `import` —
these are parsed as keywords specifically so the error message can say "not
implemented yet" instead of "unexpected token".

## Name resolution + type checking (`timet/typechecker.py`)

Single pass, scope-chained (`Scope`/`Binding`). Function declarations are
hoisted (both at top level and per-block) so forward/mutual references work
without a separate declaration-order requirement. See
`docs/DESIGN_DECISIONS.md` DD-6 for what's intentionally excluded (generics,
traits, etc).

Every expression node gets a `.ty` attribute set during `infer()`, which the
IR lowering pass (see below and `docs/ARCHITECTURE.md` §6) consumes directly
— there is no separate "typed AST" data structure copy; the same AST nodes
are annotated in place.

## IR lowering (`timet/ir.py`)

`lower_program`/`lower_function` walk the *type-checked* AST and emit a flat
list of `TirInstr` per function using a simple `_Lowerer` with an
incrementing temporary counter (`%t0, %t1, ...`), which is what makes
lowering deterministic given identical input (verified in
`tests/test_ir.py::test_deterministic_lowering`).

Statement kinds not yet fully lowered (`for`, `no_grad`, nested `fn`) emit an
explicit `unsupported_stmt`/`unsupported_expr` marker instruction carrying
the AST node kind in `attrs` — this makes the gap visible in `--ir` dumps
rather than silently dropping code, which would be worse than an honest
"not lowered yet" marker.

## What is NOT part of the compiler yet

- No optimization passes (constant folding, CSE, algebraic simplification,
  inlining, fusion, dead-code elimination) run on the IR.
- No lowering from IR to a lower-level form (bytecode or native code)
  exists; execution today happens directly off the typed AST via
  `timet/interpreter.py` (see `docs/DESIGN_DECISIONS.md` DD-3).
- No shape/dtype propagation happens in the IR; shapes are dynamic (checked
  at runtime by `Tensor` operations with clear diagnostics).

These are sequenced in `docs/ROADMAP.md` Milestone 6 and are not silently
missing — `time-t inspect --ir` will show you exactly what is and is not
represented for any given program today.
