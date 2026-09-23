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

Since v0.2.0, lowering also covers: `while` (with the condition instructions
placed INSIDE the loop region so the IR executor re-evaluates them per
iteration), `for` (as `for_begin`/`for_end` with an item temp), `no_grad`
(as region markers), keyword arguments, dynamic calls, `var` bindings as
named storage (`var_def`/`store`/`load` — required for correct loop-carried
mutation), and top-level statements (folded into a synthetic `__main__`
function so a whole program is executable as IR).

Statement/expression kinds still not lowered (nested `fn` declarations,
lambdas, if-EXPRESSIONS) emit an explicit `unsupported_stmt`/
`unsupported_expr` marker instruction carrying the AST node kind in `attrs`
— this makes the gap visible in `--ir` dumps, and the IR executor raises a
clear `IrExecError` naming the construct instead of silently running
something wrong.

## IR execution (`timet/ir_exec.py`) — v0.2.0

The IR is directly executable: `time-t run <file> --via-ir [-O 0|1]`.
Structured control markers are parsed into a block tree (`_build_tree`);
mutable variables live in chained named-storage frames mirroring the AST
interpreter's `Environment`; a user-defined `fn main()` is invoked after
top-level statements, exactly like the AST engine. Correctness is not
asserted but measured: `tests/test_ir_exec_diff.py` requires byte-identical
stdout across AST interpreter, IR-O0, and IR-O1 for every example program
and for generated random programs. Deliberate limits: DD-9.

## IR optimization (`timet/optimize.py`) — v0.2.0

`-O 1` (the default for `--via-ir` and `inspect --ir --opt`) applies:
constant folding, type-aware algebraic simplification, segment-local CSE,
segment-local copy propagation, and dead-code elimination, followed by a
final fold/DCE sweep. Every rule and its exact safety boundary (no
`x*0`-folding because NaN/Inf; no `1/0`-folding because the error belongs
at runtime; CSE/copy-prop never cross control markers; loads invalidated by
intervening stores) is documented in DESIGN_DECISIONS.md DD-12. Passes
report statistics (`OptStats`) shown by `inspect --ir --opt`.

## What is NOT part of the compiler yet

- No inlining, loop-invariant motion, operator fusion, or anything needing
  dataflow analysis (needs a CFG-form IR — DD-9/DD-12).
- No lowering from IR to a lower-level form (bytecode or native code)
  exists; execution today happens directly off the typed AST via
  `timet/interpreter.py` (see `docs/DESIGN_DECISIONS.md` DD-3).
- No shape/dtype propagation happens in the IR; shapes are dynamic (checked
  at runtime by `Tensor` operations with clear diagnostics).

These are sequenced in `docs/ROADMAP.md` Milestone 6 and are not silently
missing — `time-t inspect --ir` will show you exactly what is and is not
represented for any given program today.
