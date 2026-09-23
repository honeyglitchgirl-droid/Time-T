# Time-T — Testing Strategy

## Principles (per master prompt §28)
- Never delete or weaken a test because it fails — fix the underlying bug.
- Every numerical test reports the actual error metric on failure (§18),
  never a bare `assert a == b` for floating point values.
- Categories below map directly to files in `tests/`.

## How to run everything

```bash
python3 -m pip install --break-system-packages -r requirements.txt   # numpy, pytest
python3 -m pytest -q                       # full suite
python3 -m pytest -q --cov=timet           # with coverage (if pytest-cov installed)
python3 -m timet check examples/*.tt       # CLI-level smoke check
bin/time-t bench                            # regenerate benchmark numbers
```

## Test categories and where they live

| Category                     | File                              |
|-------------------------------|-----------------------------------|
| Lexer tests                   | tests/test_lexer.py               |
| Parser tests                  | tests/test_parser.py              |
| Type checker tests            | tests/test_typechecker.py         |
| Interpreter / runtime tests   | tests/test_interpreter.py         |
| IR tests                      | tests/test_ir.py                  |
| Tensor tests (+ NumPy diff)   | tests/test_tensor.py              |
| Autodiff tests (+ fin. diff)  | tests/test_autodiff.py            |
| Backend tests                 | tests/test_backend.py             |
| NN / training tests           | tests/test_nn.py                  |
| CLI tests                     | tests/test_cli.py                 |
| Example program tests         | tests/test_examples.py            |
| Fuzz tests                    | tests/test_fuzz.py                |
| Diagnostics/error msg tests   | tests/test_diagnostics.py         |

There is currently no separate "serialization" test file beyond IR JSON
round-trip (in test_ir.py) because no other serialization format exists yet
(no checkpoints, no ONNX) — see ROADMAP.md for what's deferred.

## Differential testing (§29)

`tests/test_tensor.py` checks every Time-T tensor op against the equivalent
NumPy computation on the same input data (e.g. `Tensor.matmul` vs `a @ b`
in raw NumPy), asserting `np.allclose` with the tolerances centralized in
`tests/numerics.py`.

`tests/test_autodiff.py` checks every backward rule against central-difference
finite differences of the forward function, reporting:
```
max abs error, max rel error, tolerance used, op name, input shape
```
on any failure.

## Fuzzing (§30)

`tests/test_fuzz.py` uses Python's `random` (seeded, so failures are
reproducible — the seed is printed on failure) to:
1. Feed random byte/character soup into the lexer — must only ever raise
   `LexError`, never an unhandled exception, never hang, never segfault
   (N/A in Python, but must never crash the interpreter process).
2. Take valid example programs and apply random token deletions/insertions,
   feeding the mutated text through lexer→parser — must only raise
   `LexError`/`ParseError`, never an unhandled `AttributeError`/`IndexError`/
   etc.

This currently covers the lexer and parser surfaces named in §30. IR
(de)serialization fuzzing and tensor-indexing fuzzing are listed as backlog
in ROADMAP.md self-criticism, not yet implemented — stated honestly rather
than silently skipped.

## Numeric tolerance policy

See DESIGN_DECISIONS.md DD-7. Centralized in `tests/numerics.py`:
```python
RTOL = 1e-3
ATOL = 1e-4
```

## Regression tests

The literal example from master prompt §10 (`x=[1,2,3]` grad, `sum(x*x)`,
backward → grad `[2,4,6]`) is a standalone regression test
(`tests/test_autodiff.py::test_master_prompt_example_grad`) so it can never
silently break.

## What "passing" means here

At the time of writing this document, running `python3 -m pytest -q` from
the repository root against this commit produces the result pasted verbatim
into the PR/commit description — not summarized from memory. Reproduce it
yourself with the command above; do not trust a claimed pass count without
re-running it.
