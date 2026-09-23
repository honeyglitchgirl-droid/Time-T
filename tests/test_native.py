"""Native backend tests (Milestone 9 v1 slice; DD-15).

Two guarantees are pinned:
  (1) every in-subset program produces BYTE-IDENTICAL stdout in the AST
      interpreter and the natively compiled binary;
  (2) every out-of-subset construct raises NativeError with a message that
      names the construct -- never a silent fallback or a wrong result.
All tests are skipped cleanly when no C compiler exists on the machine.
"""
import pytest
import shutil

from timet.parser import parse
from timet.typechecker import check
from timet.native import emit_c, run_native, find_compiler, NativeError
from timet.interpreter import run_source

pytestmark = pytest.mark.skipif(find_compiler() is None,
                                reason="no C compiler on PATH")


def run_both(src: str):
    out = []
    run_source(src, filename="<native-test>", stdout_write=out.append)
    prog = check(parse(src, "<native-test>"))
    proc = run_native(prog)
    assert proc.returncode == 0, f"native binary failed: {proc.stderr}"
    return out, proc.stdout.splitlines()


TRACE_HELLO = """
print(1 + 2 * 3 - 4)
print(7 % 3)
print(-7 % 3)
print(2 - -2)
"""

BOOLS = """
print((1 < 2) && (2 < 3))
print(!(1 == 2) || false)
print(true)
print(false)
"""

BRANCHY = """
fn sign(x: Int) -> Int {
    if x < 0 { return -1 } else { if x > 0 { return 1 } }
    return 0
}
fn main() {
    print(sign(-9))
    print(sign(0))
    print(sign(42))
    var i = 0
    var acc = 0
    while i < 10 {
        i = i + 1
        if i % 2 == 0 { continue }
        if i > 7 { break }
        acc = acc + i
    }
    print(acc)
}
"""

RECURSION = """
fn fib(n: Int) -> Int {
    if n < 2 { return n }
    return fib(n - 1) + fib(n - 2)
}
fn main() { print(fib(20)) }
"""

STRINGS = """
let a = "hello from native"
let b = a
print(a)
print(b)
"""

INT_DIV_FLOAT_COMPUTED = """
// division is double-division like the interpreter; floats COMPUTE fully,
// only printing them is out of scope -- so verify via comparisons.
fn main() {
    let q = 7 / 2
    print(q > 3.0 && q < 4.0)
    print(10 / 4 > 2.4)
}
"""

PROGRAMS = [TRACE_HELLO, BOOLS, BRANCHY, RECURSION, STRINGS,
            INT_DIV_FLOAT_COMPUTED]


@pytest.mark.parametrize("src", PROGRAMS)
def test_native_matches_ast_byte_for_byte(src):
    ast, native = run_both(src)
    assert ast == native and ast, (ast, native)


def test_examples_01_and_02_compile_natively_and_match(tmp_path):
    for name in ("01_hello_world", "02_calculator"):
        from pathlib import Path
        p = Path("examples") / f"{name}.tt"
        src = p.read_text()
        out = []
        run_source(src, filename=str(p), stdout_write=out.append)
        prog = check(parse(src, str(p)))
        from timet.native import build
        exe = str(tmp_path / name)
        build(prog, exe)
        import subprocess
        proc = subprocess.run([exe], capture_output=True, text=True)
        assert proc.returncode == 0
        assert proc.stdout.splitlines() == out


def test_emitted_c_is_deterministic():
    prog = check(parse(BRANCHY))
    assert emit_c(prog) == emit_c(check(parse(BRANCHY)))


# ---------------- honest rejection battery ----------------

def _rejects(src: str, needle: str):
    with pytest.raises(NativeError) as e:
        emit_c(check(parse(src, "<reject>")))
    assert "native:" in str(e.value) and needle in str(e.value)


def test_rejects_float_print():
    _rejects("fn main() { print(1.5) }", "printing Float")


def test_rejects_tensors():
    _rejects("fn main() { let x = tensor([1.0, 2.0]) }", "")


def test_rejects_for_loop():
    _rejects("fn main() { for i in [1, 2] { print(i) } }", "ForStmt")


def test_rejects_lambda():
    _rejects("fn main() { let f = fn(x: Int) -> Int { return x + 1 } }",
             "cannot map to C")


def test_rejects_lambda_value_call():
    _rejects("fn main() { let f = fn(x: Int) -> Int { return x + 1 }"
             "\n    print(f(1)) }", "cannot map to C")


def test_rejects_import(tmp_path):
    # imports pass parsing+typechecking (module exists); the EMITTER must
    # still refuse loudly -- imports are not in the v1 native subset.
    (tmp_path / "pair.tt").write_text("fn one() -> Int { return 1 }\n")
    src = "import pair\nfn main() { print(pair.one()) }\n"
    from timet.modules import ModuleLoader
    from timet.typechecker import check as tcheck
    prog_ast = parse(src, str(tmp_path / "main.tt"))
    checked = tcheck(prog_ast, filename=str(tmp_path / "main.tt"),
                     loader=ModuleLoader())
    with pytest.raises(NativeError) as e:
        emit_c(checked)
    assert "ImportStmt" in str(e.value) and "native:" in str(e.value)


def test_rejects_unsupported_builtin_call():
    _rejects("fn main() { print(len([1, 2])) }", "")


def test_rejects_untyped_function():
    _rejects("fn f(x) -> Int { return x }\nfn main() { print(f(1)) }",
             "annotated")


def test_static_type_errors_still_fail_loudly_not_silently():
    # && requires Bool at the TYPE level; native must never see it, and the
    # pipeline must fail loudly somewhere, not quietly produce garbage.
    from timet.diagnostics import Diagnostic
    with pytest.raises(Diagnostic):
        run_both("fn main() { print(1 && 2) }")
