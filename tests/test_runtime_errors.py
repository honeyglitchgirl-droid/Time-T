"""Regression tests for runtime error diagnostics (deep audit findings).

Verifies that division/modulo by zero and maximum recursion depth exceedance
emit structured Time-T Diagnostic exceptions (E0507, E0508, E0509) across both
the AST interpreter and IR executor, rather than crashing with unhandled Python
ZeroDivisionError or RecursionError tracebacks.
"""
import pytest
from timet.parser import parse
from timet.typechecker import check
from timet.interpreter import run_source, RuntimeErr, Interpreter
from timet.ir import lower_program
from timet.ir_exec import run_program, IrExecError


def test_division_by_zero_ast_diagnostic():
    src = "let x = 10 / 0"
    with pytest.raises(RuntimeErr) as exc_info:
        run_source(src)
    assert exc_info.value.code == "E0507"
    assert "division by zero" in exc_info.value.message


def test_division_by_zero_ir_diagnostic():
    src = "fn main() { let x = 10 / 0 }"
    prog = check(parse(src))
    tir = lower_program(prog)
    with pytest.raises(IrExecError) as exc_info:
        run_program(tir)
    assert exc_info.value.code == "E0507"


def test_modulo_by_zero_ast_diagnostic():
    src = "let x = 10 % 0"
    with pytest.raises(RuntimeErr) as exc_info:
        run_source(src)
    assert exc_info.value.code == "E0508"
    assert "modulo by zero" in exc_info.value.message


def test_modulo_by_zero_ir_diagnostic():
    src = "fn main() { let x = 10 % 0 }"
    prog = check(parse(src))
    tir = lower_program(prog)
    with pytest.raises(IrExecError) as exc_info:
        run_program(tir)
    assert exc_info.value.code == "E0508"


def test_recursion_depth_ast_diagnostic():
    src = """
    fn recurse(n: Int) -> Int {
        return recurse(n + 1)
    }
    fn main() {
        recurse(0)
    }
    """
    interp = Interpreter(max_recursion_depth=50)
    prog = check(parse(src))
    with pytest.raises(RuntimeErr) as exc_info:
        interp.run(prog)
    assert exc_info.value.code == "E0509"
    assert "maximum recursion depth exceeded" in exc_info.value.message


def test_recursion_depth_ir_diagnostic():
    src = """
    fn recurse(n: Int) -> Int {
        return recurse(n + 1)
    }
    fn main() {
        recurse(0)
    }
    """
    prog = check(parse(src))
    tir = lower_program(prog)
    from timet.ir_exec import IRExecutor
    ex = IRExecutor(tir, max_call_depth=50)
    with pytest.raises(IrExecError) as exc_info:
        ex.run()
    assert exc_info.value.code == "E0509"
