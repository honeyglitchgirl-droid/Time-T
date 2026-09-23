"""Unit tests for the IR executor (timet/ir_exec.py)."""
import pytest

from timet.parser import parse
from timet.typechecker import check
from timet.ir import lower_program
from timet.ir_exec import run_program, IrExecError


def run_ir(src: str):
    out = []
    tir = lower_program(check(parse(src)))
    run_program(tir, stdout_write=out.append)
    return out


def test_arithmetic_and_lets():
    out = run_ir("""
    fn main() {
        let a = 3 + 4 * 2
        let b = (a - 1) / 2
        print(a)
        print(b)
    }
    """)
    assert out == ["11", "5.0"]


def test_if_else_branches():
    out = run_ir("""
    fn classify(x: Int) -> Int {
        if x > 0 { return 1 } else { return 0 }
    }
    fn main() {
        print(classify(5))
        print(classify(-2))
    }
    """)
    assert out == ["1", "0"]


def test_while_loop_reevaluates_condition():
    out = run_ir("""
    fn main() {
        var i = 0
        var total = 0
        while i < 5 {
            total = total + i
            i = i + 1
        }
        print(total)
        print(i)
    }
    """)
    assert out == ["10", "5"]


def test_var_visible_across_function_call_via_static_scope():
    out = run_ir("""
    var counter = 0
    fn bump() -> Int {
        counter = counter + 1
        return counter
    }
    fn main() {
        print(bump())
        print(bump())
        print(counter)
    }
    """)
    assert out == ["1", "2", "2"]


def test_for_loop_over_tensor_elements():
    out = run_ir("""
    fn main() {
        var total = 0.0
        for x in [1.0, 2.0, 3.0] {
            total = total + x.item()
        }
        print(total)
    }
    """)
    # iterating a 1-D tensor yields 0-d tensors; .item() unwraps to float
    assert out == ["6.0"]


def test_recursion():
    out = run_ir("""
    fn fib(n: Int) -> Int {
        if n < 2 { return n }
        return fib(n - 1) + fib(n - 2)
    }
    fn main() { print(fib(10)) }
    """)
    assert out == ["55"]


def test_string_handling_round_trip():
    out = run_ir('fn main() { let s = "a\\"b\\n" print(s) }')
    assert out == ['a"b\n']


def test_keyword_arguments():
    out = run_ir("""
    fn main() {
        let t = tensor([1, 2], grad=true)
        print(t.requires_grad)
    }
    """)
    assert out == ["true"]


def test_lambda_raises_clear_error_not_wrong_answer():
    """Lambdas are deliberately not lowered to IR (DD-9): the executor must
    fail loudly with the unsupported-kind in the message, never guess."""
    with pytest.raises(IrExecError) as e:
        run_ir("""
        fn main() {
            let add = fn(a: Int, b: Int) -> Int { return a + b }
            print(add(1, 2))
        }
        """)
    assert "Lambda" in str(e.value)
    assert "--via-ir" in str(e.value) or "AST interpreter" in str(e.value)


def test_if_expression_raises_clear_error():
    with pytest.raises(IrExecError) as e:
        run_ir("""
        fn main() {
            let x = if 1 > 0 { 10 } else { 20 }
            print(x)
        }
        """)
    assert "IfExpr" in str(e.value)


def test_undefined_name_message():
    tir = lower_program(check(parse("fn main() { let x = 1 }")))
    # hand-corrupt: append a load of a nonexistent var
    from timet.ir import TirInstr
    tir.functions[-1].instrs.append(TirInstr("load", ["nope"], "%t999", "Int"))
    with pytest.raises(IrExecError) as e:
        run_program(tir, stdout_write=lambda s: None)
    assert "nope" in str(e.value)
