import pytest
from timet.parser import parse
from timet.typechecker import check, TypeError_
from timet.types import TInt, TFloat, TBool, TTensor


def check_src(src):
    return check(parse(src))


def test_int_literal_type():
    prog = check_src("let x = 5")
    assert prog.statements[0].value.ty == TInt()


def test_float_widening_allowed():
    prog = check_src("let x: Float = 5")
    # int literal widens to float type annotation
    assert prog.statements[0].value.ty == TInt()


def test_type_mismatch_raises():
    with pytest.raises(TypeError_) as exc:
        check_src('let x: Int = "hello"')
    assert exc.value.code == "E0202"
    assert exc.value.expected == "Int"
    assert exc.value.actual == "String"


def test_undefined_name_raises():
    with pytest.raises(TypeError_) as exc:
        check_src("let x = y + 1")
    assert exc.value.code == "E0203"


def test_immutable_assignment_raises():
    with pytest.raises(TypeError_) as exc:
        check_src("let x = 1\nx = 2")
    assert exc.value.code == "E0301"
    assert "immutable" in exc.value.message


def test_mutable_assignment_ok():
    prog = check_src("var x = 1\nx = 2")
    assert prog is not None


def test_function_signature_checked():
    src = """
    fn add(a: Int, b: Int) -> Int {
        return a + b
    }
    """
    check_src(src)  # should not raise


def test_binop_numeric_requirement():
    with pytest.raises(TypeError_) as exc:
        check_src('let x = "a" - "b"')
    assert exc.value.code == "E0204"


def test_bool_ops():
    prog = check_src("let x = true && false")
    assert prog.statements[0].value.ty == TBool()


def test_comparisons_produce_bool():
    prog = check_src("let x = 1 < 2")
    assert prog.statements[0].value.ty == TBool()


def test_if_condition_must_be_bool():
    with pytest.raises(TypeError_):
        check_src("if 1 { print(1) }")


def test_tensor_literal_type():
    prog = check_src("let t = [1.0, 2.0, 3.0]")
    assert isinstance(prog.statements[0].value.ty, TTensor)


def test_unknown_type_annotation_error():
    with pytest.raises(TypeError_) as exc:
        check_src("let x: Frobnicate = 1")
    assert exc.value.code == "E0200"


def test_forward_reference_functions():
    src = """
    fn a() -> Int { return b() }
    fn b() -> Int { return 1 }
    """
    check_src(src)  # should not raise thanks to hoisting


def test_undefined_function_call():
    with pytest.raises(TypeError_) as exc:
        check_src("let x = totally_undefined(1)")
    assert exc.value.code == "E0205"
