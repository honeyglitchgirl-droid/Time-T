"""Tests for struct definitions, instantiation, and field access."""
import pytest
from timet.interpreter import run_source
from timet.parser import parse
from timet.typechecker import check, TypeError_
from timet.ir import lower_program
from timet.ir_exec import run_program


def test_struct_declaration_and_instantiation():
    src = """
    struct Point {
        x: Int,
        y: Int
    }

    fn main() {
        let p = Point { x: 10, y: 20 }
        print(p.x)
        print(p.y)
    }
    """
    out = []
    run_source(src, stdout_write=out.append)
    assert out == ["10", "20"]


def test_struct_typechecking_field_mismatch():
    src = """
    struct Point {
        x: Int,
        y: Int
    }

    fn main() {
        let p = Point { x: 10, y: "bad" }
    }
    """
    with pytest.raises(TypeError_) as exc_info:
        check(parse(src))
    assert "E0202" in exc_info.value.code


def test_struct_typechecking_unknown_field():
    src = """
    struct Point {
        x: Int
    }

    fn main() {
        let p = Point { x: 10, z: 20 }
    }
    """
    with pytest.raises(TypeError_) as exc_info:
        check(parse(src))
    assert "E0211" in exc_info.value.code


def test_struct_ir_execution_matches_ast():
    src = """
    struct Config {
        lr: Float,
        steps: Int
    }

    fn main() {
        let c = Config { lr: 0.01, steps: 100 }
        print(c.steps)
    }
    """
    out_ast = []
    run_source(src, stdout_write=out_ast.append)

    prog = check(parse(src))
    tir = lower_program(prog)
    out_ir = []
    run_program(tir, stdout_write=out_ir.append)

    assert out_ast == out_ir == ["100"]
