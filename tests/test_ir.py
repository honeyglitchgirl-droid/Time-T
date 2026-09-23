import json
from timet.parser import parse
from timet.typechecker import check
from timet.ir import lower_program, TirProgram


SRC = """
fn add(a: Int, b: Int) -> Int {
    let c = a + b
    return c
}
"""


def test_lowering_produces_function():
    program = check(parse(SRC))
    tir = lower_program(program)
    assert len(tir.functions) == 1
    fn = tir.functions[0]
    assert fn.name == "add"
    assert fn.params == ["a", "b"]
    ops = [i.op for i in fn.instrs]
    assert "add" in ops
    assert "return" in ops


def test_json_round_trip():
    program = check(parse(SRC))
    tir = lower_program(program)
    s = tir.to_json_str()
    restored = TirProgram.from_json_str(s)
    assert restored.to_json() == tir.to_json()


def test_deterministic_lowering():
    program1 = check(parse(SRC))
    program2 = check(parse(SRC))
    tir1 = lower_program(program1)
    tir2 = lower_program(program2)
    assert tir1.to_json() == tir2.to_json()


def test_render_is_human_readable():
    program = check(parse(SRC))
    tir = lower_program(program)
    text = tir.render()
    assert "fn add" in text
    assert "return" in text


def test_if_lowering():
    src = """
    fn f(x: Int) -> Int {
        if x > 0 {
            return 1
        } else {
            return 0
        }
    }
    """
    program = check(parse(src))
    tir = lower_program(program)
    ops = [i.op for i in tir.functions[0].instrs]
    assert "if_begin" in ops
    assert "if_end" in ops
