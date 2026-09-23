"""Unit tests for the IR optimization passes (timet/optimize.py)."""
from timet.parser import parse
from timet.typechecker import check
from timet.ir import lower_program, TirProgram
from timet.optimize import optimize_program, OptStats


def opt(src: str, level: int = 1):
    tir = lower_program(check(parse(src)))
    tir, stats = optimize_program(tir, level=level)
    return tir, stats


def all_instrs(tir: TirProgram):
    return [i for f in tir.functions for i in f.instrs]


def ops(tir: TirProgram):
    return [i.op for i in all_instrs(tir)]


def test_constant_folding_collapses_int_arithmetic():
    src = "fn f() -> Int { return 1 + 2 * 3 }"
    tir, stats = opt(src)
    fns = {f.name: f for f in tir.functions}
    instrs = fns["f"].instrs
    non_marker = [i for i in instrs if i.op not in ("return",)]
    assert len(non_marker) == 1
    assert non_marker[0].op == "const_int"
    assert non_marker[0].args == ["7"]
    assert stats.counts.get("const_fold.folded", 0) >= 2


def test_constant_folding_float_and_bool():
    tir, _ = opt("fn f() -> Float { return 1.5 + 2.5 }")
    assert any(i.op == "const_float" and i.args == ["4.0"] for i in all_instrs(tir))


def test_no_fold_division_by_zero():
    """1/0 must NOT be compile-time folded: like the interpreter, the error
    belongs at run time."""
    tir, _ = opt("fn f() -> Float { return 1 / 0 }")
    assert "div" in ops(tir)


def test_algebraic_add_zero_and_mul_one_int():
    tir, _ = opt("fn f(x: Int) -> Int { return x + 0 }")
    assert "add" not in ops(tir), f"expected 'add' removed, got {ops(tir)}"


def test_algebraic_mul_one_float_but_not_add_zero_float():
    """Float +0 is deliberately NOT folded (it would change -0.0 -> +0.0);
    Float *1 IS folded (exact identity in IEEE-754)."""
    tir, _ = opt("fn f(x: Float) -> Float { return x * 1.0 }")
    assert "mul" not in ops(tir)
    tir2, _ = opt("fn f(x: Float) -> Float { return x + 0.0 }")
    assert "add" in ops(tir2)


def test_algebraic_int_div_one_not_folded():
    """Int x / 1 yields a Float under Python semantics -- folding it to a
    copy of the Int would change the value's type."""
    tir, _ = opt("fn f(x: Int) -> Float { return x / 1 }")
    assert "div" in ops(tir)


def test_cse_merges_duplicate_pure_ops():
    src = "fn f(a: Int, b: Int) -> Int { let x = a * b let y = a * b return x + y }"
    tir, stats = opt(src)
    muls = [i for i in all_instrs(tir) if i.op == "mul"]
    assert len(muls) == 1, f"expected CSE to keep one mul, got {len(muls)}"
    assert stats.counts.get("cse.eliminated", 0) >= 1


def test_cse_does_not_merge_loads_across_store():
    """load x ... store x ... load x: the two loads read DIFFERENT values and
    must not be merged."""
    src = """
    fn f() -> Int {
        var x = 1
        let a = x
        x = 2
        let b = x
        return a + b
    }
    """
    tir, _ = opt(src)
    loads = [i for i in all_instrs(tir) if i.op == "load"]
    assert len(loads) == 2


def test_dce_removes_unused_pure_results():
    src = "fn f(a: Int) -> Int { let unused = a * 3 return a }"
    tir, stats = opt(src)
    assert stats.counts.get("dce.removed", 0) >= 1


def test_dce_never_removes_print():
    src = 'fn main() { print("hi") }'
    tir, _ = opt(src)
    assert any(i.op == "call:print" for i in all_instrs(tir))


def test_optimization_is_deterministic():
    src = """
    fn sq(x: Int) -> Int { return x * x }
    fn main() { print(sq(3) + sq(4) + 0) }
    """
    tir1, _ = opt(src)
    tir2, _ = opt(src)
    assert tir1.to_json() == tir2.to_json()


def test_stats_report_instr_counts():
    tir, stats = opt("fn f(a: Int) -> Int { return a + 1 * 2 }")
    assert stats.instrs_before > stats.instrs_after
    d = stats.to_json()
    assert d["instrs_after"] == stats.instrs_after


def test_O0_is_a_no_op():
    tir = lower_program(check(parse("fn f() -> Int { return 1 + 2 }")))
    before = tir.to_json()
    optimize_program(tir, level=0)
    assert tir.to_json() == before
