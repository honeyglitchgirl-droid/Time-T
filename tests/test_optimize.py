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


# ---------------- regression: the dangling-return-arg bug ----------------

def test_cse_rewrites_marker_args_of_eliminated_temps():
    """Regression (found by differential testing): an eliminated def whose
    ONLY later use was a control marker (return) must still be rewritten;
    leaving `return %tN` referencing a nop'd def is a hard executor error."""
    src = """fn f() -> Int {
        let a = 5
        let b = a
        let c = 6
        let d = 6
        return c
    }"""
    tir, _ = opt(src)
    f = {fn.name: fn for fn in tir.functions}["f"]
    defined = {i.result for i in f.instrs if i.result}
    for ins in f.instrs:
        for a in ins.args:
            if a.startswith("%"):
                assert a in defined, f"dangling temp {a} in {ins}"


def test_cse_rewrites_call_args_of_eliminated_temps():
    """Regression (found on 02_calculator): duplicate consts merged by CSE
    must rewrite call: args, not just other pure ops."""
    src = """
    fn mul(a: Int, b: Int) -> Int { return a * b }
    fn main() {
        print(mul(6, 7))
        print(mul(6, 7))
    }
    """
    tir, _ = opt(src)
    f = {fn.name: fn for fn in tir.functions}["main"]
    defined = {i.result for i in f.instrs if i.result}
    for ins in f.instrs:
        for a in ins.args:
            if a.startswith("%"):
                assert a in defined, f"dangling temp {a} in ins {ins}"


def test_cse_never_crosses_control_markers():
    """An expression computed before an if-block must not be merged with the
    same expression inside the block's then-branch."""
    src = """fn f(x: Int) -> Int {
        let a = x * 2
        if x > 0 {
            let b = x * 2
            return b
        }
        return a
    }"""
    tir, _ = opt(src)
    f = {fn.name: fn for fn in tir.functions}["f"]
    muls = [i for i in f.instrs if i.op == "mul"]
    assert len(muls) == 2, f"CSE must not cross if_begin/if_end, got {len(muls)} muls"


def test_mod_by_zero_not_folded_either():
    tir, _ = opt("fn f() -> Int { return 7 % 0 }")
    assert "mod" in ops(tir)


# ---------------- whole-corpus invariant: no dangling temps, ever ----------------

def test_optimized_examples_have_no_dangling_temp_uses():
    """For every example program, after O1: every %-temp used by any instr
    (including control markers and calls) is defined in the same function.
    This invariant would have caught both historical optimizer bugs."""
    import pathlib
    from timet.modules import ModuleLoader
    for path in sorted(pathlib.Path("examples").glob("*.tt")):
        src = path.read_text()
        loader = ModuleLoader()
        tir = lower_program(check(parse(src, str(path)), filename=str(path),
                                  loader=loader),
                            loader=loader, importer_path=str(path))
        tir, _ = optimize_program(tir, level=1)
        for fn in tir.functions:
            defined = {i.result for i in fn.instrs if i.result} | set(fn.params)
            for ins in fn.instrs:
                for arg in ins.args:
                    if arg.startswith("%"):
                        assert arg in defined, \
                            f"{path.name}:{fn.name}: dangling {arg} in {ins}"
