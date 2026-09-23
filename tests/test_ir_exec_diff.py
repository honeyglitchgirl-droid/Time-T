"""Differential testing (master prompt section 29): the AST interpreter, the
IR executor, and the OPTIMIZED IR executor must produce byte-identical
output on every example program and on generated random programs.

This is what makes `time-t inspect --ir` trustworthy: the IR you see is the
IR that runs.
"""
import random
import pathlib

import pytest

from timet.parser import parse
from timet.typechecker import check
from timet.ir import lower_program
from timet.ir_exec import run_program
from timet.optimize import optimize_program
from timet.interpreter import run_source

EXAMPLES = sorted(pathlib.Path("examples").glob("*.tt"))


def _run_ast(src: str, filename: str):
    out = []
    run_source(src, filename=filename, stdout_write=out.append)
    return out


def _run_ir(src: str, filename: str, level: int):
    out = []
    tir = lower_program(check(parse(src, filename)))
    if level >= 1:
        tir, _ = optimize_program(tir, level=level)
    run_program(tir, stdout_write=out.append)
    return out


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_matches_on_all_three_engines(path):
    src = path.read_text()
    expected = path.with_suffix(".expected").read_text().splitlines()
    assert _run_ast(src, str(path)) == expected, "AST interpreter diverged from expected"
    assert _run_ir(src, str(path), level=0) == expected, "IR executor (-O0) diverged"
    assert _run_ir(src, str(path), level=1) == expected, "IR executor (-O1) diverged"


# ---------------- generated random straight-line programs ----------------

def _gen_program(rng: random.Random, n_lets: int) -> str:
    """Random integer/float arithmetic over previous bindings. No string
    ops, no &&/|| (IR is eager there), no division (zero-divisor risk), no
    mutable vars (covered by the example/loop tests)."""
    lines = ["fn main() {"]
    names = []
    for i in range(n_lets):
        def atom():
            if names and rng.random() < 0.6:
                return rng.choice(names)
            if rng.random() < 0.5:
                return str(rng.randint(0, 20))
            return f"{rng.uniform(0, 9):.3f}"
        a, b, c = atom(), atom(), atom()
        op1, op2 = rng.choice(["+", "-", "*"]), rng.choice(["+", "-", "*"])
        lines.append(f"    let v{i} = {a} {op1} {b} {op2} {c}")
        names.append(f"v{i}")
    lines.append(f"    print({rng.choice(names)})")
    lines.append("}")
    return "\n".join(lines)


def test_random_programs_agree_across_engines():
    rng = random.Random(20260923)
    for seed in range(40):
        rng_i = random.Random(rng.randint(0, 10 ** 9))
        src = _gen_program(rng_i, rng_i.randint(3, 15))
        ast = _run_ast(src, "<gen>")
        ir0 = _run_ir(src, "<gen>", level=0)
        ir1 = _run_ir(src, "<gen>", level=1)
        assert ast == ir0 == ir1, (
            f"differential mismatch on generated program:\n{src}\n"
            f"ast={ast}\nir-O0={ir0}\nir-O1={ir1}")


def test_random_programs_with_conditionals_agree():
    rng = random.Random(7)
    for _ in range(30):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        c, d = rng.uniform(0, 9), rng.uniform(0, 9)
        src = f"""
        fn pick(x: Int) -> Int {{
            if x > 5 {{ return x * 2 }} else {{ return x + 1 }}
        }}
        fn main() {{
            let p = pick({a})
            let q = pick({b})
            var t = 0
            var i = 0
            while i < {rng.randint(1, 5)} {{
                t = t + p + q
                i = i + 1
            }}
            print(t)
            print({c:.3f} * {d:.3f})
        }}
        """
        ast = _run_ast(src, "<gen>")
        ir0 = _run_ir(src, "<gen>", level=0)
        ir1 = _run_ir(src, "<gen>", level=1)
        assert ast == ir0 == ir1, (
            f"differential mismatch:\n{src}\nast={ast}\nir0={ir0}\nir1={ir1}")
