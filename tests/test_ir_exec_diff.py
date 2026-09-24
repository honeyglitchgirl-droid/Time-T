"""Differential tests (§30): AST interpreter vs IR executor (-O0 and -O1).

Every example program and a battery of generated straight-line programs
must produce BYTE-IDENTICAL stdout on all three engines. This suite is the
load-bearing claim for "the IR executor is correct": it does not trust the
lowering or the optimizer, it compares outputs.
"""
import random
from pathlib import Path

import pytest

from timet.parser import parse
from timet.typechecker import check
from timet.ir import lower_program
from timet.ir_exec import run_program
from timet.optimize import optimize_program
from timet.interpreter import run_source
from timet.modules import ModuleLoader

EXAMPLES = sorted(Path("examples").glob("*.tt"))


def run_ast(src: str, filename: str):
    out = []
    run_source(src, filename=filename, stdout_write=out.append)
    return out


def run_ir(src: str, filename: str, level: int):
    out = []
    loader = ModuleLoader()
    tir = lower_program(check(parse(src, filename), filename=filename,
                              loader=loader),
                        loader=loader, importer_path=filename)
    if level:
        tir, _ = optimize_program(tir, level=level)
    run_program(tir, stdout_write=out.append)
    return out


def _lines_match(actual_lines, expected_lines, atol=1e-3, rtol=1e-3) -> bool:
    if actual_lines == expected_lines:
        return True
    import re
    import numpy as np
    if len(actual_lines) != len(expected_lines):
        return False
    num_pat = re.compile(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?')
    for act, exp in zip(actual_lines, expected_lines):
        if act == exp:
            continue
        act_nums = [float(x) for x in num_pat.findall(act)]
        exp_nums = [float(x) for x in num_pat.findall(exp)]
        if len(act_nums) == len(exp_nums) and len(act_nums) > 0:
            skel_act = num_pat.sub('#', act)
            skel_exp = num_pat.sub('#', exp)
            if skel_act == skel_exp and np.allclose(act_nums, exp_nums, atol=atol, rtol=rtol):
                continue
        return False
    return True


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_matches_on_all_three_engines(path):
    src = path.read_text()
    expected = path.with_suffix(".expected").read_text().splitlines()
    ast_out = run_ast(src, str(path))
    ir_o0 = run_ir(src, str(path), level=0)
    ir_o1 = run_ir(src, str(path), level=1)
    assert _lines_match(ast_out, expected), f"AST interpreter diverged from expected: {ast_out} vs {expected}"
    assert _lines_match(ir_o0, expected), f"IR executor (-O0) diverged: {ir_o0} vs {expected}"
    assert _lines_match(ir_o1, expected), f"IR executor (-O1) diverged: {ir_o1} vs {expected}"
    # Cross-engine bit-exact equivalence
    assert ast_out == ir_o0 == ir_o1, f"Engines diverged from each other: AST={ast_out}, O0={ir_o0}, O1={ir_o1}"


SNIPPETS = [
    # arithmetic + precedence
    "print(1 + 2 * 3 - 4)\nprint(7 % 3)\nprint(2.5 * 4.0)\n",
    # bools and comparisons (boolean ops via && / ||)
    "print((1 < 2) && (2 < 3))\nprint(!(1 == 2) || false)\n",
    # if/else-if/else (chained blocks; parser desugars else-if into else { if })
    """
let x = 5
if x < 3 {
    print("low")
} else if x < 7 {
    print("mid")
} else {
    print("high")
}
""",
    # while loop with var mutation
    """
var i = 0
var acc = 0
while i < 5 {
    acc = acc + i * i
    i = i + 1
}
print(acc)
""",
    # functions with early return from an if block
    """
fn fib(n: Int) -> Int {
    if n < 2 { return n }
    return fib(n - 1) + fib(n - 2)
}
print(fib(12))
""",
    # tensors: construction, indexing, reductions, matmul
    """
let t = [1.0, 2.0, 3.0]
print(t.sum())
print(t[1].item())
let m = [[1.0, 2.0], [3.0, 4.0]]
print(m.matmul(m).sum())
""",
    # autodiff smoke: grad through re-wirings of the same value
    """
let x = tensor([[0.0, 1.0]], grad=true)
let y = x * x + x
y.sum().backward()
print(x.grad)
""",
    # no_grad block + in-place schematic update (var + method call mix)
    """
var w = tensor([1.0, 2.0], grad=true)
let loss = (w * w).sum()
loss.backward()
no_grad {
    w.sub_(w.grad * 0.5)
}
print(w)
""",
    # while loop with break/continue
    """
var i = 0
var acc = 0
while true {
    i = i + 1
    if i % 2 == 0 { continue }
    if i > 9 { break }
    acc = acc + i
}
print(acc)
""",
    # int-only const-heavy code that O1 should mostly fold
    """
let a = 2
let b = 3
let c = a * b + a + b
print(c * c)
""",
]


@pytest.mark.parametrize("src", SNIPPETS)
def test_snippet_three_engines_match(src):
    a = run_ast(src, "<snippet>")
    b = run_ir(src, "<snippet>", 0)
    c = run_ir(src, "<snippet>", 1)
    assert a == b == c and a, (a, b, c)


def _gen_straightline_program(rng: random.Random) -> str:
    """Random straight-line program over ints/floats with determinable output.

    Avoids division/modulo by zero by construction: right operand of div/mod
    is a literal in 1..9. Floats avoided for modulo (float % works but repr
    noise is uninteresting here).
    """
    lines = []
    vars_ = []
    for i in range(rng.randint(3, 9)):
        op = rng.choice(["+", "-", "*", "/", "%"])
        lhs = rng.choice(vars_ + [str(rng.randint(0, 9))])
        rhs = rng.choice(vars_ + [str(rng.randint(0, 9))])
        if op in ("/", "%"):
            rhs = str(rng.randint(1, 9))
        v = f"v{i}"
        lines.append(f"let {v} = ({lhs}) {op} ({rhs})")
        vars_.append(v)
        if rng.random() < 0.35:
            lines.append(f"print({v})")
    if vars_:
        lines.append(f"print({vars_[-1]})")
    else:
        lines.append("print(0)")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("seed", range(40))
def test_generated_straightline_programs_match(seed):
    rng = random.Random(seed)
    src = _gen_straightline_program(rng)
    a = run_ast(src, f"<gen{seed}>")
    b = run_ir(src, f"<gen{seed}>", 0)
    c = run_ir(src, f"<gen{seed}>", 1)
    assert a == b == c, (src, a, b, c)


# ---------------- legacy random generators (kept from v0.2.0 suite) ----------------

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
        ast = run_ast(src, "<gen>")
        ir0 = run_ir(src, "<gen>", level=0)
        ir1 = run_ir(src, "<gen>", level=1)
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
        ast = run_ast(src, "<gen>")
        ir0 = run_ir(src, "<gen>", level=0)
        ir1 = run_ir(src, "<gen>", level=1)
        assert ast == ir0 == ir1, (
            f"differential mismatch:\n{src}\nast={ast}\nir0={ir0}\nir1={ir1}")
