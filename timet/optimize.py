"""Optimization passes over the typed IR (Milestone 6).

Passes, all local and therefore safe by construction:
  const_fold   -- evaluate ops whose inputs are all known constants
  algebraic    -- identity simplifications (add 0, sub 0, mul 1, div 1) with
                  type-aware restrictions documented in docs/COMPILER.md
                  (e.g. no float mul-by-0: NaN/Inf; no Int div-by-1: it would
                  change Int -> Float semantics of Python `/`)
  cse          -- common subexpression elimination within marker-free
                  segments only (never across control-flow boundaries; loads
                  invalidated by intervening stores)
  copy_prop    -- propagate copy temps within their segment
  dce          -- dead code elimination (single backward sweep; only pure ops)

Higher-level rewrites (inlining, loop fusion, shape specialization) are NOT
implemented; see docs/ROADMAP.md. `optimize_program` applies the O1 pipeline
and reports per-pass statistics -- nothing is silently rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from timet.ir import TirFunction, TirInstr, TirProgram

_MARKERS = {
    "if_begin", "else", "if_end", "while_begin", "while_check", "while_end",
    "for_begin", "for_end", "nograd_begin", "nograd_end", "return",
}

#: ops that may be removed when their result is unused
_PURE_RESULT_OPS = {
    "const_int", "const_float", "const_bool", "const_str", "copy", "load",
    "make_tensor", "index",
    "add", "sub", "mul", "div", "mod", "eq", "ne", "lt", "le", "gt", "ge",
    "and", "or", "unary_-", "unary_!",
    "matmul",
}

#: ops eligible for CSE (field access is a pure read; method calls and calls
#: are NOT, they may have side effects)
_CSE_OPS = _PURE_RESULT_OPS | {op for op in ()}  # same set; explicit for clarity

_BIN_FOLD = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "mod": lambda a, b: a % b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
}


@dataclass
class OptStats:
    counts: Dict[str, int] = field(default_factory=dict)
    instrs_before: int = 0
    instrs_after: int = 0

    def add(self, key: str, n: int = 1):
        self.counts[key] = self.counts.get(key, 0) + n

    def to_json(self) -> dict:
        return {"instrs_before": self.instrs_before,
                "instrs_after": self.instrs_after,
                "passes": dict(sorted(self.counts.items()))}


def _const_repr(value) -> Tuple[str, str]:
    """(op, arg) pair for a constant python value."""
    if isinstance(value, bool):
        return "const_bool", "true" if value else "false"
    if isinstance(value, int):
        return "const_int", str(value)
    if isinstance(value, float):
        return "const_float", repr(value)
    if isinstance(value, str):
        import json as _json
        return "const_str", _json.dumps(value)
    raise ValueError(f"cannot represent constant of type {type(value).__name__}")


def _parse_const(ins: TirInstr):
    if ins.op == "const_int":
        return int(ins.args[0])
    if ins.op == "const_float":
        return float(ins.args[0])
    if ins.op == "const_bool":
        return ins.args[0] == "true"
    if ins.op == "const_str":
        import json as _json
        return _json.loads(ins.args[0])
    return None


# ---------------- pass 1: constant folding ----------------

def const_fold(fn: TirFunction, stats: OptStats) -> None:
    consts: Dict[str, object] = {}
    for ins in fn.instrs:
        v = _parse_const(ins)
        if v is not None or ins.op.startswith("const_"):
            consts[ins.result] = v
            continue
        folded: Optional[object] = None
        have = False
        if ins.op in _BIN_FOLD and len(ins.args) == 2 \
                and ins.args[0] in consts and ins.args[1] in consts:
            a, b = consts[ins.args[0]], consts[ins.args[1]]
            try:
                if ins.op in ("div", "mod") and b == 0:
                    raise ZeroDivisionError  # leave to the runtime, like the interpreter
                folded = _BIN_FOLD[ins.op](a, b)
                have = True
            except (ZeroDivisionError, TypeError, OverflowError):
                have = False
        elif ins.op == "unary_-" and ins.args[0] in consts:
            folded = -consts[ins.args[0]]
            have = True
        elif ins.op == "unary_!" and ins.args[0] in consts:
            folded = not consts[ins.args[0]]
            have = True
        if have and isinstance(folded, (bool, int, float, str)):
            op, arg = _const_repr(folded)
            ins.op, ins.args = op, [arg]
            ins.ty = {"const_bool": "Bool", "const_int": "Int",
                      "const_float": "Float", "const_str": "String"}[op]
            consts[ins.result] = folded
            stats.add("const_fold.folded")


# ---------------- pass 2: algebraic simplification ----------------

def _is_num(v, target) -> bool:
    """True iff v is the EXACT numeric constant `target` -- bools excluded:
    False == 0 and True == 1 in Python, but a Bool literal is never a valid
    Int/Float operand in well-typed IR, and we refuse to fold on that basis."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == target


def algebraic(fn: TirFunction, stats: OptStats) -> None:
    consts = {ins.result: _parse_const(ins) for ins in fn.instrs
              if ins.op.startswith("const_")}
    for ins in fn.instrs:
        if ins.op not in ("add", "sub", "mul", "div") or len(ins.args) != 2:
            continue
        a, b = ins.args
        ca, cb = consts.get(a), consts.get(b)
        numeric_int = ins.ty == "Int"
        numeric_float = ins.ty == "Float"
        repl: Optional[str] = None
        # NOTE the type guards: Int-only for +0/-0 (Python int semantics),
        # Float-safe for *1 and /1 (never changes value or type).
        if ins.op == "add" and numeric_int and (_is_num(cb, 0) or _is_num(ca, 0)):
            repl = a if _is_num(cb, 0) else b
        elif ins.op == "sub" and numeric_int and _is_num(cb, 0):
            repl = a
        elif ins.op == "mul" and (numeric_int or numeric_float):
            if _is_num(cb, 1):
                repl = a
            elif _is_num(ca, 1):
                repl = b
        elif ins.op == "div" and numeric_float and _is_num(cb, 1):
            repl = a
        if repl is not None:
            ins.op, ins.args = "copy", [repl]
            stats.add("algebraic.simplified")


# ---------------- pass 3: common subexpression elimination ----------------

def _segments(instrs: List[TirInstr]):
    """Yield (start, end) index ranges of marker-free straight-line runs.

    NOTE: markers keep their args; passes must rewrite marker args with the
    replace table of the segment they TERMINATE before resetting it, or a
    use like `return %t8` dangles when %t8's def was eliminated (this bit
    us; d60a097-era CSE dropped return args)."""
    i = 0
    n = len(instrs)
    while i < n:
        if instrs[i].op in _MARKERS:
            i += 1
            continue
        j = i
        while j < n and instrs[j].op not in _MARKERS:
            j += 1
        yield i, j
        i = j


def cse(fn: TirFunction, stats: OptStats) -> None:
    instrs = fn.instrs
    table: Dict[tuple, str] = {}
    replace: Dict[str, str] = {}
    for ins in instrs:
        ins.args = [replace.get(a, a) for a in ins.args]
        if ins.op in _MARKERS:
            # marker args were rewritten above; the segment ends here
            table.clear()
            replace.clear()
            continue
        if ins.op in ("store", "var_def"):
            for key in [key for key in table
                        if key[0] == "load" and key[1] == (ins.result,)]:
                del table[key]
            continue
        if ins.op not in _CSE_OPS or ins.result is None:
            continue
        key = (ins.op, tuple(ins.args))
        if key in table:
            replace[ins.result] = table[key]
            ins.op = "nop"
            stats.add("cse.eliminated")
        else:
            table[key] = ins.result
    fn.instrs = [i for i in fn.instrs if i.op != "nop"]


# ---------------- pass 4: copy propagation ----------------

def copy_prop(fn: TirFunction, stats: OptStats) -> None:
    instrs = fn.instrs
    aliases: Dict[str, str] = {}
    for ins in instrs:
        ins.args = [aliases.get(a, a) for a in ins.args]
        if ins.op in _MARKERS:
            aliases.clear()
            continue
        if ins.op == "copy" and ins.result is not None:
            aliases[ins.result] = ins.args[0]
            stats.add("copy_prop.propagated")


# ---------------- pass 5: dead code elimination ----------------

def dce(fn: TirFunction, stats: OptStats) -> None:
    instrs = fn.instrs
    needed: set = set()
    keep = [True] * len(instrs)
    for i in range(len(instrs) - 1, -1, -1):
        ins = instrs[i]
        if ins.result is None or ins.op not in _PURE_RESULT_OPS:
            needed.update(ins.args)
            continue
        if ins.result in needed:
            needed.update(ins.args)
        else:
            keep[i] = False
            stats.add("dce.removed")
    fn.instrs = [ins for ins, k in zip(instrs, keep) if k]


# ---------------- pipeline ----------------

def optimize_function(fn: TirFunction, level: int = 1,
                      stats: Optional[OptStats] = None) -> OptStats:
    stats = stats or OptStats()
    stats.instrs_before += len(fn.instrs)
    if level >= 1:
        const_fold(fn, stats)
        algebraic(fn, stats)
        copy_prop(fn, stats)
        const_fold(fn, stats)   # copy propagation can expose new constants
        cse(fn, stats)
        copy_prop(fn, stats)
        dce(fn, stats)
    stats.instrs_after += len(fn.instrs)
    return stats


def optimize_program(prog: TirProgram, level: int = 1) -> Tuple[TirProgram, OptStats]:
    """Optimize every function in place AND return (prog, stats).

    Deterministic: passes are ordered, dicts iterate in insertion order, and
    no randomness or wall-clock input is used anywhere.
    """
    stats = OptStats()
    for fn in prog.functions:
        optimize_function(fn, level=level, stats=stats)
    return prog, stats
