"""Execute Time-T IR directly (Milestone 6).

The IR executor exists for two reasons:
  1. it makes `time-t inspect --ir` output *load-bearing*: the same artifact
     can be run, so lowering bugs cannot hide behind the AST interpreter;
  2. it enables DIFFERENTIAL testing (master prompt section 29): every
     example program is run through the AST interpreter and through the IR
     executor (unoptimized and optimized) and must produce byte-identical
     output. See tests/test_ir_exec_diff.py.

Semantics notes (docs/DESIGN_DECISIONS.md DD-9):
  - `var`s live in named storage frames that chain to the caller-visible
    parent frame, mirroring the interpreter's Environment chain.
  - `let`s are SSA temps held in the current frame.
  - Structured control markers are walked as a block tree (see _build_tree);
    while conditions are re-executed every iteration by construction.
  - Anything the lowerer could not represent (unsupported_stmt/expr) raises
    IrExecError immediately -- the executor never guesses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from timet import tensor as T
from timet.autodiff import no_grad
from timet.diagnostics import Diagnostic
from timet.ir import TirFunction, TirInstr, TirProgram, MAIN_FN


class IrExecError(Diagnostic):
    pass


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class Frame:
    """One call frame: SSA temps + local var storage, chained to a parent.

    Reads walk the chain; `var_def` always defines locally; `store` writes
    the frame where the name already lives (mirrors Environment.set).
    """

    def __init__(self, parent: Optional["Frame"] = None):
        self.values: Dict[str, Any] = {}
        self.local_vars: set = set()
        self.parent = parent

    def get(self, name: str, op: str) -> Any:
        frame: Optional[Frame] = self
        while frame is not None:
            if name in frame.values:
                return frame.values[name]
            frame = frame.parent
        raise IrExecError(code="E0700",
                          message=f"IR executor: undefined name '{name}' (op {op})",
                          stage="ir-exec")

    def set_temp(self, name: str, value: Any):
        self.values[name] = value

    def define_var(self, name: str, value: Any):
        self.values[name] = value
        self.local_vars.add(name)

    def store_var(self, name: str, value: Any):
        """Write to the nearest frame that holds `name` (mirrors
        Environment.set). Static typing guarantees the name exists; the
        fall-through define is unreachable for well-typed programs."""
        frame: Optional[Frame] = self
        while frame is not None:
            if name in frame.values:
                frame.values[name] = value
                return
            frame = frame.parent
        self.define_var(name, value)

    def contains_deep(self, name: str) -> bool:
        frame: Optional[Frame] = self
        while frame is not None:
            if name in frame.values:
                return True
            frame = frame.parent
        return False


def _var_names(frame: Frame) -> set:
    # names that are real `var` storage in an ancestor frame: anything in
    # .values that is not an SSA temp (temps start with "%")
    return {k for k in frame.values if not k.startswith("%")}


# ---------------- structured block tree ----------------

@dataclass
class _Seq:
    instrs: List[TirInstr]


@dataclass
class _If:
    cond: str
    then_blocks: List
    else_blocks: List


@dataclass
class _While:
    cond_instrs: List[TirInstr]   # includes the final while_check
    cond: str
    body: List


@dataclass
class _For:
    coll: str
    item: str
    body: List


@dataclass
class _NoGrad:
    body: List


def _build_tree(instrs: List[TirInstr]) -> List:
    """Parse the flat instruction stream into nested blocks using the
    structured markers. Mismatched markers are an internal error (the
    lowerer is the only producer, and it maintains nesting by
    construction).

    While loops need one twist: the instructions computing the loop
    condition live BETWEEN while_begin and while_check, i.e. in their own
    straight-line run, which we capture verbatim and re-execute every
    iteration. A condition is a single expression, so no markers can appear
    inside that run.
    """
    root: List = []
    # frames: (kind, blocks, meta); kind in
    #   "if_then" | "if_else" | "while_cond" | "while_body" | "for" | "nograd"
    stack: List = [("root", root, {})]

    pending_seq: List[TirInstr] = []

    def cur_blocks() -> List:
        return stack[-1][1]

    def flush_seq():
        nonlocal pending_seq
        if pending_seq:
            cur_blocks().append(_Seq(pending_seq))
            pending_seq = []

    def push_node(node):
        flush_seq()
        cur_blocks().append(node)

    for ins in instrs:
        op = ins.op
        if op == "if_begin":
            flush_seq()
            stack.append(("if_then", [], {"cond": ins.args[0]}))
        elif op == "else":
            _require(stack[-1][0] == "if_then", "'else' without matching 'if_begin'")
            flush_seq()  # trailing then-branch instrs belong to the then list
            kind, blocks, meta = stack.pop()
            meta["then"] = blocks
            stack.append(("if_else", [], meta))
        elif op == "if_end":
            _require(stack[-1][0] in ("if_then", "if_else"),
                     "'if_end' without matching 'if_begin'")
            flush_seq()
            kind, blocks, meta = stack.pop()
            then_blocks = meta["then"] if kind == "if_else" else blocks
            else_blocks = blocks if kind == "if_else" else []
            push_node(_If(meta["cond"], then_blocks, else_blocks))
        elif op == "while_begin":
            flush_seq()
            stack.append(("while_cond", [], {}))
        elif op == "while_check":
            kind = stack[-1][0]
            _require(kind == "while_cond", "'while_check' without matching 'while_begin'")
            # pending_seq IS the condition computation (straight-line only).
            cond_instrs = list(pending_seq)
            pending_seq = []
            stack.pop()
            stack.append(("while_body", [], {"cond_instrs": cond_instrs,
                                             "cond": ins.args[0]}))
        elif op == "while_end":
            _require(stack[-1][0] == "while_body",
                     "'while_end' without matching 'while_begin'")
            flush_seq()
            kind, blocks, meta = stack.pop()
            push_node(_While(meta["cond_instrs"], meta["cond"], blocks))
        elif op == "for_begin":
            flush_seq()
            stack.append(("for", [], {"coll": ins.args[0], "item": ins.result}))
        elif op == "for_end":
            _require(stack[-1][0] == "for", "'for_end' without matching 'for_begin'")
            flush_seq()
            kind, blocks, meta = stack.pop()
            push_node(_For(meta["coll"], meta["item"], blocks))
        elif op == "nograd_begin":
            flush_seq()
            stack.append(("nograd", [], {}))
        elif op == "nograd_end":
            _require(stack[-1][0] == "nograd",
                     "'nograd_end' without matching 'nograd_begin'")
            flush_seq()
            kind, blocks, meta = stack.pop()
            push_node(_NoGrad(blocks))
        else:
            pending_seq.append(ins)
    flush_seq()
    if len(stack) != 1:
        raise IrExecError(code="E0701", message="IR executor: unbalanced control markers",
                          stage="ir-exec")
    return root


def _require(ok: bool, msg: str):
    if not ok:
        raise IrExecError(code="E0701", message=f"IR executor: {msg}", stage="ir-exec")


# ---------------- executor ----------------

_BINOPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "mod": lambda a, b: a % b,
    "matmul": lambda a, b: a @ b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: not (a == b),
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    # NOTE: eager, not short-circuiting (DD-9); matches Python's `and`/`or`
    # value semantics for side-effect-free operands.
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
}


class IRExecutor:
    def __init__(self, program: TirProgram, stdout_write=None):
        self.program = program
        self.functions = program.function_table()
        self.stdout_write = stdout_write or (lambda s: print(s))
        self.main_frame: Optional[Frame] = None

    # -- public --

    def run(self):
        self.main_frame = Frame()
        # engine globals (nn / optim / train) are ordinary values in the
        # main frame, reachable from every call frame via the chain (DD-10).
        from timet.interpreter import ENGINE_GLOBALS
        for name, value in ENGINE_GLOBALS.items():
            self.main_frame.define_var(name, value)
        main = self.functions.get(MAIN_FN)
        if main is not None:
            try:
                self._exec_function_body(main, self.main_frame)
            except _Return:
                pass
        # Mirror the AST interpreter: after top-level statements, a
        # user-defined `fn main()` is invoked.
        user_main = self.functions.get("main")
        if user_main is not None:
            self._call_static("main", [], {}, self.main_frame)

    # -- functions --

    def _exec_function_body(self, fn: TirFunction, frame: Frame):
        blocks = _build_tree(fn.instrs)
        self._exec_blocks(blocks, frame)

    def _call_static(self, name: str, args, kwargs, frame: Frame):
        target = self.functions.get(name)
        if target is not None:
            if len(args) != len(target.params):
                raise IrExecError(
                    code="E0702",
                    message=f"IR executor: '{name}' expects {len(target.params)} arg(s), got {len(args)}",
                    stage="ir-exec")
            # static scoping: all lowered functions are top-level, so every
            # call frame's parent is the main frame (DD-9).
            child = Frame(parent=self.main_frame)
            for pname, val in zip(target.params, args):
                child.define_var(pname, val)
            try:
                self._exec_function_body(target, child)
            except _Return as r:
                return r.value
            return None
        builtin = _BUILTINS.get(name)
        if builtin is None:
            raise IrExecError(code="E0703",
                              message=f"IR executor: cannot call '{name}' -- unknown function "
                                      f"(higher-order functions are not supported in IR yet; DD-9)",
                              stage="ir-exec")
        if name == "print":
            return builtin(self.stdout_write, *args, **kwargs)
        return builtin(*args, **kwargs)

    # -- blocks --

    def _exec_blocks(self, blocks: List, frame: Frame):
        for block in blocks:
            if isinstance(block, _Seq):
                for ins in block.instrs:
                    self._exec_instr(ins, frame)
            elif isinstance(block, _If):
                cond = frame.get(block.cond, "if_begin")
                self._exec_blocks(block.then_blocks if cond else block.else_blocks,
                                  frame)
            elif isinstance(block, _While):
                while True:
                    for ins in block.cond_instrs:
                        self._exec_instr(ins, frame)
                    if not frame.get(block.cond, "while_check"):
                        break
                    self._exec_blocks(block.body, frame)
            elif isinstance(block, _For):
                coll = frame.get(block.coll, "for_begin")
                try:
                    iterator = iter(coll)
                except TypeError:
                    raise IrExecError(code="E0704",
                                      message=f"IR executor: value of type "
                                              f"{type(coll).__name__} is not iterable",
                                      stage="ir-exec")
                for item in iterator:
                    frame.set_temp(block.item, item)
                    self._exec_blocks(block.body, frame)
            elif isinstance(block, _NoGrad):
                with no_grad():
                    self._exec_blocks(block.body, frame)
            else:  # pragma: no cover - internal
                raise AssertionError(f"unknown block {block!r}")

    # -- straight-line instructions --

    def _exec_instr(self, ins: TirInstr, frame: Frame):
        op = ins.op

        def val(arg: str):
            return frame.get(arg, op)

        if op == "const_int":
            frame.set_temp(ins.result, int(ins.args[0]))
            return
        if op == "const_float":
            frame.set_temp(ins.result, float(ins.args[0]))
            return
        if op == "const_bool":
            frame.set_temp(ins.result, ins.args[0] == "true")
            return
        if op == "const_str":
            frame.set_temp(ins.result, json.loads(ins.args[0]))
            return
        if op == "copy":
            frame.set_temp(ins.result, val(ins.args[0]))
            return
        if op == "load":
            frame.set_temp(ins.result, frame.get(ins.args[0], op))
            return
        if op == "var_def":
            frame.define_var(ins.result, val(ins.args[0]))
            return
        if op == "store":
            frame.store_var(ins.result, val(ins.args[0]))
            return
        if op == "make_tensor":
            frame.set_temp(ins.result, T.tensor(self._tensor_args(ins.args, frame)))
            return
        if op in _BINOPS:
            frame.set_temp(ins.result, _BINOPS[op](val(ins.args[0]), val(ins.args[1])))
            return
        if op == "unary_-":
            frame.set_temp(ins.result, -val(ins.args[0]))
            return
        if op == "unary_!":
            frame.set_temp(ins.result, not val(ins.args[0]))
            return
        if op == "index":
            frame.set_temp(ins.result, val(ins.args[0])[val(ins.args[1])])
            return
        if op.startswith("field:"):
            recv = val(ins.args[0])
            name = op[len("field:"):]
            if not hasattr(recv, name):
                raise IrExecError(code="E0705",
                                  message=f"IR executor: no field '{name}' on value of type "
                                          f"{type(recv).__name__}",
                                  stage="ir-exec")
            frame.set_temp(ins.result, getattr(recv, name))
            return
        if op.startswith("method:"):
            name = op[len("method:"):]
            recv = val(ins.args[0])
            kw_names = ins.attrs.get("kwargs", [])
            n_pos = len(ins.args) - 1 - len(kw_names)
            args = [val(a) for a in ins.args[1:1 + n_pos]]
            kwargs = {k: val(a) for k, a in zip(kw_names, ins.args[1 + n_pos:])}
            method = getattr(recv, name, None)
            if method is None:
                raise IrExecError(code="E0706",
                                  message=f"IR executor: no method '{name}' on value of type "
                                          f"{type(recv).__name__}",
                                  stage="ir-exec")
            frame.set_temp(ins.result, method(*args, **kwargs))
            return
        if op.startswith("call:"):
            kw_names = ins.attrs.get("kwargs", [])
            n_pos = len(ins.args) - len(kw_names)
            args = [val(a) for a in ins.args[:n_pos]]
            kwargs = {k: val(a) for k, a in zip(kw_names, ins.args[n_pos:])}
            frame.set_temp(ins.result,
                           self._call_static(op[len("call:"):], args, kwargs, frame))
            return
        if op == "call_dyn":
            kw_names = ins.attrs.get("kwargs", [])
            n_pos = len(ins.args) - 1 - len(kw_names)
            callee = val(ins.args[0])
            args = [val(a) for a in ins.args[1:1 + n_pos]]
            kwargs = {k: val(a) for k, a in zip(kw_names, ins.args[1 + n_pos:])}
            if not callable(callee):
                raise IrExecError(
                    code="E0707",
                    message=f"IR executor: dynamic call target of type {type(callee).__name__} "
                            f"is not callable (higher-order Time-T functions are not "
                            f"supported in IR yet; DD-9)",
                    stage="ir-exec")
            frame.set_temp(ins.result, callee(*args, **kwargs))
            return
        if op == "return":
            raise _Return(val(ins.args[0]) if ins.args else None)
        if op.startswith("unsupported"):
            raise IrExecError(
                code="E0708",
                message=f"IR executor: '{ins.attrs.get('kind', op)}' cannot be executed from IR "
                        f"(not lowered yet; run without --via-ir to use the AST interpreter)",
                stage="ir-exec",
            )
        raise IrExecError(code="E0709", message=f"IR executor: unknown op '{op}'",
                          stage="ir-exec")

    def _fail_load(self, name):
        raise IrExecError(code="E0700", message=f"IR executor: undefined name '{name}'",
                          stage="ir-exec")

    def _tensor_args(self, args, frame: Frame):
        out = []
        for a in args:
            v = frame.get(a, "make_tensor")
            if isinstance(v, T.Tensor):
                v = v.data.tolist()
            out.append(v)
        return out


# Builtins shared in SPIRIT with the AST interpreter. print goes through the
# injected stdout writer, everything else is a direct re-export of the same
# timet.tensor entry points. tests/test_ir_exec_diff.py proves equivalence.
def _print(stdout_write, *args):
    stdout_write(" ".join(_stringify(a) for a in args))
    return None


def _stringify(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, T.Tensor):
        return repr(value)
    return str(value)


def _assert(cond, msg="assertion failed"):
    if not cond:
        raise IrExecError(code="E0510", message=str(msg), stage="ir-exec")
    return None


_BUILTINS = {
    "print": _print,
    "len": len,
    "assert": _assert,
    "tensor": T.tensor,
    "zeros": T.zeros,
    "ones": T.ones,
    "sum": T.sum,
    "mean": T.mean,
    "matmul": T.matmul,
    "relu": T.relu,
    "sigmoid": T.sigmoid,
    "tanh": T.tanh,
    "softmax": T.softmax,
    "log_softmax": T.log_softmax,
    "argmax": T.argmax,
    "one_hot": T.one_hot,
    "exp": T.exp,
    "log": T.log,
    "sqrt": T.sqrt,
    "max": lambda t, **kw: t.max(**kw),
    "min": lambda t, **kw: t.min(**kw),
}


def run_program(program: TirProgram, stdout_write=None) -> IRExecutor:
    ex = IRExecutor(program, stdout_write=stdout_write)
    ex.run()
    return ex
