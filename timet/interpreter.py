"""Tree-walking interpreter over the typed AST.

See docs/DESIGN_DECISIONS.md DD-3 for why this executes the AST rather than
the IR today.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import timet.ast_nodes as A
from timet import nn as nn_lib
from timet import optim as optim_lib
from timet import tensor as T
from timet import train as train_lib
from timet.autodiff import no_grad
from timet.diagnostics import Diagnostic, SourceSpan
from timet.modules import ModuleLoader, ModuleError, ModuleValue

#: Module objects pre-bound in every program's global scope (DD-10). They
#: are ordinary values, so `nn.Linear(2, 8)` is a field access + call.
ENGINE_GLOBALS = {"nn": nn_lib, "optim": optim_lib, "train": train_lib}


class RuntimeErr(Diagnostic):
    pass


class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


class Function:
    def __init__(self, decl, closure: "Environment"):
        self.decl = decl
        self.closure = closure

    @property
    def name(self):
        return getattr(self.decl, "name", "<lambda>")


class Environment:
    def __init__(self, parent: Optional["Environment"] = None):
        self.parent = parent
        self.vars: Dict[str, Any] = {}

    def define(self, name: str, value: Any):
        self.vars[name] = value

    def get(self, name: str, span: SourceSpan):
        env = self
        while env is not None:
            if name in env.vars:
                return env.vars[name]
            env = env.parent
        raise RuntimeErr(
            code="E0500",
            message=f"undefined name '{name}' at runtime",
            span=span,
            stage="interpreter",
        )

    def set(self, name: str, value: Any, span: SourceSpan):
        env = self
        while env is not None:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.parent
        raise RuntimeErr(
            code="E0501",
            message=f"undefined name '{name}' at runtime",
            span=span,
            stage="interpreter",
        )

    def child(self) -> "Environment":
        return Environment(self)


class Interpreter:
    def __init__(self, stdout_write=None, loader: Optional[ModuleLoader] = None,
                 importer_path: str = "<input>"):
        self.globals = Environment()
        for name, value in ENGINE_GLOBALS.items():
            self.globals.define(name, value)
        self._stdout_write = stdout_write or (lambda s: print(s))
        self._loader = loader or ModuleLoader()
        self._importer_path = importer_path

    def run(self, program: A.Program):
        for stmt in program.statements:
            self.exec_stmt(stmt, self.globals)
        main = self.globals.vars.get("main")
        if isinstance(main, Function):
            self.call_function(main, [], {}, SourceSpan(0, 0))

    # ---------------- statements ----------------

    def exec_block(self, block: A.Block, env: Environment):
        result = None
        for stmt in block.statements:
            result = self.exec_stmt(stmt, env)
        return result

    def exec_stmt(self, stmt: A.Stmt, env: Environment):
        if isinstance(stmt, A.ImportStmt):
            self._exec_import(stmt, env)
            return None
        if isinstance(stmt, A.LetStmt):
            value = self.eval(stmt.value, env)
            env.define(stmt.name, value)
            return None
        if isinstance(stmt, A.AssignStmt):
            value = self.eval(stmt.value, env)
            if isinstance(stmt.target, A.Ident):
                env.set(stmt.target.name, value, SourceSpan(stmt.line, stmt.col))
            else:
                raise RuntimeErr(
                    code="E0599",
                    message="unsupported assignment target",
                    span=SourceSpan(stmt.line, stmt.col),
                    stage="interpreter",
                )
            return None
        if isinstance(stmt, A.ExprStmt):
            return self.eval(stmt.expr, env)
        if isinstance(stmt, A.ReturnStmt):
            value = self.eval(stmt.value, env) if stmt.value is not None else None
            raise ReturnSignal(value)
        if isinstance(stmt, A.IfStmt):
            cond = self.eval(stmt.cond, env)
            if cond:
                self.exec_block(stmt.then_branch, env.child())
            elif stmt.else_branch is not None:
                self.exec_block(stmt.else_branch, env.child())
            return None
        if isinstance(stmt, A.WhileStmt):
            while self.eval(stmt.cond, env):
                self.exec_block(stmt.body, env.child())
            return None
        if isinstance(stmt, A.ForStmt):
            iterable = self.eval(stmt.iterable, env)
            for item in iterable:
                inner = env.child()
                inner.define(stmt.var_name, item)
                self.exec_block(stmt.body, inner)
            return None
        if isinstance(stmt, A.NoGradStmt):
            with no_grad():
                self.exec_block(stmt.body, env.child())
            return None
        if isinstance(stmt, A.FnDecl):
            env.define(stmt.name, Function(stmt, env))
            return None
        raise RuntimeErr(
            code="E0598",
            message=f"internal: unhandled statement {type(stmt).__name__}",
            span=SourceSpan(stmt.line, stmt.col),
            stage="interpreter",
        )

    def _exec_import(self, stmt: A.ImportStmt, env: Environment):
        """Execute a module once per loader (Python-like), in a FRESH global
        scope (engine builtins only), then bind a ModuleValue. `fn main` in a
        module is not auto-invoked."""
        mod = self._loader.load(stmt.path, self._importer_path,
                                SourceSpan(stmt.line, stmt.col))
        if mod.value is None:
            if mod.initializing:
                raise ModuleError(
                    code="E0210",
                    message=f"import cycle involving module '{mod.dotted}'",
                    span=SourceSpan(stmt.line, stmt.col),
                    stage="modules",
                )
            mod.initializing = True
            try:
                mod_env = Environment()
                for name, value in ENGINE_GLOBALS.items():
                    mod_env.define(name, value)
                for mst in mod.program.statements:
                    self.exec_stmt(mst, mod_env)
                mod.value = ModuleValue(mod.dotted, mod_env.vars)
            finally:
                mod.initializing = False
        env.define(stmt.binding, mod.value)

    def call_function(self, fn: Function, args: List[Any], kwargs: Dict[str, Any], span: SourceSpan):
        inner = fn.closure.child()
        for i, p in enumerate(fn.decl.params):
            if i < len(args):
                inner.define(p.name, args[i])
            elif p.name in kwargs:
                inner.define(p.name, kwargs[p.name])
            else:
                raise RuntimeErr(
                    code="E0502",
                    message=f"missing argument '{p.name}' in call to '{fn.name}'",
                    span=span,
                    stage="interpreter",
                )
        try:
            self.exec_block(fn.decl.body, inner)
        except ReturnSignal as r:
            return r.value
        return None

    # ---------------- expressions ----------------

    def eval(self, expr: A.Expr, env: Environment):
        if isinstance(expr, A.IntLit):
            return expr.value
        if isinstance(expr, A.FloatLit):
            return expr.value
        if isinstance(expr, A.BoolLit):
            return expr.value
        if isinstance(expr, A.StringLit):
            return expr.value
        if isinstance(expr, A.UnitLit):
            return None
        if isinstance(expr, A.Ident):
            return env.get(expr.name, SourceSpan(expr.line, expr.col))
        if isinstance(expr, A.ListExpr):
            return T.tensor(self._eval_nested_list(expr, env))
        if isinstance(expr, A.UnaryOp):
            v = self.eval(expr.operand, env)
            if expr.op == "-":
                return -v
            if expr.op == "!":
                return not v
        if isinstance(expr, A.BinOp):
            return self._eval_binop(expr, env)
        if isinstance(expr, A.Call):
            return self._eval_call(expr, env)
        if isinstance(expr, A.MethodCall):
            return self._eval_method_call(expr, env)
        if isinstance(expr, A.FieldAccess):
            recv = self.eval(expr.receiver, env)
            if not hasattr(recv, expr.field):
                raise RuntimeErr(
                    code="E0506",
                    message=f"no field '{expr.field}' on value of type {type(recv).__name__}",
                    span=SourceSpan(expr.line, expr.col),
                    stage="interpreter",
                )
            return getattr(recv, expr.field)
        if isinstance(expr, A.Index):
            recv = self.eval(expr.receiver, env)
            idx = self.eval(expr.index, env)
            return recv[idx]
        if isinstance(expr, A.IfExpr):
            cond = self.eval(expr.cond, env)
            branch = expr.then_branch if cond else expr.else_branch
            return self.exec_block(branch, env.child())
        if isinstance(expr, A.Lambda):
            return Function(expr, env)
        raise RuntimeErr(
            code="E0597",
            message=f"internal: unhandled expression {type(expr).__name__}",
            span=SourceSpan(expr.line, expr.col),
            stage="interpreter",
        )

    def _eval_nested_list(self, expr: A.Expr, env: Environment):
        if isinstance(expr, A.ListExpr):
            return [self._eval_nested_list(e, env) for e in expr.elements]
        return self.eval(expr, env)

    def _eval_binop(self, expr: A.BinOp, env: Environment):
        op = expr.op
        if op == "&&":
            return self.eval(expr.left, env) and self.eval(expr.right, env)
        if op == "||":
            return self.eval(expr.left, env) or self.eval(expr.right, env)
        left = self.eval(expr.left, env)
        right = self.eval(expr.right, env)
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "%":
            return left % right
        if op == "@":
            return left @ right
        if op == "==":
            return left == right
        if op == "!=":
            return not (left == right)
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        raise RuntimeErr(
            code="E0596",
            message=f"internal: unhandled operator '{op}'",
            span=SourceSpan(expr.line, expr.col),
            stage="interpreter",
        )

    def _eval_call(self, expr: A.Call, env: Environment):
        args = [self.eval(a, env) for a in expr.args]
        kwargs = {k: self.eval(v, env) for k, v in expr.kwargs.items()}
        if isinstance(expr.callee, A.Ident):
            name = expr.callee.name
            builtin = self._builtin(name)
            if builtin is not None:
                return builtin(*args, **kwargs)
            fn = env.get(name, SourceSpan(expr.line, expr.col))
            if isinstance(fn, Function):
                return self.call_function(fn, args, kwargs, SourceSpan(expr.line, expr.col))
            # host callables: nn.Linear, optim.Adam, an nn.Module instance
            # (model(x)), etc. (DD-10)
            if callable(fn):
                return fn(*args, **kwargs)
            raise RuntimeErr(
                code="E0503",
                message=f"'{name}' is not callable",
                span=SourceSpan(expr.line, expr.col),
                stage="interpreter",
            )
        callee = self.eval(expr.callee, env)
        if isinstance(callee, Function):
            return self.call_function(callee, args, kwargs, SourceSpan(expr.line, expr.col))
        if callable(callee):
            return callee(*args, **kwargs)
        raise RuntimeErr(
            code="E0504",
            message="call target is not callable",
            span=SourceSpan(expr.line, expr.col),
            stage="interpreter",
        )

    def _eval_method_call(self, expr: A.MethodCall, env: Environment):
        recv = self.eval(expr.receiver, env)
        args = [self.eval(a, env) for a in expr.args]
        kwargs = {k: self.eval(v, env) for k, v in expr.kwargs.items()}
        # module.fn(...): module member may be a Time-T Function
        if isinstance(recv, ModuleValue):
            if not hasattr(recv, expr.method):
                raise RuntimeErr(
                    code="E0505",
                    message=f"module '{recv._dotted}' has no member '{expr.method}'",
                    span=SourceSpan(expr.line, expr.col),
                    stage="interpreter",
                )
            member = getattr(recv, expr.method)
            if isinstance(member, Function):
                return self.call_function(member, args, kwargs,
                                          SourceSpan(expr.line, expr.col))
            if callable(member):
                return member(*args, **kwargs)
            raise RuntimeErr(
                code="E0503",
                message=f"module member '{expr.method}' is not callable",
                span=SourceSpan(expr.line, expr.col),
                stage="interpreter",
            )
        method = getattr(recv, expr.method, None)
        if method is None:
            raise RuntimeErr(
                code="E0505",
                message=f"no method '{expr.method}' on value of type {type(recv).__name__}",
                span=SourceSpan(expr.line, expr.col),
                stage="interpreter",
            )
        return method(*args, **kwargs)

    def _builtin(self, name: str):
        def _print(*args):
            self._stdout_write(" ".join(_stringify(a) for a in args))
            return None

        def _len(x):
            return len(x)

        def _assert(cond, msg="assertion failed"):
            if not cond:
                raise RuntimeErr(code="E0510", message=str(msg), stage="interpreter")
            return None

        mapping = {
            "print": _print,
            "len": _len,
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
        return mapping.get(name)


def _stringify(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, T.Tensor):
        return repr(value)
    return str(value)


def run_source(source: str, filename: str = "<input>", stdout_write=None,
               loader: Optional[ModuleLoader] = None):
    from timet.parser import parse
    from timet.typechecker import check

    loader = loader or ModuleLoader()
    program = parse(source, filename)
    check(program, filename=filename, loader=loader)
    interp = Interpreter(stdout_write=stdout_write, loader=loader,
                         importer_path=filename)
    interp.run(program)
    return interp
