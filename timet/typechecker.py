"""Time-T type checker.

Performs name resolution + type inference/checking in a single pass over the
AST, decorating each expression node's `.ty` attribute. Module imports are
checked ACROSS file boundaries with real exported types (DD-13, v0.3.0);
see docs/DESIGN_DECISIONS.md DD-6/DD-13 for what remains intentionally out
of scope (generics, traits, structs, enums, pattern matching, higher-order
functions — anything that would turn `<unknown>` into aspirational typing).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import timet.ast_nodes as A
from timet.diagnostics import Diagnostic, SourceSpan
from timet.modules import ModuleLoader, ModuleError
from timet.types import (
    Type, TInt, TFloat, TBool, TString, TUnit, TTensor, TFunction, TUnknown,
    TModule, PRIMITIVE_NAMES, is_numeric,
)


class TypeError_(Diagnostic):
    pass


class Binding:
    def __init__(self, name: str, ty: Type, mutable: bool):
        self.name = name
        self.ty = ty
        self.mutable = mutable


class Scope:
    def __init__(self, parent: Optional["Scope"] = None):
        self.parent = parent
        self.vars: Dict[str, Binding] = {}

    def define(self, name: str, ty: Type, mutable: bool):
        self.vars[name] = Binding(name, ty, mutable)

    def lookup(self, name: str) -> Optional[Binding]:
        scope = self
        while scope is not None:
            if name in scope.vars:
                return scope.vars[name]
            scope = scope.parent
        return None

    def child(self) -> "Scope":
        return Scope(self)


TENSOR_METHODS = {
    "sum": 0, "mean": 0, "max": 0, "min": 0, "exp": 0, "log": 0, "sqrt": 0,
    "relu": 0, "sigmoid": 0, "tanh": 0, "softmax": 0, "transpose": 0,
    "reshape": 1, "matmul": 1, "backward": 0, "detach": 0, "permute": 1,
    "item": 0,
}

BUILTIN_FUNCS = {
    "print", "len", "tensor", "zeros", "ones", "assert",
    "sum", "mean", "max", "min", "exp", "log", "sqrt", "relu", "sigmoid",
    "tanh", "softmax", "log_softmax", "argmax", "one_hot", "matmul",
}

#: Names pre-bound to host modules by the engine (interpreter and IR
#: executor). Members are accessed dynamically, so anything reached through
#: them is currently typed TUnknown (documented in docs/LANGUAGE.md DD-10).
BUILTIN_MODULES = {"nn", "optim", "train"}


def _resolve_type_expr(te: A.TypeExpr) -> Type:
    if te.name in PRIMITIVE_NAMES:
        return PRIMITIVE_NAMES[te.name]
    if te.name == "Tensor":
        dtype = "f32"
        if te.args:
            dtype = te.args[0].name
        return TTensor(dtype)
    if te.name == "Function":
        *param_tes, ret_te = te.args
        return TFunction(tuple(_resolve_type_expr(p) for p in param_tes), _resolve_type_expr(ret_te))
    raise TypeError_(
        code="E0200",
        message=f"unknown type '{te.name}'",
        span=SourceSpan(te.line, te.col),
        stage="typechecker",
        note="known types: Int, Float, Bool, String, Unit, Tensor[dtype]",
    )


class TypeChecker:
    def __init__(self, loader: Optional[ModuleLoader] = None,
                 importer_path: str = "<input>"):
        self.global_scope = Scope()
        self.errors: List[TypeError_] = []
        self._loop_depth = 0
        self.loader = loader or ModuleLoader()
        self.importer_path = importer_path

    def check_program(self, program: A.Program) -> A.Program:
        scope = self.global_scope
        # First pass: resolve imports so later statements can use module
        # bindings, then hoist function signatures so mutual calls work.
        for stmt in program.statements:
            if isinstance(stmt, A.ImportStmt):
                self._check_import(stmt, scope)
        for stmt in program.statements:
            if isinstance(stmt, A.FnDecl):
                self._hoist_fn(stmt, scope)
        for stmt in program.statements:
            if not isinstance(stmt, A.ImportStmt):
                self.check_stmt(stmt, scope)
        return program

    def _check_import(self, stmt: A.ImportStmt, scope: Scope):
        span = SourceSpan(stmt.line, stmt.col)
        mod = self.loader.load(stmt.path, self.importer_path, span)
        exports = self._module_type(mod, span)
        scope.define(stmt.binding, exports, mutable=False)

    def _module_type(self, mod, span: SourceSpan) -> TModule:
        """Type-check a module (once) and return its typed export table."""
        if mod.checked_types:
            return self._tmodule_of(mod)
        if mod.checking_types:
            raise ModuleError(
                code="E0210",
                message=f"import cycle involving module '{mod.dotted}'",
                span=span,
                stage="typechecker",
                note="restructure the modules so the dependency graph is acyclic",
            )
        mod.checking_types = True
        try:
            sub = TypeChecker(loader=self.loader, importer_path=str(mod.path))
            sub.check_program(mod.program)
            exports: Dict[str, Type] = {}
            for stmt in mod.program.statements:
                if isinstance(stmt, A.FnDecl):
                    binding = sub.global_scope.lookup(stmt.name)
                    if binding is not None:
                        exports[stmt.name] = binding.ty
                elif isinstance(stmt, (A.LetStmt, A.ImportStmt)):
                    name = stmt.name if isinstance(stmt, A.LetStmt) else stmt.binding
                    binding = sub.global_scope.lookup(name)
                    if binding is not None:
                        exports[name] = binding.ty
            mod.type_exports = exports
            mod.checked_types = True
        finally:
            mod.checking_types = False
        return self._tmodule_of(mod)

    @staticmethod
    def _tmodule_of(mod) -> TModule:
        return TModule(mod.dotted, tuple(sorted(mod.type_exports.items())))

    def _hoist_fn(self, fn: A.FnDecl, scope: Scope):
        params = tuple(
            _resolve_type_expr(p.type_expr) if p.type_expr else TUnknown()
            for p in fn.params
        )
        ret = _resolve_type_expr(fn.ret_type) if fn.ret_type else TUnit()
        scope.define(fn.name, TFunction(params, ret), mutable=False)

    def check_stmt(self, stmt: A.Stmt, scope: Scope):
        if isinstance(stmt, A.ImportStmt):
            raise TypeError_(
                code="E0212",
                message="'import' is only allowed at the top level of a file",
                span=SourceSpan(stmt.line, stmt.col),
                stage="typechecker",
                note="move the import to the top of the file (DD-13)",
            )
        if isinstance(stmt, A.LetStmt):
            self._check_let(stmt, scope)
        elif isinstance(stmt, A.AssignStmt):
            self._check_assign(stmt, scope)
        elif isinstance(stmt, A.ExprStmt):
            self.infer(stmt.expr, scope)
        elif isinstance(stmt, A.ReturnStmt):
            if stmt.value is not None:
                self.infer(stmt.value, scope)
        elif isinstance(stmt, A.IfStmt):
            cond_ty = self.infer(stmt.cond, scope)
            self._require(cond_ty, TBool(), stmt.cond, "if condition")
            self.check_block(stmt.then_branch, scope.child())
            if stmt.else_branch is not None:
                self.check_block(stmt.else_branch, scope.child())
        elif isinstance(stmt, (A.BreakStmt, A.ContinueStmt)):
            if self._loop_depth == 0:
                keyword = "Break" if isinstance(stmt, A.BreakStmt) else "Continue"
                raise TypeError_(
                    code="E0215",
                    message=f"'{keyword.lower()}' used outside a loop",
                    span=SourceSpan(stmt.line, stmt.col),
                    note="break/continue must be inside a while/for body (or a "
                         "function called from one)")
        elif isinstance(stmt, A.WhileStmt):
            cond_ty = self.infer(stmt.cond, scope)
            self._require(cond_ty, TBool(), stmt.cond, "while condition")
            self._loop_depth += 1
            try:
                self.check_block(stmt.body, scope.child())
            finally:
                self._loop_depth -= 1
        elif isinstance(stmt, A.ForStmt):
            it_ty = self.infer(stmt.iterable, scope)
            # Iterating a Tensor yields (possibly 0-d) Tensor slices of the
            # same dtype -- mirror the runtime (Tensor.__getitem__).
            el_ty = TTensor(it_ty.dtype) if isinstance(it_ty, TTensor) else TUnknown()
            inner = scope.child()
            inner.define(stmt.var_name, el_ty, mutable=False)
            self._loop_depth += 1
            try:
                self.check_block(stmt.body, inner)
            finally:
                self._loop_depth -= 1
        elif isinstance(stmt, A.NoGradStmt):
            self.check_block(stmt.body, scope.child())
        elif isinstance(stmt, A.FnDecl):
            self._check_fn_body(stmt, scope)
        else:
            raise TypeError_(
                code="E0299",
                message=f"internal: unhandled statement kind {type(stmt).__name__}",
                span=SourceSpan(stmt.line, stmt.col),
                stage="typechecker",
            )

    def check_block(self, block: A.Block, scope: Scope):
        for s in block.statements:
            if isinstance(s, A.FnDecl):
                self._hoist_fn(s, scope)
        for s in block.statements:
            self.check_stmt(s, scope)

    def _check_let(self, stmt: A.LetStmt, scope: Scope):
        value_ty = self.infer(stmt.value, scope)
        declared = _resolve_type_expr(stmt.type_expr) if stmt.type_expr else None
        if declared is not None:
            self._require(value_ty, declared, stmt.value, f"'let {stmt.name}' initializer")
            final_ty = declared
        else:
            final_ty = value_ty
        scope.define(stmt.name, final_ty, mutable=stmt.mutable)

    def _check_assign(self, stmt: A.AssignStmt, scope: Scope):
        if not isinstance(stmt.target, A.Ident):
            self.infer(stmt.target, scope)
            self.infer(stmt.value, scope)
            return
        binding = scope.lookup(stmt.target.name)
        if binding is None:
            raise TypeError_(
                code="E0201",
                message=f"assignment to undefined variable '{stmt.target.name}'",
                span=SourceSpan(stmt.line, stmt.col),
                stage="typechecker",
            )
        if not binding.mutable:
            raise TypeError_(
                code="E0301",
                message=f"cannot assign to immutable binding '{stmt.target.name}'",
                span=SourceSpan(stmt.line, stmt.col),
                stage="typechecker",
                note="declare it with 'var' instead of 'let' if it needs to change",
            )
        value_ty = self.infer(stmt.value, scope)
        self._require(value_ty, binding.ty, stmt.value, f"assignment to '{stmt.target.name}'")

    def _check_fn_body(self, fn: A.FnDecl, scope: Scope):
        inner = scope.child()
        for p in fn.params:
            pty = _resolve_type_expr(p.type_expr) if p.type_expr else TUnknown()
            inner.define(p.name, pty, mutable=False)
        self.check_block(fn.body, inner)

    def _require(self, actual: Type, expected: Type, node: A.Node, where: str):
        if isinstance(expected, TUnknown) or isinstance(actual, TUnknown):
            return
        if isinstance(expected, TFloat) and isinstance(actual, TInt):
            return  # int literals widen to float
        if actual != expected:
            raise TypeError_(
                code="E0202",
                message=f"type mismatch in {where}",
                span=SourceSpan(node.line, node.col),
                expected=str(expected),
                actual=str(actual),
                stage="typechecker",
            )

    # ---------------- expressions ----------------

    def infer(self, expr: A.Expr, scope: Scope) -> Type:
        ty = self._infer_inner(expr, scope)
        expr.ty = ty
        return ty

    def _infer_inner(self, expr: A.Expr, scope: Scope) -> Type:
        if isinstance(expr, A.IntLit):
            return TInt()
        if isinstance(expr, A.FloatLit):
            return TFloat()
        if isinstance(expr, A.BoolLit):
            return TBool()
        if isinstance(expr, A.StringLit):
            return TString()
        if isinstance(expr, A.UnitLit):
            return TUnit()
        if isinstance(expr, A.Ident):
            if expr.name in BUILTIN_MODULES:
                return TUnknown()  # host module value; members typed dynamically
            binding = scope.lookup(expr.name)
            if binding is None:
                raise TypeError_(
                    code="E0203",
                    message=f"undefined name '{expr.name}'",
                    span=SourceSpan(expr.line, expr.col),
                    stage="typechecker",
                    note="check for typos or missing 'let'/'fn' declaration",
                )
            return binding.ty
        if isinstance(expr, A.ListExpr):
            for el in expr.elements:
                self.infer(el, scope)
            return TTensor("f32")
        if isinstance(expr, A.UnaryOp):
            operand_ty = self.infer(expr.operand, scope)
            if expr.op == "!":
                self._require(operand_ty, TBool(), expr.operand, "'!' operand")
                return TBool()
            return operand_ty
        if isinstance(expr, A.BinOp):
            return self._infer_binop(expr, scope)
        if isinstance(expr, A.Call):
            return self._infer_call(expr, scope)
        if isinstance(expr, A.MethodCall):
            return self._infer_method_call(expr, scope)
        if isinstance(expr, A.FieldAccess):
            recv_ty = self.infer(expr.receiver, scope)
            if isinstance(recv_ty, TModule):
                member = recv_ty.member(expr.field)
                if member is None:
                    raise TypeError_(
                        code="E0211",
                        message=f"module '{recv_ty.dotted}' has no export '{expr.field}'",
                        span=SourceSpan(expr.line, expr.col),
                        stage="typechecker",
                        note="exports: " + (", ".join(k for k, _ in recv_ty.exports) or "(none)"),
                    )
                return member
            if expr.field == "grad" and isinstance(recv_ty, TTensor):
                return TTensor(recv_ty.dtype)
            return TUnknown()
        if isinstance(expr, A.Index):
            self.infer(expr.receiver, scope)
            self.infer(expr.index, scope)
            return TTensor("f32")
        if isinstance(expr, A.IfExpr):
            self.infer(expr.cond, scope)
            then_scope = scope.child()
            self.check_block(expr.then_branch, then_scope)
            else_scope = scope.child()
            self.check_block(expr.else_branch, else_scope)
            then_ty = self._last_expr_type(expr.then_branch)
            return then_ty if then_ty is not None else TUnit()
        if isinstance(expr, A.Lambda):
            inner = scope.child()
            for p in expr.params:
                pty = _resolve_type_expr(p.type_expr) if p.type_expr else TUnknown()
                inner.define(p.name, pty, mutable=False)
            self.check_block(expr.body, inner)
            params = tuple(
                _resolve_type_expr(p.type_expr) if p.type_expr else TUnknown()
                for p in expr.params
            )
            ret = _resolve_type_expr(expr.ret_type) if expr.ret_type else TUnknown()
            return TFunction(params, ret)
        raise TypeError_(
            code="E0298",
            message=f"internal: unhandled expression kind {type(expr).__name__}",
            span=SourceSpan(expr.line, expr.col),
            stage="typechecker",
        )

    def _last_expr_type(self, block: A.Block) -> Optional[Type]:
        if not block.statements:
            return None
        last = block.statements[-1]
        if isinstance(last, A.ExprStmt):
            return last.expr.ty
        return None

    def _infer_binop(self, expr: A.BinOp, scope: Scope) -> Type:
        lt = self.infer(expr.left, scope)
        rt = self.infer(expr.right, scope)
        op = expr.op
        if op in ("&&", "||"):
            self._require(lt, TBool(), expr.left, f"'{op}' left operand")
            self._require(rt, TBool(), expr.right, f"'{op}' right operand")
            return TBool()
        if op in ("==", "!="):
            return TBool()
        if op in ("<", "<=", ">", ">="):
            return TBool()
        if op == "@":
            return TTensor("f32")
        if op in ("+", "-", "*", "/", "%"):
            if isinstance(lt, TTensor) or isinstance(rt, TTensor):
                dtype = lt.dtype if isinstance(lt, TTensor) else (rt.dtype if isinstance(rt, TTensor) else "f32")
                return TTensor(dtype)
            if isinstance(lt, TString) and isinstance(rt, TString) and op == "+":
                return TString()
            if not is_numeric(lt) or not is_numeric(rt):
                raise TypeError_(
                    code="E0204",
                    message=f"operator '{op}' requires numeric operands",
                    span=SourceSpan(expr.line, expr.col),
                    expected="Int, Float, or Tensor",
                    actual=f"{lt} {op} {rt}",
                    stage="typechecker",
                )
            if op == "/":
                # The interpreter performs TRUE division (7 / 2 == 3.5) even
                # on ints -- matching that at the type level is required for
                # sound codegen; previously this branch returned TInt()
                # for int/int, which was a genuine soundness bug found by
                # the native backend's differential tests (v0.4.0).
                return TFloat()
            if isinstance(lt, TFloat) or isinstance(rt, TFloat):
                return TFloat()
            return TInt()
        raise TypeError_(
            code="E0299",
            message=f"internal: unknown operator '{op}'",
            span=SourceSpan(expr.line, expr.col),
            stage="typechecker",
        )

    def _infer_call(self, expr: A.Call, scope: Scope) -> Type:
        for a in expr.args:
            self.infer(a, scope)
        for v in expr.kwargs.values():
            self.infer(v, scope)
        if isinstance(expr.callee, A.Ident):
            name = expr.callee.name
            if name in BUILTIN_FUNCS:
                return self._builtin_return_type(name)
            binding = scope.lookup(name)
            if binding is None:
                raise TypeError_(
                    code="E0205",
                    message=f"call to undefined function '{name}'",
                    span=SourceSpan(expr.line, expr.col),
                    stage="typechecker",
                )
            if isinstance(binding.ty, TFunction):
                return binding.ty.ret
            if isinstance(binding.ty, TUnknown):
                return TUnknown()
            raise TypeError_(
                code="E0206",
                message=f"'{name}' is not callable",
                span=SourceSpan(expr.line, expr.col),
                actual=str(binding.ty),
                stage="typechecker",
            )
        callee_ty = self.infer(expr.callee, scope)
        if isinstance(callee_ty, TFunction):
            return callee_ty.ret
        return TUnknown()

    def _infer_method_call(self, expr: A.MethodCall, scope: Scope) -> Type:
        recv_ty = self.infer(expr.receiver, scope)
        for a in expr.args:
            self.infer(a, scope)
        if isinstance(recv_ty, TModule):
            # module.fn(...) — statically typed across the module boundary
            member = recv_ty.member(expr.method)
            if member is None:
                raise TypeError_(
                    code="E0211",
                    message=f"module '{recv_ty.dotted}' has no export '{expr.method}'",
                    span=SourceSpan(expr.line, expr.col),
                    stage="typechecker",
                    note="exports: " + (", ".join(k for k, _ in recv_ty.exports) or "(none)"),
                )
            if isinstance(member, TFunction):
                return member.ret
            return member
        if expr.method == "backward":
            return TUnit()
        if expr.method == "item":
            # scalar extraction: dtype f32/f64 -> Float, i32/i64 -> Int
            if isinstance(recv_ty, TTensor) and recv_ty.dtype.startswith("i"):
                return TInt()
            return TFloat()
        if expr.method in ("sum", "mean", "max", "min") and not expr.args:
            return TTensor(recv_ty.dtype if isinstance(recv_ty, TTensor) else "f32")
        if expr.method in TENSOR_METHODS:
            return TTensor(recv_ty.dtype if isinstance(recv_ty, TTensor) else "f32")
        return TUnknown()

    def _builtin_return_type(self, name: str) -> Type:
        if name in ("tensor", "zeros", "ones", "sum", "mean", "max", "min",
                    "exp", "log", "sqrt", "relu", "sigmoid", "tanh", "softmax",
                    "log_softmax", "one_hot", "matmul"):
            return TTensor("f32")
        if name == "argmax":
            return TTensor("i64")
        if name == "len":
            return TInt()
        if name in ("print", "assert"):
            return TUnit()
        return TUnknown()


def check(program: A.Program, filename: str = "<input>",
          loader: Optional[ModuleLoader] = None) -> A.Program:
    return TypeChecker(loader=loader, importer_path=filename).check_program(program)
