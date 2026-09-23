"""Time-T typed intermediate representation.

Flat, three-address-code style IR, lowered from the typed AST. Inspectable
via `time-t inspect --ir`, serializable to/from JSON, deterministic given
identical input, and -- since v0.2 -- directly executable by
`timet/ir_exec.py` and optimized by `timet/optimize.py`.

Control flow is represented with STRUCTURED markers (if_begin/else/if_end,
while_begin/while_check/while_end, for_begin/for_end,
nograd_begin/nograd_end), which the executor walks as a block tree. Mutable
`var` bindings lower to named storage (var_def / store / load) so loop-carried
values work; immutable `let` bindings stay in SSA-temp form.

Documented limits (executor raises IrExecError, never silently mis-runs):
lambdas, nested function declarations, and if-EXPRESSIONS are not lowered
(the AST interpreter remains the semantic reference there); `&&`/`||` are
eager in IR, not short-circuiting. See docs/DESIGN_DECISIONS.md DD-9.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

import timet.ast_nodes as A


@dataclass
class TirInstr:
    op: str
    args: List[str]
    result: Optional[str]
    ty: str
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"op": self.op, "args": self.args, "result": self.result,
                "ty": self.ty, "attrs": self.attrs}

    @staticmethod
    def from_json(d: dict) -> "TirInstr":
        return TirInstr(d["op"], d["args"], d["result"], d["ty"], d.get("attrs", {}))

    def render(self) -> str:
        args = ", ".join(self.args)
        attrs = f" {self.attrs}" if self.attrs else ""
        if self.result:
            return f"{self.result}: {self.ty} = {self.op}({args}){attrs}"
        return f"{self.op}({args}){attrs}"


@dataclass
class TirFunction:
    name: str
    params: List[str]
    param_types: List[str]
    ret_type: str
    instrs: List[TirInstr]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "params": self.params,
            "param_types": self.param_types,
            "ret_type": self.ret_type,
            "instrs": [i.to_json() for i in self.instrs],
        }

    @staticmethod
    def from_json(d: dict) -> "TirFunction":
        return TirFunction(
            d["name"], d["params"], d["param_types"], d["ret_type"],
            [TirInstr.from_json(i) for i in d["instrs"]],
        )

    def render(self) -> str:
        params = ", ".join(f"{n}: {t}" for n, t in zip(self.params, self.param_types))
        lines = [f"fn {self.name}({params}) -> {self.ret_type} {{"]
        for instr in self.instrs:
            lines.append(f"  {instr.render()}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class TirProgram:
    functions: List[TirFunction]

    def to_json(self) -> dict:
        return {"functions": [f.to_json() for f in self.functions]}

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), indent=2, sort_keys=False)

    @staticmethod
    def from_json(d: dict) -> "TirProgram":
        return TirProgram([TirFunction.from_json(f) for f in d["functions"]])

    @staticmethod
    def from_json_str(s: str) -> "TirProgram":
        return TirProgram.from_json(json.loads(s))

    def function_table(self) -> Dict[str, TirFunction]:
        return {f.name: f for f in self.functions}

    def render(self) -> str:
        return "\n\n".join(f.render() for f in self.functions)


#: Top-level statements are lowered into a synthetic function of this name.
MAIN_FN = "__main__"


class _Lowerer:
    def __init__(self, prefix: str = "", storage_prefix: str = "",
                 modules: Optional[Dict[str, str]] = None,
                 module_fn_names: Optional[Set[str]] = None,
                 module_top_names: Optional[Set[str]] = None):
        """prefix:         fn-name mangling prefix for static calls of
                           file-local functions ("mathlib." or "")
        storage_prefix: when compiling a module __init__ body, ALL top-level
                        bindings go to named storage under this prefix so
                        module functions (and importers) can find them
        modules:        bound import names -> dotted path, in this file
        module_fn_names: file-local fn names (for call mangling)
        module_top_names: file-local top-level let/var names (load mangling)
        """
        self.counter = 0
        self.instrs: List[TirInstr] = []
        self.prefix = prefix
        self.storage_prefix = storage_prefix
        self.modules = modules or {}
        self.module_fn_names = module_fn_names or set()
        self.module_top_names = module_top_names or set()
        # fn-local names that live in (possibly storage_prefix-ed) storage;
        # separate from module-level storage, so a local `var x` can shadow
        # a module-level `x` exactly like scoping rules say it should.
        self.local_vars: Set[str] = set()

    def fresh(self) -> str:
        name = f"%t{self.counter}"
        self.counter += 1
        return name

    def emit(self, op: str, args: List[str], ty: str, attrs: Optional[dict] = None,
             result: Optional[str] = None) -> str:
        if result is None:
            result = self.fresh()
        self.instrs.append(TirInstr(op, args, result, ty, attrs or {}))
        return result

    def lower_block(self, block: A.Block, env: Dict[str, str]) -> Optional[str]:
        last = None
        for stmt in block.statements:
            last = self.lower_stmt(stmt, env)
        return last

    def lower_stmt(self, stmt: A.Stmt, env: Dict[str, str]) -> Optional[str]:
        if isinstance(stmt, A.ImportStmt):
            # import execution == a call to the dependency's synthetic
            # __init__ function, placed at the import's source position.
            dotted = ".".join(stmt.path)
            self.instrs.append(TirInstr(f"call:__init__{dotted}", [], None, "Unit"))
            return None
        if isinstance(stmt, A.LetStmt):
            val = self.lower_expr(stmt.value, env)
            if self.storage_prefix:
                # module top level: every binding lives in prefixed named
                # storage so module fns and importers can reach it
                self.local_vars.add(stmt.name)
                env.pop(stmt.name, None)
                self.instrs.append(
                    TirInstr("var_def", [val], self.storage_prefix + stmt.name, "Unit"))
            elif stmt.mutable:
                self.local_vars.add(stmt.name)
                env.pop(stmt.name, None)
                self.instrs.append(TirInstr("var_def", [val], stmt.name, "Unit"))
            else:
                env[stmt.name] = val
            return None
        if isinstance(stmt, A.ExprStmt):
            return self.lower_expr(stmt.expr, env)
        if isinstance(stmt, A.ReturnStmt):
            val = self.lower_expr(stmt.value, env) if stmt.value is not None else None
            self.instrs.append(TirInstr("return", [val] if val else [], None, "Unit"))
            return None
        if isinstance(stmt, A.AssignStmt) and isinstance(stmt.target, A.Ident):
            val = self.lower_expr(stmt.value, env)
            name = stmt.target.name
            env.pop(name, None)
            if name not in self.local_vars and name in self.module_top_names:
                # assignment to a MODULE-level var from a module fn: route to
                # the module's prefixed storage (shared with importers)
                self.instrs.append(TirInstr("store", [val], self.prefix + name, "Unit"))
            else:
                self.local_vars.add(name)
                self.instrs.append(
                    TirInstr("store", [val], self.storage_prefix + name, "Unit"))
            return None
        if isinstance(stmt, A.BreakStmt):
            self.instrs.append(TirInstr("break", [], None, "Unit"))
            return None
        if isinstance(stmt, A.ContinueStmt):
            self.instrs.append(TirInstr("continue", [], None, "Unit"))
            return None
        if isinstance(stmt, A.IfStmt):
            cond = self.lower_expr(stmt.cond, env)
            self.instrs.append(TirInstr("if_begin", [cond], None, "Unit"))
            self.lower_block(stmt.then_branch, dict(env))
            if stmt.else_branch is not None:
                self.instrs.append(TirInstr("else", [], None, "Unit"))
                self.lower_block(stmt.else_branch, dict(env))
            self.instrs.append(TirInstr("if_end", [], None, "Unit"))
            return None
        if isinstance(stmt, A.WhileStmt):
            # The condition instructions live INSIDE the loop region so the
            # executor re-evaluates them every iteration (a flat
            # "evaluate-once" lowering would be an infinite loop).
            self.instrs.append(TirInstr("while_begin", [], None, "Unit"))
            cond = self.lower_expr(stmt.cond, env)
            self.instrs.append(TirInstr("while_check", [cond], None, "Unit"))
            self.lower_block(stmt.body, dict(env))
            self.instrs.append(TirInstr("while_end", [], None, "Unit"))
            return None
        if isinstance(stmt, A.ForStmt):
            coll = self.lower_expr(stmt.iterable, env)
            item = self.fresh()
            self.instrs.append(TirInstr("for_begin", [coll], item,
                                        "<item>", {"var": stmt.var_name}))
            body_env = dict(env)
            body_env[stmt.var_name] = item
            self.lower_block(stmt.body, body_env)
            self.instrs.append(TirInstr("for_end", [], None, "Unit"))
            return None
        if isinstance(stmt, A.NoGradStmt):
            self.instrs.append(TirInstr("nograd_begin", [], None, "Unit"))
            self.lower_block(stmt.body, dict(env))
            self.instrs.append(TirInstr("nograd_end", [], None, "Unit"))
            return None
        # Nested function declarations are NOT lowered into IR (documented
        # limit, DD-9): closures need captured environments the flat IR does
        # not model yet.
        self.instrs.append(TirInstr("unsupported_stmt", [], None, "Unit",
                                     {"kind": type(stmt).__name__}))
        return None

    def lower_expr(self, expr: A.Expr, env: Dict[str, str]) -> str:
        ty = str(expr.ty) if expr.ty is not None else "<unknown>"
        if isinstance(expr, A.IntLit):
            return self.emit("const_int", [str(expr.value)], ty)
        if isinstance(expr, A.FloatLit):
            return self.emit("const_float", [str(expr.value)], ty)
        if isinstance(expr, A.BoolLit):
            return self.emit("const_bool", ["true" if expr.value else "false"], ty)
        if isinstance(expr, A.StringLit):
            return self.emit("const_str", [json.dumps(expr.value)], ty)
        if isinstance(expr, A.Ident):
            if expr.name in env:
                return env[expr.name]
            if expr.name in self.local_vars:
                return self.emit("load", [self.storage_prefix + expr.name], ty)
            if expr.name in self.modules:
                # a module object used as a VALUE (not .member access) is
                # not representable in IR
                return self.emit("unsupported_expr", [], ty,
                                 {"kind": f"module-as-value '{expr.name}'"})
            if expr.name in self.module_top_names:
                # module-level let/var referenced from a module fn: read the
                # prefixed storage the __init__ body wrote
                return self.emit("load", [self.prefix + expr.name], ty)
            # Unresolved: either a function/builtin name used as a value, or
            # a name a caller frame will provide. Lower as a storage load;
            # the executor resolves it (or raises a clear error).
            return self.emit("load", [expr.name], ty)
        if isinstance(expr, A.ListExpr):
            elems = [self.lower_expr(e, env) for e in expr.elements]
            return self.emit("make_tensor", elems, ty)
        if isinstance(expr, A.UnaryOp):
            v = self.lower_expr(expr.operand, env)
            return self.emit(f"unary_{expr.op}", [v], ty)
        if isinstance(expr, A.BinOp):
            l = self.lower_expr(expr.left, env)
            r = self.lower_expr(expr.right, env)
            opname = {"+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod",
                      "@": "matmul", "==": "eq", "!=": "ne", "<": "lt", "<=": "le",
                      ">": "gt", ">=": "ge", "&&": "and", "||": "or"}[expr.op]
            return self.emit(opname, [l, r], ty)
        if isinstance(expr, A.Call):
            kw_names = list(expr.kwargs.keys())
            args = [self.lower_expr(a, env) for a in expr.args]
            args += [self.lower_expr(expr.kwargs[k], env) for k in kw_names]
            if isinstance(expr.callee, A.Ident):
                name = expr.callee.name
                if name in self.modules:
                    return self.emit("unsupported_expr", [], ty,
                                     {"kind": f"call of module '{name}'"})
                target = env.get(name)
                if name in self.module_fn_names and (target is None or target == name):
                    # file-local function: mangle with the module prefix
                    return self.emit(f"call:{self.prefix}{name}", args, ty,
                                     {"kwargs": kw_names})
                if target is None or target == name:
                    # static call: resolved against program fns / builtins
                    return self.emit(f"call:{name}", args, ty,
                                     {"kwargs": kw_names})
                return self.emit("call_dyn", [target] + args, ty,
                                 {"kwargs": kw_names})
            callee = self.lower_expr(expr.callee, env)
            return self.emit("call_dyn", [callee] + args, ty, {"kwargs": kw_names})
        if isinstance(expr, A.MethodCall):
            kw_names = list(expr.kwargs.keys())
            if isinstance(expr.receiver, A.Ident) and expr.receiver.name in self.modules:
                # module.fn(args) — a fully static qualified call
                dotted = self.modules[expr.receiver.name]
                args = [self.lower_expr(a, env) for a in expr.args]
                args += [self.lower_expr(expr.kwargs[k], env) for k in kw_names]
                return self.emit(f"call:{dotted}.{expr.method}", args, ty,
                                 {"kwargs": kw_names})
            recv = self.lower_expr(expr.receiver, env)
            args = [self.lower_expr(a, env) for a in expr.args]
            args += [self.lower_expr(expr.kwargs[k], env) for k in kw_names]
            return self.emit(f"method:{expr.method}", [recv] + args, ty,
                             {"kwargs": kw_names})
        if isinstance(expr, A.FieldAccess):
            if isinstance(expr.receiver, A.Ident) and expr.receiver.name in self.modules:
                dotted = self.modules[expr.receiver.name]
                return self.emit("load", [f"{dotted}.{expr.field}"], ty)
            recv = self.lower_expr(expr.receiver, env)
            return self.emit(f"field:{expr.field}", [recv], ty)
        if isinstance(expr, A.Index):
            recv = self.lower_expr(expr.receiver, env)
            idx = self.lower_expr(expr.index, env)
            return self.emit("index", [recv, idx], ty)
        if isinstance(expr, A.IfExpr):
            # If-EXPRESSIONS are deliberately not lowered (DD-9): both branch
            # bodies would have to be emitted inline and eagerly, diverging
            # from AST semantics. The executor raises a clear error here.
            return self.emit("unsupported_expr", [], ty, {"kind": "IfExpr"})
        return self.emit("unsupported_expr", [], ty, {"kind": type(expr).__name__})


def _fn_names(program: A.Program) -> Set[str]:
    return {s.name for s in program.statements if isinstance(s, A.FnDecl)}


def _top_value_names(program: A.Program) -> Set[str]:
    return {s.name for s in program.statements if isinstance(s, A.LetStmt)}


def _imports_map(program: A.Program) -> Dict[str, str]:
    return {s.binding: ".".join(s.path) for s in program.statements
            if isinstance(s, A.ImportStmt)}


def lower_function(fn: A.FnDecl, prefix: str = "",
                   modules: Optional[Dict[str, str]] = None,
                   module_fn_names: Optional[Set[str]] = None,
                   module_top_names: Optional[Set[str]] = None,
                   name: Optional[str] = None) -> TirFunction:
    lowerer = _Lowerer(prefix=prefix, modules=modules,
                       module_fn_names=module_fn_names,
                       module_top_names=module_top_names)
    env: Dict[str, str] = {}
    for p in fn.params:
        env[p.name] = p.name
    lowerer.lower_block(fn.body, env)
    ret_ty = str(fn.ret_type.name) if fn.ret_type else "Unit"
    param_types = [str(p.type_expr.name) if p.type_expr else "<unknown>" for p in fn.params]
    return TirFunction(name or fn.name, [p.name for p in fn.params], param_types,
                       ret_ty, lowerer.instrs)


def _lower_module(mod) -> List[TirFunction]:
    """Lower one module: its functions are prefixed with its dotted name and
    its top-level statements become `__init__<dotted>` with every binding in
    prefixed named storage (DD-13)."""
    dotted = mod.dotted
    prefix = f"{dotted}."
    imports = _imports_map(mod.program)
    fn_names = _fn_names(mod.program)
    top_names = _top_value_names(mod.program)
    fns: List[TirFunction] = []
    init_stmts = []
    for stmt in mod.program.statements:
        if isinstance(stmt, A.FnDecl):
            fns.append(lower_function(stmt, prefix=prefix, modules=imports,
                                      module_fn_names=fn_names,
                                      module_top_names=top_names,
                                      name=f"{prefix}{stmt.name}"))
        else:
            init_stmts.append(stmt)
    # every module gets an __init__ fn, even an empty one: `import x` always
    # lowers to call:__init__x and the executor must find it (module with no
    # top-level statements simply initializes nothing)
    lowerer = _Lowerer(prefix=prefix, storage_prefix=prefix, modules=imports,
                       module_fn_names=fn_names, module_top_names=top_names)
    lowerer.lower_block(A.Block(statements=init_stmts, line=0, col=0), {})
    fns.append(TirFunction(f"__init__{dotted}", [], [], "Unit", lowerer.instrs))
    return fns


def lower_program(program: A.Program, loader=None, importer_path: str = "<input>") -> TirProgram:
    functions: List[TirFunction] = []
    seen: Set[str] = set()

    def load_module(dotted_parts, importer_file: str):
        mod = loader.load(dotted_parts, importer_file, None)
        key = str(mod.path)
        if key in seen:
            return
        seen.add(key)
        # dependencies first (post-order), so __init__ functions exist
        for dep in [s for s in mod.program.statements if isinstance(s, A.ImportStmt)]:
            load_module(dep.path, str(mod.path))
        functions.extend(_lower_module(mod))

    if loader is not None:
        for stmt in program.statements:
            if isinstance(stmt, A.ImportStmt):
                load_module(stmt.path, importer_path)

    main_stmts = []
    imports = _imports_map(program)
    fn_names = _fn_names(program)
    for stmt in program.statements:
        if isinstance(stmt, A.FnDecl):
            functions.append(lower_function(stmt, modules=imports,
                                            module_fn_names=fn_names))
        else:
            main_stmts.append(stmt)
    if main_stmts:
        # Top-level statements (incl. import-init calls, in source order)
        # become a synthetic `__main__` so the whole program runs as IR.
        lowerer = _Lowerer(modules=imports, module_fn_names=fn_names)
        lowerer.lower_block(A.Block(statements=main_stmts, line=0, col=0), {})
        functions.append(TirFunction(MAIN_FN, [], [], "Unit", lowerer.instrs))
    return TirProgram(functions)
