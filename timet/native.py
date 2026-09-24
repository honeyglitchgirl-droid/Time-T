"""Native code emission for a STRICT subset of Time-T (Milestone 9, DD-15).

Pipeline: typed AST -> C11 source -> system C compiler (gcc/cc/clang) ->
native executable. This is a *vertical slice*, deliberately small:

SUPPORTED (byte-verified against the AST interpreter on every test):
  Int (int64 two's-complement; Python's arbitrary precision does NOT apply
       beyond 2^63 -- documented divergence), Float (double),
  Bool, String literals (assign + print only -- no concatenation),
  + - * / % (Int arithmetic with PYTHON-SEMANTIC floor modulo; / always
       performs double division, exactly like the interpreter), comparisons,
  unary - / !, eager-enough && and || (operands must be Bool-typed so the
  short-circuit/operand-return semantics coincide),
  let/var with block scoping, if/else-if/else, while,
  typed function declarations incl. recursion (all params + return type
  must be annotated),
  print(x) of Int / Bool / String values ONLY.

NOT SUPPORTED (each rejected with a clear NativeError naming the construct
and its location -- NEVER a silent fallback):
  printing Float values (needs shortest-repr float formatting parity with
  Python's repr -- a known, documented limitation), for loops, tensors and
  any tensor/builtin calls other than print, lambdas, nested fn decls,
  closures over outer scopes, f-strings, imports, string concatenation,
  mixed Int/Float modulo, and everything else not listed above.

The emitter reads `.ty` annotations the type checker attached to every
expression node; it never invents types. Generated C is deterministic:
functions appear in source order, no timestamps, no random naming.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import timet.ast_nodes as A
from timet.diagnostics import Diagnostic
from timet.types import TInt, TFloat, TBool, TString, TUnit


CC_SEARCH_ORDER = ("gcc", "cc", "clang")


class NativeError(Diagnostic):
    def __init__(self, message: str, span=None, note: str = None):
        super().__init__(severity="error", code="E0950", message=message,
                         span=span, stage="native", note=note)


def _span(node):
    from timet.diagnostics import SourceSpan
    line = getattr(node, "line", None)
    col = getattr(node, "col", None)
    return SourceSpan(line, col) if line is not None else None


def find_compiler() -> Optional[str]:
    for cc in CC_SEARCH_ORDER:
        path = shutil.which(cc)
        if path:
            return path
    return None


_TYPE_NAMES = {TInt: "int64_t", TFloat: "double", TBool: "int",
               TString: "const char *"}


def _ctype(ty) -> Optional[str]:
    for cls, name in _TYPE_NAMES.items():
        if isinstance(ty, cls):
            return name
    return None


class _Emitter:
    def __init__(self):
        self.fns: List[str] = []     # function definitions
        self.protos: List[str] = []  # forward declarations
        self.user_main = False

    # ---------- helpers ----------
    def _reject(self, node, what: str):
        raise NativeError(
            f"native: {what} is not supported by the native backend (v1 subset)",
            span=_span(node))

    def _ty_of(self, node) -> str:
        ty = getattr(node, "ty", None)
        c = _ctype(ty) if ty is not None else None
        if c is None:
            raise NativeError(
                f"native: expression has type "
                f"'{getattr(ty, 'name', ty)}' which the native backend "
                f"cannot map to C (v1 supports Int/Float/Bool/String only)",
                span=_span(node))
        return c

    def _mangle(self, name: str) -> str:
        return f"tt_{name}"

    # ---------- expressions ----------
    def emit_expr(self, e: A.Expr) -> str:
        if isinstance(e, A.IntLit):
            return f"{e.value}LL"
        if isinstance(e, A.FloatLit):
            return repr(float(e.value))
        if isinstance(e, A.BoolLit):
            return "1" if e.value else "0"
        if isinstance(e, A.StringLit):
            return '"' + e.value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(e, A.Ident):
            return self._mangle(e.name)
        if isinstance(e, A.UnaryOp):
            inner = self.emit_expr(e.operand)
            if e.op == "-":
                return f"(-({inner}))"
            if e.op == "!":
                return f"(!({inner}))"
            self._reject(e, f"unary operator '{e.op}'")
        if isinstance(e, A.BinOp):
            return self._emit_binop(e)
        if isinstance(e, A.Call):
            return self._emit_call(e)
        if isinstance(e, A.IfExpr):
            cond_str = self.emit_expr(e.cond)
            then_expr = self._emit_block_as_expr(e.then_branch)
            else_expr = self._emit_block_as_expr(e.else_branch) if e.else_branch else "0"
            return f"(({cond_str}) ? ({then_expr}) : ({else_expr}))"
        self._reject(e, f"expression kind {type(e).__name__}")

    def _emit_binop(self, e: A.BinOp) -> str:
        lt, rt = getattr(e.left, "ty", None), getattr(e.right, "ty", None)
        a = self.emit_expr(e.left)
        b = self.emit_expr(e.right)
        op = e.op
        if op in ("&&", "||"):
            if not (isinstance(lt, TBool) and isinstance(rt, TBool)):
                raise NativeError(
                    "native: && and || require Bool operands (operand-return "
                    "semantics of other types is not native-compatible)",
                    span=_span(e))
            return f"(({a}) {'&&' if op == '&&' else '||'} ({b}))"
        if op == "%":
            if isinstance(lt, TInt) and isinstance(rt, TInt):
                return f"tt_mod(({a}), ({b}))"
            raise NativeError(
                "native: % is only supported on Int operands in v1",
                span=_span(e))
        if op == "/":
            # interpreter: / is ALWAYS double division, even on ints
            return f"((double)({a}) / (double)({b}))"
        if op in ("+", "-", "*"):
            if isinstance(lt, (TString,)) or isinstance(rt, (TString,)):
                raise NativeError(
                    "native: string arithmetic is not supported in v1 "
                    "(literals, assignment, and print only)",
                    span=_span(e))
            return f"(({a}) {op} ({b}))"
        if op in ("==", "!="):
            if isinstance(lt, TString) or isinstance(rt, TString):
                eq_call = f"(strcmp(({a}), ({b})) == 0)"
                return eq_call if op == "==" else f"(!{eq_call})"
            return f"(({a}) {op} ({b}))"
        if op in ("<", "<=", ">", ">="):
            if isinstance(lt, TString) or isinstance(rt, TString):
                cmp_call = f"strcmp(({a}), ({b}))"
                return f"({cmp_call} {op} 0)"
            return f"(({a}) {op} ({b}))"
        self._reject(e, f"operator '{op}'")

    def _emit_call(self, e: A.Call) -> str:
        callee = e.callee
        if isinstance(callee, A.Ident) and callee.name == "print":
            self._reject(e, "'print' inside a sub-expression (print returns "
                              "Unit; use it as a statement)")
        if not isinstance(callee, A.Ident):
            self._reject(e, "non-identifier call target")
        name = callee.name
        if name in ("tensor", "matmul", "zeros", "ones", "relu"):
            raise NativeError(
                f"native: '{name}()' is a tensor/builtin call; tensors are "
                f"not in the native v1 subset",
                span=_span(e))
        if name not in self._fn_names:
            # unresolved at emission time: either a builtin we don't
            # support, or something the type checker should have caught
            raise NativeError(
                f"native: call to '{name}'() is not in the v1 subset (only "
                f"calls to declared Time-T functions compile natively)",
                span=_span(e))
        args = ", ".join(self.emit_expr(a) for a in e.args)
        return f"{self._mangle(name)}({args})"

    def _emit_block_as_expr(self, b: A.Block) -> str:
        if not b.statements:
            return "0"
        if len(b.statements) == 1 and isinstance(b.statements[0], A.ExprStmt):
            return self.emit_expr(b.statements[0].expr)
        inner = []
        for s in b.statements[:-1]:
            self.emit_stmt(s, inner)
        last = b.statements[-1]
        if isinstance(last, A.ExprStmt):
            inner.append(f"{self.emit_expr(last.expr)};")
        else:
            self.emit_stmt(last, inner)
        stmts_joined = " ".join(inner)
        return f"({{ {stmts_joined} }})"

    # ---------- statements ----------
    def emit_stmt(self, s: A.Stmt, out: List[str]):
        if isinstance(s, A.ExprStmt) and isinstance(s.expr, A.Call) \
                and isinstance(s.expr.callee, A.Ident) \
                and s.expr.callee.name == "print":
            if len(s.expr.args) != 1:
                self._reject(s, "print with != 1 argument")
            arg = s.expr.args[0]
            if s.expr.kwargs:
                self._reject(s, "print keyword arguments")
            self._emit_print(arg, out)
            return
        if isinstance(s, A.LetStmt):
            c = self._ty_of(s.value)
            out.append(f"{c} {self._mangle(s.name)} = {self.emit_expr(s.value)};")
            return
        if isinstance(s, A.AssignStmt) and isinstance(s.target, A.Ident):
            out.append(f"{self._mangle(s.target.name)} = "
                       f"{self.emit_expr(s.value)};")
            return
        if isinstance(s, A.ReturnStmt):
            if s.value is None:
                out.append("return;")
            else:
                out.append(f"return {self.emit_expr(s.value)};")
            return
        if isinstance(s, A.ExprStmt):
            # any other expression statement (e.g. a function call for effect)
            out.append(f"{self.emit_expr(s.expr)};")
            return
        if isinstance(s, A.IfStmt):
            out.append(f"if ({self.emit_expr(s.cond)}) {{")
            self.emit_block(s.then_branch, out)
            out.append("}")
            if s.else_branch is not None:
                out.append("else {")
                self.emit_block(s.else_branch, out)
                out.append("}")
            return
        if isinstance(s, A.WhileStmt):
            out.append(f"while ({self.emit_expr(s.cond)}) {{")
            self.emit_block(s.body, out)
            out.append("}")
            return
        if isinstance(s, A.BreakStmt):
            out.append("break;")
            return
        if isinstance(s, A.ContinueStmt):
            out.append("continue;")
            return
        self._reject(s, f"statement kind {type(s).__name__}")

    def _emit_print(self, arg: A.Expr, out: List[str]):
        ty = getattr(arg, "ty", None)
        code = self.emit_expr(arg)
        if isinstance(ty, TFloat):
            out.append(f'tt_print_float({code});')
        elif isinstance(ty, TBool):
            out.append(f'printf("%s\\n", ({code}) ? "true" : "false");')
        elif isinstance(ty, TString):
            out.append(f'printf("%s\\n", {code});')
        elif isinstance(ty, TInt):
            out.append(f'printf("%lld\\n", (long long)({code}));')
        else:
            self._reject(arg, f"print of type '{getattr(ty, 'name', ty)}'")


    def emit_block(self, block: A.Block, out: List[str]):
        for s in block.statements:
            self.emit_stmt(s, out)

    # ---------- functions ----------
    _fn_names = None

    def declare_functions(self, program: A.Program):
        for s in program.statements:
            if isinstance(s, A.FnDecl):
                proto, ret_c, params_c = self._fn_signature(s)
                self.protos.append(proto)
                if s.name == "main":
                    self.user_main = True

    def _fn_signature(self, fn: A.FnDecl):
        prims = {"Int": "int64_t", "Float": "double", "Bool": "int",
                 "String": "const char *"}
        params = []
        for prm in fn.params:
            if prm.type_expr is None or prm.type_expr.name not in prims \
                    or prm.type_expr.args:
                raise NativeError(
                    f"native: parameter '{prm.name}' of function "
                    f"'{fn.name}' must be annotated Int/Float/Bool/String "
                    f"for native emission",
                    span=_span(prm))
            params.append(f"{prims[prm.type_expr.name]} "
                          f"{self._mangle(prm.name)}")
        if fn.ret_type is None:
            if fn.name != "main":
                raise NativeError(
                    f"native: function '{fn.name}' needs a return-type "
                    f"annotation (Int/Float/Bool/String) for native "
                    f"emission",
                    span=_span(fn))
            ret_c = "void"
        elif fn.ret_type.name in prims and not fn.ret_type.args:
            ret_c = prims[fn.ret_type.name]
        else:
            raise NativeError(
                f"native: '{fn.name}' has return type "
                f"'{fn.ret_type.name}' which the native backend cannot "
                f"map to C",
                span=_span(fn))
        params_c = ", ".join(params) if params else "void"
        return (f"static {ret_c} {self._mangle(fn.name)}({params_c});",
                ret_c, params_c)

    def emit_function(self, fn: A.FnDecl) -> str:
        proto, ret_c, params_c = self._fn_signature(fn)
        header = proto[:-1].replace("static ", "static ", 1)
        lines = [f"static {ret_c} {self._mangle(fn.name)}({params_c}) {{"]
        self.emit_block(fn.body, lines)
        if ret_c == "void":
            lines.append("return;")
        lines.append("}")
        return "\n".join(lines)


PRELUDE = """// generated by timet-native (DD-15); deterministic -- do not edit
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// Python-semantics floor modulo on int64 (Time-T follows Python, not C)
static inline int64_t tt_mod(int64_t a, int64_t b) {
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

static inline void tt_print_float(double d) {
    if (isnan(d)) {
        printf("nan\\n");
        return;
    }
    if (isinf(d)) {
        printf("%sinf\\n", d < 0 ? "-" : "");
        return;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.15g", d);
    if (!strchr(buf, '.') && !strchr(buf, 'e')) {
        strncat(buf, ".0", sizeof(buf) - strlen(buf) - 1);
    }
    printf("%s\\n", buf);
}

"""



def emit_c(program: A.Program) -> str:
    """Emit deterministic C11 for the native subset. Raises NativeError
    on the FIRST unsupported construct (with span), never a fallback."""
    em = _Emitter()
    em._fn_names = {s.name for s in program.statements if isinstance(s, A.FnDecl)}
    em.declare_functions(program)
    fn_defs = []
    top: List[str] = []
    for s in program.statements:
        if isinstance(s, A.FnDecl):
            fn_defs.append(em.emit_function(s))
        else:
            em.emit_stmt(s, top)
    body = ["static void tt_toplevel(void) {"] + top
    if em.user_main:
        body.append("tt_main();")
    body += ["}", "", "int main(void) {", "    tt_toplevel();", "    return 0;", "}"]
    parts = [PRELUDE] + em.protos + ["", *fn_defs, "", "\n".join(body)]
    return "\n".join(parts) + "\n"


@dataclass
class BuildResult:
    output: str
    c_source: str
    compile_seconds: float
    compiler: str


def build(program: A.Program, output: str, cc: Optional[str] = None) -> BuildResult:
    """Compile a typed program to a native executable via the system C
    compiler. Raises NativeError if no compiler or compilation fails."""
    cc = cc or find_compiler()
    if cc is None:
        raise NativeError(
            "native: no C compiler found on PATH (looked for gcc, cc, clang); "
            "native backends require a host C toolchain")
    src = emit_c(program)
    with tempfile.TemporaryDirectory(prefix="timet-native-") as td:
        cpath = Path(td) / "program.c"
        cpath.write_text(src)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [cc, "-O2", "-std=c11", "-o", output, str(cpath)],
            capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if proc.returncode != 0:
            raise NativeError(
                "native: C compiler failed (this is a compiler-bug-shaped "
                "event, not a program error)",
                note=proc.stderr.strip()[:2000])
    return BuildResult(output=output, c_source=src, compile_seconds=dt,
                       compiler=cc)


def run_native(program: A.Program) -> subprocess.CompletedProcess:
    """Compile to a temp binary and run it; returns the CompletedProcess
    with captured text stdout/stderr."""
    cc = find_compiler()
    if cc is None:
        raise NativeError(
            "native: no C compiler found on PATH (looked for gcc, cc, clang)")
    src = emit_c(program)
    with tempfile.TemporaryDirectory(prefix="timet-native-") as td:
        exe = str(Path(td) / "a.out")
        build(program, exe, cc=cc)
        return subprocess.run([exe], capture_output=True, text=True, timeout=120)
