"""Time-T command-line interface.

See docs/ARCHITECTURE.md section 9 and docs/DESIGN_DECISIONS.md DD-8: some
subcommands are intentionally stubs that clearly report "not implemented"
rather than silently doing nothing or fabricating output.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from timet import __version__
from timet.diagnostics import Diagnostic
from timet.parser import parse, ParseError
from timet.lexer import LexError
from timet.typechecker import check, TypeError_
from timet.ir import lower_program
from timet.interpreter import Interpreter, RuntimeErr
from timet.backend import get_default_backend
from timet import memory

NOT_IMPLEMENTED = {"build", "profile", "export", "package", "doctor"}


def _read(path: str) -> str:
    return Path(path).read_text()


def _emit_error(err: Diagnostic, as_json: bool):
    if as_json:
        print(json.dumps({"status": "error", "diagnostic": err.to_json()}), file=sys.stderr)
    else:
        print(err.human(), file=sys.stderr)


def cmd_check(args) -> int:
    ok = True
    results = []
    for path in args.files:
        try:
            src = _read(path)
            program = parse(src, path)
            check(program)
            results.append({"file": path, "status": "ok"})
        except Diagnostic as e:
            ok = False
            results.append({"file": path, "status": "error", "diagnostic": e.to_json()})
            if not args.json:
                print(f"{path}: {e.human()}", file=sys.stderr)
    if args.json:
        print(json.dumps({"status": "ok" if ok else "error", "results": results}))
    elif ok:
        print(f"checked {len(args.files)} file(s): all OK")
    return 0 if ok else 1


def cmd_run(args) -> int:
    try:
        src = _read(args.file)
        outputs = []

        def collect(s):
            outputs.append(s)
            if not args.json:
                print(s)

        program = parse(src, args.file)
        check(program)
        if getattr(args, "via_ir", False):
            from timet.ir import lower_program
            from timet.ir_exec import run_program
            tir = lower_program(program)
            engine = "ir"
            if args.opt_level >= 1:
                from timet.optimize import optimize_program
                tir, _ = optimize_program(tir, level=args.opt_level)
            run_program(tir, stdout_write=collect)
        else:
            engine = "ast"
            interp = Interpreter(stdout_write=collect)
            interp.run(program)
        if args.json:
            print(json.dumps({"status": "ok", "engine": engine, "stdout": outputs}))
        return 0
    except Diagnostic as e:
        _emit_error(e, args.json)
        return 1


def cmd_inspect(args) -> int:
    try:
        src = _read(args.file)
        program = parse(src, args.file)
        check(program)
        payload = {"status": "ok", "file": args.file}
        if args.ir:
            tir = lower_program(program)
            if getattr(args, "opt", False):
                from timet.optimize import optimize_program
                tir, stats = optimize_program(tir, level=args.opt_level)
                payload["opt_stats"] = stats.to_json()
            if args.json:
                payload["ir"] = tir.to_json()
            else:
                print(tir.render())
                if getattr(args, "opt", False):
                    print(f"-- opt stats: {json.dumps(payload['opt_stats']['passes'], sort_keys=True)} "
                          f"({payload['opt_stats']['instrs_before']} -> "
                          f"{payload['opt_stats']['instrs_after']} instrs)")
        if args.mem:
            payload["memory"] = memory.stats().to_json()
            if not args.json:
                print(json.dumps(payload["memory"], indent=2))
        if args.backend:
            caps = get_default_backend().capabilities()
            payload["backend"] = caps.to_json()
            if not args.json:
                print(json.dumps(payload["backend"], indent=2))
        if args.json:
            print(json.dumps(payload, indent=2))
        return 0
    except Diagnostic as e:
        _emit_error(e, args.json)
        return 1


def cmd_test(args) -> int:
    import subprocess
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if args.pattern:
        cmd += ["-k", args.pattern]
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent),
                           capture_output=True, text=True)
    if args.json:
        print(json.dumps({
            "status": "ok" if proc.returncode == 0 else "fail",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }))
    else:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


def cmd_bench(args) -> int:
    from benchmarks.run_benchmarks import run_all
    results = run_all()
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results["benchmarks"]:
            print(f"{r['name']:30s} {r['seconds']*1000:10.3f} ms   {r.get('note','')}")
    return 0


def cmd_repl(args) -> int:
    from timet.parser import Parser
    from timet.lexer import tokenize
    interp = Interpreter()
    print(f"Time-T v{__version__} REPL. Type an expression or Ctrl-D to exit.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return 0
        if not line.strip():
            continue
        try:
            tokens = tokenize(line)
            p = Parser(tokens)
            expr = p.parse_expr()
            from timet.typechecker import TypeChecker
            tc = TypeChecker()
            tc.infer(expr, tc.global_scope)
            value = interp.eval(expr, interp.globals)
            if value is not None:
                if hasattr(value, "tolist"):
                    print(repr(value))
                else:
                    print(value)
        except Diagnostic as e:
            print(e.human(), file=sys.stderr)


def cmd_not_implemented(name):
    def handler(args) -> int:
        msg = f"'time-t {name}' is not implemented yet in this version of Time-T."
        if args.json:
            print(json.dumps({"status": "not_implemented", "command": name, "message": msg}))
        else:
            print(msg, file=sys.stderr)
            print("See docs/ROADMAP.md for what is implemented and what is planned.", file=sys.stderr)
        return 2
    return handler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="time-t", description="Time-T language CLI")
    p.add_argument("--version", action="version", version=f"time-t {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_json_flag(sp):
        sp.add_argument("--json", action="store_true", help="machine-readable JSON output")

    sp = sub.add_parser("check", help="type-check one or more .tt files")
    sp.add_argument("files", nargs="+")
    add_json_flag(sp)
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("run", help="run a .tt file")
    sp.add_argument("file")
    sp.add_argument("--via-ir", dest="via_ir", action="store_true",
                    help="execute via the IR executor instead of the AST interpreter")
    sp.add_argument("-O", "--opt-level", dest="opt_level", type=int, default=1,
                    choices=[0, 1], help="IR optimization level (with --via-ir)")
    add_json_flag(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("inspect", help="inspect IR / memory / backend info for a .tt file")
    sp.add_argument("file")
    sp.add_argument("--ir", action="store_true")
    sp.add_argument("--opt", action="store_true",
                    help="with --ir: show/emit the OPTIMIZED IR plus pass statistics")
    sp.add_argument("-O", "--opt-level", dest="opt_level", type=int, default=1,
                    choices=[0, 1])
    sp.add_argument("--mem", action="store_true")
    sp.add_argument("--backend", action="store_true")
    add_json_flag(sp)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("test", help="run the pytest suite")
    sp.add_argument("-k", dest="pattern", default=None)
    add_json_flag(sp)
    sp.set_defaults(func=cmd_test)

    sp = sub.add_parser("bench", help="run the benchmark suite")
    add_json_flag(sp)
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("repl", help="interactive REPL")
    add_json_flag(sp)
    sp.set_defaults(func=cmd_repl)

    for name in NOT_IMPLEMENTED:
        sp = sub.add_parser(name, help=f"(not implemented yet) {name}")
        add_json_flag(sp)
        sp.set_defaults(func=cmd_not_implemented(name))

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
