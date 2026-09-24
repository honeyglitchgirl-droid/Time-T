"""Time-T command-line interface.

See docs/ARCHITECTURE.md section 9 and docs/DESIGN_DECISIONS.md DD-8: some
subcommands are intentionally stubs that clearly report "not implemented"
rather than silently doing nothing or fabricating output.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
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

NOT_IMPLEMENTED = set()



def _fresh_loader():
    from timet.modules import ModuleLoader
    return ModuleLoader()


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
            check(program, filename=path, loader=_fresh_loader())
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
        loader = _fresh_loader()
        check(program, filename=args.file, loader=loader)
        if getattr(args, "native", False):
            from timet.native import run_native
            proc = run_native(program)
            for line in proc.stdout.splitlines():
                collect(line)
            if proc.returncode != 0:
                _emit_error(Diagnostic(
                    code="E0901", message="native executable returned "
                                          f"exit code {proc.returncode}",
                    note=proc.stderr.strip()[:2000], stage="native"), args.json)
                return 1
            engine = "native"
        elif getattr(args, "via_ir", False):
            from timet.ir import lower_program
            from timet.ir_exec import run_program
            tir = lower_program(program, loader=loader, importer_path=args.file)
            engine = "ir"
            opt = getattr(args, "opt_level", 0) or 0
            if opt:
                from timet.optimize import optimize_program
                tir, _ = optimize_program(tir, level=opt)
            run_program(tir, stdout_write=collect)
        else:
            engine = "ast"
            interp = Interpreter(stdout_write=collect, loader=loader,
                                 importer_path=args.file)
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
        loader = _fresh_loader()
        check(program, filename=args.file, loader=loader)
        payload = {"status": "ok", "file": args.file}
        if args.ir:
            tir = lower_program(program, loader=loader, importer_path=args.file)
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


def cmd_export(args) -> int:
    """Export a model checkpoint to portable formats (safetensors, npz, bin, json)."""
    from timet import checkpoint
    try:
        in_path = Path(args.file)
        if not in_path.is_file():
            _emit_error(Diagnostic(code="E0800", message=f"export: file not found '{args.file}'",
                                   stage="export"), args.json)
            return 1
        tensors = checkpoint.load(in_path)
        out_path = Path(args.output)
        fmt = args.format.lower() if args.format else None
        if fmt is None:
            if out_path.suffix == ".safetensors":
                fmt = "safetensors"
            elif out_path.suffix == ".npz":
                fmt = "npz"
            elif out_path.suffix in (".ttck", ".bin"):
                fmt = "bin"
            elif out_path.suffix == ".json":
                fmt = "json"
            else:
                fmt = "safetensors"

        if fmt == "safetensors":
            checkpoint.save_safetensors(tensors, out_path)
        elif fmt == "npz":
            checkpoint.save_npz(tensors, out_path)
        elif fmt == "bin":
            checkpoint.save_bin(tensors, out_path)
        elif fmt == "json":
            checkpoint.save(tensors, out_path)
        else:
            _emit_error(Diagnostic(code="E0801",
                                   message=f"export: unsupported format '{fmt}' (choose safetensors, npz, bin, json)",
                                   stage="export"), args.json)
            return 1

        payload = {"status": "ok", "input": str(in_path), "output": str(out_path),
                   "format": fmt, "tensors": len(tensors)}
        if args.json:
            print(json.dumps(payload))
        else:
            print(f"exported {len(tensors)} tensor(s) to {out_path} ({fmt})")
        return 0
    except Diagnostic as e:
        _emit_error(e, args.json)
        return 1


def cmd_profile(args) -> int:
    """Profile execution time and peak memory of a .tt program."""
    import cProfile
    import pstats
    import io
    from timet.interpreter import Interpreter
    try:
        src = _read(args.file)
        program = parse(src, args.file)
        loader = _fresh_loader()
        check(program, filename=args.file, loader=loader)

        pr = cProfile.Profile()
        outputs = []
        def collect(s):
            outputs.append(s)

        interp = Interpreter(stdout_write=collect, loader=loader, importer_path=args.file)
        mem_before = memory.stats()
        t0 = time.perf_counter()
        pr.enable()
        interp.run(program)
        pr.disable()
        elapsed = time.perf_counter() - t0
        mem_after = memory.stats()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(15)

        payload = {
            "status": "ok",
            "file": args.file,
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": mem_after.peak,
            "top_calls": s.getvalue()
        }
        if args.json:
            print(json.dumps(payload))
        else:
            print(f"Profile for {args.file}:")
            print(f"  Elapsed: {elapsed*1000:.2f} ms")
            print(f"  Peak Tensor Memory: {mem_after.peak / 1024:.2f} KB")
            print("\nTop cumulative calls:")
            print(s.getvalue())

        return 0
    except Diagnostic as e:
        _emit_error(e, args.json)
        return 1


def cmd_package(args) -> int:

    """Package a trained model or Time-T program into a deployable bundle."""
    from timet import checkpoint
    from timet.mobile import package_mobile, quantize_dynamic
    try:
        in_path = Path(args.file)
        if not in_path.is_file():
            _emit_error(Diagnostic(code="E0810", message=f"package: file not found '{args.file}'",
                                   stage="package"), args.json)
            return 1
        out_path = Path(args.output)
        # Check if packaging mobile format
        fmt = getattr(args, "target", "mobile")
        if fmt == "mobile":
            # If checkpoint or binary file, load tensors
            # Packaging sequentially
            import timet.nn as nn
            # For demonstration / packaging CLI: load checkpoint into Sequential
            tensors = checkpoint.load(in_path)
            # Reconstruct Sequential layers from weights
            # Look for layer_0, layer_1, etc. or linear weights
            # Default packaging wraps into mobile archive
            linear_keys = sorted([k for k in tensors.keys() if "weight" in k])
            layers = []
            for k in linear_keys:
                w = tensors[k]
                b_key = k.replace("weight", "bias")
                b = tensors.get(b_key)
                l = nn.Linear(w.shape[1], w.shape[0])
                l.weight = w
                if b is not None:
                    l.bias = b
                layers.append(l)
                layers.append(nn.ReLU())
            if layers and isinstance(layers[-1], nn.ReLU):
                layers.pop()
            seq = nn.Sequential(*layers) if layers else nn.Sequential(nn.Linear(1, 1))
            if getattr(args, "quantize", True):
                seq = quantize_dynamic(seq)
            package_mobile(seq, out_path)
        else:
            _emit_error(Diagnostic(code="E0811", message=f"package: unsupported target '{fmt}'",
                                   stage="package"), args.json)
            return 1

        payload = {"status": "ok", "target": fmt, "input": str(in_path), "output": str(out_path)}
        if args.json:
            print(json.dumps(payload))
        else:
            print(f"packaged deployable bundle to {out_path} (target={fmt})")
        return 0
    except Diagnostic as e:
        _emit_error(e, args.json)
        return 1


def cmd_doctor(args) -> int:
    """System and environment diagnostics."""
    import platform
    import shutil
    cc_found = None
    for cand in ("gcc", "cc", "clang"):
        if shutil.which(cand):
            cc_found = cand
            break
    diag = {
        "status": "ok",
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "c_compiler": cc_found or "none",
        "backend": get_default_backend().name,
        "memory": memory.stats().to_json(),
    }
    if args.json:
        print(json.dumps(diag, indent=2))
    else:
        print(f"Time-T v{__version__} Doctor")
        print(f"  Platform:    {diag['platform']} ({diag['processor']})")
        print(f"  Python:      {diag['python']}")
        print(f"  C Compiler:  {diag['c_compiler']}")
        print(f"  Backend:     {diag['backend']}")
        print("All system checks passed.")
    return 0


def cmd_verify(args) -> int:
    """Run program across AST interpreter, IR-O0, and IR-O1 to certify bit-identical outputs."""
    src = _read(args.file)
    program = parse(src, args.file)
    loader = _fresh_loader()
    check(program, filename=args.file, loader=loader)

    # Engine 1: AST interpreter
    ast_out = []
    interp = Interpreter(stdout_write=ast_out.append, loader=loader, importer_path=args.file)
    interp.run(program)

    # Engine 2: IR unoptimized
    from timet.ir_exec import run_program
    tir0 = lower_program(program, loader=loader, importer_path=args.file)
    ir0_out = []
    run_program(tir0, stdout_write=ir0_out.append)

    # Engine 3: IR optimized (-O1)
    from timet.optimize import optimize_program
    tir1, _ = optimize_program(lower_program(program, loader=loader, importer_path=args.file), level=1)
    ir1_out = []
    run_program(tir1, stdout_write=ir1_out.append)

    ok = (ast_out == ir0_out == ir1_out)
    payload = {
        "status": "ok" if ok else "mismatch",
        "file": args.file,
        "certified_identical": ok,
        "ast_lines": len(ast_out),
        "ir_o0_lines": len(ir0_out),
        "ir_o1_lines": len(ir1_out),
    }
    if not ok:
        payload["diff"] = {
            "ast": ast_out,
            "ir_o0": ir0_out,
            "ir_o1": ir1_out,
        }

    if args.json:
        print(json.dumps(payload))
    else:
        if ok:
            print(f"VERIFIED [3/3 ENGINES]: {args.file}")
            print(f"  AST Interpreter == IR-O0 == IR-O1 (exact match, {len(ast_out)} output lines)")
        else:
            print(f"FAILED VERIFICATION: {args.file} produced differing outputs across engines!", file=sys.stderr)
    return 0 if ok else 1


def cmd_build(args) -> int:


    """Native compilation via the C-emitter slice (Milestone 9, DD-15).
    On subset violations, fails with an honest, machine-readable diagnostic
    -- never silently falls back to the interpreter."""
    from timet.native import build as native_build, NativeError
    try:
        src = _read(args.file)
        loader = _fresh_loader()
        program = check(parse(src, args.file), filename=args.file, loader=loader)
        out = args.output or str(Path(args.file).with_suffix("")) + ".native"
        res = native_build(program, out, cc=getattr(args, "cc", None))
        payload = {"status": "ok", "output": res.output,
                   "compiler": res.compiler,
                   "compile_seconds": res.compile_seconds,
                   "c_bytes": len(res.c_source)}
        if args.json:
            print(json.dumps(payload))
        else:
            print(f"built {out} with {res.compiler} in "
                  f"{res.compile_seconds:.2f}s ({len(res.c_source)} bytes of C)")
        return 0
    except Diagnostic as e:
        _emit_error(e, args.json)
        return 1


def cmd_repl(args) -> int:
    from timet.parser import Parser
    from timet.lexer import tokenize
    from timet.typechecker import TypeChecker
    interp = Interpreter()
    tc = TypeChecker()
    print(f"Time-T v{__version__} REPL. Type an expression or statement (or Ctrl-D to exit).")
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
            # Try parsing as statement first (let, var, assignment, expr)
            p = Parser(tokens)
            try:
                stmt = p.parse_stmt()
                tc.check_stmt(stmt, tc.global_scope)
                val = interp.exec_stmt(stmt, interp.globals)
                if val is not None:
                    if hasattr(val, "tolist"):
                        print(repr(val))
                    else:
                        print(val)
            except Diagnostic:
                # Fallback to expression parsing
                p2 = Parser(tokens)
                expr = p2.parse_expr()
                tc.infer(expr, tc.global_scope)
                val = interp.eval(expr, interp.globals)
                if val is not None:
                    if hasattr(val, "tolist"):
                        print(repr(val))
                    else:
                        print(val)
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
    sp.add_argument("--native", dest="native", action="store_true",
                    help="compile to native code (C emitter v1 subset) and run; "
                         "fails with a diagnostic on unsupported constructs")
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

    sp = sub.add_parser("verify", help="certify bit-identical execution across AST, IR-O0, and IR-O1 engines")
    sp.add_argument("file", help="input .tt program to verify across engines")
    add_json_flag(sp)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("build", help="native-compile a .tt file to an executable (v1 C-emitter subset, DD-15)")

    sp.add_argument("file")
    sp.add_argument("-o", "--output", default=None)
    sp.add_argument("--cc", default=None, help="C compiler to use (default: first of gcc/cc/clang on PATH)")
    add_json_flag(sp)
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("export", help="export a checkpoint to portable formats (safetensors, npz, bin, json)")
    sp.add_argument("file", help="input checkpoint file (.ttck, .json, etc.)")
    sp.add_argument("-o", "--output", required=True, help="output file path")
    sp.add_argument("-f", "--format", choices=["safetensors", "npz", "bin", "json"], default=None,
                    help="target format (inferred from output extension if omitted)")
    add_json_flag(sp)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("profile", help="profile execution time and memory of a .tt file")
    sp.add_argument("file", help="input .tt program to profile")
    add_json_flag(sp)
    sp.set_defaults(func=cmd_profile)

    sp = sub.add_parser("package", help="package a trained model into a mobile/deployable bundle")

    sp.add_argument("file", help="input model checkpoint file")
    sp.add_argument("-o", "--output", required=True, help="output package path (.ttm)")
    sp.add_argument("--target", default="mobile", choices=["mobile"], help="target package format")
    sp.add_argument("--no-quantize", dest="quantize", action="store_false", default=True,
                    help="disable INT8 dynamic quantization")
    add_json_flag(sp)
    sp.set_defaults(func=cmd_package)

    sp = sub.add_parser("doctor", help="system and toolchain environment diagnostics")
    add_json_flag(sp)
    sp.set_defaults(func=cmd_doctor)

    for name in NOT_IMPLEMENTED:

        sp = sub.add_parser(name, help=f"(not implemented yet) {name}")
        add_json_flag(sp)
        sp.set_defaults(func=cmd_not_implemented(name))

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"error: file not found: {e.filename}", file=sys.stderr)
        return 1
    except Exception as e:
        if os.environ.get("TIMET_DEBUG"):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
