"""Module system tests (Milestone 2 remainder; DD-13).

Covers: basic import + typed member access, aliasing, dotted paths, module
state with mutation, chained imports, once-only execution, and every error
path (missing file, cycles, nested import, unknown export) — across all
three execution engines (AST, IR-O0, IR-O1) where execution is involved.
"""
import pytest

from timet.parser import parse
from timet.typechecker import check, TypeError_
from timet.ir import lower_program
from timet.ir_exec import run_program
from timet.optimize import optimize_program
from timet.interpreter import run_source
from timet.modules import ModuleLoader, ModuleError


def write(tmp_path, name, src):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src)
    return p


def run_all_engines(main_file):
    """Run main_file on AST, IR-O0, IR-O1; return [ast, ir0, ir1] outputs."""
    src = main_file.read_text()
    outputs = []
    # AST
    out = []
    run_source(src, filename=str(main_file), stdout_write=out.append)
    outputs.append(out)
    # IR O0 / O1
    for level in (0, 1):
        out = []
        loader = ModuleLoader()
        program = check(parse(src, str(main_file)), filename=str(main_file), loader=loader)
        tir = lower_program(program, loader=loader, importer_path=str(main_file))
        if level:
            tir, _ = optimize_program(tir, level=level)
        run_program(tir, stdout_write=out.append)
        outputs.append(out)
    return outputs


MATHLIB = """
let PI = 3.14

fn add(a: Int, b: Int) -> Int { return a + b }
fn square(x: Int) -> Int { return x * x }
fn area_square(x: Int) -> Int { return square(x) }
"""


def test_basic_import_typed_calls_and_values(tmp_path):
    write(tmp_path, "mathlib.tt", MATHLIB)
    main = write(tmp_path, "main.tt", """
import mathlib
fn main() {
    print(mathlib.add(3, 4))
    print(mathlib.area_square(5))
    print(mathlib.PI)
}
""")
    ast, ir0, ir1 = run_all_engines(main)
    assert ast == ir0 == ir1 == ["7", "25", "3.14"]


def test_import_alias(tmp_path):
    write(tmp_path, "mathlib.tt", MATHLIB)
    main = write(tmp_path, "main.tt", """
import mathlib as ml
fn main() { print(ml.add(1, 2)) }
""")
    ast, ir0, ir1 = run_all_engines(main)
    assert ast == ir0 == ir1 == ["3"]


def test_dotted_path_import_from_subdirectory(tmp_path):

    (tmp_path / "lib").mkdir(exist_ok=True)
    write(tmp_path, "lib/greet.tt", """
fn hello(name: String) -> String { return "hi " + name }
""")
    main = write(tmp_path, "main.tt", """
import lib.greet
fn main() { print(greet.hello("tim")) }
""")
    ast, ir0, ir1 = run_all_engines(main)
    assert ast == ir0 == ir1 == ["hi tim"]


def test_module_state_mutation_through_module_fn(tmp_path):
    write(tmp_path, "counter.tt", """
var count = 0
fn bump() -> Int {
    count = count + 1
    return count
}
""")
    main = write(tmp_path, "main.tt", """
import counter
fn main() {
    print(counter.bump())
    print(counter.bump())
    print(counter.count)
}
""")
    ast, ir0, ir1 = run_all_engines(main)
    assert ast == ir0 == ir1 == ["1", "2", "2"]


def test_chained_imports_and_once_only_initialization(tmp_path):
    write(tmp_path, "a.tt", """
import b
fn from_a() -> Int { return b.value() + 1 }
""")
    write(tmp_path, "b.tt", """
print("initializing b")
fn value() -> Int { return 41 }
""")
    main = write(tmp_path, "main.tt", """
import a
import b
fn main() { print(a.from_a()) }
""")
    ast, ir0, ir1 = run_all_engines(main)
    # b's top-level print must run exactly ONCE despite two import paths
    assert ast == ir0 == ir1 == ["initializing b", "42"]


def test_module_does_not_auto_run_its_main(tmp_path):
    write(tmp_path, "m.tt", """
fn main() { print("module main ran") }
fn util() -> Int { return 1 }
""")
    main = write(tmp_path, "main.tt", """
import m
fn main() { print(m.util()) }
""")
    ast, ir0, ir1 = run_all_engines(main)
    assert ast == ir0 == ir1 == ["1"]


def test_module_scope_is_fresh_no_importer_leakage(tmp_path):
    write(tmp_path, "m.tt", """
fn f() -> Int { return secret + 1 }
""")
    main = write(tmp_path, "main.tt", """
import m
let secret = 10
fn main() { print(m.f()) }
""")
    # 'secret' must NOT be visible inside the module -> compile error
    with pytest.raises(TypeError_) as e:
        check(parse(main.read_text(), str(main)), filename=str(main),
              loader=ModuleLoader())
    assert "undefined name 'secret'" in str(e.value)


def test_missing_module_error_mentions_path(tmp_path):
    main = write(tmp_path, "main.tt", "import does_not_exist\n")
    with pytest.raises(ModuleError) as e:
        check(parse(main.read_text(), str(main)), filename=str(main),
              loader=ModuleLoader())
    msg = str(e.value)
    assert "E0213" in msg and "does_not_exist" in msg and "looked for" in msg


def test_import_cycle_detected(tmp_path):
    write(tmp_path, "a.tt", "import b\nfn f() -> Int { return 1 }\n")
    write(tmp_path, "b.tt", "import a\nfn g() -> Int { return 2 }\n")
    main = write(tmp_path, "main.tt", "import a\n")
    with pytest.raises(ModuleError) as e:
        check(parse(main.read_text(), str(main)), filename=str(main),
              loader=ModuleLoader())
    assert "E0210" in str(e.value) and "cycle" in str(e.value)


def test_nested_import_rejected(tmp_path):
    write(tmp_path, "m.tt", "fn f() -> Int { return 1 }\n")
    main = write(tmp_path, "main.tt", """
fn main() {
    import m
}
""")
    with pytest.raises(TypeError_) as e:
        check(parse(main.read_text(), str(main)), filename=str(main),
              loader=ModuleLoader())
    assert "E0212" in str(e.value)


def test_unknown_export_error_lists_exports(tmp_path):
    write(tmp_path, "mathlib.tt", MATHLIB)
    main = write(tmp_path, "main.tt", """
import mathlib
fn main() { print(mathlib.nope(1)) }
""")
    with pytest.raises(TypeError_) as e:
        check(parse(main.read_text(), str(main)), filename=str(main),
              loader=ModuleLoader())
    msg = str(e.value)
    assert "E0211" in msg and "nope" in msg and "add" in msg  # exports listed


def test_module_fn_type_mismatch_caught_statically(tmp_path):
    write(tmp_path, "mathlib.tt", MATHLIB)
    main = write(tmp_path, "main.tt", """
import mathlib
fn main() -> Int { return mathlib.add(1, 2) + true }
""")
    with pytest.raises(TypeError_):
        check(parse(main.read_text(), str(main)), filename=str(main),
              loader=ModuleLoader())
