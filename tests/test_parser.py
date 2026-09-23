import pytest
from timet.parser import parse, ParseError
import timet.ast_nodes as A


def test_parse_let():
    prog = parse("let x = 5")
    assert len(prog.statements) == 1
    stmt = prog.statements[0]
    assert isinstance(stmt, A.LetStmt)
    assert stmt.name == "x"
    assert stmt.mutable is False
    assert isinstance(stmt.value, A.IntLit)
    assert stmt.value.value == 5


def test_parse_var_mutable():
    prog = parse("var z = 0")
    assert prog.statements[0].mutable is True


def test_operator_precedence():
    prog = parse("let x = 1 + 2 * 3")
    value = prog.statements[0].value
    assert isinstance(value, A.BinOp)
    assert value.op == "+"
    assert isinstance(value.right, A.BinOp)
    assert value.right.op == "*"


def test_function_decl_and_call():
    src = """
    fn add(a: Int, b: Int) -> Int {
        return a + b
    }
    let r = add(1, 2)
    """
    prog = parse(src)
    fn = prog.statements[0]
    assert isinstance(fn, A.FnDecl)
    assert fn.name == "add"
    assert [p.name for p in fn.params] == ["a", "b"]
    call_stmt = prog.statements[1]
    assert isinstance(call_stmt.value, A.Call)


def test_if_else_stmt():
    src = """
    if x > 0 {
        print(1)
    } else if x < 0 {
        print(2)
    } else {
        print(3)
    }
    """
    prog = parse(src)
    stmt = prog.statements[0]
    assert isinstance(stmt, A.IfStmt)
    assert stmt.else_branch is not None


def test_if_expression_requires_else():
    with pytest.raises(ParseError):
        parse("let x = if true { 1 }")


def test_if_expression_both_branches():
    prog = parse("let x = if true { 1 } else { 2 }")
    val = prog.statements[0].value
    assert isinstance(val, A.IfExpr)


def test_while_loop():
    src = "while x < 10 { x = x + 1 }"
    prog = parse(src)
    assert isinstance(prog.statements[0], A.WhileStmt)


def test_method_call_and_field_access():
    prog = parse("let g = x.grad")
    assert isinstance(prog.statements[0].value, A.FieldAccess)
    prog2 = parse("let s = x.sum()")
    assert isinstance(prog2.statements[0].value, A.MethodCall)


def test_list_literal_and_tensor_call():
    prog = parse("let t = tensor([1.0, 2.0, 3.0], grad=true)")
    call = prog.statements[0].value
    assert isinstance(call, A.Call)
    assert isinstance(call.args[0], A.ListExpr)
    assert "grad" in call.kwargs


def test_indexing():
    prog = parse("let a = x[0]")
    assert isinstance(prog.statements[0].value, A.Index)


def test_unexpected_token_error_has_location():
    with pytest.raises(ParseError) as exc:
        parse("let x = ")
    assert exc.value.span is not None
    assert exc.value.code == "E0103"


def test_unterminated_block_error():
    with pytest.raises(ParseError):
        parse("fn f() { let x = 1")


def test_struct_enum_match_not_implemented():
    # NOTE: 'import' left this list in v0.3.0 when the module system landed
    # (tests/test_modules.py); struct/enum/match remain honest stubs.
    for kw in ("struct", "enum", "match"):
        with pytest.raises(ParseError) as exc:
            parse(f"{kw} Foo {{}}")
        assert exc.value.code == "E0101"


def test_import_statement_parses_dotted_path_and_alias():
    prog = parse("import modules.mathlib as ml\nimport mathlib")
    assert isinstance(prog.statements[0], A.ImportStmt)
    assert prog.statements[0].path == ["modules", "mathlib"]
    assert prog.statements[0].alias == "ml"
    assert prog.statements[0].binding == "ml"
    assert prog.statements[1].path == ["mathlib"]
    assert prog.statements[1].alias is None
    assert prog.statements[1].binding == "mathlib"


def test_no_grad_block():
    prog = parse("no_grad { let y = x * 2.0 }")
    assert isinstance(prog.statements[0], A.NoGradStmt)


def test_closures_lambda():
    prog = parse("let f = fn(x: Int) -> Int { return x }")
    assert isinstance(prog.statements[0].value, A.Lambda)


def test_assignment_statement():
    prog = parse("var x = 0\nx = x + 1")
    assert isinstance(prog.statements[1], A.AssignStmt)
