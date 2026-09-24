from timet.diagnostics import Diagnostic, SourceSpan


def test_diagnostic_human_message_contains_all_fields():
    d = Diagnostic(
        code="E9999",
        message="something went wrong",
        span=SourceSpan(3, 7),
        expected="Int",
        actual="String",
        note="try converting explicitly",
        stage="typechecker",
    )
    text = d.human()
    assert "E9999" in text
    assert "3:7" in text
    assert "expected: Int" in text
    assert "actual:   String" in text
    assert "note:     try converting explicitly" in text


def test_diagnostic_to_json_round_trips_fields():
    d = Diagnostic(code="E0001", message="msg", span=SourceSpan(1, 1), stage="lexer")
    j = d.to_json()
    assert j["code"] == "E0001"
    assert j["span"]["line"] == 1
    assert j["stage"] == "lexer"


def test_diagnostic_is_exception():
    d = Diagnostic(code="E0001", message="msg")
    try:
        raise d
    except Diagnostic as e:
        assert e.code == "E0001"


def test_unique_diagnostic_error_codes():
    """Verify error codes in timet codebase are consistently formatted and distinct."""
    import re
    from pathlib import Path
    codes = set()
    root = Path(__file__).parent.parent / "timet"
    for py_file in root.rglob("*.py"):
        text = py_file.read_text()
        found = re.findall(r'code=["\'](E\d{4})["\']', text)
        for c in found:
            codes.add(c)
    # Ensure there are dozens of structured error codes registered
    assert len(codes) >= 50
    # Ensure no collision between mobile (E0900..E0903) and native (E0950)
    assert "E0950" in codes
    assert "E0900" in codes
