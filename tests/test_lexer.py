import pytest
from timet.lexer import tokenize, TokenKind, LexError


def kinds(tokens):
    return [t.kind for t in tokens if t.kind != TokenKind.EOF]


def texts(tokens):
    return [t.text for t in tokens if t.kind != TokenKind.EOF]


def test_integers():
    toks = tokenize("1 23 0x1F 0b1010 1_000")
    assert [t.value for t in toks[:-1]] == [1, 23, 31, 10, 1000]


def test_floats():
    toks = tokenize("1.5 2.0e3 1_000.5")
    values = [t.value for t in toks[:-1]]
    assert values == [1.5, 2000.0, 1000.5]


def test_bools_and_keywords():
    toks = tokenize("true false let var fn")
    assert kinds(toks) == [TokenKind.BOOL, TokenKind.BOOL, TokenKind.KEYWORD,
                            TokenKind.KEYWORD, TokenKind.KEYWORD]


def test_strings_with_escapes():
    toks = tokenize(r'"hello\nworld"')
    assert toks[0].value == "hello\nworld"


def test_unterminated_string_raises():
    with pytest.raises(LexError) as exc:
        tokenize('"unterminated')
    assert exc.value.code == "E0001"


def test_unterminated_block_comment_raises():
    with pytest.raises(LexError):
        tokenize("/* never closed")


def test_line_col_tracking():
    toks = tokenize("let x\n= 5")
    # '=' is on line 2
    eq_tok = [t for t in toks if t.text == "="][0]
    assert eq_tok.line == 2
    assert eq_tok.col == 1


def test_comments_are_skipped():
    toks = tokenize("// a comment\nlet x = 1 /* inline */ + 2")
    assert texts(toks) == ["let", "x", "=", "1", "+", "2"]


def test_operators_multichar_before_singlechar():
    toks = tokenize("-> == != <= >= && ||")
    assert texts(toks) == ["->", "==", "!=", "<=", ">=", "&&", "||"]


def test_unexpected_char_raises():
    with pytest.raises(LexError) as exc:
        tokenize("let x = 1 $ 2")
    assert exc.value.code == "E0001"
    assert exc.value.span is not None


def test_matmul_operator():
    toks = tokenize("a @ b")
    assert texts(toks) == ["a", "@", "b"]
