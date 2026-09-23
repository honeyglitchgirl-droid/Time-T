"""Time-T lexer.

Hand-written, single-pass tokenizer. Tracks line/column for every token so
downstream diagnostics can point at exact source locations (master prompt
section 27).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from timet.diagnostics import Diagnostic, SourceSpan


class LexError(Diagnostic):
    pass


class TokenKind(Enum):
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    BOOL = auto()
    IDENT = auto()
    KEYWORD = auto()
    OP = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    COLON = auto()
    ARROW = auto()
    SEMI = auto()
    NEWLINE = auto()
    EOF = auto()


KEYWORDS = {
    "let", "var", "fn", "return", "if", "else", "while", "true", "false",
    "break", "continue",
    "no_grad", "struct", "enum", "match", "import", "for", "in",
}

# Multi-char operators must be listed before their single-char prefixes.
OPERATORS = [
    "->", "==", "!=", "<=", ">=", "&&", "||",
    "+", "-", "*", "/", "%", "=", "<", ">", "!", "@", ".",
]


@dataclass
class Token:
    kind: TokenKind
    text: str
    line: int
    col: int
    value: object = None

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Token({self.kind.name}, {self.text!r}, {self.line}:{self.col})"


_SINGLE_CHAR = {
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    ",": TokenKind.COMMA,
    ":": TokenKind.COLON,
    ";": TokenKind.SEMI,
}


class Lexer:
    def __init__(self, source: str, filename: str = "<input>"):
        self.src = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1
        self.n = len(source)

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.src[idx] if idx < self.n else ""

    def _advance(self) -> str:
        ch = self.src[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _error(self, message: str, span: Optional[SourceSpan] = None, **kw) -> LexError:
        return LexError(
            code="E0001",
            message=message,
            span=span if span is not None else SourceSpan(self.line, self.col),
            stage="lexer",
            **kw,
        )

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while self.pos < self.n:
            ch = self._peek()

            if ch in " \t\r":
                self._advance()
                continue
            if ch == "\n":
                self._advance()
                continue

            if ch == "/" and self._peek(1) == "/":
                while self.pos < self.n and self._peek() != "\n":
                    self._advance()
                continue

            if ch == "/" and self._peek(1) == "*":
                start_line, start_col = self.line, self.col
                self._advance(); self._advance()
                closed = False
                while self.pos < self.n:
                    if self._peek() == "*" and self._peek(1) == "/":
                        self._advance(); self._advance()
                        closed = True
                        break
                    self._advance()
                if not closed:
                    raise self._error(
                        "unterminated block comment",
                        note=f"comment started at {start_line}:{start_col}",
                    )
                continue

            line, col = self.line, self.col

            if ch.isdigit():
                tokens.append(self._lex_number(line, col))
                continue

            if ch == '"':
                tokens.append(self._lex_string(line, col))
                continue

            if ch.isalpha() or ch == "_":
                tokens.append(self._lex_ident(line, col))
                continue

            matched_op = None
            for op in OPERATORS:
                if self.src.startswith(op, self.pos):
                    matched_op = op
                    break
            if matched_op:
                for _ in matched_op:
                    self._advance()
                kind = TokenKind.ARROW if matched_op == "->" else TokenKind.OP
                tokens.append(Token(kind, matched_op, line, col))
                continue

            if ch in _SINGLE_CHAR:
                self._advance()
                tokens.append(Token(_SINGLE_CHAR[ch], ch, line, col))
                continue

            raise self._error(
                f"unexpected character {ch!r}",
                actual=repr(ch),
                note="Time-T source must be valid UTF-8 using the documented grammar (see docs/LANGUAGE.md)",
            )

        tokens.append(Token(TokenKind.EOF, "", self.line, self.col))
        return tokens

    def _lex_number(self, line: int, col: int) -> Token:
        start = self.pos
        is_float = False
        if self._peek() == "0" and self._peek(1) in ("x", "X"):
            self._advance(); self._advance()
            digits_start = self.pos
            while self.pos < self.n and (self._peek() in "0123456789abcdefABCDEF_"):
                self._advance()
            text = self.src[start:self.pos]
            value = int(text.replace("_", ""), 16)
            return Token(TokenKind.INT, text, line, col, value)
        if self._peek() == "0" and self._peek(1) in ("b", "B"):
            self._advance(); self._advance()
            while self.pos < self.n and (self._peek() in "01_"):
                self._advance()
            text = self.src[start:self.pos]
            value = int(text.replace("_", ""), 2)
            return Token(TokenKind.INT, text, line, col, value)

        while self.pos < self.n and (self._peek().isdigit() or self._peek() == "_"):
            self._advance()
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self._advance()
            while self.pos < self.n and (self._peek().isdigit() or self._peek() == "_"):
                self._advance()
        if self._peek() in ("e", "E"):
            look = 1
            if self._peek(1) in ("+", "-"):
                look = 2
            if self._peek(look).isdigit():
                is_float = True
                self._advance()
                if self._peek() in ("+", "-"):
                    self._advance()
                while self.pos < self.n and self._peek().isdigit():
                    self._advance()

        text = self.src[start:self.pos]
        cleaned = text.replace("_", "")
        if is_float:
            return Token(TokenKind.FLOAT, text, line, col, float(cleaned))
        return Token(TokenKind.INT, text, line, col, int(cleaned))

    def _lex_string(self, line: int, col: int) -> Token:
        self._advance()  # opening quote
        out_chars = []
        while True:
            if self.pos >= self.n:
                raise self._error(
                    "unterminated string literal",
                    span=SourceSpan(line, col),
                    note="expected closing \" before end of file",
                )
            ch = self._peek()
            if ch == '"':
                self._advance()
                break
            if ch == "\\":
                self._advance()
                esc = self._peek()
                mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "r": "\r", "0": "\0"}
                if esc not in mapping:
                    raise self._error(f"unknown escape sequence '\\{esc}'")
                out_chars.append(mapping[esc])
                self._advance()
                continue
            if ch == "\n":
                raise self._error("string literal cannot contain a raw newline")
            out_chars.append(ch)
            self._advance()
        text = "".join(out_chars)
        return Token(TokenKind.STRING, text, line, col, text)

    def _lex_ident(self, line: int, col: int) -> Token:
        start = self.pos
        while self.pos < self.n and (self._peek().isalnum() or self._peek() == "_"):
            self._advance()
        text = self.src[start:self.pos]
        if text in ("true", "false"):
            return Token(TokenKind.BOOL, text, line, col, text == "true")
        if text in KEYWORDS:
            return Token(TokenKind.KEYWORD, text, line, col)
        return Token(TokenKind.IDENT, text, line, col)


def tokenize(source: str, filename: str = "<input>") -> List[Token]:
    return Lexer(source, filename).tokenize()
