"""Time-T recursive-descent / Pratt parser."""
from __future__ import annotations

from typing import List, Optional

from timet.lexer import Token, TokenKind, tokenize
from timet.diagnostics import Diagnostic, SourceSpan
import timet.ast_nodes as A


class ParseError(Diagnostic):
    pass


# Binding power table for binary operators (Pratt parsing).
PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "==": 3, "!=": 3,
    "<": 4, "<=": 4, ">": 4, ">=": 4,
    "+": 5, "-": 5,
    "*": 6, "/": 6, "%": 6,
    "@": 6,
}
RIGHT_ASSOC = set()


class Parser:
    def __init__(self, tokens: List[Token], filename: str = "<input>"):
        self.tokens = tokens
        self.pos = 0
        self.filename = filename

    def _peek(self, offset: int = 0) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def _at_end(self) -> bool:
        return self._peek().kind == TokenKind.EOF

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return tok

    def _check(self, kind: TokenKind, text: Optional[str] = None) -> bool:
        tok = self._peek()
        if tok.kind != kind:
            return False
        if text is not None and tok.text != text:
            return False
        return True

    def _match(self, kind: TokenKind, text: Optional[str] = None) -> Optional[Token]:
        if self._check(kind, text):
            return self._advance()
        return None

    def _expect(self, kind: TokenKind, text: Optional[str] = None, what: str = "") -> Token:
        if self._check(kind, text):
            return self._advance()
        tok = self._peek()
        expected = text if text else kind.name
        raise ParseError(
            code="E0100",
            message=f"unexpected token while parsing {what or 'expression'}",
            span=SourceSpan(tok.line, tok.col),
            expected=expected,
            actual=f"{tok.kind.name} {tok.text!r}",
            stage="parser",
        )

    # ---------------- top level ----------------

    def parse_program(self) -> A.Program:
        stmts = []
        while not self._at_end():
            stmts.append(self.parse_stmt())
        return A.Program(stmts, line=1, col=1)

    def parse_stmt(self) -> A.Stmt:
        tok = self._peek()
        if tok.kind == TokenKind.KEYWORD:
            if tok.text == "let":
                return self._parse_let(mutable=False)
            if tok.text == "var":
                return self._parse_let(mutable=True)
            if tok.text == "fn":
                return self._parse_fn()
            if tok.text == "return":
                return self._parse_return()
            if tok.text == "if":
                return self._parse_if_stmt()
            if tok.text == "break" or tok.text == "continue":
                self._advance()
                cls = A.BreakStmt if tok.text == "break" else A.ContinueStmt
                return cls(line=tok.line, col=tok.col)
            if tok.text == "while":
                return self._parse_while()
            if tok.text == "for":
                return self._parse_for()
            if tok.text == "no_grad":
                return self._parse_no_grad()
            if tok.text == "import":
                return self._parse_import()
            if tok.text in ("struct", "enum", "match"):
                raise ParseError(
                    code="E0101",
                    message=f"'{tok.text}' is not implemented yet in this version of Time-T",
                    span=SourceSpan(tok.line, tok.col),
                    stage="parser",
                    note="see docs/ROADMAP.md for the feature sequencing",
                )
        return self._parse_expr_or_assign_stmt()

    def _parse_import(self) -> A.ImportStmt:
        kw = self._advance()  # 'import'
        parts = [self._expect(TokenKind.IDENT, what="module name").text]
        while self._match(TokenKind.OP, "."):
            parts.append(self._expect(TokenKind.IDENT, what="module name").text)
        alias = None
        if self._check(TokenKind.IDENT) and self._peek().text == "as":
            self._advance()  # 'as'
            alias = self._expect(TokenKind.IDENT, what="import alias").text
        return A.ImportStmt(parts, alias, line=kw.line, col=kw.col)

    def _parse_block(self) -> A.Block:
        tok = self._expect(TokenKind.LBRACE, what="block")
        stmts = []
        while not self._check(TokenKind.RBRACE):
            if self._at_end():
                raise ParseError(
                    code="E0102",
                    message="unterminated block, expected '}'",
                    span=SourceSpan(tok.line, tok.col),
                    stage="parser",
                )
            stmts.append(self.parse_stmt())
        self._expect(TokenKind.RBRACE, what="block")
        return A.Block(stmts, line=tok.line, col=tok.col)

    def _parse_type(self) -> A.TypeExpr:
        if self._check(TokenKind.LPAREN):
            lp = self._advance()
            param_types = []
            if not self._check(TokenKind.RPAREN):
                param_types.append(self._parse_type())
                while self._match(TokenKind.COMMA):
                    param_types.append(self._parse_type())
            self._expect(TokenKind.RPAREN, what="function type parameter list")
            self._expect(TokenKind.ARROW, what="function type")
            ret = self._parse_type()
            return A.TypeExpr("Function", param_types + [ret], line=lp.line, col=lp.col)
        tok = self._expect(TokenKind.IDENT, what="type")
        args = []
        if self._match(TokenKind.LBRACKET):
            args.append(self._parse_type())
            while self._match(TokenKind.COMMA):
                args.append(self._parse_type())
            self._expect(TokenKind.RBRACKET, what="type arguments")
        return A.TypeExpr(tok.text, args, line=tok.line, col=tok.col)

    def _parse_let(self, mutable: bool) -> A.LetStmt:
        kw = self._advance()
        name_tok = self._expect(TokenKind.IDENT, what="binding name")
        type_expr = None
        if self._match(TokenKind.COLON):
            type_expr = self._parse_type()
        self._expect(TokenKind.OP, "=", what="'let'/'var' initializer")
        value = self.parse_expr()
        return A.LetStmt(name_tok.text, type_expr, value, mutable, line=kw.line, col=kw.col)

    def _parse_params(self) -> List[A.Param]:
        params = []
        self._expect(TokenKind.LPAREN, what="parameter list")
        if not self._check(TokenKind.RPAREN):
            params.append(self._parse_param())
            while self._match(TokenKind.COMMA):
                params.append(self._parse_param())
        self._expect(TokenKind.RPAREN, what="parameter list")
        return params

    def _parse_param(self) -> A.Param:
        name_tok = self._expect(TokenKind.IDENT, what="parameter")
        type_expr = None
        if self._match(TokenKind.COLON):
            type_expr = self._parse_type()
        return A.Param(name_tok.text, type_expr, name_tok.line, name_tok.col)

    def _parse_fn(self) -> A.FnDecl:
        kw = self._advance()
        name_tok = self._expect(TokenKind.IDENT, what="function name")
        params = self._parse_params()
        ret_type = None
        if self._match(TokenKind.ARROW):
            ret_type = self._parse_type()
        body = self._parse_block()
        return A.FnDecl(name_tok.text, params, ret_type, body, line=kw.line, col=kw.col)

    def _parse_return(self) -> A.ReturnStmt:
        kw = self._advance()
        value = None
        if not self._check(TokenKind.RBRACE):
            value = self.parse_expr()
        return A.ReturnStmt(value, line=kw.line, col=kw.col)

    def _parse_if_stmt(self) -> A.IfStmt:
        kw = self._advance()
        cond = self.parse_expr()
        then_b = self._parse_block()
        else_b = None
        if self._match(TokenKind.KEYWORD, "else"):
            if self._check(TokenKind.KEYWORD, "if"):
                nested = self._parse_if_stmt()
                else_b = A.Block([nested], line=nested.line, col=nested.col)
            else:
                else_b = self._parse_block()
        return A.IfStmt(cond, then_b, else_b, line=kw.line, col=kw.col)

    def _parse_while(self) -> A.WhileStmt:
        kw = self._advance()
        cond = self.parse_expr()
        body = self._parse_block()
        return A.WhileStmt(cond, body, line=kw.line, col=kw.col)

    def _parse_for(self) -> A.ForStmt:
        kw = self._advance()
        name_tok = self._expect(TokenKind.IDENT, what="for-loop variable")
        self._expect(TokenKind.KEYWORD, "in", what="for-loop")
        iterable = self.parse_expr()
        body = self._parse_block()
        return A.ForStmt(name_tok.text, iterable, body, line=kw.line, col=kw.col)

    def _parse_no_grad(self) -> A.NoGradStmt:
        kw = self._advance()
        body = self._parse_block()
        return A.NoGradStmt(body, line=kw.line, col=kw.col)

    def _parse_expr_or_assign_stmt(self) -> A.Stmt:
        start = self._peek()
        expr = self.parse_expr()
        if self._check(TokenKind.OP, "=") and isinstance(expr, (A.Ident, A.Index)):
            self._advance()
            value = self.parse_expr()
            return A.AssignStmt(expr, value, line=start.line, col=start.col)
        return A.ExprStmt(expr, line=start.line, col=start.col)

    # ---------------- expressions (Pratt) ----------------

    def parse_expr(self, min_bp: int = 0) -> A.Expr:
        left = self._parse_unary()
        while True:
            tok = self._peek()
            if tok.kind != TokenKind.OP or tok.text not in PRECEDENCE:
                break
            bp = PRECEDENCE[tok.text]
            if bp < min_bp:
                break
            op_tok = self._advance()
            next_min = bp + 1 if op_tok.text not in RIGHT_ASSOC else bp
            right = self.parse_expr(next_min)
            left = A.BinOp(op_tok.text, left, right, line=op_tok.line, col=op_tok.col)
        return left

    def _parse_unary(self) -> A.Expr:
        tok = self._peek()
        if tok.kind == TokenKind.OP and tok.text in ("-", "!"):
            self._advance()
            operand = self._parse_unary()
            return A.UnaryOp(tok.text, operand, line=tok.line, col=tok.col)
        return self._parse_postfix()

    def _parse_postfix(self) -> A.Expr:
        expr = self._parse_primary()
        while True:
            if self._check(TokenKind.OP, "."):
                self._advance()
                name_tok = self._expect(TokenKind.IDENT, what="method name")
                if self._check(TokenKind.LPAREN):
                    args, kwargs = self._parse_call_args()
                    expr = A.MethodCall(expr, name_tok.text, args, kwargs,
                                         line=name_tok.line, col=name_tok.col)
                else:
                    expr = A.FieldAccess(expr, name_tok.text,
                                          line=name_tok.line, col=name_tok.col)
            elif self._check(TokenKind.LPAREN):
                lp = self._peek()
                args, kwargs = self._parse_call_args()
                expr = A.Call(expr, args, kwargs, line=lp.line, col=lp.col)
            elif self._check(TokenKind.LBRACKET):
                lb = self._advance()
                idx = self.parse_expr()
                self._expect(TokenKind.RBRACKET, what="index expression")
                expr = A.Index(expr, idx, line=lb.line, col=lb.col)
            else:
                break
        return expr

    def _parse_call_args(self):
        self._expect(TokenKind.LPAREN, what="call arguments")
        args = []
        kwargs = {}
        if not self._check(TokenKind.RPAREN):
            self._parse_one_arg(args, kwargs)
            while self._match(TokenKind.COMMA):
                self._parse_one_arg(args, kwargs)
        self._expect(TokenKind.RPAREN, what="call arguments")
        return args, kwargs

    def _parse_one_arg(self, args, kwargs):
        if self._check(TokenKind.IDENT) and self._peek(1).kind == TokenKind.OP and self._peek(1).text == "=":
            name_tok = self._advance()
            self._advance()  # '='
            kwargs[name_tok.text] = self.parse_expr()
        else:
            args.append(self.parse_expr())

    def _parse_primary(self) -> A.Expr:
        tok = self._peek()
        if tok.kind == TokenKind.INT:
            self._advance()
            return A.IntLit(tok.value, line=tok.line, col=tok.col)
        if tok.kind == TokenKind.FLOAT:
            self._advance()
            return A.FloatLit(tok.value, line=tok.line, col=tok.col)
        if tok.kind == TokenKind.BOOL:
            self._advance()
            return A.BoolLit(tok.value, line=tok.line, col=tok.col)
        if tok.kind == TokenKind.STRING:
            self._advance()
            return A.StringLit(tok.value, line=tok.line, col=tok.col)
        if tok.kind == TokenKind.IDENT:
            self._advance()
            return A.Ident(tok.text, line=tok.line, col=tok.col)
        if tok.kind == TokenKind.KEYWORD and tok.text == "if":
            return self._parse_if_expr()
        if tok.kind == TokenKind.KEYWORD and tok.text == "fn":
            return self._parse_lambda()
        if tok.kind == TokenKind.LBRACKET:
            self._advance()
            elements = []
            if not self._check(TokenKind.RBRACKET):
                elements.append(self.parse_expr())
                while self._match(TokenKind.COMMA):
                    if self._check(TokenKind.RBRACKET):
                        break
                    elements.append(self.parse_expr())
            self._expect(TokenKind.RBRACKET, what="list literal")
            return A.ListExpr(elements, line=tok.line, col=tok.col)
        if tok.kind == TokenKind.LPAREN:
            self._advance()
            if self._check(TokenKind.RPAREN):
                self._advance()
                return A.UnitLit(line=tok.line, col=tok.col)
            expr = self.parse_expr()
            self._expect(TokenKind.RPAREN, what="parenthesized expression")
            return expr
        raise ParseError(
            code="E0103",
            message="expected an expression",
            span=SourceSpan(tok.line, tok.col),
            actual=f"{tok.kind.name} {tok.text!r}",
            stage="parser",
        )

    def _parse_if_expr(self) -> A.IfExpr:
        kw = self._advance()
        cond = self.parse_expr()
        then_b = self._parse_block()
        self._expect(TokenKind.KEYWORD, "else", what="if-expression (both branches required)")
        else_b = self._parse_block()
        return A.IfExpr(cond, then_b, else_b, line=kw.line, col=kw.col)

    def _parse_lambda(self) -> A.Lambda:
        kw = self._advance()
        params = self._parse_params()
        ret_type = None
        if self._match(TokenKind.ARROW):
            ret_type = self._parse_type()
        body = self._parse_block()
        return A.Lambda(params, ret_type, body, line=kw.line, col=kw.col)


def parse(source: str, filename: str = "<input>") -> A.Program:
    tokens = tokenize(source, filename)
    return Parser(tokens, filename).parse_program()
