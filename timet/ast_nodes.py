"""Time-T AST node definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class Node:
    line: int = field(default=0, kw_only=True)
    col: int = field(default=0, kw_only=True)
    ty: Any = field(default=None, kw_only=True, repr=False, compare=False)


# ---------- Type annotations (surface syntax, pre-typechecking) ----------

@dataclass
class TypeExpr(Node):
    name: str
    args: List["TypeExpr"] = field(default_factory=list)


# ---------- Expressions ----------

@dataclass
class Expr(Node):
    pass


@dataclass
class IntLit(Expr):
    value: int


@dataclass
class FloatLit(Expr):
    value: float


@dataclass
class BoolLit(Expr):
    value: bool


@dataclass
class StringLit(Expr):
    value: str


@dataclass
class UnitLit(Expr):
    pass


@dataclass
class Ident(Expr):
    name: str


@dataclass
class ListExpr(Expr):
    elements: List[Expr]


@dataclass
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class UnaryOp(Expr):
    op: str
    operand: Expr


@dataclass
class Call(Expr):
    callee: Expr
    args: List[Expr]
    kwargs: dict = field(default_factory=dict)


@dataclass
class MethodCall(Expr):
    receiver: Expr
    method: str
    args: List[Expr]
    kwargs: dict = field(default_factory=dict)


@dataclass
class FieldAccess(Expr):
    receiver: Expr
    field: str


@dataclass
class Index(Expr):
    receiver: Expr
    index: Expr


@dataclass
class IfExpr(Expr):
    cond: Expr
    then_branch: "Block"
    else_branch: Optional["Block"]


@dataclass
class Lambda(Expr):
    params: List["Param"]
    ret_type: Optional[TypeExpr]
    body: "Block"


# ---------- Statements ----------

@dataclass
class Stmt(Node):
    pass


@dataclass
class Param:
    name: str
    type_expr: Optional[TypeExpr]
    line: int = 0
    col: int = 0


@dataclass
class LetStmt(Stmt):
    name: str
    type_expr: Optional[TypeExpr]
    value: Expr
    mutable: bool


@dataclass
class AssignStmt(Stmt):
    target: Expr
    value: Expr


@dataclass
class ExprStmt(Stmt):
    expr: Expr


@dataclass
class ReturnStmt(Stmt):
    value: Optional[Expr]


@dataclass
class WhileStmt(Stmt):
    cond: Expr
    body: "Block"


@dataclass
class ForStmt(Stmt):
    var_name: str
    iterable: Expr
    body: "Block"


@dataclass
class IfStmt(Stmt):
    cond: Expr
    then_branch: "Block"
    else_branch: Optional["Block"]


@dataclass
class NoGradStmt(Stmt):
    body: "Block"


@dataclass
class FnDecl(Stmt):
    name: str
    params: List[Param]
    ret_type: Optional[TypeExpr]
    body: "Block"


@dataclass
class ImportStmt(Stmt):
    """`import a.b.c` (dotted path -> `<dir>/a/b/c.tt`) with optional
    `as alias`. Bound name is `alias` or the last path component.
    Top-level only (enforced by the type checker, E0212)."""
    path: List[str]
    alias: Optional[str]

    @property
    def binding(self) -> str:
        return self.alias or self.path[-1]


@dataclass
class Block(Node):
    statements: List[Stmt]


@dataclass
class Program(Node):
    statements: List[Stmt]
