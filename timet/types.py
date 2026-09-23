"""Time-T static type system (implemented subset).

See docs/LANGUAGE.md section 2 and docs/ARCHITECTURE.md section 3 for what is
and is not covered. Shape typing is intentionally not part of the static
type (checked dynamically instead) per master prompt section 7.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class Type:
    def __repr__(self) -> str:  # pragma: no cover
        return str(self)


@dataclass(frozen=True)
class TInt(Type):
    def __str__(self):
        return "Int"


@dataclass(frozen=True)
class TFloat(Type):
    def __str__(self):
        return "Float"


@dataclass(frozen=True)
class TBool(Type):
    def __str__(self):
        return "Bool"


@dataclass(frozen=True)
class TString(Type):
    def __str__(self):
        return "String"


@dataclass(frozen=True)
class TUnit(Type):
    def __str__(self):
        return "Unit"


VALID_DTYPES = ("f32", "f64", "i32", "i64", "bool")


@dataclass(frozen=True)
class TTensor(Type):
    dtype: str = "f32"

    def __post_init__(self):
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"invalid tensor dtype {self.dtype!r}")

    def __str__(self):
        return f"Tensor[{self.dtype}]"


@dataclass(frozen=True)
class TFunction(Type):
    params: Tuple[Type, ...]
    ret: Type

    def __str__(self):
        param_str = ", ".join(str(p) for p in self.params)
        return f"({param_str}) -> {self.ret}"


@dataclass(frozen=True)
class TUnknown(Type):
    """Used internally during error recovery; never a valid final type."""
    def __str__(self):
        return "<unknown>"


PRIMITIVE_NAMES = {
    "Int": TInt(),
    "Float": TFloat(),
    "Bool": TBool(),
    "String": TString(),
    "Unit": TUnit(),
}


def tensor_type(dtype: str = "f32") -> TTensor:
    return TTensor(dtype)


def is_numeric(t: Type) -> bool:
    return isinstance(t, (TInt, TFloat))


def types_equal(a: Type, b: Type) -> bool:
    return a == b
