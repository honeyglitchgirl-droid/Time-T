"""Structured diagnostics for Time-T.

Every error Time-T produces should explain what happened, where, why, what
was expected, what was found, and (when possible) a suggestion -- per the
master prompt section 27. This module gives all compiler stages a single,
machine-readable diagnostic type instead of ad-hoc strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceSpan:
    line: int
    col: int
    end_line: Optional[int] = None
    end_col: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.col}"


@dataclass
class Diagnostic(Exception):
    code: str
    message: str
    span: Optional[SourceSpan] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    note: Optional[str] = None
    severity: str = "error"
    stage: str = "unknown"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        Exception.__init__(self, self.human())

    def human(self) -> str:
        loc = f" at {self.span}" if self.span else ""
        lines = [f"{self.severity.upper()} [{self.code}] ({self.stage}){loc}: {self.message}"]
        if self.expected is not None:
            lines.append(f"  expected: {self.expected}")
        if self.actual is not None:
            lines.append(f"  actual:   {self.actual}")
        if self.note is not None:
            lines.append(f"  note:     {self.note}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "stage": self.stage,
            "span": None if self.span is None else {
                "line": self.span.line,
                "col": self.span.col,
                "end_line": self.span.end_line,
                "end_col": self.span.end_col,
            },
            "expected": self.expected,
            "actual": self.actual,
            "note": self.note,
            "extra": self.extra,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.human()
