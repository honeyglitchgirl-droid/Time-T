"""Module loading for Time-T (Milestone 2 remainder; DD-13).

Scope of v1 (documented honestly in docs/LANGUAGE.md):
  - `import a.b.c` resolves to the FILE `<importer's dir>/a/b/c.tt`; the
    bound name is `c` (or the `as` alias). No search paths, no packages
    with `__init__`, no `from x import y` (yet).
  - Top-level imports only (type checker error E0212 otherwise).
  - All top-level `fn` declarations and `let`/`var` bindings are exported;
    there are no visibility modifiers.
  - A module file is loaded, type-checked, and executed ONCE per canonical
    path (Python-style); import cycles raise E0210 naming the cycle.
  - A module runs with a FRESH global scope (engine builtins only), never
    seeing the importer's variables; a `fn main()` inside a module is NOT
    auto-invoked (documented).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import timet.ast_nodes as A
from timet.diagnostics import Diagnostic, SourceSpan


class ModuleError(Diagnostic):
    pass


class LoadedModule:
    def __init__(self, dotted: str, path: Path, program: A.Program):
        self.dotted = dotted
        self.path = path
        self.program = program
        self.type_exports: Dict[str, "object"] = {}  # filled by the type checker
        self.checked_types = False   # type check completed
        self.checking_types = False  # type check in progress (cycle guard)
        self.value = None            # ModuleValue, filled by the interpreter
        self.initializing = False    # interpreter exec in progress (cycle guard)


class ModuleLoader:
    """Shared AST cache + resolution. One loader per program run so the
    type checker, the interpreter, and the IR lowerer agree on identity."""

    def __init__(self):
        self._ast_cache: Dict[str, LoadedModule] = {}
        self._loading: List[str] = []  # canonical paths currently resolving

    # -- resolution --

    def resolve_path(self, dotted_parts: List[str], importer_path: str,
                     span: Optional[SourceSpan] = None) -> Path:
        base = Path(importer_path).resolve().parent if not importer_path.startswith("<") \
            else Path.cwd()
        rel = Path(*dotted_parts).with_suffix(".tt")
        path = (base / rel).resolve()
        if not path.is_file():
            raise ModuleError(
                code="E0213",
                message=f"module '{'.'.join(dotted_parts)}' not found",
                span=span,
                stage="modules",
                note=f"looked for {path} (resolved relative to {base})",
            )
        return path

    def load(self, dotted_parts: List[str], importer_path: str,
             span: Optional[SourceSpan] = None) -> LoadedModule:
        """Parse (once) and return the module, checking for import cycles."""
        from timet.parser import parse  # local import: modules <-> parser cycle

        path = self.resolve_path(dotted_parts, importer_path, span)
        key = str(path)
        if key in self._ast_cache:
            return self._ast_cache[key]
        if key in self._loading:
            cycle = " -> ".join(
                Path(p).name for p in self._loading[self._loading.index(key):] + [key])
            raise ModuleError(
                code="E0210",
                message=f"import cycle detected: {cycle}",
                span=span,
                stage="modules",
                note="restructure the modules so the dependency graph is acyclic",
            )
        self._loading.append(key)
        try:
            source = path.read_text()
            program = parse(source, str(path))
        finally:
            self._loading.remove(key)
        dotted = ".".join(dotted_parts)
        mod = LoadedModule(dotted, path, program)
        self._ast_cache[key] = mod
        return mod


class ModuleValue:
    """Runtime value of an imported module: attribute reads resolve
    against the module's own top-level environment."""

    def __init__(self, dotted: str, env_vars: Dict[str, object]):
        self._dotted = dotted
        self._exports = env_vars

    @property
    def exports(self) -> Dict[str, object]:
        return self._exports

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._exports[name]
        except KeyError:
            raise AttributeError(
                f"module '{self._dotted}' has no member '{name}'") from None

    def __repr__(self):
        return f"<module {self._dotted} ({len(self._exports)} export(s))>"
