"""#175: product code must not import the benchmark harness.

DAG encoded here:

* ``socr.benchmark`` MAY import ``socr.core`` and ``socr.tables``.
* ``socr.tables`` and ``socr.core`` MUST NOT import ``socr.benchmark``.
* A private (``_``-prefixed) name MUST NOT be imported across first-level
  ``socr.*`` packages.

Hermetic: stdlib ``ast`` walk of ``src/socr``. No provider, no pipeline run.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"

# Pre-existing private cross-package imports this ticket does not own.
# Do not add entries. Remove one only when its owning ticket lands.
_ALLOWED_PRIVATE_IMPORTS = {
    ("benchmark/runner.py", "socr.engines.registry", "_ENGINES"),
    ("math/detect_equations.py", "socr.core.born_digital", "_MATH_FONT_RE"),
    ("pipeline/agentic.py", "socr.tables.locate", "_horizontal_rules"),
    ("pipeline/orchestrator.py", "socr.core.manifest", "_whole_doc_page_texts"),
    ("pipeline/orchestrator.py", "socr.core.manifest", "_winning_page_output"),
    ("pipeline/repair.py", "socr.engines.registry", "_ENGINES"),
}


def _first_level(mod: str) -> str | None:
    """``socr.tables.native_rows`` -> ``tables``; non-socr or ``socr`` itself -> None."""
    parts = mod.split(".")
    if parts[0] != "socr" or len(parts) < 2:
        return None
    return parts[1]


def _file_package(rel: str) -> str | None:
    """``tables/native_rows.py`` -> ``tables``; ``cli.py`` -> None."""
    top = rel.split("/", 1)[0]
    return None if top.endswith(".py") else top


def _iter_import_from(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
            continue
        yield node


def _iter_imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, ["<module>"]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            yield node.lineno, node.module, [a.name for a in node.names]


def test_tables_and_core_do_not_import_benchmark() -> None:
    offenders: list[str] = []
    for pkg in ("tables", "core"):
        for path in (SRC / pkg).rglob("*.py"):
            rel = path.relative_to(SRC).as_posix()
            for lineno, mod, names in _iter_imports(path):
                if mod == "socr.benchmark" or mod.startswith("socr.benchmark."):
                    offenders.append(f"{rel}:{lineno} imports {mod}")
                if mod == "socr" and "benchmark" in names:
                    offenders.append(f"{rel}:{lineno} imports socr.benchmark")
    assert not offenders, (
        "socr.tables / socr.core must not import socr.benchmark:\n  " + "\n  ".join(offenders)
    )


def test_no_private_symbol_imported_across_packages() -> None:
    offenders: list[str] = []
    seen_allowed: set[tuple[str, str, str]] = set()
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        src_pkg = _file_package(rel)
        for node in _iter_import_from(path):
            dest_pkg = _first_level(node.module)
            if dest_pkg is None or dest_pkg == src_pkg:
                continue
            for alias in node.names:
                if not alias.name.startswith("_") or alias.name == "_":
                    continue
                key = (rel, node.module, alias.name)
                if key in _ALLOWED_PRIVATE_IMPORTS:
                    seen_allowed.add(key)
                    continue
                offenders.append(f"{rel}:{node.lineno} imports {node.module}.{alias.name}")
    assert not offenders, (
        "private names must not be imported across socr.* packages:\n  " + "\n  ".join(offenders)
    )
    missing = _ALLOWED_PRIVATE_IMPORTS - seen_allowed
    assert not missing, (
        "allowlist entries that no longer match source (stale, would hide a "
        "new violation under an old name):\n  " + "\n  ".join(map(str, sorted(missing)))
    )


def test_table_grid_public_api() -> None:
    """The shared module exists and exports the names both sides import."""
    from socr.core import table_grid

    for name in (
        "NUM_TOKEN_RE",
        "NUMERIC_RE",
        "NUMERIC_CELL_RE",
        "is_numeric_cell",
        "normalize_cell",
        "markdown_table_cells",
        "rows_establish_grid",
        "ExactnessReport",
        "score_page",
        "score_rows",
        "markdown_rows",
    ):
        assert hasattr(table_grid, name), name
