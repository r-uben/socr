"""#175: product code must not import the benchmark harness.

DAG encoded here:

* ``socr.benchmark`` MAY import ``socr.core`` and ``socr.tables``.
* ``socr.tables`` and ``socr.core`` MUST NOT import ``socr.benchmark``.
* NOTHING under ``socr`` except ``socr.devtools`` itself may import
  ``socr.devtools``: it is a developer-only regeneration tool that ships in the
  package only so ``[project.scripts]`` can name an entry point, and no runtime
  path may come to depend on it.
* A private (``_``-prefixed) name MUST NOT be imported across first-level
  ``socr.*`` packages.

Hermetic: stdlib ``ast`` walk of ``src/socr``. No provider, no pipeline run.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"

# Pre-existing private cross-package imports this ticket does not own.
# Do not add entries. Remove one only when its owning ticket lands.
_ALLOWED_PRIVATE_IMPORTS = {
    ("benchmark/runner.py", "socr.engines.registry", "_ENGINES"),
    # GH-330. The coverage harness exists to measure the native extraction path, so it
    # must enumerate regions the SAME way `extract_structured` does -- including the
    # lane-stacked branch. Re-implementing the predicate publicly would let the
    # instrument drift from the thing it measures, which is exactly the failure this
    # harness was built to rule out.
    ("benchmark/binding_coverage.py", "socr.core.born_digital", "_is_lane_stacked"),
    ("math/detect_equations.py", "socr.core.born_digital", "_MATH_FONT_RE"),
    ("pipeline/agentic.py", "socr.tables.locate", "_horizontal_rules"),
    ("pipeline/orchestrator.py", "socr.core.manifest", "_whole_doc_page_texts"),
}

_MODULE_IMPORT = "<module>"


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


def _module_name(rel: str) -> str:
    """``tables/native_rows.py`` -> ``socr.tables.native_rows``."""
    body = rel[:-3] if rel.endswith(".py") else rel
    if body == "__init__":
        return "socr"
    if body.endswith("/__init__"):
        body = body[: -len("/__init__")]
    return "socr." + body.replace("/", ".")


def _containing_package(rel: str) -> str:
    """Package relative imports resolve against (PEP 328)."""
    mod = _module_name(rel)
    if rel.endswith("/__init__.py") or rel == "__init__.py":
        return mod
    if "." not in mod:
        return ""
    return mod.rsplit(".", 1)[0]


def _resolve_from(module: str | None, level: int, package: str) -> str | None:
    """Absolute module for an ImportFrom, resolving leading dots against *package*."""
    if level == 0:
        return module
    if not package:
        return None
    relative = "." * level + (module or "")
    try:
        return importlib.util.resolve_name(relative, package)
    except ImportError:
        return None


def _dynamic_module(node: ast.Call) -> str | None:
    """Module string from ``import_module`` / ``__import__``, or None if not a literal."""
    func = node.func
    is_dynamic = (isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}) or (
        isinstance(func, ast.Attribute) and func.attr == "import_module"
    )
    if not is_dynamic or not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def _collect_imports(source: str, rel: str) -> list[tuple[int, str, tuple[str, ...]]]:
    """``(lineno, absolute_module, names)`` for static, relative, and literal dynamic imports.

    A whole-module import (``import a.b``, ``import_module("a.b")``) has
    ``names == ("<module>",)``. A non-literal dynamic argument is skipped.
    """
    tree = ast.parse(source)
    package = _containing_package(rel)
    out: list[tuple[int, str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name, (_MODULE_IMPORT,)))
        elif isinstance(node, ast.ImportFrom):
            abs_mod = _resolve_from(node.module, node.level, package)
            if not abs_mod:
                continue
            out.append((node.lineno, abs_mod, tuple(a.name for a in node.names)))
        elif isinstance(node, ast.Call):
            lit = _dynamic_module(node)
            if lit is not None:
                out.append((node.lineno, lit, (_MODULE_IMPORT,)))
    return out


def _targets_benchmark(mod: str, names: tuple[str, ...]) -> bool:
    if mod == "socr.benchmark" or mod.startswith("socr.benchmark."):
        return True
    return mod == "socr" and "benchmark" in names


def _benchmark_offenders(rel: str, imports: list[tuple[int, str, tuple[str, ...]]]) -> list[str]:
    if _file_package(rel) not in {"tables", "core"}:
        return []
    return [
        f"{rel}:{lineno} imports {mod}"
        for lineno, mod, names in imports
        if _targets_benchmark(mod, names)
    ]


def _targets_devtools(mod: str, names: tuple[str, ...]) -> bool:
    if mod == "socr.devtools" or mod.startswith("socr.devtools."):
        return True
    return mod == "socr" and "devtools" in names


def _devtools_offenders(rel: str, imports: list[tuple[int, str, tuple[str, ...]]]) -> list[str]:
    if _file_package(rel) == "devtools":
        return []
    return [
        f"{rel}:{lineno} imports {mod}"
        for lineno, mod, names in imports
        if _targets_devtools(mod, names)
    ]


def _private_name_keys(
    rel: str, imports: list[tuple[int, str, tuple[str, ...]]]
) -> list[tuple[int, str, str]]:
    """``(lineno, module_imported_from, private_name)`` crossing a first-level package."""
    src_pkg = _file_package(rel)
    keys: list[tuple[int, str, str]] = []
    for lineno, mod, names in imports:
        dest_pkg = _first_level(mod)
        if dest_pkg is None or dest_pkg == src_pkg:
            continue
        if names == (_MODULE_IMPORT,):
            parts = mod.split(".")
            last = parts[-1]
            if last.startswith("_") and last != "_":
                keys.append((lineno, ".".join(parts[:-1]), last))
            continue
        for name in names:
            if name.startswith("_") and name != "_":
                keys.append((lineno, mod, name))
    return keys


def test_tables_and_core_do_not_import_benchmark() -> None:
    offenders: list[str] = []
    for pkg in ("tables", "core"):
        for path in (SRC / pkg).rglob("*.py"):
            rel = path.relative_to(SRC).as_posix()
            offenders.extend(_benchmark_offenders(rel, _collect_imports(path.read_text(), rel)))
    assert not offenders, (
        "socr.tables / socr.core must not import socr.benchmark:\n  " + "\n  ".join(offenders)
    )


def test_no_socr_module_imports_devtools() -> None:
    """``socr.devtools`` is a packaged developer tool, not a runtime dependency.

    It exists under ``src/socr`` only so ``[project.scripts]`` can name
    ``socr-regenerate-p6-prechange`` as an entry point (the repo rule: Python
    entry points live in ``pyproject.toml``, never ``python <file>``). It ships
    in the wheel, so without this rule a runtime import could quietly grow and
    put a git-archive-and-subprocess harness on a user's OCR path.
    """
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        offenders.extend(_devtools_offenders(rel, _collect_imports(path.read_text(), rel)))
    assert not offenders, (
        "no socr module outside socr.devtools may import socr.devtools:\n  "
        + "\n  ".join(offenders)
    )


def test_devtools_imports_only_the_standard_library() -> None:
    """The tool drives an archived source tree through a subprocess.

    Importing ``socr`` here would defeat that: the capture must come from the
    ARCHIVED revision's package, never from the running one. The only ``import
    socr`` in the file is inside the runner source string executed by the
    isolated child interpreter, which ``ast`` does not see as an import.
    """
    offenders: list[str] = []
    for path in (SRC / "devtools").rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        for lineno, mod, _names in _collect_imports(path.read_text(), rel):
            if mod == "socr" or mod.startswith("socr."):
                offenders.append(f"{rel}:{lineno} imports {mod}")
    assert not offenders, (
        "socr.devtools must not import the running socr package:\n  " + "\n  ".join(offenders)
    )


def test_no_private_symbol_imported_across_packages() -> None:
    offenders: list[str] = []
    seen_allowed: set[tuple[str, str, str]] = set()
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        for lineno, module, name in _private_name_keys(
            rel, _collect_imports(path.read_text(), rel)
        ):
            key = (rel, module, name)
            if key in _ALLOWED_PRIVATE_IMPORTS:
                seen_allowed.add(key)
                continue
            offenders.append(f"{rel}:{lineno} imports {module}.{name}")
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


# ---------------------------------------------------------------------------
# Evasion proofs: the walker must fail when each hole is present.
# Synthetic source, same collector the live tests use. No files on disk.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel, source",
    [
        ("tables/foo.py", "from ..benchmark import score_page\n"),
        ("tables/foo.py", "from ..benchmark.scorer import BenchmarkScorer\n"),
        ("tables/foo.py", "from .. import benchmark\n"),
        ("core/born_digital.py", "from ..benchmark import score_page\n"),
        ("core/born_digital.py", "from ..benchmark.table_exactness import score_page\n"),
    ],
)
def test_relative_import_of_benchmark_is_caught(rel: str, source: str) -> None:
    hits = _benchmark_offenders(rel, _collect_imports(source, rel))
    assert hits, f"relative import of benchmark slipped through:\n{source}"


@pytest.mark.parametrize(
    "rel, source",
    [
        ("tables/foo.py", "import importlib\nimportlib.import_module('socr.benchmark')\n"),
        ("tables/foo.py", "import importlib\nimportlib.import_module('socr.benchmark.scorer')\n"),
        ("tables/foo.py", "from importlib import import_module\nimport_module('socr.benchmark')\n"),
        ("core/foo.py", "__import__('socr.benchmark')\n"),
        ("core/foo.py", "__import__('socr.benchmark.table_exactness')\n"),
    ],
)
def test_dynamic_import_of_benchmark_is_caught(rel: str, source: str) -> None:
    hits = _benchmark_offenders(rel, _collect_imports(source, rel))
    assert hits, f"dynamic import of benchmark slipped through:\n{source}"


@pytest.mark.parametrize(
    "rel, source",
    [
        ("cli.py", "from socr.devtools import regenerate_p6_prechange\n"),
        ("cli.py", "from .devtools.regenerate_p6_prechange import main\n"),
        ("core/manifest.py", "from .. import devtools\n"),
        ("pipeline/orchestrator.py", "import socr.devtools.regenerate_p6_prechange\n"),
        ("core/foo.py", "__import__('socr.devtools')\n"),
    ],
)
def test_import_of_devtools_is_caught(rel: str, source: str) -> None:
    hits = _devtools_offenders(rel, _collect_imports(source, rel))
    assert hits, f"import of devtools slipped through:\n{source}"


def test_devtools_may_import_itself() -> None:
    """The rule is about crossing INTO devtools, not about its internals."""
    source = "from socr.devtools.regenerate_p6_prechange import normalize\n"
    assert _devtools_offenders("devtools/__init__.py", _collect_imports(source, "devtools")) == []


def test_dynamic_import_nonliteral_is_out_of_scope() -> None:
    source = "import importlib\nname = 'socr.benchmark'\nimportlib.import_module(name)\n"
    hits = _benchmark_offenders("tables/foo.py", _collect_imports(source, "tables/foo.py"))
    assert hits == []


@pytest.mark.parametrize(
    "rel, source, module, name",
    [
        (
            "core/foo.py",
            "import socr.tables.reconstruct._NUM_TOKEN_RE\n",
            "socr.tables.reconstruct",
            "_NUM_TOKEN_RE",
        ),
        (
            "core/foo.py",
            "from ..tables.reconstruct import _NUM_TOKEN_RE\n",
            "socr.tables.reconstruct",
            "_NUM_TOKEN_RE",
        ),
        (
            "benchmark/foo.py",
            "from ..tables.native_rows import _MARKER_RE\n",
            "socr.tables.native_rows",
            "_MARKER_RE",
        ),
    ],
)
def test_private_import_styles_are_caught(rel: str, source: str, module: str, name: str) -> None:
    keys = _private_name_keys(rel, _collect_imports(source, rel))
    assert (module, name) in {(m, n) for _ln, m, n in keys}, (
        f"private import slipped through ({module}.{name}):\n{source}"
    )


def test_math_font_term_has_a_public_core_accessor() -> None:
    """Cold review round 1, finding 7.

    The P4-M trigger measurement must read the math-font term through public
    core API, not a private regex import, so a core refactor cannot silently
    change what the measurement measured. This pins the accessor's existence and
    that it agrees with the private pattern it wraps.
    """
    from socr.core.born_digital import _MATH_FONT_RE, is_math_font, math_font_char_count

    assert callable(math_font_char_count)
    assert is_math_font("ABCDEF+CMMI10") == "CMMI"
    assert is_math_font("Times-Roman") is None
    assert is_math_font("") is None
    for name in ("ABCDEF+CMMI10", "STIXMath", "Times-Roman", "Helvetica"):
        assert bool(is_math_font(name)) is bool(_MATH_FONT_RE.search(name))


def test_the_benchmark_does_not_import_the_private_math_font_regex() -> None:
    """The allowlist entry that blessed this import was reverted; keep it out."""
    src = (SRC / "benchmark" / "trigger_rates.py").read_text()
    assert "_MATH_FONT_RE" not in src
    assert "math_font_char_count" in src
