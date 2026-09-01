"""GH-351: the coverage harness must rowize exactly as production does.

``benchmark/binding_coverage._discover_native_regions`` reimplements the
discovery chain. On the lane-stacked path it diverged: production calls
``rowize_from_word_list(region_words)`` bare, the harness passed a clip-scoped
``rotation`` and ``page_rect``. On a rotated lane-stacked page the scoreboard
therefore described a different native candidate than the one that ships.

The ``_is_lane_stacked`` allowlist exists so the instrument cannot drift from
the thing it measures. The call beside it had drifted.

Pinned by comparing the two call sites' ARGUMENTS, which is what "same
candidate" means here -- a behavioural test would need a rotated lane-stacked
fixture and would still not catch a divergence the fixture happens not to
exercise.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path


def _rowize_call_kwargs(source: str, func_name: str) -> set[str]:
    """Keyword names passed to ``rowize_from_word_list`` inside ``func_name``."""
    tree = ast.parse(source)
    found: set[str] = set()
    seen_call = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != func_name:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "rowize_from_word_list"
            ):
                seen_call = True
                found |= {kw.arg for kw in inner.keywords if kw.arg}
    assert seen_call, f"no rowize_from_word_list call found in {func_name}()"
    return found


def _source_of(module_path: str) -> str:
    import importlib

    module = importlib.import_module(module_path)
    return Path(inspect.getfile(module)).read_text(encoding="utf-8")


class TestTheInstrumentMatchesTheThingItMeasures:
    def test_the_harness_passes_the_same_kwargs_as_production(self) -> None:
        production = _rowize_call_kwargs(_source_of("socr.core.born_digital"), "extract_structured")
        harness = _rowize_call_kwargs(
            _source_of("socr.benchmark.binding_coverage"), "_discover_native_regions"
        )

        assert harness == production, (
            f"the coverage harness rowizes with {harness or 'no kwargs'} while "
            f"production uses {production or 'no kwargs'}; the scoreboard would "
            "describe a different native candidate than the one that ships"
        )

    def test_neither_side_passes_rotation(self) -> None:
        """The specific drift GH-351 names, stated outright so a future reader
        sees WHICH kwargs were the problem rather than only that they matched.

        ``rowize_from_words`` (the reconstruct fallback) applies a page-wide
        rotation. That is a second, deliberate policy. The harness must not
        invent a third.
        """
        harness = _rowize_call_kwargs(
            _source_of("socr.benchmark.binding_coverage"), "_discover_native_regions"
        )

        assert "rotation" not in harness
        assert "page_rect" not in harness
