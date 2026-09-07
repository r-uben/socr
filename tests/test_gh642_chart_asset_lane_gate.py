"""GH-642: the A1b candidate pools must gate on the same native-lane tuple
``_NATIVE_TEXT_LANES`` every other winner-selection branch already uses.

``_grid_shaped_attempt`` (S1's strict pool, TICKET-A1b #634) and
``_grid_reading_attempt`` (the row-corroboration fallback's own pool,
TICKET-A1b #634) each excluded native attempts with a bare
``(out.engine or "").startswith("native")`` check -- written before #265's
own review finding that ``chart_asset`` ships ``native_text`` too (a whole-
page PNG ref appended) and is therefore just as untrustworthy a candidate for
"a non-native reading authored this grid" as ``native`` itself.
``_qualifies`` in ``_d3_floor_superseding_output`` (a few lines above both
functions in ``manifest.py``) already switched to the ``_NATIVE_TEXT_LANES``
tuple for exactly this reason; the two A1b pools did not, so a
``chart_asset``-engine attempt whose relabelled native text happens to look
like a table grid could still enter the A1b candidate pools and outrank a
real ``native`` attempt for no reason other than its engine label.

Falsification: a ``chart_asset`` attempt with the SAME table-shaped body a
``native`` attempt would carry must be excluded by both pools identically to
how a ``native`` attempt is excluded. On main (``.startswith("native")``)
the ``chart_asset`` attempt is wrongly ADMITTED while the ``native`` one is
excluded.
"""

from __future__ import annotations

from test_s1_structure_class_winner_corroboration import (
    BAD_MD,
    GOOD_MD,
    _floored_structure_class_page,
    _grid_reading_output,
)

from socr.core.manifest import (
    _grid_reading_attempt,
    _grid_shaped_attempt,
    structure_class_floor_applies,
    structure_class_grid_winner,
)
from socr.core.result import FailureMode, PageOutput, PageStatus

STRICT_GRID_MD = (
    "| Year | A | B |\n|---|---|---|\n| 2018 | 100.0 | 200.0 |\n| 2019 | 110.0 | 210.0 |\n"
)

# Ragged body (a units line above the numeric rows): has_strict_table_grid()
# is False but has_authored_table_grid() is True, matching
# test_s1_structure_class_winner_corroboration.py's own GOOD_MD shape -- the
# corroboration fallback's pool is what this exercises.
RAGGED_GRID_MD = (
    "| Year | A | B |\n"
    "|---|---|---|\n"
    "| units |\n"
    "| 2018 | 100.0 | 200.0 |\n"
    "| 2019 | 110.0 | 210.0 |\n"
)


def _output(engine: str, text: str) -> PageOutput:
    return PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=False,
        confidence=0.5,
        failure_mode=FailureMode.NONE,
    )


def test_chart_asset_engine_excluded_from_strict_grid_pool() -> None:
    """A chart_asset attempt with a strict-grid-shaped body must not be a
    candidate in S1's strict pool -- same exclusion a native attempt gets.
    """
    native_out = _output("native", STRICT_GRID_MD)
    chart_out = _output("chart_asset", STRICT_GRID_MD)

    assert _grid_shaped_attempt(native_out) is False
    assert _grid_shaped_attempt(chart_out) is False


def test_chart_asset_engine_excluded_from_row_corroboration_pool() -> None:
    """Same exclusion, for the row-corroboration fallback's own (ragged-body
    tolerant) pool.
    """
    native_out = _output("native", RAGGED_GRID_MD)
    chart_out = _output("chart_asset", RAGGED_GRID_MD)

    assert _grid_reading_attempt(native_out) is False
    assert _grid_reading_attempt(chart_out) is False


def test_non_native_engine_still_admitted() -> None:
    """Sanity: the fix must not blanket-exclude every engine -- a genuine
    model reading (qwen) with the same table-shaped body still qualifies.
    """
    assert _grid_shaped_attempt(_output("qwen", STRICT_GRID_MD)) is True
    assert _grid_reading_attempt(_output("qwen", RAGGED_GRID_MD)) is True


def test_chart_asset_cannot_win_the_real_corroboration_fallback() -> None:
    """Astra review (#642): the two predicate-level tests above do not pin
    CALLER wiring -- ``_grid_reading_attempt`` could return the right answer
    while an upstream candidate-collection site still let a chart_asset
    reading through some other way. Reuses
    ``test_s1_structure_class_winner_corroboration``'s own real fixture
    (``test_corroborating_candidate_wins_over_the_floor``, where an
    otherwise-identical ``qwen`` attempt DOES win) with only the engine
    label swapped to ``chart_asset``: the same row-corroborating text must
    now be invisible to the actual ``structure_class_grid_winner`` /
    ``_row_corroborated_grid_winner`` call chain, so no winner survives and
    the floor applies.
    """
    chart = _grid_reading_output("chart_asset", GOOD_MD)
    bad = _grid_reading_output("gemini", BAD_MD)
    p = _floored_structure_class_page(with_native_words=True, attempts=[bad, chart])

    assert structure_class_grid_winner(p) is None
    assert structure_class_floor_applies(p) is True
