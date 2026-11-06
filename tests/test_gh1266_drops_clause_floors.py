"""#1266 x #1218: the ``text_layer_drops`` clause never fires on a #1266 floor body.

A floor withholds numeral-dense lines ON PURPOSE; that is not "the text layer's numbers were
dropped". For every ending (S1 floor recovered and whole-page, D3 recovered and whole-page,
the withheld-guard floor, the box-less KEEP-SENTENCES body and its line-wise order-gate
fallback) the page carries an own layer and a drop reference holding exactly the withheld
table numbers. DIFFERENCE: the same page finalized with that reference and without it ships
identical text, status, failure mode and verdict -- and the clause is absent. Non-vacuity: on
the same page state a MODEL reading that omits a row gets the clause.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gh1266_d3_regional_floor import (  # noqa: E402
    TWO_COL_1,
    TWO_COL_2,
    _boxless,
    _d3,
    _draw_prose_beside_a_grid,
    _structure_failed_only,
)
from test_gh1266_floor_geometry import (  # noqa: E402
    GRID_BOX,
    P94_BOX,
    SENTENCE_WITH_YEAR,
    _build,
    _draw_p10,
    _draw_p94,
)
from test_gh1266_withheld_floor import _withheld_page  # noqa: E402

from socr.core.manifest import (  # noqa: E402
    _apply_ladder_disposition_guard,
    _select_and_finalize_page,
)
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.core.verdict import Check, text_layer_drops_clause  # noqa: E402

#: The numbers of ``_draw_p10``'s grid rows -- what every floor withholds.
GRID_REFERENCE = "2.6 4.1\n2.9 3.9\n2.4 4.0\n3.1 3.8"


def _with_reference(ps, reference: str) -> None:
    ps.own_layer_text = ps.native_text or ""
    ps.own_layer_drop_reference = reference


def _clean_layer_flags(ps) -> None:
    for name in ("prior_ocr_layer_suspect", "has_unrecovered_symbol_glyphs"):
        setattr(ps, name, False)


def _finalized(state, ps, reference: str):
    _with_reference(ps, reference)
    rec = _select_and_finalize_page(state, 1)
    clauses = [c.check for c in (rec.verdict.clauses if rec.verdict else ())]
    return rec.output, clauses


def _assert_inert(state, ps) -> None:
    _clean_layer_flags(ps)
    with_ref, with_clauses = _finalized(state, ps, GRID_REFERENCE)
    without_ref, without_clauses = _finalized(state, ps, "")
    assert Check.TEXT_LAYER_DROPS not in with_clauses, with_ref.text
    assert text_layer_drops_clause(with_ref, ps) is None
    assert with_ref.text == without_ref.text
    assert with_ref.status is without_ref.status
    assert with_ref.failure_mode is without_ref.failure_mode
    assert with_clauses == without_clauses
    # non-vacuity: on this page state, a model reading that omits a row IS flagged
    _with_reference(ps, GRID_REFERENCE)
    model = PageOutput(
        page_num=1, text="Real GDP 2.6 4.1", status=PageStatus.SUCCESS, engine="qwen"
    )
    assert text_layer_drops_clause(model, ps) is not None, "premise: the reference is live"


def _draw_two_columns(page) -> None:
    """The column-major two-column prose page (the order-gate fallback's own fixture)."""
    for i, line in enumerate(TWO_COL_1):
        page.insert_text((50, 100 + i * 14), line, fontsize=9)
    for i, line in enumerate(TWO_COL_2):
        page.insert_text((320, 100 + i * 14), line, fontsize=9)


def _s1(tmp_path: Path, draw, boxes, name):
    state, ps = _build(tmp_path, draw, boxes, name=name)
    return state, ps


@pytest.mark.parametrize(
    "shape", ["s1-recovered", "s1-whole", "d3-recovered", "d3-whole", "boxless", "boxless-fallback"]
)
def test_the_drops_clause_is_inert_on_every_floor_ending(tmp_path: Path, shape: str) -> None:
    if shape == "s1-recovered":
        state, ps = _s1(tmp_path, _draw_p10, [GRID_BOX], "a.pdf")
    elif shape == "s1-whole":
        state, ps = _s1(tmp_path, _draw_p94, [P94_BOX], "b.pdf")
    elif shape == "d3-recovered":
        state, ps = _s1(tmp_path, _draw_p10, [GRID_BOX], "c.pdf")
        _d3(ps)
    elif shape == "d3-whole":
        state, ps = _s1(tmp_path, _draw_p94, [P94_BOX], "d.pdf")
        _d3(ps)
    elif shape == "boxless":
        state, ps = _boxless(tmp_path, name="e.pdf")
    else:
        state, ps = _build(tmp_path, _draw_two_columns, [], name="f.pdf")
        _structure_failed_only(ps, model_attempt=False)
        ps.boxless_table_words = list(ps.native_words)
        ps.native_words = []
    _assert_inert(state, ps)


def test_the_drops_clause_is_inert_on_the_withheld_guard_floor(tmp_path: Path) -> None:
    """The guard keeps the model winner's engine, so the clause's engine gate does not exclude
    it; socr's own withholding (TABLE_WITHHELD and the floor banner) does."""
    ps, winner = _withheld_page(tmp_path)
    _clean_layer_flags(ps)
    _with_reference(ps, "1.5 2.5\n3.5 4.5\n7.7 8.8")
    out = _apply_ladder_disposition_guard(winner, 1, ps)
    assert out.engine == "qwen", "premise: a model-engine output reaches the clause"
    assert text_layer_drops_clause(out, ps) is None
    # non-vacuity: the same page WITHOUT socr's withhold, and a model reading omitting rows
    ps.table_ladder_disposition = None
    model = PageOutput(page_num=1, text="| a | b |\n|---|---|\n| 1.5 | 2.5 |", engine="qwen")
    assert text_layer_drops_clause(model, ps) is not None, "premise: the reference is live"


def test_the_prose_beside_a_grid_page_is_whole_page_and_inert(tmp_path: Path) -> None:
    state, ps = _build(tmp_path, _draw_prose_beside_a_grid, [], name="g.pdf")
    _structure_failed_only(ps, model_attempt=False)
    ps.boxless_table_words = list(ps.native_words)
    ps.native_words = []
    _clean_layer_flags(ps)
    out, clauses = _finalized(state, ps, "12.5 13.1 14.2")
    assert Check.TEXT_LAYER_DROPS not in clauses
    assert SENTENCE_WITH_YEAR not in out.text
