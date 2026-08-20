"""GH-197 / GH-198 / GH-195 — the three leftovers from GH-144 A2 (PR #192).

All three live on the same rejection path in
``reconstruct.py::reconstruct_table_regions`` / ``_destroyed_numeric_tokens``:

* **#197** the ``None`` path still scoped the destruction check to
  ``table.bbox`` — the loose, whitespace-inferred rectangle the tight scope was
  introduced to stop using — so a numeral in the overrun could reject a good grid.
* **#198** the candidate filter skipped decorated numerics (``0.67***``, unicode
  minus, ``$0.67``, ``.034``), so a grid that split a starred coefficient was
  never rejected and the page shipped a silent wrong number.
* **#195** the rejection was visible only as a ``logger.warning``.

Fakes stand in for PyMuPDF ``Table``/``TableFinder``, mirroring
``test_reconstruct_gh144_review.py``, so the checks are pinned independent of
PyMuPDF's own table-detection heuristics.
"""

from __future__ import annotations

import fitz
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.reconstruct import (
    _destroyed_numeric_tokens,
    _numeric_row_bbox,
    reconstruct_table_regions,
)
from test_region_overlap_gh145 import table_page  # noqa: F401 (pytest fixture)


class _FakeRow:
    def __init__(self, bbox, cells):
        self.bbox = bbox
        self.cells = cells


class _FakeTable:
    def __init__(self, bbox, grid, rows):
        self.bbox = bbox
        self._grid = grid
        self.rows = rows

    def extract(self):
        return self._grid


class _FakeResult:
    def __init__(self, tables):
        self.tables = tables


# ---------------------------------------------------------------------------
# GH-198 — decorated numerics
# ---------------------------------------------------------------------------


def _coef_table(cell_value: str):
    """A 4x3 coefficient table that passes ``_looks_tabular``.

    ``cell_value`` is what the text-strategy grid put in the cell the starred
    coefficient's centre falls into. ``"0"`` models the GH-144 boundary split
    (the value itself was cut); ``"0.67"`` models the grid merely dropping the
    significance star.

    The grid must clear ``_looks_tabular`` — >=3 rows, >=2 cols, >=20% numeric
    cells, and a majority of rows carrying >=2 numeric cells — or
    ``reconstruct_table_regions`` skips the table before any of this is reached.
    """
    grid = [
        ["Firm", "Coef", "SE"],
        ["Alpha", cell_value, "0.12"],
        ["Beta", "1.10", "0.14"],
        ["Gamma", "2.30", "0.16"],
    ]

    def _row(y0, y1):
        return _FakeRow(
            (0.0, y0, 150.0, y1),
            [(0.0, y0, 50.0, y1), (50.0, y0, 100.0, y1), (100.0, y0, 150.0, y1)],
        )

    rows = [_row(0.0, 10.0), _row(10.0, 20.0), _row(20.0, 30.0), _row(30.0, 40.0)]
    table = _FakeTable((0.0, 0.0, 150.0, 40.0), grid, rows)
    words = [
        (10.0, 2.0, 40.0, 8.0, "Firm", 0, 0, 0),
        (55.0, 2.0, 90.0, 8.0, "Coef", 0, 0, 1),
        (105.0, 2.0, 140.0, 8.0, "SE", 0, 0, 2),
        (10.0, 12.0, 40.0, 18.0, "Alpha", 0, 1, 0),
        (55.0, 12.0, 90.0, 18.0, "PLACEHOLDER", 0, 1, 1),
        (105.0, 12.0, 140.0, 18.0, "0.12", 0, 1, 2),
        (10.0, 22.0, 40.0, 28.0, "Beta", 0, 2, 0),
        (55.0, 22.0, 90.0, 28.0, "1.10", 0, 2, 1),
        (105.0, 22.0, 140.0, 28.0, "0.14", 0, 2, 2),
        (10.0, 32.0, 40.0, 38.0, "Gamma", 0, 3, 0),
        (55.0, 32.0, 90.0, 38.0, "2.30", 0, 3, 1),
        (105.0, 32.0, 140.0, 38.0, "0.16", 0, 3, 2),
    ]
    return words, table, grid


def _with_token(token: str, cell_value: str):
    words, table, grid = _coef_table(cell_value)
    words[4] = (55.0, 12.0, 90.0, 18.0, token, 0, 1, 1)
    return words, table, grid


@pytest.mark.parametrize(
    "token",
    ["0.67***", "\u22120.253", "$0.67"],
    ids=["significance-stars", "unicode-minus", "currency-prefix"],
)
def test_decorated_numeric_destruction_is_detected(token: str) -> None:
    """#198: a decorated value split by a lane boundary must reject the grid.

    The bare anchored ``_NUM_TOKEN_RE`` matches none of these forms, so the
    candidate filter skipped them entirely and A2 never fired — the original
    GH-144 defect wearing a significance star. Here the grid cut the VALUE
    (the cell holds only ``0``), so neither the token nor its
    presentation-stripped form survives.
    """
    words, table, grid = _with_token(token, "0")

    scope = _numeric_row_bbox(table, grid, words)
    assert scope is not None, "setup: the table must have a numeric-bearing row"

    destroyed = _destroyed_numeric_tokens(words, table, grid, scope)

    assert [rec["value"] for rec in destroyed] == [token], (
        f"a lane boundary split the decorated value {token!r} and the destruction "
        f"check did not notice; got {destroyed}"
    )


def test_undecorated_numeric_that_survives_is_not_flagged() -> None:
    """Reverse regression for #198: widening the filter must not invent hits.

    The plain token survives verbatim in its raw cell. If the widened candidate
    filter or the widened survival test were wrong in the other direction, this
    clean grid would be rejected and the page would lose its table for nothing.
    """
    words, table, grid = _with_token("0.67", "0.67")

    scope = _numeric_row_bbox(table, grid, words)
    destroyed = _destroyed_numeric_tokens(words, table, grid, scope)

    assert destroyed == [], f"a surviving token was flagged as destroyed: {destroyed}"


def test_decoration_dropped_by_the_grid_is_not_destruction() -> None:
    """Reverse regression for #198: the VALUE must survive, not the star.

    A grid may legally render ``0.67***`` as ``0.67`` (the star typeset
    separately, the currency symbol living in the caption). The value is intact,
    so this is not destruction and must not reject the grid — otherwise the
    widening would manufacture a false rejection on every starred econometrics
    table, which is most of this corpus.
    """
    words, table, grid = _with_token("0.67***", "0.67")

    scope = _numeric_row_bbox(table, grid, words)
    destroyed = _destroyed_numeric_tokens(words, table, grid, scope)

    assert destroyed == [], (
        "the grid dropped only presentation (the stars), the value 0.67 survived — "
        f"this is not destruction: {destroyed}"
    )


# ---------------------------------------------------------------------------
# GH-197 — the None scope must never fall back to table.bbox
# ---------------------------------------------------------------------------


def test_no_numeric_row_keeps_the_grid_instead_of_checking_table_bbox(table_page, monkeypatch):
    """#197: with no numeric-bearing row, a numeral in the bbox overrun must
    not reject the grid.

    ``_numeric_row_bbox`` returns ``None`` (no row carries a numeric cell), and
    the pre-fix caller then passed ``fitz.Rect(table.bbox)`` — the loose
    whitespace-inferred rectangle — as the destruction scope. The stray ``106``
    in the overrun has no containing cell, counts as ``"no-cell"`` destroyed,
    and the good grid is thrown away.
    """
    # Cells carry digits EMBEDDED IN TEXT: `_NUMERIC_RE.search` finds them so
    # `_looks_tabular` accepts the grid, but `_cell_has_numeric_token` requires
    # every whitespace-split piece to be numeric-token-shaped, so no cell counts
    # and `_numeric_row_bbox` returns None. That is the exact state the `None`
    # path handles, and it is reachable without disabling `_looks_tabular`.
    grid = [
        ["Firm", "Fit", "Sample"],
        ["Alpha", "R2 0.45x", "N=106 obs"],
        ["Beta", "R2 0.51x", "N=204 obs"],
        ["Gamma", "R2 0.62x", "N=311 obs"],
    ]

    def _row(y0, y1):
        return _FakeRow(
            (0.0, y0, 150.0, y1),
            [(0.0, y0, 50.0, y1), (50.0, y0, 100.0, y1), (100.0, y0, 150.0, y1)],
        )

    rows = [_row(800.0, 810.0), _row(810.0, 820.0), _row(820.0, 830.0), _row(830.0, 840.0)]
    # bbox overruns 20pt past the last row, into where a stray numeral sits.
    table = _FakeTable((0.0, 800.0, 150.0, 860.0), grid, rows)

    with fitz.open(table_page) as doc:
        page = doc[0]
        words = [
            (10.0, 802.0, 40.0, 808.0, "Firm", 0, 0, 0),
            (55.0, 802.0, 90.0, 808.0, "Fit", 0, 0, 1),
            (10.0, 812.0, 40.0, 818.0, "Alpha", 0, 1, 0),
            (55.0, 812.0, 90.0, 818.0, "R2", 0, 1, 1),
            (10.0, 822.0, 40.0, 828.0, "Beta", 0, 2, 0),
            (10.0, 832.0, 40.0, 838.0, "Gamma", 0, 3, 0),
            (10.0, 850.0, 30.0, 856.0, "106", 0, 4, 0),  # in the overrun, no cell
        ]
        assert _numeric_row_bbox(table, grid, words) is None, "setup: no numeric row"
        # Pre-fix scope: proves the stray numeral really would have been counted.
        assert _destroyed_numeric_tokens(words, table, grid, fitz.Rect(table.bbox)), (
            "setup: table.bbox scoping picks up the stray '106'"
        )

        monkeypatch.setattr(page, "get_text", lambda *a, **k: words)
        monkeypatch.setattr(page, "find_tables", lambda *a, **k: _FakeResult([table]))
        monkeypatch.setattr("socr.tables.reconstruct.has_numeric_columns", lambda _page: True)

        out = reconstruct_table_regions(page)

    assert len(out) == 1, f"the good grid was rejected on the None path: {out}"
    assert "Alpha" in out[0][1] and "R2 0.45x" in out[0][1], out[0][1]


# ---------------------------------------------------------------------------
# GH-195 — the rejection must reach a surface, not only a log
# ---------------------------------------------------------------------------


def test_real_rejection_reaches_a_surface(table_page) -> None:
    """#195: a genuine rejection must reach a surface, not only a log.

    Drives the REAL ``BornDigitalDetector`` over GH-144's own regression fixture
    — an unruled whitespace-gutter table whose text-strategy grid genuinely
    splits ``1.00`` / ``1.10`` / ``1.06`` / ``0.73`` — through the real
    ``_phase_analyze``. Nothing here is faked, and nothing here names a symbol
    the fix introduces: at ``main_sha`` the rejection happens, the log line is
    emitted, and ``state.events`` stays empty.

    The house rule is that a failure surfaces at page, document, metadata and
    CLI level, not just one. The audit event carries the page number, so it
    reaches the page sidecar, ``audit_log.json`` and the CLI summary line.
    """
    from pathlib import Path as _P

    pipeline = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GEMINI],
            primary_engine=EngineType.DEEPSEEK,
            save_figures=False,
            dual_pass_tables=False,
            detect_equations=False,
            recover_clean_equations=False,
            quiet=True,
            audit_enabled=False,
            write_manifest=False,
        )
    )
    state = DocumentState(handle=DocumentHandle(path=_P(str(table_page)), page_count=1))
    pipeline._phase_analyze(state)

    events = [e for e in state.events if e.kind == "text_grid_rejected"]
    assert events, (
        "the text-strategy grid rejection reached no surface — it exists only as a "
        f"logger.warning; recorded events were {[(e.page_num, e.kind) for e in state.events]}"
    )
    assert events[0].page_num == 1
    assert events[0].data["destroyed_tokens"] >= 1
    assert "adversarial" in events[0].detail


def test_clean_page_emits_no_rejection_event(tmp_path) -> None:
    """Reverse regression for #195: a page with no rejection stays quiet.

    An upright, RULED table: ``find_tables`` succeeds on the lines strategy, so
    the text-strategy reconstruct never runs and there is nothing to reject.
    """
    from pathlib import Path as _P

    pdf_path = tmp_path / "ruled.pdf"
    doc = fitz.open()
    page = doc.new_page()
    x0, y0, cw, rh, cols, rows_n = 100, 100, 60, 20, 3, 4
    for r in range(rows_n + 1):
        page.draw_line((x0, y0 + r * rh), (x0 + cols * cw, y0 + r * rh))
    for c in range(cols + 1):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + rows_n * rh))
    for r in range(rows_n):
        for c in range(cols):
            page.insert_text((x0 + c * cw + 5, y0 + r * rh + 15), f"{r}.{c}", fontsize=8)
    doc.save(str(pdf_path))
    doc.close()

    pipeline = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GEMINI],
            primary_engine=EngineType.DEEPSEEK,
            save_figures=False,
            quiet=True,
            audit_enabled=False,
            write_manifest=False,
        )
    )
    state = DocumentState(handle=DocumentHandle(path=_P(str(pdf_path)), page_count=1))
    pipeline._phase_analyze(state)

    assert not [e for e in state.events if e.kind == "text_grid_rejected"], state.events
