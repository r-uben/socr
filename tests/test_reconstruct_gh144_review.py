"""GH-144 PR #192 review — findings 1 and 2.

``reconstruct_table_regions`` loops over every table PyMuPDF's text-strategy
`find_tables()` found on a page. Two review findings targeted that loop and
the rejection check it drives, independent of the lossless-rowizer geometry
covered by ``test_region_overlap_gh145.py``:

1. An early ``return`` inside the per-table loop discarded every OTHER table
   already collected on the page the moment one table triggered rejection.
2. The destroyed-token check was scoped to ``table.bbox`` — a whitespace
   -inferred rectangle that routinely overruns past the table's last real row
   into unrelated text below it (a note, a sample-size sentence, an axis
   label). A numeral caught in that overrun with no containing cell would get
   counted "destroyed" and reject an otherwise-clean grid.

Both are exercised here with lightweight fakes standing in for PyMuPDF
``Table``/``TableFinder`` objects, so the control-flow and scoping bugs are
pinned down independent of PyMuPDF's own table-detection heuristics.
"""

from __future__ import annotations

import fitz

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


def test_multi_table_page_keeps_undamaged_tables_when_one_is_rejected(table_page, monkeypatch):
    """GH-144 review finding 1: rejecting one table's text-strategy grid must
    not discard the OTHER, undamaged tables already collected on the page.

    Before the fix, ``reconstruct_table_regions`` did ``return rowized`` /
    ``return []`` the moment ONE table in ``result.tables`` was rejected —
    any table already appended to ``out`` from an earlier iteration, and any
    table still to come, was silently dropped from the page's output.
    """
    with fitz.open(table_page) as doc:
        page = doc[0]
        real_result = page.find_tables(vertical_strategy="text", horizontal_strategy="text")
        damaged_table = real_result.tables[0]

        # A second, undamaged table positioned well below every real word on
        # the page — `_destroyed_numeric_tokens` finds nothing in its scope
        # by construction, so it must survive via the direct (non-fallback)
        # path regardless of what happens to `damaged_table`.
        clean_bbox = (60.0, 900.0, 220.0, 940.0)
        clean_grid = [
            ["Firm", "2024", "2025"],
            ["Alpha", "1.23", "4.56"],
            ["Beta", "7.89", "0.12"],
        ]
        clean_rows = [
            _FakeRow(
                (60.0, 900.0, 220.0, 910.0),
                [
                    (60.0, 900.0, 110.0, 910.0),
                    (110.0, 900.0, 165.0, 910.0),
                    (165.0, 900.0, 220.0, 910.0),
                ],
            ),
            _FakeRow(
                (60.0, 910.0, 220.0, 920.0),
                [
                    (60.0, 910.0, 110.0, 920.0),
                    (110.0, 910.0, 165.0, 920.0),
                    (165.0, 910.0, 220.0, 920.0),
                ],
            ),
            _FakeRow(
                (60.0, 920.0, 220.0, 940.0),
                [
                    (60.0, 920.0, 110.0, 940.0),
                    (110.0, 920.0, 165.0, 940.0),
                    (165.0, 920.0, 220.0, 940.0),
                ],
            ),
        ]
        clean_table = _FakeTable(clean_bbox, clean_grid, clean_rows)

        monkeypatch.setattr(
            page,
            "find_tables",
            lambda *a, **k: _FakeResult([damaged_table, clean_table]),
        )

        out = reconstruct_table_regions(page)

    assert len(out) == 2, (
        f"expected both tables to survive, got {len(out)}: {[md for _, md in out]}"
    )
    mds = [md for _, md in out]
    assert any("3M Treasury" in md for md in mds), "the repaired damaged table must survive"
    assert any("Alpha" in md and "1.23" in md for md in mds), (
        "the undamaged second table must not be dropped"
    )


def test_destroyed_check_is_scoped_to_numeric_rows_not_overrun_bbox():
    """GH-144 review finding 2: a numeral inside ``table.bbox``'s overrun with
    no containing cell must not be counted as a destroyed token.

    ``table.bbox`` is whitespace-inferred by ``find_tables`` and routinely
    overruns past the last real row into whatever text sits just beneath —
    an axis tick, a page number, or (as here) a stray sample-size numeral in
    a notes line. Scoping the check to ``_numeric_row_bbox``'s tighter,
    data-driven union instead of ``table.bbox`` must not flag it.
    """
    grid = [["Firm", "Value"], ["Alpha", "1.23"]]
    rows = [
        _FakeRow((0.0, 0.0, 100.0, 10.0), [(0.0, 0.0, 50.0, 10.0), (50.0, 0.0, 100.0, 10.0)]),
        _FakeRow((0.0, 10.0, 100.0, 20.0), [(0.0, 10.0, 50.0, 20.0), (50.0, 10.0, 100.0, 20.0)]),
    ]
    # `table.bbox` overruns 20pt past the last real row, into where a
    # standalone "106" (no containing cell) sits -- e.g. a sample-size note.
    table = _FakeTable((0.0, 0.0, 100.0, 40.0), grid, rows)

    words = [
        (10.0, 2.0, 40.0, 8.0, "Firm", 0, 0, 0),
        (55.0, 2.0, 70.0, 8.0, "Value", 0, 0, 1),
        (10.0, 12.0, 40.0, 18.0, "Alpha", 0, 1, 0),
        (55.0, 12.0, 70.0, 18.0, "1.23", 0, 1, 1),
        (10.0, 30.0, 30.0, 36.0, "106", 0, 2, 0),  # inside table.bbox's overrun, outside every row
    ]

    numeric_scope = _numeric_row_bbox(table, grid, words)
    assert numeric_scope is not None

    destroyed_using_table_bbox = _destroyed_numeric_tokens(
        words, table, grid, fitz.Rect(table.bbox)
    )
    destroyed_using_numeric_scope = _destroyed_numeric_tokens(words, table, grid, numeric_scope)

    assert destroyed_using_table_bbox, (
        "sanity: table.bbox scoping picks up the stray '106' (pre-fix behaviour)"
    )
    assert not destroyed_using_numeric_scope, (
        "numeric-row scoping must not flag a numeral outside every row"
    )
