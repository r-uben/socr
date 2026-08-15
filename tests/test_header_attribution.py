"""Tests for the GH-200 header-attribution check.

Hermetic: synthetic fitz pages (pattern of ``tests/test_header_repair.py``:
``fitz.open()`` + ``page.insert_text``); no ollama/GPU/provider.

The underlying geometry chain (``header_repair.native_header_row``) only
recognises a native row as a table-header band when it carries a range/bin
marker or connector word (``%``, ``+/-``, ``<``, ``>``, ``to``, ``or``,
``more``, ``less`` — see ``header_repair._is_table_header_row``), so these
fixtures use ``%``-marked lane labels rather than plain words, matching the
same construction ``test_header_repair.py`` already relies on.
"""

from __future__ import annotations

import fitz

from socr.tables.header_attribution import HeaderVerdict, header_attribution
from socr.tables.reconcile import find_table_blocks
from socr.tables.structure_check import table_header_verdicts

_LABEL_X = 40.0
_DATA_XS = [150.0, 205.0, 260.0]
_HDR_Y = 100.0
_DATA_Y = 140.0


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _make_three_lane_page(header_tokens: list[str] | None = None) -> fitz.Page:
    """Label + 3-lane table; header words (if given) carry a ``%`` marker."""
    doc = fitz.open()
    page = doc.new_page(width=700, height=400)
    if header_tokens is not None:
        for x, tok in zip(_DATA_XS, header_tokens):
            page.insert_text((x, _HDR_Y), tok, fontsize=9)
    page.insert_text((_LABEL_X, _DATA_Y), "Row1", fontsize=9)
    for x, val in zip(_DATA_XS, ["1", "2", "3"]):
        page.insert_text((x, _DATA_Y), val, fontsize=9)
    return page


def _first_grid(md: str):
    return find_table_blocks(md)[0].grid


class TestHeaderAttribution:
    def test_missing_header_band_is_hard(self):
        """Native header words present over every data lane; candidate's
        header row is all-blank for those lanes -> HARD."""
        page = _make_three_lane_page(["Low%", "Mid%", "High%"])
        words = page.get_text("words")
        # Label survives (an all-blank header row is dropped by the parser's
        # own separator-row blind spot -- structure_check.py's module
        # docstring), the 3 data-lane cells are blank.
        md = _md_table(["Currency", "", "", ""], [["Row1", "1", "2", "3"]])
        grid = _first_grid(md)
        assert header_attribution(grid, words) is HeaderVerdict.HARD

    def test_spacer_column_without_native_header_is_not_hard(self):
        """One data lane has NO native word above it, and the candidate's
        header cell for that lane is empty too -> not HARD (Gemini's named
        falsifier, a negative control)."""
        doc = fitz.open()
        page = doc.new_page(width=700, height=400)
        # Only lanes 0 and 2 get native header words; lane 1 (Mid) is a
        # legitimate spacer column with no header owed.
        page.insert_text((_DATA_XS[0], _HDR_Y), "Low%", fontsize=9)
        page.insert_text((_DATA_XS[2], _HDR_Y), "High%", fontsize=9)
        page.insert_text((_LABEL_X, _DATA_Y), "Row1", fontsize=9)
        for x, val in zip(_DATA_XS, ["1", "2", "3"]):
            page.insert_text((x, _DATA_Y), val, fontsize=9)
        words = page.get_text("words")
        md = _md_table(["", "Low%", "", "High%"], [["Row1", "1", "2", "3"]])
        grid = _first_grid(md)
        assert header_attribution(grid, words) is not HeaderVerdict.HARD

    def test_spanning_merged_header_is_not_hard(self):
        """A single merged non-empty header cell ('Results%') spans all
        lanes with no leaf header row; body numerals correct. Documented
        MISS (not a false pass): only one lane's native cell is non-empty,
        so HARD cannot fire -- pins the resolved falsifier."""
        page = _make_three_lane_page(["Results%"])  # one word, lands in lane 0 only
        # Overwrite header words: single spanning token near the row centre.
        doc = fitz.open()
        page = doc.new_page(width=700, height=400)
        page.insert_text((_DATA_XS[1], _HDR_Y), "Results%", fontsize=9)
        page.insert_text((_LABEL_X, _DATA_Y), "Row1", fontsize=9)
        for x, val in zip(_DATA_XS, ["1", "2", "3"]):
            page.insert_text((x, _DATA_Y), val, fontsize=9)
        words = page.get_text("words")
        md = _md_table(["", "", "", ""], [["Row1", "1", "2", "3"]])
        grid = _first_grid(md)
        assert header_attribution(grid, words) is not HeaderVerdict.HARD

    def test_headerless_two_column_index_abstains(self):
        """No lane-aligned header band at all (book back-matter index shape:
        short label lines + a trailing run of page numbers, no header row).
        ``native_header_row`` returns None -> UNVERIFIABLE."""
        page = _make_three_lane_page(header_tokens=None)
        words = page.get_text("words")
        md = _md_table(["", "", "", ""], [["Row1", "1", "2", "3"]])
        grid = _first_grid(md)
        assert header_attribution(grid, words) is HeaderVerdict.UNVERIFIABLE

    def test_misplaced_header_token_is_soft_not_hard(self):
        """Header tokens present in grid[0] but shifted one column against
        the geometry lanes -> SOFT, not HARD."""
        page = _make_three_lane_page(["Low%", "Mid%", "High%"])
        words = page.get_text("words")
        # All 3 native tokens are present in the emitted header row (so HARD
        # cannot fire), but rotated into the wrong columns vs the geometry
        # lanes.
        md = _md_table(["", "High%", "Low%", "Mid%"], [["Row1", "1", "2", "3"]])
        grid = _first_grid(md)
        assert header_attribution(grid, words) is HeaderVerdict.SOFT

    def test_leading_decimal_table_abstains_and_is_counted(self):
        """Values typeset '.034' (no leading zero) throughout -> fewer than
        2 numeric lanes derived by ``_NUM_TOKEN_RE`` -> UNVERIFIABLE, and the
        abstain is visible via ``table_header_verdicts`` (not silently
        swallowed) -- makes the #206/#207 notation gap measurable."""
        doc = fitz.open()
        page = doc.new_page(width=700, height=400)
        page.insert_text((_DATA_XS[0], _HDR_Y), "Low%", fontsize=9)
        page.insert_text((_DATA_XS[1], _HDR_Y), "Mid%", fontsize=9)
        page.insert_text((_LABEL_X, _DATA_Y), "Row1", fontsize=9)
        for x, val in zip(_DATA_XS[:2], [".034", ".012"]):
            page.insert_text((x, _DATA_Y), val, fontsize=9)
        words = page.get_text("words")
        md = _md_table(["", "", ""], [["Row1", ".034", ".012"]])
        grid = _first_grid(md)
        assert header_attribution(grid, words) is HeaderVerdict.UNVERIFIABLE
        verdicts = table_header_verdicts(md, words)
        assert verdicts == [HeaderVerdict.UNVERIFIABLE]
