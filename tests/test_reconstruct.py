"""Structure-restore: rebuild markdown grids for born-digital booktabs tables."""

from __future__ import annotations

import fitz

from socr.core.born_digital import BornDigitalDetector
from socr.tables.reconstruct import (
    _clean_grid,
    _grid_to_markdown,
    _is_runhead,
    _looks_tabular,
    reconstruct_table_regions,
)

_DATA = [
    ["Industry", "b", "s", "h"],
    ["FabPr", "0.253", "0.179", "0.211"],
    ["Clths", "0.144", "0.135", "0.290"],
    ["Chem", "0.041", "0.000", "0.154"],
    ["Toys", "0.082", "0.321", "0.144"],
    ["Energy", "0.180", "0.171", "0.365"],
]


def _booktabs_page():
    """Born-digital page: a booktabs table (horizontal rules only) — find_tables
    lines strategy returns nothing, the text strategy recovers the grid."""
    doc = fitz.open()
    page = doc.new_page()
    cols = [90, 230, 320, 410]
    rows = [120 + i * 22 for i in range(len(_DATA))]
    for r, row in enumerate(_DATA):
        for c, cell in enumerate(row):
            page.insert_text((cols[c], rows[r]), cell, fontsize=10)
    for yy in [rows[0] - 8, rows[1] - 6, rows[-1] + 8]:
        page.draw_line((90, yy), (440, yy))
    return doc, page


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_grid_to_markdown_shape():
    md = _grid_to_markdown([["a", "b"], ["1", "2"], ["3", "4"]])
    lines = md.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | 2 |"


def test_clean_grid_drops_empty_rows_and_cols():
    grid = [["x", "", "1"], ["", "", ""], ["y", "", "2"]]
    assert _clean_grid(grid) == [["x", "1"], ["y", "2"]]


def test_clean_grid_strips_runhead_row():
    grid = [
        ["160 A. Author, Journal of Economics 43 (1997) 1-20", "", ""],
        ["Var", "b", "s"],
        ["x", "0.1", "0.2"],
        ["y", "0.3", "0.4"],
    ]
    cleaned = _clean_grid(grid)
    assert cleaned[0] == ["Var", "b", "s"]  # runhead gone, header now first


def test_is_runhead_matches_journal_line_even_ocr_corrupted():
    assert _is_runhead(["160 E.F. Fama", "chfJoumal ofFin", "ancial Economic"])  # corrupted
    assert _is_runhead(["A. Author, Quarterly Journal of Economics (1990)"])
    assert not _is_runhead(["FabPr", "0.253", "0.179", "0.211"])  # a data row


def test_looks_tabular_rejects_prose_accepts_numbers():
    numbers = [["v", "b", "s"], ["x", "0.1", "0.2"], ["y", "0.3", "0.4"]]
    prose = [["The", "quick", "brown"], ["fox", "jumps", "over"], ["the", "lazy", "dog"]]
    assert _looks_tabular(numbers)
    assert not _looks_tabular(prose)         # no numeric content
    assert not _looks_tabular([["a", "1"]])  # too few rows


# --------------------------------------------------------------------------
# Region reconstruction + integration
# --------------------------------------------------------------------------


def test_reconstruct_recovers_grid_from_booktabs_page():
    _doc, page = _booktabs_page()
    # the default lines strategy sees no table here
    assert not page.find_tables().tables
    regions = reconstruct_table_regions(page)
    assert regions, "text-strategy should recover the booktabs grid"
    _rect, md = regions[0]
    assert "| FabPr |" in md and "0.253" in md
    assert "| --- |" in md


def test_extract_structured_emits_grid_for_booktabs_table():
    _doc, page = _booktabs_page()
    out = BornDigitalDetector().extract_structured(page)
    assert "| --- |" in out               # a markdown table, not a flat dump
    assert "0.253" in out                 # char-exact native value preserved


def test_reconstruct_skips_pathological_dense_page():
    # A page with thousands of words (reference list / equation dump) must be
    # skipped before the expensive text-strategy call — never hang the pipeline.
    doc = fitz.open()
    page = doc.new_page()
    y = 40
    line = "Author A 2020 Journal 10 200 some reference text et al pages " * 4
    for _ in range(60):                      # ~14k words, far over the guard
        page.insert_text((40, y), line, fontsize=6)
        y += 11
        if y > 780:
            page = doc.new_page()
            y = 40
    page0 = doc[0]
    assert len(page0.get_text("words")) > 1500
    assert reconstruct_table_regions(page0) == []   # guarded, returns fast


def test_extract_structured_leaves_prose_alone():
    doc = fitz.open()
    page = doc.new_page()
    y = 100
    para = "This is an ordinary paragraph of body prose with no tabular structure."
    for _ in range(8):
        page.insert_text((72, y), para, fontsize=11)
        y += 16
    out = BornDigitalDetector().extract_structured(page)
    assert "| --- |" not in out           # prose must never become a table
