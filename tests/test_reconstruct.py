"""Structure-restore: rebuild markdown grids for born-digital booktabs tables."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from socr.core.born_digital import BornDigitalDetector
from socr.tables.reconstruct import (
    _clean_grid,
    _collapse_header_prefix,
    _grid_to_markdown,
    _is_runhead,
    _looks_tabular,
    has_numeric_columns,
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
    assert not _looks_tabular(prose)  # no numeric content
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
    assert "| --- |" in out  # a markdown table, not a flat dump
    assert "0.253" in out  # char-exact native value preserved


def test_numeric_columns_gate_accepts_table_rejects_references():
    # Table: numbers stack into lanes, each data row populates several -> True.
    _doc, page = _booktabs_page()
    assert has_numeric_columns(page) is True

    # References list: numbers scattered one-per-line (year, volume, pages) at
    # ragged x, never co-occupying multiple lanes in a row -> False.
    rdoc = fitz.open()
    rpage = rdoc.new_page()
    y = 80
    cites = [
        "Fama, E. and K. French (1992), Journal of Finance 47, 427-465.",
        "Jensen, M. (1968), Journal of Finance 23, 389-416.",
        "Sharpe, W. (1964), Journal of Finance 19, 425-442.",
        "Black, F. (1972), Journal of Business 45, 444-455.",
        "Banz, R. (1981), Journal of Financial Economics 9, 3-18.",
    ]
    for c in cites:
        rpage.insert_text((72, y), c, fontsize=10)
        y += 18
    assert has_numeric_columns(rpage) is False


def test_extract_structured_leaves_references_alone():
    doc = fitz.open()
    page = doc.new_page()
    y = 80
    for i in range(8):
        page.insert_text((72, y), f"Author {i} (199{i}), Journal of X {i}, 1{i}-2{i}.", fontsize=10)
        y += 18
    out = BornDigitalDetector().extract_structured(page)
    assert "| --- |" not in out  # references never become a table


def test_reconstruct_skips_pathological_dense_page():
    # A page with thousands of words (reference list / equation dump) must be
    # skipped before the expensive text-strategy call — never hang the pipeline.
    doc = fitz.open()
    page = doc.new_page()
    y = 40
    line = "Author A 2020 Journal 10 200 some reference text et al pages " * 4
    for _ in range(60):  # ~14k words, far over the guard
        page.insert_text((40, y), line, fontsize=6)
        y += 11
        if y > 780:
            page = doc.new_page()
            y = 40
    page0 = doc[0]
    assert len(page0.get_text("words")) > 1500
    assert reconstruct_table_regions(page0) == []  # guarded, returns fast


def test_extract_structured_leaves_prose_alone():
    doc = fitz.open()
    page = doc.new_page()
    y = 100
    para = "This is an ordinary paragraph of body prose with no tabular structure."
    for _ in range(8):
        page.insert_text((72, y), para, fontsize=11)
        y += 16
    out = BornDigitalDetector().extract_structured(page)
    assert "| --- |" not in out  # prose must never become a table


def test_prose_with_small_embedded_table_is_not_whole_page_gridded():
    """Regression (issue #32): an exercises+references page with one small data
    table made has_numeric_columns fire, then text-strategy find_tables grids the
    WHOLE page, shredding prose/references into character-split cells. The
    data-row-majority guard must reject the over-capture and leave clean prose."""
    doc = fitz.open()
    page = doc.new_page()
    y = 80
    # prose / exercise text
    for line in [
        "3. Consider the following items bought in a supermarket and some of",
        "their characteristics:",
    ]:
        page.insert_text((54, y), line, fontsize=10)
        y += 16
    # a small 3-numeric-column data table (the real signal that trips the gate)
    cols = [60, 130, 200]
    for no, cost, vol in [("1", "20", "6"), ("2", "50", "8"), ("3", "90", "10")]:
        for x, cell in zip(cols, (no, cost, vol)):
            page.insert_text((x, y), cell, fontsize=10)
        y += 16
    y += 8
    # references (single-token wraps that make _detect_columnar_numbers fire)
    for cite in [
        "Alpaydin, E.: Introduction to Machine Learning, 2nd edn. MIT Press (2010)",
        "Bishop, C.M.: Neural Networks for Pattern Recognition. Oxford (2006)",
        "Duda, R.O., Hart, P.E., Stork, D.G.: Pattern Classification. Wiley (2001)",
    ]:
        page.insert_text((54, y), cite, fontsize=10)
        y += 16

    out = BornDigitalDetector().extract_structured(page)
    # the prose must survive intact, not be shredded into a grid
    assert "Consider the following items bought in a supermarket" in out
    assert "| --- |" not in out  # no whole-page fake grid
    import re

    assert not re.search(r"\| [a-z] \|", out)  # no mid-word character-split cells


# ---------------------------------------------------------------------------
# _collapse_header_prefix safety tests (TR-2 reviewer requirements)
# ---------------------------------------------------------------------------


def test_all_na_data_row_not_swallowed():
    """An all-na forecaster row must NEVER be merged into the header prefix.

    The grid here has one empty-col-0 header row (indicator names) followed by
    a row where col 0 = 'Forecaster' AND data cols contain year values —
    that row has non-empty data cells so the scan stops before 'EarlyFirm'.
    'EarlyFirm' is an all-na data row and must remain a data row.
    """
    grid = [
        ["", "GDP", "GDP"],
        ["Forecaster", "2024", "2025"],
        ["EarlyFirm", "", "", ""],
        ["LateFirm", "2.1", "1.8"],
    ]
    data_rows = _collapse_header_prefix(grid)[1:]
    assert any("EarlyFirm" in r[0] for r in data_rows), (
        "all-na first data row was swallowed into the header prefix"
    )


def test_integer_only_data_row_not_swallowed():
    """An integer-only data row must NOT be merged into the header prefix.

    Car Sales has integer values (16, 17) in data cols, so the row is a data
    row and must survive collapse_header_prefix as a body row.
    """
    grid = [
        ["Indicator", "2021", "2022"],
        ["Car Sales", "16", "17"],
    ]
    data_rows = _collapse_header_prefix(grid)[1:]
    assert any("Car Sales" in r[0] for r in data_rows), (
        "integer-only data row was swallowed into the header prefix"
    )


# ---------------------------------------------------------------------------
# GH-330 Task 5: Orientation-aware word rowizer
# ---------------------------------------------------------------------------


def _create_synthetic_table_page(rotation: int = 0) -> tuple[fitz.Document, fitz.Page]:
    """Create a page with a 4-column table and a notes line at the given rotation."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    data = [
        ["Model", "Beta", "SE", "t-stat"],
        ["OLS", "1.25", "0.05", "25.0"],
        ["IV", "1.80", "0.12", "15.0"],
        ["GMM", "1.45", "0.08", "18.1"],
    ]
    notes = "Note: Standard errors clustered at firm level with N = 500."

    if rotation == 0:
        cols = [100, 220, 340, 460]
        y = 100
        for row in data:
            for c, cell in enumerate(row):
                page.insert_text((cols[c], y), cell, fontsize=10, rotate=0)
            y += 25
        page.insert_text((100, y + 20), notes, fontsize=9, rotate=0)
    elif rotation == 90:
        # Rotated 90 degrees clockwise
        x = 520
        rows_y = [100, 220, 340, 460]
        for row in data:
            for c, cell in enumerate(row):
                page.insert_text((x, rows_y[c]), cell, fontsize=10, rotate=90)
            x -= 25
        page.insert_text((x - 20, 100), notes, fontsize=9, rotate=90)
    elif rotation in (270, -90):
        # Rotated 270 degrees (or -90)
        x = 80
        rows_y = [500, 380, 260, 140]
        for row in data:
            for c, cell in enumerate(row):
                page.insert_text((x, rows_y[c]), cell, fontsize=10, rotate=270)
            x += 25
        page.insert_text((x + 20, 500), notes, fontsize=9, rotate=270)
    else:
        raise ValueError(f"Unsupported rotation: {rotation}")

    return doc, page


def test_rowize_from_words_orientation_aware_rotation_matrix():
    """GH-330 Task 5: 0, +90, and -90 degree tables produce equivalent strict grids.

    All three orientations:
      1. Produce exactly one strict-parsable grid.
      2. Preserve identical cell contents and reading order across all three.
      3. Retain every numeric token exactly once (lossless).
      4. Return bounding rects enclosing the original page-coordinate words.
    """
    from socr.tables.binding import parse_grid
    from socr.tables.reconstruct import rowize_from_word_list

    expected_cells = [
        ("Model", "Beta", "SE", "t-stat"),
        ("OLS", "1.25", "0.05", "25.0"),
        ("IV", "1.80", "0.12", "15.0"),
        ("GMM", "1.45", "0.08", "18.1"),
    ]
    expected_numerics = {"1.25", "0.05", "1.80", "0.12", "1.45", "0.08", "25.0", "15.0", "18.1"}

    for rot in (0, 90, 270):
        doc, page = _create_synthetic_table_page(rotation=rot)
        words = page.get_text("words")
        assert words, f"Page must contain words for rotation={rot}"

        try:
            regions = rowize_from_word_list(words, rotation=rot, page_rect=page.rect)
        except TypeError:
            # If rotation / page_rect kwargs are not yet supported on rowize_from_word_list
            pytest.xfail(f"rowize_from_word_list does not yet accept rotation={rot}")

        # GH-350: no empty-regions escape. A rowizer that returns nothing is
        # BROKEN, and xfailing on it made breakage indistinguishable from a
        # feature that is not built yet. The TypeError branch above is the
        # legitimate not-yet-supported signal; this is not.
        assert regions, f"rowize_from_word_list returned no regions for rotation={rot}"

        rect, md = regions[0]
        grid = parse_grid(md)
        assert grid is not None, f"Emitted region must parse as strict grid for rotation={rot}"

        all_rows = list(grid.header_rows) + list(grid.rows)
        assert len(all_rows) == len(expected_cells)
        for actual_row, expected_row in zip(all_rows, expected_cells):
            assert tuple(actual_row) == expected_row, (
                f"Row mismatch for rotation={rot}: got {actual_row}, expected {expected_row}"
            )

        found_numerics = set()
        for row in grid.rows:
            for cell in row:
                if cell in expected_numerics:
                    found_numerics.add(cell)
        assert found_numerics == expected_numerics, (
            f"Missing numeric tokens for rotation={rot}: {expected_numerics - found_numerics}"
        )

        assert rect.is_valid and not rect.is_empty
        doc.close()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "GH-350: a single rotated marginal note currently destroys rowization of "
        "the whole table -- 1 region without the note, 0 with it. The old test "
        "xfailed on empty regions, so this defect read as green. Strict, so it "
        "fails loudly the day the behaviour starts holding."
    ),
)
def test_mixed_horizontal_page_with_rotated_marginal_note_stays_horizontal():
    """GH-330 Task 5: A single rotated marginal note does not flip page orientation.

    Dominant text direction on the page remains horizontal, so the table is
    processed in the horizontal frame.
    """
    from socr.tables.binding import parse_grid
    from socr.tables.reconstruct import rowize_from_word_list

    doc, page = _create_synthetic_table_page(rotation=0)
    page.insert_text((550, 200), "Running Header 2026", fontsize=8, rotate=90)

    words = page.get_text("words")
    try:
        regions = rowize_from_word_list(words, rotation=0, page_rect=page.rect)
    except TypeError:
        regions = rowize_from_word_list(words)

    # GH-350: see above -- empty regions is breakage, not an unbuilt feature.
    assert regions, "rowize_from_word_list returned no regions"

    _rect, md = regions[0]
    grid = parse_grid(md)
    assert grid is not None
    assert "OLS" in md and "1.25" in md and "25.0" in md
    doc.close()


def test_rotated_table_production_refusal_preserved(tmp_path: Path):
    """GH-330 Task 5: Production routing refusal for rotated table pages is unchanged.

    Rotated table pages are not trusted as native markdown directly in production;
    they remain routed to OCR / flagged fail-closed per GH-147 / GH-263.
    """
    doc, _page = _create_synthetic_table_page(rotation=90)
    pdf_path = tmp_path / "rotated.pdf"
    doc.save(str(pdf_path))
    doc.close()

    detector = BornDigitalDetector()
    assessment = detector.detect_page(pdf_path, 1)
    assert assessment is not None
    assert assessment.text_is_rotated is True
