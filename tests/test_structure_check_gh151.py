"""Pure-function tests for socr.tables.structure_check (GH-151 TICKET-A1).

No pipeline imports, no ``_phase_agentic``/``process()`` coverage — this
module is a pure diagnostic and the test file stays pure-function so no
``_available_engines_for_agentic`` patching is ever needed. Keep it that way.
"""

from __future__ import annotations

import pytest

from socr.tables.structure_check import check_grid, check_markdown

# Issue #151 Table 3 markdown, verbatim (macron, circumflex, Unicode minus,
# mismatched 7-cell separator, as shipped). Do not "fix" the characters.
GH151_P26_MD = (
    "|  |  |  | ¯Etxr(6) t+12 |  |  | ¯Etxr(11) t+12 |  |\n"
    "| --- | --- | --- | --- | --- | --- | --- |\n"
    "| Panel A: | Baseline | ˆyt |  |  |  |\n"
    "| ˆγ | −1.93*** | −2.24*** | −2.61*** | −3.02** | −3.49*** | −3.79*** |\n"
    "|  | (0.55) | (0.61) | (0.37) | (1.24) | (1.31) | (0.55) |\n"
    "| TERM |  | 0.32* |  |  | 0.51 |  |\n"
    "| R2 |  |  |  |  |  |  |\n"
    "|  | 0.12 | 0.16 | 0.61 | 0.09 | 0.12 | 0.61 |"
)


def test_p26_grid_reported_defective():
    reports = check_markdown(GH151_P26_MD)
    assert len(reports) == 1
    report = reports[0]
    assert report.defective is True
    assert report.ragged is True
    assert report.row_widths == (8, 6, 7, 7, 7, 7, 7)
    assert report.orphan_rows == (3, 6)
    assert report.findings == ("ragged", "orphan_rows")
    assert report.empty_cells == 21
    assert report.total_cells == 49
    assert report.footprint_cells == 56
    assert report.empty_cell_ratio == pytest.approx(21 / 49)


def test_clean_rectangular_grid_not_defective():
    grid = [
        ["", "col1", "col2", "col3"],
        ["row1", "a", "b", "c"],
        ["row2", "d", "e", "f"],
    ]
    report = check_grid(grid)
    assert report.defective is False
    assert report.ragged is False
    assert report.orphan_rows == ()
    assert report.findings == ()


def test_sparse_rectangular_grid_density_never_votes():
    grid = [
        ["", "col1", "col2", "col3"],
        ["row1", "", "", ""],
        ["row2", "", "d", ""],
        ["row3", "e", "", ""],
    ]
    report = check_grid(grid)
    assert report.empty_cell_ratio > 0
    assert report.defective is False
    assert report.findings == ()


def test_header_row_blank_label_never_orphan():
    report = check_grid([["", "(1)", "(2)"], ["x", "1", "2"]])
    assert report.orphan_rows == ()
    assert report.defective is False


def test_check_markdown_prose_plus_one_table():
    md = "Some intro prose.\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\nMore prose after."
    reports = check_markdown(md)
    assert len(reports) == 1


def test_empty_and_single_row_grids():
    empty_report = check_grid([])
    assert empty_report.row_widths == ()
    assert empty_report.empty_cell_ratio == 0.0
    assert empty_report.defective is False
    assert empty_report.orphan_rows == ()

    single_report = check_grid([["a", "b"]])
    assert single_report.row_widths == (2,)
    assert single_report.defective is False
    assert single_report.orphan_rows == ()


def test_separator_free_contract():
    grid = [["a", "b"], ["---", "---"], ["c"]]
    report = check_grid(grid)
    assert report.row_widths == (2, 2, 1)
    assert report.ragged is True


def test_blankness_is_whitespace_only():
    grid = [
        ["", "col1", "col2"],
        ["$", "1", "2"],
        ["**", "3", "4"],
    ]
    report = check_grid(grid)
    assert report.orphan_rows == ()
    assert report.empty_cells == 1  # only the header label cell


def test_last_row_blank_label_is_orphan():
    grid = [
        ["", "col1", "col2"],
        ["row1", "1", "2"],
        ["", "3", "4"],
    ]
    report = check_grid(grid)
    assert report.orphan_rows == (2,)
