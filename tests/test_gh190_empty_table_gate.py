"""GH-190: an all-empty but structurally-valid table passes validation.

A table that is structurally well-formed but entirely content-free (e.g. empty cells,
whitespace, or dash/hyphen placeholders) previously passed all validators in the pipeline
because every validator in the path was shape-only:

1. ``reconcile._parse_grid`` drops blank rows through its vacuous separator test
   ``all(_SEP_CELL.match(c.strip()) for c in cells if c.strip())`` (an empty generator
   yields True), so ``check_markdown`` returned no defective report.
2. ``table_emission_defect`` requires an authored header (``any(cell.strip() for cell in rows[0])``),
   so an all-blank table was skipped, and a populated-header/empty-body table was accepted
   because delimiter width matched and no LaTeX table commands leaked.
3. ``table_output_defect`` returned ``DEFECT_NONE`` (""), allowing empty tables to ship
   with PageStatus.SUCCESS and audit_passed=True -- violating CLAUDE.md's "no silent content loss".

This test module freezes:
- The pre-fix mechanism and blind path reproduction.
- Paired difference assertions (load-bearing difference on the gate between an empty table
  and the same table with exactly one value).
- The firing matrix across blank cells, whitespace, adjacent pipes, ASCII hyphens, Unicode dashes,
  mixed placeholders, populated header over empty body, and multi-table documents.
- The non-firing matrix across legitimate sparse tables, blank headers over populated bodies,
  symbol-only cells (currency, significance stars, daggers), NA/0 cells,
  fences, comments, indented code, raw HTML literal blocks, prose, and None/empty inputs.
- Pure predicate behavior for ``table_content_defect``.
- Precedence among emission defects (LaTeX leaks, width mismatches), content defects, and grid shape.
- Load-bearing seam control using concrete return value patching.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from socr.tables.reconcile import (
    TABLE_EMISSION_LATEX_LEAK,
    TABLE_EMISSION_NONE,
    TABLE_EMISSION_WIDTH_MISMATCH,
    _parse_grid,
    table_emission_defect,
)

try:
    from socr.tables.reconcile import TABLE_CONTENT_EMPTY, table_content_defect
except ImportError:
    TABLE_CONTENT_EMPTY = "table_content_empty"
    table_content_defect = None

try:
    from socr.tables.structure_check import DEFECT_TABLE_CONTENT_EMPTY
except ImportError:
    DEFECT_TABLE_CONTENT_EMPTY = "table_content_empty"

from socr.tables.structure_check import (
    DEFECT_GRID_SHAPE,
    DEFECT_NONE,
    DEFECT_TABLE_LATEX_LEAK,
    DEFECT_TABLE_WIDTH_MISMATCH,
    check_markdown,
    table_output_defect,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

# Issue #190 literal reproduction fixture from Qwen 3.5:27b-mlx evaluation:
ISSUE_LITERAL_EMPTY_TABLE = (
    "| | | | | | | | | | |\n|---|---|---|---|---|---|---|---|---|---|\n| | | | | | | | | | |\n"
)

# Paired control fixture differing in EXACTLY ONE body cell:
PAIRED_ONE_VALUE_TABLE = (
    "| | | | | | | | | | |\n|---|---|---|---|---|---|---|---|---|---|\n| 1 | | | | | | | | | |\n"
)

# Populated header over blank body:
POPULATED_HEADER_EMPTY_BODY = "| Col A | Col B | Col C |\n| --- | --- | --- |\n| | | |\n| | | |\n"


# --------------------------------------------------------------------------
# Task t1: Freeze the pre-fix mechanism and blind path
# --------------------------------------------------------------------------


def test_pre_fix_blind_path_mechanism() -> None:
    """Pin the exact mechanism that allowed empty tables to pass validation.

    1. reconcile._parse_grid treats blank rows as separator rows because
       all(_SEP_CELL.match(c.strip()) for c in cells if c.strip()) is vacuously True.
    2. check_markdown therefore finds no table blocks for the issue literal.
    3. table_emission_defect skips blank-header tables and accepts populated-header/empty-body
       tables.
    """
    # 1. _parse_grid drops blank header and blank body rows
    parsed = _parse_grid(["| | | |", "|---|---|", "| | | |"])
    assert parsed == [], (
        f"Expected _parse_grid to drop all-blank rows via vacuous separator match, got {parsed!r}"
    )

    # 2. check_markdown returns no reports for the issue literal
    reports = check_markdown(ISSUE_LITERAL_EMPTY_TABLE)
    assert reports == [], f"Expected check_markdown to return no reports, got {reports!r}"

    # 3. table_emission_defect skips the issue literal (blank header) and accepts empty body
    assert table_emission_defect(ISSUE_LITERAL_EMPTY_TABLE) == TABLE_EMISSION_NONE
    assert table_emission_defect(POPULATED_HEADER_EMPTY_BODY) == TABLE_EMISSION_NONE


def test_paired_difference_assertion_on_gate() -> None:
    """CLAUDE.md load-bearing difference assertion: assert a difference between outcomes.

    The gate must treat an all-empty grid and a paired grid with exactly one populated value
    differently.
    """
    defect_empty = table_output_defect(ISSUE_LITERAL_EMPTY_TABLE, None, None)
    defect_pop = table_output_defect(PAIRED_ONE_VALUE_TABLE, None, None)

    assert defect_empty != defect_pop, "Gate must produce different verdicts for empty vs populated"
    assert defect_empty == DEFECT_TABLE_CONTENT_EMPTY, (
        f"Expected {DEFECT_TABLE_CONTENT_EMPTY!r}, got {defect_empty!r}"
    )
    assert defect_pop == DEFECT_NONE, f"Expected {DEFECT_NONE!r}, got {defect_pop!r}"


# --------------------------------------------------------------------------
# Firing Matrix: all empty and placeholder variants
# --------------------------------------------------------------------------

FIRING_CASES = [
    # Issue literal
    ("issue_literal", ISSUE_LITERAL_EMPTY_TABLE),
    # Blank cells
    ("blank_2x2", "| | |\n|---|---|\n| | |"),
    ("blank_3x3", "| | | |\n|---|---|---|\n| | | |\n| | | |"),
    # Whitespace-only cells (spaces and tabs)
    ("whitespace_spaces_tabs", "|   |  \t |\n|---|---|\n| \t  |   |"),
    ("whitespace_3col", "|  |   |    |\n|---|---|---|\n|    |   |  |"),
    # Adjacent pipes empty strings
    ("adjacent_pipes_2col", "|||\n|---|---|\n|||"),
    ("adjacent_pipes_3col", "||||\n|---|---|---|\n||||"),
    ("adjacent_pipes_mixed_spacing", "|| |\n|---|---|\n| ||"),
    # Single and repeated ASCII hyphens
    ("ascii_single_hyphen", "| - | - |\n|---|---|\n| - | - |"),
    ("ascii_double_hyphen", "| -- | -- |\n|---|---|\n| -- | -- |"),
    ("ascii_triple_hyphen", "| --- | --- |\n|---|---|\n| --- | --- |"),
    ("ascii_quad_hyphen", "| ---- | ---- |\n|---|---|\n| ---- | ---- |"),
    # Unicode dash variants
    ("unicode_em_dash", "| — | — |\n|---|---|\n| — | — |"),
    ("unicode_en_dash", "| – | – |\n|---|---|\n| – | – |"),
    ("unicode_minus_sign", "| − | − |\n|---|---|\n| − | − |"),
    ("unicode_horizontal_bar", "| ― | ― |\n|---|---|\n| ― | ― |"),
    ("unicode_figure_dash", "| ‒ | ‒ |\n|---|---|\n| ‒ | ‒ |"),
    ("unicode_small_em_dash", "| ﹣ | ﹣ |\n|---|---|\n| ﹣ | ﹣ |"),
    ("unicode_fullwidth_hyphen", "| － | － |\n|---|---|\n| － | － |"),
    # Mixed blank and dash placeholders
    ("mixed_blank_and_em_dash", "|   | - |\n|---|---|\n| — |   |"),
    ("mixed_multi_dash_types", "| - |   | -- |\n|---|---|---|\n|   | — | − |"),
    # Populated header over empty or placeholder body
    ("populated_header_blank_body", "| Header 1 | Header 2 |\n| --- | --- |\n| | |"),
    ("populated_header_dash_body", "| Col A | Col B | Col C |\n| --- | --- | --- |\n| - |   | — |"),
    ("populated_header_multi_row_empty", "| Year | Revenue |\n| --- | --- |\n| | |\n| - | - |"),
    # Borderless pipe table forms
    ("borderless_blank_body", "A | B\n--- | ---\n | "),
    ("borderless_dash_body", "A | B\n--- | ---\n- | -"),
    # Width-mismatched content-free bodies (cold review item 1)
    ("borderless_2col_blank_pipes", "A | B\n--- | ---\n | |"),
    ("borderless_3col_blank_pipes", "A | B | C\n--- | --- | ---\n | | "),
    ("bordered_narrower_than_header", "| A | B | C |\n| --- | --- | --- |\n| | |"),
    ("bordered_wider_than_header", "| A | B |\n| --- | --- |\n| | | |"),
    # Copied alignment-marker rows used as bodies (cold review item 3)
    ("alignment_marker_left", "| A | B |\n| :--- | :--- |\n| :--- | :--- |"),
    ("alignment_marker_center", "| A | B |\n| :---: | :---: |\n| :---: | :---: |"),
    ("alignment_marker_right", "| A | B |\n| ---: | ---: |\n| ---: | ---: |"),
    (
        "alignment_marker_mixed",
        "| A | B | C |\n| :--- | :---: | ---: |\n| :--- | :---: | ---: |",
    ),
    # Header + separator block with no body rows (cold review item 2)
    ("header_separator_no_body", "| A | B |\n| --- | --- |"),
    ("header_separator_no_body_spaces", "| Col1 | Col2 | Col3 |\n| --- | --- | --- |"),
    ("blank_header_no_body", "| | |\n|---|---|"),
    # Empty table appearing after an earlier valid table
    (
        "empty_after_valid_table",
        "| Col1 | Col2 |\n| --- | --- |\n| val1 | val2 |\n\n| H1 | H2 |\n| --- | --- |\n| | |",
    ),
    # Empty table appearing before a valid table
    (
        "empty_before_valid_table",
        "| H1 | H2 |\n| --- | --- |\n| - | - |\n\n| A | B |\n| --- | --- |\n| 1 | 2 |",
    ),
]


@pytest.mark.parametrize("name,emitted", FIRING_CASES, ids=[c[0] for c in FIRING_CASES])
def test_firing_matrix_covers_all_empty_and_placeholder_variants(name: str, emitted: str) -> None:
    """Every empty or dash-placeholder table must be rejected by table_output_defect."""
    assert table_output_defect(emitted, None, None) == DEFECT_TABLE_CONTENT_EMPTY


# --------------------------------------------------------------------------
# Non-Firing Matrix: legitimate content, sparse tables, and controls
# --------------------------------------------------------------------------

NON_FIRING_CASES = [
    # Sparse tables with real values
    ("sparse_one_value_bottom_right", "| A | B |\n| --- | --- |\n| | 1 |"),
    ("sparse_one_value_middle", "| A | B | C |\n| --- | --- | --- |\n| | | |\n| | 42 | |"),
    ("sparse_with_dash_placeholders", "| A | B |\n| --- | --- |\n| - | - |\n| - | value |"),
    # Blank header over populated body
    ("blank_header_populated_body", "| | |\n|---|---|\n| 1 | 2 |"),
    ("blank_header_sparse_body", "| | | |\n|---|---|---|\n| | data | |"),
    # Standard missing data notation and numeric zeros (content!)
    ("na_cells_na_and_dot", "| A | B |\n| --- | --- |\n| n.a. | NA |"),
    ("na_cells_slash_and_none", "| A | B |\n| --- | --- |\n| N/A | None |"),
    ("zero_numeric_cells", "| A | B |\n| --- | --- |\n| 0 | 0.0 |"),
    ("mixed_zero_and_placeholder", "| A | B |\n| --- | --- |\n| - | 0 |"),
    # Symbol-only body cells (currency, significance stars, daggers)
    (
        "symbols_currency_and_stars",
        "| A | B | C | D |\n| --- | --- | --- | --- |\n| $ | * | % | † |",
    ),
    ("symbols_euro_and_triple_star", "| Currency | Note |\n| --- | --- |\n| € | *** |"),
    ("symbols_pound_and_double_dagger", "| Sym1 | Sym2 |\n| --- | --- |\n| £ | ‡ |"),
    ("symbols_yen_and_hash", "| S1 | S2 |\n| --- | --- |\n| ¥ | # |"),
    # Boundary controls for colons and non-placeholder colon patterns (cold review item 3)
    ("colon_single", "| A | B |\n| --- | --- |\n| : | : |"),
    ("colon_double", "| A | B |\n| --- | --- |\n| :: | :: |"),
    ("colon_ratio", "| A | B |\n| --- | --- |\n| 1:2 | 3:4 |"),
    ("colon_sub_delimiter_hyphens", "| A | B |\n| --- | --- |\n| :-: | :-: |"),
    ("colon_sub_delimiter_hyphen_left_right", "| A | B |\n| --- | --- |\n| :- | -: |"),
    # Fenced code blocks containing empty tables
    (
        "fenced_backticks",
        "```markdown\n| | |\n|---|---|\n| | |\n```",
    ),
    (
        "fenced_tildes",
        "~~~markdown\n| | |\n|---|---|\n| | |\n~~~",
    ),
    # Commented markdown
    (
        "commented_table",
        "<!--\n| | |\n|---|---|\n| | |\n-->",
    ),
    # Indented code blocks
    (
        "indented_code",
        "    | | |\n    |---|---|\n    | | |",
    ),
    # Raw HTML literal blocks
    (
        "raw_html_pre",
        "<pre>\n| | |\n|---|---|\n| | |\n</pre>",
    ),
    (
        "raw_html_textarea",
        "<textarea>\n| | |\n|---|---|\n| | |\n</textarea>",
    ),
    # Prose with pipe characters
    ("prose_with_pipes", "Prose with pipe | characters | and no table delimiter."),
    ("prose_regular", "Ordinary text describing data without any table markup."),
    # Empty string
    ("empty_string", ""),
]


@pytest.mark.parametrize("name,emitted", NON_FIRING_CASES, ids=[c[0] for c in NON_FIRING_CASES])
def test_non_firing_matrix_covers_all_legitimate_content_and_controls(
    name: str, emitted: str
) -> None:
    """Legitimate sparse tables, symbols, and non-table blocks must produce DEFECT_NONE."""
    assert table_output_defect(emitted, None, None) == DEFECT_NONE


# --------------------------------------------------------------------------
# Task t2: Direct table_content_defect predicate tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,emitted", FIRING_CASES, ids=[c[0] for c in FIRING_CASES])
def test_table_content_defect_direct_firing(name: str, emitted: str) -> None:
    """table_content_defect directly returns TABLE_CONTENT_EMPTY for all firing cases."""
    assert table_content_defect is not None, "table_content_defect must be implemented"
    assert table_content_defect(emitted) == TABLE_CONTENT_EMPTY


@pytest.mark.parametrize(
    "name,emitted",
    NON_FIRING_CASES + [("none_input", None)],
    ids=[c[0] for c in NON_FIRING_CASES] + ["none_input"],
)
def test_table_content_defect_direct_non_firing(name: str, emitted: str | None) -> None:
    """table_content_defect directly returns '' for all non-firing controls including None."""
    assert table_content_defect is not None, "table_content_defect must be implemented"
    assert table_content_defect(emitted) == ""


def test_table_content_defect_structural_preconditions() -> None:
    """table_content_defect checks candidate structure before reporting empty body.

    A candidate requires:
    - Delimiter strictly at row index 1.
    - Strict delimiter (>= 3 hyphens per cell).
    - Matching border style between header and delimiter.
    - Matching column widths across header and delimiter.
    """
    assert table_content_defect is not None, "table_content_defect must be implemented"

    # Delimiter not at row index 1 -> not a standard candidate
    del_at_row_2 = "| A | B |\n| 1 | 2 |\n| --- | --- |\n| | |"
    assert table_content_defect(del_at_row_2) == ""

    # Border style mismatch between header and delimiter
    border_mismatch = "| A | B |\n--- | ---\n| | |"
    assert table_content_defect(border_mismatch) == ""

    # Zero body rows -> candidate is rejected as content-empty (vacuous empty body)
    zero_body = "| A | B |\n| --- | --- |"
    assert table_content_defect(zero_body) == TABLE_CONTENT_EMPTY


def test_table_content_defect_three_hyphen_body_is_not_treated_as_delimiter() -> None:
    """A body row made of triple hyphens is an empty placeholder row, not a second delimiter."""
    assert table_content_defect is not None, "table_content_defect must be implemented"

    triple_hyphen_body = "| A | B |\n| --- | --- |\n| --- | --- |"
    assert table_content_defect(triple_hyphen_body) == TABLE_CONTENT_EMPTY


# --------------------------------------------------------------------------
# Task t3: Precedence and Seam Control
# --------------------------------------------------------------------------


def test_precedence_latex_leak_over_content_empty() -> None:
    """table_emission_defect runs first: residual LaTeX leak takes precedence over empty body."""
    emitted = r"| A | \multicolumn{2}{c}{B} |" + "\n| --- | --- |\n| | |"

    assert table_emission_defect(emitted) == TABLE_EMISSION_LATEX_LEAK
    assert table_output_defect(emitted, None, None) == DEFECT_TABLE_LATEX_LEAK


def test_precedence_width_mismatch_over_content_empty() -> None:
    """table_emission_defect runs first: delimiter width mismatch takes precedence."""
    emitted = "| A | B | C |\n| --- | --- |\n| | | |"

    assert table_emission_defect(emitted) == TABLE_EMISSION_WIDTH_MISMATCH
    assert table_output_defect(emitted, None, None) == DEFECT_TABLE_WIDTH_MISMATCH


def test_precedence_ragged_populated_table_returns_grid_shape() -> None:
    """A populated ragged table keeps the existing grid_shape defect."""
    emitted = "| A | B |\n| --- | --- |\n| 1 | 2 | 3 |"

    assert table_emission_defect(emitted) == TABLE_EMISSION_NONE
    assert table_content_defect(emitted) == ""
    assert table_output_defect(emitted, None, None) == DEFECT_GRID_SHAPE

    # Multi-row populated ragged case
    multi_ragged = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 | 5 |"
    assert table_emission_defect(multi_ragged) == TABLE_EMISSION_NONE
    assert table_content_defect(multi_ragged) == ""
    assert table_output_defect(multi_ragged, None, None) == DEFECT_GRID_SHAPE


def test_load_bearing_seam_control_patching_table_content_defect() -> None:
    """Load-bearing seam control: patching only table_content_defect restores DEFECT_NONE.

    Proves that the rejection is caused strictly by the new content term and not by an
    unintended side-effect in existing checks.
    """
    empty_fixture = "| Col1 | Col2 |\n| --- | --- |\n| | |\n"

    # Real unpatched evaluation produces DEFECT_TABLE_CONTENT_EMPTY
    assert table_output_defect(empty_fixture, None, None) == DEFECT_TABLE_CONTENT_EMPTY

    # Patching table_content_defect to return "" restores DEFECT_NONE
    with patch("socr.tables.structure_check.table_content_defect", return_value=""):
        assert table_output_defect(empty_fixture, None, None) == DEFECT_NONE


def test_no_bare_magic_mock_comparisons() -> None:
    """Guard against vacuously true assertions: all comparison operands must be concrete."""
    result = table_output_defect(ISSUE_LITERAL_EMPTY_TABLE, None, None)
    assert isinstance(result, str)
    assert result == DEFECT_TABLE_CONTENT_EMPTY
