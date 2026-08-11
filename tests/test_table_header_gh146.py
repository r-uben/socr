"""GH-146: a data row must never be promoted to the table header.

``_grid_to_markdown`` used to take ``grid[0]`` as the header unconditionally.
When the column-header band was missing from the grid — the reference case is
Nakamura & Steinsson (WP) p.13 Table I, whose header line sits above the
rowizer's region — the first *data* row became the schema:

    | 3M Treasury yield | 0.67 |  |  |
    | --- | --- | --- | --- |
    |  | (0.14) |  |  |

The values are all present, so a numeric-multiset check passes, yet the table
declares a wrong schema and has lost an observation from its body. These tests
pin the lossless behaviour: emit an empty header row and keep row 0 in the body.
"""

from socr.tables.reconstruct import _grid_to_markdown, _is_data_row


def _rows(md: str) -> list[str]:
    return md.splitlines()


def _body_cells(md: str) -> list[list[str]]:
    """Cells of every row below the separator line."""
    lines = _rows(md)
    return [[c.strip() for c in line.strip("|").split("|")] for line in lines[2:]]


# ---------------------------------------------------------------------------
# _is_data_row
# ---------------------------------------------------------------------------


def test_label_plus_value_is_a_data_row():
    assert _is_data_row(["3M Treasury yield", "0.67", "", ""])


def test_parenthesised_standard_error_is_a_value():
    assert _is_data_row(["Slope", "(0.14)", ""])


def test_percent_and_thousands_separators_are_values():
    assert _is_data_row(["Share", "45%", ""])
    assert _is_data_row(["Employment", "1,204", ""])


def test_word_header_with_named_label_column_is_not_data():
    """A header may name its label column; its cells are words, not values."""
    assert not _is_data_row(["Firm", "Nominal", "Real"])


def test_column_metadata_row_is_not_data():
    """Empty col 0 — the ``Nominal Real Inflation`` band itself."""
    assert not _is_data_row(["", "Nominal", "Real", "Inflation"])


def test_label_only_row_is_not_data():
    assert not _is_data_row(["Forecaster", "", "", ""])


def test_row_with_no_data_columns_is_not_data():
    assert not _is_data_row(["0.67"])


# ---------------------------------------------------------------------------
# _grid_to_markdown
# ---------------------------------------------------------------------------


def test_headerless_grid_keeps_every_data_row_in_the_body():
    grid = [
        ["3M Treasury yield", "0.67", "", ""],
        ["", "(0.14)", "", ""],
    ]
    md = _grid_to_markdown(grid)
    lines = _rows(md)

    # Empty header, correct arity, separator intact.
    assert lines[0] == "|  |  |  |  |"
    assert lines[1] == "| --- | --- | --- | --- |"

    # No row was consumed as structure.
    assert _body_cells(md) == [
        ["3M Treasury yield", "0.67", "", ""],
        ["", "(0.14)", "", ""],
    ]


def test_no_value_is_lost_when_the_header_band_is_missing():
    grid = [
        ["3M Treasury yield", "0.67", "", ""],
        ["", "(0.14)", "", ""],
    ]
    md = _grid_to_markdown(grid)
    body = "\n".join(_rows(md)[2:])
    for value in ("3M Treasury yield", "0.67", "(0.14)"):
        assert value in body


def test_real_header_is_still_promoted():
    grid = [["Firm", "Nominal", "Real"], ["Ashford", "2.1", "1.9"]]
    md = _grid_to_markdown(grid)
    assert _rows(md)[0] == "| Firm | Nominal | Real |"
    assert _body_cells(md) == [["Ashford", "2.1", "1.9"]]


def test_column_metadata_header_is_still_promoted():
    grid = [["", "Nominal", "Real"], ["3M Treasury yield", "0.67", "0.41"]]
    md = _grid_to_markdown(grid)
    assert _rows(md)[0] == "|  | Nominal | Real |"
    assert _body_cells(md) == [["3M Treasury yield", "0.67", "0.41"]]


def test_pipes_in_a_demoted_first_row_are_escaped():
    grid = [["a|b", "0.67"], ["", "(0.14)"]]
    md = _grid_to_markdown(grid)
    assert r"a\|b" in md
    assert _rows(md)[0] == "|  |  |"
