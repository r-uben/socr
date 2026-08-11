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

import fitz

from socr.tables.header_repair import repair_collapsed_header, repair_table_headers_on_page
from socr.tables.reconcile import find_table_blocks
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


def test_decorated_values_are_still_values():
    """Reviewer finding on #149: the anchored `_NUM_TOKEN_RE` rejects every one
    of these, so a starred coefficient row — the common shape in an econometrics
    table — would have been promoted to header despite carrying observations."""
    assert _is_data_row(["Surprise", "0.67***", ""])
    assert _is_data_row(["Surprise", "-0.253*", ""])
    assert _is_data_row(["Section total", "**23,126**", ""])
    assert _is_data_row(["Deficit", "−0.253", ""])  # U+2212
    assert _is_data_row(["Revenue", "£43.2", ""])


def test_word_header_survives_presentation_stripping():
    """Emphasis stripping must not turn a bold word header into data."""
    assert not _is_data_row(["Firm", "**Nominal**", "**Real**"])


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


def test_empty_row_is_not_data():
    """`_is_header_row` guards the empty case, so no IndexError reaches here."""
    assert not _is_data_row([])


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


# ---------------------------------------------------------------------------
# assume_header — the header_repair seam
# ---------------------------------------------------------------------------


def test_assume_header_keeps_a_numeric_shaped_repaired_header():
    """Reviewer finding on #149: `header_repair` rebuilds row 0 from native word
    geometry and gates it on `_header_is_faithful`. Re-inferring there would
    demote a numeric-shaped header band and silently discard the repair."""
    repaired = [["Firm", "2024", "2025"], ["Ashford", "2.1", "1.9"]]

    # Without the opt-out, inference demotes it — this is why the flag exists.
    assert _rows(_grid_to_markdown(repaired))[0] == "|  |  |  |"

    md = _grid_to_markdown(repaired, assume_header=True)
    assert _rows(md)[0] == "| Firm | 2024 | 2025 |"
    assert _body_cells(md) == [["Ashford", "2.1", "1.9"]]


def test_header_repair_survives_demotion_end_to_end():
    """The regression through the real repair path, on synthetic CE-style geometry.

    `_is_table_header_row` requires a range marker, so the reachable case is a
    percentage-bin band that also prints its label-column name (`Currency`) on
    the header line: every data cell is then a bare numeric token and column 0
    is non-empty — exactly `_is_data_row`. Without the opt-out the repaired
    header lands in the body and the table is left with no schema at all.
    """
    label_x = 40.0
    data_xs = [label_x + 110.0 + i * 55.0 for i in range(7)]
    doc = fitz.open()
    page = doc.new_page(width=700, height=400)
    page.insert_text((label_x, 100), "Currency", fontsize=9)
    for x, tok in zip(data_xs, ["-23%", "-14%", "-5%", "+5%", "+14%", "+23%", "+30%"]):
        page.insert_text((x, 100), tok, fontsize=9)
    page.insert_text((label_x, 140), "Euro1", fontsize=9)
    for x, val in zip(data_xs, ["1", "2", "16", "49", "23", "8", "1"]):
        page.insert_text((x, 140), val, fontsize=9)
    page.insert_text((label_x, 170), "Yen1", fontsize=9)
    for x, val in zip(data_xs, ["0", "3", "15", "40", "29", "9", "4"]):
        page.insert_text((x, 170), val, fontsize=9)

    collapsed_md = "\n".join(
        [
            "| Currency -23% -14% -5% +5% +14% +23% +30% | Side |",
            "| --- | --- |",
            "| Euro1 | 1 | 2 | 16 | 49 | 23 | 8 | 1 |",
            "| Yen1 | 0 | 3 | 15 | 40 | 29 | 9 | 4 |",
        ]
    )
    grid = find_table_blocks(collapsed_md)[0].grid
    repaired = repair_collapsed_header(grid, page.get_text("words"))

    assert repaired is not None, "fixture no longer exercises the repair path"
    assert repaired[0] == [
        "Currency",
        "-23%",
        "-14%",
        "-5%",
        "+5%",
        "+14%",
        "+23%",
        "+30%",
    ]
    # The repaired header is indistinguishable from data by shape — which is
    # precisely why the repair path must assert what it knows.
    assert _is_data_row(repaired[0])

    new_md, count = repair_table_headers_on_page(page, collapsed_md)
    assert count == 1
    assert _rows(new_md)[0] == ("| Currency | -23% | -14% | -5% | +5% | +14% | +23% | +30% |")
    assert _body_cells(new_md) == [
        ["Euro1", "1", "2", "16", "49", "23", "8", "1"],
        ["Yen1", "0", "3", "15", "40", "29", "9", "4"],
    ]
