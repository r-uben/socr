"""#1257 (plan B1): positive row-binding evidence beyond the same-line rule.

Four exact kinds may move a row or heading from UNVERIFIABLE to BOUND and nothing else:

* E1 wrapped label: the shipped label is the exact join of consecutive printed fragments;
* E2 stacked heading: every heading cell is exactly the words stacked over its data column;
* E3 whole line: the row's label and cells are the printed line, word for word;
* completeness: none of them binds a table that left a row-shaped line without a shipped row.

Every test pins a DIFFERENCE: the same page and markdown with the new evidence on and off (the
evidence functions patched out), so the only change between arms is the evidence.  An adversary
must give the same verdict in both arms.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import fitz
import pytest
from native_table_fixtures import place
from test_gh1267_row_binding_check import FONT, HEIGHT, LABEL_U, VALUE_STEP, VALUE_U0, WIDTH

from socr.tables import row_binding_check as rbc
from socr.tables.row_binding_check import (
    EVIDENCE_STACKED_HEADING,
    EVIDENCE_WHOLE_LINE,
    EVIDENCE_WRAPPED_LABEL,
    RowVerdict,
    TableOutcome,
)

ROTATIONS = (0, 90, 270)
PITCH = 18.0  # between rows
WRAP = 11.0  # between the lines of a wrapped label
Y0 = 100.0

# ---------------------------------------------------------------------------
# Page builder: free layout, any rotation
# ---------------------------------------------------------------------------


def page_of(items: list[tuple[float, float, str]], rotation: int) -> fitz.Page:
    """*items* are ``(u, v, text)`` in the upright frame."""
    doc = fitz.open()
    page = doc.new_page(width=WIDTH, height=HEIGHT)
    for u, v, s in items:
        page.insert_text(
            place(u, v, rotation, WIDTH, HEIGHT), s, fontsize=FONT, fontname="helv", rotate=rotation
        )
    return page


def col(k: int) -> float:
    return VALUE_U0 + k * VALUE_STEP


def layout(rows: list[tuple[list[str], list[str], int]]) -> list[tuple[float, float, str]]:
    """``rows`` = ``(label lines, values, index of the label line the values sit on)``.  Lines of one
    label are WRAP apart, rows PITCH apart (from the last line of the previous row)."""
    items: list[tuple[float, float, str]] = []
    v = Y0
    for labels, values, on in rows:
        for i, text in enumerate(labels):
            items.append((LABEL_U, v + i * WRAP, text))
        for k, val in enumerate(values):
            items.append((col(k), v + on * WRAP, val))
        v += (len(labels) - 1) * WRAP + PITCH
    return items


def table_md(rows: list[tuple[str, list[str]]], header: str = "|  |  |  |  |") -> str:
    out = [header, "|---|---|---|---|"]
    for label, values in rows:
        cells = list(values) + [""] * (3 - len(values))
        out.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(out) + "\n"


@contextlib.contextmanager
def evidence_off():
    """The same-line rule alone: the three kinds of new evidence answer 'no'."""
    with (
        patch.object(rbc, "_wrapped_label_fragments", lambda *a, **k: None),
        patch.object(rbc, "_wrapped_group_label", lambda *a, **k: None),
        patch.object(rbc, "_stacked_heading_binds", lambda *a, **k: None),
        patch.object(rbc, "_whole_line_matches", lambda *a, **k: False),
    ):
        yield


def check(page, markdown: str, *, on: bool = True):
    ctx = contextlib.nullcontext() if on else evidence_off()
    with ctx:
        results, _ = rbc.check_page_row_binding(page, markdown)
    assert len(results) == 1
    return results[0]


def both(items, markdown, rotation):
    page = page_of(items, rotation)
    return check(page, markdown, on=True), check(page, markdown, on=False)


def same_verdicts(a, b) -> bool:
    return a.outcome is b.outcome and [r.verdict for r in a.rows] == [r.verdict for r in b.rows]


V = [["1.1", "1.2", "1.3"], ["2.1", "2.2", "2.3"], ["3.1", "3.2", "3.3"], ["4.1", "4.2", "4.3"]]
LONG = "Cash operating balance, end of period"

# a table whose second row has a label printed over two lines (the first valueless)
WRAP_ABOVE = [
    (["Gross product"], V[0], 0),
    (["Cash operating balance,", "end of period"], V[1], 1),
    (["Receipts"], V[2], 0),
    (["Outlays"], V[3], 0),
]
WRAP_ABOVE_MD = [("Gross product", V[0]), (LONG, V[1]), ("Receipts", V[2]), ("Outlays", V[3])]

# ---------------------------------------------------------------------------
# E1: wrapped labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_label_wrapped_above_its_values_binds(rotation):
    on, off = both(layout(WRAP_ABOVE), table_md(WRAP_ABOVE_MD), rotation)
    assert off.outcome is TableOutcome.UNVERIFIABLE
    assert off.rows[1].verdict is RowVerdict.UNVERIFIABLE
    assert on.outcome is TableOutcome.BOUND
    assert on.evidence_counts() == {EVIDENCE_WRAPPED_LABEL: 1}
    assert [r.verdict for r in on.rows] == [RowVerdict.BOUND] * 4
    # the rows the same-line rule bound are the same rows in both arms
    assert [r.verdict for i, r in enumerate(off.rows) if i != 1] == [RowVerdict.BOUND] * 3


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_label_wrapped_below_its_values_binds(rotation):
    rows = [
        (["Gross product"], V[0], 0),
        (["Cash operating balance,", "end of period"], V[1], 0),
        (["Receipts"], V[2], 0),
        (["Outlays"], V[3], 0),
    ]
    on, off = both(layout(rows), table_md(WRAP_ABOVE_MD), rotation)
    assert off.outcome is TableOutcome.UNVERIFIABLE
    assert on.outcome is TableOutcome.BOUND and on.evidence_counts() == {EVIDENCE_WRAPPED_LABEL: 1}


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_label_wrapped_over_three_lines_binds(rotation):
    rows = [
        (["Gross product"], V[0], 0),
        (["Cash operating", "balance, end", "of period"], V[1], 1),
        (["Receipts"], V[2], 0),
        (["Outlays"], V[3], 0),
    ]
    on, off = both(layout(rows), table_md(WRAP_ABOVE_MD), rotation)
    assert off.outcome is TableOutcome.UNVERIFIABLE
    assert on.outcome is TableOutcome.BOUND and on.evidence_counts() == {EVIDENCE_WRAPPED_LABEL: 1}


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_adversary_values_moved_one_row_never_bind(rotation):
    """The model slid the wrapped row's values one row down: the label is still the exact join of the
    fragments around ITS OWN values' line, which are not these values' line.  Same verdicts as the
    same-line rule alone, and no more BOUND rows than it gives."""
    slid = [("Gross product", V[0]), (LONG, V[2]), ("Receipts", V[1]), ("Outlays", V[3])]
    on, off = both(layout(WRAP_ABOVE), table_md(slid), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_adversary_label_one_fragment_short_or_long_never_binds(rotation):
    for label in ("Cash operating balance, end", "Cash operating balance, end of period (1)"):
        shipped = [("Gross product", V[0]), (label, V[1]), ("Receipts", V[2]), ("Outlays", V[3])]
        on, off = both(layout(WRAP_ABOVE), table_md(shipped), rotation)
        assert on.outcome is not TableOutcome.BOUND
        assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_adversary_a_non_exact_join_never_binds(rotation):
    """Merge gate rule: no footnote or case tolerance on the BOUND path.  The page prints
    `end of period¹`; the shipped label has no mark (and the reverse)."""
    printed = [
        (["Gross product"], V[0], 0),
        (["Cash operating balance,", "end of period¹"], V[1], 1),
        (["Receipts"], V[2], 0),
        (["Outlays"], V[3], 0),
    ]
    for label in (LONG, LONG + "²", LONG.upper()):
        shipped = [("Gross product", V[0]), (label, V[1]), ("Receipts", V[2]), ("Outlays", V[3])]
        on, off = both(layout(printed), table_md(shipped), rotation)
        assert on.outcome is not TableOutcome.BOUND, label
        assert same_verdicts(on, off)
    exact = [("Gross product", V[0]), (LONG + "¹", V[1]), ("Receipts", V[2]), ("Outlays", V[3])]
    on, _ = both(layout(printed), table_md(exact), rotation)
    assert on.outcome is TableOutcome.BOUND  # identical string, footnote mark included


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_adversary_a_fragment_that_is_not_consecutive_never_binds(rotation):
    """A valueless line two pitches above the values' line is a section label, not a continuation."""
    far = [
        (["Gross product"], V[0], 0),
        (["Cash operating balance,"], [], 0),
        (["end of period"], V[1], 0),
        (["Receipts"], V[2], 0),
        (["Outlays"], V[3], 0),
    ]
    items = layout(far)
    # push the values' line one more pitch down so the fragment is no longer consecutive
    items = [
        (
            u,
            v
            + (
                PITCH
                if t in ("end of period", *V[1], "Receipts", "Outlays", *V[2], *V[3])
                and v > Y0 + PITCH + 1
                else 0
            ),
            t,
        )
        for u, v, t in items
    ]
    shipped = [("Gross product", V[0]), (LONG, V[1]), ("Receipts", V[2]), ("Outlays", V[3])]
    on, off = both(items, table_md(shipped), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_adversary_a_fragment_another_row_owns_never_binds(rotation):
    """`Cash operating balance,` is ALSO shipped as a group label of its own: the line may be that
    row's, so it is nobody's continuation."""
    shipped = [
        ("Gross product", V[0]),
        ("Cash operating balance,", []),
        (LONG, V[1]),
        ("Receipts", V[2]),
        ("Outlays", V[3]),
    ]
    on, off = both(layout(WRAP_ABOVE), table_md(shipped), rotation)
    assert on.rows[2].verdict is not RowVerdict.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_adversary_the_line_of_a_categorical_row_is_not_a_fragment(rotation):
    """`Receipts Yes` is the printed line of a categorical row of its own; the next row's label is
    that line plus `end of period`: the line is already a row's, so it is nobody's continuation."""
    items = [
        (LABEL_U, Y0, "Receipts"),
        (LABEL_U + 60, Y0, "Yes"),
        (LABEL_U, Y0 + WRAP, "end of period"),
        *[(col(k), Y0 + WRAP, val) for k, val in enumerate(V[1])],
        (LABEL_U, Y0 + WRAP + PITCH, "Outlays"),
        *[(col(k), Y0 + WRAP + PITCH, val) for k, val in enumerate(V[2])],
        (LABEL_U, Y0 + WRAP + 2 * PITCH, "Surplus"),
        *[(col(k), Y0 + WRAP + 2 * PITCH, val) for k, val in enumerate(V[3])],
    ]
    shipped = [
        ("Receipts", ["Yes"]),
        ("Receipts Yes end of period", V[1]),
        ("Outlays", V[2]),
        ("Surplus", V[3]),
    ]
    on, off = both(items, table_md(shipped), rotation)
    assert on.rows[1].verdict is not RowVerdict.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_adversary_a_shared_continuation_binds_neither_row(rotation):
    """One valueless line between two rows, each shipped label wanting it."""
    printed = [
        (["Alpha"], V[0], 0),
        (["continued here"], [], 0),
        (["Beta"], V[1], 0),
        (["Gamma"], V[2], 0),
        (["Delta"], V[3], 0),
    ]
    shipped = [
        ("Alpha continued here", V[0]),
        ("continued here Beta", V[1]),
        ("Gamma", V[2]),
        ("Delta", V[3]),
    ]
    on, off = both(layout(printed), table_md(shipped), rotation)
    assert on.rows[0].verdict is not RowVerdict.BOUND
    assert on.rows[1].verdict is not RowVerdict.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_adversary_a_label_that_reaches_into_the_value_columns_never_binds(rotation):
    """The 'continuation' below prints inside the value columns: it is part of another column, not
    the label's second line."""
    below = [
        (["Gross product"], V[0], 0),
        (["Cash operating balance,", "end of period"], V[1], 0),
        (["Receipts"], V[2], 0),
        (["Outlays"], V[3], 0),
    ]
    items = layout(below)
    items = [(col(1) if t == "end of period" else u, v, t) for u, v, t in items]
    on, off = both(items, table_md(WRAP_ABOVE_MD), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_adversary_a_continuation_left_of_the_label_never_binds(rotation):
    """A continuation line starting well left of the label's first line is another level of the list,
    not the label's second line."""
    items = layout(WRAP_ABOVE)
    items = [(u - 30 if t == "end of period" else u, v, t) for u, v, t in items]
    on, off = both(items, table_md(WRAP_ABOVE_MD), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_adversary_a_row_missing_cells_of_its_line_never_binds(rotation):
    """`_cells_cover_line` still applies: the page prints a word between the row's values that the row
    dropped (trailing text after the last value is allowed, as before)."""
    items = layout(WRAP_ABOVE) + [(col(0) + 20, Y0 + PITCH + WRAP, "n.a.")]
    on, off = both(items, table_md(WRAP_ABOVE_MD), rotation)
    assert on.rows[1].verdict is not RowVerdict.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_adversary_two_splits_of_one_label_never_bind(rotation):
    """The label could be read as 1+2 or as 2+1 lines (a repeated word makes both exact): not unique,
    so no evidence."""
    printed = [
        (["Gross product"], V[0], 0),
        (["Net", "Net", "Net"], V[1], 1),
        (["Receipts"], V[2], 0),
        (["Outlays"], V[3], 0),
    ]
    shipped = [("Gross product", V[0]), ("Net Net", V[1]), ("Receipts", V[2]), ("Outlays", V[3])]
    on, off = both(layout(printed), table_md(shipped), rotation)
    assert on.rows[1].verdict is not RowVerdict.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_a_wrapped_group_label_binds(rotation):
    printed = [
        (["Gross product"], V[0], 0),
        (["Prices excluding", "food and energy"], [], 0),
        (["Receipts"], V[2], 0),
        (["Outlays"], V[3], 0),
    ]
    shipped = [
        ("Gross product", V[0]),
        ("**Prices excluding food and energy**", []),
        ("Receipts", V[2]),
        ("Outlays", V[3]),
    ]
    on, off = both(layout(printed), table_md(shipped), rotation)
    assert off.outcome is TableOutcome.UNVERIFIABLE
    assert on.outcome is TableOutcome.BOUND and on.evidence_counts() == {EVIDENCE_WRAPPED_LABEL: 1}


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_a_wrapped_group_label_with_a_foreign_line_between_never_binds(rotation):
    printed = [
        (["Gross product"], V[0], 0),
        (["Prices excluding", "(percent)", "food and energy"], [], 0),
        (["Receipts"], V[2], 0),
        (["Outlays"], V[3], 0),
    ]
    shipped = [
        ("Gross product", V[0]),
        ("Prices excluding food and energy", []),
        ("Receipts", V[2]),
        ("Outlays", V[3]),
    ]
    on, off = both(layout(printed), table_md(shipped), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e1_a_wrapped_group_label_whose_lines_are_a_section_apart_never_binds(rotation):
    printed = [
        (["Gross product"], V[0], 0),
        (["Prices excluding"], [], 0),
        (["(percent)"], [], 0),
        (["food and energy"], [], 0),
        (["Receipts"], V[2], 0),
        (["Outlays"], V[3], 0),
    ]
    items = layout(printed)
    # take the middle line away: "Prices excluding" and "food and energy" are now two pitches apart
    items = [i for i in items if i[2] != "(percent)"]
    shipped = [
        ("Gross product", V[0]),
        ("Prices excluding food and energy", []),
        ("Receipts", V[2]),
        ("Outlays", V[3]),
    ]
    on, off = both(items, table_md(shipped), rotation)
    assert on.rows[1].verdict is not RowVerdict.BOUND
    assert same_verdicts(on, off)


# ---------------------------------------------------------------------------
# E3: whole line
# ---------------------------------------------------------------------------

# a label spread over two cells: `1` | `LR`
STUB_ROWS = [
    (["Gross product"], V[0], 0),
    (["1 LR"], V[1][:2], 0),
    (["Receipts"], V[2], 0),
    (["Outlays"], V[3], 0),
]


def _two_cell_stub(rotation, label="1", cell="LR"):
    items = layout(STUB_ROWS)
    items = [(u, v, "1") if t == "1 LR" else (u, v, t) for u, v, t in items]
    items.append((LABEL_U + 12, Y0 + PITCH, "LR"))
    shipped = [
        ("Gross product", V[0]),
        (label, [cell, *V[1][:2]]),
        ("Receipts", V[2]),
        ("Outlays", V[3]),
    ]
    return items, table_md(shipped)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e3_a_label_spread_over_two_cells_binds(rotation):
    items, markdown = _two_cell_stub(rotation)
    on, off = both(items, markdown, rotation)
    assert off.outcome is TableOutcome.UNVERIFIABLE
    assert on.outcome is TableOutcome.BOUND and on.evidence_counts() == {EVIDENCE_WHOLE_LINE: 1}


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e3_adversary_a_word_changed_or_dropped_never_binds(rotation):
    for label, cell in (("2", "LR"), ("1", "LS"), ("1", "")):
        items, markdown = _two_cell_stub(rotation, label, cell)
        on, off = both(items, markdown, rotation)
        assert on.rows[1].verdict is not RowVerdict.BOUND, (label, cell)
        assert same_verdicts(on, off)


RANGE_PRINTED = [
    (["Gross product"], V[0], 0),
    (["Forecast errors .0-3.0 .7-3.6 . . ."], [], 0),
    (["Receipts"], V[2], 0),
    (["Outlays"], V[3], 0),
]


def _range_page(rotation):
    items = layout(RANGE_PRINTED)
    items = [i for i in items if not i[2].startswith("Forecast errors")]
    y = Y0 + PITCH
    items += [(LABEL_U, y, "Forecast errors"), (col(0), y, ".0-3.0"), (col(1), y, ".7-3.6")]
    items += [(col(2), y, "."), (col(2) + 5, y, "."), (col(2) + 10, y, ".")]
    return items


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e3_a_row_of_ranges_and_ellipses_binds(rotation):
    shipped = [
        ("Gross product", V[0]),
        ("Forecast errors", [".0-3.0", ".7-3.6", "..."]),
        ("Receipts", V[2]),
        ("Outlays", V[3]),
    ]
    on, off = both(_range_page(rotation), table_md(shipped), rotation)
    assert off.outcome is TableOutcome.UNVERIFIABLE
    assert on.outcome is TableOutcome.BOUND and on.evidence_counts() == {EVIDENCE_WHOLE_LINE: 1}


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e3_adversary_a_changed_range_never_binds(rotation):
    for cells in ([".0-3.1", ".7-3.6", "..."], [".0-3.0", ".7-3.6", ". ."], [".0-3.0", ".7-3.6"]):
        shipped = [
            ("Gross product", V[0]),
            ("Forecast errors", cells),
            ("Receipts", V[2]),
            ("Outlays", V[3]),
        ]
        on, off = both(_range_page(rotation), table_md(shipped), rotation)
        assert on.rows[1].verdict is not RowVerdict.BOUND, cells
        assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e3_adversary_a_printed_word_the_row_left_out_never_binds(rotation):
    """The page prints one more word after the row's last cell; the row is a prefix of the line, not
    the line.  (Trailing text is allowed by the same-line rule's cell cover, never by whole-line.)"""
    items, markdown = _two_cell_stub(rotation)
    items = items + [(col(2), Y0 + PITCH, "x")]
    on, off = both(items, markdown, rotation)
    assert on.rows[1].verdict is not RowVerdict.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e3_adversary_two_identical_rows_for_one_printed_line_never_bind(rotation):
    row = ("Forecast errors", [".0-3.0", ".7-3.6", "..."])
    shipped = [("Gross product", V[0]), row, row, ("Receipts", V[2]), ("Outlays", V[3])]
    on, off = both(_range_page(rotation), table_md(shipped), rotation)
    assert [r.verdict for r in on.rows[1:3]] == [RowVerdict.UNVERIFIABLE] * 2
    assert same_verdicts(on, off)


# ---------------------------------------------------------------------------
# E2: stacked headings
# ---------------------------------------------------------------------------

HEAD_TOP = ["Real", "Net", "Gov."]
HEAD_BOTTOM = ["GDP", "exports", "spend"]
HEAD_MD = "| Item | Real GDP | Net exports | Gov. spend |"


def _stacked(extra_top: list[str] | None = None, body=None) -> list[tuple[float, float, str]]:
    rows = body or [("Alpha", V[0]), ("Bravo", V[1]), ("Charlie", V[2]), ("Delta", V[3])]
    items: list[tuple[float, float, str]] = []
    y = Y0
    if extra_top:
        items += [(col(k), y, t) for k, t in enumerate(extra_top)]
        y += WRAP
    items += [(col(k), y, t) for k, t in enumerate(HEAD_TOP)]
    y += WRAP
    items += [(LABEL_U, y, "Item")] + [(col(k), y, t) for k, t in enumerate(HEAD_BOTTOM)]
    y += PITCH
    for label, values in rows:
        items.append((LABEL_U, y, label))
        items += [(col(k), y, val) for k, val in enumerate(values)]
        y += PITCH
    return items


BODY = [("Alpha", V[0]), ("Bravo", V[1]), ("Charlie", V[2]), ("Delta", V[3])]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e2_a_stacked_heading_binds(rotation):
    on, off = both(_stacked(), table_md(BODY, HEAD_MD), rotation)
    assert off.outcome is TableOutcome.UNVERIFIABLE
    assert on.outcome is TableOutcome.BOUND and on.evidence_counts() == {
        EVIDENCE_STACKED_HEADING: 1
    }
    assert on.rows[0].shipped_label == "Item"


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e2_adversary_swapped_columns_never_bind(rotation):
    """Every word is kept and each column keeps its own words in order, but two headings sit over the
    other's data column."""
    swapped = "| Item | Net exports | Real GDP | Gov. spend |"
    on, off = both(_stacked(), table_md(BODY, swapped), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e2_adversary_words_moved_across_the_column_border_never_bind(rotation):
    """The flattened word sequence is unchanged (`Real GDP Net exports Gov. spend`), the cells split it
    one word off."""
    shifted = "| Item | Real | GDP Net exports | Gov. spend |"
    on, off = both(_stacked(), table_md(BODY, shifted), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e2_a_spanning_word_binds_only_in_the_column_its_centre_sits_over(rotation):
    """A word printed above the border of two columns: the cells must put it where the page does."""
    items = _stacked() + [(col(1) - 22, Y0 - WRAP, "Growth")]
    # `Growth` is 27 wide and starts 22 left of the second column: its centre is over the second
    centred = "| Item | Real GDP | Growth Net exports | Gov. spend |"
    moved = "| Item | Growth Real GDP | Net exports | Gov. spend |"
    on, off = both(items, table_md(BODY, centred), rotation)
    assert on.outcome is TableOutcome.BOUND and on.evidence_counts() == {
        EVIDENCE_STACKED_HEADING: 1
    }
    assert off.outcome is TableOutcome.UNVERIFIABLE
    on2, off2 = both(items, table_md(BODY, moved), rotation)
    assert on2.outcome is not TableOutcome.BOUND
    assert same_verdicts(on2, off2)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e2_adversary_a_native_heading_word_left_over_never_binds(rotation):
    """The page prints a third line at the foot of the stack (`pct.`, under `GDP`); the shipped
    heading dropped it: a word of the stack is left over."""
    items = _stacked() + [(col(k), Y0 + 2 * WRAP, "pct.") for k in range(3)]
    on, off = both(items, table_md(BODY, HEAD_MD), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e2_adversary_a_missing_column_never_binds(rotation):
    two = "| Item | Real GDP | Gov. spend |"
    on, off = both(_stacked(), table_md(BODY, two), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e2_adversary_a_heading_with_fewer_cells_than_data_columns_never_binds(rotation):
    """The third column has no heading on the page and none shipped: the shipped heading accounts for
    two columns, the rows carry three.  The words match; the shape does not."""
    items = [(u, v, t) for u, v, t in _stacked() if u < col(2) - 1 or v > Y0 + 2 * WRAP]
    two = "| Item | Real GDP | Net exports |"
    on, off = both(items, table_md(BODY, two), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e2_adversary_a_stub_the_page_does_not_print_never_binds(rotation):
    on, off = both(_stacked(), table_md(BODY, HEAD_MD.replace("Item", "Items")), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_e2_adversary_a_stack_that_is_not_consecutive_never_binds(rotation):
    """The first stack line sits a full section break above the second: not one heading."""
    items = _stacked()
    items = [(u, v - 2 * PITCH if t in HEAD_TOP else v, t) for u, v, t in items]
    on, off = both(items, table_md(BODY, HEAD_MD), rotation)
    assert on.outcome is not TableOutcome.BOUND
    assert same_verdicts(on, off)


# ---------------------------------------------------------------------------
# Completeness: a table that lost a row keeps its mark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_table_that_lost_a_row_keeps_its_mark_under_every_kind(rotation):
    # E2: one printed row is missing from the shipped table
    items = _stacked(body=[*BODY, ("Echo", V[0])])
    on, off = both(items, table_md(BODY, HEAD_MD), rotation)
    assert on.outcome is not TableOutcome.BOUND and same_verdicts(on, off)
    # E1: the same for a wrapped label
    printed = [*WRAP_ABOVE, (["Surplus"], V[0], 0)]
    on, off = both(layout(printed), table_md(WRAP_ABOVE_MD), rotation)
    assert on.outcome is not TableOutcome.BOUND and same_verdicts(on, off)
    # and a complete table binds (the printed row is shipped too)
    on, _ = both(layout(printed), table_md([*WRAP_ABOVE_MD, ("Surplus", V[0])]), rotation)
    assert on.outcome is TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_running_head_with_a_date_does_not_count_as_a_lost_row(rotation):
    items = layout(WRAP_ABOVE) + [(col(2), 40.0, "October 18, 2012")]
    on, off = both(items, table_md(WRAP_ABOVE_MD), rotation)
    assert on.outcome is TableOutcome.BOUND and off.outcome is TableOutcome.UNVERIFIABLE


# ---------------------------------------------------------------------------
# Convictions are untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_convicted_table_stays_convicted_with_the_evidence_on(rotation):
    shifted = [
        ("Gross product", V[0]),
        (LONG, []),
        ("Receipts", V[1]),
        ("", V[2]),
        ("Outlays", V[3]),
    ]
    on, off = both(layout(WRAP_ABOVE), table_md(shifted), rotation)
    assert off.outcome is TableOutcome.CONTRADICTED
    assert on.outcome is TableOutcome.CONTRADICTED
    assert [r.verdict for r in on.rows] == [r.verdict for r in off.rows]


def test_the_point_repair_gate_judges_by_the_same_line_rule(monkeypatch):
    """A repair is never licensed by the new evidence: its gate calls the check without it."""
    seen: list[bool] = []
    real = rbc.check_row_binding

    def spy(words, region, markdown, **kw):
        seen.append(kw.get("extra_evidence", True))
        return real(words, region, markdown, **kw)

    monkeypatch.setattr(rbc, "check_row_binding", spy)
    page = page_of(layout(WRAP_ABOVE), 0)
    words, (_, union), _ = rbc._page_frame(page, None)
    rbc.propose_point_repairs(words, union, table_md(WRAP_ABOVE_MD))
    assert seen and not any(seen)


# ---------------------------------------------------------------------------
# The record: what a pass leaves behind
# ---------------------------------------------------------------------------


def test_the_record_counts_bound_rows_and_names_the_evidence():
    page = page_of(layout(WRAP_ABOVE), 0)
    result = check(page, table_md(WRAP_ABOVE_MD))
    payload = rbc.table_payload(
        1, 0, result.outcome.value, rbc.row_evidence_dicts(result), "", result
    )
    assert payload["outcome"] == "bound"
    assert payload["rows_bound"] == 4 and payload["rows_total"] == 4
    assert payload["evidence"] == {EVIDENCE_WRAPPED_LABEL: 1}
    assert payload["rows"] == []  # rows still lists only what is not BOUND
    plain = rbc.table_payload(1, 0, "bound", [])
    assert "rows_bound" not in plain and "evidence" not in plain
