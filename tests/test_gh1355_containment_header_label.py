"""#1355: winner containment counts a supported number carried in a HEADER label cell.

A paid reading often writes a column header as ``2001 (Percent change<sup>1</sup>)`` where the
free reading wrote ``2001``. The tokeniser treats the former as a label cell, so containment
(#1126-r6) found the year "missing" and kept the floor. The number inside a header label cell of
the winner's own table now counts. Data-row label cells and page prose never do.
"""

from __future__ import annotations

import pytest

from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import winner_accounts_for_marked_tables


def _out(text: str, **kw) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="x", **kw)


def _table(header: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(f"| {a} | {b} |\n" for a, b in rows)
    return f"| item | {header} |\n|---|---|\n{body}"


MARKED = _out(
    _table("2001", [("Corn", "5.5"), ("Soy", "6.5")]),
    table_disputed="x",
    table_disputed_values=("-9.9",),
)
ROWS = [("Corn", "5.5"), ("Soy", "6.5")]


def test_header_year_inside_a_label_cell_lifts_and_its_absence_does_not() -> None:
    carried = _out(_table("2001 (Percent change)", ROWS))
    absent = _out(_table("Percent change", ROWS))
    assert winner_accounts_for_marked_tables(carried, MARKED) is True
    assert winner_accounts_for_marked_tables(absent, MARKED) is False


def test_a_number_only_in_page_prose_does_not_lift() -> None:
    absent = _out("Results for 2001 are below.\n\n" + _table("Percent change", ROWS))
    assert winner_accounts_for_marked_tables(absent, MARKED) is False


def test_a_data_row_label_cell_cannot_hide_a_dropped_value() -> None:
    hidden = _out(_table("Percent change", [("Corn 2001", "5.5"), ("Soy", "6.5")]))
    assert winner_accounts_for_marked_tables(hidden, MARKED) is False


def test_multiset_shortfall_does_not_lift() -> None:
    marked = _out(
        _table("2001", [("Corn", "2001"), ("Soy", "6.5")]),
        table_disputed="x",
        table_disputed_values=("-9.9",),
    )
    one = _out(_table("2001 (Percent change)", [("Corn", "5.5"), ("Soy", "6.5")]))
    both = _out(_table("2001 (Percent change)", [("Corn", "2001"), ("Soy", "6.5")]))
    assert winner_accounts_for_marked_tables(one, marked) is False
    assert winner_accounts_for_marked_tables(both, marked) is True


@pytest.mark.parametrize("header", ["2001 (Percent change)", "2001"])
def test_a_bare_header_cell_is_unchanged(header: str) -> None:
    assert winner_accounts_for_marked_tables(_out(_table(header, ROWS)), MARKED) is True
