"""#988: ``table_truncated``'s row-shortfall term must count like with like.

Before: the candidate side dropped every blank-stub row (each standard-error /
t-statistic line) while the native side counted every page band with enough
numbers -- the same SE lines plus axis ticks, running heads and prose -- so a
complete regression table read as a large shortfall and was refused.

Every pin here is a DIFFERENCE: the same input is run with the two #988
changes neutralised (``_pre_988``) and with them live, so nothing asserts an
absolute outcome measured on one machine.
"""

from __future__ import annotations

import pytest

from socr.tables import row_corroboration, structure_check
from socr.tables.row_corroboration import numeric_body_rows, table_shaped_native_row_count
from socr.tables.structure_check import _truncated_row_shortfall, table_truncated

COEFS = 10  # coefficient rows; each is followed by one SE row


def _band(y: float, label: str | None, values: list[str]) -> list[tuple]:
    """One printed line: an optional label word, then values in fixed x lanes."""
    words = []
    if label is not None:
        words.append((0.0, y, 40.0, y + 10.0, label))
    for lane, value in enumerate(values):
        x = 100.0 + lane * 60.0
        words.append((x, y, x + 30.0, y + 10.0, value))
    return words


def _regression_rows(n: int = COEFS) -> list[tuple[str, list[str], list[str]]]:
    return [
        (f"Var{i}", [f"{0.1 * (i + 1):.2f}", f"{0.2 * (i + 1):.2f}", f"{0.3 * (i + 1):.2f}"],
         [f"({1.0 + i:.2f})", f"({2.0 + i:.2f})", f"({3.0 + i:.2f})"])
        for i in range(n)
    ]  # fmt: skip


def _page(n: int = COEFS, y0: float = 100.0) -> list[tuple]:
    words: list[tuple] = []
    y = y0
    for label, coef, se in _regression_rows(n):
        words += _band(y, label, coef)
        words += _band(y + 20.0, None, se)
        y += 40.0
    return words


def _markdown(n: int = COEFS, *, with_se: bool = True, skip: int = 0) -> str:
    md = "| Variable | (1) | (2) | (3) |\n|---|---|---|---|\n"
    for label, coef, se in _regression_rows(COEFS)[skip : skip + n]:
        md += f"| {label} | {' | '.join(coef)} |\n"
        if with_se:
            md += f"| | {' | '.join(se)} |\n"
    return md


@pytest.fixture
def pre_988(monkeypatch: pytest.MonkeyPatch):
    """Neutralise both #988 changes: blank-stub rows dropped, page-wide native count."""

    def apply() -> None:
        monkeypatch.setattr(
            row_corroboration,
            "numeric_body_rows",
            lambda rows, include_blank_stub=False: numeric_body_rows(rows),
        )
        monkeypatch.setattr(
            structure_check,
            "_native_table_rows_in_candidate_region",
            lambda words, blocks, row_shape_min: table_shaped_native_row_count(
                words, row_shape_min
            ),
        )

    return apply


def test_numeric_body_rows_blank_stub_is_opt_in() -> None:
    se_row = ["", "(1.46)", "(1.55)"]
    assert numeric_body_rows([se_row]) == []
    assert numeric_body_rows([se_row], include_blank_stub=True) == [("(1.46)", "(1.55)")]


def test_se_rows_table_difference_pin(pre_988) -> None:
    """A complete regression table (coefficient + SE rows) that main refuses and
    the branch accepts.
    """
    words, md = _page(), _markdown()
    assert table_truncated(md, words) is False  # live
    pre_988()
    assert table_truncated(md, words) is True  # what main did


def test_truncated_se_table_still_refused(pre_988) -> None:
    """Final rows missing (coefficient AND SE rows) is still a truncation, both ways."""
    words = _page()
    md = _markdown(COEFS - 4)
    assert table_truncated(md, words) is True
    pre_988()
    assert table_truncated(md, words) is True


def test_truncation_hidden_above_first_matched_row_is_counted() -> None:
    """Same, for rows missing at the TOP of the table."""
    words = _page()
    md = _markdown(COEFS // 2, skip=COEFS // 2)
    assert _truncated_row_shortfall(words, md) is True


def test_truncation_hidden_below_last_matched_row_is_counted() -> None:
    """The region must extend past the candidate's last bound row, or a truncated
    tail would shrink the region with it and the shortfall would never show.
    """
    words = _page()
    md = _markdown(COEFS // 2)
    assert _truncated_row_shortfall(words, md) is True


def test_axis_ticks_and_running_head_outside_table_do_not_inflate(pre_988) -> None:
    """Tick rows above the table and a running head below it, each separated from
    the table by a numeric-free line, are not part of the table region.
    """
    words = _page(y0=400.0)
    # a figure's x-axis ticks: 12 three-number bands, then a caption line
    for i in range(12):
        words += _band(10.0 + i * 20.0, None, ["10", "20", "30"])
    words.append((0.0, 300.0, 200.0, 310.0, "Figure"))
    # a running head with three figures, below the table, after a prose line
    table_bottom = 400.0 + COEFS * 40.0
    words.append((0.0, table_bottom + 20.0, 200.0, table_bottom + 30.0, "Source"))
    words += _band(table_bottom + 60.0, "Journal", ["145", "2023", "103822"])
    md = _markdown()

    assert table_truncated(md, words) is False
    pre_988()
    assert table_truncated(md, words) is True


def test_no_bound_row_keeps_page_wide_count() -> None:
    """A candidate that binds no native band gives no region; the guard must not
    relax on evidence it could not locate.
    """
    words = _page()
    md = "| Variable | (1) | (2) | (3) |\n|---|---|---|---|\n| X | 9.91 | 9.92 | 9.93 |\n"
    assert _truncated_row_shortfall(words, md) is True
