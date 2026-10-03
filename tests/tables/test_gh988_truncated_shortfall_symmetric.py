"""#988: ``table_truncated``'s row-shortfall term must count like with like.

Before: the candidate side dropped every blank-stub row (each standard-error /
t-statistic line) while the native side counted every page band with enough
numbers -- including those SE lines -- so a complete regression table read as a
shortfall and was refused.

The fix counts a blank-stub candidate row, but only when it binds uniquely to a
native band. The native count stays page-wide: scoping it to a "table region"
was tried and every variant let a truncated table through (see
docs/log/2026-10-03_truncated-shortfall.md).

Every pin is a DIFFERENCE: the same input is run with main's counting restored
(``pre_988``) and with the fix live, so nothing asserts an absolute outcome
measured on one machine.
"""

from __future__ import annotations

import pytest

from socr.tables import row_corroboration, structure_check
from socr.tables.row_corroboration import numeric_body_rows, table_blocks
from socr.tables.structure_check import table_truncated

COEFS = 10  # coefficient rows; each is followed by one SE row


def _band(y: float, label: str | None, values: list[str], x0: float = 100.0) -> list[tuple]:
    """One printed line: an optional label word, then values in fixed x lanes."""
    words = []
    if label is not None:
        words.append((0.0, y, 40.0, y + 10.0, label))
    for lane, value in enumerate(values):
        x = x0 + lane * 60.0
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


def _markdown(n: int = COEFS, *, skip: int = 0, drop: range = range(0)) -> str:
    md = "| Variable | (1) | (2) | (3) |\n|---|---|---|---|\n"
    for i, (label, coef, se) in enumerate(_regression_rows(COEFS)[skip : skip + n], skip):
        if i in drop:
            continue
        md += f"| {label} | {' | '.join(coef)} |\n"
        md += f"| | {' | '.join(se)} |\n"
    return md


def _legacy_rows(words: list, markdown: str):
    """What main counted: blank-stub rows dropped, every labelled row counted."""
    rows = [r for blk in table_blocks(markdown) for r in numeric_body_rows(blk) if r]
    return rows, min((len(r) for r in rows), default=0)


@pytest.fixture
def pre_988(monkeypatch: pytest.MonkeyPatch):
    """Restore main's counting for the candidate side."""

    def apply() -> None:
        monkeypatch.setattr(structure_check, "_corroborated_candidate_rows", _legacy_rows)

    return apply


def test_numeric_body_rows_blank_stub_is_opt_in() -> None:
    se_row = ["", "(1.46)", "(1.55)"]
    assert numeric_body_rows([se_row]) == []
    assert numeric_body_rows([se_row], include_blank_stub=True) == [("(1.46)", "(1.55)")]
    assert row_corroboration.numeric_body_rows is numeric_body_rows


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


def test_middle_deletion_is_refused(pre_988) -> None:
    words = _page()
    md = _markdown(drop=range(3, 7))
    assert table_truncated(md, words) is True
    pre_988()
    assert table_truncated(md, words) is True


def test_repeated_block_is_refused(pre_988) -> None:
    """Astra 1: the first half emitted twice must not count 20 rows against 20
    native bands, nor bind the same SE bands twice.
    """
    words = _page()
    md = _markdown(COEFS // 2) + "\n" + _markdown(COEFS // 2)
    assert table_truncated(md, words) is True
    pre_988()
    assert table_truncated(md, words) is True


def test_three_copies_are_refused(pre_988) -> None:
    """Astra 2: three copies of the first half (10 bound + 10 unbound labelled)
    must not mask the missing half of the page.
    """
    words = _page()
    md = "\n".join([_markdown(COEFS // 2)] * 3)
    assert table_truncated(md, words) is True
    pre_988()
    assert table_truncated(md, words) is True


def test_second_table_below_is_not_ignored(pre_988) -> None:
    """Astra 3: native has two tables; the candidate carries only the first.
    The native count is page-wide, so the second table is a shortfall.
    """
    words = _page() + _page(y0=900.0)
    md = _markdown()
    assert table_truncated(md, words) is True
    pre_988()
    assert table_truncated(md, words) is True


def test_invented_se_rows_do_not_count(pre_988) -> None:
    """Blank-stub rows that bind to no native band must not make up a shortfall."""
    words = _page()
    md = "| Variable | (1) | (2) | (3) |\n|---|---|---|---|\n"
    for label, coef, _se in _regression_rows():
        md += f"| {label} | {' | '.join(coef)} |\n"
    for i in range(COEFS):
        md += f"| | (9{i}.1) | (9{i}.2) | (9{i}.3) |\n"
    assert table_truncated(md, words) is True
    pre_988()
    assert table_truncated(md, words) is True


def test_se_rows_repeated_in_a_second_block_do_not_count(pre_988) -> None:
    """A native SE band credits one candidate row: the same SE lines emitted again
    in a second block must not make up the missing coefficient rows.
    """
    rows = _regression_rows(6)
    words = _page(6)
    head = "| Variable | (1) | (2) | (3) |\n|---|---|---|---|\n"
    block1 = head + "".join(
        f"| {label} | {' | '.join(coef)} |\n| | {' | '.join(se)} |\n"
        for label, coef, se in rows[:4]
    )
    block2 = head + "".join(f"| | {' | '.join(se)} |\n" for _l, _c, se in rows[:4])
    md = block1 + "\n" + block2
    assert table_truncated(md, words) is True
    pre_988()
    assert table_truncated(md, words) is True
