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
from socr.tables.row_corroboration import numeric_body_rows, table_blocks
from socr.tables.structure_check import _truncated_row_shortfall, table_truncated

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


def _page(n: int = COEFS, y0: float = 100.0, heading_after: int | None = None) -> list[tuple]:
    words: list[tuple] = []
    y = y0
    for i, (label, coef, se) in enumerate(_regression_rows(n)):
        if heading_after == i:
            words.append((0.0, y, 90.0, y + 10.0, "Panel"))  # numeric-free panel heading
            y += 20.0
        words += _band(y, label, coef)
        words += _band(y + 20.0, None, se)
        y += 40.0
    return words


def _markdown(
    n: int = COEFS, *, with_se: bool = True, skip: int = 0, drop: range = range(0)
) -> str:
    md = "| Variable | (1) | (2) | (3) |\n|---|---|---|---|\n"
    for i, (label, coef, se) in enumerate(_regression_rows(COEFS)[skip : skip + n], skip):
        if i in drop:
            continue
        md += f"| {label} | {' | '.join(coef)} |\n"
        if with_se:
            md += f"| | {' | '.join(se)} |\n"
    return md


def _legacy_rows(words: list, markdown: str):
    """What main counted: blank-stub rows dropped, nothing bound (so page-wide)."""
    rows = [r for blk in table_blocks(markdown) for r in numeric_body_rows(blk) if r]
    return rows, [], min((len(r) for r in rows), default=0)


@pytest.fixture
def pre_988(monkeypatch: pytest.MonkeyPatch):
    """Restore main's counting: blank-stub rows dropped, page-wide native count."""

    def apply() -> None:
        monkeypatch.setattr(structure_check, "_corroborated_candidate_rows", _legacy_rows)

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
        words += _band(10.0 + i * 20.0, None, ["10", "20", "30"], x0=600.0)
    words.append((0.0, 300.0, 200.0, 310.0, "Figure"))
    # a running head with three figures, below the table, after a prose line
    table_bottom = 400.0 + COEFS * 40.0
    words.append((0.0, table_bottom + 20.0, 200.0, table_bottom + 30.0, "Source"))
    words += _band(table_bottom + 60.0, "Journal", ["145", "2023", "103822"], x0=600.0)
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


def test_middle_deletion_is_refused(pre_988) -> None:
    """Rows missing from the MIDDLE are a shortfall on both sides of the change."""
    words = _page()
    md = _markdown(drop=range(3, 7))
    assert table_truncated(md, words) is True
    pre_988()
    assert table_truncated(md, words) is True


def test_panel_boundary_truncation_is_refused(pre_988) -> None:
    """Native: panel 1, a numeric-free panel heading, panel 2. The candidate emits
    only panel 1. The heading must not end the extent.
    """
    words = _page(heading_after=COEFS // 2)
    md = _markdown(COEFS // 2)
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


def test_one_binding_does_not_enable_scoping(pre_988) -> None:
    """One row binding to an off-table band (axis ticks) must not scope the native
    count to that band and hide that the real table is mostly missing.
    """
    words = _page()
    words.append((0.0, 900.0, 200.0, 910.0, "Source"))  # numeric-free line between
    for i in range(COEFS):
        words += _band(1000.0 + i * 20.0, None, ["10", "20", "30"], x0=600.0)
    md = "| Variable | (1) | (2) | (3) |\n|---|---|---|---|\n| Tick | 10 | 20 | 30 |\n"
    for i in range(COEFS - 1):
        md += f"| Z{i} | 7.{i}1 | 7.{i}2 | 7.{i}3 |\n"
    assert table_truncated(md, words) is True
    pre_988()
    assert table_truncated(md, words) is True


def test_two_blocks_cannot_credit_the_same_native_bands(pre_988) -> None:
    """Astra's reproducer: the first half of the table emitted twice (two blocks)
    must not count 20 rows against 20 native bands, nor bind the same bands twice.
    """
    words = _page()
    md = _markdown(COEFS // 2) + "\n" + _markdown(COEFS // 2)
    assert table_truncated(md, words) is True
    pre_988()
    assert table_truncated(md, words) is True


def _two_tables_same_lanes(caption: str | None, gap: float) -> list[tuple]:
    """Table A, then table B printed in the same lanes ``gap`` below it."""
    words = _page()
    bottom = 100.0 + COEFS * 40.0
    if caption:
        words.append((0.0, bottom + gap - 20.0, 60.0, bottom + gap - 10.0, caption))
        words.append((70.0, bottom + gap - 20.0, 90.0, bottom + gap - 10.0, "2"))
    else:
        words.append((0.0, bottom + gap - 20.0, 90.0, bottom + gap - 10.0, "Notes"))
    return words + _page(y0=bottom + gap)


@pytest.mark.parametrize("caption,gap", [(None, 400.0), ("Table", 20.0)])
def test_second_table_in_same_lanes_is_not_absorbed(caption, gap) -> None:
    """A complete candidate for table A must not be refused because table B
    below shares its lanes: a wide gap or a ``Table N`` caption ends the bridge.
    """
    words = _two_tables_same_lanes(caption, gap)
    assert table_truncated(_markdown(), words) is False
    # and table A really is missing rows when the candidate stops early
    assert table_truncated(_markdown(COEFS // 2), words) is True
