"""#1370: chart axis tick lines (year labels under a chart) are not table rows.

Term (b) of ``table_truncated`` counted every table-shaped native band as a
table row, so a complete reading of a page with a small table and charts was
called truncated. Hermetic: synthetic fitz-shaped word tuples and Markdown.
"""

from __future__ import annotations

import pytest

from socr.tables.structure_check import _is_axis_tick_band, table_truncated

ROWS = 8
COLS = 4
COL_X = [200.0 + 50.0 * c for c in range(COLS)]


def _word(x: float, y: float, text: str) -> tuple:
    return (x, y, x + 24.0, y + 8.0, text, 0, 0, 0)


def _table_words() -> list:
    words = []
    for r in range(ROWS):
        y = 100.0 + 12.0 * r
        words.append(_word(60.0, y, f"Item{r}"))
        for c in range(COLS):
            words.append(_word(COL_X[c], y, f"{r + 1}{c + 3}.{r}{c}"))
    return words


def _tick_band(y: float, labels: list[str], x0: float = 60.0, pitch: float = 30.0) -> list:
    return [_word(x0 + pitch * i, y, lab) for i, lab in enumerate(labels)]


def _years(first: int, last: int, step: int = 1) -> list[str]:
    return [str(y) for y in range(first, last + 1, step)]


def _chart_words() -> list:
    """Two stacked charts, then two side-by-side charts merged into one band."""
    words = _tick_band(400.0, _years(1992, 2005))
    words += _tick_band(500.0, _years(1992, 2005))
    side = _years(2000, 2007)
    words += _tick_band(600.0, side + side, x0=40.0, pitch=28.0)
    return words


def _markdown(skip: set[int] = frozenset()) -> str:
    lines = ["| Item | A | B | C | D |", "| --- | --- | --- | --- | --- |"]
    for r in range(ROWS):
        if r in skip:
            continue
        cells = [f"{r + 1}{c + 3}.{r}{c}" for c in range(COLS)]
        lines.append("| " + " | ".join([f"Item{r}", *cells]) + " |")
    return "\n".join(lines)


class TestAxisBandPredicate:
    @pytest.mark.parametrize(
        "tokens",
        [
            tuple(_years(1992, 2005)),
            tuple(_years(1990, 2008, 2)),
            tuple(_years(2000, 2007) + _years(2000, 2007)),
            # overprinted labels repeat adjacent tokens
            ("2000", "2000", "2001", "2002", "2002", "2003", "2004"),
        ],
    )
    def test_axis_like(self, tokens):
        assert _is_axis_tick_band(tokens)

    @pytest.mark.parametrize(
        "tokens",
        [
            (),
            ("4.5", "4.6", "4.7"),  # decimals are measurements
            ("1,000", "2,000", "3,000"),
            ("-1", "0", "1"),
            ("3", "17", "40", "41"),  # not equal steps
            ("2000", "2001"),  # two terms constrain nothing
            tuple(_years(1992, 2005)) + ("7",),  # a stray value breaks the run
        ],
    )
    def test_not_axis_like(self, tokens):
        assert not _is_axis_tick_band(tokens)


class TestCompleteReadingNotTruncated:
    def test_complete_reading_with_chart_axes_is_not_truncated(self):
        words = _table_words() + _chart_words()
        assert table_truncated(_markdown(), words) is False

    def test_baseline_without_charts_is_not_truncated(self):
        assert table_truncated(_markdown(), _table_words()) is False


class TestRealTruncationStillCaught:
    def test_tail_cut_with_chart_axes_is_truncated(self):
        words = _table_words() + _chart_words()
        assert table_truncated(_markdown(skip={ROWS - 2, ROWS - 1}), words) is True

    def test_middle_cut_with_chart_axes_is_truncated(self):
        words = _table_words() + _chart_words()
        assert table_truncated(_markdown(skip={3, 4}), words) is True

    def test_cut_is_caught_with_and_without_charts_alike(self):
        """The difference under test: adding chart axis bands changes nothing."""
        cut = _markdown(skip={3, 4})
        assert table_truncated(cut, _table_words()) == table_truncated(
            cut, _table_words() + _chart_words()
        )
