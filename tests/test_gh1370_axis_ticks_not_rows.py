"""#1370: chart axis tick lines (year labels under a chart) are not table rows.

Term (b) of ``table_truncated`` counted every table-shaped native band as a
table row, so a complete reading of a page with a small table and charts was
called truncated. The exclusion must not open a hole for real truncation, so
the guard tests are DIFFERENCE tests: a cut reading gets the same verdict with
the axis exclusion active and with it patched off.

Hermetic: synthetic fitz-shaped word tuples and Markdown.
"""

from __future__ import annotations

import pytest

import socr.tables.structure_check as sc
from socr.tables.structure_check import _is_axis_tick_band, table_truncated

COLS = 4
COL_X = [200.0 + 50.0 * c for c in range(COLS)]


def _word(x: float, y: float, text: str) -> tuple:
    return (x, y, x + 24.0, y + 8.0, text, 0, 0, 0)


def _row_words(y: float, label: str, values: list[str]) -> list:
    words = [_word(20.0 + 26.0 * j, y, tok) for j, tok in enumerate(label.split())]
    words += [_word(COL_X[c], y, v) for c, v in enumerate(values)]
    return words


def _tick_band(y: float, labels: list[str], x0: float = 20.0, pitch: float = 30.0) -> list:
    return [_word(x0 + pitch * i, y, lab) for i, lab in enumerate(labels)]


def _years(first: int, last: int, step: int = 1) -> list[str]:
    return [str(y) for y in range(first, last + 1, step)]


def _page(rows: list[tuple[str, list[str]]], y0: float = 100.0) -> list:
    words: list = []
    for i, (label, values) in enumerate(rows):
        words += _row_words(y0 + 12.0 * i, label, values)
    return words


def _markdown(rows: list[tuple[str, list[str]]], skip: set[int] = frozenset()) -> str:
    lines = ["| Item | C0 | C1 | C2 | C3 |", "| --- | --- | --- | --- | --- |"]
    for i, (label, values) in enumerate(rows):
        if i not in skip:
            lines.append("| " + " | ".join([label, *values]) + " |")
    return "\n".join(lines)


def _chart_words() -> list:
    """Two stacked charts, then two side-by-side charts merged into one band."""
    words = _tick_band(400.0, _years(1992, 2005))
    words += _tick_band(500.0, _years(1992, 2005))
    side = _years(2000, 2007)
    words += _tick_band(600.0, side + side, x0=40.0, pitch=28.0)
    return words


def _verdicts(markdown: str, words: list, monkeypatch) -> tuple[bool, bool]:
    """(exclusion active, exclusion patched off)."""
    active = table_truncated(markdown, words)
    # raising=False: this file can then run against an older commit that lacks
    # one of the two names (the first commit has only ``_is_axis_tick_band``,
    # which took tokens), instead of dying with AttributeError. The patch
    # still switches the exclusion off there, which is what makes the same
    # tests fail on those commits.
    monkeypatch.setattr(sc, "_axis_runs", lambda band_words: None, raising=False)
    monkeypatch.setattr(sc, "_is_axis_tick_band", lambda *_a: False, raising=False)
    return active, table_truncated(markdown, words)


DECIMAL_ROWS = [(f"Item{r}", [f"{r + 1}{c + 3}.{r}{c}" for c in range(COLS)]) for r in range(8)]

# input 1: a table whose labelled rows include integer progressions
MIXED_ROWS = [
    ("GDP growth", ["2.1", "2.4", "1.9", "3.0"]),
    ("Unemployment", ["4.6", "4.5", "4.7", "4.4"]),
    ("Target rate bp", ["25", "50", "75", "100"]),
    ("Meetings held", ["2", "4", "6", "8"]),
    ("Maturity yrs", ["1", "3", "5", "7"]),
    ("Core PCE", ["1.8", "2.2", "2.0", "2.3"]),
    ("Payrolls th", ["100", "200", "300", "400"]),
    ("CPI", ["2.9", "3.1", "2.7", "3.3"]),
    ("Housing starts", ["1.31", "1.27", "1.42", "1.38"]),
    ("Vacancies", ["4.11", "4.02", "3.97", "4.20"]),
]

# input 2: every row a labelled progression
TENOR_NAMES = "one two three four five six seven eight nine ten".split()
TENOR_ROWS = [
    (f"Tenor {n}", [str(k * m) for m in (5, 10, 15, 20)])
    for k, n in enumerate(TENOR_NAMES, start=1)
]

# input 3: three panels, each a labelled year row plus decimal rows
PANEL_ROWS = []
for _p, _name in enumerate("ABC"):
    PANEL_ROWS.append((f"Panel {_name}", [str(2001 + 4 * _p + c) for c in range(COLS)]))
    for _r in range(3):
        PANEL_ROWS.append(
            (f"Series {_name}{_r}", [f"{_p + 1}{_r}.{c}{_r}{_p}" for c in range(COLS)])
        )


class TestAxisBandPredicate:
    @staticmethod
    def _band(texts: list[str], pitch: float = 30.0) -> list:
        return _tick_band(100.0, texts, pitch=pitch)

    @pytest.mark.parametrize(
        "texts",
        [
            _years(1992, 2005),
            _years(1990, 2008, 2),
            _years(2000, 2007) + _years(2000, 2007),
        ],
    )
    def test_axis_like(self, texts):
        assert _is_axis_tick_band(self._band(texts))

    def test_overprinted_labels_at_the_same_x_count_once(self):
        words = [
            _word(110.0, 10.0, "2000"),
            _word(110.0, 10.0, "2000"),
            _word(130.0, 10.0, "2001"),
            _word(150.0, 10.0, "2002"),
            _word(150.00002, 10.0, "2002"),  # measured on p049: x0 differs by 2e-5
        ]
        assert _is_axis_tick_band(words)

    @pytest.mark.parametrize(
        "texts",
        [
            [],
            ["4.5", "4.6", "4.7"],
            ["1,000", "2,000", "3,000"],
            ["-1", "0", "1"],
            ["3", "17", "40", "41"],
            ["2000", "2001"],
            ["5", "10", "15", "3", "6", "9"],  # two progressions, different steps
            ["1", "5", "9", "2", "4", "6"],
            ["3", "6", "9", "9", "9"],  # repeats in different slots are not overprints
            ["$100", "$200", "$300"],
            ["5*", "10*", "15*"],
            _years(1992, 2005) + ["7"],
        ],
    )
    def test_not_axis_like(self, texts):
        assert not _is_axis_tick_band(self._band(texts))

    def test_a_labelled_row_is_not_an_axis_band(self):
        assert not _is_axis_tick_band(
            _row_words(100.0, "Payrolls th", ["100", "200", "300", "400"])
        )


class TestCompleteReadingNotTruncated:
    def test_complete_reading_with_chart_axes_is_not_truncated(self):
        words = _page(DECIMAL_ROWS) + _chart_words()
        assert table_truncated(_markdown(DECIMAL_ROWS), words) is False

    def test_baseline_without_charts_is_not_truncated(self):
        assert table_truncated(_markdown(DECIMAL_ROWS), _page(DECIMAL_ROWS)) is False


class TestRealTruncationStillCaught:
    def test_tail_cut_with_chart_axes_is_truncated(self):
        words = _page(DECIMAL_ROWS) + _chart_words()
        assert table_truncated(_markdown(DECIMAL_ROWS, skip={6, 7}), words) is True

    def test_middle_cut_with_chart_axes_is_truncated(self):
        words = _page(DECIMAL_ROWS) + _chart_words()
        assert table_truncated(_markdown(DECIMAL_ROWS, skip={3, 4}), words) is True

    @pytest.mark.parametrize("skip", [{2, 3}, {0, 1}, {7, 8, 9}, {0, 1, 5, 7, 8}])
    def test_input1_labelled_progression_rows(self, skip, monkeypatch):
        words = _page(MIXED_ROWS)
        active, off = _verdicts(_markdown(MIXED_ROWS, skip), words, monkeypatch)
        assert active == off is True

    def test_input2_all_rows_progressions(self, monkeypatch):
        words = _page(TENOR_ROWS)
        keep = set(range(1, 10))
        active, off = _verdicts(_markdown(TENOR_ROWS, keep), words, monkeypatch)
        assert active == off is True

    @pytest.mark.parametrize("skip", [{1, 2}, {1, 2, 3}, {8, 9, 10, 11}])
    def test_input3_labelled_year_rows(self, skip, monkeypatch):
        words = _page(PANEL_ROWS)
        active, off = _verdicts(_markdown(PANEL_ROWS, skip), words, monkeypatch)
        assert active == off is True

    @pytest.mark.parametrize("skip", [{1, 2, 3}, {0, 3, 5, 6}])
    def test_input4_candidate_wrote_axis_rows_as_table_rows(self, skip, monkeypatch):
        """A chart transcribed as a table with the years as a body row: the
        candidate counts those rows, so the native bands must count too."""
        words = _page(DECIMAL_ROWS)
        for i, y in enumerate((400.0, 500.0, 600.0)):
            words += _tick_band(y, _years(2000, 2007))
        axis_rows = [(f"chart {i}, x-axis", _years(2000, 2007)) for i in range(3)]
        md = _markdown(DECIMAL_ROWS, skip)
        md += "\n\n| Chart | " + " | ".join(f"c{i}" for i in range(8)) + " |\n"
        md += "| --- |" + " --- |" * 8 + "\n"
        md += "\n".join("| " + " | ".join([lab, *vals]) + " |" for lab, vals in axis_rows)
        active, off = _verdicts(md, words, monkeypatch)
        assert active == off is True

    def test_cut_verdict_is_independent_of_chart_axes(self):
        cut = _markdown(DECIMAL_ROWS, skip={3, 4})
        assert table_truncated(cut, _page(DECIMAL_ROWS)) == table_truncated(
            cut, _page(DECIMAL_ROWS) + _chart_words()
        )


def _page_split_labels(rows: list[tuple[str, list[str]]], pitch: float = 24.0) -> list:
    """Layout B: the label sits 12 pt above its values, so every integer
    progression row is a digit-only band."""
    words: list = []
    for i, (label, values) in enumerate(rows):
        y = 100.0 + pitch * i
        words += [_word(20.0 + 26.0 * j, y, tok) for j, tok in enumerate(label.split())]
        words += [_word(COL_X[c], y + 12.0, v) for c, v in enumerate(values)]
    return words


def _with_chart_table(md: str, rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    out = md + "\n\n| Chart | " + " | ".join(f"c{i}" for i in range(width)) + " |\n"
    out += "| --- |" + " --- |" * width + "\n"
    out += "\n".join("| " + " | ".join(["chart x-axis", *r]) + " |" for r in rows)
    return out


def _chart_axis_words() -> list:
    side = _years(2000, 2007)
    return _tick_band(400.0, side) + _tick_band(500.0, side)


def _drop_last(rows, k: int) -> set[int]:
    return set(range(len(rows) - k, len(rows)))


T = _years(2000, 2007)


class TestImperfectAxisCopy:
    """A: the reader wrote an imperfect copy of a chart axis as a table row.
    The candidate counts that row, so the native axis band must stay counted."""

    @pytest.mark.parametrize(
        ("name", "chart_rows", "k"),
        [
            ("skipped tick", [_years(2000, 2006) + ["2008"]] * 2, 2),
            ("skipped tick", [_years(2000, 2006) + ["2008"]] * 2, 3),
            ("y value in row", [[*T, "67.6"]] * 2, 2),
            ("y value in row", [[*T, "67.6"]] * 2, 3),
            ("overprints", [["2000", "2000", "2001", "2002", "2002", *T[3:]]], 1),
            ("overprints", [["2000", "2000", "2001", "2002", "2002", *T[3:]]], 2),
            ("first four ticks", [T[:4]] * 3, 3),
        ],
    )
    def test_cut_still_truncated(self, name, chart_rows, k, monkeypatch):
        words = _page(DECIMAL_ROWS) + _chart_axis_words()
        md = _with_chart_table(_markdown(DECIMAL_ROWS, _drop_last(DECIMAL_ROWS, k)), chart_rows)
        active, off = _verdicts(md, words, monkeypatch)
        assert active == off is True


class TestLabelFreeProgressionRowsInsideTable:
    """B: labels 12 pt above the values, so integer rows are digit-only bands."""

    @pytest.mark.parametrize("skip", [{2, 3}, {2, 3, 4}, {2, 3, 4, 6}])
    def test_integer_rows_dropped(self, skip, monkeypatch):
        words = _page_split_labels(MIXED_ROWS)
        active, off = _verdicts(_markdown(MIXED_ROWS, skip), words, monkeypatch)
        assert active == off is True

    def test_integer_rows_misread_and_decimal_rows_dropped(self, monkeypatch):
        misread = {2: (0, "26"), 3: (3, "9"), 4: (3, "8"), 6: (3, "409")}
        rows = []
        for i, (label, values) in enumerate(MIXED_ROWS):
            if i in misread:
                c, v = misread[i]
                values = [*values[:c], v, *values[c + 1 :]]
            rows.append((label, values))
        words = _page_split_labels(MIXED_ROWS)
        active, off = _verdicts(_markdown(rows, {0, 1, 5, 7, 8}), words, monkeypatch)
        assert active == off is True

    def test_year_caption_above_the_table_is_still_excluded(self):
        """The caption has no aligned band above it, so it is not a body row:
        a complete reading with a caption and a chart is not truncated."""
        caption = _tick_band(80.0, _years(2001, 2004), x0=COL_X[0], pitch=50.0)
        words = _page(DECIMAL_ROWS) + caption + _chart_words()
        assert table_truncated(_markdown(DECIMAL_ROWS), words) is False
