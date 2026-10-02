"""GH-958: two DEFER-only ship-gate changes found by the vision audit of the native-first SHIPs.

(A) ``text_in_numeric_column``: a panel-label row that spills text into the numeric columns is no
    longer exempt when its joined cells equal ONE whole source line and that line is one run (a
    sentence the PDF prints in one piece, emitted one word per cell).
(B) ``header_over_empty_column``: a header cell over a column no data row fills, next to a numeric
    column with no header: the values sit one lane off their header.

Every test pins a DIFFERENCE: the same markdown with and without the source line that backs it, the
same source line over one run and over lanes, the same grid with the header cell on the empty and on
the filled column. Words are synthetic PyMuPDF-shaped tuples; nothing reads a corpus file and
nothing needs a provider (``native_ship_gate`` and ``plan_native_table`` never consult one).
"""

from __future__ import annotations

from native_table_fixtures import COL_XS, HEADER, PITCH, ROWS, UNCHECKED, Y0
from test_gh916_native_ship_gate import _md, _plan, _predicates, _words
from test_gh936_prose_in_header import WORD_SPACE, _grid_words, _line, _w

from socr.tables import ship_gate
from socr.tables.native_first import DEFER, SHIP

TNC = ship_gate.TEXT_IN_NUMERIC_COLUMN
HOEC = ship_gate.HEADER_OVER_EMPTY_COLUMN
#: A panel label whose words sit one per cell, two of them in numeric columns.
SCATTERED = ["Big, low-profitability", "growth", "firms:", "dSM", ""]
#: Between the data rows ROWS[2] (y = Y0 + 3 pitches) and ROWS[3].
SENTENCE_Y = Y0 + 3.5 * PITCH


def _fired(words, md) -> set[str]:
    return {f["predicate"] for f in ship_gate.native_ship_gate(words, md, line_dirs=UNCHECKED)}


def _md_with(label_row) -> str:
    body = [list(r) for r in ROWS]
    body.insert(3, list(label_row))
    return _md(HEADER, body)


def _sentence_run() -> list[tuple]:
    """The scattered cells' words as one run on one source line."""
    return _line(
        "Big, low-profitability growth firms: dSM".split(), COL_XS[0], SENTENCE_Y, 30, WORD_SPACE
    )


def _sentence_over_lanes() -> list[tuple]:
    """The same words, one per lane of the table, on one source line (a positioned sub-header)."""
    toks = "Big, low-profitability growth firms: dSM".split()
    cells = [toks[0] + " " + toks[1], toks[2], toks[3], toks[4]]
    return [_w(COL_XS[i], SENTENCE_Y, t, 30, 0, i) for i, t in enumerate(cells)]


class TestScatteredPanelHeading:
    def test_difference_pin_a_single_run_source_line_makes_the_exempt_row_fire(self) -> None:
        md = _md_with(SCATTERED)
        base = _grid_words()
        assert _fired(base, md) == set(), "no source line: the exemption stands"
        words = base + _sentence_run()
        assert _fired(words, md) == {TNC}
        plan = _plan(words, md)
        assert plan.action == DEFER and _predicates(plan) == {TNC}
        assert _plan(words, md, gate=False).action == SHIP, "the exact-pass the gate overrides"

    def test_the_same_row_over_separate_lanes_is_a_positioned_sub_header_and_quiet(self) -> None:
        # fama p469 shape: the words of the row are the lane headings, not a sentence.
        md = _md_with(["5-Yr SR", "High", "Mid", "Low", ""])
        cells = ["5-Yr SR", "High", "Mid", "Low"]
        words = _grid_words() + [
            _w(COL_XS[i], SENTENCE_Y, t, 30, 0, i) for i, t in enumerate(cells)
        ]
        assert _fired(words, md) == set()
        # Same markdown as the fault, same words, spread over the lanes instead of one run.
        assert _fired(_grid_words() + _sentence_over_lanes(), _md_with(SCATTERED)) == set()

    def test_the_run_clause_is_the_only_difference(self) -> None:
        md = _md_with(SCATTERED)
        assert _fired(_grid_words() + _sentence_run(), md) == {TNC}
        assert _fired(_grid_words() + _sentence_over_lanes(), md) == set()

    def test_one_text_cell_in_the_numeric_columns_is_not_enough(self) -> None:
        md = _md_with(["Big, low-profitability growth firms:", "dSM", "", "", ""])
        words = _grid_words() + _sentence_run()
        assert _fired(words, md) == set()

    def test_a_page_with_no_measurable_word_space_abstains(self) -> None:
        words = _grid_words(prose_lines=0) + _sentence_run()
        assert _fired(words, _md_with(SCATTERED)) == set()

    def test_a_row_that_is_not_a_whole_source_line_is_exempt(self) -> None:
        # The source line carries one more word than the row's cells.
        extra = _line(
            "Big, low-profitability growth firms: dSM extra".split(),
            COL_XS[0],
            SENTENCE_Y,
            30,
            WORD_SPACE,
        )
        assert TNC not in _fired(_grid_words() + extra, _md_with(SCATTERED))


#: Year-labelled rows: the label column is a numeric lane too, so the lane count (5) and the
#: grid's column count (6, with the never-filled column) stay within the exact-pass tolerance.
YEAR_ROWS = [[str(1990 + k), *r[1:]] for k, r in enumerate(ROWS)]
YEAR_HEADER = ["Year", "b", "s", "h", "q"]


def _hoec_grid(header_cells, *, filled=None):
    """``(words, markdown)``: YEAR_ROWS in the source; the grid inserts a never-filled column 1.

    *filled* puts a value in that column for data row 2."""
    rows = [[r[0], "", *r[1:]] for r in YEAR_ROWS]
    if filled:
        rows[2][1] = filled
    words = _words([YEAR_HEADER] + [list(r) for r in YEAR_ROWS])
    return words, _md(header_cells, rows)


class TestHeaderOverEmptyColumn:
    def test_difference_pin_header_over_the_empty_column_defers(self) -> None:
        words, clean = _hoec_grid(["Year", "", "b", "s", "h", "q"])
        _, faulty = _hoec_grid(["Year", "b", "", "s", "h", "q"])
        assert _plan(words, clean).action == SHIP
        assert _plan(words, faulty, gate=False).action == SHIP, "the exact-pass the gate overrides"
        plan = _plan(words, faulty)
        assert plan.action == DEFER and _predicates(plan) == {HOEC}

    def test_fault_names_row_and_column(self) -> None:
        words, faulty = _hoec_grid(["Year", "b", "", "s", "h", "q"])
        (fault,) = ship_gate.native_ship_gate(words, faulty, line_dirs=UNCHECKED)
        assert fault["predicate"] == HOEC
        assert "header row 0 column 1" in fault["detail"]

    def test_two_line_spanning_header_over_a_filled_column_is_quiet(self) -> None:
        # stock_watson shape: a two-line heading over a value column and its blank-headed
        # standard-error column. The heading sits on a column the data fills: nothing is orphaned,
        # and a lane-centre clause would have called this one misplaced.
        two = [["", "Growth", "", "", ""], ["Year", "of GDP", "", "h", "q"]]
        rows = [[r[0], *r[1:]] for r in YEAR_ROWS]
        words = _words(two + [list(r) for r in rows])
        md = _md(["", "Growth", "", "", ""], [["Year", "of GDP", "", "h", "q"]] + rows)
        assert _fired(words, md) == set()

    def test_the_adjacent_numeric_column_must_be_headerless(self) -> None:
        words, md = _hoec_grid(["Year", "b", "", "s", "h", "q"])
        assert _fired(words, md) == {HOEC}
        # A second header row that also labels column 2 makes it a headed column.
        rows = [[r[0], "", *r[1:]] for r in YEAR_ROWS]
        labelled = _md(["Year", "b", "", "s", "h", "q"], [["", "", "est", "", "", ""]] + rows)
        words = _words([YEAR_HEADER] + [list(r) for r in YEAR_ROWS])
        assert HOEC not in _fired(words, labelled)

    def test_a_column_a_data_row_fills_is_not_empty(self) -> None:
        words, md = _hoec_grid(["Year", "b", "", "s", "h", "q"], filled="note")
        assert HOEC not in _fired(words, md)


def _run_line(text: str) -> list[tuple]:
    return _line(text.split(), COL_XS[0], SENTENCE_Y, 30, WORD_SPACE)


class TestAcceptedFires:
    def test_the_exact_original_row_fires_when_one_source_line_backs_it(self) -> None:
        # The row of the renamed GH-917 test (fama p782 origin). Markdown-only: exempt. With a
        # single-run source line it is the scattered heading itself, and it defers.
        row = ["Big, low-profitability", "growth firms:", "dSM", "< 0", ""]
        md = _md_with(row)
        assert _fired(_grid_words(), md) == set()
        assert _fired(
            _grid_words() + _run_line("Big, low-profitability growth firms: dSM < 0"), md
        ) == {TNC}

    def test_label_words_in_numeric_cells_are_the_defect_shape_and_fire(self) -> None:
        # Intended fire (fama p475 shape): a panel label whose words spill into numeric cells.
        md = _md_with(["Panel A:", "Small", "firms", "", ""])
        assert _fired(_grid_words() + _run_line("Panel A: Small firms"), md) == {TNC}

    def test_the_merged_single_cell_label_stays_quiet(self) -> None:
        md = _md_with(["Panel A: Small firms", "", "", "", ""])
        assert _fired(_grid_words() + _run_line("Panel A: Small firms"), md) == set()

    def test_a_deliberately_unfilled_column_is_a_known_accepted_false_defer(self) -> None:
        # Header Year | Forecast | blank | Actual over an always-blank Forecast column with a
        # headerless neighbour: indistinguishable from the one-lane-off defect by the grid alone.
        # A false DEFER costs one model read; the rule is not narrowed (GH-958 ruling).
        words, md = _hoec_grid(["Year", "Forecast", "", "s", "h", "q"])
        assert HOEC in _fired(words, md)
