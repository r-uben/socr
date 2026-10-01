"""GH-917: ``text_in_numeric_column``, a DEFER-only predicate in the native-first ship gate.

A shipped grid row below the table's first data row that is not itself a data row and
carries letters in a column the data rows establish as numeric (a footnote paragraph, a
caption, equation fragments, a "(Continued)" marker emitted as a row) defers.

Every test pins a DIFFERENCE: the same source words, one markdown with the fault and one
without, and (through ``plan_native_table``) the same markdown with the gate switched off, so
the exact-pass the gate overrides is proven to exist. Words are synthetic tuples in the shape
of ``page.get_text("words")``; nothing reads a corpus file and nothing needs a provider.
"""

from __future__ import annotations

from unittest.mock import patch

from native_table_fixtures import HEADER, ROWS, UNCHECKED
from test_gh916_native_ship_gate import _md, _plan, _predicates, _words

from socr.tables import native_first as nf
from socr.tables import ship_gate
from socr.tables.native_first import DEFER, SHIP, plan_native_table

TNC = ship_gate.TEXT_IN_NUMERIC_COLUMN
NO_CELLS = ["", "", "", "", ""]


def _fired(words, md) -> set[str]:
    return {f["predicate"] for f in ship_gate.native_ship_gate(words, md, line_dirs=UNCHECKED)}


def _grid(rows=ROWS, *, tail=(), head=(), inner=None):
    """``(words, markdown)``: source words for *rows* only; *head*, *tail* and *inner* are
    markdown-only rows (``inner`` is ``{row index before which to insert: row}``)."""
    words = _words([HEADER] + [list(r) for r in rows])
    body = [list(r) for r in rows]
    for at, extra in sorted((inner or {}).items(), reverse=True):
        body.insert(at, list(extra))
    return words, _md(HEADER, [list(r) for r in head] + body + [list(r) for r in tail])


def _gate_off(words, md):
    with patch.object(nf, "native_ship_gate", return_value=()):
        return plan_native_table(words, md, line_dirs=UNCHECKED)


class TestFaultAndNoFault:
    def test_difference_pin_footnote_row_in_numeric_columns(self) -> None:
        footnote = ["a See footnote", "table 8b.", "", "", ""]
        words, clean = _grid()
        _, faulty = _grid(tail=[footnote])
        assert _plan(words, clean).action == SHIP
        assert _gate_off(words, faulty).action == SHIP, "the exact-pass the gate overrides exists"
        plan = _plan(words, faulty)
        assert plan.action == DEFER and _predicates(plan) == {TNC}
        assert plan.reason.startswith(ship_gate.SHIP_GATE_REASON_PREFIX)

    def test_same_text_in_one_label_cell_ships(self) -> None:
        # The note is one merged cell in the label column: no numeric column holds text.
        words, md = _grid(tail=[["a See footnote table 8b.", "", "", "", ""]])
        assert _plan(words, md).action == SHIP

    def test_continued_marker_in_the_last_column_defers(self) -> None:
        words, md = _grid(tail=[["", "", "", "", "(Continued)"]])
        assert _fired(words, md) == {TNC}

    def test_fault_names_row_column_and_text(self) -> None:
        words, md = _grid(tail=[["", "", "", "", "(Continued)"]])
        (fault,) = ship_gate.native_ship_gate(words, md, line_dirs=UNCHECKED)
        assert fault["predicate"] == TNC
        assert "row 7" in fault["detail"] and "(Continued)" in fault["detail"]

    def test_sub_header_floating_over_numeric_columns_between_data_rows_defers(self) -> None:
        words, md = _grid(inner={3: NO_CELLS[:2] + ["m"] + NO_CELLS[3:]})
        assert _fired(words, md) == {TNC}

    def test_signed_numbers_with_a_detached_glyph_still_establish_the_column(self) -> None:
        # The PDF prints "- 0.253" as two words; the grid cell is "- 0.253". Every value
        # in column 1 is signed, so the column is numeric only if such a cell is a number.
        rows = [list(r) for r in ROWS]
        words = _words([HEADER] + rows)
        md_rows = [[r[0], "\u2013 " + r[1]] + r[2:] for r in rows]
        faulty = _md(HEADER, md_rows + [["", "see note", "", "", ""]])
        assert _fired(words, faulty) == {TNC}
        assert _fired(words, _md(HEADER, md_rows)) == set()


class TestMustNotFire:
    def test_header_band_above_the_first_data_row_is_exempt(self) -> None:
        # HEADER already sits in the numeric columns; add a second heading row above it.
        words, md = _grid(head=[["", "Estimates", "", "Spreads", ""]])
        assert _fired(words, md) == set()

    def test_panel_label_in_the_label_column_between_data_rows_is_exempt(self) -> None:
        words, md = _grid(inner={3: ["Panel B: spreads", "across horizons", "", "", ""]})
        assert _fired(words, md) == set()
        # The same cells AFTER the last data row are a footnote, not a panel label.
        words, md = _grid(tail=[["Panel B: spreads", "across horizons", "", "", ""]])
        assert _fired(words, md) == {TNC}

    def test_panel_label_spanning_into_numeric_columns_is_exempt(self) -> None:
        words, md = _grid(inner={3: ["Big, low-profitability", "growth firms:", "dSM", "< 0", ""]})
        assert _fired(words, md) == set()

    def test_footnote_markers_and_stars_on_numbers(self) -> None:
        # A trailing row of marked numbers carries no letters of its own.
        for cell in ("0.23*", "0.23**", "0.23†", "0.23a", "(0.23)", "[0.23]", "12.5%"):
            words, md = _grid(tail=[["", cell, "", "", ""]])
            assert _fired(words, md) == set(), cell
        # One letter is a marker; a word is text (the named constant, not a vocabulary).
        assert ship_gate._MARKER_MAX_LETTERS == 1
        words, md = _grid(tail=[["", "0.23abc", "", "", ""]])
        assert _fired(words, md) == {TNC}

    def test_dash_placeholders_are_not_text(self) -> None:
        for dash in ("–", "—", "-", "−"):
            words, md = _grid(tail=[["", dash, dash, "", ""]])
            assert _fired(words, md) == set(), dash

    def test_range_fragments_without_letters_are_not_text(self) -> None:
        words, md = _grid(inner={3: ["", "", "1927–", "1986", ""]})
        assert _fired(words, md) == set()

    def test_standard_error_parentheses_are_numbers(self) -> None:
        words, md = _grid(tail=[["", "(0.12)", "(0.13)", "", ""]])
        assert _fired(words, md) == set()

    def test_placeholder_the_column_itself_repeats_is_exempt(self) -> None:
        # "n.a." sits in column 3 of a data row AND of the row below: the same column holds
        # it twice, so it is a placeholder. A single occurrence, or the same text spread one
        # per column, is text (documented limit: a false DEFER costs one model read).
        rows = [list(r) for r in ROWS]
        rows[1][3] = "n.a."
        words, md = _grid(rows, tail=[["", "", "", "n.a.", ""]])
        assert _fired(words, md) == set()
        words, md = _grid(rows, tail=[["", "n.a.", "n.a.", "", ""]])
        assert _fired(words, md) == {TNC}
        words, md = _grid(tail=[["", "", "", "n.a.", ""]])
        assert _fired(words, md) == {TNC}

    def test_a_different_text_is_not_covered_by_the_placeholder(self) -> None:
        rows = [list(r) for r in ROWS]
        rows[1][3] = "n.a."
        words, md = _grid(rows, tail=[["", "", "", "see note", ""]])
        assert _fired(words, md) == {TNC}

    def test_a_row_with_placeholders_after_its_numbers_is_still_data(self) -> None:
        # The last two rows end in "n.a." twice over in columns 3 and 4 (so each is its
        # column's placeholder, not text). They stay data rows, so the note above them is
        # interior (a label-column row, exempt). If placeholders counted as text those rows
        # would not be data, the last data row would move up, and the note would sit below
        # it and fire.
        rows = [list(r) for r in ROWS]
        rows[4] = ["CB", "0.180", "0.171", "n.a.", "n.a."]
        rows[5] = ["TR", "0.310", "0.220", "n.a.", "n.a."]
        words, md = _grid(rows, inner={4: ["See note", "text", "", "", ""]})
        assert _fired(words, md) == set()

    def test_a_panel_label_with_a_year_range_is_not_a_data_row(self) -> None:
        # "Panel A 1968 2021" pairs with the source and has two numbers, but they reach two
        # of four numeric columns. Were it data, the heading row below it would be below the
        # first data row and fire.
        words = _words(
            [HEADER, ["Panel A", "1968", "2021", "", ""], NO_CELLS] + [list(r) for r in ROWS]
        )
        md = _md(
            HEADER,
            [["Panel A", "1968", "2021", "", ""], ["", "Mean", "Std", "P25", ""]]
            + [list(r) for r in ROWS],
        )
        assert _fired(words, md) == set()


class TestDerivation:
    def test_numeric_column_needs_more_than_half_the_data_rows(self) -> None:
        # Column 4 holds a number in `k` of the six data rows; text sits in it below the table.
        def fired(k: int) -> set[str]:
            rows = [list(r) for r in ROWS]
            for r in rows[k:]:
                r[4] = ""
            words, md = _grid(rows, tail=[["", "", "", "", "see note"]])
            return _fired(words, md)

        assert fired(3) == set(), "exactly half is not most"
        assert fired(4) == {TNC}

    def test_fewer_than_two_data_rows_abstains(self) -> None:
        words, md = _grid(ROWS[:1], tail=[["", "see note", "", "", ""]])
        assert _fired(words, md) == set()
        words, md = _grid(ROWS[:2], tail=[["", "see note", "", "", ""]])
        assert _fired(words, md) == {TNC}

    def test_a_row_is_data_only_when_the_source_pairs_it(self) -> None:
        # The grid has six rows; the source prints N of them. Grid rows no source row
        # confirms establish nothing: one confirmed row abstains, two fire.
        def fired(n: int) -> set[str]:
            words = _words([HEADER] + [list(r) for r in ROWS[:n]])
            md = _md(HEADER, [list(r) for r in ROWS] + [["", "see note", "", "", ""]])
            return _fired(words, md)

        assert fired(1) == set()
        assert TNC in fired(2)

    def test_a_one_number_row_is_not_data(self) -> None:
        # A table with ONE numeric column has no row with two values: nothing establishes
        # a data row, so a note in that column is not reported (the rowizer's own minimum
        # for a row of values is two).
        rows = [[r[0], r[1], "", "", ""] for r in ROWS]
        words, md = _grid(rows, tail=[["", "see note", "", "", ""]])
        assert _fired(words, md) == set()

    def test_two_label_cells_and_two_values_is_a_data_row(self) -> None:
        # The label spans two cells (columns 0 and 1) and the values sit in columns 2 and 3,
        # so columns 2 and 3 are the numeric ones and the note in column 2 fires.
        rows = [[r[0], "firms", r[1], r[2], ""] for r in ROWS]
        words, md = _grid(rows, tail=[["", "", "see note", "", ""]])
        assert _fired(words, md) == {TNC}


def test_a_separator_only_block_does_not_mask_another_blocks_fault() -> None:
    """Astra on PR #931: a separator-only block (``| --- | --- |``) made the predicate
    raise, and the gate's handler then replaced the real fault with ``gate_error``."""
    words, md = _grid(tail=[["", "", "", "", "(Continued)"]])
    assert _fired(words, md + "\n\n| --- | --- |\n") == {TNC}
