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
        # Column 4 holds a number in `k` of the six data rows and text in the rest; text also sits in it below the table.
        def fired(k: int) -> set[str]:
            rows = [list(r) for r in ROWS]
            for j, r in enumerate(rows[k:]):
                r[4] = "ab cd"[: 2 + j % 2] + "xyz"[j % 3]  # distinct text: not a placeholder
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


class TestNumericClassifierIsTheVerifiers:
    """GH-932: a cell the verifier calls numeric (currency, ``∗``/``✱``) is a number, so a column of them is a numeric column and a footrow of text
    under it defers. Each pin is a difference: the same markdown with and without the
    ``tail`` row, and the dressed column against the plain one."""

    FOOTROW = ["a See footnote", "table 8b.", "", "", ""]

    @staticmethod
    def _dressed(prefix="", suffix="", *, tail=()):
        # Source words carry the bare numbers (the glyph is not in the numeric multiset);
        # only the markdown cells in columns 1-4 are dressed.
        words = _words([HEADER] + [list(r) for r in ROWS])
        body = [[r[0]] + [prefix + c + suffix for c in r[1:]] for r in ROWS]
        return words, _md(HEADER, body + [list(t) for t in tail])

    def test_currency_column_with_a_text_footrow_defers(self) -> None:
        for sign in ("$", "€", "£", "¥"):
            words, clean = self._dressed(sign)
            _, faulty = self._dressed(sign, tail=[self.FOOTROW[:1] + ["table 8b.", "", "", ""]])
            assert _fired(words, clean) == set(), sign
            assert _fired(words, faulty) == {TNC}, sign

    def test_star_column_with_a_text_footrow_defers(self) -> None:
        # ★ and ⋆ are not in the verifier's marks: such a markdown never exact-passes, so no
        # SHIP exists for the gate to override (measured; widening the shared marks is out of scope).
        for star in ("∗", "✱"):
            words, clean = self._dressed(suffix=star)
            _, faulty = self._dressed(suffix=star, tail=[["", "see note", "", "", ""]])
            assert _fired(words, clean) == set(), star
            assert _fired(words, faulty) == {TNC}, star

    def test_cell_kind_reads_the_verifiers_numbers(self) -> None:
        for cell in ("$1,234", "€5.2", "£3", "¥100", "0.3∗", "0.3✱", "∗0.05"):
            assert ship_gate._cell_kind(cell) == ship_gate._NUMBER, cell
        # Still text or nothing: a currency sign or star alone, a word, a long marker.
        for cell in ("$", "★", "$ total", "0.23abc"):
            assert ship_gate._cell_kind(cell) != ship_gate._NUMBER, cell

    def test_dressed_columns_keep_the_must_not_fire_controls_quiet(self) -> None:
        for sign in ("$", "€"):
            words, md = self._dressed(sign, tail=[["", "", "", "", ""]])
            assert _fired(words, md) == set()
            words, md = self._dressed(sign, tail=[["", "–", "–", "", ""]])
            assert _fired(words, md) == set()
            words, md = self._dressed(sign, tail=[["a See footnote table 8b.", "", "", "", ""]])
            assert _fired(words, md) == set()


class TestDisjointPanels:
    """GH-932 (Astra, PR #935): value rows that each fill only their own panel's columns
    (Alpha/Beta in A-B, Gamma/Delta in C-D) leave every column at exactly half of the data
    rows, so a count over ALL data rows finds no numeric column and the predicate abstains.
    A column's numbers are counted against the data rows that have a cell there."""

    PANELS = [
        ["Alpha", "0.11", "0.12", "", ""],
        ["Beta", "0.21", "0.22", "", ""],
        ["Gamma", "", "", "0.31", "0.32"],
        ["Delta", "", "", "0.41", "0.42"],
    ]

    def _table(self, prefix="", *, tail):
        words = _words([HEADER] + [list(r) for r in self.PANELS])
        body = [
            [c if not (c[:1].isdigit() and prefix) else prefix + c for c in r] for r in self.PANELS
        ]
        return words, _md(HEADER, body + [list(t) for t in tail])

    def test_currency_panels_with_a_text_footrow_defer(self) -> None:
        note = [["", "see note", "", "", ""]]
        words, clean = self._table("$", tail=[])
        _, faulty = self._table("$", tail=note)
        assert _fired(words, clean) == set()
        assert _fired(words, faulty) == {TNC}

    def test_plain_number_panels_with_a_text_footrow_defer(self) -> None:
        note = [["", "see note", "", "", ""]]
        words, clean = self._table(tail=[])
        _, faulty = self._table(tail=note)
        assert _fired(words, clean) == set()
        assert _fired(words, faulty) == {TNC}

    def test_an_empty_column_is_not_numeric(self) -> None:
        kinds = [["text", "number", "empty"], ["text", "number", "empty"]]
        assert ship_gate._numeric_columns(kinds, [0, 1], panels=True) == [1]

    def test_one_filled_cell_is_not_a_column(self) -> None:
        kinds = [
            ["text", "number", "empty"],
            ["text", "empty", "empty"],
            ["text", "empty", "empty"],
        ]
        assert ship_gate._numeric_columns(kinds, [0, 1, 2], panels=True) == []


class TestMonotone:
    """GH-932 (Astra, PR #935 round 2): the disjoint-panel rule only ADDS faults. Every
    fault the original rule finds is still found: the gate is DEFER-only, so a union of
    the two rules can add a DEFER and never lose one."""

    GRID = [
        ["Alpha", "11", "12", "13", "14"],
        ["Beta", "21", "22", "23", "24"],
        ["Gamma", "31", "32", "33", "34"],
        ["Delta", "41", "42", "43", "44"],
    ]
    # Two notes, each with two year-like numbers in the same two columns and text in a third.
    NOTES = [
        ["Note one", "1968", "2021", "see appendix", ""],
        ["Note two", "1970", "2022", "sample restriction", ""],
    ]

    def test_notes_sharing_a_column_set_are_still_caught(self) -> None:
        # Original rule: columns 1-4 are numeric, each note covers 2 of 4 (not data), the
        # text in column 3 fires. The panel rule alone would admit both notes as data rows
        # (they share the support {1, 2}) and lose that catch.
        rows = self.GRID + self.NOTES
        words = _words([HEADER] + rows)
        md = _md(HEADER, rows)
        assert _fired(words, md) == {TNC}
        details = [f["detail"] for f in ship_gate.native_ship_gate(words, md, line_dirs=UNCHECKED)]
        assert len(details) == 2, details
        assert "row 5" in details[0] and "see appendix" in details[0]
        assert "row 6" in details[1] and "sample restriction" in details[1]

    def test_plain_panels_next_to_currency_panels_with_a_footrow_defer(self) -> None:
        # Panel one is plain numbers, panel two is currency-prefixed: the column sets differ
        # in kind, not only in position. The footrow's text sits in a currency column.
        panels = [
            ["Alpha", "0.11", "0.12", "", ""],
            ["Beta", "0.21", "0.22", "", ""],
            ["Gamma", "", "", "$31", "$32"],
            ["Delta", "", "", "$41", "$42"],
        ]
        # The source prints the bare numbers; only the markdown carries the currency sign.
        words = _words([HEADER] + [[c.replace("$", "") for c in r] for r in panels])
        assert _fired(words, _md(HEADER, panels)) == set()
        assert _fired(words, _md(HEADER, panels + [["", "", "", "see note", ""]])) == {TNC}
