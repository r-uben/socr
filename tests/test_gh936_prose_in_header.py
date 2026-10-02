"""GH-936: ``prose_in_header``, a DEFER-only predicate in the native-first ship gate.

A source row above the table's first data row that the grid absorbed whole into its header
rows, and that is ONE run of two or more words (no gap over ``ALIGNED_RUN_GAP_MAX_WORD_SPACES``
page word spaces), is a caption or notes sentence, not a set of column headings.

Every test pins a DIFFERENCE: the same grid with and without the caption, the same caption
with a one-run and a lane-wide layout, the same markdown with the gate switched off (so the
exact-pass the gate overrides is proven to exist). Words are synthetic PyMuPDF-shaped tuples;
nothing reads a corpus file and nothing needs a provider.
"""

from __future__ import annotations

from native_table_fixtures import CHAR_W, COL_XS, HEADER, PITCH, ROWS, UNCHECKED, WORD_H, Y0
from test_gh916_native_ship_gate import _md, _plan, _predicates

from socr.core.born_digital import ALIGNED_RUN_GAP_MAX_WORD_SPACES
from socr.tables import ship_gate
from socr.tables.native_first import DEFER, SHIP

PIH = ship_gate.PROSE_IN_HEADER
#: The page's word space. Every grid cell sits on its own text line, so the only measurable
#: gaps are the body sentence's and the caption's, and this is their median.
WORD_SPACE = 3.0
CAPTION = ["Percent", "of", "aggregate", "values"]
SPREAD = ["Percent of", "aggregate values", "", "", ""]


def _w(x: float, y: float, text: str, block: int, line: int, no: int) -> tuple:
    return (x, y, x + CHAR_W * len(text), y + WORD_H, text, block, line, no)


def _line(words: list[str], x: float, y: float, block: int, gap: float) -> list[tuple]:
    out = []
    for no, text in enumerate(words):
        out.append(_w(x, y, text, block, 0, no))
        x += CHAR_W * len(text) + gap
    return out


def _grid_words() -> list[tuple]:
    """HEADER at ``Y0`` and ROWS below it, one text line per cell, plus a body sentence far
    below the table that fixes the page's word space."""
    out, line = [], 0
    for ri, row in enumerate([HEADER] + ROWS):
        for ci, cell in enumerate(row):
            out.append(_w(COL_XS[ci], Y0 + ri * PITCH, cell, 0, line, 0))
            line += 1
    body = "Prose elsewhere on the page sets the word space".split()
    return out + _line(body, COL_XS[0], Y0 + 20 * PITCH, 9, WORD_SPACE)


def _caption_words(words: list[str], gap: float) -> list[tuple]:
    return _line(words, COL_XS[0], Y0 - PITCH, 5, gap)


def _gate_md(caption_cells: list[str] | None) -> str:
    """Markdown whose header rows are the caption (when given) then HEADER."""
    rows = [list(HEADER)] + [list(r) for r in ROWS]
    if caption_cells is not None:
        rows.insert(0, caption_cells)
    return _md(rows[0], rows[1:])


def _fired(words, md) -> set[str]:
    return {f["predicate"] for f in ship_gate.native_ship_gate(words, md, line_dirs=UNCHECKED)}


class TestFaultAndNoFault:
    def test_difference_pin_caption_run_in_the_header_defers(self) -> None:
        clean_words, clean_md = _grid_words(), _gate_md(None)
        words = clean_words + _caption_words(CAPTION, WORD_SPACE)
        md = _gate_md(SPREAD)
        assert _plan(clean_words, clean_md).action == SHIP
        assert _plan(words, md, gate=False).action == SHIP, (
            "the exact-pass the gate overrides exists"
        )
        plan = _plan(words, md)
        assert plan.action == DEFER and _predicates(plan) == {PIH}
        assert plan.reason.startswith(ship_gate.SHIP_GATE_REASON_PREFIX)

    def test_fault_names_the_row_and_the_word_count(self) -> None:
        words = _grid_words() + _caption_words(CAPTION, WORD_SPACE)
        (fault,) = ship_gate.native_ship_gate(words, _gate_md(SPREAD), line_dirs=UNCHECKED)
        assert fault["predicate"] == PIH
        assert f"y={round(Y0 - PITCH)}" in fault["detail"] and "4 words" in fault["detail"]

    def test_the_same_caption_laid_out_over_lanes_does_not_fire(self) -> None:
        # A real spanning heading: the same words, gaps wider than the bound. Each word is
        # its own run, so the row is lane-shaped.
        wide = (ALIGNED_RUN_GAP_MAX_WORD_SPACES + 1) * WORD_SPACE * 3
        words = _grid_words() + _caption_words(CAPTION, wide)
        assert _fired(words, _gate_md(SPREAD)) == set()

    def test_a_gap_just_over_the_bound_does_not_fire_and_at_the_bound_does(self) -> None:
        bound = ALIGNED_RUN_GAP_MAX_WORD_SPACES * WORD_SPACE
        at = _grid_words() + _caption_words(CAPTION, bound)
        over = _grid_words() + _caption_words(CAPTION, bound + 0.5)
        assert _fired(at, _gate_md(SPREAD)) == {PIH}
        assert _fired(over, _gate_md(SPREAD)) == set()

    def test_a_two_word_row_is_enough(self) -> None:
        words = _grid_words() + _caption_words(["Percent", "of"], WORD_SPACE)
        assert _fired(words, _gate_md(["Percent of", "", "", "", ""])) == {PIH}

    def test_a_one_word_header_row_does_not_fire(self) -> None:
        words = _grid_words() + _caption_words(["Percent"], WORD_SPACE)
        assert _fired(words, _gate_md(["Percent", "", "", "", ""])) == set()


class TestMustNotFire:
    def test_a_clean_grid_is_quiet(self) -> None:
        assert _fired(_grid_words(), _gate_md(None)) == set()

    def test_a_caption_the_grid_dropped_is_not_this_predicate(self) -> None:
        # The source has the caption run, the grid does not carry it: nothing was absorbed.
        words = _grid_words() + _caption_words(CAPTION, WORD_SPACE)
        assert PIH not in _fired(words, _gate_md(None))

    def test_a_caption_only_partly_carried_does_not_fire(self) -> None:
        # "every word carried": the grid keeps three of the four caption words.
        words = _grid_words() + _caption_words(CAPTION, WORD_SPACE)
        assert PIH not in _fired(words, _gate_md(["Percent of", "aggregate", "", "", ""]))

    def test_a_word_printed_twice_must_be_carried_twice(self) -> None:
        words = _grid_words() + _caption_words(["Total", "Total"], WORD_SPACE)
        assert PIH not in _fired(words, _gate_md(["Total", "", "", "", ""]))
        assert _fired(words, _gate_md(["Total Total", "", "", "", ""])) == {PIH}

    def test_a_run_below_the_first_data_row_is_not_a_header_row(self) -> None:
        # The words are in the grid's header, but the source row sits under the data: the
        # source row, not the words, decides that it is not above the first data row.
        tail = _line(CAPTION, COL_XS[0], Y0 + 9 * PITCH, 5, WORD_SPACE)
        assert PIH not in _fired(_grid_words() + tail, _gate_md(SPREAD))

    def test_a_page_with_no_measurable_word_space_abstains(self) -> None:
        # Single-word lines only: no gap to measure, so the yardstick is undefined.
        words = [w for w in _grid_words() if w[5] != 9]
        words += [_w(COL_XS[0], Y0 - PITCH, "Percent", 5, 0, 0)]
        words += [_w(COL_XS[1], Y0 - PITCH, "of", 5, 1, 0)]
        assert PIH not in _fired(words, _gate_md(SPREAD))
