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
#: The page's word space, printed by the body prose outside the table. Every grid cell sits on its
#: own text line, so these (and the caption) are the only same-line gaps on the page.
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


def _grid_words(*, prose_lines: int = 2, whole_row_lines: bool = False) -> list[tuple]:
    """HEADER at ``Y0`` and ROWS below it, plus *prose_lines* body sentences far below the table.

    ``whole_row_lines=False``: one text line per cell. ``True``: one text line per table row, as a
    PDF that sets whole rows prints them (the gaps are then the column gutters).
    """
    out, line = [], 0
    for ri, row in enumerate([HEADER] + ROWS):
        for ci, cell in enumerate(row):
            if whole_row_lines:
                out.append(_w(COL_XS[ci], Y0 + ri * PITCH, cell, 0, ri, ci))
            else:
                out.append(_w(COL_XS[ci], Y0 + ri * PITCH, cell, 0, line, 0))
            line += 1
    body = "Prose elsewhere on the page sets the word space".split()
    for k in range(prose_lines):
        out += _line(body, COL_XS[0], Y0 + (20 + k) * PITCH, 9 + k, WORD_SPACE)
    return out


def _caption_words(words: list[str], gap: float) -> list[tuple]:
    return _line(words, COL_XS[0], Y0 - PITCH, 5, gap)


def _gate_md(caption_cells: list[str] | list[list[str]] | None) -> str:
    """Markdown whose header rows are the caption row(s) (when given) then HEADER."""
    rows = [list(HEADER)] + [list(r) for r in ROWS]
    if caption_cells is not None:
        extra = caption_cells if isinstance(caption_cells[0], list) else [caption_cells]
        rows = [list(r) for r in extra] + rows
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
        words = _grid_words(prose_lines=0)
        words += [_w(COL_XS[0], Y0 - PITCH, "Percent", 5, 0, 0)]
        words += [_w(COL_XS[1], Y0 - PITCH, "of", 5, 1, 0)]
        assert PIH not in _fired(words, _gate_md(SPREAD))


class TestSpacingEvidence:
    """The word space is measured only on text outside the table's extent (GH-936 round 2)."""

    def test_table_only_page_with_a_caption_run_abstains(self) -> None:
        # Per-cell lines: the caption's own gaps are INSIDE the extent, so they are not evidence.
        words = _grid_words(prose_lines=0) + _caption_words(CAPTION, WORD_SPACE)
        assert _fired(words, _gate_md(SPREAD)) == set()
        # The same page with prose elsewhere fires: the evidence is the only difference.
        words = _grid_words(prose_lines=2) + _caption_words(CAPTION, WORD_SPACE)
        assert _fired(words, _gate_md(SPREAD)) == {PIH}

    def test_whole_row_lines_make_a_real_header_quiet(self) -> None:
        # Whole-row PDF lines: the column pitch is the only same-line gap. The real header row
        # (one run at that pitch) must not read as prose, with or without a caption above it.
        words = _grid_words(prose_lines=0, whole_row_lines=True)
        assert _fired(words, _gate_md(None)) == set()
        words += _caption_words(CAPTION, WORD_SPACE)
        assert _fired(words, _gate_md(SPREAD)) == set()

    def test_whole_row_lines_with_prose_outside_still_judge_by_the_prose(self) -> None:
        words = _grid_words(prose_lines=2, whole_row_lines=True)
        assert _fired(words, _gate_md(None)) == set(), "the real header is lane-shaped"
        words += _caption_words(CAPTION, WORD_SPACE)
        assert _fired(words, _gate_md(SPREAD)) == {PIH}

    def test_one_prose_line_is_not_enough_evidence(self) -> None:
        words = _grid_words(prose_lines=ship_gate._MIN_SPACING_LINES - 1)
        words += _caption_words(CAPTION, WORD_SPACE)
        assert PIH not in _fired(words, _gate_md(SPREAD))
        words = _grid_words(prose_lines=ship_gate._MIN_SPACING_LINES)
        words += _caption_words(CAPTION, WORD_SPACE)
        assert PIH in _fired(words, _gate_md(SPREAD))

    def test_words_without_line_indices_abstain_and_never_raise(self) -> None:
        # A five-field word has no block/line: it cannot give a gap. The predicate must not turn
        # the page into a gate_error and hide the other faults.
        words = [w[:5] for w in _grid_words() + _caption_words(CAPTION, WORD_SPACE)]
        fired = _fired(words, _gate_md(SPREAD))
        assert ship_gate.GATE_ERROR not in fired and PIH not in fired


class TestSpacingIsNotCalibratedByHeaderRows:
    """A line the predicate scans as a header candidate must not supply the yardstick that judges it.

    The zone is geometric: the header reach above the first core row down to the last core row plus
    the outward reach. Text above the reach is never a candidate, so it is independent evidence.
    """

    WIDE = 40.0  # a column-wide gap, far over ALIGNED_RUN_GAP_MAX_WORD_SPACES x any word space here

    def _page(self, *, prose_lines: int, uncarried_tail: bool = False):
        # Two header lines ABOVE the excluded extent plus the near header row, all with the same
        # wide gap g: unless carried lines are excluded, the median is g and g <= 2g passes.
        far = [
            _line(
                ["Alpha", "Beta"] + (["Zed"] if uncarried_tail else []),
                COL_XS[0],
                Y0 - (2 + k) * PITCH,
                20 + k,
                self.WIDE,
            )
            for k in range(2)
        ]
        near = _caption_words(["Gamma", "Delta"], self.WIDE)
        words = _grid_words(prose_lines=prose_lines) + sum(far, []) + near
        md = _gate_md([["Alpha", "Beta", "", "", ""]] * 2 + [["Gamma", "Delta", "", "", ""]])
        return words, md

    def test_two_wide_gap_header_lines_inside_the_reach_abstain(self) -> None:
        words, md = self._page(prose_lines=0)
        assert _fired(words, md) == set()

    def test_the_same_layout_with_independent_prose_uses_the_prose_spacing(self) -> None:
        words, md = self._page(prose_lines=2)
        # Prose spacing is WORD_SPACE; a gap of WIDE is two or more runs, so the row is lane-shaped.
        assert _fired(words, md) == set()
        # ... and with the near row at the prose spacing the same page DOES fire: the prose is used.
        near_tight = _caption_words(["Gamma", "Delta"], WORD_SPACE)
        tight = [w for w in words if w not in _caption_words(["Gamma", "Delta"], self.WIDE)]
        assert _fired(tight + near_tight, md) == {PIH}

    def test_uncarried_lines_inside_the_extent_are_not_evidence_either(self) -> None:
        # Two note lines under the data, inside the extent and NOT in the grid, with a tight word
        # space. The extent rule alone (not the carried rule) must keep them out of the yardstick.
        notes = [
            w
            for k in range(2)
            for w in _line(
                ["Source", "and", "notes", "text"],
                COL_XS[0],
                Y0 + (8 + k) * PITCH,
                30 + k,
                WORD_SPACE,
            )
        ]
        words = _grid_words(prose_lines=0) + notes + _caption_words(CAPTION, WORD_SPACE)
        assert _fired(words, _gate_md(SPREAD)) == set()

    def test_a_far_header_line_with_one_uncarried_word_still_cannot_calibrate(self) -> None:
        # Astra's bypass: one extra word the grid does not carry, at the same wide gap, on each far
        # line. A token rule is escaped; the geometric zone is not.
        words, md = self._page(prose_lines=0, uncarried_tail=True)
        assert _fired(words, md) == set()

    def test_the_bypass_page_with_prose_below_the_table_uses_the_prose(self) -> None:
        words, md = self._page(prose_lines=2, uncarried_tail=True)
        # (the uncarried "Zed" makes the far lines a header_band_missing matter, not this predicate's)
        assert PIH not in _fired(words, md), "WIDE is lane-shaped at the prose spacing"
        tight = [w for w in words if w not in _caption_words(["Gamma", "Delta"], self.WIDE)]
        tight += _caption_words(["Gamma", "Delta"], WORD_SPACE)
        assert PIH in _fired(tight, md)

    ABOVE = 5.0  # above the reach of the first core row (Y0 + PITCH less _PANEL_GAP_ROWS pitches)

    def _prose_above(self) -> list[tuple]:
        return [
            w
            for k in range(2)
            for w in _line(
                ["Intro", "prose", "text"], COL_XS[0], self.ABOVE + k * PITCH, 40 + k, WORD_SPACE
            )
        ]

    def test_prose_above_the_reach_is_evidence_and_a_caption_run_fires(self) -> None:
        words = _grid_words(prose_lines=0) + self._prose_above()
        words += _caption_words(CAPTION, WORD_SPACE)
        assert _fired(words, _gate_md(SPREAD)) == {PIH}

    def test_a_line_above_the_reach_is_never_a_candidate(self) -> None:
        # The carried tight run sits above the reach: it is not scanned, however prose-like.
        words = _grid_words(prose_lines=2) + _line(CAPTION, COL_XS[0], self.ABOVE, 50, WORD_SPACE)
        assert _fired(words, _gate_md(SPREAD)) == set()
        # The same run inside the reach is a candidate and fires: the position is the difference.
        words = _grid_words(prose_lines=2) + _caption_words(CAPTION, WORD_SPACE)
        assert _fired(words, _gate_md(SPREAD)) == {PIH}

    def test_zone_membership_uses_the_rows_rounded_y(self) -> None:
        # The first core row sits at Y0 + PITCH and the reach is 5 pitches, so the zone starts at
        # 44. Two tight lines at y 43.6 round to 44: they ARE one candidate row, so they must not
        # also calibrate the yardstick (raw 43.6 would put them outside the zone).
        y = 43.6
        a = _line(["Percent", "of"], COL_XS[0], y, 60, WORD_SPACE)
        b = _line(["aggregate", "values"], a[-1][2] + WORD_SPACE, y, 61, WORD_SPACE)
        words = _grid_words(prose_lines=0) + a + b
        md = _gate_md(["Percent of aggregate values", "", "", "", ""])
        assert _fired(words, md) == set()
