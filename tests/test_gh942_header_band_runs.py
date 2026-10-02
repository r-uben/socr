"""GH-942: ``header_band_missing`` also fires on a header row that splits into runs.

Main's clause needs each header word on its own numeric lane. A multi-word heading puts two
words over one lane, and a right-aligned heading has its x-centre off the lane, so both
shipped a grid that had lost its column labels. The extension (OR-ed with main's clause)
fires when the row's in-extent words split into at least ``_MIN_CORE_LANES`` runs (GH-945: it
was ``_MIN_LANES_PER_ROW``, 3, in GH-942) at gaps
wider than ``ALIGNED_RUN_GAP_MAX_WORD_SPACES`` x the page's median word gap.

Each test pins a DIFFERENCE on the same synthetic page (grid drops the band / grid keeps it).
Words are tuples in the shape of ``page.get_text("words")``; nothing reads a corpus file and
nothing needs a provider. The page's word-space yardstick comes from a prose paragraph far
above the table (outside the reach); without it the table's own column gaps would be the
median and no gap would count as a run gap, which is exactly the fixture of the GH-917 tests.
"""

from __future__ import annotations

import pytest
from native_table_fixtures import CHAR_W, COL_XS, PITCH, ROWS, Y0
from test_gh916_native_ship_gate import _md, _predicates, _word, _words
from test_gh917_gate_direction_header import _dirs

from socr.core.born_digital import ALIGNED_RUN_GAP_MAX_WORD_SPACES
from socr.tables import ship_gate
from socr.tables.native_first import DEFER, SHIP, plan_native_table

FIRST = Y0 + PITCH  # first data row
BAND_Y = FIRST - 2 * PITCH
WORD_SPACE = 5.0  # the prose paragraph's gap between words, the page's median word gap
GRID_HEAD = ["Country", "", "", "", ""]  # a grid that kept only the stub


def _prose() -> list:
    """Three prose lines of 4-letter words, far above the table: the word-space yardstick."""
    out = []
    for line in range(3):
        for i in range(40):
            out.append(
                _word(
                    90.0 + i * (CHAR_W * 4 + WORD_SPACE),
                    -2000.0 - line * PITCH,
                    "word",
                    200 + line,
                    i,
                )
            )
    return out


def _multi_word(headings: int) -> list:
    """*headings* two-word headings, one per numeric lane, the second word a word space on."""
    out = []
    for i in range(headings):
        x = COL_XS[1 + i]
        first = _word(x, BAND_Y, "Net", 300 + i, 0)
        out.append(first)
        out.append(_word(first[2] + WORD_SPACE, BAND_Y, "Sales", 300 + i, 1))
    return out


def _right_aligned(headings: int) -> list:
    """*headings* short headings right-aligned to the numbers' right edge.

    A short word ending where a wider number ends has its x-centre more than a lane snap
    right of the lane (the numbers' left edge), so it snaps to no lane.
    """
    out = []
    for i in range(headings):
        text = RIGHT_LABELS[i]
        right = COL_XS[1 + i] + CHAR_W * 5
        out.append(_word(right - CHAR_W * len(text), BAND_Y, text, 300 + i, 0))
    return out


RIGHT_LABELS = ["N", "R2", "F"]
SHAPES = {"multi_word": _multi_word, "right_aligned": _right_aligned}


def _page(shape: str, headings: int, *, prose: bool = True) -> list:
    words = _words(ROWS, y_start=FIRST)
    words += SHAPES[shape](headings)
    return words + (_prose() if prose else [])


def _plan(words, head=GRID_HEAD, rows=ROWS):
    return plan_native_table(words, _md(head, rows), line_dirs=_dirs(words))


def _lane_clause_only(words) -> bool:
    """Whether main's lane clause alone would fire: the row's region words on distinct lanes."""
    blocks = ship_gate._output_blocks(_md(GRID_HEAD, ROWS))
    src = ship_gate._source_rows(words)
    pairs = ship_gate._unique_pairs(blocks, src)
    lanes, _core = ship_gate._block_geometries(pairs, src)[0]
    row = src[round(BAND_Y)]
    region = [w for w in row if w[0] >= lanes[0] - ship_gate._SNAP_PT]
    hits = [ship_gate._lane_of((w[0] + w[2]) / 2, lanes) for w in region]
    return (
        len(region) >= ship_gate._MIN_LANES_PER_ROW
        and None not in hits
        and len(set(hits)) == len(region)
    )


class TestRunClause:
    @pytest.mark.parametrize("shape", sorted(SHAPES))
    def test_main_clause_misses_this_header_shape(self, shape: str) -> None:
        # The fixture is only evidence if main's own clause is silent on it.
        assert not _lane_clause_only(_page(shape, 3))

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    def test_difference_pin_dropped_header_defers_and_kept_header_ships(self, shape: str) -> None:
        words = _page(shape, 3)
        dropped = _plan(words)
        assert dropped.action == DEFER
        assert _predicates(dropped) == {ship_gate.HEADER_BAND_MISSING}
        # the same page with the header words carried into the grid: quiet
        head = {
            "multi_word": ["Country", "Net Sales", "Net Sales", "Net Sales", ""],
            "right_aligned": ["Country", *RIGHT_LABELS, ""],
        }[shape]
        kept = _plan(words, head=head)
        assert kept.action == SHIP and kept.faults == ()

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    def test_the_runs_floor_is_min_core_lanes(self, shape: str) -> None:
        # GH-945: the floor was _MIN_LANES_PER_ROW (3), which left a two-run header (Gurkaynak
        # p46: "Monetary Policy Surprise" / "Differences") shipping with its labels lost.
        assert ship_gate._MIN_CORE_LANES == 2
        assert ship_gate._MIN_LANES_PER_ROW == 3
        one = _plan(_page(shape, 1))
        two = _plan(_page(shape, 2))
        assert one.action == SHIP and one.faults == ()
        assert two.action == DEFER
        assert _predicates(two) == {ship_gate.HEADER_BAND_MISSING}

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    def test_difference_pin_two_run_header_fired_only_below_the_old_floor(self, shape: str) -> None:
        # The old floor (3) is quiet on this exact page, the new one (2) fires: only the
        # floor differs. The lane clause is silent on it as well, so the run clause decides.
        words = _page(shape, 2)
        row = sorted((w for w in words if round(w[1]) == round(BAND_Y)), key=lambda w: w[0])
        unit = ship_gate._median_word_gap(words)
        runs = ship_gate._run_count(row, unit)
        assert runs == 2
        assert runs < ship_gate._MIN_LANES_PER_ROW  # main: quiet
        assert runs >= ship_gate._MIN_CORE_LANES  # branch: fires
        assert not _lane_clause_only(words)
        assert _plan(words).action == DEFER
        # and the same page with the header carried into the grid is quiet
        head = {
            "multi_word": ["Country", "Net Sales", "Net Sales", "", ""],
            "right_aligned": ["Country", *RIGHT_LABELS[:2], "", ""],
        }[shape]
        assert _plan(words, head=head).action == SHIP

    @pytest.mark.parametrize(
        ("inner_gap", "action"),
        [(0.5, SHIP), (WORD_SPACE, SHIP), (ALIGNED_RUN_GAP_MAX_WORD_SPACES * WORD_SPACE, SHIP)]
        + [(ALIGNED_RUN_GAP_MAX_WORD_SPACES * WORD_SPACE + 0.5, DEFER)],
    )
    def test_run_gap_bound_is_exclusive_at_k_median_word_gaps(
        self, inner_gap: float, action: str
    ) -> None:
        # One two-word heading: one run while the inner gap is within K x median word gap
        # (ships), two runs once it is wider (2 runs, the floor: fires).
        words = _words(ROWS, y_start=FIRST) + _prose()
        for i in range(1):
            first = _word(COL_XS[1 + i], BAND_Y, "Net", 300 + i, 0)
            words += [first, _word(first[2] + inner_gap, BAND_Y, "Sales", 300 + i, 1)]
        assert ship_gate._median_word_gap(words) == WORD_SPACE
        assert _plan(words).action == action

    def test_run_count_splits_only_strictly_above_the_bound(self) -> None:
        row = [_word(0.0, 0.0, "ab", 0, 0), _word(10.0 + CHAR_W * 2, 0.0, "cd", 0, 1)]
        bound = ALIGNED_RUN_GAP_MAX_WORD_SPACES * WORD_SPACE
        assert row[1][0] - row[0][2] == bound
        assert ship_gate._run_count(row, WORD_SPACE) == 1
        assert ship_gate._run_count(row, WORD_SPACE - 0.01) == 2


class TestControls:
    def test_a_header_present_in_the_grid_stays_quiet(self) -> None:
        words = _page("multi_word", 3)
        head = ["Country", "Net Sales", "Net Sales", "Net Sales", ""]
        plan = _plan(words, head=head)
        assert plan.action == SHIP and plan.faults == ()

    def test_a_caption_left_out_is_one_run_and_stays_quiet(self) -> None:
        words = _words(ROWS, y_start=FIRST) + _prose()
        x = COL_XS[0]
        for i, text in enumerate(["Table", "Forecast", "errors", "by", "horizon"]):
            w = _word(x, BAND_Y, text, 310, i)
            words.append(w)
            x = w[2] + WORD_SPACE
        plan = _plan(words)
        assert plan.action == SHIP and plan.faults == ()

    def test_a_row_carrying_a_number_stays_quiet(self) -> None:
        words = _page("multi_word", 3)
        words.append(_word(COL_XS[0], BAND_Y, "2004", 310, 0))
        assert _plan(words).action == SHIP

    def test_main_clause_is_still_required(self) -> None:
        # No prose: the median word gap is the table's own column gap, so no row splits into
        # runs. The GH-917 shape (one word per lane) must still fire through the lane clause.
        words = _words(ROWS, y_start=FIRST)
        for i, text in enumerate(["Country", "Alpha", "Gamma", "Delta", "Omega"]):
            words.append(_word(COL_XS[i], BAND_Y, text, 320, i))
        assert _lane_clause_only(words)
        row = sorted((w for w in words if round(w[1]) == round(BAND_Y)), key=lambda w: w[0])
        unit = ship_gate._median_word_gap(words)
        assert ship_gate._run_count(row, unit) < ship_gate._MIN_CORE_LANES
        assert _predicates(_plan(words)) == {ship_gate.HEADER_BAND_MISSING}

    def test_absence_is_per_block_not_per_page(self) -> None:
        # Two panels; the header is dropped from panel A but present in panel B. A page-wide
        # absence test would find "Net"/"Sales" in B's cells and stay silent on A.
        gap = 8 * PITCH
        a_first = FIRST
        b_first = FIRST + len(ROWS) * PITCH + gap
        rows_b = [[c if j == 0 else f"{float(c) + 1:.3f}" for j, c in enumerate(r)] for r in ROWS]
        rows_b = [[f"{r[0]}b", *r[1:]] for r in rows_b]
        words = _words(ROWS, y_start=a_first)
        words += _words(rows_b, y_start=b_first, start_line=50)
        words += _multi_word(3)
        for i in range(3):
            x = COL_XS[1 + i]
            first = _word(x, b_first - 2 * PITCH, "Net", 400 + i, 0)
            words += [first, _word(first[2] + WORD_SPACE, b_first - 2 * PITCH, "Sales", 400 + i, 1)]
        words += _prose()
        head_b = ["Country", "Net Sales", "Net Sales", "Net Sales", ""]
        md = _md(GRID_HEAD, ROWS) + "\n\n" + _md(head_b, rows_b)
        faults = ship_gate.native_ship_gate(
            words, md, line_dirs=ship_gate.LineDirections.unchecked_for_tests()
        )
        hb = [f for f in faults if f["predicate"] == ship_gate.HEADER_BAND_MISSING]
        assert len(hb) == 1
        assert f"y={round(BAND_Y)}" in hb[0]["detail"]


class TestAcceptedFalseDefer:
    def test_known_accepted_false_defer_two_run_caption_costs_one_model_read(self) -> None:
        # GH-945 ACCEPTED COST: a numeric-free, two-run caption ("Panel A:" ... "Returns", wide
        # gap) left out of the grid is indistinguishable from a two-run header, so the lowered
        # floor DEFERs a page that would otherwise SHIP. The price is one model read. Pinned as a
        # difference: quiet at the old floor (3), fires at the new one (2).
        words = _words(ROWS, y_start=FIRST) + _prose()
        panel = _word(COL_XS[0], BAND_Y, "Panel", 330, 0)
        a = _word(panel[2] + WORD_SPACE, BAND_Y, "A:", 330, 1)
        words += [panel, a, _word(COL_XS[2], BAND_Y, "Returns", 330, 2)]
        row = sorted((w for w in words if round(w[1]) == round(BAND_Y)), key=lambda w: w[0])
        runs = ship_gate._run_count(row, ship_gate._median_word_gap(words))
        assert runs == 2
        assert runs < ship_gate._MIN_LANES_PER_ROW  # old floor: quiet
        assert not _lane_clause_only(words)
        plan = _plan(words)
        assert plan.action == DEFER  # new floor: fires
        assert _predicates(plan) == {ship_gate.HEADER_BAND_MISSING}
