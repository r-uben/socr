"""GH-916: the ship gate in front of native-first SHIP.

``plan_native_table`` exact-passes a grid whose numbers are all present, which
says nothing about a detached sign, a dropped row, or reversed order. Each test
pins a DIFFERENCE: the same words, one markdown with the fault and one without,
and (for the fault) the same markdown with the gate switched off, so the
exact-pass the gate overrides is proven to exist. Words are synthetic tuples in
the shape of ``page.get_text("words")``; no corpus file is read.

Every predicate DEFERs; none REFUSEs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables import native_first as nf
from socr.tables import ship_gate
from socr.tables.native_first import DEFER, SHIP, plan_native_table
from socr.tables.reconstruct import detached_sign_pairs

COL_XS = [90.0, 180.0, 270.0, 360.0, 450.0]
CHAR_W = 5.0
PITCH = 14.0
Y0 = 100.0

HEADER = ["Variable", "b", "s", "h", "q"]
ROWS = [
    ["GDP", "0.253", "0.179", "0.211", "0.301"],
    ["CPI", "0.144", "0.135", "0.290", "0.188"],
    ["IP", "0.041", "0.050", "0.154", "0.099"],
    ["UR", "0.082", "0.321", "0.144", "0.211"],
    ["CB", "0.180", "0.171", "0.365", "0.244"],
    ["TR", "0.310", "0.220", "0.410", "0.188"],
]


def _word(x: float, y: float, text: str, line: int, no: int) -> tuple:
    return (x, y, x + CHAR_W * len(text), y + 9.0, text, 0, line, no)


def _words(rows, *, start_line: int = 0, y_start: float = Y0) -> list:
    """One PyMuPDF-style word per cell; one text line per row."""
    out = []
    for ri, row in enumerate(rows):
        y = y_start + ri * PITCH
        for ci, cell in enumerate(row):
            if cell:
                out.append(_word(COL_XS[ci], y, cell, start_line + ri, ci))
    return out


def _md(header, rows) -> str:
    def line(cells):
        return "| " + " | ".join(cells) + " |"

    return "\n".join(
        [line(header), "| " + " | ".join(["---"] * len(header)) + " |"] + [line(r) for r in rows]
    )


def _plan(words, markdown, *, gate: bool = True):
    if gate:
        return plan_native_table(words, markdown)
    with patch.object(nf, "native_ship_gate", return_value=()):
        return plan_native_table(words, markdown)


def _predicates(plan) -> set[str]:
    return {f["predicate"] for f in plan.faults}


def _base():
    return _words([HEADER] + ROWS), _md(HEADER, ROWS)


def test_clean_grid_ships_and_gate_is_inert() -> None:
    words, md = _base()
    plan = _plan(words, md)
    assert plan.action == SHIP and plan.faults == ()
    assert _plan(words, md, gate=False).action == SHIP


# ---------------------------------------------------------------- P2 sign


def _sign_rows(*, sign_dx: float, num: str = "0.230"):
    """Row NEG carries a sign word ``sign_dx`` pt left of its second number."""
    rows = [list(r) for r in ROWS]
    rows[2][1] = num
    words = _words([HEADER] + rows)
    num_x = COL_XS[1]
    ri = 3  # header is line 0
    y = Y0 + ri * PITCH
    sign_x = num_x + 0.2 - CHAR_W * 0.7 - sign_dx
    sign = (sign_x, y, sign_x + CHAR_W * 0.7, y + 9.0, "-", 0, ri, 9)
    words.append(sign)
    return rows, words


def _sign_md(rows, *, in_label: bool):
    rows = [list(r) for r in rows]
    if in_label:
        rows[2][0] = rows[2][0] + " -"
        return _md(HEADER, rows)
    row = rows[2]
    row = row[:1] + ["-"] + row[1:]
    md_rows = [list(r) for r in rows]
    md_rows[2] = row
    return "\n".join(
        [
            _md(HEADER, rows).splitlines()[0],
            "| " + " | ".join(["---"] * 6) + " |",
        ]
        + ["| " + " | ".join(r) + " |" for r in md_rows]
    )


class TestSignDetached:
    def test_difference_pin_bare_sign_cell_with_source_contact(self) -> None:
        rows, words = _sign_rows(sign_dx=0.0)
        assert detached_sign_pairs(words), "fixture must put the sign in contact"
        clean = _plan(words, _md(HEADER, rows))
        faulty_md = _sign_md(rows, in_label=False)
        faulty = _plan(words, faulty_md)
        off = _plan(words, faulty_md, gate=False)
        assert off.action == SHIP, "the exact-pass the gate overrides must exist"
        assert faulty.action == DEFER and _predicates(faulty) == {ship_gate.SIGN_DETACHED}
        assert faulty.reason.startswith(ship_gate.SHIP_GATE_REASON_PREFIX)
        # Fault absent from the markdown: nothing fires (the unsigned grid is
        # a different failure, a lost sign, that EXACT_PASS also cannot see;
        # P2 is the detached-cell shape only).
        assert clean.action == SHIP

    def test_sign_at_the_end_of_a_populated_cell(self) -> None:
        rows, words = _sign_rows(sign_dx=0.0)
        md = _sign_md(rows, in_label=True)
        assert _plan(words, md, gate=False).action == SHIP
        plan = _plan(words, md)
        assert plan.action == DEFER and _predicates(plan) == {ship_gate.SIGN_DETACHED}

    def test_placeholder_dash_in_its_own_column_does_not_fire(self) -> None:
        # The sign sits a column gap away from the number: a placeholder.
        rows, words = _sign_rows(sign_dx=40.0)
        assert detached_sign_pairs(words) == []
        plan = _plan(words, _sign_md(rows, in_label=False))
        assert plan.action == SHIP and plan.faults == ()

    def test_range_hyphen_flush_on_both_sides_does_not_fire(self) -> None:
        rows = [list(r) for r in ROWS]
        rows[2][1] = "1990"
        rows[2][2] = "2000"
        words = _words([HEADER] + rows)
        # "1990" ends where the hyphen starts and the hyphen ends where "2000" starts.
        left = next(w for w in words if w[4] == "1990")
        right = next(w for w in words if w[4] == "2000")
        hyphen = (left[2], left[1], right[0] + 0.1, left[3], "-", 0, left[6], 9)
        words.append(hyphen)
        assert detached_sign_pairs(words) == []
        md_rows = [list(r) for r in rows]
        md_rows[2] = [rows[2][0], "1990", "-", "2000", rows[2][3], rows[2][4]]
        md = "\n".join(
            ["| " + " | ".join(HEADER) + " |", "| " + " | ".join(["---"] * 5) + " |"]
            + ["| " + " | ".join(r) + " |" for r in md_rows]
        )
        plan = _plan(words, md)
        assert plan.action == SHIP and plan.faults == ()

    def test_contact_is_bound_to_the_output_row_when_values_repeat(self) -> None:
        # The same number prints twice: signed in row 2, unsigned in row 4.
        rows, words = _sign_rows(sign_dx=0.0)
        rows[4][2] = "0.230"
        words = [w for w in words if not (w[4] == "0.171" and abs(w[1] - (Y0 + 5 * PITCH)) < 1)]
        words = _words([HEADER] + rows) + [w for w in words if w[4] == "-"]
        md_rows = [list(r) for r in rows]
        # The bare sign is shipped on the row whose source number is UNSIGNED.
        md_rows[4] = md_rows[4][:2] + ["-"] + md_rows[4][2:]
        md = "\n".join(
            [_md(HEADER, rows).splitlines()[0], "| " + " | ".join(["---"] * 6) + " |"]
            + ["| " + " | ".join(r) + " |" for r in md_rows]
        )
        plan = _plan(words, md)
        # Row 4's own multiset differs from row 2's, so the contact on row 2's
        # line is not evidence for row 4: the gate abstains on this cell.
        assert ship_gate.SIGN_DETACHED not in _predicates(plan)
        # Same words and the same repeated value, bare sign on the signed row:
        # now the contact belongs to that output occurrence and the gate fires.
        md_rows = [list(r) for r in rows]
        md_rows[2] = md_rows[2][:1] + ["-"] + md_rows[2][1:]
        md = "\n".join(
            [_md(HEADER, rows).splitlines()[0], "| " + " | ".join(["---"] * 6) + " |"]
            + ["| " + " | ".join(r) + " |" for r in md_rows]
        )
        assert ship_gate.SIGN_DETACHED in {
            f["predicate"] for f in ship_gate.native_ship_gate(words, md)
        }


# ---------------------------------------------------------------- P1 order


class TestOrder:
    def test_reversed_rows_defer_and_correct_order_ships(self) -> None:
        words, _ = _base()
        reversed_md = _md(HEADER, list(reversed(ROWS)))
        assert _plan(words, reversed_md, gate=False).action == SHIP
        plan = _plan(words, reversed_md)
        assert plan.action == DEFER and ship_gate.ROW_ORDER in _predicates(plan)
        assert _plan(words, _md(HEADER, ROWS)).action == SHIP

    def test_reversed_cells_defer(self) -> None:
        words, _ = _base()
        rows = [r[:1] + list(reversed(r[1:])) for r in ROWS]
        md = _md(HEADER, rows)
        assert _plan(words, md, gate=False).action == SHIP
        plan = _plan(words, md)
        assert plan.action == DEFER and ship_gate.CELL_ORDER in _predicates(plan)

    def test_ambiguous_identical_rows_abstain(self) -> None:
        rows = [list(ROWS[0]), list(ROWS[1]), list(ROWS[0]), list(ROWS[2]), list(ROWS[3])]
        words = _words([HEADER] + rows)
        plan = _plan(words, _md(HEADER, rows))
        assert not _predicates(plan) & {ship_gate.ROW_ORDER, ship_gate.CELL_ORDER}


# ----------------------------------------------------------- P5a data rows


class TestDataRowMissing:
    def test_dropped_data_row_defers_and_full_grid_ships(self) -> None:
        words, _ = _base()
        for drop in (-1, 0):
            rows = [r for i, r in enumerate(ROWS) if i != drop % len(ROWS)]
            md = _md(HEADER, rows)
            assert _plan(words, md, gate=False).action == SHIP
            plan = _plan(words, md)
            assert plan.action == DEFER and _predicates(plan) == {ship_gate.DATA_ROW_MISSING}
        assert _plan(words, _md(HEADER, ROWS)).action == SHIP

    def test_prose_numbers_beside_a_table_do_not_fire(self) -> None:
        words, md = _base()
        # Two-column page: right-column prose, numbers off every table lane,
        # on baselines between the table rows.
        for i in range(4):
            y = Y0 + 7.0 + i * PITCH
            words.append(_word(520.0, y, "growth", 20 + i, 0))
            words.append(_word(560.0, y, f"{3 + i}.5", 20 + i, 1))
            words.append(_word(590.0, y, f"{4 + i}.2", 20 + i, 2))
        # Straight at the gate: the verifier's own row count is not the subject.
        assert ship_gate.native_ship_gate(words, md) == ()

    def test_two_lane_prose_numbers_outside_the_table_do_not_fire(self) -> None:
        words, md = _base()
        # Prose below the table with two numbers that happen to fall in table
        # lanes: fewer than the table's own width and outside its rows.
        y = Y0 + (len(ROWS) + 3) * PITCH
        words.append(_word(COL_XS[1] + 1, y, "3.5", 40, 0))
        words.append(_word(COL_XS[2] + 1, y, "4.2", 40, 1))
        assert _plan(words, md).action == SHIP


# ---------------------------------------------------------- P5c label rows


class TestLabelRowMissing:
    def _with_panel(self):
        rows = [list(r) for r in ROWS]
        rows.insert(3, ["Panel B: spreads", "", "", "", ""])
        return rows

    def test_dropped_panel_label_defers_and_kept_label_ships(self) -> None:
        rows = self._with_panel()
        words = _words([HEADER] + rows)
        kept = _md(HEADER, rows)
        dropped = _md(HEADER, [r for r in rows if not r[0].startswith("Panel")])
        assert _plan(words, kept).action == SHIP
        assert _plan(words, dropped, gate=False).action == SHIP
        plan = _plan(words, dropped)
        assert plan.action == DEFER and _predicates(plan) == {ship_gate.LABEL_ROW_MISSING}

    def test_caption_below_and_other_column_prose_do_not_fire(self) -> None:
        words, md = _base()
        # A note under the last row (outside the first..last data rows).
        y_note = Y0 + (len(ROWS) + 2) * PITCH
        words.append(_word(COL_XS[0], y_note, "Note: standard errors in parentheses", 30, 0))
        # Other-column prose between rows, right of the table.
        words.append(_word(520.0, Y0 + 3 * PITCH + 7.0, "unrelated", 31, 0))
        assert _plan(words, md).action == SHIP


class TestSignDetachedRound2:
    """Occurrence-specific binding and the shapes Astra named."""

    @staticmethod
    def _md_with(rows, row_index, cells):
        md_rows = [list(r) for r in rows]
        md_rows[row_index] = cells
        width = max(len(r) for r in md_rows)
        return "\n".join(
            ["| " + " | ".join(HEADER + [""] * (width - len(HEADER))) + " |"]
            + ["| " + " | ".join(["---"] * width) + " |"]
            + ["| " + " | ".join(r) + " |" for r in md_rows]
        )

    def test_leading_decimal_fires_and_is_in_the_shared_helper(self) -> None:
        rows, words = _sign_rows(sign_dx=0.0, num=".230")
        assert detached_sign_pairs(words), "the shared helper must pair a sign with .230"
        md = self._md_with(rows, 2, [rows[2][0], "-", ".230"] + rows[2][2:])
        faults = ship_gate.native_ship_gate(words, md)
        assert {f["predicate"] for f in faults} == {ship_gate.SIGN_DETACHED}
        # No contact (placeholder a column away): never fires.
        rows, words = _sign_rows(sign_dx=40.0, num=".230")
        assert ship_gate.native_ship_gate(words, md) == ()

    def test_sign_attached_as_a_tail_of_the_label_fires_only_with_contact(self) -> None:
        rows, words = _sign_rows(sign_dx=0.0)
        md = self._md_with(rows, 2, [rows[2][0] + "-"] + rows[2][1:])
        assert {f["predicate"] for f in ship_gate.native_ship_gate(words, md)} == {
            ship_gate.SIGN_DETACHED
        }
        rows, words = _sign_rows(sign_dx=40.0)
        assert ship_gate.native_ship_gate(words, md) == ()

    def test_placeholder_row_with_identical_numbers_abstains(self) -> None:
        # Rows 2 and 4 print the same numbers; only row 2's number carries a
        # sign in contact. The bare sign cell shipped on row 4 is a placeholder
        # as far as the page shows, so the gate must not fire on it.
        rows, words = _sign_rows(sign_dx=0.0)
        rows[4] = [rows[4][0]] + rows[2][1:]
        words = _words([HEADER] + rows) + [w for w in words if w[4] == "-"]
        md = self._md_with(rows, 4, [rows[4][0], "-"] + rows[4][1:])
        assert ship_gate.native_ship_gate(words, md) == ()
        # The sign cell on the row that really carries the sign: the two identical
        # lines still disagree about it, so the gate abstains there too.
        md2 = self._md_with(rows, 2, [rows[2][0], "-"] + rows[2][1:])
        assert ship_gate.SIGN_DETACHED not in {
            f["predicate"] for f in ship_gate.native_ship_gate(words, md2)
        }

    def test_contact_must_sit_on_the_cells_own_column(self) -> None:
        # One row prints 0.230 twice; the sign touches the SECOND. A bare sign
        # shipped before the FIRST is not the printed sign.
        rows = [list(r) for r in ROWS]
        rows[2][1] = "0.230"
        rows[2][3] = "0.230"
        words = _words([HEADER] + rows)
        y = Y0 + 3 * PITCH
        x = COL_XS[3]
        words.append((x + 0.2 - CHAR_W * 0.7, y, x + 0.2, y + 9.0, "-", 0, 3, 9))
        md_first = self._md_with(rows, 2, [rows[2][0], "-"] + rows[2][1:])
        assert ship_gate.native_ship_gate(words, md_first) == ()
        md_second = self._md_with(rows, 2, rows[2][:3] + ["-"] + rows[2][3:])
        assert {f["predicate"] for f in ship_gate.native_ship_gate(words, md_second)} == {
            ship_gate.SIGN_DETACHED
        }


class TestDataRowRound2:
    def test_full_width_numeric_prose_far_above_or_below_does_not_fire(self) -> None:
        words, md = _base()
        far = 8 * PITCH
        for y in (Y0 - far, Y0 + (len(ROWS) + 1) * PITCH + far):
            for ci in range(1, 5):
                words.append(_word(COL_XS[ci], y, f"{ci}.5", 60, ci))
        assert ship_gate.native_ship_gate(words, md) == ()

    def test_prose_between_the_table_and_a_distant_numeric_line_does_not_bridge(self) -> None:
        # A paragraph at the table's own row pitch runs from the last row out to
        # a full-width numeric line: the prose rows are not table rows, so they
        # must not carry the span out to it.
        words, md = _base()
        last = Y0 + len(ROWS) * PITCH
        far = last + 12 * PITCH
        for i in range(1, 12):
            words.append(_word(COL_XS[0], last + i * PITCH, "prose", 80 + i, 0))
        for ci in range(1, 5):
            words.append(_word(COL_XS[ci], far, f"{ci}.5", 90, ci))
        assert ship_gate.native_ship_gate(words, md) == ()

    def test_a_full_width_row_at_the_reach_limit_is_in_and_one_pitch_beyond_is_out(self) -> None:
        last = Y0 + len(ROWS) * PITCH
        limit = 5  # the measured p99 of inter-row gaps, in row pitches (see ship_gate)
        for pitches, fires in ((limit, True), (limit + 1, False)):
            words, md = _base()
            for ci in range(1, 5):
                words.append(_word(COL_XS[ci], last + pitches * PITCH, f"{ci}.5", 70, ci))
            got = {f["predicate"] for f in ship_gate.native_ship_gate(words, md)}
            assert (got == {ship_gate.DATA_ROW_MISSING}) is fires, (pitches, got)

    def test_bound_zero_extends_nothing(self) -> None:
        # The benchmark's baseline: no floor may extend the span at bound 0.
        words, md = _base()
        y = Y0 + (len(ROWS) + 1) * PITCH
        for ci in range(1, 5):
            words.append(_word(COL_XS[ci], y, f"{ci}.5", 70, ci))
        blocks = ship_gate._output_blocks(md)
        src = ship_gate._source_rows(words)
        anchors = ship_gate._Anchors(blocks, src)
        assert ship_gate.data_row_missing_faults(blocks, anchors, src, 0) == []
        assert len(ship_gate.data_row_missing_faults(blocks, anchors, src)) == 1

    def test_width_is_counted_in_lanes_not_numeric_words(self) -> None:
        # Every cell prints two numbers in ONE lane ("0.253 (1)"), so a row has twice
        # as many numeric words as lanes. A dropped last row must still extend the span.
        words = []
        out_rows = []
        for ri, row in enumerate([HEADER] + ROWS):
            y = Y0 + ri * PITCH
            md_cells = [row[0]]
            words.append(_word(COL_XS[0], y, row[0], ri, 0))
            for ci in range(1, 5):
                words.append(_word(COL_XS[ci], y, row[ci], ri, ci))
                if ri:
                    words.append(_word(COL_XS[ci] + 4.0, y, "(1)", ri, 10 + ci))
                md_cells.append(row[ci] + (" (1)" if ri else ""))
            out_rows.append(md_cells)
        md = _md(out_rows[0], out_rows[1:])
        assert ship_gate.native_ship_gate(words, md) == ()
        dropped = _md(out_rows[0], out_rows[1:-1])
        assert {f["predicate"] for f in ship_gate.native_ship_gate(words, dropped)} == {
            ship_gate.DATA_ROW_MISSING
        }

    def test_dropped_copy_of_a_repeated_row_fires(self) -> None:
        rows = [list(r) for r in ROWS]
        rows.insert(3, list(rows[2]))  # an identical row prints twice
        words = _words([HEADER] + rows)
        both = _md(HEADER, rows)
        one = _md(HEADER, rows[:3] + rows[4:])
        assert ship_gate.native_ship_gate(words, both) == ()
        assert {f["predicate"] for f in ship_gate.native_ship_gate(words, one)} == {
            ship_gate.DATA_ROW_MISSING
        }

    def test_a_data_row_equal_to_the_header_numbers_is_not_hidden(self) -> None:
        header = ["Year", "2001", "2002", "2003", "2004"]
        rows = [list(r) for r in ROWS]
        rows.insert(3, ["Rebased", "2001", "2002", "2003", "2004"])
        words = _words([header] + rows)
        kept = _md(header, rows)
        dropped = _md(header, rows[:3] + rows[4:])
        assert ship_gate.native_ship_gate(words, kept) == ()
        assert ship_gate.DATA_ROW_MISSING in {
            f["predicate"] for f in ship_gate.native_ship_gate(words, dropped)
        }


class TestLabelRowRound2:
    @staticmethod
    def _grid(label_rows, out_labels):
        """ROWS with *label_rows* inserted after row 2; the markdown carries *out_labels*."""
        rows = [list(r) for r in ROWS]
        blank = ["", "", "", ""]
        words = _words([HEADER] + rows[:3] + [[t] + blank for t in label_rows] + rows[3:])
        md_rows = rows[:3] + [list(c) + [""] * (5 - len(c)) for c in out_labels] + rows[3:]
        return words, _md(HEADER, md_rows)

    def test_dehyphenated_line_break_is_not_missing(self) -> None:
        words, md = self._grid(["Evalu-", "ation"], [["Evaluation"]])
        assert ship_gate.native_ship_gate(words, md) == ()
        words, md = self._grid(["Evalu-", "ation"], [])
        assert {f["predicate"] for f in ship_gate.native_ship_gate(words, md)} == {
            ship_gate.LABEL_ROW_MISSING
        }

    def test_a_match_cannot_straddle_two_cells(self) -> None:
        # "bc" is in neither "ab" nor "cd"; it only exists in their concatenation.
        words, md = self._grid(["bc"], [["ab", "cd"]])
        assert {f["predicate"] for f in ship_gate.native_ship_gate(words, md)} == {
            ship_gate.LABEL_ROW_MISSING
        }

    def test_a_repeated_label_is_counted(self) -> None:
        words, md_two = self._grid(["Panel B", "Panel B"], [["Panel B"], ["Panel B"]])
        _w, md_one = self._grid(["Panel B", "Panel B"], [["Panel B"]])
        assert ship_gate.native_ship_gate(words, md_two) == ()
        assert {f["predicate"] for f in ship_gate.native_ship_gate(words, md_one)} == {
            ship_gate.LABEL_ROW_MISSING
        }

    def test_a_full_width_heading_inside_the_span_is_a_missing_label(self) -> None:
        # Five words spanning every lane, between two data rows, dropped from the
        # grid: width alone never makes a row inside the table prose.
        rows = [list(r) for r in ROWS]
        heading = ["Averages", "are", "taken", "over", "years"]
        words = _words([HEADER] + rows[:3] + [heading] + rows[3:])
        assert {f["predicate"] for f in ship_gate.native_ship_gate(words, _md(HEADER, rows))} == {
            ship_gate.LABEL_ROW_MISSING
        }
        kept = [list(r) for r in rows[:3]] + [heading] + [list(r) for r in rows[3:]]
        assert ship_gate.native_ship_gate(words, _md(HEADER, kept)) == ()

    def test_a_notes_heading_between_panels_does_not_hide_later_labels(self) -> None:
        # "Notes:" sits between two data blocks and numeric rows resume after it,
        # so the label that vanishes after it is still a fault.
        rows = [list(r) for r in ROWS]
        blank = ["", "", "", ""]
        inserted = [["Notes:"] + blank, ["Panel C"] + blank]
        words = _words([HEADER] + rows[:3] + inserted + rows[3:])
        md_rows = rows[:3] + [["Notes:"] + blank] + rows[3:]
        assert {
            f["predicate"] for f in ship_gate.native_ship_gate(words, _md(HEADER, md_rows))
        } == {ship_gate.LABEL_ROW_MISSING}
        assert ship_gate.native_ship_gate(words, _md(HEADER, rows[:3] + inserted + rows[3:])) == ()

    def test_a_notes_paragraph_with_numerals_is_a_false_defer_we_accept(self) -> None:
        # The gomez-cram p10 / piller p33 shape: a Notes paragraph swallowed into
        # the grid whose continuation lines carry numerals in table lanes, with an
        # unnumbered line dropped between them. The gate FIRES. Pinned on purpose:
        # a false DEFER costs one model call, a missed fault can ship a wrong
        # number, so no Notes/Source rule is allowed to suppress it (GH-916 round 5).
        words, _ = _base()
        y0 = Y0 + (len(ROWS) + 2) * PITCH
        out_rows = [list(r) for r in ROWS]
        words.append(_word(95.0, y0, "Notes:", 100, 0))
        words.append(_word(140.0, y0, "values", 100, 1))
        out_rows.append(["Notes: values", "", "", "", ""])
        filler = ["and", "scenario", "for", "each", "horizon", "shown"]
        for n, (a, b) in enumerate([("7.5", "8.5"), ("9.5", "6.5")]):
            y = y0 + (1 + 2 * n) * PITCH
            for k, tok in enumerate(filler):
                words.append(_word(95.0 + 38.0 * k, y, tok, 101 + n, k))
            words.append(_word(COL_XS[1], y, a, 101 + n, 20))
            words.append(_word(COL_XS[2], y, b, 101 + n, 21))
            out_rows.append([" ".join(filler), a, b, "", ""])
        words.append(_word(95.0, y0 + 2 * PITCH, "between", 110, 0))
        got = {f["predicate"] for f in ship_gate.native_ship_gate(words, _md(HEADER, out_rows))}
        assert got == {ship_gate.LABEL_ROW_MISSING}

    @staticmethod
    def _narrow_section(heading: str, drop_label: bool, drop_row: bool):
        """Four-lane rows, a heading, then a NARROWER (two-lane) trailing section."""
        rows = [list(r) for r in ROWS[:3]]
        blank = ["", "", "", ""]
        narrow = [
            ["N1", "1.5", "2.5", "", ""],
            ["Panel B"] + blank,
            ["N2", "3.5", "4.5", "", ""],
            ["N3", "5.5", "6.5", "", ""],
        ]
        words = _words([HEADER] + rows + [[heading] + blank] + narrow)
        md_narrow = [
            r
            for r in narrow
            if not (drop_label and r[0] == "Panel B") and not (drop_row and r[0] == "N2")
        ]
        return words, _md(HEADER, rows + [[heading] + blank] + md_narrow)

    @pytest.mark.parametrize("heading", ["Source of shock", "Notes:", "Source:"])
    def test_a_narrow_section_after_a_source_or_notes_heading_is_still_checked(
        self, heading: str
    ) -> None:
        words, md = self._narrow_section(heading, drop_label=False, drop_row=False)
        assert ship_gate.native_ship_gate(words, md) == ()
        words, md = self._narrow_section(heading, drop_label=True, drop_row=False)
        assert {f["predicate"] for f in ship_gate.native_ship_gate(words, md)} == {
            ship_gate.LABEL_ROW_MISSING
        }
        words, md = self._narrow_section(heading, drop_label=False, drop_row=True)
        assert {f["predicate"] for f in ship_gate.native_ship_gate(words, md)} == {
            ship_gate.DATA_ROW_MISSING
        }

    def test_a_neighbouring_columns_notes_opener_does_not_suppress_this_table(self) -> None:
        rows = [list(r) for r in ROWS]
        blank = ["", "", "", ""]
        words = _words([HEADER] + rows[:3] + [["Panel B"] + blank] + rows[3:])
        # A second column's paragraph opens with "Notes:" on a row inside this table.
        words.append(_word(520.0, Y0 + 4 * PITCH, "Notes:", 200, 0))
        words.append(_word(560.0, Y0 + 4 * PITCH, "elsewhere", 200, 1))
        got = {f["predicate"] for f in ship_gate.native_ship_gate(words, _md(HEADER, rows))}
        assert ship_gate.LABEL_ROW_MISSING in got

    def test_data_resuming_after_an_in_table_notes_heading_is_core_again(self) -> None:
        # Full-width rows after the heading are table rows: a label dropped
        # further down is still seen.
        rows = [list(r) for r in ROWS]
        blank = ["", "", "", ""]
        words = _words(
            [HEADER]
            + rows[:2]
            + [["Notes:"] + blank]
            + rows[2:4]
            + [["Panel D"] + blank]
            + rows[4:]
        )
        md_rows = rows[:2] + [["Notes:"] + blank] + rows[2:]
        got = {f["predicate"] for f in ship_gate.native_ship_gate(words, _md(HEADER, md_rows))}
        assert got == {ship_gate.LABEL_ROW_MISSING}

    def test_text_heavy_final_rows_do_not_hide_a_dropped_panel_heading(self) -> None:
        # Short rows, a panel heading, then two final numeric rows with long text in
        # the lane region. Word count must not push those rows out of the table.
        rows = [list(r) for r in ROWS]
        blank = ["", "", "", ""]
        heavy = []
        for k, base in enumerate(rows[3:5]):
            heavy.append(["Long label " * 3 + "ab"[k]] + base[1:])
        words = _words([HEADER] + rows[:3] + [["Panel B"] + blank] + heavy)
        for k in range(2):
            y = Y0 + (5 + k) * PITCH
            for j in range(10):
                words.append(_word(100.0 + 24.0 * j, y, f"w{j}", 120 + k, 30 + j))
        with_heading = rows[:3] + [["Panel B"] + blank] + heavy
        assert ship_gate.native_ship_gate(words, _md(HEADER, with_heading)) == ()
        without = rows[:3] + heavy
        got = {f["predicate"] for f in ship_gate.native_ship_gate(words, _md(HEADER, without))}
        assert ship_gate.LABEL_ROW_MISSING in got


class TestPunctuatedValues:
    """Numeric recognition matches the verifier: ``12.5,`` is a number."""

    @staticmethod
    def _punct(rows):
        return [[r[0]] + [c + "," for c in r[1:]] for r in rows]

    def test_order_and_sign_checks_stay_active_on_a_punctuated_table(self) -> None:
        rows = self._punct(ROWS)
        words = _words([HEADER] + rows)
        assert ship_gate.native_ship_gate(words, _md(HEADER, rows)) == ()
        reversed_rows = _md(HEADER, list(reversed(rows)))
        assert ship_gate.ROW_ORDER in {
            f["predicate"] for f in ship_gate.native_ship_gate(words, reversed_rows)
        }
        swapped = [r[:1] + list(reversed(r[1:])) for r in rows]
        assert ship_gate.CELL_ORDER in {
            f["predicate"] for f in ship_gate.native_ship_gate(words, _md(HEADER, swapped))
        }
        # A detached sign before a punctuated number.
        signed = [list(r) for r in rows]
        signed[2][1] = "0.230,"
        words = _words([HEADER] + signed)
        y = Y0 + 3 * PITCH
        x = COL_XS[1]
        words.append((x + 0.2 - CHAR_W * 0.7, y, x + 0.2, y + 9.0, "-", 0, 3, 9))
        md_rows = [list(r) for r in signed]
        md_rows[2] = md_rows[2][:1] + ["-"] + md_rows[2][1:]
        md = "\n".join(
            ["| " + " | ".join(HEADER + [""]) + " |", "| " + " | ".join(["---"] * 6) + " |"]
            + ["| " + " | ".join(r) + " |" for r in md_rows]
        )
        assert ship_gate.SIGN_DETACHED in {
            f["predicate"] for f in ship_gate.native_ship_gate(words, md)
        }

    def test_a_dropped_punctuated_edge_row_fires(self) -> None:
        rows = self._punct(ROWS)
        words = _words([HEADER] + rows)
        for drop in (0, -1):
            kept = [r for i, r in enumerate(rows) if i != drop % len(rows)]
            got = {f["predicate"] for f in ship_gate.native_ship_gate(words, _md(HEADER, kept))}
            assert got == {ship_gate.DATA_ROW_MISSING}, (drop, got)


# ------------------------------------------------------ lane + resume pins


def _config() -> PipelineConfig:
    return PipelineConfig(
        agentic=True,
        native_first=True,
        native_only=False,
        primary_engine=EngineType.QWEN,
        local_engine=EngineType.QWEN,
        enabled_engines=[EngineType.QWEN],
        tiered=False,
        dual_pass_tables=False,
        detect_equations=False,
        save_figures=False,
        quiet=True,
        table_judge_ladder=False,
    )


def _dense_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 50), "Table 1. GDP growth forecasts.", fontsize=10, fontname="helv")
    for ci, hdr in enumerate(HEADER):
        page.insert_text((COL_XS[ci], 80), hdr, fontsize=9, fontname="helv")
    for ri, row in enumerate(ROWS):
        for ci, cell in enumerate(row):
            page.insert_text((COL_XS[ci], 100 + ri * 22), cell, fontsize=9, fontname="helv")
    doc.save(str(path))
    doc.close()


class TestLane:
    def _run(self, tmp_path: Path, providers: list, *, drop_row: bool):
        from socr.core.born_digital import BornDigitalDetector

        pdf = tmp_path / "t.pdf"
        if not pdf.exists():
            _dense_pdf(pdf)
        pipeline = UnifiedPipeline(_config())
        if drop_row:
            real = BornDigitalDetector()

            class _Drop(BornDigitalDetector):
                def detect(self, path):
                    assessment = real.detect(path)
                    page = assessment.pages[0]
                    lines = (page.native_text or "").splitlines()
                    keep = [ln for ln in lines if not ln.startswith("| TR ")]
                    assert len(keep) == len(lines) - 1
                    page.native_text = "\n".join(keep)
                    return assessment

            pipeline.bd_detector = _Drop()
        routes: list[int] = []

        def _route(page_num, ladder, run_provider, judge, **kwargs):
            routes.append(page_num)
            out = PageOutput(
                page_num=page_num,
                text="model table",
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            )
            from socr.pipeline.agentic import PageDecision, ProviderAttempt

            prof = ladder[0]
            att = ProviderAttempt(
                engine=prof.engine,
                output=out,
                cost_usd=0.0,
                accepted=True,
                reason="test",
                provider_id=prof.id,
                model=prof.model,
                backend=prof.backend,
            )
            return PageDecision(page_num=page_num, final_output=out, attempts=[att], accepted=True)

        out_dir = tmp_path / ("out_drop" if drop_row else "out_ok")
        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
            patch.object(pipeline, "_available_engines_for_agentic", return_value=providers),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            result = pipeline.process(pdf, out_dir)
        sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text(encoding="utf-8"))
        return result, sidecar, routes

    @pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "none"])
    def test_gate_defers_to_route_page_and_records_the_fault(
        self, tmp_path: Path, providers: list
    ) -> None:
        clean_result, clean_side, clean_routes = self._run(tmp_path, providers, drop_row=False)
        _result, side, routes = self._run(tmp_path, providers, drop_row=True)
        clean_kinds = [e["kind"] for e in clean_side["audit_events"]]
        kinds = [e["kind"] for e in side["audit_events"]]
        # Same page, only the dropped row differs: native exact-pass vs deferral.
        assert "native_table_exact_pass" in clean_kinds and clean_routes == []
        assert ship_gate.SHIP_GATE_KIND not in clean_kinds
        assert "native_table_exact_pass" not in kinds
        assert kinds.count(ship_gate.SHIP_GATE_KIND) == 1
        event = next(e for e in side["audit_events"] if e["kind"] == ship_gate.SHIP_GATE_KIND)
        assert ship_gate.DATA_ROW_MISSING in event["data"]["predicates"]
        # DEFER, not REFUSE: no D3 image-floor marker, and with a provider the
        # model is asked (a REFUSE would never reach route_page).
        assert "native_table_cell_unresolved" not in kinds
        if providers:
            assert routes == [1]
        else:
            assert routes == []
        assert clean_result.markdown != "model table"


class TestEventSurvivesResume:
    def _emit_flush_restore(self, tmp_path: Path):
        from socr.core.born_digital import BornDigitalDetector

        pdf = tmp_path / "t.pdf"
        _dense_pdf(pdf)
        out_dir = tmp_path / "out"
        pipeline = UnifiedPipeline(_config())
        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        pipeline._phase_analyze(state)
        ps = state.pages[1]
        assert BornDigitalDetector is not None
        ps.native_text = "\n".join(
            ln for ln in (ps.native_text or "").splitlines() if "| TR " not in ln
        )
        with patch.object(pipeline, "_available_engines_for_agentic", return_value=[]):
            assert pipeline._plan_native_table_first(state, 1, ps) is None
        assert pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True) is not None
        resumed = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        resumed.pages[1] = ps
        restored = PageOutput(
            page_num=1,
            text="model table",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        pipeline._restore_terminal_page_state(resumed, 1, restored, out_dir)
        return state, resumed

    def test_event_replays_exactly_once(self, tmp_path: Path) -> None:
        kind = ship_gate.SHIP_GATE_KIND
        state, resumed = self._emit_flush_restore(tmp_path)
        assert [e.kind for e in state.events].count(kind) == 1
        assert [e.kind for e in resumed.events].count(kind) == 1
        assert isinstance(resumed.events[0], AuditEvent)

    def test_kind_is_in_the_resume_allowlist(self) -> None:
        assert ship_gate.SHIP_GATE_KIND in UnifiedPipeline.resume_restore_kinds()


# ---------------------------------------------------- round 2: wiring pins


class TestRotatedGateEvent:
    """The gate fires on a rotated attempt and the event is recorded.

    The quarantine also DEFERs a rotated page, so the gate event is the only
    record of WHY native was rejected when a real fault exists. Difference pin:
    the same rotated page with and without a dropped row.
    """

    @staticmethod
    def _plan(tmp_path: Path, *, drop_last_row: bool):
        import socr.tables.reconstruct as reconstruct
        from test_rotated_native_table_first import _rotated_dense_forecast_pdf

        pdf = tmp_path / "rotated.pdf"
        if not pdf.exists():
            _rotated_dense_forecast_pdf(pdf)
        pipeline = UnifiedPipeline(_config())
        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        pipeline._phase_analyze(state)
        real = reconstruct.rowize_from_words

        def _dropping(page, **kwargs):
            regions = real(page, **kwargs)
            out = []
            for rect, md in regions:
                lines = md.splitlines()
                if drop_last_row:
                    lines = [ln for ln in lines if "| TR |" not in ln]
                out.append((rect, "\n".join(lines)))
            return out

        with patch.object(reconstruct, "rowize_from_words", side_effect=_dropping):
            work = pipeline._plan_native_table_first(state, 1, state.pages[1])
        return work, [e.kind for e in state.events], state.events

    def test_gate_event_recorded_when_the_rotated_grid_has_a_fault(self, tmp_path: Path) -> None:
        clean_work, clean_kinds, _ = self._plan(tmp_path, drop_last_row=False)
        work, kinds, events = self._plan(tmp_path, drop_last_row=True)
        assert clean_work is None and work is None  # both defer to route_page
        # Quarantine alone explains the clean page; the faulty page is explained by
        # the gate, and the gate event is on the record.
        assert "rotated_native_table_quarantined" in clean_kinds
        assert ship_gate.SHIP_GATE_KIND not in clean_kinds
        assert kinds.count(ship_gate.SHIP_GATE_KIND) == 1
        event = next(e for e in events if e.kind == ship_gate.SHIP_GATE_KIND)
        assert ship_gate.DATA_ROW_MISSING in event.data["predicates"]


class TestGateError:
    def test_a_predicate_that_raises_defers_and_never_ships(self) -> None:
        words, md = _base()
        assert _plan(words, md).action == SHIP
        with patch.object(ship_gate, "sign_detached_faults", side_effect=RuntimeError("boom")):
            faults = ship_gate.native_ship_gate(words, md)
            plan = plan_native_table(words, md)
        assert [f["predicate"] for f in faults] == [ship_gate.GATE_ERROR]
        assert plan.action == DEFER and plan.action != SHIP
        assert ship_gate.GATE_ERROR in plan.reason

    def test_a_gate_error_is_recorded_like_any_other_fault(self) -> None:
        words, md = _base()
        state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
        with patch.object(ship_gate, "order_faults", side_effect=ValueError("bad")):
            plan = plan_native_table(words, md)
        UnifiedPipeline._record_native_ship_gate(state, 1, plan)
        assert [e.kind for e in state.events] == [ship_gate.SHIP_GATE_KIND]
        assert state.events[0].data["predicates"] == [ship_gate.GATE_ERROR]


class TestGateDeferEqualsOrdinaryDefer:
    """Only the audit event may differ between a gate DEFER and an ordinary DEFER."""

    @pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "none"])
    def test_flags_text_and_selection_are_identical(self, tmp_path: Path, providers: list) -> None:
        def run(name: str, *, ordinary: bool):
            sub = tmp_path / name
            sub.mkdir()
            runner = TestLane()
            if not ordinary:
                return runner._run(sub, providers, drop_row=True)
            ordinary_plan = nf.NativeTablePlan(DEFER, reason="AMBIGUOUS")
            with patch.object(nf, "plan_native_table", return_value=ordinary_plan):
                return runner._run(sub, providers, drop_row=True)

        g_result, g_side, g_routes = run("gate", ordinary=False)
        o_result, o_side, o_routes = run("ordinary", ordinary=True)

        g_kinds = [e["kind"] for e in g_side["audit_events"]]
        o_kinds = [e["kind"] for e in o_side["audit_events"]]
        assert ship_gate.SHIP_GATE_KIND in g_kinds
        assert ship_gate.SHIP_GATE_KIND not in o_kinds
        assert [k for k in g_kinds if k != ship_gate.SHIP_GATE_KIND] == o_kinds
        assert g_routes == o_routes
        assert g_result.markdown == o_result.markdown

        def strip(side: dict) -> dict:
            volatile = {"audit_events", "input_checksum", "timings_s"}  # per-run bytes, clocks
            kept = {k: v for k, v in side.items() if k not in volatile}
            return json.loads(json.dumps(kept, sort_keys=True, default=str))

        assert strip(g_side) == strip(o_side)


class TestSwallowedNotesDoNotStretchTheTable:
    def test_lines_with_one_stray_number_are_not_core(self) -> None:
        # The grid ships two lines (one stray number each, no opener) as rows but
        # not the unnumbered line between them. They pair by that single number, so
        # they must not move the table's last row down to them.
        words, _ = _base()
        y = Y0 + (len(ROWS) + 2) * PITCH
        lines = [("stars noted", "10%"), ("were clustered", None), ("again", "5%")]
        for n, (txt, num) in enumerate(lines):
            words.append(_word(90.0, y + n * PITCH, txt, 70 + n, 0))
            if num:
                words.append(_word(300.0, y + n * PITCH, num, 70 + n, 1))
        md_rows = [list(r) for r in ROWS]
        md_rows += [["stars noted", "", "10%", "", ""], ["again", "", "5%", "", ""]]
        assert ship_gate.native_ship_gate(words, _md(HEADER, md_rows)) == ()
