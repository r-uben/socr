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


def _sign_rows(*, sign_dx: float, at_end_of_label: bool = False):
    """Row NEG carries a sign word ``sign_dx`` pt left of its second number."""
    rows = [list(r) for r in ROWS]
    rows[2][1] = "0.230"
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
