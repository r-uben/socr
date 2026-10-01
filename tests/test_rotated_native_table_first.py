"""Upright retry for GH-147 refused born-digital table pages.

After analyze refuses the sideways rowizer output, agentic native-table-first
re-rowizes upright and runs ``plan_native_table``. Only an exact pass (SHIP)
ships the grid without ``route_page``. Any other outcome — no grid, read error,
REFUSE, DEFER, or CELLS — leaves the page on the existing whole-page route.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from native_table_fixtures import (
    HEADER,
    ROWS,
    UNCHECKED,
    flush_and_restore,
    forecast_pdf,
    native_first_config,
    rotated_forecast_pdf,
    routed_decision,
)

from socr.core.born_digital import BornDigitalDetector
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.native_first import (
    DEFER,
    REFUSE,
    ROTATED_QUARANTINE_KIND,
    ROTATED_SHIP_QUARANTINED,
    SHIP,
    NativeTablePlan,
    RotatedNativeTableAttempt,
    attempt_rotated_native_table,
    plan_native_table,
)

#: A real rotated Fed minutes page (p14); the tests using it skip where the store is absent.
_FOMC_PDF = Path(
    "/cursor/stores/bc-f87a255a-cbe0-40f3-b811-f505faf8233d/media/"
    "fed-minutes-example/fomcminutes20190619.pdf"
)
requires_fomc_fixture = pytest.mark.skipif(
    not _FOMC_PDF.is_file(), reason="FOMC fixture not present in the project store"
)


class TestAttemptRotatedNativeTable:
    def test_upright_rowizer_exact_passes(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "rotated.pdf"
        rotated_forecast_pdf(pdf_path)
        assessment = BornDigitalDetector().detect(pdf_path).pages[0]
        assert assessment.native_table_lane_refused is True
        doc = fitz.open(pdf_path)
        attempt = attempt_rotated_native_table(doc[0])
        doc.close()
        assert attempt is not None
        # GH-917: the grid is still built, but the SHIP is quarantined.
        assert attempt.plan.action == DEFER
        assert attempt.plan.reason == ROTATED_SHIP_QUARANTINED
        assert "0.253" in attempt.markdown
        # Difference pin: the planner itself still says SHIP on the same inputs,
        # so the quarantine is the only thing that changed the action.
        raw = plan_native_table(
            attempt.words,
            attempt.markdown,
            structure_defective=attempt.structure_defective,
            header_unattributed=attempt.header_unattributed,
            orphan_words=list(attempt.orphan_words),
            line_dirs=UNCHECKED,
        )
        assert raw.action == SHIP

    @requires_fomc_fixture
    def test_fomc_page_14_does_not_exact_pass_upright(self) -> None:
        """Real rotated table: upright arm must not invent a shippable grid."""
        doc = fitz.open(_FOMC_PDF)
        attempt = attempt_rotated_native_table(doc[13])
        doc.close()
        assert attempt is not None
        assert attempt.plan.action != SHIP


class TestAgenticRotatedNativeTableFirst:
    def _run(self, tmp_path: Path, providers: list, *, quarantine: bool):
        """One process() run; ``quarantine=False`` restores the pre-#917 SHIP."""
        import dataclasses

        pdf_path = tmp_path / "rotated.pdf"
        if not pdf_path.exists():
            rotated_forecast_pdf(pdf_path)
        out_dir = tmp_path / ("out_q" if quarantine else "out_noq")
        pipeline = UnifiedPipeline(native_first_config())
        route_calls: list[int] = []

        def _route(page_num, ladder, run_provider, judge, **kwargs):
            route_calls.append(page_num)
            return routed_decision(page_num, ladder)

        def _unquarantined(page):
            attempt = attempt_rotated_native_table(page)
            raw = plan_native_table(
                attempt.words,
                attempt.markdown,
                structure_defective=attempt.structure_defective,
                header_unattributed=attempt.header_unattributed,
                orphan_words=list(attempt.orphan_words),
                line_dirs=UNCHECKED,
            )
            return dataclasses.replace(attempt, plan=raw)

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
            patch.object(pipeline, "_available_engines_for_agentic", return_value=providers),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            if quarantine:
                result = pipeline.process(pdf_path, out_dir)
            else:
                with patch(
                    "socr.tables.native_first.attempt_rotated_native_table",
                    side_effect=_unquarantined,
                ):
                    result = pipeline.process(pdf_path, out_dir)
        sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text(encoding="utf-8"))
        return result, sidecar, route_calls

    @pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "none"])
    def test_rotated_exact_pass_is_quarantined_not_shipped(
        self, tmp_path: Path, providers: list
    ) -> None:
        """GH-917: a rotated exact-pass must not ship as native SUCCESS.

        Difference pin: the same page run twice in this process, changing only
        whether the quarantine is in effect.
        """
        q_result, q_side, q_routes = self._run(tmp_path, providers, quarantine=True)
        u_result, u_side, u_routes = self._run(tmp_path, providers, quarantine=False)

        q_kinds = [ev["kind"] for ev in q_side["audit_events"]]
        u_kinds = [ev["kind"] for ev in u_side["audit_events"]]
        # Quarantine on: no exact-pass claim, the quarantine is recorded once.
        assert "landscape_page_refused" in q_kinds
        assert q_kinds.count(ROTATED_QUARANTINE_KIND) == 1
        assert "native_table_exact_pass" not in q_kinds
        # Quarantine off: the old behaviour, so the pin is not vacuous.
        assert "native_table_exact_pass" in u_kinds
        assert ROTATED_QUARANTINE_KIND not in u_kinds
        assert u_routes == []
        # The page no longer takes the native-grid outcome.
        assert q_result.markdown != u_result.markdown
        assert q_side.get("engine") != u_side.get("engine") or (
            q_side.get("status") != u_side.get("status")
        )
        if providers:
            # With a provider the quarantined page reaches route_page and the
            # routed output is what ships.
            assert q_routes == [1]
            assert "model table" in (q_result.markdown or "")
        else:
            # Empty ladder: nothing routes, and the native grid still must not
            # ship as an exact-pass SUCCESS.
            assert q_routes == []

    def test_upright_failed_check_calls_route_page(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "rotated.pdf"
        rotated_forecast_pdf(pdf_path)
        pipeline = UnifiedPipeline(native_first_config())
        route_calls: list[int] = []

        def _route(page_num, ladder, run_provider, judge, **kwargs):
            route_calls.append(page_num)
            return routed_decision(page_num, ladder)

        refuse_attempt = RotatedNativeTableAttempt(
            plan=NativeTablePlan(REFUSE, reason="row_count"),
            markdown="| a | b |\n| --- | --- |\n| 1 | 2 |",
            words=[],
            structure_defective=False,
            header_unattributed=False,
        )

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch(
                "socr.tables.native_first.attempt_rotated_native_table",
                return_value=refuse_attempt,
            ),
        ):
            result = pipeline.process(pdf_path, tmp_path / "out")
        assert route_calls == [1]
        sidecar = json.loads(
            next((tmp_path / "out").rglob("pages/00001.json")).read_text(encoding="utf-8")
        )
        kinds = [ev["kind"] for ev in sidecar["audit_events"]]
        # The event lives in the sidecar's audit_events, not in the markdown:
        # a REFUSE plan must not record an exact-pass claim.
        assert "native_table_exact_pass" not in kinds
        assert "native_table_cell_unresolved" not in kinds

    @requires_fomc_fixture
    def test_fomc_page_14_reaches_route_page(self, tmp_path: Path) -> None:
        single = tmp_path / "fomc-p14.pdf"
        src = fitz.open(_FOMC_PDF)
        doc = fitz.open()
        doc.insert_pdf(src, from_page=13, to_page=13)
        doc.save(single)
        doc.close()
        src.close()

        pipeline = UnifiedPipeline(native_first_config())
        route_calls: list[int] = []

        def _route(page_num, ladder, run_provider, judge, **kwargs):
            route_calls.append(page_num)
            return routed_decision(
                page_num, ladder, text="", status=PageStatus.ERROR, accepted=False
            )

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            pipeline.process(single, tmp_path / "out")
        assert route_calls == [1]


# ---------------------------------------------------------------------------
# GH-902: the upright correction must be applied with the right sign
# ---------------------------------------------------------------------------

_LABEL_ORDER = [HEADER[0]] + [r[0] for r in ROWS]


def _first_column(markdown: str) -> list[str]:
    cells = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("| ---"):
            continue
        # The caption's words share the first column in this fixture, so the label
        # is the leftmost non-empty cell; the column holding it is checked below.
        row = [c.strip() for c in stripped.strip("|").split("|")]
        cells.append(next((c for c in row if c), ""))
    # Caption / footnote rows may share the grid; the labels' ORDER is the pin.
    return [c for c in cells if c in _LABEL_ORDER]


def _rowized_markdown(pdf_path: Path, *, rotation_sign: int = 1) -> str:
    from socr.core.born_digital import upright_rotation_for
    from socr.tables.reconstruct import rowize_from_word_list

    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        regions = rowize_from_word_list(
            page.get_text("words"),
            rotation=rotation_sign * upright_rotation_for(page),
            page_rect=page.rect,
        )
    finally:
        doc.close()
    return "\n\n".join(md for _rect, md in regions)


@pytest.mark.parametrize("rotation", [90, 270])
class TestRotationSign:
    def test_rowized_grid_equals_the_upright_twin_in_reading_order(
        self, tmp_path: Path, rotation: int
    ) -> None:
        upright_pdf = tmp_path / "upright.pdf"
        rotated_pdf = tmp_path / "rotated.pdf"
        forecast_pdf(upright_pdf, 0)
        forecast_pdf(rotated_pdf, rotation)

        fixed = _rowized_markdown(rotated_pdf)
        upright = _rowized_markdown(upright_pdf)
        assert upright.strip(), "the upright twin must rowize, or the pin is vacuous"
        assert _first_column(upright) == _LABEL_ORDER
        assert _first_column(fixed) == _LABEL_ORDER
        assert fixed == upright
        # Difference pin: the previous sign reads the same page 180 degrees flipped.
        old_sign = _rowized_markdown(rotated_pdf, rotation_sign=-1)
        assert old_sign != upright
        assert _first_column(old_sign) != _LABEL_ORDER

    def test_attempt_matches_the_upright_twin_and_ships_in_reading_order(
        self, tmp_path: Path, rotation: int
    ) -> None:
        from socr.tables.reconstruct import rowize_from_words

        upright_pdf = tmp_path / "upright.pdf"
        rotated_pdf = tmp_path / "rotated.pdf"
        forecast_pdf(upright_pdf, 0)
        forecast_pdf(rotated_pdf, rotation)

        doc = fitz.open(upright_pdf)
        upright_md = "\n\n".join(md for _r, md in rowize_from_words(doc[0]) if md.strip())
        doc.close()
        doc = fitz.open(rotated_pdf)
        attempt = attempt_rotated_native_table(doc[0])
        doc.close()

        assert attempt is not None
        assert attempt.markdown == upright_md
        assert attempt.plan.action == DEFER
        assert attempt.plan.reason == ROTATED_SHIP_QUARANTINED
        assert _first_column(attempt.markdown) == _LABEL_ORDER

    def test_the_witness_words_read_in_the_upright_twins_order(
        self, tmp_path: Path, rotation: int
    ) -> None:
        """The comparison words must be upright on their own, not just agree with the grid.

        ``plan_native_table`` cannot catch a flipped witness: its verifier pairs rows
        by number multiset, so a grid and witness flipped TOGETHER (or a grid flipped
        against a correct witness) both still exact-pass (measured: reversing rows,
        cells, or both leaves SHIP). The witness's own geometry is therefore the only
        place the sign is observable, so it is pinned directly against the upright twin.
        """
        from socr.tables.native_first import upright_words_for_page

        upright_pdf = tmp_path / "upright.pdf"
        rotated_pdf = tmp_path / "rotated.pdf"
        forecast_pdf(upright_pdf, 0)
        forecast_pdf(rotated_pdf, rotation)

        def reading_order(pdf: Path) -> list[str]:
            doc = fitz.open(pdf)
            try:
                words, _ = upright_words_for_page(doc[0])
            finally:
                doc.close()
            return [w[4] for w in sorted(words, key=lambda w: (round(w[1]), w[0]))]

        expected = reading_order(upright_pdf)
        assert expected, "the upright twin must have words, or the pin is vacuous"
        assert reading_order(rotated_pdf) == expected

    def test_region_rect_encloses_the_table_in_page_coordinates(
        self, tmp_path: Path, rotation: int
    ) -> None:
        """The output rect is rotated back to the page frame with the opposite sign."""
        from socr.tables.reconstruct import rowize_from_words

        rotated_pdf = tmp_path / "rotated.pdf"
        forecast_pdf(rotated_pdf, rotation)
        doc = fitz.open(rotated_pdf)
        try:
            page = doc[0]
            regions = rowize_from_words(page)
            body = {c for row in ROWS for c in row}
            cells = [w for w in page.get_text("words") if w[4] in body]
        finally:
            doc.close()
        assert regions and cells
        rect = regions[0][0]
        for x0, y0, x1, y1, *_ in cells:
            assert rect.x0 - 1 <= x0 and x1 <= rect.x1 + 1
            assert rect.y0 - 1 <= y0 and y1 <= rect.y1 + 1


class TestQuarantineEventSurvivesResume:
    """GH-917: the quarantine record must replay exactly once on resume."""

    def _emit_flush_restore(self, tmp_path: Path):
        pdf = tmp_path / "rotated.pdf"
        rotated_forecast_pdf(pdf)
        out_dir = tmp_path / "out"
        pipeline = UnifiedPipeline(native_first_config())
        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        pipeline._phase_analyze(state)
        # Real emit site; run 1 only (a resumed page is dropped from ocr_pages
        # before planning, so nothing re-emits it).
        assert pipeline._plan_native_table_first(state, 1, state.pages[1]) is None
        return state, flush_and_restore(pipeline, state, pdf, out_dir)

    def test_quarantine_event_replays_exactly_once(self, tmp_path: Path) -> None:
        kind = ROTATED_QUARANTINE_KIND
        state, resumed = self._emit_flush_restore(tmp_path)
        assert [e.kind for e in state.events].count(kind) == 1
        assert [e.kind for e in resumed.events].count(kind) == 1

    def test_kind_is_in_the_resume_allowlist(self) -> None:
        assert ROTATED_QUARANTINE_KIND in UnifiedPipeline.resume_restore_kinds()
