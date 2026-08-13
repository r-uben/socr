"""Tests for GH-147 A2: refuse the native table lane on rotated pages.

Synthetic PyMuPDF fixtures only (no Fama/corpus). Confirms three binding
evidence groups:

  1. ``BornDigitalDetector.detect_page`` on a rotated ruled-grid+prose page
     retains prose, emits no markdown table separator, keeps ``has_tables``
     True, sets ``needs_ocr_enhancement`` True, and never calls
     ``extract_structured``. A rotated prose-only page (no table signal) is
     the negative control: rotation alone must not set
     ``needs_ocr_enhancement``.
  2. Under ``--native-only``, ``_is_trusted_native_without_ocr`` is False for
     a rotated table page and True for an upright table page, while
     ``_is_native_eligible_without_ocr`` (the chart-lane primitive) is True
     for both.
  3. A hermetic ``process()`` run on the rotated-grid PDF emits exactly one
     ``landscape_page_refused`` audit event, in both ``state.events`` and the
     page sidecar, ships ``DocumentStatus.AUDIT_FAILED`` with a native-prose
     WARNING fallback, and accepts no OCR output.
  4. Review negative control (PR #193): a non-born-digital rotated page that
     also carries a ruled table -- ``has_tables`` is stamped before the early
     non-born-digital returns in ``_assess_page_signals``, so the refusal
     branch never runs and ``native_table_lane_refused`` stays False. The
     audit predicate in ``_phase_analyze`` must not emit
     ``landscape_page_refused`` for such a page.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import BornDigitalDetector, DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

_PROSE_LINES = [
    "prose retained above the rotated grid despite table refusal",
    "second prose line carrying real sentence content for this page",
    "third prose line so the fixture reads as a genuine landscape page",
]


def _rotated_ruled_grid_pdf(path: Path) -> None:
    """A page whose ALL text (prose + cell contents) runs at 90 degrees, with
    a ruled cell grid so PyMuPDF's built-in ``find_tables()`` fires — the
    signal ``_detect_tables`` uses, independent of text direction or the
    numeric-column lane-cooccupancy gate.
    """
    doc = fitz.open()
    page = doc.new_page()

    x0, y0 = 100, 100
    cw, rh = 60, 20
    cols, rows = 3, 4
    for r in range(rows + 1):
        page.draw_line((x0, y0 + r * rh), (x0 + cols * cw, y0 + r * rh))
    for c in range(cols + 1):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + rows * rh))
    for r in range(rows):
        for c in range(cols):
            page.insert_text(
                (x0 + c * cw + 5, y0 + r * rh + 15),
                f"c{r}{c}",
                fontsize=8,
                fontname="helv",
                rotate=90,
            )

    prose_y = 400
    for line in _PROSE_LINES:
        page.insert_text((72, prose_y), line, fontsize=11, fontname="helv", rotate=90)
        prose_y += 16

    doc.save(str(path))
    doc.close()


def _rotated_prose_only_pdf(path: Path) -> None:
    """Rotated text, no ruled grid: negative control for the table conjunction."""
    doc = fitz.open()
    page = doc.new_page()
    y = 400
    for line in _PROSE_LINES:
        page.insert_text((72, y), line, fontsize=11, fontname="helv", rotate=90)
        y += 16
    doc.save(str(path))
    doc.close()


def _rotated_garbled_grid_pdf(path: Path) -> None:
    """Rotated ruled grid + CID-artifact prose: NOT born-digital.

    Same ruled-grid table signal as ``_rotated_ruled_grid_pdf``, but the prose
    carries literal ``(cid:N)`` font-mapping artifacts, a definitive
    not-born-digital signal (``has_cid``) that ``_assess_page_signals``
    checks via an early return -- before it ever reaches the rotated-table
    refusal branch. ``has_tables`` is stamped before that early return, so
    this is the exact page shape the review flagged: rotated AND has_tables,
    but the refusal branch never ran.
    """
    doc = fitz.open()
    page = doc.new_page()

    x0, y0 = 100, 100
    cw, rh = 60, 20
    cols, rows = 3, 4
    for r in range(rows + 1):
        page.draw_line((x0, y0 + r * rh), (x0 + cols * cw, y0 + r * rh))
    for c in range(cols + 1):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + rows * rh))
    for r in range(rows):
        for c in range(cols):
            page.insert_text(
                (x0 + c * cw + 5, y0 + r * rh + 15),
                f"c{r}{c}",
                fontsize=8,
                fontname="helv",
                rotate=90,
            )

    # CID font-mapping artifacts: a definitive "not born-digital" signal
    # (``has_cid`` in ``_assess_page_signals``), reliably reached with a
    # normal-length word list so this text never trips the earlier avg-word-
    # length / short-text gates first.
    cid_line = " ".join(f"(cid:{n})" for n in range(20))
    prose_y = 400
    for _ in range(3):
        page.insert_text((72, prose_y), cid_line, fontsize=11, fontname="helv", rotate=90)
        prose_y += 16

    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# Group 1: detect_page on the rotated grid
# ---------------------------------------------------------------------------


class TestRotatedGridRefusesNativeTable:
    def test_prose_retained_no_separator_extract_structured_never_called(
        self, tmp_path: Path
    ) -> None:
        pdf_path = tmp_path / "rotated_grid.pdf"
        _rotated_ruled_grid_pdf(pdf_path)

        detector = BornDigitalDetector()
        with patch.object(
            BornDigitalDetector, "extract_structured", wraps=detector.extract_structured
        ) as spy:
            assessment = detector.detect_page(pdf_path, 1)

            spy.assert_not_called()

        for line in _PROSE_LINES:
            assert line in assessment.native_text, f"prose line missing from native_text: {line!r}"
        assert "---" not in assessment.native_text
        assert "| --- |" not in assessment.native_text
        assert assessment.has_tables is True
        assert assessment.text_is_rotated is True
        assert assessment.needs_ocr_enhancement is True

    def test_rotated_prose_only_page_does_not_set_needs_ocr_enhancement(
        self, tmp_path: Path
    ) -> None:
        """Rotation alone is not the key -- only rotation AND tables refuses."""
        pdf_path = tmp_path / "rotated_prose.pdf"
        _rotated_prose_only_pdf(pdf_path)

        assessment = BornDigitalDetector().detect_page(pdf_path, 1)

        assert assessment.text_is_rotated is True
        assert assessment.has_tables is False
        assert assessment.needs_ocr_enhancement is False
        for line in _PROSE_LINES:
            assert line in assessment.native_text


# ---------------------------------------------------------------------------
# Group 1b: PR #193 review -- audit predicate overbreadth negative control
# ---------------------------------------------------------------------------


class TestNonBornDigitalRotatedTableNoRefusalAudit:
    """A rotated page with a ruled table that is classified NOT born-digital
    (high garbage ratio) must never claim a native-table-lane refusal: the
    refusal branch in ``_assess_page_signals`` lives past the early
    non-born-digital returns and never runs for this page, even though
    ``has_tables`` and ``text_is_rotated`` are both True.
    """

    def test_detect_page_flag_stays_false_on_non_bd_rotated_table_page(
        self, tmp_path: Path
    ) -> None:
        pdf_path = tmp_path / "rotated_garbled_grid.pdf"
        _rotated_garbled_grid_pdf(pdf_path)

        assessment = BornDigitalDetector().detect_page(pdf_path, 1)

        # The exact pre-condition the review flagged: both true, refusal not run.
        assert assessment.is_born_digital is False
        assert assessment.has_tables is True
        assert assessment.text_is_rotated is True
        assert assessment.native_table_lane_refused is False

    def test_phase_analyze_emits_no_refusal_event_for_non_bd_rotated_table_page(
        self, tmp_path: Path
    ) -> None:
        """Direct regression guard on the ``_phase_analyze`` audit predicate.

        Without the fix (predicate re-derived as ``text_is_rotated and
        has_tables``) this synthetic assessment -- non-born-digital, rotated,
        has_tables True, ``native_table_lane_refused`` False -- would still
        emit ``landscape_page_refused``. With the fix keyed on the explicit
        flag, it must not.
        """
        pdf_path = tmp_path / "fake.pdf"
        config = PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            agentic=True,
            enabled_engines=[EngineType.GEMINI],
            quiet=True,
            audit_enabled=False,
            save_figures=False,
            write_manifest=False,
        )
        pipeline = UnifiedPipeline(config)

        non_bd_rotated_table_pa = PageAssessment(
            page_num=1,
            is_born_digital=False,
            native_text="",
            confidence=0.8,
            has_tables=True,
            dominant_text_direction=(0.0, -1.0),
            native_table_lane_refused=False,
        )
        fake_assessment = DocumentAssessment(path=pdf_path, pages=[non_bd_rotated_table_pa])

        state = DocumentState(handle=DocumentHandle(path=pdf_path, page_count=1))

        with patch.object(pipeline.bd_detector, "detect", return_value=fake_assessment):
            pipeline._phase_analyze(state)

        refusal_events = [ev for ev in state.events if ev.kind == "landscape_page_refused"]
        assert refusal_events == [], (
            f"non-born-digital rotated table page must not emit landscape_page_refused; "
            f"got {state.events}"
        )


# ---------------------------------------------------------------------------
# Group 2: native_only predicates, in-memory PageAssessment/PageState
# ---------------------------------------------------------------------------


def _make_config(*, native_only: bool) -> PipelineConfig:
    return PipelineConfig(
        primary_engine=EngineType.DEEPSEEK,
        agentic=True,
        enabled_engines=[EngineType.GEMINI],
        native_first=True,
        native_only=native_only,
        quiet=True,
        audit_enabled=False,
        save_figures=False,
        write_manifest=False,
    )


class TestNativeOnlyPredicates:
    def _pipeline_with_assessment(
        self, *, rotated: bool, has_tables: bool, native_only: bool
    ) -> tuple[UnifiedPipeline, PageState]:
        pipeline = UnifiedPipeline(_make_config(native_only=native_only))
        direction = (0.0, -1.0) if rotated else (1.0, 0.0)
        pa = PageAssessment(
            page_num=1,
            is_born_digital=True,
            native_text="native prose " * 10,
            confidence=0.9,
            has_tables=has_tables,
            needs_ocr_enhancement=False,
            dominant_text_direction=direction,
        )
        pipeline._last_assessment = DocumentAssessment(path=Path("/tmp/fake.pdf"), pages=[pa])
        ps = PageState(
            page_num=1,
            is_born_digital=True,
            native_text="native prose " * 10,
            has_tables=has_tables,
        )
        return pipeline, ps

    def test_rotated_table_page_not_trusted_but_eligible(self) -> None:
        pipeline, ps = self._pipeline_with_assessment(
            rotated=True, has_tables=True, native_only=True
        )
        assert pipeline._is_native_eligible_without_ocr(1, ps) is True
        assert pipeline._is_trusted_native_without_ocr(1, ps) is False

    def test_upright_table_page_trusted_and_eligible(self) -> None:
        pipeline, ps = self._pipeline_with_assessment(
            rotated=False, has_tables=True, native_only=True
        )
        assert pipeline._is_native_eligible_without_ocr(1, ps) is True
        assert pipeline._is_trusted_native_without_ocr(1, ps) is True

    def test_rotated_table_less_page_still_trusted(self) -> None:
        """Rotated but table-less pages keep the unconditional native_only bypass."""
        pipeline, ps = self._pipeline_with_assessment(
            rotated=True, has_tables=False, native_only=True
        )
        assert pipeline._is_native_eligible_without_ocr(1, ps) is True
        assert pipeline._is_trusted_native_without_ocr(1, ps) is True


# ---------------------------------------------------------------------------
# Group 3: hermetic process()
# ---------------------------------------------------------------------------


def _rejected_decision_with_attempt(page_num: int, ladder):
    """A PageDecision that records a real attempt but accepts nothing."""
    from socr.pipeline.agentic import PageDecision, ProviderAttempt

    rejected_out = PageOutput(
        page_num=page_num,
        text="",
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
    )
    prof = ladder[0]
    att = ProviderAttempt(
        engine=prof.engine,
        output=rejected_out,
        cost_usd=0.0,
        accepted=False,
        reason="judge reject",
        provider_id=prof.id,
        model=prof.model,
        backend=prof.backend,
    )
    return PageDecision(
        page_num=page_num, final_output=rejected_out, attempts=[att], accepted=False
    )


class TestHermeticProcessRefusal:
    def test_rotated_table_page_emits_refusal_event_and_audit_failed(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "rotated_grid.pdf"
        _rotated_ruled_grid_pdf(pdf_path)

        config = PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            audit_enabled=False,
            save_figures=False,
            write_manifest=False,
        )
        pipeline = UnifiedPipeline(config)

        captured_state = {}
        orig_assemble = pipeline._phase_assemble

        def _capture_assemble(state, output_dir):
            captured_state["state"] = state
            return orig_assemble(state, output_dir)

        route_calls: list[int] = []

        def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
            route_calls.append(page_num)
            return _rejected_decision_with_attempt(page_num, ladder)

        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route),
            patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
            patch.object(pipeline, "_phase_assemble", side_effect=_capture_assemble),
        ):
            result = pipeline.process(pdf_path, tmp_path)

        assert route_calls == [1], f"route_page must be called for p1 only; got {route_calls}"

        state = captured_state["state"]
        refusal_events = [ev for ev in state.events if ev.kind == "landscape_page_refused"]
        assert len(refusal_events) == 1, (
            f"expected exactly one landscape_page_refused event; got {state.events}"
        )
        assert refusal_events[0].page_num == 1

        sidecar_path = tmp_path / pdf_path.stem / "pages" / "00001.json"
        assert sidecar_path.exists(), f"sidecar missing at {sidecar_path}"
        sidecar = json.loads(sidecar_path.read_text())
        sidecar_kinds = [ev["kind"] for ev in sidecar["audit_events"]]
        assert "landscape_page_refused" in sidecar_kinds

        assert result.status == DocumentStatus.AUDIT_FAILED, (
            f"expected AUDIT_FAILED; got {result.status}"
        )

        # The frozen winning output for the page is the native-prose fallback:
        # needs_ocr_enhancement + a rejected attempt routes _winning_page_output
        # to the WARNING/audit_passed=False native-text branch (manifest.py),
        # not the rejected OCR attempt itself (ps.best_output stays the rejected
        # decision.final_output -- the fallback only wins at freeze time).
        assert sidecar["status"] == PageStatus.WARNING.value, (
            f"expected sidecar status warning (native fallback); got {sidecar['status']}"
        )
        assert sidecar["engine"] == "native"
        winning_text = sidecar["winning_output"].get("text", "")
        for line in _PROSE_LINES:
            assert line in winning_text

        ps = state.pages[1]
        # No accepted OCR anywhere for this page.
        assert not any(
            att.status == PageStatus.SUCCESS and att.engine == "qwen" for att in ps.attempts
        )
        assert ps.best_output is not None and ps.best_output.audit_passed is False
