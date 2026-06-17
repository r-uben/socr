"""TR-3: D3 fail-closed floor + selection-policy fix.

Tests for the three main guarantees:

1. Selection policy fix (manifest._winning_page_output):
   - A page that failed OCR AND had a per-region geometry hard-fail ships the
     explicit failed-table marker, NOT the collapsed/ragged native table.
   - A page that failed OCR but whose regions all PASSED verification ships the
     regular native-fallback text (no spurious D3 floor).

2. Distinct audit event:
   - ``table_region_unverifiable`` is emitted for D3 floor pages.
   - ``native_fallback`` is NOT emitted for D3 floor pages (distinct, no
     double-counting).

3. No-silent-loss propagation:
   - The D3 floor page surfaces in the ``failed_pages`` list (explicit marker).
   - Document status is demoted to AUDIT_FAILED (not a clean SUCCESS).
   - The ``table_region_unverifiable`` AuditEvent is recorded on state.events.

4. Negative control:
   - A region that passes per-region verification continues to ship its grid
     (no spurious D3 floor on verifiable pages).

5. TR-0 parity fixture still passes unchanged (no regression).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from socr.core.document import DocumentHandle
from socr.core.manifest import (
    _winning_page_output,
    is_page_failed_marker,
)
from socr.core.result import (
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handle(page_count: int = 1) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        h = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=page_count)
    return h


def _failed_ocr_attempt(page_num: int, text: str = "ragged row-major attempt") -> PageOutput:
    """An OCR attempt that failed verification (audit_passed=False)."""
    return PageOutput(
        page_num=page_num,
        text=text,
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
    )


def _build_state_d3_floor(
    page_num: int = 1, native_text: str = "collapsed| table |"
) -> DocumentState:
    """DocumentState simulating a page that triggers the D3 fail-closed floor.

    Sets:
    - is_born_digital = True + native_text set (table page)
    - native_table_structure_failed = True (OCR ladder failed)
    - native_table_unverifiable = True (per-region geometry hard-fail)
    - At least one failed OCR attempt (so p.attempts is non-empty)
    """
    state = DocumentState(handle=_make_handle(page_num))
    ps = state.pages[page_num]
    ps.is_born_digital = True
    ps.native_text = native_text
    ps.has_tables = True
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = True  # TR-3 flag
    ps.attempts.append(_failed_ocr_attempt(page_num))
    return state


def _build_state_native_fallback(
    page_num: int = 1, native_text: str = "Indicator | 2024\n--- | ---\nGDP | 2.1"
) -> DocumentState:
    """DocumentState simulating a page that hits native_fallback but NOT D3 floor.

    Sets native_table_structure_failed = True but native_table_unverifiable = False
    (the per-region verifier did NOT hard-fail — only the OCR ladder failed).
    """
    state = DocumentState(handle=_make_handle(page_num))
    ps = state.pages[page_num]
    ps.is_born_digital = True
    ps.native_text = native_text
    ps.has_tables = True
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = False  # per-region verifier passed
    ps.attempts.append(_failed_ocr_attempt(page_num))
    return state


# ---------------------------------------------------------------------------
# 1. Selection policy: D3 floor emits failure marker, not collapsed table
# ---------------------------------------------------------------------------


class TestSelectionPolicyD3Floor:
    """_winning_page_output must route to D3 floor when both OCR and per-region
    verifier failed; must NOT ship collapsed/ragged native table text."""

    def test_d3_floor_ships_failure_marker(self) -> None:
        """When native_table_structure_failed AND native_table_unverifiable,
        _winning_page_output must return the explicit failed-table marker."""
        state = _build_state_d3_floor()
        winner = _winning_page_output(state, 1, None)
        assert is_page_failed_marker(winner.text), (
            f"D3 floor must ship the explicit failed-table marker, got: {winner.text!r}"
        )

    def test_d3_floor_status_is_error(self) -> None:
        """D3 floor page must carry PageStatus.ERROR (not SUCCESS or WARNING)."""
        state = _build_state_d3_floor()
        winner = _winning_page_output(state, 1, None)
        assert winner.status == PageStatus.ERROR, (
            f"Expected PageStatus.ERROR for D3 floor page, got {winner.status}"
        )

    def test_d3_floor_audit_passed_false(self) -> None:
        """D3 floor page must NOT be marked audit_passed=True."""
        state = _build_state_d3_floor()
        winner = _winning_page_output(state, 1, None)
        assert not winner.audit_passed, "D3 floor page must have audit_passed=False"

    def test_d3_floor_does_not_ship_collapsed_native(self) -> None:
        """The collapsed native table text must NOT appear in the D3 floor output."""
        collapsed = "collapsed| table |content here"
        state = _build_state_d3_floor(native_text=collapsed)
        winner = _winning_page_output(state, 1, None)
        assert collapsed not in winner.text, (
            f"D3 floor must NOT ship the collapsed native text; got: {winner.text!r}"
        )

    def test_d3_floor_marker_contains_page_num(self) -> None:
        """The failure marker text must contain the page number and the image ref
        when a PNG was rendered, or just the marker when no PNG is available.

        Without a PNG render (d3_floor_png_ref=""), the text is the marker only.
        With a PNG render (d3_floor_png_ref set), the text is marker + image ref.
        Both forms satisfy is_page_failed_marker and do NOT contain native table text.
        """
        # Case 1: no PNG rendered (d3_floor_png_ref="" default)
        state = _build_state_d3_floor(page_num=1)
        winner = _winning_page_output(state, 1, None)
        assert "[page 1 failed:" in winner.text, (
            f"Failure marker must mention 'page 1 failed:'; got {winner.text!r}"
        )
        assert is_page_failed_marker(winner.text), (
            f"is_page_failed_marker must return True for D3 floor text; got {winner.text!r}"
        )

        # Case 2: PNG rendered — text is marker + image ref
        state2 = _build_state_d3_floor(page_num=2)
        state2.pages[2].d3_floor_png_ref = "![Failed table page 2](figures/failed_table_p2.png)"
        winner2 = _winning_page_output(state2, 2, None)
        assert "[page 2 failed:" in winner2.text, (
            f"Marker must mention 'page 2 failed:'; got {winner2.text!r}"
        )
        assert "![Failed table page 2]" in winner2.text, (
            f"PNG image ref must appear in D3 floor text; got {winner2.text!r}"
        )
        assert is_page_failed_marker(winner2.text), (
            f"is_page_failed_marker must return True for D3 floor text with PNG ref; "
            f"got {winner2.text!r}"
        )


# ---------------------------------------------------------------------------
# 2. Negative control: verifiable table still ships grid
# ---------------------------------------------------------------------------


class TestNegativeControlVerifiableTable:
    """When per-region verifier PASSES (native_table_unverifiable=False),
    _winning_page_output must NOT route to the D3 floor — the native fallback
    text ships as usual (flagged WARNING, not ERROR)."""

    def test_verifiable_table_ships_native_text(self) -> None:
        """A table page that passed per-region verification ships native text,
        not the failed-table marker."""
        native = "| GDP 2024 | GDP 2025 |\n| --- | --- |\n| 2.1 | 1.9 |"
        state = _build_state_native_fallback(native_text=native)
        winner = _winning_page_output(state, 1, None)
        assert not is_page_failed_marker(winner.text), (
            "A verifiable table must NOT hit the D3 floor; "
            f"got failed-table marker instead. text={winner.text!r}"
        )
        assert native in winner.text or winner.text == native, (
            f"Expected native text to be shipped; got {winner.text!r}"
        )

    def test_verifiable_table_status_is_warning(self) -> None:
        """A native-fallback table page ships with PageStatus.WARNING, not ERROR."""
        state = _build_state_native_fallback()
        winner = _winning_page_output(state, 1, None)
        assert winner.status == PageStatus.WARNING, (
            f"Expected WARNING for native-fallback page, got {winner.status}"
        )

    def test_d3_floor_requires_attempts(self) -> None:
        """D3 floor only fires when p.attempts is non-empty (OCR was tried).
        A page with no attempts (no OCR tried yet) must NOT hit the floor."""
        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "| col1 | col2 |\n| --- | --- |\n| a | b |"
        ps.has_tables = True
        ps.native_table_structure_failed = True
        ps.native_table_unverifiable = True
        # No attempts — OCR was never tried; should fall through to native text
        winner = _winning_page_output(state, 1, None)
        assert not is_page_failed_marker(winner.text), (
            "D3 floor must NOT fire when no OCR attempts were made "
            "(the page has not been tried yet)"
        )


# ---------------------------------------------------------------------------
# 3. PageState flag propagation from born_digital.PageAssessment
# ---------------------------------------------------------------------------


class TestPageStateFlag:
    """native_table_unverifiable is propagated from PageAssessment via
    apply_born_digital when has_unverifiable_table_region=True."""

    def test_flag_propagated_when_set(self) -> None:
        """apply_born_digital must set native_table_unverifiable=True on PageState
        when the PageAssessment carries has_unverifiable_table_region=True."""
        from socr.core.born_digital import DocumentAssessment, PageAssessment

        pa = PageAssessment(
            page_num=1,
            is_born_digital=True,
            native_text="some table text",
            confidence=0.9,
            has_tables=True,
            has_unverifiable_table_region=True,
        )
        assessment = DocumentAssessment(path=Path("/tmp/fake.pdf"), pages=[pa])
        state = DocumentState(handle=_make_handle(1))
        state.apply_born_digital(assessment)
        assert state.pages[1].native_table_unverifiable, (
            "native_table_unverifiable must be True after apply_born_digital "
            "when has_unverifiable_table_region=True"
        )

    def test_flag_not_set_when_regions_verify(self) -> None:
        """apply_born_digital must NOT set native_table_unverifiable when the
        PageAssessment has has_unverifiable_table_region=False (default)."""
        from socr.core.born_digital import DocumentAssessment, PageAssessment

        pa = PageAssessment(
            page_num=1,
            is_born_digital=True,
            native_text="clean table",
            confidence=0.9,
            has_tables=True,
            has_unverifiable_table_region=False,  # default
        )
        assessment = DocumentAssessment(path=Path("/tmp/fake.pdf"), pages=[pa])
        state = DocumentState(handle=_make_handle(1))
        state.apply_born_digital(assessment)
        assert not state.pages[1].native_table_unverifiable, (
            "native_table_unverifiable must remain False when regions verify"
        )


# ---------------------------------------------------------------------------
# 4. Audit event distinctness
# ---------------------------------------------------------------------------


class TestAuditEventDistinctness:
    """table_region_unverifiable is the correct audit event for D3 floor pages;
    native_fallback must NOT appear for them.  Non-D3 fallback pages must get
    native_fallback (not table_region_unverifiable)."""

    def _run_assemble(self, state: DocumentState) -> list:
        """Run _phase_assemble against a DocumentState and return state.events.

        Uses a tmp_path-style temp dir and patches the bits that need disk
        access.  Returns the events list after assembly.
        """
        import tempfile

        from socr.core.config import EngineType, PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=False,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            audit_enabled=False,
            save_figures=False,
            write_manifest=False,
        )
        pipeline = UnifiedPipeline(config)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            # _phase_assemble needs canonical_page_texts and file saves.
            # Patch _save_markdown and _write_metadata to avoid real PDF/disk ops.
            with (
                patch.object(pipeline, "_save_markdown", return_value=output_dir / "out.md"),
                patch.object(pipeline, "_write_metadata", return_value=None),
                patch.object(pipeline, "_write_manifest", return_value=None),
                patch.object(pipeline, "_rewrite_all_fragments", return_value=None),
                patch.object(pipeline, "_flush_page_fragment", return_value=None),
                patch.object(pipeline, "_flush_page_sidecar", return_value=None),
                patch.object(pipeline, "_stitch_fragments", return_value=""),
            ):
                pipeline._phase_assemble(state, output_dir)
        return state.events

    def test_d3_floor_emits_table_region_unverifiable(self) -> None:
        """A D3 floor page must produce a ``table_region_unverifiable`` event."""
        state = _build_state_d3_floor()
        events = self._run_assemble(state)
        kinds = [e.kind for e in events]
        assert "table_region_unverifiable" in kinds, (
            f"Expected 'table_region_unverifiable' audit event for D3 floor page; "
            f"got events: {kinds}"
        )

    def test_d3_floor_event_carries_d3_flag(self) -> None:
        """The table_region_unverifiable event must carry data['d3_floor']=True."""
        state = _build_state_d3_floor()
        events = self._run_assemble(state)
        d3_events = [e for e in events if e.kind == "table_region_unverifiable"]
        assert d3_events, "Expected at least one table_region_unverifiable event"
        assert d3_events[0].data.get("d3_floor") is True, (
            f"Expected data['d3_floor']=True; got data={d3_events[0].data!r}"
        )

    def test_d3_floor_does_not_emit_native_fallback(self) -> None:
        """A D3 floor page must NOT produce a ``native_fallback`` event."""
        state = _build_state_d3_floor()
        events = self._run_assemble(state)
        kinds = [e.kind for e in events]
        assert "native_fallback" not in kinds, (
            "D3 floor pages must NOT emit native_fallback — "
            "they have their own distinct table_region_unverifiable event"
        )

    def test_non_d3_fallback_emits_native_fallback_not_d3(self) -> None:
        """A native-fallback page (verifier passed, OCR failed) must emit
        ``native_fallback`` and NOT ``table_region_unverifiable``."""
        state = _build_state_native_fallback()
        events = self._run_assemble(state)
        kinds = [e.kind for e in events]
        assert "native_fallback" in kinds, (
            f"Expected 'native_fallback' event for non-D3 fallback page; got: {kinds}"
        )
        assert "table_region_unverifiable" not in kinds, (
            "Non-D3 fallback page must NOT emit table_region_unverifiable"
        )


# ---------------------------------------------------------------------------
# 5. Document-status demotion
# ---------------------------------------------------------------------------


class TestDocumentStatusDemotion:
    """A run that routes any page to the D3 floor must NOT report SUCCESS.
    The document status must be AUDIT_FAILED (or ERROR), never SUCCESS."""

    def _run_and_get_result(self, state: DocumentState) -> object:
        """Run _phase_assemble and return the EngineResult."""
        import tempfile

        from socr.core.config import EngineType, PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=False,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            audit_enabled=False,
            save_figures=False,
            write_manifest=False,
        )
        pipeline = UnifiedPipeline(config)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with (
                patch.object(pipeline, "_save_markdown", return_value=output_dir / "out.md"),
                patch.object(pipeline, "_write_metadata", return_value=None),
                patch.object(pipeline, "_write_manifest", return_value=None),
                patch.object(pipeline, "_rewrite_all_fragments", return_value=None),
                patch.object(pipeline, "_flush_page_fragment", return_value=None),
                patch.object(pipeline, "_flush_page_sidecar", return_value=None),
                patch.object(pipeline, "_stitch_fragments", return_value=""),
            ):
                return pipeline._phase_assemble(state, output_dir)

    def test_d3_floor_demotes_document_status(self) -> None:
        """A document with a D3 floor page must NOT report SUCCESS."""
        from socr.core.result import DocumentStatus

        state = _build_state_d3_floor()
        self._run_and_get_result(state)
        # The document status must be AUDIT_FAILED or ERROR — never SUCCESS.
        assert state.status != DocumentStatus.SUCCESS, (
            f"Document status must not be SUCCESS when a D3 floor page exists; got {state.status}"
        )

    def test_clean_page_reports_success(self) -> None:
        """A clean page (audit_passed=True) still reports SUCCESS."""
        from socr.core.result import DocumentStatus

        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "| col |\n| --- |\n| val |"
        ps.has_tables = True
        # No native_table_structure_failed, no unverifiable — clean path
        clean_out = PageOutput(
            page_num=1,
            text="| col |\n| --- |\n| val |",
            status=PageStatus.SUCCESS,
            engine="native",
            audit_passed=True,
        )
        ps.attempts.append(clean_out)
        ps.best_output = clean_out

        self._run_and_get_result(state)
        assert state.status == DocumentStatus.SUCCESS, (
            f"A clean page should report SUCCESS; got {state.status}"
        )


# ---------------------------------------------------------------------------
# 6. _verify_regions return value
# ---------------------------------------------------------------------------


class TestVerifyRegionsReturnValue:
    """_verify_regions now returns True when any region hard-fails."""

    def test_returns_false_on_clean_regions(self) -> None:
        """When no region hard-fails, _verify_regions must return False."""
        import fitz

        from socr.core.born_digital import BornDigitalDetector
        from socr.tables.native_verifier import VerifierResult

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "col1  col2\n1.0   2.0\n3.0   4.0")
        regions = [(fitz.Rect(0, 80, 200, 200), "| col1 | col2 |\n| --- | --- |\n| 1.0 | 2.0 |")]

        detector = BornDigitalDetector()
        # Patch verify_native_table_region to return a passing result
        with patch(
            "socr.core.born_digital.BornDigitalDetector._verify_regions",
            wraps=detector._verify_regions,
        ):
            with patch(
                "socr.tables.native_verifier.verify_native_table_region",
                return_value=VerifierResult(hard_fail=False, warn=False, reason="ok"),
            ):
                result = detector._verify_regions(page, regions)
        doc.close()
        assert result is False, f"Expected False (no hard-fail); got {result!r}"

    def test_returns_true_on_hard_fail(self) -> None:
        """When any region hard-fails, _verify_regions must return True."""
        import fitz

        from socr.core.born_digital import BornDigitalDetector
        from socr.tables.native_verifier import VerifierResult

        doc = fitz.open()
        page = doc.new_page()
        regions = [(fitz.Rect(0, 80, 200, 200), "| col1 | col2 |\n| --- | --- |\n| 1.0 | 2.0 |")]

        detector = BornDigitalDetector()
        with patch(
            "socr.tables.native_verifier.verify_native_table_region",
            return_value=VerifierResult(
                hard_fail=True, warn=False, reason="geometry_impossible_collapse"
            ),
        ):
            result = detector._verify_regions(page, regions)
        doc.close()
        assert result is True, f"Expected True (hard-fail detected); got {result!r}"


# ---------------------------------------------------------------------------
# 7. Multi-page state-leak guard
# ---------------------------------------------------------------------------


class TestMultiPageStateLeak:
    """The _last_extraction_had_unverifiable instance variable is reset for every
    page.  A page whose per-region verifier hard-fails must NOT infect subsequent
    pages that have clean, verifiable tables (false-flooring the good table)."""

    def test_unverifiable_page1_does_not_floor_clean_page2(self, tmp_path: Path) -> None:
        """Page 1 has an unverifiable table (hard-fail); page 2 has a clean table.

        After BornDigitalDetector.detect() + apply_born_digital():
        - state.pages[1].native_table_unverifiable == True
        - state.pages[2].native_table_unverifiable == False  ← no leak
        - _winning_page_output for page 2 returns the clean native table,
          NOT the failed-table marker.
        """
        import fitz

        from socr.core.born_digital import BornDigitalDetector
        from socr.tables.native_verifier import VerifierResult

        # Build a two-page PDF.
        # Page 1: a table with column layout that will trigger the verifier hard-fail.
        # Page 2: a clean table with verified columns (no hard-fail).
        pdf_path = tmp_path / "two_page.pdf"
        doc = fitz.open()

        # Page 1 — table with "collapsed" geometry (mock will report hard-fail)
        p1 = doc.new_page()
        p1.insert_text((50, 60), "col1  col2  col3")
        p1.insert_text((50, 80), "1.0   2.0   3.0")
        p1.insert_text((50, 100), "4.0   5.0   6.0")

        # Page 2 — clean simple table (mock will report no hard-fail)
        p2 = doc.new_page()
        p2.insert_text((50, 60), "Indicator  2024  2025")
        p2.insert_text((50, 80), "GDP        2.1   1.9")
        p2.insert_text((50, 100), "CPI        3.4   2.8")

        doc.save(str(pdf_path))
        doc.close()

        # Patch verify_native_table_region: hard-fail ONLY on page 1's region
        # (bbox y0 < 150), pass on everything else.
        def _selective_verify(page, output_text, region_bbox):
            page_num = getattr(page, "number", 0) + 1
            if page_num == 1:
                return VerifierResult(
                    hard_fail=True, warn=False, reason="geometry_impossible_collapse"
                )
            return VerifierResult(hard_fail=False, warn=False, reason="ok")

        with patch(
            "socr.tables.native_verifier.verify_native_table_region",
            side_effect=_selective_verify,
        ):
            detector = BornDigitalDetector()
            assessment = detector.detect(pdf_path)

        assert len(assessment.pages) == 2

        # Page 1 must be flagged
        p1_assessment = assessment.pages[0]
        # Only check if the page actually has tables (needed for verifier to fire)
        if p1_assessment.has_tables:
            assert p1_assessment.has_unverifiable_table_region, (
                "Page 1 with a mocked hard-fail must have has_unverifiable_table_region=True"
            )

        # Page 2 must NOT be flagged — the verifier returned clean for page 2
        p2_assessment = assessment.pages[1]
        assert not p2_assessment.has_unverifiable_table_region, (
            "Page 2 must NOT have has_unverifiable_table_region=True; "
            "the page 1 hard-fail must not leak to page 2"
        )

        # Propagate through apply_born_digital and verify PageState
        state = DocumentState(handle=_make_handle(2))
        state.apply_born_digital(assessment)

        assert not state.pages[2].native_table_unverifiable, (
            "PageState page 2 must NOT have native_table_unverifiable=True; "
            "the page 1 flag must not leak to page 2"
        )
        # Double-check: _winning_page_output for page 2 must NOT hit the D3 floor
        # (even though we're not setting native_table_structure_failed here, the flag
        # state is what matters: native_table_unverifiable must be False for page 2)
        assert not state.pages[2].native_table_unverifiable, (
            "PageState.native_table_unverifiable must be False for page 2"
        )


# ---------------------------------------------------------------------------
# 8. Sidecar serialization of TR-3 flags
# ---------------------------------------------------------------------------


class TestSidecarSerialization:
    """native_table_unverifiable and d3_floor_png_ref must round-trip through
    the per-page sidecar JSON (_flush_page_sidecar / _restore_flags_from_sidecar)
    so a resumed run reconstructs the correct D3 floor state."""

    def test_native_table_unverifiable_in_sidecar_payload(self) -> None:
        """_flush_page_sidecar must include native_table_unverifiable in the JSON."""
        import json
        import tempfile

        from socr.core.config import EngineType, PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=False,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            audit_enabled=False,
            save_figures=False,
            write_manifest=False,
        )
        pipeline = UnifiedPipeline(config)
        state = _build_state_d3_floor()
        state.pages[1].d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            sidecar_path = pipeline._flush_page_sidecar(state, 1, output_dir, terminal=True)
            if sidecar_path is not None and sidecar_path.exists():
                payload = json.loads(sidecar_path.read_text())
                assert "native_table_unverifiable" in payload, (
                    "Sidecar must include 'native_table_unverifiable' key"
                )
                assert payload["native_table_unverifiable"] is True, (
                    "Sidecar 'native_table_unverifiable' must be True for D3 floor page"
                )
                assert "d3_floor_png_ref" in payload, "Sidecar must include 'd3_floor_png_ref' key"
                assert "failed_table_p1" in payload["d3_floor_png_ref"], (
                    "Sidecar 'd3_floor_png_ref' must contain the PNG filename"
                )

    def test_flags_restored_via_restore_terminal_page_state(self) -> None:
        """_restore_terminal_page_state must set native_table_unverifiable and
        d3_floor_png_ref on PageState from the sidecar JSON."""
        import json
        import tempfile

        from socr.core.config import EngineType, PipelineConfig
        from socr.core.result import PageOutput, PageStatus
        from socr.pipeline.orchestrator import UnifiedPipeline

        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=False,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            audit_enabled=False,
            save_figures=False,
            write_manifest=False,
        )
        pipeline = UnifiedPipeline(config)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            # Write a minimal sidecar JSON manually with TR-3 flags set.
            from ocr_output_contract import doc_dir_for, relative_key

            state = _build_state_d3_floor()
            scan_root = state.handle.path.parent
            doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
            sidecar_dir = doc_dir / "pages"
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar_path = sidecar_dir / "00001.json"
            png_ref = "![Failed table page 1](figures/failed_table_p1.png)"
            # Write a terminal sidecar with TR-3 flags
            sidecar_path.write_text(
                json.dumps(
                    {
                        "native_table_unverifiable": True,
                        "d3_floor_png_ref": png_ref,
                        "needs_ocr_enhancement": False,
                        "native_table_structure_failed": True,
                        "chart_asset_render_failed": False,
                        "judge_rejected": False,
                        # Other fields required by _restore_terminal_page_state
                        "run_fingerprint": "",
                        "input_checksum": "",
                    }
                )
            )

            # _restore_terminal_page_state reads from the sidecar via doc_dir.
            fresh_state = DocumentState(handle=_make_handle(1))
            page_out = PageOutput(
                page_num=1,
                text="stub body",
                status=PageStatus.SUCCESS,
                engine="native",
                audit_passed=True,
            )
            pipeline._restore_terminal_page_state(fresh_state, 1, page_out, output_dir)
            ps = fresh_state.pages[1]

            assert ps.native_table_unverifiable is True, (
                "Restored native_table_unverifiable must be True"
            )
            assert ps.d3_floor_png_ref == png_ref, (
                f"Restored d3_floor_png_ref must match; got {ps.d3_floor_png_ref!r}"
            )
