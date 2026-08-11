"""Unit tests for the page-lane modality router (native PDF vs OCR LLM)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from socr.pipeline.page_router import (
    PageLane,
    REASON_CHART_MARKS,
    REASON_HAS_TABLES,
    REASON_NATIVE_ONLY,
    REASON_NATIVE_TRUSTED,
    REASON_NEEDS_ENHANCEMENT,
    REASON_NO_NATIVE_FIRST,
    REASON_NOT_BORN_DIGITAL,
    decide_page_lane,
)


def _decide(
    *,
    native_first: bool = True,
    native_only: bool = False,
    is_born_digital: bool = True,
    native_text: str | None = "Clean prose about GDP.",
    needs_ocr_enhancement: bool = False,
    has_tables: bool = False,
    has_chart_marks: bool = False,
):
    return decide_page_lane(
        native_first=native_first,
        native_only=native_only,
        is_born_digital=is_born_digital,
        native_text=native_text,
        needs_ocr_enhancement=needs_ocr_enhancement,
        has_tables=has_tables,
        has_chart_marks=has_chart_marks,
    )


class TestDecidePageLaneNative:
    def test_clean_prose_ships_native(self) -> None:
        d = _decide()
        assert d.lane is PageLane.NATIVE
        assert d.reason == REASON_NATIVE_TRUSTED

    def test_chart_marks_on_trusted_native(self) -> None:
        d = _decide(has_chart_marks=True)
        assert d.lane is PageLane.CHART_ASSET
        assert d.reason == REASON_CHART_MARKS

    def test_native_only_trusts_enhancement_pages(self) -> None:
        d = _decide(native_only=True, needs_ocr_enhancement=True)
        assert d.lane is PageLane.NATIVE
        assert d.reason == REASON_NATIVE_ONLY

    def test_native_only_with_chart_marks(self) -> None:
        d = _decide(native_only=True, needs_ocr_enhancement=True, has_chart_marks=True)
        assert d.lane is PageLane.CHART_ASSET
        assert d.reason == REASON_CHART_MARKS


class TestDecidePageLaneOcr:
    def test_no_native_first_forces_ocr(self) -> None:
        d = _decide(native_first=False)
        assert d.lane is PageLane.OCR
        assert d.reason == REASON_NO_NATIVE_FIRST

    def test_scanned_page_forces_ocr(self) -> None:
        d = _decide(is_born_digital=False, native_text=None)
        assert d.lane is PageLane.OCR
        assert d.reason == REASON_NOT_BORN_DIGITAL

    def test_empty_native_text_forces_ocr(self) -> None:
        d = _decide(native_text="   ")
        assert d.lane is PageLane.OCR
        assert d.reason == REASON_NOT_BORN_DIGITAL

    def test_needs_enhancement_forces_ocr(self) -> None:
        d = _decide(needs_ocr_enhancement=True)
        assert d.lane is PageLane.OCR
        assert d.reason == REASON_NEEDS_ENHANCEMENT

    def test_tables_force_ocr(self) -> None:
        d = _decide(has_tables=True)
        assert d.lane is PageLane.OCR
        assert d.reason == REASON_HAS_TABLES

    def test_tables_beat_chart_marks(self) -> None:
        """Table pages stay on the OCR ladder even if chart marks are present."""
        d = _decide(has_tables=True, has_chart_marks=True)
        assert d.lane is PageLane.OCR
        assert d.reason == REASON_HAS_TABLES


class TestDecidePageLaneHelpers:
    def test_trusted_native_predicate(self) -> None:
        from socr.pipeline.page_router import is_native_bypass_lane

        assert is_native_bypass_lane(_decide()) is True
        assert is_native_bypass_lane(_decide(has_chart_marks=True)) is True
        assert is_native_bypass_lane(_decide(has_tables=True)) is False


class TestAgenticEmitsPageLaneAudit:
    """Agentic loop records modality decisions as page_lane audit events."""

    def test_native_page_emits_page_lane_event(self, tmp_path: Path) -> None:
        import fitz

        from socr.core.config import EngineType, PipelineConfig
        from socr.core.document import DocumentHandle
        from socr.core.providers import PROFILE_QWEN_LOCAL
        from socr.core.state import DocumentState, PageState
        from socr.pipeline.orchestrator import UnifiedPipeline

        pdf = tmp_path / "prose.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This is enough clean prose for a native page.")
        doc.save(pdf)
        doc.close()

        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=list(EngineType),
            agentic=True,
            quiet=True,
            native_first=True,
            write_manifest=False,
        )
        pipeline = UnifiedPipeline(config)
        handle = DocumentHandle(pdf)
        state = DocumentState(handle)
        state.pages[1] = PageState(
            page_num=1,
            is_born_digital=True,
            native_text="This is enough clean prose for a native page.",
            needs_ocr_enhancement=False,
            has_tables=False,
        )
        state.events = []

        with (
            patch.object(
                pipeline,
                "_available_engines_for_agentic",
                return_value=[PROFILE_QWEN_LOCAL],
            ),
            patch.object(pipeline, "_build_page_judge", return_value=MagicMock()),
            patch("socr.pipeline.orchestrator.route_page") as mock_route,
            patch.object(pipeline, "_flush_page_sidecar"),
            patch.object(pipeline, "_rewrite_all_fragments", create=True),
        ):
            mock_route.side_effect = AssertionError("OCR ladder must not run for native prose")
            pipeline._phase_agentic(state, tmp_path / "out")

        lanes = [e for e in state.events if getattr(e, "kind", None) == "page_lane"]
        assert len(lanes) == 1
        assert lanes[0].data["lane"] == "native"
        assert lanes[0].data["reason"] == REASON_NATIVE_TRUSTED
        assert state.pages[1].best_output is not None
        assert state.pages[1].best_output.engine == "native"
