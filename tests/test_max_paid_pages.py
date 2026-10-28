"""Hermetic tests for --max-paid-pages Gemini OCR cap."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.result import FailureMode, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline


@pytest.fixture
def pipeline() -> UnifiedPipeline:
    cfg = PipelineConfig(max_paid_pages=2, max_paid_pages_pinned=True)
    pipe = UnifiedPipeline(cfg)
    pipe._reset_table_judge_rung_probes()
    return pipe


def _minimal_state(tmp_path: Path) -> DocumentState:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    from socr.core.document import DocumentHandle

    handle = DocumentHandle(path=pdf, page_count=3, page_count_known=True)
    state = DocumentState(handle=handle)
    for n in (1, 2, 3):
        state.pages[n] = MagicMock(native_text="")
    return state


def test_max_paid_pages_stops_after_n_gemini_calls(
    pipeline: UnifiedPipeline, tmp_path: Path
) -> None:
    state = _minimal_state(tmp_path)
    engine = MagicMock()
    engine.name = "gemini"
    engine.is_available.return_value = True
    engine.process_pages.return_value = [
        MagicMock(
            page_num=1,
            status=PageStatus.SUCCESS,
            failure_mode=FailureMode.NONE,
            text="ok",
        )
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "socr.pipeline.orchestrator.get_engine",
            lambda _et: engine,
        )
        out1 = pipeline._run_engine_on_pages(
            state, [1], [], EngineType.GEMINI, "agentic", profile=None
        )
        out2 = pipeline._run_engine_on_pages(
            state, [2], [], EngineType.GEMINI, "agentic", profile=None
        )
        out3 = pipeline._run_engine_on_pages(
            state, [3], [], EngineType.GEMINI, "agentic", profile=None
        )

    assert out1[0].status == PageStatus.SUCCESS
    assert out2[0].status == PageStatus.SUCCESS
    assert out3[0].status == PageStatus.ERROR
    assert out3[0].failure_mode == FailureMode.MODEL_UNAVAILABLE
    assert "max_paid_pages" in (out3[0].error or "")
    assert engine.process_pages.call_count == 2


def test_paid_counter_resets_each_run(pipeline: UnifiedPipeline, tmp_path: Path) -> None:
    state = _minimal_state(tmp_path)
    engine = MagicMock()
    engine.name = "gemini"
    engine.is_available.return_value = True
    engine.process_pages.return_value = [
        MagicMock(
            page_num=1,
            status=PageStatus.SUCCESS,
            failure_mode=FailureMode.NONE,
            text="ok",
        )
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("socr.pipeline.orchestrator.get_engine", lambda _et: engine)
        pipeline._run_engine_on_pages(state, [1], [], EngineType.GEMINI, "agentic")
        pipeline._run_engine_on_pages(state, [2], [], EngineType.GEMINI, "agentic")
        pipeline._reset_table_judge_rung_probes()
        pipeline._run_engine_on_pages(state, [3], [], EngineType.GEMINI, "agentic")

    assert engine.process_pages.call_count == 3
