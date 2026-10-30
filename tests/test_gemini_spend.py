"""Hermetic Gemini token spend, budget stop, and at-most-once tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from socr.core.config import DEFAULT_GEMINI_MODEL, EngineType, PipelineConfig
from socr.core.gemini_pricing import GeminiUsage, compute_gemini_usd, reference_page_cost_usd
from socr.core.gemini_spend import GeminiSpendTracker, attempt_cost_usd_for_profile
from socr.core.providers import PROFILE_GEMINI
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.routing.cascade_policy import parse_gemini_usage


def test_compute_gemini_usd_includes_thinking_tokens() -> None:
    usage = GeminiUsage(
        prompt_token_count=1000,
        candidates_token_count=200,
        thoughts_token_count=800,
        cached_content_token_count=0,
        total_token_count=2000,
        usage_metadata_present=True,
    )
    usd = compute_gemini_usd(DEFAULT_GEMINI_MODEL, usage)
    assert usd is not None
    # input 1000 @ $0.50/M + output (200+800) @ $3/M
    assert usd == pytest.approx(0.0005 + 0.003)


def test_compute_gemini_usd_missing_metadata_returns_none() -> None:
    usage = GeminiUsage(usage_metadata_present=False)
    assert compute_gemini_usd(DEFAULT_GEMINI_MODEL, usage) is None


def test_parse_gemini_usage_json_snippet() -> None:
    stderr = (
        'usageMetadata={"promptTokenCount": 10, "candidatesTokenCount": 2, '
        '"thoughtsTokenCount": 5, "cachedContentTokenCount": 1, "totalTokenCount": 18}'
    )
    usage = parse_gemini_usage(stderr, "")
    assert usage.usage_metadata_present is True
    assert usage.prompt_token_count == 10
    assert usage.thoughts_token_count == 5
    assert usage.cached_content_token_count == 1


def test_profile_gemini_cost_is_token_derived_not_flat_constant() -> None:
    assert PROFILE_GEMINI.cost_per_page_usd == pytest.approx(reference_page_cost_usd())
    assert PROFILE_GEMINI.cost_per_page_usd > 0.001


def test_spend_tracker_outcome_and_rejected() -> None:
    tracker = GeminiSpendTracker()
    usage = GeminiUsage(
        prompt_token_count=100,
        candidates_token_count=10,
        thoughts_token_count=40,
        usage_metadata_present=True,
    )
    call_id = tracker.record_call(
        document_key="/doc.pdf",
        page_num=1,
        engine="gemini",
        model=DEFAULT_GEMINI_MODEL,
        usage=usage,
        outcome="success",
    )
    tracker.set_outcome(call_id, "rejected")
    summary = tracker.summarize()
    assert summary["calls_by_outcome"]["rejected"] == 1
    assert summary["run_usd_total"] > 0


def test_attempt_cost_uses_page_output_tokens_for_gemini() -> None:
    po = PageOutput(
        page_num=1,
        text="ok",
        status=PageStatus.SUCCESS,
        engine="gemini",
        gemini_usage_metadata_present=True,
        cost_usd=0.042,
    )
    assert attempt_cost_usd_for_profile(PROFILE_GEMINI, po) == pytest.approx(0.042)


@pytest.fixture
def pipeline() -> UnifiedPipeline:
    cfg = PipelineConfig(max_paid_usd=0.001, max_paid_usd_pinned=True)
    pipe = UnifiedPipeline(cfg)
    pipe._reset_table_judge_rung_probes()
    return pipe


def _minimal_state(tmp_path: Path) -> DocumentState:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    from socr.core.document import DocumentHandle

    handle = DocumentHandle(path=pdf, page_count=2, page_count_known=True)
    state = DocumentState(handle=handle)
    for n in (1, 2):
        state.pages[n] = MagicMock(native_text="")
    return state


def _gemini_output(page_num: int, stderr: str) -> list[PageOutput]:
    usage = parse_gemini_usage(stderr, "")
    po = PageOutput(
        page_num=page_num,
        text="text",
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=True,
        llm_input_tokens=usage.prompt_token_count,
        llm_output_tokens=usage.billable_output_tokens,
        llm_thoughts_tokens=usage.thoughts_token_count,
        gemini_usage_metadata_present=usage.usage_metadata_present,
    )
    from socr.core.gemini_spend import apply_usage_to_page_output

    apply_usage_to_page_output(po, usage, model=DEFAULT_GEMINI_MODEL)
    return [po]


def test_max_paid_usd_blocks_second_call(pipeline: UnifiedPipeline, tmp_path: Path) -> None:
    state = _minimal_state(tmp_path)
    engine = MagicMock()
    engine.name = "gemini"
    engine.is_available.return_value = True
    expensive = "promptTokenCount: 5000\ncandidatesTokenCount: 1000\nthoughtsTokenCount: 9000\n"

    def _process(**kwargs):
        page_nums = kwargs["page_nums"]
        return _gemini_output(page_nums[0], expensive)

    engine.process_pages.side_effect = _process

    with patch("socr.pipeline.orchestrator.get_engine", lambda _et: engine):
        out1 = pipeline._run_engine_on_pages(
            state, [1], [], EngineType.GEMINI, "agentic", profile=None
        )
        out2 = pipeline._run_engine_on_pages(
            state, [2], [], EngineType.GEMINI, "agentic", profile=None
        )

    assert out1[0].status == PageStatus.SUCCESS
    assert out2[0].status == PageStatus.ERROR
    assert "max_paid_usd" in (out2[0].error or "")
    assert engine.process_pages.call_count == 1


def test_gemini_at_most_once_per_page(pipeline: UnifiedPipeline, tmp_path: Path) -> None:
    state = _minimal_state(tmp_path)
    engine = MagicMock()
    engine.name = "gemini"
    engine.is_available.return_value = True
    stderr = "promptTokenCount: 100\ncandidatesTokenCount: 10\nthoughtsTokenCount: 10\n"
    engine.process_pages.return_value = _gemini_output(1, stderr)

    with patch("socr.pipeline.orchestrator.get_engine", lambda _et: engine):
        first = pipeline._run_engine_on_pages(
            state, [1], [], EngineType.GEMINI, "agentic", profile=None
        )
        second = pipeline._run_engine_on_pages(
            state, [1], [], EngineType.GEMINI, "agentic", profile=None
        )

    assert first[0].status == PageStatus.SUCCESS
    assert second[0].status == PageStatus.ERROR
    assert "at_most_once" in (second[0].error or "")
    assert engine.process_pages.call_count == 1


def test_failed_gemini_call_still_recorded_with_usd(
    pipeline: UnifiedPipeline, tmp_path: Path
) -> None:
    state = _minimal_state(tmp_path)
    engine = MagicMock()
    engine.name = "gemini"
    engine.is_available.return_value = True
    stderr = "promptTokenCount: 400\ncandidatesTokenCount: 0\nthoughtsTokenCount: 600\n"
    engine.process_pages.return_value = [
        PageOutput(
            page_num=1,
            text="",
            status=PageStatus.ERROR,
            engine="gemini",
            failure_mode=FailureMode.EMPTY_RESPONSE,
            llm_input_tokens=400,
            llm_output_tokens=600,
            llm_thoughts_tokens=600,
            gemini_usage_metadata_present=True,
        )
    ]
    from socr.core.gemini_spend import apply_usage_to_page_output
    from socr.routing.cascade_policy import parse_gemini_usage

    usage = parse_gemini_usage(stderr, "")
    apply_usage_to_page_output(
        engine.process_pages.return_value[0], usage, model=DEFAULT_GEMINI_MODEL
    )

    with patch("socr.pipeline.orchestrator.get_engine", lambda _et: engine):
        pipeline._run_engine_on_pages(state, [1], [], EngineType.GEMINI, "agentic", profile=None)

    assert len(pipeline._gemini_spend.calls) == 1
    assert pipeline._gemini_spend.calls[0].outcome == "empty"
    assert pipeline._gemini_spend.calls[0].usd is not None
