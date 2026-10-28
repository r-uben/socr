"""Gemini spending-cap circuit breaker and run stats."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.routing.gemini_quota import (
    PAID_DISABLED_SKIP_REASON,
    format_gemini_run_summary,
    gemini_attempt_failure_kind,
    gemini_empty_response_message,
    gemini_quota_failure_class,
    summarize_gemini_attempts,
)


def test_spending_cap_429_is_exhausted_not_transient_retry() -> None:
    msg = (
        "CLI exited 1: quota/rate-limit: HTTP 429 exceeded monthly spending cap "
        "for generativelanguage.googleapis.com"
    )
    assert gemini_quota_failure_class(msg) == "exhausted"


def test_plain_rate_limit_is_transient() -> None:
    assert gemini_quota_failure_class("quota/rate-limit: 429 Too Many Requests") == "transient"


def test_stop_empty_response_is_not_quota_or_safety() -> None:
    msg = (
        "Error processing /tmp/x/images/page_0001.png: Empty response: "
        "finish_reason=STOP, len(parts)=0"
    )
    assert gemini_empty_response_message(msg)
    assert gemini_quota_failure_class(msg) == ""
    attempt = PageOutput(
        page_num=1,
        text="",
        status=PageStatus.ERROR,
        engine="gemini",
        failure_mode=FailureMode.CLI_ERROR,
        error=msg,
        audit_passed=False,
    )
    assert gemini_attempt_failure_kind(attempt) == "empty_response"


def test_gemini_empty_exhausted_skips_further_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = PipelineConfig(agentic=True, quiet=True)
    pipe = UnifiedPipeline(cfg)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    from socr.core.document import DocumentHandle

    handle = DocumentHandle(path=pdf, page_count=1, page_count_known=True)
    state = DocumentState(handle=handle)
    state.pages[1] = MagicMock(native_text="native text")

    empty_err = "Empty response: finish_reason=STOP, len(parts)=0"
    engine = MagicMock()
    engine.name = "gemini"
    engine.is_available.return_value = True
    calls = {"n": 0}

    def _process_pages(**_kwargs):
        calls["n"] += 1
        return [
            MagicMock(
                page_num=1,
                status=PageStatus.ERROR,
                failure_mode=FailureMode.EMPTY_RESPONSE,
                error=empty_err,
                text="",
                skip_reason="",
            )
        ]

    engine.process_pages.side_effect = _process_pages
    monkeypatch.setattr("socr.pipeline.orchestrator.get_engine", lambda _et: engine)

    pipe._run_engine_on_pages(state, [1], [], EngineType.GEMINI, "agentic")
    pipe._run_engine_on_pages(state, [1], [], EngineType.GEMINI, "agentic")
    pipe._run_engine_on_pages(state, [1], [], EngineType.GEMINI, "agentic")

    assert calls["n"] == 2
    assert 1 in pipe._gemini_empty_exhausted_pages


def test_gemini_quota_circuit_stops_after_spending_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = PipelineConfig(agentic=True)
    pipe = UnifiedPipeline(cfg)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    from socr.core.document import DocumentHandle

    handle = DocumentHandle(path=pdf, page_count=2, page_count_known=True)
    state = DocumentState(handle=handle)
    for n in (1, 2):
        state.pages[n] = MagicMock(native_text="native text")

    cap_err = "CLI exited 1: quota/rate-limit: 429 exceeded monthly spending cap for project"
    engine = MagicMock()
    engine.name = "gemini"
    engine.is_available.return_value = True
    calls = {"n": 0}

    def _process_pages(**_kwargs):
        calls["n"] += 1
        page_num = _kwargs["page_nums"][0]
        return [
            MagicMock(
                page_num=page_num,
                status=PageStatus.ERROR,
                failure_mode=FailureMode.CLI_ERROR,
                error=cap_err,
                text="",
                skip_reason="",
            )
        ]

    engine.process_pages.side_effect = _process_pages
    monkeypatch.setattr("socr.pipeline.orchestrator.get_engine", lambda _et: engine)
    monkeypatch.setattr(
        "socr.routing.gemini_quota.GEMINI_TRANSIENT_BACKOFF_SEC",
        0.0,
    )

    out1 = pipe._run_engine_on_pages(state, [1], [], EngineType.GEMINI, "agentic")
    out2 = pipe._run_engine_on_pages(state, [2], [], EngineType.GEMINI, "agentic")

    assert calls["n"] == 1
    assert pipe._gemini_quota_circuit_open is True
    assert out1[0].failure_mode is FailureMode.CLI_ERROR
    assert out2[0].skip_reason == PAID_DISABLED_SKIP_REASON
    assert state.paid_disabled is not None
    assert state.paid_disabled["reason"] == "quota_exhausted"
    assert out2[0].error == PAID_DISABLED_SKIP_REASON


def test_summarize_gemini_attempts_headline_and_buckets() -> None:
    from socr.core.document import DocumentHandle
    from socr.core.result import PageOutput

    handle = DocumentHandle(path=Path("x.pdf"), page_count=1, page_count_known=True)
    state = DocumentState(handle=handle)
    state.pages[1].attempts = [
        PageOutput(
            page_num=1,
            text="ok",
            status=PageStatus.SUCCESS,
            engine="gemini",
            audit_passed=True,
        ),
        PageOutput(
            page_num=1,
            text="",
            status=PageStatus.ERROR,
            engine="gemini",
            failure_mode=FailureMode.CLI_ERROR,
            error="quota/rate-limit: 429 exceeded monthly spending cap",
            audit_passed=False,
        ),
        PageOutput(
            page_num=1,
            text="",
            status=PageStatus.ERROR,
            engine="gemini",
            failure_mode=FailureMode.EMPTY_RESPONSE,
            audit_passed=False,
        ),
        PageOutput(
            page_num=1,
            text="words",
            status=PageStatus.SUCCESS,
            engine="gemini",
            audit_passed=False,
            skip_reason="heuristics failed",
        ),
    ]
    stats = summarize_gemini_attempts(state)
    assert stats["gemini_pages_accepted"] == 1
    assert stats["gemini_call_attempts"] == 4
    assert stats["gemini_pages"] == 4
    assert stats["gemini_failures_quota"] == 1
    assert stats["gemini_failures_empty_response"] == 1
    assert stats["gemini_failures_judge_rejected"] == 1
    line = format_gemini_run_summary(stats)
    assert "1 accepted / 4 calls" in line
    assert "quota" in line
    assert "empty_response" in line
