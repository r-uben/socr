"""Gemini CLI failures must surface actionable errors in manifest journal reasons."""

from __future__ import annotations

from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.pipeline.agentic import REASON_PROVIDER_TIMEOUT
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.routing.cascade_policy import extract_cli_actionable_error


def _journal_reason(stderr: str, *, returncode: int = 1) -> dict:
    detail = extract_cli_actionable_error(stderr, returncode=returncode)
    err = f"CLI exited {returncode}: {detail}"
    output = PageOutput(
        page_num=1,
        text="",
        status=PageStatus.ERROR,
        engine="gemini",
        failure_mode=FailureMode.CLI_ERROR,
        error=err,
        audit_passed=False,
        skip_reason="empty/error output",
        llm_input_tokens=1200,
        llm_output_tokens=0,
    )
    reason = UnifiedPipeline._skip_reason_with_provider_error(output)
    return {
        "engine": "gemini",
        "accepted": False,
        "failure_mode": output.failure_mode.value,
        "reason": reason,
        "llm_input_tokens": output.llm_input_tokens,
        "llm_output_tokens": output.llm_output_tokens,
    }


def test_mocked_429_journal_row() -> None:
    stderr = (
        "google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded for "
        "generativelanguage.googleapis.com"
    )
    row = _journal_reason(stderr)
    assert "quota/rate-limit" in row["reason"]
    assert "429" in row["reason"]
    assert row["llm_input_tokens"] == 1200


def test_mocked_safety_block_journal_row() -> None:
    stderr = (
        "Candidate was blocked due to SAFETY finish_reason=SAFETY "
        "harm_category=HARM_CATEGORY_DANGEROUS_CONTENT"
    )
    row = _journal_reason(stderr)
    assert "safety-block" in row["reason"]
    assert "SAFETY" in row["reason"] or "blocked" in row["reason"].lower()


def test_provider_timeout_reason_not_duplicated() -> None:
    output = PageOutput(
        page_num=1,
        text="",
        status=PageStatus.ERROR,
        engine="gemini",
        error="gemini: timed out after 240s",
        skip_reason=REASON_PROVIDER_TIMEOUT,
    )
    assert UnifiedPipeline._skip_reason_with_provider_error(output) == REASON_PROVIDER_TIMEOUT
