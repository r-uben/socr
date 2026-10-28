"""Agentic ladder must not escalate source-evidence rejects to paid rungs."""

from __future__ import annotations

from unittest.mock import MagicMock

from socr.core.config import EngineType
from socr.core.providers import PROFILE_GEMINI, PROFILE_MARKER, PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision, route_page


def test_source_evidence_reject_skips_gemini() -> None:
    ladder = [PROFILE_QWEN_LOCAL, PROFILE_MARKER, PROFILE_GEMINI]

    def run_provider(prof, page_num: int) -> PageOutput:
        return PageOutput(
            page_num=page_num,
            text="| a | b |\n|---|---|\n| 1 | 2 |",
            status=PageStatus.SUCCESS,
            engine=prof.engine.value,
        )

    judge = MagicMock()
    judge.assess.side_effect = [
        AcceptDecision(
            accept=False,
            reason="source_evidence_table: numeric tokens unsupported by page evidence",
        ),
        AcceptDecision(accept=False, reason="empty/error output"),
        AcceptDecision(accept=True, reason="should not run"),
    ]

    decision = route_page(1, ladder, run_provider, judge)
    assert judge.assess.call_count == 2
    gemini_attempts = [a for a in decision.attempts if a.engine is EngineType.GEMINI]
    assert len(gemini_attempts) == 1
    assert "paid escalation blocked" in (gemini_attempts[0].reason or "")
