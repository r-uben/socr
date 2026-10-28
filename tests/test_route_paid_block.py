"""Agentic ladder must not escalate source-evidence rejects to paid rungs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from socr.core.config import EngineType
from socr.core.providers import PROFILE_GEMINI, PROFILE_MARKER, PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision, route_page


def test_no_witness_source_evidence_does_not_block_paid() -> None:
    from socr.routing.cascade_policy import blocks_paid_escalation

    assert blocks_paid_escalation(
        "source_evidence_table: [no_witness_backend] no classical OCR witness read this page"
    ) is False
    assert blocks_paid_escalation(
        "source_evidence_table: numeric tokens unsupported by page evidence: ['1']"
    ) is True


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


def test_source_evidence_reject_ladder_not_accepted() -> None:
    """Paid block: ladder must not finish verified/accepted on Gemini."""
    ladder = [PROFILE_QWEN_LOCAL, PROFILE_MARKER, PROFILE_GEMINI]

    def run_provider(prof, page_num: int) -> PageOutput:
        return PageOutput(
            page_num=page_num,
            text="| a | b |\n|---|---|\n| 1 | 2 |",
            status=PageStatus.SUCCESS,
            engine=prof.engine.value,
            audit_passed=True,
        )

    judge = MagicMock()
    judge.assess.side_effect = [
        AcceptDecision(
            accept=False,
            reason="source_evidence_table: numeric tokens unsupported by page evidence",
        ),
        AcceptDecision(accept=False, reason="empty/error output"),
        AcceptDecision(accept=True, reason="must not run"),
    ]

    decision = route_page(1, ladder, run_provider, judge)
    assert decision.accepted is False
    assert decision.final_output.engine != EngineType.GEMINI.value
    assert not any(a.accepted for a in decision.attempts)


def test_scanned_source_evidence_floor_not_verified_success() -> None:
    """Fail-closed: scanned source-evidence floor ships a failure marker, not clean SUCCESS."""
    from socr.core.manifest import _winning_page_output, is_page_failed_marker
    from socr.core.result import FailureMode
    from socr.core.state import DocumentState
    from socr.core.document import DocumentHandle

    handle = DocumentHandle(path=Path("x.pdf"), page_count=1, page_count_known=True)
    state = DocumentState(handle=handle)
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.scanned_table_evidence_failed = True
    ps.best_output = PageOutput(
        page_num=1,
        text="[page 1 failed: unverifiable table — see image]\n\n",
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.HALLUCINATION,
    )
    winner = _winning_page_output(state, 1, None)
    assert winner.audit_passed is False
    assert winner.status is not PageStatus.SUCCESS
    assert is_page_failed_marker(winner.text)
