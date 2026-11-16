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

    assert (
        blocks_paid_escalation(
            "source_evidence_table: [no_witness_backend] no classical OCR witness read this page"
        )
        is False
    )
    assert (
        blocks_paid_escalation(
            "source_evidence_table: numeric tokens unsupported by page evidence: ['1']"
        )
        is True
    )


def test_no_witness_reject_reaches_gemini() -> None:
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
            reason=("source_evidence_table: [no_witness_backend] no classical OCR witness"),
        ),
        AcceptDecision(accept=False, reason="empty/error output"),
        AcceptDecision(accept=True, reason="gemini ran"),
    ]

    decision = route_page(1, ladder, run_provider, judge)
    assert judge.assess.call_count == 3
    assert any(a.engine is EngineType.GEMINI for a in decision.attempts)


def test_source_evidence_reject_skips_gemini_when_block_requested() -> None:
    """#1346: the old block is still reachable, but only by asking for it."""
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

    decision = route_page(1, ladder, run_provider, judge, block_paid_on_source_evidence=True)
    assert judge.assess.call_count == 2
    gemini_attempts = [a for a in decision.attempts if a.engine is EngineType.GEMINI]
    assert len(gemini_attempts) == 1
    assert "paid escalation blocked" in (gemini_attempts[0].reason or "")


def test_source_evidence_reject_ladder_not_accepted_when_block_requested() -> None:
    """Explicit paid block: ladder must not finish verified/accepted on Gemini."""
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

    decision = route_page(1, ladder, run_provider, judge, block_paid_on_source_evidence=True)
    assert decision.accepted is False
    assert decision.final_output.engine != EngineType.GEMINI.value
    assert not any(a.accepted for a in decision.attempts)


_WITNESS_REJECT = (
    "source_evidence_table: numeric tokens unsupported by page evidence: ['37.3', '17.5']"
)


def _table_provider(prof, page_num: int) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text="| a | b |\n|---|---|\n| 1 | 2 |",
        status=PageStatus.SUCCESS,
        engine=prof.engine.value,
    )


def _route(verdicts, **kwargs):
    judge = MagicMock()
    judge.assess.side_effect = list(verdicts)
    ladder = [PROFILE_QWEN_LOCAL, PROFILE_MARKER, PROFILE_GEMINI]
    return judge, route_page(1, ladder, _table_provider, judge, **kwargs)


def test_witness_refuted_free_rung_reaches_and_judges_paid_rung() -> None:
    """#1346: a witness-backed dispute no longer stops the paid rung."""
    judge, decision = _route(
        [
            AcceptDecision(accept=False, reason=_WITNESS_REJECT),
            AcceptDecision(accept=False, reason="empty/error output"),
            AcceptDecision(accept=True, reason="gemini read it"),
        ]
    )
    # The paid candidate went through judge.assess like every other rung.
    assert judge.assess.call_count == 3
    assert [c.args[1].engine for c in judge.assess.call_args_list][-1] is EngineType.GEMINI
    assert decision.accepted is True
    assert decision.final_output.engine == EngineType.GEMINI.value
    # The refuted free attempt stays in the journal with its reason.
    qwen = [a for a in decision.attempts if a.engine is EngineType.QWEN]
    assert len(qwen) == 1 and qwen[0].accepted is False
    assert "numeric tokens unsupported" in (qwen[0].reason or "")
    assert not any("paid escalation blocked" in (a.reason or "") for a in decision.attempts)


def test_paid_rung_refuted_too_ends_like_the_block() -> None:
    """Difference test: block on vs off, paid also refuted -> same page outcome."""
    verdicts = [
        AcceptDecision(accept=False, reason=_WITNESS_REJECT),
        AcceptDecision(accept=False, reason="empty/error output"),
        AcceptDecision(accept=False, reason=_WITNESS_REJECT),
    ]
    _, off = _route(verdicts, block_paid_on_source_evidence=False)
    _, on = _route(verdicts[:2], block_paid_on_source_evidence=True)
    assert off.accepted is on.accepted is False
    # Same winner-of-last-resort reading, same non-accepted outcome.
    assert off.final_output.engine == on.final_output.engine
    assert off.final_output.text == on.final_output.text
    # Nothing was accepted on either path, and the free refutation is journaled on both.
    for d in (off, on):
        assert not any(a.accepted for a in d.attempts)
        assert any("numeric tokens unsupported" in (a.reason or "") for a in d.attempts)


def test_budget_and_sparse_blocks_unchanged_by_source_evidence_lift() -> None:
    """The lift is only the source-evidence tag; budget still skips the paid rung."""
    judge, decision = _route(
        [
            AcceptDecision(accept=False, reason=_WITNESS_REJECT),
            AcceptDecision(accept=False, reason="empty/error output"),
        ],
        remaining_budget=0.0,
    )
    assert judge.assess.call_count == 2
    gemini = [a for a in decision.attempts if a.engine is EngineType.GEMINI]
    assert len(gemini) == 1 and "budget" in (gemini[0].output.error or gemini[0].reason or "")
    assert decision.accepted is False


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
