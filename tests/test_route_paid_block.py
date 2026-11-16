"""Agentic ladder must not escalate source-evidence rejects to paid rungs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

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


def _disputed_provider(prof, page_num: int) -> PageOutput:
    """Distinct text per rung (the paid read has MORE rows) and a witness dispute on each."""
    rows = {EngineType.QWEN: 1, EngineType.MARKER: 1, EngineType.GEMINI: 3}[prof.engine]
    body = "".join(f"| r{k} | {k} |\n" for k in range(rows))
    return PageOutput(
        page_num=page_num,
        text=f"| a | b |\n|---|---|\n{body}",
        status=PageStatus.SUCCESS,
        engine=prof.engine.value,
        table_disputed="'1'",
        table_disputed_values=("1",),
    )


def _settled(verdicts, inner_accepts: set[EngineType], *, block: bool):
    """Route then settle exactly as the orchestrator does; return the settled outcome."""
    from socr.pipeline.agentic import settle_scanned_table_decision

    judge = MagicMock()
    judge.assess.side_effect = list(verdicts)
    inner = MagicMock()
    inner.assess.side_effect = lambda out, prov: AcceptDecision(
        accept=prov.engine in inner_accepts, reason="inner"
    )
    decision = route_page(
        1,
        [PROFILE_QWEN_LOCAL, PROFILE_MARKER, PROFILE_GEMINI],
        _disputed_provider,
        judge,
        block_paid_on_source_evidence=block,
    )
    promoted = settle_scanned_table_decision(decision, inner=inner)
    return decision, promoted


_PAID_ALSO_REFUTED = [
    AcceptDecision(accept=False, reason=_WITNESS_REJECT),
    AcceptDecision(accept=False, reason="empty/error output"),
    AcceptDecision(accept=False, reason=_WITNESS_REJECT),
]


def _outcome(decision, promoted):
    return (
        decision.accepted,
        decision.final_output.engine,
        decision.final_output.text,
        None if promoted is None else (promoted.engine, promoted.output.text),
        decision.final_output.table_disputed_values,
    )


@pytest.mark.parametrize(
    "inner_accepts",
    [
        set(),
        {EngineType.GEMINI},
        {EngineType.QWEN},
        {EngineType.QWEN, EngineType.GEMINI},
    ],
    ids=[
        "inner-refuses-all",
        "inner-accepts-paid-only",
        "inner-accepts-qwen",
        "inner-accepts-both",
    ],
)
def test_paid_rung_refuted_too_ends_like_the_block(inner_accepts) -> None:
    """Difference test: block on vs off, paid also refuted, SETTLED outcome identical.

    Distinct text per rung and a witness dispute on every output, so the settle /
    promote walk is exercised. The paid reading (inner chain happy with it) must never
    be the one that ships: with the block it was an empty stub.
    """
    off_dec, off_prom = _settled(_PAID_ALSO_REFUTED, inner_accepts, block=False)
    on_dec, on_prom = _settled(_PAID_ALSO_REFUTED[:2], inner_accepts, block=True)
    assert _outcome(off_dec, off_prom) == _outcome(on_dec, on_prom)
    assert off_dec.final_output.engine != EngineType.GEMINI.value
    # The paid attempt ran and is journaled (with its refutation) on the off path only.
    assert [a.engine for a in off_dec.attempts][-1] is EngineType.GEMINI
    assert any("numeric tokens unsupported" in (a.reason or "") for a in off_dec.attempts[-1:])


def test_paid_reading_after_dispute_is_never_best_effort_or_promoted() -> None:
    """Mutant anchor: drop the paid_after_witness_dispute exclusions and this fails."""
    dec, prom = _settled(_PAID_ALSO_REFUTED, {EngineType.GEMINI}, block=False)
    paid = dec.attempts[-1]
    assert paid.paid_after_witness_dispute is True
    assert dec.accepted is False
    assert prom is None  # inner accepts only the paid table; it is not a candidate
    assert dec.final_output.engine == EngineType.QWEN.value


def test_cap_blocked_paid_call_is_not_charged() -> None:
    """A paid rung that never went out (cap, quota) costs nothing and spends no budget."""
    from socr.core.result import FailureMode

    def provider(prof, page_num: int) -> PageOutput:
        if prof.engine is EngineType.GEMINI:
            return PageOutput(
                page_num=page_num,
                text="",
                status=PageStatus.ERROR,
                engine="gemini",
                failure_mode=FailureMode.MODEL_UNAVAILABLE,
                error="max_paid_pages exhausted (1/1 Gemini pages this run)",
            )
        return _table_provider(prof, page_num)

    judge = MagicMock()
    judge.assess.side_effect = [
        AcceptDecision(accept=False, reason=_WITNESS_REJECT),
        AcceptDecision(accept=False, reason="empty/error output"),
        AcceptDecision(accept=False, reason="empty/error output"),
    ]
    decision = route_page(1, [PROFILE_QWEN_LOCAL, PROFILE_MARKER, PROFILE_GEMINI], provider, judge)
    paid = [a for a in decision.attempts if a.engine is EngineType.GEMINI]
    assert len(paid) == 1 and paid[0].cost_usd == 0.0
    assert decision.total_cost_usd == 0.0


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
