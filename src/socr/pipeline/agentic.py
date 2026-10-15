"""Cost-aware agentic per-page OCR routing.

The core of "best cost-effective provider on the go": for each page, try the
cheapest available provider, let a judge decide whether the output is good
enough, and escalate up the cost ladder only when it is not. Stop at the first
accepted output; if none is accepted, keep the best attempt.

Control flow lives in Python (the panel was unanimous: the LLM is a stateless
per-page decision function, not the orchestrator). ``route_page`` is pure given
its two injected dependencies:

  - ``run_provider(profile, page_num) -> PageOutput`` — actually OCR one page with
    one provider. It takes the whole ``ProviderProfile``, not just its
    ``EngineType``: local and cloud Qwen share ``EngineType.QWEN``, so an engine
    alone cannot say which backend/model must run (GH-159). The orchestrator wires
    the real implementation (render + engine call); tests pass a stub.
  - a ``PageJudge`` — accept/escalate verdict for one page's output. Either the
    VLM judge (``VLMPageJudge``) or the heuristic fallback (``HeuristicPageJudge``)
    so the loop still works with no model available.

Every attempt (engine, cost, verdict) is recorded so the manifest can store the
winning provider and the total cost is known.
"""

from __future__ import annotations

import concurrent.futures
import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

from socr.core.config import EngineType
from socr.core.daemon_call import submit_daemon
from socr.core.providers import ProviderProfile
from socr.core.result import (
    JUDGE_OUTCOME_COMPLETED,
    JUDGE_OUTCOME_EXCEPTION,
    JUDGE_OUTCOME_SOURCE_EVIDENCE,
    JUDGE_OUTCOME_TIMEOUT,
    JUDGE_OUTCOME_VERIFIER_ERROR,
    REJECTION_AMBIGUOUS_DEFERRED,
    REJECTION_JUDGE_ONLY,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.judge.judge import JudgeVerdict, PageJudgeCircuitOpenError, is_page_judge_timeout
from socr.tables.label_canonical import canonicalize_candidate

if TYPE_CHECKING:
    from socr.tables.furniture import DocumentFurniture

logger = logging.getLogger(__name__)

# Soft-timeout defaults per provider engine type.
# Values are derived from measured worst-case latencies on the owner's 64GB
# Mac, recorded in scratch/bench/out200/results.tsv (2026-06-13):
#   qwen3-vl:30b-a3b-instruct (local): ~50-60s prose/math, ~91-125s dense tables
#   thinking build qwen3-vl:30b:        never terminates (the case this guard catches)
# Values sit comfortably above the real worst-case but well below runaway.
DEFAULT_PROVIDER_TIMEOUTS: dict[EngineType, float] = {
    # local qwen3-vl:30b-a3b-instruct: 91-125s observed; 300s catches runaway
    EngineType.QWEN: 300.0,
    # Gemini API latency not directly measured in bench data;
    # 240s is a conservative upper-bound (cloud endpoint typically fast;
    # guard exists primarily for thinking-runaway, not Gemini)
    EngineType.GEMINI: 240.0,
}

RunProvider = Callable[[ProviderProfile, int], PageOutput]


@dataclass
class AcceptDecision:
    """A judge's accept/escalate verdict for one page output."""

    accept: bool
    reason: str = ""
    confidence: float = 0.0
    raw_verdict: object | None = None  # JudgeVerdict when a VLM judged
    #: #713 round 4 (Astra P2): the TYPED outcome THIS invocation produced when
    #: the decision is not a judge's answer at all. "" -- the default, and what
    #: every real verdict carries -- means a judge looked and decided.
    #:
    #: Only ``_UnverifiedTableRejection._reject_unverified`` sets it today, to
    #: ``JUDGE_OUTCOME_VERIFIER_ERROR``: the deterministic table verifier RAISED,
    #: so the inner judge was never consulted and the negative decision is a
    #: fail-closed refusal to proceed, not a refusal of the content.
    #:
    #: It rides on the DECISION rather than on the output because the outcome is
    #: a fact about one call. The boundary previously inferred it by comparing
    #: ``PageOutput.rejection_class`` before and after -- which cannot tell "the
    #: verifier set this value again" from "the judge left a stale value
    #: untouched", so a SECOND consecutive verifier crash on the same output was
    #: recorded as a completed rejection and destroyed the credential (Astra
    #: round 4). A value carried out of the call has no such blind spot.
    judge_outcome: str = ""


class PageJudge(Protocol):
    """Decides whether a page's OCR output is good enough to stop escalating."""

    def assess(self, output: PageOutput, provider: ProviderProfile) -> AcceptDecision: ...


#: The ``ProviderAttempt.reason`` recorded when a provider itself exceeded its
#: deadline. #800: consumers that must act on a PROVIDER timeout -- and not on a
#: judge timeout, whose reason is ``"judge raised: page judge timeout ..."`` and
#: also contains the word -- compare against this, never a substring.
REASON_PROVIDER_TIMEOUT = "provider timeout"


@dataclass
class ProviderAttempt:
    engine: EngineType
    output: PageOutput
    cost_usd: float
    accepted: bool
    reason: str = ""
    raw_verdict: object | None = None
    provider_id: str = ""
    model: str = ""
    backend: str = ""


@dataclass
class PageDecision:
    """Outcome of routing one page through the cost ladder."""

    page_num: int
    final_output: PageOutput
    attempts: list[ProviderAttempt] = field(default_factory=list)
    accepted: bool = False

    @property
    def total_cost_usd(self) -> float:
        return sum(a.cost_usd for a in self.attempts)

    @property
    def escalations(self) -> int:
        """Number of times we moved to a costlier provider (attempts - 1)."""
        return max(0, len(self.attempts) - 1)

    @property
    def winning_engine(self) -> str:
        return self.final_output.engine


def _evidence_readable_table_count(text: str) -> int | None:
    """How many tables ``verify_scanned_table`` / ``collect_table_tokens`` will read.

    ``None`` when an HTML table remains unparsed: the check cannot count, and
    the caller treats that as doubt (promote).
    """
    from socr.tables.source_evidence import evidence_readable_table_count, html_table_for_check

    check_text, html_unparsed = html_table_for_check(text or "")
    if html_unparsed:
        return None
    # #1124-r2: evidence-readable tables, not find_table_blocks
    return evidence_readable_table_count(check_text)


def _collect_evidence_table_tokens(text: str):
    """``collect_table_tokens`` on the same converted text the table count uses."""
    from socr.tables.source_evidence import collect_table_tokens, html_table_for_check

    check_text, html_unparsed = html_table_for_check(text or "")
    if html_unparsed:
        return None
    return collect_table_tokens(check_text)


def accepted_winner_has_passed_table(output: PageOutput) -> bool:
    """True when an accepted winner carries a table the source-evidence check passed.

    #1124: a later rung that drops the table (or keeps one the check did not
    read) must not lift the scanned-table floor. Prose-only text, HTML the
    checker could not parse, maths pipes #1098 excludes, and a candidate still
    carrying ``table_disputed`` / ``table_unverifiable`` are not a passed table.
    """
    n = _evidence_readable_table_count(output.text)
    if not n:
        return False
    if output.table_disputed or output.table_unverifiable:
        return False
    return True


def _witness_supported_numeric_tokens(marked: PageOutput) -> dict[str, int]:
    """Numeric tokens the witness checked and did not dispute in this attempt.

    Containment applies only where the witness actually read the page. An
    unverifiable table (no witness reading) and a disputed table with only the
    legacy reason string (no ``table_disputed_values``) carry no supported
    tokens — only the table-count rule in ``winner_accounts_for_marked_tables``.
    """
    if marked.table_unverifiable:
        return {}
    if marked.table_disputed and not marked.table_disputed_values:
        return {}
    table = _collect_evidence_table_tokens(marked.text or "")
    if table is None:
        return {}
    disputed = set(marked.table_disputed_values or ())
    return {tok: count for tok, count in table.numeric.items() if tok not in disputed}


def winner_accounts_for_marked_tables(winner: PageOutput, marked: PageOutput) -> bool:
    """True when the winner keeps every evidence-readable table the marked attempt had.

    Rule: promote when the winner has fewer evidence-readable tables than the
    marked attempt, or either side cannot be counted (unparsed HTML). No
    overlap threshold — a dropped table is a smaller count. A later rung that
    re-emits the same number of tables (HTML retry of one verified grid) may
    lift the floor; a rung that drops one of two may not. When in doubt,
    promote: a marked table is the safe outcome.

    #1126-r6: table count alone is not enough — the winner must still carry
    every numeric token the witness supported in the marked attempt (set
    containment on ``collect_table_tokens`` multisets, skipping disputed values).
    """
    # #1124-r2: winner must keep every marked-attempt table
    w_n = _evidence_readable_table_count(winner.text)
    m_n = _evidence_readable_table_count(marked.text)
    if w_n is None or m_n is None:
        return False
    if w_n < m_n:
        return False
    supported = _witness_supported_numeric_tokens(marked)
    if not supported:
        return True
    winner_table = _collect_evidence_table_tokens(winner.text or "")
    if winner_table is None:
        return False
    for tok, need in supported.items():
        if winner_table.numeric.get(tok, 0) < need:
            return False
    return True


def _promotion_inner_judge(judge: PageJudge | None) -> PageJudge | None:
    """The chain inside ``SourceEvidenceTableJudge``, deadline kept.

    Peels only ``SourceEvidenceTableJudge`` (re-entering it would reject the
    same table again). Production (#1128/#1135) is
    ``SE(Native(_TimeoutJudge(leaf)))``: peeling SE leaves Native around the
    leaf deadline. An older ``_TimeoutJudge(SE(...))`` wrap is still
    collected and rebuilt around the peeled inner so promotion keeps the
    deadline and the timeout callback. Native + circuit-breaker stay.
    """
    if judge is None:
        return None
    timeouts: list[object] = []
    core: object = judge
    while core is not None:
        name = core.__class__.__name__
        if name == "_TimeoutJudge":
            timeouts.append(core)
            core = getattr(core, "_inner", None)
            continue
        if name == "SourceEvidenceTableJudge":
            # #1125-r2: peel SourceEvidenceTableJudge only — keep the deadline
            core = getattr(core, "_inner", None)
            continue
        break
    wrapped = core
    for tw in reversed(timeouts):
        wrapped = tw.__class__(
            wrapped,
            tw._timeout_sec,
            owner=getattr(tw, "_owner", None),
            on_timeout=getattr(tw, "_on_timeout", None),
        )
    return wrapped


def settle_scanned_table_decision(
    decision: PageDecision, inner: PageJudge | None = None
) -> ProviderAttempt | None:
    """#1124 / #1125: promote a marked table the inner chain accepts, else floor.

    Chosen over "just keep the floor on a prose winner": the owner ruling is
    that a disputed / uncheckable table ships with a visible mark and the page
    image. A later rung that drops the table and ships SUCCESS is the worst
    outcome. Promotion walks marked candidates last-to-first and ships only
    one the inner chain accepts; none accepts → the old fail-closed floor.
    """
    marked = [
        att
        for att in decision.attempts
        if (att.output.table_disputed or att.output.table_unverifiable)
        and (att.output.text or "").strip()
    ]
    if not marked:
        return None
    # #1124: a later accept without a passed table does not lift the floor
    if decision.accepted and accepted_winner_has_passed_table(decision.final_output):
        if all(
            winner_accounts_for_marked_tables(decision.final_output, att.output) for att in marked
        ):
            return None
    if decision.accepted:
        decision.accepted = False
        for att in decision.attempts:
            if att.output is decision.final_output:
                att.accepted = False
                break
    return promote_unchecked_table_attempt(decision, inner=inner)


def _note_promotion_failure(att: ProviderAttempt, extra: str) -> None:
    """Keep the original rejection reason; append why promotion failed."""
    # #1125-r3: record why promotion failed — skip this append and
    # the refuse-journal test fails. Never touch ``judge_outcome``: the SE
    # stamp must survive promotion failures (#1126-r6).
    att.reason = f"{att.reason}; {extra}" if att.reason else extra


def promote_unchecked_table_attempt(
    decision: PageDecision, inner: PageJudge | None = None
) -> ProviderAttempt | None:
    """#1048 / #1125: at an exhausted (or un-floored) ladder, ship a marked table.

    ``SourceEvidenceTableJudge`` rejects a table it could not check, or one the
    second reading disputes, at its rung exactly as before, so the router
    escalates and a later rung that verifies wins unmarked.

    Candidates are walked LAST to FIRST (most-escalated first). #1125: a
    candidate is promoted only when the inner judge chain accepts it. None
    accepts → ``None`` and the floor applies. The original rejection reason is
    kept (the promotion note is a prefix). ``judge_outcome`` is
    ``page_judge_completed`` only when that inner call ran.
    """
    if decision.accepted:
        return None
    if inner is None:
        # #1125-r5: promotion requires an inner chain; peeling to nothing is fail-closed.
        for att in reversed(decision.attempts):
            out = att.output
            if not ((out.table_disputed or out.table_unverifiable) and (out.text or "").strip()):
                continue
            _note_promotion_failure(att, "promotion refused: no inner judge chain")
        return None
    for att in reversed(decision.attempts):
        out = att.output
        if not ((out.table_disputed or out.table_unverifiable) and (out.text or "").strip()):
            continue
        from types import SimpleNamespace

        provider = SimpleNamespace(
            id=att.provider_id,
            engine=att.engine,
            model=att.model,
            backend=att.backend,
        )
        # Deep-copy probe: the inner chain may rewrite ``text`` (native header
        # repair) or set fields the shallow copy would drop. ``table_disputed``
        # / ``table_disputed_values`` stay on the probe as the pre-repair SE
        # witness record; marks still name those bytes.
        probe = copy.deepcopy(out)
        probe.status = PageStatus.SUCCESS
        probe.failure_mode = FailureMode.NONE
        probe.error = ""
        try:
            # #1125: inner-chain accept is required to promote — skip this
            # call and the refuse-everything test fails
            verdict = inner.assess(probe, provider)
        except Exception as exc:  # noqa: BLE001 - a raised inner is a refuse
            why = (
                f"promotion timeout: {exc}"
                if is_page_judge_timeout(exc)
                else f"promotion judge raised: {exc}"
            )
            _note_promotion_failure(att, why)
            continue
        if not verdict.accept:
            why = f"promotion refused: {verdict.reason}" if verdict.reason else "promotion refused"
            _note_promotion_failure(att, why)
            continue
        original_reason = att.reason or ""
        kind = "disputed" if probe.table_disputed else "unverified"
        out = replace(
            probe,
            status=PageStatus.SUCCESS,
            failure_mode=FailureMode.NONE,
            error="",
            judge_outcome=JUDGE_OUTCOME_COMPLETED,
            judge_accepted=True,
        )
        att.output = out
        att.accepted = True
        note = f"ladder exhausted; table shipped marked {kind} (#1048)"
        att.reason = f"{note}; {original_reason}" if original_reason else note
        decision.final_output = att.output
        decision.accepted = True
        return att
    return None


def _error_output(page_num: int, msg: str) -> PageOutput:
    return PageOutput(
        page_num=page_num, text="", status=PageStatus.ERROR, error=msg, audit_passed=False
    )


def is_failed_candidate(output: PageOutput) -> bool:
    """True for an ERROR / CLI_ERROR output: a provider failure, never page content.

    GH-1020: such an output must not be picked by ``_best_effort`` and must not reach
    the table ladder or a table-escalation rung, which read ``output.text`` without a
    status check.
    """
    return output.status == PageStatus.ERROR or output.failure_mode == FailureMode.CLI_ERROR


def _best_effort(attempts: list[ProviderAttempt], page_num: int) -> ProviderAttempt:
    """When nothing was accepted, keep the most trustworthy attempt.

    Prefer an attempt that passed its own audit, then highest confidence, then
    most words, then the last (most-escalated) attempt. Never return empty if a
    non-empty attempt exists.
    """
    # The pre-GH-1020 selection, byte for byte: non-empty attempts, else all attempts.
    usable = [a for a in attempts if a.output.text.strip()]
    pool = usable or attempts
    # GH-1020: the only change. A failed candidate can never beat a usable one; when
    # every attempt failed, ``pool`` above is exactly what it always was.
    healthy = [a for a in usable if not is_failed_candidate(a.output)]
    pool = healthy or pool
    if not pool:
        return ProviderAttempt(
            engine=EngineType.AUTO,
            output=_error_output(page_num, "no provider produced output"),
            cost_usd=0.0,
            accepted=False,
            reason="all providers failed",
        )
    return max(
        pool,
        key=lambda a: (
            a.output.audit_passed,
            a.output.confidence,
            a.output.word_count,
        ),
    )


#: #1013: the outcomes of re-judging a timed-out candidate on resume, and the audit-event
#: kind each is surfaced under. Only ``accepted`` ships the candidate.
REJUDGE_ACCEPTED = "accepted"
REJUDGE_REJECTED = "rejected"
REJUDGE_TIMEOUT = "timeout"
REJUDGE_ERROR = "error"
REJUDGE_EVENT_KINDS = tuple(
    f"rejudge_{o}" for o in (REJUDGE_ACCEPTED, REJUDGE_REJECTED, REJUDGE_TIMEOUT, REJUDGE_ERROR)
)


def rejudge_candidate(
    make_candidate: Callable[[], PageOutput],
    prof: ProviderProfile,
    judge: PageJudge,
    *,
    attempts: int,
) -> tuple[str, str, PageOutput | None]:
    """#1013: ask ``judge`` about the kept bytes at most ``attempts`` times.

    ``make_candidate`` returns a FRESH ``PageOutput`` each call: every attempt judges
    its own snapshot, because a deadline worker can outlive its attempt and the table
    verifier can rewrite ``text`` in place. Acceptance binds to the sha256 of the bytes
    first judged: a snapshot whose text no longer hashes to it was rewritten, so its
    verdict is not a verdict on the kept bytes.

    ACCEPTED requires an explicit COMPLETED acceptance: ``accept``, no ``judge_outcome``
    (a verifier crash wears a verdict's shape), and a ``JudgeVerdict`` the judge itself
    produced in ``raw_verdict`` (a heuristic decision carries none). Anything else is
    ``REJUDGE_ERROR``. Only a TIMEOUT is retried (a missing verdict); an accept, a
    completed rejection or any other failure ends the loop. Returns ``(outcome, reason, shipped)``;
    ``shipped`` is a deep copy of the judged snapshot on ACCEPTED, else ``None``.
    """
    from socr.core.page_credential import sha256_text

    outcome, reason = REJUDGE_ERROR, "no re-judge attempt was allowed"
    judged_sha = sha256_text(make_candidate().text)
    for _ in range(max(attempts, 0)):
        snapshot = make_candidate()
        try:
            decision = judge.assess(snapshot, prof)
        except Exception as exc:
            outcome = REJUDGE_TIMEOUT if is_page_judge_timeout(exc) else REJUDGE_ERROR
            reason = f"judge raised: {exc}"
            if outcome == REJUDGE_TIMEOUT:
                continue
            break
        reason = decision.reason
        if decision.judge_outcome:
            outcome = REJUDGE_ERROR
            reason = f"decision carries a judge_outcome ({decision.judge_outcome}), not a verdict"
            break
        if sha256_text(snapshot.text) != judged_sha:
            outcome, reason = REJUDGE_ERROR, "the judge chain rewrote the kept bytes"
            break
        if decision.accept:
            if isinstance(decision.raw_verdict, JudgeVerdict):
                # Ship a COPY OF THE JUDGED snapshot, never a rebuild: verification sets
                # fields on it in place (``table_label_unverified``, corroboration, audit
                # notes, status, failure mode) that the manifest guards read. The copy
                # is taken now, so nothing that still holds ``snapshot`` can alter it.
                shipped = copy.deepcopy(snapshot)
                if sha256_text(shipped.text) != judged_sha:
                    return (
                        REJUDGE_ERROR,
                        "the accepted snapshot no longer hashes to the kept bytes",
                        None,
                    )
                return REJUDGE_ACCEPTED, reason, shipped
            outcome, reason = REJUDGE_ERROR, "acceptance is not a completed VLM verdict"
            break
        outcome = REJUDGE_REJECTED
        break
    return outcome, reason, None


def route_page(
    page_num: int,
    ladder: list[ProviderProfile],
    run_provider: RunProvider,
    judge: PageJudge,
    *,
    max_attempts: int = 0,
    remaining_budget: float | None = None,
    provider_timeout: dict[EngineType, float] | None = None,
    on_candidate: Callable[[int, PageOutput], None] | None = None,
) -> PageDecision:
    """Route one page: cheapest provider first, escalate until the judge accepts.

    Escalation is bounded by ladder exhaustion and the cost controls — NOT by
    a retry count. The historical ``max_retries + 1`` cap made the paid rungs
    mathematically unreachable whenever 3+ free local engines were installed,
    so "escalate to cloud when needed" never happened (issue #39).

    Args:
        page_num: 1-indexed page.
        ladder: providers ordered cheapest-first (see ``provider_ladder``).
        run_provider: runs one provider profile on one page.
        judge: accept/escalate decision per output.
        max_attempts: optional cap on providers tried (0 = whole ladder, the
            default). Kept for explicit user overrides and tests only.
        remaining_budget: document budget left, in USD. Checked BEFORE each
            rung: a paid provider that does not fit is skipped (free rungs
            always fit), instead of discovering the overrun after spending.
            ``None`` = unbounded.
        on_candidate: called at the candidate boundary below -- once per rung,
            on the canonicalised candidate, BEFORE the judge assesses it. A
            transform that must apply to what is judged, selected, bound and
            persisted belongs here and nowhere later; anything after this point
            is changing text a verdict has already been passed on.
        provider_timeout: optional per-provider wall-clock timeout in seconds.
            When a provider's ``EngineType`` is present in this dict, the
            ``run_provider`` call is wrapped with a
            ``concurrent.futures.ThreadPoolExecutor`` timeout. On timeout the
            provider is treated like a raised exception: an ERROR attempt is
            recorded and the loop escalates to the next rung.
            ``None`` (the default) disables all timeouts (backward-compatible).
    """
    if not ladder:
        return PageDecision(page_num, _error_output(page_num, "no providers available"))

    limit = len(ladder) if max_attempts <= 0 else min(max_attempts, len(ladder))
    attempts: list[ProviderAttempt] = []
    tried = 0

    for prof in ladder:
        if tried >= limit:
            break
        if remaining_budget is not None and prof.cost_per_page_usd > remaining_budget:
            logger.info(
                "page %s: skipping %s ($%g/page exceeds remaining budget $%g)",
                page_num,
                prof.engine.value,
                prof.cost_per_page_usd,
                remaining_budget,
            )
            # Record a stub attempt so the manifest journal captures the skip
            # reason for every rung (not just the ones that ran).
            attempts.append(
                ProviderAttempt(
                    engine=prof.engine,
                    output=_error_output(page_num, "budget exceeded"),
                    cost_usd=0.0,
                    accepted=False,
                    reason="budget exceeded",
                    provider_id=prof.id,
                    model=prof.model,
                    backend=prof.backend,
                )
            )
            continue
        tried += 1
        timeout_sec: float | None = (
            provider_timeout.get(prof.engine) if provider_timeout is not None else None
        )
        try:
            if timeout_sec is not None:
                # GH-974: a DAEMON thread (GH-172 showed ThreadPoolExecutor workers
                # are not, and hold the interpreter open while a wedged socket
                # never returns). The abandoned call is discarded at exit.
                future = submit_daemon(run_provider, prof, page_num)
                try:
                    output = future.result(timeout=timeout_sec)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    logger.warning(
                        "provider %s timed out on page %s (%.2fs) — escalating",
                        prof.engine.value,
                        page_num,
                        timeout_sec,
                    )
                    attempts.append(
                        ProviderAttempt(
                            engine=prof.engine,
                            output=_error_output(
                                page_num,
                                f"{prof.engine.value}: timed out after {timeout_sec}s",
                            ),
                            cost_usd=0.0,
                            accepted=False,
                            reason=REASON_PROVIDER_TIMEOUT,
                            # GH-344: the timeout branch was the only attempt
                            # that omitted these. Budget skip, provider raise,
                            # judge raise and the accepted path all record them,
                            # so a timed-out rung was the one journal entry that
                            # could not say WHICH provider timed out -- exactly
                            # the entry an operator reads first.
                            provider_id=prof.id,
                            model=prof.model,
                            backend=prof.backend,
                        )
                    )
                    continue
            else:
                output = run_provider(prof, page_num)
        except Exception as exc:  # a provider blowing up must not kill the page
            logger.warning("provider %s failed on page %s: %s", prof.engine.value, page_num, exc)
            attempts.append(
                ProviderAttempt(
                    engine=prof.engine,
                    output=_error_output(page_num, f"{prof.engine.value}: {exc}"),
                    cost_usd=0.0,
                    accepted=False,
                    reason="provider raised",
                    provider_id=prof.id,
                    model=prof.model,
                    backend=prof.backend,
                )
            )
            continue

        if remaining_budget is not None:
            remaining_budget -= prof.cost_per_page_usd

        # #688 -- THE candidate boundary. This is where a provider's proposed
        # ``PageOutput.text`` becomes the candidate everything downstream
        # judges, selects, binds, adjudicates and persists, so it is where its
        # table row labels are canonicalised (decode the intended entities,
        # drop the presentation indentation). Row/column counts and every
        # value cell are untouched, so no physical cell reference moves. The
        # provider's raw bytes remain immutable provenance in the transcript
        # the caller keeps; what changes is the candidate.
        canonicalize_candidate(output)
        if on_candidate is not None:
            on_candidate(page_num, output)

        try:
            decision = judge.assess(output, prof)
        except Exception as exc:  # a judge blowing up must not kill the document
            # Mirror the provider guard above. The text is real — only the
            # verdict is missing — so the attempt is recorded UNJUDGED and stays
            # eligible for ``_best_effort``; the loop escalates. Without this an
            # HTTP error from the judge backend (e.g. a 404 on a model that was
            # never pulled) propagates out of the per-page loop in
            # ``_phase_agentic`` and takes the whole document with it (#133).
            logger.warning("judge failed on page %s at %s: %s", page_num, prof.engine.value, exc)
            # #713: record the TYPED outcome on the attempt itself. The reason
            # string below still carries the raw exception text for the audit
            # trail, but no gate may key on it: it interpolates an arbitrary
            # ``str(exc)``, so "timed out" in it proves nothing about who said
            # it. Only this field separates an infrastructure timeout -- the
            # one outcome #713's credentialed stand-in may act on -- from a
            # judge defect, a refusal, or a malformed response, and it is set
            # from the exception's TYPE. A completed rejection never reaches
            # this branch at all (it returns a decision), and a provider blowing
            # up is caught by the guard above, which never touches this field.
            output.judge_outcome = (
                JUDGE_OUTCOME_TIMEOUT if is_page_judge_timeout(exc) else JUDGE_OUTCOME_EXCEPTION
            )
            attempts.append(
                ProviderAttempt(
                    engine=prof.engine,
                    output=output,
                    cost_usd=prof.cost_per_page_usd,
                    accepted=False,
                    reason=f"judge raised: {exc}",
                    provider_id=prof.id,
                    model=prof.model,
                    backend=prof.backend,
                )
            )
            continue

        # #713 round 2 (Astra P1-2): the judge COMPLETED. Record that typed
        # outcome on the attempt's own output BEFORE the attempt is appended,
        # and on a refusal drop any acceptance credential riding on it.
        #
        # ``output`` is a live object a previous rung may already have stamped
        # ``JUDGE_OUTCOME_TIMEOUT`` (and the table gate may already have minted a
        # credential against). A completed verdict on these exact bytes is a
        # LATER, applicable answer to the same question the timed-out judge never
        # answered, so it supersedes it: the credentialed stand-in in
        # ``manifest.credentialed_judge_timeout_winner`` admits only
        # ``JUDGE_OUTCOME_TIMEOUT``, and after this line that page can no longer
        # reach it. Retiring the credential too is belt-and-braces on the same
        # fact -- authority to ship must never outlive the verdict that would
        # have refused it.
        # #713 (Astra rounds 2-4): a decision object is NOT proof a judge answered.
        # ``_UnverifiedTableRejection._reject_unverified`` returns a negative
        # ``AcceptDecision`` when the deterministic table VERIFIER RAISED, before
        # the inner judge is ever consulted -- an infrastructure failure wearing
        # a verdict's shape. Stamping it COMPLETED made every downstream reader
        # ("a later completed verdict refused these bytes") retire a credential
        # nothing had contradicted, and the page shipped the fail-closed floor on
        # the strength of a crash.
        #
        # The outcome is read from the DECISION this call returned, never from
        # the reason text and never from a before/after comparison of
        # ``output.rejection_class``: that field persists across rungs, so a
        # same-value assignment -- a second verifier crash on the same output --
        # is invisible to a snapshot, and the second crash was recorded as a
        # completed rejection (Astra round 4). A stale class from an earlier rung
        # still cannot disguise a real refusal, because a real verdict carries no
        # ``judge_outcome`` at all.
        missing_verdict = decision.judge_outcome if not decision.accept else ""
        if decision.accept:
            output.judge_outcome = JUDGE_OUTCOME_COMPLETED
        elif missing_verdict == JUDGE_OUTCOME_SOURCE_EVIDENCE:
            # #1126-r6: witness-contradicted bytes are a completed refusal, not a
            # missing verdict — retire any credential without stamping page_judge_completed.
            output.judge_outcome = JUDGE_OUTCOME_SOURCE_EVIDENCE
            output.table_acceptance_credential = None
            output.judge_accepted = False
        elif missing_verdict:
            # A missing verdict cannot retire another missing verdict either: an
            # already-typed timeout stands, so a verifier crashing on a LATER
            # rung does not destroy the very stand-in this ticket exists for.
            if output.judge_outcome != JUDGE_OUTCOME_TIMEOUT:
                output.judge_outcome = missing_verdict
        else:
            output.judge_outcome = JUDGE_OUTCOME_COMPLETED
            output.table_acceptance_credential = None
        # #1129 r9: stamp the judge verdict here; #1124 may later set att.accepted
        # False without changing what the judge said.
        if decision.accept:
            output.judge_accepted = True
        elif not missing_verdict:
            output.judge_accepted = False
        attempts.append(
            ProviderAttempt(
                engine=prof.engine,
                output=output,
                cost_usd=prof.cost_per_page_usd,
                accepted=decision.accept,
                reason=decision.reason,
                raw_verdict=decision.raw_verdict,
                provider_id=prof.id,
                model=prof.model,
                backend=prof.backend,
            )
        )
        if decision.accept:
            return PageDecision(page_num, output, attempts, accepted=True)

    best = _best_effort(attempts, page_num)
    return PageDecision(page_num, best.output, attempts, accepted=False)


# ---------------------------------------------------------------------------
# Judge adapters — bridge the page-level loop to the actual quality check.
# ---------------------------------------------------------------------------


class HeuristicPageJudge:
    """Cheap, no-model judge: accept if the output passes the heuristic checker.

    The graceful fallback so the agentic loop runs anywhere, even with no VLM.

    ``sparse_ok(page_num) -> bool`` marks pages where low word count is
    expected. Without it, a correct sparse caption page is rejected at EVERY
    rung and — with uncapped escalation — deterministically walks the paid
    ladder before shipping best-effort: the judge must be sparse-aware at the
    decision point, not only in post-hoc scoring (issue #39 review).
    """

    def __init__(self, checker, sparse_ok=None) -> None:
        self._checker = checker
        self._sparse_ok = sparse_ok if sparse_ok is not None else (lambda page_num: False)

    def assess(self, output: PageOutput, provider: ProviderProfile) -> AcceptDecision:
        if output.status != PageStatus.SUCCESS or not output.text.strip():
            return AcceptDecision(accept=False, reason="empty/error output")
        check = self._checker.check(output.text, sparse_ok=self._sparse_ok(output.page_num))
        if check.passed:
            reason = "heuristics passed"
        else:
            reason = "; ".join(check.errors or ["heuristics failed"])
        return AcceptDecision(
            accept=bool(check.passed),
            reason=reason,
            confidence=output.confidence,
        )


class CircuitBreakerPageJudge:
    """Short-circuits ``inner`` to a judge TIMEOUT while ``is_open()`` (#987).

    The judge circuit breaker. Once the page judge is shown to be wedged, every
    later call raises the SAME ``PageJudgeTimeoutError`` a real deadline raises,
    instantly. ``route_page`` then types it ``JUDGE_OUTCOME_TIMEOUT`` and the page
    fails closed exactly as it does today, minus the wait. It never degrades to a
    weaker judge: a heuristic judge does not check fidelity to the page image, so
    pages it accepted would ship as passed.

    Sits at the VLM leaf, inside the deterministic native-table verifier: pages
    that verifier settles without a model never reach it and are unaffected.
    """

    def __init__(self, inner: PageJudge, is_open: Callable[[], bool]) -> None:
        self._inner = inner
        self._is_open = is_open

    def assess(self, output: PageOutput, provider: ProviderProfile) -> AcceptDecision:
        if self._is_open():
            raise PageJudgeCircuitOpenError(
                "page judge timeout (circuit open): the judge failed a liveness probe "
                "earlier in this document; failing closed without waiting"
            )
        return self._inner.assess(output, provider)


class VLMPageJudge:
    """Judge a page by *looking* at it: render the page, ask the VLM judge whether
    the OCR faithfully represents it.

    ``render_image(page_num) -> path`` is injected so this stays decoupled from
    the document handle and testable.
    """

    def __init__(self, judge, render_image: Callable[[int], object]) -> None:
        self._judge = judge
        self._render_image = render_image

    def assess(self, output: PageOutput, provider: ProviderProfile) -> AcceptDecision:
        if output.status != PageStatus.SUCCESS or not output.text.strip():
            return AcceptDecision(accept=False, reason="empty/error output")
        image_path = self._render_image(output.page_num)
        verdict = self._judge.judge(image_path, output.text)
        return AcceptDecision(
            accept=bool(verdict.faithful),
            reason="; ".join(verdict.issues) if verdict.issues else "faithful",
            confidence=verdict.confidence,
            raw_verdict=verdict,
        )


#: GH-162: exceptions that mean the PROCESS is broken, not that this page's
#: verifier hit bad geometry. Swallowing them per-page would turn a wedged or
#: exhausted interpreter into a quiet document-wide table rejection, hiding the
#: real fault -- the same failure shape the VLM cascade-halt exists to prevent.
#: ``KeyboardInterrupt``/``SystemExit`` derive from ``BaseException`` and are
#: already outside ``except Exception``.
_VERIFIER_FATAL = (MemoryError, RecursionError)


class _UnverifiedTableRejection:
    """GH-162: shared fail-closed handling for a table verifier that raised.

    Both table judges run a deterministic verifier BEFORE an inner (VLM or
    heuristic) judge, and both promise in their docstrings that an unverifiable
    table hard-rejects rather than reaching heuristic acceptance. A raised
    verifier is the strongest form of unverifiable: no term ran, so there is no
    evidence either way.

    The rejection is deliberately shallow. It does NOT set ``status``,
    ``failure_mode`` or ``audit_passed``:

    - ``audit_passed`` is the winner-SELECTION flag, and clearing it discards
      the page's text (the #252 round-1 defect).
    - a judge decides ONE ladder rung. Whether the page ends up escalated,
      flagged, or reduced to a failed-table marker is the assembler's call,
      made once all rungs are known.

    What it does set is ``rejection_class``, so the disposition is nameable
    downstream -- and ``REJECTION_VERIFIER_ERROR`` is deliberately absent from
    ``D3_SUPERSEDING_REJECTIONS``, so a crashed verifier can never license
    shipping its table over a fail-closed floor.
    """

    def _reject_unverified(self, output: PageOutput, exc: BaseException, page_num: int):
        from socr.core.result import REJECTION_VERIFIER_ERROR

        name = type(self).__name__
        detail = f"{type(exc).__name__}: {exc}"[:500]
        logger.warning(
            "%s: verifier raised on p%d (%s); rejecting fail-closed (GH-162)",
            name,
            page_num,
            detail,
        )
        self._emit_event(
            page_num=page_num,
            kind="table_verifier_error",
            engine=output.engine or "",
            detail=(
                f"{name} verifier raised; no deterministic table term ran, so the "
                "table is unverified. Rejected fail-closed rather than delegated "
                f"to the inner judge. {detail}"
            ),
            data={
                "judge": name,
                "exception_type": type(exc).__name__,
                "rejection_class": REJECTION_VERIFIER_ERROR,
            },
        )
        output.rejection_class = REJECTION_VERIFIER_ERROR
        return AcceptDecision(
            accept=False,
            reason=f"table_verifier_error: {name} raised ({type(exc).__name__})",
            confidence=0.0,
            # #713 round 4: this call produced no verdict. Carried on the result
            # so ``route_page`` needs no inference about what the verifier did.
            judge_outcome=JUDGE_OUTCOME_VERIFIER_ERROR,
        )


class SourceEvidenceTableJudge(_UnverifiedTableRejection):
    """Fail-closed source-evidence gate for VLM-emitted markdown tables (GH-90).

    Runs BEFORE the inner judge chain on ANY model output that contains markdown
    table blocks, regardless of ``PageState.has_tables``.

    Deferral is selected on the page's born-digital TRUST classification, not on
    word presence (GH-163). A trusted-native page defers to
    ``NativeTableVerifierJudge``; a page classified untrusted verifies cell
    tokens against local non-generative evidence (crop and page raster +
    optional classical OCR) EVEN IF it carries native words -- a scanned page
    with a baked-in OCR layer has words and no trustworthy reading, and its text
    layer is excluded from that evidence so it cannot corroborate the table
    under suspicion. When the classification is unavailable, word presence is
    the fallback, as before. Unsupported or unverifiable tables hard-reject
    before heuristic acceptance.
    """

    def __init__(
        self,
        inner: PageJudge,
        get_fitz_page: Callable[[int], object] | None,
        record_event: Callable[[object], None] | None = None,
        *,
        ocr_image_fn: Callable[[object], str] | None = None,
        native_trusted: Callable[[int], bool | None] | None = None,
    ) -> None:
        self._inner = inner
        self._get_fitz_page = get_fitz_page
        self._record_event = record_event
        self._ocr_image_fn = ocr_image_fn
        # GH-163: the page's born-digital classification. Without it the
        # verifier deferred on word PRESENCE, so a scanned page carrying a
        # baked-in OCR layer skipped the evidence check and was graded against
        # that same untrusted layer. None (no callable, or it cannot tell)
        # keeps the pre-GH-163 behaviour.
        self._native_trusted = native_trusted

    def assess(self, output: PageOutput, provider: ProviderProfile) -> AcceptDecision:
        from socr.tables.reconcile import find_table_blocks
        from socr.tables.source_evidence import html_table_for_check, verify_scanned_table

        # #1048: HTML tables are converted to markdown for the CHECK (the shipped text is
        # untouched); one that cannot be parsed is a table the check cannot read.
        check_text, html_unparsed = html_table_for_check(output.text or "")
        if not find_table_blocks(check_text) and not html_unparsed:
            return self._inner.assess(output, provider)

        if output.status != PageStatus.SUCCESS or not output.text.strip():
            return self._inner.assess(output, provider)

        if self._get_fitz_page is None:
            return self._inner.assess(output, provider)

        page_num = output.page_num
        try:
            fitz_page = self._get_fitz_page(page_num)
            trusted: bool | None = None
            if self._native_trusted is not None:
                try:
                    trusted = self._native_trusted(page_num)
                except Exception:  # noqa: BLE001 - unknown, not untrusted
                    trusted = None
            result = verify_scanned_table(
                fitz_page,
                check_text,
                ocr_image_fn=self._ocr_image_fn,
                native_trusted=trusted,
            )
        except _VERIFIER_FATAL:
            raise
        except Exception as exc:
            # GH-162: a raised verifier produced NO evidence, so this table is
            # unverified -- not verified-good. Delegating to an inner judge that
            # accepts would ship a generated table under a clean status, which
            # this class's docstring promises not to do. The scanned lane has no
            # native reading to fall back on, so the model output is the only
            # reading of the page and it ships on zero corroboration.
            return self._reject_unverified(output, exc, page_num)

        from socr.tables.source_evidence import REASON_NO_TABLE_BLOCKS

        # A deferral to the native verifier stands. "No markdown table to read" is not one
        # when an HTML table the check could not parse is what the candidate holds.
        if result.deferred and not (html_unparsed and result.reason == REASON_NO_TABLE_BLOCKS):
            return self._inner.assess(output, provider)

        if html_unparsed and result.verifiable and (result.passed or result.deferred):
            # The tables the check could read passed, but an HTML table it cannot read
            # remains: that table is unchecked, so the candidate is could-not-check.
            from socr.tables.source_evidence import SourceEvidenceResult

            result = SourceEvidenceResult(
                verifiable=False,
                passed=False,
                reason="an HTML table could not be parsed for checking",
            )

        if not result.verifiable or not result.passed:
            from socr.core.result import FailureMode
            from socr.tables.source_evidence import CAUSE_NO_WITNESS_BACKEND

            # #658: the gate has two fail-closed endings and they are not the
            # same finding. A missing tesseract means nothing ever read the
            # pixels, so socr holds no evidence about this table; calling that
            # HALLUCINATION accuses the model of inventing content on the sole
            # ground that the host is missing an install. The page stays
            # fail-closed either way -- unwitnessed is still unverified -- and
            # only the reason, the failure mode and one EXTRA audit event
            # change; the generic reject event below is emitted in both cases.
            no_witness = result.cause == CAUSE_NO_WITNESS_BACKEND
            self._emit_event(
                page_num=page_num,
                kind="source_evidence_table_reject",
                engine=output.engine or "",
                detail=result.reason,
                data={
                    "verifiable": result.verifiable,
                    "passed": result.passed,
                    "cause": result.cause,
                },
            )
            if no_witness:
                # A SECOND event, not a replacement for the one above: every
                # consumer of the generic reject kind (``tables_trust``'s
                # distrust set among them) must keep seeing this page, and a
                # page that lost its reject event would read as trusted. This
                # one exists so an operator can count unwitnessed pages with a
                # single grep instead of parsing a sentence.
                self._emit_event(
                    page_num=page_num,
                    kind="source_evidence_no_witness_backend",
                    engine=output.engine or "",
                    detail=result.reason,
                    data={
                        "cause": result.cause,
                        # The precise state, never a flat "absent": an OCR
                        # reader that crashed and a page that would not
                        # rasterise are both unwitnessed, and neither is fixed
                        # by installing anything.
                        "witness_state": result.witness_state,
                    },
                )
            # #1048 (owner rulings 2026-10-08 and 2026-10-12): the candidate is
            # rejected at this rung exactly as before, so the router still escalates.
            # What changes is the END of the ladder: if no rung verifies, the
            # orchestrator ships the last candidate carrying one of these fields
            # with its in-text mark instead of the failed-table marker. A table the
            # check could not examine (``verifiable`` False) records the reason in
            # ``table_unverifiable``; one the second reading disputes records EVERY
            # unsupported value in ``table_disputed`` (no cap; the mark names them
            # all). Other contradiction sources are not touched.
            if result.verifiable and not result.passed:
                output.table_disputed_values = tuple(result.disputed_values)
                output.table_disputed = ", ".join(result.disputed_values) or result.reason
            else:
                output.table_unverifiable = result.reason
            output.audit_passed = False
            output.status = PageStatus.ERROR
            output.failure_mode = (
                FailureMode.NO_WITNESS_BACKEND if no_witness else FailureMode.HALLUCINATION
            )
            # #658 reviewer item 3: the ``source_evidence_table`` token in this
            # reason is a LOAD-BEARING marker, not a label -- the scanned-table
            # fail-closed floor latches on ``"source_evidence_table" in
            # att.reason`` (orchestrator ``_phase_agentic``). Renaming the
            # prefix for the no-witness ending silently stopped the floor from
            # applying to exactly these pages, which is the fail-OPEN direction.
            # The distinguishing token is ADDED inside the same prefix instead.
            marker = f"[{CAUSE_NO_WITNESS_BACKEND}] " if no_witness else ""
            return AcceptDecision(
                accept=False,
                reason=f"source_evidence_table: {marker}{result.reason}",
                confidence=0.0,
                # #1125-r2: source-evidence reject is not page_judge_completed
                judge_outcome=JUDGE_OUTCOME_SOURCE_EVIDENCE,
            )

        if result.content_unverified:
            # Astra review round 1 (cf. #658): an event with no consumer
            # disappears. The table ships (numerics are fully corroborated) --
            # this must NOT flip ``audit_passed``, which is what selects the
            # winner -- so it becomes a STANDING, candidate-associated outcome
            # via a data field on the output itself: it survives to the
            # finalized page and the resumed sidecar (``table_label_unverified``,
            # which is what makes the warning retire automatically the day a
            # different, fully-supported candidate wins instead), feeds
            # ``TABLE_DISTRUST_KINDS`` (``tables_trust.json``), and a
            # document-metadata / CLI note (``orchestrator.py``'s
            # ``_label_unverified_note`` / ``_label_unverified_pages``).
            #
            # Astra review round 2 (P1): this used to also flip
            # ``output.status`` to WARNING right here, before the ``return``
            # below hands the SAME output to ``self._inner``. Both
            # ``HeuristicPageJudge`` and ``VLMPageJudge`` treat any non-SUCCESS
            # status as empty/error input and reject on sight -- so setting
            # WARNING at judge time rejected the very candidate this ticket
            # exists to ship. The reporting status change belongs at
            # FINALIZATION, once a page has already been selected and no judge
            # will see it again -- see ``manifest._apply_label_unverified_guard``,
            # the same pattern #658 and #165 use for a post-selection status
            # demotion. ``audit_notes`` is safe to append here: no judge reads it.
            output.table_label_unverified = result.content_unverified
            output.audit_notes.append(
                f"table label unverified by page evidence: {result.content_unverified}"
            )
            self._emit_event(
                page_num=page_num,
                kind="source_evidence_table_label_unverified",
                engine=output.engine or "",
                detail=result.content_unverified,
                data={"cause": ""},
            )

        return self._inner.assess(output, provider)

    def _emit_event(self, page_num: int, kind: str, engine: str, detail: str, data: dict) -> None:
        if self._record_event is None:
            return
        try:
            from socr.core.audit_log import AuditEvent

            self._record_event(
                AuditEvent(page_num=page_num, kind=kind, engine=engine, detail=detail, data=data)
            )
        except Exception as exc:
            logger.debug("SourceEvidenceTableJudge: failed to record audit event: %s", exc)


class NativeTableVerifierJudge(_UnverifiedTableRejection):
    """Two-tier deterministic pre-check for born-digital table pages.

    Runs BEFORE the inner judge (VLM or heuristic) on pages where the
    born-digital detector found table-like structure (``is_table_page``
    returns True for the given page number).

    Tier 1 — Hard-fail (value_guard): the output table's numeric-token content
    does not match the native page's numeric tokens on a per-row basis (TR-4
    value-guard: row-aware multiset equality + label-binding interleaved detection).
    → Return AcceptDecision(accept=False) immediately; inner judge NOT called.

    Tier 2 — Warn-and-defer: ambiguous lane-count mismatch (paired-year
    columns, spanning headers, stub columns).
    → Record an AuditEvent, then delegate to the inner judge.

    Pages that bypass native geometry verification (non-table-classified pages,
    missing ``get_fitz_page`` callable, non-success or empty outputs, or scanned
    pages with empty ``get_text('words')``) delegate to the inner judge; any
    accepting inner decision still passes through the string-only structural
    shipping gate (``_apply_structural_gate`` with ``words=None`` and
    ``rules=None``) before shipping, preventing structurally defective tables
    from shipping.

    Args:
        inner:         The wrapped judge (VLMPageJudge or HeuristicPageJudge).
        get_fitz_page: Callable[[page_num: int]] -> fitz.Page — opens the PDF
                       page for native geometry access.  None disables the
                       verifier (safe fallback for test environments without a
                       real PDF).
        is_table_page: Callable[[page_num: int]] -> bool — True when the
                       born-digital detector found table structure on the page.
        record_event:  Callable[[AuditEvent]] — appends to state.events.
                       None silently skips audit recording (non-breaking).
    """

    def __init__(
        self,
        inner: PageJudge,
        get_fitz_page: Callable[[int], object] | None,
        is_table_page: Callable[[int], bool],
        record_event: Callable[[object], None] | None = None,
    ) -> None:
        self._inner = inner
        self._get_fitz_page = get_fitz_page
        self._is_table_page = is_table_page
        self._record_event = record_event
        # #988 M2: repeated word positions, scanned once per document.
        self._furniture: dict[str, DocumentFurniture] = {}

    def assess(self, output: PageOutput, provider: ProviderProfile) -> AcceptDecision:
        from socr.tables.native_verifier import (
            VerifierState,
            describe_drift,
            verify_native_table,
        )

        page_num = output.page_num

        # Only run the verifier on born-digital table pages with usable geometry.
        # Bypassing native geometry does NOT bypass the string-only structural
        # gate: an accepting inner decision still passes through
        # _apply_structural_gate with words=None and rules=None before returning.
        if not self._is_table_page(page_num) or self._get_fitz_page is None:
            decision = self._inner.assess(output, provider)
            return self._apply_structural_gate(decision, output, page_num, None, None)

        if output.status != PageStatus.SUCCESS or not output.text.strip():
            decision = self._inner.assess(output, provider)
            return self._apply_structural_gate(decision, output, page_num, None, None)

        try:
            fitz_page = self._get_fitz_page(page_num)
            words = fitz_page.get_text("words") if fitz_page is not None else None
            # GH-212: the header term needs the page's drawn rules. Precompute
            # them here so structure_check stays pure and never sees a page.
            from socr.tables.locate import _horizontal_rules

            rules = _horizontal_rules(fitz_page) if fitz_page is not None else None
            furniture = self._document_furniture(fitz_page)
            vr = verify_native_table(fitz_page, output.text)
            vr = self._maybe_repair_collapsed_headers(fitz_page, output, vr)
        except _VERIFIER_FATAL:
            raise
        except Exception as exc:
            # GH-162: supersedes the GH-200 partial hardening. That fix ran the
            # string-only grid-shape term over the inner judge's decision, which
            # caught a ragged candidate but still let a well-formed one through
            # on an accepting inner judge -- the value-guard and
            # header-attribution terms never ran. A raised verifier means no
            # deterministic term ran at all, so there is nothing to accept on.
            # The native layer still exists as a checkable reference here, which
            # is why the floor may later prefer a native reading; that is the
            # assembler's call, not this judge's.
            return self._reject_unverified(output, exc, page_num)

        # #259 round 3: the value guard found a numeric multiset mismatch but a
        # row-count discrepancy made the pairing unreliable, so it is AMBIGUOUS
        # rather than CERTAIN_FAIL. Under the owner's ruling the flagged table
        # is KEPT, which makes surfacing this mandatory rather than optional: a
        # kept table that socr privately believes contains a wrong number is
        # silent content corruption. Emitted before the tri-state dispatch so it
        # fires on every exit -- warn, row-count-warn-only, accept alike -- and
        # exactly once per assess.
        if getattr(vr, "unadjudicated_drift", None):
            self._emit_event(
                page_num=page_num,
                kind="table_value_drift_unadjudicated",
                engine=output.engine or "",
                detail=(
                    "numeric multiset mismatch detected but NOT adjudicated: the "
                    "row-count discrepancy makes per-row pairing unreliable, so it "
                    "is not a certain fail. " + describe_drift(vr.unadjudicated_drift)
                ),
                data={
                    "drifted_rows": vr.unadjudicated_drift,
                    "adjudicated": False,
                    "verifier_state": vr.state,
                },
            )

        # TR-6 tri-state dispatch based on vr.state:
        #   EXACT_PASS   → ship immediately (no inner judge needed)
        #   CERTAIN_FAIL → hard-reject
        #   AMBIGUOUS    → row-count warning and/or Tier-2 lane gap → emit event,
        #                  delegate to inner judge (ships flagged)

        # Soft row-count warning: table ships, but emit audit event before
        # delegating so the audit log + status machinery can demote to WARNING.
        if vr.row_count_warn:
            self._emit_event(
                page_num=page_num,
                kind="value_guard_row_count_warning",
                engine=output.engine or "",
                detail=vr.row_count_warn_reason,
                data={
                    "native_lane_count": vr.native_lane_count,
                    "output_col_count": vr.output_col_count,
                    "verifier_state": vr.state,
                },
            )

        if vr.hard_fail:
            # CERTAIN_FAIL: systematic label-binding or multiset mismatch → hard-reject
            if vr.drifted_rows:
                predicate = vr.drifted_rows[0].get("predicate", "value_guard")
            else:
                predicate = "value_guard"
            self._emit_event(
                page_num=page_num,
                kind="native_table_verifier_hard_fail",
                engine=output.engine or "",
                detail=vr.reason,
                data={
                    "native_lane_count": vr.native_lane_count,
                    "output_col_count": vr.output_col_count,
                    "drifted_rows": vr.drifted_rows,
                    "predicate": predicate,
                    "verifier_state": vr.state,
                },
            )
            return AcceptDecision(
                accept=False,
                reason=f"native_table_verifier: {vr.reason}",
                confidence=0.0,
            )

        if vr.warn:
            # AMBIGUOUS (Tier-2 lane-count gap): emit event, defer to inner judge
            self._emit_event(
                page_num=page_num,
                kind="native_table_verifier_warn",
                engine=output.engine or "",
                detail=vr.reason,
                data={
                    "native_lane_count": vr.native_lane_count,
                    "output_col_count": vr.output_col_count,
                    "drifted_rows": vr.drifted_rows,
                    "verifier_state": vr.state,
                },
            )
            # Defer to the inner judge — do NOT escalate here
            decision = self._inner.assess(output, provider)
            # #259 round 2: record the DISPOSITION, not just the bool.
            # ``ProviderAttempt.accepted`` is all that survives into
            # ``PageOutput`` (orchestrator: ``att.output.audit_passed =
            # att.accepted``), so downstream a CERTAIN_FAIL rejection above and
            # this ambiguous deferral look identical -- and keeping a candidate
            # the value guard positively proved wrong would corrupt the page.
            # This is the one path on which socr can say the refusal was soft:
            # the verifier reached AMBIGUOUS ("paired/spanning headers possible
            # — deferring to VLM") and the inner judge, not a deterministic
            # gate, is what refused. Marked BEFORE the structural gate runs, so
            # a gate rejection below can never be mistaken for this one.
            if not decision.accept:
                output.rejection_class = REJECTION_AMBIGUOUS_DEFERRED
            return self._apply_structural_gate(
                decision, output, page_num, words, rules, furniture=furniture
            )

        if vr.state == VerifierState.EXACT_PASS:
            # EXACT_PASS: ship immediately — no model needed
            self._emit_event(
                page_num=page_num,
                kind="native_table_verifier_exact_pass",
                engine=output.engine or "",
                detail="deterministic EXACT_PASS: row counts match + multiset clean",
                data={
                    "native_lane_count": vr.native_lane_count,
                    "output_col_count": vr.output_col_count,
                    "verifier_state": vr.state,
                },
            )
            decision = AcceptDecision(
                accept=True,
                reason="native_table_verifier: EXACT_PASS",
                confidence=1.0,
            )
            # #245: this is the one accepting exit where no model has seen the
            # page. The gate is told so; a header-attribution abstain on this
            # path delegates to the inner judge instead of shipping at 1.0.
            return self._apply_structural_gate(
                decision,
                output,
                page_num,
                words,
                rules,
                provider=provider,
                inner_consulted=False,
                furniture=furniture,
            )

        # No issue detected → delegate to inner judge
        decision = self._inner.assess(output, provider)
        # #262: record the DISPOSITION here too, exactly as the warn branch
        # above does. The verifier found nothing to refute -- no multiset
        # mismatch, no lane-count gap -- so if this reading was refused, the
        # inner judge alone refused it. That is a soft refusal by the same
        # definition #259 used, and without recording it the D3 floor cannot
        # tell it from a CERTAIN_FAIL and must assume the worst. Set BEFORE
        # ``_apply_structural_gate`` so a deterministic gate rejection below is
        # never mislabelled as a judge-only one (the gate returns a rejecting
        # decision unchanged and leaves ``rejection_class`` alone).
        if not decision.accept:
            output.rejection_class = REJECTION_JUDGE_ONLY
        return self._apply_structural_gate(
            decision, output, page_num, words, rules, furniture=furniture
        )

    def _apply_structural_gate(
        self,
        decision: AcceptDecision,
        output: PageOutput,
        page_num: int,
        words: list | None,
        rules: list[tuple[float, float, float]] | None = None,
        *,
        provider: ProviderProfile | None = None,
        inner_consulted: bool = True,
        furniture: DocumentFurniture | None = None,
    ) -> AcceptDecision:
        """GH-200: the winner-side structural/header check on whatever is ABOUT TO SHIP.

        Runs on every ACCEPTING path out of ``assess`` -- the early-delegation
        geometry-bypass paths (non-table page, missing ``get_fitz_page``,
        non-success/empty output, with ``words=None`` and ``rules=None``), the
        delegated-warn path, the deterministic EXACT_PASS accept, and the
        delegated-no-issue path. (Verifier exceptions are rejected fail-closed
        directly by GH-162 and do not reach the gate.) TR-3 (the multiset check
        above) proves the numbers are right; it is blind by construction to
        header loss, detached labels, and star-only row deletion (2026-08-15
        hand judgement, 4/4 damaged pages). A rejecting inner decision is
        returned unchanged -- there is nothing to gate on a page that is not
        shipping anyway.

        #245: ``inner_consulted`` says whether a model has already seen this
        page on the way here. On the EXACT_PASS path it has not. When the
        header-attribution term then ABSTAINS (borderless table, no drawn
        rule, non-unique anchor), the page is not verified -- it is
        unverifiable -- and an abstain must not ship at confidence 1.0. It is
        treated as AMBIGUOUS: the inner judge is consulted and its decision
        stands. A grounded verdict (HARD/SOFT/OK) keeps the short-circuit.

        #988 M2: pipe runs that transcribe page furniture (a site menu printed
        on every page) are left out of what is checked. When the page is
        accepted, those carrying no number are also left out of the text that
        ships, recorded as ``table_furniture_removed``; removing them is what
        lets the manifest backstop, which has no document, judge the accepted
        text the same way. A run with a number stays in the text.
        """
        if not decision.accept:
            return decision

        from socr.tables.furniture import (
            TABLE_FURNITURE_REMOVED_KIND,
            DocumentFurniture,
            strip_furniture_runs,
        )
        from socr.tables.header_attribution import HeaderVerdict
        from socr.tables.structure_check import table_header_verdicts, table_output_defect

        # #988 M2: page furniture the model wrote as a table is not judged.
        if furniture is None:
            furniture = DocumentFurniture()
        page_furniture = furniture.page_words(words)
        gated, _ = strip_furniture_runs(output.text, page_furniture)
        defect = table_output_defect(gated, words, rules, furniture.is_menu_band)
        if not defect and words:
            verdicts = table_header_verdicts(gated, words)
            if HeaderVerdict.UNVERIFIABLE in verdicts:
                delegated = not inner_consulted and provider is not None
                # Abstain, surfaced so its firing rate is measurable rather
                # than silently swallowed (#206/#207 notation-gap risk).
                self._emit_event(
                    page_num=page_num,
                    kind="table_header_unverifiable",
                    engine=output.engine or "",
                    detail=(
                        "header-attribution geometry chain abstained"
                        + ("; delegated to inner judge (#245)" if delegated else "")
                    ),
                    data={"verdicts": [v.value for v in verdicts], "delegated": delegated},
                )
                if delegated:
                    decision = self._inner.assess(output, provider)
                    if not decision.accept:
                        # Same disposition as the warn tier: the verifier
                        # could not refute the reading, the judge refused it.
                        output.rejection_class = REJECTION_AMBIGUOUS_DEFERRED
                        return decision

        if not defect:
            shipped, removed = strip_furniture_runs(output.text, page_furniture, keep_numbered=True)
            if removed:
                output.text = shipped
                self._emit_event(
                    page_num=page_num,
                    kind=TABLE_FURNITURE_REMOVED_KIND,
                    engine=output.engine or "",
                    detail=f"left out {len(removed)} page-furniture run(s) written as a table",
                    data={"runs": removed},
                )
            return decision

        self._emit_event(
            page_num=page_num,
            kind="table_structure_failed",
            engine=output.engine or "",
            detail=defect,
            data={"defect": defect},
        )
        return AcceptDecision(
            accept=False,
            reason=f"table_structure_failed: {defect}",
            confidence=0.0,
        )

    def _document_furniture(self, fitz_page) -> DocumentFurniture:
        """#988 M2: the page's document furniture, scanned once per document.

        ``get_fitz_page`` reopens the PDF on every call, so the scan is cached
        here by file name. A scan that raises leaves the gate as it was before
        (nothing removed, every header band owed), which is the stricter side.
        """
        from socr.tables.furniture import DocumentFurniture, document_furniture

        if fitz_page is None:
            return DocumentFurniture()
        doc = fitz_page.parent
        key = str(getattr(doc, "name", "") or "")
        if key not in self._furniture:
            try:
                self._furniture[key] = document_furniture(doc)
            except _VERIFIER_FATAL:
                raise
            except Exception as exc:
                logger.debug("page furniture scan failed: %s", exc)
                self._furniture[key] = DocumentFurniture()
        return self._furniture[key]

    def _maybe_repair_collapsed_headers(self, fitz_page, output: PageOutput, vr):
        """Repair too-narrow or collapsed multi-band headers when detected."""
        from socr.tables.header_repair import repair_table_headers_on_page
        from socr.tables.native_verifier import verify_native_table
        from socr.tables.reconcile import find_table_blocks

        blocks = find_table_blocks(output.text)
        if not blocks:
            return vr

        repaired_text, repair_count = repair_table_headers_on_page(fitz_page, output.text)
        if repair_count == 0:
            return vr

        # #688: the repair produced a NEW candidate, so it crosses the same
        # boundary before anything measures it -- the verifier included.
        output.text = repaired_text
        canonicalize_candidate(output)
        repaired_text = output.text
        new_vr = verify_native_table(fitz_page, repaired_text)
        self._emit_event(
            page_num=output.page_num,
            kind="table_header_repair",
            engine=output.engine or "",
            detail=(
                f"repaired {repair_count} too-narrow or collapsed table header(s) "
                f"(cols {vr.output_col_count}→{new_vr.output_col_count})"
            ),
            data={
                "repair_count": repair_count,
                "native_lane_count_before": vr.native_lane_count,
                "output_col_count_before": vr.output_col_count,
                "output_col_count_after": new_vr.output_col_count,
                "verifier_state_after": new_vr.state,
            },
        )
        return new_vr

    def _emit_event(self, page_num: int, kind: str, engine: str, detail: str, data: dict) -> None:
        if self._record_event is None:
            return
        try:
            from socr.core.audit_log import AuditEvent

            self._record_event(
                AuditEvent(page_num=page_num, kind=kind, engine=engine, detail=detail, data=data)
            )
        except Exception as exc:
            logger.debug("NativeTableVerifierJudge: failed to record audit event: %s", exc)
