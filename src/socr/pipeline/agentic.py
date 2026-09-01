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
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from socr.core.config import EngineType
from socr.core.providers import ProviderProfile
from socr.core.result import (
    REJECTION_AMBIGUOUS_DEFERRED,
    REJECTION_JUDGE_ONLY,
    PageOutput,
    PageStatus,
)

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


class PageJudge(Protocol):
    """Decides whether a page's OCR output is good enough to stop escalating."""

    def assess(self, output: PageOutput, provider: ProviderProfile) -> AcceptDecision: ...


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


def _error_output(page_num: int, msg: str) -> PageOutput:
    return PageOutput(
        page_num=page_num, text="", status=PageStatus.ERROR, error=msg, audit_passed=False
    )


def _best_effort(attempts: list[ProviderAttempt], page_num: int) -> ProviderAttempt:
    """When nothing was accepted, keep the most trustworthy attempt.

    Prefer an attempt that passed its own audit, then highest confidence, then
    most words, then the last (most-escalated) attempt. Never return empty if a
    non-empty attempt exists.
    """
    usable = [a for a in attempts if a.output.text.strip()]
    pool = usable or attempts
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


def route_page(
    page_num: int,
    ladder: list[ProviderProfile],
    run_provider: RunProvider,
    judge: PageJudge,
    *,
    max_attempts: int = 0,
    remaining_budget: float | None = None,
    provider_timeout: dict[EngineType, float] | None = None,
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
                ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = ex.submit(run_provider, prof, page_num)
                try:
                    output = future.result(timeout=timeout_sec)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    # Abandon the executor without waiting for the stalled thread:
                    # wait=False lets us escalate immediately; the daemon thread
                    # will be cleaned up when the process exits or the thread
                    # eventually unblocks.
                    ex.shutdown(wait=False)
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
                            reason="provider timeout",
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
                    ex.shutdown(wait=False)
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
        )


class SourceEvidenceTableJudge(_UnverifiedTableRejection):
    """Fail-closed source-evidence gate for VLM-emitted markdown tables (GH-90).

    Runs BEFORE the inner judge chain on ANY model output that contains markdown
    table blocks, regardless of ``PageState.has_tables``.  Born-digital pages
    with native PyMuPDF words defer to ``NativeTableVerifierJudge``; scanned pages
    with no native words verify cell tokens against local non-generative evidence
    (page/crop raster + optional classical OCR).  Unsupported or unverifiable
    tables hard-reject before heuristic acceptance.
    """

    def __init__(
        self,
        inner: PageJudge,
        get_fitz_page: Callable[[int], object] | None,
        record_event: Callable[[object], None] | None = None,
        *,
        ocr_image_fn: Callable[[object], str] | None = None,
    ) -> None:
        self._inner = inner
        self._get_fitz_page = get_fitz_page
        self._record_event = record_event
        self._ocr_image_fn = ocr_image_fn

    def assess(self, output: PageOutput, provider: ProviderProfile) -> AcceptDecision:
        from socr.tables.reconcile import find_table_blocks
        from socr.tables.source_evidence import verify_scanned_table

        if not find_table_blocks(output.text or ""):
            return self._inner.assess(output, provider)

        if output.status != PageStatus.SUCCESS or not output.text.strip():
            return self._inner.assess(output, provider)

        if self._get_fitz_page is None:
            return self._inner.assess(output, provider)

        page_num = output.page_num
        try:
            fitz_page = self._get_fitz_page(page_num)
            result = verify_scanned_table(
                fitz_page,
                output.text,
                ocr_image_fn=self._ocr_image_fn,
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

        if result.deferred:
            return self._inner.assess(output, provider)

        if not result.verifiable or not result.passed:
            from socr.core.result import FailureMode

            self._emit_event(
                page_num=page_num,
                kind="source_evidence_table_reject",
                engine=output.engine or "",
                detail=result.reason,
                data={
                    "verifiable": result.verifiable,
                    "passed": result.passed,
                },
            )
            output.audit_passed = False
            output.status = PageStatus.ERROR
            output.failure_mode = FailureMode.HALLUCINATION
            return AcceptDecision(
                accept=False,
                reason=f"source_evidence_table: {result.reason}",
                confidence=0.0,
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

    Scanned pages bypass cleanly (empty get_text("words")).

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

    def assess(self, output: PageOutput, provider: ProviderProfile) -> AcceptDecision:
        from socr.tables.native_verifier import (
            VerifierState,
            describe_drift,
            verify_native_table,
        )

        page_num = output.page_num

        # Only run the verifier on born-digital table pages
        if not self._is_table_page(page_num) or self._get_fitz_page is None:
            return self._inner.assess(output, provider)

        if output.status != PageStatus.SUCCESS or not output.text.strip():
            return self._inner.assess(output, provider)

        try:
            fitz_page = self._get_fitz_page(page_num)
            words = fitz_page.get_text("words") if fitz_page is not None else None
            # GH-212: the header term needs the page's drawn rules. Precompute
            # them here so structure_check stays pure and never sees a page.
            from socr.tables.locate import _horizontal_rules

            rules = _horizontal_rules(fitz_page) if fitz_page is not None else None
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
            return self._apply_structural_gate(decision, output, page_num, words, rules)

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
            return self._apply_structural_gate(decision, output, page_num, words, rules)

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
        return self._apply_structural_gate(decision, output, page_num, words, rules)

    def _apply_structural_gate(
        self,
        decision: AcceptDecision,
        output: PageOutput,
        page_num: int,
        words: list | None,
        rules: list[tuple[float, float, float]] | None = None,
    ) -> AcceptDecision:
        """GH-200: the winner-side structural/header check on whatever is ABOUT TO SHIP.

        Runs on every ACCEPTING path out of ``assess`` -- the delegated-warn
        path, the deterministic EXACT_PASS accept, the delegated-no-issue
        path, and (grid-shape term only, ``words=None``) the verifier-
        exception path. TR-3 (the multiset check above) proves the numbers
        are right; it is blind by construction to header loss, detached
        labels, and star-only row deletion (2026-08-15 hand judgement, 4/4
        damaged pages). A rejecting inner decision is returned unchanged --
        there is nothing to gate on a page that is not shipping anyway.
        """
        if not decision.accept:
            return decision

        from socr.tables.header_attribution import HeaderVerdict
        from socr.tables.structure_check import table_header_verdicts, table_output_defect

        defect = table_output_defect(output.text, words, rules)
        if not defect and words:
            verdicts = table_header_verdicts(output.text, words)
            if HeaderVerdict.UNVERIFIABLE in verdicts:
                # Abstain, surfaced so its firing rate is measurable rather
                # than silently swallowed (#206/#207 notation-gap risk).
                self._emit_event(
                    page_num=page_num,
                    kind="table_header_unverifiable",
                    engine=output.engine or "",
                    detail="header-attribution geometry chain abstained",
                    data={"verdicts": [v.value for v in verdicts]},
                )

        if not defect:
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

        output.text = repaired_text
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
