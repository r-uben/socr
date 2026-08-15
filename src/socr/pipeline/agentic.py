"""Cost-aware agentic per-page OCR routing.

The core of "best cost-effective provider on the go": for each page, try the
cheapest available provider, let a judge decide whether the output is good
enough, and escalate up the cost ladder only when it is not. Stop at the first
accepted output; if none is accepted, keep the best attempt.

Control flow lives in Python (the panel was unanimous: the LLM is a stateless
per-page decision function, not the orchestrator). ``route_page`` is pure given
its two injected dependencies:

  - ``run_provider(engine, page_num) -> PageOutput`` — actually OCR one page with
    one engine. The orchestrator wires the real implementation (render + engine
    call); tests pass a stub.
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
from socr.core.result import PageOutput, PageStatus

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

RunProvider = Callable[[EngineType, int], PageOutput]


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
        run_provider: runs one engine on one page.
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
                future = ex.submit(run_provider, prof.engine, page_num)
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
                        )
                    )
                    continue
                else:
                    ex.shutdown(wait=False)
            else:
                output = run_provider(prof.engine, page_num)
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


class SourceEvidenceTableJudge:
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
        except Exception as exc:
            logger.warning(
                "SourceEvidenceTableJudge: verifier raised on p%d (%s); delegating to inner",
                page_num,
                exc,
            )
            return self._inner.assess(output, provider)

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


class NativeTableVerifierJudge:
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
        from socr.tables.native_verifier import VerifierState, verify_native_table

        page_num = output.page_num

        # Only run the verifier on born-digital table pages
        if not self._is_table_page(page_num) or self._get_fitz_page is None:
            return self._inner.assess(output, provider)

        if output.status != PageStatus.SUCCESS or not output.text.strip():
            return self._inner.assess(output, provider)

        try:
            fitz_page = self._get_fitz_page(page_num)
            words = fitz_page.get_text("words") if fitz_page is not None else None
            vr = verify_native_table(fitz_page, output.text)
            vr = self._maybe_repair_collapsed_headers(fitz_page, output, vr)
        except Exception as exc:
            logger.warning(
                "NativeTableVerifierJudge: verifier raised on p%d (%s); delegating to inner judge",
                page_num,
                exc,
            )
            # GH-200: geometry failed, so the header-attribution term cannot
            # run -- but the grid-shape term (structural_gate_fires) is
            # string-only and needs no words. Run it anyway so a ragged
            # candidate is still rejected rather than accepted by silent
            # delegation to the inner judge.
            decision = self._inner.assess(output, provider)
            return self._apply_structural_gate(decision, output, page_num, words=None)

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
            return self._apply_structural_gate(decision, output, page_num, words)

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
            return self._apply_structural_gate(decision, output, page_num, words)

        # No issue detected → delegate to inner judge
        decision = self._inner.assess(output, provider)
        return self._apply_structural_gate(decision, output, page_num, words)

    def _apply_structural_gate(
        self,
        decision: AcceptDecision,
        output: PageOutput,
        page_num: int,
        words: list | None,
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

        defect = table_output_defect(output.text, words)
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
        """Rebuild collapsed multi-band headers from native geometry when detected."""
        from socr.tables.header_repair import (
            detect_header_column_collapse,
            repair_table_headers_on_page,
        )
        from socr.tables.native_verifier import verify_native_table
        from socr.tables.reconcile import find_table_blocks

        blocks = find_table_blocks(output.text)
        if not blocks or not any(detect_header_column_collapse(b.grid)[0] for b in blocks):
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
                f"rebuilt {repair_count} collapsed table header(s) from native geometry "
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
