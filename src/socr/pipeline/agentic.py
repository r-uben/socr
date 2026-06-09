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

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from socr.core.config import EngineType
from socr.core.providers import ProviderProfile
from socr.core.result import PageOutput, PageStatus

logger = logging.getLogger(__name__)

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
            continue
        tried += 1
        try:
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
                )
            )
            continue

        if remaining_budget is not None:
            remaining_budget -= prof.cost_per_page_usd

        decision = judge.assess(output, prof)
        attempts.append(
            ProviderAttempt(
                engine=prof.engine,
                output=output,
                cost_usd=prof.cost_per_page_usd,
                accepted=decision.accept,
                reason=decision.reason,
                raw_verdict=decision.raw_verdict,
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
        if output.status == PageStatus.ERROR or not output.text.strip():
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
        if output.status == PageStatus.ERROR or not output.text.strip():
            return AcceptDecision(accept=False, reason="empty/error output")
        image_path = self._render_image(output.page_num)
        verdict = self._judge.judge(image_path, output.text)
        return AcceptDecision(
            accept=bool(verdict.faithful),
            reason="; ".join(verdict.issues) if verdict.issues else "faithful",
            confidence=verdict.confidence,
            raw_verdict=verdict,
        )
