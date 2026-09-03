"""GH-353 TICKET-A4: the ladder state machine and the page reducer.

Pure functions over injected rung callables — no I/O, no config reads. The
rung callables (A2's ollama rung, A3's gemini rung) and the crop path /
markdown they judge are the caller's (B1's) responsibility; this module only
owns the S1/S2 transition logic per table and the multi-table page
reduction.

Terminals are pinned by GH-359 (``docs/log/2026-08-31_gh359-ladder-terminals.md``),
which supersedes the contradictory sentences in the 2026-08-30 design note
for these seven questions.

Per-rung outcome vocabulary:

- **A** — S1 ok, verdict PASS.
- **B** — S1 ok, verdict FAIL: escalate as *tiebreak* (next rung is called;
  GH-359 ruling 4: it does **not** see the findings).
- **C** — S1 failed (¬S1): escalate as *substitute* (next rung is also
  called with no prior verdict).

Per-table resolution (GH-359):

- **A, high confidence** — accept immediately at the current rung.
- **A, low confidence** — needs corroboration from a *real PASS witness*
  (any confidence). Escalate with no findings. If that was the last rung,
  the table is accepted only if the immediately preceding rung was itself
  a real PASS; a lone low-confidence PASS exhausts to ``TABLE_UNVERIFIED``
  (ruling 1).
- **B** — escalate as tiebreak. CLI₂ may overrule CLI₁ FAIL with a
  high-confidence PASS (ruling 2: "FAIL trusted at any rung" is dropped).
  Exhausting the ladder on B is a content problem: ``TABLE_REJECTED``.
  Mixed B then C (CLI₂ never looked) is ``TABLE_UNVERIFIED`` (ruling 3).
- **C** — escalate as substitute. Exhausting the ladder on C is an infra
  problem: ``TABLE_UNVERIFIED``.

Judge input at every rung is crop + markdown. Nothing else (ruling 4).
``prior_findings`` is always ``None``; the RungCallable parameter remains
for signature compatibility.

A high-confidence PASS at any rung ends the ladder in ``TABLE_ACCEPTED``
immediately — a table is never held hostage by an earlier B or C once a
later rung actually looked and approved with confidence. A low-confidence
PASS only accepts when a preceding rung already answered PASS (any
confidence): two real witnesses in agreement, never one weak witness
standing in for a corroboration that an infra failure could not provide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

from socr.judge.table_verdict import (
    RungCallable,
    RungResult,
    TableJudgeVerdict,
    is_availability_exception,
)

logger = logging.getLogger(__name__)


class TableLadderOutcome(str, Enum):
    """The three terminal dispositions a table (or a page) can reach.

    Reused for both the per-table ladder result and the page reducer's
    result — the reducer's rule (any REJECTED wins, else any UNVERIFIED,
    else ACCEPTED) is expressed over the same three values.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"


@dataclass
class TableLadderResult:
    """One table's full ladder run: every rung's answer plus the verdict.

    ``rung_results`` keeps every rung invoked, in order, for the sidecar /
    audit trail — never just the deciding one. ``final_verdict`` is the
    ``TableJudgeVerdict`` that produced ACCEPTED or REJECTED (``None`` for
    UNVERIFIED, since nobody produced one).
    """

    table_id: str
    outcome: TableLadderOutcome
    rung_results: list[RungResult] = field(default_factory=list)
    final_verdict: TableJudgeVerdict | None = None

    @property
    def accepted(self) -> bool:
        return self.outcome is TableLadderOutcome.ACCEPTED


@dataclass
class PageLadderResult:
    """The page-level reduction over every table's ladder result.

    ``table_results`` keeps ALL per-table results (for the sidecar/audit
    events), not just the ones that determined the page outcome.
    """

    outcome: TableLadderOutcome
    table_results: list[TableLadderResult] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.outcome is TableLadderOutcome.ACCEPTED


def run_table_ladder(
    rungs: Sequence[RungCallable],
    crop_path: Path,
    markdown: str,
    table_id: str = "",
) -> TableLadderResult:
    """Run one table through the ladder of rung callables in order.

    ``rungs`` must be non-empty. Each rung is called with the same
    ``crop_path``/``markdown`` and ``prior_findings=None`` (GH-359 ruling 4:
    judge input is crop + markdown, nothing else). Returns the full history
    plus the terminal outcome.
    """
    if not rungs:
        raise ValueError("run_table_ladder requires at least one rung")

    rung_results: list[RungResult] = []
    # Whether the immediately preceding rung answered with a real PASS
    # verdict (any confidence) — the corroboration a low-confidence PASS
    # needs at the last rung. False before the first rung: there is no
    # preceding witness yet.
    prior_was_pass = False
    last_index = len(rungs) - 1

    for index, rung in enumerate(rungs):
        try:
            result = rung(crop_path, markdown, None)
        except Exception as exc:
            rung_id = (
                getattr(rung, "rung_id", "") or getattr(rung, "__name__", "") or type(exc).__name__
            )
            # Cold review round 2, finding 2. A rung is contractually
            # non-raising, so ANY exception here is unexpected -- but the cause
            # decides whether it is an outage. A transport failure latches and
            # is retried when the rung returns; a TypeError/AssertionError from
            # our own code is deterministic, and latching it would make every
            # resume re-run the ladder to reproduce the same crash. The
            # traceback is logged either way; the table still ends UNVERIFIED.
            unavailable = is_availability_exception(exc)
            logger.warning(
                "table judge rung %s raised %s: %s (unavailable=%s)",
                rung_id,
                type(exc).__name__,
                exc,
                unavailable,
                exc_info=True,
            )
            result = RungResult(
                rung=rung_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                unavailable=unavailable,
            )
        rung_results.append(result)
        is_last = index == last_index

        if not result.ok:
            # C — ¬S1: substitute at the next rung with fresh eyes.
            if is_last:
                return TableLadderResult(
                    table_id=table_id,
                    outcome=TableLadderOutcome.UNVERIFIED,
                    rung_results=rung_results,
                    final_verdict=None,
                )
            prior_was_pass = False
            continue

        verdict = result.verdict
        assert verdict is not None  # ok=True guarantees a verdict (A1 contract)

        if verdict.passed:
            # A — PASS. High confidence accepts outright at any rung. Low
            # confidence needs corroboration: at the last rung it accepts
            # only if the PRECEDING rung was already a real PASS (two
            # witnesses in agreement); otherwise a lone, uncorroborated
            # weak PASS cannot verify the table on its own (GH-359 ruling 1).
            if verdict.confidence == "high":
                return TableLadderResult(
                    table_id=table_id,
                    outcome=TableLadderOutcome.ACCEPTED,
                    rung_results=rung_results,
                    final_verdict=verdict,
                )
            if is_last:
                if prior_was_pass:
                    return TableLadderResult(
                        table_id=table_id,
                        outcome=TableLadderOutcome.ACCEPTED,
                        rung_results=rung_results,
                        final_verdict=verdict,
                    )
                return TableLadderResult(
                    table_id=table_id,
                    outcome=TableLadderOutcome.UNVERIFIED,
                    rung_results=rung_results,
                    final_verdict=None,
                )
            prior_was_pass = True
            continue

        # B — FAIL. Tiebreak at the next rung; GH-359 ruling 2: CLI2 may
        # overrule. Exhausting the ladder on FAIL is a content problem.
        # Findings are NOT forwarded (ruling 4).
        if is_last:
            return TableLadderResult(
                table_id=table_id,
                outcome=TableLadderOutcome.REJECTED,
                rung_results=rung_results,
                final_verdict=verdict,
            )
        prior_was_pass = False
        continue

    # Unreachable: the loop always returns on its last iteration.
    raise AssertionError("run_table_ladder fell through without a terminal result")


def reduce_page_ladder(table_results: Sequence[TableLadderResult]) -> PageLadderResult:
    """Reduce every table's ladder result on a page to one page outcome.

    Rule: any REJECTED table makes the page REJECTED; else any UNVERIFIED
    table makes the page UNVERIFIED; else the page is ACCEPTED. A page with
    no tables is ACCEPTED (nothing to reject or fail to verify). Every
    per-table result is kept in the returned ``table_results`` regardless
    of which one decided the page outcome.
    """
    results = list(table_results)

    if any(result.outcome is TableLadderOutcome.REJECTED for result in results):
        outcome = TableLadderOutcome.REJECTED
    elif any(result.outcome is TableLadderOutcome.UNVERIFIED for result in results):
        outcome = TableLadderOutcome.UNVERIFIED
    else:
        outcome = TableLadderOutcome.ACCEPTED

    return PageLadderResult(outcome=outcome, table_results=results)
