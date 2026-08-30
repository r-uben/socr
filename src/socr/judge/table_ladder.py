"""GH-353 TICKET-A4: the ladder state machine and the page reducer.

Pure functions over injected rung callables — no I/O, no config reads. The
rung callables (A2's ollama rung, A3's gemini rung) and the crop path /
markdown they judge are the caller's (B1's) responsibility; this module only
owns the S1/S2 transition logic per table and the multi-table page
reduction.

Per-rung outcome vocabulary (design: ``docs/log/2026-08-30_table-judge-ladder.md``):

- **A** — S1 ok, verdict PASS.
- **B** — S1 ok, verdict FAIL: escalate as *tiebreak* (next rung sees the
  findings).
- **C** — S1 failed (¬S1): escalate as *substitute* (next rung gets fresh
  eyes, no prior verdict).

Per-table resolution:

- **A, high confidence** — accept immediately at the current rung.
- **A, low confidence** — needs confirmation: escalate to the next rung
  with no findings (nothing to complain about). If that was the last rung,
  the unanimous-PASS-so-far result stands and the table is accepted.
- **B** — escalate as tiebreak. Exhausting the ladder on B (no more rungs)
  is a content problem: ``TABLE_REJECTED``.
- **C** — escalate as substitute. Exhausting the ladder on C is an infra
  problem: ``TABLE_UNVERIFIED``.

Any rung's outright PASS (high or low confidence) reached with rungs still
available to confirm, or reached at the last rung, ends the ladder in
``TABLE_ACCEPTED`` — a table is never held hostage by an earlier B or C once
a later rung actually looked and approved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

from socr.judge.table_verdict import Finding, RungCallable, RungResult, TableJudgeVerdict


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
    ``crop_path``/``markdown`` and the ``prior_findings`` carried from the
    previous rung's outcome (a FAIL's findings on tiebreak, ``None``
    otherwise). Returns the full history plus the terminal outcome.
    """
    if not rungs:
        raise ValueError("run_table_ladder requires at least one rung")

    rung_results: list[RungResult] = []
    prior_findings: list[Finding] | None = None
    last_index = len(rungs) - 1

    for index, rung in enumerate(rungs):
        result = rung(crop_path, markdown, prior_findings)
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
            prior_findings = None
            continue

        verdict = result.verdict
        assert verdict is not None  # ok=True guarantees a verdict (A1 contract)

        if verdict.passed:
            # A — PASS. High confidence (or the last rung, unanimous so
            # far) accepts outright; low confidence with rungs remaining
            # needs one more confirmation, carrying no findings.
            if verdict.confidence == "high" or is_last:
                return TableLadderResult(
                    table_id=table_id,
                    outcome=TableLadderOutcome.ACCEPTED,
                    rung_results=rung_results,
                    final_verdict=verdict,
                )
            prior_findings = None
            continue

        # B — FAIL. Tiebreak at the next rung with findings attached;
        # exhausting the ladder on FAIL is a content problem.
        if is_last:
            return TableLadderResult(
                table_id=table_id,
                outcome=TableLadderOutcome.REJECTED,
                rung_results=rung_results,
                final_verdict=verdict,
            )
        prior_findings = verdict.findings
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
