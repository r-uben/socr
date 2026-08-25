"""GH-96: deciding whether an escalated table may replace the incumbent's.

The escalation lane re-reads a suspect table page with a second engine. This module
decides whether the result is an improvement, and it is deliberately the *only*
place that decision is made.

## Why exactness, and not the canary

``escalation_canary`` checks that a candidate invents nothing the page's native text
layer cannot support. That is necessary — a vision model whose image read was
silently blocked once returned confident, entirely fabricated fiscal data at exit 0
— but it is not sufficient, and measurement showed why.

Escalating with socr's own ``gemini-ocr`` regresses four pages of the reference
document, two of which the incumbent had **perfect** (page 59: 100.0% -> 79.3%,
page 61: 82.1% -> 50.0%). Every number in those regressed outputs is present in the
native layer, so token containment is satisfied and the canary accepts them. A
regression is not an invention.

"Escalation is never worse" turned out to be a property of the *oracle* used in the
bake-off, not of the engine the lane will actually call.

On a born-digital page, hierarchy-aware exactness against the native text layer is
computable at runtime, model-free, at zero cost — the same property that made it the
best available trigger. So it is used as the gate as well:

    accept iff exactness(candidate) > exactness(incumbent)

Measured over 16 table pages, escalating the 13 the trigger fires on:

    socr baseline, no escalation      45.0%
    canary-only accept rule           81.7%
    accept iff exactness improves     85.0%

These three numbers predate #123 TICKET-B2's lane-aware column alignment (`exact`
used to compare markdown and ground truth positionally, column-index to
column-index; it now compares under each side's best monotone lane-to-column map).
The relative ordering the decision relies on — canary-only < exactness-gate — is
not expected to change, since B2 only tightens what counts as a match, but the
literal percentages above are stale and want a re-measurement against the same
16-page corpus before being cited again.

**#123 TICKET-C1 refines the rule to be lexicographic, not a bare `exact` compare.**
A wider candidate can raise `exact` while dropping a whole ground-truth lane the
narrower incumbent covered (see ``table_exactness``'s "map-search freedom") — that
is content loss and must never win on exactness alone. The gate therefore ranks
``unexplained_lanes`` ahead of ``exact``:

    fewer unexplained lanes  -> accept
    more unexplained lanes   -> reject, regardless of exactness
    equal unexplained lanes  -> exactness is the tiebreak (the rule above)

Better, and **monotone by construction**: no page can leave worse than it arrived.
The canary cannot offer that. It also subsumes the canary's own job — a fabrication
scores near zero and a truncated candidate scores low, so both are rejected on
quality without needing a separate fabrication test.

## Where the canary still earns its place

Exactness needs ground truth. When the native layer will not parse into scorable
rows — a scanned page, a table the locator misses, a layout the row parser cannot
read — there is no score to compare and the quality gate is blind. The canary is
the fallback there: it cannot rank two candidates, but it can still refuse an
invention. That is strictly weaker, and :class:`EscalationDecision` records which
gate ran so the audit trail never conflates them.
"""

from __future__ import annotations

from dataclasses import dataclass

from socr.core.table_grid import score_page
from socr.tables.escalation_canary import judge_escalation

# How the decision was reached. Recorded per page so a reader can tell a
# quality-gated acceptance from a merely-not-fabricated one.
GATE_EXACTNESS = "exactness"
GATE_CANARY = "canary"
GATE_NONE = "none"

# Fewest ground-truth rows that make a comparison meaningful. One row is a parser
# artifact, not a table; below this there is nothing to rank two candidates by.
_MIN_GROUND_TRUTH_ROWS = 2


@dataclass(frozen=True)
class EscalationDecision:
    """Whether to replace the incumbent page text with the escalated one."""

    accepted: bool
    gate: str
    reason: str
    incumbent_pct: float | None = None
    candidate_pct: float | None = None
    # #123 TICKET-C1: carried through so a caller can surface content loss even on
    # a rejected/accepted decision without re-scoring the page.
    incumbent_unexplained_lanes: int = 0
    candidate_unexplained_lanes: int = 0
    incumbent_ceiling_note: str = ""
    candidate_ceiling_note: str = ""

    @property
    def delta(self) -> float | None:
        if self.incumbent_pct is None or self.candidate_pct is None:
            return None
        return round(self.candidate_pct - self.incumbent_pct, 1)


def decide_escalation(page, incumbent_markdown: str, candidate_markdown: str) -> EscalationDecision:
    """Accept the candidate only if it measurably improves the page.

    Primary gate is hierarchy-aware cell exactness against the page's own native
    text layer. When that ground truth is unscorable the canary is used instead,
    which can refuse an invention but cannot tell a better table from a worse one —
    the returned ``gate`` says which ran.

    Ties are rejected. An equal-scoring candidate is not an improvement, and
    keeping the incumbent avoids churning the corpus and re-flushing fragments for
    no measured gain.
    """
    incumbent = score_page(page, incumbent_markdown)
    candidate = score_page(page, candidate_markdown)

    # Gate on whether the GROUND TRUTH is usable, not on whether either report is
    # "scorable". A report is marked unscorable when no prediction label matched
    # any ground-truth row — but that is evidence the *prediction* is bad, not that
    # the measurement is invalid, and both sides are scored against the same truth.
    # Gating on it would hand the incumbent's worst failures to the weaker canary:
    # pages whose labels are shredded score 0 and would be waved through rather than
    # compared, including two real 0% -> ~86% recoveries.
    if incumbent.gt_rows >= _MIN_GROUND_TRUTH_ROWS:
        # #123 TICKET-C1: unexplained lanes are content loss, and rank ahead of
        # exactness rather than being folded into it — a candidate can score more
        # exact cells while dropping a whole column the native layer supports, and
        # that must never win. "Zero versus non-zero, fewer versus more" only:
        # exactness is the tiebreak, not a way to outvote a lane increase.
        lane_delta = candidate.unexplained_lanes - incumbent.unexplained_lanes
        if lane_delta > 0:
            improved = False
            reason = (
                f"candidate increases unexplained lanes "
                f"({incumbent.unexplained_lanes} -> {candidate.unexplained_lanes}); "
                f"rejected regardless of exactness ({incumbent.pct}% -> {candidate.pct}%)"
            )
        elif lane_delta < 0:
            improved = True
            reason = (
                f"candidate decreases unexplained lanes "
                f"({incumbent.unexplained_lanes} -> {candidate.unexplained_lanes}); "
                f"accepted ({incumbent.pct}% -> {candidate.pct}%)"
            )
        else:
            improved = candidate.exact > incumbent.exact
            reason = (
                f"exactness {incumbent.pct}% -> {candidate.pct}% "
                f"({candidate.exact} vs {incumbent.exact} of {incumbent.cells} cells)"
            )
        return EscalationDecision(
            accepted=improved,
            gate=GATE_EXACTNESS,
            incumbent_pct=incumbent.pct,
            candidate_pct=candidate.pct,
            reason=reason,
            incumbent_unexplained_lanes=incumbent.unexplained_lanes,
            candidate_unexplained_lanes=candidate.unexplained_lanes,
            incumbent_ceiling_note=incumbent.ceiling_note,
            candidate_ceiling_note=candidate.ceiling_note,
        )

    # No usable ground truth: fall back to the weaker guarantee.
    verdict = judge_escalation(page, incumbent_markdown, candidate_markdown)
    if not verdict.usable:
        return EscalationDecision(
            accepted=False,
            gate=GATE_NONE,
            reason=(
                "page offers neither scorable ground truth nor a value oracle; "
                "escalation cannot be judged, keeping the incumbent"
            ),
            incumbent_ceiling_note=incumbent.ceiling_note,
            candidate_ceiling_note=candidate.ceiling_note,
        )
    return EscalationDecision(
        accepted=verdict.accepted,
        gate=GATE_CANARY,
        reason=f"ground truth unscorable; canary only: {verdict.reason}",
        incumbent_ceiling_note=incumbent.ceiling_note,
        candidate_ceiling_note=candidate.ceiling_note,
    )
