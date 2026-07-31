"""GH-96: the escalation canary.

Decides whether a table page's escalated re-read may replace the incumbent's.

The danger is demonstrated, not hypothetical. A vision model whose image read was
silently blocked returned 741 bytes of clean, confident, entirely fabricated fiscal
data at exit code 0 with empty stderr, and it satisfied every structural
instruction it had been given — so it would have scored *better* than the real 38%
output on every model-free structural heuristic available
(``docs/log/2026-07-30_obr-table-bakeoff.md``, Result 4). Structural plausibility
cannot distinguish a good transcription from a fabrication.

On a born-digital page the PDF's own text layer is a free oracle of true values.
The rule is two-sided and threshold-free:

    accept  iff  introduced(candidate) ⊆ introduced(incumbent)
            and  covered(candidate)    ⊇ covered(incumbent)

"Introduce nothing new, lose nothing verified."

**Both halves are load-bearing.** Containment alone is monotone in the wrong
direction: ``introduced(∅) = ∅ ⊆ anything``, so a candidate that times out and
emits three rows is accepted against *any* incumbent, including a near-perfect one.
Partial emission is the ordinary failure mode of a second-pass VLM under pressure,
so one-sided containment would silently trade a good table for a stub. The coverage
half closes that: a truncated or fabricated candidate cannot reproduce the
native-supported values the incumbent already found.

Set semantics, deliberately. Multiset counting rejects correct candidates on this
document class: hierarchical fiscal tables are dense with merged and spanning
cells, and a candidate that correctly expands a merged ``0.0`` across three year
columns emits a count the native layer never had. Measured — the multiset variant
rejected a real +75pp improvement over one surplus token. The multiset delta is
retained as diagnostics only.

Scoped to the native table's *value* tokens rather than every number on the page.
Page-scoped coverage rejects correct candidates for failing to reproduce incidental
tokens — footnote markers and years that a shredded incumbent grid emitted as
standalone cells and a correct candidate keeps attached to its label. Measured:
that cost three of nine genuine improvements.

What acceptance does NOT establish is stated on :func:`judge_escalation`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from socr.tables.native_rows import native_rows_from_page
from socr.tables.native_verifier import _normalize_numeric_token
from socr.tables.source_evidence import collect_table_tokens


@dataclass(frozen=True)
class CanaryVerdict:
    """Outcome of judging one escalation candidate."""

    accepted: bool
    reason: str
    usable: bool = True  # False when the page offers no oracle to judge against
    introduced: tuple[str, ...] = ()  # candidate tokens absent from the oracle
    lost: tuple[str, ...] = ()  # oracle-backed tokens the incumbent had and the candidate dropped
    oracle_size: int = 0
    multiset_surplus: dict = field(default_factory=dict)  # diagnostics only, never gating


def native_value_oracle(page) -> frozenset[str]:
    """Normalized numeric tokens that are genuine table *values* on *page*.

    Empty when the page is scanned, has no located table, or exposes no native
    text — in which case there is no oracle and no escalation may be judged.
    """
    try:
        rows = native_rows_from_page(page)
    except Exception:
        return frozenset()
    return frozenset(_normalize_numeric_token(v) for row in rows for v in row.values)


def table_value_tokens(markdown: str) -> Counter:
    """Normalized numeric tokens emitted by *markdown*'s tables, with counts."""
    tokens = collect_table_tokens(markdown)
    return Counter(tokens.numeric) if tokens else Counter()


def judge_escalation(page, incumbent_markdown: str, candidate_markdown: str) -> CanaryVerdict:
    """Decide whether *candidate_markdown* may replace *incumbent_markdown*.

    Guarantee on acceptance, stated so it cannot be over-read: over numeric tokens
    the page's native text layer can adjudicate, the candidate introduces no token
    absent from that layer beyond those the incumbent also introduced, and retains
    every native-backed value the incumbent's tables contained.

    It does **not** guarantee correct cell, row or column alignment, correct label
    pairing, correct value multiplicity, completeness beyond the incumbent's,
    correct sign in context, or anything about values the native layer does not
    contain. In particular a candidate that places every correct number on the
    wrong label passes perfectly — token containment proves *not invented*, never
    *correctly aligned*.
    """
    oracle = native_value_oracle(page)
    if not oracle:
        return CanaryVerdict(
            accepted=False,
            usable=False,
            reason="no native value oracle for this page; escalation cannot be judged",
        )

    incumbent = table_value_tokens(incumbent_markdown)
    candidate = table_value_tokens(candidate_markdown)

    inc_introduced = set(incumbent) - oracle
    cand_introduced = set(candidate) - oracle
    inc_covered = set(incumbent) & oracle
    cand_covered = set(candidate) & oracle

    novel = sorted(cand_introduced - inc_introduced)
    lost = sorted(inc_covered - cand_covered)

    surplus = {
        tok: count - incumbent.get(tok, 0)
        for tok, count in candidate.items()
        if count > incumbent.get(tok, 0) and tok in oracle
    }

    if novel:
        return CanaryVerdict(
            accepted=False,
            reason=(
                f"candidate introduced {len(novel)} value(s) absent from the page's "
                f"native text: {novel[:5]}"
            ),
            introduced=tuple(novel),
            lost=tuple(lost),
            oracle_size=len(oracle),
            multiset_surplus=surplus,
        )

    if lost:
        return CanaryVerdict(
            accepted=False,
            reason=(
                f"candidate dropped {len(lost)} native-backed value(s) the incumbent "
                f"had: {lost[:5]}"
            ),
            introduced=(),
            lost=tuple(lost),
            oracle_size=len(oracle),
            multiset_surplus=surplus,
        )

    return CanaryVerdict(
        accepted=True,
        reason=(
            f"introduced nothing new and retained all {len(inc_covered)} native-backed "
            f"value(s) the incumbent had"
        ),
        oracle_size=len(oracle),
        multiset_surplus=surplus,
    )
