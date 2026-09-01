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


# ---------------------------------------------------------------------------
# GH-326: the presence gate.
#
# This is what the native text layer is still allowed to say about a model's
# table, and nothing more.
#
# Measurement forced the narrowing. On 13 rows transcribed blind from the page
# images (``docs/log/2026-08-30_model-vs-native-table-rows.md``), native was the
# LEAST accurate of three readings -- 8/13 rows exact against qwen's 12/13 and
# gemini's 11/13, losing row labels where neither model did. Its positional
# assertions are therefore inadmissible: a checker that grades a model against
# native's row geometry convicts correct output, which is exactly what happened
# ten times on a page later verified cell by cell.
#
# What survives is presence. The word layer can still say "this number is on the
# page" or "this number is not", because that claim needs no rows, no columns and
# no labels -- only the tokens themselves.
# ---------------------------------------------------------------------------

#: The gate's possible answers. UNVERIFIABLE is a first-class outcome, not a
#: failure: an absence of evidence must never be reported as a conviction.
PRESENCE_OK = "ok"
PRESENCE_INVENTED = "invented"
PRESENCE_LOST = "lost"
PRESENCE_UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class PresenceVerdict:
    """What the word layer can attest about a candidate's numbers.

    Deliberately says nothing about WHERE any value sits. There is no row, column
    or label field on this class, and adding one would reintroduce the assertion
    the measurement disqualified.
    """

    status: str
    invented: tuple[str, ...] = ()  # in the candidate, not on the page
    lost: tuple[str, ...] = ()  # on the page, absent from the candidate
    oracle_size: int = 0
    reason: str = ""

    @property
    def blocks_success(self) -> bool:
        """Only invention blocks. Loss flags, absence of evidence does neither.

        Asymmetric on purpose. A value the model wrote that is nowhere on the page
        is the failure #270 documents and cannot be explained away. A value the
        page has and the candidate lacks may be a real omission OR a table the
        candidate legitimately split, so it is surfaced and not gated on.
        """
        return self.status == PRESENCE_INVENTED


def native_value_counts(page) -> Counter:
    """Numeric tokens on *page*'s word layer, WITH multiplicity.

    Counts rather than the set :func:`native_value_oracle` returns, because the
    #270 failure is a substitution: one occurrence of a coefficient becomes two,
    or one value is overwritten by its neighbour. Set containment nets those to
    zero -- ``{1.02} ⊆ {1.02}`` however many times the model wrote it -- so a
    set-based gate is blind to the very failure this gate exists to catch.
    """
    try:
        rows = native_rows_from_page(page)
    except Exception:
        return Counter()
    return Counter(_normalize_numeric_token(v) for row in rows for v in row.values)


def native_text_value_counts(native_text: str) -> Counter:
    """Numeric tokens in a page's own extracted text, with multiplicity.

    The word-layer oracle needs a ``fitz`` page, which the disposition site does
    not have -- ``PageState`` carries only ``native_text``. That turns out not to
    matter, and the reason is the point of this whole gate: **presence needs
    tokens, not layout.** ``native_text`` contains every number on the page even
    when their arrangement is wrong, and arrangement is exactly what the
    measurement disqualified native from asserting.

    So this is the same oracle read off a different surface, not a weaker one.
    """
    import re as _re

    from socr.tables.source_evidence import collect_table_tokens

    tokens = collect_table_tokens(native_text or "")
    if tokens and tokens.numeric:
        # GH-355 hole 3: UNION with the whole-page pass, never narrow to the
        # grid alone. This caller's input is `native_table_structure_failed` --
        # the grid is precisely what is broken -- so a correct model value
        # sitting OUTSIDE the ragged native grid read as invented and the good
        # table was discarded. Presence claims must cover the page the
        # disposition site actually has.
        #
        # Union, not replace: grid cells stay counted, so a value inside the
        # grid keeps its multiplicity. Counters throughout -- set containment
        # would be blind to GH-270 substitution, which the module docs forbid.
        # GH-355: .elements(), not the Counter itself. Iterating a Counter
        # yields KEYS, so every native token's count collapsed to 1 while the
        # regex path below kept multiplicity -- two surfaces, two answers, from
        # a function whose contract says "with multiplicity".
        #
        # The collapse is not cosmetic: this module uses Counters rather than
        # sets precisely so a value occurring once that appears twice reads as
        # invented (GH-270 substitution). With counts flattened, a CORRECT
        # candidate repeating a value the page also repeats fails the multiset
        # check and falls back to flagged native.
        grid_counts = Counter(_normalize_numeric_token(t) for t in tokens.numeric.elements())
        page_raw = _re.findall(r"[-\u2212]?(?:\d[\d,]*\.?\d*|\.\d+)", native_text or "")
        page_counts = Counter(_normalize_numeric_token(t) for t in page_raw)
        # Max, not sum: the same token appears in both passes when it is inside
        # the grid, and adding would double it into a phantom occurrence.
        return Counter({t: max(grid_counts[t], page_counts[t]) for t in grid_counts | page_counts})
    # No table markup in the native text: fall back to every numeric token in it.
    # A page whose native reading lost its grid still has its numbers, and this
    # gate only ever claims presence.
    # GH-355: leading-decimal tokens. The old pattern required a digit before
    # the point, so ".75" matched only from the "7" and became "75" -- a value
    # an order of magnitude out, silently, in a presence oracle. Econ tables
    # write .75 and .05 constantly.
    raw = _re.findall(r"[-\u2212]?(?:\d[\d,]*\.?\d*|\.\d+)", native_text or "")
    return Counter(_normalize_numeric_token(t) for t in raw)


def presence_verdict_from_text(
    native_text: str, candidate_markdown: str, *, encoding_suspect: bool = False
) -> PresenceVerdict:
    """As :func:`presence_verdict`, but against a page's extracted text rather
    than its word layer -- for callers holding ``PageState``, not a ``fitz`` page.
    """
    oracle = native_text_value_counts(native_text)
    if not oracle:
        return PresenceVerdict(PRESENCE_UNVERIFIABLE, reason="no numeric tokens in native text")
    if encoding_suspect:
        return PresenceVerdict(
            PRESENCE_UNVERIFIABLE,
            oracle_size=len(oracle),
            reason="text layer shows decode damage; absence is not evidence here",
        )
    candidate = table_value_tokens(candidate_markdown)
    invented, lost = candidate - oracle, oracle - candidate
    if invented:
        return PresenceVerdict(
            PRESENCE_INVENTED,
            invented=tuple(sorted(invented.elements())),
            lost=tuple(sorted(lost.elements())),
            oracle_size=len(oracle),
            reason=f"{sum(invented.values())} value(s) not present in the page text",
        )
    if lost:
        return PresenceVerdict(
            PRESENCE_LOST,
            lost=tuple(sorted(lost.elements())),
            oracle_size=len(oracle),
            reason=f"{sum(lost.values())} page value(s) absent from the candidate",
        )
    return PresenceVerdict(PRESENCE_OK, oracle_size=len(oracle), reason="every value accounted for")


def presence_verdict(
    page, candidate_markdown: str, *, encoding_suspect: bool = False
) -> PresenceVerdict:
    """Judge a candidate's numbers against the page, on presence alone.

    ``encoding_suspect`` must be passed when the page's text layer shows decode
    damage (``PageAssessment.has_encoding_hygiene_suspect`` /
    ``has_corrupt_math``). A token can then be "absent from the word layer"
    because the layer misdecoded it -- the corpus contains ``⟨0.00⟩`` arriving as
    ``h0.00i`` -- and reporting that as invention would convict the model for the
    text layer's own failure. Such a page is UNVERIFIABLE, which is a different
    fact from clean.
    """
    oracle = native_value_counts(page)
    if not oracle:
        return PresenceVerdict(PRESENCE_UNVERIFIABLE, reason="no native value oracle on this page")
    if encoding_suspect:
        return PresenceVerdict(
            PRESENCE_UNVERIFIABLE,
            oracle_size=len(oracle),
            reason="text layer shows decode damage; absence is not evidence here",
        )

    candidate = table_value_tokens(candidate_markdown)
    invented = candidate - oracle  # multiset difference: catches substitutions
    lost = oracle - candidate

    if invented:
        return PresenceVerdict(
            PRESENCE_INVENTED,
            invented=tuple(sorted(invented.elements())),
            lost=tuple(sorted(lost.elements())),
            oracle_size=len(oracle),
            reason=f"{sum(invented.values())} value(s) not present on the page",
        )
    if lost:
        return PresenceVerdict(
            PRESENCE_LOST,
            lost=tuple(sorted(lost.elements())),
            oracle_size=len(oracle),
            reason=f"{sum(lost.values())} page value(s) absent from the candidate",
        )
    return PresenceVerdict(PRESENCE_OK, oracle_size=len(oracle), reason="every value accounted for")


# ---------------------------------------------------------------------------
# GH-326: the acceptance gate.
#
# Composes three witnesses in COST order -- free deterministic checks first, the
# paid image judge last -- so the expensive one runs only on candidates the free
# ones could not settle. That ordering is the cost premise of the whole router,
# not an optimisation bolted on afterwards.
#
# What is deliberately NOT a witness here: native row geometry. It was the
# reference the previous gate graded against, and measurement disqualified it --
# 8/13 rows exact against a free local model's 12/13, and it convicted a
# cell-perfect page ten times. See docs/log/2026-08-30_model-vs-native-table-rows.md.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceVerdict:
    """Whether a candidate table may ship SUCCESS, and on whose authority."""

    accepted: bool
    reason: str
    witness: str  # which check decided: "structure" | "presence" | "image" | "default"
    presence: PresenceVerdict | None = None
    judged: bool = False  # whether the image judge actually ran

    @property
    def demote_only(self) -> bool:
        """True when the page must not ship SUCCESS but its content is still usable.

        The #322 disposition: a rejected candidate is demoted, not discarded. Only
        a candidate with no usable content at all should lose its text.
        """
        return not self.accepted


def table_acceptance(
    page,
    candidate_markdown: str,
    *,
    encoding_suspect: bool = False,
    image_judge=None,
    page_image=None,
) -> AcceptanceVerdict:
    """Decide whether *candidate_markdown* may ship SUCCESS for *page*.

    Three witnesses, cheapest first:

    1. **Candidate-side structure** (free, no reference). A grid that is ragged or
       carries detached label rows is malformed on its own terms -- this needs
       nothing to compare against, so it cannot be poisoned by a bad reference.
    2. **Presence** (free, page-grounded). A value that appears nowhere in the
       page's word layer is the #270 failure and blocks. See
       :func:`presence_verdict` for why this is counts, not sets, and why a
       damaged text layer yields UNVERIFIABLE rather than a conviction.
    3. **The image judge** (paid, positional). The only witness that can answer
       "is this value in the right cell", and the only one that has ever caught
       #270's diagonal misplacement -- twice, independently. Runs last, and only
       when the free checks have not already decided.

    An absent judge does not block: refusing to ship because no judge was
    configured would make the gate a availability test rather than a quality one.
    The verdict records ``judged=False`` so the caller can tell "passed" from
    "never asked".
    """
    from socr.tables import structure_check

    # 1. Structure -- free, and needs no reference at all.
    reports = structure_check.check_markdown(candidate_markdown)
    if structure_check.structural_gate_fires(reports):
        return AcceptanceVerdict(
            accepted=False,
            reason="candidate grid is ragged or has detached label rows",
            witness="structure",
        )

    # 2. Presence -- free, and grounded in the page rather than in a reconstruction.
    presence = presence_verdict(page, candidate_markdown, encoding_suspect=encoding_suspect)
    if presence.blocks_success:
        return AcceptanceVerdict(
            accepted=False, reason=presence.reason, witness="presence", presence=presence
        )

    # 3. The image judge -- paid, and the only positional witness.
    if image_judge is not None and page_image is not None:
        try:
            verdict = image_judge.judge(page_image, candidate_markdown)
        except Exception as exc:  # a judge that errors must not convict
            return AcceptanceVerdict(
                accepted=True,
                reason=f"image judge unavailable ({type(exc).__name__}); free checks passed",
                witness="default",
                presence=presence,
            )
        if not verdict.is_good:
            return AcceptanceVerdict(
                accepted=False,
                reason="; ".join(verdict.issues) or "image judge rejected the transcription",
                witness="image",
                presence=presence,
                judged=True,
            )
        return AcceptanceVerdict(
            accepted=True,
            reason="image judge accepted",
            witness="image",
            presence=presence,
            judged=True,
        )

    return AcceptanceVerdict(
        accepted=True,
        reason="free checks passed; no image judge available",
        witness="default",
        presence=presence,
    )
