"""#1145: the per-page verdict -- why a page is not CHECKED, and the marks that say so.

One ``PageVerdict`` per page, built once at the ``_select_and_finalize_page`` seam, after
every guard has had its say. **CHECKED means no non-PASS clause, which means no mark.** A
page that is not CHECKED carries one canonical line per uncovered clause, as the first
rendered block of its fragment.

Why the clauses are not derived from ``failure_mode``
----------------------------------------------------
Measured on corpus run 23117187: 322 WARNING pages carry ``failure_mode=none``,
``audit_passed=true``, ``judge_outcome=page_judge_completed``, ``judge_reason=faithful``
and ``disposition.primary_reason=accepted_output``. Nothing in the mode or the disposition
names the check that failed. 261 of them were demoted by ``_apply_chart_region_guard``,
which is *status-only* and deliberately leaves the mode alone; 30 by native glyph damage
on a winner that was already WARNING; 34 by nothing with a serialized flag at all.

So the clauses are read from the SAME per-page flags the guards read, never from the mode
the guards chose not to set. Where even those are absent, ``_fallback_clause`` fires, so a
page cannot reach the reader unmarked merely because socr failed to name its own reason.

Two deliberate differences from the guards
-----------------------------------------
1. A status-only guard returns early when the page is already WARNING for some other
   reason, so its flag goes unrecorded. ``derive_clauses`` reads the flag regardless: the
   check still did not pass, and the reader is owed both reasons, not the first one. This
   does NOT extend to a signal a ruling made report-only on measured grounds -- see the
   note where ``_native_glyph_clause`` used to be.
2. Slice 1 emits clauses only for pages already WARNING or ERROR, so CHECKED pages stay
   byte-identical. The SUCCESS-demoting classes (unverified formula readings, untranscribed
   chart values, the escalated never-judged winner) are slice 3 and enter through
   ``silent_success_clauses``, which is empty here on purpose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from socr.core.marks import (
    SCOPE_PAGE,
    parse_canonical_mark,
    Check,
    CheckResult,
    FAMILY_COVERAGE,
    families_of,
    page_failed_coverage,
    page_unchecked_clause_mark,
)
from socr.core.result import FailureMode, PageOutput, PageStatus

logger = logging.getLogger(__name__)

#: The engine label the crop-backed corrupt-math region lane stamps on its attempt
#: (``pipeline/orchestrator.py``), and the exact token
#: ``_select_page_output_tagged``'s ``CORRUPT_MATH_HYBRID`` branch tests before it
#: demotes the page (``core/manifest.py``). It is the one field that distinguishes
#: that branch's ``AUDIT_FAILED`` from every other page carrying the same mode.
ENGINE_NATIVE_MATH = "native+math"

#: Failure modes that name their own failed check. A mode absent from this map falls to
#: ``_fallback_clause`` rather than being guessed at.
_MODE_CLAUSES: dict[FailureMode, tuple[Check, CheckResult, str]] = {
    FailureMode.NATIVE_GARBLED_MATH: (
        Check.NATIVE_GLYPHS,
        CheckResult.FAIL,
        "the native text layer garbled this page's mathematics and no reading replaced it",
    ),
    FailureMode.NATIVE_MINUS_AS_DIGIT: (
        Check.NATIVE_GLYPHS,
        CheckResult.FAIL,
        "a minus sign or control byte in the native text makes a negative value unreliable",
    ),
    FailureMode.FIGURE_WORDS_UNREAD: (
        Check.FIGURE_TEXT,
        CheckResult.UNKNOWN,
        "a figure crop on this page contains words no reader transcribed",
    ),
    FailureMode.TABLE_NOT_RECONSTRUCTED: (
        Check.TABLE_STRUCTURE,
        CheckResult.FAIL,
        "a table on this page was flattened to prose and its grid was not reconstructed",
    ),
    FailureMode.MODEL_OUTPUT_FLAGGED: (
        Check.NATIVE_TABLE_VERIFIER,
        CheckResult.FAIL,
        "no table-ladder rung accepted this page's table and the model reading still ships",
    ),
    FailureMode.NATIVE_TABLE_STRUCTURE_FAILED: (
        Check.TABLE_STRUCTURE,
        CheckResult.FAIL,
        "this page's native table structure could not be read",
    ),
    FailureMode.TABLE_EMISSION_INVALID: (
        Check.TABLE_STRUCTURE,
        CheckResult.FAIL,
        "a table on this page cannot be represented faithfully and was withheld",
    ),
    FailureMode.NATIVE_INVISIBLE_TEXT_SCAN: (
        Check.NATIVE_COVERAGE,
        CheckResult.UNKNOWN,
        "what ships is a scan's invisible baked-in OCR layer, which socr did not verify",
    ),
    FailureMode.INVISIBLE_SCAN_UNREAD: (
        Check.NATIVE_COVERAGE,
        CheckResult.FAIL,
        "this scan's invisible OCR layer is unusable and no model reading was accepted",
    ),
    FailureMode.NATIVE_TEXT_SHREDDED: (
        Check.NATIVE_COVERAGE,
        CheckResult.FAIL,
        "this page's rotated native text extracted as one glyph per line, not as a reading",
    ),
    FailureMode.TEXT_INK_UNREAD: (
        Check.NATIVE_COVERAGE,
        CheckResult.FAIL,
        "text-like ink on this page has no visible text layer over it",
    ),
    FailureMode.TEXT_COVERAGE_UNKNOWN: (
        Check.NATIVE_COVERAGE,
        CheckResult.UNKNOWN,
        "the text-coverage check on this page could not run or could not decide",
    ),
    FailureMode.PAGE_NOT_PROCESSED_AFTER_HALT: (
        Check.PAGE_PROCESSED,
        CheckResult.UNKNOWN,
        "the page loop halted before reaching this page, so no per-page pass ran on it",
    ),
    FailureMode.TABLE_REJECTED: (
        Check.SOURCE_EVIDENCE,
        CheckResult.FAIL,
        "both table-ladder rungs looked at this page's table and refused it",
    ),
    FailureMode.TABLE_UNVERIFIED: (
        Check.SOURCE_EVIDENCE,
        CheckResult.UNKNOWN,
        "the table ladder ran out of witnesses on this page without reaching a verdict",
    ),
    FailureMode.MODEL_TABLE_OVER_FAILED_FLOOR: (
        Check.NATIVE_TABLE_VERIFIER,
        CheckResult.FAIL,
        "a model table ships here over a native floor that failed verification",
    ),
    FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED: (
        Check.NATIVE_TABLE_VERIFIER,
        CheckResult.FAIL,
        "every rung of the table ladder was exhausted without accepting this page's grid",
    ),
    FailureMode.HEADER_BINDING_UNVERIFIED: (
        Check.TABLE_LABEL,
        CheckResult.UNKNOWN,
        "socr could not confirm which column a header on this page binds to",
    ),
    FailureMode.ROW_SHAPE_NOT_RECONCILABLE_TEXT_TABLE: (
        Check.TABLE_STRUCTURE,
        CheckResult.FAIL,
        "this page's text-table rows could not be reconciled to one grid shape",
    ),
    FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED: (
        Check.PAGE_JUDGE,
        CheckResult.UNKNOWN,
        "the page-level judge never returned a verdict; only this page's tables were checked",
    ),
    FailureMode.PAGE_JUDGE_TIMEOUT: (
        Check.PAGE_JUDGE,
        CheckResult.UNKNOWN,
        "the page judge did not finish, so the bytes that ship were never accepted",
    ),
    FailureMode.PAGE_JUDGE_EXCEPTION: (
        Check.PAGE_JUDGE,
        CheckResult.UNKNOWN,
        "the page judge crashed, so the bytes that ship were never accepted",
    ),
    FailureMode.NATIVE_UNTRUSTED_JUDGE_TIMEOUT: (
        Check.PAGE_JUDGE,
        CheckResult.UNKNOWN,
        "the page judge timed out on the model read, and the native text is not trusted",
    ),
    FailureMode.UNREADABLE_INPUT: (
        Check.UNREADABLE_INPUT,
        CheckResult.FAIL,
        "no page of this document could be loaded",
    ),
}


@dataclass(frozen=True)
class VerdictClause:
    """One check that did not pass, and the region it is about."""

    check: Check
    result: CheckResult
    scope: str = SCOPE_PAGE
    reason: str = ""

    def mark(self, page_num: int) -> str:
        return page_unchecked_clause_mark(
            page_num, self.check, self.result, self.scope, self.reason
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "check": self.check.value,
            "result": self.result.value,
            "scope": self.scope,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerdictClause:
        return cls(
            check=Check(d["check"]),
            result=CheckResult(d["result"]),
            scope=str(d.get("scope") or SCOPE_PAGE),
            reason=str(d.get("reason") or ""),
        )


@dataclass(frozen=True)
class PageVerdict:
    """Every check that did not pass on one page. Empty clauses means CHECKED."""

    page_num: int
    clauses: tuple[VerdictClause, ...] = ()

    @property
    def checked(self) -> bool:
        return not self.clauses

    def marks(self) -> tuple[str, ...]:
        """Every clause rendered, ordered by ``(check, scope)`` so the bytes are stable.

        A stable order is not cosmetic: the fragment is compared byte-for-byte against the
        assembled document and against the previous run on resume, so an order that
        depended on dict iteration would make those comparisons flap.

        This is the FULL set. What a given page still needs written at its top is
        :func:`marks_to_write`, which drops the clauses a mark already there states.
        """
        ordered = sorted(self.clauses, key=lambda c: (c.check.value, c.scope))
        return tuple(c.mark(self.page_num) for c in ordered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_num": self.page_num,
            "clauses": [c.to_dict() for c in self.clauses],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PageVerdict:
        raw = d.get("clauses") or []
        return cls(
            page_num=int(d.get("page_num") or 0),
            clauses=tuple(VerdictClause.from_dict(c) for c in raw if isinstance(c, dict)),
        )


def _flag(p: Any, name: str) -> bool:
    return bool(getattr(p, name, False))


def _chart_region_clause(p: Any) -> VerdictClause | None:
    """``_apply_chart_region_guard``'s own flags (manifest.py:4467).

    261 of the 322 ``failure_mode=none`` WARNING pages on run 23117187 are this one guard.
    It is status-only by design, so the mode says nothing and these flags are the record.
    """
    if _flag(p, "chart_region_render_failed"):
        reason = "a chart region on this page could not be rendered"
    elif _flag(p, "chart_region_placement_unresolved"):
        reason = "a chart region on this page sits at a position socr could not establish"
    elif _flag(p, "chart_region_inventory_failed"):
        reason = "socr could not inventory this page's chart regions"
    else:
        return None
    return VerdictClause(Check.CHART_REGION, CheckResult.FAIL, SCOPE_PAGE, reason)


def _table_label_clause(output: PageOutput) -> VerdictClause | None:
    """``_apply_label_unverified_guard``'s field (manifest.py:4212), #659.

    NOT serialized on the sidecar, which is why a resumed page must take its PUBLISHED
    verdict rather than a recomputed one -- see ``core/manifest`` and the resume gate.
    """
    if not getattr(output, "table_label_unverified", False):
        return None
    return VerdictClause(
        Check.TABLE_LABEL,
        CheckResult.UNKNOWN,
        SCOPE_PAGE,
        "a table label on this page could not be confirmed against the page",
    )


def _scanned_figure_clause(p: Any) -> VerdictClause | None:
    if not _flag(p, "scanned_figure_render_failed"):
        return None
    return VerdictClause(
        Check.FIGURE_TEXT,
        CheckResult.UNKNOWN,
        SCOPE_PAGE,
        "this scanned page names a figure whose image could not be rendered",
    )


def _native_glyph_clause(
    p: Any, output: PageOutput, provenance: Any = None
) -> VerdictClause | None:
    """``native_glyphs=FAIL`` from the PUA class -- and NOT from the math-font class.

    The two signals are routinely confused and only one belongs here. An earlier draft of
    this file removed the clause entirely, citing #140's report-only ruling and its 36.1%
    trigger rate (``docs/log/2026-09-02_p4m-trigger-rates.md``). That ruling is about
    ``has_math_font_typesetting``: 36.1% of the free lane, with an ~8% slice the region
    locator can never clear, which is why ``_apply_math_font_unrecovered_guard`` only
    appends a note. This clause reads ``has_unmapped_math_glyphs``, the PUA class, measured
    in the same document at **2.4%** -- positively damaged glyphs in the shipped text, which
    ``_apply_unresolved_math_guard`` already demotes on its own account. Removing it cost
    the 30 pages measured on run 23117187 a true reason and gave them
    ``unknown=UNKNOWN`` ("socr did not record which check failed") instead, which is worse
    for the reader and is not what the design says.

    ``FAIL``, not ``UNKNOWN`` (Argon's finding 3): the glyph check RAN and found damage in
    the bytes that ship, exactly as ``NATIVE_GARBLED_MATH`` did. ``UNKNOWN`` would tell the
    reader the check never decided.

    Only reached on a page that is already WARNING or ERROR: ``derive_clauses`` returns
    before this for a SUCCESS page, so the clause can never demote one.
    """
    if not _flag(p, "has_unmapped_math_glyphs"):
        return None
    from socr.core.manifest import native_math_damage_ships

    if not native_math_damage_ships(output, provenance):
        # The accounting describes the NATIVE text. When a model reading replaced it, the
        # page owes that damage nothing: it takes the shipping reading's own status. Reusing
        # #1005's predicate rather than restating it -- the guard this clause mirrors
        # (``_apply_unresolved_math_guard``) gates on the same call, and it is not a
        # one-liner: an unknown ending and every ending that may carry native bytes answer
        # True, and a MODEL body that still carries private-use codepoints answers True too,
        # so it errs toward keeping the warning. Measured: 7 corpus pages with a qwen winner
        # were being told their mathematics is unrecovered PUA while a model reading ships.
        return None
    return VerdictClause(
        Check.NATIVE_GLYPHS,
        CheckResult.FAIL,
        SCOPE_PAGE,
        "private-use codepoints in this page's native text layer are unrecovered mathematics",
    )


def _judge_refusal_clause(output: PageOutput, provenance: Any = None) -> VerdictClause | None:
    """``page_judge=FAIL``, from the ONE predicate that decides a judge refusal.

    ``_positive_judge_refusal_evidence`` is CALLED, never restated. It is subtler than it
    looks -- ``judge_accepted is True`` vetoes a stale ``rejection_class`` (#1129) -- and a
    second copy of that rule here would drift from it and lose exactly that nuance. The
    import is local because ``core.manifest`` imports this module.

    This is also why ``AUDIT_FAILED`` is NOT a ``page_judge`` row in ``_MODE_CLAUSES``.
    That mode is a catch-all: ``_select_page_output_tagged``'s corrupt-math branch stamps
    it with no judge involved at all, so reading it as a refusal told the reader the page
    judge rejected bytes it had never seen. Measured on the P6 corpus, page 11
    (``hybrid_clean``): ``audit_passed=False``, ``rejection_class=""``, no
    ``judge_accepted``, no ``judge_outcome`` -- no refusal, and #1129's own banner stays
    silent on it for that reason.
    """
    from socr.core.manifest import (
        _UNACCEPTED_OUTPUT_MARK_EXCLUDED_PROVENANCES,
        _positive_judge_refusal_evidence,
    )

    if not _positive_judge_refusal_evidence(output):
        return None
    if provenance in _UNACCEPTED_OUTPUT_MARK_EXCLUDED_PROVENANCES:
        # ``FAIL`` would be a FALSE statement here, and this is why the clause needs the
        # ending and not only the output. These endings REBUILD the shipped body -- the
        # #649/#652 prose recovery ships the page's own text layer -- and then copy the
        # refused MODEL attempt's ``judge_accepted`` / ``rejection_class`` onto it
        # (``core/manifest.py``, the prose-recovery branch). So the refusal evidence is
        # real, but it is about bytes that did not ship. #1142 skips its banner on exactly
        # this set for exactly this reason; the clause had no provenance input and so wrote
        # "the page judge did not accept the bytes that ship" above both recovery banners.
        #
        # UNKNOWN rather than silence, because something true is left to say: no judge ever
        # looked at what shipped. That is the state the reader needs, and it is the same
        # distinction ``FAIL`` versus ``UNKNOWN`` carries everywhere else in this module --
        # the check decided against the page, or the check never decided.
        return VerdictClause(
            Check.PAGE_JUDGE,
            CheckResult.UNKNOWN,
            SCOPE_PAGE,
            "the bytes that ship were rebuilt after the judge ran and were never judged",
        )
    return VerdictClause(
        Check.PAGE_JUDGE,
        CheckResult.FAIL,
        SCOPE_PAGE,
        "the page judge did not accept the bytes that ship",
    )


def _corrupt_math_hybrid_clause(output: PageOutput) -> VerdictClause | None:
    """``equation_fidelity=FAIL`` for the corrupt-math hybrid winner.

    The ``CORRUPT_MATH_HYBRID`` ending ships native text with spliced region readings and
    demotes to ``AUDIT_FAILED`` when the attempt carried no mode of its own. What failed
    is the mathematics, not the page judge: a LaTeX candidate passed a syntax gate, which
    is not a proof that it says what the crop says (#165's own wording). 1,118 pages of
    run 23117187 are this class -- the largest single clause in the design table -- so
    naming it ``page_judge`` would mislabel the biggest group.
    """
    if output.failure_mode is not FailureMode.AUDIT_FAILED:
        return None
    if (getattr(output, "engine", "") or "") != ENGINE_NATIVE_MATH:
        return None
    return VerdictClause(
        Check.EQUATION_FIDELITY,
        CheckResult.FAIL,
        SCOPE_PAGE,
        "a corrupt equation candidate ships on this page without a verified reading",
    )


def _mode_clause(output: PageOutput) -> VerdictClause | None:
    entry = _MODE_CLAUSES.get(output.failure_mode)
    if entry is None:
        return None
    check, result, reason = entry
    return VerdictClause(check, result, SCOPE_PAGE, reason)


def _fallback_clause(output: PageOutput) -> VerdictClause:
    """The fail-closed clause: not CHECKED, and nothing named why.

    34 of the 322 pages reach this on today's corpus. Deleting it is a mutant the suite
    must kill: without it a page whose reason socr cannot name ships with no mark at all,
    which is the whole defect this ticket removes.
    """
    return VerdictClause(
        Check.UNKNOWN_CHECK,
        CheckResult.UNKNOWN,
        SCOPE_PAGE,
        f"socr did not confirm this page and did not record which check failed "
        f"(status {output.status.value}, failure mode {output.failure_mode.value})",
    )


#: The equation lane's own record that it attached a reading to a detected region
#: (``pipeline/orchestrator.py``). In ``EQUATION_LANE_EVENT_KINDS``, so it is replayed on
#: resume rather than disappearing with the audit log -- which is why the clause may key on
#: it at all.
EQUATION_READING_ATTACHED_KIND = "equation_region_reading_attached"


#: Clauses that can NEVER be resolved by reprocessing the page, because the check does not
#: exist to be run. Keyed on ``(check, result)``, and the RESULT is load-bearing -- keying on
#: the check alone was a bug that
#: ``test_agentic_corrupt_math_flag_uses_region_recovery_not_whole_page`` caught:
#:
#: * ``equation_fidelity=UNKNOWN`` is slice 3's clause: a reading was attached and the lane
#:   validates LaTeX syntax only, with ``semantic_fidelity_verified`` a hardcoded ``False``.
#:   No run can do better, so re-OCR produces the same reading and the same clause.
#: * ``equation_fidelity=FAIL`` is slice 1's corrupt-math hybrid: the mathematics is DAMAGED
#:   and unrecovered. A run with a provider available, or a recovery lane that succeeds, can
#:   genuinely fix it -- and that test pins that such a page must re-OCR rather than skip.
#:
#: That is the general rule for this set, not a special case: UNKNOWN means no check decided,
#: FAIL means one did and found something wrong, and something wrong may be fixable.
#:
#: Read by the resume gate (``_resume_verdict_is_permanently_unverifiable``) to let such a
#: page skip-and-keep instead of being re-OCR'd on every resume for ever. Deliberately small
#: and enumerated, and NOT "every UNKNOWN clause": a judge that timed out may well answer
#: next time, and admitting those would turn a narrow content exception into a general
#: "WARNING pages resume" rule.
PERMANENTLY_UNVERIFIABLE_CLAUSES: frozenset[tuple[Check, CheckResult]] = frozenset(
    {(Check.EQUATION_FIDELITY, CheckResult.UNKNOWN)}
)


def _equation_fidelity_clause(event_kinds: frozenset[str]) -> VerdictClause | None:
    """``equation_fidelity=UNKNOWN`` on a page that shipped an attached equation reading.

    UNKNOWN, not FAIL, and that is the whole character of this clause: the lane does not
    check. ``semantic_fidelity_verified`` is a literal ``False`` in two event payloads
    (``orchestrator.py:3538`` and ``:4377``) -- never computed, never varying, and not a
    field on ``PageOutput`` at all. It records that the validation scope is
    ``latex_syntax_only``: a LaTeX candidate passed a syntax gate, which is not a proof that
    it says what the crop says. So the mathematics on these pages is unverified by
    construction, and FAIL would assert a check decided against them when none ran.

    Keyed on the EVENT, not on the ``native+equations`` engine label, though the two agree
    on every page of run 23192118 (403 of 403). The event is the lane's own record of what
    it did; the engine label is a routing artefact that a future lane could set or not set
    for unrelated reasons. Measured: 403 pages carry the event -- 399 shipping SUCCESS, which
    this clause demotes, and 4 already WARNING, which slice 1 marks.
    """
    if EQUATION_READING_ATTACHED_KIND not in event_kinds:
        return None
    return VerdictClause(
        Check.EQUATION_FIDELITY,
        CheckResult.UNKNOWN,
        SCOPE_PAGE,
        "an equation reading was attached from a crop and its mathematics was never "
        "verified against the page -- only its LaTeX syntax was checked",
    )


def silent_success_clauses(
    output: PageOutput, p: Any, event_kinds: frozenset[str] = frozenset()
) -> tuple[VerdictClause, ...]:
    """Slice 3: clauses that DEMOTE a page which would otherwise ship SUCCESS.

    Each of these DEMOTES as well as marks, because a page whose check never ran is not a
    page socr confirmed, and the status is what every other consumer reads. Slice 3 equation
    ships the first: unverified attached equation readings. The untranscribed chart labels
    and values (#1150) and the escalated never-judged winner follow in their own slices, as
    does the judge-crashed winner whose ``audit_passed`` stayed True (measured 0 pages, and
    the hole confirmed at the seam -- see the slice-2 log).
    """
    clauses: list[VerdictClause] = []
    for clause in (_equation_fidelity_clause(event_kinds),):
        if clause is not None:
            clauses.append(clause)
    return tuple(clauses)


def derive_clauses(
    output: PageOutput,
    p: Any,
    provenance: Any = None,
    event_kinds: frozenset[str] = frozenset(),
) -> tuple[VerdictClause, ...]:
    """Every check that did not pass on this page, from the fields that record it.

    The fail-closed fallback is NOT added here. It belongs after suppression, because a
    page that already carries a specific legacy disclosure must not also be told it is
    ``unknown`` -- see ``build_page_verdict``.
    """
    # The status-INDEPENDENT clauses first. These were SUCCESS-only in the first draft of
    # slice 3, and that was wrong in a way the seam's own idempotence test caught: the
    # demotion makes the page WARNING, so on the next pass -- and
    # ``_rewrite_all_fragments`` re-runs this seam over every page at assemble -- the clause
    # was not re-derived, nothing else named a check, and the FAIL-CLOSED FALLBACK fired.
    # The page's line still said ``equation_fidelity=UNKNOWN`` while its sidecar verdict
    # had become ``unknown=UNKNOWN``: the mark and the record disagreeing about the same
    # page, which is the exact conflation of "what is true" with "what is written" that
    # ``build_page_verdict`` and ``marks_to_write`` were split apart to prevent.
    #
    # The fact these clauses state is true whatever the status says -- an unverified
    # equation reading shipped on this page -- so they are derived for every status. A
    # CHECKED page stays byte-identical because it has none of them; a page already WARNING
    # for another reason gains a second, true clause, which §1 of the design asks for
    # explicitly ("the reader is owed both reasons, not the first one").
    clauses: list[VerdictClause] = list(silent_success_clauses(output, p, event_kinds))
    if output.status is PageStatus.SUCCESS:
        return tuple(clauses)

    for clause in (
        _mode_clause(output),
        _corrupt_math_hybrid_clause(output),
        _judge_refusal_clause(output, provenance),
        _native_glyph_clause(p, output, provenance),
        _chart_region_clause(p),
        _table_label_clause(output),
        _scanned_figure_clause(p),
    ):
        if clause is None:
            continue
        # One line per (check, region), so two sources that name the SAME failed check on
        # the same region collapse to one. The first wins, and the order above is from
        # most specific to least: ``NATIVE_GARBLED_MATH`` says the native layer garbled the
        # mathematics AND that no reading replaced it, which the bare PUA flag does not, so
        # a page carrying both must show the mode's wording. Without this the reader gets
        # two lines about one check.
        if any(c.check is clause.check and c.scope == clause.scope for c in clauses):
            continue
        clauses.append(clause)
    return tuple(clauses)


def _covered_checks(
    leading: list[str], page_num: int, ships_nothing_else: bool
) -> tuple[set[Check], set[tuple[Check, str]]]:
    """What the marks ALREADY at the top of this page state, in two parts.

    The first set is the checks a LEGACY line states. Those are region-blind: every legacy
    family states a page-level or single-table fact and none carries a region ordinal, so
    the caller cannot know which table one is about, and a clause whose scope names a region
    is therefore never suppressed by them.

    The second set is the exact ``(check, scope)`` pairs the CANONICAL lines state, read off
    the lines themselves. A canonical line names one check about one region and may suppress
    only that pair -- it used to be mapped to ``frozenset(Check)``, which let the first line
    written to a two-clause page silence the second.
    """
    covered: set[Check] = set()
    pairs: set[tuple[Check, str]] = set()
    matched_any = False
    for line in leading:
        for family in families_of(line, page_num):
            matched_any = True
            if family == "canonical_unchecked":
                parsed = parse_canonical_mark(line, page_num)
                if parsed is not None:
                    pairs.add(parsed)
            elif family == "page_failed":
                covered |= page_failed_coverage(ships_nothing_else)
            else:
                covered |= FAMILY_COVERAGE.get(family, frozenset())
    if matched_any:
        # ANY disclosure at the top supersedes the fail-closed "no reason was recorded"
        # line. The fallback means socr could not name the failed check in its own fields;
        # a legacy mark has already named something to the reader, so adding a generic
        # line above a specific one is strictly worse for them -- and for a
        # ``[page N failed: …]`` body it is harmful, since a line above the marker stops
        # the page reading as one. The verdict still RECORDS the fallback clause; only its
        # line is withheld, which is why this lives in emission and not in derivation.
        covered.add(Check.UNKNOWN_CHECK)
    return covered, pairs


def build_page_verdict(
    page_num: int,
    output: PageOutput,
    p: Any,
    published: PageVerdict | None = None,
    provenance: Any = None,
    event_kinds: frozenset[str] = frozenset(),
) -> PageVerdict:
    """What is TRUE about this page: every check that did not pass.

    Deliberately independent of what is already written on the page. An earlier draft
    reduced the verdict by the marks already at the top, which made the verdict of a
    re-finalized page come out EMPTY -- the page kept its mark while its sidecar said
    CHECKED, and the resumed re-flush differed from the first flush byte for byte
    (``test_sidecar_bytes_unchanged_after_resumed_run_reflushes``). Suppression belongs to
    the mark EMISSION, not to the verdict: see :func:`marks_to_write`.

    ``published`` is the verdict a previous run wrote to this page's sidecar. When present
    it WINS, because two clause inputs do not survive a resume --
    ``PageOutput.table_label_unverified`` is not serialized, and slice 3's escalation
    predicate reads ``p.attempts``, which resume rebuilds as the single frozen winner. A
    recomputed verdict would silently drop them and the page would come back from a resume
    looking cleaner than it left. This mirrors the existing rule that a restored page keeps
    the base disposition its original run published.
    """
    if published is not None:
        if published.clauses or output.status is PageStatus.SUCCESS:
            return replace(published, page_num=page_num)
        # A published verdict that says CHECKED on a page that is NOT SUCCESS cannot both
        # be true. "Not SUCCESS means at least one clause" is the invariant the whole
        # ticket rests on, and it must hold by construction here rather than depend on
        # resume rule (a) having refused every sidecar that could carry such a pair -- a
        # rule in another module, which a future skip path could widen without ever
        # touching this file. So the page falls through to the fail-closed clause and the
        # reader is told the page is unconfirmed, which is the conservative direction.
        logger.warning(
            "#1145: p%d carried a published verdict with no clause while shipping %s; "
            "falling back to the fail-closed clause",
            page_num,
            output.status.value,
        )
        return PageVerdict(page_num=page_num, clauses=(_fallback_clause(output),))
    clauses = derive_clauses(output, p, provenance, event_kinds)
    if not clauses and output.status is not PageStatus.SUCCESS:
        # Fail-closed: not CHECKED, and nothing named why. 34 pages of run 23117187 reach
        # this. Deleting it is a mutant the suite must kill.
        clauses = (_fallback_clause(output),)
    return PageVerdict(page_num=page_num, clauses=clauses)


def marks_to_write(
    verdict: PageVerdict,
    leading: list[str],
    ships_nothing_else: bool,
) -> list[str]:
    """Which of the verdict's lines this page still needs at the top.

    A clause a mark already at the top STATES is not written again -- that keeps the
    reader's page to one disclosure per fact, and keeps an already-marked page
    byte-identical. A clause scoped to a REGION is never suppressed: no legacy mark carries
    a region ordinal, so a withheld Table 1's marker must not be read as covering a
    rendered, corrupted Table 2 (measured: 192 pages ship prose beside such a marker).

    Idempotent by construction, and NOT because a canonical line covers everything -- it
    covers exactly the ``(check, scope)`` it names, which is what round 2 corrected. On a
    second pass each line already written is in *leading*, ``parse_canonical_mark`` reads
    its own pair back off it, and so every clause already disclosed is suppressed by its own
    line and nothing is written. That is why ``leading_mark_blocks`` must return EVERY
    leading mark line rather than the first block: with per-pair coverage, a second clause
    whose line sat in the second block would otherwise be re-emitted on every pass.
    """
    covered, pairs = _covered_checks(leading, verdict.page_num, ships_nothing_else)
    wanted = [
        c
        for c in verdict.clauses
        if (c.check, c.scope) not in pairs and (c.scope != SCOPE_PAGE or c.check not in covered)
    ]
    ordered = sorted(wanted, key=lambda c: (c.check.value, c.scope))
    return [c.mark(verdict.page_num) for c in ordered]
