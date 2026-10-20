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
import re
from dataclasses import dataclass, replace
from typing import Any

from socr.core.marks import (
    SCOPE_PAGE,
    parse_canonical_mark,
    Check,
    CheckResult,
    FAMILY_COVERAGE,
    families_of,
    line_is_mark,
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

#: The deterministic native-table verifier's record that it reached EXACT_PASS on this
#: page: row counts matched and every paired row's numeric multiset was clean, so
#: ``pipeline/agentic.py`` accepted the page at ``confidence=1.0`` with NO judge call.
NATIVE_TABLE_EXACT_PASS_KIND = "native_table_verifier_exact_pass"

#: The accepting decision's own reason on that path (``pipeline/agentic.py``), which
#: survives onto the shipped output as ``judge_reason`` and is the exact string the
#: acceptance meter reads (``benchmark/acceptance.py``: ``TABLE_ONLY_JUDGE_REASON_PREFIX``
#: plus ``audit_passed``). Required IN ADDITION to the event, and that is Fable's finding
#: on the diagnosis: the event is emitted BEFORE ``_apply_structural_gate``, which on a
#: header-attribution abstain delegates to the inner judge (#245) -- so a page can carry
#: the event and still ship on a model-JUDGED attempt. Keying on the event alone would
#: mark such a page with a reason that is false of it.
EXACT_PASS_ACCEPT_REASON = "native_table_verifier: EXACT_PASS"

#: One sentence, stating what was compared and what was not. It deliberately never says
#: the table was verified: the check is a per-row numeric MULTISET
#: (``tables/native_verifier.py``: ``Counter`` equality), so it is blind to which column a
#: value sits in and to which label a row carries -- measured on a synthetic page, where a
#: within-row column swap and a row-label swap both still reach EXACT_PASS.
EXACT_PASS_UNJUDGED_REASON = (
    "no judge saw this page; only the numbers within each table row were compared, "
    "not their columns or row labels, and nothing outside the table"
)

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
    #: #1203. UNKNOWN, not FAIL: nothing refused this table. It is withheld because
    #: nothing available on this host compares its cells against a second reading, and
    #: the model's reading is retained in the sidecar for #1208 to release.
    FailureMode.TABLE_WITHHELD_UNCONFIRMED: (
        Check.SOURCE_EVIDENCE,
        CheckResult.UNKNOWN,
        "this page's table is withheld until an independent reader confirms its cells",
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
    #: #1145 slice 3. The row that makes the demotion idempotent: on the first pass the
    #: clause comes from the lane's attach EVENT, and from here on every later pass, which
    #: is what ``_rewrite_all_fragments`` does to every page at assemble. Same check, same
    #: result, same reason -- so the verdict a re-finalized page publishes is the verdict
    #: it already had.
    #: #1145 slice 3 (chart). Same purpose as the equation row below: the clause comes from
    #: the lane's event on the first pass and from here on every later one.
    FailureMode.VISUAL_VALUES_UNTRANSCRIBED: (
        Check.FIGURE_TEXT,
        CheckResult.UNKNOWN,
        "this page's chart ships as an image and the labels and values drawn in it were "
        "never transcribed -- read them off the image before citing",
    ),
    #: #1181. Same purpose as the two rows above: the clause comes from the EXACT_PASS
    #: audit event on the first pass and from here on every later one, so a re-finalized
    #: page publishes the verdict it already had. The wording must not imply the table was
    #: verified -- see ``_exact_pass_unjudged_clause``.
    FailureMode.MODEL_OUTPUT_UNJUDGED: (
        Check.PAGE_JUDGE,
        CheckResult.UNKNOWN,
        EXACT_PASS_UNJUDGED_REASON,
    ),
    FailureMode.EQUATION_FIDELITY_UNVERIFIED: (
        Check.EQUATION_FIDELITY,
        CheckResult.UNKNOWN,
        "an equation reading was attached from a crop and its mathematics was never "
        "verified against the page -- only its LaTeX syntax was checked",
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
    #: #1156: without this row the demoted page carried a second, bogus banner saying
    #: ``unknown=UNKNOWN`` beside the clause that already named the reason.
    FailureMode.PRIOR_OCR_LAYER_UNVERIFIED: (
        Check.PRIOR_OCR_LAYER,
        CheckResult.UNKNOWN,
        "earlier OCR, not re-read",
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


#: The only checks the chart floor may state about a page. ``chart_region`` per region,
#: and ``native_coverage`` for text of the page that did not ship. Anything else from that
#: attribute is a bug at the writer, so it is dropped and logged rather than published.
_CHART_FLOOR_CHECKS: frozenset = frozenset({Check.CHART_REGION, Check.NATIVE_COVERAGE})


def _chart_floor_clauses(p: Any) -> list[VerdictClause]:
    """#1155: every outcome the chart floor RECORDED on this page.

    Separate from ``_chart_region_clause`` above, and deliberately: that one reads
    ``_apply_chart_region_guard``'s page-level flags and says one page-scoped thing. These
    are per REGION, and they come from the region results themselves
    (``figures.chart_witness.region_clause_records``) -- the same records the page's own
    lines are rendered from, so the line and the clause cannot disagree.

    Why it is needed: the first version of that lane wrote a mark per region into the page
    and recorded nothing, so all 12 finalized verdicts carried ZERO ``chart_region``
    clauses while the Markdown showed one per region. Both reviewers found it, and both
    said the same thing -- a rendered mark string is not the machine-readable evidence.

    Round 4 (Astra P2) widened it from the per-region records to EVERY record the lane
    writes, because the page-scoped ``native_coverage=FAIL`` line -- the withheld prose
    runs, and the reading-order refusal -- had exactly the same gap: the page said it in
    Markdown and its terminal JSON carried no such clause. The withheld-runs line fires on
    all 12 real Fed pages where the refusal fires on none, so recording only the one a
    reviewer named would have left the commoner case unrecorded.

    Reachability is POSITIVE, not a flag over a class: the attribute exists only on a page
    whose bare whole-page marker was actually replaced by a chart body. A page that never
    reached that branch has nothing here and derives exactly what it derived before, which
    is why this cannot move the meter for the lanes that share this module. A record whose
    check is not one this lane may state is ignored rather than trusted: the attribute is
    written by one caller, and a clause saying otherwise would be a bug there, not a fact
    about this page.
    """
    records = getattr(p, "chart_floor_clauses", None)
    # A LIST or TUPLE, positively: ``p`` is a stand-in object in much of the suite, and a
    # bare ``Mock`` hands back another ``Mock`` for any attribute. Iterating that would
    # raise inside the verdict seam for every page in those tests, and a ``MagicMock``
    # would iterate as EMPTY and read as "no clauses" -- which is the vacuous shape this
    # repo keeps finding. Neither is a record.
    if not isinstance(records, (list, tuple)):
        return []
    clauses: list[VerdictClause] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            clause = VerdictClause.from_dict(record)
        except Exception:
            logger.warning("#1155: unreadable chart-floor region record %r", record)
            continue
        if clause.check not in _CHART_FLOOR_CHECKS:
            logger.warning("#1155: chart-floor record states a check it may not: %r", record)
            continue
        if clause.check is Check.CHART_REGION and clause.scope == SCOPE_PAGE:
            # A chart_region clause must name WHICH region: a page-scoped one would
            # impugn every region, including the ones this lane did represent.
            logger.warning("#1155: chart-floor region record with no region: %r", record)
            continue
        clauses.append(clause)
    return clauses


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


#: Every event kind that records "a model's reading of an equation crop was spliced into
#: this page's text", across BOTH seams that do it.
#:
#: ``equation_region_reading_attached`` is the P4-R region lane's. ``equation_latex_accepted``
#: is the legacy ``--detect-equations --recover-clean-equations`` seam's
#: (``_attach_equation_latex_sidecars``), which splices readings the same way and emits no
#: attach event -- so keying on the first alone let every page of that seam escape the
#: clause. 0 corpus pages, because those flags are off by default, and covered anyway: the
#: cost is one more member of a frozenset, and the alternative is a known silent gap waiting
#: for someone to turn the flags on.
#:
#: Both are in ``EQUATION_LANE_EVENT_KINDS``, the replay allowlist, so both survive a resume
#: rather than disappearing with the audit log -- which is why the clause may key on them at
#: all. The region kind already was; ``equation_latex_accepted`` was NOT, and the first draft
#: of this comment asserted it before anyone checked. Adding it is a fix in its own right --
#: by that set's own docstring a missing kind "disappears from ``audit_log.json`` the moment
#: the page resumes", which was already true for that seam and independent of #1145 -- and
#: #1145 makes it load-bearing, because a clause whose evidence is dropped on resume would
#: silently un-demote the page on the next run.
EQUATION_READING_ATTACHED_KINDS: frozenset[str] = frozenset(
    {"equation_region_reading_attached", "equation_latex_accepted"}
)
#: Kept as a name for the region lane's kind specifically, since tests and the log refer to
#: it, and it is the one the 399 measured pages carry.
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
#: Failure modes a page can never shed by being reprocessed, because the check they name
#: has nothing to run. Keyed on the MODE rather than on the clause, and the chart slice is
#: why: ``figure_text=UNKNOWN`` is produced by THREE sources -- this lane, the
#: ``FIGURE_WORDS_UNREAD`` mode, and the scanned-figure flag -- and only this one is
#: unresolvable. A reader that could not transcribe a figure crop today may succeed next
#: run, so admitting ``(figure_text, UNKNOWN)`` as a pair would have let those pages skip
#: for ever. The mode is already the field that explains the status (that is why the
#: demotion records one), so it is the right key for "can this status ever be resolved".
#:
#: Deliberately small and enumerated. ``AUDIT_FAILED`` and the judge-timeout modes are
#: absent on purpose: a judge may answer next time, and admitting those would turn a narrow
#: content exception into a general "WARNING pages resume" rule.
PERMANENTLY_UNRESOLVABLE_MODES: frozenset[FailureMode] = frozenset(
    {
        FailureMode.EQUATION_FIDELITY_UNVERIFIED,
        FailureMode.VISUAL_VALUES_UNTRANSCRIBED,
        # #1185: no own text layer, or no number in the reading. The check that could
        # resolve it is a second, independent reader (#1208), which does not exist yet;
        # when it does, the source digest in the run fingerprint re-OCRs the page anyway.
        FailureMode.NUMBERS_UNCHECKED,
        # #1185, and Argon's plan review: a disagreement is permanent for the SAME input
        # and config too. The same fingerprint re-reads the same page into the same
        # numbers, so reprocessing it on every resume would never converge (#1192's
        # argument). A changed model or prompt changes the fingerprint and re-reads it.
        FailureMode.NUMBERS_DISAGREE_WITH_TEXT_LAYER,
    }
)

#: #1185: the demotion modes of the ``text_layer_numbers`` clause, and the check each one
#: records. Both are permanent (``PERMANENTLY_UNRESOLVABLE_MODES``).
TEXT_LAYER_NUMBERS_MODES: dict[FailureMode, Check] = {
    FailureMode.NUMBERS_DISAGREE_WITH_TEXT_LAYER: Check.TEXT_LAYER_NUMBERS,
    FailureMode.NUMBERS_UNCHECKED: Check.TEXT_LAYER_NUMBERS,
}


#: Checks whose clause a REPROCESS could still resolve, so a page carrying one must never
#: be admitted to the resume skip-and-keep exception however its mode reads.
#:
#: #1181, found by Fable on this diff. The mode-only test above rests on a reachability
#: argument -- "a page carrying one of these modes cannot also be carrying a clause from
#: another source" -- which held while each silent-SUCCESS class was the only one on its
#: page. #1181 adds a third class whose resolvability DIFFERS: a judge may run on a later
#: pass, so its page must stay eligible for reprocessing (see ``MODEL_OUTPUT_UNJUDGED``).
#: A results page with a table and a display equation under ``--detect-equations`` carries
#: both clauses, and ``_merge_clause``'s mode arbitration records the EQUATION mode, which
#: IS permanent -- so the page was skip-and-kept for ever with its table still unjudged.
#: Measured through the seam: ``_resume_verdict_is_permanently_unverifiable`` returned True.
#: This is the clause-level check the chart slice deleted as unreachable, restored to
#: exactly the one case that reaches it.
REPROCESSABLE_CLAUSE_CHECKS: frozenset[Check] = frozenset({Check.PAGE_JUDGE})


#: The chart lane's record that the values drawn in a chart were never transcribed.
#:
#: Keyed on this event ALONE, not on ``chart_asset_page`` beside it, and that is a checked
#: decision rather than a simplification: ``chart_asset_page`` is NOT in the resume replay
#: allowlist, so a predicate requiring it would lose the clause on every resumed page. On
#: run 23192118 the two agree exactly (1,353 pages, every one with a ``chart_asset``
#: winner), and this event is the one that states what the clause states -- a page whose
#: visual values went untranscribed owes the reader that sentence whichever lane shipped it.
VISUAL_VALUES_UNTRANSCRIBED_KIND = "visual_values_not_transcribed"


def _chart_values_clause(event_kinds: frozenset[str]) -> VerdictClause | None:
    """``figure_text=UNKNOWN`` on a page whose chart ships as an image, untranscribed.

    The chart lane routes such a page to an image asset on purpose -- that is the lane
    working, not failing -- but the numbers a reader would cite are then present only as
    pixels. Measured on run 23192118: 1,353 pages carry the event, 210 shipping SUCCESS
    with no mark anywhere (demoted here) and 1,143 already WARNING (which slice 1 marks and
    which now gain this second, true clause).

    UNKNOWN, not FAIL: nothing read the chart and found it wrong; nothing read it at all.
    """
    if VISUAL_VALUES_UNTRANSCRIBED_KIND not in event_kinds:
        return None
    return VerdictClause(
        Check.FIGURE_TEXT,
        CheckResult.UNKNOWN,
        SCOPE_PAGE,
        "this page's chart ships as an image and the labels and values drawn in it were "
        "never transcribed -- read them off the image before citing",
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
    if not (event_kinds & EQUATION_READING_ATTACHED_KINDS):
        return None
    return VerdictClause(
        Check.EQUATION_FIDELITY,
        CheckResult.UNKNOWN,
        SCOPE_PAGE,
        "an equation reading was attached from a crop and its mathematics was never "
        "verified against the page -- only its LaTeX syntax was checked",
    )


def _exact_pass_unjudged_clause(
    output: PageOutput, event_kinds: frozenset[str]
) -> VerdictClause | None:
    """``page_judge=UNKNOWN`` on a page the deterministic table verifier shipped unjudged.

    #1181. On the EXACT_PASS accept no judge is called at all, so the page ships SUCCESS,
    ``audit_passed=true``, ``verdict.clauses=[]`` and no mark, while everything outside its
    tables is unexamined. Measured on run 23319732: 5 pages of 17,091 ship on this path --
    3 of them SUCCESS and unmarked, which is the whole of the acceptance meter's
    ``table_only_acceptance`` class -- and 5 of 5 carry content outside their tables.

    UNKNOWN, not FAIL: nothing read the caption or the prose and found it wrong; nothing
    read it. Two conditions, both required: the lane's own event, and the accepting reason
    that proves no judge was consulted (see ``EXACT_PASS_ACCEPT_REASON``).

    **Every page accepted through that exit, with no further predicate**, and that is the
    ruling (owner/orchestrator 2026-10-16) after Fable argued it twice. An earlier version
    fired only when the page shipped non-blank content outside its table syntax. Three
    reasons it is gone. The acceptance meter flags EVERY bypassed SUCCESS page
    (``benchmark/acceptance.rule_qwen``: bypassed and ``audit_passed`` -> a violation), so
    the predicate left exactly the hole the meter exists to catch. The binding of values to
    columns and row labels is unchecked on ALL of these pages, table-only ones included,
    because the oracle that could refute it (``tables/binding.py``) is reached only from
    the table-ladder path, which ran on none of the 5 in run 23319732. And 5 of 5 carry
    content outside their tables, so the predicate's negative branch had no corpus
    instance: code no page exercises is code to delete. The mark's wording is true of a
    table-only page unchanged.
    """
    if NATIVE_TABLE_EXACT_PASS_KIND not in event_kinds:
        return None
    if (getattr(output, "judge_reason", "") or "").strip() != EXACT_PASS_ACCEPT_REASON:
        return None
    return VerdictClause(
        Check.PAGE_JUDGE,
        CheckResult.UNKNOWN,
        SCOPE_PAGE,
        EXACT_PASS_UNJUDGED_REASON,
    )


def _prior_ocr_layer_clause(output: PageOutput, p: Any) -> VerdictClause | None:
    """#1156: this page's text layer is an earlier OCR run, and nothing checked it.

    #1050's ink-vs-text coverage shows the layer covers the page's ink; it cannot show the
    layer is RIGHT, and on a scan that layer is a previous OCR run's reading, not the
    document's own characters. Measured on the Fed pilot: 63 of 79 scanned pages shipped
    one unverified prior reading and the meter counted every one CHECKED.

    So the page is demoted and marked. Part A has no comparator, so there is no record a
    page could carry to earn its way out of this clause: one reading of a scan is one
    reading, and the page is MARKED. (Part B adds the agreement record and the escape.)

    Round 3 (Argon): this covers a MODEL winner too. Restricting it to native winners
    left a qwen reading of such a scan shipping SUCCESS with no mark at all, while the
    meter's own rule flagged the same page ``prior_ocr_unchecked`` -- the page and the
    meter disagreeing about whether anything checked it. One reading of a scan, whoever
    produced it, is one reading.
    """
    if not _flag(p, "prior_ocr_layer_suspect"):
        return None
    return VerdictClause(
        check=Check.PRIOR_OCR_LAYER,
        result=CheckResult.UNKNOWN,
        scope=SCOPE_PAGE,
        reason="earlier OCR, not re-read",
    )


#: #1185: the engines whose output is a MODEL READING of the page, by positive evidence:
#: the reading engines socr can be configured with. A native lane, a chart picture, a blank
#: receipt or an equation crop is not a reading of the page's characters by a model, and a
#: native lane compared with its own layer would agree by construction.
def model_reading_engines() -> frozenset[str]:
    from socr.core.config import EngineType

    return frozenset(e.value for e in EngineType if e is not EngineType.AUTO)


#: The page judge can say "faithful" of a reading that turned ``-.8`` into ``-8``; the
#: page's own characters cannot. These are the clause's two UNKNOWN reasons.
#: The audit event a document records for each page this clause FAILs (#1185).
NUMBERS_DISAGREE_KIND = "numbers_disagree_with_text_layer"
NUMBERS_DISAGREE_EVENT_DETAIL = (
    "the model reading ships a number this page's own text layer does not hold; the "
    "numbers are named in the page's mark, nothing was corrected, and the page is never "
    "counted CHECKED (#1185)"
)


def numbers_disagree_note(pages: list[int]) -> str:
    """The document's ``error`` note naming the pages this clause FAILs: written by a run
    and by ``socr-refinalize-own-text`` alike, so the two cannot word it differently."""
    return (
        "model reading ships numbers the page's own text layer does not hold on page(s) "
        + ", ".join(str(n) for n in pages)
        + " (named on each page)"
    )


NO_OWN_TEXT_REASON = "this page has no text layer of its own to check the reading's numbers against"
NO_NUMBER_REASON = "the reading ships no number to check against the page's text layer"
#: The same, on a page whose own layer DOES print numbers: the reading may have dropped
#: them. Counted, not named -- they include page numbers and running heads, and checking
#: drops is issue #1218 (owner/orchestrator ruling, 2026-10-20).
NO_NUMBER_LAYER_HAS_SOME_REASON = (
    "the reading ships no number, while the page's text layer prints {count} "
    "(numbers a reading drops are not checked yet: #1218)"
)
DAMAGED_LAYER_PREFIX = "this page's text layer is damaged, and it does not confirm: "
#: The reading agrees with a layer socr's page analysis refused or found damaged. Not a
#: CHECK: the owner's ruling asks for the page's OWN born-digital text, and a refused layer
#: may be an earlier OCR run's the analysis could not name (Argon, PR #1219 review: #961's
#: invisible-OCR test runs after the analysis's early returns, so such a layer reached the
#: comparator as the page's own). Overrules the round-1 asymmetry (agreement counted).
DAMAGED_LAYER_AGREES_REASON = (
    "the reading's numbers agree with this page's text layer, but socr's page analysis "
    "refused that layer or found its digits damaged, so the agreement confirms nothing"
)

#: The detector's findings that a layer's DIGITS are unreliable on this page: a minus or a
#: control byte standing in for a sign or a decimal point, an unmapped symbol font. A
#: disagreement with such a layer may be the layer's fault (Argon, plan review: an eaten
#: ``1`` would make "text layer says 2.8, reading says 12.8" blame a faithful reading), so
#: it is reported as UNKNOWN, and so is an agreement with it (round 2: see
#: ``DAMAGED_LAYER_AGREES_REASON``).
_DAMAGED_LAYER_FLAGS = (
    "control_byte_digit_hits",
    "control_byte_scan_failed",
    "minus_as_digit_hits",
    "minus_as_digit_scan_failed",
    "has_unrecovered_symbol_glyphs",
)
#: #136's COSMETIC class (lost spaces, fused words): the detector routes digit damage to
#: OCR before it sets this, so the layer's numbers are its own and an agreement with them
#: is a CHECK. A fused pair (``2.32.5``) can still hide a number from the comparator, so a
#: DISPUTE on such a layer is UNKNOWN, as on a damaged one.
_COSMETIC_LAYER_FLAGS = ("has_encoding_hygiene_suspect",)


def _layer_is_damaged(p: Any) -> bool:
    # A page the detector refused as born-digital may carry its own characters (an eaten
    # digit, a garbage ratio) or an earlier OCR run's; it is compared so a dispute is
    # NAMED, and the refusal is the damage finding either way.
    return not _flag(p, "is_born_digital") or any(_flag(p, f) for f in _DAMAGED_LAYER_FLAGS)


#: How socr's OWN quoted notes open, exactly: every quote opener socr emits (marks.py,
#: manifest.py and the chart and figure lanes). Their other lines continue the same quote
#: block. Any other quote is the model's transcription and is checked: the earlier prefixes
#: ``> **`` and ``> [`` also dropped a model's ``> **Note:** The rate was 9.31`` and a whole
#: ``> **Table 3 …**`` quoted table shipping a wrong ``-8`` (socr bot, PR #1219 decision).
_SOCR_QUOTE_OPENINGS = (
    "> [page ",
    "> [table unverified: ",
    "> [table disputed: ",
    "> **Chart region",
    "> **Chart grids on page",
    "> **Unresolved chart placement on page",
    "> *Figure description",
)
_PAGE_HEADER_RE = re.compile(r"#{1,6}\s*Page\s+\d+\s*", re.IGNORECASE)


def reading_numbers_text(text: str, page_num: int) -> str:
    """The part of a shipped page that is the model's reading: socr's own lines removed.

    Dropped: any recognised mark line (wherever it sits); every quote block that socr
    opened (``_SOCR_QUOTE_OPENINGS``) -- its page and region numbers are socr's, not the
    reading's; and a ``## Page N`` header, which assembly strips and which is not one of
    the page's numbers. A quote the MODEL transcribed is kept and checked: dropping every
    quote let a wrong number inside one ship on a page counted CHECKED (Argon, diff
    review).
    """
    kept = []
    in_socr_quote = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(">"):
            in_socr_quote = False
        elif not in_socr_quote and stripped.startswith(_SOCR_QUOTE_OPENINGS):
            in_socr_quote = True
        if (
            not stripped
            or in_socr_quote
            or line_is_mark(stripped, page_num)
            or _PAGE_HEADER_RE.fullmatch(stripped)
        ):
            continue
        kept.append(line.lstrip("> ") if stripped.startswith(">") else line)
    return "\n".join(kept)


def text_layer_numbers_clause(output: PageOutput, p: Any) -> VerdictClause | None:
    """#1185: a model reading is CHECKED only if its numbers occur in the page's own text.

    Measured on the Fed calibration run: 2018 Tealbook p125 shipped ``-8`` for a printed
    ``-.8`` with ``judge_reason=faithful`` and nothing comparing the table's numbers with
    the layer. The comparison is ``tables.reading_agreement.compare_numbers`` (a page
    multiset; what it cannot see is stated there).

    * Not a model reading, no text, or a page-failed marker (no reading ships): no clause.
    * The page's layer is an earlier OCR run (#1156): no clause here -- #1156's own clause
      already marks the page, and #1217 is that layer's agreement path.
    * No text layer of its own: UNKNOWN. The seam for #1208: an independent reader's
      agreement would be the evidence that clears this; nothing produces one yet.
    * The reading ships no number: UNKNOWN. Agreement over nothing is vacuous (the owner's
      ruling: CHECKED requires ``compared > 0``).
    * Numbers the layer has and the reading lacks are NOT checked here (issue #1218).
    * A shipped number the layer does not hold: FAIL, and every one is NAMED -- UNKNOWN
      when the detector refused the layer or found its digits damaged (still named).
    * Every number agrees, on a layer the detector refused or found damaged: UNKNOWN. Only
      agreement with a born-digital, undamaged layer is a CHECK.
    """
    from socr.tables.reading_agreement import (
        PROVENANCE_OWN_CHARACTERS,
        compare_numbers,
        describe_disputes,
    )

    from socr.core.manifest import is_page_failed_marker

    if (output.engine or "") not in model_reading_engines() or not (output.text or "").strip():
        return None
    # A page that ships socr's ``[page N failed: ...]`` marker ships none of the reading,
    # so there is nothing to compare; its marker already marks it. Without this the
    # sidecar carried an UNKNOWN clause that no line on the page states, because
    # ``_apply_verdict_marks`` never writes above a whole-page failure marker -- one
    # surface saying what another does not (found by the p6 stage C oracle, page 9).
    if is_page_failed_marker(output.text):
        return None
    if _flag(p, "prior_ocr_layer_suspect"):
        return None
    reference = getattr(p, "own_layer_text", "")
    if not isinstance(reference, str) or not reference.strip():
        return VerdictClause(
            Check.TEXT_LAYER_NUMBERS, CheckResult.UNKNOWN, SCOPE_PAGE, NO_OWN_TEXT_REASON
        )
    record = compare_numbers(
        reference,
        reading_numbers_text(output.text, output.page_num),
        provenance=PROVENANCE_OWN_CHARACTERS,
    )
    if record.compared == 0:
        layer_count = record.reference_only
        reason = (
            NO_NUMBER_LAYER_HAS_SOME_REASON.format(count=layer_count)
            if layer_count
            else NO_NUMBER_REASON
        )
        return VerdictClause(Check.TEXT_LAYER_NUMBERS, CheckResult.UNKNOWN, SCOPE_PAGE, reason)
    if _layer_is_damaged(p):
        reason = (
            DAMAGED_LAYER_PREFIX + describe_disputes(record)
            if record.disputed
            else DAMAGED_LAYER_AGREES_REASON
        )
        return VerdictClause(Check.TEXT_LAYER_NUMBERS, CheckResult.UNKNOWN, SCOPE_PAGE, reason)
    if record.disputed and any(_flag(p, f) for f in _COSMETIC_LAYER_FLAGS):
        return VerdictClause(
            Check.TEXT_LAYER_NUMBERS,
            CheckResult.UNKNOWN,
            SCOPE_PAGE,
            DAMAGED_LAYER_PREFIX + describe_disputes(record),
        )
    if record.disputed:
        return VerdictClause(
            Check.TEXT_LAYER_NUMBERS, CheckResult.FAIL, SCOPE_PAGE, describe_disputes(record)
        )
    return None


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
    numbers = text_layer_numbers_clause(output, p)
    return (
        *_clauses_ahead_of_text_layer_numbers(output, p, event_kinds),
        *((numbers,) if numbers is not None else ()),
    )


def _clauses_ahead_of_text_layer_numbers(
    output: PageOutput, p: Any, event_kinds: frozenset[str]
) -> tuple[VerdictClause, ...]:
    clauses = (
        _equation_fidelity_clause(event_kinds),
        _chart_values_clause(event_kinds),
        _exact_pass_unjudged_clause(output, event_kinds),
        _prior_ocr_layer_clause(output, p),
    )
    return tuple(clause for clause in clauses if clause is not None)


def with_text_layer_numbers_clause(
    published: PageVerdict,
    clause: VerdictClause,
    output: PageOutput,
    p: Any,
    event_kinds: frozenset[str],
) -> PageVerdict:
    """*published* (a verdict written without #1185) with *clause* where a run of this code
    puts it: after the silent-success clauses that precede it in
    :func:`silent_success_clauses`, before every status-dependent one. Those precede it by
    COUNT, not by check: a status clause about the same check replaces a silent one in
    place (``_merge_clause``), so the slot is the silent clause's whatever its wording.
    *p* and *event_kinds* are what the writing run saw (Argon, PR #1219 review: appending
    put the mark below Tealbook p125's chart mark, unlike a fresh run).

    It assumes the run derived the silent-success clauses this code derives. A run older
    than one of them (run 23319732 predates #1181's ``page_judge`` clause) published fewer,
    so on a page that also carries a status clause the slot is one late; at most the 5
    EXACT_PASS pages of that run (Fable, round-2 review P2-3)."""
    ahead = len(_clauses_ahead_of_text_layer_numbers(output, p, event_kinds))
    clauses = published.clauses
    return replace(published, clauses=(*clauses[:ahead], clause, *clauses[ahead:]))


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
        *_chart_floor_clauses(p),
    ):
        if clause is None:
            continue
        clauses = _merge_clause(clauses, clause, output)
    return tuple(clauses)


def _merge_clause(
    clauses: list[VerdictClause], clause: VerdictClause, output: PageOutput
) -> list[VerdictClause]:
    """Add *clause*, collapsing it with any existing clause about the same (check, region).

    One line per (check, region): two sources naming the same failed check on the same
    region must not give the reader two lines about one thing.

    **Which one survives is decided by the page's recorded MODE, not by arrival order**, and
    that is a correction. The rule used to be "the first wins, and the order is from most
    specific to least" -- which held while the status-dependent clauses came first, and
    then slice 3 put the silent-success clauses at the front of the list and silently
    inverted it. Fable found the consequence on the 388 ``figure_words_unread`` chart pages:
    the chart clause REPLACED #1053's reason, so the page's line said "this page's chart
    ships as an image" while its recorded mode still said ``figure_words_unread`` -- the mark
    and the record contradicting each other about the same page, on 388 pages, while the log
    claimed they each gained a second reason.

    So the mode arbitrates. It is the field that explains the status (that is why the
    demotion records one), so the clause whose wording the mode vouches for is the one the
    reader should see; an incoming clause replaces an existing one only when the mode names
    the incoming one's text and not the existing one's. Where the mode names neither, the
    first still wins, which preserves the old most-specific-first behaviour for the
    mode-table clauses that are all that can reach that case.
    """
    for index, existing in enumerate(clauses):
        if existing.check is not clause.check or existing.scope != clause.scope:
            continue
        entry = _MODE_CLAUSES.get(output.failure_mode)
        mode_reason = entry[2] if entry is not None else None
        if mode_reason is not None and clause.reason == mode_reason != existing.reason:
            # The page's own mode names the INCOMING clause's wording and not the one
            # already here. Replace, so the line agrees with the record.
            return [*clauses[:index], clause, *clauses[index + 1 :]]
        return clauses
    return [*clauses, clause]


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


def _status_is_explained(
    output: PageOutput,
    p: Any,
    provenance: Any,
    event_kinds: frozenset[str],
    clauses: tuple[VerdictClause, ...],
) -> bool:
    """Does anything in this page's record name WHY its status is not SUCCESS?

    Not "are there clauses". A silent-success clause says something true about the page but
    nothing about its status -- it would be there on a SUCCESS page too -- so it cannot
    discharge the fail-closed duty on its own. That was the hole: a page demoted by a
    status-only guard that names nothing (``_apply_ditto_guard``) plus an attach event came
    out with exactly ``{equation_fidelity=UNKNOWN}``, the fallback stayed silent, and the
    page then looked entirely unresolvable to the resume exception.

    Two things can explain the status, and the first is why the demotion records a mode:

    * the failure MODE maps to a check. This covers the ticket's own demotion on every pass
      after the first -- the mode clause may be DEDUPED away as a duplicate of the
      silent-success clause, which is correct for emission and would be wrong to read as
      "nothing explains the status", so the mode is asked directly rather than inferred
      from the surviving clause list;
    * a clause came from somewhere other than the silent-success set -- a flag, the judge,
      the ladder.
    """
    if _MODE_CLAUSES.get(output.failure_mode) is not None:
        return True
    # #1185: a mode whose clause wording names the page's own numbers cannot sit in
    # ``_MODE_CLAUSES`` (a fixed reason there would REPLACE the named one in
    # ``_merge_clause``). It is explained when the clause it records is present.
    check = TEXT_LAYER_NUMBERS_MODES.get(output.failure_mode)
    if check is not None and any(c.check is check for c in clauses):
        return True
    status_free = set(silent_success_clauses(output, p, event_kinds))
    return bool([c for c in clauses if c not in status_free])


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
    # Fail-closed: the page is not CHECKED and nothing in the clause set names WHY ITS
    # STATUS IS WHAT IT IS. 34 pages of run 23117187 reach this. Deleting it is a mutant the
    # suite must kill.
    #
    # The condition is "no clause explains the status", not "no clauses at all", and the
    # difference is a hole this ticket opened and Fable found. Slice 3 derives the
    # silent-success clauses for EVERY status (it had to -- see the idempotence finding in
    # the slice-3 log), so a page demoted by a STATUS-ONLY guard that sets no failure mode
    # -- ``_apply_ditto_guard`` is one, and it names no check anywhere -- and which also
    # carries an attach event came out with exactly ``{equation_fidelity=UNKNOWN}``. Not
    # empty, so the fallback stayed silent, and the page then looked to
    # ``_resume_verdict_is_permanently_unverifiable`` like a page whose every clause is
    # unresolvable: restored for ever under a reason that is not the one that demoted it,
    # where before this ticket it was re-OCR'd. 0 corpus pages and plainly reachable.
    #
    # A silent-success clause says something true about the page but nothing about its
    # STATUS -- it would be there on a SUCCESS page too. So it cannot discharge the
    # fail-closed duty, and the status still needs a reason.
    if output.status is not PageStatus.SUCCESS and not _status_is_explained(
        output, p, provenance, event_kinds, clauses
    ):
        clauses = (*clauses, _fallback_clause(output))
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
