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
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from socr.core.marks import (
    SCOPE_PAGE,
    region_scope,
    parse_canonical_mark,
    Check,
    CheckResult,
    FAMILY_COVERAGE,
    families_of,
    line_is_mark,
    page_failed_coverage,
    page_unchecked_clause_mark,
)
from socr.core.pass_trace import TraceEntry, TraceResult
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

#: #1251 (C3): the reason of the ``page_judge=UNKNOWN`` clause on a model reading the page judge
#: accepted and nothing independent confirmed. Worded by what is MISSING, never by who judged:
#: on an escalated page a cloud engine read and the local model judged, so "approved by the model
#: that produced it" would be false there. Also the key that tells this clause from #1181's
#: (``EXACT_PASS_UNJUDGED_REASON``): same check, different resolvability.
JUDGE_ONLY_REASON = (
    "no independent check confirmed this reading: neither the words nor the numbers of the "
    "page were compared with the page's own text layer; verify it against the page image "
    "before citing"
)

#: Failure modes that name their own failed check. A mode absent from this map falls to
#: ``_fallback_clause`` rather than being guessed at.
#: #1156 / #1262: why a page whose text layer is an earlier OCR run is never CHECKED. The
#: reason depends on what shipped. The prior layer itself has not been re-read at all; a
#: model reading of the page image is ONE reading, and nothing independent agreed with it
#: (Part B's escape). The status is the same either way, only the explanation differs.
PRIOR_OCR_NATIVE_REASON = "earlier OCR, not re-read"
PRIOR_OCR_MODEL_REASON = "read once from the image, no second reading agreed"

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
    #: #1270. FAIL: a second reader DID read this page's pixels and its numbers disagree
    #: with the table. The wording must not say which reader is wrong: the witness is a
    #: classical OCR engine and misreads dense tables, so this is not a finding against the
    #: model (it was labelled ``hallucination`` until #1270, 85 of 90 pages).
    FailureMode.WITNESS_DISAGREES: (
        Check.SOURCE_EVIDENCE,
        CheckResult.FAIL,
        "a second reader of this page's image disagrees with its table numbers; "
        "which reader is wrong is not decided",
    ),
    #: #1270. UNKNOWN: the second reader could not examine this table, so nothing was
    #: contradicted and nothing was confirmed.
    FailureMode.TABLE_UNVERIFIED_WITNESS: (
        Check.SOURCE_EVIDENCE,
        CheckResult.UNKNOWN,
        "a second reader of this page's image could not examine its table, so the "
        "table's numbers are unchecked",
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
    #: #1187 is deliberately ABSENT from this map, and that is the fix rather than an
    #: omission. These rows exist for a lane whose per-page state does not survive a resume,
    #: so the clause must be rebuilt from the mode. This lane's state DOES survive
    #: (``unshipped_figure_unlocated`` is a persisted sidecar key, restored by the resume
    #: path), so ``_unshipped_figure_clauses`` re-derives the real clause on every pass. A row
    #: here would be a SECOND, page-scoped clause about the same fact -- which is what the
    #: shared ``VISUAL_VALUES_UNTRANSCRIBED`` row produced on the second pass, saying the
    #: page's chart ships as an image on a page that shipped none (Argon P0-3, reproduced).
    #: #1181. Same purpose as the two rows above: the clause comes from the EXACT_PASS
    #: audit event on the first pass and from here on every later one, so a re-finalized
    #: page publishes the verdict it already had. The wording must not imply the table was
    #: verified -- see ``_exact_pass_unjudged_clause``.
    FailureMode.MODEL_OUTPUT_UNJUDGED: (
        Check.PAGE_JUDGE,
        CheckResult.UNKNOWN,
        EXACT_PASS_UNJUDGED_REASON,
    ),
    #: #1251 (C3): same purpose as the row above, for the judge-approved model reading that no
    #: independent check confirmed.
    FailureMode.READING_UNCORROBORATED: (
        Check.PAGE_JUDGE,
        CheckResult.UNKNOWN,
        JUDGE_ONLY_REASON,
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
        PRIOR_OCR_NATIVE_REASON,
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
class CheckOutcome:
    """What one check did on one scope of one page (#1183 pass trace).

    ``clause`` is the verdict clause the check found, or ``None``; ``entry`` is the trace entry
    and is ALWAYS present, so a check cannot answer "nothing to report" without also saying
    whether that is a pass (it ran and found nothing), not applicable, or not run. A check
    function that returns ``None`` for "no clause" and leaves the reader to infer which of the
    three it meant is the defect this type exists to prevent.
    """

    entry: TraceEntry
    clause: VerdictClause | None = None


_CLAUSE_TRACE_RESULT = {
    CheckResult.PASS: TraceResult.PASS,
    CheckResult.FAIL: TraceResult.FAIL,
    CheckResult.UNKNOWN: TraceResult.UNKNOWN,
}


def _check_name(check: Check | str) -> str:
    return check.value if isinstance(check, Check) else str(check)


def entry_from_clause(clause: VerdictClause) -> TraceEntry:
    """The trace entry that states *clause*: same check, scope and reason, same result."""
    return TraceEntry(
        _check_name(clause.check),
        _CLAUSE_TRACE_RESULT[clause.result],
        clause.scope,
        clause.reason,
    )


def _found(clause: VerdictClause) -> CheckOutcome:
    return CheckOutcome(entry_from_clause(clause), clause)


def _passed(check: Check | str, scope: str = SCOPE_PAGE, reason: str = "") -> CheckOutcome:
    return CheckOutcome(TraceEntry(_check_name(check), TraceResult.PASS, scope, reason))


def _not_applicable(check: Check | str, reason: str, scope: str = SCOPE_PAGE) -> CheckOutcome:
    return CheckOutcome(TraceEntry(_check_name(check), TraceResult.NOT_APPLICABLE, scope, reason))


def _not_run(check: Check | str, reason: str, scope: str = SCOPE_PAGE) -> CheckOutcome:
    return CheckOutcome(TraceEntry(_check_name(check), TraceResult.NOT_RUN, scope, reason))


CHECKED_TEXT_CHANGED_REASON = "checked text changed after the check"
ROW_BINDING_OFF_TABLE_REASON = (
    "a row_binding pass is about one table block; this one names no table, so it is not a "
    "pass of any block of the shipped text"
)
PASS_WITHOUT_DIGEST_REASON = (
    "this pass carries no digest of the text it was computed on, so it cannot be tied to the "
    "text that ships"
)


def page_text_digest(text: str) -> str:
    """The digest a page-scope PASS carries: the sidecar's own text digest (``sha256:`` over the
    UTF-8 bytes), so the trace and the sidecar normalise the page identically."""
    from socr.core.page_credential import sha256_text

    return sha256_text(text or "")


def table_block_digests(text: str) -> list[str]:
    """The digest of each table block of *text*, in order (ordinal k is index k-1): over the
    block's own lines, so a mark put above the body, or a table added elsewhere on the page, does
    not change another table's digest."""
    from socr.core.page_credential import sha256_text
    from socr.tables.reconcile import find_table_blocks

    lines = (text or "").splitlines()
    return [
        sha256_text("\n".join(lines[b.start : b.end + 1])) for b in find_table_blocks(text or "")
    ]


def table_block_digest(text: str, table_index: int) -> str:
    """The digest of the 0-based *table_index*-th table block of *text*, ``""`` when it has none."""
    digests = table_block_digests(text)
    return digests[table_index] if 0 <= table_index < len(digests) else ""


def stamp_text_digest(entry: TraceEntry, text: str) -> TraceEntry:
    """*entry* with the digest of *text* when it is a PASS that carries none: a check that has just
    run on *text* names it. A PASS that already carries a digest (a replayed or table entry) is
    left alone, and so is every result that is not a PASS."""
    if entry.result is not TraceResult.PASS or entry.text_sha or entry.table_sha:
        return entry
    return replace(entry, text_sha=page_text_digest(text))


def verify_pass_digests(entries: Iterable[TraceEntry], text: str) -> list[TraceEntry]:
    """*entries* with every PASS that was not computed on *text* made NOT_RUN.

    A page-scope PASS must carry ``text_sha`` equal to the digest of *text*; a ``row_binding``
    ``table:N`` PASS must carry ``table_sha`` equal to the digest of the N-th table block of
    *text*. A PASS with no digest, or a different one, states nothing about the bytes that ship
    (the text changed after the check: a figure caption, a table inserted above the bound one, a
    failure marker an assembly guard wrote, a refinalize tool's rewrite). Every other result is
    returned as it is."""
    page_digest = page_text_digest(text)
    tables = table_block_digests(text)
    out: list[TraceEntry] = []
    for e in entries:
        if e.result is not TraceResult.PASS:
            out.append(e)
            continue
        kind, _, ordinal = e.scope.partition(":")
        if e.check == ROW_BINDING_CHECK:
            if kind != "table" or not ordinal.isdigit():
                # Row binding binds one table block. A PASS of it at page scope (or any scope
                # that names no table) is about no block of this text: no writer produces it.
                out.append(
                    TraceEntry(e.check, TraceResult.NOT_RUN, e.scope, ROW_BINDING_OFF_TABLE_REASON)
                )
                continue
            k = int(ordinal)
            have, want = e.table_sha, tables[k - 1] if 1 <= k <= len(tables) else ""
        else:
            have, want = e.text_sha, page_digest
        if have and have == want:
            out.append(e)
            continue
        reason = CHECKED_TEXT_CHANGED_REASON if have else PASS_WITHOUT_DIGEST_REASON
        out.append(TraceEntry(e.check, TraceResult.NOT_RUN, e.scope, reason))
    return out


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


def pages_not_success_note(pages: list[int]) -> str:
    """The named reason a document with otherwise clean buckets is still not COMPLETED."""
    return (
        "page(s) "
        + ", ".join(str(n) for n in pages)
        + " did not finish with status success and a passing audit"
    )


def published_audit_passed(
    status: Any, accepted: Any, verdict: PageVerdict | dict[str, Any] | None
) -> bool:
    """The PUBLISHED ``audit_passed`` label of a page sidecar (#1244 C1).

    True only when the page is SUCCESS, the winner-selection acceptance was an explicit
    ``True``, and the verdict holds no FAIL clause. UNKNOWN clauses do not block: the
    status ladder already demotes what an UNKNOWN stops being allowed to ship as SUCCESS,
    and a WARNING or ERROR page is false regardless of its clauses.

    ``accepted`` is the INTERNAL acceptance (``PageOutput.audit_passed``, the #252
    winner-selection flag, persisted as ``winning_output.audit_passed``). It is an input
    here and is never written back: resume (``_load_terminal_page``) and selection read
    the internal value, readers of the top-level key read this derivation.
    """
    if getattr(status, "value", status) != PageStatus.SUCCESS.value or accepted is not True:
        return False
    if verdict is None:
        return True
    if isinstance(verdict, PageVerdict):
        results = [c.result.value for c in verdict.clauses]
    else:
        results = [
            str(c.get("result")) for c in (verdict.get("clauses") or []) if isinstance(c, dict)
        ]
    return CheckResult.FAIL.value not in results


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


def _unshipped_figure_clauses(p: Any) -> list[VerdictClause]:
    """#1187: one clause per figure this page NAMES and socr could neither show nor locate.

    Status-INDEPENDENT (it lives in ``silent_success_clauses``) because the pages it is for
    ship SUCCESS: on run 23319732 all 3,082 model-read pages count CHECKED while 142 of them
    name a figure they do not show, 39 of which this lane cannot give a picture. A clause that
    only fired on an already-demoted page would be blind to exactly that population.

    Records in, clauses out. The orchestrator writes the captions; the lane's own module
    renders them into clause records, and ``_apply_verdict_marks`` renders the page's lines
    from those -- one source, so the line and the record cannot disagree (#1155's lesson).

    ``isinstance`` rather than truthiness: ``p`` is a stand-in object in much of the suite,
    and a bare ``Mock`` hands back another ``Mock`` for any attribute, which would iterate
    as a non-empty sequence of ``Mock``s here -- while a ``MagicMock`` iterates as EMPTY and
    reads as "no clauses", the vacuous shape this repo keeps finding.
    """
    captions = getattr(p, "unshipped_figure_unlocated", None)
    if not isinstance(captions, (list, tuple)) or not captions:
        return []
    from socr.figures.unshipped_figures import clause_records

    page_num = int(getattr(p, "page_num", 0) or 0)
    kind = getattr(p, "unshipped_figure_kind", "")
    clauses: list[VerdictClause] = []
    for record in clause_records(
        page_num, tuple(str(c) for c in captions), str(kind) if isinstance(kind, str) else ""
    ):
        try:
            clauses.append(VerdictClause.from_dict(record))
        except Exception:
            logger.warning("#1187: unreadable unshipped-figure record %r", record)
    return clauses


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


def _mode_entry(output: PageOutput) -> tuple[Check, CheckResult, str] | None:
    """``_MODE_CLAUSES``'s row for *output*, with #1262's reason for a model winner."""
    entry = _MODE_CLAUSES.get(output.failure_mode)
    if (
        entry is not None
        and output.failure_mode is FailureMode.PRIOR_OCR_LAYER_UNVERIFIED
        and _is_model_reading(output)
    ):
        return (entry[0], entry[1], PRIOR_OCR_MODEL_REASON)
    return entry


def _is_model_reading(output: PageOutput) -> bool:
    """Whether *output* is a model's reading of the page image (#1185's engine set).

    The consensus producer labels its winner ``consensus(<engine>)``; that is still a model
    reading, so the wrapper is stripped first (cubic, PR #1263 review)."""
    return engine_is_model_reading(output.engine or "")


def engine_is_model_reading(engine: str) -> bool:
    """:func:`_is_model_reading` for a bare engine label (the refinalize tool has no
    ``PageOutput`` yet and must gate a page exactly as a fresh run does)."""
    from socr.core.manifest import _base_engine_name

    return _base_engine_name(engine) in model_reading_engines()


def _mode_clause(output: PageOutput) -> VerdictClause | None:
    entry = _mode_entry(output)
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
        # #1187 round 4: a figure the page names and socr could not SHOW is deterministic
        # under the same input and fingerprint -- the same locator on the same PDF gives the
        # same answer -- so re-reading the page cannot resolve it and the per-page ledger
        # should restore it. Round 2 left this out and every page the lane marked was re-OCR'd
        # on every partial-document resume, for ever (Fable P1-1 / Argon P1-C, reproduced).
        # ``FIGURE_RENDER_FAILED`` is deliberately NOT here: that one is a render failure a
        # re-run may fix, which is what Argon's round-2 P0-3 asked for.
        FailureMode.FIGURE_NOT_SHOWN,
        # #1185: no own text layer, or no number in the reading. The check that could
        # resolve it is a second, independent reader (#1208), which does not exist yet;
        # when it does, the source digest in the run fingerprint re-OCRs the page anyway.
        FailureMode.NUMBERS_UNCHECKED,
        # #1185, and Argon's plan review: a disagreement is permanent for the SAME input
        # and config too. The same fingerprint re-reads the same page into the same
        # numbers, so reprocessing it on every resume would never converge (#1192's
        # argument). A changed model or prompt changes the fingerprint and re-reads it.
        FailureMode.NUMBERS_DISAGREE_WITH_TEXT_LAYER,
        # #1218: the same page and config re-read give the same omission.
        FailureMode.NUMBERS_DROPPED_FROM_READING,
        # #1294: the same page and config produce the same shipped words.
        FailureMode.WORDS_MISSING_FROM_PAGE,
        # #1306: the same layer holds the same control bytes.
        FailureMode.CONTROL_BYTES_REPLACED,
        # #1251: re-reading cannot produce an independent check the layer does not offer, and
        # the reading is not deterministic, so re-reading would only change shipped text.
        FailureMode.READING_UNCORROBORATED,
    }
)

#: #1185: the demotion modes of the ``text_layer_numbers`` clause, and the check each one
#: records. Both are permanent (``PERMANENTLY_UNRESOLVABLE_MODES``).
TEXT_LAYER_NUMBERS_MODES: dict[FailureMode, Check] = {
    FailureMode.NUMBERS_DISAGREE_WITH_TEXT_LAYER: Check.TEXT_LAYER_NUMBERS,
    FailureMode.NUMBERS_UNCHECKED: Check.TEXT_LAYER_NUMBERS,
}

#: #1218: the demotion mode of the ``text_layer_drops`` clause. Kept apart from
#: ``TEXT_LAYER_NUMBERS_MODES`` because the two checks say different things (a number the
#: reading SHIPS vs a number it OMITS); every consumer that must treat the three modes alike
#: (the resume WARNING guard, the status-explained test) reads ``NUMBER_CHECK_MODES``.
TEXT_LAYER_DROPS_MODES: dict[FailureMode, Check] = {
    FailureMode.NUMBERS_DROPPED_FROM_READING: Check.TEXT_LAYER_DROPS,
}
NUMBER_CHECK_MODES: dict[FailureMode, Check] = {
    **TEXT_LAYER_NUMBERS_MODES,
    **TEXT_LAYER_DROPS_MODES,
}
#: #1294: the demotion mode of the ``text_layer_words`` clause.
TEXT_LAYER_WORDS_MODES: dict[FailureMode, Check] = {
    FailureMode.WORDS_MISSING_FROM_PAGE: Check.TEXT_LAYER_WORDS,
}
#: Every mode a text-layer clause demotes a page with, for the consumers that treat them
#: alike (the resume WARNING guard, ``_mode_is_explained``).
LAYER_CHECK_MODES: dict[FailureMode, Check] = {
    **NUMBER_CHECK_MODES,
    **TEXT_LAYER_WORDS_MODES,
    # #1306: the control-byte clause is a fact about the page's own layer too.
    FailureMode.CONTROL_BYTES_REPLACED: Check.CONTROL_BYTES,
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


def clause_is_reprocessable(clause: VerdictClause) -> bool:
    """Could re-reading the page resolve *clause*? Every ``page_judge`` clause but #1251's: that
    one says no independent check exists for this reading, which a re-read of the same page
    under the same layer and config cannot change (it can only change the shipped text)."""
    return clause.check in REPROCESSABLE_CLAUSE_CHECKS and clause.reason != JUDGE_ONLY_REASON


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
    return _chart_values_outcome(event_kinds).clause


#: #1183 pass trace reasons. Named, not inlined: tests and the scoreboard read them by value.
NO_PROVIDER_LADDER_REASON = "no provider ladder in this run"
CHART_VALUES_NOT_APPLICABLE_REASON = (
    "no chart of this page shipped as an image with its values untranscribed"
)
EQUATION_NOT_APPLICABLE_REASON = "no equation reading was attached to this page"
EXACT_PASS_NOT_APPLICABLE_REASON = (
    "this page was not accepted by the deterministic table verifier without a judge "
    "(this entry does not say whether a judge ran)"
)


def _chart_values_outcome(event_kinds: frozenset[str]) -> CheckOutcome:
    """:func:`_chart_values_clause` with its outcome. The check is keyed on the chart lane's
    event alone, so it can find a problem and can never PASS: nothing reads the values."""
    if VISUAL_VALUES_UNTRANSCRIBED_KIND not in event_kinds:
        return _not_applicable(Check.FIGURE_TEXT, CHART_VALUES_NOT_APPLICABLE_REASON)
    return _found(
        VerdictClause(
            Check.FIGURE_TEXT,
            CheckResult.UNKNOWN,
            SCOPE_PAGE,
            "this page's chart ships as an image and the labels and values drawn in it were "
            "never transcribed -- read them off the image before citing",
        )
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
    return _equation_fidelity_outcome(event_kinds).clause


EQUATION_LANE_OFF_REASON = (
    "no equation-reading lane was enabled on this run, so no equation reading could be attached"
)


def _equation_fidelity_outcome(
    event_kinds: frozenset[str],
    providers_available: bool = True,
    equation_lane_enabled: bool = True,
) -> CheckOutcome:
    """:func:`_equation_fidelity_clause` with its outcome. Model-dependent (a reading of a crop
    is what the lane attaches), and it can never PASS: the lane checks LaTeX syntax only."""
    if not (event_kinds & EQUATION_READING_ATTACHED_KINDS):
        if not providers_available:
            return _not_run(
                Check.EQUATION_FIDELITY,
                f"{NO_PROVIDER_LADDER_REASON}: no equation reading could be attached",
            )
        if not equation_lane_enabled:
            return _not_run(Check.EQUATION_FIDELITY, EQUATION_LANE_OFF_REASON)
        return _not_applicable(Check.EQUATION_FIDELITY, EQUATION_NOT_APPLICABLE_REASON)
    return _found(
        VerdictClause(
            Check.EQUATION_FIDELITY,
            CheckResult.UNKNOWN,
            SCOPE_PAGE,
            "an equation reading was attached from a crop and its mathematics was never "
            "verified against the page -- only its LaTeX syntax was checked",
        )
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
    return _exact_pass_unjudged_outcome(output, event_kinds).clause


def _exact_pass_unjudged_outcome(output: PageOutput, event_kinds: frozenset[str]) -> CheckOutcome:
    """:func:`_exact_pass_unjudged_clause` with its outcome. Keyed on the lane's event and the
    accepting reason, so it can find the unjudged accept and can never PASS."""
    if NATIVE_TABLE_EXACT_PASS_KIND not in event_kinds:
        return _not_applicable(Check.PAGE_JUDGE, EXACT_PASS_NOT_APPLICABLE_REASON)
    if (getattr(output, "judge_reason", "") or "").strip() != EXACT_PASS_ACCEPT_REASON:
        return _not_applicable(Check.PAGE_JUDGE, EXACT_PASS_NOT_APPLICABLE_REASON)
    return _found(
        VerdictClause(
            Check.PAGE_JUDGE,
            CheckResult.UNKNOWN,
            SCOPE_PAGE,
            EXACT_PASS_UNJUDGED_REASON,
        )
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
    return _prior_ocr_layer_outcome(output, p).clause


PRIOR_OCR_NO_LAYER_REASON = (
    "this page has no text layer of its own, so none can be an earlier OCR run"
)
PRIOR_OCR_PASS_REASON = "page analysis found no earlier-OCR layer on this page's own text layer"


def _prior_ocr_layer_outcome(output: PageOutput, p: Any) -> CheckOutcome:
    """:func:`_prior_ocr_layer_clause` with its outcome. The page analysis scans every page
    (``BornDigitalDetector.assess``), and a failed scan counts as suspect, so a page that has
    its own layer and is not suspect is a recorded negative, not an absence."""
    if _flag(p, "prior_ocr_layer_suspect"):
        return _found(
            VerdictClause(
                check=Check.PRIOR_OCR_LAYER,
                result=CheckResult.UNKNOWN,
                scope=SCOPE_PAGE,
                reason=PRIOR_OCR_MODEL_REASON
                if _is_model_reading(output)
                else PRIOR_OCR_NATIVE_REASON,
            )
        )
    own = getattr(p, "own_layer_text", "")
    if not (_flag(p, "is_born_digital") or (isinstance(own, str) and own.strip())):
        return _not_applicable(Check.PRIOR_OCR_LAYER, PRIOR_OCR_NO_LAYER_REASON)
    return _passed(Check.PRIOR_OCR_LAYER, reason=PRIOR_OCR_PASS_REASON)


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


#: The audit event a document records for each page ``text_layer_drops`` FAILs (#1218).
#: Its own kind and wording: it states an OMISSION, never that the reading "ships a number
#: the layer does not hold" (#1185's kind), which would be false of these pages.
NUMBERS_DROPPED_KIND = "numbers_dropped_from_reading"
NUMBERS_DROPPED_EVENT_DETAIL = (
    "the model reading omits numbers this page's own text layer prints in table rows; the "
    "mark gives their count and this event's data lists them. This check adds and withholds "
    "nothing (other checks on the page may do either), and the page is never counted "
    "CHECKED (#1218)"
)


#: #1294: the audit event a document records for each page ``text_layer_words`` FAILs, and
#: the CLI line and document note naming those pages. One definition each, so a run and a
#: re-finalize tool cannot word them differently.
WORDS_MISSING_KIND = "words_missing_from_page"
WORDS_MISSING_EVENT_DETAIL = (
    "this page ships without words its own text layer prints; the mark names them and this "
    "event's data lists them. This check adds and withholds nothing, and the page is never "
    "counted CHECKED (#1294)"
)
WORDS_MISSING_NOTE_PREFIX = "page(s) ship without words their own text layer prints"


def words_missing_event_data(page_num: int, missing) -> dict:
    """The audit event's ``data``: the page and its missing words, in layer order."""
    return {"page": page_num, "missing": list(missing or ())}


def words_missing_cli_line(pages: list[int]) -> str:
    return f"{len(pages)} {WORDS_MISSING_NOTE_PREFIX} (named on each page): {pages}"


def words_missing_note(pages: list[int]) -> str:
    return (
        WORDS_MISSING_NOTE_PREFIX
        + ": "
        + ", ".join(str(n) for n in pages)
        + " (named on each page)"
    )


#: #1306: the two page-level audit events of the damaged-layer guard, one per fact. Both are
#: replayed on resume (``orchestrator._RESUME_REPLAYED``): the first run's guard changed the
#: shipped text, a resumed page restores that text and never runs the guard again, so the
#: events are the only record of what was changed.
CONTROL_BYTES_KIND = "control_bytes_replaced"
CONTROL_BYTES_EVENT_DETAIL = (
    "control bytes of this page's own text layer shipped as U+FFFD, never deleted: a deleted "
    "minus would turn a negative number positive. The mark states the count (#1306)"
)
HIDDEN_FOOTNOTE_KIND = "hidden_footnote_stripped"
HIDDEN_FOOTNOTE_EVENT_DETAIL = (
    "the invisible tagged copy of a footnote in the page's text layer was removed because the "
    "page also carries the printed footnote with the same number and the same full text; "
    "one event per wrapper (#1306)"
)
CONTROL_BYTES_NOTE_PREFIX = "page(s) ship control characters of their own text layer as U+FFFD"


def control_bytes_cli_line(pages: list[int]) -> str:
    return f"{len(pages)} {CONTROL_BYTES_NOTE_PREFIX} (named on each page): {pages}"


def control_bytes_note(pages: list[int]) -> str:
    return (
        CONTROL_BYTES_NOTE_PREFIX
        + ": "
        + ", ".join(str(n) for n in pages)
        + " (named on each page)"
    )


#: #1218 re-finalize parity. A run appends its document events in one block and its document
#: notes in one sequence; ``socr-refinalize-own-text`` must put #1185's and #1218's at the
#: position the run gives them, not after everything the document already carries. These name
#: what a run records AFTER them. Pinned against the orchestrator's source by
#: ``test_the_after_lists_cover_what_the_orchestrator_appends_later``.
EVENT_KINDS_AFTER_NUMBERS_DROPPED = (
    # #1294: recorded right after #1218's.
    WORDS_MISSING_KIND,
    "native_garbled_math_retained",
    "table_not_reconstructed_retained",
    "figure_words_unread_retained",
    "text_coverage_unknown_retained",
    "text_ink_unread_retained",
    "blank_page_retained",
    "native_only_table_distrusted",
    "flagged_model_table_kept",
    "d3_floor_model_table_kept",
    "table_region_unverifiable",
    "structure_class_model_table_kept",
    "judge_timeout_ladder_accepted",
    "page_judge_timeout_floor",
    "row_shape_not_reconcilable_text_table_floor",
    "structure_class_ladder_exhausted_floor",
    "figure_phase_failed",
    "figure_description_failed",
    "chart_region_duplicate_suppressed",
    "figure_cap_reached",
    "figure_caption_withheld_on_floor",
    "figure_description_dropped",
    "figure_recoverable_labels",
)
#: ``table_structure_failed`` is recorded earlier too; only the exact-final-body site follows.
FINAL_BODY_EVENT_SITE = "final_body"


def event_follows_numbers_dropped(event: dict) -> bool:
    """Whether a run records this event (as a dict) after its ``numbers_dropped`` one."""
    kind = event.get("kind")
    if kind in EVENT_KINDS_AFTER_NUMBERS_DROPPED:
        return True
    return (
        kind == "table_structure_failed"
        and (event.get("data") or {}).get("site") == FINAL_BODY_EVENT_SITE
    )


def event_follows_words_missing(event: dict) -> bool:
    """#1294: whether a run records this event after its ``words_missing_from_page`` one."""
    return event.get("kind") != WORDS_MISSING_KIND and event_follows_numbers_dropped(event)


def event_follows_numbers_disagree(event: dict) -> bool:
    """The same for #1185's event, which a run records just before #1218's."""
    return event.get("kind") == NUMBERS_DROPPED_KIND or event_follows_numbers_dropped(event)


#: The document note a run appends AFTER the numbers notes (``process``, exact-final-body
#: validation). Built from this constant at both ends.
INVALID_FINAL_EMISSION_NOTE_PREFIX = "invalid final table emission on page(s) "
NUMBERS_DROPPED_NOTE_PREFIX = "model reading omits numbers the page's own text layer prints"


def numbers_dropped_event_data(page_num: int, dropped) -> dict:
    """The audit event's ``data``: the page and its omitted numbers (written forms, reference
    order). One definition, so a run and ``socr-refinalize-own-text`` write the same."""
    return {"page": page_num, "dropped": list(dropped or ())}


def numbers_dropped_cli_line(pages: list[int]) -> str:
    """The CLI summary line for the pages ``text_layer_drops`` FAILs. The page carries the
    count only; the numbers are in the audit log, and this line must not say otherwise."""
    return (
        f"{len(pages)} model-read page(s) omit numbers the page's own text layer prints in "
        f"table rows (count on each page; numbers in the audit log): {pages}"
    )


def numbers_dropped_note(pages: list[int]) -> str:
    """The document's ``error`` note naming the pages ``text_layer_drops`` FAILs: written by
    a run and by ``socr-refinalize-own-text`` alike, so the two cannot word it differently."""
    return (
        NUMBERS_DROPPED_NOTE_PREFIX
        + " in table rows on page(s) "
        + ", ".join(str(n) for n in pages)
        + " (count on each page; numbers in the audit log)"
    )


NO_OWN_TEXT_REASON = "this page has no text layer of its own to check the reading's numbers against"
NO_NUMBER_REASON = "the reading ships no number to check against the page's text layer"
#: The same, on a page whose own layer DOES print numbers: the reading may have dropped
#: them. Counted, not named -- they include page numbers and running heads, and checking
#: drops is issue #1218 (owner/orchestrator ruling, 2026-10-20).
NO_NUMBER_LAYER_HAS_SOME_REASON = (
    "the reading ships no number, while the page's text layer prints {count} "
    "(table-row numbers it omits are named by the text_layer_drops check; other numbers a "
    "reading drops are not checked: #1218)"
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
    return text_layer_numbers_outcome(output, p).clause


#: #1183 pass trace reasons of the two model-reading checks (numbers and drops).
NOT_A_MODEL_READING_REASON = "this page ships native text, not a model's reading of the page"
NO_TEXT_REASON = "this page ships no text"
FAILED_MARKER_REASON = "this page ships socr's failure marker, so no reading ships to compare"
PRIOR_OCR_LAYER_NOT_EVIDENCE_REASON = (
    "this page's text layer is an earlier OCR run, which is not evidence (#1156)"
)
NUMBERS_PASS_REASON = "every number the reading ships occurs in the page's own text layer"


def _not_a_reading(
    check: Check, output: PageOutput, providers_available: bool, scope: str = SCOPE_PAGE
) -> CheckOutcome:
    """The outcome of a model-reading check on a page that ships no model reading. With a
    provider the page simply is not one (NOT_APPLICABLE); with none, no reading could have
    existed, so the check COULD have applied and had nothing to run on (NOT_RUN)."""
    if not providers_available:
        return _not_run(
            check, f"{NO_PROVIDER_LADDER_REASON}: no model reading exists to compare", scope
        )
    return _not_applicable(check, NOT_A_MODEL_READING_REASON, scope)


def text_layer_numbers_outcome(
    output: PageOutput, p: Any, providers_available: bool = True
) -> CheckOutcome:
    """:func:`text_layer_numbers_clause` with its outcome (#1183).

    PASS only when the comparison ran with ``compared > 0`` on an undamaged layer and every
    shipped number is in it. Every other no-clause exit names why it did not run."""
    from socr.tables.reading_agreement import (
        PROVENANCE_OWN_CHARACTERS,
        compare_numbers,
        describe_disputes,
    )

    from socr.core.manifest import is_page_failed_marker

    check = Check.TEXT_LAYER_NUMBERS
    if not _is_model_reading(output):
        return _not_a_reading(check, output, providers_available)
    if not (output.text or "").strip():
        return _not_applicable(check, NO_TEXT_REASON)
    # A page that ships socr's ``[page N failed: ...]`` marker ships none of the reading,
    # so there is nothing to compare; its marker already marks it. Without this the
    # sidecar carried an UNKNOWN clause that no line on the page states, because
    # ``_apply_verdict_marks`` never writes above a whole-page failure marker -- one
    # surface saying what another does not (found by the p6 stage C oracle, page 9).
    if is_page_failed_marker(output.text):
        return _not_run(check, FAILED_MARKER_REASON)
    if _flag(p, "prior_ocr_layer_suspect"):
        return _not_run(check, PRIOR_OCR_LAYER_NOT_EVIDENCE_REASON)
    reference = getattr(p, "own_layer_text", "")
    if not isinstance(reference, str) or not reference.strip():
        return _found(VerdictClause(check, CheckResult.UNKNOWN, SCOPE_PAGE, NO_OWN_TEXT_REASON))
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
        return _found(VerdictClause(check, CheckResult.UNKNOWN, SCOPE_PAGE, reason))
    if _layer_is_damaged(p):
        reason = (
            DAMAGED_LAYER_PREFIX + describe_disputes(record)
            if record.disputed
            else DAMAGED_LAYER_AGREES_REASON
        )
        return _found(VerdictClause(check, CheckResult.UNKNOWN, SCOPE_PAGE, reason))
    if record.disputed and any(_flag(p, f) for f in _COSMETIC_LAYER_FLAGS):
        return _found(
            VerdictClause(
                check,
                CheckResult.UNKNOWN,
                SCOPE_PAGE,
                DAMAGED_LAYER_PREFIX + describe_disputes(record),
            )
        )
    if record.disputed:
        return _found(VerdictClause(check, CheckResult.FAIL, SCOPE_PAGE, describe_disputes(record)))
    return _passed(check, reason=NUMBERS_PASS_REASON)


#: #1218: the reason of a ``text_layer_drops`` FAIL. COUNT ONLY, never the numbers: a mark
#: that printed layer numbers would ship values socr withheld (a table refused for a dropped
#: minus) and loose table text (owner ruling: table regions never ship loose text). The
#: numbers are in the document audit event's ``data`` (``NUMBERS_DROPPED_KIND``). "At least":
#: the comparison is a multiset over a generous reading, so it can hide a drop and never
#: invent one.
TEXT_LAYER_DROPS_REASON = (
    "at least {count} number(s) printed in table rows on this page's own text layer are "
    "missing from this reading (listed in the audit log, #1218)"
)


def drops_count_from_reason(reason: str) -> int | None:
    """The count a ``text_layer_drops`` clause's reason states, or ``None`` if it states none.

    The inverse of :data:`TEXT_LAYER_DROPS_REASON`, built from it so the two cannot drift.
    ``socr-refinalize-own-text`` holds a persisted event to this count before it will repair
    or replace the event's numbers."""
    head, tail = TEXT_LAYER_DROPS_REASON.split("{count}")
    m = re.fullmatch(re.escape(head) + r"(\d+)" + re.escape(tail), str(reason or "").strip())
    return int(m.group(1)) if m else None


def _table_withheld_by_socr(output: PageOutput, p: Any) -> bool:
    """Whether socr itself withheld a table on this page (#1218 merge gate, Fable P2).

    The withheld numbers are missing BY DESIGN, so "the model reading omits them" would be a
    false attribution (and the audit event would list the withheld values). Three witnesses,
    any one enough: the page's failure mode, the table ladder's disposition, and the marker
    socr wrote into the body (``table_counts.carries_withheld_marker``)."""
    from socr.core.result import FailureMode
    from socr.core.table_counts import carries_withheld_marker

    withheld_modes = {FailureMode.TABLE_WITHHELD, FailureMode.TABLE_WITHHELD_UNCONFIRMED}
    return (
        output.failure_mode in withheld_modes
        or getattr(p, "table_ladder_disposition", None) is FailureMode.TABLE_WITHHELD
        or carries_withheld_marker(output.text)
    )


def page_dropped_numbers(output: PageOutput, p: Any) -> tuple[str, ...]:
    """#1218: the written forms, in reference order, of the table-row numbers the page's own
    layer prints and a model reading omits. ``()`` when the check does not apply.

    Same preconditions as :func:`text_layer_numbers_clause` (a model reading with text, not a
    page-failed marker, not an earlier OCR layer), and ``()`` -- never UNKNOWN -- when:

    * there is no own layer, or the layer is damaged (``_layer_is_damaged``): a damaged layer
      has fewer or wrong numbers, the wrong direction for counting omissions;
    * socr itself withheld a table on the page (``_table_withheld_by_socr``): the missing
      numbers are withheld, not dropped by the model;
    * the page's drop reference is EMPTY. Empty means NOT CHECKED (the assessment could not
      classify the layer, or no table row qualified), never "zero dropped".

    The cosmetic-layer flag does NOT suppress it (orchestrator decision, 2026-11-06): a fused
    pair can hide a layer number from the reference, which makes the count lower, not higher.
    Uses ``tables.reading_agreement.dropped_numbers`` over ``reading_numbers_text``. The
    reading's side is NOT reduced by the page's running heads and folios (see
    ``dropped_numbers``): the count is a lower bound, and a transcribed head MAY absorb a
    dropped body number of the same value.
    """
    return _dropped_numbers_outcome(output, p)[0]


DROPS_LAYER_NOT_USABLE_REASON = (
    "this page's own text layer is missing, damaged or an earlier OCR run, so it cannot "
    "count what a reading omitted"
)
DROPS_TABLE_WITHHELD_REASON = (
    "socr withheld a table on this page, so the numbers missing from it are missing by design"
)
DROPS_NO_REFERENCE_REASON = (
    "the page's drop reference is empty (not computed, or no table row qualified): not checked"
)
DROPS_PASS_REASON = "no number printed in this page's table rows is missing from the reading"
#: #1183: the scope of every NON-clause ``text_layer_drops`` entry. The check compares the
#: numbers of the layer's TABLE ROWS and nothing else (#1218), so a PASS names that region and
#: never reads as a page pass (#1181). The FAIL clause keeps its page scope: it is a verdict
#: clause and the verdict is not changed by the trace.
SCOPE_TABLE_ROWS = "table_rows"


def _dropped_numbers_outcome(
    output: PageOutput, p: Any, providers_available: bool = True
) -> tuple[tuple[str, ...], CheckOutcome]:
    """The dropped numbers (``()`` when the check does not apply) and the check's outcome.
    The ONE body of :func:`page_dropped_numbers` and :func:`text_layer_drops_outcome`, so the
    list the clause counts and the outcome the trace records cannot come from two copies."""
    from socr.core.manifest import is_page_failed_marker
    from socr.tables.reading_agreement import dropped_numbers

    check = Check.TEXT_LAYER_DROPS
    if not _is_model_reading(output):
        return (), _not_a_reading(check, output, providers_available, SCOPE_TABLE_ROWS)
    if not (output.text or "").strip():
        return (), _not_applicable(check, NO_TEXT_REASON, SCOPE_TABLE_ROWS)
    if is_page_failed_marker(output.text):
        return (), _not_run(check, FAILED_MARKER_REASON, SCOPE_TABLE_ROWS)
    if _flag(p, "prior_ocr_layer_suspect") or _layer_is_damaged(p):
        return (), _not_run(check, DROPS_LAYER_NOT_USABLE_REASON, SCOPE_TABLE_ROWS)
    if _table_withheld_by_socr(output, p):
        return (), _not_run(check, DROPS_TABLE_WITHHELD_REASON, SCOPE_TABLE_ROWS)
    own = getattr(p, "own_layer_text", "")
    reference = getattr(p, "own_layer_drop_reference", "")
    if not isinstance(own, str) or not own.strip():
        return (), _not_run(check, DROPS_LAYER_NOT_USABLE_REASON, SCOPE_TABLE_ROWS)
    if not isinstance(reference, str) or not reference.strip():
        return (), _not_run(check, DROPS_NO_REFERENCE_REASON, SCOPE_TABLE_ROWS)
    record = dropped_numbers(
        reference,
        reading_numbers_text(output.text, output.page_num),
    )
    dropped = tuple(record.dropped)
    if dropped:
        return dropped, _found(
            VerdictClause(
                check,
                CheckResult.FAIL,
                SCOPE_PAGE,
                TEXT_LAYER_DROPS_REASON.format(count=len(dropped)),
            )
        )
    return (), _passed(check, SCOPE_TABLE_ROWS, DROPS_PASS_REASON)


def text_layer_drops_outcome(
    output: PageOutput, p: Any, providers_available: bool = True
) -> CheckOutcome:
    """:func:`text_layer_drops_clause` with its outcome (#1183)."""
    return _dropped_numbers_outcome(output, p, providers_available)[1]


def text_layer_drops_clause(output: PageOutput, p: Any) -> VerdictClause | None:
    """#1218: FAIL (count only) when :func:`page_dropped_numbers` finds any; else nothing."""
    dropped = page_dropped_numbers(output, p)
    if not dropped:
        return None
    return VerdictClause(
        Check.TEXT_LAYER_DROPS,
        CheckResult.FAIL,
        SCOPE_PAGE,
        TEXT_LAYER_DROPS_REASON.format(count=len(dropped)),
    )


#: #1294: the reason of a ``text_layer_words`` clause. The words are named: they are the
#: page's own printed words, never a value socr withheld (a page where socr withheld a
#: table gets no clause), and a reader can judge in seconds whether they matter.
TEXT_LAYER_WORDS_REASON = (
    "{count} word(s) this page's own text layer prints are missing from it: {words}"
)
#: The same on a layer socr's page analysis refused or found glyph-damaged: the missing
#: words may be the layer's own damage, so the result is UNKNOWN, still named.
TEXT_LAYER_WORDS_DAMAGED_REASON = (
    "this page's text layer is damaged, and it prints {count} word(s) missing from the "
    "page: {words}"
)
_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def _word_layer_is_damaged(p: Any) -> bool:
    """A layer whose WORDS may not be the page's: refused by the page analysis, or set in a
    symbol font it could not map. The digit-damage flags of ``_layer_is_damaged`` say
    nothing about words, so they are not used here."""
    return not _flag(p, "is_born_digital") or _flag(p, "has_unrecovered_symbol_glyphs")


def _stripped_footnote_words(output: PageOutput, p: Any) -> str:
    """#1306: the words of the hidden footnote wrappers the damaged-layer guard stripped from
    this page, which its own layer (the words reference) holds twice (hidden and printed) and
    the page now ships once. Only for a page that ships the layer's own text: a model reading
    never carried the wrapper, and a field left by an earlier finalisation must not excuse it."""
    from socr.core.manifest import engine_ships_native_text

    words = getattr(p, "hidden_footnote_words", "")
    if not isinstance(words, str) or not words:
        return ""
    return words if engine_ships_native_text(output.engine) else ""


def page_missing_words(output: PageOutput, p: Any) -> tuple[str, ...] | None:
    """#1294: the words the page's own layer prints and the shipped page lacks, in layer
    order; ``()`` when none; ``None`` when the check does not apply.

    Applies to every engine (a native page loses words to socr's own table, equation and
    column handling, a model reading to the model). Does not apply, and records nothing,
    when: the page ships no text or socr's page-failed marker; its layer is an earlier OCR
    run (#1156 marks it); socr withheld a table on it (those words are missing by design);
    or the reference is EMPTY (not computed, or no word), which means NOT CHECKED.
    """
    return _missing_words_outcome(output, p)[0]


WORDS_NO_REFERENCE_REASON = (
    "this page's own text layer gave no word reference (not computed on this run, or the "
    "layer holds no word): not checked"
)
WORDS_NO_LAYER_REASON = "this page has no text layer of its own to take words from"
WORDS_TABLE_WITHHELD_REASON = (
    "socr withheld a table on this page, so the words missing from it are missing by design"
)
WORDS_PASS_REASON = "every word this page's own text layer prints is on the page"
WORDS_NONE_OWED_REASON = (
    "the page's own text layer owes no word to compare (a figure ships in place of its "
    "words, or no word remains after the exclusions): nothing was checked"
)
WORDS_LAYER_NOT_EVIDENCE_REASON = (
    "this page's text layer was refused by the page analysis or is glyph-damaged, so its "
    "words agreeing with the page is not evidence: the check did not pass"
)


def _words_owed(output: PageOutput, p: Any) -> int:
    """How many words the page's own layer owes the shipped page: the ``compared`` count of the
    sidecar's ``text_layer_words`` record, from the one body (a figure that ships takes its words
    off the debt, as :func:`missing_words` does)."""
    from socr.tables.layer_words import word_tokens

    figure = getattr(p, "own_layer_figure_words", "") or ""
    owed = word_tokens(getattr(p, "own_layer_word_reference", "") or "")
    if not _IMAGE_LINK_RE.search(output.text or ""):
        owed += word_tokens(figure)
    # The hidden footnote wrapper the page ships once instead of twice is not owed, exactly as
    # :func:`missing_words` subtracts it: a page whose only words are those has nothing compared.
    owed -= word_tokens(_stripped_footnote_words(output, p))
    return sum(owed.values())


def _words_clause(missing: tuple[str, ...], p: Any) -> VerdictClause:
    """The ``text_layer_words`` clause for a non-empty *missing* (UNKNOWN on a damaged layer)."""
    if _word_layer_is_damaged(p):
        result, template = CheckResult.UNKNOWN, TEXT_LAYER_WORDS_DAMAGED_REASON
    else:
        result, template = CheckResult.FAIL, TEXT_LAYER_WORDS_REASON
    reason = template.format(count=len(missing), words=_name_words(missing))
    return VerdictClause(Check.TEXT_LAYER_WORDS, result, SCOPE_PAGE, reason)


def _missing_words_outcome(
    output: PageOutput, p: Any
) -> tuple[tuple[str, ...] | None, CheckOutcome]:
    """The missing words (``None`` when the check does not apply) and the check's outcome: the
    ONE body of :func:`page_missing_words` and :func:`text_layer_words_outcome`."""
    from socr.core.manifest import is_page_failed_marker
    from socr.tables.layer_words import WordReference, missing_words

    check = Check.TEXT_LAYER_WORDS
    text = output.text or ""
    if not text.strip():
        return None, _not_applicable(check, NO_TEXT_REASON)
    if is_page_failed_marker(text):
        return None, _not_run(check, FAILED_MARKER_REASON)
    if _flag(p, "prior_ocr_layer_suspect"):
        return None, _not_run(check, PRIOR_OCR_LAYER_NOT_EVIDENCE_REASON)
    if _table_withheld_by_socr(output, p):
        return None, _not_run(check, WORDS_TABLE_WITHHELD_REASON)
    reference = getattr(p, "own_layer_word_reference", "")
    figure = getattr(p, "own_layer_figure_words", "")
    reference = reference if isinstance(reference, str) else ""
    figure = figure if isinstance(figure, str) else ""
    # Either field counts: a page whose only words sit in a chart is still checked.
    if not reference.strip() and not figure.strip():
        own = getattr(p, "own_layer_text", "")
        has_layer = _flag(p, "is_born_digital") or (isinstance(own, str) and bool(own.strip()))
        if not has_layer:
            return None, _not_applicable(check, WORDS_NO_LAYER_REASON)
        return None, _not_run(check, WORDS_NO_REFERENCE_REASON)
    shipped = _HTML_COMMENT_RE.sub(" ", _IMAGE_LINK_RE.sub(" ", text))
    missing = tuple(
        missing_words(
            WordReference(reference, figure),
            reading_numbers_text(shipped, output.page_num),
            figure_shipped=bool(_IMAGE_LINK_RE.search(text)),
            excused=_stripped_footnote_words(output, p),
        )
    )
    if missing:
        return missing, _found(_words_clause(missing, p))
    # PASS only for a check that RAN with something to compare, on a layer whose words are
    # evidence. A damaged layer is asked first: agreement with it proves nothing (the numbers
    # check does the same), and it is not a clause (the verdict is unchanged), so it is NOT_RUN.
    if _word_layer_is_damaged(p):
        return missing, _not_run(check, WORDS_LAYER_NOT_EVIDENCE_REASON)
    if _words_owed(output, p) == 0:
        return missing, _not_applicable(check, WORDS_NONE_OWED_REASON)
    return missing, _passed(check, reason=WORDS_PASS_REASON)


def text_layer_words_outcome(output: PageOutput, p: Any) -> CheckOutcome:
    """:func:`text_layer_words_clause` with its outcome (#1183)."""
    return _missing_words_outcome(output, p)[1]


def page_word_check_record(output: PageOutput, p: Any) -> dict | None:
    """#1294: what the words check did on this page, PASS or not, for the page's sidecar.

    ``{"compared": owed words, "missing": missing words}``, or ``None`` when the check did
    not apply (:func:`page_missing_words`). A clean page thereby shows it was checked,
    rather than only that nothing was found (fix-plan rule: a check records its passes).
    """
    missing = page_missing_words(output, p)
    if missing is None:
        return None
    return {"compared": _words_owed(output, p), "missing": len(missing)}


def _name_words(words: tuple[str, ...]) -> str:
    return ", ".join(dict.fromkeys(words))


#: #1306: the reason of a ``control_bytes`` clause. A control byte stands for a glyph the layer
#: could not map; in this corpus it is often a minus, so the numbers beside it may be wrong, and
#: the page ships U+FFFD where it stood rather than a clean-looking number.
CONTROL_BYTES_REASON = (
    "{count} control character(s) in this page's own text layer ({codes}) could not be read "
    "and ship as U+FFFD; each stood for a glyph (often a minus), so numbers beside them "
    "may be wrong"
)

CONTROL_BYTES_RAW_REASON = (
    "{count} raw control character(s) ({codes}) ship in this page's text; each can stand for a "
    "glyph (often a minus), so numbers beside them may be wrong"
)


def _recorded_control_bytes_clause(output: PageOutput, p: Any) -> VerdictClause | None:
    """The clause for control bytes the guard replaced (its record and U+FFFD in the bytes)."""
    from socr.core.damaged_layer import REPLACEMENT_CHAR

    found = getattr(p, "control_bytes_replaced", None)
    if isinstance(found, dict) and found and REPLACEMENT_CHAR in (output.text or ""):
        codes = ", ".join(f"{code} x{n}" for code, n in sorted(found.items()))
        reason = CONTROL_BYTES_REASON.format(count=sum(found.values()), codes=codes)
        return VerdictClause(Check.CONTROL_BYTES, CheckResult.FAIL, SCOPE_PAGE, reason)
    return None


def control_bytes_clause(output: PageOutput, p: Any) -> VerdictClause | None:
    """#1306: FAIL when the damaged-layer guard replaced control bytes in this page's text.

    Both halves are required: the guard's record on the page state, and U+FFFD in the bytes
    that ship. The record alone could be left by an earlier finalisation of a text a later
    one replaced; the character alone could be any reading's. Status-independent, so a page
    already WARNING for another reason gains the true second reason (see ``derive_clauses``).

    A raw control byte the guard never replaced is a finding of the TRACE only
    (:func:`control_bytes_outcome`): it does not move the status, which is the guard's to set.
    """
    return _recorded_control_bytes_clause(output, p)


CONTROL_BYTES_PASS_REASON = (
    "the damaged-layer guard scanned this page's text and found no control byte"
)


def control_bytes_outcome(output: PageOutput, p: Any) -> CheckOutcome:
    """:func:`control_bytes_clause` with its outcome (#1183). ``_apply_damaged_layer_guard``
    scans every shipped text that is not a failure marker, whatever its engine, so on such a
    page a clean scan is a recorded negative (PASS).

    The shipped bytes are the evidence, not the guard's bookkeeping: a tool that rewrote the page
    offline (a spliced layer cell, a masked body) never ran the guard, so a raw control byte in
    the text is a FAIL entry whatever the page state says, and PASS needs a clean scan of the
    text itself. That entry carries no clause: it records the finding without changing a status
    the guard did not set."""
    from socr.core.damaged_layer import CONTROL_BYTE_RE
    from socr.core.manifest import is_page_failed_marker

    text = output.text or ""
    recorded = _recorded_control_bytes_clause(output, p)
    if recorded is not None:
        return _found(recorded)
    raw = Counter(f"U+{ord(m.group()):04X}" for m in CONTROL_BYTE_RE.finditer(text))
    if raw:
        codes = ", ".join(f"{code} x{n}" for code, n in sorted(raw.items()))
        reason = CONTROL_BYTES_RAW_REASON.format(count=sum(raw.values()), codes=codes)
        return CheckOutcome(
            TraceEntry(Check.CONTROL_BYTES.value, TraceResult.FAIL, SCOPE_PAGE, reason)
        )
    if not text.strip():
        return _not_applicable(Check.CONTROL_BYTES, NO_TEXT_REASON)
    if is_page_failed_marker(text):
        return _not_run(Check.CONTROL_BYTES, FAILED_MARKER_REASON)
    return _passed(Check.CONTROL_BYTES, reason=CONTROL_BYTES_PASS_REASON)


def text_layer_words_clause(output: PageOutput, p: Any) -> VerdictClause | None:
    """#1294: FAIL naming the words when :func:`page_missing_words` finds any (UNKNOWN on a
    damaged layer, :func:`_word_layer_is_damaged`); else nothing."""
    missing = page_missing_words(output, p)
    if not missing:
        return None
    return _words_clause(missing, p)


#: The order ``silent_success_clauses`` emits its clauses in, as ``(check, scope kind)``.
#: Named here, beside the function that defines it, so the two cannot drift.
#:
#: ``FIGURE_TEXT`` appears TWICE and the pair is what tells the two apart: the chart lane's
#: clause is page-scoped and comes early, this lane's names a region and comes last before
#: the numbers clause.
_SILENT_SUCCESS_ORDER: tuple[tuple[Check, str], ...] = (
    (Check.EQUATION_FIDELITY, SCOPE_PAGE),
    (Check.FIGURE_TEXT, SCOPE_PAGE),
    (Check.PAGE_JUDGE, SCOPE_PAGE),
    (Check.PRIOR_OCR_LAYER, SCOPE_PAGE),
    (Check.FIGURE_TEXT, "figure"),
    (Check.TEXT_LAYER_NUMBERS, SCOPE_PAGE),
    # #1218: LAST, so ``with_text_layer_numbers_clause``'s count of the clauses ahead of
    # #1185's slot is unchanged.
    (Check.TEXT_LAYER_DROPS, SCOPE_PAGE),
    # #1294: after #1218's, for the same reason: the count of clauses ahead of #1185's slot
    # is unchanged.
    (Check.TEXT_LAYER_WORDS, SCOPE_PAGE),
    # #1306: last again, for the same reason.
    (Check.CONTROL_BYTES, SCOPE_PAGE),
)


def in_fresh_run_order(clauses):
    """*clauses* reordered as a fresh run would emit them.

    The silent-success clauses first, in :data:`_SILENT_SUCCESS_ORDER`, then every
    status-dependent clause in the order it already had. Stable, so two clauses that rank the
    same keep their relative positions.

    For a re-finalize tool, which merges a clause into a verdict some earlier run or tool
    wrote. Inserting at a computed SLOT cannot do this: the recorded list may itself not be
    in fresh-run order (it came from another tool), and then no single insertion point gives
    the order a fresh run would (owner item 4, Fable P2). Sorting does not care how the list
    got into the state it is in.
    """
    rank = {pair: i for i, pair in enumerate(_SILENT_SUCCESS_ORDER)}

    def key(item):
        index, clause = item
        kind = str(clause.scope).split(":", 1)[0]
        return (rank.get((clause.check, kind), len(rank)), index)

    return [clause for _i, clause in sorted(enumerate(clauses), key=key)]


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
    drops = text_layer_drops_clause(output, p)
    words = text_layer_words_clause(output, p)
    control = control_bytes_clause(output, p)
    clauses = (
        *_clauses_ahead_of_text_layer_numbers(output, p, event_kinds),
        *((numbers,) if numbers is not None else ()),
        *((drops,) if drops is not None else ()),
        *((words,) if words is not None else ()),
        *((control,) if control is not None else ()),
    )
    # #1251: LAST, and only on a page the other checks leave SUCCESS: a page that already carries
    # a clause is already demoted and marked, and a second line would say nothing new.
    judge_only = () if clauses else _judge_only_clauses(output, p, event_kinds)
    return (*clauses, *judge_only)


# --- the page judge is not a check (#1251, C3) --------------------------------------------


def independent_page_evidence(entries: Iterable[TraceEntry], text: str) -> bool:
    """Does a pass trace hold an independent confirmation of the page text *text*?

    Yes only when, after every PASS not computed on *text* is dropped
    (:func:`verify_pass_digests`): the words check passed at page scope, the numbers check passed
    at page scope or does not apply, and no entry of any check is a FAIL or UNKNOWN. The words
    and numbers checks compare the shipped page with the page's OWN text layer, which no model
    wrote; a table-scope pass (row binding) says nothing about the rest of the page (#1181), and a
    NOT_RUN is an absence of evidence, so neither counts. Pure over the entries: it reads no
    judge field, so a reworded judge reason cannot move it."""
    checked = verify_pass_digests(entries, text)
    if any(e.result in (TraceResult.FAIL, TraceResult.UNKNOWN) for e in checked):
        return False

    def at_page(check: Check, *wanted: TraceResult) -> bool:
        return any(
            e.check == check.value and e.scope == SCOPE_PAGE and e.result in wanted for e in checked
        )

    return at_page(Check.TEXT_LAYER_WORDS, TraceResult.PASS) and at_page(
        Check.TEXT_LAYER_NUMBERS, TraceResult.PASS, TraceResult.NOT_APPLICABLE
    )


def _judge_accepted(output: PageOutput) -> bool:
    """The page judge COMPLETED and accepted this reading (typed fields, never the reason)."""
    from socr.core.result import JUDGE_OUTCOME_COMPLETED

    return output.judge_outcome == JUDGE_OUTCOME_COMPLETED and output.judge_accepted is True


def _judge_only_clauses(
    output: PageOutput, p: Any, event_kinds: frozenset[str]
) -> tuple[VerdictClause, ...]:
    """``page_judge=UNKNOWN`` on a model reading whose acceptance is the page judge's alone.

    #1251. Every defective model page the four auditors found had been judged faithful by a
    model call on the same page, so the acceptance carries no information: it may warn or reject
    (untouched), it does not confirm. The predicate is the ABSENCE of independent evidence on the
    shipped text (:func:`independent_page_evidence`), evaluated from the checks' own outcomes, so
    it does not read the judge's reason. Fires on a page still SUCCESS, or on one this clause
    itself demoted (every later pass derives it again, like #1181's).

    Not fired when the page was accepted without a judge (#1181 owns that), and not on a page the
    judge did not accept: those carry their own refusal clause.
    """
    if not (
        output.status is PageStatus.SUCCESS
        or output.failure_mode is FailureMode.READING_UNCORROBORATED
    ):
        return ()
    if not _is_model_reading(output) or not _judge_accepted(output):
        return ()
    if _exact_pass_unjudged_outcome(output, event_kinds).clause is not None:
        return ()
    text = output.text or ""
    entries = [
        stamp_text_digest(o.entry, text)
        for o in pass_trace_outcomes(output, p, event_kinds, providers_available=True)
    ]
    return _judge_only_clause_on(entries, text)


def _judge_only_clause_on(entries: Iterable[TraceEntry], text: str) -> tuple[VerdictClause, ...]:
    """The one clause, when *entries* (a pass trace) hold no independent evidence on *text*."""
    if independent_page_evidence(entries, text):
        return ()
    return (VerdictClause(Check.PAGE_JUDGE, CheckResult.UNKNOWN, SCOPE_PAGE, JUDGE_ONLY_REASON),)


def is_judge_only_clause(clause: VerdictClause) -> bool:
    return clause.check is Check.PAGE_JUDGE and clause.reason == JUDGE_ONLY_REASON


def with_judge_only_clause(
    published: PageVerdict,
    output: PageOutput,
    old_checks: Any,
    event_kinds: frozenset[str] = frozenset(),
    recomputed: tuple[TraceEntry, ...] = (),
    row_binding_records: Any = None,
) -> PageVerdict:
    """*published* (the verdict an OFFLINE tool is about to write) with #1251's clause as a live
    run of this code derives it, from the trace the tool will write.

    The same predicate as :func:`_judge_only_clauses` (accepted model reading, no #1181 clause,
    :func:`independent_page_evidence`), evaluated on :func:`reconcile_pass_trace` of
    *old_checks* with the tool's *recomputed* checks and *row_binding_records*, stamped with the
    text it ships. A sidecar with no readable trace has none to reconcile: the tool's own
    *recomputed* outcomes are then the whole evidence, and with none the page has none.

    A live run gives the clause only to a page the other checks leave SUCCESS. So when the
    verdict carries any other clause the judge-only one is dropped (the page's mode is then
    re-chosen by the seam, as a fresh run chooses it); when it carries none and the clause is
    already there, or evidence exists, the verdict is returned as it stands (an offline tool
    never lifts a mark).
    """
    text = output.text or ""
    others = tuple(c for c in published.clauses if not is_judge_only_clause(c))
    if others:
        if len(others) == len(published.clauses):
            return published
        return replace(published, clauses=others)
    if published.clauses:  # only the judge-only clause: already there
        return published
    if output.status is not PageStatus.SUCCESS:
        return published
    if not _is_model_reading(output) or not _judge_accepted(output):
        return published
    if _exact_pass_unjudged_outcome(output, event_kinds).clause is not None:
        return published
    reconciled = reconcile_pass_trace(
        old_checks,
        published,
        row_binding_records=row_binding_records,
        recomputed=recomputed,
        text=text,
    )
    if reconciled is None:
        entries = [stamp_text_digest(e, text) for e in recomputed]
    else:
        from socr.core.pass_trace import trace_from_dicts

        entries = list(trace_from_dicts(reconciled) or ())
    clauses = _judge_only_clause_on(entries, text)
    if not clauses:
        return published
    return replace(published, clauses=clauses)


# --- the pass trace (#1183) ---------------------------------------------------------------

#: The trace name of the focused row-binding check (#1267). It is a TABLE check whose failures
#: surface as table marks and withheld tables, not as ``Check`` clauses, so it is traced by
#: name. Its entries carry ``table:N`` scopes (N is the mark's own 1-based ordinal).
ROW_BINDING_CHECK = "row_binding"

ROW_BINDING_NO_TABLE_REASON = "this page ships no markdown table to bind"
ROW_BINDING_NATIVE_REASON = (
    "this table is the layer's own text; the check binds model-read tables to the layer"
)
ROW_BINDING_NOT_RUN_REASON = (
    "the check recorded no result for this model-read table (a layer it may not judge, a "
    "native-only run, or a page that did not load)"
)
ROW_BINDING_PASS_REASON = (
    "every row label of this table sits on the text-layer line that prints its values"
)
ROW_BINDING_FAIL_REASON = "row labels of this table do not bind to the page's text-layer values"
ROW_BINDING_UNKNOWN_REASON = "row binding of this table could not be settled on the text layer"
_ROW_BINDING_RESULTS = {
    "bound": (TraceResult.PASS, ROW_BINDING_PASS_REASON),
    "contradicted": (TraceResult.FAIL, ROW_BINDING_FAIL_REASON),
    "unverifiable": (TraceResult.UNKNOWN, ROW_BINDING_UNKNOWN_REASON),
}


def row_binding_record_entries(records: dict) -> list[TraceEntry]:
    """One trace entry per table record of the row-binding check (``PageState.row_binding``, or
    the sidecar's ``row_binding``): PASS for bound, FAIL for contradicted, UNKNOWN for
    unverifiable, at the table's own ``table:N`` scope (N is the mark's 1-based ordinal)."""
    entries: list[TraceEntry] = []
    for table_id, record in sorted(records.items(), key=lambda kv: str(kv[0])):
        if not isinstance(record, dict):
            continue
        try:
            scope = region_scope("table", int(record.get("table_index")) + 1)
        except (TypeError, ValueError):
            logger.warning("#1183: unreadable row-binding record for %r", table_id)
            continue
        result, reason = _ROW_BINDING_RESULTS.get(
            str(record.get("outcome")), (TraceResult.UNKNOWN, ROW_BINDING_UNKNOWN_REASON)
        )
        sha = record.get("table_sha")
        entries.append(
            TraceEntry(
                ROW_BINDING_CHECK,
                result,
                scope,
                reason,
                table_sha=sha if isinstance(sha, str) else "",
            )
        )
    return entries


ROW_BINDING_NOT_SHIPPED_REASON = (
    "the table this record is about does not ship in the output (socr withheld or replaced "
    "it, or the page ships another text), so its binding says nothing about the shipped page"
)


def row_binding_outcomes(output: PageOutput, p: Any) -> list[CheckOutcome]:
    """One outcome per table of the SHIPPED text for the row-binding check.

    A PASS may exist only for a table that ships: the record is stated (PASS for bound, FAIL for
    contradicted, UNKNOWN for unverifiable) only when the output is a model reading, is not a
    failure marker and the record's ``table_index`` names one of the table blocks the shipped
    text holds. Every table block that ships with no record is NOT_RUN. A record whose table
    does not ship (a page-granular withhold, a late demotion to a marker) is NOT_RUN with the
    reason, at the table's own scope. A page that ships no table is NOT_APPLICABLE, and so is a
    native text (the check binds model-read tables). A bound table never reads as a page pass."""
    from socr.core.manifest import is_page_failed_marker
    from socr.tables.reconcile import find_table_blocks

    raw = getattr(p, "row_binding", None)
    records = raw if isinstance(raw, dict) else {}
    text = output.text or ""
    blocks = len(find_table_blocks(text))
    reading = _is_model_reading(output) and not is_page_failed_marker(text)
    shipped_reading_tables = blocks if reading else 0
    entries: dict[str, TraceEntry] = {}
    for table_id, record in sorted(records.items(), key=lambda kv: str(kv[0])):
        stated = row_binding_record_entries({table_id: record})
        if not stated:
            continue
        scope = stated[0].scope
        ordinal = int(scope.split(":", 1)[1])
        if 1 <= ordinal <= shipped_reading_tables:
            entries.setdefault(scope, stated[0])
        else:
            entries.setdefault(
                scope,
                TraceEntry(
                    ROW_BINDING_CHECK, TraceResult.NOT_RUN, scope, ROW_BINDING_NOT_SHIPPED_REASON
                ),
            )
    for k in range(1, blocks + 1):
        scope = region_scope("table", k)
        if scope in entries:
            continue
        entries[scope] = (
            _not_run(ROW_BINDING_CHECK, ROW_BINDING_NOT_RUN_REASON, scope).entry
            if reading
            else _not_applicable(ROW_BINDING_CHECK, ROW_BINDING_NATIVE_REASON, scope).entry
        )
    if not entries:
        return [_not_applicable(ROW_BINDING_CHECK, ROW_BINDING_NO_TABLE_REASON)]
    return [CheckOutcome(e) for _, e in sorted(entries.items(), key=_scope_order)]


def _scope_order(item: tuple[str, TraceEntry]) -> tuple[str, int]:
    kind, _, ordinal = item[0].partition(":")
    return kind, int(ordinal) if ordinal.isdigit() else 0


def pass_trace_outcomes(
    output: PageOutput,
    p: Any,
    event_kinds: frozenset[str] = frozenset(),
    providers_available: bool = True,
    equation_lane_enabled: bool = True,
) -> list[CheckOutcome]:
    """Every check this module can state an outcome for, on this page, in a fixed order."""
    return [
        text_layer_numbers_outcome(output, p, providers_available),
        text_layer_drops_outcome(output, p, providers_available),
        text_layer_words_outcome(output, p),
        control_bytes_outcome(output, p),
        _prior_ocr_layer_outcome(output, p),
        _equation_fidelity_outcome(event_kinds, providers_available, equation_lane_enabled),
        _chart_values_outcome(event_kinds),
        _exact_pass_unjudged_outcome(output, event_kinds),
        *row_binding_outcomes(output, p),
    ]


#: The slots of every traced check except ``row_binding`` (its entries are per table, at the
#: shipped text's own ``table:N`` scopes, so there is no page slot for it). A page restored from
#: a sidecar that recorded no trace states each of them NOT_RUN: its checks did not run on this
#: run, and the run that did them left no record of the outcome.
PASS_TRACE_SLOTS: tuple[tuple[str, str], ...] = (
    (Check.TEXT_LAYER_NUMBERS.value, SCOPE_PAGE),
    (Check.TEXT_LAYER_DROPS.value, SCOPE_TABLE_ROWS),
    (Check.TEXT_LAYER_WORDS.value, SCOPE_PAGE),
    (Check.CONTROL_BYTES.value, SCOPE_PAGE),
    (Check.PRIOR_OCR_LAYER.value, SCOPE_PAGE),
    (Check.EQUATION_FIDELITY.value, SCOPE_PAGE),
    (Check.FIGURE_TEXT.value, SCOPE_PAGE),
    (Check.PAGE_JUDGE.value, SCOPE_PAGE),
)
RESUMED_WITHOUT_TRACE_REASON = (
    "this page was restored from a sidecar that recorded no pass trace; its checks were "
    "not run again and their outcomes were not recorded"
)


def build_pass_trace(
    output: PageOutput,
    p: Any,
    verdict: PageVerdict,
    event_kinds: frozenset[str] = frozenset(),
    providers_available: bool = True,
    replay: tuple[TraceEntry, ...] | None = None,
    resumed_without_trace: bool = False,
    equation_lane_enabled: bool = True,
) -> tuple[TraceEntry, ...]:
    """The page's pass trace: every verdict clause, plus each other check's own outcome.

    * *replay* (a terminal page restored from a sidecar that carries a trace) keeps its entries:
      a resumed page does not run its checks again, so recomputing would invent outcomes for a
      run that did not happen. Only the clauses of *verdict* are overlaid on it (each replaces
      the replayed entry of its check and scope), because the verdict is the authority for what
      was found and a resumed page can be demoted after the original run (the fail-closed
      branch of :func:`build_page_verdict`, a late assemble guard): the trace never says PASS
      beside a clause of the same check.
    * Every clause of *verdict* is an entry (FAIL or UNKNOWN), including the status-dependent
      ones and the fail-closed fallback: the verdict is the authority for what was found.
    * Every other check contributes the outcome it reported itself. A check that found a
      clause the verdict does not carry (a published verdict, a patched check) is kept as the
      finding it is (FAIL or UNKNOWN), as :func:`reconcile_pass_trace` keeps a recomputed one: the
      check's slot never disappears, so a live run and an offline tool state the same trace.
      The clauses of the verdict itself always win over any entry of their own check and scope.
    * *resumed_without_trace* (a restored page whose sidecar predates the trace): the clauses
      stand and every other slot is NOT_RUN with :data:`RESUMED_WITHOUT_TRACE_REASON`.

    No entry is ever derived from the absence of a clause. Guard-derived clauses (chart
    region, native glyphs, table label, the judge refusal) keep no record of having run, so
    they appear only when they fire.
    """
    entries: dict[tuple[str, str], TraceEntry] = {}
    if replay is not None:
        entries.update({(e.check, e.scope): e for e in replay})
    for clause in verdict.clauses:
        entries[(_check_name(clause.check), clause.scope)] = entry_from_clause(clause)
    if replay is None and resumed_without_trace:
        from socr.tables.reconcile import find_table_blocks

        slots = list(PASS_TRACE_SLOTS)
        # The row-binding slots are the tables the shipped text holds, not an invented page slot.
        slots += [
            (ROW_BINDING_CHECK, region_scope("table", k))
            for k in range(1, len(find_table_blocks(output.text or "")) + 1)
        ]
        for check, scope in slots:
            entries.setdefault(
                (check, scope),
                TraceEntry(check, TraceResult.NOT_RUN, scope, RESUMED_WITHOUT_TRACE_REASON),
            )
    elif replay is None:
        for outcome in pass_trace_outcomes(
            output, p, event_kinds, providers_available, equation_lane_enabled
        ):
            entries.setdefault(
                (outcome.entry.check, outcome.entry.scope),
                stamp_text_digest(outcome.entry, output.text or ""),
            )
    # Whatever the entry's source, a PASS stays only for the text that ships now.
    checked = verify_pass_digests(entries.values(), output.text or "")
    return tuple(sorted(checked, key=lambda e: (e.check, e.scope)))


REFINALIZED_CLAUSE_REMOVED_REASON = (
    "re-finalized offline: the clause this entry recorded is no longer on the page, and the "
    "check was not run again"
)


REFINALIZED_TEXT_CHANGED_REASON = (
    "re-finalized offline, not run on the new text: the shipped text changed after this check "
    "passed on the old one"
)


def reconcile_pass_trace(
    old_raw: Any,
    verdict: PageVerdict,
    row_binding_records: Any = None,
    recomputed: tuple[TraceEntry, ...] = (),
    text_changed: bool = False,
    text: str | None = None,
) -> list[dict[str, str]] | None:
    """The trace of a sidecar an OFFLINE re-finalize tool rewrote, kept consistent with the
    verdict the tool wrote (#1183).

    The tools change a page's clauses without running the page's checks again, so the trace the
    original run recorded can no longer be replayed as it stands: a clause the tool added must
    be an entry, and an entry that recorded a clause the tool removed no longer states anything
    the page holds. Rules:

    * every clause of *verdict* is an entry, replacing any entry of the same check and scope;
    * an entry that recorded a clause (FAIL or UNKNOWN of a verdict ``Check``) which *verdict*
      no longer carries becomes NOT_RUN, with the reason above: it is not turned into a PASS;
    * *row_binding_records* (the sidecar's ``row_binding`` after the tool wrote it, when the tool
      runs that check): the row-binding entries are the records' own, replacing the old ones;
    * *recomputed* (outcomes of checks the tool itself ran again on the page it rewrote): each
      replaces the old entries of the same check, unless a clause of *verdict* has it;
    * *text_changed* (the tool changed ``winning_output.text``): a kept PASS the tool did not
      recompute no longer states anything about the shipped text, so it becomes NOT_RUN with
      :data:`REFINALIZED_TEXT_CHANGED_REASON`;
    * a clause of *verdict* removes every old entry of its check that is not itself a clause, at
      ANY scope (a PASS at ``table_rows`` never stays beside a FAIL at ``page``);
    * a recomputed FAIL or UNKNOWN is kept as the recomputed entry even when *verdict* carries no
      clause for it: a finding is never dropped, and the check's slot never disappears;
    * *text* (the text the tool ships): every PASS, kept or recomputed, whose digest is not the
      digest of *text* becomes NOT_RUN (:func:`verify_pass_digests`); a recomputed PASS is first
      stamped with *text*. This is the structural rule and replaces *text_changed* wherever the
      tool passes the text it ships;
    * every other entry (a NOT_APPLICABLE, a NOT_RUN, a row-binding record) is kept.

    ``None`` when the old sidecar carries no readable trace: a tool never invents one."""
    from socr.core.pass_trace import trace_from_dicts, trace_to_dicts

    old = trace_from_dicts(old_raw)
    if old is None:
        return None
    entries: dict[tuple[str, str], TraceEntry] = {
        (_check_name(c.check), c.scope): entry_from_clause(c) for c in verdict.clauses
    }
    verdict_checks = {c.value for c in Check}
    records = row_binding_record_entries(row_binding_records) if row_binding_records else []
    for entry in records:
        entries[(entry.check, entry.scope)] = entry
    clause_checks = {_check_name(c.check) for c in verdict.clauses}
    for entry in recomputed:
        if text is not None:
            entry = stamp_text_digest(entry, text)
        # A recomputed FAIL or UNKNOWN the verdict lacks is KEPT: it is what the check found on
        # the text the tool ships, and dropping it would delete the check's slot.
        entries.setdefault((entry.check, entry.scope), entry)
    rerun_checks = {e.check for e in recomputed}
    for entry in old:
        key = (entry.check, entry.scope)
        if key in entries or (records and entry.check == ROW_BINDING_CHECK):
            continue
        is_clause = (
            entry.result in (TraceResult.FAIL, TraceResult.UNKNOWN)
            and entry.check in verdict_checks
        )
        if entry.check in clause_checks and not is_clause:
            # The verdict now states this check: an old pass or non-run of it, at any scope, no
            # longer stands beside the clause.
            continue
        if entry.check in rerun_checks:
            # The tool ran this check again: its outcomes are the check's, at whatever scope it
            # states them. An old entry at another scope (a FAIL clause the repair removed, which
            # sat at page scope while the pass sits at ``table_rows``) is not kept beside them.
            continue
        if (
            entry.result in (TraceResult.FAIL, TraceResult.UNKNOWN)
            and entry.check in verdict_checks
        ):
            entry = TraceEntry(
                entry.check, TraceResult.NOT_RUN, entry.scope, REFINALIZED_CLAUSE_REMOVED_REASON
            )
        elif text_changed and entry.result is TraceResult.PASS:
            entry = TraceEntry(
                entry.check, TraceResult.NOT_RUN, entry.scope, REFINALIZED_TEXT_CHANGED_REASON
            )
        entries[key] = entry
    final: list[TraceEntry] = list(entries.values())
    if text is not None:
        final = verify_pass_digests(final, text)
    return trace_to_dicts(sorted(final, key=lambda e: (e.check, e.scope)))


#: The checks whose outcome is a function of the shipped text and the page's own layer, which an
#: offline tool that holds the page state (the layer inputs) can run again on the text it wrote.
TEXT_DEPENDENT_CHECKS: frozenset[str] = frozenset(
    {
        Check.TEXT_LAYER_NUMBERS.value,
        Check.TEXT_LAYER_DROPS.value,
        Check.TEXT_LAYER_WORDS.value,
        Check.CONTROL_BYTES.value,
        Check.PRIOR_OCR_LAYER.value,
    }
)


def rerun_text_dependent_checks(
    output: PageOutput, p: Any, old_raw: Any = None
) -> tuple[TraceEntry, ...]:
    """Outcomes of :data:`TEXT_DEPENDENT_CHECKS` on *output*, for an offline tool that rebuilt the
    page state *p* the way a fresh run holds it (the *recomputed* argument of
    :func:`reconcile_pass_trace`). The provider state is not on the page: it is read off the old
    trace (a model-dependent entry that says there was no ladder), so a no-provider run keeps its
    NOT_RUN reason and is not rewritten as NOT_APPLICABLE."""
    from socr.core.pass_trace import trace_from_dicts

    old = trace_from_dicts(old_raw) or ()
    providers_available = not any(NO_PROVIDER_LADDER_REASON in e.reason for e in old)
    return tuple(
        o.entry
        for o in pass_trace_outcomes(output, p, providers_available=providers_available)
        if o.entry.check in TEXT_DEPENDENT_CHECKS
    )


def write_reconciled_trace(out: dict, old_raw: Any, verdict: PageVerdict, **kwargs: Any) -> None:
    """Set ``out["checks"]`` to :func:`reconcile_pass_trace` of the sidecar's old trace. When the
    old trace is absent or malformed the key is DROPPED: an unreadable list is never left in a
    sidecar the tool rewrote, and no trace is invented."""
    checks = reconcile_pass_trace(old_raw, verdict, **kwargs)
    if checks is None:
        out.pop("checks", None)
    else:
        out["checks"] = checks


def _clauses_ahead_of_text_layer_numbers(
    output: PageOutput, p: Any, event_kinds: frozenset[str]
) -> tuple[VerdictClause, ...]:
    clauses = (
        _equation_fidelity_clause(event_kinds),
        _chart_values_clause(event_kinds),
        _exact_pass_unjudged_clause(output, event_kinds),
        _prior_ocr_layer_clause(output, p),
        # #1187 (merge with #1185): INSIDE the ahead tuple, not after the numbers clause.
        # This tuple is the single ordering authority -- ``with_text_layer_numbers_clause``
        # slots #1185's clause by the COUNT of what precedes it -- so a lane that derived its
        # clauses anywhere else would make a re-finalized page's marks sit in a different
        # order from a fresh run's. The two lanes are independent and BOTH may mark one page:
        # a figure this page names and does not show, and numbers its text layer disputes,
        # are different facts and neither undoes the other.
        *_unshipped_figure_clauses(p),
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
        entry = _mode_entry(output)
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


#: #1187's two modes. Both are recorded for a figure a page names and does not show, and
#: both are explained by the same ``figure_text`` clause; they differ only in whether the
#: outcome can change on a later run (see ``PERMANENTLY_UNRESOLVABLE_MODES``).
_FIGURE_LANE_MODES: frozenset[FailureMode] = frozenset(
    {FailureMode.FIGURE_NOT_SHOWN, FailureMode.FIGURE_RENDER_FAILED}
)


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
    # #1187: the one demotion whose mode deliberately has no ``_MODE_CLAUSES`` row, because
    # its lane's state survives a resume and so the real clause is re-derived every pass (see
    # the note where the row would be). The status is explained when that clause is present --
    # asked here rather than given a row, because a row would publish a SECOND, page-scoped
    # clause about the same fact. Without this the fail-closed fallback fired on a page whose
    # reason WAS named, which is the hole this helper exists to close, not to open.
    # BOTH of this lane's modes, not just the first (round 5, Argon P1-2b). The two were
    # split apart so a deterministic outcome could be permanent while a transient crop
    # failure is retried; they are explained by the SAME clause, and naming one of them here
    # made the other's page gain the fail-closed ``unknown`` clause on the second pass --
    # measured: pass 1 ``[figure_text, text_layer_numbers]``, pass 2 the same plus
    # ``unknown``, with the page's bytes unchanged. The record and the marks disagreeing
    # about one page is the exact conflation ``build_page_verdict`` exists to prevent.
    if output.failure_mode in _FIGURE_LANE_MODES:
        return any(
            c.check is Check.FIGURE_TEXT and str(c.scope).startswith("figure:") for c in clauses
        )
    # #1185: a mode whose clause wording names the page's own numbers cannot sit in
    # ``_MODE_CLAUSES`` (a fixed reason there would REPLACE the named one in
    # ``_merge_clause``). It is explained when the clause it records is present.
    #
    # Two independent early returns, deliberately: a page can carry BOTH a figure it does not
    # show and numbers its layer disputes, and the mode records only the one that demoted it
    # first. Each asks about its own clause, so neither lane's answer depends on the other's.
    check = LAYER_CHECK_MODES.get(output.failure_mode)
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
