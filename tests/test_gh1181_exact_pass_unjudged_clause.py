"""#1181: a native-table EXACT_PASS ships the whole page unjudged, so it owes a mark.

On the EXACT_PASS accept (``pipeline/agentic.py``) the deterministic value guard has
matched row counts and every paired row's numeric multiset, and the page then ships at
``confidence=1.0`` with NO judge call. Measured on corpus run 23319732: 5 pages of 17,091
ship on this path -- 3 of them SUCCESS, ``audit_passed=true``, ``verdict.clauses=[]`` and no
mark, which is the whole of the acceptance meter's ``table_only_acceptance`` class -- while
captions, panel headings, notes, prose and chart images on the same pages were never looked
at. All 5 are ``engine=qwen``, so the shipped bytes ARE a model reading; what never happened
is a check of it.

**Driven through ``_select_and_finalize_page``**, not ``_apply_verdict_marks``, for the
reason given in the chart slice's file: step 9e can demote and mark a page on its own and a
marks-level probe never sees it.

Hermetic: no provider, no page loop, no model. The input is a constructed ``DocumentState``
and the evidence is an audit event on ``state.events``, where the lane puts it.
"""

from __future__ import annotations

import pathlib

from socr.core.document import DocumentHandle
from socr.core.manifest import _select_and_finalize_page
from socr.core.marks import Check, CheckResult, leading_marks
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.core.verdict import (
    EXACT_PASS_ACCEPT_REASON,
    NATIVE_TABLE_EXACT_PASS_KIND,
)
from verdict_marks_helper import canonical_marks, strip_verdict_marks

TABLE = "| Model | 3-cat | 5-cat |\n| --- | --- | --- |\n| Vicuna | 0.80 | 0.48 |\n"

#: The shape of all three meter pages: a caption, the table, and a notes block.
WITH_PROSE = f"Table 2: Model performance\n\n{TABLE}\nNotes: best of a battery of prompts.\n"

#: The same page with everything outside the table removed. The clause fires on it too
#: (the ruling drops any predicate about the body), which is its own test below.
TABLE_ONLY = TABLE

EXACT_PASS = (NATIVE_TABLE_EXACT_PASS_KIND,)


class _Ev:
    """An audit event with the one field the seam's filter reads, plus a page number."""

    def __init__(self, kind: str, page_num: int) -> None:
        self.kind, self.page_num = kind, page_num
        self.engine = self.detail = ""
        self.data: dict = {}


def _state(tmp_path, *, kinds=(), text: str = WITH_PROSE, **output_kw) -> DocumentState:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1, page_count_known=True))
    ps = state.pages[1]
    base = dict(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        failure_mode=FailureMode.NONE,
        judge_reason=EXACT_PASS_ACCEPT_REASON,
    )
    base.update(output_kw)
    out = PageOutput(**base)
    ps.attempts.append(out)
    ps.best_output = out
    for kind in kinds:
        state.events.append(_Ev(kind, 1))
    return state


def _record(tmp_path, **kw):
    return _select_and_finalize_page(_state(tmp_path, **kw), 1)


# ---------------------------------------------------------------------------
# The difference test: the SAME page, with and without the EXACT_PASS accept
# ---------------------------------------------------------------------------


def test_the_mark_turns_on_the_exact_pass_accept_and_nothing_else(tmp_path) -> None:
    """Two runs of the seam in one process, differing only in how the page was accepted.

    An absolute outcome is not pinned: what is pinned is the DIFFERENCE. The bytes, the
    engine and ``audit_passed`` are identical in both arms, so the only thing that can move
    status, mode and mark is whether the page came through the exit where no judge was
    called -- the lane's event plus the accepting reason that proves it.
    """
    bypassed = _record(tmp_path, kinds=EXACT_PASS)
    judged = _record(tmp_path, kinds=(), judge_reason="faithful")

    assert bypassed.output.status is PageStatus.WARNING
    assert judged.output.status is PageStatus.SUCCESS, (
        "the arms do not differ, so this test cannot tell the clause from one that fires "
        "on every page"
    )
    assert [c.check for c in bypassed.verdict.clauses] == [Check.PAGE_JUDGE]
    assert judged.verdict.clauses == ()
    assert canonical_marks(judged.output.text, 1) == []
    # The marked arm's body is its input body plus the mark block, nothing else.
    assert strip_verdict_marks(bypassed.output.text, 1).strip() == WITH_PROSE.strip()


def test_a_table_only_page_is_marked_too(tmp_path) -> None:
    """The ruling Fable argued for: no predicate about content outside the table.

    An earlier version fired only when the page shipped something outside its table syntax,
    which left a table-only page SUCCESS and unmarked -- and that page is a violation by the
    meter's own rule (``rule_qwen``: bypassed and ``audit_passed`` -> ``table_only_acceptance``),
    so the predicate left exactly the hole the meter exists to catch. The binding of values
    to columns and row labels is unchecked on this page too: the oracle that could refute it
    runs only on the table-ladder path. The mark's wording stays true of it.
    """
    rec = _record(tmp_path, kinds=EXACT_PASS, text=TABLE_ONLY)

    assert rec.output.status is PageStatus.WARNING
    assert [c.check for c in rec.verdict.clauses] == [Check.PAGE_JUDGE]
    marks = canonical_marks(rec.output.text, 1)
    assert len(marks) == 1 and "nothing outside the table" in marks[0], marks


def test_the_clause_says_what_was_compared_and_never_that_the_table_was_verified(
    tmp_path,
) -> None:
    """The wording requirement, and it is load-bearing.

    The gate the mark speaks for compares a per-row numeric MULTISET
    (``tables/native_verifier.py``: ``Counter`` equality), which is blind to which column a
    value sits in and to which label a row carries -- measured on a synthetic page, where a
    within-row column swap and a row-label swap both still reach EXACT_PASS. A mark that
    said "the table was verified" would overstate exactly that.
    """
    rec = _record(tmp_path, kinds=EXACT_PASS)
    marks = canonical_marks(rec.output.text, 1)

    assert len(marks) == 1, marks
    assert f"{Check.PAGE_JUDGE.value}={CheckResult.UNKNOWN.value}" in marks[0], marks[0]
    assert "no judge saw this page" in marks[0]
    assert "not their columns or row labels" in marks[0]
    assert "nothing outside the table" in marks[0]
    assert "verified" not in marks[0], (
        "the mark claims the table was verified, which the multiset check does not show"
    )
    assert rec.output.failure_mode is FailureMode.MODEL_OUTPUT_UNJUDGED, (
        "the demotion does not explain itself, so a later pass over this page cannot tell "
        "it from one demoted by a guard that names nothing"
    )
    assert rec.verdict.clauses[0].result is CheckResult.UNKNOWN, (
        "FAIL would say a check ran and decided against the page; none ran"
    )


# ---------------------------------------------------------------------------
# The negatives. Each is a way the clause could fire on a page it is false of.
# ---------------------------------------------------------------------------


def test_a_page_without_the_event_is_untouched(tmp_path) -> None:
    """Including the verifier's sibling kinds, so "any verifier event" would fail here."""
    for kinds in (
        (),
        ("native_table_verifier_warn",),
        ("native_table_verifier_hard_fail",),
        ("value_guard_row_count_warning",),
    ):
        rec = _record(tmp_path, kinds=kinds)
        assert rec.output.status is PageStatus.SUCCESS, kinds
        assert rec.verdict.clauses == (), kinds
        assert leading_marks(rec.output.text, 1) == [], kinds


def test_a_page_the_gate_delegated_to_the_judge_is_untouched(tmp_path) -> None:
    """Fable's finding on the diagnosis, and the reason the accepting reason is required.

    The event is emitted BEFORE ``_apply_structural_gate``, which on a header-attribution
    abstain consults the inner judge (#245) and returns ITS decision. Such a page carries
    the event and ships model-JUDGED, so a clause keyed on the event alone would tell its
    reader no judge saw a page a judge accepted.
    """
    for reason in ("faithful", "", "native_table_verifier: AMBIGUOUS"):
        rec = _record(tmp_path, kinds=EXACT_PASS, judge_reason=reason)
        assert rec.output.status is PageStatus.SUCCESS, reason
        assert rec.verdict.clauses == (), reason


def test_the_marked_page_is_stable_when_the_seam_runs_again(tmp_path) -> None:
    """Resume / re-finalization: the same mark and the same status, not a second line.

    ``_rewrite_all_fragments`` re-runs this seam over every page at assemble, and a resumed
    page is re-finalized from its restored state. The second pass is given the page as the
    first pass left it -- demoted, mode stamped, mark written -- so the clause must come
    back from ``_MODE_CLAUSES`` and the mark must not be duplicated.
    """
    first = _record(tmp_path, kinds=EXACT_PASS)
    second = _select_and_finalize_page(
        _state(
            tmp_path,
            kinds=EXACT_PASS,
            text=first.output.text,
            status=first.output.status,
            failure_mode=first.output.failure_mode,
        ),
        1,
    )
    assert second.output.text == first.output.text
    assert second.output.status is first.output.status
    assert [c.check for c in second.verdict.clauses] == [Check.PAGE_JUDGE]
    assert canonical_marks(second.output.text, 1) == canonical_marks(first.output.text, 1)


def test_the_clause_survives_a_resume_because_the_event_is_replayed(tmp_path) -> None:
    """The lane's record must outlive the run that recorded it.

    ``_restore_terminal_page_state`` replays only allowlisted kinds, so a kind missing from
    ``resume_restore_kinds`` is in the page sidecar but gone from ``audit_log.json`` the
    moment the page resumes, and the run summary stops counting it.

    It is NOT what keeps a marked page marked, and this docstring said it was until Fable's
    review: a page this clause marks is WARNING, so ``_load_terminal_page`` refuses it and
    the page is reprocessed, which re-emits the event. The pages that DO resume carrying
    this kind are those where the gate emitted EXACT_PASS and then delegated to the page
    judge: the accepting reason is replaced, so the clause does not fire.
    """
    from socr.pipeline.orchestrator import UnifiedPipeline

    assert NATIVE_TABLE_EXACT_PASS_KIND in UnifiedPipeline.resume_restore_kinds()


def test_the_emitter_and_the_clause_name_the_same_event_and_reason(tmp_path) -> None:
    """One name for the kind AND for the accepting reason, so neither can drift.

    Round 2, and the reason half is the reviewer's blocker. The clause requires exact
    equality with ``EXACT_PASS_ACCEPT_REASON``; while the exit wrote its own literal, the
    reviewer changed that literal and the FULL suite passed (7,894), so the clause could be
    switched off in production with CI green and only the meter on the next corpus run
    would have noticed.
    """
    import inspect

    from socr.pipeline import agentic

    src = inspect.getsource(agentic)
    assert src.count("kind=NATIVE_TABLE_EXACT_PASS_KIND") == 1, (
        "the EXACT_PASS emitter no longer uses the shared constant"
    )
    assert src.count("reason=EXACT_PASS_ACCEPT_REASON") == 1, (
        "the EXACT_PASS accepting decision no longer uses the shared constant"
    )
    for literal in (
        '"native_table_verifier_exact_pass"',
        "'native_table_verifier_exact_pass'",
        '"native_table_verifier: EXACT_PASS"',
        "'native_table_verifier: EXACT_PASS'",
    ):
        assert literal not in src, (
            f"{literal} is spelled as a literal again, which can drift from the clause"
        )


# ---------------------------------------------------------------------------
# The acceptance meter, which is what #1181 is measured by
# ---------------------------------------------------------------------------


def test_the_meter_reads_the_fix_as_marked_not_as_a_violation(tmp_path) -> None:
    """``table_only_acceptance`` is gone and nothing else takes its place.

    A difference on the meter's own classifier, not a re-run of the corpus: the same page
    is judged in the shape it shipped BEFORE this fix (SUCCESS, ``audit_passed=true``,
    ``judge_reason`` naming the verifier, no mark) and in the shape the seam now produces.
    ``judge_page`` applies the lane rules only to SUCCESS pages, so the demotion is what
    moves the page out of the violation set -- and the mark is what keeps it out of
    ``warning_unmarked``.
    """
    from socr.benchmark.acceptance import TABLE_ONLY_ACCEPTANCE, Page, judge_page

    rec = _record(tmp_path, kinds=EXACT_PASS)

    def _page(md: str, **sidecar) -> Page:
        base = dict(
            status="success",
            failure_mode="none",
            audit_passed=True,
            engine="qwen",
            text_coverage="passed",
            winning_output=dict(
                status="success",
                failure_mode="none",
                audit_passed=True,
                engine="qwen",
                judge_outcome="page_judge_completed",
                judge_reason=EXACT_PASS_ACCEPT_REASON,
                text=md,
            ),
            audit_events=[{"kind": NATIVE_TABLE_EXACT_PASS_KIND}],
        )
        base.update(sidecar)
        return Page(doc="d", num=1, sidecar=base, md=md, manifest_entry=None)

    before = judge_page(_page(WITH_PROSE))
    assert TABLE_ONLY_ACCEPTANCE in before.violations, (
        "the meter does not flag the pre-fix shape, so this test cannot show the fix moved anything"
    )

    # The seam's OWN status and mode, not hand-written ones: a seam that marked without
    # demoting would otherwise be invisible here, while the docstring claims the demotion
    # is what moves the page (Fable, review of this diff).
    after = judge_page(
        _page(
            rec.output.text,
            status=rec.output.status.value,
            failure_mode=rec.output.failure_mode.value,
        )
    )
    assert after.marked and not after.violations, after.violations


# ---------------------------------------------------------------------------
# A hole Fable found in the first version of this fix
# ---------------------------------------------------------------------------


def test_an_unjudged_page_is_never_admitted_to_the_resume_skip_exception(tmp_path) -> None:
    """A page carrying BOTH an equation clause and this one must still be reprocessed.

    ``_resume_verdict_is_permanently_unverifiable`` decides on the recorded MODE, on the
    argument that a page carrying an unresolvable mode carries no other kind of clause.
    #1181 breaks that argument: a results page with a table and a display equation under
    ``--detect-equations`` carries both, ``_merge_clause`` records the EQUATION mode (which
    is permanent), and the page would skip-and-keep for ever with its table still unjudged.

    The difference, not the absolute: the equation page alone must stay admissible, so the
    test fails if the fix simply closed the exception for everybody.
    """
    from socr.pipeline.orchestrator import _resume_verdict_is_permanently_unverifiable

    equation_only = _record(
        tmp_path, kinds=("equation_region_reading_attached",), judge_reason="faithful"
    )
    both = _record(
        tmp_path, kinds=("equation_region_reading_attached", NATIVE_TABLE_EXACT_PASS_KIND)
    )

    assert _resume_verdict_is_permanently_unverifiable(
        equation_only.verdict.to_dict(), equation_only.output.failure_mode.value
    ), "the equation lane's own exception is gone, so this fix closed more than its own hole"
    assert not _resume_verdict_is_permanently_unverifiable(
        both.verdict.to_dict(), both.output.failure_mode.value
    ), "an unjudged table page is skip-and-kept for ever once an equation shares the page"


# ---------------------------------------------------------------------------
# End to end: the REAL exit's own decision, through the seam
# ---------------------------------------------------------------------------


def test_the_real_exact_pass_exit_produces_a_reason_the_clause_marks(tmp_path) -> None:
    """``NativeTableVerifierJudge.assess`` → its OWN decision → the seam → a marked page.

    Round 2, the reviewer's blocker. Every other test in this file hands the seam a
    ``judge_reason`` built from the constant, so none of them crosses the boundary where
    the string is actually written. This one takes ``decision.reason`` from the real exit
    and feeds exactly that to ``_select_and_finalize_page``: if the exit's spelling and the
    clause's constant ever part company, the page comes back unmarked here.

    Hermetic: a PyMuPDF page built in memory, no provider, no model. The wrapped judge
    raises if it is ever consulted, which is also what proves the page took the exit where
    no judge is called.
    """
    import fitz

    from socr.pipeline.agentic import NativeTableVerifierJudge

    class _NeverCalled:
        def assess(self, output, provider):  # pragma: no cover - must not run
            raise AssertionError("the inner judge was consulted on the EXACT_PASS exit")

    doc = fitz.open()
    page = doc.new_page()
    rows = (("Alpha", "1.11", "2.22"), ("Beta", "3.33", "4.44"), ("Gamma", "5.55", "6.66"))
    page.insert_text((72, 80), "Label", fontsize=10)
    page.insert_text((200, 80), "X", fontsize=10)
    page.insert_text((300, 80), "Y", fontsize=10)
    for index, (label, first, second) in enumerate(rows):
        y = 100 + 20 * index
        page.insert_text((72, y), label, fontsize=10)
        page.insert_text((200, y), first, fontsize=10)
        page.insert_text((300, y), second, fontsize=10)

    markdown = "| Label | X | Y |\n| --- | --- | --- |\n" + "".join(
        f"| {label} | {first} | {second} |\n" for label, first, second in rows
    )
    candidate = PageOutput(
        page_num=1,
        text=markdown,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        failure_mode=FailureMode.NONE,
    )
    events: list = []
    judge = NativeTableVerifierJudge(
        inner=_NeverCalled(),
        get_fitz_page=lambda _n: page,
        is_table_page=lambda _n: True,
        record_event=events.append,
    )

    decision = judge.assess(candidate, None)

    assert decision.accept and decision.confidence == 1.0, decision
    assert NATIVE_TABLE_EXACT_PASS_KIND in {getattr(e, "kind", "") for e in events}, events

    rec = _select_and_finalize_page(
        _state(tmp_path, kinds=EXACT_PASS, text=markdown, judge_reason=decision.reason), 1
    )

    assert [c.check for c in rec.verdict.clauses] == [Check.PAGE_JUDGE], (
        f"the exit's own reason {decision.reason!r} is not the one the clause requires"
    )
    assert rec.output.status is PageStatus.WARNING
    assert canonical_marks(rec.output.text, 1), rec.output.text


def test_the_meter_recognises_the_reason_without_sharing_the_constant(tmp_path) -> None:
    """The meter stays the INDEPENDENT instrument, and this pins the relationship.

    Round 2. The reviewer suggested deriving ``TABLE_ONLY_JUDGE_REASON_PREFIX`` from
    ``EXACT_PASS_ACCEPT_REASON``; the ruling is not to. An instrument that shares the
    product's constant cannot catch the product's drift, which is the whole reason the
    meter exists beside the tests. So the two stay separately spelled, and what is pinned
    is the only thing that must hold between them: whatever the product calls this accept,
    the meter must still recognise it.
    """
    from socr.benchmark import acceptance

    assert EXACT_PASS_ACCEPT_REASON.startswith(acceptance.TABLE_ONLY_JUDGE_REASON_PREFIX)
    assert "EXACT_PASS_ACCEPT_REASON" not in pathlib.Path(acceptance.__file__).read_text(), (
        "the meter now imports the product constant, so it can no longer catch its drift"
    )
