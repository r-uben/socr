"""#1145 slice 3 (chart): a chart shipped as an image owes the reader that sentence.

210 pages of run 23192118 ship SUCCESS, `failure_mode=none`, `audit_passed=True` and NO
mark, having routed a chart to an image asset. That routing is the lane WORKING -- a chart
is not prose and socr does not pretend to read it -- but the numbers a reader would cite are
then present only as pixels, and the page said nothing about it. `figure_text=UNKNOWN`, plus
a demotion.

**Driven through `_select_and_finalize_page`, not `_apply_verdict_marks`** (Fable's method
note on PR #1171). The seam has eleven steps and the verdict is the last; step 9e
(`_apply_unaccepted_output_mark`) runs before it and can demote and mark a page on its own.
A probe at the marks level alone would have called the `audit_passed=False` shape admissible
to the resume exception, because it never sees what 9e did. Only the full chain closes that.

Hermetic: no provider, no page loop. The input is a constructed `DocumentState`, and the
clause's evidence is an audit event on `state.events`, which is where the lanes put it.
"""

from __future__ import annotations

import pytest

from socr.core.document import DocumentHandle
from socr.core.manifest import _select_and_finalize_page
from socr.core.marks import Check, CheckResult, leading_marks
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.core.verdict import VISUAL_VALUES_UNTRANSCRIBED_KIND
from socr.pipeline.orchestrator import _resume_verdict_is_permanently_unverifiable
from verdict_marks_helper import assert_differs_only_by_marks, canonical_marks

BODY = "Figure 3 shows the response.\n\n![chart](figures/chart_p1.png)\n"


class _Ev:
    """An audit event with the one field the seam's filter reads, plus a page number."""

    def __init__(self, kind: str, page_num: int) -> None:
        self.kind, self.page_num = kind, page_num
        self.engine = self.detail = ""
        self.data: dict = {}


def _state(tmp_path, *, kinds=(), text: str = BODY, **output_kw) -> DocumentState:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1, page_count_known=True))
    ps = state.pages[1]
    base = dict(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="chart_asset",
        audit_passed=True,
        failure_mode=FailureMode.NONE,
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


CHART = (VISUAL_VALUES_UNTRANSCRIBED_KIND,)


# ---------------------------------------------------------------------------
# The clause, and the negative
# ---------------------------------------------------------------------------


def test_an_untranscribed_chart_demotes_the_page_and_says_why(tmp_path) -> None:
    """Demoted, marked, and the mark names the right check with a usable instruction.

    UNKNOWN, not FAIL: nothing read the chart and found it wrong; nothing read it at all.
    """
    rec = _record(tmp_path, kinds=CHART)

    assert rec.output.status is PageStatus.WARNING, (
        "the page still ships SUCCESS, so the library, the table counts and the CLI all "
        "still call it clean"
    )
    assert [c.check for c in rec.verdict.clauses] == [Check.FIGURE_TEXT], rec.verdict
    assert rec.verdict.clauses[0].result is CheckResult.UNKNOWN
    assert rec.output.failure_mode is FailureMode.VISUAL_VALUES_UNTRANSCRIBED, (
        "the demotion does not explain itself, so a later pass cannot tell it from a page "
        "demoted by a guard that names nothing"
    )
    marks = canonical_marks(rec.output.text, 1)
    assert len(marks) == 1 and f"{Check.FIGURE_TEXT.value}=UNKNOWN" in marks[0], marks
    assert "read them off the image" in marks[0], (
        "the mark does not tell the reader what to do instead"
    )


def test_a_page_without_the_event_is_untouched(tmp_path) -> None:
    """The negative half, and it is the test.

    Without it the clause could fire on every page the chart lane routed -- or on every
    page at all -- and the slice would look like it worked. Includes a sibling kind the
    lane also emits, so "any chart-ish event" would fail here.
    """
    for kinds in ((), ("chart_asset_page",), ("chart_counts_derived",)):
        rec = _record(tmp_path, kinds=kinds)
        assert rec.verdict.checked, (kinds, rec.verdict)
        assert rec.output.status is PageStatus.SUCCESS, kinds
        assert rec.output.text == BODY, kinds


def test_the_demoted_page_differs_only_by_the_mark(tmp_path) -> None:
    """The image reference and the prose around it are DISCLOSED, never rewritten."""
    marked = _record(tmp_path, kinds=CHART).output.text
    clean = _record(tmp_path, kinds=()).output.text

    assert_differs_only_by_marks(marked, clean, 1)
    assert "![chart](figures/chart_p1.png)" in marked, "the chart image ref was lost"


def test_audit_passed_is_not_touched(tmp_path) -> None:
    """``audit_passed`` is winner-SELECTION input, not a spare "is this page fine" flag.

    Flipping it to flag a page makes assemble discard that page's text.
    """
    rec = _record(tmp_path, kinds=CHART)
    assert rec.output.audit_passed is True


def test_an_existing_more_specific_mode_is_never_overwritten(tmp_path) -> None:
    """A page already carrying a mode knows something this clause does not."""
    rec = _record(tmp_path, kinds=CHART, failure_mode=FailureMode.FIGURE_WORDS_UNREAD)
    assert rec.output.status is PageStatus.WARNING
    assert rec.output.failure_mode is FailureMode.FIGURE_WORDS_UNREAD, rec.output.failure_mode


@pytest.mark.parametrize("kinds", [CHART, ()])
def test_the_seam_is_idempotent_over_three_passes(tmp_path, kinds) -> None:
    """``_rewrite_all_fragments`` re-runs this seam over every page at assemble.

    Three passes, because two can hide a cycle of period two. Both arms, because a
    demotion that re-fires would walk the status down again.
    """
    state = _state(tmp_path, kinds=kinds)
    seen = []
    for _ in range(3):
        rec = _select_and_finalize_page(state, 1)
        state.pages[1].best_output = rec.output
        seen.append((rec.output.text, rec.output.status, rec.verdict))

    assert seen[0] == seen[1] == seen[2], seen


def test_the_mark_is_the_first_rendered_block(tmp_path) -> None:
    """A disclosure the reader scrolls past is not a disclosure."""
    rec = _record(tmp_path, kinds=CHART)
    assert leading_marks(rec.output.text, 1) == canonical_marks(rec.output.text, 1)


# ---------------------------------------------------------------------------
# Resume admissibility -- the reason this file drives the whole chain
# ---------------------------------------------------------------------------


def test_the_chart_demotion_earns_the_resume_exception(tmp_path) -> None:
    """Nothing can transcribe the chart by trying again, so the page must not re-OCR.

    Without the exception the 210 pages are WARNING, the ledger gate requires SUCCESS, and
    every resume reprocesses them -- and the next run demotes them again, so the work never
    converges. The same non-convergence #713's exception exists to prevent.
    """
    rec = _record(tmp_path, kinds=CHART)
    assert (
        _resume_verdict_is_permanently_unverifiable(
            rec.verdict.to_dict(), rec.output.failure_mode.value
        )
        is True
    )


def test_figure_text_unknown_from_another_source_does_NOT_earn_it(tmp_path) -> None:
    """The case that forced admissibility onto the MODE rather than the clause pair.

    ``figure_text=UNKNOWN`` has three sources: this lane, the ``FIGURE_WORDS_UNREAD`` mode,
    and the scanned-figure flag. Only this one is unresolvable -- a reader that could not
    transcribe a figure crop today may succeed next run, and skipping those for ever would
    keep a page nobody ever read. Keying on ``(figure_text, UNKNOWN)`` admitted all three.
    """
    # WARNING, which is the real shape of such a page: the mode comes WITH a demotion, and
    # a mode clause is only derived for a page that is not SUCCESS.
    rec = _record(
        tmp_path,
        status=PageStatus.WARNING,
        failure_mode=FailureMode.FIGURE_WORDS_UNREAD,
    )
    assert [c.check for c in rec.verdict.clauses] == [Check.FIGURE_TEXT], rec.verdict
    assert rec.verdict.clauses[0].result is CheckResult.UNKNOWN
    assert (
        _resume_verdict_is_permanently_unverifiable(
            rec.verdict.to_dict(), rec.output.failure_mode.value
        )
        is False
    ), "an unread figure crop earns a permanent skip, so nothing will ever read it"


def test_a_judge_refused_chart_page_is_not_resume_admissible(tmp_path) -> None:
    """Fable's method note, as a test: step 9e runs BEFORE the verdict.

    ``_apply_unaccepted_output_mark`` (#1129) demotes and marks a page with judge refusal
    evidence, and it runs before the verdict step. A probe at the marks level alone never
    sees that, so it would report this page as carrying only the chart clause and call it
    admissible -- a page the judge refused, restored for ever. Through the whole chain the
    judge clause is there too, and the extra clause makes it inadmissible.
    """
    from socr.core.result import REJECTION_JUDGE_ONLY

    rec = _record(
        tmp_path,
        kinds=CHART,
        audit_passed=False,
        judge_accepted=False,
        rejection_class=REJECTION_JUDGE_ONLY,
    )
    checks = {c.check for c in rec.verdict.clauses}
    assert Check.FIGURE_TEXT in checks, rec.verdict
    assert Check.PAGE_JUDGE in checks, (
        "the judge refusal is not in the verdict, so this test is not exercising the shape "
        "the method note is about"
    )
    assert (
        _resume_verdict_is_permanently_unverifiable(
            rec.verdict.to_dict(), rec.output.failure_mode.value
        )
        is False
    ), "a judge-refused page earns a permanent skip"


def test_the_clause_keys_on_an_event_that_survives_a_resume() -> None:
    """Checked, not assumed -- and the sibling that does NOT survive is why.

    ``chart_asset_page`` is absent from the replay allowlist, so a predicate requiring it
    would lose the clause on every resumed page and the next run would silently un-demote
    it. The clause keys on ``visual_values_not_transcribed`` alone, which is replayed, and
    which on run 23192118 identifies exactly the same 1,353 pages.
    """
    from socr.pipeline.orchestrator import UnifiedPipeline

    replayed = UnifiedPipeline.resume_restore_kinds()
    assert VISUAL_VALUES_UNTRANSCRIBED_KIND in replayed
    assert "chart_asset_page" not in replayed, (
        "chart_asset_page is replayed now; the comment explaining why the clause does not "
        "key on it is stale and the predicate could be tightened"
    )


def test_an_admissible_mode_is_only_ever_set_by_the_verdict_demotion() -> None:
    """The reachability argument the resume gate rests on, read from the parse tree.

    The gate decides on the MODE alone, and the extra-clause half of that check was deleted
    as unreachable: these modes are written in exactly ONE place, reached only by the
    demotion, which only fires on a page still SUCCESS at the verdict step -- and a SUCCESS
    page's clause set is the silent-success clauses alone. That argument is only true while
    nothing else writes them, and this test is what notices if something starts.

    **By AST, not by scanning text.** The first version matched source LINES, and Fable's M7
    walked straight past it: a writer spread over several lines, or passed as a positional
    argument, has no line that looks like an assignment. So every reference to these two
    enum members is located as a node, its enclosing function recorded, and the set of
    enclosing functions compared -- which is the unit that matters, since one table with a
    row per class is one writer and a second FUNCTION is what breaks the argument.
    """
    import ast
    from pathlib import Path as _Path

    import socr

    NAMES = {"EQUATION_FIDELITY_UNVERIFIED", "VISUAL_VALUES_UNTRANSCRIBED"}
    root = _Path(socr.__file__).parent
    functions: dict[str, set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        # Map every node to its enclosing function by walking the function bodies.
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Attribute) or inner.attr not in NAMES:
                    continue
                value = inner.value
                if not (isinstance(value, ast.Name) and value.id == "FailureMode"):
                    continue
                functions.setdefault(node.name, set()).add(f"{path.relative_to(root)}")

    # ``_MODE_CLAUSES`` and ``PERMANENTLY_UNRESOLVABLE_MODES`` are module-level tables, so
    # they are not inside any function and never appear here: they are READERS of the name.
    assert set(functions) == {"_verdict_demotion_mode"}, (
        "the resume gate decides on the mode alone, which is sound only while exactly one "
        f"FUNCTION sets these modes. Functions referencing them: {functions}"
    )
    assert functions["_verdict_demotion_mode"] == {"core/manifest.py"}, functions


def test_the_mark_never_contradicts_the_recorded_mode(tmp_path) -> None:
    """Fable's P1, reproduced on monacelli p21: 388 pages said one thing and recorded another.

    #1053's ``figure_words_unread`` and this clause both produce ``figure_text=UNKNOWN``, so
    they collapse to one line. The old rule was "the first wins, and the order is from most
    specific to least" -- true while the status-dependent clauses came first, and silently
    INVERTED when slice 3 moved the silent-success clauses to the front. The chart clause
    then REPLACED #1053's reason: the page's line said "this page's chart ships as an image"
    while its recorded mode still said ``figure_words_unread``.

    A mark that contradicts the record is worse than either alone -- the reader and the
    corpus disagree about the same page, and nothing says which is right. So the MODE
    arbitrates: it is the field that explains the status, and the clause whose wording it
    vouches for is the one that shows.
    """
    rec = _record(
        tmp_path,
        kinds=CHART,
        status=PageStatus.WARNING,
        failure_mode=FailureMode.FIGURE_WORDS_UNREAD,
    )

    assert [c.check for c in rec.verdict.clauses] == [Check.FIGURE_TEXT], rec.verdict
    assert rec.output.failure_mode is FailureMode.FIGURE_WORDS_UNREAD, "the mode moved"
    reason = rec.verdict.clauses[0].reason
    assert "no reader transcribed" in reason, (
        f"the line states the chart wording while the record says figure_words_unread: {reason!r}"
    )
    assert "ships as an image" not in reason, reason
    # And the line on the page says the same thing as the clause.
    (mark,) = canonical_marks(rec.output.text, 1)
    assert "no reader transcribed" in mark, mark

    # The control, so the arbitration is not simply "always prefer the mode table": with no
    # mode to arbitrate, the chart clause is what the page has to say.
    plain = _record(tmp_path, kinds=CHART)
    assert "ships as an image" in plain.verdict.clauses[0].reason


def test_a_success_page_with_an_unmapped_mode_still_gets_the_chart_clause(tmp_path) -> None:
    """The #1173 shape: SUCCESS, a non-NONE mode nothing maps, plus the chart event.

    Such a page is still not CHECKED, and the clause is still the only thing that says so.
    The demotion must not silently skip it because the mode slot is already occupied by
    something the clause table does not recognise -- and the existing mode must survive,
    because it says something this clause does not.
    """
    rec = _record(tmp_path, kinds=CHART, failure_mode=FailureMode.TABLE_UNVERIFIED)

    assert rec.output.status is PageStatus.WARNING, "the page was left SUCCESS and unmarked"
    assert Check.FIGURE_TEXT in {c.check for c in rec.verdict.clauses}, rec.verdict
    assert rec.output.failure_mode is FailureMode.TABLE_UNVERIFIED, (
        "the demotion overwrote a mode that says more than it does"
    )
    assert canonical_marks(rec.output.text, 1)


def test_a_dual_silent_success_page_records_one_mode_and_says_both(tmp_path) -> None:
    """Repo soft 2: a page with BOTH an attached equation reading and an untranscribed chart.

    The mode is a single field, so it can only name one of them -- the first row of
    ``_verdict_demotion_mode``'s table, which is the equation one. That is documented rather
    than fixed: inventing a combined mode would multiply with every future class, and the
    VERDICT is the record that carries both, which is what a reader and the corpus read.

    What must hold, and is asserted here: both clauses survive, both lines are written, and
    the mode is one of the two rather than NONE -- a page demoted for two reasons must not
    come out unexplained.
    """
    rec = _record(
        tmp_path,
        kinds=(VISUAL_VALUES_UNTRANSCRIBED_KIND, "equation_region_reading_attached"),
    )

    checks = {c.check for c in rec.verdict.clauses}
    assert checks == {Check.FIGURE_TEXT, Check.EQUATION_FIDELITY}, rec.verdict
    assert len(canonical_marks(rec.output.text, 1)) == 2, canonical_marks(rec.output.text, 1)
    assert rec.output.failure_mode in {
        FailureMode.EQUATION_FIDELITY_UNVERIFIED,
        FailureMode.VISUAL_VALUES_UNTRANSCRIBED,
    }, rec.output.failure_mode
    # Documented, not accidental: the equation row comes first in the table.
    assert rec.output.failure_mode is FailureMode.EQUATION_FIDELITY_UNVERIFIED
