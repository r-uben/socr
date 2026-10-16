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
    """The reachability argument the resume gate rests on, pinned by reading the source.

    The gate decides on the MODE alone. An earlier version also required the clauses to say
    nothing more than the mode does, and that check was UNREACHABLE -- a mutant deleting it
    survived every test, which is how it was found. The argument for deleting it rather than
    inventing a fixture for it: these two modes are written in exactly ONE place, reached
    only by the demotion, which only fires on a page still SUCCESS at the verdict step -- and
    a SUCCESS page's clause set is the silent-success clauses and nothing else. Every other
    clause source (a flag, the judge, the ladder) arrives with its own demotion and leaves
    the mode as it found it, so a page WARNING for another reason reaches the gate with that
    reason's mode or with NONE, and is refused.

    That argument is only true while nothing else writes these modes. This test is what
    notices if something starts -- the event that would make the deleted check necessary
    again. Read from the source text rather than by behaviour, because the point is the
    number of WRITERS, which no single page can show.
    """
    import re
    from pathlib import Path as _Path

    import socr

    root = _Path(socr.__file__).parent
    writers: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for num, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not re.search(
                r"(EQUATION_FIDELITY_UNVERIFIED|VISUAL_VALUES_UNTRANSCRIBED)\b", stripped
            ):
                continue
            # The enum's own definition, the clause table and the admissible set are
            # READERS of the name, not writers of the field.
            if re.match(r"(EQUATION_FIDELITY_UNVERIFIED|VISUAL_VALUES_UNTRANSCRIBED) =", stripped):
                continue
            if stripped.startswith("FailureMode.") and stripped.endswith(","):
                continue
            if stripped.startswith("FailureMode.") and stripped.endswith("("):
                continue
            # Which function is this line in? That is the unit that matters: one table
            # with a row per class is one writer, and a second FUNCTION setting the field
            # is what breaks the argument.
            enclosing = ""
            for prev in reversed(path.read_text().splitlines()[:num]):
                if prev.startswith("def ") or prev.startswith("    def "):
                    enclosing = prev.strip().split("(")[0].removeprefix("def ").strip()
                    break
            writers.append(f"{path.relative_to(root)}:{num} in {enclosing}(): {stripped}")

    functions = {w.split(" in ")[1].split("()")[0] for w in writers}
    assert functions == {"_verdict_demotion_mode"}, (
        "the resume gate decides on the mode alone, which is sound only while exactly one "
        "FUNCTION sets these modes. Writers found:\n  " + "\n  ".join(writers)
    )
    assert all("manifest.py" in w for w in writers), writers
