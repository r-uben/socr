"""#1145 slice 3 (equation): an attached equation reading is not a verified one.

Design §4 and §7, test T8's equation half. 399 pages of run 23192118 ship SUCCESS,
`failure_mode=none`, `audit_passed=True` and NO mark, having spliced a model's LaTeX reading
of an equation crop into the page. The lane validates LaTeX SYNTAX only -- a candidate that
parses is not a candidate that says what the crop says -- so the mathematics on those pages
is unverified by construction, and nothing told the reader.

This is the first clause in the ticket that DEMOTES: a page whose check never ran is not a
page socr confirmed, and the status is what every other consumer reads (the library's
`unverified` curation, `table_counts`' `verified_text`, the review HTML, the CLI counters).
Marking without demoting would leave all of them calling the page clean.

Hermetic: no provider, no page loop. The clause's input is an audit-event KIND, so these
drive the seam directly with the kinds set, which is what production passes it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from socr.core.manifest import _apply_verdict_marks, is_page_failed_marker
from socr.core.marks import SCOPE_PAGE, Check, CheckResult, leading_marks
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.verdict import (
    EQUATION_READING_ATTACHED_KIND,
    PageVerdict,
    VerdictClause,
)
from verdict_marks_helper import assert_differs_only_by_marks, canonical_marks

ATTACHED = frozenset({EQUATION_READING_ATTACHED_KIND})
#: A sibling kind the lane also emits, used as the negative: the clause must key on the
#: ATTACH, not on the lane having run at all.
DETECTED_ONLY = frozenset({"equation_region_detected"})

BODY = "Prose above.\n\n$$y = \\alpha + \\beta x$$\n\nProse below.\n"


class _Flags:
    """A PageState stand-in with no flag set, so only the event kinds can drive a clause."""

    def __getattr__(self, name: str):
        raise AttributeError(name)


def _success(text: str = BODY, **kw) -> PageOutput:
    base = PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="native+equations",
        audit_passed=True,
        failure_mode=FailureMode.NONE,
    )
    return replace(base, **kw) if kw else base


# ---------------------------------------------------------------------------
# The clause, and the negative that makes it mean something
# ---------------------------------------------------------------------------


def test_an_attached_equation_reading_demotes_the_page_and_says_why() -> None:
    """The three halves this slice is: demoted, marked, and the mark names the right check.

    ``UNKNOWN``, not ``FAIL``: the lane does not check semantics, so no check decided
    against the page. ``semantic_fidelity_verified`` is a literal ``False`` in two event
    payloads -- never computed, never varying -- recording that the validation scope is
    ``latex_syntax_only``. FAIL would assert a verdict nobody reached.
    """
    out, verdict = _apply_verdict_marks(_success(), _Flags(), 1, event_kinds=ATTACHED)

    assert out.status is PageStatus.WARNING, (
        "the page still ships SUCCESS, so the library, the table counts and the CLI all "
        "still call it clean"
    )
    assert [c.check for c in verdict.clauses] == [Check.EQUATION_FIDELITY], verdict
    assert verdict.clauses[0].result is CheckResult.UNKNOWN, (
        "FAIL asserts a check decided against this page; the lane never checked"
    )
    marks = canonical_marks(out.text, 1)
    assert len(marks) == 1 and f"{Check.EQUATION_FIDELITY.value}=UNKNOWN" in marks[0], marks
    assert "never verified" in marks[0], marks[0]


def test_a_page_the_lane_merely_looked_at_is_untouched() -> None:
    """The negative half, and it is the test.

    Without it the clause could fire on every page the equation lane ran on -- 399 pages
    become thousands, every one of them demoted, and the slice would look like it worked.
    The clause keys on the ATTACH, which is the lane saying it spliced a reading in; a
    detection with no attached reading spliced nothing and has nothing unverified in the
    shipped bytes.
    """
    for kinds in (frozenset(), DETECTED_ONLY, frozenset({"equation_region_reading_rejected"})):
        out, verdict = _apply_verdict_marks(_success(), _Flags(), 1, event_kinds=kinds)
        assert verdict.checked, (kinds, verdict)
        assert out.status is PageStatus.SUCCESS, kinds
        assert out.text == BODY, kinds


def test_the_demoted_page_differs_from_the_clean_one_only_by_the_mark() -> None:
    """The body is DISCLOSED, never rewritten -- the invariant the whole ticket rests on.

    Same output, same flags; only the event kinds differ. The reading that was spliced in
    stays exactly where the lane put it.
    """
    marked, _ = _apply_verdict_marks(_success(), _Flags(), 1, event_kinds=ATTACHED)
    clean, _ = _apply_verdict_marks(_success(), _Flags(), 1, event_kinds=DETECTED_ONLY)

    assert_differs_only_by_marks(marked.text, clean.text, 1)
    assert "$$y = \\alpha + \\beta x$$" in marked.text, "the equation itself was lost"


def test_audit_passed_is_not_touched_by_the_demotion() -> None:
    """``audit_passed`` is winner-SELECTION input, not a spare "is this page fine" flag.

    #1129 reads it as part of refusal evidence and ``_warn_audit_failed_success_status``
    keys on it; flipping it to flag a page makes assemble discard that page's text. The
    status is the demotion surface and the clause is the reason.
    """
    out, _ = _apply_verdict_marks(_success(), _Flags(), 1, event_kinds=ATTACHED)
    assert out.audit_passed is True, (
        "the demotion flipped audit_passed; assemble may now drop this page"
    )
    assert out.failure_mode is FailureMode.NONE, (
        "a FailureMode was invented; the clause already says what is unresolved"
    )


def test_a_page_already_warning_gets_BOTH_reasons_and_is_not_demoted_twice() -> None:
    """The 4 corpus pages that carry the attach AND already ship WARNING.

    This expectation was the opposite in the first draft -- the clause was SUCCESS-only, so
    these pages kept only their original reason. That was wrong for two reasons, and the
    seam's own idempotence test is what exposed it. Mechanically: the demotion makes a page
    WARNING, so on the NEXT pass (and ``_rewrite_all_fragments`` re-runs this seam over
    every page at assemble) a SUCCESS-only clause is not re-derived, nothing names a check,
    and the fail-closed fallback replaces ``equation_fidelity=UNKNOWN`` with
    ``unknown=UNKNOWN`` -- the page's line and its sidecar verdict disagreeing about the
    same page. And on the merits: the fact is true whatever the status says, and design §1
    asks for exactly this ("the reader is owed both reasons, not the first one").

    So: both clauses, both lines, one per fact -- and the status does NOT walk down again.
    """
    warned = _success(status=PageStatus.WARNING, failure_mode=FailureMode.NATIVE_GARBLED_MATH)
    out, verdict = _apply_verdict_marks(warned, _Flags(), 1, event_kinds=ATTACHED)

    assert out.status is PageStatus.WARNING, "the demotion fired on an already-WARNING page"
    checks = [c.check for c in verdict.clauses]
    assert set(checks) == {Check.EQUATION_FIDELITY, Check.NATIVE_GLYPHS}, verdict
    assert len(checks) == len(set(checks)), f"a check is stated twice: {checks}"

    marks = canonical_marks(out.text, 1)
    assert len(marks) == 2, marks
    assert Check.UNKNOWN_CHECK.value + "=" not in "".join(marks), (
        "the fail-closed fallback fired on a page whose reasons are both named"
    )


def test_a_failed_marker_page_is_never_prepended_to(tmp_path=None) -> None:
    """The guard that keeps biting: a line above ``[page N failed: …]`` un-marks the page.

    A SUCCESS page whose body is a marker should not exist, but the clause now fires on
    SUCCESS pages and this guard is the one that has already been broken twice in this
    ticket, so it is pinned for the new path too.
    """
    marker = "[page 1 failed: unverifiable table — see image]"
    out, verdict = _apply_verdict_marks(_success(text=marker), _Flags(), 1, event_kinds=ATTACHED)
    assert is_page_failed_marker(out.text), "the seam un-marked a failure-marker page"
    assert out.text == marker
    assert not verdict.checked, "the verdict forgot the clause while withholding its line"


def test_the_published_verdict_still_wins_for_this_clause() -> None:
    """Resume rule (b) covers the new clause too.

    The clause's input is an audit-event kind. Those are replayed on resume from an explicit
    allowlist (``EQUATION_LANE_EVENT_KINDS``), so this one does survive -- but the rule must
    not depend on that, because a kind dropped from that allowlist would silently un-demote
    every one of these pages on the next run.
    """
    published = PageVerdict(
        page_num=1,
        clauses=(
            VerdictClause(
                Check.EQUATION_FIDELITY, CheckResult.UNKNOWN, SCOPE_PAGE, "never verified"
            ),
        ),
    )
    # The input is GONE, as it would be if the kind stopped being replayed.
    out, verdict = _apply_verdict_marks(
        _success(), _Flags(), 1, published=published, event_kinds=frozenset()
    )
    assert verdict == published, verdict
    assert out.status is PageStatus.WARNING, (
        "a resumed page came back SUCCESS while carrying an unresolved-check verdict"
    )
    assert canonical_marks(out.text, 1), "the published clause wrote no line"


@pytest.mark.parametrize("kinds", [ATTACHED, DETECTED_ONLY])
def test_the_seam_is_idempotent_over_three_passes(kinds) -> None:
    """Every finalization seam re-runs on its own output, and assemble re-enters it.

    Three passes, because two can hide a cycle of period two. Both arms, because a demotion
    that re-fires would walk the status down on every pass.
    """
    once, v1 = _apply_verdict_marks(_success(), _Flags(), 1, event_kinds=kinds)
    twice, v2 = _apply_verdict_marks(once, _Flags(), 1, event_kinds=kinds)
    thrice, v3 = _apply_verdict_marks(twice, _Flags(), 1, event_kinds=kinds)

    assert once.text == twice.text == thrice.text
    assert once.status is twice.status is thrice.status
    assert v1 == v2 == v3


def test_the_mark_is_the_first_rendered_block() -> None:
    """A disclosure the reader scrolls past is not a disclosure."""
    out, _ = _apply_verdict_marks(_success(), _Flags(), 1, event_kinds=ATTACHED)
    assert leading_marks(out.text, 1) == canonical_marks(out.text, 1)
    assert out.text.startswith("> [page 1 unchecked: equation_fidelity=UNKNOWN;")


# ---------------------------------------------------------------------------
# The resume exception, and why its narrowness is the whole safety argument
# ---------------------------------------------------------------------------


def test_only_a_permanently_unverifiable_clause_earns_the_resume_exception() -> None:
    """The set must stay small, and nothing else pinned that.

    Found by a SURVIVING mutant: widening ``PERMANENTLY_UNVERIFIABLE_CHECKS`` to every
    ``Check`` left every test green, which would have turned a narrow content exception into
    a general "WARNING pages resume" rule. The distinction it rests on:

    * ``equation_fidelity`` here can NEVER be resolved by reprocessing -- the lane validates
      LaTeX syntax only and ``semantic_fidelity_verified`` is a hardcoded ``False`` -- so a
      re-OCR produces the same unverified reading and the same clause, and refusing the skip
      re-OCRs the page on every resume for ever without converging;
    * a ``page_judge`` timeout is the opposite: the judge may well answer next time, and
      skipping such a page keeps text no judge has accepted. That is an infra doubt, and
      doubt must re-OCR.

    One row per side, plus the mixed case, which is the one a reviewer should care about
    most: a page carrying BOTH must reprocess, because the admissible clause must not carry
    the inadmissible one past the gate.
    """
    from socr.pipeline.orchestrator import _resume_verdict_is_permanently_unverifiable

    def verdict(*clauses: tuple[Check, CheckResult]) -> dict:
        return PageVerdict(
            page_num=1,
            clauses=tuple(
                VerdictClause(check, result, SCOPE_PAGE, "reason") for check, result in clauses
            ),
        ).to_dict()

    unverifiable = (Check.EQUATION_FIDELITY, CheckResult.UNKNOWN)
    judge_timeout = (Check.PAGE_JUDGE, CheckResult.UNKNOWN)
    table = (Check.TABLE_STRUCTURE, CheckResult.FAIL)
    # The SAME check with the other result, and it must NOT be admitted. This is slice 1's
    # corrupt-math hybrid clause: the mathematics is DAMAGED and unrecovered, which a run
    # with a provider available can genuinely fix, and
    # ``test_agentic_corrupt_math_flag_uses_region_recovery_not_whole_page`` pins that such
    # a page must re-OCR. Keying the set on the CHECK alone admitted it; that test is what
    # caught the bug, and this row is what stops it coming back.
    corrupt_math = (Check.EQUATION_FIDELITY, CheckResult.FAIL)

    assert _resume_verdict_is_permanently_unverifiable(verdict(unverifiable)) is True

    for inadmissible in (judge_timeout, table, corrupt_math):
        assert _resume_verdict_is_permanently_unverifiable(verdict(inadmissible)) is False, (
            f"{inadmissible[0].value}={inadmissible[1].value} earned the resume exception; it "
            "can be resolved by reprocessing, so skipping keeps unconfirmed text for ever"
        )
        assert (
            _resume_verdict_is_permanently_unverifiable(verdict(unverifiable, inadmissible))
            is False
        ), (
            f"an admissible clause carried {inadmissible[0].value}="
            f"{inadmissible[1].value} past the gate; the exception must require EVERY "
            "clause to be unresolvable"
        )

    # A CHECKED page is never admitted by this route: it has no need of it, and the ordinary
    # SUCCESS gate is the one that should decide it.
    assert _resume_verdict_is_permanently_unverifiable(verdict()) is False

    # And doubt reprocesses: anything unparseable is refused rather than guessed at.
    for junk in (None, {}, "failed", {"clauses": [{"check": "not_a_check"}]}, {"clauses": 7}):
        assert _resume_verdict_is_permanently_unverifiable(junk) is False, junk
