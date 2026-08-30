"""GH-353 TICKET-C3: the table-judge ladder's page disposition must survive
winner selection.

``_select_page_output_tagged`` has many endings, several of which reconstruct
a page as clean SUCCESS / ``audit_passed=True`` whenever no OTHER
native-distrust flag happened to fire on that page -- notably the
native-only reconstruction path (manifest.py ~:1271). The ladder's
REJECTED/UNVERIFIED verdict (B1, not yet wired into production) is judged
AFTER routing, so no cascade branch can see it on its own; without a guard
applied at the shared final seam, a rejected table could be silently
reconstructed as a passing page downstream of selection.

Nothing in production SETS ``PageState.table_ladder_disposition`` yet (B1
owns that); these tests plant it directly to prove the guard, at
``_winning_page_output`` and ``finalized_page_outputs``, is inert with the
attribute absent and enforces the invariant once it is present.
"""

from __future__ import annotations

import pytest

from socr.core.document import DocumentHandle
from socr.core.manifest import _winning_page_output, finalized_page_outputs
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState


def _fake_handle(page_count: int = 1) -> DocumentHandle:
    import pathlib

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DocumentHandle, "__post_init__", lambda self: None)
        return DocumentHandle(path=pathlib.Path("/tmp/fake.pdf"), page_count=page_count)


def _clean_native_only_state(page_num: int = 1) -> DocumentState:
    """A page that would ordinarily ship the NATIVE_CLEAN ending: born-digital,
    native text present, no table-distrust flag, no attempts -- plain SUCCESS /
    audit_passed=True (manifest.py's native-only reconstruction path)."""
    state = DocumentState(handle=_fake_handle())
    ps = state.pages[page_num]
    ps.is_born_digital = True
    ps.native_text = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    return state


def _passing_model_state(page_num: int = 1) -> DocumentState:
    """A page whose best_output is a clean, passing, non-native attempt --
    the ordinary PASSING_BEST_OUTPUT ending."""
    state = DocumentState(handle=_fake_handle())
    ps = state.pages[page_num]
    ps.is_born_digital = False
    passing = PageOutput(
        page_num=page_num,
        text="model reading of the page",
        status=PageStatus.SUCCESS,
        engine="qwen-local",
        audit_passed=True,
    )
    ps.attempts.append(passing)
    ps.best_output = passing
    return state


# ---------------------------------------------------------------------------
# Inert with no disposition present (default / today's behaviour, untouched)
# ---------------------------------------------------------------------------


def test_no_disposition_ships_native_clean_unchanged():
    state = _clean_native_only_state()

    winner = _winning_page_output(state, 1)

    assert winner.status is PageStatus.SUCCESS
    assert winner.audit_passed is True
    assert winner.failure_mode is FailureMode.NONE


def test_no_disposition_ships_passing_best_output_unchanged():
    state = _passing_model_state()

    winner = _winning_page_output(state, 1)

    assert winner.status is PageStatus.SUCCESS
    assert winner.audit_passed is True
    assert winner.text == "model reading of the page"


# ---------------------------------------------------------------------------
# REJECTED / UNVERIFIED preserved through the native-only reconstruction path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [FailureMode.TABLE_REJECTED, FailureMode.TABLE_UNVERIFIED])
def test_native_only_reconstruction_preserves_disposition(mode):
    """The :1271 native-only ending must not regain SUCCESS while REJECTED."""
    state = _clean_native_only_state()
    state.pages[1].table_ladder_disposition = mode

    winner = _winning_page_output(state, 1)

    assert winner.status is not PageStatus.SUCCESS
    assert winner.audit_passed is False
    assert winner.failure_mode is mode
    # Content must not be lost -- only the disposition, never the text.
    assert "| A | B |" in winner.text


@pytest.mark.parametrize("mode", [FailureMode.TABLE_REJECTED, FailureMode.TABLE_UNVERIFIED])
def test_passing_best_output_preserves_disposition(mode):
    """A rejected candidate can still lose SELECTION -- but the PAGE cannot
    regain SUCCESS while the page's disposition is REJECTED/UNVERIFIED."""
    state = _passing_model_state()
    state.pages[1].table_ladder_disposition = mode

    winner = _winning_page_output(state, 1)

    assert winner.audit_passed is False
    assert winner.status is not PageStatus.SUCCESS
    assert winner.failure_mode is mode
    assert winner.text == "model reading of the page"


def test_disposition_backfills_failure_mode_when_already_non_passing():
    """A page already demoted for a different, unset (NONE) reason still gets
    the disposition recorded rather than surfacing FailureMode.NONE."""
    state = _clean_native_only_state()
    ps = state.pages[1]
    ps.needs_ocr_enhancement = True
    ps.attempts.append(
        PageOutput(
            page_num=1,
            text="rejected model attempt",
            status=PageStatus.WARNING,
            engine="qwen-local",
            audit_passed=False,
        )
    )
    ps.table_ladder_disposition = FailureMode.TABLE_REJECTED

    winner = _winning_page_output(state, 1)

    assert winner.audit_passed is False
    assert winner.failure_mode is FailureMode.TABLE_REJECTED


def test_disposition_does_not_override_a_more_specific_failure_mode():
    """A page demoted for a DIFFERENT, already-specific reason (e.g. the
    GH-226 table-emission guard, or an earlier native-table floor) keeps that
    more precise diagnosis -- the ladder disposition never overwrites a
    failure_mode that already explains the demotion."""
    state = DocumentState(handle=_fake_handle())
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.native_text = "collapsed table text"
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = True
    ps.attempts.append(
        PageOutput(
            page_num=1,
            text="ragged attempt",
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
        )
    )
    ps.table_ladder_disposition = FailureMode.TABLE_UNVERIFIED

    winner = _winning_page_output(state, 1)

    assert winner.audit_passed is False
    assert winner.failure_mode is FailureMode.NATIVE_TABLE_STRUCTURE_FAILED


# ---------------------------------------------------------------------------
# The injection is at the shared seam, not merely on best_output
# ---------------------------------------------------------------------------


def test_guard_reads_page_state_not_best_output_attribute():
    """The disposition lives on ``PageState``, not on the ``PageOutput`` --
    setting it must demote the page even though nothing about the winning
    ``PageOutput`` object itself changed."""
    state = _passing_model_state()
    winner_before = _winning_page_output(state, 1)
    assert winner_before.audit_passed is True  # sanity: unset is a no-op

    state.pages[1].table_ladder_disposition = FailureMode.TABLE_REJECTED
    winner_after = _winning_page_output(state, 1)

    assert winner_after.audit_passed is False
    assert winner_after.failure_mode is FailureMode.TABLE_REJECTED
    # best_output itself is untouched -- selection-time flag, not a
    # page-quality flag (the #252 round-1 defect this repo guards against).
    assert state.pages[1].best_output.audit_passed is True


def test_finalized_page_outputs_also_preserves_disposition():
    """``finalized_page_outputs`` (the assembled-Markdown / manifest-freeze
    path) must apply the SAME guard as ``_winning_page_output``, not merely
    ``best_output`` in isolation, so the shipped .md and the manifest blob
    can never disagree about whether a rejected page shipped SUCCESS."""
    state = _clean_native_only_state()
    state.pages[1].table_ladder_disposition = FailureMode.TABLE_REJECTED

    outputs = finalized_page_outputs(state)

    assert len(outputs) == 1
    assert outputs[0].audit_passed is False
    assert outputs[0].status is not PageStatus.SUCCESS
    assert outputs[0].failure_mode is FailureMode.TABLE_REJECTED


def test_finalized_page_outputs_inert_with_no_disposition():
    state = _clean_native_only_state()

    outputs = finalized_page_outputs(state)

    assert outputs[0].audit_passed is True
    assert outputs[0].status is PageStatus.SUCCESS


def test_unknown_disposition_value_is_ignored():
    """Only the two ladder terminals demote; an unrelated failure_mode value
    (or anything else someone might mistakenly plant here) must not."""
    state = _clean_native_only_state()
    state.pages[1].table_ladder_disposition = FailureMode.HALLUCINATION

    winner = _winning_page_output(state, 1)

    assert winner.status is PageStatus.SUCCESS
    assert winner.audit_passed is True
