"""#1270 (B4): a table with no ladder terminal carries its OWN mark when class 2 released its page.

Class 2 removes the numbers clause. A page whose table the ladder never judged used to keep a mark
only by accident (the numbers clause's "no number" UNKNOWN); the release must not leave that table
unmarked. Every assertion pins a DIFFERENCE: the same page with and without a terminal, and the
same page with and without the release.

Review of PR #1325 (Carbon): the mark's scope is 1-based like every other ``table:N`` in the repo,
an HTML ``<table>`` counts, and the seam is idempotent for a page demoted by this mark alone.
"""

from __future__ import annotations

import pytest
from test_gh1270_b4_no_number_pages import _out, _p, _run

from socr.core.marks import Check, CheckResult
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import FailureMode, PageStatus
from socr.core.verdict import (
    UNCHECKED_TABLE_REASON,
    build_page_verdict,
    silent_success_clauses,
    text_layer_numbers_clause,
)

#: A table with no number on either side (class 2), and the layer that prints its words.
WORD_TABLE = "| alpha | beta |\n|---|---|\n| gamma | delta |\n"
LAYER_WORDS = "alpha beta gamma delta"
HTML_TABLE = "<table><tr><td>gamma</td><td>delta</td></tr></table>"


def _marks(clauses):
    return [
        (c.check, c.scope)
        for c in clauses
        if c.check is Check.SOURCE_EVIDENCE and c.scope.startswith("table:")
    ]


def _page(**over):
    return _p(own_layer_text=LAYER_WORDS, own_layer_word_reference=LAYER_WORDS, **over)


def test_premise_the_numbers_clause_is_released() -> None:
    assert text_layer_numbers_clause(_out(WORD_TABLE), _page()) is None


def test_a_released_page_with_an_unjudged_table_keeps_a_table_scoped_mark() -> None:
    clauses = silent_success_clauses(_out(WORD_TABLE), _page(), frozenset())
    assert _marks(clauses) == [(Check.SOURCE_EVIDENCE, "table:1")]
    (clause,) = [c for c in clauses if c.scope == "table:1"]
    assert clause.result is CheckResult.UNKNOWN and clause.reason == UNCHECKED_TABLE_REASON


def test_an_html_table_is_marked_too() -> None:
    out = _out(HTML_TABLE)
    page = _p(own_layer_text="gamma delta", own_layer_word_reference="gamma delta")
    assert text_layer_numbers_clause(out, page) is None, "premise"
    assert _marks(silent_success_clauses(out, page, frozenset())) == [
        (Check.SOURCE_EVIDENCE, "table:1")
    ]


@pytest.mark.parametrize(
    "terminal",
    [
        {"event_kinds": frozenset({"table_ladder_accepted"})},
        {"event_kinds": frozenset({"table_ladder_withheld"})},
        {"disposition": FailureMode.TABLE_UNVERIFIED},
        {"disposition": FailureMode.TABLE_REJECTED},
    ],
)
def test_the_same_page_with_a_ladder_terminal_carries_no_such_mark(terminal) -> None:
    page = _page(table_ladder_disposition=terminal.get("disposition"))
    clauses = silent_success_clauses(
        _out(WORD_TABLE), page, terminal.get("event_kinds", frozenset())
    )
    assert _marks(clauses) == []


def test_a_page_that_was_not_released_is_unchanged() -> None:
    """Scope guard: the mark is the release's backstop, not a new rule for every page."""
    page = _page(text_coverage_verdict="flagged")  # the numbers UNKNOWN stands
    assert _marks(silent_success_clauses(_out(WORD_TABLE), page, frozenset())) == []


def test_each_table_is_marked_by_its_own_ordinal() -> None:
    out = _out(WORD_TABLE + "\nbetween\n\n" + WORD_TABLE)
    page = _p(
        own_layer_text=LAYER_WORDS + " between " + LAYER_WORDS,
        own_layer_word_reference=LAYER_WORDS + " between " + LAYER_WORDS,
    )
    assert text_layer_numbers_clause(out, page) is None, "premise"
    assert _marks(silent_success_clauses(out, page, frozenset())) == [
        (Check.SOURCE_EVIDENCE, "table:1"),
        (Check.SOURCE_EVIDENCE, "table:2"),
    ]


def test_a_failed_marker_page_has_no_table_to_mark() -> None:
    out = _out("[page 1 failed: unverifiable table - see image]")
    assert _marks(silent_success_clauses(out, _page(), frozenset())) == []


def test_the_page_verdict_is_not_checked_so_the_page_is_demoted() -> None:
    verdict = build_page_verdict(1, _out(WORD_TABLE), _page())
    assert _marks(verdict.clauses) == [(Check.SOURCE_EVIDENCE, "table:1")]


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_the_seam_is_idempotent_for_a_page_demoted_by_this_mark_alone(tmp_path, providers) -> None:
    """Carbon: a second pass over the finalized page must state the same mode and clauses, and
    never append the ``unknown=UNKNOWN`` fallback. Both provider states."""
    from socr.core.manifest import finalized_page_record

    state, first = _run(
        tmp_path,
        check_coverage=True,
        providers=providers,
        reading=WORD_TABLE,
        layer=LAYER_WORDS,
        want_state=True,
    )
    again = finalized_page_record(state, 1)
    clauses = [(c.check.value, c.result.value) for c in again.verdict.clauses]
    assert clauses == first[2]
    if providers:
        assert first[0] is PageStatus.WARNING
        assert ("source_evidence", "UNKNOWN") in first[2]
        assert ("unknown", "UNKNOWN") not in clauses
    else:
        # no provider: the page is never read, so this class has nothing to say about it
        assert ("source_evidence", "UNKNOWN") not in first[2]


def test_the_seam_states_the_same_mode_and_clauses_on_a_second_pass() -> None:
    """Carbon (re-check of PR #1325): a page demoted by this mark ALONE was left with
    ``failure_mode=NONE`` and gained the ``unknown=UNKNOWN`` fallback on the second pass. The
    seam is called twice, directly; both passes must state the same."""
    from socr.core.manifest import apply_verdict_marks

    page = _page()
    first_out, first = apply_verdict_marks(_out(WORD_TABLE), page, 1)
    assert first_out.status is PageStatus.WARNING
    assert first_out.failure_mode is FailureMode.NUMBERS_UNCHECKED
    second_out, second = apply_verdict_marks(first_out, page, 1)
    assert second.clauses == first.clauses
    assert second_out.failure_mode is first_out.failure_mode
    assert second_out.text == first_out.text
    assert "unknown" not in {c.check.value for c in second.clauses}
