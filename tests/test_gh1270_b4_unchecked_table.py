"""#1270 (B4, review of PR #1325): a table with no ladder terminal carries its OWN mark.

Classes 2 and 3 clear the numbers clause. A page whose table the ladder never judged used to
keep a mark only by accident (the numbers clause's "layer refused" UNKNOWN); clearing the class
must not leave that table unmarked. Every assertion pins a DIFFERENCE: the same page with and
without a terminal, and the same page with and without the clearance.
"""

from __future__ import annotations

import pytest
from test_gh1270_b4_no_number_pages import _p as _blank_p
from test_gh1270_b4_shape_refused_layer import AGREES, _p as _shape_p

from socr.core.marks import Check, CheckResult
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.verdict import (
    UNCHECKED_TABLE_REASON,
    build_page_verdict,
    silent_success_clauses,
    text_layer_numbers_clause,
)

#: A pipe table whose numbers agree with ``AGREES`` (class 3: shape-refused layer that agrees).
NUMERIC_TABLE = "| k | v |\n|---|---|\n| A | 1 |\n| B | 11 |\n| C | 2 |\n| D | 12 |\n"
#: A table with no number on either side (class 2).
WORD_TABLE = "| alpha | beta |\n|---|---|\n| gamma | delta |\n"
LAYER_WORDS = "alpha beta gamma delta"


def _out(text: str) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")


def _marks(clauses):
    return [
        (c.check, c.scope)
        for c in clauses
        if c.check is Check.SOURCE_EVIDENCE and c.scope.startswith("table:")
    ]


def _class3(**over):
    return _out(NUMERIC_TABLE), _shape_p(own_layer_text=AGREES, **over)


def _class2(**over):
    return _out(WORD_TABLE), _blank_p(
        own_layer_text=LAYER_WORDS, own_layer_word_reference=LAYER_WORDS, **over
    )


def test_premise_the_numbers_clause_is_cleared_in_both_classes() -> None:
    assert text_layer_numbers_clause(*_class3()) is None
    assert text_layer_numbers_clause(*_class2()) is None


@pytest.mark.parametrize("build", [_class3, _class2], ids=["class3", "class2"])
def test_a_cleared_page_with_an_unjudged_table_keeps_a_table_scoped_mark(build) -> None:
    out, p = build()
    clauses = silent_success_clauses(out, p, frozenset())
    assert _marks(clauses) == [(Check.SOURCE_EVIDENCE, "table:0")]
    (clause,) = [c for c in clauses if c.scope == "table:0"]
    assert clause.result is CheckResult.UNKNOWN and clause.reason == UNCHECKED_TABLE_REASON


@pytest.mark.parametrize("build", [_class3, _class2], ids=["class3", "class2"])
@pytest.mark.parametrize(
    "terminal",
    [
        {"event_kinds": frozenset({"table_ladder_accepted"})},
        {"event_kinds": frozenset({"table_ladder_withheld"})},
        {"disposition": FailureMode.TABLE_UNVERIFIED},
        {"disposition": FailureMode.TABLE_REJECTED},
    ],
)
def test_the_same_page_with_a_ladder_terminal_carries_no_such_mark(build, terminal) -> None:
    out, p = build(table_ladder_disposition=terminal.get("disposition"))
    clauses = silent_success_clauses(out, p, terminal.get("event_kinds", frozenset()))
    assert _marks(clauses) == []


def test_a_page_that_is_not_cleared_by_b4_is_unchanged() -> None:
    """Scope guard: the mark is the clearance's backstop, not a new rule for every page."""
    out, p = _class3(layer_refused_for_shape_alone=False)  # the old UNKNOWN stands
    assert _marks(silent_success_clauses(out, p, frozenset())) == []


def test_each_table_is_marked_by_its_own_ordinal() -> None:
    out = _out(NUMERIC_TABLE + "\nbetween\n\n" + NUMERIC_TABLE)
    p = _shape_p(own_layer_text=AGREES + " " + AGREES)
    assert text_layer_numbers_clause(out, p) is None
    assert _marks(silent_success_clauses(out, p, frozenset())) == [
        (Check.SOURCE_EVIDENCE, "table:0"),
        (Check.SOURCE_EVIDENCE, "table:1"),
    ]


def test_a_failed_marker_page_has_no_table_to_mark() -> None:
    out = _out("[page 1 failed: unverifiable table — see image]")
    assert silent_success_clauses(out, _shape_p(own_layer_text=AGREES), frozenset()) == ()


def test_the_page_verdict_is_not_checked_so_the_page_is_demoted() -> None:
    out, p = _class3()
    verdict = build_page_verdict(1, out, p)
    assert _marks(verdict.clauses) == [(Check.SOURCE_EVIDENCE, "table:0")]
    assert out.status is PageStatus.SUCCESS  # the seam (manifest) demotes on any silent clause
