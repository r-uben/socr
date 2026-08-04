"""GH-96: the escalation accept rule.

Accept an escalated table only when it *measurably improves* the page, judged by
hierarchy-aware exactness against the page's own native text layer.

The rule replaced token containment (the canary) after measurement: escalating with
socr's own ``gemini-ocr`` regresses four pages of the reference document, two of
which the incumbent had perfect. Those regressions are entirely native-supported, so
the canary accepts them — a regression is not an invention.
"""

from __future__ import annotations

import fitz
import pytest

from socr.tables.escalation_decision import (
    GATE_CANARY,
    GATE_EXACTNESS,
    GATE_NONE,
    decide_escalation,
)

_ROWS = [
    ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
    ("September energy package", ["43.2", "26.8", "3.7"]),
    ("Energy price guarantee", ["24.8", "26.8", "3.7"]),
    ("Energy bill relief scheme", ["18.4", "9.1", "5.5"]),
]


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    path = tmp_path_factory.mktemp("gh96decide") / "t.pdf"
    doc = fitz.open()
    pg = doc.new_page()
    y = 200.0
    for label, values in _ROWS:
        pg.insert_text((60.0, y), label, fontsize=9)
        for x, v in zip((300.0, 360.0, 420.0), values):
            pg.insert_text((x, y), v, fontsize=9)
        y += 18.0
    pg.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    pg.draw_line(fitz.Point(50, y), fitz.Point(470, y))
    doc.save(path)
    doc.close()
    opened = fitz.open(path)
    yield opened[0]
    opened.close()


def _md(rows):
    out = ["| | c1 | c2 | c3 |", "| --- | --- | --- | --- |"]
    for label, values in rows:
        out.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(out)


_PERFECT = _md(_ROWS)
# The GH-96 failure: values slid one row down inside the nested block.
_SHIFTED = _md(
    [
        ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
        ("September energy package", ["", "", ""]),
        ("Energy price guarantee", ["43.2", "26.8", "3.7"]),
        ("Energy bill relief scheme", ["24.8", "26.8", "3.7"]),
    ]
)


def test_a_real_improvement_is_accepted(page):
    decision = decide_escalation(page, _SHIFTED, _PERFECT)

    assert decision.accepted
    assert decision.gate == GATE_EXACTNESS
    assert decision.delta is not None and decision.delta > 0


def test_a_regression_is_rejected(page):
    """The case the canary cannot see: every value is native-supported."""
    decision = decide_escalation(page, _PERFECT, _SHIFTED)

    assert not decision.accepted
    assert decision.gate == GATE_EXACTNESS
    assert decision.delta is not None and decision.delta < 0


def test_a_tie_is_rejected(page):
    """An equal candidate is not an improvement; keep the incumbent, avoid churn."""
    decision = decide_escalation(page, _PERFECT, _PERFECT)

    assert not decision.accepted
    assert decision.delta == 0.0


def test_a_fabrication_is_rejected_on_quality_alone(page):
    """No separate fabrication test is needed: invented data scores near zero."""
    fabricated = _md(
        [("Revenue", ["23,126", "25,123", "26,204"]), ("Taxes", ["19,451", "21,326", "22,239"])]
    )

    decision = decide_escalation(page, _SHIFTED, fabricated)

    assert not decision.accepted


def test_a_truncated_candidate_is_rejected(page):
    decision = decide_escalation(page, _PERFECT, _md(_ROWS[:1]))

    assert not decision.accepted


def test_an_incumbent_with_no_matching_labels_is_still_compared(page):
    """Shredded labels score 0 but the ground truth is fine — compare, don't defer.

    Gating on the report's ``scorable`` flag would hand exactly the incumbent's
    worst failures to the weaker canary. Two real 0% -> ~86% recoveries on the
    reference document depend on this.
    """
    shredded = _md([("Ttl", ["42.8"]), ("Sep", ["43.2"])])

    decision = decide_escalation(page, shredded, _PERFECT)

    assert decision.accepted
    assert decision.gate == GATE_EXACTNESS, "must be decided on measurement, not deferred"


def test_a_page_with_no_ground_truth_falls_back_to_the_canary(tmp_path):
    path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    opened = fitz.open(path)

    decision = decide_escalation(opened[0], _PERFECT, _PERFECT)
    opened.close()

    assert decision.gate in (GATE_CANARY, GATE_NONE)
    assert not decision.accepted, "with no oracle at all, keep the incumbent"


def test_the_gate_used_is_always_recorded(page):
    """The audit trail must never conflate a measured accept with a mere non-fabrication."""
    decision = decide_escalation(page, _SHIFTED, _PERFECT)

    assert decision.gate in (GATE_EXACTNESS, GATE_CANARY, GATE_NONE)
    assert decision.reason


# ----------------------------------------------------------------------
# #123 TICKET-C1: unexplained lanes rank ahead of exactness
# ----------------------------------------------------------------------

# The incumbent gets every value wrong except the first column; the candidate
# drops the third column entirely (fewer emitted columns than native lanes) but
# nails everything it does emit. Candidate exactness is strictly higher, yet it
# leaves a whole native lane with no home — that must never win.
_INCUMBENT_MOSTLY_WRONG = _md([(label, [values[0], "99.9", "88.8"]) for label, values in _ROWS])
_CANDIDATE_DROPS_A_COLUMN = "\n".join(
    ["| | c1 | c2 |", "| --- | --- | --- |"]
    + [f"| {label} | {values[0]} | {values[1]} |" for label, values in _ROWS]
)


def test_a_candidate_that_increases_unexplained_lanes_is_rejected_despite_higher_exactness(page):
    decision = decide_escalation(page, _INCUMBENT_MOSTLY_WRONG, _CANDIDATE_DROPS_A_COLUMN)

    assert decision.candidate_pct is not None and decision.incumbent_pct is not None
    assert decision.candidate_pct > decision.incumbent_pct, (
        "the fixture must actually exercise the trap: higher exactness, more loss"
    )
    assert decision.candidate_unexplained_lanes > decision.incumbent_unexplained_lanes
    assert not decision.accepted
    assert decision.gate == GATE_EXACTNESS


def test_a_candidate_that_decreases_unexplained_lanes_is_preferred_even_at_equal_exactness(page):
    """Fewer unexplained lanes wins outright; exactness is only the tiebreak."""
    # Incumbent: drops the third column (like the candidate above). Candidate:
    # recovers that column but gets nothing else more right, so exactness ties
    # relative to that incumbent's own baseline — the win must come purely from
    # unexplained lanes going to zero.
    incumbent = _CANDIDATE_DROPS_A_COLUMN
    candidate = _md([(label, [values[0], values[1], "88.8"]) for label, values in _ROWS])

    decision = decide_escalation(page, incumbent, candidate)

    assert decision.candidate_unexplained_lanes < decision.incumbent_unexplained_lanes
    assert decision.accepted
    assert decision.gate == GATE_EXACTNESS
