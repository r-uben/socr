"""GH-96: the escalation canary.

Two-sided, threshold-free: accept iff the candidate introduces no value absent from
the page's native text beyond those the incumbent also introduced, AND retains every
native-backed value the incumbent had.

Both halves are load-bearing, and the second is the non-obvious one — see
``test_one_sided_containment_would_accept_a_truncated_candidate``.
"""

from __future__ import annotations

import fitz
import pytest

from socr.tables.escalation_canary import (
    judge_escalation,
    native_value_oracle,
    table_value_tokens,
)

# The fabrication from docs/log/2026-07-30_obr-table-bakeoff.md, Result 4: a vision
# model that could not read its input, at exit 0, bold-celled.
_FABRICATED = "\n".join(
    [
        "| | 2017 | 2018 |",
        "| --- | --- | --- |",
        "| **Revenue** | **23,126** | **25,123** |",
        "| Taxes | 19,451 | 21,326 |",
    ]
)

_ROWS = [
    ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
    ("September energy package", ["43.2", "26.8", "3.7"]),
    ("Energy price guarantee", ["24.8", "26.8", "3.7"]),
    ("Energy bill relief scheme", ["18.4", "9.1", "5.5"]),
]


@pytest.fixture(scope="module")
def table_page(tmp_path_factory):
    path = tmp_path_factory.mktemp("gh96canary") / "table.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 200.0
    for label, values in _ROWS:
        page.insert_text((60.0, y), label, fontsize=9)
        for x, value in zip((300.0, 360.0, 420.0), values):
            page.insert_text((x, y), value, fontsize=9)
        y += 18.0
    page.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    page.draw_line(fitz.Point(50, y), fitz.Point(470, y))
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


_FAITHFUL = _md(_ROWS)


def test_the_oracle_holds_every_table_value(table_page):
    oracle = native_value_oracle(table_page)

    for _label, values in _ROWS:
        for value in values:
            assert value in oracle


def test_a_faithful_candidate_is_accepted(table_page):
    verdict = judge_escalation(table_page, _FAITHFUL, _FAITHFUL)

    assert verdict.accepted
    assert verdict.introduced == ()
    assert verdict.lost == ()


def test_a_fabrication_is_rejected(table_page):
    """The demonstrated failure: confident invented data at exit 0."""
    verdict = judge_escalation(table_page, _FAITHFUL, _FABRICATED)

    assert not verdict.accepted
    assert "absent from the page's native text" in verdict.reason
    assert "23126" in verdict.introduced


def test_one_sided_containment_would_accept_a_truncated_candidate(table_page):
    """Why the coverage half exists.

    ``introduced(∅) = ∅ ⊆ anything``, so containment alone is monotone in the wrong
    direction — it rewards emitting less. A second-pass VLM that truncates is the
    ordinary failure, not an exotic one.
    """
    truncated = _md(_ROWS[:1])

    verdict = judge_escalation(table_page, _FAITHFUL, truncated)

    # One-sided containment alone would have passed it:
    assert verdict.introduced == ()
    # The coverage half rejects it.
    assert not verdict.accepted
    assert "dropped" in verdict.reason
    assert "24.8" in verdict.lost


def test_an_empty_candidate_is_rejected(table_page):
    verdict = judge_escalation(table_page, _FAITHFUL, "no tables here at all")

    assert not verdict.accepted


def test_a_candidate_may_add_values_the_incumbent_missed(table_page):
    """Recovering a page the incumbent dropped entirely is the headline win."""
    verdict = judge_escalation(table_page, "prose only, no table", _FAITHFUL)

    assert verdict.accepted
    assert verdict.lost == ()


def test_a_candidate_inherits_the_incumbents_own_unsupported_tokens(table_page):
    """Self-calibrating: shared oracle-dark misreads cancel on both sides."""
    off_oracle = "| Row | 99.99 |"
    incumbent = _FAITHFUL + "\n\n" + off_oracle
    candidate = _FAITHFUL + "\n\n" + off_oracle

    verdict = judge_escalation(table_page, incumbent, candidate)

    assert verdict.accepted, "a token the incumbent also failed to support must not block"


def test_a_page_without_an_oracle_is_not_judged(tmp_path):
    """No native values means no adjudication — the lane must skip, not guess."""
    path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    opened = fitz.open(path)

    verdict = judge_escalation(opened[0], _FAITHFUL, _FAITHFUL)
    opened.close()

    assert verdict.usable is False
    assert verdict.accepted is False


def test_multiset_surplus_is_diagnostic_and_never_gates(table_page):
    """A merged cell expanded across columns must not be treated as invention.

    The multiset variant rejected a real +75pp improvement over one surplus token,
    so counts are reported and not enforced.
    """
    duplicated = _md(_ROWS + [("Repeat of a real row", ["42.8", "42.8", "42.8"])])

    verdict = judge_escalation(table_page, _FAITHFUL, duplicated)

    assert verdict.accepted
    assert verdict.multiset_surplus, "the surplus should still be reported"


def test_token_extraction_sees_bold_cells(table_page):
    """GH-103: bolded values were invisible, so the gate passed vacuously."""
    assert "23126" in table_value_tokens(_FABRICATED)
