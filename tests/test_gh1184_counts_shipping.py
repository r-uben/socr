"""#1184 PR 2: when a chart panel's counts may ship, and what it says when they may not.

Hermetic: the panels here are stand-ins carrying exactly what ``decide_panel``
reads from a real ``PanelReading``, and the pages are drawn with PyMuPDF. No
provider, no network, no model -- the chart reader and this gate are
deterministic.

The rule, ratified by the orchestrator on 2026-10-17: a series ships its counts
only when its reading is COMPLETE and it passes the witness gate. The two are
separate conditions and the gate alone is not enough -- it compares only the
lowest and highest bars it can see, so a bar the reader could not measure in
between passes it unnoticed. Measured on the Fed pilot, 5 of the 35 series that
pass the gate have unread bars; requiring completeness ships 30 series.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import fitz
import pytest

from socr.core.marks import CheckResult
from socr.figures.chart_reader import INTEGER
from socr.figures.chart_witness import (
    SHIPPED_REASON,
    floor_replacement_body,
    region_clause_records,
    WITHHELD_NO_VARIABLE,
    WitnessTable,
    decide_panel,
    figure_variable,
    witness_for_document,
)

_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class _Cell:
    bin_label: str
    status: str
    count: int | None


@dataclass(frozen=True)
class _Series:
    name: str
    cells: tuple[_Cell, ...]


@dataclass(frozen=True)
class _Panel:
    label: str
    series: tuple[_Series, ...] = field(default_factory=tuple)


def _series(name, pairs):
    """``pairs`` is ``(bin label, count or None)``; None is a bar not read."""
    return _Series(
        name=name,
        cells=tuple(
            _Cell(label, INTEGER if count is not None else _UNRESOLVED, count)
            for label, count in pairs
        ),
    )


#: A witness holding one variable's published range, as the table prints it.
_WITNESS = WitnessTable(
    rows={
        ("change in real gdp", "current", "2015"): (1.9, 2.5),
        ("change in real gdp", "prior", "2015"): (1.7, 2.3),
    },
    prior_months=frozenset({"June"}),
)
_COMPLETE = (("1.6-1.7", 0), ("1.8-1.9", 1), ("2.0-2.1", 8), ("2.2-2.3", 5), ("2.4-2.5", 3))


def test_a_complete_reading_that_passes_the_gate_ships_its_counts():
    panel = _Panel("2015", (_series("September projections", _COMPLETE),))
    decision = decide_panel(panel, _WITNESS, "change in real GDP")
    assert decision.shipped
    assert decision.reason == SHIPPED_REASON
    assert len(decision.blocks) == 1
    assert "1.8-1.9" in decision.blocks[0] and "| 8 |" in decision.blocks[0].replace("  ", " ")


def test_a_bar_the_reader_could_not_measure_withholds_the_series():
    """The condition the gate cannot supply. The hole is in the MIDDLE on purpose.

    Its lowest and highest filled bars still span the published range, so the
    gate passes it; only the completeness condition catches it.
    """
    holed = (("1.6-1.7", 0), ("1.8-1.9", 1), ("2.0-2.1", None), ("2.2-2.3", 5), ("2.4-2.5", 3))
    panel = _Panel("2015", (_series("September projections", holed),))
    decision = decide_panel(panel, _WITNESS, "change in real GDP")
    assert not decision.shipped
    assert decision.blocks == ()
    assert "could not be read" in decision.reason
    assert "2.0-2.1" in decision.reason, "and it names WHICH bar"


def test_a_reading_that_contradicts_the_published_range_withholds():
    shifted = (("1.6-1.7", 0), ("1.8-1.9", 0), ("2.0-2.1", 8), ("2.2-2.3", 5), ("2.4-2.5", 3))
    panel = _Panel("2015", (_series("September projections", shifted),))
    decision = decide_panel(panel, _WITNESS, "change in real GDP")
    assert not decision.shipped
    assert "do not span the range" in decision.reason
    assert "1.9" in decision.reason and "2.5" in decision.reason


def test_a_previous_meetings_series_is_checked_against_ITS_OWN_row():
    """Both series ship, each against the row its own month names.

    Checked against the current row instead, the June series fails -- that was
    the bug #1155 found -- so this is a difference test on the binding, not on
    the gate.
    """
    panel = _Panel(
        "2015",
        (
            _series("September projections", _COMPLETE),
            _series(
                "June projections",
                (("1.6-1.7", 1), ("1.8-1.9", 9), ("2.0-2.1", 6), ("2.2-2.3", 1), ("2.4-2.5", 0)),
            ),
        ),
    )
    decision = decide_panel(panel, _WITNESS, "change in real GDP")
    assert decision.shipped
    assert decision.reason == SHIPPED_REASON
    assert len(decision.blocks) == 2


def test_one_series_withheld_does_not_withhold_the_other_but_IS_stated():
    """Per series, not per panel -- and the line must say what is absent.

    The two series are drawn in the same axes but checked against different
    rows, so a bar unread in one says nothing about the other. Shipping the
    good one silently would let a reader take the panel for whole.
    """
    panel = _Panel(
        "2015",
        (
            _series("September projections", _COMPLETE),
            _series(
                "June projections",
                (("1.6-1.7", 1), ("1.8-1.9", None), ("2.0-2.1", 6), ("2.2-2.3", 1), ("2.4-2.5", 0)),
            ),
        ),
    )
    decision = decide_panel(panel, _WITNESS, "change in real GDP")
    assert decision.shipped
    assert len(decision.blocks) == 1, "only the complete series ships"
    assert "September projections" in decision.blocks[0]
    assert SHIPPED_REASON in decision.reason
    assert "June projections" in decision.reason and "could not be read" in decision.reason


@pytest.mark.parametrize(
    "witness,variable",
    [(None, "change in real GDP"), (_WITNESS, ""), (_WITNESS, "a variable nobody prints")],
)
def test_without_a_witness_or_a_variable_nothing_ships(witness, variable):
    panel = _Panel("2015", (_series("September projections", _COMPLETE),))
    decision = decide_panel(panel, witness, variable)
    assert not decision.shipped
    assert decision.blocks == ()
    assert decision.reason == WITHHELD_NO_VARIABLE


def test_a_year_the_table_does_not_publish_withholds_by_name():
    panel = _Panel("2099", (_series("September projections", _COMPLETE),))
    decision = decide_panel(panel, _WITNESS, "change in real GDP")
    assert not decision.shipped
    assert "publishes no range" in decision.reason and "2099" in decision.reason


# --------------------------------------------------------------------------------------
# Finding the variable and the witness
# --------------------------------------------------------------------------------------


def _caption_page(tmp_path, lines):
    doc = fitz.open()
    page = doc.new_page(width=792, height=792)
    for index, text in enumerate(lines):
        page.insert_text((40, 60 + index * 16), text, fontsize=9)
    path = tmp_path / "caption.pdf"
    doc.save(path)
    doc.close()
    return fitz.open(path)


@pytest.mark.parametrize(
    "lines,expected",
    [
        (
            [
                "Figure 2.A. Distribution of participants' projections for the change in "
                "real GDP, 2008-11"
            ],
            "change in real GDP",
        ),
        (
            [
                "Figure 3.D. Distribution of participants' projections for core PCE "
                "inflation, 2015-18"
            ],
            "core PCE inflation",
        ),
        # The policy-path caption wraps mid-phrase and names a variable twice.
        # Read line by line it yields "federal", which binds to nothing.
        (
            [
                "Figure 3.E. Distribution of participants' judgments of the midpoint of the "
                "appropriate target range for the federal",
                "funds rate or the appropriate target level for the federal funds rate, "
                "2015-18 and over the longer run",
            ],
            "federal funds rate",
        ),
        (["Number of participants", "2015"], ""),
    ],
)
def test_the_caption_names_the_variable_even_when_it_wraps(tmp_path, lines, expected):
    doc = _caption_page(tmp_path, lines)
    found = figure_variable(doc[0])
    doc.close()
    assert found == expected


def test_the_witness_is_found_on_another_page_and_only_a_real_table_is_taken(tmp_path):
    """Every page is offered to the parser; prose must not be mistaken for a table."""
    from test_gh1184_table1_layouts import _GROUPS, _ROWS, _YEARS, _side_by_side

    table = _side_by_side(tmp_path / "t.pdf", years=_YEARS, groups=_GROUPS, rows=_ROWS)
    out = fitz.open()
    prose = out.new_page(width=792, height=792)
    prose.insert_text(
        (60, 80), "In conjunction with the April 2010 meeting, participants", fontsize=9
    )
    prose.insert_text(
        (60, 96), "projected growth of 3.2 to 3.7 percent for 2010 and 2011.", fontsize=9
    )
    out.insert_pdf(table)
    table.close()
    witness = witness_for_document(out)
    out.close()
    assert witness is not None
    assert witness.range_for("Change in real GDP", "current", "2010") == (2.7, 4.0)


def test_no_projection_table_in_the_document_means_no_witness(tmp_path):
    doc = _caption_page(tmp_path, ["Record of policy actions. The Committee agreed."])
    assert witness_for_document(doc) is None
    doc.close()


# --------------------------------------------------------------------------------------
# The counts have to reach the page, and the withheld ones have to not
# --------------------------------------------------------------------------------------


@dataclass
class _Reading:
    panels: dict


def _two_panel_page():
    ships = _Panel("2015", (_series("September projections", _COMPLETE),))
    holed = _Panel(
        "2015",
        (
            _series(
                "September projections",
                (("1.6-1.7", 0), ("1.8-1.9", 1), ("2.0-2.1", None), ("2.2-2.3", 5), ("2.4-2.5", 3)),
            ),
        ),
    )
    reading = _Reading(panels={1: ships, 2: holed})
    decisions = {
        index: decide_panel(panel, _WITNESS, "change in real GDP")
        for index, panel in reading.panels.items()
    }
    return reading, decisions


def test_a_passing_panels_counts_reach_the_page_and_a_withheld_panels_do_not():
    reading, decisions = _two_panel_page()
    body = floor_replacement_body(
        None,
        7,
        "some prose outside the charts",
        region_pictures={1: "![r1](a.png)", 2: "![r2](b.png)"},
        reading=reading,
        region_count=2,
        decisions=decisions,
    )
    assert body is not None
    assert "September projections (2015):" in body
    assert body.count("September projections (2015):") == 1, "only the panel that passed"
    assert "| 8 |" in body.replace("  ", " "), "and its counts are the numbers read"
    assert "could not be read" in body, "while the other panel says why it shipped none"
    assert "2.0-2.1" in body


def test_the_page_and_its_verdict_say_the_same_thing_about_each_panel():
    """One source for both, so a reader and a consumer cannot be told different things."""
    reading, decisions = _two_panel_page()
    records = region_clause_records(
        7,
        region_pictures={1: "![r1](a.png)", 2: "![r2](b.png)"},
        reading=reading,
        region_count=2,
        decisions=decisions,
    )
    assert len(records) == 2
    assert SHIPPED_REASON in records[0]["reason"]
    assert "could not be read" in records[1]["reason"]
    # A withheld panel was judged and found wanting; a shipped one is not proven
    # right by the gate, only narrowed, so it stays UNKNOWN.
    assert records[0]["result"] == CheckResult.UNKNOWN.value
    assert records[1]["result"] == CheckResult.FAIL.value
    body = floor_replacement_body(
        None,
        7,
        "",
        region_pictures={1: "![r1](a.png)", 2: "![r2](b.png)"},
        reading=reading,
        region_count=2,
        decisions=decisions,
        region_records=records,
    )
    for record in records:
        assert record["reason"].split(";")[0][:48] in body


def test_without_decisions_nothing_claims_the_counts_were_judged():
    """The earlier state of this lane: the gate built, no counts wired to an output."""
    reading, _ = _two_panel_page()
    records = region_clause_records(
        7,
        region_pictures={1: "![r1](a.png)", 2: "![r2](b.png)"},
        reading=reading,
        region_count=2,
    )
    assert all("not shipped in this version" in r["reason"] for r in records)
    assert all(r["result"] == CheckResult.UNKNOWN.value for r in records)
