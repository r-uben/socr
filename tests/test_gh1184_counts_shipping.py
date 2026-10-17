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
    WITHHELD_NO_TABLE,
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
    assert "change in real GDP" in decision.blocks[0], "the header names the variable"


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
    assert "September projections:" in decision.blocks[0]
    assert SHIPPED_REASON in decision.reason
    assert "June projections" in decision.reason and "could not be read" in decision.reason


@pytest.mark.parametrize(
    "witness,variable,expected",
    [
        (None, "change in real GDP", WITHHELD_NO_TABLE),
        (_WITNESS, "", WITHHELD_NO_VARIABLE),
        (_WITNESS, "a variable nobody prints", WITHHELD_NO_VARIABLE),
    ],
)
def test_without_a_witness_or_a_variable_nothing_ships(witness, variable, expected):
    """And each says which of the two is missing.

    A document with no projection table at all was being told its CAPTION was
    at fault, which is false of every non-SEP document the chart floor reaches
    (PR #1199 review).
    """
    panel = _Panel("2015", (_series("September projections", _COMPLETE),))
    decision = decide_panel(panel, witness, variable)
    assert not decision.shipped
    assert decision.blocks == ()
    assert decision.reason == expected
    assert not decision.decided, "the check could not run, so nothing was judged"


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
        # No comma closes the phrase, so where the variable ends is not
        # established. Refusing costs this caption its counts; guessing could
        # bind the panel to the wrong variable and check it against another
        # variable's published range.
        (["Figure 2.A. Distribution of participants' projections for the change in real GDP"], ""),
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
        index: decide_panel(
            panel, _WITNESS, "change in real GDP", region_index=index, region_count=2
        )
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
    assert "region 1 of 2, change in real GDP, 2015 -- September projections:" in body
    assert body.count("-- September projections:") == 1, "only the panel that passed"
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
    # Bars unread is a check that could not RUN, so UNKNOWN, not FAIL.
    assert records[1]["result"] == CheckResult.UNKNOWN.value
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


def _panels(first_ships: bool):
    """Two panels, with the SHIPPING one placed where the test asks for it."""
    holed = _series(
        "September projections",
        (("1.6-1.7", 0), ("1.8-1.9", 1), ("2.0-2.1", None), ("2.2-2.3", 5), ("2.4-2.5", 3)),
    )
    good = _series("September projections", _COMPLETE)
    order = (good, holed) if first_ships else (holed, good)
    reading = _Reading(panels={1: _Panel("2015", (order[0],)), 2: _Panel("2015", (order[1],))})
    decisions = {
        index: decide_panel(
            panel, _WITNESS, "change in real GDP", region_index=index, region_count=2
        )
        for index, panel in reading.panels.items()
    }
    return reading, decisions


def test_the_LAST_regions_counts_ship_too():
    """A fixture that always ships from region 1 cannot see the last one dropped."""
    reading, decisions = _panels(first_ships=False)
    body = floor_replacement_body(
        None,
        9,
        "",
        region_pictures={1: "![r1](a.png)", 2: "![r2](b.png)"},
        reading=reading,
        region_count=2,
        decisions=decisions,
    )
    assert "region 2 of 2, change in real GDP, 2015 -- September projections:" in body


def test_counts_do_not_ship_under_a_line_that_says_they_did_not():
    """Blocker 2: a region whose picture failed said "values not shipped" and shipped them.

    The counts used to be collected straight from the decisions, so a region
    that took the no-picture branch of ``region_clause_records`` got a line
    saying its values were withheld and a count table underneath it -- with no
    "read from the drawing, not verified" sentence anywhere on the page, since
    that sentence lives in the line that was replaced. Grok reproduced it on
    2008-10-29 p16 by pointing ``figures_dir`` at a file.
    """
    reading, decisions = _panels(first_ships=True)
    assert decisions[1].shipped, "the panel itself is fine; only its picture is missing"
    body = floor_replacement_body(
        None,
        9,
        "",
        region_pictures={2: "![r2](b.png)"},
        reading=reading,
        region_count=2,
        decisions=decisions,
    )
    assert "picture not rendered" in body
    assert "-- September projections:" not in body, (
        "no count table under a line that says the values were not shipped"
    )
    assert "| 8 |" not in body.replace("  ", " ")


def test_what_ships_and_what_the_line_says_cannot_disagree():
    """The general form of blocker 2, stated as the invariant it now holds.

    Every count table on the page belongs to a region whose own record carries
    the shipped reason, and every region carrying that reason contributes its
    tables. One source, so the two cannot drift.
    """
    for first_ships in (True, False):
        for pictures in ({1: "a", 2: "b"}, {2: "b"}, {1: "a"}, {}):
            reading, decisions = _panels(first_ships=first_ships)
            records = region_clause_records(
                9,
                region_pictures=pictures,
                reading=reading,
                region_count=2,
                decisions=decisions,
            )
            body = (
                floor_replacement_body(
                    None,
                    9,
                    "",
                    region_pictures=pictures,
                    reading=reading,
                    region_count=2,
                    decisions=decisions,
                    region_records=records,
                )
                or ""
            )
            shipping = {
                index
                for index, record in enumerate(records, start=1)
                if SHIPPED_REASON in record["reason"]
            }
            tables = body.count("-- September projections:")
            assert tables == len(shipping), (first_ships, pictures, tables, shipping)


# --------------------------------------------------------------------------------------
# The real caller. Blocker 3: none of the wiring above was pinned where it runs.
# --------------------------------------------------------------------------------------

#: The chart's own series names, as ``build_chart`` draws its legend.
_SOLID, _DASHED = "September", "June"
#: Counts for the two series, over ``build_chart``'s default bins
#: 1.0 .. 5.0. The filled bins of each have to span the range its own row of
#: the table publishes, which is what the gate checks.
_SOLID_COUNTS = [0, 1, 3, 2, 0]
_DASHED_COUNTS = [1, 2, 1, 0, 0]
_CAPTION = (
    "Figure 2.A. Distribution of participants' projections for the change in real GDP, 2010-12"
)


def _chart_and_table_document(tmp_path, *, caption=_CAPTION, table=True):
    """A two-page document: a histogram page, then its own Table 1 page.

    This is the shape the chart floor meets in production -- the witness is on
    ANOTHER page of the same document -- and nothing in the suite drove it
    before, so the whole lookup could be deleted with every test still green.
    """
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_gh635_chart_reader import build_chart
    from test_gh1184_table1_layouts import _GROUPS, _side_by_side

    chart, _boxes = build_chart(
        tmp_path / "chart.pdf", _SOLID_COUNTS, _DASHED_COUNTS, heading="2010"
    )
    if caption:
        chart[0].insert_text(fitz.Point(60, 40), caption, fontsize=8)

    rows = (
        (
            "Change in real GDP",
            {
                (0, 0): "3.0",
                (0, 1): "3.4",
                (1, 0): "2.5 to 3.5",
                (1, 1): "3.0 to 4.0",
                (2, 0): "2.0 to 4.0",
                (2, 1): "2.5 to 4.5",
            },
        ),
        (
            "June projection",
            {
                (0, 0): "2.0",
                (0, 1): "3.2",
                (1, 0): "1.5 to 2.5",
                (1, 1): "2.8 to 3.8",
                (2, 0): "1.0 to 3.0",
                (2, 1): "2.0 to 4.2",
            },
        ),
    )
    out = fitz.open()
    out.insert_pdf(chart)
    chart.close()
    if table:
        witness_doc = _side_by_side(
            tmp_path / "t1.pdf", years=("2010", "2011"), groups=_GROUPS, rows=rows
        )
        out.insert_pdf(witness_doc)
        witness_doc.close()
    path = tmp_path / "document.pdf"
    out.save(path)
    out.close()
    return path


def _drive_floor(tmp_path, pdf):
    from socr.core.manifest import chart_floor_replacement

    from test_gh1155_chart_witness import _floor_page_state, _floor_state

    png = "![Failed table page 1](figures/failed_table_p1.png)"
    builder = chart_floor_replacement(
        _floor_state(pdf, tmp_path),
        1,
        _floor_page_state(pdf, png),
        marker_line="[page 1 failed: no usable OCR output]",
        png_ref=png,
    )
    return builder() if builder else None


def test_the_real_caller_ships_counts_vouched_for_by_the_documents_own_table(tmp_path):
    """End to end through ``chart_floor_replacement``: the counts AND the mark.

    Asserting both together is what pins the wiring. With ``decisions`` emptied,
    or not passed to either call, or the caption read off the wrong page, the
    counts vanish while everything else about the page is unchanged -- and
    every one of those survived the full suite before this test existed
    (PR #1199 review, blocker 3).
    """
    body = _drive_floor(tmp_path, _chart_and_table_document(tmp_path))
    assert body, "the page ships a chart body"
    assert "change in real GDP" in body and "2010" in body
    assert f"-- {_SOLID}:" in body, "the series the table vouches for ships its counts"
    assert "| 3 |" in body.replace("  ", " "), "and they are the numbers the reader read"
    # The counts may never appear without the sentence saying what they are.
    assert "read from the drawing" in body
    assert "lowest and highest" in body


def test_the_real_caller_ships_NOTHING_when_the_document_has_no_table(tmp_path):
    """The difference the witness makes, on the same chart page.

    Only the second page changes, so anything that still ships here was not
    vouched for by anything.
    """
    with_table = _drive_floor(tmp_path, _chart_and_table_document(tmp_path))
    without = (
        _drive_floor(tmp_path, _chart_and_table_document(tmp_path / "b", table=False))
        if (tmp_path / "b").mkdir() is None
        else None
    )
    assert with_table and f"-- {_SOLID}:" in with_table
    assert without, "the page still ships its chart body"
    assert f"-- {_SOLID}:" not in without, "but no counts, because nothing vouches for them"
    assert "no projection table" in without


def test_the_real_caller_ships_nothing_when_the_caption_names_no_variable(tmp_path):
    """The caption is read off THIS page; a panel bound to the wrong variable
    would be checked against another variable's published range."""
    body = _drive_floor(
        tmp_path, _chart_and_table_document(tmp_path, caption="Number of participants")
    )
    assert body
    assert f"-- {_SOLID}:" not in body
    assert "names no variable" in body


def test_a_chart_page_with_no_floor_ships_the_marker_never_its_axis_ticks(tmp_path):
    """Grok's P1, pinned as the invariant rather than argued about.

    Grok re-selected 2008-10-29 p16 with only its native winner and got the
    chart's raw axis ticks -- "2\\n4\\n6\\n8" -- loose numbers from inside a
    chart, which is the #1155 P0. Every branch of ``structure_class_floor_text``
    that cannot use the chart floor returns the whole-page MARKER instead, so
    the question is whether that holds when the floor is unavailable for each
    of the reasons it can be.

    Checked here for all three: no builder at all (the selector's
    ``text_will_be_overridden`` arm passes ``chart_floor=None``), a builder that
    declines, and a builder that raises. None of them may let the page's own
    numerals through. This does not prove no other path exists; it does close
    the one the finding names.
    """
    from socr.core.manifest import structure_class_floor_text

    from test_gh1155_chart_witness import _floor_page_state

    pdf = _chart_and_table_document(tmp_path)
    page = _floor_page_state(pdf, "![img](figures/p1.png)")

    def raises():
        raise RuntimeError("builder failed")

    for name, floor in (("none", None), ("declines", lambda: None), ("raises", raises)):
        try:
            body = structure_class_floor_text(page, 1, chart_floor=floor)
        except RuntimeError:
            # A raising builder must not be caught HERE and turned into text;
            # the caller logs it and falls back. Nothing shipped, so nothing leaked.
            continue
        assert "page 1 failed" in body or "unverifiable" in body, (name, body[:120])
        ticks = [line for line in body.splitlines() if line.strip() in {"2", "4", "6", "8", "10"}]
        assert not ticks, f"{name}: axis ticks reached the page: {ticks}"
        for label in ("1.0", "2.0", "3.0", "4.0", "5.0"):
            assert f"\\n{label}\\n" not in body, f"{name}: a bin label reached the page"


def test_a_check_that_could_not_run_is_UNKNOWN_not_FAIL():
    """#1145's distinction, applied to the counts (PR #1199 review).

    FAIL means this panel's counts were checked and refused. A document with no
    projection table, a caption naming nothing, a year the table does not
    publish and a reading with unread bars are all cases where the check could
    not run, and recording them FAIL says the page was judged when it was not.
    """
    panel = _Panel("2015", (_series("September projections", _COMPLETE),))
    holed = _Panel(
        "2015",
        (
            _series(
                "September projections",
                (("1.6-1.7", 0), ("1.8-1.9", 1), ("2.0-2.1", None), ("2.2-2.3", 5), ("2.4-2.5", 3)),
            ),
        ),
    )
    shifted = _Panel(
        "2015",
        (
            _series(
                "September projections",
                (("1.6-1.7", 0), ("1.8-1.9", 0), ("2.0-2.1", 8), ("2.2-2.3", 5), ("2.4-2.5", 3)),
            ),
        ),
    )
    cases = [
        (_Reading(panels={1: panel}), None, "change in real GDP", CheckResult.UNKNOWN),
        (_Reading(panels={1: panel}), _WITNESS, "", CheckResult.UNKNOWN),
        (_Reading(panels={1: holed}), _WITNESS, "change in real GDP", CheckResult.UNKNOWN),
        (_Reading(panels={1: shifted}), _WITNESS, "change in real GDP", CheckResult.FAIL),
    ]
    for reading, witness, variable, expected in cases:
        decisions = {
            index: decide_panel(p2, witness, variable, region_index=index, region_count=1)
            for index, p2 in reading.panels.items()
        }
        records = region_clause_records(
            4,
            region_pictures={1: "![r](a.png)"},
            reading=reading,
            region_count=1,
            decisions=decisions,
        )
        assert records[0]["result"] == expected.value, (variable, witness, records[0]["reason"])
