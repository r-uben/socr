"""#1155: the projection-table witness and the gate it feeds.

Hermetic by construction: every table is drawn here with PyMuPDF, and the one
set of real measurements (the 2015-09-17 bin/range pairs) is carried as literal
fixture data with its provenance stated, never fetched. No provider, no
network, no Ollama -- nothing in this file touches the agentic path.
"""

from __future__ import annotations

import fitz
import pytest

from socr.core.marks import is_canonical_mark
from socr.figures.chart_witness import (
    GATE_NO_FILLED_BINS,
    WITHHELD_RUNS_REASON,
    clause_mark,
    GATE_OUT_OF_RANGE,
    GATE_UNPARSEABLE_BIN,
    LAYOUT_UPRIGHT_SINGLE_BLOCK,
    REFUSED_ROTATED,
    REFUSED_SPLIT_BLOCKS,
    SHIPPED_REASON,
    bin_interval,
    gate_series,
    panel_mark_line,
    parse_table1_witness,
)

# --------------------------------------------------------------------------------------
# Table fixtures. Drawn, not sampled: the parser must read a LAYOUT, so the fixture has
# to be a real page whose geometry the parser walks.
# --------------------------------------------------------------------------------------

#: Column header the fixtures print. Two groups (central tendency, range), as the
#: 2008-2011 releases print; the parser reads the count from this header, never assumes it.
_YEARS = ("2010", "2011", "2012")


def _upright_table(path, rows, *, years=_YEARS, median=False):
    """One upright page: a title, a year header, then one line per row."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 60), "Table 1. Economic projections of Federal Reserve Board", fontsize=9)
    header = "Variable " + (" ".join(years) + " ") * (3 if median else 2)
    if median:
        header = (
            "Variable Median "
            + " ".join(years)
            + " Central tendency "
            + " ".join(years)
            + " Range "
            + " ".join(years)
        )
    page.insert_text((60, 80), header, fontsize=9)
    y = 100
    for line in rows:
        page.insert_text((60, y), line, fontsize=9)
        y += 20
    doc.save(path)
    doc.close()
    return fitz.open(path)


def _rotated_table(path):
    """The same table drawn turned through 90 degrees inside an upright page."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (60, 700), "Table 1. Economic projections of Federal Reserve Board", fontsize=9, rotate=90
    )
    page.insert_text((80, 700), "Variable 2010 2011 2012 2010 2011 2012", fontsize=9, rotate=90)
    page.insert_text(
        (100, 700),
        "Change in real GDP 2.7 to 4.0 3.0 to 4.6 2.8 to 5.0 2.4 to 3.0 2.5 to 4.1 2.6 to 5.1",
        fontsize=9,
        rotate=90,
    )
    doc.save(path)
    doc.close()
    return fitz.open(path)


# --------------------------------------------------------------------------------------
# The witness must parse what it claims to parse
# --------------------------------------------------------------------------------------


def test_upright_single_block_parses_every_year_of_every_row(tmp_path):
    """One row per series, label then one cell per column, group by group."""
    pdf = tmp_path / "upright.pdf"
    doc = _upright_table(
        pdf,
        [
            "Change in real GDP 3.2 to 3.7 3.4 to 4.5 3.5 to 4.5 2.7 to 4.0 3.0 to 4.6 2.8 to 5.0",
            "January projection 2.8 to 3.5 3.4 to 4.5 3.5 to 4.5 2.3 to 4.0 2.7 to 4.7 3.0 to 5.0",
        ],
    )
    witness = parse_table1_witness(doc[0])
    doc.close()

    assert witness.refusal == ""
    assert witness.layout == LAYOUT_UPRIGHT_SINGLE_BLOCK
    # The RANGE group is the witness, not the central tendency.
    assert witness.range_for("Change in real GDP", "current", "2010") == (2.7, 4.0)
    assert witness.range_for("Change in real GDP", "current", "2012") == (2.8, 5.0)
    assert witness.range_for("Change in real GDP", "prior", "2011") == (2.7, 4.7)
    assert witness.prior_months == frozenset({"January"})


def test_a_collapsed_range_is_read_as_a_degenerate_interval(tmp_path):
    """A release prints a collapsed range as one figure. Dropping it loses a variable."""
    pdf = tmp_path / "collapsed.pdf"
    doc = _upright_table(pdf, ["PCE inflation 1.2 to 1.6 1.0 to 2.0 2.0 1.1 to 2.0 0.9 to 2.4 2.0"])
    witness = parse_table1_witness(doc[0])
    doc.close()
    assert witness.range_for("PCE in", "current", "2012") == (2.0, 2.0)


def test_a_row_whose_nesting_breaks_is_not_a_witness(tmp_path):
    """The witness must prove itself: central tendency outside its own range."""
    pdf = tmp_path / "broken.pdf"
    doc = _upright_table(
        pdf,
        [  # 2010's central tendency 3.2-3.7 is NOT inside the range 1.0-1.5
            "Change in real GDP 3.2 to 3.7 3.4 to 4.5 3.5 to 4.5 1.0 to 1.5 3.0 to 4.6 2.8 to 5.0"
        ],
    )
    witness = parse_table1_witness(doc[0])
    doc.close()
    assert witness.rows == {}
    assert any("not inside the range" in reason for _, _, reason in witness.rejected)


def test_a_row_with_the_wrong_cell_count_is_not_a_witness(tmp_path):
    """The header declares the column count; a row that disagrees is not trusted."""
    pdf = tmp_path / "short.pdf"
    doc = _upright_table(pdf, ["Unemployment rate 9.1 to 9.5 8.1 to 8.5 6.6 to 7.5"])
    witness = parse_table1_witness(doc[0])
    doc.close()
    assert witness.rows == {}


def test_the_rotated_layout_refuses_by_name(tmp_path):
    """A layout this parser does not read must refuse, never guess."""
    pdf = tmp_path / "rotated.pdf"
    doc = _rotated_table(pdf)
    witness = parse_table1_witness(doc[0])
    doc.close()
    assert witness.rows == {}
    assert witness.refusal == REFUSED_ROTATED


def test_the_split_block_layout_refuses_by_name(tmp_path):
    """Every data row short of the header's count is a stacked-block table."""
    pdf = tmp_path / "split.pdf"
    doc = _upright_table(
        pdf,
        [  # each row carries ONE group: the central-tendency block only
            "Change in real GDP 0.0 to 0.3 -0.2 to 1.1 2.3 to 3.2",
            "Unemployment rate 6.3 to 6.5 7.1 to 7.6 6.5 to 7.3",
        ],
    )
    witness = parse_table1_witness(doc[0])
    doc.close()
    assert witness.rows == {}
    assert witness.refusal == REFUSED_SPLIT_BLOCKS


def test_a_series_is_the_prior_meeting_only_when_its_own_month_matches(tmp_path):
    """Every series here is month-prefixed, so "has a month" classes them all prior."""
    pdf = tmp_path / "months.pdf"
    doc = _upright_table(
        pdf,
        [
            "Change in real GDP 3.2 to 3.7 3.4 to 4.5 3.5 to 4.5 2.7 to 4.0 3.0 to 4.6 2.8 to 5.0",
            "January projection 2.8 to 3.5 3.4 to 4.5 3.5 to 4.5 2.3 to 4.0 2.7 to 4.7 3.0 to 5.0",
        ],
    )
    witness = parse_table1_witness(doc[0])
    doc.close()
    assert witness.series_row_for("January projections") == "prior"
    assert witness.series_row_for("April projections") == "current"


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------

#: Measured on the 2015-09-17 FOMC minutes, Figures 3.A and 3.B, by running this tree's
#: own ``read_chart_page`` and hand-parsing that release's Table 1 (which this parser
#: refuses -- it is drawn rotated, see #1184). Each entry is
#: ``(lowest filled bin, highest filled bin, published range)`` for one CURRENT-meeting
#: series. Carried as literal data because the PDF is not in the repository and CI has no
#: network; the provenance is this docstring and `leads/1155/evidence.md` section N.
_2015_CURRENT_SERIES = (
    ("1.8-1.9", "2.4-2.5", (1.9, 2.5)),
    ("2.0-2.1", "2.8-2.9", (2.1, 2.8)),
    ("1.8-1.9", "2.6-2.7", (1.9, 2.6)),
    ("1.6-1.7", "2.4-2.5", (1.6, 2.4)),
    ("1.8-1.9", "2.6-2.7", (1.8, 2.7)),
    ("4.8-4.9", "5.2-5.3", (4.9, 5.2)),
    ("4.4-4.5", "5.0-5.1", (4.5, 5.0)),
    ("4.4-4.5", "5.0-5.1", (4.5, 5.0)),
    ("4.6-4.7", "5.2-5.3", (4.6, 5.3)),
    ("4.6-4.7", "5.8-5.9", (4.7, 5.8)),
)


def test_the_gate_passes_every_measured_2015_current_series():
    """10 of 10. If this drops, the gate got stricter than the data it was built on."""
    passed = [gate_series([lo, hi], rng)[0] for lo, hi, rng in _2015_CURRENT_SERIES]
    assert passed == [True] * len(_2015_CURRENT_SERIES)


@pytest.mark.parametrize("low,high,rng", _2015_CURRENT_SERIES)
def test_a_shift_that_moves_an_extreme_outside_its_bin_fails(low, high, rng):
    """The protection: a reading shifted one bin off its endpoint is refused.

    The shift is applied to the BIN LABELS, as a misread axis would produce: both
    extremes move by one bin width, so at least one no longer contains its endpoint.
    """
    width = round(bin_interval(low)[1] - bin_interval(low)[0], 10)
    step = width + 0.1  # one whole bin further on, as the next label would read

    def shifted(label):
        lo, hi = bin_interval(label)
        return f"{lo + step:.1f}-{hi + step:.1f}"

    ok, reason = gate_series([shifted(low), shifted(high)], rng)
    assert ok is False
    assert reason == GATE_OUT_OF_RANGE


def test_a_shift_that_stays_inside_both_bins_survives_the_gate():
    """Recorded honestly: this gate narrows what ships, it does not prove a reading.

    The published range's endpoints still fall inside the first and last filled bins, so
    nothing here can tell the two readings apart. Pinning it means the limitation cannot
    be quietly lost, and it is stated in the module docstring and in the shipped mark.
    """
    true_reading = ["1.8-1.9", "2.0-2.1", "2.2-2.3", "2.4-2.5"]
    ok, _ = gate_series(true_reading, (1.9, 2.5))
    assert ok is True, "sanity: the true reading passes"

    # The gate inspects the FIRST and LAST filled bins and nothing between them, so a
    # reading that gets every interior bin wrong -- a swap, a count off by one, two bins
    # transposed -- is invisible to it as long as the extremes are unchanged.
    interior_wrong = ["1.8-1.9", "2.2-2.3", "2.0-2.1", "2.4-2.5"]
    assert interior_wrong != true_reading
    also_ok, _ = gate_series(interior_wrong, (1.9, 2.5))
    assert also_ok is True, (
        "the gate checks only the two extremes, so any interior error survives it; it "
        "narrows what can ship and never claims a reading is correct"
    )

    # And an extreme that moves but stays inside its own bin is equally invisible.
    inside_bin_shift = ["1.85-1.95", "2.45-2.55"]
    assert gate_series(inside_bin_shift, (1.9, 2.5))[0] is True


def test_a_reading_truncated_at_the_TOP_is_refused():
    """Only the low end was exercised before: a mutant that deleted the high-end check
    killed no test. A truncated reading -- the reader missed the topmost bar, so the
    lowest bin is right and the highest is short of the published maximum -- is the case
    that needs it, and it is a real failure shape, not a contrived one.
    """
    ok, reason = gate_series(["1.8-1.9", "2.0-2.1", "2.2-2.3"], (1.9, 2.5))
    assert ok is False
    assert reason == GATE_OUT_OF_RANGE


def test_a_reading_truncated_at_the_BOTTOM_is_refused():
    """The mirror case, so neither endpoint's check can be deleted unnoticed."""
    ok, reason = gate_series(["2.0-2.1", "2.2-2.3", "2.4-2.5"], (1.9, 2.5))
    assert ok is False
    assert reason == GATE_OUT_OF_RANGE


def test_a_series_with_no_filled_bin_is_refused_not_passed():
    ok, reason = gate_series([], (1.9, 2.5))
    assert ok is False
    assert reason == GATE_NO_FILLED_BINS


def test_an_unreadable_bin_label_is_refused_not_guessed():
    ok, reason = gate_series(["UNRESOLVED", "UNRESOLVED"], (1.9, 2.5))
    assert ok is False
    assert reason == GATE_UNPARSEABLE_BIN


def test_a_bin_labels_separator_is_a_hyphen_not_a_minus_sign():
    """The bug this cost a whole measurement run to find."""
    assert bin_interval("1.8-1.9") == (1.8, 1.9)


# --------------------------------------------------------------------------------------
# The visible mark
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"shipped": True}, SHIPPED_REASON),
        ({"shipped": False, "decided": False}, "no independent witness; values not shipped"),
        ({"shipped": False}, "the filled bins do not span the range; values not shipped"),
    ],
)
def test_every_panel_line_is_a_canonical_mark(kwargs, reason):
    line = panel_mark_line(17, 3, 5, reason, **kwargs)
    assert is_canonical_mark(line, 17)
    assert "region 3 of 5" in line


def test_a_shipped_panel_says_the_numbers_are_not_verified():
    """The owner's standing rule: a mark never claims a value was verified (#1181)."""
    line = panel_mark_line(17, 1, 5, SHIPPED_REASON, shipped=True)
    assert "not verified" in line
    assert "lowest and highest filled bars" in line
    assert "=UNKNOWN;" in line, "a shipped-but-unverified reading is UNKNOWN, never PASS"


def test_a_decided_failure_is_FAIL_and_an_undecidable_one_is_UNKNOWN():
    """#1145 draws the distinction; losing it would say socr never looked."""
    decided = panel_mark_line(17, 2, 5, "the bins do not span the range", shipped=False)
    undecided = panel_mark_line(17, 2, 5, "no witness", shipped=False, decided=False)
    assert "=FAIL;" in decided
    assert "=UNKNOWN;" in undecided


# --------------------------------------------------------------------------------------
# The floor replacement. Built on a page shaped like production: a chart region whose
# INTERIOR carries numeric tick labels, and prose outside it. The first version of this
# lane was tested with a hand-written prose-only string, which is why it shipped the
# chart's own numbers in production and the tests did not notice (#1155 PR 1 review).
# --------------------------------------------------------------------------------------


def _chart_page_with_prose(tmp_path):
    """A readable chart, numeric tick labels INSIDE it, and real prose OUTSIDE it.

    The prose is drawn into the PDF, not merely put on the fixture object: the
    partitioner works on the page's own words, so a page that is ALL chart recovers
    nothing and any "no numerals leaked" assertion over it passes vacuously. That is
    what the first version of this fixture did, and a mutant removing the chart boxes
    killed no test.
    """
    from test_gh635_chart_reader import build_chart

    pdf = tmp_path / "floor_chart.pdf"
    doc, bboxes = build_chart(pdf, [2, 0, 0, 0, 0], [0, 2, 0, 0, 0])
    # Well below the plot, so it falls outside every chart region.
    doc[0].insert_text((72, 760), "A sentence of ordinary prose carrying no figures", fontsize=10)
    doc.save(tmp_path / "with_prose.pdf")
    doc.close()
    return tmp_path / "with_prose.pdf", bboxes


def _page_state(pdf, prose_line):
    """A PageState-like page carrying the real native words of *pdf*, plus prose."""
    doc = fitz.open(pdf)
    page = doc[0]
    words = list(page.get_text("words"))
    native = " ".join(w[4] for w in words)

    class _P:
        native_words = words
        native_text = prose_line + "\n" + native
        detected_table_count = 0
        detected_table_bboxes = ()
        native_table_region_count = 0
        d3_floor_png_ref = ""
        page_rotation = 0

    return doc, _P()


def test_no_numeral_drawn_inside_a_chart_region_reaches_the_page(tmp_path):
    """The P0 this round fixes, pinned on the geometry rather than on the text.

    Every numeric tick the chart draws lies INSIDE a chart region, so once those
    regions join the withheld boxes no band containing one can ship. Previously the
    whole native layer shipped verbatim: 149 loose numeric lines on 2010-04-28 p15.
    """
    import re

    from socr.core.manifest import native_prose_floor_text
    from socr.tables.reconstruct import chart_region_bboxes

    pdf, _ = _chart_page_with_prose(tmp_path)
    doc, p = _page_state(pdf, "")
    boxes = list(chart_region_bboxes(doc[0]))
    assert boxes, "premise: the page has a chart region"

    def numerals_inside_regions(text):
        """Numerals the page draws INSIDE a chart region, found in *text*."""
        inside = set()
        for x0, y0, x1, y1, word, *_ in doc[0].get_text("words"):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            for b in boxes:
                if b.x0 <= cx <= b.x1 and b.y0 <= cy <= b.y1 and re.search(r"\d", word):
                    inside.add(word)
        return {w for w in inside if re.search(rf"(?<!\S){re.escape(w)}(?!\S)", text or "")}

    # Control: the unwithheld native layer DOES carry them, so the assertion below is
    # inspecting a real difference and not an empty set.
    assert numerals_inside_regions(p.native_text), "control: the raw layer leaks them"

    recovered = native_prose_floor_text(
        p, 1, marker_line="[marker]", png_ref="", extra_withheld_boxes=boxes
    )
    leaked = numerals_inside_regions(recovered or "")
    leaked_raw = numerals_inside_regions(p.native_text)
    doc.close()

    # NOT vacuous, both halves asserted:
    #  - the lane really did recover this page's prose, so there is a body to inspect;
    assert recovered and "ordinary prose carrying no figures" in recovered
    #  - and the raw native layer, which the first version of this lane shipped
    #    verbatim, really does carry the in-chart numerals.
    assert leaked_raw, "control: the raw native layer carries in-chart numerals"
    assert leaked == set(), "no numeral drawn inside a chart region may ship as loose text"


def test_the_replacement_carries_a_line_per_region_and_the_recovered_prose(tmp_path):
    from socr.core.marks import is_canonical_mark
    from socr.figures.chart_reader import read_chart_page
    from socr.figures.chart_witness import floor_replacement_body
    from socr.tables.reconstruct import chart_region_bboxes

    pdf, _ = _chart_page_with_prose(tmp_path)
    doc = fitz.open(pdf)
    boxes = list(chart_region_bboxes(doc[0]))
    reading = read_chart_page(doc[0], boxes, page_num=1)
    body = floor_replacement_body(
        None,
        1,
        "Recovered prose outside every region",
        region_pictures={
            i: f"![chart region {i}](figures/chart_region_p1_{i}.png)"
            for i in range(1, len(boxes) + 1)
        },
        reading=reading,
        region_count=len(boxes),
    )
    doc.close()
    lines = [ln for ln in body.splitlines() if ln.startswith("> [page 1 unchecked:")]
    assert len(lines) == len(boxes)
    assert all(is_canonical_mark(ln, 1) for ln in lines)
    assert "Recovered prose outside every region" in body


def test_a_picture_the_prose_already_references_is_not_appended_again(tmp_path):
    """Argon 3: the native layer embeds its own region placeholders; appending
    unconditionally emitted ten image links for five regions."""
    from socr.figures.chart_reader import read_chart_page
    from socr.figures.chart_witness import floor_replacement_body
    from socr.tables.reconstruct import chart_region_bboxes

    pdf, _ = _chart_page_with_prose(tmp_path)
    doc = fitz.open(pdf)
    boxes = list(chart_region_bboxes(doc[0]))
    reading = read_chart_page(doc[0], boxes, page_num=1)
    ref = "![chart region 1](figures/chart_region_p1_1.png)"
    body = floor_replacement_body(
        None,
        1,
        f"Prose before\n\n{ref}\n\nProse after",
        region_pictures={1: ref},
        reading=reading,
        region_count=len(boxes),
    )
    doc.close()
    assert body.count(ref) == 1, "a reference already in the prose must not be duplicated"


def test_a_region_with_no_picture_says_so_and_keeps_the_prose(tmp_path):
    from socr.figures.chart_reader import read_chart_page
    from socr.figures.chart_witness import floor_replacement_body
    from socr.tables.reconstruct import chart_region_bboxes

    pdf, _ = _chart_page_with_prose(tmp_path)
    doc = fitz.open(pdf)
    boxes = list(chart_region_bboxes(doc[0]))
    reading = read_chart_page(doc[0], boxes, page_num=1)
    body = floor_replacement_body(
        None,
        1,
        "Prose survives",
        region_pictures={"error:1": "disk full"},
        reading=reading,
        region_count=len(boxes),
    )
    doc.close()
    assert "Prose survives" in body
    assert "picture not rendered" in body
    assert "![chart region" not in body


def test_a_reading_that_resolves_nothing_leaves_the_floor_alone(tmp_path):
    from socr.figures.chart_witness import floor_replacement_body

    class _Empty:
        panels: dict = {}
        refusals: dict = {}

    assert (
        floor_replacement_body(
            None, 1, "prose", region_pictures={}, reading=_Empty(), region_count=3
        )
        is None
    )


def test_a_page_with_regions_but_nothing_resolved_is_left_alone(tmp_path):
    """Regions EXIST, but the reader refuses them, so the floor keeps its own behaviour."""
    from socr.core.manifest import chart_floor_replacement
    from socr.tables.reconstruct import chart_region_bboxes

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 40), "Some prose above the shapes", fontsize=11)
    red = (0.9, 0.1, 0.1)
    for i, x in enumerate((100, 180, 260, 340, 420)):
        page.draw_rect(fitz.Rect(x, 500 - i * 40, x + 60, 680), color=red, fill=red)
    pdf = tmp_path / "unreadable.pdf"
    doc.save(pdf)
    doc.close()
    reopened = fitz.open(pdf)
    assert chart_region_bboxes(reopened[0]), "premise: this page HAS a chart region"
    reopened.close()

    class _State:
        class handle:
            path = pdf

        figures_dir = tmp_path

    builder = chart_floor_replacement(_State(), 1, object(), marker_line="[marker]", png_ref="")
    assert builder is None, "a page the reader cannot read must offer no replacement"


def test_the_floor_returns_the_chart_body_instead_of_the_bare_marker():
    """The wiring a mutant found untested: the floor must actually USE the replacement.

    Pins the DIFFERENCE, not a literal: the same page, the same floor call, changing
    only whether a chart replacement is offered.
    """
    from socr.core.manifest import structure_class_floor_text

    class _Page:
        native_text = "prose that the bare marker used to destroy"
        detected_table_count = 0
        detected_table_bboxes = ()
        native_table_region_count = 0
        d3_floor_png_ref = ""
        page_rotation = 0

    page = _Page()
    # The body must follow the page's own reading order: #1144 round 5's gate applies
    # to this lane too, so a body that does not is refused -- which is why this is the
    # page's own prose with a mark in front of it rather than an arbitrary string.
    body = "> [chart marks]\n\n" + _Page.native_text
    without = structure_class_floor_text(page, 7)
    with_chart = structure_class_floor_text(page, 7, chart_floor=lambda: body)

    assert with_chart == body
    assert with_chart != without
    assert "failed: unverifiable table" in without, (
        "premise: without a replacement this page still ships the bare marker"
    )


def test_a_rotated_page_keeps_its_bare_marker_even_with_a_chart_body():
    """Scope limit: the rotated-lane refusal is NOT the branch #1155 replaces.

    A rotated page's recovered "prose" is confetti (GH-147), so the bare marker stays
    the honest floor there however readable its chart looks.
    """
    from socr.core.manifest import structure_class_floor_text

    class _Rotated:
        native_text = "confetti / order / not / reading"
        detected_table_count = 0
        detected_table_bboxes = ()
        native_table_region_count = 0
        d3_floor_png_ref = ""
        page_rotation = 0
        native_table_lane_refused_rotated = True

    out = structure_class_floor_text(
        _Rotated(), 9, chart_floor=lambda: "> [chart marks]\n\n" + _Rotated.native_text
    )
    assert "[chart marks]" not in out
    assert "failed: unverifiable table" in out


def test_selecting_the_replacement_marks_the_page_for_the_preservation_pass():
    """Argon 3: the generic chart-preservation pass must skip a chart-floor body.

    It anchors crops on in-chart words this lane withholds, so it cannot place them
    and would inject an "Unresolved chart placement" banner plus a contradictory
    ``chart_region=FAIL`` mark beside the per-region marks already on the page.
    """
    from socr.core.manifest import structure_class_floor_text

    class _Page:
        native_text = "prose that the bare marker used to destroy"
        native_words = ()
        detected_table_count = 0
        detected_table_bboxes = ()
        native_table_region_count = 0
        d3_floor_png_ref = ""
        page_rotation = 0
        chart_floor_body = False

    page = _Page()
    body = "> [chart marks]\n\n" + _Page.native_text
    structure_class_floor_text(page, 7, chart_floor=lambda: body)
    assert page.chart_floor_body is True

    untouched = _Page()
    structure_class_floor_text(untouched, 7)
    assert untouched.chart_floor_body is False, "a page that floors normally is not flagged"


def test_the_builder_is_not_called_when_the_page_does_not_reach_that_branch():
    """Astra 1 / Argon 1: no crop may be rendered for a page taking another branch.

    The builder is the only thing that renders, so "not called" is the property that
    makes eager rendering impossible. A rotated page is the case that reproduced it.
    """
    from socr.core.manifest import structure_class_floor_text

    calls = []

    def _builder():
        calls.append(1)
        return "SHOULD NEVER BE BUILT"

    class _Rotated:
        native_text = "some text"
        native_words = ()
        detected_table_count = 0
        detected_table_bboxes = ()
        native_table_region_count = 0
        d3_floor_png_ref = ""
        page_rotation = 90

    structure_class_floor_text(_Rotated(), 3, chart_floor=_builder)
    assert calls == [], "the builder must not run for a page the lane refuses"


def test_prose_that_breaks_the_pages_reading_order_is_dropped_not_shipped():
    """#1144 round 5's gate applies to this lane, on the prose and before any mark."""
    from socr.core.manifest import _prose_follows_native_order

    native = "alpha beta gamma delta"
    assert _prose_follows_native_order("alpha beta", native) is True
    assert _prose_follows_native_order("delta alpha", native) is False


def test_words_drawn_inside_a_chart_region_are_withheld_by_geometry(tmp_path):
    """The chart-box rule, SEEN to guard: in-region text that READS as prose.

    The numeral-count rule withholds a band by how many figures it carries, so a
    chart's own legend names and panel heading -- words, no figures -- are judged
    prose and ship as loose text from inside the plot. Only the geometry rule
    withholds them.

    Recorded honestly: on the 12 real Fed SEP pages this lane reaches today, the
    chart boxes change nothing -- the numeral-count rule already withholds every
    band there, and the in-chart numerals go to 0 either way. This rule is kept
    because withholding by the chart's own geometry is positive evidence, where
    the numeral count is a deny-list over numeric shapes, and the record on this
    repo is that deny-list lanes leak in later rounds. The page below is what it
    guards; no corpus page has been seen to need it yet.
    """
    import re

    from socr.core.manifest import native_prose_floor_text
    from socr.tables.reconstruct import chart_region_bboxes

    pdf, _ = _chart_page_with_prose(tmp_path)
    doc, p = _page_state(pdf, "")
    boxes = list(chart_region_bboxes(doc[0]))

    in_region_words = {
        word
        for x0, y0, x1, y1, word, *_ in doc[0].get_text("words")
        if any(b.x0 <= (x0 + x1) / 2 <= b.x1 and b.y0 <= (y0 + y1) / 2 <= b.y1 for b in boxes)
        and any(c.isalpha() for c in word)
        and len(word) > 1  # a single letter matches inside other words
    }

    def leaked(text):
        return {
            w for w in in_region_words if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text or "")
        }

    with_rule = native_prose_floor_text(
        p, 1, marker_line="[m]", png_ref="", extra_withheld_boxes=boxes
    )
    without_rule = native_prose_floor_text(
        p, 1, marker_line="[m]", png_ref="", extra_withheld_boxes=None
    )
    leaked_with, leaked_without = leaked(with_rule), leaked(without_rule)
    doc.close()

    assert in_region_words, "premise: the chart draws words inside its own region"
    # The numeral-count rule alone ships the panel heading and a legend name.
    assert leaked_without, "control: without the geometry rule, in-region words ship as loose prose"
    assert leaked_with == set(), "no word drawn inside a chart region may ship as loose text"
    # And the page's real prose, outside every region, still ships.
    assert with_rule and "ordinary prose carrying no figures" in with_rule


def test_the_renderer_runs_once_across_repeated_finalizations(tmp_path):
    """Astra 3: selection runs again on a re-finalize and on a resume -- and even
    when a ``saved_text`` override is about to discard its result -- so the crops
    must be rendered once, not once per pass."""
    from socr.core.manifest import chart_floor_replacement

    pdf, _ = _chart_page_with_prose(tmp_path)
    doc, p = _page_state(pdf, "")
    doc.close()
    p.chart_floor_body = False
    p.chart_floor_pictures = None

    class _State:
        class handle:
            path = pdf

        figures_dir = tmp_path / "figures"

    calls = []
    import socr.figures.chart_regions as regions

    real = regions.render_chart_region_crops

    def _counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    regions.render_chart_region_crops = _counting
    try:
        for _ in range(3):
            builder = chart_floor_replacement(_State(), 1, p, marker_line="[m]", png_ref="")
            assert builder is not None
            builder()
    finally:
        regions.render_chart_region_crops = real

    assert calls == [1], f"the renderer must run once, ran {len(calls)} times"


def test_a_partial_table_inventory_does_not_reach_the_chart_floor():
    """Round 2 (Astra P1, Argon 2): a missing table BOX is not an absence of tables.

    ``detected_table_count=2`` with one box means the detector found two tables and
    could box only one. The chart regions cannot protect the table nobody boxed, and
    an unboxed row of word cells ships as loose prose. Both reviewers reproduced it
    on constructed states; all 12 real Fed pages carry count 0 and no boxes.
    """
    from socr.core.manifest import structure_class_floor_text

    class _Partial:
        native_text = "prose and an unboxed row East West North"
        native_words = ()
        detected_table_count = 2
        detected_table_bboxes = ((0.0, 0.0, 10.0, 10.0),)
        native_table_region_count = 0
        d3_floor_png_ref = ""
        page_rotation = 0
        chart_floor_body = False

    calls = []
    out = structure_class_floor_text(
        _Partial(), 5, chart_floor=lambda: calls.append(1) or "CHART BODY"
    )
    assert calls == [], "the chart floor must not be entered on a partial inventory"
    assert "CHART BODY" not in out


def test_repeated_withheld_run_markers_collapse_to_one_page_scoped_line():
    """Argon 4: the prose lane marks EVERY withheld run, so a Fed SEP page carried
    3-6 identical "[page N failed: ...]" blocks and a picture link each time."""
    from socr.figures.chart_witness import collapse_withheld_runs

    marker = "[page 15 failed: no usable OCR output]"
    png = "![Failed table page 15](figures/failed_table_p15.png)"
    recovered = (
        f"[banner]\n\n{png}\n\n{marker}\n\n[page image](figures/failed_table_p15.png)\n\n"
        f"Percent range\n\n{marker}\n\n[page image](figures/failed_table_p15.png)\n\n"
        f"Percent range\n\n{marker}\n\n[page image](figures/failed_table_p15.png)"
    )
    prose, record, image = collapse_withheld_runs(recovered, marker, png, 15)
    # Round 4: the middle element is the clause RECORD, and the page's line is rendered
    # from it. Every assertion this test made about the line is made about the rendered
    # record, so the change of contract costs the test nothing.
    mark = clause_mark(15, record) if record else None

    assert record == {
        "check": "native_coverage",
        "result": "FAIL",
        "scope": "page",
        "reason": WITHHELD_RUNS_REASON.format(runs=3),
    }, "the record says what the line says, and is what the verdict will read"
    assert marker not in prose, "no per-run failure marker survives"
    assert "[page image](" not in prose, "no per-run picture link survives"
    assert prose.count("Percent range") == 2, "the surviving prose is untouched"
    assert mark is not None and "native_coverage=FAIL" in mark
    assert "scope=page" in mark, "withheld page text is page-scoped, not a chart region"
    assert "3 run(s)" in mark, "the line says how many runs were withheld"
    assert "#1154" in mark, "and cites the caption issue rather than hiding it"
    assert image == png


def test_the_page_image_is_carried_exactly_once():
    from socr.figures.chart_reader import read_chart_page
    from socr.figures.chart_witness import floor_replacement_body
    from socr.tables.reconstruct import chart_region_bboxes

    import tempfile
    from pathlib import Path as _Path

    tmp = _Path(tempfile.mkdtemp())
    pdf, _ = _chart_page_with_prose(tmp)
    doc = fitz.open(pdf)
    boxes = list(chart_region_bboxes(doc[0]))
    reading = read_chart_page(doc[0], boxes, page_num=1)
    png = "![Failed table page 1](figures/failed_table_p1.png)"
    body = floor_replacement_body(
        None,
        1,
        "Prose",
        region_pictures={1: "![chart region 1](f/c1.png)"},
        reading=reading,
        region_count=len(boxes),
        page_image=png,
    )
    doc.close()
    assert body.count(png) == 1


def _floor_document_state(pdf, *, best_output=None):
    """The GH-317 structure-class floor fixture's state, over *pdf*.

    The one page shape that reaches the no-detected-box branch where this lane is
    offered. ``best_output`` overrides the winner, which is how a RESTORED page is
    represented: a terminal body plus the typed credential the sidecar carries.
    """
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import DocumentHandle, DocumentState

    from test_gh317_structure_class_floor import NATIVE_TEXT_WITH_PROSE

    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.has_tables = True
    ps.native_text = NATIVE_TEXT_WITH_PROSE
    native = PageOutput(
        page_num=1,
        text=NATIVE_TEXT_WITH_PROSE,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    model = PageOutput(
        page_num=1,
        text="The table on this page could not be read reliably.\n",
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
    )
    model.rejection_class = "ambiguous_deferred"
    ps.attempts.extend([native, model])
    ps.best_output = best_output if best_output is not None else native
    return state


def test_a_restored_prose_recovery_keeps_its_frozen_bytes_through_the_floor(tmp_path):
    """Argon round 3: round 2's guard ERASED the page it was added to protect.

    Round 2 made ``chart_floor_replacement`` return ``None`` for a page whose winner is
    an already-finalized prose recovery, to stop it being recomputed from
    ``native_words`` a restored page does not carry. But the caller reads a ``None``
    builder as "this page has no chart body" and ships the bare whole-page marker, so the
    frozen recovery -- its prose, its banner, its crops -- was deleted rather than merely
    recomputed, and ``scanned_prose_recovered`` flipped to False.

    Driven through ``_select_and_finalize_page``, the production entry point, because the
    round-2 test asserted ``builder is None`` on the HELPER and so pinned the defect as
    though it were the contract. #649 round 2's ruling is that the frozen result stands.
    """
    import socr.core.manifest as m
    from socr.core.result import PageOutput, PageStatus

    from test_gh317_structure_class_floor import _born_digital_pdf

    pdf = _born_digital_pdf(tmp_path)
    frozen = (
        m.prose_recovered_flag(1, born_digital=True)
        + "\n\nNote: The midpoints of the target ranges are shown.\n\n"
        + "![chart region 1](figures/chart_region_p1_1.png)"
    )
    restored = PageOutput(
        page_num=1,
        text=frozen,
        status=PageStatus.ERROR,
        engine="native",
        audit_passed=False,
        scanned_prose_recovered=True,
    )
    assert m._is_restored_prose_recovery(
        _floor_document_state(pdf, best_output=restored).pages[1], 1
    ), "premise: this winner IS an already-finalized prose recovery"

    record = m._select_and_finalize_page(_floor_document_state(pdf, best_output=restored), 1)

    assert record.output.text == frozen, "the frozen recovery ships unchanged"
    assert "Note: The midpoints" in record.output.text, "its prose is not erased"
    assert "chart_region_p1_1.png" in record.output.text, "nor its crop"
    assert record.output.scanned_prose_recovered is True, (
        "and the page still records that it IS a recovery"
    )

    # The difference: the same state whose winner is NOT a restored recovery keeps the
    # bare marker, so the arm above is not simply "every page ships its winner".
    plain = m._select_and_finalize_page(_floor_document_state(pdf), 1)
    assert "Note: The midpoints" not in plain.output.text
    assert plain.output.scanned_prose_recovered is not True


def test_a_saved_body_carries_its_own_recovery_facts(tmp_path):
    """Argon round 3: assemble rewrote the sidecar with the facts of a body it discarded.

    ``_phase_assemble`` re-finalizes every page with the fragment it already wrote as
    ``saved_text``. Selection is then told its text will be overridden, the chart floor
    declines, and the structure-class ending derives ``scanned_prose_recovered`` and
    ``audit_notes`` from the bare marker it builds instead -- so the LAST write of the run
    said ``scanned_prose_recovered=False`` and ``audit_notes=[]`` over a body that
    visibly carries the recovery banner.

    Pinned as the difference between the two saved bodies, through
    ``_select_and_finalize_page``: a saved recovery body records the recovery, a saved
    body that is not one does not.
    """
    import socr.core.manifest as m

    from test_gh317_structure_class_floor import _born_digital_pdf

    pdf = _born_digital_pdf(tmp_path)
    saved_recovery = (
        m.prose_recovered_flag(1, born_digital=True)
        + "\n\nNote: The midpoints of the target ranges are shown.\n"
    )

    with_recovery = m._select_and_finalize_page(
        _floor_document_state(pdf), 1, saved_text=saved_recovery
    )
    assert with_recovery.output.text == saved_recovery, "the saved body still has the last word"
    assert with_recovery.output.scanned_prose_recovered is True, (
        "the record states what the shipped bytes are"
    )
    assert m.SCANNED_PROSE_RECOVERED_NOTE in (with_recovery.output.audit_notes or []), (
        "and the note a reader and the corpus see is written too"
    )

    without = m._select_and_finalize_page(
        _floor_document_state(pdf), 1, saved_text="A saved body carrying no banner.\n"
    )
    assert without.output.scanned_prose_recovered is not True, (
        "control: a saved body that is not a recovery is not recorded as one"
    )
    assert m.SCANNED_PROSE_RECOVERED_NOTE not in (without.output.audit_notes or [])

    # No arm for the opposite direction: the helper is additive by design. A mutant that
    # removed a clearing branch killed nothing, because a saved body forces the ending to
    # derive its flag from the bare marker, so a stale True cannot reach that seam here.
    # See ``_recovery_facts_of_saved_body``.


def test_a_saved_body_means_the_chart_floor_is_never_offered(monkeypatch):
    """Argon 3: selection ran its side effects -- reading the PDF and rendering crops
    -- even when a saved body was about to replace its text.

    HONEST SCOPE: this pins the floor's half of the contract (a ``None`` builder means
    nothing is constructed, and a real one means it is), NOT the selector's wiring that
    decides which to pass. That branch lives inside ``_select_page_output_tagged`` and
    needs a full DocumentState to drive; it is covered only by the mutant on the
    production line, and is recorded as a test gap in the report rather than implied
    to be covered here.
    """
    import socr.core.manifest as m

    built = []
    monkeypatch.setattr(
        m, "chart_floor_replacement", lambda *a, **kw: built.append(1) or (lambda: None)
    )

    class _P:
        native_text = "prose"
        native_words = ()
        detected_table_count = 0
        detected_table_bboxes = ()
        native_table_region_count = 0
        d3_floor_png_ref = ""
        page_rotation = 0
        chart_floor_body = False
        structure_class_floor = True

    # Drive the branch directly with each value of the flag, as the selector does.
    for overridden in (False, True):
        built.clear()
        chart_floor = (
            None
            if overridden
            else m.chart_floor_replacement(object(), 9, _P(), marker_line="[m]", png_ref="")
        )
        m.structure_class_floor_text(_P(), 9, chart_floor=chart_floor)
        if overridden:
            assert built == [], "a saved body must mean no builder is constructed"
        else:
            assert built == [1], "control: without the override the builder IS built"


# --------------------------------------------------------------------------------------
# Round 3. Two findings the panel left open after round 2, each driven through the
# production builder on a drawn page rather than through a hand-written body.
# --------------------------------------------------------------------------------------


def _two_column_chart_page(tmp_path):
    """A readable chart plus prose in TWO columns, which the band partition welds.

    The order gate's own case, built rather than asserted: ``get_text()`` keeps the two
    columns apart (PyMuPDF block order), while the recovery lane clusters bands on y
    only, so the prose it recovers interleaves the columns line by line and is not the
    page's reading order. Returns ``(pdf, chart_boxes)``.
    """
    from test_gh635_chart_reader import build_chart

    pdf = tmp_path / "two_column_chart.pdf"
    doc, _ = build_chart(pdf, [2, 0, 0, 0, 0], [0, 2, 0, 0, 0])
    doc[0].insert_textbox(
        fitz.Rect(20, 620, 180, 700),
        "Monetary policy remained accommodative while participants discussed the balance of risks",
        fontsize=9,
    )
    doc[0].insert_textbox(
        fitz.Rect(230, 620, 400, 700),
        "Financial conditions eased somewhat and credit spreads narrowed over the period",
        fontsize=9,
    )
    out = tmp_path / "two_column_saved.pdf"
    doc.save(out)
    doc.close()
    from socr.tables.reconstruct import chart_region_bboxes

    with fitz.open(out) as d:
        return out, list(chart_region_bboxes(d[0]))


def _floor_page_state(pdf, png_ref):
    """A production-shaped page state for *pdf*: its real words and its real text."""
    with fitz.open(pdf) as doc:
        words = list(doc[0].get_text("words"))
        native = doc[0].get_text()

    class _P:
        native_words = words
        native_text = native
        detected_table_count = 0
        detected_table_bboxes = ()
        native_table_region_count = 0
        d3_floor_png_ref = png_ref
        page_rotation = 0

    return _P()


def _floor_state(pdf, figures_dir):
    class _State:
        class handle:
            path = pdf

        figures_dir = None

    _State.figures_dir = figures_dir
    return _State()


def test_an_order_refusal_keeps_the_page_image_and_says_the_text_was_withheld(tmp_path):
    """Astra round 2: the refusal discarded the page's own backup image, silently.

    ``_prose_follows_native_order`` refusing set ``recovered = ""``, and the full-page
    image was embedded in that string -- so the page shipped chart crops, no text, and
    nothing to read the text FROM, beside no statement that any text had been withheld.

    Pinned as a DIFFERENCE between the two prose outcomes on the same lane, not as an
    absolute body: the refusing page and the accepting page are built from the same
    chart, and the test asserts its own premise (that the gate really does refuse the
    one and accept the other) so it cannot pass if the fixture stops reproducing it.
    """
    from socr.core.manifest import (
        _prose_follows_native_order,
        chart_floor_replacement,
        native_prose_floor_text,
    )
    from socr.figures.chart_witness import ORDER_REFUSED_REASON
    from socr.tables.reconstruct import chart_region_bboxes as _chart_region_bboxes

    png = "![Failed table page 1](figures/failed_table_p1.png)"
    marker = "[page 1 failed: no usable OCR output]"

    refusing_pdf, boxes = _two_column_chart_page(tmp_path)
    refusing = _floor_page_state(refusing_pdf, png)
    recovered = native_prose_floor_text(
        refusing, 1, marker_line=marker, png_ref=png, extra_withheld_boxes=boxes
    )
    assert recovered, "premise: the lane recovers prose from this page at all"
    assert not _prose_follows_native_order(recovered, refusing.native_text), (
        "premise: the two columns weld into an order the page does not have"
    )

    body = chart_floor_replacement(
        _floor_state(refusing_pdf, tmp_path), 1, refusing, marker_line=marker, png_ref=png
    )()
    assert body, "the page still ships a chart body"
    assert body.count(png) == 1, "the page's own image survives the refusal, exactly once"
    assert ORDER_REFUSED_REASON.split(";")[0] in body, "and the refusal is stated, not silent"
    assert "native_coverage=FAIL" in body and "scope=page" in body, (
        "withheld page text is a page-scoped coverage failure"
    )
    assert "Monetary policy remained" not in body, "the refused prose does not ship"

    # The control, same lane and same assertions, with prose the gate accepts.
    # NOT ``build_chart``'s own second return value: that is a full-PAGE bbox, which
    # would withhold the prose too and make the control's premise unmeetable.
    accepting_pdf, _ = _chart_page_with_prose(tmp_path)
    with fitz.open(accepting_pdf) as _doc:
        accept_boxes = list(_chart_region_bboxes(_doc[0]))
    accepting = _floor_page_state(accepting_pdf, png)
    accepted = native_prose_floor_text(
        accepting, 1, marker_line=marker, png_ref=png, extra_withheld_boxes=accept_boxes
    )
    assert accepted and _prose_follows_native_order(accepted, accepting.native_text), (
        "premise: the one-column page's recovered prose IS in the page's order"
    )
    control = chart_floor_replacement(
        _floor_state(accepting_pdf, tmp_path), 1, accepting, marker_line=marker, png_ref=png
    )()
    assert "A sentence of ordinary prose" in control, "control: accepted prose ships"
    assert ORDER_REFUSED_REASON.split(";")[0] not in control, (
        "control: an accepted page carries no refusal line"
    )


def test_every_region_the_page_marks_is_also_a_clause_in_its_verdict(tmp_path):
    """Both seats, round 2: the 12 verdicts carried ZERO chart_region clauses.

    The page's Markdown showed a mark per region while ``sidecar["verdict"]["clauses"]``
    had none of them, so nothing downstream could learn per region what this page did
    not check without parsing mark strings back out of the text. The clauses are now
    derived from the region RESULTS, and this drives the real finalize seam
    (``_apply_verdict_marks``) over the real body to pin three things at once: the
    clauses exist, they agree line-for-line with the marks already on the page, and the
    page's bytes do not change -- a clause whose line is already there must not be
    written twice.
    """
    from socr.core.manifest import _apply_verdict_marks, chart_floor_replacement
    from socr.core.marks import Check
    from socr.core.result import FailureMode, PageOutput, PageStatus

    png = "![Failed table page 1](figures/failed_table_p1.png)"
    pdf, boxes = _chart_page_with_prose(tmp_path)
    p = _floor_page_state(pdf, png)
    body = chart_floor_replacement(
        _floor_state(pdf, tmp_path),
        1,
        p,
        marker_line="[page 1 failed: no usable OCR output]",
        png_ref=png,
    )()
    assert body, "premise: the page ships a chart body"
    region_marks = [ln for ln in body.splitlines() if "chart_region=" in ln]
    assert len(region_marks) == len(boxes), "premise: one region line per chart region"

    output = PageOutput(
        page_num=1,
        text=body,
        status=PageStatus.ERROR,
        engine="native",
        audit_passed=False,
        failure_mode=FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED,
    )
    marked, verdict = _apply_verdict_marks(output, p, 1)

    region_clauses = [c for c in verdict.clauses if c.check is Check.CHART_REGION]
    assert len(region_clauses) == len(boxes), (
        "the verdict records one clause per chart region, not zero"
    )
    assert {c.scope for c in region_clauses} == {f"chart:{i}" for i in range(1, len(boxes) + 1)}
    assert {c.mark(1) for c in region_clauses} == set(region_marks), (
        "each recorded clause renders the exact line already on the page"
    )
    assert marked.text == body, "and so nothing is written twice: the bytes do not move"

    # The difference: the same page, same body, with the record absent derives none.
    p.chart_floor_clauses = None
    _, without = _apply_verdict_marks(output, p, 1)
    assert [c for c in without.clauses if c.check is Check.CHART_REGION] == [], (
        "control: a page the chart floor never spoke for records no region clause"
    )


def test_a_regions_result_is_UNKNOWN_unless_its_own_outcome_was_established(tmp_path):
    """The UNKNOWN/FAIL distinction #1145 draws, pinned per REGION on the records.

    Found by a surviving mutant, not by review: inverting ``decided`` in
    ``region_clause_records`` passed the whole file. The existing test for it
    (``test_a_decided_failure_is_FAIL_and_an_undecidable_one_is_UNKNOWN``) covers
    ``panel_mark_line``, which the floor no longer calls now that both the line and the
    clause are rendered from one record -- so the distinction was no longer covered on
    the path that ships.

    Each case is driven through the real reader's own reading of a drawn page. The
    unlabelled-panel case replaces the ``label`` of a REAL ``PanelReading`` (frozen, so
    ``replace`` yields the same production type) because the reader's label fallback
    finds the legend text on every page that can be drawn here; everything else about
    that panel is what the reader produced.
    """
    from dataclasses import replace as _replace

    from socr.figures.chart_reader import read_chart_page
    from socr.figures.chart_witness import region_clause_records
    from socr.tables.reconstruct import chart_region_bboxes

    pdf, _ = _chart_page_with_prose(tmp_path)
    with fitz.open(pdf) as doc:
        boxes = list(chart_region_bboxes(doc[0]))
        reading = read_chart_page(doc[0], boxes, page_num=1)
    assert reading.panels, "premise: the reader resolves this page's panel"
    pictures = {i: f"![chart region {i}](f/c{i}.png)" for i in range(1, len(boxes) + 1)}

    def results(reading_, pictures_):
        return [
            r["result"]
            for r in region_clause_records(
                1, region_pictures=pictures_, reading=reading_, region_count=len(boxes)
            )
        ]

    # Nothing was DECIDED about these regions: the counts are not wired in this version,
    # and a region with no picture was not examined at all.
    assert results(reading, pictures) == ["UNKNOWN"] * len(boxes), (
        "a resolved panel whose counts are not shipped decided nothing"
    )
    assert results(reading, {}) == ["UNKNOWN"] * len(boxes), (
        "a region with no picture decided nothing either"
    )

    # The one region outcome that IS established: a panel the reader read whose printed
    # label is absent, so its values can never be bound to a year.
    unlabelled = _replace(
        reading, panels={k: _replace(v, label="") for k, v in reading.panels.items()}
    )
    assert results(unlabelled, pictures) == ["FAIL"] * len(boxes), (
        "an unbindable panel is a FAIL about that region, not an UNKNOWN"
    )


def test_a_saved_body_stops_the_selector_offering_the_chart_floor_at_all(tmp_path):
    """The round-2 test gap, CLOSED: the selector's own override branch.

    ``test_a_saved_body_means_the_chart_floor_is_never_offered`` pins only the floor's
    half (a ``None`` builder constructs nothing). The branch that decides WHICH to pass
    lives in ``_select_page_output_tagged`` and needs a real ``DocumentState``, so it was
    covered by nothing but a mutant on the production line, and the report said so.

    It is driven here through ``_select_and_finalize_page`` -- the seam the pipeline
    calls -- so both halves are pinned at once: ``saved_text`` is what reaches the
    selector as ``text_will_be_overridden``, and the selector is what declines to build.
    The state is the GH-317 floor fixture, the one shape that reaches this branch.
    """
    import socr.core.manifest as m
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import DocumentHandle, DocumentState

    from test_gh317_structure_class_floor import NATIVE_TEXT_WITH_PROSE, _born_digital_pdf

    pdf = _born_digital_pdf(tmp_path)
    built: list[int] = []
    real = m.chart_floor_replacement

    def _spy(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    def _state():
        state = DocumentState(handle=DocumentHandle.from_path(pdf))
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.has_tables = True
        ps.native_text = NATIVE_TEXT_WITH_PROSE
        native = PageOutput(
            page_num=1,
            text=NATIVE_TEXT_WITH_PROSE,
            status=PageStatus.SUCCESS,
            engine="native",
            audit_passed=True,
        )
        model = PageOutput(
            page_num=1,
            text="The table on this page could not be read reliably.\n",
            status=PageStatus.SUCCESS,
            engine="gemini",
            audit_passed=False,
        )
        model.rejection_class = "ambiguous_deferred"
        ps.attempts.extend([native, model])
        ps.best_output = native
        return state

    m.chart_floor_replacement = _spy
    try:
        m._select_and_finalize_page(_state(), 1, saved_text="a body saved by an earlier run")
        overridden = list(built)
        built.clear()
        m._select_and_finalize_page(_state(), 1)
        normal = list(built)
    finally:
        m.chart_floor_replacement = real

    assert normal == [1], "premise: this state DOES reach the branch that offers the chart floor"
    assert overridden == [], "a saved body must stop the selector building the chart floor at all"


def _chart_floor_document_state(pdf):
    """A structure-class floor state over a CHART pdf, with that page's real text.

    Same shape as ``_floor_document_state`` -- the one that reaches the no-detected-box
    branch -- but the native text is the chart page's own, so the recovery lane and the
    chart reader both run on real bytes.
    """
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import DocumentHandle, DocumentState

    with fitz.open(pdf) as doc:
        words = list(doc[0].get_text("words"))
        native = doc[0].get_text()

    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.has_tables = True
    ps.native_text = native
    ps.native_words = words
    ps.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"
    native_attempt = PageOutput(
        page_num=1, text=native, status=PageStatus.SUCCESS, engine="native", audit_passed=True
    )
    model = PageOutput(
        page_num=1,
        text="The table on this page could not be read reliably.\n",
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
    )
    model.rejection_class = "ambiguous_deferred"
    ps.attempts.extend([native_attempt, model])
    ps.best_output = native_attempt
    return state


def test_what_the_page_says_about_withheld_text_is_in_its_terminal_verdict(tmp_path):
    """Astra round 3 (P2): the page-scoped line existed only as a string.

    A page could say ``native_coverage=FAIL; scope=page`` in its Markdown while its
    terminal JSON carried no such clause, so a machine-readable audit could not tell that
    the page's text had been withheld -- for ordering, or as withheld runs. Both are
    recorded now, from the same records the lines are rendered from.

    Driven through ``_select_and_finalize_page``, whose record IS what the orchestrator
    serialises, and asserted on both arms: the two-column page (the order refusal) and
    the one-column page (withheld runs). The two arms also prove the clause is not a
    constant: the reasons differ, and each matches its own page's line.
    """
    import socr.core.manifest as m
    from socr.core.marks import SCOPE_PAGE, Check
    from socr.figures.chart_witness import ORDER_REFUSED_REASON

    refusing_pdf, _ = _two_column_chart_page(tmp_path)
    accepting_pdf, _ = _chart_page_with_prose(tmp_path)

    def coverage_clauses(pdf):
        record = m._select_and_finalize_page(_chart_floor_document_state(pdf), 1)
        clauses = [
            c
            for c in record.verdict.clauses
            if c.check is Check.NATIVE_COVERAGE and c.scope == SCOPE_PAGE
        ]
        return record, clauses

    refused_record, refused = coverage_clauses(refusing_pdf)
    assert "chart_region=" in refused_record.output.text, (
        "premise: the two-column page did reach the chart floor"
    )
    assert len(refused) == 1, "the order refusal is one page-scoped coverage clause"
    assert refused[0].reason == ORDER_REFUSED_REASON
    assert refused[0].mark(1) in refused_record.output.text, (
        "and the clause renders the exact line the page carries"
    )

    runs_record, runs = coverage_clauses(accepting_pdf)
    assert "chart_region=" in runs_record.output.text, (
        "premise: the one-column page did reach the chart floor"
    )
    assert len(runs) == 1, "the withheld runs are one page-scoped coverage clause too"
    assert "run(s) of this page's text were withheld" in runs[0].reason, (
        "with its own reason, not the refusal's"
    )
    assert runs[0].reason != ORDER_REFUSED_REASON
    assert runs[0].mark(1) in runs_record.output.text


def test_an_empty_saved_body_does_not_spin_the_recovery_check_forever(tmp_path):
    """Round 4: the suite hung for 55 minutes, at 100% CPU, on a BLANK page.

    ``_is_prose_recovery_body`` walked the body with ``while True``, dropping leading
    blank and canonical-mark lines. On an empty or whitespace-only body,
    ``"".partition("\\n")`` returns ``("", "", "")`` -- blank head, empty separator -- so it
    reassigned ``body = ""`` and looped forever. Latent while every caller passed a body
    carrying at least a marker line; round 4's ``_recovery_facts_of_saved_body`` passes a
    SAVED fragment, which can be empty, and
    ``test_1100_blank_pages.py::test_halted_blank_page_is_not_reported_as_shipped``
    stalled the whole run.

    Asserted with a TIMEOUT rather than by calling it directly: a regression here does not
    fail a test, it hangs the suite, and a hang is the one failure that reports nothing. A
    thread that is still alive after the deadline is the failure.
    """
    import threading

    import socr.core.manifest as m

    def returns_within(seconds, fn, *args):
        done: dict = {}
        worker = threading.Thread(target=lambda: done.setdefault("r", fn(*args)), daemon=True)
        worker.start()
        worker.join(seconds)
        assert not worker.is_alive(), f"{fn.__name__} did not return within {seconds}s"
        return done.get("r")

    assert returns_within(5, m._is_prose_recovery_body, "", 1) is False
    assert returns_within(5, m._is_prose_recovery_body, "\n\n   \n", 1) is False
    # A body that is nothing BUT this page's own canonical lines is the same shape: every
    # line is dropped and the walk must still end.
    from socr.core.marks import page_unchecked_clause_mark

    only_marks = page_unchecked_clause_mark(1, "chart_region", "UNKNOWN", "chart:1", "r")
    assert returns_within(5, m._is_prose_recovery_body, only_marks + "\n", 1) is False

    # And through the production entry point, with the empty saved body itself.
    from test_gh317_structure_class_floor import _born_digital_pdf

    pdf = _born_digital_pdf(tmp_path)
    record = returns_within(
        20, lambda: m._select_and_finalize_page(_floor_document_state(pdf), 1, saved_text="")
    )
    assert record is not None, "an empty saved body finalizes instead of hanging"
    assert record.output.scanned_prose_recovered is not True, (
        "and an empty body is not recorded as a prose recovery"
    )
