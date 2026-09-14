"""#750: a heading row overhanging the top tick still binds to its panel.

`_panel_label`'s vertical test rejected a heading whenever its WORD-ROW BBOX
(``y1``, PyMuPDF's own bottom edge, padded with the font's descent) reached
past the highest tick -- and on two SEP releases every heading's padded edge
does, by 0.45pt, even though the glyphs themselves sit clearly above the tick.
The fix compares the row's CENTRE instead, which is not sensitive to that
per-font padding and needs no tolerance constant of its own.

The two guards below are pinned from geometry MEASURED off the real corpus
pages (``sep-20250319-p09``, region 1) rather than invented, per the brief:
a heading row that dips 0.45pt past the tick on its bbox edge but not its
centre, and a legend row (``March projections``) that is horizontally inside
the frame just like a heading, but sits well below the tick on BOTH measures
-- the negative control showing the loosened rule stays bounded and does not
start accepting rows the old rule correctly rejected.

The full-pipeline corpus test at the bottom is the authoritative one; the two
geometry pins run in CI, where the corpus is absent.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from test_gh635_chart_reader import build_chart

from socr.figures.chart_reader import Frame, WordRow, _panel_label, read_chart_page
from socr.tables.reconstruct import chart_region_bboxes

SEP_CORPUS = Path.home() / "Data/socr/sep-dotplots/in"

#: Region 1's frame on ``sep-20250319-p09``, page 9 -- read directly off the
#: page, not chosen to make the test pass.
_MEASURED_FRAME = Frame(
    baseline=225.97525024414062,
    x0=107.62460327148438,
    x1=502.5957946777344,
    tick_ys=(
        146.574,
        154.513,
        162.451,
        170.389,
        178.336,
        186.275,
        194.213,
        202.151,
        210.099,
        218.037,
    ),
)

#: The heading row itself: its bbox bottom edge (``y1``) sits 0.45pt BELOW
#: the top tick (146.574), which is exactly what made the old ``row.y1 >
#: top_tick`` test reject it. Its centre (141.72) sits comfortably above.
_HEADING_ROW = WordRow(
    y0=136.4110107421875,
    y1=147.02432250976562,
    x0=115.2181625366211,
    x1=131.13015747070312,
    text="2025",
    tokens=((123.2, 15.9, "2025"),),
)

#: A legend row on the same page, in the same panel: horizontally inside the
#: frame exactly like a heading, but its centre (154.67) sits well below the
#: tick too -- it is not a false positive the loosened rule now lets through.
_LEGEND_ROW = WordRow(
    y0=151.12872314453125,
    y1=158.20425415039062,
    x0=141.19007873535156,
    x1=179.93051147460938,
    text="March projections",
    tokens=((160.6, 38.7, "March projections"),),
)


def test_a_heading_row_dipping_past_the_top_tick_on_its_bbox_edge_still_resolves() -> None:
    assert _HEADING_ROW.y1 > min(_MEASURED_FRAME.tick_ys), "the bbox-edge overhang this guards"
    assert _HEADING_ROW.cy < min(_MEASURED_FRAME.tick_ys), "the centre stays clear of the tick"

    label = _panel_label(_MEASURED_FRAME, [_HEADING_ROW, _LEGEND_ROW], shared=set())

    assert label == "2025"


def test_a_row_whose_centre_also_sits_below_the_tick_is_not_mistaken_for_a_heading() -> None:
    """The negative control: looking at the centre must not over-loosen the rule."""
    assert _LEGEND_ROW.cy > min(_MEASURED_FRAME.tick_ys)

    label = _panel_label(_MEASURED_FRAME, [_LEGEND_ROW], shared=set())

    assert label == ""


def test_a_heading_the_shared_set_already_excludes_stays_excluded() -> None:
    """An axis title repeated across panels is caught by ``shared`` alone.

    Region 1's own axis title ('Number of participants') is drawn OUTSIDE the
    frame's horizontal span here, so it never needed the containment rule
    either -- ``shared`` is what keeps it out, on this corpus. This pins that
    property directly rather than leaving it implicit.
    """
    axis_title = WordRow(
        y0=126.98492431640625,
        y1=134.06045532226562,
        x0=463.2648620605469,
        x1=512.4647216796875,
        text="Number of participants",
        tokens=((487.9, 49.2, "Number of participants"),),
    )
    rows = [axis_title, _HEADING_ROW, _LEGEND_ROW]

    label = _panel_label(_MEASURED_FRAME, rows, shared={"Number of participants"})

    assert label == "2025"


@pytest.mark.skipif(not SEP_CORPUS.exists(), reason="SEP dot-plot corpus is not present")
@pytest.mark.parametrize(
    "stem",
    ["sep-20250319-p09", "sep-20250618-p09"],
)
def test_the_two_affected_releases_now_label_every_panel(stem: str) -> None:
    """The defect itself, on the real pages it was found on (#750).

    Both releases lost every panel's label to the bbox-edge overhang; all four
    panels must now resolve to their year (or "Longer run"), and none may be
    refused.
    """
    page = fitz.open(str(SEP_CORPUS / f"{stem}.pdf"))[0]
    reading = read_chart_page(page, chart_region_bboxes(page), page_num=9)

    assert reading.refusals == {}
    labels = [reading.panels[idx].label for idx in sorted(reading.panels)]
    assert labels == ["2025", "2026", "2027", "Longer run"]


# ---------------------------------------------------------------------------
# The refusal path: a panel whose heading cannot be located at all
# ---------------------------------------------------------------------------


def test_an_unresolved_heading_refuses_instead_of_publishing_an_empty_label(
    tmp_path: Path,
) -> None:
    """No corpus page exercises this -- every real panel draws SOME heading.

    ``build_chart(heading="")`` draws every other feature of a readable panel
    (axis, ticks, bins, legend, bars) but no heading row at all, so nothing
    the drawing offers can serve as one. ``ticks`` extends past the legend's
    own fixed height (``build_chart`` always draws it level with the default
    top tick) so the legend cannot stand in for the missing heading -- it must
    sit BELOW the raised top tick, which condition (c) already excludes it by.
    The panel must be recorded as a refusal naming why, not published with
    ``label=""`` for a caller to match against nothing.
    """
    doc, bboxes = build_chart(
        tmp_path / "no_heading.pdf",
        [2, 3, 1, 0, 4],
        None,
        heading="",
        ticks=(2, 4, 6, 8, 10, 12),
    )

    reading = read_chart_page(doc[0], bboxes, page_num=1)

    assert 1 not in reading.panels
    assert 1 in reading.refusals
    reason = reading.refusals[1]
    assert "heading" in reason
    assert "year" in reason
