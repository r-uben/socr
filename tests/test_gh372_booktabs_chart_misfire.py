"""GH-372: booktabs table rules must not reclassify the table as a chart.

On Cochrane–Piazzesi p18, Table 6's own >1pt booktabs rules clustered into a
blob above the chart area threshold, and the thick-stroke branch of
``_has_filled_rects_or_thick_strokes`` fired on them — so every word in the
table was excluded from the rowizer and the "chart region" crop shipped a
screenshot of the table. A rule-shaped stroke (bbox many times longer than it
is thick) must not qualify a cluster as a chart on its own; fills, coloured
strokes, and thick strokes with real two-dimensional extent still do.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from socr.tables.reconstruct import (
    _has_filled_rects_or_thick_strokes,
    chart_region_bboxes,
)

BLACK = (0, 0, 0)

# Full-width rule geometry: x-span and the y positions of three booktabs-style
# rules. 25pt apart keeps them within the 30pt cluster gap, and the merged
# cluster bbox (468pt x ~50pt) clears the 14 400pt² chart area threshold, so
# only the mark check separates table from chart — exactly the p18 shape.
RULE_X0, RULE_X1 = 72.0, 540.0
RULE_YS = (100.0, 125.0, 150.0)
HEAVY_RULE_WIDTH = 1.2  # >1pt: what the thick-stroke branch keys on


def _page_with_booktabs_rules(doc):
    page = doc.new_page()
    for y in RULE_YS:
        page.draw_line(
            fitz.Point(RULE_X0, y), fitz.Point(RULE_X1, y), color=BLACK, width=HEAVY_RULE_WIDTH
        )
    # A few table words between the rules, as on a real table page.
    page.insert_text(fitz.Point(90, 115), "Maturity   Coefficient   Std. Error")
    page.insert_text(fitz.Point(90, 140), "2 years    0.42          0.11")
    return page


def test_booktabs_rules_alone_do_not_make_a_chart_region() -> None:
    doc = fitz.open()
    page = _page_with_booktabs_rules(doc)
    assert chart_region_bboxes(page) == []
    doc.close()


def test_rule_shaped_thick_stroke_is_not_a_chart_mark() -> None:
    doc = fitz.open()
    page = _page_with_booktabs_rules(doc)
    cluster = (RULE_X0, RULE_YS[0], RULE_X1, RULE_YS[-1])
    assert _has_filled_rects_or_thick_strokes(page, cluster) is False
    doc.close()


def test_rules_plus_filled_bars_still_detected_as_chart() -> None:
    """A genuine chart whose axis is a thick rule still qualifies via its bars."""
    doc = fitz.open()
    page = _page_with_booktabs_rules(doc)
    for i, x in enumerate((120.0, 200.0, 280.0)):
        page.draw_rect(
            fitz.Rect(x, 148.0 - 20.0 * (i + 1), x + 40.0, 148.0),
            fill=(0.2, 0.4, 0.8),
        )
    assert chart_region_bboxes(page) != []
    doc.close()


def test_thick_stroke_with_real_extent_still_detected_as_chart() -> None:
    """A thick drawn line with genuine two-dimensional extent is a data mark."""
    doc = fitz.open()
    page = doc.new_page()
    page.draw_line(fitz.Point(100, 500), fitz.Point(400, 200), color=BLACK, width=2.0)
    assert chart_region_bboxes(page) != []
    doc.close()
