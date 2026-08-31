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
    _CHART_MIN_CLUSTER_AREA_PT2,
    _RULE_THINNESS_RATIO,
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


# ---------------------------------------------------------------------------
# GH-377: several booktabs rules packed into ONE get_drawings() path.
#
# #376 taught the thick-stroke branch to reject a rule-shaped stroke via
# _RULE_THINNESS_RATIO applied to d["rect"] -- the drawing's own union AABB.
# That works when each rule is its own drawing (flat rect). Some PDF writers
# instead emit all the parallel rules of one table as line items inside a
# SINGLE path (one page.new_shape() with several draw_line() calls, one
# finish()/commit()). d["rect"] is then the union of all those rules'
# bboxes -- tall enough relative to its width that thickness/span clears
# _RULE_THINNESS_RATIO even though every individual item is still a flat
# rule. The union-rect check cannot tell "one fat rectangle" from "three
# thin rules stacked" apart; only per-item geometry can.
# ---------------------------------------------------------------------------

PACKED_RULE_YS = (100.0, 125.0, 150.0)


def _page_with_packed_rule_cluster(doc):
    """One get_drawings() path containing PACKED_RULE_YS as separate line items."""
    page = doc.new_page()
    shape = page.new_shape()
    for y in PACKED_RULE_YS:
        shape.draw_line(fitz.Point(RULE_X0, y), fitz.Point(RULE_X1, y))
    # closePath=False: without it PyMuPDF appends a synthetic closing
    # segment between the last and first sub-path points, which would not
    # match the real booktabs-table shape this fixture models.
    shape.finish(color=BLACK, width=HEAVY_RULE_WIDTH, closePath=False)
    shape.commit()
    page.insert_text(fitz.Point(90, 97), "Maturity   Coefficient   Std. Error")
    page.insert_text(fitz.Point(90, 115), "2 years    0.42          0.11")
    page.insert_text(fitz.Point(90, 140), "5 years    0.58          0.09")
    page.insert_text(fitz.Point(90, 165), "10 years   0.71          0.14")
    return page


def _item_thinness_ratio(item) -> float:
    """thickness/span for a single get_drawings() 'items' entry, line-only.

    The fixture only exercises 'l' (line) items -- the drawing-path shape
    PyMuPDF actually emits for draw_line() calls sharing one Shape/finish().
    """
    kind = item[0]
    assert kind == "l", f"unexpected item kind {kind!r} in packed-rule fixture"
    p1, p2 = item[1], item[2]
    dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
    span, thickness = max(dx, dy), min(dx, dy)
    return thickness / span if span > 0.0 else 0.0


def test_packed_rule_fixture_is_one_drawing_of_rule_shaped_items() -> None:
    """Self-validate the fixture's geometry before trusting it as a regression."""
    doc = fitz.open()
    page = _page_with_packed_rule_cluster(doc)

    # find_tables()'s default "lines" strategy sees no ruled grid here (same
    # as a real booktabs table with no vertical rules).
    assert len(page.find_tables().tables) == 0

    drawings = page.get_drawings()
    assert len(drawings) == 1, "fixture must pack all rules into one path"
    d = drawings[0]
    assert d.get("type") not in ("f", "fs"), "fixture must be stroke-only, no fill"
    color = d.get("color")
    assert color == BLACK or color == (0.0, 0.0, 0.0)
    assert d.get("width", 0.0) > 1.0

    items = d.get("items") or []
    assert len(items) == len(PACKED_RULE_YS)
    for item in items:
        assert item[0] == "l"
        assert _item_thinness_ratio(item) <= _RULE_THINNESS_RATIO, (
            "each packed rule must individually be rule-shaped"
        )

    rect = d["rect"]
    span = max(rect.x1 - rect.x0, rect.y1 - rect.y0)
    thickness = min(rect.x1 - rect.x0, rect.y1 - rect.y0)
    union_ratio = thickness / span if span > 0.0 else 0.0
    assert union_ratio > _RULE_THINNESS_RATIO, (
        "fixture must reproduce the union-rect false positive: the union AABB "
        "of the packed rules must itself clear the ratio gate"
    )

    area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
    assert area >= _CHART_MIN_CLUSTER_AREA_PT2, "cluster must clear the chart area gate"
    doc.close()


def test_packed_rule_cluster_is_not_a_chart_mark() -> None:
    """GH-377: per-item geometry must reject a packed multi-rule drawing.

    On the unmodified GH-377 head this fails: the union rect (thickness/span
    over _RULE_THINNESS_RATIO) makes the width > 1.0 branch return True even
    though every individual stroke item is still a flat rule.
    """
    doc = fitz.open()
    page = _page_with_packed_rule_cluster(doc)
    rect = page.get_drawings()[0]["rect"]
    cluster = (rect.x0, rect.y0, rect.x1, rect.y1)
    assert _has_filled_rects_or_thick_strokes(page, cluster) is False
    doc.close()


def test_packed_rule_cluster_chart_region_bboxes_is_empty() -> None:
    """Public-boundary pin: chart_region_bboxes must not eat a packed-rule table.

    Fails on the unmodified GH-377 head for the same reason as
    test_packed_rule_cluster_is_not_a_chart_mark.
    """
    doc = fitz.open()
    page = _page_with_packed_rule_cluster(doc)
    assert chart_region_bboxes(page) == []
    doc.close()


def test_packed_rule_cluster_keeps_table_words_through_extract_structured() -> None:
    """Rowizer-caller pin (measured where the issue asks: chart_region_bboxes'
    only consumer in the extraction path, BornDigitalDetector.extract_structured).

    reconstruct_table_regions is forced to [] so the text-strategy grid path is
    bypassed and the word-geometry chart-aware rowizer -- the thing
    chart_region_bboxes actually gates -- is what handles this page.

    On the unmodified GH-377 head: the packed rule cluster is misclassified as
    a chart, so extract_structured() emits an
    "![chart region ...](...)" placeholder and the table words are diluted
    out of the markdown table (this was manually confirmed against head
    53b67b865d0a00d0f28e76ec70159de4978447e9). After the fix, no chart
    placeholder should appear, every native word token must survive, and the
    data rows must render as markdown table rows.
    """
    from unittest.mock import patch

    from socr.core.born_digital import BornDigitalDetector

    doc = fitz.open()
    page = _page_with_packed_rule_cluster(doc)
    native_tokens = [w[4] for w in page.get_text("words")]

    with patch("socr.tables.reconstruct.reconstruct_table_regions", return_value=[]):
        out = BornDigitalDetector().extract_structured(page)

    assert "chart region" not in out, "packed booktabs rules must not become a chart placeholder"
    for token in native_tokens:
        assert token in out, f"native word token {token!r} was dropped from extract_structured()"
    assert "| 2 years | 0.42 | 0.11 |" in out
    assert "| 5 years | 0.58 | 0.09 |" in out
    assert "| 10 years | 0.71 | 0.14 |" in out
    doc.close()


def test_packed_path_with_one_2d_item_still_detected_as_chart() -> None:
    """A non-rule item INSIDE the same packed path must still qualify the cluster.

    Guards the per-item check's polarity: GH-377 rejects a packed drawing only
    when EVERY stroke item is rule-shaped. An inverted "every item must have
    2-D extent" implementation would pass the packed-rule tests above but fail
    here, because the diagonal shares one get_drawings() path with the rules.
    """
    doc = fitz.open()
    page = doc.new_page()
    shape = page.new_shape()
    for y in PACKED_RULE_YS:
        shape.draw_line(fitz.Point(RULE_X0, y), fitz.Point(RULE_X1, y))
    # One genuine 2-D mark (a trend-line-like diagonal) in the SAME path.
    shape.draw_line(fitz.Point(RULE_X0, 300.0), fitz.Point(RULE_X1, 100.0))
    shape.finish(color=BLACK, width=HEAVY_RULE_WIDTH, closePath=False)
    shape.commit()

    drawings = page.get_drawings()
    assert len(drawings) == 1, "fixture must keep rules and diagonal in one path"

    rect = drawings[0]["rect"]
    cluster = (rect.x0, rect.y0, rect.x1, rect.y1)
    assert _has_filled_rects_or_thick_strokes(page, cluster) is True
    assert chart_region_bboxes(page) != []
    doc.close()
