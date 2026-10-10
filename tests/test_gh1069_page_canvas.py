"""GH-1069: a page-sized background fill is the canvas, not a chart frame.

`_has_framed_data_cluster` accepts any axis-aligned drawing covering enough of a
cluster's bbox as the frame. A full-page white (or coloured) background fill,
unioned with everything else the page draws, becomes that "frame", and table
rules count as interior marks. The page then ships as a whole-page picture.

`has_chart_marks` now drops a drawing that reaches the page boundary on all four
sides before clustering. The comparison is made in UNROTATED page space, where
`get_drawings()` reports its rects.

Hermetic: real PyMuPDF pages built in-test, detector only, no provider.
"""

from __future__ import annotations

import math

import pytest

fitz = pytest.importorskip("fitz")

from socr.figures.extractor import (  # noqa: E402
    CHART_MIN_CLUSTER_AREA,
    CLUSTER_GAP,
    DATA_STROKE_MIN_WIDTH,
    MIN_DRAWINGS_FOR_VECTOR,
    _cluster_drawings,
    _has_framed_data_cluster,
    _is_page_canvas,
    has_chart_marks,
)

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0
NEUTRAL = (0.0, 0.0, 0.0)
THIN = DATA_STROKE_MIN_WIDTH * 0.6
N_RULES = MIN_DRAWINGS_FOR_VECTOR + 2
_SIDE = math.sqrt(CHART_MIN_CLUSTER_AREA)


def _page(*, background: bool, spike: bool = False, rotation: int = 0):
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    if background:
        page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
    for i in range(25):
        page.insert_text(
            fitz.Point(60, 40 + i * 12), f"prose line {i} of the body text", fontsize=9
        )
    # Table rules, spaced inside CLUSTER_GAP so they form one cluster.
    for i in range(N_RULES):
        y = 380 + i * (CLUSTER_GAP / 2)
        page.draw_line(fitz.Point(80, y), fitz.Point(80 + _SIDE, y), color=NEUTRAL, width=THIN)
    if spike:
        frame = fitz.Rect(300, 40, 300 + _SIDE, 40 + _SIDE * 1.25)
        page.draw_rect(frame, color=NEUTRAL, width=THIN)
        n = MIN_DRAWINGS_FOR_VECTOR + 2
        step = (frame.width - 12) / (n - 1)
        for i in range(n):
            x = frame.x0 + 6 + i * step
            top = frame.y1 - 6 - (i % 4 + 1) * (frame.height - 12) / 5
            page.draw_line(
                fitz.Point(x, frame.y1 - 6), fitz.Point(x, top), color=NEUTRAL, width=THIN
            )
    if rotation:
        page.set_rotation(rotation)
    data = doc.tobytes()
    doc.close()
    return fitz.open(stream=data, filetype="pdf")[0]


def _raw_framed(page) -> bool:
    """Would the framed-cluster gate admit any cluster of the RAW drawings?"""
    drawings = page.get_drawings()
    clusters = _cluster_drawings(drawings, PAGE_WIDTH, PAGE_HEIGHT, CLUSTER_GAP)
    return any(
        _has_framed_data_cluster(cd, bbox)
        and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= CHART_MIN_CLUSTER_AREA
        for cd, bbox in clusters
    )


def test_background_fill_with_table_rules_is_not_a_chart():
    page = _page(background=True)
    # Premise: without the canvas drop, the fill frames the rules.
    assert _raw_framed(page) is True
    assert has_chart_marks(page) is False


def test_real_framed_spike_plot_on_filled_page_is_still_a_chart():
    page = _page(background=True, spike=True)
    assert has_chart_marks(page) is True


def test_control_without_background_is_unaffected():
    """Pin the difference: the fill is the only thing the fix changes."""
    plain = _page(background=False)
    filled = _page(background=True)
    # No canvas drawing, so the drop is a no-op and the verdict is whatever the
    # remaining drawings give: the same as the filled page minus its fill.
    assert _raw_framed(plain) is False
    assert has_chart_marks(plain) is False
    assert has_chart_marks(filled) == has_chart_marks(plain)
    plain_s = _page(background=False, spike=True)
    assert has_chart_marks(plain_s) is True


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_page_background_is_still_the_canvas(rotation):
    page = _page(background=True, rotation=rotation)
    assert page.rotation == rotation
    assert has_chart_marks(page) is False


def test_is_page_canvas_tolerance_and_partial_cover():
    page_rect = fitz.Rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT)
    assert _is_page_canvas(fitz.Rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT), page_rect)
    assert _is_page_canvas(fitz.Rect(0.5, 0.5, PAGE_WIDTH - 0.5, PAGE_HEIGHT - 0.5), page_rect)
    assert not _is_page_canvas(fitz.Rect(20, 0, PAGE_WIDTH, PAGE_HEIGHT), page_rect)
    assert not _is_page_canvas(None, page_rect)
