"""GH-656 (leftover from #654/E1): a scan/decorative page raster must not veto
the whole page. When the largest raster clears ``CHART_MIN_CLUSTER_AREA`` and
``_raster_is_scan_or_decorative`` hits (GH-511), ``has_chart_marks`` used to
``return False`` immediately -- never reaching the vector path, so a real
vector chart drawn on the same page (e.g. under a full-page decorative
background, or beside a scanned photograph) was silently dropped.

Wanted: do not claim chart from that raster, but fall through to (a) any
other qualifying raster placement that is not itself scan/decorative, then
(b) the vector path. A pure scan (one page raster, no drawings) must still
answer False -- already covered by
``tests/figures/test_has_chart_marks_scan_raster.py``.

All fixtures are real PyMuPDF page objects -- no MagicMock -- so this pins
the actual control-flow DIFFERENCE: main's ``has_chart_marks`` answers False
on the combined fixture below; the fix answers True.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.figures.extractor import (  # noqa: E402
    CHART_MIN_CLUSTER_AREA,
    CLUSTER_GAP,
    DATA_STROKE_MIN_WIDTH,
    MIN_DATA_MARKS,
    RASTER_TEXT_DENSITY_MIN,
    SCAN_RASTER_PAGE_COVERAGE_MIN,
    has_chart_marks,
)

PAGE_W, PAGE_H = 612.0, 792.0

# Same full-page rect shape as test_has_chart_marks_scan_raster.py.
FULL_PAGE_RECT = fitz.Rect(10, 10, 600, 780)

STROKE_WIDTH = DATA_STROKE_MIN_WIDTH * 0.6
_GATE_SIDE = math.sqrt(CHART_MIN_CLUSTER_AREA)
BAR_WIDTH = _GATE_SIDE / 3
BAR_GAP = CLUSTER_GAP / 3
BAR_HEIGHT = _GATE_SIDE
BAR_COLORS = [(0.8, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.1, 0.8)]
assert len(BAR_COLORS) == MIN_DATA_MARKS


def _place_words_grid(page, rect: fitz.Rect, count: int, fontsize: float = 8.0) -> None:
    """Same word-grid placement as test_has_chart_marks_scan_raster.py."""
    if count <= 0:
        return
    cols = max(1, int(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    margin = 5.0
    usable_w = max(rect.width - 2 * margin, 1.0)
    usable_h = max(rect.height - 2 * margin, 1.0)
    placed = 0
    for r in range(rows):
        for c in range(cols):
            if placed >= count:
                return
            x = rect.x0 + margin + (usable_w * c / max(cols - 1, 1) if cols > 1 else usable_w / 2)
            y = (
                rect.y0
                + margin
                + (usable_h * r / max(rows - 1, 1) if rows > 1 else usable_h / 2)
                + fontsize
            )
            page.insert_text((x, y), "w", fontsize=fontsize)
            placed += 1


def _rgb_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (400, 500), color=(30, 100, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _draw_qualifying_bar_chart(page, *, x0: float = 100.0, y_bottom: float = 400.0) -> None:
    """A filled bar chart cluster that clears CHART_MIN_CLUSTER_AREA and
    MIN_DATA_MARKS -- same shape as GH-150's `_make_filled_bar_doc` fixture,
    proven to pass the vector path on its own."""
    x = x0
    for color in BAR_COLORS:
        rect = fitz.Rect(x, y_bottom - BAR_HEIGHT, x + BAR_WIDTH, y_bottom)
        page.draw_rect(rect, color=color, fill=color, width=STROKE_WIDTH)
        x += BAR_WIDTH + BAR_GAP


def _make_combined_doc(tmp_path: Path, name: str, *, with_vector_chart: bool) -> Path:
    """A full-page decorative/scan raster (dense native words, same shape as
    the GH-511 scan fixture) plus, optionally, a qualifying vector bar chart
    drawn on the same page."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(FULL_PAGE_RECT, stream=_rgb_png(), keep_proportion=False)

    raster_area = FULL_PAGE_RECT.width * FULL_PAGE_RECT.height
    word_count = round(3.0 * raster_area / 10000.0)  # Fed-scan-shaped density
    _place_words_grid(page, FULL_PAGE_RECT, word_count)

    if with_vector_chart:
        _draw_qualifying_bar_chart(page)

    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _open_page(pdf_path: Path):
    doc = fitz.open(pdf_path)
    return doc, doc[0]


# A raster well above CHART_MIN_CLUSTER_AREA but well below
# SCAN_RASTER_PAGE_COVERAGE_MIN -- same shape as the #510 small-raster gate
# fixture in test_has_chart_marks_scan_raster.py, so it is a *qualifying*,
# non-scan raster by construction (the coverage gate rejects it as scan
# regardless of any native text density).
SMALL_QUALIFYING_RECT = fitz.Rect(50, 50, 250, 250)  # 200x200 = 40000pt2


def _make_scan_plus_small_raster_doc(tmp_path: Path, name: str, *, small_first: bool) -> Path:
    """Two DISTINCT raster placements (two xrefs): a full-page scan/decorative
    raster and a smaller qualifying non-scan raster. ``small_first`` controls
    image ENUMERATION order (the order ``page.get_images()`` will return
    them) so the fixture also exercises order independence."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    if small_first:
        page.insert_image(SMALL_QUALIFYING_RECT, stream=_rgb_png(), keep_proportion=False)
        page.insert_image(FULL_PAGE_RECT, stream=_rgb_png(), keep_proportion=False)
    else:
        page.insert_image(FULL_PAGE_RECT, stream=_rgb_png(), keep_proportion=False)
        page.insert_image(SMALL_QUALIFYING_RECT, stream=_rgb_png(), keep_proportion=False)

    raster_area = FULL_PAGE_RECT.width * FULL_PAGE_RECT.height
    word_count = round(3.0 * raster_area / 10000.0)  # Fed-scan-shaped density
    _place_words_grid(page, FULL_PAGE_RECT, word_count)

    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_same_xref_two_placements_doc(tmp_path: Path, name: str) -> Path:
    """ONE image (one xref) placed TWICE: once full-page (scan/decorative
    shaped) and once at the smaller, non-scan-qualifying size. Exercises the
    ``get_image_rects(xref)`` multi-rect branch directly, as distinct from
    two different images."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    xref = page.insert_image(FULL_PAGE_RECT, stream=_rgb_png(), keep_proportion=False)
    page.insert_image(SMALL_QUALIFYING_RECT, xref=xref)

    raster_area = FULL_PAGE_RECT.width * FULL_PAGE_RECT.height
    word_count = round(3.0 * raster_area / 10000.0)
    _place_words_grid(page, FULL_PAGE_RECT, word_count)

    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


class TestE1Fallthrough:
    def test_setup_sanity_raster_is_scan_shaped_and_qualifying(self) -> None:
        """Fixture self-check: the raster clears both gates that make main
        return False before ever reaching the vector path."""
        coverage = FULL_PAGE_RECT.width * FULL_PAGE_RECT.height / (PAGE_W * PAGE_H)
        assert coverage >= SCAN_RASTER_PAGE_COVERAGE_MIN
        assert FULL_PAGE_RECT.width * FULL_PAGE_RECT.height >= CHART_MIN_CLUSTER_AREA
        assert 3.0 > RASTER_TEXT_DENSITY_MIN

    def test_decorative_raster_with_vector_chart_falls_through_to_true(
        self, tmp_path: Path
    ) -> None:
        """FALSIFICATION: main's has_chart_marks returns False here (the
        scan/decorative largest raster short-circuits before the vector path
        is ever reached). The fix must fall through and find the chart."""
        pdf = _make_combined_doc(tmp_path, "combined.pdf", with_vector_chart=True)
        doc, page = _open_page(pdf)
        try:
            assert has_chart_marks(page) is True
        finally:
            doc.close()

    def test_decorative_raster_without_vector_chart_still_false(self, tmp_path: Path) -> None:
        """Control: same scan/decorative raster, no vector chart underneath --
        must still answer False (pure scan, GH-511 unchanged)."""
        pdf = _make_combined_doc(tmp_path, "no_chart.pdf", with_vector_chart=False)
        doc, page = _open_page(pdf)
        try:
            assert has_chart_marks(page) is False
        finally:
            doc.close()

    def test_smaller_nonscan_raster_found_when_largest_is_scan(self, tmp_path: Path) -> None:
        """Reviewer addendum: the largest placement is scan/decorative, but a
        SMALLER, DISTINCT raster placement also clears CHART_MIN_CLUSTER_AREA
        and is not itself scan/decorative -- must be found (True), never
        vetoed by the largest placement's rejection."""
        pdf = _make_scan_plus_small_raster_doc(tmp_path, "scan_plus_small.pdf", small_first=False)
        doc, page = _open_page(pdf)
        try:
            assert has_chart_marks(page) is True
        finally:
            doc.close()

    def test_smaller_nonscan_raster_found_regardless_of_enumeration_order(
        self, tmp_path: Path
    ) -> None:
        """Reviewer addendum: same fixture, but the smaller non-scan raster is
        inserted (and therefore enumerated by page.get_images()) BEFORE the
        full-page scan raster. Selection must not depend on image order --
        largest-first ranking inside has_chart_marks should give the same
        answer either way."""
        pdf = _make_scan_plus_small_raster_doc(tmp_path, "small_first.pdf", small_first=True)
        doc, page = _open_page(pdf)
        try:
            assert has_chart_marks(page) is True
        finally:
            doc.close()

    def test_same_xref_two_placements_one_scan_one_qualifying(self, tmp_path: Path) -> None:
        """Reviewer addendum: a single image (one xref, via get_image_rects)
        placed twice -- once full-page/scan-shaped, once at a smaller
        qualifying size -- must still be found True through the smaller
        placement, exercising the multi-rect-per-xref branch directly."""
        pdf = _make_same_xref_two_placements_doc(tmp_path, "same_xref.pdf")
        doc, page = _open_page(pdf)
        try:
            assert has_chart_marks(page) is True
        finally:
            doc.close()
