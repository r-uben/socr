"""GH-511 (large half): a page-covering raster with dense native words is the
scan itself (OCR text layer under a page photograph) or a decorative page
export (real text under a slide background) -- not a chart.

Anchors measured 2026-09-07 on the real corpus (docs/log/2026-09-07_E1-scan-raster-not-chart.md):
  - not-chart floor: 1.37 words/100pt^2 (ECB 2021 slide export, sparsest page)
  - chart ceiling:   0.21 words/100pt^2 (synthetic raster chart, axis tick labels only)
``RASTER_TEXT_DENSITY_MIN`` (0.75) sits strictly between both with margin.

All fixtures are built with real PyMuPDF page objects -- no MagicMock -- so the
tests pin the actual DIFFERENCE the new gate makes: on vs off, real geometry.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.figures.extractor import (
    CHART_MIN_CLUSTER_AREA,
    RASTER_TEXT_DENSITY_MIN,
    SCAN_RASTER_PAGE_COVERAGE_MIN,
    has_chart_marks,
)

PAGE_W, PAGE_H = 612, 792


def _place_words_grid(page, rect: "fitz.Rect", count: int, fontsize: float = 8.0) -> None:
    """Place ``count`` single-token words evenly spread across ``rect``.

    Each call to ``insert_text`` is a separate text-showing operation, so
    PyMuPDF's word segmentation reports each placement as its own word --
    verified against ``get_text("words")`` returning exactly ``count`` words
    for a grid this size.
    """
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


def _make_raster_page(
    tmp_path: Path,
    name: str,
    *,
    rect: "fitz.Rect",
    word_count: int,
    colorspace: str = "gray",
) -> Path:
    """A single-page PDF: one page-covering (or partial) raster + N native words inside it."""
    if colorspace == "gray":
        img = _gray_png()
    else:
        img = _rgb_png()

    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_image(rect, stream=img, keep_proportion=False)
    _place_words_grid(page, rect, word_count)

    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _gray_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (400, 500), color=200).save(buf, format="PNG")
    return buf.getvalue()


def _rgb_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (400, 500), color=(30, 100, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _open_page(pdf_path: Path):
    doc = fitz.open(pdf_path)
    return doc, doc[0]


# A raster covering >= SCAN_RASTER_PAGE_COVERAGE_MIN of the PAGE_W x PAGE_H page.
FULL_PAGE_RECT = fitz.Rect(10, 10, 600, 780)


class TestScanRasterNotChart:
    """The new gate: page-covering raster + prose-density native words -> not a chart."""

    def test_dense_full_page_scan_is_not_chart(self, tmp_path: Path) -> None:
        """Fed-scan-shaped fixture: coverage 1.0-ish, density ~3.0/100pt2 (measured Fed range 2.22-8.03)."""
        raster_area = FULL_PAGE_RECT.width * FULL_PAGE_RECT.height
        word_count = round(3.0 * raster_area / 10000.0)
        pdf = _make_raster_page(
            tmp_path, "scan.pdf", rect=FULL_PAGE_RECT, word_count=word_count, colorspace="gray"
        )
        doc, page = _open_page(pdf)
        try:
            coverage = FULL_PAGE_RECT.width * FULL_PAGE_RECT.height / (PAGE_W * PAGE_H)
            assert coverage >= SCAN_RASTER_PAGE_COVERAGE_MIN
            assert has_chart_marks(page) is False
        finally:
            doc.close()

    def test_sparse_decorative_slide_is_not_chart(self, tmp_path: Path) -> None:
        """ECB-2021-slide-shaped fixture: RGB raster, density 1.37/100pt2 (measured floor)."""
        raster_area = FULL_PAGE_RECT.width * FULL_PAGE_RECT.height
        word_count = round(1.37 * raster_area / 10000.0)
        pdf = _make_raster_page(
            tmp_path, "slide.pdf", rect=FULL_PAGE_RECT, word_count=word_count, colorspace="rgb"
        )
        doc, page = _open_page(pdf)
        try:
            assert has_chart_marks(page) is False
        finally:
            doc.close()

    def test_raster_chart_with_sparse_axis_labels_still_a_chart(self, tmp_path: Path) -> None:
        """Synthetic raster-chart anchor: density 0.21/100pt2 (measured ceiling) -- must stay a chart."""
        raster_area = FULL_PAGE_RECT.width * FULL_PAGE_RECT.height
        word_count = round(0.21 * raster_area / 10000.0)
        pdf = _make_raster_page(
            tmp_path, "chart.pdf", rect=FULL_PAGE_RECT, word_count=word_count, colorspace="rgb"
        )
        doc, page = _open_page(pdf)
        try:
            assert has_chart_marks(page) is True
        finally:
            doc.close()

    def test_raster_chart_with_no_native_text_still_a_chart(self, tmp_path: Path) -> None:
        """A page-covering raster with ZERO native words is unaffected (density 0 < floor either way)."""
        pdf = _make_raster_page(
            tmp_path, "no_text.pdf", rect=FULL_PAGE_RECT, word_count=0, colorspace="rgb"
        )
        doc, page = _open_page(pdf)
        try:
            assert has_chart_marks(page) is True
        finally:
            doc.close()

    def test_pins_the_density_threshold_difference(self, tmp_path: Path) -> None:
        """Same coverage, density straddling RASTER_TEXT_DENSITY_MIN -- pins the exact difference."""
        raster_area = FULL_PAGE_RECT.width * FULL_PAGE_RECT.height
        below = round((RASTER_TEXT_DENSITY_MIN - 0.05) * raster_area / 10000.0)
        above = round((RASTER_TEXT_DENSITY_MIN + 0.05) * raster_area / 10000.0)

        pdf_below = _make_raster_page(tmp_path, "below.pdf", rect=FULL_PAGE_RECT, word_count=below)
        pdf_above = _make_raster_page(tmp_path, "above.pdf", rect=FULL_PAGE_RECT, word_count=above)

        doc_b, page_b = _open_page(pdf_below)
        doc_a, page_a = _open_page(pdf_above)
        try:
            assert has_chart_marks(page_b) is True, "below the density floor: still a chart"
            assert has_chart_marks(page_a) is False, (
                "above the density floor: scan/decorative, not a chart"
            )
        finally:
            doc_b.close()
            doc_a.close()

    def test_small_raster_with_dense_text_is_unaffected_by_new_gate(self, tmp_path: Path) -> None:
        """#510's small-raster gate stays intact: below SCAN_RASTER_PAGE_COVERAGE_MIN,
        dense text inside the raster does NOT suppress the chart mark -- the scan check
        only applies once the raster covers most of the page."""
        # A raster well above CHART_MIN_CLUSTER_AREA but covering well under
        # SCAN_RASTER_PAGE_COVERAGE_MIN of the page.
        rect = fitz.Rect(50, 50, 250, 250)  # 200x200 = 40000pt2
        page_area = PAGE_W * PAGE_H
        coverage = rect.width * rect.height / page_area
        assert coverage < SCAN_RASTER_PAGE_COVERAGE_MIN
        assert rect.width * rect.height >= CHART_MIN_CLUSTER_AREA

        # Pack it with dense text (density far above RASTER_TEXT_DENSITY_MIN) --
        # this would trip the scan check if the coverage gate did not guard it.
        word_count = round(10.0 * (rect.width * rect.height) / 10000.0)
        pdf = _make_raster_page(tmp_path, "small_dense.pdf", rect=rect, word_count=word_count)
        doc, page = _open_page(pdf)
        try:
            assert has_chart_marks(page) is True
        finally:
            doc.close()

    def test_constants_are_named_and_ordered(self) -> None:
        """No magic literals: both new thresholds are named module constants,
        and the density floor sits strictly between the measured anchors."""
        assert isinstance(SCAN_RASTER_PAGE_COVERAGE_MIN, float)
        assert isinstance(RASTER_TEXT_DENSITY_MIN, float)
        assert 0.21 < RASTER_TEXT_DENSITY_MIN < 1.37
