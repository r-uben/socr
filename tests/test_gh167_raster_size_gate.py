"""GH-167: an embedded raster is not a chart merely by existing.

`has_chart_marks` returned True on the first `page.get_images()` hit, with no
size, placement or semantic filter. So a logo, signature, publisher mark or
decorative photo on an otherwise clean prose page routed that page into the
chart-asset lane: shipped as a full-page PNG and audited as if chart data had
not been transcribed. The page's prose is perfectly good native text, and
calling it an untranscribed chart loses it.

The gate is the bar the VECTOR path already applies to a cluster,
`CHART_MIN_CLUSTER_AREA` -- a mark too small to be a chart cluster is too small
to be a raster chart. No new threshold is introduced, which is why this file
imports the constant rather than writing a number.

Measured on the PLACED rect rather than the pixel dimensions, and both
directions are pinned: a huge image scaled into a corner is a logo, and a small
image stretched across the page is a figure. Sizing on pixels would get both
backwards.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.figures.extractor import CHART_MIN_CLUSTER_AREA, has_chart_marks  # noqa: E402


def _png(width_px: int, height_px: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width_px, height_px), color=(30, 100, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _page_with_image(tmp_path: Path, *, rect: fitz.Rect, px: tuple[int, int] = (400, 300)):
    """A prose page carrying one raster placed at *rect*.

    The prose matters: this is the page whose text the chart lane would discard.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(30):
        page.insert_text(
            (72, 72 + i * 14), f"ordinary prose line {i} of a normal page", fontsize=10
        )
    page.insert_image(rect, stream=_png(*px))
    pdf = tmp_path / "p.pdf"
    doc.save(pdf)
    doc.close()
    return pdf


def _has_marks(pdf: Path) -> bool:
    doc = fitz.open(pdf)
    try:
        assert doc[0].get_images(), "fixture must actually embed a raster"
        return has_chart_marks(doc[0])
    finally:
        doc.close()


# A square whose area sits either side of the gate, derived from the constant
# itself so a change to the constant moves both fixtures together.
_SIDE = CHART_MIN_CLUSTER_AREA**0.5


def test_a_small_logo_does_not_make_a_prose_page_a_chart(tmp_path: Path) -> None:
    small = _SIDE / 3  # comfortably inside the gate, not a boundary probe
    pdf = _page_with_image(tmp_path / "logo", rect=fitz.Rect(72, 60, 72 + small, 60 + small))

    assert not _has_marks(pdf), (
        "a logo-sized raster routed a prose page into the chart-asset lane, "
        "where its text ships as an untranscribed full-page image"
    )


def test_a_full_size_raster_chart_still_enters_the_lane(tmp_path: Path) -> None:
    """The control. Without it, a gate that rejected every raster would satisfy
    the test above while deleting the lane."""
    big = _SIDE * 2
    pdf = _page_with_image(tmp_path / "chart", rect=fitz.Rect(72, 300, 72 + big, 300 + big))

    assert _has_marks(pdf), "a page-sized raster chart no longer reaches the chart lane"


def test_the_gate_measures_the_placement_not_the_pixels(tmp_path: Path) -> None:
    """A 2000x1500 logo scaled into a corner is still a logo.

    Sizing on `Image.width` would call this a chart, and would call the
    stretched low-resolution figure below prose -- both backwards.
    """
    small = _SIDE / 3
    pdf = _page_with_image(
        tmp_path / "bigpx", rect=fitz.Rect(72, 60, 72 + small, 60 + small), px=(2000, 1500)
    )

    assert not _has_marks(pdf), (
        "a high-resolution image placed in a corner was called a chart; the gate "
        "is reading pixels, not the page"
    )


def test_a_low_resolution_image_across_the_page_is_still_a_figure(tmp_path: Path) -> None:
    """The other half of the same asymmetry."""
    big = _SIDE * 2
    pdf = _page_with_image(
        tmp_path / "lowpx", rect=fitz.Rect(72, 300, 72 + big, 300 + big), px=(40, 30)
    )

    assert _has_marks(pdf), (
        "a low-resolution scan stretched across the page was dropped from the "
        "chart lane; the gate is reading pixels, not the page"
    )


class TestFailOpenOnUnmeasurablePlacement:
    """The deliberate fail-open branch (cubic P3 on #510).

    When a placement cannot be measured, the page keeps the pre-GH-167 answer
    and enters the chart lane. That is load-bearing -- it is what makes the gate
    narrow only on evidence -- and nothing pinned it, so a future change could
    have flipped routing silently.
    """

    def _small_image_page(self, tmp_path: Path) -> Path:
        small = _SIDE / 3
        return _page_with_image(tmp_path, rect=fitz.Rect(72, 60, 72 + small, 60 + small))

    def test_a_raising_rect_lookup_keeps_the_page_in_the_chart_lane(self, tmp_path: Path) -> None:
        pdf = self._small_image_page(tmp_path / "raises")
        assert not _has_marks(pdf), "control: this page is rejected when it CAN be measured"

        doc = fitz.open(pdf)
        try:
            page = doc[0]

            def _boom(*_a, **_k):
                raise RuntimeError("no rects for you")

            page.get_image_rects = _boom
            assert has_chart_marks(page), (
                "an unmeasurable placement dropped the page from the chart lane; "
                "the gate must narrow only on evidence, never on ignorance"
            )
        finally:
            doc.close()

    def test_an_image_resolving_to_no_rect_is_unmeasurable_too(self, tmp_path: Path) -> None:
        """cubic P2: returning `[]` is as unknown as raising.

        Treating only the raising case as unknown let a page with one unresolved
        image and one small image be rejected on evidence that never covered the
        unresolved one.
        """
        pdf = self._small_image_page(tmp_path / "empty")
        doc = fitz.open(pdf)
        try:
            page = doc[0]
            page.get_image_rects = lambda *_a, **_k: []
            assert has_chart_marks(page), (
                "an image with no resolvable placement was silently treated as "
                "measured-and-too-small"
            )
        finally:
            doc.close()
