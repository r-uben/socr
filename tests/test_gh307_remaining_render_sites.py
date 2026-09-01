"""GH-307: every render site must derotate, not just the two GH-305 fixed.

BaseEngine OCR and the VLM judge agreed on orientation after GH-304/305. Four
siblings did not, and each one matters for a different reason:

- ``DocumentHandle.render_all_pages`` feeds HPC OCR straight through
  ``_page_images``, so HPC still OCR'd the sideways page GH-304 measured.
- The review HTML is the JUDGEMENT instrument -- a sideways PDF beside an
  upright extract asks a human to compare two things that cannot agree.
- The chart-lane PNG is the ONLY artifact carrying a chart page's semantics: the
  lane records "data values not transcribed" and hands over the image instead.
- Chart REGION clips, same, scoped to a region that can run against the page's
  dominant direction.

Pinned as DIFFERENCES against the same page rendered upright: a sideways page
must come back with swapped dimensions, and an already-horizontal page must be
untouched. Asserting pixel content would pin the renderer; asserting the
difference pins the derotation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")


def _sideways_pdf(tmp_path: Path, name: str = "sideways.pdf") -> Path:
    """A page whose dominant text direction reads upward (90 degrees)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(8):
        page.insert_text(
            (200 + i * 18, 500),
            "Coefficient p-value 0.86 estimate",
            fontsize=11,
            rotate=90,
        )
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


def _horizontal_pdf(tmp_path: Path, name: str = "horizontal.pdf") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(8):
        page.insert_text((72, 100 + i * 18), "Coefficient p-value 0.86 estimate", fontsize=11)
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


def _measured_rotation(pdf: Path) -> int:
    from socr.core.born_digital import upright_rotation_for

    doc = fitz.open(pdf)
    try:
        return upright_rotation_for(doc[0])
    finally:
        doc.close()


class TestRenderAllPages:
    def test_a_sideways_page_is_rendered_upright(self, tmp_path: Path) -> None:
        from socr.core.document import DocumentHandle

        sideways = _sideways_pdf(tmp_path / "s")
        assert _measured_rotation(sideways) != 0, "fixture must actually be sideways"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(DocumentHandle, "__post_init__", lambda self: None)
            handle = DocumentHandle(path=sideways, page_count=1)
            img = handle.render_all_pages(dpi=72)[1]

        # A 90-degree render swaps the page's aspect: portrait becomes landscape.
        assert img.width > img.height, (
            f"page rendered {img.width}x{img.height}; a derotated sideways page "
            "must come back landscape"
        )

    def test_a_horizontal_page_is_left_alone(self, tmp_path: Path) -> None:
        """Difference control: without this, a renderer that rotated everything
        would satisfy the test above."""
        from socr.core.document import DocumentHandle

        horizontal = _horizontal_pdf(tmp_path / "h")
        assert _measured_rotation(horizontal) == 0

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(DocumentHandle, "__post_init__", lambda self: None)
            handle = DocumentHandle(path=horizontal, page_count=1)
            img = handle.render_all_pages(dpi=72)[1]

        assert img.height > img.width, "an upright portrait page must stay portrait"


class TestReviewHtmlRender:
    def test_the_judgement_instrument_shows_an_upright_page(self, tmp_path: Path) -> None:
        import base64
        import io

        from PIL import Image

        from socr.review.html import _render_page_image

        sideways = _sideways_pdf(tmp_path / "rs")
        assert _measured_rotation(sideways) != 0

        doc = fitz.open(sideways)
        try:
            data, error = _render_page_image(doc, 0, scale=1.0, quality=70)
        finally:
            doc.close()

        assert not error, f"render failed: {error}"
        raw = base64.b64decode(data.split(",", 1)[1]) if "," in data else base64.b64decode(data)
        img = Image.open(io.BytesIO(raw))
        assert img.width > img.height, (
            "the review page image is still sideways; it is the instrument a "
            "human judges the extract against"
        )
