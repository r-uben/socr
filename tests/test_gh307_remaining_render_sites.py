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

All four are pinned. The chart REGION clip was left unwitnessed by #425 --
the method renders nothing without a real chart cluster, and the first
attempt skipped rather than build one -- and is covered here (#427).
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


def _chart_region_pdf(tmp_path: Path, rotate: int, name: str = "chart.pdf") -> Path:
    """A page carrying a real chart CLUSTER, with its labels at *rotate* degrees.

    ``_render_chart_region_pngs`` renders nothing unless ``chart_region_bboxes``
    finds a qualifying cluster, so the earlier attempt at this pin had to skip.
    Filled bars give it one: the union-find cluster clears
    ``_CHART_MIN_CLUSTER_AREA_PT2`` and ``_has_filled_rects_or_thick_strokes``.

    Only the label direction changes between the two variants -- the bars, and
    therefore the bbox the clip is taken from, are identical. So the difference
    the test measures can only come from the clip-scoped derotate.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(6):
        x = 100 + i * 40
        page.draw_rect(
            fitz.Rect(x, 400 - i * 25, x + 30, 500), color=(0, 0, 0), fill=(0.2, 0.2, 0.2)
        )
    for i in range(6):
        page.insert_text((105 + i * 40, 520), f"20{10 + i}", fontsize=8, rotate=rotate)
    for j in range(5):
        page.insert_text((80, 100 + j * 14), f"prose line {j}", fontsize=9)
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


def _region_clip_rotation(pdf: Path) -> tuple[int, object]:
    """The rotation and bbox the production code will actually see."""
    from socr.core.born_digital import upright_rotation_for
    from socr.tables.reconstruct import chart_region_bboxes

    doc = fitz.open(pdf)
    try:
        bboxes = chart_region_bboxes(doc[0])
        assert bboxes, "fixture must produce a chart region cluster"
        return upright_rotation_for(doc[0], clip=bboxes[0]), bboxes[0]
    finally:
        doc.close()


class TestChartRenders:
    """GH-425 review: the PR claimed four sites; only two were pinned.

    These two matter most of the four. The chart lane records "data values not
    transcribed" and hands the reader the PNG instead, so a sideways image is
    not a degraded output -- it is the entire output, unreadable.
    """

    def _pipeline(self):
        from socr.core.config import EngineType, PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        return UnifiedPipeline(
            PipelineConfig(
                primary_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                quiet=True,
            )
        )

    def test_a_sideways_chart_page_png_is_saved_upright(self, tmp_path: Path) -> None:
        from PIL import Image

        sideways = _sideways_pdf(tmp_path / "cs")
        assert _measured_rotation(sideways) != 0, "fixture must actually be sideways"

        figures = tmp_path / "figs"
        figures.mkdir()
        saved = self._pipeline()._render_chart_page_png(sideways, 1, figures)

        with Image.open(saved) as img:
            assert img.width > img.height, (
                f"chart PNG saved {img.width}x{img.height}; a derotated sideways "
                "page must come back landscape, and this image IS the page's "
                "whole payload"
            )

    def test_an_upright_chart_page_png_is_left_alone(self, tmp_path: Path) -> None:
        """Difference control: a renderer that rotated everything would satisfy
        the test above."""
        from PIL import Image

        horizontal = _horizontal_pdf(tmp_path / "ch")
        assert _measured_rotation(horizontal) == 0

        figures = tmp_path / "figs2"
        figures.mkdir()
        saved = self._pipeline()._render_chart_page_png(horizontal, 1, figures)

        with Image.open(saved) as img:
            assert img.height > img.width, "an upright portrait page must stay portrait"


class TestChartRegionClip:
    """The fourth site: the chart REGION clip in ``_render_chart_region_pngs``.

    Left unwitnessed by #425 and filed as #427. A region can run against the
    page's dominant direction, so the derotate here is clip-scoped rather than
    page-scoped -- which is exactly what a page-level pin would not catch.
    """

    def _pipeline(self):
        from socr.core.config import EngineType, PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        return UnifiedPipeline(
            PipelineConfig(
                primary_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                quiet=True,
            )
        )

    def _render(self, pdf: Path, figures: Path):
        from PIL import Image

        native_text = "![chart region 1](chart_region_p1_1.png)"
        out = self._pipeline()._render_chart_region_pngs(pdf, 1, native_text, figures)
        assert out != native_text, (
            "the placeholder was not rewritten, so no PNG was rendered and the "
            "assertion below would be measuring nothing"
        )
        saved = figures / "chart_region_p1_1.png"
        assert saved.exists(), "placeholder rewritten but no file on disk"
        with Image.open(saved) as img:
            return img.width, img.height

    def test_a_sideways_region_clip_is_rendered_upright(self, tmp_path: Path) -> None:
        pdf = _chart_region_pdf(tmp_path / "rs", rotate=90)
        rotation, bbox = _region_clip_rotation(pdf)
        assert rotation != 0, "fixture region must actually read sideways"
        assert bbox.width > bbox.height, "fixture bbox must be landscape to start"

        width, height = self._render(pdf, tmp_path / "figs_rs")
        assert height > width, (
            f"region PNG saved {width}x{height}; the clip is landscape on the "
            "page, so a derotated sideways region must come back portrait"
        )

    def test_an_upright_region_clip_is_left_alone(self, tmp_path: Path) -> None:
        """Difference control: a renderer that rotated every clip would satisfy
        the test above without derotating anything."""
        pdf = _chart_region_pdf(tmp_path / "ru", rotate=0)
        rotation, bbox = _region_clip_rotation(pdf)
        assert rotation == 0, "control region must read horizontally"
        assert bbox.width > bbox.height

        width, height = self._render(pdf, tmp_path / "figs_ru")
        assert width > height, "an upright landscape clip must stay landscape"
