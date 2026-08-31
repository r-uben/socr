"""GH-304b: de-rotate the table-crop and source-evidence raster lanes.

Completes GH-304 (PR #305), which de-rotated whole pages before OCR and VLM judging.
This test suite verifies:
1. Helper-level rotation derivation (upright_rotation_for):
   - Clip-local text direction versus whole-page direction
   - Clip fallback to page-level direction when the clip contains no text lines
   - Image-only / no-line blocks yielding 0
   - Exception in whole-page or clip inspection failing open to 0
   - Parity across legacy wrapper delegates (_upright_rotation_for)
2. Evasion matrix (ADR 0002 rule 3):
   - E1 (test_crop_rotation_sign_90_vs_270): 90 vs 270 quadrant correctness
   - E2 (test_mixed_page_crop_uses_clip_local_direction): mixed direction on a page
   - E3 (test_origin_clamped_crop_rotates_the_same_region): origin-clamped bbox padding
   - E4 (test_horizontal_extract_crop_png_is_byte_identical,
         test_horizontal_evidence_pixmaps_are_byte_identical):
     byte-identical renders on horizontal pages
   - E5 (test_located_crop_bbox_round_trips_in_page_space): page-space round trip
   - E6 (test_uninspectable_extract_crop_png_is_byte_identical,
         test_uninspectable_evidence_pixmaps_are_byte_identical):
     fail-open byte-identical renders for uninspectable pages/crops
   - E7 (test_source_evidence_crop_and_full_page_follow_rotation):
     source evidence crop and full-page rotation
   - E8 (test_image_locator_textless_scan_stays_unrotated): raster locator stays unrotated
   - E9: resume invalidation pinned by test_resume_source_version_gh214.py

Hermetic tests: no Ollama, no VLM provider, no GPU.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import fitz
import pytest
from PIL import Image

from socr.core.born_digital import upright_rotation_for
from socr.core.document import _upright_rotation_for as doc_upright_rotation_for
from socr.engines.base import _upright_rotation_for as base_upright_rotation_for
from socr.tables.extract import (
    CROP_PADDING_PT,
    TableCropExtractor,
)
from socr.tables.image_locate import DETECT_DPI, locate_tables_image
from socr.tables.locate import TableBox, locate_tables
from socr.tables.source_evidence import (
    _EVIDENCE_OCR_DPI,
    _render_crop_pixmap,
    build_scanned_evidence,
)

# --------------------------------------------------------------------------
# Synthetic page & document fixtures
# --------------------------------------------------------------------------


def _make_rotated_pdf(tmp_path: Path, angle: int = 90, text: str = "Rotated Cell Value") -> Path:
    """A one-page PDF whose text is rotated by `angle` degrees."""
    doc = fitz.open()
    page = doc.new_page(width=500, height=700)
    page.insert_text((200, 350), text, fontsize=11, rotate=angle)
    path = tmp_path / f"rotated_{angle}.pdf"
    doc.save(path)
    doc.close()
    return path


def _make_horizontal_pdf(tmp_path: Path, text: str = "Horizontal Prose and Table") -> Path:
    """A standard horizontal one-page PDF."""
    doc = fitz.open()
    page = doc.new_page(width=500, height=700)
    page.insert_text((100, 200), text, fontsize=11)
    path = tmp_path / "horizontal.pdf"
    doc.save(path)
    doc.close()
    return path


def _make_blank_pdf(tmp_path: Path) -> Path:
    """An empty one-page PDF with no text or images."""
    doc = fitz.open()
    doc.new_page(width=500, height=700)
    path = tmp_path / "blank.pdf"
    doc.save(path)
    doc.close()
    return path


def _make_mixed_page_pdf(tmp_path: Path) -> Path:
    """A PDF page with dominant horizontal prose and a rotated table block."""
    doc = fitz.open()
    page = doc.new_page(width=500, height=700)
    # 6 lines of horizontal body text (dominant direction = (1.0, 0.0))
    for i in range(6):
        page.insert_text(
            (50, 60 + i * 30),
            f"Horizontal paragraph text line {i + 1} across the body column.",
            fontsize=10,
        )
    # 2 lines of 90-degree rotated text in a table block region (y=400..600)
    page.insert_text((250, 450), "Table Rotated Header Col 1", fontsize=10, rotate=90)
    page.insert_text((280, 450), "Table Rotated Header Col 2", fontsize=10, rotate=90)
    path = tmp_path / "mixed.pdf"
    doc.save(path)
    doc.close()
    return path


def _make_asymmetric_pdf(tmp_path: Path) -> Path:
    """PDF with an asymmetric drawing so 0, 90, 180, and 270 rotations produce distinct rasters."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    # Asymmetrical 'F'-like solid shape in top-left
    page.draw_rect(fitz.Rect(50, 50, 150, 80), color=(0, 0, 0), fill=(0, 0, 0))
    page.draw_rect(fitz.Rect(50, 50, 80, 250), color=(0, 0, 0), fill=(0, 0, 0))
    page.draw_rect(fitz.Rect(50, 130, 120, 160), color=(0, 0, 0), fill=(0, 0, 0))
    path = tmp_path / "asymmetric.pdf"
    doc.save(path)
    doc.close()
    return path


def _make_origin_marker_pdf(tmp_path: Path) -> Path:
    """A page with distinct ink inside and outside an origin-clamped crop."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    # An asymmetric marker inside the clip expected from the near-origin bbox.
    page.draw_rect(fitz.Rect(20, 20, 95, 40), color=(0, 0, 0), fill=(0, 0, 0))
    page.draw_rect(fitz.Rect(20, 20, 42, 100), color=(0, 0, 0), fill=(0, 0, 0))
    # This marker is intentionally outside that clip. An accidentally expanded
    # crop must not be able to pass as the intended region.
    page.draw_rect(fitz.Rect(220, 20, 300, 100), color=(0, 0, 0), fill=(0, 0, 0))
    path = tmp_path / "origin_markers.pdf"
    doc.save(path)
    doc.close()
    return path


def _make_ruled_rotated_table_pdf(tmp_path: Path) -> Path:
    """PDF containing a vector-ruled table with rotated text."""
    doc = fitz.open()
    page = doc.new_page(width=500, height=700)
    # Vector grid lines
    rows = [200, 240, 280, 320, 360]
    cols = [100, 200, 300, 400]
    for y in rows:
        page.draw_line((100, y), (400, y))
    for x in cols:
        page.draw_line((x, 200), (x, 360))
    # Rotated cell contents
    page.insert_text((150, 230), "Rot Col A", fontsize=9, rotate=90)
    page.insert_text((250, 230), "Rot Col B", fontsize=9, rotate=90)
    page.insert_text((350, 230), "Rot Col C", fontsize=9, rotate=90)
    path = tmp_path / "ruled_rotated.pdf"
    doc.save(path)
    doc.close()
    return path


def _make_scanned_image_page(tmp_path: Path) -> tuple[fitz.Document, fitz.Page]:
    """A synthetic scanned page: a single raster image with no vector drawings."""
    src = fitz.open()
    p = src.new_page(width=400, height=500)
    # Draw table rules into src raster
    p.draw_line((50, 100), (350, 100))
    p.draw_line((50, 140), (350, 140))
    p.draw_line((50, 300), (350, 300))
    pix = p.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
    src.close()

    doc = fitz.open()
    page = doc.new_page(width=400, height=500)
    page.insert_image(page.rect, pixmap=pix)
    return doc, page


# --------------------------------------------------------------------------
# Test reader & proxies
# --------------------------------------------------------------------------


class _DummyTableReader:
    """Mock TableReader recording invoked image paths."""

    def __init__(self, response: str = "| A | B |\n|---|---|\n| 1 | 2 |") -> None:
        self.response = response
        self.recorded_crops: list[bytes] = []

    def read(self, image_path: Path) -> str:
        self.recorded_crops.append(Path(image_path).read_bytes())
        return self.response


class _ExplodingPageProxy:
    """Proxy that raises from get_text to test fail-open behavior."""

    def __init__(self, real_page: fitz.Page | None = None) -> None:
        self._page = real_page
        self.rect = real_page.rect if real_page is not None else fitz.Rect(0, 0, 500, 700)

    def get_text(self, *args, **kwargs):
        raise RuntimeError("inspection failure")

    def get_pixmap(self, *args, **kwargs):
        if self._page is not None:
            return self._page.get_pixmap(*args, **kwargs)
        raise RuntimeError("no pixmap")


class _ExplodingClipPageProxy:
    """Proxy that raises only when a clip is passed to get_text('dict')."""

    def __init__(self, real_page: fitz.Page) -> None:
        self._page = real_page
        self.rect = real_page.rect

    def get_text(self, *args, **kwargs):
        if kwargs.get("clip") is not None or (len(args) > 1 and args[1] is not None):
            raise RuntimeError("clipped inspection failure")
        return self._page.get_text(*args, **kwargs)

    def get_pixmap(self, *args, **kwargs):
        return self._page.get_pixmap(*args, **kwargs)


class _DirectionProxyPage:
    """Proxy delegating rendering to a real page while forcing line direction."""

    def __init__(self, real_page: fitz.Page, direction: tuple[float, float]) -> None:
        self._page = real_page
        self.rect = real_page.rect
        self._direction = direction

    def get_text(self, kind="text", **kwargs):
        if kind == "dict":
            return {
                "blocks": [
                    {
                        "type": 0,
                        "lines": [
                            {
                                "dir": self._direction,
                                "spans": [{"text": "Sample text", "bbox": (50, 50, 150, 150)}],
                            }
                        ],
                    }
                ]
            }
        return "Sample text"

    def get_drawings(self):
        return self._page.get_drawings()

    def get_pixmap(self, *args, **kwargs):
        return self._page.get_pixmap(*args, **kwargs)


# --------------------------------------------------------------------------
# Task t2: Helper-level tests for upright_rotation_for(page, clip=None)
# --------------------------------------------------------------------------


class TestUprightRotationForHelper:
    """Unit tests for upright_rotation_for(page, clip=None)."""

    def test_upright_rotation_for_clip_local_vs_page_direction(self, tmp_path: Path) -> None:
        """Clip-local blocks override whole-page direction when they contain text lines."""
        pdf_path = _make_mixed_page_pdf(tmp_path)
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            table_clip = fitz.Rect(200, 400, 350, 600)
            prose_clip = fitz.Rect(40, 50, 450, 250)

            # Whole-page dominant direction is horizontal (6 prose lines vs 2 rotated)
            assert upright_rotation_for(page) == 0

            # Table crop clip contains only 90-degree rotated lines -> derived angle 90
            assert upright_rotation_for(page, clip=table_clip) == 90

            # Prose crop clip contains only horizontal lines -> angle 0
            assert upright_rotation_for(page, clip=prose_clip) == 0

    def test_upright_rotation_for_clip_no_lines_falls_back_to_page(self, tmp_path: Path) -> None:
        """A clip containing no text lines falls back to page-level direction."""
        pdf_path = _make_rotated_pdf(tmp_path, angle=90)
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            # Empty margin with no text lines
            empty_clip = fitz.Rect(10, 10, 50, 50)

            # Page is rotated 90; clip has no text -> falls back to page-level 90
            assert upright_rotation_for(page, clip=empty_clip) == 90

    def test_upright_rotation_for_image_only_or_no_lines_yields_zero(self, tmp_path: Path) -> None:
        """Image-only or blank pages with no directional text lines yield 0."""
        pdf_path = _make_blank_pdf(tmp_path)
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            assert upright_rotation_for(page) == 0
            assert upright_rotation_for(page, clip=fitz.Rect(0, 0, 100, 100)) == 0

        image_doc, image_page = _make_scanned_image_page(tmp_path)
        try:
            assert upright_rotation_for(image_page) == 0
            assert upright_rotation_for(image_page, clip=fitz.Rect(0, 0, 100, 100)) == 0
        finally:
            image_doc.close()

    def test_upright_rotation_for_exploding_get_text_yields_zero(self) -> None:
        """An uninspectable page failing during get_text fails open to 0."""
        proxy = _ExplodingPageProxy()
        assert upright_rotation_for(proxy) == 0
        assert upright_rotation_for(proxy, clip=fitz.Rect(10, 10, 100, 100)) == 0

    def test_upright_rotation_for_uninspectable_clip_yields_zero(self, tmp_path: Path) -> None:
        """When clipped inspection raises, fail open to 0 rather than rotating from fallback."""
        pdf_path = _make_rotated_pdf(tmp_path, angle=90)
        with fitz.open(pdf_path) as doc:
            proxy = _ExplodingClipPageProxy(doc[0])
            # Whole page inspects fine (angle 90), but clipped inspection raises -> returns 0
            assert upright_rotation_for(proxy) == 90
            assert upright_rotation_for(proxy, clip=fitz.Rect(100, 100, 300, 400)) == 0

    @pytest.mark.parametrize("scenario", ["rotated", "horizontal", "blank", "uninspectable"])
    def test_legacy_rotation_wrappers_match_shared_helper(
        self, tmp_path: Path, scenario: str
    ) -> None:
        """Parity test: base and document _upright_rotation_for match upright_rotation_for."""
        if scenario == "rotated":
            with fitz.open(_make_rotated_pdf(tmp_path, angle=90)) as doc:
                page = doc[0]
                expected = upright_rotation_for(page)
                assert base_upright_rotation_for(page) == expected
                assert doc_upright_rotation_for(page) == expected
        elif scenario == "horizontal":
            with fitz.open(_make_horizontal_pdf(tmp_path)) as doc:
                page = doc[0]
                expected = upright_rotation_for(page)
                assert base_upright_rotation_for(page) == expected
                assert doc_upright_rotation_for(page) == expected
        elif scenario == "blank":
            with fitz.open(_make_blank_pdf(tmp_path)) as doc:
                page = doc[0]
                expected = upright_rotation_for(page)
                assert base_upright_rotation_for(page) == expected
                assert doc_upright_rotation_for(page) == expected
        elif scenario == "uninspectable":
            proxy = _ExplodingPageProxy()
            expected = upright_rotation_for(proxy)
            assert base_upright_rotation_for(proxy) == expected
            assert doc_upright_rotation_for(proxy) == expected


# --------------------------------------------------------------------------
# Task t5: E1 (90 vs 270), E2 (mixed direction), E3 (origin-clamped crop)
# --------------------------------------------------------------------------


class TestCropRotationEvasions:
    """Evasion tests for rotation sign, mixed page direction, and origin clamping."""

    def test_crop_rotation_sign_90_vs_270(self, tmp_path: Path) -> None:
        """E1: 90 vs 270 rotation produces distinct rasters matching their quadrants."""
        pdf_path = _make_asymmetric_pdf(tmp_path)
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            box = TableBox(bbox=(40, 40, 300, 300), source="ruled")
            page_rect = page.rect

            proxy_90 = _DirectionProxyPage(page, (0.0, -1.0))
            proxy_270 = _DirectionProxyPage(page, (0.0, 1.0))

            extractor = TableCropExtractor(reader=_DummyTableReader(), crop_dpi=150)

            path_90 = extractor._render_crop(proxy_90, box, page_rect)
            path_270 = extractor._render_crop(proxy_270, box, page_rect)
            try:
                assert path_90 is not None and path_270 is not None

                # Independent reference renders of the exact same clip.
                clip = fitz.Rect(
                    max(page_rect.x0, box.bbox[0] - CROP_PADDING_PT),
                    max(page_rect.y0, box.bbox[1] - CROP_PADDING_PT),
                    min(page_rect.x1, box.bbox[2] + CROP_PADDING_PT),
                    min(page_rect.y1, box.bbox[3] + CROP_PADDING_PT),
                )
                # Each oracle uses a fresh matrix, never chained or mutated
                pix_unrot = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), clip=clip)
                pix_90 = page.get_pixmap(
                    matrix=fitz.Matrix(150 / 72, 150 / 72).prerotate(90), clip=clip
                )
                pix_270 = page.get_pixmap(
                    matrix=fitz.Matrix(150 / 72, 150 / 72).prerotate(270), clip=clip
                )
                img_90_bytes = Image.open(path_90).tobytes()
                img_270_bytes = Image.open(path_270).tobytes()

                # 1. (0, -1) maps to a fresh 90-degree reference.
                assert img_90_bytes == pix_90.samples
                # 2. (0, 1) maps to a fresh 270-degree reference.
                assert img_270_bytes == pix_270.samples
                # 3. Each quadrant differs from the fresh legacy render.
                assert img_90_bytes != pix_unrot.samples
                assert img_270_bytes != pix_unrot.samples
                # 4. The two quadrants are not silently interchanged.
                assert img_90_bytes != img_270_bytes
            finally:
                if path_90 is not None:
                    path_90.unlink(missing_ok=True)
                if path_270 is not None:
                    path_270.unlink(missing_ok=True)

    def test_mixed_page_crop_uses_clip_local_direction(self, tmp_path: Path) -> None:
        """E2: A page with mixed direction uses clip-local rotation for the table crop."""
        pdf_path = _make_mixed_page_pdf(tmp_path)
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            box_table = TableBox(bbox=(200, 400, 350, 600), source="ruled")
            box_prose = TableBox(bbox=(40, 50, 450, 250), source="ruled")
            assert upright_rotation_for(page) == 0

            extractor = TableCropExtractor(reader=_DummyTableReader(), crop_dpi=150)

            # Render table crop
            path_table = extractor._render_crop(page, box_table, page.rect)
            # Render prose crop
            path_prose = extractor._render_crop(page, box_prose, page.rect)
            legacy_prose_path: Path | None = None
            try:
                assert path_table is not None and path_prose is not None

                # Table crop clip
                clip_table = fitz.Rect(
                    max(page.rect.x0, box_table.bbox[0] - CROP_PADDING_PT),
                    max(page.rect.y0, box_table.bbox[1] - CROP_PADDING_PT),
                    min(page.rect.x1, box_table.bbox[2] + CROP_PADDING_PT),
                    min(page.rect.y1, box_table.bbox[3] + CROP_PADDING_PT),
                )
                # Prose crop clip
                clip_prose = fitz.Rect(
                    max(page.rect.x0, box_prose.bbox[0] - CROP_PADDING_PT),
                    max(page.rect.y0, box_prose.bbox[1] - CROP_PADDING_PT),
                    min(page.rect.x1, box_prose.bbox[2] + CROP_PADDING_PT),
                    min(page.rect.y1, box_prose.bbox[3] + CROP_PADDING_PT),
                )
                assert upright_rotation_for(page, clip=clip_table) == 90
                assert upright_rotation_for(page, clip=clip_prose) == 0

                # Fresh oracle matrices for table: unrotated and 90-degree
                pix_table_unrot = page.get_pixmap(
                    matrix=fitz.Matrix(150 / 72, 150 / 72), clip=clip_table
                )
                pix_table_rot = page.get_pixmap(
                    matrix=fitz.Matrix(150 / 72, 150 / 72).prerotate(90), clip=clip_table
                )

                # Prose crop is byte-identical to the exact legacy PNG-save path.
                # Use a fresh matrix for the legacy render to avoid any mutation.
                pix_prose_legacy = page.get_pixmap(
                    matrix=fitz.Matrix(150 / 72, 150 / 72), clip=clip_prose
                )
                img_prose_legacy = Image.frombytes(
                    "RGB",
                    (pix_prose_legacy.width, pix_prose_legacy.height),
                    pix_prose_legacy.samples,
                )
                fd, name = tempfile.mkstemp(prefix="legacy_prose_", suffix=".png")
                os.close(fd)
                legacy_prose_path = Path(name)
                img_prose_legacy.save(legacy_prose_path)
                table_crop_bytes = Image.open(path_table).tobytes()
                # Table crop matches a fresh prerotated render of its unchanged clip.
                assert table_crop_bytes != pix_table_unrot.samples
                assert table_crop_bytes == pix_table_rot.samples
                assert path_prose.read_bytes() == legacy_prose_path.read_bytes()
            finally:
                if path_table is not None:
                    path_table.unlink(missing_ok=True)
                if path_prose is not None:
                    path_prose.unlink(missing_ok=True)
                if legacy_prose_path is not None:
                    legacy_prose_path.unlink(missing_ok=True)

    def test_origin_clamped_crop_rotates_the_same_region(self, tmp_path: Path) -> None:
        """E3: A crop near the origin with clamped padding rotates the exact clamped region."""
        pdf_path = _make_origin_marker_pdf(tmp_path)
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            # Bbox touching near origin: x0 - 6.0 and y0 - 6.0 cross page origin (0, 0)
            box = TableBox(bbox=(2.0, 3.0, 150.0, 180.0), source="ruled")
            extractor = TableCropExtractor(reader=_DummyTableReader(), crop_dpi=150)

            crop_path = extractor._render_crop(
                _DirectionProxyPage(page, (0.0, -1.0)), box, page.rect
            )
            assert crop_path is not None

            # Expected clamped clip
            clamped_clip = fitz.Rect(
                0.0,
                0.0,
                min(page.rect.x1, 150.0 + CROP_PADDING_PT),
                min(page.rect.y1, 180.0 + CROP_PADDING_PT),
            )

            # Fresh oracle: unrotated render of the exact clamped region
            pix_unrot = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), clip=clamped_clip)
            # Fresh oracle: 90-degree rotated render of the exact clamped region
            pix_rot = page.get_pixmap(
                matrix=fitz.Matrix(150 / 72, 150 / 72).prerotate(90), clip=clamped_clip
            )
            # Wrong expanded region that includes the outside marker
            wrong_expanded = page.get_pixmap(
                matrix=fitz.Matrix(150 / 72, 150 / 72).prerotate(90),
                clip=fitz.Rect(0, 0, 310, 186),
            )

            try:
                crop_img = Image.open(crop_path)
                crop_bytes = crop_img.tobytes()

                # 1. Production crop matches a fresh prerotated render of the exact clamped clip.
                assert crop_bytes == pix_rot.samples

                # 2. The fixture proves we did not merely rotate blank pixels.
                assert any(sample < 255 for sample in crop_bytes), (
                    "Crop should be nonblank; inside marker must be present"
                )

                # 3. Dimensions reflect the 90-degree rotation (swapped axes relative to unrotated).
                assert crop_img.width == pix_unrot.height, (
                    "Width should equal unrotated height after 90-degree rotation"
                )
                assert crop_img.height == pix_unrot.width, (
                    "Height should equal unrotated width after 90-degree rotation"
                )

                # 4. The distinct outside marker is excluded by the unchanged clip.
                # Dimensions must differ from the expanded region.
                assert (crop_img.width, crop_img.height) != (
                    wrong_expanded.width,
                    wrong_expanded.height,
                ), "Crop dimensions must differ from wrongly expanded region"
                # Bytes must differ from the expanded region (outside marker included there).
                assert crop_bytes != wrong_expanded.samples, (
                    "Crop samples must differ from expanded region (outside marker excluded)"
                )
            finally:
                crop_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Task t6: E4 (horizontal identity), E6 (fail-open), E5 (page-space round trip)
# --------------------------------------------------------------------------


class TestExtractInvariants:
    """Tests for byte-identity and page-space bbox round-tripping."""

    def test_horizontal_extract_crop_png_is_byte_identical(self, tmp_path: Path) -> None:
        """E4: Horizontal page crops produce PNG files byte-identical to the pre-fix path."""
        pdf_path = _make_horizontal_pdf(tmp_path)
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            box = TableBox(bbox=(80, 150, 400, 300), source="ruled")
            extractor = TableCropExtractor(reader=_DummyTableReader(), crop_dpi=150)

            prod_path = extractor._render_crop(page, box, page.rect)
            assert prod_path is not None
            prod_bytes = prod_path.read_bytes()

            # Exact legacy unrotated pipeline: clip -> fresh matrix -> get_pixmap -> Image -> save
            clip = fitz.Rect(
                max(page.rect.x0, box.bbox[0] - CROP_PADDING_PT),
                max(page.rect.y0, box.bbox[1] - CROP_PADDING_PT),
                min(page.rect.x1, box.bbox[2] + CROP_PADDING_PT),
                min(page.rect.y1, box.bbox[3] + CROP_PADDING_PT),
            )
            pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), clip=clip)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            fd, name = tempfile.mkstemp(prefix="legacy_crop_", suffix=".png")
            os.close(fd)
            legacy_path = Path(name)
            try:
                img.save(legacy_path)
                assert prod_bytes == legacy_path.read_bytes()
            finally:
                prod_path.unlink(missing_ok=True)
                legacy_path.unlink(missing_ok=True)

    def test_uninspectable_extract_crop_png_is_byte_identical(self, tmp_path: Path) -> None:
        """E6: Uninspectable pages fail open and produce PNG files byte-identical to legacy."""
        pdf_path = _make_horizontal_pdf(tmp_path)
        with fitz.open(pdf_path) as doc:
            real_page = doc[0]
            proxy = _ExplodingPageProxy(real_page)
            box = TableBox(bbox=(80, 150, 400, 300), source="ruled")
            extractor = TableCropExtractor(reader=_DummyTableReader(), crop_dpi=150)

            prod_path = extractor._render_crop(proxy, box, proxy.rect)
            assert prod_path is not None
            prod_bytes = prod_path.read_bytes()

            # Legacy unrotated render with fresh matrix
            clip = fitz.Rect(
                max(real_page.rect.x0, box.bbox[0] - CROP_PADDING_PT),
                max(real_page.rect.y0, box.bbox[1] - CROP_PADDING_PT),
                min(real_page.rect.x1, box.bbox[2] + CROP_PADDING_PT),
                min(real_page.rect.y1, box.bbox[3] + CROP_PADDING_PT),
            )
            pix = real_page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), clip=clip)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            fd, name = tempfile.mkstemp(prefix="legacy_crop_", suffix=".png")
            os.close(fd)
            legacy_path = Path(name)
            try:
                img.save(legacy_path)
                assert prod_bytes == legacy_path.read_bytes()
            finally:
                prod_path.unlink(missing_ok=True)
                legacy_path.unlink(missing_ok=True)

    def test_located_crop_bbox_round_trips_in_page_space(self, tmp_path: Path) -> None:
        """E5: Bboxes remain in PDF page space through localization and crop extraction."""
        pdf_path = _make_ruled_rotated_table_pdf(tmp_path)
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            boxes = locate_tables(page)
            assert len(boxes) >= 1
            box = boxes[0]

        reader = _DummyTableReader(response="| A | B |\n|---|---|\n| 1 | 2 |")
        extractor = TableCropExtractor(reader=reader, crop_dpi=150)

        # Run extraction
        crop_tables = extractor.extract(pdf_path, page_num=1, boxes=[box])
        assert len(crop_tables) == 1

        emitted_crop = crop_tables[0]
        # 1. Emitted bbox is strictly equal to the located box in page space
        assert emitted_crop.bbox == box.bbox
        # 2. Coordinates remain inside the page boundaries
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            crop_rect = fitz.Rect(emitted_crop.bbox)
            assert crop_rect in page.rect

        # 3. The recorded raster handed to the reader is the prerotated(90) crop
        assert len(reader.recorded_crops) == 1
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            clip = fitz.Rect(
                max(page.rect.x0, box.bbox[0] - CROP_PADDING_PT),
                max(page.rect.y0, box.bbox[1] - CROP_PADDING_PT),
                min(page.rect.x1, box.bbox[2] + CROP_PADDING_PT),
                min(page.rect.y1, box.bbox[3] + CROP_PADDING_PT),
            )
            assert upright_rotation_for(page, clip=clip) == 90
            # Fresh matrix oracle for the expected 90-degree rotated crop
            pix_expected = page.get_pixmap(
                matrix=fitz.Matrix(150 / 72, 150 / 72).prerotate(90), clip=clip
            )
            img = Image.open(io.BytesIO(reader.recorded_crops[0]))
            assert img.tobytes() == pix_expected.samples


# --------------------------------------------------------------------------
# Task t7: E7 (evidence rotation), E4/E6 (evidence invariants), E8 (image locator)
# --------------------------------------------------------------------------


class _RecordingScannedProxy:
    """Proxy for build_scanned_evidence fallback testing."""

    def __init__(self, real_page: fitz.Page, direction: tuple[float, float]) -> None:
        self._page = real_page
        self.rect = real_page.rect
        self._direction = direction

    def get_text(self, kind="text", **kwargs):
        if kind == "dict":
            return {
                "blocks": [
                    {
                        "type": 0,
                        "lines": [
                            {
                                "dir": self._direction,
                                "spans": [{"text": "Sample text", "bbox": (50, 50, 150, 150)}],
                            }
                        ],
                    }
                ]
            }
        # Plain get_text returns empty so plain tokens are empty and fallback triggers
        return ""

    def get_drawings(self):
        # Empty drawings so locate_tables finds no vector tables
        return []

    def get_pixmap(self, *args, **kwargs):
        return self._page.get_pixmap(*args, **kwargs)


class _MatrixRecordingPageProxy:
    """Proxy recording matrices passed to get_pixmap during raster table localization."""

    def __init__(self, page: fitz.Page) -> None:
        self._page = page
        self.rect = page.rect
        self.recorded_matrices: list[fitz.Matrix] = []
        self.recorded_pixmaps: list[tuple[bytes, int, int, int]] = []

    def get_drawings(self):
        return self._page.get_drawings()

    def get_text(self, *args, **kwargs):
        return self._page.get_text(*args, **kwargs)

    def get_pixmap(self, *args, **kwargs):
        mat = kwargs.get("matrix")
        if mat is None and args:
            mat = args[0]
        if mat is not None:
            self.recorded_matrices.append(mat)
        pix = self._page.get_pixmap(*args, **kwargs)
        self.recorded_pixmaps.append((pix.samples, pix.width, pix.height, pix.n))
        return pix


class TestSourceEvidenceAndImageLocator:
    """Tests for source evidence raster rotation and image locator invariants."""

    def test_source_evidence_crop_and_full_page_follow_rotation(self, tmp_path: Path) -> None:
        """E7: Both _render_crop_pixmap and build_scanned_evidence full-page fallback rotate."""
        pdf_path = _make_rotated_pdf(tmp_path, angle=90)
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            bbox = (50.0, 100.0, 300.0, 400.0)

            # 1. Per-crop evidence pixmap
            pix_crop = _render_crop_pixmap(page, bbox, _EVIDENCE_OCR_DPI)
            # Fresh oracle: 90-degree rotated crop
            pix_crop_expected = page.get_pixmap(
                matrix=fitz.Matrix(_EVIDENCE_OCR_DPI / 72, _EVIDENCE_OCR_DPI / 72).prerotate(90),
                clip=fitz.Rect(bbox),
            )
            # Fresh oracle: unrotated crop (for comparison)
            pix_crop_unrot = page.get_pixmap(
                matrix=fitz.Matrix(_EVIDENCE_OCR_DPI / 72, _EVIDENCE_OCR_DPI / 72),
                clip=fitz.Rect(bbox),
            )

            assert pix_crop.samples == pix_crop_expected.samples
            assert pix_crop.samples != pix_crop_unrot.samples

            # 2. Full-page evidence fallback
            recorded_pixmaps = []

            def _ocr_recorder(pix):
                recorded_pixmaps.append(pix)
                return "sample content 123"

            proxy = _RecordingScannedProxy(page, (0.0, -1.0))
            build_scanned_evidence(proxy, ocr_image_fn=_ocr_recorder)

            assert len(recorded_pixmaps) == 1
            pix_full = recorded_pixmaps[0]
            # Fresh oracle: 90-degree rotated full page
            pix_full_expected = page.get_pixmap(
                matrix=fitz.Matrix(_EVIDENCE_OCR_DPI / 72, _EVIDENCE_OCR_DPI / 72).prerotate(90)
            )
            # Fresh oracle: unrotated full page (for comparison)
            pix_full_unrot = page.get_pixmap(
                matrix=fitz.Matrix(_EVIDENCE_OCR_DPI / 72, _EVIDENCE_OCR_DPI / 72)
            )

            assert pix_full.samples == pix_full_expected.samples
            assert pix_full.samples != pix_full_unrot.samples

    def test_horizontal_evidence_pixmaps_are_byte_identical(self, tmp_path: Path) -> None:
        """E4: Horizontal page evidence renders match legacy unrotated calls."""
        pdf_path = _make_horizontal_pdf(tmp_path)
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            bbox = (50.0, 100.0, 300.0, 400.0)

            # Crop evidence
            pix_crop = _render_crop_pixmap(page, bbox, _EVIDENCE_OCR_DPI)
            # Fresh matrix oracle for crop legacy render
            legacy_crop = page.get_pixmap(
                matrix=fitz.Matrix(_EVIDENCE_OCR_DPI / 72, _EVIDENCE_OCR_DPI / 72),
                clip=fitz.Rect(bbox),
            )

            assert pix_crop.samples == legacy_crop.samples
            assert pix_crop.width == legacy_crop.width
            assert pix_crop.height == legacy_crop.height
            assert pix_crop.n == legacy_crop.n

            # Full-page fallback
            recorded_pixmaps = []

            def _ocr_recorder(pix):
                recorded_pixmaps.append(pix)
                return "sample text 123"

            proxy = _RecordingScannedProxy(page, (1.0, 0.0))
            build_scanned_evidence(proxy, ocr_image_fn=_ocr_recorder)

            assert len(recorded_pixmaps) == 1
            # Fresh matrix oracle for full-page legacy render
            legacy_full = page.get_pixmap(
                matrix=fitz.Matrix(_EVIDENCE_OCR_DPI / 72, _EVIDENCE_OCR_DPI / 72)
            )
            assert recorded_pixmaps[0].samples == legacy_full.samples
            assert recorded_pixmaps[0].width == legacy_full.width
            assert recorded_pixmaps[0].height == legacy_full.height
            assert recorded_pixmaps[0].n == legacy_full.n

    def test_uninspectable_evidence_pixmaps_are_byte_identical(self, tmp_path: Path) -> None:
        """E6: Uninspectable pages fail open to unrotated renders in source evidence."""
        pdf_path = _make_rotated_pdf(tmp_path, angle=90)
        with fitz.open(pdf_path) as doc:
            real_page = doc[0]
            proxy = _ExplodingPageProxy(real_page)
            bbox = (50.0, 100.0, 300.0, 400.0)

            # Crop evidence
            pix_crop = _render_crop_pixmap(proxy, bbox, _EVIDENCE_OCR_DPI)
            # Fresh matrix oracle for crop legacy render
            legacy_crop = real_page.get_pixmap(
                matrix=fitz.Matrix(_EVIDENCE_OCR_DPI / 72, _EVIDENCE_OCR_DPI / 72),
                clip=fitz.Rect(bbox),
            )

            assert pix_crop.samples == legacy_crop.samples
            assert pix_crop.width == legacy_crop.width
            assert pix_crop.height == legacy_crop.height
            assert pix_crop.n == legacy_crop.n

            # Full-page fallback
            recorded_pixmaps = []

            def _ocr_recorder(pix):
                recorded_pixmaps.append(pix)
                return "sample text"

            build_scanned_evidence(proxy, ocr_image_fn=_ocr_recorder)
            assert len(recorded_pixmaps) == 1
            # Fresh matrix oracle for full-page legacy render
            legacy_full = real_page.get_pixmap(
                matrix=fitz.Matrix(_EVIDENCE_OCR_DPI / 72, _EVIDENCE_OCR_DPI / 72)
            )
            assert recorded_pixmaps[0].samples == legacy_full.samples
            assert recorded_pixmaps[0].width == legacy_full.width
            assert recorded_pixmaps[0].height == legacy_full.height
            assert recorded_pixmaps[0].n == legacy_full.n

    def test_image_locator_textless_scan_stays_unrotated(self, tmp_path: Path) -> None:
        """E8: Scanned image localization stays unrotated in PDF page space."""
        pytest.importorskip("cv2")
        doc, page = _make_scanned_image_page(tmp_path)
        try:
            # 1. Textless scanned page has 0 rotation
            assert upright_rotation_for(page) == 0

            # 2. Locator uses unrotated matrix
            proxy = _MatrixRecordingPageProxy(page)
            boxes = locate_tables_image(proxy)

            assert len(boxes) >= 1
            assert all(b.source == "image" for b in boxes)
            assert all(fitz.Rect(b.bbox) in page.rect for b in boxes)

            assert len(proxy.recorded_matrices) == 1
            mat = proxy.recorded_matrices[0]
            expected_scale = DETECT_DPI / 72.0
            assert mat.a == pytest.approx(expected_scale)
            assert mat.b == pytest.approx(0.0)
            assert mat.c == pytest.approx(0.0)
            assert mat.d == pytest.approx(expected_scale)

            # Pixmap consumed by image locator matches fresh unrotated render
            pix_direct = page.get_pixmap(matrix=fitz.Matrix(expected_scale, expected_scale))
            consumed_samples, consumed_width, consumed_height, consumed_channels = (
                proxy.recorded_pixmaps[0]
            )
            assert consumed_samples == pix_direct.samples
            assert (consumed_width, consumed_height, consumed_channels) == (
                pix_direct.width,
                pix_direct.height,
                pix_direct.n,
            )
        finally:
            doc.close()
