"""Pages with a ``/Rotate`` must reach every reader upright.

``upright_rotation_for`` reads the text direction from ``get_text``, in the
UNROTATED page frame. ``get_pixmap`` draws the DISPLAYED frame, which already
applies ``/Rotate``. Every render site pre-rotated by the first and rendered the
second, so a Tealbook table set horizontally on a ``/Rotate 270`` page went to
the model on its side, and a ``/Rotate 90`` page that displays upright was turned
sideways. Measured on 8 ``/Rotate 270`` Fed pages: 130 of 219 table rows wrong
sideways, 25 upright (scratch/army/fed-e2e/UPRIGHT).

The fixture makes "upright" checkable without OCR. A landscape source page
carries horizontal text and a red square in its top-left corner. It is placed on
a target page turned by each content rotation, and the target gets each
``/Rotate``. An image is upright iff it is landscape AND the red square sits in
its top-left quadrant: each of the four right-angle turns fails at least one.

Every render site that hands an image to a reader is pinned through its real
code path. Each test is a DIFFERENCE over all 16 (content, /Rotate)
combinations, so a renderer that rotated everything, or nothing, fails.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")
np = pytest.importorskip("numpy")

from PIL import Image  # noqa: E402

RIGHT_ANGLES = (0, 90, 180, 270)
COMBOS = [(c, r) for c in RIGHT_ANGLES for r in RIGHT_ANGLES]

# Source page geometry, in points. Landscape so an upright image is wider than tall.
SRC_W, SRC_H = 720, 432
MARKER = fitz.Rect(24, 24, 96, 96)
RED = (1, 0, 0)


def _source_doc():
    src = fitz.open()
    page = src.new_page(width=SRC_W, height=SRC_H)
    page.draw_rect(MARKER, color=RED, fill=RED)
    for i in range(8):
        page.insert_text((120, 60 + i * 40), "Coefficient 0.86 estimate 1.24 (0.02)", fontsize=14)
    return src


def _rotated_pdf(tmp_path: Path, content: int, rotate: int) -> Path:
    """The source page drawn turned by ``content`` degrees, on a page with ``/Rotate rotate``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = _source_doc()
    doc = fitz.open()
    w, h = (SRC_W, SRC_H) if content in (0, 180) else (SRC_H, SRC_W)
    page = doc.new_page(width=w, height=h)
    page.show_pdf_page(page.rect, src, 0, rotate=content)
    page.set_rotation(rotate)
    path = tmp_path / f"c{content}_r{rotate}.pdf"
    doc.save(path)
    doc.close()
    src.close()
    return path


def _image_upright(img: Image.Image) -> bool:
    """Landscape, with the red marker's centroid in the top-left quadrant."""
    arr = np.asarray(img.convert("RGB")).astype(int)
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
    if not red.any():
        return False
    ys, xs = np.nonzero(red)
    h, w = red.shape
    return w > h and xs.mean() < w / 2 and ys.mean() < h / 2


def _pix_image(pix) -> Image.Image:
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _upright_rotation_by_eye(page, clip=None, dpi: int = 72) -> int:
    """The prerotation that makes the render upright, found by trying all four.

    Independent of socr: it is what the checker above accepts.
    """
    hits = []
    for rot in RIGHT_ANGLES:
        mat = fitz.Matrix(dpi / 72, dpi / 72).prerotate(rot)
        if _image_upright(_pix_image(page.get_pixmap(matrix=mat, clip=clip))):
            hits.append(rot)
    assert len(hits) == 1, f"fixture ambiguous: upright at {hits}"
    return hits[0]


def _sideways_at_least_once(results: dict) -> None:
    """Guard on the fixture: the OLD rule must fail some combination, or the pin is vacuous."""
    from socr.core.born_digital import upright_rotation_for

    failures = 0
    for (content, rotate), path in results.items():
        with fitz.open(path) as doc:
            page = doc[0]
            mat = fitz.Matrix(1, 1).prerotate(upright_rotation_for(page))
            if not _image_upright(_pix_image(page.get_pixmap(matrix=mat))):
                failures += 1
    assert failures, "the old rule renders every combination upright; the fixture tests nothing"


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("upright")
    return {(c, r): _rotated_pdf(base, c, r) for c, r in COMBOS}


def test_fixture_defeats_the_old_rule(pdfs) -> None:
    _sideways_at_least_once(pdfs)


def test_every_combination_has_exactly_one_upright_render(pdfs) -> None:
    for (c, r), path in pdfs.items():
        with fitz.open(path) as doc:
            _upright_rotation_by_eye(doc[0])


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("content", "rotate"), COMBOS)
def test_render_rotation_for_makes_the_image_upright(pdfs, content, rotate) -> None:
    from socr.core.born_digital import render_rotation_for

    with fitz.open(pdfs[(content, rotate)]) as doc:
        page = doc[0]
        mat = fitz.Matrix(1, 1).prerotate(render_rotation_for(page))
        assert _image_upright(_pix_image(page.get_pixmap(matrix=mat))), (content, rotate)


@pytest.mark.parametrize("content", RIGHT_ANGLES)
def test_rotate_zero_is_upright_rotation_for_exactly(pdfs, content) -> None:
    from socr.core.born_digital import render_rotation_for, upright_rotation_for

    with fitz.open(pdfs[(content, 0)]) as doc:
        page = doc[0]
        assert render_rotation_for(page) == upright_rotation_for(page)


@pytest.mark.parametrize("kind", ["drawing", "image"])
@pytest.mark.parametrize("rotate", RIGHT_ANGLES)
def test_no_text_means_no_turn(tmp_path, rotate, kind) -> None:
    """A scan with no text layer is drawn as displayed: its /Rotate is trusted, never undone.

    ``image`` gives ``get_text("dict")`` an image block with no lines -- the shape of
    a real scan -- for which ``dominant_text_direction`` answers (1, 0).
    """
    from socr.core.born_digital import render_rotation_for

    doc = fitz.open()
    page = doc.new_page(width=SRC_W, height=SRC_H)
    if kind == "image":
        scan = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 32), False)
        scan.clear_with(200)
        page.insert_image(page.rect, pixmap=scan)
    page.draw_rect(MARKER, color=RED, fill=RED)
    page.set_rotation(rotate)
    path = tmp_path / "scan.pdf"
    doc.save(path)
    doc.close()
    with fitz.open(path) as d:
        blocks = d[0].get_text("dict").get("blocks", [])
        if kind == "image":
            assert blocks and not any(b.get("lines") for b in blocks), (
                "fixture must be an image block"
            )
        assert render_rotation_for(d[0]) == 0


def test_a_page_without_a_rotation_attribute_behaves_as_before() -> None:
    """Test doubles and odd page objects keep the pre-/Rotate behaviour (fail-open)."""
    from socr.core.born_digital import render_rotation_for, upright_rotation_for

    class _Page:
        def get_text(self, *_a, **_k):
            return {"blocks": [{"lines": [{"dir": (0.0, -1.0), "spans": [{"text": "x 1.0"}]}]}]}

    assert render_rotation_for(_Page()) == upright_rotation_for(_Page()) == 90


def test_clip_is_inspected_in_the_text_layer_frame(pdfs) -> None:
    """The clip goes to the direction inspection unchanged (get_text filters unrotated)."""
    from socr.core.born_digital import render_rotation_for

    with fitz.open(pdfs[(0, 270)]) as doc:
        page = doc[0]
        assert render_rotation_for(page, clip=page.mediabox) == render_rotation_for(page)


# ---------------------------------------------------------------------------
# Every render site, through its real code path
# ---------------------------------------------------------------------------

BIG_BOX = (0.0, 0.0, 10_000.0, 10_000.0)  # clamped to page.rect: the whole displayed page


def _assert_all_upright(images: dict) -> None:
    bad = [k for k, img in images.items() if not _image_upright(img)]
    assert not bad, f"sideways or flipped for (content, /Rotate): {bad}"


def test_process_pages_hands_the_engine_an_upright_page(pdfs, monkeypatch) -> None:
    from socr.core.config import PipelineConfig
    from socr.engines.base import BaseEngine

    class _Stub(BaseEngine):
        name = "deepseek"
        cli_command = "true"

        def _build_command(self, pdf_path, output_dir, config):
            return ["true"]

    captured: list[Image.Image] = []
    original = Image.Image.save

    def _save(self, fp, *args, **kwargs):
        captured.append(self.copy())
        return original(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", _save)
    images = {}
    for key, path in pdfs.items():
        captured.clear()
        _Stub().process_pages(path, [1], PipelineConfig(quiet=True), dpi=72)
        assert captured, "process_pages rendered nothing"
        images[key] = captured[0]
    _assert_all_upright(images)


def _handle(pdf: Path):
    from socr.core.document import DocumentHandle

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DocumentHandle, "__post_init__", lambda self: None)
        return DocumentHandle(path=pdf, page_count=1)


def test_render_page_the_judge_reads_is_upright(pdfs) -> None:
    _assert_all_upright({k: _handle(p).render_page(1, dpi=72) for k, p in pdfs.items()})


def test_render_all_pages_for_hpc_is_upright(pdfs) -> None:
    _assert_all_upright({k: _handle(p).render_all_pages(dpi=72)[1] for k, p in pdfs.items()})


def test_review_html_page_image_is_upright(pdfs) -> None:
    import base64

    from socr.review.html import _render_page_image

    images = {}
    for key, path in pdfs.items():
        with fitz.open(path) as doc:
            data, error = _render_page_image(doc, 0, scale=1.0, quality=95)
        assert not error, error
        images[key] = Image.open(io.BytesIO(base64.b64decode(data)))
    _assert_all_upright(images)


def test_chart_page_png_is_upright(pdfs, tmp_path) -> None:
    from socr.pipeline.orchestrator import UnifiedPipeline

    images = {}
    for key, path in pdfs.items():
        out = Path(UnifiedPipeline._render_chart_page_png(None, path, 1, tmp_path / f"{key}"))
        images[key] = Image.open(out)
    _assert_all_upright(images)


def test_adjudication_crop_is_upright(pdfs) -> None:
    from socr.pipeline.orchestrator import UnifiedPipeline

    images = {}
    for key, path in pdfs.items():
        out = UnifiedPipeline._render_adjudication_crop(None, path, 1, BIG_BOX)
        assert out is not None
        images[key] = Image.open(out).copy()
        Path(out).unlink()
    _assert_all_upright(images)


def test_table_crop_for_the_reader_is_upright(pdfs) -> None:
    from socr.tables.extract import TableCropExtractor
    from socr.tables.locate import TableBox

    extractor = TableCropExtractor(reader=None, crop_dpi=72)
    images = {}
    for key, path in pdfs.items():
        with fitz.open(path) as doc:
            page = doc[0]
            out = extractor._render_crop(page, TableBox(bbox=BIG_BOX, source="test"), page.rect)
        assert out is not None
        images[key] = Image.open(out).copy()
        Path(out).unlink()
    _assert_all_upright(images)


def test_table_witness_crop_is_upright(pdfs) -> None:
    from socr.tables.locate import TableBox
    from socr.tables.witness import _render_crop_safe

    images = {}
    for key, path in pdfs.items():
        out = _render_crop_safe(path, 1, TableBox(bbox=BIG_BOX, source="test"), 72)
        assert out is not None
        images[key] = Image.open(out).copy()
        Path(out).unlink()
    _assert_all_upright(images)


def test_evidence_crop_pixmap_is_upright(pdfs) -> None:
    from socr.tables.source_evidence import _render_crop_pixmap

    images = {}
    for key, path in pdfs.items():
        with fitz.open(path) as doc:
            page = doc[0]
            images[key] = _pix_image(_render_crop_pixmap(page, tuple(page.rect), 72))
    _assert_all_upright(images)


def test_evidence_full_page_witness_reads_an_upright_page(pdfs, monkeypatch) -> None:
    from socr.tables import source_evidence

    # No table box and no text layer: the full-page witness is the only reading.
    monkeypatch.setattr(source_evidence, "locate_tables", lambda page: [], raising=False)
    import socr.tables.locate as locate

    monkeypatch.setattr(locate, "locate_tables", lambda page: [])
    images = {}
    for key, path in pdfs.items():
        seen: list = []

        def _reader(pix, _seen=seen):
            _seen.append(_pix_image(pix))
            return ""

        with fitz.open(path) as doc:
            source_evidence.build_scanned_evidence(
                doc[0], ocr_image_fn=_reader, include_text_layer=False
            )
        assert seen, "the full-page witness never ran"
        images[key] = seen[-1]
    _assert_all_upright(images)


def test_gemma_crops_are_upright(pdfs, tmp_path, monkeypatch) -> None:
    from socr.routing import gemma_crops

    def _full_page(page):
        return gemma_crops.GemmaCropPlan(
            clips=(("full", fitz.Rect(page.rect)),), full_page_unverified=True
        )

    monkeypatch.setattr(gemma_crops, "plan_gemma_page_crops", _full_page)
    images = {}
    for key, path in pdfs.items():
        stems, _ = gemma_crops.render_gemma_crops(path, 1, tmp_path / f"{key}", dpi=72)
        images[key] = Image.open(stems[0][1])
    _assert_all_upright(images)


def test_chart_region_crop_is_upright_and_matches_its_digest(pdfs, tmp_path) -> None:
    """The shipped crop is upright, and ``crop_digest`` names the same bytes."""
    import hashlib

    from socr.figures.chart_reader import crop_digest
    from socr.figures.chart_regions import render_chart_region_crops
    from socr.figures.extractor import RENDER_DPI

    images = {}
    for key, path in pdfs.items():
        with fitz.open(path) as doc:
            page = doc[0]
            rect = fitz.Rect(page.rect)
            digest, _, _ = crop_digest(page, rect, RENDER_DPI)
            by_eye = _upright_rotation_by_eye(page, clip=rect, dpi=RENDER_DPI)
            mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72).prerotate(by_eye)
            expected = hashlib.sha256(
                page.get_pixmap(matrix=mat, clip=rect).tobytes("png")
            ).hexdigest()
        assert digest == expected, f"crop_digest does not name an upright crop for {key}"
        assets = render_chart_region_crops(path, 1, [rect], tmp_path / f"{key}")
        assert assets and assets[0].rendered, assets
        images[key] = Image.open(tmp_path / f"{key}" / assets[0].rel_path.split("/")[-1])
    _assert_all_upright(images)
