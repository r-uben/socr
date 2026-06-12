import io
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")
PIL = pytest.importorskip("PIL")

from socr.core.config import PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FigureInfo
from socr.figures.extractor import FigureExtractor


def _make_pdf_with_image(tmp_path: Path) -> Path:
    img_bytes = io.BytesIO()
    from PIL import Image

    img = Image.new("RGB", (200, 100), color="red")
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=img_bytes)

    pdf_path = tmp_path / "with_image.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_figure_extractor_finds_images(tmp_path: Path) -> None:
    pdf_path = _make_pdf_with_image(tmp_path)

    figures_dir = tmp_path / "figures"
    extractor = FigureExtractor(max_total=5, max_per_page=3, save_dir=figures_dir)
    extracted = extractor.extract(pdf_path)

    assert len(extracted) >= 1
    assert extracted[0].page_num == 1
    assert extracted[0].figure_num >= 1


def _make_rotated_vector_pdf(tmp_path: Path) -> Path:
    """A portrait mediabox page rotated 90deg with a red vector 'chart'.

    The chart lives at unrotated coords x:[60,360], y:[500,700]. Through the
    rotation matrix that lands at rotated coords x:[92,292], y:[60,360] —
    chosen so a clip taken in the WRONG (unrotated) space shares no area with
    the chart's true rendered position, and few enough drawings (<10) that
    the landscape presentation fallback cannot mask a bad crop.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    red = (1, 0, 0)
    page.draw_rect(fitz.Rect(60, 500, 360, 700), color=red, fill=red)
    for i in range(5):
        y = 520 + i * 35
        page.draw_line(fitz.Point(70, y), fitz.Point(350, y), color=red, width=2)
    page.set_rotation(90)

    pdf_path = tmp_path / "rotated_chart.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _has_red(img) -> bool:
    pixels = img.convert("RGB").getdata()
    return any(r > 180 and g < 90 and b < 90 for r, g, b in pixels)


def test_figure_extractor_crops_vector_figure_on_rotated_page(tmp_path: Path) -> None:
    """Issue #41: get_drawings() rects are unrotated; the render clip is not.

    On a rotated page the cluster bbox must be mapped through
    page.rotation_matrix before clipping, or the crop lands in the wrong
    place (bottom sliver of the chart + swallowed prose on real reports).
    The extracted figure must actually contain the red chart.
    """
    pdf_path = _make_rotated_vector_pdf(tmp_path)

    extractor = FigureExtractor(max_total=5, max_per_page=3)
    extracted = extractor.extract(pdf_path)

    assert len(extracted) >= 1
    assert any(_has_red(fig.image) for fig in extracted), (
        "no extracted figure contains the chart: the crop was taken in "
        "unrotated coordinates on a rotated page"
    )
