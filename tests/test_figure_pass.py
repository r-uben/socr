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
