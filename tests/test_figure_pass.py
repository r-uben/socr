import io
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("PIL")

from socr.figures.extractor import FigureExtractor  # noqa: E402


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


def _noisy_jpeg_bytes(size: tuple[int, int] = (400, 300)) -> bytes:
    """A JPEG that survives the extractor's <5KB thumbnail filter."""
    from PIL import Image

    buf = io.BytesIO()
    Image.effect_noise(size, 60).convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_pdf_with_placed_image(tmp_path: Path) -> Path:
    """A born-digital-style page: native text plus one placed raster figure."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Some native body text above the figure.")
    page.insert_image(fitz.Rect(100, 100, 300, 250), stream=_noisy_jpeg_bytes())

    pdf_path = tmp_path / "placed_image.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_fake_scan_pdf(tmp_path: Path, n_pages: int = 3) -> Path:
    """Scanned-style PDF: each page is ONE full-page raster, no text layer."""
    img = _noisy_jpeg_bytes((1700, 2200))
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=img)

    pdf_path = tmp_path / "fake_scan.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_figure_extractor_dedupes_placed_raster(tmp_path: Path) -> None:
    """Issue #42: a placed raster must come out ONCE, not once per strategy.

    Strategy 1 captures the image via its placement bbox; Strategy 2 used to
    re-extract the same xref because it never consulted the processed set.
    """
    pdf_path = _make_pdf_with_placed_image(tmp_path)

    extracted = FigureExtractor(max_total=10, max_per_page=3).extract(pdf_path)

    assert len(extracted) == 1


def test_figure_extractor_one_figure_per_scanned_page_without_skips(tmp_path: Path) -> None:
    """Issue #42: a fake scan used to yield TWO full-page figures per page."""
    pdf_path = _make_fake_scan_pdf(tmp_path, n_pages=3)

    extracted = FigureExtractor(max_total=10, max_per_page=3).extract(pdf_path)

    assert len(extracted) == 3
    assert sorted(f.page_num for f in extracted) == [1, 2, 3]


def test_figure_extractor_skip_pages(tmp_path: Path) -> None:
    """skip_pages drops listed pages entirely (scanned pages: the only
    'figure' is the page raster itself, already covered by the VLM text)."""
    pdf_path = _make_fake_scan_pdf(tmp_path, n_pages=3)

    extracted = FigureExtractor(max_total=10, max_per_page=3).extract(pdf_path, skip_pages={1, 3})

    assert [f.page_num for f in extracted] == [2]


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


def _draw_vector_chart(page, rect: fitz.Rect, color: tuple[float, float, float]) -> None:
    """Dense vector chart panel with bars, gridlines, and a line series."""
    page.draw_rect(rect, color=color, width=0.7)
    for i in range(1, 6):
        y = rect.y0 + i * rect.height / 7
        page.draw_line((rect.x0 + 8, y), (rect.x1 - 8, y), color=(0.6, 0.6, 0.6), width=0.4)
    for i in range(12):
        x = rect.x0 + 12 + i * (rect.width - 24) / 12
        height = (i % 5 + 2) * (rect.height - 22) / 8
        page.draw_rect(
            fitz.Rect(x, rect.y1 - 10 - height, x + 5, rect.y1 - 10),
            color=color,
            fill=color,
            width=0.3,
        )
    pts = []
    for i in range(12):
        x = rect.x0 + 12 + i * (rect.width - 24) / 11
        y = rect.y0 + 18 + ((i * 7) % 10) * (rect.height - 36) / 10
        pts.append(fitz.Point(x, y))
    for a, b in zip(pts, pts[1:]):
        page.draw_line(a, b, color=color, width=1.2)


def _make_vector_dashboard_pdf(tmp_path: Path) -> Path:
    """Consensus-style page: one large vector table plus three small charts.

    With the old coarse vector clustering, the header/table/charts are bridged
    into a single ~full-page region, while two real charts are below the 5%
    area gate and disappear.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Page-spanning header and a large left table. These are not figures.
    page.draw_rect(
        fitz.Rect(34, 34, 562, 52),
        color=(0.2, 0.4, 0.7),
        fill=(0.7, 0.8, 0.9),
        width=0.5,
    )
    table = fitz.Rect(34, 62, 292, 564)
    page.draw_rect(table, color=(0, 0, 0), width=0.5)
    for i in range(10):
        x = table.x0 + i * table.width / 9
        page.draw_line((x, table.y0), (x, table.y1), color=(0.65, 0.65, 0.65), width=0.3)
    for j in range(32):
        y = table.y0 + j * table.height / 31
        page.draw_line((table.x0, y), (table.x1, y), color=(0.65, 0.65, 0.65), width=0.3)

    # Small vector decorations that bridge clusters under the old 30pt gap.
    page.draw_rect(fitz.Rect(29, 568, 301, 617), color=(1, 0, 0), width=0.7)
    page.draw_line((318, 590), (558, 590), color=(0.5, 0.5, 0.5), width=0.4)
    page.draw_line((324, 612), (553, 612), color=(0.5, 0.5, 0.5), width=0.4)

    _draw_vector_chart(page, fitz.Rect(318, 519, 558, 586), (1, 0, 0))
    _draw_vector_chart(page, fitz.Rect(45, 713, 282, 792), (0, 0.6, 0))
    _draw_vector_chart(page, fitz.Rect(324, 632, 553, 807), (0, 0, 1))

    pdf_path = tmp_path / "vector_dashboard.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _has_color(img, channel: str) -> bool:
    pixels = img.convert("RGB").getdata()
    if channel == "red":
        return any(r > 180 and g < 90 and b < 90 for r, g, b in pixels)
    if channel == "green":
        return any(g > 120 and r < 90 and b < 90 for r, g, b in pixels)
    if channel == "blue":
        return any(b > 180 and r < 90 and g < 90 for r, g, b in pixels)
    raise ValueError(channel)


def test_figure_extractor_splits_vector_dashboard_charts(tmp_path: Path) -> None:
    """Issue #43: dense vector dashboards need chart crops, not a full-page PNG."""
    pdf_path = _make_vector_dashboard_pdf(tmp_path)

    extracted = FigureExtractor(max_total=10, max_per_page=5).extract(pdf_path)

    assert len(extracted) == 3
    assert all(max(fig.image.size) < 900 for fig in extracted), (
        "a chart crop should not be a full-page render"
    )
    assert any(_has_color(fig.image, "red") for fig in extracted)
    assert any(_has_color(fig.image, "green") for fig in extracted)
    assert any(_has_color(fig.image, "blue") for fig in extracted)
