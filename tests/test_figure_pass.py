import io
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("PIL")

from socr.figures.extractor import (  # noqa: E402
    LOGO_HEIGHT_RATIO,
    LOGO_TOP_MARGIN_RATIO,
    LOGO_WIDE_WIDTH_RATIO,
    ExtractedFigure,
    ExtractionResult,
    FigureExtractor,
)


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
    result = extractor.extract(pdf_path)

    assert isinstance(result, ExtractionResult)
    assert len(result.figures) >= 1
    assert result.figures[0].page_num == 1
    assert result.figures[0].figure_num >= 1


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

    result = FigureExtractor(max_total=10, max_per_page=3).extract(pdf_path)

    assert len(result.figures) == 1


def test_figure_extractor_one_figure_per_scanned_page_without_skips(tmp_path: Path) -> None:
    """Issue #42: a fake scan used to yield TWO full-page figures per page."""
    pdf_path = _make_fake_scan_pdf(tmp_path, n_pages=3)

    result = FigureExtractor(max_total=10, max_per_page=3).extract(pdf_path)

    assert len(result.figures) == 3
    assert sorted(f.page_num for f in result.figures) == [1, 2, 3]


def test_figure_extractor_skip_pages(tmp_path: Path) -> None:
    """skip_pages drops listed pages entirely (scanned pages: the only
    'figure' is the page raster itself, already covered by the VLM text)."""
    pdf_path = _make_fake_scan_pdf(tmp_path, n_pages=3)

    result = FigureExtractor(max_total=10, max_per_page=3).extract(pdf_path, skip_pages={1, 3})

    assert [f.page_num for f in result.figures] == [2]


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
    result = extractor.extract(pdf_path)

    assert len(result.figures) >= 1
    assert any(_has_red(fig.image) for fig in result.figures), (
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

    result = FigureExtractor(max_total=10, max_per_page=5).extract(pdf_path)

    assert len(result.figures) == 3
    assert all(max(fig.image.size) < 900 for fig in result.figures), (
        "a chart crop should not be a full-page render"
    )
    assert any(_has_color(fig.image, "red") for fig in result.figures)
    assert any(_has_color(fig.image, "green") for fig in result.figures)
    assert any(_has_color(fig.image, "blue") for fig in result.figures)


# ---------------------------------------------------------------------------
# GH-47A: cap-reached signal
# ---------------------------------------------------------------------------


def _make_multi_page_image_pdf(tmp_path: Path, n_pages: int) -> Path:
    """One substantial embedded image per page (survives the 5KB filter)."""
    img_bytes = _noisy_jpeg_bytes((400, 300))
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=612, height=792)
        # Place in the middle of the page so the logo filter does not touch it.
        page.insert_image(fitz.Rect(50, 200, 400, 500), stream=img_bytes)
    pdf_path = tmp_path / "multi_page_images.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_cap_reached_flag_set_when_limit_hit(tmp_path: Path) -> None:
    """GH-47A: ExtractionResult.cap_reached is True when max_total is hit.

    Build a 5-page PDF, cap at 2 figures.  The extractor must stop early and
    flag cap_reached so the orchestrator can emit a durable audit event.
    """
    pdf_path = _make_multi_page_image_pdf(tmp_path, n_pages=5)

    result = FigureExtractor(max_total=2, max_per_page=1).extract(pdf_path)

    assert result.cap_reached, "cap_reached must be True when max_total is reached"
    assert len(result.figures) == 2


def test_cap_reached_flag_not_set_when_under_limit(tmp_path: Path) -> None:
    """GH-47A: cap_reached must be False when all figures fit within max_total."""
    pdf_path = _make_multi_page_image_pdf(tmp_path, n_pages=2)

    result = FigureExtractor(max_total=10, max_per_page=3).extract(pdf_path)

    assert not result.cap_reached, "cap_reached must be False when cap was not hit"


def test_cap_reached_logged_as_warning(tmp_path: Path, caplog) -> None:
    """GH-47A: hitting the cap emits a WARNING-level log line."""
    import logging

    pdf_path = _make_multi_page_image_pdf(tmp_path, n_pages=4)

    with caplog.at_level(logging.WARNING, logger="socr.figures.extractor"):
        FigureExtractor(max_total=1, max_per_page=1).extract(pdf_path)

    assert any(
        "cap" in record.message.lower() or "max_total" in record.message.lower()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ), "Expected a WARNING-level log about the figure cap being reached"


# ---------------------------------------------------------------------------
# GH-47A: logo / letterhead false-positive filter
# ---------------------------------------------------------------------------


def _make_pdf_with_banner_logo(tmp_path: Path) -> Path:
    """Title page with a wide, short banner image at the top of the page.

    Geometry chosen so all three logo conditions hold:
    - y0 < page_height * LOGO_TOP_MARGIN_RATIO
    - width  >= page_width  * LOGO_WIDE_WIDTH_RATIO
    - height <= page_height * LOGO_HEIGHT_RATIO
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page_h = 792.0
    page_w = 612.0

    # Banner: spans >50% of width, sits in top 20%, height <15% of page.
    banner_y0 = page_h * 0.02  # well inside the top band
    banner_y1 = banner_y0 + page_h * 0.08  # short (8% of page height)
    banner_x0 = page_w * 0.05
    banner_x1 = page_w * 0.95  # 90% of page width

    img_bytes = _noisy_jpeg_bytes((int(banner_x1 - banner_x0), int(banner_y1 - banner_y0)))
    page.insert_image(fitz.Rect(banner_x0, banner_y0, banner_x1, banner_y1), stream=img_bytes)

    pdf_path = tmp_path / "logo_banner.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_pdf_with_top_figure(tmp_path: Path) -> Path:
    """Page with a genuine substantive figure near the top.

    The figure is tall enough (>LOGO_HEIGHT_RATIO) that it should NOT be
    suppressed by the logo filter, even though it sits at the top of the page.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page_h = 792.0
    page_w = 612.0

    # Tall figure: top-of-page but 40% of page height -- fails the height gate.
    fig_y0 = page_h * 0.05
    fig_y1 = fig_y0 + page_h * 0.40  # 40% tall -- above LOGO_HEIGHT_RATIO (0.15)
    fig_x0 = page_w * 0.10
    fig_x1 = page_w * 0.90

    img_bytes = _noisy_jpeg_bytes((int(fig_x1 - fig_x0), int(fig_y1 - fig_y0)))
    page.insert_image(fitz.Rect(fig_x0, fig_y0, fig_x1, fig_y1), stream=img_bytes)

    pdf_path = tmp_path / "top_figure.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_pdf_with_narrow_top_figure(tmp_path: Path) -> Path:
    """Page with a narrow figure at the top of the page.

    The figure is narrow (< LOGO_WIDE_WIDTH_RATIO of page width), so even
    though it is short and in the top band it must NOT be suppressed.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page_h = 792.0
    page_w = 612.0

    # Narrow figure: short but only 40% of page width -- fails the width gate.
    fig_y0 = page_h * 0.02
    fig_y1 = fig_y0 + page_h * 0.08
    fig_x0 = page_w * 0.30
    fig_x1 = page_w * 0.70  # 40% width -- below LOGO_WIDE_WIDTH_RATIO (0.50)

    img_bytes = _noisy_jpeg_bytes((int(fig_x1 - fig_x0), int(fig_y1 - fig_y0)))
    page.insert_image(fitz.Rect(fig_x0, fig_y0, fig_x1, fig_y1), stream=img_bytes)

    pdf_path = tmp_path / "narrow_top_figure.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_logo_banner_at_top_is_suppressed(tmp_path: Path) -> None:
    """GH-47A: a wide, short image at the very top of the page is not emitted.

    The canonical title-page letterhead / institutional logo must be suppressed
    so it does not appear as figure_1 in the output.
    """
    pdf_path = _make_pdf_with_banner_logo(tmp_path)

    result = FigureExtractor(max_total=10, max_per_page=3).extract(pdf_path)

    assert len(result.figures) == 0, (
        "Expected 0 figures: the banner logo should be suppressed by the logo filter. "
        f"Got {len(result.figures)} figure(s)."
    )


def test_tall_top_figure_is_not_suppressed(tmp_path: Path) -> None:
    """GH-47A: a tall figure at the top of the page survives the logo filter.

    Conservative filter: a genuine chart/inset that happens to be at the top
    but is taller than LOGO_HEIGHT_RATIO must pass through unchanged.
    """
    pdf_path = _make_pdf_with_top_figure(tmp_path)

    result = FigureExtractor(max_total=10, max_per_page=3).extract(pdf_path)

    assert len(result.figures) >= 1, (
        "Expected >= 1 figure: a tall top-of-page figure must survive the logo filter."
    )


def test_narrow_top_figure_is_not_suppressed(tmp_path: Path) -> None:
    """GH-47A: a narrow figure at the top of the page survives the logo filter.

    A figure that is narrow (< LOGO_WIDE_WIDTH_RATIO) is not a banner logo,
    even if it sits in the top band and is short.
    """
    pdf_path = _make_pdf_with_narrow_top_figure(tmp_path)

    result = FigureExtractor(max_total=10, max_per_page=3).extract(pdf_path)

    assert len(result.figures) >= 1, (
        "Expected >= 1 figure: a narrow top-of-page figure must survive the logo filter."
    )


def test_logo_filter_thresholds_are_named_constants() -> None:
    """GH-47A: logo filter parameters must be importable named constants.

    This is a smoke-test that the constants are exported and have reasonable
    values so reviewers can see the filter at a glance without reading code.
    """
    assert 0.0 < LOGO_TOP_MARGIN_RATIO < 0.5, (
        f"LOGO_TOP_MARGIN_RATIO={LOGO_TOP_MARGIN_RATIO} out of expected range"
    )
    assert 0.0 < LOGO_WIDE_WIDTH_RATIO < 1.0, (
        f"LOGO_WIDE_WIDTH_RATIO={LOGO_WIDE_WIDTH_RATIO} out of expected range"
    )
    assert 0.0 < LOGO_HEIGHT_RATIO < 0.5, (
        f"LOGO_HEIGHT_RATIO={LOGO_HEIGHT_RATIO} out of expected range"
    )


# ---------------------------------------------------------------------------
# GH-47C: bbox persistence on ExtractedFigure and recoverable-label audit
# ---------------------------------------------------------------------------


def _make_pdf_with_vector_labels(tmp_path: Path) -> Path:
    """Born-digital-style page: a vector chart region with native text labels inside it.

    The chart region is at (100, 100, 400, 500); axis labels and a legend entry
    are placed as native text inside that region so ``page.get_text("words")``
    can recover them.  A body paragraph sits above the figure region (outside
    the bbox) and must NOT appear in the recoverable-label set.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    # Draw a prominent vector chart (enough drawings to trigger Strategy 0).
    red = (0.8, 0.1, 0.1)
    chart = fitz.Rect(100, 100, 400, 500)
    page.draw_rect(chart, color=red, width=1.0)
    # Gridlines + bar data marks.
    for i in range(1, 6):
        y = chart.y0 + i * chart.height / 6
        page.draw_line((chart.x0 + 5, y), (chart.x1 - 5, y), color=(0.6, 0.6, 0.6), width=0.4)
    for i in range(8):
        x = chart.x0 + 12 + i * (chart.width - 24) / 8
        h = (i % 4 + 2) * (chart.height - 30) / 8
        page.draw_rect(
            fitz.Rect(x, chart.y1 - 15 - h, x + 10, chart.y1 - 15),
            color=red,
            fill=red,
            width=0.3,
        )

    # Native axis/legend text INSIDE the figure region.
    page.insert_text(fitz.Point(110, 490), "Quarters", fontsize=8)
    page.insert_text(fitz.Point(110, 120), "EuroArea", fontsize=8)
    page.insert_text(fitz.Point(200, 300), "GDP", fontsize=9)

    # Body paragraph OUTSIDE the figure region (above it).
    page.insert_text(fitz.Point(72, 60), "This paragraph is above the chart.", fontsize=10)

    pdf_path = tmp_path / "vector_labels.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_pdf_with_raster_only_figure(tmp_path: Path) -> Path:
    """Page with a single rasterized embedded image: no native text inside the figure rect."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # Placed raster image; no text operators inside the rect.
    page.insert_image(fitz.Rect(100, 200, 400, 500), stream=_noisy_jpeg_bytes())
    pdf_path = tmp_path / "raster_only.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_extracted_figure_has_bbox_field() -> None:
    """GH-47C step-0: ExtractedFigure.bbox exists and defaults to None."""
    from PIL import Image

    img = Image.new("RGB", (10, 10))
    fig = ExtractedFigure(figure_num=1, page_num=1, image=img)
    assert hasattr(fig, "bbox"), "ExtractedFigure must have a bbox field"
    assert fig.bbox is None, "bbox must default to None"

    fig2 = ExtractedFigure(figure_num=2, page_num=1, image=img, bbox=(10.0, 20.0, 100.0, 200.0))
    assert fig2.bbox == (10.0, 20.0, 100.0, 200.0)


def test_figure_info_has_bbox_field() -> None:
    """GH-47C step-0: FigureInfo.bbox exists, round-trips through to_dict/from_dict."""
    from socr.core.result import FigureInfo

    fi = FigureInfo(figure_num=1, page_num=2, figure_type="chart", bbox=(5.0, 10.0, 300.0, 400.0))
    assert fi.bbox == (5.0, 10.0, 300.0, 400.0)

    d = fi.to_dict()
    assert d["bbox"] == [5.0, 10.0, 300.0, 400.0], "bbox must be serialised as a list"

    fi2 = FigureInfo.from_dict(d)
    assert fi2.bbox == (5.0, 10.0, 300.0, 400.0), "bbox must round-trip through from_dict"


def test_figure_info_bbox_none_round_trips() -> None:
    """GH-47C step-0: FigureInfo.bbox=None serialises and deserialises cleanly."""
    from socr.core.result import FigureInfo

    fi = FigureInfo(figure_num=3, page_num=1, figure_type="image")
    assert fi.bbox is None
    d = fi.to_dict()
    assert d["bbox"] is None
    fi2 = FigureInfo.from_dict(d)
    assert fi2.bbox is None


def test_extractor_persists_bbox_on_image_block_strategy(tmp_path: Path) -> None:
    """GH-47C step-0: bbox is set on ExtractedFigure for Strategy 1 (IMAGE block) figures."""
    pdf_path = _make_pdf_with_placed_image(tmp_path)
    result = FigureExtractor(max_total=5, max_per_page=3).extract(pdf_path)

    assert len(result.figures) >= 1
    for fig in result.figures:
        assert fig.bbox is not None, "bbox must be set for Strategy 1 (IMAGE block) figures"
        x0, y0, x1, y1 = fig.bbox
        assert x1 > x0 and y1 > y0, "bbox must be a valid non-degenerate rect"


def test_extractor_persists_bbox_on_vector_strategy(tmp_path: Path) -> None:
    """GH-47C step-0: bbox is set on ExtractedFigure for Strategy 0 (vector cluster) figures."""
    pdf_path = _make_vector_dashboard_pdf(tmp_path)
    result = FigureExtractor(max_total=10, max_per_page=5).extract(pdf_path)

    assert len(result.figures) >= 1
    for fig in result.figures:
        assert fig.bbox is not None, "bbox must be set for Strategy 0 (vector) figures"
        x0, y0, x1, y1 = fig.bbox
        assert x1 > x0 and y1 > y0, "bbox must be a valid non-degenerate rect"


def test_recoverable_label_set_contains_inside_tokens(tmp_path: Path) -> None:
    """GH-47C: native word tokens inside the figure bbox are recoverable via get_text("words").

    This is a unit test of the filtering logic (centre-point containment) independent of
    the orchestrator, confirming that the three label strings inside the chart region are
    found and the body text above the region is excluded.
    """
    pdf_path = _make_pdf_with_vector_labels(tmp_path)

    # Open the PDF and apply the same centre-point containment filter the orchestrator uses.
    doc = fitz.open(pdf_path)
    page = doc[0]
    words = page.get_text("words")

    # Chart region bbox as inserted in _make_pdf_with_vector_labels.
    x0, y0, x1, y1 = 100.0, 100.0, 400.0, 500.0
    inside = [
        w[4] for w in words if x0 <= (w[0] + w[2]) / 2.0 <= x1 and y0 <= (w[1] + w[3]) / 2.0 <= y1
    ]
    doc.close()

    assert "Quarters" in inside, f"'Quarters' label should be inside figure bbox; got: {inside}"
    assert "EuroArea" in inside, f"'EuroArea' label should be inside figure bbox; got: {inside}"
    assert "GDP" in inside, f"'GDP' label should be inside figure bbox; got: {inside}"
    # Body paragraph text must be outside the region.
    assert not any("paragraph" in t.lower() for t in inside), (
        f"Body text from above the chart should NOT appear in figure region words; got: {inside}"
    )


def test_recoverable_label_set_empty_for_raster_figure(tmp_path: Path) -> None:
    """GH-47C: a rasterized embedded image has no native word tokens in its region.

    Zero recoverable words is INAPPLICABLE — correct and expected, never a failure.
    """
    pdf_path = _make_pdf_with_raster_only_figure(tmp_path)

    doc = fitz.open(pdf_path)
    page = doc[0]
    words = page.get_text("words")
    doc.close()

    # Rect matches the insert_image call in _make_pdf_with_raster_only_figure.
    x0, y0, x1, y1 = 100.0, 200.0, 400.0, 500.0
    inside = [
        w[4] for w in words if x0 <= (w[0] + w[2]) / 2.0 <= x1 and y0 <= (w[1] + w[3]) / 2.0 <= y1
    ]

    # For a pure raster image there are no text operators in the region.
    assert inside == [], f"Raster figure should yield empty recoverable-label set; got: {inside}"
