"""#1050 step 1: coverage verdict on trusted-native pages; flagged never ships SUCCESS."""

from __future__ import annotations

import json
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.manifest import SelectionProvenance, _select_page_output_tagged
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.text_coverage import (
    DETECT_DPI,
    _MAX_THICK_PT,
    _MIN_FILL_FRAC,
    analyze_page,
    coverage_for_document,
    uncovered_text_image_ref,
    uncovered_text_marker,
    uncovered_text_marker_no_image,
)
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables import image_locate


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def test_reused_constants_match_image_locate() -> None:
    assert DETECT_DPI == image_locate.DETECT_DPI
    assert _MIN_FILL_FRAC == image_locate._MIN_FILL_FRAC
    assert _MAX_THICK_PT == image_locate._MAX_THICK_PT


def _glyph_row(page, y: float, n: int = 8, color=(0, 0, 0)) -> None:
    """Path-only 'letters': stroked boxes, no text layer.

    Solid fills have fill=1.0 and sit above the corpus row-fill p99 (0.917).
    A thin stroke keeps fill in the real-glyph envelope; widths vary so the
    row's width CV stays above the corpus p1 (0.12).
    """
    x = 72.0
    widths = (3.0, 4.0, 5.0, 3.5, 4.5, 5.5, 3.2, 4.8)
    for i in range(n):
        w = widths[i % len(widths)]
        page.draw_rect(fitz.Rect(x, y, x + w, y + 6.0), color=color, width=0.6)
        x += w + 2.0


def _vector_unread_pdf(path: Path) -> Path:
    """Trusted-native body plus unread path-glyphs in the middle of the page.

    A page with no text layer is routed to OCR, not native; the coverage
    apply hook only demotes trusted-native winners. The unread marks sit
    between two body paragraphs so furniture-band suppression cannot eat
    them.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "A clean paragraph of native body text on this page.")
    _glyph_row(page, 200.0)
    _glyph_row(page, 212.0)
    page.insert_text((72, 520), "A second paragraph sits below the unread marks.")
    doc.save(path)
    doc.close()
    return path


def _light_on_dark_pdf(path: Path) -> Path:
    """White path-glyphs on a filled dark field, plus native body text.

    Dark-only adaptive ink misses the letters (or treats the field as one
    solid) and the page can pass. Both polarities, chosen from the page
    mid-grey, keep the white strokes as ink.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "A clean paragraph of native body text on this page.")
    page.draw_rect(fitz.Rect(60, 180, 420, 280), color=(0, 0, 0), fill=(0, 0, 0))
    _glyph_row(page, 200.0, color=(1, 1, 1))
    _glyph_row(page, 212.0, color=(1, 1, 1))
    page.insert_text((72, 520), "A second paragraph sits below the unread marks.")
    doc.save(path)
    doc.close()
    return path


def _fill_unread_pdf(path: Path, fill, color) -> Path:
    """Unread path-glyphs on a filled field (black-on-blue, grey, …)."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "A clean paragraph of native body text on this page.")
    page.draw_rect(fitz.Rect(60, 180, 420, 280), color=fill, fill=fill)
    _glyph_row(page, 200.0, color=color)
    _glyph_row(page, 212.0, color=color)
    page.insert_text((72, 520), "A second paragraph sits below the unread marks.")
    doc.save(path)
    doc.close()
    return path


def _repeating_logo_pdf(path: Path, pages: int = 3) -> Path:
    """Same unread mark in the header, different body text on each page.

    There is no furniture suppression: a repeated logo is an unneeded
    warning, which the owner accepts over silent loss.
    """
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        _glyph_row(page, 36.0)
        _glyph_row(page, 48.0)
        page.insert_text((72, 140), f"Body text unique to page {i + 1} of this document.")
        page.insert_text((72, 400), f"A second paragraph on page {i + 1} sits below.")
    doc.save(path)
    doc.close()
    return path


def _stacked_chart_pdf(path: Path, pages: int = 3) -> Path:
    """Three copies of the same unread-glyph page (the Golosov × 3 shape)."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 80), "A clean paragraph of native body text on this page.")
        _glyph_row(page, 200.0)
        _glyph_row(page, 212.0)
        page.insert_text((72, 520), "A second paragraph sits below the unread marks.")
    doc.save(path)
    doc.close()
    return path


def _raster_label_pdf(path: Path, rotation: int = 0) -> Path:
    """16pt Helvetica-Oblique raster, no text layer, 12 pt tall on the page.

    At this scale the glyph filter sees letter-height at 0°/180° and the
    swapped axis at 90°/270°. Normalising min/max before the envelope
    keeps the flag; using the raw pixmap width/height clears 90/270.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "A clean paragraph of native body text on this page.")
    label = fitz.open()
    lp = label.new_page(width=360, height=32)
    lp.insert_text((4, 22), "Minimum wage and employment", fontsize=16, fontname="heit")
    pix = lp.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    page.insert_image(fitz.Rect(72, 200, 72 + 360 * 12 / 32, 212), pixmap=pix)
    label.close()
    page.insert_text((72, 520), "Figure 1. A caption below the chart.")
    if rotation:
        page.set_rotation(rotation)
    doc.save(path)
    doc.close()
    return path


def _figure_above_caption_pdf(path: Path) -> Path:
    """Unread figure labels sit above the only text-layer line (the caption).

    Fable flagged these pages. Treating ``above_text`` as furniture cleared
    every region (Stock–Watson 2010, Acosta, Lee).
    """
    doc = fitz.open()
    page = doc.new_page()
    _glyph_row(page, 80.0)
    _glyph_row(page, 92.0)
    page.insert_text((72, 520), "Figure 1. A caption below the chart.")
    doc.save(path)
    doc.close()
    return path


def _short_word_row_pdf(path: Path) -> Path:
    """Narrow unread row (< 48 px) that is still a word on this page.

    Native words are short so page-local min_run is small. Three stroked
    marks span ~15 pt ≈ 31 px, below WORD_W_P50_PX. Astra's
    min_run_components keeps them; Fable's global 48 px cut dropped
    Golosov / Sutskever / Brochet / Ben–Gunn–Khan.
    """
    doc = fitz.open()
    page = doc.new_page()
    y = 72.0
    for _ in range(8):
        page.insert_text((72, y), "to be or to be or to be or to be")
        y += 14.0
    x = 72.0
    for w in (3.0, 3.5, 4.0):
        page.draw_rect(fitz.Rect(x, 200, x + w, 206), color=(0, 0, 0), width=0.6)
        x += w + 1.5
    x = 72.0
    for w in (3.2, 4.0, 3.8):
        page.draw_rect(fitz.Rect(x, 212, x + w, 218), color=(0, 0, 0), width=0.6)
        x += w + 1.5
    doc.save(path)
    doc.close()
    return path


def _clean_text_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Ordinary prose with a visible text layer and no leftover ink.")
    page.insert_text((72, 110), "A second line so the page is a real paragraph, not a stub.")
    doc.save(path)
    doc.close()
    return path


def _pipeline(**overrides) -> UnifiedPipeline:
    defaults = dict(
        primary_engine=EngineType.QWEN,
        local_engine=EngineType.QWEN,
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        recover_corrupt_math=False,
        equation_region_lane=False,
        describe_figure_crops=False,
        judge_backend="heuristic",
        check_text_coverage=True,
    )
    defaults.update(overrides)
    return UnifiedPipeline(PipelineConfig(**defaults))


def _run(pipeline: UnifiedPipeline, pdf: Path, out: Path, providers):
    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=providers),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        return pipeline.process(pdf, out)


def _run_with_state(pipeline: UnifiedPipeline, pdf: Path, out: Path, providers):
    captured: dict = {}
    assemble = pipeline._phase_assemble

    def capture(state, output_dir):
        captured["state"] = state
        return assemble(state, output_dir)

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=providers),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_phase_assemble", side_effect=capture),
    ):
        result = pipeline.process(pdf, out)
    return result, captured["state"]


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []])
def test_vector_drawn_text_is_flagged(tmp_path: Path, providers) -> None:
    pdf = _vector_unread_pdf(tmp_path / "vector.pdf")
    on = _run(_pipeline(check_text_coverage=True), pdf, tmp_path / "on", providers)
    off = _run(_pipeline(check_text_coverage=False), pdf, tmp_path / "off", providers)
    # process() returns a one-page EngineResult (whole-doc text). Per-page
    # status lives on the sidecar; document status is the difference pin.
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert off.status is DocumentStatus.SUCCESS
    assert on.status != off.status
    sidecar = next((tmp_path / "on").rglob("pages/00001.json"))
    rec = json.loads(sidecar.read_text())
    assert rec["status"] == PageStatus.WARNING.value
    assert rec["failure_mode"] == FailureMode.TEXT_INK_UNREAD.value
    assert rec["text_coverage"] == "flagged"
    off_side = next((tmp_path / "off").rglob("pages/00001.json"))
    off_rec = json.loads(off_side.read_text())
    assert off_rec["status"] == PageStatus.SUCCESS.value
    assert off_rec.get("text_coverage", "") in ("", "passed")
    mark = uncovered_text_marker(1)
    assert mark in (on.pages[0].text or "")
    assert mark not in (off.pages[0].text or "")
    frag = next((tmp_path / "on").rglob("pages/00001.md"))
    assert mark in frag.read_text()
    pngs = list((tmp_path / "on").rglob("figures/uncovered_text_p1_r*.png"))
    assert pngs, "flagged page must save a region PNG"


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []])
def test_repeating_logo_is_flagged(tmp_path: Path, providers) -> None:
    """A repeated header logo is an unneeded warning, not silent loss."""
    pdf = _repeating_logo_pdf(tmp_path / "logo.pdf")
    off = _run(_pipeline(check_text_coverage=False), pdf, tmp_path / "off", providers)
    on = _run(_pipeline(), pdf, tmp_path / "on", providers)
    assert off.status is DocumentStatus.SUCCESS
    assert on.status is DocumentStatus.AUDIT_FAILED
    for side in (tmp_path / "on").rglob("pages/*.json"):
        rec = json.loads(side.read_text())
        assert rec["status"] == PageStatus.WARNING.value
        assert rec.get("text_coverage") == "flagged"


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []])
def test_clean_text_page_is_passed_and_byte_identical(tmp_path: Path, providers) -> None:
    pdf = _clean_text_pdf(tmp_path / "clean.pdf")
    on = _run(_pipeline(check_text_coverage=True), pdf, tmp_path / "on", providers)
    off = _run(_pipeline(check_text_coverage=False), pdf, tmp_path / "off", providers)
    assert on.status is DocumentStatus.SUCCESS
    assert off.status is DocumentStatus.SUCCESS
    assert on.pages[0].text == off.pages[0].text
    assembled_on = next(p for p in (tmp_path / "on").rglob("*.md") if p.parent.name != "pages")
    assembled_off = next(p for p in (tmp_path / "off").rglob("*.md") if p.parent.name != "pages")
    assert assembled_on.read_bytes() == assembled_off.read_bytes()


def test_analyze_page_sees_vector_glyphs(tmp_path: Path) -> None:
    pdf = _vector_unread_pdf(tmp_path / "v.pdf")
    doc = fitz.open(pdf)
    rec = analyze_page(doc[0])
    doc.close()
    assert rec.verdict == "flagged"
    assert rec.regions


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []])
def test_light_on_dark_vector_is_flagged(tmp_path: Path, providers) -> None:
    """White vector lettering on a dark field must not ship silent SUCCESS."""
    pdf = _light_on_dark_pdf(tmp_path / "lod.pdf")
    doc = fitz.open(pdf)
    rec = analyze_page(doc[0])
    doc.close()
    assert rec.verdict == "flagged"
    assert rec.regions
    on = _run(_pipeline(check_text_coverage=True), pdf, tmp_path / "on", providers)
    off = _run(_pipeline(check_text_coverage=False), pdf, tmp_path / "off", providers)
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert off.status is DocumentStatus.SUCCESS
    assert on.status != off.status
    sidecar = next((tmp_path / "on").rglob("pages/00001.json"))
    rec = json.loads(sidecar.read_text())
    assert rec["status"] == PageStatus.WARNING.value
    assert rec["failure_mode"] == FailureMode.TEXT_INK_UNREAD.value
    assert rec["text_coverage"] == "flagged"
    assert uncovered_text_marker(1) in (on.pages[0].text or "")


@pytest.mark.parametrize(
    "name,fill,color",
    [
        ("black-on-blue", (0.1, 0.2, 0.6), (0, 0, 0)),
        ("black-on-grey-0.25", (0.25, 0.25, 0.25), (0, 0, 0)),
        ("black-on-grey-0.40", (0.40, 0.40, 0.40), (0, 0, 0)),
        ("black-on-grey-0.55", (0.55, 0.55, 0.55), (0, 0, 0)),
        ("white-on-grey-0.45", (0.45, 0.45, 0.45), (1, 1, 1)),
        ("white-on-grey-0.55", (0.55, 0.55, 0.55), (1, 1, 1)),
    ],
)
def test_contrast_fill_vector_is_flagged(tmp_path: Path, name, fill, color) -> None:
    """Black-on-mid and white-on-grey must flag; a global mid hid these."""
    pdf = _fill_unread_pdf(tmp_path / f"{name}.pdf", fill, color)
    doc = fitz.open(pdf)
    rec = analyze_page(doc[0])
    doc.close()
    assert rec.verdict == "flagged", name
    assert rec.regions


def test_figure_labels_above_caption_stay_flagged(tmp_path: Path) -> None:
    pdf = _figure_above_caption_pdf(tmp_path / "fig.pdf")
    doc = fitz.open(pdf)
    pages = coverage_for_document(doc)
    doc.close()
    assert pages[0].verdict == "flagged"
    assert pages[0].regions
    assert all(r.band == "above_text" for r in pages[0].regions)
    assert pages[0].kept


def test_short_word_row_kept_by_page_min_run(tmp_path: Path) -> None:
    from socr.core.text_coverage import WORD_W_P50_PX, SCALE

    pdf = _short_word_row_pdf(tmp_path / "short.pdf")
    doc = fitz.open(pdf)
    rec = analyze_page(doc[0])
    doc.close()
    assert rec.verdict == "flagged"
    assert rec.regions
    # Every kept region is narrower than Fable's global median word width.
    for reg in rec.regions:
        width_px = reg.px[2] - reg.px[0]
        assert width_px < WORD_W_P50_PX, (width_px, SCALE)


def test_wide_short_run_kept_by_word_width(tmp_path: Path) -> None:
    """A wide two-glyph row must survive a high page-local min_run.

    Replacing the width cut with min_run alone dropped Zheng p8 and
    Andrade–Ferroni p13 (Fable T, already flagged in r1).
    """
    pdf = tmp_path / "wide.pdf"
    doc = fitz.open()
    page = doc.new_page()
    for i in range(8):
        page.insert_text((72, 72 + i * 14), "paragraphs calibration components median width")
    page.draw_rect(fitz.Rect(72, 200, 92, 206), color=(0, 0, 0), width=0.6)
    page.draw_rect(fitz.Rect(96, 200, 118, 206), color=(0, 0, 0), width=0.6)
    page.draw_rect(fitz.Rect(72, 212, 93, 218), color=(0, 0, 0), width=0.6)
    page.draw_rect(fitz.Rect(97, 212, 119, 218), color=(0, 0, 0), width=0.6)
    doc.save(pdf)
    doc.close()
    held = fitz.open(pdf)
    rec = analyze_page(held[0])
    held.close()
    assert rec.verdict == "flagged"
    assert rec.regions
    from socr.core.text_coverage import WORD_W_P50_PX

    assert any(reg.px[2] - reg.px[0] >= WORD_W_P50_PX for reg in rec.regions)


def test_fingerprint_changes_when_the_check_is_toggled() -> None:
    on = _pipeline(check_text_coverage=True)._run_fingerprint()
    off = _pipeline(check_text_coverage=False)._run_fingerprint()
    assert on != off


def test_stacked_identical_pages_keep_citable_flags(tmp_path: Path) -> None:
    """Same ink at the same place is not furniture when it sits in the body."""
    pdf = _stacked_chart_pdf(tmp_path / "stack.pdf")
    one = _stacked_chart_pdf(tmp_path / "one.pdf", pages=1)
    doc1 = fitz.open(one)
    solo = coverage_for_document(doc1)
    doc1.close()
    doc3 = fitz.open(pdf)
    stacked = coverage_for_document(doc3)
    doc3.close()
    assert solo[0].verdict == "flagged"
    assert all(p.verdict == "flagged" for p in stacked)
    assert all(p.kept for p in stacked)


def test_distinct_captions_do_not_suppress_body_ink(tmp_path: Path) -> None:
    """Appendix 1/2/3 on Golosov-shaped copies must not eat the axis titles.

    Distinct captions change the text-layer fingerprint. That is not
    furniture. The unread marks sit between two body paragraphs, so they
    are inside the document text body and must stay flagged.
    """
    pdf = tmp_path / "app.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 80), "A clean paragraph of native body text on this page.")
        _glyph_row(page, 200.0)
        _glyph_row(page, 212.0)
        page.insert_text((72, 520), f"Appendix {i + 1}. A caption below the chart.")
    doc.save(pdf)
    doc.close()
    pages = coverage_for_document(fitz.open(pdf))
    assert all(p.verdict == "flagged" for p in pages)
    assert all(p.kept for p in pages)
    on = _run(_pipeline(), pdf, tmp_path / "on", [PROFILE_QWEN_LOCAL])
    assert on.status is DocumentStatus.AUDIT_FAILED


def test_rotated_raster_label_stays_flagged(tmp_path: Path) -> None:
    from socr.core.text_coverage import render_region_png

    upright = _raster_label_pdf(tmp_path / "r0.pdf", 0)
    doc = fitz.open(upright)
    rec0 = analyze_page(doc[0])
    doc.close()
    assert rec0.verdict == "flagged"
    for rot in (90, 180, 270):
        pdf = _raster_label_pdf(tmp_path / f"r{rot}.pdf", rot)
        d = fitz.open(pdf)
        rec = analyze_page(d[0])
        assert rec.verdict == "flagged", rot
        dest = tmp_path / f"crop_{rot}.png"
        render_region_png(d[0], rec.regions[0].pdf, dest)
        d.close()
        pix = fitz.Pixmap(dest)
        arr = bytes(pix.samples)
        assert min(arr) < 250, f"rotation {rot} crop is blank"


def test_crop_write_failure_is_unknown(tmp_path: Path) -> None:
    """cv2.imwrite returning False must not claim a saved image."""
    import cv2

    pdf = _vector_unread_pdf(tmp_path / "w.pdf")
    on_pipe = _pipeline(check_text_coverage=True)
    off_pipe = _pipeline(check_text_coverage=False)
    with patch.object(cv2, "imwrite", return_value=False):
        on = _run(on_pipe, pdf, tmp_path / "on", [PROFILE_QWEN_LOCAL])
    off = _run(off_pipe, pdf, tmp_path / "off", [PROFILE_QWEN_LOCAL])
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert off.status is DocumentStatus.SUCCESS
    sidecar = next((tmp_path / "on").rglob("pages/00001.json"))
    rec = json.loads(sidecar.read_text())
    assert rec["status"] == PageStatus.WARNING.value
    assert rec["failure_mode"] == FailureMode.TEXT_COVERAGE_UNKNOWN.value
    assert rec["text_coverage"] == "unknown"
    assert uncovered_text_marker(1) not in (on.pages[0].text or "")
    assert uncovered_text_marker_no_image(1) in (on.pages[0].text or "")
    assert not list((tmp_path / "on").rglob("figures/uncovered_text_p1_r*.png"))


def test_unknown_does_not_ship_success(tmp_path: Path) -> None:
    pdf = _clean_text_pdf(tmp_path / "u.pdf")
    on_pipe = _pipeline(check_text_coverage=True)
    off_pipe = _pipeline(check_text_coverage=False)
    with patch("socr.core.text_coverage._render_gray", side_effect=RuntimeError("boom")):
        on = _run(on_pipe, pdf, tmp_path / "on", [PROFILE_QWEN_LOCAL])
    off = _run(off_pipe, pdf, tmp_path / "off", [PROFILE_QWEN_LOCAL])
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert off.status is DocumentStatus.SUCCESS
    assert on.status != off.status
    sidecar = next((tmp_path / "on").rglob("pages/00001.json"))
    rec = json.loads(sidecar.read_text())
    assert rec["status"] == PageStatus.WARNING.value
    assert rec["failure_mode"] == FailureMode.TEXT_COVERAGE_UNKNOWN.value
    assert rec["text_coverage"] == "unknown"
    assert uncovered_text_marker_no_image(1) in (on.pages[0].text or "")


def test_coverage_decorates_math_hybrid(tmp_path: Path) -> None:
    """Coverage must keep the recovered equation and add the mark.

    Replacing the winning attempt left corrupt_math_hybrid on the
    predecessor; selection then fell back to raw native text.
    """
    from socr.core.document import DocumentHandle
    from socr.core.manifest import _select_page_output_tagged
    from socr.core.result import PageOutput
    from socr.core.state import DocumentState, PageState
    from socr.core.text_coverage import analyze_page

    pdf = _vector_unread_pdf(tmp_path / "hyb.pdf")
    doc = fitz.open(pdf)
    cov = analyze_page(doc[0])
    doc.close()
    assert cov.verdict == "flagged"
    recovered = "Clean prose.\n$P(A \\cup B)=P(A)+P(B)$\nMore prose."
    hybrid = PageOutput(
        page_num=1,
        text=recovered,
        status=PageStatus.WARNING,
        engine="native+math",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    ps = PageState(page_num=1)
    ps.attempts = [hybrid]
    ps.best_output = hybrid
    ps.corrupt_math_hybrid = hybrid
    ps.text_ink_unread = True
    ps.text_coverage_verdict = "flagged"
    state = DocumentState(handle=DocumentHandle(path=pdf))
    state.pages[1] = ps
    pipe = _pipeline()
    pipe._text_coverage_pages = {1: cov}
    pipe._scan_root = tmp_path
    pipe._apply_text_coverage(state, 1, ps, tmp_path / "out")
    winner, _prov = _select_page_output_tagged(state, 1)
    mark = uncovered_text_marker(1)
    assert "$P(A \\cup B)=P(A)+P(B)$" in (winner.text or "")
    assert mark in (winner.text or "")
    assert ps.corrupt_math_hybrid is ps.best_output
    assert ps.corrupt_math_hybrid in ps.attempts


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []])
def test_chart_asset_flag_demotes_document(tmp_path: Path, providers) -> None:
    """A flagged chart_asset winner must demote the DOCUMENT, not only the page."""
    pdf = _vector_unread_pdf(tmp_path / "ch.pdf")
    on_pipe = _pipeline(check_text_coverage=True)
    off_pipe = _pipeline(check_text_coverage=False)
    with patch.object(on_pipe, "_is_chart_asset_page", return_value=True):
        on = _run(on_pipe, pdf, tmp_path / "on", providers)
    with patch.object(off_pipe, "_is_chart_asset_page", return_value=True):
        off = _run(off_pipe, pdf, tmp_path / "off", providers)
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert off.status is DocumentStatus.SUCCESS
    assert on.status != off.status
    sidecar = next((tmp_path / "on").rglob("pages/00001.json"))
    rec = json.loads(sidecar.read_text())
    assert rec["engine"] == "chart_asset"
    assert rec["status"] == PageStatus.WARNING.value
    assert rec["failure_mode"] == FailureMode.TEXT_INK_UNREAD.value
    assert uncovered_text_marker(1) in (on.pages[0].text or "")


def _raster_above_distinct_captions_pdf(path: Path, pages: int = 3) -> Path:
    """Same raster label above distinct captions — Astra r3 P0."""
    doc = fitz.open()
    label = fitz.open()
    lp = label.new_page(width=360, height=32)
    lp.insert_text((4, 22), "Minimum wage and employment", fontsize=16, fontname="heit")
    pix = lp.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    label.close()
    for i in range(pages):
        page = doc.new_page()
        page.insert_image(fitz.Rect(72, 80, 72 + 360 * 12 / 32, 92), pixmap=pix)
        # One native line so the page stays trusted-native (caption-only
        # pages go to OCR). The raster still sits above every text-layer
        # line — the median-span "header" that used to eat the flag.
        page.insert_text((72, 140), f"Notes unique to appendix {i + 1} of this figure.")
        page.insert_text((72, 520), f"Appendix {i + 1}. A caption below the chart.")
    doc.save(path)
    doc.close()
    return path


def test_raster_label_above_distinct_captions_stays_flagged(tmp_path: Path) -> None:
    """Three figure-only pages with the same raster and distinct captions.

    The median caption span used to make the label a header margin and
    ship page and document SUCCESS. Suppression is gone; process() must
    mark every page.
    """
    pdf = _raster_above_distinct_captions_pdf(tmp_path / "cap.pdf")
    solo = _raster_above_distinct_captions_pdf(tmp_path / "one.pdf", pages=1)
    assert coverage_for_document(fitz.open(solo))[0].verdict == "flagged"
    on = _run(_pipeline(), pdf, tmp_path / "on", [])
    off = _run(_pipeline(check_text_coverage=False), pdf, tmp_path / "off", [])
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert off.status is DocumentStatus.SUCCESS
    for i in range(1, 4):
        sidecar = next((tmp_path / "on").rglob(f"pages/{i:05d}.json"))
        rec = json.loads(sidecar.read_text())
        assert rec["status"] == PageStatus.WARNING.value
        assert rec["text_coverage"] == "flagged"
        assert rec["failure_mode"] == FailureMode.TEXT_INK_UNREAD.value
        frag = next((tmp_path / "on").rglob(f"pages/{i:05d}.md"))
        assert uncovered_text_marker(i) in frag.read_text()


def _ruled_table_plus_unread_pdf(path: Path) -> Path:
    """Intersecting ruled table plus unread path-glyphs.

    Two unjoined rules do not set ``has_tables``. A closed grid does, so
    selection reaches ``NATIVE_FALLBACK`` instead of ``PASSING_BEST_OUTPUT``.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (72, 90), "See the specification for the estimates reported below.", fontsize=11
    )
    rows = (("Year", "Coefficient"), ("2019", "0.31"), ("2020", "0.42"), ("2021", "0.55"))
    top, left, mid, right, pitch = 128.0, 68.0, 190.0, 320.0, 22.0
    for i, (a, b) in enumerate(rows):
        y = top + 16 + i * pitch
        page.insert_text((left + 6, y), a, fontsize=10)
        page.insert_text((mid + 6, y), b, fontsize=10)
    bottom = top + pitch * len(rows)
    for i in range(len(rows) + 1):
        page.draw_line(fitz.Point(left, top + pitch * i), fitz.Point(right, top + pitch * i))
    for x in (left, mid, right):
        page.draw_line(fitz.Point(x, top), fitz.Point(x, bottom))
    _glyph_row(page, 300.0)
    _glyph_row(page, 312.0)
    page.insert_text((60, 520), "A caption sits below the unread marks.")
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("native_only", [False, True])
@pytest.mark.parametrize("unknown", [False, True])
def test_actual_table_fallback_coverage(tmp_path: Path, native_only: bool, unknown: bool) -> None:
    """A real detected table must take NATIVE_FALLBACK, not PASSING_BEST_OUTPUT."""
    pdf = _ruled_table_plus_unread_pdf(tmp_path / "tbl.pdf")
    kw = dict(native_only=native_only, table_judge_ladder=False)
    unknown_ctx = (
        patch("socr.core.text_coverage._render_gray", side_effect=RuntimeError("boom"))
        if unknown
        else nullcontext()
    )
    with unknown_ctx:
        on, state = _run_with_state(
            _pipeline(check_text_coverage=True, **kw), pdf, tmp_path / "on", []
        )
    off, off_state = _run_with_state(
        _pipeline(check_text_coverage=False, **kw), pdf, tmp_path / "off", []
    )
    ps = state.pages[1]
    assert ps.has_tables is True
    assert ps.native_table_region_count == 1
    assert ps.is_structure_class() is True
    _, provenance = _select_page_output_tagged(state, 1)
    _, off_prov = _select_page_output_tagged(off_state, 1)
    rec = json.loads(next((tmp_path / "on").rglob("pages/00001.json")).read_text())
    off_rec = json.loads(next((tmp_path / "off").rglob("pages/00001.json")).read_text())
    assert off_prov is SelectionProvenance.NATIVE_CLEAN
    assert off_rec["status"] == PageStatus.SUCCESS.value
    assert rec["text_coverage"] == ("unknown" if unknown else "flagged")
    assert rec["status"] == PageStatus.WARNING.value
    assert provenance is SelectionProvenance.NATIVE_FALLBACK
    assert rec["failure_mode"] == (
        FailureMode.TEXT_COVERAGE_UNKNOWN.value if unknown else FailureMode.TEXT_INK_UNREAD.value
    )
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert off.status is DocumentStatus.SUCCESS
    assert on.status != off.status
    if not unknown:
        mark = uncovered_text_marker(1)
        assert mark in (on.pages[0].text or "")
        assert mark not in (off.pages[0].text or "")
        assert list((tmp_path / "on").rglob("uncovered_text_p1_r*.png"))


def test_fallback_without_crop_does_not_promise_image(tmp_path: Path) -> None:
    """Fallback must not say 'see image' when apply never saved a crop.

    Selection is pure: apply (best_output already gone) writes unknown,
    then two selections return the same text with the no-image mark.
    """
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    pdf = _vector_unread_pdf(tmp_path / "fb.pdf")
    ps = PageState(page_num=1)
    ps.is_born_digital = True
    ps.native_text = "A clean paragraph of native body text on this page."
    ps.best_output = None
    ps.text_ink_unread = True
    ps.text_coverage_verdict = "flagged"
    state = DocumentState(handle=DocumentHandle(path=pdf))
    state.pages[1] = ps
    pipe = _pipeline()
    pipe._apply_text_coverage(state, 1, ps, tmp_path / "out")
    assert ps.text_coverage_unknown is True
    assert ps.text_ink_unread is False
    first, _ = _select_page_output_tagged(state, 1)
    flags = (ps.text_ink_unread, ps.text_coverage_unknown, ps.text_coverage_verdict)
    second, _ = _select_page_output_tagged(state, 1)
    assert first.text == second.text
    assert flags == (ps.text_ink_unread, ps.text_coverage_unknown, ps.text_coverage_verdict)
    assert uncovered_text_marker(1) not in (first.text or "")
    assert uncovered_text_marker_no_image(1) in (first.text or "")
    assert "see image" not in (first.text or "")
    assert first.failure_mode is FailureMode.TEXT_COVERAGE_UNKNOWN
    assert first.status is PageStatus.WARNING


def test_fallback_keeps_equation_sidecar_and_coverage_mark(tmp_path: Path) -> None:
    """A later native equation append must survive when apply decorated best_output."""
    from socr.core.document import DocumentHandle
    from socr.core.result import PageOutput
    from socr.core.state import DocumentState, PageState

    pdf = _vector_unread_pdf(tmp_path / "eq.pdf")
    base = "A clean paragraph of native body text on this page."
    equation = "$$P(A \\cup B)=P(A)+P(B)$$"
    sidecar = base + "\n\n" + equation + "\n"
    mark = uncovered_text_marker(1)
    link = uncovered_text_image_ref(1, 1)
    decorated = base + "\n\n" + mark + "\n\n" + link + "\n"
    bo = PageOutput(
        page_num=1,
        text=decorated,
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=True,
        failure_mode=FailureMode.TEXT_INK_UNREAD,
    )
    eq = PageOutput(
        page_num=1,
        text=sidecar,
        status=PageStatus.SUCCESS,
        engine="native+equations",
        audit_passed=True,
        failure_mode=FailureMode.NONE,
    )
    ps = PageState(page_num=1)
    ps.is_born_digital = True
    ps.native_text = base
    ps.has_tables = True
    ps.attempts = [bo, eq]
    ps.best_output = bo
    ps.text_ink_unread = True
    ps.text_coverage_verdict = "flagged"
    state = DocumentState(handle=DocumentHandle(path=pdf))
    state.pages[1] = ps
    winner, provenance = _select_page_output_tagged(state, 1)
    assert provenance is SelectionProvenance.NATIVE_FALLBACK
    assert equation in (winner.text or "")
    assert mark in (winner.text or "")
    assert link in (winner.text or "")


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []])
def test_coverage_keeps_preexisting_failure_mode(tmp_path: Path, providers) -> None:
    """Chart SUCCESS plus minus+coverage must keep MINUS and demote the document."""
    pdf = _vector_unread_pdf(tmp_path / "minus.pdf")
    off_pipe = _pipeline(check_text_coverage=False)
    on_pipe = _pipeline(check_text_coverage=True)
    with (
        patch.object(off_pipe, "_is_chart_asset_page", return_value=True),
        patch("socr.pipeline.orchestrator.minus_as_digit_suspect", return_value=False),
    ):
        off = _run(off_pipe, pdf, tmp_path / "off", providers)
    with (
        patch.object(on_pipe, "_is_chart_asset_page", return_value=True),
        patch("socr.pipeline.orchestrator.minus_as_digit_suspect", return_value=True),
    ):
        on = _run(on_pipe, pdf, tmp_path / "on", providers)
    assert off.status is DocumentStatus.SUCCESS
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert on.status != off.status
    sidecar = next((tmp_path / "on").rglob("pages/00001.json"))
    rec = json.loads(sidecar.read_text())
    assert rec["engine"] == "chart_asset"
    assert rec["status"] == PageStatus.WARNING.value
    assert rec["failure_mode"] == FailureMode.NATIVE_MINUS_AS_DIGIT.value
    assert rec["text_coverage"] == "flagged"
    assert uncovered_text_marker(1) in (on.pages[0].text or "")


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []])
@pytest.mark.parametrize("unknown_via", [None, "doc_open", "crop_save"])
def test_flattened_table_outranks_coverage_mode(tmp_path: Path, providers, unknown_via) -> None:
    """Apply must stamp table_not_reconstructed on flagged and both unknown paths."""
    pdf = _vector_unread_pdf(tmp_path / "flat.pdf")
    on_pipe = _pipeline(check_text_coverage=True)
    off_pipe = _pipeline(check_text_coverage=False)
    patches = [
        patch(
            "socr.pipeline.orchestrator.table_not_reconstructed_suspect",
            return_value=True,
        ),
        patch("socr.core.manifest.table_not_reconstructed_suspect", return_value=True),
    ]
    if unknown_via == "doc_open":
        patches.append(
            patch(
                "socr.core.text_coverage.coverage_for_document",
                side_effect=RuntimeError("open boom"),
            )
        )
    elif unknown_via == "crop_save":
        patches.append(
            patch(
                "socr.core.text_coverage.render_region_png",
                side_effect=RuntimeError("crop boom"),
            )
        )
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        on = _run(on_pipe, pdf, tmp_path / "on", providers)
        off = _run(off_pipe, pdf, tmp_path / "off", providers)
    on_rec = json.loads(next((tmp_path / "on").rglob("pages/00001.json")).read_text())
    off_rec = json.loads(next((tmp_path / "off").rglob("pages/00001.json")).read_text())
    assert on_rec["failure_mode"] == FailureMode.TABLE_NOT_RECONSTRUCTED.value
    assert off_rec["failure_mode"] == FailureMode.TABLE_NOT_RECONSTRUCTED.value
    assert on_rec["status"] == PageStatus.WARNING.value
    if unknown_via is None:
        assert uncovered_text_marker(1) in (on.pages[0].text or "")
    else:
        assert uncovered_text_marker_no_image(1) in (on.pages[0].text or "")
        assert uncovered_text_marker(1) not in (on.pages[0].text or "")
    assert uncovered_text_marker(1) not in (off.pages[0].text or "")
    assert on.status is DocumentStatus.AUDIT_FAILED


def test_document_open_failure_is_unknown(tmp_path: Path) -> None:
    """A failed coverage_for_document must not leave the page SUCCESS."""
    pdf = _clean_text_pdf(tmp_path / "open.pdf")
    on_pipe = _pipeline(check_text_coverage=True)
    off_pipe = _pipeline(check_text_coverage=False)
    with patch(
        "socr.core.text_coverage.coverage_for_document",
        side_effect=RuntimeError("open boom"),
    ):
        on = _run(on_pipe, pdf, tmp_path / "on", [PROFILE_QWEN_LOCAL])
    off = _run(off_pipe, pdf, tmp_path / "off", [PROFILE_QWEN_LOCAL])
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert off.status is DocumentStatus.SUCCESS
    sidecar = next((tmp_path / "on").rglob("pages/00001.json"))
    rec = json.loads(sidecar.read_text())
    assert rec["text_coverage"] == "unknown"
    assert rec["failure_mode"] == FailureMode.TEXT_COVERAGE_UNKNOWN.value
    assert uncovered_text_marker_no_image(1) in (on.pages[0].text or "")


def test_text_coverage_pages_metadata_skips_ocr(tmp_path: Path) -> None:
    """Per-document metadata lists only native / chart / native+math winners."""
    pdf = _vector_unread_pdf(tmp_path / "ocr.pdf")
    on_pipe = _pipeline(check_text_coverage=True)
    _on, state = _run_with_state(on_pipe, pdf, tmp_path / "on", [])
    ps = state.pages[1]
    assert ps.text_ink_unread is True
    meta = tmp_path / "on" / pdf.stem / "metadata.json"
    assert meta.is_file()
    data = json.loads(meta.read_text())
    assert "text_coverage_pages" in data
    assert data["text_coverage_pages"] == [1]
    qwen = PageOutput(
        page_num=1,
        text=ps.best_output.text if ps.best_output else "ocr",
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=True,
        failure_mode=FailureMode.TEXT_INK_UNREAD,
    )
    ps.best_output = qwen
    on_pipe._phase_assemble(state, tmp_path / "on")
    data = json.loads(meta.read_text())
    assert 1 not in (data.get("text_coverage_pages") or [])


@pytest.mark.parametrize("branch", ["unknown", "flagged"])
@pytest.mark.parametrize("status", [PageStatus.ERROR, PageStatus.SKIPPED])
def test_apply_does_not_soften_non_success(tmp_path: Path, branch: str, status) -> None:
    """Apply must not raise ERROR / SKIPPED to WARNING on either branch."""
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    pdf = _vector_unread_pdf(tmp_path / "err.pdf")
    bo = PageOutput(
        page_num=1,
        text="A clean paragraph of native body text on this page.",
        status=status,
        engine="native",
        audit_passed=False,
        failure_mode=FailureMode.HALLUCINATION,
    )
    ps = PageState(page_num=1)
    ps.is_born_digital = True
    ps.native_text = bo.text
    ps.best_output = bo
    ps.attempts = [bo]
    if branch == "unknown":
        ps.text_coverage_unknown = True
        ps.text_coverage_verdict = "unknown"
    else:
        ps.text_ink_unread = True
        ps.text_coverage_verdict = "flagged"
    state = DocumentState(handle=DocumentHandle(path=pdf))
    state.pages[1] = ps
    pipe = _pipeline()
    pipe._scan_root = tmp_path
    pipe._apply_text_coverage(state, 1, ps, tmp_path / "out")
    assert ps.best_output is not None
    assert ps.best_output.status is status
    assert ps.best_output.failure_mode is FailureMode.HALLUCINATION


def test_selection_alone_does_not_write_coverage_flags(tmp_path: Path) -> None:
    """Selection must not mutate flags when apply never ran."""
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    pdf = _vector_unread_pdf(tmp_path / "pure.pdf")
    ps = PageState(page_num=1)
    ps.is_born_digital = True
    ps.native_text = "A clean paragraph of native body text on this page."
    ps.best_output = None
    ps.text_ink_unread = True
    ps.text_coverage_unknown = False
    ps.text_coverage_verdict = "flagged"
    state = DocumentState(handle=DocumentHandle(path=pdf))
    state.pages[1] = ps
    before = (ps.text_ink_unread, ps.text_coverage_unknown, ps.text_coverage_verdict)
    _select_page_output_tagged(state, 1)
    assert (ps.text_ink_unread, ps.text_coverage_unknown, ps.text_coverage_verdict) == before


def test_ink_mask_is_removed() -> None:
    import socr.core.text_coverage as tc

    assert not hasattr(tc, "_ink_mask")
