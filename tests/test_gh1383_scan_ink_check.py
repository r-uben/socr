"""#1383: tests for the sub-rules of the scan completeness check and the #1381 gates that a
mutant run of #1382 left alive. Every case is a DIFFERENCE from a control built the same way."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import fitz
import pytest

import socr
from socr.core.text_coverage import analyze_page

from test_gh1381_prior_ocr_agreement import (
    LAYER,
    SCAN_LINES,
    _agrees,
    _layer_of,
    _page,
    _run,
    _scan,
)

ROWS = [f"Row {i} value {i * 7}.5 and {i * 3}.25" for i in range(1, 12)]


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _scan_doc(extra_ink=None, extra_page=None, layer=ROWS, ink=ROWS) -> fitz.Document:
    """A scan: *ink* is the raster, *layer* is invisible text over it."""
    src = fitz.open()
    s = src.new_page(width=612, height=792)
    for row, line in enumerate(ink):
        s.insert_text((72, 100 + row * 14), line, fontsize=10)
    if extra_ink:
        extra_ink(s)
    png = s.get_pixmap(dpi=150).tobytes("png")
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_image(p.rect, stream=png)
    for row, line in enumerate(layer):
        p.insert_text((72, 100 + row * 14), line, fontsize=10, render_mode=3)
    if extra_page:
        extra_page(p)
    return doc


def _fact(doc: fitz.Document) -> bool:
    return analyze_page(doc[0]).hidden_layer_ink_accounted


def test_the_control_scan_is_accounted() -> None:
    assert _fact(_scan_doc()) is True


def test_a_glyph_sized_mark_outside_the_layer_is_caught() -> None:
    """The residual size floor: one 10 pt letter is about 10 px at the check's scale, far under
    a word, and must still count as ink the layer does not account for."""
    doc = _scan_doc(extra_ink=lambda s: s.insert_text((400, 400), "x", fontsize=10))
    assert _fact(doc) is False


def _light_only_scan() -> fitz.Document:
    """A near-white patch with a white glyph, every dark-polarity component of the patch under the
    layer's boxes and the glyph in a hole between them: only the LIGHT polarity sees the glyph."""

    def ink(s):
        s.draw_rect(fitz.Rect(300, 300, 360, 340), color=None, fill=(0.97,) * 3)
        s.insert_text((338, 325), "x", fontsize=14, color=(1, 1, 1))

    def boxes(p):
        for x in (296, 306, 316):
            for y in (298, 310, 322, 334):
                p.insert_text((x, y + 12), "M", fontsize=14, render_mode=3)
        for y in (298, 338):
            p.insert_text((328, y + 12), "M", fontsize=14, render_mode=3)
        for y in (298, 310, 322, 334):
            p.insert_text((356, y + 12), "M", fontsize=14, render_mode=3)
        for x in (338, 346):
            for y in (298, 338):
                p.insert_text((x, y + 12), "M", fontsize=14, render_mode=3)

    return _scan_doc(extra_ink=ink, extra_page=boxes, layer=[], ink=[])


def test_light_on_dark_ink_outside_the_layer_is_caught() -> None:
    from socr.core import text_coverage as tc

    page = _light_only_scan()[0]
    gray, mat = tc._render_gray(page)
    dark, light = tc._polarity_masks(gray)
    vis, inv = tc.visible_char_boxes(page)
    layer = tc._paint(dark.shape, [*vis, *inv], page, mat)

    def qualifying(ink) -> int:
        _l, st, res, _c, _g = tc._residual_components(ink, layer)
        return sum(
            1
            for i in res
            if max(st[i, 2], st[i, 3]) >= tc.GLYPH_W_P02_PX
            and not tc._is_horizontal_rule_px(st[i, 2], st[i, 3])
        )

    assert qualifying(dark) == 0 and qualifying(light) > 0, "fixture lost its premise"
    assert analyze_page(page).hidden_layer_ink_accounted is False


def test_a_drawing_that_is_not_a_rule_is_not_accounted() -> None:
    """Drawings are never excused unless they are horizontal rules, even one that leaves no ink."""
    white_box = lambda p: p.draw_rect(fitz.Rect(300, 300, 360, 340), color=(1, 1, 1), fill=None)  # noqa: E731
    assert _fact(_scan_doc(extra_page=white_box)) is False


def test_a_page_without_an_invisible_layer_has_no_hidden_layer_fact() -> None:
    """Visible text and no scan: the strict #1378 fact holds, the scan-mode fact must not."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for row, line in enumerate(ROWS):
        page.insert_text((72, 100 + row * 14), line, fontsize=10)
    cov = analyze_page(doc[0])
    assert cov.ink_fully_accounted is True
    assert cov.hidden_layer_ink_accounted is False


# --- the agreement predicate ------------------------------------------------------------


@pytest.mark.parametrize(("layer_form", "reading_form"), [("US", "us"), ("Fed", "FED"), ("a", "A")])
def test_case_differences_disagree(layer_form: str, reading_form: str) -> None:
    layer = f"The note holds {layer_form} in the table."
    assert _agrees(layer, _page(layer))
    assert not _agrees(f"The note holds {reading_form} in the table.", _page(layer))


def test_a_suspect_page_that_is_not_a_positive_finding_never_agrees() -> None:
    """Suspicion restored from a sidecar carries no positive flag and no failed-scan flag."""
    restored = _page(prior_ocr_layer=False)
    assert restored.prior_ocr_layer_scan_failed is False
    assert restored.invisible_text_over_raster is False
    assert not _agrees(LAYER, restored)
    # either positive finding alone is enough
    assert _agrees(LAYER, _page(prior_ocr_layer=False, invisible_text_over_raster=True))


# --- the refinalize tool ------------------------------------------------------------------


def _confirmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from socr.core.born_digital import BornDigitalDetector

    pdf = _scan(tmp_path / "scan.pdf", SCAN_LINES, SCAN_LINES)
    run = _run(tmp_path / "agree", pdf, _layer_of(pdf), monkeypatch)
    assert run["sidecar"].get("prior_ocr_layer_agreement") is True
    return run["sidecar"], BornDigitalDetector().detect(pdf).pages[0]


def test_refinalize_needs_the_runs_ink_fact(tmp_path: Path, monkeypatch) -> None:
    from socr.pipeline.refinalize_own_text import refinalize_page

    sidecar, pa = _confirmed(tmp_path, monkeypatch)
    assert refinalize_page(sidecar, pa) is None
    no_ink = {k: v for k, v in sidecar.items() if k != "text_hidden_layer_ink_accounted"}
    assert refinalize_page(no_ink, pa) is not None


@pytest.mark.parametrize("flag", ["prior_ocr_layer_scan_failed", "invisible_text_scan_failed"])
def test_refinalize_restores_the_failed_scan_flags(tmp_path: Path, monkeypatch, flag: str) -> None:
    from socr.pipeline.refinalize_own_text import refinalize_page

    sidecar, pa = _confirmed(tmp_path, monkeypatch)
    assert refinalize_page(sidecar, pa) is None
    failed = dataclasses.replace(pa, **{flag: True})
    assert refinalize_page(sidecar, failed) is not None
