"""#1383 (owner ruling): the scan-mode completeness fact counts the old layer's space boxes as
covered ink, exactly as stored. The born-digital fact does not."""

from __future__ import annotations

import fitz

from socr.core import text_coverage as tc
from socr.core.text_coverage import analyze_page, invisible_space_boxes

from test_gh1383_scan_ink_check import _scan_doc


def _comma_scan(layer: str) -> fitz.Document:
    """Raster ``ab`` plus a comma drawn just right of ``b``; the layer is *layer* at the same origin.
    With a trailing space glyph the comma lies inside that space's box, without it outside every box."""

    def ink(s):
        s.insert_text((100, 200), "ab", fontsize=10)
        s.insert_text((111.5, 200), ",", fontsize=10)

    doc = _scan_doc(extra_ink=ink, layer=[], ink=[])
    doc[0].insert_text((100, 200), layer, fontsize=10, render_mode=3)
    return doc


def test_the_layers_space_box_accounts_for_ink_that_overhangs_into_it() -> None:
    """The measured p071 case: ink right of its character's box, inside the following space box."""
    with_space = analyze_page(_comma_scan("ab ")[0])
    without = analyze_page(_comma_scan("ab")[0])
    assert with_space.hidden_layer_ink_accounted is True
    assert without.hidden_layer_ink_accounted is False


def test_the_born_digital_fact_does_not_count_a_space_box() -> None:
    """``analyze_page`` builds the #1378 fact's layer from the visible boxes only: a visible page
    whose ink overhangs into a visible space's box is not accounted by that space."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((100, 200), "f f", fontsize=40, fontname="helv", fontfile=None)
    gray, mat = tc._render_gray(page)
    dark, light = tc._polarity_masks(gray)
    vis, inv = tc.visible_char_boxes(page)
    assert inv == [] and invisible_space_boxes(page) == []
    narrow = tc._paint(dark.shape, vis, page, mat)
    assert analyze_page(page).ink_fully_accounted == tc.page_ink_is_fully_accounted(
        page, mat, dark, light, narrow
    )


def test_boxes_are_exactly_as_stored() -> None:
    doc = _comma_scan("ab ")
    boxes = invisible_space_boxes(doc[0])
    assert len(boxes) >= 1
    raw = doc[0].get_text("rawdict")
    stored = [
        tuple(ch["bbox"])
        for b in raw["blocks"]
        if b.get("type") == 0
        for ln in b["lines"]
        for sp in ln["spans"]
        for ch in sp["chars"]
        if ch["c"].isspace()
    ]
    assert sorted(tuple(b) for b in boxes) == sorted(stored)


def test_a_page_with_no_space_glyph_is_unchanged() -> None:
    doc = _scan_doc(layer=["Rowwithoutspaces"], ink=["Rowwithoutspaces"])
    assert invisible_space_boxes(doc[0]) == []


def test_a_visible_space_is_not_a_layer_space_box() -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((100, 200), "two words", fontsize=10)
    assert invisible_space_boxes(page) == []


def test_a_decimal_point_under_a_space_box_is_accounted() -> None:
    """The dangerous case, pinned so the owner can see it: raster ``1.5``, layer ``1 5`` whose
    space box covers the point. The ink check no longer sees the point; a reading ``1 5`` that
    agrees with the layer is then confirmed."""

    def ink(s):
        s.insert_text((100, 200), "1.5", fontsize=10)

    doc = _scan_doc(extra_ink=ink, layer=[], ink=[])
    doc[0].insert_text((100, 200), "1 5", fontsize=10, render_mode=3)
    assert analyze_page(doc[0]).hidden_layer_ink_accounted is True


def test_process_a_decimal_point_under_a_space_box_is_confirmed(tmp_path, monkeypatch) -> None:
    """What the owner accepted, end to end: raster ``Rate was 1.5 percent``, layer ``Rate was 1 5
    percent`` (a space box over the point), reading identical to the layer. The page is confirmed
    although the printed number is 1.5. The same page with the space glyph absent from the layer
    is not accounted (the difference is the ruling, nothing else)."""
    from test_gh1381_prior_ocr_agreement import _layer_of, _run

    def scan(path, layer_line):
        def ink(s):
            s.insert_text((72, 100), "Rate was 1.5 percent in the quarter.", fontsize=10)

        doc = _scan_doc(extra_ink=ink, layer=[], ink=[])
        doc[0].insert_text((72, 100), layer_line, fontsize=10, render_mode=3)
        doc.save(path)
        return path

    pdf = scan(tmp_path / "space.pdf", "Rate was 1 5 percent in the quarter.")
    out = _run(tmp_path / "out", pdf, _layer_of(pdf), monkeypatch)
    assert out["sidecar"].get("text_hidden_layer_ink_accounted") is True
    assert out["sidecar"].get("prior_ocr_layer_agreement") is True
