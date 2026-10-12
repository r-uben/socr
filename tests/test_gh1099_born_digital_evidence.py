"""GH-1099: born-digital pages use visible PDF text + pixel OCR as table evidence."""

from __future__ import annotations

from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.tables.locate import locate_tables  # noqa: E402
from socr.tables.source_evidence import (  # noqa: E402
    _invisible_text_layer_bboxes,
    _plain_text_from_visible_chars,
    _text_dict_spans,
    _text_layer_ink_matrix,
    _tokens_from_page_text_layer,
    build_born_digital_evidence,
    text_layer_span_is_visible,
    verify_scanned_table,
)

PYMUPDF_VERSION = fitz.version[0]


def _table(*rows: tuple[str, str]) -> str:
    lines = ["| label | value |", "| --- | --- |"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def _ocr_alpha_five(_pix) -> str:
    return "alpha 5"


def _make_scan_with_hidden_layer(shape: str):
    """Raster below dominance ratio + invisible bogus number in the text layer."""
    src = fitz.open()
    sp = src.new_page(width=400, height=400)
    sp.insert_text((50, 100), "alpha 5", fontsize=22)
    pix = sp.get_pixmap()
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    if shape == "crop":
        page.insert_image(fitz.Rect(50, 50, 450, 450), pixmap=pix)
    elif shape == "tiles":
        page.insert_image(fitz.Rect(50, 50, 450, 250), pixmap=pix, keep_proportion=False)
        page.insert_image(fitz.Rect(50, 250, 450, 450), pixmap=pix, keep_proportion=False)
    page.insert_text((75, 150), "alpha 999", render_mode=3)
    return doc, page


@pytest.mark.parametrize("shape", ["crop", "tiles"])
def test_invisible_layer_on_partial_raster_does_not_support_bogus_number(shape: str) -> None:
    doc, page = _make_scan_with_hidden_layer(shape)
    try:
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            result = verify_scanned_table(
                page,
                _table(("alpha", "999")),
                native_trusted=False,
                ocr_image_fn=_ocr_alpha_five,
            )
        assert not result.passed
        assert "999" in result.reason
    finally:
        doc.close()


def test_invisible_layer_on_vector_page_does_not_support_bogus_number() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "alpha 999", render_mode=3)
    page.insert_text((72, 120), "alpha 5")
    try:
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            result = verify_scanned_table(
                page,
                _table(("alpha", "999")),
                native_trusted=False,
                ocr_image_fn=_ocr_alpha_five,
            )
        assert not result.passed
    finally:
        doc.close()


def test_raster_inventory_failure_treats_page_as_scan_not_born_digital() -> None:
    src = fitz.open()
    sp = src.new_page(width=400, height=400)
    sp.insert_text((50, 100), "alpha 5", fontsize=22)
    pix = sp.get_pixmap()
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_image(page.rect, pixmap=pix)
    page.insert_text((75, 150), "alpha 999", render_mode=3)
    try:
        with (
            patch("socr.tables.locate.locate_tables", return_value=[]),
            patch.object(page, "get_image_info", side_effect=RuntimeError("inventory failed")),
        ):
            result = verify_scanned_table(
                page,
                _table(("alpha", "999")),
                native_trusted=False,
                ocr_image_fn=_ocr_alpha_five,
            )
        assert not result.passed
    finally:
        doc.close()


def _reviewer_table(*values: str) -> str:
    return "| label | value |\n| --- | --- |\n" + "\n".join(f"| alpha | {v} |" for v in values)


def _page_mixed_invisible_digit() -> tuple[fitz.Document, fitz.Page]:
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    kwargs = {"fontsize": 22}
    page.insert_text((72, 100), "99", **kwargs)
    x = 72 + fitz.get_text_length("99", fontsize=22)
    page.insert_text((x, 100), "9", render_mode=3, **kwargs)
    return doc, page


@pytest.mark.parametrize(
    ("kind", "values", "ocr"),
    [
        ("white", ("999",), ""),
        ("under_image", ("999",), ""),
        ("mixed", ("999",), "99"),
        ("hiddenminus", ("-5",), "5"),
    ],
)
def test_hidden_text_escapes_rejected(kind: str, values: tuple[str, ...], ocr: str) -> None:
    doc = fitz.open()
    blank = fitz.open()
    page = doc.new_page(width=400, height=400)
    blank_page = blank.new_page(width=400, height=400)
    kwargs = {"fontsize": 22}
    try:
        if kind == "white":
            page.insert_text((72, 100), "999", color=(1, 1, 1), **kwargs)
        elif kind == "under_image":
            page.insert_text((72, 100), "999", **kwargs)
            page.insert_image(
                fitz.Rect(40, 40, 200, 180),
                pixmap=blank_page.get_pixmap(),
                keep_proportion=False,
            )
        elif kind == "mixed":
            page.insert_text((72, 100), "99", **kwargs)
            x = 72 + fitz.get_text_length("99", fontsize=22)
            page.insert_text((x, 100), "9", render_mode=3, **kwargs)
        elif kind == "hiddenminus":
            page.insert_text((72, 100), "-", render_mode=3, **kwargs)
            page.insert_text(
                (72 + fitz.get_text_length("-", fontsize=22), 100),
                "5",
                **kwargs,
            )
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            result = verify_scanned_table(
                page,
                _reviewer_table(*values),
                native_trusted=False,
                ocr_image_fn=lambda _: ocr,
            )
        assert not result.passed
    finally:
        doc.close()
        blank.close()


def test_alpha_filter_required_for_mixed_and_hiddenminus(monkeypatch) -> None:
    """Dropping only the alpha gate still leaves ink, so fabrications stay rejected."""
    doc_m, page_m = _page_mixed_invisible_digit()
    doc_h = fitz.open()
    page_h = doc_h.new_page(width=400, height=400)
    page_h.insert_text((72, 100), "-", render_mode=3, fontsize=22)
    page_h.insert_text((72 + fitz.get_text_length("-", fontsize=22), 100), "5", fontsize=22)
    monkeypatch.setattr(
        "socr.tables.source_evidence.text_layer_span_is_visible",
        lambda _span: True,
    )
    try:
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            mixed = verify_scanned_table(
                page_m,
                _reviewer_table("999"),
                native_trusted=False,
                ocr_image_fn=lambda _: "99",
            )
            hidden = verify_scanned_table(
                page_h,
                _reviewer_table("-5"),
                native_trusted=False,
                ocr_image_fn=lambda _: "5",
            )
        assert not mixed.passed
        assert not hidden.passed
    finally:
        doc_m.close()
        doc_h.close()


def test_dropping_alpha_and_ink_guards_allows_invisible_span_fabrication(monkeypatch) -> None:
    """Mutant: no alpha or ink gate → invisible ``999`` span supports the model."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((72, 100), "999", render_mode=3, fontsize=22)
    page.insert_text((72, 130), "99", fontsize=22)
    try:
        monkeypatch.setattr(
            "socr.tables.source_evidence.text_layer_span_is_visible",
            lambda _span: True,
        )
        monkeypatch.setattr(
            "socr.tables.source_evidence._glyph_bbox_has_ink",
            lambda *_args, **_kwargs: True,
        )
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            result = verify_scanned_table(
                page,
                _reviewer_table("999"),
                native_trusted=False,
                ocr_image_fn=lambda _: "99",
            )
        assert result.passed
    finally:
        doc.close()


def test_dropping_ink_check_allows_white_on_white_fabrication(monkeypatch) -> None:
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((72, 100), "999", color=(1, 1, 1), fontsize=22)
    try:
        monkeypatch.setattr(
            "socr.tables.source_evidence._glyph_bbox_has_ink",
            lambda *_args, **_kwargs: True,
        )
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            result = verify_scanned_table(
                page,
                _reviewer_table("999"),
                native_trusted=False,
                ocr_image_fn=lambda _: "",
            )
        assert result.passed
    finally:
        doc.close()


def test_visible_born_digital_table_still_passes() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "alpha 42", fontsize=22)
    try:
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            result = verify_scanned_table(
                page,
                _table(("alpha", "42")),
                native_trusted=False,
                ocr_image_fn=lambda _: "alpha 42",
            )
        assert result.passed
    finally:
        doc.close()


def test_rotated_visible_table_still_passes() -> None:
    doc = fitz.open()
    page = doc.new_page(width=500, height=700)
    page.insert_text((200, 350), "revenue 99", fontsize=11, rotate=90)
    try:
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            result = verify_scanned_table(
                page,
                _table(("revenue", "99")),
                native_trusted=False,
                ocr_image_fn=lambda _: "",
            )
        assert result.passed
    finally:
        doc.close()


def test_max_merge_rejects_duplicate_model_cell_for_one_printed_five() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "alpha 5")
    try:
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            result = verify_scanned_table(
                page,
                _table(("alpha", "5"), ("alpha", "5")),
                native_trusted=False,
                ocr_image_fn=_ocr_alpha_five,
            )
        assert not result.passed
    finally:
        doc.close()


def test_sum_merge_would_accept_duplicate_five(monkeypatch) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "alpha 5")
    try:
        from collections import Counter

        def _sum_merge(left: Counter, right: Counter) -> Counter:
            return left + right

        monkeypatch.setattr("socr.tables.source_evidence._max_merge_numeric", _sum_merge)
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            result = verify_scanned_table(
                page,
                _table(("alpha", "5"), ("alpha", "5")),
                native_trusted=False,
                ocr_image_fn=_ocr_alpha_five,
            )
        assert result.passed
    finally:
        doc.close()


def test_control_byte_layer_does_not_support_invented_negative() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "alpha \x020.47", fontsize=22)
    try:
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            result = verify_scanned_table(
                page,
                _table(("alpha", "-0.47")),
                native_trusted=False,
                ocr_image_fn=lambda _: "alpha 0.47",
            )
        assert not result.passed
        assert "-0.47" in result.reason
    finally:
        doc.close()


def test_control_byte_layer_plus_ocr_negative_passes_when_ocr_saw_minus() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "alpha \x020.0008", fontsize=22)
    try:
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            result = verify_scanned_table(
                page,
                _table(("alpha", "-0.0008")),
                native_trusted=False,
                ocr_image_fn=lambda _: "alpha -0.0008",
            )
        assert result.passed
    finally:
        doc.close()


def test_render_mode_3_invisible_spans_have_zero_alpha() -> None:
    """``get_text('dict')`` must expose invisible OCR on this PyMuPDF build."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "visible 5")
    page.insert_text((72, 120), "hidden 999", render_mode=3)
    try:
        spans = list(_text_dict_spans(page))
        visible = [s for s in spans if text_layer_span_is_visible(s)]
        invisible = [s for s in spans if not text_layer_span_is_visible(s)]
        assert any("5" in (s.get("text") or "") for s in visible)
        assert invisible, f"expected an invisible span (PyMuPDF {PYMUPDF_VERSION})"
        assert all(s.get("alpha") == 0 for s in invisible), (
            f"render_mode=3 spans must have alpha==0 on PyMuPDF {PYMUPDF_VERSION}: "
            f"{[(s.get('text'), s.get('alpha')) for s in invisible]}"
        )
        assert len(_invisible_text_layer_bboxes(page)) == len(invisible)
    finally:
        doc.close()


def _stress_multipage_pdf() -> fitz.Document:
    """Synthetic doc: visible tables, invisible layer, and sub-dominance raster pages."""
    label_pix = fitz.open()
    label_page = label_pix.new_page(width=300, height=200)
    label_page.insert_text((40, 80), "label value 5", fontsize=18)
    pixmap = label_page.get_pixmap()
    label_pix.close()

    doc = fitz.open()
    for i in range(40):
        page = doc.new_page(width=500, height=500)
        kind = i % 3
        if kind == 0:
            page.insert_text((72, 100), "revenue 42", fontsize=14)
            page.insert_text((72, 130), "cost 7", fontsize=14)
        elif kind == 1:
            page.insert_text((72, 100), "label value 5", fontsize=14)
            page.insert_text((72, 130), "label value 999", render_mode=3, fontsize=14)
        else:
            page.insert_image(fitz.Rect(50, 50, 450, 450), pixmap=pixmap)
            page.insert_text((75, 150), "label value 999", render_mode=3, fontsize=14)
    return doc


def test_many_pages_one_document_in_one_process() -> None:
    """Evidence + real ``locate_tables``/``find_tables`` must survive a long in-process run."""
    doc = _stress_multipage_pdf()
    try:
        for index, page in enumerate(doc):
            model = _table(("revenue", "42"), ("cost", "7"))
            if index % 3 == 1:
                model = _table(("label", "5"))
            elif index % 3 == 2:
                model = _table(("label", "999"))
            verify_scanned_table(
                page,
                model,
                native_trusted=False,
                ocr_image_fn=_ocr_alpha_five,
            )
        for page in doc:
            locate_tables(page)
            list(page.find_tables().tables)
    finally:
        doc.close()


def test_space_separated_numbers_are_two_tokens_one_number_stays_one() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "revenue 42 cost 7", fontsize=22)
    page.insert_text((72, 130), "ratio 3.14", fontsize=22)
    try:
        numeric, _, _ = _tokens_from_page_text_layer(page)
        assert numeric["42"] == 1
        assert numeric["7"] == 1
        assert numeric["3.14"] == 1
        assert numeric["42"] + numeric["7"] == 2
    finally:
        doc.close()


def _plain_text_skip_whitespace_breaks(page, mat, page_pix=None) -> str:
    """Mutant: ignore whitespace in the rawdict stream (must not ship)."""
    from socr.tables.source_evidence import (
        _glyph_bbox_has_ink,
        text_layer_span_is_visible,
    )

    try:
        raw = page.get_text("rawdict") or {}
    except Exception:
        return ""
    words: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            words.append("".join(current))
            current.clear()

    for block in raw.get("blocks", ()):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                if not isinstance(span, dict):
                    continue
                alpha = span.get("alpha", 255)
                for ch in span.get("chars", ()):
                    if not isinstance(ch, dict):
                        continue
                    bbox = ch.get("bbox")
                    char = ch.get("c")
                    if not bbox or len(bbox) < 4 or not char:
                        continue
                    if not text_layer_span_is_visible({"alpha": alpha}):
                        continue
                    if char.isspace():
                        continue
                    if not _glyph_bbox_has_ink(page, mat, bbox, page_pix=page_pix):
                        continue
                    current.append(char)
            flush()
    return " ".join(words)


def test_whitespace_word_break_mutant_merges_separated_numbers(monkeypatch) -> None:
    """Dropping whitespace breaks must merge ``42`` and ``7`` (guard is load-bearing)."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "revenue 42 cost 7", fontsize=22)
    try:
        numeric, _, _ = _tokens_from_page_text_layer(page)
        assert numeric["42"] == 1 and numeric["7"] == 1
        monkeypatch.setattr(
            "socr.tables.source_evidence._plain_text_from_visible_chars",
            _plain_text_skip_whitespace_breaks,
        )
        merged, _, _ = _tokens_from_page_text_layer(page)
        assert merged.get("42", 0) != 1 or merged.get("7", 0) != 1
    finally:
        doc.close()


def test_build_born_digital_evidence_reads_visible_words_not_scanned_path() -> None:
    """P2-4: production path ``build_born_digital_evidence``, not ``build_scanned_evidence``."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "alpha 5")
    page.insert_text((72, 120), "hidden 999", render_mode=3)
    try:
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            bundle = build_born_digital_evidence(page, ocr_image_fn=_ocr_alpha_five)
        assert bundle.numeric.get("5", 0) >= 1
        assert bundle.numeric.get("999", 0) == 0
    finally:
        doc.close()
