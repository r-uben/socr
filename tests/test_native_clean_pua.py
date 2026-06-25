"""Tests for issue #92: native-layer cleaning + unmapped-math-glyph (PUA) detection.

Covers the boundary cleaner (zero-width / soft-hyphen strip, exotic-space
normalization), PUA detection as the weak-ToUnicode fingerprint, the math-font
allowlist additions, and the end-to-end wiring through _assess_page.
"""

from pathlib import Path

import fitz

from socr.core.born_digital import (
    _MATH_FONT_RE,
    BornDigitalDetector,
    clean_native_text,
    count_pua_chars,
)

ZWSP = chr(0x200B)
SHY = chr(0x00AD)
NBSP = chr(0x00A0)
NARROW_NBSP = chr(0x202F)
PUA_F766 = chr(0xF766)  # UniMath script-T
PUA_E14B = chr(0xE14B)  # STIXNonUnicode symbol


class TestCleanNativeText:
    def test_strips_zero_width_and_soft_hyphen(self) -> None:
        cleaned, zw, sp = clean_native_text(f"hetero{SHY}geneous c{ZWSP}i,t")
        assert cleaned == "heterogeneous ci,t"
        assert zw == 2
        assert sp == 0

    def test_normalizes_exotic_spaces_to_ascii(self) -> None:
        cleaned, zw, sp = clean_native_text(f"a{NBSP}b{NARROW_NBSP}c")
        assert cleaned == "a b c"
        assert zw == 0
        assert sp == 2

    def test_preserves_pua_and_real_math_alphanumerics(self) -> None:
        # PUA is content signal, never altered; 𝒯 (U+1D4AF) is real Unicode, kept.
        cleaned, _, _ = clean_native_text(f"{PUA_F766} and \U0001d4af")
        assert cleaned == f"{PUA_F766} and \U0001d4af"

    def test_idempotent(self) -> None:
        once, _, _ = clean_native_text(f"x{ZWSP}y{NBSP}z")
        twice, zw, sp = clean_native_text(once)
        assert once == twice
        assert zw == 0 and sp == 0

    def test_clean_text_unchanged(self) -> None:
        cleaned, zw, sp = clean_native_text("ordinary prose, nothing to do.")
        assert cleaned == "ordinary prose, nothing to do."
        assert zw == 0 and sp == 0


class TestCountPua:
    def test_counts_bmp_pua(self) -> None:
        assert count_pua_chars(f"a{PUA_F766}b{PUA_E14B}c") == 2

    def test_counts_supplementary_pua_planes(self) -> None:
        assert count_pua_chars("\U000f0000\U00100000") == 2

    def test_zero_for_clean_text_and_math_alphanumerics(self) -> None:
        assert count_pua_chars("plain text with script T \U0001d4af and Greek γ") == 0


class TestMathFontRegex:
    def test_includes_size_variant_and_nonunicode_symbol_fonts(self) -> None:
        assert _MATH_FONT_RE.search("EPTAMP+STIXSizeThreeSym-Regular")
        assert _MATH_FONT_RE.search("ABCDEF+STIXNonUnicode-Regular")

    def test_excludes_general_body_fonts(self) -> None:
        # STIXGeneral / Times set body text too -> must NOT high-confidence-trigger
        # (those cases are caught via the PUA signal instead).
        assert not _MATH_FONT_RE.search("IKEVWL+STIXGeneral-Regular")
        assert not _MATH_FONT_RE.search("IKEVWL+TimesLTStd-Roman")


def _born_digital_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in [
        "This is a born-digital academic paper with more than enough words to clear",
        "the born-digital floor and be classified as a clean native text layer here,",
        "presenting a comprehensive analysis with several full sentences of real prose.",
    ]:
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
        y += 16
    doc.save(str(path))
    doc.close()


class TestAssessPageWiring:
    def test_invisibles_cleaned_and_pua_flagged(self, tmp_path: Path, monkeypatch) -> None:
        pdf = tmp_path / "doc.pdf"
        _born_digital_pdf(pdf)
        detector = BornDigitalDetector()
        doc = fitz.open(str(pdf))
        page = doc[0]

        crafted = (
            f"Optimists set a lift-off date {PUA_F766}{ZWSP}o(T) and the hetero{SHY}geneous "
            f"agents{NBSP}consume more, with enough trailing words to stay above the floor "
            "for a born digital clean text layer classification indeed yes."
        )
        orig_get_text = page.get_text

        def fake_get_text(*args, **kwargs):
            mode = args[0] if args else kwargs.get("option", "text")
            if mode == "text":
                return crafted
            return orig_get_text(*args, **kwargs)

        monkeypatch.setattr(page, "get_text", fake_get_text)

        pa = detector._assess_page(page, 1)

        assert pa.is_born_digital
        # invisibles stripped from the shipped native text
        assert ZWSP not in pa.native_text
        assert SHY not in pa.native_text
        assert NBSP not in pa.native_text
        assert "heterogeneous" in pa.native_text
        # PUA preserved (content), and flagged
        assert PUA_F766 in pa.native_text
        assert pa.has_unmapped_math_glyphs is True
        # surfaced in notes (never silent)
        assert any("unmapped math glyphs" in n for n in pa.notes)
        assert any("native layer cleaned" in n for n in pa.notes)
        # the flag must NOT force whole-page OCR (wrong lane)
        assert pa.needs_ocr_enhancement is False

    def test_clean_page_has_no_flag(self, tmp_path: Path) -> None:
        pdf = tmp_path / "clean.pdf"
        _born_digital_pdf(pdf)
        detector = BornDigitalDetector()
        assessment = detector.detect(str(pdf))
        pa = assessment.pages[0]
        assert pa.is_born_digital
        assert pa.has_unmapped_math_glyphs is False
        assert not any("unmapped math glyphs" in n for n in pa.notes)
