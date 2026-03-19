"""Tests for OutputNormalizer — engine-specific and generic cleanup."""

from pathlib import Path

from socr.core.normalizer import OutputNormalizer


def _n(text: str, engine: str = "") -> str:
    """Shortcut: normalize *text* through a fresh OutputNormalizer."""
    return OutputNormalizer().normalize(text, engine=engine)


# ── DeepSeek / GLM engine-specific ─────────────────────────────────────


class TestDeepSeekGLM:
    """Grounding tags, bounding boxes, HTML tags stripped."""

    def test_ref_tags_removed(self) -> None:
        text = "Hello <|ref|>some ref<|/ref|> world"
        assert _n(text, "deepseek") == "Hello  world"

    def test_det_tags_removed(self) -> None:
        text = "Token <|det|>[[10,20,30,40]]<|/det|> rest"
        assert _n(text, "deepseek") == "Token  rest"

    def test_bare_special_tokens_removed(self) -> None:
        text = "<|im_start|>system\nContent<|im_end|>"
        assert _n(text, "deepseek") == "system\nContent"

    def test_bare_bounding_boxes_removed(self) -> None:
        text = "Text [[100, 200, 300, 400]] more"
        assert _n(text, "deepseek") == "Text  more"

    def test_html_br_converted(self) -> None:
        text = "Line one<br/>Line two<BR>Line three"
        assert _n(text, "deepseek") == "Line one\nLine two\nLine three"

    def test_html_tags_stripped(self) -> None:
        text = "<div>Some <b>bold</b> text</div>"
        assert _n(text, "deepseek") == "Some bold text"

    def test_glm_uses_same_cleanup(self) -> None:
        text = "Hello <|ref|>x<|/ref|> world"
        assert _n(text, "glm") == "Hello  world"

    def test_deepseek_vllm_uses_same_cleanup(self) -> None:
        text = "Hello <|ref|>x<|/ref|> world"
        assert _n(text, "deepseek-vllm") == "Hello  world"


# ── Mistral engine-specific ─────────────────────────────────────────────


class TestMistral:
    """Standalone header and metadata lines stripped."""

    def test_standalone_ocr_results_header(self) -> None:
        text = "# OCR Results\n\nActual content here."
        assert _n(text, "mistral") == "Actual content here."

    def test_metadata_lines_stripped(self) -> None:
        text = (
            "**Original File:** paper.pdf\n"
            "**Processed:** 2025-01-01\n"
            "\n"
            "Real content."
        )
        assert _n(text, "mistral") == "Real content."

    def test_mixed_header_and_meta(self) -> None:
        text = (
            "# OCR Results\n\n"
            "**Original File:** test.pdf\n"
            "**Processing Time:** 5s\n"
            "\n"
            "Body text."
        )
        assert _n(text, "mistral") == "Body text."


# ── Nougat engine-specific ──────────────────────────────────────────────


class TestNougat:
    """LaTeX preamble and \\end{document} stripped."""

    def test_latex_preamble_stripped(self) -> None:
        text = (
            "\\documentclass{article}\n"
            "\\usepackage{amsmath}\n"
            "\\begin{document}\n"
            "Real content here."
        )
        assert _n(text, "nougat") == "Real content here."

    def test_latex_end_stripped(self) -> None:
        text = "Some content.\n\\end{document}\n"
        assert _n(text, "nougat") == "Some content."

    def test_preamble_with_options(self) -> None:
        text = (
            "\\documentclass[12pt]{article}\n"
            "\\begin{document}\n"
            "Body."
        )
        assert _n(text, "nougat") == "Body."


# ── Marker engine-specific ──────────────────────────────────────────────


class TestMarker:
    """[MISSING_PAGE_*] markers stripped."""

    def test_missing_page_post(self) -> None:
        text = "Page 1 content\n\n[MISSING_PAGE_POST]\n\nPage 3 content"
        result = _n(text, "marker")
        assert "[MISSING_PAGE_POST]" not in result
        assert "Page 1 content" in result
        assert "Page 3 content" in result

    def test_missing_page_post_with_number(self) -> None:
        text = "Content [MISSING_PAGE_POST:5] more"
        assert "[MISSING_PAGE_POST:5]" not in _n(text, "marker")

    def test_missing_page_empty(self) -> None:
        text = "Before [MISSING_PAGE_EMPTY:2] after"
        result = _n(text, "marker")
        assert "[MISSING_PAGE_EMPTY:2]" not in result
        assert "Before" in result
        assert "after" in result


# ── Generic normalization ────────────────────────────────────────────────


class TestGenericNormalization:
    """CRLF, blank lines, trailing whitespace, unicode."""

    def test_crlf_to_lf(self) -> None:
        text = "Line one\r\nLine two\r\nLine three"
        assert "\r" not in _n(text)

    def test_bare_cr_to_lf(self) -> None:
        text = "Line one\rLine two"
        assert _n(text) == "Line one\nLine two"

    def test_collapse_excessive_blank_lines(self) -> None:
        text = "Para 1\n\n\n\n\nPara 2"
        assert _n(text) == "Para 1\n\nPara 2"

    def test_two_blank_lines_preserved(self) -> None:
        text = "Para 1\n\nPara 2"
        assert _n(text) == "Para 1\n\nPara 2"

    def test_trailing_whitespace_stripped(self) -> None:
        text = "Line one   \nLine two\t\nLine three"
        result = _n(text)
        for line in result.split("\n"):
            assert line == line.rstrip(), f"Trailing whitespace in: {line!r}"

    def test_smart_quotes_normalized(self) -> None:
        text = "\u201cHello\u201d and \u2018world\u2019"
        assert _n(text) == '"Hello" and \'world\''

    def test_em_dash_normalized(self) -> None:
        text = "word\u2014word"
        assert _n(text) == "word--word"

    def test_en_dash_normalized(self) -> None:
        text = "pages 1\u20135"
        assert _n(text) == "pages 1-5"

    def test_ellipsis_normalized(self) -> None:
        text = "and so on\u2026"
        assert _n(text) == "and so on..."

    def test_fi_ligature_normalized(self) -> None:
        text = "the \ufb01rst finding"
        assert _n(text) == "the first finding"

    def test_fl_ligature_normalized(self) -> None:
        text = "\ufb02ow of data"
        assert _n(text) == "flow of data"

    def test_ff_ligature_normalized(self) -> None:
        text = "e\ufb00ect"
        assert _n(text) == "effect"

    def test_ffi_ligature_normalized(self) -> None:
        text = "e\ufb03cient"
        assert _n(text) == "efficient"

    def test_ffl_ligature_normalized(self) -> None:
        text = "ba\ufb04e"
        assert _n(text) == "baffle"

    def test_nfkc_normalization(self) -> None:
        # Superscript 2 -> regular 2 under NFKC
        text = "x\u00b2 + y\u00b2"
        result = _n(text)
        assert "\u00b2" not in result
        assert "2" in result

    def test_empty_string(self) -> None:
        assert _n("") == ""

    def test_whitespace_only(self) -> None:
        assert _n("   \n\n  ") == ""


# ── Idempotency ──────────────────────────────────────────────────────────


class TestIdempotency:
    """Normalizing already-normalized text should produce identical output."""

    def test_idempotent_generic(self) -> None:
        text = "A normal paragraph.\n\nAnother paragraph."
        first = _n(text)
        second = _n(first)
        assert first == second

    def test_idempotent_deepseek(self) -> None:
        text = "Hello <|ref|>ref<|/ref|> world <|det|>[[1,2,3,4]]<|/det|>"
        first = _n(text, "deepseek")
        second = _n(first, "deepseek")
        assert first == second

    def test_idempotent_mistral(self) -> None:
        text = "# OCR Results\n\n**Original File:** foo.pdf\n\nBody."
        first = _n(text, "mistral")
        second = _n(first, "mistral")
        assert first == second

    def test_idempotent_nougat(self) -> None:
        text = "\\documentclass{article}\n\\begin{document}\nContent.\n\\end{document}"
        first = _n(text, "nougat")
        second = _n(first, "nougat")
        assert first == second

    def test_idempotent_marker(self) -> None:
        text = "Before [MISSING_PAGE_POST] after"
        first = _n(text, "marker")
        second = _n(first, "marker")
        assert first == second

    def test_idempotent_unicode(self) -> None:
        text = "\u201cquoted\u201d \u2014 em \u2013 en \ufb01 ligature"
        first = _n(text)
        second = _n(first)
        assert first == second


# ── Integration with BaseEngine._clean_output ────────────────────────────


class TestBaseEngineIntegration:
    """Verify _clean_output calls the normalizer."""

    def test_clean_output_strips_frontmatter_and_normalizes(self) -> None:
        from socr.engines.base import BaseEngine

        raw = "---\ntitle: Test\n---\n\nContent with trailing ws   \n\n\n\n\nMore."
        result = BaseEngine._clean_output(raw, engine="gemini")
        assert result == "Content with trailing ws\n\nMore."

    def test_clean_output_strips_mistral_header_and_normalizes(self) -> None:
        from socr.engines.base import BaseEngine

        raw = (
            "# OCR Results\n\n"
            "**Original File:** paper.pdf\n"
            "**Processed:** 2025-01-01\n"
            "\n---\n"
            "Body text with \u201csmart quotes\u201d."
        )
        result = BaseEngine._clean_output(raw, engine="mistral")
        assert result == 'Body text with "smart quotes".'

    def test_clean_output_backward_compat_no_engine(self) -> None:
        """Calling without engine= still works (generic normalization only)."""
        from socr.engines.base import BaseEngine

        raw = "Simple text\r\nwith CRLF"
        result = BaseEngine._clean_output(raw)
        assert result == "Simple text\nwith CRLF"


# ── Engine name is case-insensitive ──────────────────────────────────────


class TestEngineNameCaseInsensitive:
    def test_uppercase(self) -> None:
        text = "Hello <|ref|>x<|/ref|> world"
        assert _n(text, "DEEPSEEK") == "Hello  world"

    def test_mixed_case(self) -> None:
        text = "Hello <|ref|>x<|/ref|> world"
        assert _n(text, "DeepSeek") == "Hello  world"


# ── Phantom image stripping ─────────────────────────────────────────────


class TestStripPhantomImages:
    """strip_phantom_images removes unreachable markdown image refs."""

    def test_strips_relative_refs_without_output_dir(self) -> None:
        text = "Before\n\n![img](img-0.jpeg)\n\nAfter"
        norm = OutputNormalizer()
        result = norm.strip_phantom_images(text, output_dir=None)
        assert "![img]" not in result
        assert "Before" in result
        assert "After" in result

    def test_strips_extracted_images_path(self) -> None:
        text = "Content\n\n![Page 1](./extracted_images/page1.png)\n\nMore"
        norm = OutputNormalizer()
        result = norm.strip_phantom_images(text, output_dir=None)
        assert "![Page 1]" not in result
        assert "Content" in result
        assert "More" in result

    def test_preserves_http_urls(self) -> None:
        text = "See ![logo](https://example.com/logo.png) here"
        norm = OutputNormalizer()
        result = norm.strip_phantom_images(text, output_dir=None)
        assert "![logo](https://example.com/logo.png)" in result

    def test_preserves_data_uris(self) -> None:
        text = "Inline ![x](data:image/png;base64,abc) end"
        norm = OutputNormalizer()
        result = norm.strip_phantom_images(text, output_dir=None)
        assert "![x](data:image/png;base64,abc)" in result

    def test_preserves_existing_files(self, tmp_path: Path) -> None:
        # Create a real file
        img = tmp_path / "real.png"
        img.write_bytes(b"\x89PNG")
        text = f"See ![fig](real.png) here"
        norm = OutputNormalizer()
        result = norm.strip_phantom_images(text, output_dir=tmp_path)
        assert "![fig](real.png)" in result

    def test_strips_missing_files_with_output_dir(self, tmp_path: Path) -> None:
        text = "See ![fig](nonexistent.png) here"
        norm = OutputNormalizer()
        result = norm.strip_phantom_images(text, output_dir=tmp_path)
        assert "![fig]" not in result
        assert "See" in result

    def test_no_op_without_image_refs(self) -> None:
        text = "Plain text without any images."
        norm = OutputNormalizer()
        result = norm.strip_phantom_images(text)
        assert result == text

    def test_multiple_phantom_refs_stripped(self) -> None:
        text = (
            "Para 1\n\n"
            "![a](img1.png)\n\n"
            "Para 2\n\n"
            "![b](img2.jpg)\n\n"
            "Para 3"
        )
        norm = OutputNormalizer()
        result = norm.strip_phantom_images(text, output_dir=None)
        assert "![a]" not in result
        assert "![b]" not in result
        assert "Para 1" in result
        assert "Para 2" in result
        assert "Para 3" in result

    def test_blank_lines_collapsed_after_stripping(self) -> None:
        text = "A\n\n![x](phantom.png)\n\n\n\nB"
        norm = OutputNormalizer()
        result = norm.strip_phantom_images(text, output_dir=None)
        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in result
