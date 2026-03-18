"""Tests for born-digital PDF detection."""

from pathlib import Path

import fitz
import pytest

from socr.core.born_digital import (
    BornDigitalDetector,
    DocumentAssessment,
    PageAssessment,
)


# ---------------------------------------------------------------------------
# Helpers: create synthetic PDFs for testing
# ---------------------------------------------------------------------------


def _create_born_digital_pdf(path: Path, num_pages: int = 1) -> None:
    """Create a PDF with real text content (born-digital)."""
    doc = fitz.open()
    lines = [
        "This is a born-digital academic paper about economic growth and monetary",
        "policy in developing countries. The author presents a comprehensive analysis",
        "of fiscal multipliers across different exchange rate regimes. The empirical",
        "evidence suggests that government spending has larger effects during recessions",
        "than during expansions, consistent with theoretical predictions from New",
        "Keynesian models with credit constraints and heterogeneous agents.",
        "The methodology combines structural vector autoregression with panel data",
        "techniques to identify causal effects of policy interventions.",
    ]
    for _ in range(num_pages):
        page = doc.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=11, fontname="helv")
            y += 16
    doc.save(str(path))
    doc.close()


def _create_scanned_pdf(path: Path, num_pages: int = 1) -> None:
    """Create a PDF that simulates a scanned document (image-only, no text)."""
    doc = fitz.open()
    for _ in range(num_pages):
        page = doc.new_page()
        # Draw a gray rectangle to simulate a scanned image (no text layer)
        page.draw_rect(page.rect, color=(0.9, 0.9, 0.9), fill=(0.95, 0.95, 0.95))
    doc.save(str(path))
    doc.close()


def _create_mixed_pdf(path: Path) -> None:
    """Create a PDF with one born-digital page and one scanned page."""
    doc = fitz.open()

    # Page 1: born-digital — use insert_text with multiple lines to get enough words
    page1 = doc.new_page()
    lines = [
        "Abstract: We examine the relationship between central bank independence",
        "and inflation targeting in emerging market economies. Using a panel dataset",
        "spanning forty countries over three decades, we find that institutional",
        "reforms significantly reduce inflation persistence. Our results are robust",
        "to alternative specifications and hold across different subsamples.",
        "The empirical evidence is drawn from quarterly macroeconomic data and",
        "supplemented with institutional quality indicators from multiple sources.",
    ]
    y = 72
    for line in lines:
        page1.insert_text((72, y), line, fontsize=11, fontname="helv")
        y += 16

    # Page 2: scanned (image-only)
    page2 = doc.new_page()
    page2.draw_rect(page2.rect, color=(0.9, 0.9, 0.9), fill=(0.95, 0.95, 0.95))

    doc.save(str(path))
    doc.close()


def _create_garbage_ocr_pdf(path: Path) -> None:
    """Create a PDF with a garbage OCR text layer (simulates bad baked-in OCR)."""
    doc = fitz.open()
    page = doc.new_page()

    # Insert garbage text that looks like bad OCR output
    garbage = "a b c d e f g h i j k l " * 20  # single-char "words"
    tw = fitz.TextWriter(page.rect)
    tw.append((72, 72), garbage, fontsize=11, font=fitz.Font("helv"))
    tw.write_text(page)

    doc.save(str(path))
    doc.close()


def _create_sparse_text_pdf(path: Path) -> None:
    """Create a PDF with very little text (e.g., just a title page)."""
    doc = fitz.open()
    page = doc.new_page()
    tw = fitz.TextWriter(page.rect)
    tw.append((72, 72), "Title Page", fontsize=24, font=fitz.Font("helv"))
    tw.write_text(page)
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# Tests: BornDigitalDetector
# ---------------------------------------------------------------------------


class TestBornDigitalDetector:
    """Tests for the BornDigitalDetector class."""

    def test_born_digital_single_page(self, tmp_path: Path) -> None:
        """Born-digital page is correctly identified."""
        pdf_path = tmp_path / "born_digital.pdf"
        _create_born_digital_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)

        assert isinstance(result, DocumentAssessment)
        assert result.page_count == 1
        assert result.is_fully_born_digital
        assert not result.is_fully_scanned
        assert not result.is_mixed

        page = result.pages[0]
        assert page.is_born_digital
        assert page.page_num == 1
        assert len(page.native_text) > 0
        assert page.confidence > 0.7
        assert page.word_count > 10

    def test_scanned_single_page(self, tmp_path: Path) -> None:
        """Scanned page (no text layer) is correctly identified."""
        pdf_path = tmp_path / "scanned.pdf"
        _create_scanned_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)

        assert result.page_count == 1
        assert result.is_fully_scanned
        assert not result.is_fully_born_digital

        page = result.pages[0]
        assert not page.is_born_digital
        assert page.native_text == ""

    def test_mixed_document(self, tmp_path: Path) -> None:
        """Document with both born-digital and scanned pages."""
        pdf_path = tmp_path / "mixed.pdf"
        _create_mixed_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)

        assert result.page_count == 2
        assert result.is_mixed
        assert result.born_digital_count == 1
        assert result.scanned_count == 1
        assert result.born_digital_pages() == [1]
        assert result.scanned_pages() == [2]

    def test_multi_page_born_digital(self, tmp_path: Path) -> None:
        """Multi-page born-digital PDF."""
        pdf_path = tmp_path / "multi.pdf"
        _create_born_digital_pdf(pdf_path, num_pages=5)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)

        assert result.page_count == 5
        assert result.is_fully_born_digital
        assert result.born_digital_count == 5
        assert len(result.born_digital_pages()) == 5

    def test_sparse_text_not_born_digital(self, tmp_path: Path) -> None:
        """Page with very little text (title page) should not be born-digital."""
        pdf_path = tmp_path / "sparse.pdf"
        _create_sparse_text_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)

        page = result.pages[0]
        assert not page.is_born_digital

    def test_garbage_ocr_not_born_digital(self, tmp_path: Path) -> None:
        """Page with garbage OCR text should not be classified as born-digital."""
        pdf_path = tmp_path / "garbage_ocr.pdf"
        _create_garbage_ocr_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)

        page = result.pages[0]
        assert not page.is_born_digital
        assert page.native_text == ""

    def test_detect_page_single(self, tmp_path: Path) -> None:
        """detect_page() works for a single page."""
        pdf_path = tmp_path / "test.pdf"
        _create_born_digital_pdf(pdf_path, num_pages=3)

        detector = BornDigitalDetector()
        page = detector.detect_page(pdf_path, page_num=2)

        assert isinstance(page, PageAssessment)
        assert page.page_num == 2
        assert page.is_born_digital

    def test_detect_page_out_of_range(self, tmp_path: Path) -> None:
        """detect_page() raises ValueError for invalid page number."""
        pdf_path = tmp_path / "test.pdf"
        _create_born_digital_pdf(pdf_path, num_pages=2)

        detector = BornDigitalDetector()
        with pytest.raises(ValueError, match="out of range"):
            detector.detect_page(pdf_path, page_num=5)

    def test_file_not_found(self) -> None:
        """Raises FileNotFoundError for missing PDF."""
        detector = BornDigitalDetector()
        with pytest.raises(FileNotFoundError):
            detector.detect(Path("/nonexistent/file.pdf"))

    def test_custom_thresholds(self, tmp_path: Path) -> None:
        """Custom thresholds can be passed to the detector."""
        pdf_path = tmp_path / "test.pdf"
        _create_born_digital_pdf(pdf_path)

        # Require extremely high char count — should reject the page
        detector = BornDigitalDetector(min_chars=100_000)
        result = detector.detect(pdf_path)
        assert not result.pages[0].is_born_digital

    def test_page_assessment_fields(self, tmp_path: Path) -> None:
        """PageAssessment fields are populated correctly."""
        pdf_path = tmp_path / "test.pdf"
        _create_born_digital_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.char_count > 0
        assert page.word_count > 0
        assert page.font_count >= 0
        assert 0.0 <= page.confidence <= 1.0
        assert isinstance(page.has_images, bool)
        assert isinstance(page.notes, list)
        assert len(page.notes) > 0

    def test_document_assessment_properties(self, tmp_path: Path) -> None:
        """DocumentAssessment summary properties are consistent."""
        pdf_path = tmp_path / "test.pdf"
        _create_born_digital_pdf(pdf_path, num_pages=3)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)

        assert result.born_digital_count + result.scanned_count == result.page_count
        assert result.path == pdf_path


# ---------------------------------------------------------------------------
# Tests: DocumentHandle.detect_born_digital()
# ---------------------------------------------------------------------------


class TestDocumentHandleBornDigital:
    """Tests for the DocumentHandle.detect_born_digital() integration."""

    def test_document_handle_detect(self, tmp_path: Path) -> None:
        """DocumentHandle.detect_born_digital() returns a DocumentAssessment."""
        from socr.core.document import DocumentHandle

        pdf_path = tmp_path / "test.pdf"
        _create_born_digital_pdf(pdf_path)

        handle = DocumentHandle.from_path(pdf_path)
        result = handle.detect_born_digital()

        assert isinstance(result, DocumentAssessment)
        assert result.page_count == handle.page_count
        assert result.is_fully_born_digital


class TestNativeTextExtraction:
    """Tests that native text is correctly extracted from born-digital pages."""

    def test_extracted_text_is_nonempty(self, tmp_path: Path) -> None:
        """Born-digital pages have non-empty native_text."""
        pdf_path = tmp_path / "test.pdf"
        _create_born_digital_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)

        page = result.pages[0]
        assert page.is_born_digital
        assert len(page.native_text) > 50
        assert "economic" in page.native_text.lower() or "monetary" in page.native_text.lower()

    def test_scanned_pages_have_no_native_text(self, tmp_path: Path) -> None:
        """Scanned pages have empty native_text."""
        pdf_path = tmp_path / "scanned.pdf"
        _create_scanned_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)

        page = result.pages[0]
        assert not page.is_born_digital
        assert page.native_text == ""
