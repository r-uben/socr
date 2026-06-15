"""Tests for born-digital PDF detection."""

from pathlib import Path
from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# Helpers: create synthetic PDFs with tables/images/equations
# ---------------------------------------------------------------------------


def _create_pdf_with_table(path: Path) -> None:
    """Create a born-digital PDF with a table embedded via text layout.

    Uses PyMuPDF's table-like text insertion to create content that
    find_tables() can detect.
    """
    doc = fitz.open()
    page = doc.new_page()

    # Add enough prose to pass born-digital thresholds
    prose_lines = [
        "This document presents regression results from our empirical analysis.",
        "The following table summarizes the key coefficients and standard errors",
        "for the main specification described in the methodology section above.",
        "We estimate the model using ordinary least squares with robust standard",
        "errors clustered at the country level following standard practice.",
    ]
    y = 72
    for line in prose_lines:
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
        y += 16

    # Insert a simple table using shapes to create grid lines that
    # find_tables() can detect. The table has 3 columns and 4 rows.
    table_top = y + 20
    col_widths = [150, 100, 100]
    row_height = 20
    num_rows = 4
    x_start = 72

    # Draw horizontal lines
    shape = page.new_shape()
    for row in range(num_rows + 1):
        y_pos = table_top + row * row_height
        shape.draw_line(
            fitz.Point(x_start, y_pos),
            fitz.Point(x_start + sum(col_widths), y_pos),
        )

    # Draw vertical lines
    x_pos = x_start
    for col_w in [0] + col_widths:
        x_pos += col_w
        shape.draw_line(
            fitz.Point(x_pos - col_widths[0] if col_w == 0 else x_pos, table_top),
            fitz.Point(
                x_pos - col_widths[0] if col_w == 0 else x_pos, table_top + num_rows * row_height
            ),
        )

    # Actually, let's draw vertical lines properly
    shape = page.new_shape()
    total_width = sum(col_widths)

    # Horizontal lines
    for row in range(num_rows + 1):
        y_pos = table_top + row * row_height
        shape.draw_line(fitz.Point(x_start, y_pos), fitz.Point(x_start + total_width, y_pos))

    # Vertical lines
    x_pos = x_start
    for i in range(len(col_widths) + 1):
        shape.draw_line(
            fitz.Point(x_pos, table_top), fitz.Point(x_pos, table_top + num_rows * row_height)
        )
        if i < len(col_widths):
            x_pos += col_widths[i]

    shape.finish(color=(0, 0, 0), width=0.5)
    shape.commit()

    # Insert cell text
    cells = [
        ["Variable", "Coefficient", "Std Error"],
        ["GDP Growth", "0.523", "0.041"],
        ["Inflation", "-0.187", "0.029"],
        ["Trade Open", "0.312", "0.056"],
    ]
    for row_idx, row_data in enumerate(cells):
        for col_idx, cell_text in enumerate(row_data):
            cell_x = x_start + sum(col_widths[:col_idx]) + 5
            cell_y = table_top + row_idx * row_height + 14
            page.insert_text((cell_x, cell_y), cell_text, fontsize=9, fontname="helv")

    doc.save(str(path))
    doc.close()


def _create_pdf_with_image(path: Path) -> None:
    """Create a born-digital PDF with an embedded raster image (figure)."""
    import io

    from PIL import Image

    doc = fitz.open()
    page = doc.new_page()

    # Add prose text
    prose_lines = [
        "Figure 1 below shows the impulse response function from our structural",
        "vector autoregression model. The shaded area represents the confidence",
        "interval at the ninety-five percent level. Results indicate a strong",
        "and persistent effect of monetary policy shocks on output growth.",
        "The peak effect occurs approximately four quarters after the initial shock.",
    ]
    y = 72
    for line in prose_lines:
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
        y += 16

    # Create a small synthetic image and embed it
    img = Image.new("RGB", (200, 150), color=(200, 200, 255))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    page.insert_image(
        fitz.Rect(72, y + 10, 272, y + 160),
        stream=img_bytes.read(),
    )

    doc.save(str(path))
    doc.close()


def _create_pdf_with_equations(path: Path) -> None:
    """Create a born-digital PDF with LaTeX-like equation text."""
    doc = fitz.open()
    page = doc.new_page()

    lines = [
        "The utility function is defined as follows for the representative agent",
        "in the economy with heterogeneous preferences and risk aversion.",
        "We specify the following functional form for estimation purposes:",
        r"$$U(c) = \frac{c^{1-\sigma}}{1-\sigma}$$",
        "where sigma represents the coefficient of relative risk aversion",
        "and c denotes per capita consumption. The budget constraint is",
        r"\begin{equation} c_t + k_{t+1} = w_t + r_t k_t \end{equation}",
        "The first-order conditions yield the standard Euler equation",
        "which we use as a moment condition for our estimation strategy.",
    ]
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
        y += 16

    doc.save(str(path))
    doc.close()


def _create_pdf_with_mixed_content(path: Path) -> None:
    """Create a PDF with prose on page 1 and a table on page 2."""
    import io

    from PIL import Image

    doc = fitz.open()

    # Page 1: prose only
    page1 = doc.new_page()
    prose_lines = [
        "This is the introduction section of our paper on monetary policy.",
        "We examine the effects of unconventional monetary policy measures",
        "on financial markets in emerging economies during the recent crisis.",
        "The empirical analysis uses a difference-in-differences framework",
        "to identify the causal impact of central bank interventions.",
        "Our sample covers twenty emerging market economies over the period.",
    ]
    y = 72
    for line in prose_lines:
        page1.insert_text((72, y), line, fontsize=11, fontname="helv")
        y += 16

    # Page 2: text + table + image
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Table 1 reports the main results from our regression analysis below.",
        fontsize=11,
        fontname="helv",
    )
    page2.insert_text(
        (72, 88),
        "The coefficients are statistically significant at conventional levels.",
        fontsize=11,
        fontname="helv",
    )
    page2.insert_text(
        (72, 104),
        "Standard errors are clustered at the country level throughout.",
        fontsize=11,
        fontname="helv",
    )

    # Draw a table with lines
    table_top = 130
    col_widths = [120, 80, 80]
    row_height = 18
    num_rows = 3
    x_start = 72
    total_width = sum(col_widths)

    shape = page2.new_shape()
    for row in range(num_rows + 1):
        y_pos = table_top + row * row_height
        shape.draw_line(fitz.Point(x_start, y_pos), fitz.Point(x_start + total_width, y_pos))
    x_pos = x_start
    for i in range(len(col_widths) + 1):
        shape.draw_line(
            fitz.Point(x_pos, table_top), fitz.Point(x_pos, table_top + num_rows * row_height)
        )
        if i < len(col_widths):
            x_pos += col_widths[i]
    shape.finish(color=(0, 0, 0), width=0.5)
    shape.commit()

    cells = [
        ["Variable", "Coeff", "SE"],
        ["Interest Rate", "0.45", "0.12"],
        ["Exchange Rate", "-0.23", "0.08"],
    ]
    for row_idx, row_data in enumerate(cells):
        for col_idx, cell_text in enumerate(row_data):
            cell_x = x_start + sum(col_widths[:col_idx]) + 5
            cell_y = table_top + row_idx * row_height + 13
            page2.insert_text((cell_x, cell_y), cell_text, fontsize=9, fontname="helv")

    # Also embed a small image
    img = Image.new("RGB", (100, 80), color=(220, 240, 220))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    page2.insert_image(
        fitz.Rect(72, 210, 172, 290),
        stream=img_bytes.read(),
    )

    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# Tests: Table detection
# ---------------------------------------------------------------------------


class TestTableDetection:
    """Tests for table detection on born-digital pages."""

    def test_page_with_table_detected(self, tmp_path: Path) -> None:
        """A page with a grid-line table is detected as having tables."""
        pdf_path = tmp_path / "table.pdf"
        _create_pdf_with_table(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.is_born_digital
        assert page.has_tables

    def test_prose_only_page_no_tables(self, tmp_path: Path) -> None:
        """A prose-only born-digital page has no tables."""
        pdf_path = tmp_path / "prose.pdf"
        _create_born_digital_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.is_born_digital
        assert not page.has_tables

    def test_clean_table_page_does_not_need_ocr_enhancement(self, tmp_path: Path) -> None:
        """A clean native table should stay native/structured, not whole-page OCR."""
        pdf_path = tmp_path / "table.pdf"
        _create_pdf_with_table(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.has_tables
        assert not page.needs_ocr_enhancement

    def test_prose_only_no_ocr_enhancement(self, tmp_path: Path) -> None:
        """Prose-only pages should not need OCR enhancement."""
        pdf_path = tmp_path / "prose.pdf"
        _create_born_digital_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert not page.needs_ocr_enhancement


# ---------------------------------------------------------------------------
# Tests: Figure detection
# ---------------------------------------------------------------------------


class TestFigureDetection:
    """Tests for figure/image detection on born-digital pages."""

    def test_page_with_image_detected(self, tmp_path: Path) -> None:
        """A page with an embedded image has has_figures=True."""
        pdf_path = tmp_path / "figure.pdf"
        _create_pdf_with_image(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.is_born_digital
        assert page.has_figures
        assert page.has_images  # has_figures is based on has_images

    def test_page_without_image_no_figures(self, tmp_path: Path) -> None:
        """A text-only page has has_figures=False."""
        pdf_path = tmp_path / "text.pdf"
        _create_born_digital_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.is_born_digital
        assert not page.has_figures

    def test_clean_figure_page_does_not_need_ocr_enhancement(self, tmp_path: Path) -> None:
        """Figures are extracted/described separately; they do not require OCR."""
        pdf_path = tmp_path / "figure.pdf"
        _create_pdf_with_image(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.has_figures
        assert not page.needs_ocr_enhancement


# ---------------------------------------------------------------------------
# Tests: Equation detection
# ---------------------------------------------------------------------------


class TestEquationDetection:
    """Tests for equation/math detection via text patterns."""

    def test_page_with_equations_detected(self, tmp_path: Path) -> None:
        """A page with LaTeX equation markup is detected."""
        pdf_path = tmp_path / "equations.pdf"
        _create_pdf_with_equations(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.is_born_digital
        assert page.has_equations

    def test_prose_only_no_equations(self, tmp_path: Path) -> None:
        """A prose-only page has no equations."""
        pdf_path = tmp_path / "prose.pdf"
        _create_born_digital_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert not page.has_equations

    def test_equation_detection_patterns(self) -> None:
        """Equation detection recognizes various LaTeX patterns."""
        detector = BornDigitalDetector()

        # LaTeX commands
        assert detector._detect_equations(r"The formula is \frac{a}{b}")
        assert detector._detect_equations(r"We compute \sum_{i=1}^{n} x_i")
        assert detector._detect_equations(r"The integral \int_0^1 f(x) dx")
        assert detector._detect_equations(r"\begin{equation} y = mx + b \end{equation}")

        # Display math
        assert detector._detect_equations(r"$$E = mc^2$$")
        assert detector._detect_equations(r"\[ F = ma \]")

        # No equations
        assert not detector._detect_equations("This is plain text about economics.")
        assert not detector._detect_equations("The price is $50 per unit.")
        assert not detector._detect_equations("")

    def test_clean_equation_page_does_not_need_whole_page_ocr(self, tmp_path: Path) -> None:
        """Clean equations are metadata for future regional recovery, not VLM routing."""
        pdf_path = tmp_path / "equations.pdf"
        _create_pdf_with_equations(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.has_equations
        assert not page.needs_ocr_enhancement


class TestMathFontDetection:
    """Tests for font-based math detection (_detect_math_fonts).

    The primary motivation: PyMuPDF mangles math typeset with CM/STIX fonts
    (superscripts flatten, Greek drops) but the *extracted text* looks fine
    to any string-based checker.  Font metadata is the only reliable pre-
    extraction signal.
    """

    def test_cmmi_font_detected(self) -> None:
        """CMMI (Computer Modern Math Italic) triggers detection."""
        page = MagicMock()
        page.get_fonts.return_value = [
            (1, "otf", "Type1", "CMMI10", "CMMI10", "WinAnsi"),
        ]
        assert BornDigitalDetector._detect_math_fonts(page)

    def test_cmsy_font_detected(self) -> None:
        """CMSY (Computer Modern Symbol) triggers detection."""
        page = MagicMock()
        page.get_fonts.return_value = [
            (2, "otf", "Type1", "CMSY10", "CMSY10", "WinAnsi"),
        ]
        assert BornDigitalDetector._detect_math_fonts(page)

    def test_ams_fonts_detected(self) -> None:
        """MSAM/MSBM (AMS symbol fonts) trigger detection."""
        page = MagicMock()
        page.get_fonts.return_value = [
            (3, "otf", "Type1", "MSAM10", "MSAM10", "WinAnsi"),
        ]
        assert BornDigitalDetector._detect_math_fonts(page)

    def test_stix_math_detected(self) -> None:
        """STIXMath (modern OpenType math) triggers detection."""
        page = MagicMock()
        page.get_fonts.return_value = [
            (4, "otf", "TrueType", "STIXMath-Regular", "STIXMath-Regular", ""),
        ]
        assert BornDigitalDetector._detect_math_fonts(page)

    def test_latin_modern_math_detected(self) -> None:
        """LatinModernMath triggers detection."""
        page = MagicMock()
        page.get_fonts.return_value = [
            (5, "otf", "TrueType", "LatinModernMath-Regular", "LMMath", ""),
        ]
        assert BornDigitalDetector._detect_math_fonts(page)

    def test_subset_prefix_handled(self) -> None:
        """Subset-prefixed names like 'ABCDEF+CMMI10' are matched correctly."""
        page = MagicMock()
        page.get_fonts.return_value = [
            (6, "otf", "Type1", "ABCDEF+CMMI10", "CMMI10", "WinAnsi"),
        ]
        assert BornDigitalDetector._detect_math_fonts(page)

    def test_prose_fonts_not_detected(self) -> None:
        """Common text fonts (Times, Helvetica) do not trigger detection."""
        page = MagicMock()
        page.get_fonts.return_value = [
            (1, "otf", "TrueType", "Times-Roman", "Times-Roman", "WinAnsi"),
            (2, "otf", "TrueType", "Helvetica-Bold", "Helvetica-Bold", "WinAnsi"),
            (3, "otf", "TrueType", "Arial-Italic", "Arial-Italic", "WinAnsi"),
        ]
        assert not BornDigitalDetector._detect_math_fonts(page)

    def test_empty_font_list(self) -> None:
        """Page with no fonts returns False (scanned/image page)."""
        page = MagicMock()
        page.get_fonts.return_value = []
        assert not BornDigitalDetector._detect_math_fonts(page)

    def test_get_fonts_exception_returns_false(self) -> None:
        """If get_fonts() raises, detection degrades gracefully to False."""
        page = MagicMock()
        page.get_fonts.side_effect = RuntimeError("PDF corruption")
        assert not BornDigitalDetector._detect_math_fonts(page)

    def test_mixed_prose_and_math_fonts(self) -> None:
        """A page with both prose and math fonts (inline equations) is detected."""
        page = MagicMock()
        page.get_fonts.return_value = [
            (1, "otf", "TrueType", "Times-Roman", "Times-Roman", "WinAnsi"),
            (2, "otf", "Type1", "CMMI10", "CMMI10", "WinAnsi"),  # inline math
            (3, "otf", "TrueType", "Helvetica", "Helvetica", "WinAnsi"),
        ]
        assert BornDigitalDetector._detect_math_fonts(page)


# ---------------------------------------------------------------------------
# Tests: Structured text extraction (markdown tables)
# ---------------------------------------------------------------------------


class TestStructuredExtraction:
    """Tests for extract_structured() markdown table rendering."""

    def test_table_to_markdown_format(self) -> None:
        """_table_to_markdown produces valid markdown table syntax."""
        detector = BornDigitalDetector()

        class FakeTable:
            def extract(self):
                return [
                    ["Col A", "Col B", "Col C"],
                    ["val1", "val2", "val3"],
                    ["val4", "val5", "val6"],
                ]

        md = detector._table_to_markdown(FakeTable())
        lines = md.strip().split("\n")

        # Header row
        assert lines[0] == "| Col A | Col B | Col C |"
        # Separator
        assert lines[1] == "| --- | --- | --- |"
        # Data rows
        assert lines[2] == "| val1 | val2 | val3 |"
        assert lines[3] == "| val4 | val5 | val6 |"

    def test_table_to_markdown_handles_none_cells(self) -> None:
        """None cells are replaced with empty strings."""
        detector = BornDigitalDetector()

        class FakeTable:
            def extract(self):
                return [
                    ["Header", None],
                    [None, "data"],
                ]

        md = detector._table_to_markdown(FakeTable())
        assert "| Header |  |" in md
        assert "|  | data |" in md

    def test_table_to_markdown_empty_table(self) -> None:
        """Empty table returns empty string."""
        detector = BornDigitalDetector()

        class FakeTable:
            def extract(self):
                return []

        assert detector._table_to_markdown(FakeTable()) == ""

    def test_extract_structured_on_table_page(self, tmp_path: Path) -> None:
        """extract_structured produces markdown with table content."""
        pdf_path = tmp_path / "table.pdf"
        _create_pdf_with_table(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        # The native_text should contain markdown table syntax
        assert page.is_born_digital
        assert page.has_tables
        assert "|" in page.native_text
        assert "---" in page.native_text

    def test_extract_structured_preserves_prose(self, tmp_path: Path) -> None:
        """extract_structured keeps prose text alongside tables."""
        pdf_path = tmp_path / "table.pdf"
        _create_pdf_with_table(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        # Should contain both prose and table content
        text_lower = page.native_text.lower()
        assert "regression" in text_lower or "empirical" in text_lower or "analysis" in text_lower
        assert "|" in page.native_text

    def test_extract_structured_prose_only_fallback(self, tmp_path: Path) -> None:
        """extract_structured on prose-only page returns plain text."""
        pdf_path = tmp_path / "prose.pdf"
        _create_born_digital_pdf(pdf_path)

        detector = BornDigitalDetector()

        with fitz.open(pdf_path) as doc:
            text = detector.extract_structured(doc[0])

        # Should be plain text, no markdown table markers
        assert len(text) > 50
        assert "economic" in text.lower() or "monetary" in text.lower()


# ---------------------------------------------------------------------------
# Tests: Mixed content (text + table + figures)
# ---------------------------------------------------------------------------


class TestMixedContent:
    """Tests for pages with mixed content types."""

    def test_mixed_content_page_detection(self, tmp_path: Path) -> None:
        """Page 2 of mixed content PDF has tables and figures."""
        pdf_path = tmp_path / "mixed.pdf"
        _create_pdf_with_mixed_content(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)

        # Page 1: prose only
        page1 = result.pages[0]
        assert page1.is_born_digital
        assert not page1.has_tables
        assert not page1.has_figures
        assert not page1.needs_ocr_enhancement

        # Page 2: table + image
        page2 = result.pages[1]
        assert page2.is_born_digital
        assert page2.has_tables
        assert page2.has_figures  # has embedded image
        assert not page2.needs_ocr_enhancement

    def test_complex_content_notes_list_types(self, tmp_path: Path) -> None:
        """Notes field lists the detected content types."""
        pdf_path = tmp_path / "mixed.pdf"
        _create_pdf_with_mixed_content(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page2 = result.pages[1]

        notes_text = " ".join(page2.notes)
        assert "complex content" in notes_text.lower()

    def test_backward_compatible_defaults(self) -> None:
        """New PageAssessment fields have safe defaults for backward compat."""
        pa = PageAssessment(
            page_num=1,
            is_born_digital=True,
            native_text="some text",
            confidence=0.9,
        )
        assert pa.has_tables is False
        assert pa.has_figures is False
        assert pa.has_equations is False
        assert pa.needs_ocr_enhancement is False


# ---------------------------------------------------------------------------
# Helpers: sparse/figure page PDFs for GH-35 characterization tests
# ---------------------------------------------------------------------------


def _create_figure_page_with_short_caption(path: Path) -> None:
    """Born-digital figure page: large embedded image + short caption text.

    The caption has fewer than MIN_WORDS_PER_PAGE (15) words but is genuinely
    clean native text.  Prior to GH-35, this page was wrongly classified as
    SCANNED because the word-count gate fired before quality checks.
    """
    import io

    from PIL import Image

    doc = fitz.open()
    page = doc.new_page()

    # Caption text — 10 words, below the old MIN_WORDS_PER_PAGE=15 threshold
    page.insert_text(
        (72, 72),
        "Figure 1. Impulse response function.",
        fontsize=11,
        fontname="helv",
    )
    page.insert_text(
        (72, 90),
        "Notes: 90% confidence intervals shown.",
        fontsize=9,
        fontname="helv",
    )

    # Large embedded image covering most of the page body
    img = Image.new("RGB", (400, 500), color=(200, 200, 255))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    page.insert_image(fitz.Rect(72, 110, 472, 610), stream=img_bytes.read())

    doc.save(str(path))
    doc.close()


def _create_sparse_section_heading_pdf(path: Path) -> None:
    """Born-digital sparse page: section/part heading with 6-8 clean words.

    Characterises a page that has enough chars (>=50) but fewer than 15 words.
    Prior to GH-35, the word-count gate classified this as SCANNED even though
    the text is perfectly clean and produced by proper fonts.
    """
    doc = fitz.open()
    page = doc.new_page()

    # Single heading line — 7 words, chars > 50 due to long words
    page.insert_text(
        (72, 350),
        "Part II. Empirical Applications and Robustness Checks",
        fontsize=24,
        fontname="helv",
    )

    doc.save(str(path))
    doc.close()


def _create_image_only_scan_pdf(path: Path) -> None:
    """True image-only scan: one large raster image, zero text layer.

    This must remain classified as SCANNED; the char-count gate catches it
    because PyMuPDF returns no text for a page that is purely a raster bitmap.
    """
    import io

    from PIL import Image

    doc = fitz.open()
    page = doc.new_page()

    # Full-page grayscale bitmap simulating a scanned document
    img = Image.new("L", (1800, 2400), color=245)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    page.insert_image(page.rect, stream=img_bytes.read())

    doc.save(str(path))
    doc.close()


def _create_decorative_front_matter_pdf(path: Path) -> None:
    """Decorative front matter: image + very short publisher/watermark text.

    Two tokens — "Confidential Draft" — which falls below MIN_WORDS_SPARSE=3.
    Must remain SCANNED (or at least non-born-digital) even though the two
    words are clean, because the content is too thin to be useful native text.
    """
    import io

    from PIL import Image

    doc = fitz.open()
    page = doc.new_page()

    # Full-page background image
    img = Image.new("RGB", (600, 800), color=(240, 240, 240))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    page.insert_image(page.rect, stream=img_bytes.read())

    # Watermark: exactly 2 clean words (below MIN_WORDS_SPARSE=3)
    page.insert_text((200, 400), "Confidential Draft", fontsize=36, fontname="helv")

    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# GH-35: Scanned over-count on sparse and full-page-figure pages
# ---------------------------------------------------------------------------


class TestSparseAndFigurePageClassification:
    """Regression tests for GH-35: sparse/figure born-digital pages must not
    be classified as scanned solely because their text layer is short.

    The dangerous direction — a real scan misclassified as born-digital and
    thus skipping OCR — is tested explicitly; it must never happen.
    """

    def test_full_page_figure_with_short_caption_is_born_digital(self, tmp_path: Path) -> None:
        """A born-digital figure page with a short-but-clean caption is NOT scanned.

        Characterisation (pre-GH-35): word-count gate fired → SCANNED.
        Expected post-GH-35: clean short text layer → born-digital with sparse note.
        """
        pdf_path = tmp_path / "figure_caption.pdf"
        _create_figure_page_with_short_caption(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.is_born_digital, (
            "Full-page-figure born-digital page with short caption must NOT be "
            "classified as scanned (GH-35 regression)"
        )
        assert page.has_figures, "Page with embedded image must have has_figures=True"
        assert not page.needs_ocr_enhancement, "Clean native text does not require OCR enhancement"
        # Audit log must surface the sparse detection
        notes_text = " ".join(page.notes)
        assert "sparse" in notes_text.lower(), (
            "Notes must record that this is a sparse native layer for audit visibility"
        )

    def test_sparse_section_heading_is_born_digital(self, tmp_path: Path) -> None:
        """A born-digital sparse section heading with 7 clean words is NOT scanned.

        Characterisation (pre-GH-35): word-count gate fired → SCANNED.
        Expected post-GH-35: clean text layer → born-digital with sparse note.
        """
        pdf_path = tmp_path / "section_heading.pdf"
        _create_sparse_section_heading_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.is_born_digital, (
            "Sparse born-digital section heading must NOT be classified as scanned (GH-35)"
        )
        assert page.word_count < 15, "Fixture must have fewer than MIN_WORDS_PER_PAGE words"
        notes_text = " ".join(page.notes)
        assert "sparse" in notes_text.lower()

    def test_true_image_only_scan_remains_scanned(self, tmp_path: Path) -> None:
        """A true image-only scan with no text layer must remain classified as SCANNED.

        This is the safety-critical case: a real scan must never be misrouted to
        native extraction (which would silently skip OCR and lose content).
        """
        pdf_path = tmp_path / "real_scan.pdf"
        _create_image_only_scan_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert not page.is_born_digital, (
            "True image-only scan must remain SCANNED — "
            "misclassifying it as born-digital would skip OCR and lose content (GH-35)"
        )
        assert page.native_text == ""

    def test_decorative_front_matter_below_sparse_floor_is_scanned(self, tmp_path: Path) -> None:
        """Front-matter with only 2 clean words is rejected by the MIN_WORDS_SPARSE floor.

        The fixture ("Confidential Draft", 19 chars) has clean text and a proper font,
        so it would otherwise pass `text_layer_is_clean`.  The MIN_CHARS_FOR_TEXT_LAYER
        gate (default 50) would catch it first, so we lower min_chars to 10 to isolate
        the MIN_WORDS_SPARSE=3 branch.  This confirms the sparse-word floor is what
        actually rejects the page, not an earlier unrelated gate.
        """
        pdf_path = tmp_path / "decorative_front.pdf"
        _create_decorative_front_matter_pdf(pdf_path)

        # Lower the char threshold so the fixture (19 chars) passes it and the
        # MIN_WORDS_SPARSE gate is what rejects it.
        detector = BornDigitalDetector(min_chars=10)
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert not page.is_born_digital, (
            "Two-word watermark page must remain non-born-digital even if text is clean"
        )
        # Confirm the rejection came from the word-count gate, not the char gate.
        notes_text = " ".join(page.notes)
        assert "too few words" in notes_text, (
            "Rejection must come from the sparse-word floor (MIN_WORDS_SPARSE), "
            f"not from an earlier gate; got notes: {page.notes}"
        )

    def test_garbage_ocr_with_few_words_is_still_scanned(self, tmp_path: Path) -> None:
        """Dirty short text (single-char 'words') must not be rescued as born-digital.

        The clean-text bypass only applies when ALL quality signals pass.
        A page where avg_word_len fails (single-char OCR tokens) must remain SCANNED.
        """
        doc = fitz.open()
        page = doc.new_page()

        # Insert single-char tokens — exactly 8 words, below MIN_WORDS_PER_PAGE but
        # avg_word_len=1.0, which fails MIN_AVG_WORD_LENGTH check.
        garbage = "a b c d e f g h"
        tw = fitz.TextWriter(page.rect)
        tw.append((72, 72), garbage, fontsize=11, font=fitz.Font("helv"))
        tw.write_text(page)
        pdf_path = tmp_path / "dirty_short.pdf"
        doc.save(str(pdf_path))
        doc.close()

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page_a = result.pages[0]

        assert not page_a.is_born_digital, (
            "Short dirty text (single-char tokens, avg_word_len=1) must remain SCANNED"
        )

    def test_sparse_page_native_text_is_populated(self, tmp_path: Path) -> None:
        """A rescued sparse born-digital page must have its native text populated.

        If we classify the page as born-digital, native_text must not be empty —
        otherwise the pipeline would have nothing to work with.
        """
        pdf_path = tmp_path / "section_heading.pdf"
        _create_sparse_section_heading_pdf(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        if page.is_born_digital:
            assert page.native_text.strip() != "", (
                "Rescued sparse born-digital page must have non-empty native_text"
            )

    def test_figure_page_native_text_contains_caption(self, tmp_path: Path) -> None:
        """A rescued figure page must include the caption in native_text."""
        pdf_path = tmp_path / "figure_caption.pdf"
        _create_figure_page_with_short_caption(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.is_born_digital
        assert "Figure" in page.native_text or "Impulse" in page.native_text, (
            "Caption text must appear in native_text of rescued figure page"
        )

    def test_scan_with_clean_short_ocr_caption_is_currently_born_digital(
        self, tmp_path: Path
    ) -> None:
        """PINS the accepted GH-35 tradeoff: scan + short clean baked-in OCR → born-digital.

        A scanned page with a full-page raster image AND a short, high-quality
        baked-in OCR caption (8 words, >= 50 chars, proper font, no CID artifacts)
        is currently classified as born-digital after the GH-35 fix.

        This is documented, accepted behaviour: the baked-in OCR text IS the best
        available text for such a sparse scan, and re-OCR-ing it would not produce
        better output.  The practical impact is therefore bounded — we use the
        baked-in OCR text as native text rather than re-running OCR.

        DO NOT change this assertion without updating docs/log/2026-06-15_GH-35.md
        and re-evaluating the tradeoff.  See the residual-risk section in that log.
        """
        import io

        from PIL import Image

        doc = fitz.open()
        page = doc.new_page()

        # Full-page scan image (simulates a rasterized document page)
        img = Image.new("L", (1800, 2400), color=245)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        page.insert_image(page.rect, stream=img_bytes.read())

        # Baked-in OCR text layer: 8 clean words, 52 chars, proper Helvetica font.
        # This passes the char gate (52 >= 50), all quality checks (no CID, low
        # garbage, normal word lengths, font_count=1), and word_count=8 >= MIN_WORDS_SPARSE=3,
        # so text_layer_is_clean=True and the sparse-rescue path fires.
        page.insert_text(
            (72, 100),
            "Figure one. Monetary policy shock on output growth.",
            fontsize=10,
            fontname="helv",
        )

        pdf_path = tmp_path / "scan_with_clean_ocr.pdf"
        doc.save(str(pdf_path))
        doc.close()

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        pg = result.pages[0]

        # PIN: this is currently True — documented GH-35 accepted tradeoff.
        assert pg.is_born_digital, (
            "GH-35 accepted tradeoff: scan + short clean baked-in OCR is classified "
            "born-digital (uses baked-in OCR text as native; see docs/log/2026-06-15_GH-35.md)"
        )
