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
            fitz.Point(x_pos - col_widths[0] if col_w == 0 else x_pos, table_top + num_rows * row_height),
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
        shape.draw_line(fitz.Point(x_pos, table_top), fitz.Point(x_pos, table_top + num_rows * row_height))
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
        shape.draw_line(fitz.Point(x_pos, table_top), fitz.Point(x_pos, table_top + num_rows * row_height))
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

    def test_table_page_needs_ocr_enhancement(self, tmp_path: Path) -> None:
        """Pages with tables should be flagged for OCR enhancement."""
        pdf_path = tmp_path / "table.pdf"
        _create_pdf_with_table(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.needs_ocr_enhancement

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

    def test_figure_page_needs_ocr_enhancement(self, tmp_path: Path) -> None:
        """Pages with figures should be flagged for OCR enhancement."""
        pdf_path = tmp_path / "figure.pdf"
        _create_pdf_with_image(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.needs_ocr_enhancement


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

    def test_equation_page_needs_ocr_enhancement(self, tmp_path: Path) -> None:
        """Pages with equations should be flagged for OCR enhancement."""
        pdf_path = tmp_path / "equations.pdf"
        _create_pdf_with_equations(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page = result.pages[0]

        assert page.needs_ocr_enhancement


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
        assert page2.has_figures  # has embedded image
        assert page2.needs_ocr_enhancement

    def test_complex_content_notes_list_types(self, tmp_path: Path) -> None:
        """Notes field lists the detected content types."""
        pdf_path = tmp_path / "mixed.pdf"
        _create_pdf_with_mixed_content(pdf_path)

        detector = BornDigitalDetector()
        result = detector.detect(pdf_path)
        page2 = result.pages[1]

        notes_text = " ".join(page2.notes)
        assert "complex content" in notes_text.lower() or "ocr enhancement" in notes_text.lower()

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
