"""Page difficulty classification for tiered engine routing.

Classifies scanned/complex pages as EASY or HARD so the orchestrator can
route easy pages to cheap local engines and reserve cloud engines for hard ones.

Signals used (all available without rendering the page):
  - Layout complexity: number of columns, table presence
  - Image characteristics: page size, drawing density
  - Content density: words on the page (from any text layer)
  - Document metadata: font count, image count

This classifier runs AFTER born-digital detection and only on pages that
need OCR (scanned pages + complex born-digital pages).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import fitz

from socr.core.pdf import open_pdf

logger = logging.getLogger(__name__)


class PageDifficulty(str, Enum):
    """Difficulty level for a page that needs OCR."""

    EASY = "easy"  # Clean printed text, simple layout → local model
    HARD = "hard"  # Complex layout, tables, degraded, multi-column → cloud model


@dataclass
class DifficultyAssessment:
    """Result of difficulty classification for one page."""

    page_num: int
    difficulty: PageDifficulty
    reasons: list[str]
    # Raw signals (for debugging / calibration)
    drawing_count: int = 0
    image_count: int = 0
    table_count: int = 0
    column_count: int = 1
    text_block_count: int = 0


# Thresholds — kept simple, can be calibrated from benchmark data later
_MAX_DRAWINGS_EASY = 20  # Pages with many drawings are likely figures/charts
_MAX_IMAGES_EASY = 2  # Pages with many images are complex
_MAX_TABLES_EASY = 0  # Any table → HARD (tables need structured extraction)
_MAX_COLUMNS_EASY = 1  # Multi-column layouts are harder
_MIN_TEXT_BLOCKS_EASY = 1  # Must have at least some text
_MAX_TEXT_BLOCKS_EASY = 30  # Dense multi-block layouts are complex


def classify_page(
    page: fitz.Page,
    page_num: int,
    has_tables_hint: bool = False,
    has_equations_hint: bool = False,
) -> DifficultyAssessment:
    """Classify a single page's difficulty for OCR.

    This is fast — no rendering, no API calls. Uses PyMuPDF's structural
    analysis only.

    Args:
        page: A PyMuPDF page object.
        page_num: 1-indexed page number.
        has_tables_hint: If True (from born-digital detector), force HARD.
        has_equations_hint: If True (from born-digital detector), force HARD.

    Returns:
        DifficultyAssessment with difficulty level and reasons.
    """
    reasons: list[str] = []

    # Hints from born-digital detector override PyMuPDF analysis
    if has_tables_hint:
        return DifficultyAssessment(
            page_num=page_num,
            difficulty=PageDifficulty.HARD,
            reasons=["tables detected (born-digital hint)"],
        )
    if has_equations_hint:
        return DifficultyAssessment(
            page_num=page_num,
            difficulty=PageDifficulty.HARD,
            reasons=["equations detected (born-digital hint)"],
        )

    # Count structural elements
    try:
        drawings = page.get_drawings()
        drawing_count = len(drawings)
    except Exception:
        drawings = []
        drawing_count = 0

    try:
        images = page.get_images()
        image_count = len(images)
    except Exception:
        image_count = 0

    try:
        tables_result = page.find_tables()
        table_count = len(tables_result.tables)
    except Exception:
        table_count = 0

    # Estimate column count from text block positions
    try:
        blocks = page.get_text("dict").get("blocks", [])
        text_blocks = [b for b in blocks if b.get("type", 0) == 0]
        text_block_count = len(text_blocks)
        column_count = _estimate_columns(text_blocks, page.rect.width)
    except Exception:
        text_block_count = 0
        column_count = 1

    # --- Classification logic ---

    difficulty = PageDifficulty.EASY

    if table_count > _MAX_TABLES_EASY:
        difficulty = PageDifficulty.HARD
        reasons.append(f"{table_count} table(s) detected")

    if column_count > _MAX_COLUMNS_EASY:
        difficulty = PageDifficulty.HARD
        reasons.append(f"{column_count}-column layout")

    if drawing_count > _MAX_DRAWINGS_EASY:
        difficulty = PageDifficulty.HARD
        reasons.append(f"{drawing_count} drawings (likely charts/figures)")

    if image_count > _MAX_IMAGES_EASY:
        difficulty = PageDifficulty.HARD
        reasons.append(f"{image_count} embedded images")

    if text_block_count > _MAX_TEXT_BLOCKS_EASY:
        difficulty = PageDifficulty.HARD
        reasons.append(f"{text_block_count} text blocks (dense layout)")

    if text_block_count < _MIN_TEXT_BLOCKS_EASY and image_count > 0:
        # Page is mostly images with little/no text — likely a figure page
        difficulty = PageDifficulty.HARD
        reasons.append("image-dominated page (few text blocks)")

    if not reasons:
        reasons.append("simple layout, clean content")

    return DifficultyAssessment(
        page_num=page_num,
        difficulty=difficulty,
        reasons=reasons,
        drawing_count=drawing_count,
        image_count=image_count,
        table_count=table_count,
        column_count=column_count,
        text_block_count=text_block_count,
    )


def classify_pages(
    pdf_path: str,
    page_nums: list[int],
    page_hints: dict[int, dict] | None = None,
) -> dict[int, DifficultyAssessment]:
    """Classify multiple pages from a PDF.

    Args:
        pdf_path: Path to the PDF file.
        page_nums: 1-indexed page numbers to classify.
        page_hints: Optional dict of {page_num: {"has_tables": bool,
            "has_equations": bool}} from born-digital detection.

    Returns:
        Dict mapping page_num → DifficultyAssessment.
    """
    hints = page_hints or {}
    results: dict[int, DifficultyAssessment] = {}
    with open_pdf(pdf_path) as doc:
        for page_num in page_nums:
            if page_num < 1 or page_num > len(doc):
                continue
            page = doc[page_num - 1]
            h = hints.get(page_num, {})
            results[page_num] = classify_page(
                page,
                page_num,
                has_tables_hint=h.get("has_tables", False),
                has_equations_hint=h.get("has_equations", False),
            )
    return results


def _estimate_columns(
    text_blocks: list[dict],
    page_width: float,
) -> int:
    """Estimate number of text columns from block x-positions.

    Groups text blocks by their horizontal center position.
    Two distinct x-position clusters → 2 columns, etc.
    """
    if len(text_blocks) < 2:
        return 1

    # Get x-center of each text block
    centers = []
    for b in text_blocks:
        bbox = b.get("bbox", (0, 0, 0, 0))
        x_center = (bbox[0] + bbox[2]) / 2
        centers.append(x_center)

    if not centers:
        return 1

    # Simple clustering: sort centers, find gaps > 20% of page width
    centers.sort()
    gap_threshold = page_width * 0.15
    columns = 1
    for i in range(1, len(centers)):
        if centers[i] - centers[i - 1] > gap_threshold:
            columns += 1

    # Cap at 4 — more is likely misdetection
    return min(columns, 4)
