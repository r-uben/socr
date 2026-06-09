"""Benchmark page-type taxonomy (issue #39, consilium design).

Five coarse types matching the pipeline's real routing splits — deliberately
no finer, because the benchmark sample is ~10 papers and finer cells overfit:

  - ``native_prose``               born-digital, plain text
  - ``native_table_or_equation``   born-digital with structured content
  - ``scanned_prose``              no usable text layer, plain text
  - ``scanned_table_or_equation``  no usable text layer, structured content
  - ``sparse_or_figure``           figure-dominated / legitimately sparse

Types are derived deterministically from the born-digital detector's page
assessment, so tagging is automatic at benchmark time and reproducible.
"""

from __future__ import annotations

from pathlib import Path

NATIVE_PROSE = "native_prose"
NATIVE_TABLE_OR_EQUATION = "native_table_or_equation"
SCANNED_PROSE = "scanned_prose"
SCANNED_TABLE_OR_EQUATION = "scanned_table_or_equation"
SPARSE_OR_FIGURE = "sparse_or_figure"

ALL_PAGE_TYPES = (
    NATIVE_PROSE,
    NATIVE_TABLE_OR_EQUATION,
    SCANNED_PROSE,
    SCANNED_TABLE_OR_EQUATION,
    SPARSE_OR_FIGURE,
)


def classify_page_type(assessment, min_word_count: int) -> str:
    """Map one ``PageAssessment`` to a benchmark page type.

    ``min_word_count`` is the audit gate's configured minimum (the same
    threshold the pipeline uses), not a benchmark-specific constant: a page
    whose own text layer carries fewer words than the gate minimum —
    including zero words — is legitimately sparse.

    Sparseness is WORD-COUNT-derived, not figure-derived (issue #39 review):
    a dense page that merely contains an embedded image is a prose page; a
    figure-DOMINATED page has few words and qualifies through them.
    Structured content (tables/equations) outranks sparseness.
    """
    structured = assessment.has_tables or assessment.has_equations
    if not assessment.is_born_digital:
        return SCANNED_TABLE_OR_EQUATION if structured else SCANNED_PROSE
    if structured:
        return NATIVE_TABLE_OR_EQUATION
    if (assessment.word_count or 0) < min_word_count:
        return SPARSE_OR_FIGURE
    return NATIVE_PROSE


def classify_document_pages(pdf_path: Path, min_word_count: int) -> dict[int, str]:
    """Page number (1-indexed) -> page type for a whole PDF.

    Runs the same born-digital detector the pipeline uses, so benchmark page
    types and production routing decisions share one source of truth.
    """
    from socr.core.born_digital import BornDigitalDetector

    assessment = BornDigitalDetector().detect(pdf_path)
    return {
        pa.page_num: classify_page_type(pa, min_word_count) for pa in assessment.pages
    }
