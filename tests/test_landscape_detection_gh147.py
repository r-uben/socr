"""Tests for GH-147 A1: dominant text direction on PageAssessment.

Pure-fitz and hermetic — synthetic PDFs built in-test, no external corpus
files, no agentic pipeline. Confirms:
  - the free predicate `text_direction_is_rotated` is a pure axis comparison;
  - `_assess_page` stamps `dominant_text_direction` / `text_is_rotated` on
    every code path, including the early return below MIN_CHARS_FOR_TEXT_LAYER;
  - the PDF /Rotate flag (`set_rotation`) is not the signal — only line dirs.
"""

from pathlib import Path

import fitz

from socr.core.born_digital import (
    BornDigitalDetector,
    text_direction_is_rotated,
)

_PROSE_LINES = [
    "This is a born-digital academic paper about economic growth and monetary",
    "policy in developing countries. The author presents a comprehensive analysis",
    "of fiscal multipliers across different exchange rate regimes. The empirical",
    "evidence suggests that government spending has larger effects during recessions",
]


def _horizontal_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in _PROSE_LINES:
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
        y += 16
    doc.save(str(path))
    doc.close()


def _rotated_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 400
    for line in _PROSE_LINES:
        page.insert_text((72, y), line, fontsize=11, fontname="helv", rotate=90)
        y += 16
    doc.save(str(path))
    doc.close()


def _short_rotated_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 400), "abcd", fontsize=11, fontname="helv", rotate=90)
    doc.save(str(path))
    doc.close()


def _rotate_flag_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in _PROSE_LINES:
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
        y += 16
    page.set_rotation(90)
    doc.save(str(path))
    doc.close()


class TestTextDirectionIsRotatedPredicate:
    def test_horizontal_is_not_rotated(self) -> None:
        assert text_direction_is_rotated((1.0, 0.0)) is False

    def test_vertical_positive_is_rotated(self) -> None:
        assert text_direction_is_rotated((0.0, 1.0)) is True

    def test_vertical_negative_is_rotated(self) -> None:
        assert text_direction_is_rotated((0.0, -1.0)) is True

    def test_reversed_horizontal_is_not_rotated(self) -> None:
        assert text_direction_is_rotated((-1.0, 0.0)) is False

    def test_45_degree_tie_fails_closed_to_rotated(self) -> None:
        assert text_direction_is_rotated((0.707, 0.707)) is True

    def test_zero_vector_stays_horizontal(self) -> None:
        assert text_direction_is_rotated((0.0, 0.0)) is False


class TestAssessmentStampsDirection:
    def test_horizontal_page_reports_horizontal(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "horizontal.pdf"
        _horizontal_pdf(pdf_path)
        with fitz.open(str(pdf_path)) as doc:
            assessment = BornDigitalDetector()._assess_page(doc[0], 1)
        assert assessment.dominant_text_direction == (1.0, 0.0)
        assert assessment.text_is_rotated is False

    def test_rotated_page_reports_rotated(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "rotated.pdf"
        _rotated_pdf(pdf_path)
        with fitz.open(str(pdf_path)) as doc:
            assessment = BornDigitalDetector()._assess_page(doc[0], 1)
        # Empirically measured sign for rotate=90 under this venv's PyMuPDF.
        assert assessment.dominant_text_direction == (0.0, -1.0)
        assert assessment.text_is_rotated is True
        assert any("rotated text direction" in note for note in assessment.notes)

    def test_short_rotated_page_still_stamped_on_early_return(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "short_rotated.pdf"
        _short_rotated_pdf(pdf_path)
        with fitz.open(str(pdf_path)) as doc:
            assessment = BornDigitalDetector()._assess_page(doc[0], 1)
        # Below MIN_CHARS_FOR_TEXT_LAYER (50): the signals body takes its
        # earliest return (insufficient text layer). The direction stamp must
        # still land — that is the whole point of stamping in the wrapper.
        assert assessment.char_count < BornDigitalDetector.MIN_CHARS_FOR_TEXT_LAYER
        assert assessment.dominant_text_direction == (0.0, -1.0)
        assert assessment.text_is_rotated is True

    def test_pdf_rotate_flag_is_not_the_signal(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "rotate_flag.pdf"
        _rotate_flag_pdf(pdf_path)
        with fitz.open(str(pdf_path)) as doc:
            assessment = BornDigitalDetector()._assess_page(doc[0], 1)
        assert assessment.dominant_text_direction == (1.0, 0.0)
        assert assessment.text_is_rotated is False


class TestDetectPageStampsDirection:
    def test_detect_page_stamps_rotated_direction(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "rotated.pdf"
        _rotated_pdf(pdf_path)
        assessment = BornDigitalDetector().detect_page(pdf_path, 1)
        assert assessment.dominant_text_direction == (0.0, -1.0)
        assert assessment.text_is_rotated is True
