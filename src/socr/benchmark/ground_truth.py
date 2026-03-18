"""Ground truth extraction from born-digital PDFs using PyMuPDF.

Extracts native text per page and saves as individual text files for
benchmark comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class PageGroundTruth:
    """Ground truth text for a single page."""

    page_num: int  # 1-indexed
    text: str
    word_count: int
    char_count: int


class GroundTruthExtractor:
    """Extract native text from born-digital PDFs as ground truth."""

    def extract(self, pdf_path: Path) -> list[PageGroundTruth]:
        """Extract native text per page using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of PageGroundTruth, one per page.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages: list[PageGroundTruth] = []
        with fitz.open(pdf_path) as doc:
            for page_idx in range(len(doc)):
                text = doc[page_idx].get_text("text").strip()
                pages.append(
                    PageGroundTruth(
                        page_num=page_idx + 1,
                        text=text,
                        word_count=len(text.split()) if text else 0,
                        char_count=len(text),
                    )
                )

        return pages

    def save(self, truths: list[PageGroundTruth], output_dir: Path) -> None:
        """Save ground truth as per-page text files and a combined full text.

        Creates:
            output_dir/page_1.txt
            output_dir/page_2.txt
            ...
            output_dir/full.txt

        Args:
            truths: List of PageGroundTruth to save.
            output_dir: Directory to write files into.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        full_parts: list[str] = []
        for page_gt in truths:
            page_file = output_dir / f"page_{page_gt.page_num}.txt"
            page_file.write_text(page_gt.text, encoding="utf-8")
            if page_gt.text:
                full_parts.append(page_gt.text)

        full_file = output_dir / "full.txt"
        full_file.write_text("\n\n".join(full_parts), encoding="utf-8")

    def extract_and_save(self, pdf_path: Path, output_dir: Path) -> list[PageGroundTruth]:
        """Extract ground truth and save to disk in one step.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory to write ground truth files.

        Returns:
            List of PageGroundTruth.
        """
        truths = self.extract(pdf_path)
        self.save(truths, output_dir)
        return truths
