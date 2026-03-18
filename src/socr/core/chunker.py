"""PDF chunking for long documents.

Splits PDFs that exceed a page threshold into smaller chunks so that
CLI-based OCR engines can process them without hitting context-window
or timeout limits.  Each chunk is a standalone PDF written to a
temporary directory; the caller concatenates the per-chunk texts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PDFChunk:
    """Metadata for a single chunk of a split PDF."""

    chunk_num: int
    start_page: int  # 1-indexed
    end_page: int  # inclusive
    path: Path  # path to the chunk PDF file
    page_count: int


class PDFChunker:
    """Split long PDFs into fixed-size chunks using PyMuPDF."""

    def __init__(self, max_pages_per_chunk: int = 20) -> None:
        if max_pages_per_chunk < 1:
            raise ValueError("max_pages_per_chunk must be >= 1")
        self.max_pages_per_chunk = max_pages_per_chunk

    def needs_chunking(self, pdf_path: Path, threshold: int | None = None) -> bool:
        """Check if *pdf_path* exceeds *threshold* pages.

        Parameters
        ----------
        pdf_path:
            Path to the PDF file.
        threshold:
            Page-count threshold.  If *None*, defaults to
            ``max_pages_per_chunk`` (i.e. any document that would produce
            more than one chunk).
        """
        import fitz

        if threshold is None:
            threshold = self.max_pages_per_chunk

        with fitz.open(pdf_path) as doc:
            return len(doc) > threshold

    def chunk(self, pdf_path: Path, output_dir: Path) -> list[PDFChunk]:
        """Split *pdf_path* into chunk PDFs written to *output_dir*.

        Returns a list of :class:`PDFChunk` descriptors sorted by
        ``chunk_num``.  If the document has fewer pages than
        ``max_pages_per_chunk``, a single chunk covering the whole
        document is returned.
        """
        import fitz

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = pdf_path.stem

        with fitz.open(pdf_path) as src:
            total_pages = len(src)

            if total_pages == 0:
                return []

            chunks: list[PDFChunk] = []
            chunk_num = 0

            for start_0 in range(0, total_pages, self.max_pages_per_chunk):
                chunk_num += 1
                end_0 = min(start_0 + self.max_pages_per_chunk - 1, total_pages - 1)

                start_1 = start_0 + 1  # 1-indexed
                end_1 = end_0 + 1  # 1-indexed, inclusive
                page_count = end_0 - start_0 + 1

                chunk_path = output_dir / f"{stem}_chunk{chunk_num:03d}.pdf"

                with fitz.open() as dst:
                    dst.insert_pdf(src, from_page=start_0, to_page=end_0)
                    dst.save(str(chunk_path))

                chunks.append(
                    PDFChunk(
                        chunk_num=chunk_num,
                        start_page=start_1,
                        end_page=end_1,
                        path=chunk_path,
                        page_count=page_count,
                    )
                )

            logger.info(
                f"Split {pdf_path.name} ({total_pages} pages) into "
                f"{len(chunks)} chunks of up to {self.max_pages_per_chunk} pages"
            )

            return chunks
