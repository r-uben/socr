"""Tests for PDFChunker (L1B-07)."""

from pathlib import Path

import fitz
import pytest

from socr.core.chunker import PDFChunk, PDFChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_pdf(path: Path, num_pages: int) -> Path:
    """Create a minimal PDF with *num_pages* pages."""
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNeedsChunking:
    def test_small_pdf_does_not_need_chunking(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "small.pdf", 5)
        chunker = PDFChunker(max_pages_per_chunk=20)

        assert chunker.needs_chunking(pdf) is False

    def test_large_pdf_needs_chunking(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "large.pdf", 25)
        chunker = PDFChunker(max_pages_per_chunk=20)

        assert chunker.needs_chunking(pdf) is True

    def test_exact_threshold_does_not_need_chunking(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "exact.pdf", 20)
        chunker = PDFChunker(max_pages_per_chunk=20)

        assert chunker.needs_chunking(pdf) is False

    def test_custom_threshold(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "custom.pdf", 10)
        chunker = PDFChunker(max_pages_per_chunk=20)

        assert chunker.needs_chunking(pdf, threshold=5) is True
        assert chunker.needs_chunking(pdf, threshold=10) is False
        assert chunker.needs_chunking(pdf, threshold=15) is False


class TestChunk:
    def test_single_chunk_for_small_pdf(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "small.pdf", 5)
        chunker = PDFChunker(max_pages_per_chunk=20)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert len(chunks) == 1
        assert chunks[0].chunk_num == 1
        assert chunks[0].start_page == 1
        assert chunks[0].end_page == 5
        assert chunks[0].page_count == 5
        assert chunks[0].path.exists()

    def test_two_chunks_for_medium_pdf(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "medium.pdf", 35)
        chunker = PDFChunker(max_pages_per_chunk=20)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert len(chunks) == 2
        # Chunk 1: pages 1-20
        assert chunks[0].start_page == 1
        assert chunks[0].end_page == 20
        assert chunks[0].page_count == 20
        # Chunk 2: pages 21-35
        assert chunks[1].start_page == 21
        assert chunks[1].end_page == 35
        assert chunks[1].page_count == 15

    def test_exact_multiple(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "exact.pdf", 40)
        chunker = PDFChunker(max_pages_per_chunk=20)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert len(chunks) == 2
        assert chunks[0].page_count == 20
        assert chunks[1].page_count == 20

    def test_many_chunks(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "long.pdf", 55)
        chunker = PDFChunker(max_pages_per_chunk=10)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert len(chunks) == 6
        # Pages: 1-10, 11-20, 21-30, 31-40, 41-50, 51-55
        assert chunks[-1].start_page == 51
        assert chunks[-1].end_page == 55
        assert chunks[-1].page_count == 5

    def test_chunk_pdfs_are_valid(self, tmp_path: Path) -> None:
        """Each chunk PDF should be a valid PDF with the right page count."""
        pdf = _create_pdf(tmp_path / "test.pdf", 25)
        chunker = PDFChunker(max_pages_per_chunk=10)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        for chunk in chunks:
            with fitz.open(chunk.path) as doc:
                assert len(doc) == chunk.page_count

    def test_chunk_preserves_content(self, tmp_path: Path) -> None:
        """Text in each chunk should match the original pages."""
        pdf = _create_pdf(tmp_path / "content.pdf", 5)
        chunker = PDFChunker(max_pages_per_chunk=3)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert len(chunks) == 2
        # Chunk 1 should have pages 1-3
        with fitz.open(chunks[0].path) as doc:
            assert "Page 1" in doc[0].get_text()
            assert "Page 3" in doc[2].get_text()
        # Chunk 2 should have pages 4-5
        with fitz.open(chunks[1].path) as doc:
            assert "Page 4" in doc[0].get_text()
            assert "Page 5" in doc[1].get_text()

    def test_output_dir_created(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf", 5)
        chunker = PDFChunker(max_pages_per_chunk=3)
        out = tmp_path / "nested" / "output"

        chunks = chunker.chunk(pdf, out)

        assert out.exists()
        assert len(chunks) == 2

    def test_chunk_numbering(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf", 30)
        chunker = PDFChunker(max_pages_per_chunk=10)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        for i, chunk in enumerate(chunks, 1):
            assert chunk.chunk_num == i

    def test_single_page_chunk_size(self, tmp_path: Path) -> None:
        """Chunk size of 1 should produce one chunk per page."""
        pdf = _create_pdf(tmp_path / "test.pdf", 3)
        chunker = PDFChunker(max_pages_per_chunk=1)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert len(chunks) == 3
        for i, chunk in enumerate(chunks, 1):
            assert chunk.start_page == i
            assert chunk.end_page == i
            assert chunk.page_count == 1


class TestEdgeCases:
    def test_invalid_max_pages(self) -> None:
        with pytest.raises(ValueError):
            PDFChunker(max_pages_per_chunk=0)

        with pytest.raises(ValueError):
            PDFChunker(max_pages_per_chunk=-1)

    def test_single_page_pdf(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "single.pdf", 1)
        chunker = PDFChunker(max_pages_per_chunk=20)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert len(chunks) == 1
        assert chunks[0].page_count == 1
        assert chunks[0].start_page == 1
        assert chunks[0].end_page == 1

    def test_chunk_file_naming(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "my_doc.pdf", 25)
        chunker = PDFChunker(max_pages_per_chunk=10)

        chunks = chunker.chunk(pdf, tmp_path / "chunks")

        assert chunks[0].path.name == "my_doc_chunk001.pdf"
        assert chunks[1].path.name == "my_doc_chunk002.pdf"
        assert chunks[2].path.name == "my_doc_chunk003.pdf"
