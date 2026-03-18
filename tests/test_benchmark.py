"""Tests for the benchmark package: dataset, ground truth, scorer, rasterizer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import fitz
import pytest

from socr.benchmark.dataset import BenchmarkPaper, BenchmarkSet
from socr.benchmark.ground_truth import GroundTruthExtractor
from socr.benchmark.rasterize import PaperRasterizer
from socr.benchmark.scorer import BenchmarkScorer, _levenshtein


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_synthetic_pdf(path: Path, pages: list[str]) -> Path:
    """Create a simple PDF with text content on each page."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# BenchmarkPaper / BenchmarkSet
# ---------------------------------------------------------------------------


class TestBenchmarkPaper:
    def test_creation(self) -> None:
        paper = BenchmarkPaper(
            name="test_paper",
            pdf_path=Path("/tmp/test.pdf"),
            category="mixed",
            page_count=10,
            notes="a note",
        )
        assert paper.name == "test_paper"
        assert paper.category == "mixed"
        assert paper.page_count == 10
        assert paper.ground_truth_path is None

    def test_path_coercion(self) -> None:
        paper = BenchmarkPaper(
            name="test",
            pdf_path="/tmp/test.pdf",  # type: ignore[arg-type]
            category="text_only",
            page_count=5,
        )
        assert isinstance(paper.pdf_path, Path)


class TestBenchmarkSet:
    def test_by_category(self) -> None:
        papers = [
            BenchmarkPaper("a", Path("/a.pdf"), "mixed", 10),
            BenchmarkPaper("b", Path("/b.pdf"), "mixed", 20),
            BenchmarkPaper("c", Path("/c.pdf"), "math_heavy", 30),
        ]
        bench = BenchmarkSet(papers=papers)
        cats = bench.by_category()
        assert len(cats["mixed"]) == 2
        assert len(cats["math_heavy"]) == 1

    def test_save_and_load(self, tmp_path: Path) -> None:
        papers = [
            BenchmarkPaper("paper_a", Path("/papers/a.pdf"), "mixed", 10, notes="first"),
            BenchmarkPaper(
                "paper_b",
                Path("/papers/b.pdf"),
                "table_heavy",
                20,
                ground_truth_path=Path("/gt/b"),
            ),
        ]
        bench = BenchmarkSet(papers=papers, created="2026-03-18T00:00:00+00:00")

        out_file = tmp_path / "bench.json"
        bench.save(out_file)

        # Verify JSON structure
        data = json.loads(out_file.read_text())
        assert data["created"] == "2026-03-18T00:00:00+00:00"
        assert len(data["papers"]) == 2
        assert data["papers"][1]["ground_truth_path"] == "/gt/b"

        # Round-trip load
        loaded = BenchmarkSet.load(out_file)
        assert len(loaded.papers) == 2
        assert loaded.papers[0].name == "paper_a"
        assert loaded.papers[0].notes == "first"
        assert loaded.papers[1].ground_truth_path == Path("/gt/b")
        assert loaded.created == "2026-03-18T00:00:00+00:00"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        bench = BenchmarkSet(papers=[])
        out_file = tmp_path / "sub" / "dir" / "bench.json"
        bench.save(out_file)
        assert out_file.exists()

    def test_empty_set(self) -> None:
        bench = BenchmarkSet()
        assert len(bench.papers) == 0
        assert bench.by_category() == {}
        assert bench.created  # auto-populated


# ---------------------------------------------------------------------------
# GroundTruthExtractor
# ---------------------------------------------------------------------------


class TestGroundTruthExtractor:
    def test_extract_from_synthetic_pdf(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        _create_synthetic_pdf(pdf_path, ["Page one text.", "Page two text."])

        extractor = GroundTruthExtractor()
        truths = extractor.extract(pdf_path)

        assert len(truths) == 2
        assert truths[0].page_num == 1
        assert truths[1].page_num == 2
        assert "Page one" in truths[0].text
        assert "Page two" in truths[1].text
        assert truths[0].word_count >= 3
        assert truths[0].char_count > 0

    def test_save_creates_files(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        _create_synthetic_pdf(pdf_path, ["First page.", "Second page."])

        extractor = GroundTruthExtractor()
        truths = extractor.extract(pdf_path)

        gt_dir = tmp_path / "ground_truth"
        extractor.save(truths, gt_dir)

        assert (gt_dir / "page_1.txt").exists()
        assert (gt_dir / "page_2.txt").exists()
        assert (gt_dir / "full.txt").exists()

        page1_text = (gt_dir / "page_1.txt").read_text()
        assert "First page" in page1_text

        full_text = (gt_dir / "full.txt").read_text()
        assert "First page" in full_text
        assert "Second page" in full_text

    def test_extract_and_save(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        _create_synthetic_pdf(pdf_path, ["Hello world."])

        extractor = GroundTruthExtractor()
        gt_dir = tmp_path / "gt"
        truths = extractor.extract_and_save(pdf_path, gt_dir)

        assert len(truths) == 1
        assert (gt_dir / "page_1.txt").exists()
        assert (gt_dir / "full.txt").exists()

    def test_missing_pdf_raises(self) -> None:
        extractor = GroundTruthExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract(Path("/nonexistent/file.pdf"))


# ---------------------------------------------------------------------------
# BenchmarkScorer
# ---------------------------------------------------------------------------


class TestLevenshtein:
    def test_identical(self) -> None:
        assert _levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0

    def test_single_substitution(self) -> None:
        assert _levenshtein(["a", "b", "c"], ["a", "x", "c"]) == 1

    def test_insertion(self) -> None:
        assert _levenshtein(["a", "b"], ["a", "x", "b"]) == 1

    def test_deletion(self) -> None:
        assert _levenshtein(["a", "b", "c"], ["a", "c"]) == 1

    def test_empty_sequences(self) -> None:
        assert _levenshtein([], []) == 0
        assert _levenshtein(["a", "b"], []) == 2
        assert _levenshtein([], ["a", "b"]) == 2

    def test_completely_different(self) -> None:
        assert _levenshtein(["a", "b", "c"], ["x", "y", "z"]) == 3


class TestBenchmarkScorer:
    def test_perfect_match(self) -> None:
        scorer = BenchmarkScorer()
        wer = scorer.score("hello world", "hello world")
        assert wer == 0.0

    def test_completely_wrong(self) -> None:
        scorer = BenchmarkScorer()
        wer = scorer.score("foo bar baz", "xxx yyy zzz")
        assert wer == 1.0  # 3 substitutions / 3 words

    def test_partial_match(self) -> None:
        scorer = BenchmarkScorer()
        # "the cat sat" vs "the dog sat" -> 1 sub / 3 words = 0.333
        wer = scorer.score("the dog sat", "the cat sat")
        assert abs(wer - 1 / 3) < 1e-6

    def test_empty_ground_truth(self) -> None:
        scorer = BenchmarkScorer()
        assert scorer.score("", "") == 0.0
        assert scorer.score("some text", "") == 1.0

    def test_empty_prediction(self) -> None:
        scorer = BenchmarkScorer()
        wer = scorer.score("", "the reference text")
        # 3 deletions / 3 words = 1.0
        assert wer == 1.0

    def test_cer(self) -> None:
        scorer = BenchmarkScorer()
        cer = scorer.score_cer("abc", "abc")
        assert cer == 0.0

        cer = scorer.score_cer("abd", "abc")
        assert abs(cer - 1 / 3) < 1e-6

    def test_score_page(self) -> None:
        scorer = BenchmarkScorer()
        page = scorer.score_page("hello world", "hello world", page_num=1)
        assert page.page_num == 1
        assert page.word_error_rate == 0.0
        assert page.character_error_rate == 0.0
        assert page.word_count_ratio == 1.0

    def test_wer_can_exceed_one(self) -> None:
        """WER > 1.0 when insertions exceed reference length."""
        scorer = BenchmarkScorer()
        wer = scorer.score("a b c d e f", "a")
        # ref=1 word, hyp=6 words -> 5 insertions / 1 word = 5.0
        assert wer == 5.0


# ---------------------------------------------------------------------------
# PaperRasterizer
# ---------------------------------------------------------------------------


class TestPaperRasterizer:
    def test_rasterize_produces_image_only_pdf(self, tmp_path: Path) -> None:
        # Create a born-digital PDF
        src_path = tmp_path / "source.pdf"
        _create_synthetic_pdf(src_path, ["This is text content.", "Page two content."])

        # Verify source has text
        with fitz.open(str(src_path)) as doc:
            src_text = doc[0].get_text("text").strip()
        assert "text content" in src_text

        # Rasterize
        rasterizer = PaperRasterizer()
        out_path = tmp_path / "output" / "rasterized.pdf"
        result = rasterizer.rasterize(src_path, out_path, dpi=72)

        assert result == out_path
        assert out_path.exists()

        # The rasterized PDF should have pages but minimal/no extractable text
        with fitz.open(str(out_path)) as doc:
            assert len(doc) == 2
            raster_text = doc[0].get_text("text").strip()
            # Image-only PDF should have essentially no text layer
            assert len(raster_text) < 5  # Might have empty string or whitespace

        # The rasterized PDF should contain images
        with fitz.open(str(out_path)) as doc:
            images = doc[0].get_images()
            assert len(images) >= 1

    def test_rasterize_missing_pdf_raises(self, tmp_path: Path) -> None:
        rasterizer = PaperRasterizer()
        with pytest.raises(FileNotFoundError):
            rasterizer.rasterize(Path("/nonexistent.pdf"), tmp_path / "out.pdf")

    def test_rasterize_creates_parent_dirs(self, tmp_path: Path) -> None:
        src_path = tmp_path / "source.pdf"
        _create_synthetic_pdf(src_path, ["Text."])

        rasterizer = PaperRasterizer()
        out_path = tmp_path / "deep" / "nested" / "dir" / "out.pdf"
        rasterizer.rasterize(src_path, out_path, dpi=72)
        assert out_path.exists()
