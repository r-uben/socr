"""P1 benchmark-scoring regressions (issue #39).

Pins the fixes for the 0.0-WER catastrophe: coverage is a hard gate,
whole-document outputs are split and scored, NES/page-type/coverage are
persisted, and table fidelity is measured against hand-verified grid GT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from socr.benchmark.page_types import (
    NATIVE_PROSE,
    NATIVE_TABLE_OR_EQUATION,
    SCANNED_PROSE,
    SCANNED_TABLE_OR_EQUATION,
    SPARSE_OR_FIGURE,
    classify_page_type,
)
from socr.benchmark.runner import _dict_to_score, _score_to_dict
from socr.benchmark.scorer import BenchmarkScorer, DocumentScore, PageScore
from socr.core.result import DocumentStatus, EngineResult, PageOutput, PageStatus


def _gt(tmp_path: Path, pages: dict[int, str]) -> Path:
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    for n, text in pages.items():
        (gt_dir / f"page_{n}.txt").write_text(text)
    return gt_dir


def _result(pages: list[PageOutput]) -> EngineResult:
    return EngineResult(
        document_path=Path("/tmp/paper.pdf"),
        engine="testengine",
        status=DocumentStatus.SUCCESS,
        pages=pages,
    )


def _page(num: int, text: str) -> PageOutput:
    return PageOutput(page_num=num, text=text, status=PageStatus.SUCCESS, engine="testengine")


class TestZeroWerBugDead:
    """The historical bug: page_num=0 output matched no 1-indexed GT file,
    the loop skipped everything, and every engine scored a perfect 0.0."""

    def test_markerless_whole_doc_no_longer_scores_perfect(self, tmp_path: Path) -> None:
        gt_dir = _gt(tmp_path, {1: "alpha beta gamma", 2: "delta epsilon zeta"})
        result = _result([_page(0, "completely wrong text with no page markers")])

        score = BenchmarkScorer().score_document(result, gt_dir, expected_pages=2)
        assert score.overall_wer > 0.0
        assert score.pages_missing == [2]  # markerless blob covers page 1 only
        assert score.coverage == 0.5

    def test_marked_whole_doc_splits_and_scores_per_page(self, tmp_path: Path) -> None:
        gt_dir = _gt(tmp_path, {1: "alpha beta gamma", 2: "delta epsilon zeta"})
        body = "## Page 1\n\nalpha beta gamma\n\n## Page 2\n\ndelta epsilon zeta"
        result = _result([_page(0, body)])

        score = BenchmarkScorer().score_document(result, gt_dir, expected_pages=2)
        assert score.pages_missing == []
        assert score.coverage == 1.0
        assert score.overall_wer == 0.0  # genuinely perfect, and now PROVEN so
        assert all(p.covered for p in score.pages)

    def test_per_page_outputs_score_directly(self, tmp_path: Path) -> None:
        gt_dir = _gt(tmp_path, {1: "alpha beta gamma"})
        result = _result([_page(1, "alpha beta WRONG")])
        score = BenchmarkScorer().score_document(result, gt_dir, expected_pages=1)
        assert 0.0 < score.overall_wer < 1.0


class TestCoverageHardGate:
    def test_uncovered_page_is_total_failure(self, tmp_path: Path) -> None:
        gt_dir = _gt(tmp_path, {1: "alpha beta gamma", 2: "delta epsilon zeta"})
        result = _result([_page(1, "alpha beta gamma")])  # page 2 never produced

        score = BenchmarkScorer().score_document(result, gt_dir, expected_pages=2)
        missing = next(p for p in score.pages if p.page_num == 2)
        assert missing.covered is False
        assert missing.word_error_rate >= 1.0
        assert missing.normalized_edit_similarity == 0.0
        assert score.pages_missing == [2]

    def test_engine_with_no_output_scores_zero_coverage(self, tmp_path: Path) -> None:
        gt_dir = _gt(tmp_path, {1: "alpha", 2: "beta"})
        result = _result([])
        score = BenchmarkScorer().score_document(result, gt_dir, expected_pages=2)
        assert score.coverage == 0.0
        assert score.pages_missing == [1, 2]


class TestTableCellFidelity:
    GT = "| Year | Coef |\n|---|---|\n| 1994 | 0.040 |\n| 1995 | -0.213 |\n"

    def test_exact_grid_scores_full(self) -> None:
        acc, ok = BenchmarkScorer.score_table_cells(self.GT, self.GT)
        assert acc == 1.0 and ok is True

    def test_wrong_digit_detected(self) -> None:
        pred = self.GT.replace("0.040", "0.047")
        acc, ok = BenchmarkScorer.score_table_cells(pred, self.GT)
        assert ok is True
        assert acc < 1.0  # the flipped digit is NOT forgiven

    def test_shape_mismatch_fails_structure_keeps_content_credit(self) -> None:
        pred = (
            "| Year | Coef | Extra |\n|---|---|---|\n| 1994 | 0.040 | x |\n| 1995 | -0.213 | y |\n"
        )
        acc, ok = BenchmarkScorer.score_table_cells(pred, self.GT)
        assert ok is False
        assert acc == 1.0  # values all present, structure wrong

    def test_grid_gt_file_wires_into_document_score(self, tmp_path: Path) -> None:
        gt_dir = _gt(tmp_path, {1: "Year Coef 1994 0.040 1995 -0.213"})
        (gt_dir / "page_1.table.md").write_text(self.GT)
        result = _result([_page(1, self.GT)])
        score = BenchmarkScorer().score_document(result, gt_dir, expected_pages=1)
        assert score.pages[0].table_cell_accuracy == 1.0
        assert score.pages[0].table_structure_ok is True


@dataclass
class _PA:
    is_born_digital: bool = True
    has_tables: bool = False
    has_equations: bool = False
    has_figures: bool = False
    word_count: int = 400


class TestPageTypeTaxonomy:
    MIN = 50

    def test_native_prose(self) -> None:
        assert classify_page_type(_PA(), self.MIN) == NATIVE_PROSE

    def test_native_structured(self) -> None:
        assert classify_page_type(_PA(has_tables=True), self.MIN) == NATIVE_TABLE_OR_EQUATION
        assert classify_page_type(_PA(has_equations=True), self.MIN) == NATIVE_TABLE_OR_EQUATION

    def test_scanned(self) -> None:
        assert classify_page_type(_PA(is_born_digital=False), self.MIN) == SCANNED_PROSE
        assert (
            classify_page_type(_PA(is_born_digital=False, has_tables=True), self.MIN)
            == SCANNED_TABLE_OR_EQUATION
        )

    def test_sparse_or_figure(self) -> None:
        assert classify_page_type(_PA(word_count=24), self.MIN) == SPARSE_OR_FIGURE
        # Zero words = the sparsest possible page (review: word_count=0
        # truthiness bug used to classify blank pages as native_prose).
        assert classify_page_type(_PA(word_count=0), self.MIN) == SPARSE_OR_FIGURE

    def test_dense_page_with_figure_is_prose(self) -> None:
        """Sparseness is word-count-derived, not figure-derived: a dense
        page that merely contains an image is a prose page (review)."""
        assert classify_page_type(_PA(has_figures=True, word_count=400), self.MIN) == NATIVE_PROSE

    def test_structured_outranks_sparse(self) -> None:
        assert (
            classify_page_type(_PA(has_tables=True, word_count=10), self.MIN)
            == NATIVE_TABLE_OR_EQUATION
        )


class TestSerializationCarriesEverything:
    def test_nes_pagetype_coverage_roundtrip(self) -> None:
        score = DocumentScore(
            paper_name="p",
            engine="e",
            pages=[
                PageScore(
                    page_num=1,
                    word_error_rate=0.1,
                    character_error_rate=0.05,
                    normalized_edit_similarity=0.93,
                    word_count_ratio=1.0,
                    page_type=NATIVE_TABLE_OR_EQUATION,
                    covered=True,
                    table_cell_accuracy=0.98,
                    table_structure_ok=True,
                )
            ],
            overall_nes=0.93,
            expected_pages=2,
            pages_missing=[2],
        )
        loaded = _dict_to_score(_score_to_dict(score))
        assert loaded.pages[0].normalized_edit_similarity == 0.93
        assert loaded.pages[0].page_type == NATIVE_TABLE_OR_EQUATION
        assert loaded.pages[0].table_cell_accuracy == 0.98
        assert loaded.overall_nes == 0.93
        assert loaded.pages_missing == [2]
        # Coverage is denominated in SCORED pages; in real score_document
        # output every missing page also appears in pages with covered=False.
        assert loaded.coverage == 1.0

    def test_macro_aggregation_by_type(self) -> None:
        def _p(num, ptype, nes):
            return PageScore(
                page_num=num,
                word_error_rate=1 - nes,
                character_error_rate=1 - nes,
                normalized_edit_similarity=nes,
                word_count_ratio=1.0,
                page_type=ptype,
            )

        score = DocumentScore(
            paper_name="p",
            engine="e",
            pages=[
                _p(1, NATIVE_PROSE, 1.0),
                _p(2, NATIVE_PROSE, 1.0),
                _p(3, NATIVE_TABLE_OR_EQUATION, 0.5),
            ],
        )
        by_type = score.by_page_type()
        assert by_type[NATIVE_PROSE]["nes"] == 1.0
        assert by_type[NATIVE_TABLE_OR_EQUATION]["nes"] == 0.5
        assert by_type[NATIVE_PROSE]["pages"] == 2.0


class TestMarkerAwareSplit:
    """Review critical: positional renumbering misattributed every page
    after a skipped marker — page 3's correct text scored against page 2's
    ground truth."""

    def test_skipped_marker_attributes_by_number(self, tmp_path: Path) -> None:
        gt_dir = _gt(
            tmp_path,
            {1: "alpha beta gamma", 2: "delta epsilon zeta", 3: "eta theta iota"},
        )
        body = "## Page 1\n\nalpha beta gamma\n\n## Page 3\n\neta theta iota"
        score = BenchmarkScorer().score_document(
            _result([_page(0, body)]), gt_dir, expected_pages=3
        )
        by_num = {p.page_num: p for p in score.pages}
        assert by_num[1].word_error_rate == 0.0
        assert by_num[3].word_error_rate == 0.0  # NOT scored against page 2's GT
        assert score.pages_missing == [2]

    def test_preamble_before_first_marker_belongs_to_it(self, tmp_path: Path) -> None:
        gt_dir = _gt(tmp_path, {1: "title page text alpha beta gamma"})
        body = "title page text\n\n## Page 1\n\nalpha beta gamma"
        score = BenchmarkScorer().score_document(
            _result([_page(0, body)]), gt_dir, expected_pages=1
        )
        assert score.pages[0].word_error_rate == 0.0


class TestEmptyGroundTruthDir:
    def test_no_gt_files_scores_zero_not_perfect(self, tmp_path: Path) -> None:
        """The Stage-2 corner of the 0.0-WER catastrophe: a GT dir that
        exists but has no page_N.txt yet must never grade an engine 0.0."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        score = BenchmarkScorer().score_document(
            _result([_page(1, "anything")]), gt_dir, expected_pages=2
        )
        assert score.overall_wer == 1.0
        assert score.overall_nes == 0.0
        assert score.coverage == 0.0
        assert score.pages_missing == [1, 2]


class TestBlankGtPage:
    def test_correctly_empty_prediction_counts_covered(self, tmp_path: Path) -> None:
        gt_dir = _gt(tmp_path, {1: "real text here", 2: ""})
        result = _result([_page(1, "real text here")])  # nothing for blank page 2
        score = BenchmarkScorer().score_document(result, gt_dir, expected_pages=2)
        blank = next(p for p in score.pages if p.page_num == 2)
        assert blank.covered is True
        assert score.pages_missing == []
        assert score.coverage == 1.0


class TestNumericCellNormalization:
    def test_unicode_minus_matches_ascii(self) -> None:
        gt = "| Coef |\n|---|\n| −0.213 |\n"  # U+2212 minus in GT
        pred = "| Coef |\n|---|\n| -0.213 |\n"
        acc, ok = BenchmarkScorer.score_table_cells(pred, gt)
        assert acc == 1.0 and ok is True

    def test_unicode_minus_wrong_digit_still_detected(self) -> None:
        gt = "| Coef |\n|---|\n| −0.213 |\n"
        pred = "| Coef |\n|---|\n| -0.218 |\n"
        acc, ok = BenchmarkScorer.score_table_cells(pred, gt)
        assert acc < 1.0

    def test_escaped_pipe_is_content_not_separator(self) -> None:
        gt = "| Label | Value |\n|---|---|\n| a\\|b | 4.4 |\n"
        acc, ok = BenchmarkScorer.score_table_cells(gt, gt)
        assert ok is True
        assert acc == 1.0


class TestRunnerScoresEverything:
    def _paper(self, tmp_path: Path):
        from socr.benchmark.dataset import BenchmarkPaper

        gt_dir = _gt(tmp_path, {1: "alpha beta gamma", 2: "delta epsilon"})
        return BenchmarkPaper(
            name="stub_paper",
            pdf_path=tmp_path / "stub.pdf",
            category="mixed",
            page_count=2,
            ground_truth_path=gt_dir,
        )

    def test_unavailable_engine_scores_zero_coverage(self, tmp_path: Path) -> None:
        """Review: MODEL_UNAVAILABLE runs must not silently drop out of the
        ranking — they score zero coverage like any other failure."""
        from unittest.mock import MagicMock, patch

        from socr.benchmark.runner import BenchmarkRunner
        from socr.core.config import EngineType, PipelineConfig

        paper = self._paper(tmp_path)
        runner = BenchmarkRunner(PipelineConfig(quiet=True))
        runner._page_types[paper.name] = {}  # skip fitz on the fake pdf

        engine = MagicMock()
        engine.is_available.return_value = False
        with patch("socr.benchmark.runner.get_engine", return_value=engine):
            run = runner.run_single(paper, EngineType.GEMINI, tmp_path / "out")

        assert run.score is not None
        assert run.score.coverage == 0.0
        assert run.score.pages_missing == [1, 2]
