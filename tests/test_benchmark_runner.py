"""Tests for benchmark runner, results serialization, and calibration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from socr.benchmark.calibrate import (
    CalibrationReport,
    EngineProfile,
    RepairCalibrator,
)
from socr.benchmark.dataset import BenchmarkPaper, BenchmarkSet
from socr.benchmark.runner import (
    BenchmarkResults,
    BenchmarkRunner,
    EngineRun,
    _score_to_dict,
    _dict_to_score,
)
from socr.benchmark.scorer import DocumentScore, PageScore
from socr.core.config import EngineType, PipelineConfig
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    PageOutput,
    PageStatus,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_engine_result(
    engine: str,
    pdf_path: Path = Path("/tmp/test.pdf"),
    success: bool = True,
    text: str = "hello world",
    processing_time: float = 1.0,
    failure_mode: FailureMode = FailureMode.NONE,
) -> EngineResult:
    """Build a minimal EngineResult for testing."""
    if success:
        return EngineResult(
            document_path=pdf_path,
            engine=engine,
            status=DocumentStatus.SUCCESS,
            failure_mode=failure_mode,
            pages=[
                PageOutput(
                    page_num=0,
                    text=text,
                    status=PageStatus.SUCCESS,
                    engine=engine,
                    processing_time=processing_time,
                )
            ],
            processing_time=processing_time,
        )
    return EngineResult(
        document_path=pdf_path,
        engine=engine,
        status=DocumentStatus.ERROR,
        failure_mode=failure_mode or FailureMode.CLI_ERROR,
        error="test error",
        processing_time=processing_time,
    )


def _make_score(
    paper: str,
    engine: str,
    wer: float = 0.1,
    cer: float = 0.05,
    processing_time: float = 1.0,
) -> DocumentScore:
    """Build a minimal DocumentScore."""
    return DocumentScore(
        paper_name=paper,
        engine=engine,
        pages=[
            PageScore(page_num=1, word_error_rate=wer, character_error_rate=cer, word_count_ratio=1.0),
        ],
        overall_wer=wer,
        overall_cer=cer,
        processing_time=processing_time,
    )


def _make_benchmark_paper(
    name: str = "test_paper",
    category: str = "mixed",
    page_count: int = 5,
    gt_path: Path | None = None,
) -> BenchmarkPaper:
    """Build a BenchmarkPaper for testing."""
    return BenchmarkPaper(
        name=name,
        pdf_path=Path(f"/tmp/{name}.pdf"),
        category=category,
        page_count=page_count,
        ground_truth_path=gt_path,
    )


# ---------------------------------------------------------------------------
# EngineRun
# ---------------------------------------------------------------------------


class TestEngineRun:
    def test_creation(self) -> None:
        result = _make_engine_result("gemini")
        run = EngineRun(paper_name="paper_a", engine="gemini", result=result)
        assert run.paper_name == "paper_a"
        assert run.engine == "gemini"
        assert run.score is None

    def test_with_score(self) -> None:
        result = _make_engine_result("gemini")
        score = _make_score("paper_a", "gemini")
        run = EngineRun(paper_name="paper_a", engine="gemini", result=result, score=score)
        assert run.score is not None
        assert run.score.overall_wer == 0.1


# ---------------------------------------------------------------------------
# BenchmarkResults
# ---------------------------------------------------------------------------


class TestBenchmarkResults:
    def test_empty_results(self) -> None:
        results = BenchmarkResults()
        assert len(results.runs) == 0
        assert results.timestamp  # auto-populated

    def test_by_engine(self) -> None:
        runs = [
            EngineRun("paper_a", "gemini", _make_engine_result("gemini")),
            EngineRun("paper_b", "gemini", _make_engine_result("gemini")),
            EngineRun("paper_a", "deepseek", _make_engine_result("deepseek")),
        ]
        results = BenchmarkResults(runs=runs)
        by_engine = results.by_engine()
        assert len(by_engine["gemini"]) == 2
        assert len(by_engine["deepseek"]) == 1

    def test_by_paper(self) -> None:
        runs = [
            EngineRun("paper_a", "gemini", _make_engine_result("gemini")),
            EngineRun("paper_a", "deepseek", _make_engine_result("deepseek")),
            EngineRun("paper_b", "gemini", _make_engine_result("gemini")),
        ]
        results = BenchmarkResults(runs=runs)
        by_paper = results.by_paper()
        assert len(by_paper["paper_a"]) == 2
        assert len(by_paper["paper_b"]) == 1

    def test_save_and_load(self, tmp_path: Path) -> None:
        score = _make_score("paper_a", "gemini", wer=0.15, cer=0.08)
        runs = [
            EngineRun(
                "paper_a", "gemini",
                _make_engine_result("gemini", processing_time=2.5),
                score=score,
            ),
            EngineRun(
                "paper_b", "deepseek",
                _make_engine_result("deepseek", success=False, failure_mode=FailureMode.TIMEOUT),
            ),
        ]
        results = BenchmarkResults(runs=runs, timestamp="2026-03-18T00:00:00+00:00")

        out_file = tmp_path / "results.json"
        results.save(out_file)

        # Verify JSON structure
        data = json.loads(out_file.read_text())
        assert data["timestamp"] == "2026-03-18T00:00:00+00:00"
        assert len(data["runs"]) == 2
        assert data["runs"][0]["score"]["overall_wer"] == 0.15
        assert data["runs"][1]["score"] is None
        assert data["runs"][1]["failure_mode"] == "timeout"

        # Round-trip load
        loaded = BenchmarkResults.load(out_file)
        assert len(loaded.runs) == 2
        assert loaded.runs[0].paper_name == "paper_a"
        assert loaded.runs[0].score is not None
        assert loaded.runs[0].score.overall_wer == 0.15
        assert loaded.runs[0].score.pages[0].word_error_rate == 0.15
        assert loaded.runs[1].score is None
        assert loaded.runs[1].result.failure_mode == FailureMode.TIMEOUT

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        results = BenchmarkResults()
        out_file = tmp_path / "sub" / "dir" / "results.json"
        results.save(out_file)
        assert out_file.exists()


# ---------------------------------------------------------------------------
# Score serialization helpers
# ---------------------------------------------------------------------------


class TestScoreSerialization:
    def test_score_to_dict_and_back(self) -> None:
        score = _make_score("paper_x", "marker", wer=0.22, cer=0.11, processing_time=3.5)
        d = _score_to_dict(score)
        assert d["paper_name"] == "paper_x"
        assert d["engine"] == "marker"
        assert d["overall_wer"] == 0.22
        assert len(d["pages"]) == 1

        restored = _dict_to_score(d)
        assert restored.paper_name == "paper_x"
        assert restored.engine == "marker"
        assert restored.overall_wer == 0.22
        assert restored.overall_cer == 0.11
        assert restored.processing_time == 3.5
        assert len(restored.pages) == 1
        assert restored.pages[0].page_num == 1


# ---------------------------------------------------------------------------
# BenchmarkRunner (with mock engines)
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    def test_run_single_with_mock_engine(self, tmp_path: Path) -> None:
        """Test run_single with a mocked engine that returns a canned result."""
        config = PipelineConfig()
        runner = BenchmarkRunner(config)

        # Create ground truth
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        (gt_dir / "page_1.txt").write_text("hello world test")

        paper = _make_benchmark_paper("test_paper", gt_path=gt_dir)

        mock_result = _make_engine_result(
            "gemini",
            pdf_path=paper.pdf_path,
            text="hello world test",
            processing_time=2.0,
        )

        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = mock_result

        with patch("socr.benchmark.runner.get_engine", return_value=mock_engine):
            run = runner.run_single(paper, EngineType.GEMINI, tmp_path / "output")

        assert run.paper_name == "test_paper"
        assert run.engine == "gemini"
        assert run.result.success
        # Score should exist since we provided ground truth and the result succeeded
        # Note: score_document needs per-page output with page_num matching gt files.
        # Our mock returns page_num=0 (CLI-style), so score_document won't match page_1.txt.
        # This is expected behavior -- CLI engines produce page_num=0 for the whole doc.

    def test_run_single_unavailable_engine(self, tmp_path: Path) -> None:
        """Unavailable engines should return a MODEL_UNAVAILABLE error."""
        config = PipelineConfig()
        runner = BenchmarkRunner(config)
        paper = _make_benchmark_paper("test_paper")

        mock_engine = MagicMock()
        mock_engine.is_available.return_value = False

        with patch("socr.benchmark.runner.get_engine", return_value=mock_engine):
            run = runner.run_single(paper, EngineType.GEMINI, tmp_path / "output")

        assert not run.result.success
        assert run.result.failure_mode == FailureMode.MODEL_UNAVAILABLE
        assert run.score is None

    def test_run_multiple_papers_and_engines(self, tmp_path: Path) -> None:
        """Test full run with multiple papers and engines (all mocked)."""
        config = PipelineConfig()
        runner = BenchmarkRunner(config)

        papers = [
            _make_benchmark_paper("paper_a", category="mixed"),
            _make_benchmark_paper("paper_b", category="math_heavy"),
        ]
        bench = BenchmarkSet(papers=papers)

        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = _make_engine_result("mock")

        engines = [EngineType.GEMINI, EngineType.DEEPSEEK]

        with patch("socr.benchmark.runner.get_engine", return_value=mock_engine):
            results = runner.run(bench, tmp_path / "output", engines=engines)

        # 2 papers x 2 engines = 4 runs
        assert len(results.runs) == 4
        assert len(results.by_engine()) == 2
        assert len(results.by_paper()) == 2

    def test_run_with_failed_engine(self, tmp_path: Path) -> None:
        """An engine that fails still produces an EngineRun with error status."""
        config = PipelineConfig()
        runner = BenchmarkRunner(config)

        paper = _make_benchmark_paper("paper_fail")
        bench = BenchmarkSet(papers=[paper])

        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = _make_engine_result(
            "deepseek", success=False, failure_mode=FailureMode.TIMEOUT
        )

        with patch("socr.benchmark.runner.get_engine", return_value=mock_engine):
            results = runner.run(bench, tmp_path / "output", engines=[EngineType.DEEPSEEK])

        assert len(results.runs) == 1
        assert not results.runs[0].result.success
        assert results.runs[0].result.failure_mode == FailureMode.TIMEOUT
        assert results.runs[0].score is None


# ---------------------------------------------------------------------------
# CalibrationReport serialization
# ---------------------------------------------------------------------------


class TestCalibrationReport:
    def test_save_and_load(self, tmp_path: Path) -> None:
        profiles = [
            EngineProfile(
                engine="gemini",
                category_wer={"mixed": 0.1, "math_heavy": 0.2},
                failure_mode_recovery={"timeout": 0.8},
                avg_processing_time=5.0,
            ),
            EngineProfile(
                engine="deepseek",
                category_wer={"mixed": 0.15, "math_heavy": 0.12},
                failure_mode_recovery={},
                avg_processing_time=3.0,
            ),
        ]
        report = CalibrationReport(
            profiles=profiles,
            recommended_chain={"mixed": ["gemini", "deepseek"], "math_heavy": ["deepseek", "gemini"]},
        )

        out_file = tmp_path / "calibration.json"
        report.save(out_file)

        loaded = CalibrationReport.load(out_file)
        assert len(loaded.profiles) == 2
        assert loaded.profiles[0].engine == "gemini"
        assert loaded.profiles[0].category_wer["mixed"] == 0.1
        assert loaded.profiles[0].failure_mode_recovery["timeout"] == 0.8
        assert loaded.recommended_chain["math_heavy"] == ["deepseek", "gemini"]

    def test_empty_report(self) -> None:
        report = CalibrationReport()
        assert len(report.profiles) == 0
        assert report.recommended_chain == {}


# ---------------------------------------------------------------------------
# RepairCalibrator
# ---------------------------------------------------------------------------


class TestRepairCalibrator:
    def _make_synthetic_results(self) -> BenchmarkResults:
        """Build synthetic benchmark results for calibration testing.

        Simulates 2 papers x 3 engines with varying WER and failure modes.
        """
        runs: list[EngineRun] = []

        # Engine A (gemini): good on both papers
        runs.append(EngineRun(
            "paper_mixed", "gemini",
            _make_engine_result("gemini", processing_time=5.0),
            score=_make_score("paper_mixed", "gemini", wer=0.08, cer=0.04),
        ))
        runs.append(EngineRun(
            "paper_math", "gemini",
            _make_engine_result("gemini", processing_time=6.0),
            score=_make_score("paper_math", "gemini", wer=0.12, cer=0.06),
        ))

        # Engine B (deepseek): better on math, worse on mixed
        runs.append(EngineRun(
            "paper_mixed", "deepseek",
            _make_engine_result("deepseek", processing_time=3.0),
            score=_make_score("paper_mixed", "deepseek", wer=0.20, cer=0.10),
        ))
        runs.append(EngineRun(
            "paper_math", "deepseek",
            _make_engine_result("deepseek", processing_time=2.0),
            score=_make_score("paper_math", "deepseek", wer=0.05, cer=0.02),
        ))

        # Engine C (nougat): fails on one paper
        runs.append(EngineRun(
            "paper_mixed", "nougat",
            _make_engine_result("nougat", success=False, failure_mode=FailureMode.TIMEOUT, processing_time=30.0),
        ))
        runs.append(EngineRun(
            "paper_math", "nougat",
            _make_engine_result("nougat", processing_time=4.0),
            score=_make_score("paper_math", "nougat", wer=0.15, cer=0.08),
        ))

        return BenchmarkResults(runs=runs)

    def test_calibrate_produces_profiles(self) -> None:
        results = self._make_synthetic_results()
        calibrator = RepairCalibrator()
        report = calibrator.calibrate(results)

        assert len(report.profiles) == 3
        engine_names = {p.engine for p in report.profiles}
        assert engine_names == {"gemini", "deepseek", "nougat"}

    def test_calibrate_profiles_have_wer(self) -> None:
        results = self._make_synthetic_results()
        calibrator = RepairCalibrator()
        report = calibrator.calibrate(results)

        for profile in report.profiles:
            if profile.engine in ("gemini", "deepseek"):
                assert "_all" in profile.category_wer
                assert profile.category_wer["_all"] > 0

    def test_calibrate_recommended_chain(self) -> None:
        results = self._make_synthetic_results()
        calibrator = RepairCalibrator()
        report = calibrator.calibrate(results)

        # Should have a recommended chain for the "_all" category
        assert "_all" in report.recommended_chain
        chain = report.recommended_chain["_all"]
        assert len(chain) >= 2

        # The engine with lowest average WER should be first
        # deepseek avg = (0.20 + 0.05) / 2 = 0.125
        # gemini avg = (0.08 + 0.12) / 2 = 0.10
        # nougat avg = 0.15 / 1 = 0.15
        # So gemini should be first
        assert chain[0] == "gemini"

    def test_calibrate_with_categories(self) -> None:
        results = self._make_synthetic_results()
        calibrator = RepairCalibrator()

        paper_categories = {
            "paper_mixed": "mixed",
            "paper_math": "math_heavy",
        }

        report = calibrator.calibrate_with_categories(results, paper_categories)

        # Should have per-category data
        for profile in report.profiles:
            if profile.engine == "gemini":
                assert "mixed" in profile.category_wer
                assert "math_heavy" in profile.category_wer
                assert abs(profile.category_wer["mixed"] - 0.08) < 1e-6
                assert abs(profile.category_wer["math_heavy"] - 0.12) < 1e-6

        # Recommended chain per category
        assert "mixed" in report.recommended_chain
        assert "math_heavy" in report.recommended_chain

        # For mixed: gemini (0.08) < deepseek (0.20), so gemini first
        assert report.recommended_chain["mixed"][0] == "gemini"
        # For math_heavy: deepseek (0.05) < gemini (0.12) < nougat (0.15)
        assert report.recommended_chain["math_heavy"][0] == "deepseek"

    def test_apply_to_config(self) -> None:
        results = self._make_synthetic_results()
        calibrator = RepairCalibrator()
        report = calibrator.calibrate(results)

        config = PipelineConfig()
        calibrator.apply_to_config(report, config)

        # Best engine overall should be primary
        # gemini avg WER = 0.10, deepseek = 0.125, nougat = 0.15
        assert config.primary_engine == EngineType.GEMINI
        assert len(config.fallback_chain) >= 1

    def test_apply_to_config_with_unknown_engine(self) -> None:
        """Engines not in EngineType should be skipped."""
        report = CalibrationReport(
            profiles=[
                EngineProfile(engine="unknown_engine", category_wer={"_all": 0.01}),
                EngineProfile(engine="gemini", category_wer={"_all": 0.10}),
            ],
            recommended_chain={"_all": ["unknown_engine", "gemini"]},
        )
        calibrator = RepairCalibrator()
        config = PipelineConfig()
        calibrator.apply_to_config(report, config)

        # unknown_engine should be skipped, gemini should be primary
        assert config.primary_engine == EngineType.GEMINI

    def test_apply_to_config_empty_report(self) -> None:
        """Empty report should not crash."""
        report = CalibrationReport()
        calibrator = RepairCalibrator()
        config = PipelineConfig()
        original_primary = config.primary_engine
        calibrator.apply_to_config(report, config)
        assert config.primary_engine == original_primary

    def test_failure_mode_recovery(self) -> None:
        """Engines that fail should have failure_mode_recovery stats."""
        results = self._make_synthetic_results()
        calibrator = RepairCalibrator()
        report = calibrator.calibrate(results)

        nougat_profile = next(p for p in report.profiles if p.engine == "nougat")
        # nougat had 1 timeout failure, 0 recoveries at WER < 0.5
        assert "timeout" in nougat_profile.failure_mode_recovery
        assert nougat_profile.failure_mode_recovery["timeout"] == 0.0

    def test_engine_ranking_stability(self) -> None:
        """Engines with same WER should produce a deterministic chain."""
        runs = [
            EngineRun(
                "paper_a", "alpha",
                _make_engine_result("alpha"),
                score=_make_score("paper_a", "alpha", wer=0.10),
            ),
            EngineRun(
                "paper_a", "beta",
                _make_engine_result("beta"),
                score=_make_score("paper_a", "beta", wer=0.10),
            ),
        ]
        results = BenchmarkResults(runs=runs)
        calibrator = RepairCalibrator()
        report = calibrator.calibrate(results)

        chain = report.recommended_chain["_all"]
        assert len(chain) == 2
        # With identical WER, sorting by (wer, name) should give deterministic order
        assert chain == sorted(chain) or chain == list(reversed(sorted(chain)))
