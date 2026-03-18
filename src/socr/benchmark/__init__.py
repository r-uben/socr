"""Benchmark suite for OCR quality evaluation."""

from socr.benchmark.calibrate import CalibrationReport, EngineProfile, RepairCalibrator
from socr.benchmark.dataset import BenchmarkPaper, BenchmarkSet, build_benchmark_set
from socr.benchmark.ground_truth import GroundTruthExtractor, PageGroundTruth
from socr.benchmark.rasterize import PaperRasterizer
from socr.benchmark.runner import BenchmarkResults, BenchmarkRunner, EngineRun
from socr.benchmark.scorer import BenchmarkScorer, DocumentScore, PageScore

__all__ = [
    "BenchmarkPaper",
    "BenchmarkResults",
    "BenchmarkRunner",
    "BenchmarkScorer",
    "BenchmarkSet",
    "CalibrationReport",
    "DocumentScore",
    "EngineProfile",
    "EngineRun",
    "GroundTruthExtractor",
    "PageGroundTruth",
    "PageScore",
    "PaperRasterizer",
    "RepairCalibrator",
    "build_benchmark_set",
]
