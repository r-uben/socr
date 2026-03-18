"""Benchmark suite for OCR quality evaluation."""

from socr.benchmark.dataset import BenchmarkPaper, BenchmarkSet, build_benchmark_set
from socr.benchmark.ground_truth import GroundTruthExtractor, PageGroundTruth
from socr.benchmark.rasterize import PaperRasterizer
from socr.benchmark.scorer import BenchmarkScorer, DocumentScore, PageScore

__all__ = [
    "BenchmarkPaper",
    "BenchmarkScorer",
    "BenchmarkSet",
    "DocumentScore",
    "GroundTruthExtractor",
    "PageGroundTruth",
    "PageScore",
    "PaperRasterizer",
    "build_benchmark_set",
]
