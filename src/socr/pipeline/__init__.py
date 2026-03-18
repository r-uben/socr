"""Pipeline orchestration for OCR processing."""

from socr.pipeline.orchestrator import UnifiedPipeline
from socr.pipeline.processor import StandardPipeline

__all__ = ["StandardPipeline", "UnifiedPipeline"]
