"""Core data models for socr."""

from socr.core.born_digital import (
    BornDigitalDetector,
    DocumentAssessment,
    PageAssessment,
)
from socr.core.chunker import PDFChunk, PDFChunker
from socr.core.config import EngineType, HPCConfig, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.normalizer import OutputNormalizer
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    FigureInfo,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState, PageState

__all__ = [
    "BornDigitalDetector",
    "DocumentAssessment",
    "DocumentHandle",
    "PDFChunk",
    "PDFChunker",
    "DocumentState",
    "DocumentStatus",
    "EngineResult",
    "EngineType",
    "FailureMode",
    "FigureInfo",
    "HPCConfig",
    "OutputNormalizer",
    "PageAssessment",
    "PageOutput",
    "PageState",
    "PageStatus",
    "PipelineConfig",
]
