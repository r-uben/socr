"""Core data models for socr."""

from socr.core.born_digital import (
    BornDigitalDetector,
    DocumentAssessment,
    PageAssessment,
)
from socr.core.config import EngineType, HPCConfig, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.metadata import MetadataManager
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    FigureInfo,
    PageOutput,
    PageStatus,
)

__all__ = [
    "BornDigitalDetector",
    "DocumentAssessment",
    "DocumentHandle",
    "DocumentStatus",
    "EngineResult",
    "EngineType",
    "FailureMode",
    "FigureInfo",
    "HPCConfig",
    "MetadataManager",
    "PageAssessment",
    "PageOutput",
    "PageStatus",
    "PipelineConfig",
]
