"""Core data models for socr."""

from socr.core.config import EngineType, HPCConfig, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.metadata import MetadataManager
from socr.core.result import (
    DocumentResult,
    DocumentStatus,
    FigureInfo,
    PageResult,
    PageStatus,
)

__all__ = [
    "DocumentHandle",
    "DocumentResult",
    "DocumentStatus",
    "EngineType",
    "FigureInfo",
    "HPCConfig",
    "MetadataManager",
    "PageResult",
    "PageStatus",
    "PipelineConfig",
]
