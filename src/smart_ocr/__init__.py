"""smart-ocr - Multi-engine document OCR with cascading fallback."""

__version__ = "0.2.7"

from smart_ocr.core.config import EngineType, PipelineConfig
from smart_ocr.core.document import DocumentHandle
from smart_ocr.core.result import DocumentResult

__all__ = [
    "DocumentHandle",
    "DocumentResult",
    "EngineType",
    "PipelineConfig",
]
