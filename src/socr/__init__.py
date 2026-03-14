"""socr - Multi-engine document OCR with cascading fallback."""

__version__ = "1.0.1"

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import DocumentResult

__all__ = [
    "DocumentHandle",
    "DocumentResult",
    "EngineType",
    "PipelineConfig",
]
