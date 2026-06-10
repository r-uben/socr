"""socr - Multi-engine document OCR with cascading fallback."""

__version__ = "2.5.0"

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import EngineResult

__all__ = [
    "DocumentHandle",
    "EngineResult",
    "EngineType",
    "PipelineConfig",
]
