"""smart-ocr - Multi-engine document OCR with cascading fallback."""

__version__ = "0.2.1"

from smart_ocr.core.config import AgentConfig
from smart_ocr.core.document import Document, DocumentType
from smart_ocr.core.result import OCRResult, PageResult
from smart_ocr.pipeline.processor import OCRPipeline

__all__ = [
    "AgentConfig",
    "Document",
    "DocumentType",
    "OCRResult",
    "PageResult",
    "OCRPipeline",
]
