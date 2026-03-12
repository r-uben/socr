"""OCR result data structures for smart-ocr v1.0.

Document-level results for standard mode (CLI-based engines).
Per-page results retained only for HPC mode (vLLM direct API).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DocumentStatus(str, Enum):
    """Status of document-level OCR processing."""

    PENDING = "pending"
    SUCCESS = "success"
    AUDIT_FAILED = "audit_failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class FigureInfo:
    """Metadata for a detected figure."""

    figure_num: int
    page_num: int
    figure_type: str  # chart, table, diagram, image
    description: str = ""
    image_path: str | None = None


@dataclass
class DocumentResult:
    """Result from processing a single document with one engine.

    Standard mode: one engine runs on the whole PDF via CLI, produces markdown.
    This replaces the per-page OCRResult for CLI-based engines.
    """

    document_path: Path
    engine: str
    status: DocumentStatus = DocumentStatus.PENDING
    markdown: str = ""
    pages_processed: int = 0
    processing_time: float = 0.0
    error: str | None = None
    figures: list[FigureInfo] = field(default_factory=list)
    audit_passed: bool = True
    audit_notes: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.markdown.split()) if self.markdown else 0

    @property
    def success(self) -> bool:
        return self.status == DocumentStatus.SUCCESS


# --- HPC mode: per-page results (kept for vLLM direct API) ---


class PageStatus(str, Enum):
    """Status of per-page OCR (HPC mode only)."""

    PENDING = "pending"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class PageResult:
    """Result for a single page (HPC mode — vLLM direct API)."""

    page_num: int
    text: str = ""
    status: PageStatus = PageStatus.PENDING
    engine: str = ""
    processing_time: float = 0.0
    error_message: str = ""
    figures: list[FigureInfo] = field(default_factory=list)
    audit_passed: bool = True
    audit_notes: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0

    def needs_reprocessing(self) -> bool:
        if self.status == PageStatus.ERROR:
            return True
        return not self.audit_passed
