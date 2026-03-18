"""OCR result data structures for socr.

Canonical engine contract: all engines return EngineResult with structured
PageOutput list. CLI engines produce a single PageOutput (page_num=0) with
the full document text. HTTP engines produce per-page PageOutputs.
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


class PageStatus(str, Enum):
    """Status of per-page OCR."""

    PENDING = "pending"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class FailureMode(str, Enum):
    """Why an engine result or page failed.

    Used by downstream repair routing to decide what fallback strategy to use.
    """

    NONE = "none"
    TIMEOUT = "timeout"
    CLI_ERROR = "cli_error"
    EMPTY_OUTPUT = "empty_output"
    API_ERROR = "api_error"
    MODEL_UNAVAILABLE = "model_unavailable"
    AUDIT_FAILED = "audit_failed"
    HALLUCINATION = "hallucination"
    REFUSAL = "refusal"
    GARBAGE = "garbage"
    LOW_WORD_COUNT = "low_word_count"
    TRUNCATED = "truncated"


@dataclass
class FigureInfo:
    """Metadata for a detected figure."""

    figure_num: int
    page_num: int
    figure_type: str  # chart, table, diagram, image
    description: str = ""
    image_path: str | None = None
    engine: str = ""


@dataclass
class PageOutput:
    """Structured output for a single page.

    For CLI engines that process whole documents at once, a single PageOutput
    with page_num=0 holds the entire document text. For HTTP/per-page engines,
    each page gets its own PageOutput.
    """

    page_num: int
    text: str = ""
    status: PageStatus = PageStatus.PENDING
    failure_mode: FailureMode = FailureMode.NONE
    engine: str = ""
    processing_time: float = 0.0
    error: str = ""
    confidence: float = 0.0
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


@dataclass
class EngineResult:
    """Canonical result from any OCR engine.

    Replaces raw markdown blobs with structured per-page outputs.
    Engines return: status, failure_mode, pages, model_version, cost.
    """

    document_path: Path
    engine: str
    status: DocumentStatus = DocumentStatus.PENDING
    failure_mode: FailureMode = FailureMode.NONE
    pages: list[PageOutput] = field(default_factory=list)
    model_version: str = ""
    cost: float = 0.0
    pages_processed: int = 0
    processing_time: float = 0.0
    error: str | None = None
    figures: list[FigureInfo] = field(default_factory=list)
    audit_passed: bool = True
    audit_notes: list[str] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        """Assemble full document text from page outputs."""
        texts = [p.text for p in self.pages if p.text]
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0]
        return "\n\n---\n\n".join(texts)

    @property
    def word_count(self) -> int:
        return len(self.markdown.split()) if self.pages else 0

    @property
    def success(self) -> bool:
        return self.status == DocumentStatus.SUCCESS
