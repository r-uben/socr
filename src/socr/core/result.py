"""OCR result data structures for socr.

Canonical engine contract: all engines return EngineResult with structured
PageOutput list. CLI engines produce a single PageOutput (page_num=0) with
the full document text. HTTP engines produce per-page PageOutputs.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Shared sentinel for the lost-content contract between _phase_assemble
# (which composes EngineResult.error) and the CLI (which exits non-zero when
# content was erased). A shared constant, not a magic string in two modules.
LOST_CONTENT_NOTE = "produced no usable output"


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
    RECITATION = "recitation"  # Gemini copyright/recitation filter blocked verbatim output
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

    def to_dict(self) -> dict:
        return {
            "figure_num": self.figure_num,
            "page_num": self.page_num,
            "figure_type": self.figure_type,
            "description": self.description,
            "image_path": self.image_path,
            "engine": self.engine,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FigureInfo":
        return cls(
            figure_num=d["figure_num"],
            page_num=d["page_num"],
            figure_type=d["figure_type"],
            description=d.get("description", ""),
            image_path=d.get("image_path"),
            engine=d.get("engine", ""),
        )


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
    escalated_from: str = ""  # engine that failed, triggering escalation
    cost_usd: float = 0.0  # estimated USD cost of producing this page output
    # Agentic routing provenance (B3) — empty for non-agentic runs
    provider_id: str = ""  # ProviderProfile.id that produced this output
    provider_model: str = ""  # resolved model name (e.g. qwen3-vl:30b-a3b-instruct)
    provider_backend: str = ""  # backend (e.g. ollama, gemini-api)
    skip_reason: str = ""  # why the rung was not tried (e.g. budget exceeded)

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0

    def needs_reprocessing(self) -> bool:
        if self.status == PageStatus.ERROR:
            return True
        return not self.audit_passed

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict. Used for content-addressed caching."""
        return {
            "page_num": self.page_num,
            "text": self.text,
            "status": self.status.value,
            "failure_mode": self.failure_mode.value,
            "engine": self.engine,
            "processing_time": self.processing_time,
            "error": self.error,
            "confidence": self.confidence,
            "figures": [f.to_dict() for f in self.figures],
            "audit_passed": self.audit_passed,
            "audit_notes": list(self.audit_notes),
            "escalated_from": self.escalated_from,
            "cost_usd": self.cost_usd,
            "provider_id": self.provider_id,
            "provider_model": self.provider_model,
            "provider_backend": self.provider_backend,
            "skip_reason": self.skip_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PageOutput":
        return cls(
            page_num=d["page_num"],
            text=d.get("text", ""),
            status=PageStatus(d.get("status", PageStatus.PENDING.value)),
            failure_mode=FailureMode(d.get("failure_mode", FailureMode.NONE.value)),
            engine=d.get("engine", ""),
            processing_time=d.get("processing_time", 0.0),
            error=d.get("error", ""),
            confidence=d.get("confidence", 0.0),
            figures=[FigureInfo.from_dict(f) for f in d.get("figures", [])],
            audit_passed=d.get("audit_passed", True),
            audit_notes=list(d.get("audit_notes", [])),
            escalated_from=d.get("escalated_from", ""),
            cost_usd=d.get("cost_usd", 0.0),
            provider_id=d.get("provider_id", ""),
            provider_model=d.get("provider_model", ""),
            provider_backend=d.get("provider_backend", ""),
            skip_reason=d.get("skip_reason", ""),
        )


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
