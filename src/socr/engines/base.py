"""Base engine adapters for socr v1.0.

Two engine families:
  - BaseEngine: CLI-based, one subprocess per document (standard mode)
  - BaseHTTPEngine: HTTP API, per-page processing (HPC mode with vLLM)
"""

import logging
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.core.result import DocumentResult, DocumentStatus, FigureInfo, PageResult, PageStatus

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Sanitize a filename for use as a directory name."""
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()


class BaseEngine(ABC):
    """Abstract base class for CLI-based OCR engines.

    Each engine wraps a sibling CLI tool (gemini-ocr, nougat-ocr, etc.).
    The contract: call CLI once per PDF, read output markdown from
    {output_dir}/{stem}/{stem}.md.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier (matches EngineType value)."""
        ...

    @property
    @abstractmethod
    def cli_command(self) -> str:
        """The CLI binary name (e.g., 'gemini-ocr', 'nougat-ocr')."""
        ...

    def is_available(self) -> bool:
        """Check if the CLI tool is installed and callable."""
        try:
            result = subprocess.run(
                [self.cli_command, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def process_document(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> DocumentResult:
        """Process a PDF document via CLI subprocess.

        Calls the CLI once on the whole PDF, reads the output markdown.
        """
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir)
            cmd = self._build_command(pdf_path, tmp_out, config)

            logger.info(f"[{self.name}] Running: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout,
                )

                if result.returncode != 0:
                    stderr = result.stderr.strip() if result.stderr else "Unknown error"
                    logger.error(f"[{self.name}] CLI failed: {stderr}")
                    return DocumentResult(
                        document_path=pdf_path,
                        engine=self.name,
                        status=DocumentStatus.ERROR,
                        error=f"CLI exited {result.returncode}: {stderr[:500]}",
                        processing_time=time.time() - start_time,
                    )

                # Read output markdown
                markdown = self._read_output(pdf_path, tmp_out)
                if markdown is None:
                    return DocumentResult(
                        document_path=pdf_path,
                        engine=self.name,
                        status=DocumentStatus.ERROR,
                        error="CLI produced no output markdown",
                        processing_time=time.time() - start_time,
                    )

                return DocumentResult(
                    document_path=pdf_path,
                    engine=self.name,
                    status=DocumentStatus.SUCCESS,
                    markdown=markdown,
                    processing_time=time.time() - start_time,
                )

            except subprocess.TimeoutExpired:
                return DocumentResult(
                    document_path=pdf_path,
                    engine=self.name,
                    status=DocumentStatus.ERROR,
                    error=f"Timeout after {config.timeout}s",
                    processing_time=time.time() - start_time,
                )

    @abstractmethod
    def _build_command(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> list[str]:
        """Build the CLI command for this engine."""
        ...

    def _read_output(self, pdf_path: Path, output_dir: Path) -> str | None:
        """Read the output markdown from the CLI's output directory.

        Sibling CLIs use different output structures:
          - gemini-ocr: {output_dir}/{sanitized_stem}/{sanitized_stem}.md
          - deepseek-ocr: {output_dir}/{stem}/{stem}.md
          - mistral-ocr: {output_dir}/{stem}.md (flat, no subdirectory)
          - nougat-ocr/marker-ocr: {output_dir}/{stem}/{stem}.md
        """
        stem = sanitize_filename(pdf_path.stem)

        # Try subdirectory layout first: {output_dir}/{stem}/{stem}.md
        md_path = output_dir / stem / f"{stem}.md"
        if md_path.exists():
            return self._clean_output(md_path.read_text(encoding="utf-8"))

        # Try flat layout: {output_dir}/{stem}.md
        flat_path = output_dir / f"{stem}.md"
        if flat_path.exists():
            return self._clean_output(flat_path.read_text(encoding="utf-8"))

        # Fallback: find any .md file (handles sanitization mismatches)
        for md_file in output_dir.rglob("*.md"):
            # Guard against symlinks escaping the temp directory
            if not md_file.resolve().is_relative_to(output_dir.resolve()):
                logger.warning(f"[{self.name}] Skipping symlink outside output dir: {md_file}")
                continue
            logger.warning(f"[{self.name}] Output found via rglob fallback: {md_file}")
            return self._clean_output(md_file.read_text(encoding="utf-8"))

        return None

    @staticmethod
    def _clean_output(text: str) -> str:
        """Remove frontmatter and metadata headers from CLI output.

        Handles:
          - YAML frontmatter (--- ... ---)
          - Metadata headers (# OCR Results + **Original File:** + **Processed:** + ---)
        """
        import re

        # Strip YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                text = parts[2].strip()

        # Strip metadata header block (mistral-ocr format):
        # # OCR Results\n\n**Original File:**...\n**Full Path:**...\n**Processed:**...\n\n---
        # Requires at least one metadata line to avoid stripping real "# OCR Results" headings
        text = re.sub(
            r"^#\s*OCR Results\s*\n+"
            r"(?:\*\*(?:Original File|Full Path|Processed|Processing Time):\*\*[^\n]*\n)+"
            r"\s*(?:---\s*\n)?",
            "",
            text,
        ).strip()

        return text


class BaseHTTPEngine(ABC):
    """Abstract base class for HTTP API engines (vLLM, HPC mode).

    These engines call a local vLLM server per-page via OpenAI-compatible API.
    They are NOT CLI-based — they use httpx to talk to a running server.
    """

    def __init__(self) -> None:
        self._initialized: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def initialize(self) -> bool:
        """Connect to the server and verify it's ready."""
        ...

    def is_available(self) -> bool:
        """Check if the engine can be initialized."""
        return self._initialized or self.initialize()

    @abstractmethod
    def process_image(self, image: "Image.Image", page_num: int = 1) -> PageResult:
        """Process a single page image and return text."""
        ...

    def describe_figure(
        self,
        image: "Image.Image",
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureInfo:
        """Describe a figure image. Override in subclasses that support this."""
        return FigureInfo(
            figure_num=0,
            page_num=0,
            figure_type=figure_type,
            description="Figure description not supported by this engine",
        )

    def close(self) -> None:
        """Clean up resources."""
        pass

    @staticmethod
    def _create_success_result(
        page_num: int,
        text: str,
        confidence: float = 0.0,
        processing_time: float = 0.0,
    ) -> PageResult:
        return PageResult(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            processing_time=processing_time,
            confidence=confidence,
        )

    @staticmethod
    def _create_error_result(page_num: int, error: str) -> PageResult:
        return PageResult(
            page_num=page_num,
            status=PageStatus.ERROR,
            error_message=error,
        )
