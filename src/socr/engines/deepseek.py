"""DeepSeek OCR engine adapter.

CLI: deepseek-ocr process <path> -o <dir> [--task convert|ocr|layout|extract|parse]
     [--backend ollama|vllm] [--dpi N] [-w N] [--max-dim N] [-q]
Click group — requires explicit 'process' subcommand.

Task modes:
  - "convert": Structured markdown (uses <|grounding|> prefix). Best default.
  - "ocr": Raw transcription ("Free OCR."). May hallucinate formatting instructions.
  - "layout": Layout detection.
  - "extract": Plain text extraction.
  - "parse": Figure/chart parsing.
"""

import logging
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.core.ollama_utils import check_ollama_model as _check_ollama_model
from socr.engines.base import BaseEngine

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "deepseek-ocr"


class DeepSeekEngine(BaseEngine):
    """Adapter for deepseek-ocr-cli."""

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def cli_command(self) -> str:
        return "deepseek-ocr"

    def is_available(self) -> bool:
        """Check CLI is installed AND Ollama model is available."""
        if not super().is_available():
            return False
        error = _check_ollama_model(OLLAMA_MODEL)
        if error:
            logger.debug(f"[{self.name}] {error}")
            return False
        return True

    def process_document(self, pdf_path, output_dir, config):
        """Process document, with Ollama model pre-check for ollama backend."""
        if config.deepseek_backend != "vllm":
            error = _check_ollama_model(OLLAMA_MODEL)
            if error:
                from socr.core.result import DocumentStatus, EngineResult, FailureMode

                logger.error(f"[{self.name}] {error}")
                return EngineResult(
                    document_path=pdf_path,
                    engine=self.name,
                    status=DocumentStatus.ERROR,
                    failure_mode=FailureMode.MODEL_UNAVAILABLE,
                    error=error,
                    processing_time=0.0,
                )
        return super().process_document(pdf_path, output_dir, config)

    def _build_command(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> list[str]:
        cmd = [
            self.cli_command,
            "process",
            str(pdf_path),
            "-o",
            str(output_dir),
            "--task",
            config.deepseek_task,
            "--backend",
            config.deepseek_backend,
            "--dpi",
            str(config.render_dpi),
        ]
        if config.workers > 1:
            cmd.extend(["-w", str(config.workers)])
        if config.deepseek_backend == "vllm":
            cmd.extend(["--vllm-url", config.deepseek_vllm_url])
        if config.save_figures:
            cmd.append("--analyze-figures")
        if config.quiet:
            cmd.append("-q")
        if config.verbose:
            cmd.append("--verbose")
        if config.reprocess:
            cmd.append("--reprocess")
        return cmd
