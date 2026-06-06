"""GLM-OCR engine adapter.

CLI: glm-ocr process <path> -o <dir> [--task text|formula|table|figure]
     [--backend ollama|transformers|vllm] [--dpi N] [-w N] [-q]
Click group — requires explicit 'process' subcommand.
"""

import logging
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.engines.base import BaseEngine
from socr.engines.deepseek import _check_ollama_model

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "glm-ocr"


class GLMEngine(BaseEngine):
    """Adapter for glm-ocr-cli."""

    @property
    def name(self) -> str:
        return "glm"

    @property
    def cli_command(self) -> str:
        return "glm-ocr"

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
        """Process document, with Ollama model pre-check."""
        if config.glm_backend == "ollama":
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
            config.glm_task,
            "--backend",
            config.glm_backend,
            "--dpi",
            str(config.render_dpi),
        ]
        if config.workers > 1:
            cmd.extend(["-w", str(config.workers)])
        if config.save_figures:
            cmd.append("--analyze-figures")
        if config.quiet:
            cmd.append("-q")
        if config.verbose:
            cmd.append("--verbose")
        if config.reprocess:
            cmd.append("--reprocess")
        return cmd
