"""Mistral OCR engine adapter.

CLI: mistral-ocr <path> -o <dir> [--model] [-w N] [--max-pages N] [-q]
Flat @click.command. Uses Mistral AI OCR API with structured per-page output.
"""

import os
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.engines.base import BaseEngine


class MistralEngine(BaseEngine):
    """Adapter for mistral-ocr-cli."""

    @property
    def name(self) -> str:
        return "mistral"

    @property
    def cli_command(self) -> str:
        return "mistral-ocr"

    @property
    def model_version(self) -> str:
        return "mistral-ocr-latest"

    def is_available(self) -> bool:
        if not os.environ.get("MISTRAL_API_KEY"):
            return False
        return super().is_available()

    def _build_command(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> list[str]:
        cmd = [
            self.cli_command,
            str(pdf_path),
            "-o",
            str(output_dir),
            "--model",
            config.mistral_model,
        ]
        if config.workers > 1:
            cmd.extend(["-w", str(config.workers)])
        if config.save_figures:
            cmd.append("--include-images")
        else:
            cmd.append("--no-images")
        if config.quiet:
            cmd.append("-q")
        if config.verbose:
            cmd.append("-v")
        if config.reprocess:
            cmd.append("--reprocess")
        return cmd
