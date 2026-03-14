"""Gemini OCR engine adapter.

CLI: gemini-ocr <path> -o <dir> [--model] [-q]
Flat @click.command — no subcommands.
"""

import os
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.engines.base import BaseEngine


class GeminiEngine(BaseEngine):
    """Adapter for gemini-ocr-cli."""

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def cli_command(self) -> str:
        return "gemini-ocr"

    def is_available(self) -> bool:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
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
            "-o", str(output_dir),
            "--model", config.gemini_model,
        ]
        if config.quiet:
            cmd.append("-q")
        return cmd
