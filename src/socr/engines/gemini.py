"""Gemini OCR engine adapter.

CLI: gemini-ocr <path> -o <dir> [--model] [--task convert|extract|table] [-w N] [-q]
Flat @click.command. Uses Google Gemini Files API (native PDF upload).
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

    @property
    def model_version(self) -> str:
        return "gemini-3-flash-preview"

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
            "-o",
            str(output_dir),
            "--model",
            config.gemini_model,
            "--task",
            config.gemini_task,
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
