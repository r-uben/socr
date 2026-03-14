"""Mistral OCR engine adapter.

CLI: mistral-ocr <path> -o <dir> [--model] [--reprocess]
Flat @click.command — no subcommands. No -q flag.
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
            "-o", str(output_dir),
            "--model", config.mistral_model,
        ]
        return cmd
