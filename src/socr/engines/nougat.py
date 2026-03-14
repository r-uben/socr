"""Nougat OCR engine adapter.

CLI: nougat-ocr <path> -o <dir> [--pages 0-5] [--device auto|cpu|cuda|mps] [-q]
Flat @click.command — no subcommands.
"""

from pathlib import Path

from socr.core.config import PipelineConfig
from socr.engines.base import BaseEngine


class NougatEngine(BaseEngine):
    """Adapter for nougat-ocr-cli."""

    @property
    def name(self) -> str:
        return "nougat"

    @property
    def cli_command(self) -> str:
        return "nougat-ocr"

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
            "--model", config.nougat_model,
        ]
        if config.quiet:
            cmd.append("-q")
        return cmd
