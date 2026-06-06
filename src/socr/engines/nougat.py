"""Nougat OCR engine adapter.

CLI: nougat-ocr <path> -o <dir> [--model] [--device auto|cpu|cuda|mps]
     [--batch-size N] [--pages 0-5] [-q]
Flat @click.command. Local inference, best for academic papers with equations.
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
            "-o",
            str(output_dir),
            "--model",
            config.nougat_model,
        ]
        if config.quiet:
            cmd.append("-q")
        if config.verbose:
            cmd.append("-v")
        if config.reprocess:
            cmd.append("--reprocess")
        return cmd
