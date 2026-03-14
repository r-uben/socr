"""Marker OCR engine adapter.

CLI: marker-ocr <path> -o <dir> [--pages 0-5] [--device auto|cpu|cuda|mps] [--force-ocr] [-q]
Flat @click.command — no subcommands.
"""

from pathlib import Path

from socr.core.config import PipelineConfig
from socr.engines.base import BaseEngine


class MarkerEngine(BaseEngine):
    """Adapter for marker-ocr-cli."""

    @property
    def name(self) -> str:
        return "marker"

    @property
    def cli_command(self) -> str:
        return "marker-ocr"

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
            "--device", config.marker_device,
        ]
        if config.quiet:
            cmd.append("-q")
        return cmd
