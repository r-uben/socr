"""Marker OCR engine adapter.

CLI: marker-ocr <pdf> -o <dir> [--device auto|cpu|cuda|mps] [--force-ocr]
     [--pages 0-5] [-q]
Flat @click.command. Local inference, layout-aware (Surya + Texify).

Issue #1243: ``marker-ocr`` 0.2.x accepts PDFs only. The shared
``BaseEngine.process_pages`` path must pass a one-page PDF (not a PNG folder).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.engines.base import BaseEngine, reports_empty_response
from socr.routing.cascade_policy import sanitize_cli_stderr

logger = logging.getLogger(__name__)


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
        *,
        page_index: int | None = None,
    ) -> list[str]:
        cmd = [
            self.cli_command,
            str(pdf_path),
            "-o",
            str(output_dir),
            "--device",
            config.marker_device,
        ]
        if config.marker_force_ocr:
            cmd.append("--force-ocr")
        if page_index is not None:
            cmd.extend(["--pages", str(page_index)])
        if config.quiet:
            cmd.append("-q")
        if config.verbose:
            cmd.append("-v")
        if config.reprocess:
            cmd.append("--reprocess")
        return cmd

    def process_pages(
        self,
        pdf_path: Path,
        page_nums: list[int],
        config: PipelineConfig,
        dpi: int = 200,
        subprocess_timeout: float | None = None,
    ) -> list[PageOutput]:
        """Run marker once per requested page on a single-page temp PDF."""
        import fitz

        from socr.core.pdf import open_pdf

        start = time.time()
        effective_timeout = subprocess_timeout if subprocess_timeout is not None else config.timeout
        outputs: list[PageOutput] = []

        for page_num in page_nums:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                one_pdf = tmp / f"page_{page_num:04d}.pdf"
                cli_out = tmp / "out"
                with open_pdf(pdf_path) as doc:
                    single = fitz.open()
                    single.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
                    single.save(one_pdf)
                    single.close()

                cmd = self._build_command(one_pdf, cli_out, config, page_index=0)
                logger.info("[marker] page %s: %s", page_num, " ".join(cmd))
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=effective_timeout,
                    )
                except subprocess.TimeoutExpired:
                    outputs.append(
                        PageOutput(
                            page_num=page_num,
                            status=PageStatus.ERROR,
                            engine=self.name,
                            failure_mode=FailureMode.TIMEOUT,
                            audit_passed=False,
                            error=f"Timeout after {effective_timeout}s",
                        )
                    )
                    continue

                stderr_text = result.stderr or ""
                stdout_text = result.stdout or ""
                cli_note = ""
                if result.returncode != 0:
                    cli_note = sanitize_cli_stderr(stderr_text, stdout_text) or (
                        f"CLI exited {result.returncode}"
                    )
                stem = f"page_{page_num:04d}"
                text = self._read_page_output(stem, cli_out, scan_root=one_pdf.parent)
                if not text:
                    # marker writes ``<stem>.md`` beside PDF basename when input is a file
                    alt = cli_out / f"{one_pdf.stem}.md"
                    if alt.is_file():
                        text = alt.read_text(encoding="utf-8", errors="replace")
                    canon = cli_out / one_pdf.stem / f"{one_pdf.stem}.md"
                    if not text and canon.is_file():
                        text = canon.read_text(encoding="utf-8", errors="replace")

                empty_answer = reports_empty_response(stderr_text, one_pdf.stem) or (
                    text is not None and not text.strip()
                )
                if empty_answer or not text:
                    if not cli_note and result.returncode == 0:
                        cli_note = sanitize_cli_stderr(stderr_text, stdout_text) or (
                            "CLI exit 0 but no markdown output (marker accepts PDFs only)"
                        )
                    outputs.append(
                        PageOutput(
                            page_num=page_num,
                            status=PageStatus.ERROR,
                            engine=self.name,
                            failure_mode=FailureMode.CLI_ERROR
                            if cli_note
                            else FailureMode.EMPTY_OUTPUT,
                            audit_passed=False,
                            error=(
                                f"No output for page {page_num}"
                                + (f" ({cli_note})" if cli_note else "")
                            ),
                        )
                    )
                    continue

                text = self._clean_output(text, self.name)
                elapsed = time.time() - start
                outputs.append(
                    PageOutput(
                        page_num=page_num,
                        text=text,
                        status=PageStatus.SUCCESS,
                        engine=self.name,
                        processing_time=elapsed / max(len(page_nums), 1),
                        audit_passed=True,
                    )
                )
        return outputs
