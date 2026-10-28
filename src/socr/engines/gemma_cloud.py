"""Gemma vision OCR via Ollama Cloud — a dedicated agentic ladder rung.

Uses ``qwen-ocr`` with half-page 300-dpi crops and output guards (token cap,
repeated-text / image-URL rejection, equation-unverified flag). Judging is
unchanged: the orchestrator still runs every candidate through the same page
judge as other ladder rungs.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.core.ollama_utils import DEFAULT_PROBE_TIMEOUT_SEC, probe_model_generation
from socr.core.providers import PROFILE_GEMMA_CLOUD, is_cloud_model
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.engines.base import BaseEngine
from socr.routing.cascade_policy import extract_cli_actionable_error, parse_token_usage
from socr.routing.gemma_crops import render_page_half_crops
from socr.routing.gemma_guards import apply_gemma_output_guards
from socr.tables.extract import resolve_ollama_host


def gemma_cloud_model_available(model: str) -> tuple[bool, str]:
    """Probe the configured Gemma cloud tag with a bounded generation."""
    if not is_cloud_model(model):
        return False, f"{model}: not a cloud model tag"
    ok, reason = probe_model_generation(
        resolve_ollama_host().rstrip("/"),
        model,
        DEFAULT_PROBE_TIMEOUT_SEC,
    )
    return ok, "" if ok else f"{model}: {reason}"


class GemmaCloudEngine(BaseEngine):
    """Adapter: ``qwen-ocr process`` on 300-dpi half-page crops."""

    @property
    def name(self) -> str:
        return "gemma"

    @property
    def cli_command(self) -> str:
        return "qwen-ocr"

    def resolved_model_version(self, config: PipelineConfig) -> str:
        return (config.gemma_cloud_model or PROFILE_GEMMA_CLOUD.model).strip()

    def is_available(self) -> bool:
        try:
            result = subprocess.run(
                [self.cli_command, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
        model = PROFILE_GEMMA_CLOUD.model
        ok, _ = gemma_cloud_model_available(model)
        return ok

    def _build_command(
        self,
        input_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> list[str]:
        model = self.resolved_model_version(config)
        dpi = getattr(config, "gemma_crop_dpi", 300) or 300
        cmd = [
            self.cli_command,
            "process",
            str(input_path),
            "-o",
            str(output_dir),
            "--backend",
            "ollama",
            "--model",
            model,
            "--dpi",
            str(dpi),
        ]
        if config.quiet:
            cmd.append("-q")
        if config.verbose:
            cmd.append("-v")
        if config.reprocess:
            cmd.append("--reprocess")
        return cmd

    def _run_cli_on_images(
        self,
        images_dir: Path,
        config: PipelineConfig,
        subprocess_timeout: float | None,
    ) -> tuple[str, int | None, int | None, str]:
        with tempfile.TemporaryDirectory() as tmp:
            cli_out = Path(tmp) / "out"
            cmd = self._build_command(images_dir, cli_out, config)
            effective_timeout = (
                subprocess_timeout if subprocess_timeout is not None else config.timeout
            )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            inp_tok, out_tok = parse_token_usage(result.stderr or "", result.stdout or "")
            if result.returncode != 0:
                detail = extract_cli_actionable_error(
                    result.stderr or "",
                    result.stdout or "",
                    returncode=result.returncode,
                )
                return "", inp_tok, out_tok, f"CLI exited {result.returncode}: {detail}"
            aggregate, err = self._read_aggregated_pages(cli_out, images_dir, {})
            if err:
                return "", inp_tok, out_tok, err
            parts = [aggregate[k] for k in sorted(aggregate or {})]
            return "\n\n".join(p for p in parts if p), inp_tok, out_tok, ""

    def process_pages(
        self,
        pdf_path: Path,
        page_nums: list[int],
        config: PipelineConfig,
        dpi: int = 200,
        subprocess_timeout: float | None = None,
    ) -> list[PageOutput]:
        """OCR each page from left/right 300-dpi crops with output guards."""
        start = time.time()
        crop_dpi = getattr(config, "gemma_crop_dpi", 300) or 300
        max_tokens = getattr(config, "gemma_max_output_tokens", 4096) or 4096
        outputs: list[PageOutput] = []

        for page_num in page_nums:
            with tempfile.TemporaryDirectory() as tmpdir:
                crop_dir = Path(tmpdir) / "crops"
                try:
                    stems = render_page_half_crops(pdf_path, page_num, crop_dir, dpi=crop_dpi)
                except Exception as exc:
                    outputs.append(
                        PageOutput(
                            page_num=page_num,
                            status=PageStatus.ERROR,
                            engine=self.name,
                            failure_mode=FailureMode.CLI_ERROR,
                            error=f"gemma crop render failed: {exc}",
                            audit_passed=False,
                        )
                    )
                    continue

                half_texts: list[str] = []
                inp_sum = out_sum = None
                cli_err = ""
                per_half_timeout = None
                if subprocess_timeout is not None:
                    per_half_timeout = max(30.0, subprocess_timeout / max(len(stems), 1))

                for _stem, crop_path in stems:
                    half_dir = Path(tmpdir) / f"half_{_stem}"
                    half_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(crop_path, half_dir / crop_path.name)
                    text, inp, out, err = self._run_cli_on_images(
                        half_dir, config, per_half_timeout
                    )
                    if inp is not None:
                        inp_sum = (inp_sum or 0) + inp
                    if out is not None:
                        out_sum = (out_sum or 0) + out
                    if err:
                        cli_err = err
                    elif text.strip():
                        half_texts.append(text.strip())

                if cli_err and not half_texts:
                    outputs.append(
                        PageOutput(
                            page_num=page_num,
                            status=PageStatus.ERROR,
                            engine=self.name,
                            failure_mode=FailureMode.CLI_ERROR,
                            error=cli_err,
                            audit_passed=False,
                            llm_input_tokens=inp_sum,
                            llm_output_tokens=out_sum,
                        )
                    )
                    continue

                merged = "\n\n".join(half_texts)
                guarded = apply_gemma_output_guards(
                    merged,
                    page_num=page_num,
                    max_output_tokens=max_tokens,
                )
                guarded.processing_time = (time.time() - start) / max(len(page_nums), 1)
                guarded.llm_input_tokens = inp_sum
                guarded.llm_output_tokens = out_sum
                outputs.append(guarded)

        return outputs
