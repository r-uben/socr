"""Gemma vision OCR via Ollama Cloud — a dedicated agentic ladder rung.

Uses ``qwen-ocr`` with layout-aware 300-dpi crops and output guards (token cap,
repeated-text / image-URL rejection, equation-unverified flag). Judging is
unchanged: the orchestrator still runs every candidate through the same page
judge as other ladder rungs.

Cloud egress is opt-in (``enable_gemma_cloud`` / ``--enable-gemma-cloud``).
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
from socr.routing.gemma_crops import render_gemma_crops
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
    """Adapter: ``qwen-ocr process`` on layout-aware 300-dpi crops."""

    @property
    def name(self) -> str:
        return "gemma"

    @property
    def cli_command(self) -> str:
        return "qwen-ocr"

    def resolved_model_version(self, config: PipelineConfig) -> str:
        return (config.gemma_cloud_model or PROFILE_GEMMA_CLOUD.model).strip()

    def is_available(self) -> bool:
        """CLI presence only — cloud model probing is gated in agentic setup."""
        try:
            result = subprocess.run(
                [self.cli_command, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def is_available_for_config(self, config: PipelineConfig) -> tuple[bool, str]:
        """Probe ``config.gemma_cloud_model`` when the rung is explicitly enabled."""
        if not self.is_available():
            return False, "qwen-ocr CLI not available"
        model = self.resolved_model_version(config)
        return gemma_cloud_model_available(model)

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
        """OCR each page from layout-aware crops; one failed crop fails the page."""
        if not config.enable_gemma_cloud and not config.gemma_cloud_model_pinned:
            return [
                PageOutput(
                    page_num=n,
                    status=PageStatus.ERROR,
                    engine=self.name,
                    failure_mode=FailureMode.CLI_ERROR,
                    error="gemma cloud rung disabled (use --enable-gemma-cloud)",
                    audit_passed=False,
                )
                for n in page_nums
            ]

        crop_dpi = getattr(config, "gemma_crop_dpi", 300) or 300
        max_tokens = getattr(config, "gemma_max_output_tokens", 4096) or 4096
        outputs: list[PageOutput] = []

        for page_num in page_nums:
            page_start = time.time()
            with tempfile.TemporaryDirectory() as tmpdir:
                crop_dir = Path(tmpdir) / "crops"
                try:
                    stems, full_page_unverified = render_gemma_crops(
                        pdf_path, page_num, crop_dir, dpi=crop_dpi
                    )
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

                expected = len(stems)
                half_texts: list[str] = []
                inp_sum = out_sum = None
                per_half_timeout = None
                if subprocess_timeout is not None and expected:
                    per_half_timeout = max(30.0, subprocess_timeout / expected)

                page_failed = False
                fail_err = ""
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
                        page_failed = True
                        fail_err = err
                        break
                    if not text.strip():
                        page_failed = True
                        fail_err = f"gemma crop {_stem}: empty OCR output"
                        break
                    half_texts.append(text.strip())

                if page_failed or len(half_texts) != expected:
                    outputs.append(
                        PageOutput(
                            page_num=page_num,
                            status=PageStatus.ERROR,
                            engine=self.name,
                            failure_mode=FailureMode.CLI_ERROR,
                            error=fail_err or "gemma crop OCR incomplete",
                            audit_passed=False,
                            llm_input_tokens=inp_sum,
                            llm_output_tokens=out_sum,
                            processing_time=time.time() - page_start,
                        )
                    )
                    continue

                merged = "\n\n".join(half_texts)
                guarded = apply_gemma_output_guards(
                    merged,
                    page_num=page_num,
                    max_output_tokens=max_tokens,
                    full_page_unverified=full_page_unverified,
                )
                guarded.processing_time = time.time() - page_start
                guarded.llm_input_tokens = inp_sum
                guarded.llm_output_tokens = out_sum
                outputs.append(guarded)

        return outputs
