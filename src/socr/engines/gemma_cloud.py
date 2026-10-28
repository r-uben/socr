"""Gemma vision OCR via Ollama Cloud — a dedicated agentic ladder rung.

Uses the same ``qwen-ocr`` CLI transport as Qwen, but always pins the Gemma
cloud model so it is distinct from ``--qwen-model …:cloud`` (issue #1253/#1254).

Guards (300 dpi column crops, output token cap, repetition / image-URL loop
rejection, equation-unverified flags) are **not** duplicated here yet: they live
in the shared agentic escalation / crop-repair path for Qwen today (#1254).
This rung inherits the same per-provider timeout as Qwen in ``route_page`` and
records token usage via ``BaseEngine.process_pages`` when the CLI logs counts.
"""

from __future__ import annotations

import os
from pathlib import Path

from socr.core.config import PipelineConfig
from socr.core.ollama_utils import DEFAULT_PROBE_TIMEOUT_SEC, probe_model_generation
from socr.core.providers import PROFILE_GEMMA_CLOUD, is_cloud_model
from socr.engines.base import BaseEngine
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
    """Adapter: ``qwen-ocr process`` with ``--backend ollama`` and a Gemma model pin."""

    @property
    def name(self) -> str:
        return "gemma"

    @property
    def cli_command(self) -> str:
        return "qwen-ocr"

    def resolved_model_version(self, config: PipelineConfig) -> str:
        return (config.gemma_cloud_model or PROFILE_GEMMA_CLOUD.model).strip()

    def is_available(self) -> bool:
        if not (os.environ.get("OLLAMA_API_KEY") or os.environ.get("OLLAMA_HOST")):
            # Local daemon may still serve :cloud tags when signed in.
            pass
        try:
            result = __import__("subprocess").run(
                [self.cli_command, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
        except (__import__("subprocess").SubprocessError, FileNotFoundError):
            return False
        model = PROFILE_GEMMA_CLOUD.model
        ok, _ = gemma_cloud_model_available(model)
        return ok

    def _build_command(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> list[str]:
        model = self.resolved_model_version(config)
        cmd = [
            self.cli_command,
            "process",
            str(pdf_path),
            "-o",
            str(output_dir),
            "--backend",
            "ollama",
            "--model",
            model,
            "--dpi",
            str(config.render_dpi),
        ]
        if config.quiet:
            cmd.append("-q")
        if config.verbose:
            cmd.append("-v")
        if config.reprocess:
            cmd.append("--reprocess")
        return cmd
