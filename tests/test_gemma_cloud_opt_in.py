"""Gemma cloud egress is opt-in; model pin alone must not probe or process."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from socr.core.config import EngineType, PipelineConfig
from socr.engines.gemma_cloud import GemmaCloudEngine
from socr.pipeline.orchestrator import UnifiedPipeline


def test_gemma_cloud_model_pin_without_enable_skips_ladder_probe() -> None:
    config = PipelineConfig(
        agentic=True,
        enable_gemma_cloud=False,
        gemma_cloud_model_pinned=True,
        gemma_cloud_model="gemma3:27b:cloud",
    )
    pipe = UnifiedPipeline(config)
    with patch("socr.engines.gemma_cloud.GemmaCloudEngine.is_available_for_config") as probe:
        available, _, _ = pipe._probe_engines_for_agentic()
        probe.assert_not_called()
    assert EngineType.GEMMA not in {p.engine for p in available}


def test_gemma_cloud_model_pin_without_enable_rejects_process_pages() -> None:
    engine = GemmaCloudEngine()
    config = PipelineConfig(
        enable_gemma_cloud=False,
        gemma_cloud_model_pinned=True,
        gemma_cloud_model="gemma3:27b:cloud",
    )
    outputs = engine.process_pages(Path("/dev/null"), [1], config)
    assert len(outputs) == 1
    assert "disabled" in (outputs[0].error or "").lower()


def test_is_available_for_config_requires_enable_gemma_cloud() -> None:
    engine = GemmaCloudEngine()
    config = PipelineConfig(
        enable_gemma_cloud=False,
        gemma_cloud_model_pinned=True,
        gemma_cloud_model="gemma3:27b:cloud",
    )
    ok, reason = engine.is_available_for_config(config)
    assert ok is False
    assert "disabled" in reason.lower()
