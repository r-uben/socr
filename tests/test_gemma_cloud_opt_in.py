"""Gemma cloud ladder default-on; opt-out and availability probes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import is_cloud_model
from socr.engines.gemma_cloud import GemmaCloudEngine, gemma_cloud_model_available
from socr.pipeline.orchestrator import UnifiedPipeline


def test_gemma_cloud_enabled_by_default_in_config() -> None:
    assert PipelineConfig().enable_gemma_cloud is True


def test_no_gemma_cloud_skips_ladder_probe() -> None:
    config = PipelineConfig(
        agentic=True,
        enable_gemma_cloud=False,
        gemma_cloud_model_pinned=True,
        gemma_cloud_model="gemma3:27b:cloud",
    )
    pipe = UnifiedPipeline(config)
    with patch("socr.engines.gemma_cloud.GemmaCloudEngine.is_available_for_config") as probe:
        available, _, _, _ = pipe._probe_engines_for_agentic()
        probe.assert_not_called()
    assert EngineType.GEMMA not in {p.engine for p in available}


def test_no_gemma_cloud_rejects_process_pages() -> None:
    engine = GemmaCloudEngine()
    config = PipelineConfig(
        enable_gemma_cloud=False,
        gemma_cloud_model_pinned=True,
        gemma_cloud_model="gemma3:27b:cloud",
    )
    outputs = engine.process_pages(Path("/dev/null"), [1], config)
    assert len(outputs) == 1
    assert "disabled" in (outputs[0].error or "").lower()


def test_default_gemma_cloud_model_is_cloud_tag() -> None:
    config = PipelineConfig()
    assert is_cloud_model(config.gemma_cloud_model)


def test_local_gemma_tag_rejected_when_cloud_enabled() -> None:
    ok, reason = gemma_cloud_model_available("gemma3:27b")
    assert ok is False
    assert "cloud" in reason.lower()


def test_is_available_for_config_honors_disable_flag() -> None:
    engine = GemmaCloudEngine()
    config = PipelineConfig(
        enable_gemma_cloud=False,
        gemma_cloud_model_pinned=True,
        gemma_cloud_model="gemma3:27b:cloud",
    )
    ok, reason = engine.is_available_for_config(config)
    assert ok is False
    assert "disabled" in reason.lower()


def test_gemma_probe_when_default_on_and_model_missing() -> None:
    config = PipelineConfig(agentic=True, enable_gemma_cloud=True)
    pipe = UnifiedPipeline(config)
    with patch(
        "socr.engines.gemma_cloud.GemmaCloudEngine.is_available_for_config",
        return_value=(False, "gemma4:31b-cloud: model not found"),
    ):
        available, unservable, _qwen_pin, gemma_pin = pipe._probe_engines_for_agentic()
    assert EngineType.GEMMA not in {p.engine for p in available}
    assert gemma_pin
    assert "gemma" in gemma_pin.lower() or "model" in gemma_pin.lower()
