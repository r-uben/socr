"""Gemma cloud availability probe wiring."""

from __future__ import annotations

from unittest.mock import patch

from socr.core.config import PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline


def test_gemma_probe_failure_surfaces_as_gemma_not_qwen() -> None:
    pipe = UnifiedPipeline(PipelineConfig(agentic=True, enable_gemma_cloud=True, quiet=True))
    state = DocumentState(
        handle=DocumentHandle(path=__import__("pathlib").Path("x.pdf"), page_count=1)
    )
    pipe._gemma_cloud_pin_unavailable = "gemma4:31b-cloud: model not found"
    pipe._surface_gemma_cloud_pin_unavailable(state)
    assert pipe._gemma_cloud_pin_unavailable == ""
    kinds = [e.kind for e in state.events]
    assert "gemma_cloud_unavailable" in kinds
    assert "qwen_cloud_pin_unavailable" not in kinds


def test_gemma_probe_uses_short_timeout() -> None:
    from socr.engines import gemma_cloud as mod

    assert mod.GEMMA_CLOUD_PROBE_TIMEOUT_SEC <= 10.0
    with patch("socr.engines.gemma_cloud.probe_model_generation") as probe:
        probe.return_value = (True, "")
        mod.gemma_cloud_model_available("gemma4:31b-cloud")
        _host, _model, timeout = probe.call_args[0]
        assert timeout == mod.GEMMA_CLOUD_PROBE_TIMEOUT_SEC
