"""GH-842: engine types with no CLI engine are decided structurally, and reported
when -- and only when -- the operator asked for them.

Measured on main before this: ``PipelineConfig().enabled_engines`` is every
``EngineType``, so EVERY default run put ``VLLM`` and ``DEEPSEEK_VLLM`` through
``_available_engines_for_agentic``; ``get_engine`` raised ``ValueError`` for both,
and the reachability ``except`` swallowed it exactly as it swallows "the daemon is
down". No content was lost -- those rungs could never serve -- but an operator who
listed one on purpose got a ladder that silently did less than the config said.

Hermetic: ``get_engine`` is patched, so nothing probes a real provider, and the
QWEN cloud probe is patched off.
"""

from __future__ import annotations

import logging

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.engines import registry
from socr.engines.registry import has_cli_engine
from socr.pipeline import orchestrator as orch
from socr.pipeline.orchestrator import UnifiedPipeline


def test_the_structural_answer_is_fixed():
    assert has_cli_engine(EngineType.QWEN)
    assert not has_cli_engine(EngineType.VLLM)
    assert not has_cli_engine(EngineType.DEEPSEEK_VLLM)


class _Up:
    def is_available(self):
        return True


@pytest.fixture
def hermetic(monkeypatch):
    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _Up())
    import socr.engines.qwen as qwen

    monkeypatch.setattr(qwen, "cloud_model_available", lambda: False)


def _pipeline(**cfg):
    p = object.__new__(UnifiedPipeline)
    p.config = PipelineConfig(quiet=True, **cfg)
    return p


def _warnings(caplog):
    return [r for r in caplog.records if "#842" in r.getMessage()]


def test_unservable_types_never_reach_the_ladder(hermetic):
    """Even with ``get_engine`` patched to say everything is up, the structural
    check must still keep the no-engine types out -- the decision no longer
    depends on an exception being raised."""
    ids = {p.id for p in _pipeline()._available_engines_for_agentic()}
    assert "vllm" not in ids
    assert "deepseek-vllm" not in ids
    assert "qwen" in ids or any("qwen" in i for i in ids)


def test_default_config_stays_quiet(hermetic, caplog):
    """The default lists every EngineType; nobody asked for the vLLM types, so
    warning on every run would be noise (the GH-525 lesson)."""
    with caplog.at_level(logging.WARNING):
        _pipeline()._available_engines_for_agentic()
    assert _warnings(caplog) == []


def test_an_explicit_list_naming_one_is_reported_once(hermetic, caplog):
    """The difference the ticket asks for: same probe, only the config changes."""
    p = _pipeline(enabled_engines=[EngineType.QWEN, EngineType.VLLM])
    with caplog.at_level(logging.WARNING):
        p._available_engines_for_agentic()
        p._available_engines_for_agentic()  # a second probe in the same run
    found = _warnings(caplog)
    assert len(found) == 1, "once per pipeline, not once per probe"
    assert "vllm" in found[0].getMessage()


def test_an_explicit_list_without_them_is_quiet(hermetic, caplog):
    with caplog.at_level(logging.WARNING):
        _pipeline(enabled_engines=[EngineType.QWEN])._available_engines_for_agentic()
    assert _warnings(caplog) == []


def test_a_real_engine_that_is_down_is_still_silently_skipped(monkeypatch, caplog):
    """Reachability failures keep their old handling: skipped, no #842 warning,
    no crash. Only the structural case changed."""

    class _Down:
        def is_available(self):
            raise ConnectionError("daemon down")

    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _Down())
    import socr.engines.qwen as qwen

    monkeypatch.setattr(qwen, "cloud_model_available", lambda: False)
    with caplog.at_level(logging.WARNING):
        got = _pipeline(enabled_engines=[EngineType.QWEN])._available_engines_for_agentic()
    assert got == []
    assert _warnings(caplog) == []


def test_registry_and_structural_check_agree():
    """If someone registers a vLLM engine later, the check must follow the
    registry rather than a hardcoded list."""
    for et in EngineType:
        assert has_cli_engine(et) == (et in registry._ENGINES)
