"""Qwen's resolved model must reach the run fingerprint (#231).

``config.qwen_model`` is a sentinel: empty means "not user-pinned, let the
resolver pick for the backend". The base ``resolved_model_version`` reads that
field directly, so before this fix a default run fingerprinted no model at all
and a vllm/sglang/api run fingerprinted the sentinel instead of the model the
OpenAI-compatible server was actually serving. Two runs on different backends
could share a fingerprint and resume across each other's pages.

Hermetic by construction: ``resolve_qwen_intent`` and ``get_engine`` are pure
config/registry lookups with no Ollama or network probe, so these pass on CI
where no provider exists.
"""

from __future__ import annotations

import pytest

import socr.pipeline.orchestrator as orch
from socr.core.config import EngineType, PipelineConfig
from socr.engines.qwen import OLLAMA_MODEL, QwenEngine

# Backends that route through the OpenAI-compatible server rather than local
# Ollama, and therefore must fingerprint ``qwen_vllm_model``.
SERVED_BACKENDS = ("vllm", "sglang", "api")


@pytest.fixture(autouse=True)
def _isolate_backend_resolution(monkeypatch):
    """Decide backend resolution here, never inherit it from the shell.

    GH-521. `qwen_backend` defaults to `auto`, and `auto` means vLLM whenever
    `VLLM_BASE_URL` is exported -- which is the documented HPC deployment. So on
    such a machine these tests failed while production was behaving exactly as
    designed: they asserted the Ollama answer and got the correct vLLM one.

    Clearing the variable makes the DEFAULT deterministic. It does not make the
    exported state untested: the parametrised cases below set it back
    explicitly, so both answers are pinned rather than one of them being
    whatever the shell happened to hold.
    """
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)


@pytest.fixture(autouse=True)
def _pin_source_digest(monkeypatch: pytest.MonkeyPatch):
    """Hold socr's own source identity constant so only the knob under test varies."""
    orch._SOURCE_DIGEST_CACHE = None
    monkeypatch.setattr(orch, "_socr_source_digest", lambda: "pinned-digest")
    yield
    orch._SOURCE_DIGEST_CACHE = None


def _config(**overrides: object) -> PipelineConfig:
    config = PipelineConfig()
    for key, value in overrides.items():
        assert hasattr(config, key), f"PipelineConfig has no field {key!r}"
        setattr(config, key, value)
    return config


def _fingerprint(**overrides: object) -> str:
    """Fingerprint with Qwen pinned as primary, so its determinants are consulted.

    ``primary_engine``/``local_engine`` default to AUTO, which resolves to
    GEMINI when no local provider is present (the CI case). Pinning QWEN keeps
    the assertion about Qwen's determinants and not about the AUTO ladder.
    """
    overrides.setdefault("primary_engine", EngineType.QWEN)
    overrides.setdefault("local_engine", EngineType.QWEN)
    overrides.setdefault("fallback_chain", [])
    return orch.UnifiedPipeline(_config(**overrides))._run_fingerprint()


# --- engine level: the resolved model is what the CLI is actually given ---


@pytest.mark.parametrize("backend", SERVED_BACKENDS)
def test_served_backend_reports_the_served_model(backend: str) -> None:
    engine = QwenEngine()
    resolved = engine.resolved_model_version(_config(qwen_backend=backend, qwen_vllm_model="Org/M"))
    assert resolved == "Org/M"


@pytest.mark.parametrize("backend", SERVED_BACKENDS)
def test_served_backend_model_swap_changes_the_reported_model(backend: str) -> None:
    engine = QwenEngine()
    first = engine.resolved_model_version(_config(qwen_backend=backend, qwen_vllm_model="Org/A"))
    second = engine.resolved_model_version(_config(qwen_backend=backend, qwen_vllm_model="Org/B"))
    assert first != second


@pytest.mark.parametrize("backend", ("auto", "ollama"))
def test_local_backend_reports_the_instruct_moe(backend: str) -> None:
    """Local runs use the validated instruct build regardless of the served model."""
    engine = QwenEngine()
    resolved = engine.resolved_model_version(
        _config(qwen_backend=backend, qwen_vllm_model="Org/irrelevant")
    )
    assert resolved == OLLAMA_MODEL


def test_explicit_pin_is_honoured_verbatim() -> None:
    """A deliberate --qwen-model pin must reach the fingerprint unchanged."""
    engine = QwenEngine()
    resolved = engine.resolved_model_version(
        _config(qwen_backend="ollama", qwen_model="qwen3.5:cloud", qwen_model_pinned=True)
    )
    assert resolved == "qwen3.5:cloud"


def test_pinning_a_model_changes_the_reported_model() -> None:
    """Toggling the pin flag alone rewrites which model runs, so it must show up."""
    engine = QwenEngine()
    unpinned = engine.resolved_model_version(
        _config(qwen_backend="ollama", qwen_model="qwen3.5:cloud", qwen_model_pinned=False)
    )
    pinned = engine.resolved_model_version(
        _config(qwen_backend="ollama", qwen_model="qwen3.5:cloud", qwen_model_pinned=True)
    )
    assert unpinned == OLLAMA_MODEL
    assert pinned == "qwen3.5:cloud"


# --- fingerprint level: the run fingerprint actually moves ---


@pytest.mark.parametrize("backend", SERVED_BACKENDS)
def test_served_model_swap_invalidates_the_fingerprint(backend: str) -> None:
    assert _fingerprint(qwen_backend=backend, qwen_vllm_model="Org/A") != _fingerprint(
        qwen_backend=backend, qwen_vllm_model="Org/B"
    )


def test_served_model_is_ignored_on_a_local_backend() -> None:
    """No needless reprocess: the served model is unread when Ollama runs the job."""
    assert _fingerprint(qwen_backend="ollama", qwen_vllm_model="Org/A") == _fingerprint(
        qwen_backend="ollama", qwen_vllm_model="Org/B"
    )


def test_local_and_served_backends_do_not_share_a_fingerprint() -> None:
    """The issue's acceptance criterion, stated directly."""
    assert _fingerprint(qwen_backend="ollama") != _fingerprint(qwen_backend="vllm")
