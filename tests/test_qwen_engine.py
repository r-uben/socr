"""Qwen engine integration: registration, routing priority, ladder, CLI command."""

from __future__ import annotations

import pytest

from socr.core.config import ENGINE_PRIORITY, EngineType, PipelineConfig
from socr.core.providers import DEFAULT_PROVIDERS, provider_ladder
from socr.engines.qwen import OLLAMA_MODEL, QwenEngine, resolve_qwen_intent
from socr.engines.registry import _LOCAL_ENGINE_ORDER, get_engine


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


def test_qwen_registered():
    engine = get_engine(EngineType.QWEN)
    assert isinstance(engine, QwenEngine)
    assert engine.name == "qwen" and engine.cli_command == "qwen-ocr"


def test_qwen_leads_local_tier():
    assert _LOCAL_ENGINE_ORDER[0] == EngineType.QWEN
    assert ENGINE_PRIORITY[EngineType.QWEN] == 0


def test_qwen_is_free_and_first_among_free_providers():
    assert EngineType.QWEN in DEFAULT_PROVIDERS
    assert DEFAULT_PROVIDERS[EngineType.QWEN].is_free
    ladder = provider_ladder({EngineType.QWEN, EngineType.GLM, EngineType.GEMINI})
    # Qwen free + priority 0 -> tried before GLM (free) and Gemini (paid).
    assert [p.engine for p in ladder] == [
        EngineType.QWEN,
        EngineType.GLM,
        EngineType.GEMINI,
    ]


def test_qwen_build_command_matches_cli_contract():
    cfg = PipelineConfig(qwen_backend="auto", render_dpi=300)
    cmd = QwenEngine()._build_command("/tmp/in", "/tmp/out", cfg)
    assert cmd[:2] == ["qwen-ocr", "process"]
    assert "--backend" in cmd and cmd[cmd.index("--backend") + 1] == "auto"
    assert "--dpi" in cmd and cmd[cmd.index("--dpi") + 1] == "300"


def test_qwen_backend_default_is_auto():
    assert PipelineConfig().qwen_backend == "auto"


# ---------------------------------------------------------------------------
# resolve_qwen_intent — covers all acceptance-criteria cases
# ---------------------------------------------------------------------------


def test_resolve_auto_backend_uses_local_instruct_model():
    """Auto backend without a pin must resolve to the validated local instruct MoE.

    This is the core fix for GH-51: a cloud model string must never be passed to
    the local qwen backend via the auto path.
    """
    cfg = PipelineConfig(qwen_backend="auto")
    backend, model = resolve_qwen_intent(cfg)
    assert backend == "auto"
    assert model == OLLAMA_MODEL
    assert model != "qwen3.5:cloud"


def test_resolve_ollama_backend_uses_local_instruct_model():
    """Explicit ollama backend without a pin also resolves to the local instruct MoE."""
    cfg = PipelineConfig(qwen_backend="ollama")
    backend, model = resolve_qwen_intent(cfg)
    assert backend == "ollama"
    assert model == OLLAMA_MODEL


def test_resolve_explicit_model_pin_passes_through_unchanged():
    """Explicit --qwen-model pin is honoured verbatim regardless of backend."""
    cfg = PipelineConfig(qwen_backend="auto", qwen_model="qwen3.5:27b", qwen_model_pinned=True)
    backend, model = resolve_qwen_intent(cfg)
    assert backend == "auto"
    assert model == "qwen3.5:27b"


def test_resolve_cloud_model_pin_passes_through_unchanged():
    """Explicitly pinned qwen3.5:cloud must reach qwen-ocr unchanged."""
    cfg = PipelineConfig(qwen_backend="auto", qwen_model="qwen3.5:cloud", qwen_model_pinned=True)
    _, model = resolve_qwen_intent(cfg)
    assert model == "qwen3.5:cloud"


def test_resolve_blank_model_on_local_backend_uses_local_instruct():
    """Blank (unset) model on a local backend resolves to the local instruct MoE."""
    cfg = PipelineConfig(qwen_backend="ollama", qwen_model="")
    _, model = resolve_qwen_intent(cfg)
    assert model == OLLAMA_MODEL


def test_resolve_vllm_backend_requests_served_model():
    """Non-local backend (vllm) requests config.qwen_vllm_model (the served model name)."""
    cfg = PipelineConfig(qwen_backend="vllm", qwen_vllm_model="Qwen/Qwen3-VL-7B")
    _, model = resolve_qwen_intent(cfg)
    assert model == "Qwen/Qwen3-VL-7B"


def test_resolve_nonlocal_backend_falls_back_to_qwen_model_when_vllm_model_blank():
    """When qwen_vllm_model is blank, a non-local backend falls back to qwen_model
    (empty here, letting the CLI pick its own default)."""
    cfg = PipelineConfig(qwen_backend="api", qwen_vllm_model="", qwen_model="")
    _, model = resolve_qwen_intent(cfg)
    assert model == ""


def test_resolve_api_backend_also_requests_served_model():
    """The api backend, like vllm, requests config.qwen_vllm_model (both are
    OpenAI-compatible servers that 404 on a mismatched model name)."""
    cfg = PipelineConfig(qwen_backend="api", qwen_vllm_model="Qwen/Qwen3-VL-30B-A3B-Instruct")
    _, model = resolve_qwen_intent(cfg)
    assert model == "Qwen/Qwen3-VL-30B-A3B-Instruct"


# ---------------------------------------------------------------------------
# _build_command — integration with resolver
# ---------------------------------------------------------------------------


def test_build_command_auto_backend_no_pin_uses_local_instruct():
    """_build_command must not pass a cloud model string on the auto/local path."""
    cfg = PipelineConfig(qwen_backend="auto")
    cmd = QwenEngine()._build_command("/tmp/in", "/tmp/out", cfg)
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == OLLAMA_MODEL


def test_build_command_explicit_model_pin_reaches_cli_unchanged():
    """Explicit --qwen-model reaches qwen-ocr unchanged."""
    cfg = PipelineConfig(qwen_model="qwen3.5:27b", qwen_model_pinned=True)
    cmd = QwenEngine()._build_command("/tmp/in", "/tmp/out", cfg)
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "qwen3.5:27b"


def test_build_command_blank_model_omits_model_flag_on_nonlocal_backend():
    """Fully blank model on a non-local backend (api) produces no --model flag,
    restoring the CLI's own default. Requires blanking qwen_vllm_model too, since it
    otherwise supplies the served model name."""
    cfg = PipelineConfig(qwen_backend="api", qwen_vllm_model="", qwen_model="")
    cmd = QwenEngine()._build_command("/tmp/in", "/tmp/out", cfg)
    assert "--model" not in cmd


def test_qwen_model_omitted_when_blanked_and_pinned():
    """Pinned empty string (rare) omits --model flag, restoring the CLI's own default."""
    cfg = PipelineConfig(qwen_model="", qwen_model_pinned=True)
    cmd = QwenEngine()._build_command("/tmp/in", "/tmp/out", cfg)
    assert "--model" not in cmd


# ---------------------------------------------------------------------------
# Agentic profile — local instruct is used by PROFILE_QWEN_LOCAL
# ---------------------------------------------------------------------------


def test_agentic_local_profile_pins_instruct_model():
    """The agentic ladder's qwen-local-instruct profile carries the validated MoE."""
    from socr.core.providers import PROFILE_QWEN_LOCAL

    assert PROFILE_QWEN_LOCAL.id == "qwen-local-instruct"
    assert PROFILE_QWEN_LOCAL.backend == "ollama"
    assert PROFILE_QWEN_LOCAL.model == OLLAMA_MODEL


@pytest.mark.parametrize("exported", [None, "http://gpu-node:8000/v1"])
def test_auto_resolves_by_the_environment_and_both_answers_are_pinned(monkeypatch, exported):
    """GH-521: clearing the variable must not leave the exported state untested.

    The autouse fixture makes the DEFAULT deterministic. Without this, the HPC
    answer would be the one nobody asserts -- which is how five tests came to
    fail on a machine where production was behaving correctly.

    Both arms assert a real resolution, so a change that collapsed `auto` to one
    backend everywhere would redden one of them rather than silently pass.
    """
    if exported is None:
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("VLLM_BASE_URL", exported)

    backend, model = resolve_qwen_intent(PipelineConfig(qwen_backend="auto"))

    if exported is None:
        assert backend == "auto", "with no server exported, auto must stay on the local path"
        assert model == OLLAMA_MODEL
    else:
        assert backend == "vllm", (
            "with VLLM_BASE_URL exported, auto must resolve to vLLM -- this is "
            "the documented HPC deployment, not an edge case"
        )
        assert model != OLLAMA_MODEL, (
            "the vLLM backend was handed the Ollama model tag; the two servers "
            "name the same model differently"
        )


def test_an_explicit_backend_ignores_the_environment(monkeypatch):
    """Control: the environment decides only what `auto` means.

    Without this, a resolver that read the variable unconditionally would
    satisfy the parametrised test above while overriding an explicit choice.
    """
    for exported in (None, "http://gpu-node:8000/v1"):
        if exported is None:
            monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        else:
            monkeypatch.setenv("VLLM_BASE_URL", exported)

        backend, _model = resolve_qwen_intent(PipelineConfig(qwen_backend="ollama"))
        assert backend == "ollama", (
            f"an explicit ollama backend was overridden by VLLM_BASE_URL={exported!r}"
        )
