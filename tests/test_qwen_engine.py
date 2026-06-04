"""Qwen engine integration: registration, routing priority, ladder, CLI command."""

from __future__ import annotations

from socr.core.config import ENGINE_PRIORITY, EngineType, PipelineConfig
from socr.core.providers import DEFAULT_PROVIDERS, provider_ladder
from socr.engines.qwen import QwenEngine
from socr.engines.registry import _LOCAL_ENGINE_ORDER, get_engine


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


def test_qwen_model_override_passed_to_cli():
    cfg = PipelineConfig(qwen_model="qwen3.5:27b")
    cmd = QwenEngine()._build_command("/tmp/in", "/tmp/out", cfg)
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "qwen3.5:27b"


def test_qwen_model_omitted_when_unset():
    cmd = QwenEngine()._build_command("/tmp/in", "/tmp/out", PipelineConfig())
    assert "--model" not in cmd  # default ("") -> CLI picks its own default
