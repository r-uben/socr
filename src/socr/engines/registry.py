"""Engine registry — maps EngineType to engine instances."""

import logging

from socr.core.config import AUTO_ENGINE_ORDER, EngineType
from socr.engines.base import BaseEngine
from socr.engines.deepseek import DeepSeekEngine
from socr.engines.gemma_cloud import GemmaCloudEngine
from socr.engines.gemini import GeminiEngine
from socr.engines.glm import GLMEngine
from socr.engines.marker import MarkerEngine
from socr.engines.mistral import MistralEngine
from socr.engines.nougat import NougatEngine
from socr.engines.qwen import QwenEngine

logger = logging.getLogger(__name__)

_ENGINES: dict[EngineType, type[BaseEngine]] = {
    EngineType.GLM: GLMEngine,
    EngineType.NOUGAT: NougatEngine,
    EngineType.DEEPSEEK: DeepSeekEngine,
    EngineType.MISTRAL: MistralEngine,
    EngineType.GEMINI: GeminiEngine,
    EngineType.MARKER: MarkerEngine,
    EngineType.GEMMA: GemmaCloudEngine,
    EngineType.QWEN: QwenEngine,
}


def has_cli_engine(engine_type: EngineType) -> bool:
    """True when ``get_engine`` can build an engine for *engine_type*.

    #842: ``EngineType.VLLM`` and ``EngineType.DEEPSEEK_VLLM`` have profiles in
    ``DEFAULT_PROVIDERS`` but no entry here -- ``hpc_pipeline.py`` constructs
    those engines directly. Asking this is a STRUCTURAL question with a fixed
    answer, and callers must not learn it by catching the ``ValueError`` below,
    which is indistinguishable from a provider that is merely down.
    """
    return engine_type in _ENGINES


def get_engine(engine_type: EngineType) -> BaseEngine:
    """Get an engine instance by type."""
    cls = _ENGINES.get(engine_type)
    if cls is None:
        raise ValueError(f"No CLI engine for {engine_type.value}")
    return cls()


def resolve_auto_engine() -> EngineType:
    """Probe CLI engines in priority order and return the first available one.

    Returns the first engine whose CLI is installed and whose dependencies
    (API keys, Ollama models, etc.) are satisfied. Falls back to GEMINI
    if nothing is available, letting downstream code produce a clear error.
    """
    for engine_type in AUTO_ENGINE_ORDER:
        try:
            cli_engine = _ENGINES.get(engine_type)
            if cli_engine is None:
                continue
            instance = cli_engine()
            if instance.is_available():
                logger.info(f"Auto-selected engine: {engine_type.value}")
                return engine_type
        except Exception:
            continue

    logger.warning("No engines available; falling back to gemini (will likely fail)")
    return EngineType.GEMINI


_DEFAULT_RESOLVE_AUTO_ENGINE = resolve_auto_engine


def resolve_auto_engine_for_config(config) -> EngineType:
    """Policy-aware auto engine selection respecting strict-local and gemma opt-out.

    Preserves test monkeypatches targeting ``socr.engines.registry.resolve_auto_engine``:
    when callers patched that zero-arg symbol, delegate directly so hermetic tests keep
    their stub without probing the host.
    """
    if resolve_auto_engine is not _DEFAULT_RESOLVE_AUTO_ENGINE:
        return resolve_auto_engine()

    from socr.core.providers import (
        DEFAULT_PROVIDERS,
        TIER_LOCAL,
        zero_cap_pinned_forbids_cloud,
    )

    forbid_cloud = bool(
        getattr(config, "strict_local", False) or zero_cap_pinned_forbids_cloud(config)
    )
    allow_gemma = bool(getattr(config, "enable_gemma_cloud", True))

    for engine_type in AUTO_ENGINE_ORDER:
        if engine_type is EngineType.GEMMA and not allow_gemma:
            continue
        if forbid_cloud:
            prof = DEFAULT_PROVIDERS.get(engine_type)
            if prof is not None and prof.tier != TIER_LOCAL:
                continue
        try:
            cli_engine = _ENGINES.get(engine_type)
            if cli_engine is None:
                continue
            instance = cli_engine()
            if instance.is_available():
                logger.info(f"Auto-selected engine: {engine_type.value}")
                return engine_type
        except Exception:
            continue

    if forbid_cloud:
        logger.warning("No local engines available under strict-local; falling back to qwen")
        return EngineType.QWEN
    logger.warning("No engines available; falling back to gemini (will likely fail)")
    return EngineType.GEMINI


# Local-only engines for tiered routing (no API keys needed).
# Qwen-VL leads: best open OCR quality (socOCRbench ~0.47-0.58), though slower per page
# than GLM. Quality-first ordering — the tiered router only sends *easy* pages here anyway.
# DeepSeek-OCR is dropped (~0.085 socOCRbench, near dead last); use --primary deepseek
# to invoke it explicitly. Nougat is dropped too (GH-637: produced nothing on any table
# page in the 2026-09-06 ECB defect census, matching D1's auto_eligible=False in
# core/providers.py); use --primary nougat to invoke it explicitly.
_LOCAL_ENGINE_ORDER: list[EngineType] = [
    EngineType.QWEN,  # Qwen3-VL via Ollama; best open quality, ~slower
    EngineType.MARKER,  # Layout-aware, CPU-friendly
    EngineType.GLM,  # 0.9B, fast, ~10s/page
]


def resolve_local_engine() -> EngineType | None:
    """Probe local engines in priority order.

    Returns the first available local engine, or None if no local
    engine is available (tiered routing will be skipped).
    """
    for engine_type in _LOCAL_ENGINE_ORDER:
        try:
            cli_engine = _ENGINES.get(engine_type)
            if cli_engine is None:
                continue
            instance = cli_engine()
            if instance.is_available():
                logger.info(f"Auto-selected local engine: {engine_type.value}")
                return engine_type
        except Exception:
            continue

    logger.info("No local engines available; tiered routing disabled")
    return None
