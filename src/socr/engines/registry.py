"""Engine registry — maps EngineType to engine instances."""

import logging

from socr.core.config import AUTO_ENGINE_ORDER, EngineType
from socr.engines.base import BaseEngine
from socr.engines.deepseek import DeepSeekEngine
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
    EngineType.QWEN: QwenEngine,
}


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


# Local-only engines for tiered routing (no API keys needed).
# Qwen-VL leads: best open OCR quality (socOCRbench ~0.47-0.58), though slower per page
# than GLM. Quality-first ordering — the tiered router only sends *easy* pages here anyway.
_LOCAL_ENGINE_ORDER: list[EngineType] = [
    EngineType.QWEN,  # Qwen3-VL via Ollama; best open quality, ~slower
    EngineType.GLM,  # 0.9B, fast, ~10s/page
    EngineType.DEEPSEEK,  # Larger, needs Ollama
    EngineType.NOUGAT,  # Academic papers only
    EngineType.MARKER,  # Layout-aware, CPU-friendly
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
