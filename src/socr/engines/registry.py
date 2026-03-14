"""Engine registry — maps EngineType to engine instances."""

from socr.core.config import EngineType
from socr.engines.base import BaseEngine
from socr.engines.deepseek import DeepSeekEngine
from socr.engines.gemini import GeminiEngine
from socr.engines.marker import MarkerEngine
from socr.engines.mistral import MistralEngine
from socr.engines.nougat import NougatEngine

_ENGINES: dict[EngineType, type[BaseEngine]] = {
    EngineType.NOUGAT: NougatEngine,
    EngineType.DEEPSEEK: DeepSeekEngine,
    EngineType.MISTRAL: MistralEngine,
    EngineType.GEMINI: GeminiEngine,
    EngineType.MARKER: MarkerEngine,
}


def get_engine(engine_type: EngineType) -> BaseEngine:
    """Get an engine instance by type."""
    cls = _ENGINES.get(engine_type)
    if cls is None:
        raise ValueError(f"No CLI engine for {engine_type.value}")
    return cls()
