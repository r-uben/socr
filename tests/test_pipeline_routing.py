import pytest

pytest.importorskip("rich")

from socr.core.config import EngineType, PipelineConfig
from socr.pipeline.router import EngineRouter


class _StubEngine:
    """Minimal engine stub for availability checks."""

    def __init__(self, available: bool) -> None:
        self._available = available
        self.name = "stub"

    def is_available(self) -> bool:
        return self._available


def _make_router(config: PipelineConfig) -> EngineRouter:
    engines = {
        EngineType.NOUGAT: _StubEngine(False),
        EngineType.DEEPSEEK: _StubEngine(False),
        EngineType.MISTRAL: _StubEngine(False),
        EngineType.GEMINI: _StubEngine(False),
        EngineType.MARKER: _StubEngine(False),
    }
    return EngineRouter(config, engines)


def test_primary_selected_when_available() -> None:
    config = PipelineConfig(primary_engine=EngineType.MISTRAL)
    router = _make_router(config)
    router.engines[EngineType.MISTRAL] = _StubEngine(True)

    choice = router.select_primary()
    assert choice == EngineType.MISTRAL


def test_primary_falls_through_when_unavailable() -> None:
    config = PipelineConfig(primary_engine=EngineType.MISTRAL)
    router = _make_router(config)
    # Mistral unavailable, Nougat available
    router.engines[EngineType.NOUGAT] = _StubEngine(True)

    choice = router.select_primary()
    assert choice == EngineType.NOUGAT


def test_fallback_different_from_primary() -> None:
    config = PipelineConfig(fallback_engine=EngineType.DEEPSEEK)
    router = _make_router(config)
    router.engines[EngineType.DEEPSEEK] = _StubEngine(True)

    choice = router.select_fallback(primary=EngineType.NOUGAT)
    assert choice == EngineType.DEEPSEEK


def test_fallback_skips_unavailable() -> None:
    config = PipelineConfig(fallback_engine=EngineType.GEMINI)
    router = _make_router(config)
    # Gemini unavailable, Mistral available
    router.engines[EngineType.MISTRAL] = _StubEngine(True)

    choice = router.select_fallback(primary=EngineType.NOUGAT)
    assert choice == EngineType.MISTRAL
