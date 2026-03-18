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


# --- Primary selection (unchanged) ---

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


# --- Legacy fallback_engine backward compat ---

def test_fallback_engine_setter_wraps_in_chain() -> None:
    config = PipelineConfig()
    config.fallback_engine = EngineType.DEEPSEEK
    assert config.fallback_chain == [EngineType.DEEPSEEK]


def test_fallback_engine_getter_returns_first() -> None:
    config = PipelineConfig(fallback_chain=[EngineType.MISTRAL, EngineType.GEMINI])
    assert config.fallback_engine == EngineType.MISTRAL


def test_fallback_engine_getter_empty_chain() -> None:
    config = PipelineConfig(fallback_chain=[])
    assert config.fallback_engine is None


def test_fallback_different_from_primary() -> None:
    config = PipelineConfig(fallback_chain=[EngineType.DEEPSEEK])
    router = _make_router(config)
    router.engines[EngineType.DEEPSEEK] = _StubEngine(True)

    choice = router.select_fallback(primary=EngineType.NOUGAT)
    assert choice == EngineType.DEEPSEEK


def test_fallback_skips_unavailable() -> None:
    config = PipelineConfig(fallback_chain=[EngineType.GEMINI])
    router = _make_router(config)
    # Gemini unavailable, Mistral available
    router.engines[EngineType.MISTRAL] = _StubEngine(True)

    choice = router.select_fallback(primary=EngineType.NOUGAT)
    assert choice == EngineType.MISTRAL


# --- Fallback chain: iteration ---

def test_chain_iterates_in_order() -> None:
    """First available engine in the chain wins."""
    config = PipelineConfig(
        fallback_chain=[EngineType.MISTRAL, EngineType.GEMINI],
    )
    router = _make_router(config)
    router.engines[EngineType.MISTRAL] = _StubEngine(True)
    router.engines[EngineType.GEMINI] = _StubEngine(True)

    choice = router.select_fallback(primary=EngineType.DEEPSEEK)
    assert choice == EngineType.MISTRAL


def test_chain_skips_unavailable_engines() -> None:
    """Skip first chain entry if unavailable, pick second."""
    config = PipelineConfig(
        fallback_chain=[EngineType.MISTRAL, EngineType.GEMINI],
    )
    router = _make_router(config)
    # Mistral unavailable, Gemini available
    router.engines[EngineType.GEMINI] = _StubEngine(True)

    choice = router.select_fallback(primary=EngineType.DEEPSEEK)
    assert choice == EngineType.GEMINI


def test_chain_skips_primary() -> None:
    """Chain entry matching primary is skipped."""
    config = PipelineConfig(
        fallback_chain=[EngineType.DEEPSEEK, EngineType.GEMINI],
    )
    router = _make_router(config)
    router.engines[EngineType.DEEPSEEK] = _StubEngine(True)
    router.engines[EngineType.GEMINI] = _StubEngine(True)

    choice = router.select_fallback(primary=EngineType.DEEPSEEK)
    assert choice == EngineType.GEMINI


def test_chain_empty_falls_through_to_auto() -> None:
    """Empty chain falls through to automatic priority-based selection."""
    config = PipelineConfig(fallback_chain=[])
    router = _make_router(config)
    router.engines[EngineType.NOUGAT] = _StubEngine(True)

    choice = router.select_fallback(primary=EngineType.DEEPSEEK)
    assert choice == EngineType.NOUGAT


def test_chain_all_unavailable_returns_none() -> None:
    """All engines unavailable -> None."""
    config = PipelineConfig(
        fallback_chain=[EngineType.MISTRAL, EngineType.GEMINI],
    )
    router = _make_router(config)
    # All stubs default to False (unavailable)

    choice = router.select_fallback(primary=EngineType.DEEPSEEK)
    assert choice is None


# --- select_fallback_chain ---

def test_select_fallback_chain_ordered() -> None:
    """Returns chain engines first, then remaining available by priority."""
    config = PipelineConfig(
        fallback_chain=[EngineType.GEMINI, EngineType.MISTRAL],
    )
    router = _make_router(config)
    router.engines[EngineType.GEMINI] = _StubEngine(True)
    router.engines[EngineType.MISTRAL] = _StubEngine(True)
    router.engines[EngineType.NOUGAT] = _StubEngine(True)

    chain = router.select_fallback_chain(primary=EngineType.DEEPSEEK)
    # Gemini and Mistral from configured chain first, then Nougat from auto
    assert chain == [EngineType.GEMINI, EngineType.MISTRAL, EngineType.NOUGAT]


def test_select_fallback_chain_excludes_primary() -> None:
    config = PipelineConfig(
        fallback_chain=[EngineType.DEEPSEEK, EngineType.GEMINI],
    )
    router = _make_router(config)
    router.engines[EngineType.DEEPSEEK] = _StubEngine(True)
    router.engines[EngineType.GEMINI] = _StubEngine(True)

    chain = router.select_fallback_chain(primary=EngineType.DEEPSEEK)
    assert EngineType.DEEPSEEK not in chain
    assert chain == [EngineType.GEMINI]


def test_select_fallback_chain_empty_when_none_available() -> None:
    config = PipelineConfig(fallback_chain=[EngineType.MISTRAL])
    router = _make_router(config)

    chain = router.select_fallback_chain(primary=EngineType.DEEPSEEK)
    assert chain == []
