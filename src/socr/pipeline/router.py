"""Routing logic for selecting OCR engines."""

from typing import Callable, Mapping

from socr.core.config import EngineType, PipelineConfig


class EngineRouter:
    """Decide which engine to use for primary and fallback stages."""

    def __init__(
        self,
        config: PipelineConfig,
        engines: Mapping[EngineType, object],
    ) -> None:
        self.config = config
        self.engines = engines

    def _enabled(self, engine: EngineType) -> bool:
        """Whether an engine is in the enabled list."""
        return engine in self.config.enabled_engines

    def _available(self, engine: EngineType) -> bool:
        """Whether an engine is enabled and currently available."""
        if not self._enabled(engine):
            return False
        impl = self.engines.get(engine)
        if not impl:
            return False
        try:
            return bool(impl.is_available())
        except Exception:
            return False

    def select_primary(
        self,
        warn: Callable[[str], None] | None = None,
    ) -> EngineType:
        """Select primary engine based on config and availability."""
        # Check configured primary first
        primary = self.config.primary_engine
        if self._available(primary):
            return primary
        if warn:
            warn(f"Primary engine '{primary.value}' not available; using automatic selection")

        # Fall through priority order
        preference = [
            EngineType.NOUGAT, EngineType.DEEPSEEK, EngineType.MARKER,
            EngineType.GEMINI, EngineType.MISTRAL,
        ]
        for engine_type in preference:
            if self._available(engine_type):
                return engine_type

        enabled = [e.value for e in EngineType if self._enabled(e)]
        raise RuntimeError(
            "No OCR engines are available. "
            f"Enabled in config: {enabled or 'none'}. "
            "Check dependencies/API keys/Ollama, or enable at least one engine."
        )

    def select_fallback(
        self,
        primary: EngineType,
        warn: Callable[[str], None] | None = None,
    ) -> EngineType | None:
        """Select first available fallback engine from the chain (excluding primary).

        Iterates through ``config.fallback_chain`` in order, returning the
        first engine that is available and different from *primary*.  Falls
        back to automatic priority-based selection when the chain is exhausted.
        """
        for engine in self.config.fallback_chain:
            if engine != primary and self._available(engine):
                return engine

        if warn and self.config.fallback_chain:
            warn("No engine in fallback chain available; trying automatic fallback")

        preference = [EngineType.DEEPSEEK, EngineType.NOUGAT, EngineType.MISTRAL, EngineType.GEMINI]
        for engine_type in preference:
            if engine_type != primary and self._available(engine_type):
                return engine_type

        return None

    def select_fallback_chain(
        self,
        primary: EngineType,
    ) -> list[EngineType]:
        """Return the full ordered list of available fallback engines (excluding primary).

        Engines from ``config.fallback_chain`` come first (preserving order),
        followed by any remaining available engines sorted by default priority.
        """
        seen: set[EngineType] = {primary}
        result: list[EngineType] = []

        # Chain engines first
        for engine in self.config.fallback_chain:
            if engine not in seen and self._available(engine):
                result.append(engine)
                seen.add(engine)

        # Then remaining available engines by priority
        preference = [EngineType.DEEPSEEK, EngineType.NOUGAT, EngineType.MISTRAL, EngineType.GEMINI]
        for engine in preference:
            if engine not in seen and self._available(engine):
                result.append(engine)
                seen.add(engine)

        return result

    def select_hpc_engines(
        self,
        warn: Callable[[str], None] | None = None,
    ) -> list[EngineType]:
        """Select engines for HPC mode."""
        available = []

        if self._available(EngineType.DEEPSEEK_VLLM):
            available.append(EngineType.DEEPSEEK_VLLM)
        elif warn:
            warn("DeepSeek-vLLM not available for HPC mode")

        if self._available(EngineType.NOUGAT):
            available.append(EngineType.NOUGAT)
        elif warn:
            warn("Nougat not available (LaTeX support disabled)")

        if self._available(EngineType.VLLM):
            available.append(EngineType.VLLM)
        elif warn:
            warn("vLLM vision not available for figure description")

        return available
