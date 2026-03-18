"""Selective page repair routing.

Examines DocumentState to find pages that need reprocessing, then selects
the best fallback engine for each based on the failure mode. Produces a
RepairPlan that the orchestrator can execute — this module does NOT run
any engines itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from socr.core.config import EngineType, PipelineConfig
from socr.core.result import FailureMode
from socr.core.state import DocumentState, PageState

# Engine families — used to pick a *different* family when the failure mode
# suggests the current family is fundamentally unsuited (e.g. hallucination).
_ENGINE_FAMILIES: dict[str, set[EngineType]] = {
    "deepseek": {EngineType.DEEPSEEK, EngineType.DEEPSEEK_VLLM},
    "google": {EngineType.GEMINI},
    "mistral": {EngineType.MISTRAL},
    "meta": {EngineType.NOUGAT, EngineType.MARKER},
    "local": {EngineType.GLM, EngineType.VLLM},
}

# Reverse lookup: engine -> family name
_ENGINE_TO_FAMILY: dict[EngineType, str] = {}
for _fam, _members in _ENGINE_FAMILIES.items():
    for _eng in _members:
        _ENGINE_TO_FAMILY[_eng] = _fam

# Cloud engines (less likely to refuse, more capable)
_CLOUD_ENGINES: set[EngineType] = {
    EngineType.GEMINI,
    EngineType.MISTRAL,
    EngineType.DEEPSEEK,
}

# "Lighter" engines — faster, lower resource, good for timeout recovery
_LIGHT_ENGINES: list[EngineType] = [
    EngineType.GLM,
    EngineType.NOUGAT,
    EngineType.MARKER,
]

# "Capable" engines — better quality, for garbage/low-word-count recovery
_CAPABLE_ENGINES: list[EngineType] = [
    EngineType.GEMINI,
    EngineType.MISTRAL,
    EngineType.DEEPSEEK,
    EngineType.DEEPSEEK_VLLM,
]


@dataclass
class PageRepair:
    """A single repair action: re-process one page with a specific engine."""

    page_num: int
    engine: EngineType
    reason: str


@dataclass
class RepairPlan:
    """The output of repair planning: what to do and what to skip."""

    repairs: list[PageRepair] = field(default_factory=list)
    pages_skipped: list[int] = field(default_factory=list)

    @property
    def by_engine(self) -> dict[EngineType, list[PageRepair]]:
        """Group repairs by engine for batched execution."""
        groups: dict[EngineType, list[PageRepair]] = {}
        for repair in self.repairs:
            groups.setdefault(repair.engine, []).append(repair)
        return groups

    @property
    def is_empty(self) -> bool:
        return len(self.repairs) == 0


class RepairRouter:
    """Decides which pages need repair and which engine to use for each."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pages_needing_repair(
        self, state: DocumentState
    ) -> list[tuple[int, PageState]]:
        """Return (page_num, PageState) pairs that need reprocessing.

        A page needs repair if ``page_state.needs_repair`` is True
        (no passing best_output and not born-digital with native text).
        """
        return [
            (page_num, page_state)
            for page_num, page_state in sorted(state.pages.items())
            if page_state.needs_repair
        ]

    def select_repair_engine(
        self,
        failure_mode: FailureMode,
        tried_engines: set[EngineType],
    ) -> EngineType | None:
        """Pick the best untried engine for a given failure mode.

        Returns None if all engines in the fallback chain have been
        exhausted.
        """
        candidates = self._candidates(tried_engines)
        if not candidates:
            return None

        match failure_mode:
            case FailureMode.HALLUCINATION:
                return self._pick_different_family(tried_engines, candidates)
            case FailureMode.REFUSAL:
                return self._pick_cloud(candidates)
            case FailureMode.GARBAGE | FailureMode.LOW_WORD_COUNT:
                return self._pick_capable(candidates)
            case FailureMode.TIMEOUT:
                return self._pick_light(candidates)
            case _:
                # EMPTY_OUTPUT, NONE, API_ERROR, CLI_ERROR, etc.
                return candidates[0]

    def plan_repairs(
        self, state: DocumentState, config: PipelineConfig | None = None
    ) -> RepairPlan:
        """Build a full repair plan from the current document state.

        For each page that needs repair, determines the dominant failure
        mode from its attempts and selects the next engine to try.
        Pages where all engines have been exhausted go into
        ``pages_skipped``.
        """
        cfg = config or self.config
        _ = cfg  # config already captured in __init__; param kept for API symmetry

        pages = self.pages_needing_repair(state)
        repairs: list[PageRepair] = []
        skipped: list[int] = []

        for page_num, page_state in pages:
            failure = self._dominant_failure(page_state)
            tried = self._tried_engines(page_state)
            engine = self.select_repair_engine(failure, tried)

            if engine is None:
                skipped.append(page_num)
            else:
                reason = self._build_reason(failure, tried, engine)
                repairs.append(
                    PageRepair(
                        page_num=page_num,
                        engine=engine,
                        reason=reason,
                    )
                )

        return RepairPlan(repairs=repairs, pages_skipped=skipped)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _candidates(self, tried: set[EngineType]) -> list[EngineType]:
        """Return fallback chain engines not yet tried, in config order."""
        chain = self._full_chain()
        return [e for e in chain if e not in tried]

    def _full_chain(self) -> list[EngineType]:
        """Full ordered list of engines available for repair.

        Starts with the configured fallback_chain, then appends the
        primary engine (it might not have been used on a specific page
        during a partial re-run), then any remaining enabled engines.
        Deduplicates while preserving order.
        """
        seen: set[EngineType] = set()
        result: list[EngineType] = []

        for engine in self.config.fallback_chain:
            if engine not in seen:
                result.append(engine)
                seen.add(engine)

        if self.config.primary_engine not in seen:
            result.append(self.config.primary_engine)
            seen.add(self.config.primary_engine)

        for engine in self.config.enabled_engines:
            if engine not in seen:
                result.append(engine)
                seen.add(engine)

        return result

    def _pick_different_family(
        self,
        tried: set[EngineType],
        candidates: list[EngineType],
    ) -> EngineType:
        """For hallucination: prefer an engine from a different family."""
        tried_families = {_ENGINE_TO_FAMILY.get(e) for e in tried} - {None}

        for engine in candidates:
            family = _ENGINE_TO_FAMILY.get(engine)
            if family and family not in tried_families:
                return engine

        # All families tried — just return first untried engine
        return candidates[0]

    def _pick_cloud(self, candidates: list[EngineType]) -> EngineType:
        """For refusal: prefer cloud engines (they refuse less)."""
        for engine in candidates:
            if engine in _CLOUD_ENGINES:
                return engine
        return candidates[0]

    def _pick_capable(self, candidates: list[EngineType]) -> EngineType:
        """For garbage/low-word-count: prefer more capable engines."""
        for engine in candidates:
            if engine in _CAPABLE_ENGINES:
                return engine
        return candidates[0]

    def _pick_light(self, candidates: list[EngineType]) -> EngineType:
        """For timeout: prefer faster/lighter engines."""
        for engine in candidates:
            if engine in _LIGHT_ENGINES:
                return engine
        return candidates[0]

    @staticmethod
    def _dominant_failure(page_state: PageState) -> FailureMode:
        """Extract the most recent non-NONE failure mode from page attempts.

        If no attempts exist yet (fresh page, never processed), returns
        EMPTY_OUTPUT so the router treats it like a first try.
        """
        for attempt in reversed(page_state.attempts):
            if attempt.failure_mode != FailureMode.NONE:
                return attempt.failure_mode
        return FailureMode.EMPTY_OUTPUT

    @staticmethod
    def _tried_engines(page_state: PageState) -> set[EngineType]:
        """Collect engine types already tried on this page."""
        engines: set[EngineType] = set()
        for attempt in page_state.attempts:
            # attempt.engine is a string; convert to EngineType if valid
            try:
                engines.add(EngineType(attempt.engine))
            except ValueError:
                pass
        return engines

    @staticmethod
    def _build_reason(
        failure: FailureMode, tried: set[EngineType], engine: EngineType
    ) -> str:
        """Human-readable reason for the repair decision."""
        tried_names = ", ".join(sorted(e.value for e in tried)) if tried else "none"
        return (
            f"failure={failure.value}, "
            f"tried=[{tried_names}], "
            f"selected={engine.value}"
        )
