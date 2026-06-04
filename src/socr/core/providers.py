"""Provider cost registry — the cost model for agentic, cost-aware OCR routing.

The agentic router picks the *cheapest capable provider* per page and escalates
to a costlier one only when the judge rejects the output. That requires a single
source of truth for "how much does each provider cost, and in what order should
we try them." This module is that source.

Design choices (deliberate):

- **Relative ordering drives routing, not absolute dollars.** The loop tries
  providers cheapest-first; the judge decides accept/escalate. So the *order*
  matters more than the exact price. Order falls back to ``ENGINE_PRIORITY``
  (local-free -> cheap-cloud -> premium-cloud), which already lives in config.
- **Prices are tunable DEFAULTS, not magic constants buried in logic.** They sit
  in one editable table here and can be overridden per run. Local engines are
  free (your GPU / Ollama); cloud prices are rough per-page estimates to be
  refined as real usage data accrues. No routing code hardcodes a price.
- **No capability tables.** We do NOT pre-declare "engine X handles math." The
  judge catches a cheap provider failing on a hard page and the loop escalates.
  Letting the judge reason beats a brittle static capability matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

from socr.core.config import ENGINE_PRIORITY, EngineType

# Descriptive tiers (for reporting / grouping, not for routing math).
TIER_NATIVE = "native"
TIER_LOCAL = "local"
TIER_CLOUD = "cloud"


@dataclass(frozen=True)
class ProviderProfile:
    """Cost + tier metadata for one OCR provider."""

    engine: EngineType
    tier: str
    cost_per_page_usd: float  # DEFAULT estimate; 0.0 for local/native; tunable
    supports_per_page: bool = True

    @property
    def is_free(self) -> bool:
        return self.cost_per_page_usd <= 0.0


# Default cost table. Local engines run on your own hardware -> free. Cloud
# per-page prices are rough estimates (see README engine table) and are meant to
# be tuned, not trusted as exact. Edit here or override via PipelineConfig.
DEFAULT_PROVIDERS: dict[EngineType, ProviderProfile] = {
    # Qwen-VL (local Ollama) — free and the best open OCR; ENGINE_PRIORITY=0 makes the
    # cost-tied sort try it first among free providers.
    EngineType.QWEN: ProviderProfile(EngineType.QWEN, TIER_LOCAL, 0.0),
    EngineType.GLM: ProviderProfile(EngineType.GLM, TIER_LOCAL, 0.0),
    EngineType.NOUGAT: ProviderProfile(EngineType.NOUGAT, TIER_LOCAL, 0.0),
    EngineType.DEEPSEEK: ProviderProfile(EngineType.DEEPSEEK, TIER_LOCAL, 0.0),
    EngineType.MARKER: ProviderProfile(EngineType.MARKER, TIER_LOCAL, 0.0),
    EngineType.GEMINI: ProviderProfile(EngineType.GEMINI, TIER_CLOUD, 0.0002),
    EngineType.MISTRAL: ProviderProfile(EngineType.MISTRAL, TIER_CLOUD, 0.001),
    # vLLM/HPC providers run on local GPUs -> free.
    EngineType.DEEPSEEK_VLLM: ProviderProfile(EngineType.DEEPSEEK_VLLM, TIER_LOCAL, 0.0),
    EngineType.VLLM: ProviderProfile(EngineType.VLLM, TIER_LOCAL, 0.0),
}


def _sort_key(p: ProviderProfile) -> tuple[float, int]:
    """Cheapest first; ties broken by the existing priority ladder."""
    return (p.cost_per_page_usd, ENGINE_PRIORITY.get(p.engine, 99))


def provider_ladder(
    available: set[EngineType] | list[EngineType] | None = None,
    *,
    registry: dict[EngineType, ProviderProfile] | None = None,
    per_page_only: bool = False,
    max_cost_per_page: float = 0.0,
) -> list[ProviderProfile]:
    """Providers ordered cheapest-first — the escalation ladder for a page.

    Args:
        available: engines that are actually usable right now (probed). None =
            all known providers.
        registry: cost table override (defaults to DEFAULT_PROVIDERS).
        per_page_only: keep only providers that can OCR individual pages.
        max_cost_per_page: if > 0, drop providers above this price cap.
    """
    reg = registry or DEFAULT_PROVIDERS
    avail = set(available) if available is not None else set(reg.keys())
    ladder = [
        p
        for e, p in reg.items()
        if e in avail
        and (not per_page_only or p.supports_per_page)
        and (max_cost_per_page <= 0.0 or p.cost_per_page_usd <= max_cost_per_page)
    ]
    return sorted(ladder, key=_sort_key)


def cost_of(
    engine: EngineType,
    n_pages: int = 1,
    *,
    registry: dict[EngineType, ProviderProfile] | None = None,
) -> float:
    """Estimated USD cost of running ``engine`` over ``n_pages``."""
    reg = registry or DEFAULT_PROVIDERS
    prof = reg.get(engine)
    if prof is None:
        return 0.0
    return prof.cost_per_page_usd * max(0, n_pages)
