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
- **Provider identity = engine + backend + model.** QWEN alone is ambiguous:
  it covers both local qwen3-vl:30b-a3b-instruct (Ollama, free) and cloud
  qwen3.5:cloud (Ollama Cloud, ~free). Named profiles carry all three fields
  so the manifest and replay logic can distinguish them unambiguously.
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
    id: str = ""
    backend: str = ""
    model: str = ""
    auto_eligible: bool = True

    @property
    def is_free(self) -> bool:
        return self.cost_per_page_usd <= 0.0


# Named profiles — each one is a unique (engine, backend, model) triple.
# These are the canonical rungs of the agentic ladder.

PROFILE_QWEN_LOCAL = ProviderProfile(
    engine=EngineType.QWEN,
    tier=TIER_LOCAL,
    cost_per_page_usd=0.0,
    id="qwen-local-instruct",
    backend="ollama",
    model="qwen3-vl:30b-a3b-instruct",
)

PROFILE_QWEN_CLOUD = ProviderProfile(
    engine=EngineType.QWEN,
    tier=TIER_CLOUD,
    cost_per_page_usd=0.0,
    id="qwen-cloud",
    backend="ollama-cloud",
    model="qwen3.5:cloud",
)

PROFILE_GEMINI = ProviderProfile(
    engine=EngineType.GEMINI,
    tier=TIER_CLOUD,
    cost_per_page_usd=0.0002,
    id="gemini",
    backend="gemini-api",
    model="gemini-1.5-flash",
)

PROFILE_MARKER = ProviderProfile(
    engine=EngineType.MARKER,
    tier=TIER_LOCAL,
    cost_per_page_usd=0.0,
    id="marker",
    backend="marker",
    model="marker",
)

PROFILE_GLM = ProviderProfile(
    engine=EngineType.GLM,
    tier=TIER_LOCAL,
    cost_per_page_usd=0.0,
    id="glm",
    backend="ollama",
    model="glm-4v",
)

PROFILE_NOUGAT = ProviderProfile(
    engine=EngineType.NOUGAT,
    tier=TIER_LOCAL,
    cost_per_page_usd=0.0,
    id="nougat",
    backend="nougat",
    model="nougat",
)

PROFILE_MISTRAL = ProviderProfile(
    engine=EngineType.MISTRAL,
    tier=TIER_CLOUD,
    cost_per_page_usd=0.001,
    id="mistral",
    backend="mistral-api",
    model="mistral-ocr-latest",
    auto_eligible=False,
)

PROFILE_DEEPSEEK = ProviderProfile(
    engine=EngineType.DEEPSEEK,
    tier=TIER_LOCAL,
    cost_per_page_usd=0.0,
    id="deepseek",
    backend="deepseek",
    model="deepseek-ocr",
    auto_eligible=False,
)

PROFILE_DEEPSEEK_VLLM = ProviderProfile(
    engine=EngineType.DEEPSEEK_VLLM,
    tier=TIER_LOCAL,
    cost_per_page_usd=0.0,
    id="deepseek-vllm",
    backend="vllm",
    model="deepseek-ai/DeepSeek-OCR",
)

PROFILE_VLLM = ProviderProfile(
    engine=EngineType.VLLM,
    tier=TIER_LOCAL,
    cost_per_page_usd=0.0,
    id="vllm",
    backend="vllm",
    model="Qwen/Qwen2-VL-7B-Instruct",
)


# Default cost table. Local engines run on your own hardware -> free. Cloud
# per-page prices are rough estimates (see README engine table) and are meant to
# be tuned, not trusted as exact. Edit here or override via PipelineConfig.
#
# QWEN maps to the local-instruct profile by default; the cloud variant is
# reachable via PROFILE_QWEN_CLOUD but does not get its own EngineType key
# because EngineType.QWEN is the shared key for both backends.
DEFAULT_PROVIDERS: dict[EngineType, ProviderProfile] = {
    EngineType.QWEN: PROFILE_QWEN_LOCAL,
    EngineType.GLM: PROFILE_GLM,
    EngineType.NOUGAT: PROFILE_NOUGAT,
    EngineType.DEEPSEEK: PROFILE_DEEPSEEK,
    EngineType.MARKER: PROFILE_MARKER,
    EngineType.GEMINI: PROFILE_GEMINI,
    EngineType.MISTRAL: PROFILE_MISTRAL,
    EngineType.DEEPSEEK_VLLM: PROFILE_DEEPSEEK_VLLM,
    EngineType.VLLM: PROFILE_VLLM,
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
    include_ineligible: bool = False,
) -> list[ProviderProfile]:
    """Providers ordered cheapest-first — the escalation ladder for a page.

    Args:
        available: engines that are actually usable right now (probed). None =
            all known providers.
        registry: cost table override (defaults to DEFAULT_PROVIDERS).
        per_page_only: keep only providers that can OCR individual pages.
        max_cost_per_page: if > 0, drop providers above this price cap.
        include_ineligible: if True, include providers with auto_eligible=False
            (DeepSeek, Mistral). Default False — they are excluded from the
            automatic routing ladder and only reachable via explicit --primary.
    """
    reg = registry or DEFAULT_PROVIDERS
    avail = set(available) if available is not None else set(reg.keys())
    ladder = [
        p
        for e, p in reg.items()
        if e in avail
        and (not per_page_only or p.supports_per_page)
        and (max_cost_per_page <= 0.0 or p.cost_per_page_usd <= max_cost_per_page)
        and (include_ineligible or p.auto_eligible)
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
