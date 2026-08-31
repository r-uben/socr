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
- **Direct profile injection.** ``provider_ladder`` accepts either a set of
  ``EngineType`` values (dict-lookup path, backward-compatible) or a list of
  ``ProviderProfile`` objects (direct path, skips dict). The direct path lets
  callers like ``_available_engines_for_agentic`` supply two QWEN profiles
  (local + cloud) as distinct rungs without needing two ``EngineType`` keys.
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
    model="gemini-3-flash-preview",
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
# QWEN maps to the local-instruct profile by default. PROFILE_QWEN_CLOUD shares
# the same EngineType.QWEN key and cannot coexist in this dict — callers that
# need both as distinct rungs (e.g. _available_engines_for_agentic) pass a
# list[ProviderProfile] directly to provider_ladder() instead of using this dict.
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

#: GH-370: every named profile, including the ones DEFAULT_PROVIDERS cannot
#: hold because they share an ``EngineType`` with another rung (QWEN cloud).
#: ``profile_by_id`` below resolves a recorded attempt back to its profile.
_ALL_PROFILES: tuple[ProviderProfile, ...] = (
    PROFILE_QWEN_LOCAL,
    PROFILE_QWEN_CLOUD,
    PROFILE_GEMINI,
    PROFILE_MARKER,
    PROFILE_GLM,
    PROFILE_NOUGAT,
    PROFILE_MISTRAL,
    PROFILE_DEEPSEEK,
    PROFILE_DEEPSEEK_VLLM,
    PROFILE_VLLM,
)


def _sort_key(p: ProviderProfile) -> tuple[float, int]:
    """Cheapest first; ties broken by the existing priority ladder."""
    return (p.cost_per_page_usd, ENGINE_PRIORITY.get(p.engine, 99))


def provider_ladder(
    available: set[EngineType] | list[EngineType] | list[ProviderProfile] | None = None,
    *,
    registry: dict[EngineType, ProviderProfile] | None = None,
    per_page_only: bool = False,
    max_cost_per_page: float = 0.0,
    include_ineligible: bool = False,
) -> list[ProviderProfile]:
    """Providers ordered cheapest-first — the escalation ladder for a page.

    Args:
        available: engines or profiles available right now. Three forms:
            - ``None``: all providers in the registry.
            - ``set[EngineType] | list[EngineType]``: look up each engine in
              the registry (backward-compatible path).
            - ``list[ProviderProfile]``: use these profiles directly, bypassing
              the registry. Enables two profiles with the same EngineType (e.g.
              QWEN local + QWEN cloud) as distinct rungs.
        registry: cost table override (defaults to DEFAULT_PROVIDERS). Ignored
            when ``available`` is a ``list[ProviderProfile]``.
        per_page_only: keep only providers that can OCR individual pages.
        max_cost_per_page: if > 0, drop providers above this price cap.
        include_ineligible: if True, include providers with auto_eligible=False
            (DeepSeek, Mistral). Default False — they are excluded from the
            automatic routing ladder and only reachable via explicit --primary.
    """
    if available is not None and available and isinstance(next(iter(available)), ProviderProfile):
        profiles: list[ProviderProfile] = list(available)  # type: ignore[arg-type]
    else:
        reg = registry or DEFAULT_PROVIDERS
        if available is None:
            avail: set[EngineType] = set(reg.keys())
        else:
            avail = set(available)  # type: ignore[arg-type]
        profiles = [p for e, p in reg.items() if e in avail]

    ladder = [
        p
        for p in profiles
        if (not per_page_only or p.supports_per_page)
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


def is_cloud_qwen(profile: ProviderProfile | None) -> bool:
    """Whether *profile* is the Ollama-Cloud Qwen rung.

    One predicate so the two places that must treat that rung specially -- the
    config overrides below and the availability probe in
    ``UnifiedPipeline._run_engine_on_pages`` -- cannot drift apart. Keyed on the
    profile ``id`` rather than the descriptive ``backend`` label, so the rung's
    identity (not a transport string that may be reworded) is what decides.
    """
    return profile is not None and profile.id == PROFILE_QWEN_CLOUD.id


def execution_overrides(profile: ProviderProfile) -> dict[str, object]:
    """Config fields that must be forced so *profile* actually runs as declared.

    GH-159. ``ProviderProfile`` carries ``(engine, backend, model)``, but until now
    only ``engine`` reached execution: ``route_page`` passed ``prof.engine`` into
    ``run_provider`` and the backend/model were recorded as provenance only. For
    every engine whose ``EngineType`` maps 1:1 to a deployment that is fine. For
    QWEN it is not — ``PROFILE_QWEN_LOCAL`` and ``PROFILE_QWEN_CLOUD`` share
    ``EngineType.QWEN``, so the cloud rung executed whatever
    ``resolve_qwen_intent`` derived from ``PipelineConfig`` (normally the local
    instruct build) while the manifest recorded ``backend="ollama-cloud"``.

    Returned overrides are deliberately MINIMAL — only the ambiguous case is
    touched:

    - **Cloud Qwen** pins the model and forces the Ollama backend. Ollama Cloud is
      served by the local Ollama runtime under a ``:cloud`` tag, so the backend is
      ``ollama``, not the profile's descriptive ``ollama-cloud`` label. Pinning is
      required because ``resolve_qwen_intent`` rewrites any unpinned model back to
      ``OLLAMA_MODEL`` on a local backend — the exact clobber that made this rung
      inert.
    - **Everything else** returns ``{}``. In particular the LOCAL Qwen rung is left
      alone: forcing its Ollama tag would break a vLLM/HPC deployment, where
      ``qwen_backend`` is the operator's deliberate choice and the config-derived
      model is already correct.

    ``qwen_backend`` is the one literal here, and deliberately so: it is not the
    profile's ``backend`` field. That field carries the DESCRIPTIVE label
    ``"ollama-cloud"``, which is not a value ``PipelineConfig.qwen_backend`` accepts
    ("auto", "ollama", "vllm", "api"). Ollama Cloud is served by the local Ollama
    runtime under a ``:cloud`` model tag, so the executed backend really is
    ``ollama`` -- the translation cannot be sourced from the registry because the
    registry records what the rung IS, not which transport runs it. The model, which
    the registry can answer for, does come from the profile.
    """
    if is_cloud_qwen(profile):
        return {
            "qwen_backend": "ollama",
            "qwen_model": profile.model,
            "qwen_model_pinned": True,
        }
    return {}


def profile_by_id(provider_id: str) -> ProviderProfile | None:
    """The named profile whose ``id`` is *provider_id*, or ``None``.

    GH-370. A recorded ``ProviderAttempt`` carries the rung's ``provider_id``
    but not the profile object, and ``resolved_provenance`` needs the profile
    (its ``engine`` decides whether anything must be resolved). Keyed on ``id``
    -- the same stable identity ``is_cloud_qwen`` uses -- so a reworded
    descriptive label cannot break the lookup.
    """
    for prof in _ALL_PROFILES:
        if prof.id == provider_id:
            return prof
    return None


def resolved_provenance(profile: ProviderProfile, config: object) -> tuple[str, str]:
    """Return the ``(backend, model)`` that ACTUALLY executed *profile*.

    GH-370. ``ProviderProfile.backend``/``.model`` describe what a rung IS in
    the registry, not which transport ran it. For most profiles those coincide.
    For QWEN they do not: ``PROFILE_QWEN_LOCAL`` declares ``backend="ollama"``
    and the Ollama tag ``qwen3-vl:30b-a3b-instruct``, while
    ``execution_overrides`` deliberately returns ``{}`` for it so the operator's
    ``--qwen-backend`` choice wins at execution. A vLLM/HPC run therefore
    executed on vLLM and recorded ``ollama`` -- on hosts where Ollama is not
    even installed.

    That makes the manifest unusable for the audit it exists for: a genuinely
    misrouted run (``qwen-ocr`` missing from ``PATH``, the documented HPC
    gotcha) and a correct one are indistinguishable in the record.

    This is the recording counterpart of ``execution_overrides``: both answer
    "what really runs" from the same two inputs, so provenance cannot drift
    from execution again.

    - **Cloud Qwen** -- ``("ollama", profile.model)``, matching the overrides
      above: Ollama Cloud is served by the local Ollama runtime under a
      ``:cloud`` tag, so the executed backend is ``ollama``, not the
      descriptive ``ollama-cloud`` label.
    - **Any other QWEN rung** -- whatever ``resolve_qwen_intent`` derives from
      the live config, which is exactly what the CLI invocation was built from.
    - **Everything else** -- the registry values unchanged. Their
      ``EngineType`` maps 1:1 to a deployment, so there is nothing to resolve.
    """
    if is_cloud_qwen(profile):
        return "ollama", profile.model
    if profile.engine is EngineType.QWEN:
        # Local import: ``core`` must not import ``engines`` at module scope.
        from socr.engines.qwen import resolve_qwen_intent

        # GH-384: no rewrite here. ``resolve_qwen_intent`` now resolves
        # ``auto`` + VLLM_BASE_URL itself, and ``_build_command`` reads the same
        # function, so the recorded backend/model is by construction the one the
        # CLI was invoked with. A second rewrite at this site is what made the
        # manifest name a backend the invocation never asked for.
        return resolve_qwen_intent(config)
    return profile.backend, profile.model


def qwen_auto_resolves_to_openai(config: object) -> bool:
    """Whether an ``auto`` qwen backend actually lands on an OpenAI-compatible server.

    GH-370 follow-up. ``PipelineConfig`` adopts ``VLLM_BASE_URL`` into
    ``qwen_vllm_url`` while leaving ``qwen_backend`` at its ``"auto"`` default,
    so exporting ONE environment variable is the whole HPC deployment -- see
    ``UnifiedPipeline._local_backend_is_openai_compatible``, whose docstring
    names this "not a corner case, it is the HPC deployment".

    An EXPLICIT backend always wins, in both directions: ``"ollama"`` stays
    Ollama even with ``VLLM_BASE_URL`` exported, because a value the user typed
    outranks one the environment happens to carry.
    """
    import os

    return getattr(config, "qwen_backend", "") == "auto" and bool(os.environ.get("VLLM_BASE_URL"))
