"""GH-159: a ladder rung must EXECUTE its declared backend/model, not just record it.

``PROFILE_QWEN_LOCAL`` and ``PROFILE_QWEN_CLOUD`` share ``EngineType.QWEN``. Before
this fix ``route_page`` passed only ``prof.engine`` into ``run_provider``, so the
cloud rung ran whatever ``resolve_qwen_intent`` derived from ``PipelineConfig`` --
normally the local instruct build -- while the manifest still recorded
``backend="ollama-cloud"``. A cloud-only install could appear available and never
actually run the cloud model, and every local-vs-cloud measurement compared the
same backend against itself.

Testing note (CLAUDE.md): CI has no ollama and no provider, so nothing here drives a
real engine and nothing pins an absolute measured outcome. Each test pins a
DIFFERENCE between two rungs that vary only in provider identity.
"""

from __future__ import annotations

from dataclasses import replace

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import (
    PROFILE_GEMINI,
    PROFILE_QWEN_CLOUD,
    PROFILE_QWEN_LOCAL,
    execution_overrides,
)
from socr.core.result import PageOutput, PageStatus
from socr.engines.qwen import resolve_qwen_intent
from socr.pipeline.agentic import AcceptDecision, route_page


class _RejectingJudge:
    """Rejects everything, so the router walks the whole ladder."""

    def assess(self, output, provider):
        return AcceptDecision(accept=False, reason="stub rejects all")


def _recording_provider(seen: list):
    """A ``RunProvider`` stub that records the object it was handed per rung."""

    def run_provider(profile, page_num: int) -> PageOutput:
        seen.append(profile)
        return PageOutput(
            page_num=page_num,
            text=f"text from {getattr(profile, 'id', profile)}",
            status=PageStatus.SUCCESS,
            engine=profile.engine.value,
        )

    return run_provider


def test_router_hands_the_provider_profile_not_a_bare_engine():
    """The rung's identity survives the call, so an ambiguous engine is resolvable."""
    ladder = [PROFILE_QWEN_LOCAL, PROFILE_QWEN_CLOUD]
    seen: list = []

    route_page(1, ladder, _recording_provider(seen), _RejectingJudge())

    assert len(seen) == 2, "both rungs must run when the judge rejects every one"
    # The DIFFERENCE that matters: the two calls are distinguishable at all.
    # Passing `prof.engine` made them identical -- this is the regression.
    assert seen[0] is not seen[1]
    assert [p.id for p in seen] == ["qwen-local-instruct", "qwen-cloud"]
    assert [p.backend for p in seen] == ["ollama", "ollama-cloud"]
    # ...while the engine alone -- all the router used to pass -- cannot tell them apart.
    assert seen[0].engine is seen[1].engine is EngineType.QWEN


def test_recorded_provenance_matches_the_profile_that_ran():
    """Each attempt's provenance names the rung actually invoked, in order."""
    ladder = [PROFILE_QWEN_LOCAL, PROFILE_QWEN_CLOUD]
    seen: list = []

    decision = route_page(1, ladder, _recording_provider(seen), _RejectingJudge())

    ran = [p.id for p in seen]
    recorded = [a.provider_id for a in decision.attempts]
    assert recorded == ran, "manifest provenance must not drift from what executed"


def test_cloud_rung_overrides_config_but_local_rung_does_not():
    """Only the ambiguous rung is touched; the local rung stays byte-identical."""
    assert execution_overrides(PROFILE_QWEN_LOCAL) == {}
    assert execution_overrides(PROFILE_GEMINI) == {}

    cloud = execution_overrides(PROFILE_QWEN_CLOUD)
    assert cloud, "the cloud rung must force its own backend/model"
    # Sourced from the profile registry, never a literal in the override helper.
    assert cloud["qwen_model"] == PROFILE_QWEN_CLOUD.model
    # Ollama Cloud is served by the local Ollama runtime under a `:cloud` tag, so
    # the executed backend is `ollama`, not the descriptive profile label.
    assert cloud["qwen_backend"] == "ollama"
    assert cloud["qwen_model_pinned"] is True


def test_the_override_is_what_stops_resolve_qwen_intent_clobbering_the_cloud_model():
    """Without the pin, the resolver rewrites the cloud model back to the local build."""
    base = PipelineConfig()

    # What used to happen: the cloud rung ran through the untouched config.
    _, model_without_override = resolve_qwen_intent(base)

    patched = replace(base, **execution_overrides(PROFILE_QWEN_CLOUD))
    backend_with, model_with = resolve_qwen_intent(patched)

    # Pin the DIFFERENCE: the cloud rung now resolves to a different model than the
    # config-derived default, which is precisely what GH-159 says was missing.
    assert model_with != model_without_override
    assert model_with == PROFILE_QWEN_CLOUD.model
    assert backend_with == "ollama"


def test_local_rung_resolution_is_unchanged_by_the_fix():
    """The no-op guarantee: the local rung resolves exactly as it did before."""
    base = PipelineConfig()
    expected = resolve_qwen_intent(base)

    patched = replace(base, **execution_overrides(PROFILE_QWEN_LOCAL))
    assert resolve_qwen_intent(patched) == expected
