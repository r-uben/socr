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

import pathlib
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


# ---------------------------------------------------------------------------
# The production line, not just the helpers.
#
# The tests above pin the agentic call boundary and the two helpers in isolation.
# Reverting `orchestrator.py` -- dropping `profile=profile` from the
# `_run_engine_on_pages` call, or the `replace(self.config, **overrides)` inside it
# -- leaves every one of them green while the cloud rung silently runs the local
# build again. These drive the real method and spy the config that reaches the
# engine, so the wiring itself is guarded.
# ---------------------------------------------------------------------------


class _SpyEngine:
    """Records the config each `process_pages` call receives."""

    name = "qwen"

    def __init__(self):
        self.configs = []

    def is_available(self):
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi):
        self.configs.append(config)
        return [
            PageOutput(page_num=n, text="ocr", status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


def _pipeline_with(spy, monkeypatch):
    from socr.core.config import PipelineConfig
    from socr.pipeline import orchestrator as orch

    monkeypatch.setattr(orch, "get_engine", lambda engine_type: spy)
    return orch.UnifiedPipeline(PipelineConfig(quiet=True))


class _StubHandle:
    path = pathlib.Path("/nonexistent/doc.pdf")


class _StubPage:
    """Minimal page state: the unavailable-engine branch reads `native_text`."""

    native_text = ""
    has_tables = False
    native_table_structure_failed = False


class _StubState:
    def __init__(self):
        self.handle = _StubHandle()
        self.pages = {1: _StubPage()}


def test_the_cloud_rung_reaches_the_engine_with_its_own_backend_and_model(monkeypatch):
    """The guard: revert the orchestrator wiring and this goes red.

    Pins the DIFFERENCE at the real call site -- the config the engine is handed
    for the cloud rung versus the local rung.

    `cloud_model_available` MUST be patched. The cloud rung's availability probe is
    no longer `engine.is_available()` -- that IS the GH-159 fix -- so leaving the
    probe live makes this test depend on whether ollama happens to be installed. It
    passed on a workstation and failed in CI, where the probe returns False, the
    cloud call never reaches the engine, and `spy.configs` has one entry instead of
    two. Exactly the local-passes/CI-fails trap CLAUDE.md documents.
    """
    from socr.engines import qwen as qwen_engine

    monkeypatch.setattr(qwen_engine, "cloud_model_available", lambda: True)

    spy = _SpyEngine()
    pipe = _pipeline_with(spy, monkeypatch)

    pipe._run_engine_on_pages(
        _StubState(), [1], [], EngineType.QWEN, "agentic", profile=PROFILE_QWEN_LOCAL
    )
    pipe._run_engine_on_pages(
        _StubState(), [1], [], EngineType.QWEN, "agentic", profile=PROFILE_QWEN_CLOUD
    )

    local_cfg, cloud_cfg = spy.configs

    # The cloud rung executes what it declares...
    assert cloud_cfg.qwen_model == PROFILE_QWEN_CLOUD.model
    assert cloud_cfg.qwen_backend == "ollama"
    assert cloud_cfg.qwen_model_pinned is True

    # ...and the local rung's config is untouched, so a vLLM/HPC deployment keeps
    # the operator's own `qwen_backend`.
    assert local_cfg.qwen_model == pipe.config.qwen_model
    assert local_cfg.qwen_backend == pipe.config.qwen_backend
    assert local_cfg.qwen_model_pinned == pipe.config.qwen_model_pinned

    # The regression in one line: the two rungs must not hand over the same config.
    assert (cloud_cfg.qwen_model, cloud_cfg.qwen_backend) != (
        local_cfg.qwen_model,
        local_cfg.qwen_backend,
    )


def test_no_profile_leaves_the_config_exactly_as_the_non_agentic_paths_saw_it(monkeypatch):
    """The no-op guarantee for phase-major callers, which pass no profile."""
    spy = _SpyEngine()
    pipe = _pipeline_with(spy, monkeypatch)

    pipe._run_engine_on_pages(_StubState(), [1], [], EngineType.QWEN, "local")

    assert spy.configs[0] is pipe.config


def test_a_cloud_only_machine_can_still_run_the_cloud_rung(monkeypatch):
    """`is_available()` probes the LOCAL tier, so it must not gate the cloud rung.

    #159's acceptance list says cloud-only environments must successfully use the
    qwen-cloud rung. Asking the local probe about it refuses the rung before its
    pinned config ever runs.
    """
    from socr.engines import qwen as qwen_engine
    from socr.pipeline import orchestrator as orch

    spy = _SpyEngine()
    spy.is_available = lambda: False  # cloud-only box: no local build, no vLLM
    monkeypatch.setattr(orch, "get_engine", lambda engine_type: spy)
    monkeypatch.setattr(qwen_engine, "cloud_model_available", lambda: True)

    pipe = orch.UnifiedPipeline(PipelineConfig(quiet=True))
    out = pipe._run_engine_on_pages(
        _StubState(), [1], [], EngineType.QWEN, "agentic", profile=PROFILE_QWEN_CLOUD
    )

    assert spy.configs, "the cloud rung must reach the engine, not be refused"
    assert out[0].status is PageStatus.SUCCESS

    # The DIFFERENCE: the same machine still refuses the LOCAL rung.
    spy2 = _SpyEngine()
    spy2.is_available = lambda: False
    monkeypatch.setattr(orch, "get_engine", lambda engine_type: spy2)
    pipe2 = orch.UnifiedPipeline(PipelineConfig(quiet=True))
    pipe2._run_engine_on_pages(
        _StubState(), [1], [], EngineType.QWEN, "agentic", profile=PROFILE_QWEN_LOCAL
    )
    assert not spy2.configs, "the local rung must still respect the local probe"


def test_the_agentic_loop_hands_the_profile_down_to_the_engine_runner(tmp_path):
    """Guards the CALL SITE, which the direct-invocation tests above cannot.

    `_run_engine_on_pages` applying the overrides is only half the wiring; the
    other half is `_phase_agentic`'s `run_provider` closure passing `profile` in.
    Dropping that argument leaves every other test in this file green -- the
    helpers still work, the router still hands over profiles -- while the cloud
    rung silently runs the local build again.

    So drive the real phase and record what arrives. Hermetic per CLAUDE.md: the
    provider ladder is patched, the judge is a stub, and the crop-VLM probe is
    pinned, so nothing contacts ollama and CI behaves like a workstation.
    """
    import fitz

    from socr.core.config import PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = tmp_path / "f.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 80
    for _ in range(14):
        page.insert_text((60, y), "Estimated coefficient 0.082 significant", fontsize=9)
        y += 16
    doc.save(str(pdf))
    doc.close()

    seen: list = []

    pipe = UnifiedPipeline(PipelineConfig(agentic=True, quiet=True))

    # A clean born-digital prose page takes the trusted-native bypass and never
    # reaches the ladder (that bypass is GH-317 itself), so force the page to need
    # OCR -- otherwise this test would pass vacuously with an empty `seen`.
    _detect = pipe.bd_detector.detect

    def _needs_ocr(path):
        assessment = _detect(path)
        assessment.pages[0].needs_ocr_enhancement = True
        return assessment

    pipe.bd_detector.detect = _needs_ocr
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_CLOUD]
    pipe._build_page_judge = lambda state: _RejectingJudge()
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""

    def spy(state, nums, nat, eng, phase, profile=None):
        seen.append(profile)
        return [
            PageOutput(page_num=p, text=f"text {p}", status=PageStatus.SUCCESS, engine="qwen")
            for p in nums
        ]

    pipe._run_engine_on_pages = spy
    pipe.process(pdf, output_dir=tmp_path / "out")

    assert seen, "the agentic loop must have called the engine runner"
    # The DIFFERENCE: every call carries the rung's identity. Before GH-159 the
    # closure passed only `prof.engine`, so this list would be all None.
    assert all(p is not None for p in seen), (
        "_phase_agentic must pass the ProviderProfile down; without it "
        "_run_engine_on_pages cannot know which backend/model the rung declares"
    )
    assert {p.id for p in seen} == {PROFILE_QWEN_CLOUD.id}
