"""GH-133: the agentic page judge must BE the judge the system reports.

Four defects, one root cause — ``_build_page_judge`` built ``OllamaVisionJudge()``
bare (module default ``qwen2-vl:7b``) instead of the model ``_resolve_judge_model``
picks:

1. availability was a name-prefix match, so an installed 30B instruct model
   satisfied a request for an 8B judge that was never pulled;
2. a judge that raised propagated out of the per-page loop and killed the run;
3. provenance named a VLM for pages the heuristic checker had judged;
4. the run fingerprint recorded the (empty) config field, so pulling a judge
   model changed gating without invalidating the per-page resume ledger.

Hermetic by construction: no Ollama, no engines, no PDF. Every HTTP call is
stubbed, so these pin the same behaviour in CI as on a workstation.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import provider_ladder
from socr.core.result import PageOutput, PageStatus
from socr.judge.ollama_judge import OllamaVisionJudge
from socr.pipeline.agentic import route_page
from socr.pipeline.orchestrator import JUDGE_IDENTITY_HEURISTIC, UnifiedPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INSTALLED = ["qwen3-vl:30b-a3b-instruct", "llama3:latest"]


def _stub_tags(monkeypatch, names):
    """Make /api/tags report exactly ``names`` as the pulled models."""

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": n} for n in names]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())


class _State:
    """Minimal DocumentState stand-in for _build_page_judge.

    ``handle.path`` is only captured by the lazy ``get_fitz_page`` closure, which
    these tests never invoke — so it need not point at a real PDF.
    """

    def __init__(self):
        self.events = []
        self.agentic_judge_model = ""
        self.pages = {}
        self.handle = SimpleNamespace(path=Path("/nonexistent/doc.pdf"))


def _pipeline(**overrides):
    return UnifiedPipeline(PipelineConfig(**overrides))


# ---------------------------------------------------------------------------
# 1. Availability is an exact pull, not a family prefix
# ---------------------------------------------------------------------------


def test_installed_sibling_does_not_satisfy_a_different_tag(monkeypatch):
    """qwen3-vl:30b-a3b-instruct must NOT make qwen3-vl:8b look available.

    This is the trap: the prefix match reported the 8B judge as present, and the
    404 only surfaced later, at judge time, mid-document.
    """
    _stub_tags(monkeypatch, INSTALLED)
    assert OllamaVisionJudge(model="qwen3-vl:8b").is_available() is False


def test_exact_tag_is_available(monkeypatch):
    _stub_tags(monkeypatch, INSTALLED)
    assert OllamaVisionJudge(model="qwen3-vl:30b-a3b-instruct").is_available() is True


def test_untagged_reference_resolves_to_latest(monkeypatch):
    """Ollama treats a bare name as ``:latest``; availability must agree."""
    _stub_tags(monkeypatch, INSTALLED)
    assert OllamaVisionJudge(model="llama3").is_available() is True
    assert OllamaVisionJudge(model="qwen3-vl").is_available() is False


def test_unreachable_daemon_is_unavailable_not_an_error(monkeypatch):
    def _boom(*a, **k):
        raise httpx.ConnectError("no daemon")

    monkeypatch.setattr(httpx, "get", _boom)
    assert OllamaVisionJudge(model="anything:1b").is_available() is False


# ---------------------------------------------------------------------------
# 2. A judge that raises must not kill the document
# ---------------------------------------------------------------------------


class _ExplodingJudge:
    def assess(self, output, provider):
        raise httpx.HTTPStatusError("404 model not found", request=None, response=None)


class _AcceptingJudge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="stub")


def _run_provider(profile, page_num: int) -> PageOutput:
    # GH-159: the router passes the whole ProviderProfile, not a bare EngineType.
    engine = profile.engine
    return PageOutput(
        page_num=page_num,
        text=f"text from {engine.value}",
        status=PageStatus.SUCCESS,
        engine=engine.value,
    )


LADDER = provider_ladder({EngineType.GLM, EngineType.GEMINI}, include_ineligible=True)


def test_judge_exception_escalates_instead_of_propagating():
    """The page survives an exploding judge; the run does not abort."""
    decision = route_page(1, LADDER, _run_provider, _ExplodingJudge())

    assert decision.accepted is False
    # Every rung was tried, each recorded rather than swallowed.
    assert len(decision.attempts) == len(LADDER)
    assert all("judge raised" in a.reason for a in decision.attempts)


def test_judge_exception_keeps_the_text_it_could_not_judge():
    """Unjudged is not unusable: best-effort still ships real OCR text."""
    decision = route_page(1, LADDER, _run_provider, _ExplodingJudge())

    assert decision.final_output.text.strip()
    assert decision.final_output.text.startswith("text from ")


def test_judge_exception_preserves_provider_provenance():
    """The timeout path drops provider_id/model/backend; the raise path must not."""
    decision = route_page(1, LADDER, _run_provider, _ExplodingJudge())

    first = decision.attempts[0]
    assert first.provider_id
    assert first.backend


# ---------------------------------------------------------------------------
# 3. Provenance names the judge that actually ran
# ---------------------------------------------------------------------------


def test_provenance_says_heuristic_when_no_vlm_resolves(monkeypatch):
    """metadata.json must not claim a VLM judged heuristic-gated pages."""
    _stub_tags(monkeypatch, INSTALLED)  # none of the candidates are installed
    pipe = _pipeline()
    state = _State()

    pipe._build_page_judge(state)

    assert state.agentic_judge_model == JUDGE_IDENTITY_HEURISTIC


def test_degradation_emits_an_audit_event_under_default_backend(monkeypatch):
    """judge_backend defaults to "auto", where this used to be silent."""
    _stub_tags(monkeypatch, INSTALLED)
    pipe = _pipeline()
    assert pipe.config.judge_backend == "auto"
    state = _State()

    pipe._build_page_judge(state)

    kinds = [e.kind for e in state.events]
    assert "judge_degraded_to_heuristic" in kinds


def test_resolved_model_is_the_one_constructed(monkeypatch):
    """The judge must be built from the resolved model, never the module default."""
    _stub_tags(monkeypatch, ["minicpm-v:8b"])
    pipe = _pipeline()
    state = _State()

    built = []
    real_init = OllamaVisionJudge.__init__

    def _spy(self, model=None, *a, **k):
        built.append(model)
        return real_init(self, model=model, *a, **k) if model else real_init(self, *a, **k)

    monkeypatch.setattr(OllamaVisionJudge, "__init__", _spy)

    pipe._build_page_judge(state)

    assert "minicpm-v:8b" in built
    assert "qwen2-vl:7b" not in built, "module default must never reach the judge"
    assert state.agentic_judge_model == "minicpm-v:8b"


def test_resolution_is_memoized(monkeypatch):
    """_run_fingerprint runs per page; resolving each time would be 3 probes/page."""
    calls = {"n": 0}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            calls["n"] += 1
            return {"models": [{"name": "minicpm-v:8b"}]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    pipe = _pipeline()

    for _ in range(5):
        pipe._resolve_judge_model()

    assert calls["n"] <= 2, f"expected memoized resolution, got {calls['n']} probes"


def test_explicit_judge_model_is_never_discarded(monkeypatch):
    """An operator override wins without probing — a brief outage must not drop it."""

    def _boom(*a, **k):
        raise httpx.ConnectError("daemon down")

    monkeypatch.setattr(httpx, "get", _boom)
    pipe = _pipeline(judge_model="my-judge:v2")

    assert pipe._resolve_judge_model() == "my-judge:v2"


# ---------------------------------------------------------------------------
# 4. The fingerprint tracks the judge that will actually gate the pages
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_the_judge_model_appears(monkeypatch):
    """Pulling a judge model changes gating, so terminal pages must not resume."""
    _stub_tags(monkeypatch, INSTALLED)
    without = _pipeline()._run_fingerprint()

    _stub_tags(monkeypatch, INSTALLED + ["minicpm-v:8b"])
    with_judge = _pipeline()._run_fingerprint()

    assert without != with_judge


def test_heuristic_backend_does_not_probe(monkeypatch):
    """--judge-backend heuristic can't run a VLM; don't pay 3 round-trips to say so."""

    def _boom(*a, **k):
        raise AssertionError("heuristic backend must not probe Ollama")

    monkeypatch.setattr(httpx, "get", _boom)

    _pipeline(judge_backend="heuristic")._run_fingerprint()


@pytest.mark.parametrize("backend", ["auto", "vlm"])
def test_fingerprint_is_stable_across_repeated_calls(backend, monkeypatch):
    """Per-page sidecar flushes must not produce drifting fingerprints."""
    _stub_tags(monkeypatch, ["minicpm-v:8b"])
    pipe = _pipeline(judge_backend=backend)

    assert pipe._run_fingerprint() == pipe._run_fingerprint()
