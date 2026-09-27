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

import importlib
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import socr.judge.ollama_judge as ollama_judge_module
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


def _run_killable_inprocess(spec, timeout):
    """Bypass the real ``multiprocessing`` spawn: call the probe body
    directly, in THIS process (GH-903 round 3, P2-b made ``is_available()``
    cross ``run_killable``'s real subprocess boundary). A real spawned child
    re-imports everything fresh, so an ``httpx.post`` monkeypatch made in the
    test process would never reach it -- every hermetic test in this file
    fakes ``run_killable`` this way instead. The real boundary (a genuine
    wall-clock deadline against a trickling peer) is exercised separately, in
    ``tests/test_gh172_judge_killable.py``, against a real server.

    Mirrors just enough of the real ``run_killable``/``_child_main``
    reclassification (``socr/core/killable.py``) to be a faithful stand-in:
    a timeout raised by the probe body is reclassified as
    ``KillableTimeoutError`` (a ``TimeoutError`` subclass), the same as the
    real boundary would, so ``is_available()``'s ``except TimeoutError``
    still fires.
    """
    from socr.core.killable import KillableTimeoutError

    module_name, _, qualname = spec.func.partition(":")
    fn = getattr(importlib.import_module(module_name), qualname)
    try:
        return fn(*spec.args, **(spec.kwargs or {}))
    except httpx.TimeoutException as exc:
        raise KillableTimeoutError(spec.func, timeout, killed=False) from exc


@pytest.fixture(autouse=True)
def _probe_run_killable_is_synchronous(monkeypatch):
    monkeypatch.setattr(ollama_judge_module, "run_killable", _run_killable_inprocess)


def _with_implicit_tag(name: str) -> str:
    """Mirror Ollama's own untagged -> ``:latest`` resolution for the stub."""
    return name if ":" in name else f"{name}:latest"


def _stub_generate(monkeypatch, names, on_call=None):
    """Make POST /api/generate (the real GH-903 probe) succeed for exactly
    ``names`` (matched on the full ``name:tag``, mirroring Ollama's own
    untagged -> ``:latest`` resolution) and 404 for anything else.

    ``/api/tags`` is no longer what availability means (GH-903): a retired
    Ollama Cloud model stayed listed there while every generation 410'd, so
    the probe now POSTs the exact generation call ``is_available`` sends and
    this stub answers THAT, never ``httpx.get``.
    """
    available = {_with_implicit_tag(n) for n in names}

    class _Resp:
        def __init__(self, status_code: int):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "http://x/api/generate")
                response = httpx.Response(
                    self.status_code,
                    request=request,
                    json={"error": "model not found"},
                )
                raise httpx.HTTPStatusError(
                    f"{self.status_code} error", request=request, response=response
                )

        def json(self):
            return {"response": "{}"}

    def _post(url, json=None, **kwargs):
        if on_call is not None:
            on_call()
        model = (json or {}).get("model", "")
        ok = _with_implicit_tag(model) in available
        return _Resp(200 if ok else 404)

    monkeypatch.setattr(httpx, "post", _post)


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
    # #885: this file is about page-judge wiring, not engine selection; pin
    # the engine so the AUTO default does not shell out to `ollama`.
    pinned = {
        "primary_engine": EngineType.QWEN,
        "local_engine": EngineType.QWEN,
        "enabled_engines": [EngineType.QWEN],
    }
    pinned.update(overrides)
    return UnifiedPipeline(PipelineConfig(**pinned))


# ---------------------------------------------------------------------------
# 1. Availability is an exact pull, not a family prefix
# ---------------------------------------------------------------------------


def test_installed_sibling_does_not_satisfy_a_different_tag(monkeypatch):
    """qwen3-vl:30b-a3b-instruct must NOT make qwen3-vl:8b look available.

    This is the trap: the prefix match reported the 8B judge as present, and the
    404 only surfaced later, at judge time, mid-document.
    """
    _stub_generate(monkeypatch, INSTALLED)
    assert OllamaVisionJudge(model="qwen3-vl:8b").is_available() is False


def test_exact_tag_is_available(monkeypatch):
    _stub_generate(monkeypatch, INSTALLED)
    assert OllamaVisionJudge(model="qwen3-vl:30b-a3b-instruct").is_available() is True


def test_untagged_reference_resolves_to_latest(monkeypatch):
    """Ollama treats a bare name as ``:latest``; availability must agree."""
    _stub_generate(monkeypatch, INSTALLED)
    assert OllamaVisionJudge(model="llama3").is_available() is True
    assert OllamaVisionJudge(model="qwen3-vl").is_available() is False


def test_unreachable_daemon_is_unavailable_not_an_error(monkeypatch):
    def _boom(*a, **k):
        raise httpx.ConnectError("no daemon")

    monkeypatch.setattr(httpx, "post", _boom)
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
    _stub_generate(monkeypatch, INSTALLED)  # none of the candidates are installed
    pipe = _pipeline()
    state = _State()

    pipe._build_page_judge(state)

    assert state.agentic_judge_model == JUDGE_IDENTITY_HEURISTIC


def test_degradation_emits_an_audit_event_under_default_backend(monkeypatch):
    """judge_backend defaults to "auto", where this used to be silent."""
    _stub_generate(monkeypatch, INSTALLED)
    pipe = _pipeline()
    assert pipe.config.judge_backend == "auto"
    state = _State()

    pipe._build_page_judge(state)

    kinds = [e.kind for e in state.events]
    assert "judge_degraded_to_heuristic" in kinds


def test_resolved_model_is_the_one_constructed(monkeypatch):
    """The judge must be built from the resolved model, never the module default."""
    _stub_generate(monkeypatch, ["minicpm-v:8b"])
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

    _stub_generate(
        monkeypatch, ["minicpm-v:8b"], on_call=lambda: calls.__setitem__("n", calls["n"] + 1)
    )
    pipe = _pipeline()

    for _ in range(5):
        pipe._resolve_judge_model()

    # JUDGE_MODEL_DEFAULT fails its probe, minicpm-v:8b succeeds -- 2 probes on
    # the FIRST (unmemoized) call, then 0 on the remaining 4.
    assert calls["n"] <= 2, f"expected memoized resolution, got {calls['n']} probes"


def test_explicit_judge_model_bypasses_the_ladder_not_the_probe(monkeypatch):
    """GH-903 round 3 (P2-a, cubic): the ladder never substitutes a different
    model for an explicit override, but a genuinely unreachable daemon still
    means "no judge" (heuristics), surfaced with a reason -- an override that
    silently ran unavailable used to fail on every page instead."""

    def _boom(*a, **k):
        raise httpx.ConnectError("daemon down")

    monkeypatch.setattr(httpx, "post", _boom)
    pipe = _pipeline(judge_model="my-judge:v2")

    assert pipe._resolve_judge_model() is None
    assert "my-judge:v2" in pipe._judge_unavailable_reason


# ---------------------------------------------------------------------------
# 4. The fingerprint tracks the judge that will actually gate the pages
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_the_judge_model_appears(monkeypatch):
    """Pulling a judge model changes gating, so terminal pages must not resume."""
    _stub_generate(monkeypatch, INSTALLED)
    without = _pipeline()._run_fingerprint()

    _stub_generate(monkeypatch, INSTALLED + ["minicpm-v:8b"])
    with_judge = _pipeline()._run_fingerprint()

    assert without != with_judge


def test_heuristic_backend_does_not_probe(monkeypatch):
    """--judge-backend heuristic can't run a VLM; don't pay 3 round-trips to say so."""

    def _boom(*a, **k):
        raise AssertionError("heuristic backend must not probe Ollama")

    monkeypatch.setattr(httpx, "post", _boom)

    _pipeline(judge_backend="heuristic")._run_fingerprint()


@pytest.mark.parametrize("backend", ["auto", "vlm"])
def test_fingerprint_is_stable_across_repeated_calls(backend, monkeypatch):
    """Per-page sidecar flushes must not produce drifting fingerprints."""
    _stub_generate(monkeypatch, ["minicpm-v:8b"])
    pipe = _pipeline(judge_backend=backend)

    assert pipe._run_fingerprint() == pipe._run_fingerprint()
