"""GH-903: Ollama Cloud retired the default page judge (`qwen3.5:cloud`, 2026-09-25).

`POST /api/generate` 410'd on every call while `/api/tags` kept listing the
model, so the old tags-based availability probe kept selecting a judge that
could never answer. Round 2 (owner review of 1ae4e36) found the FIX's own
probe timeout was too short: a cold-loaded `qwen3.8:27b` (unloaded, GPU
memory not resident) took ~46s to answer on the owner's Mac, and a 10s probe
budget misread that load as unavailability -- memoizing `None` for the whole
run. Changes pinned here:

1. The page judge (`_post_generate`, and the availability probe) sends
   `"think": false` -- a thinking-model candidate (`qwen3.8:27b`, the new
   default) otherwise puts its verdict in `thinking` and leaves the field
   socr parses empty. The TABLE judge / cell adjudicator (cloud thinking
   models with measured accuracy behind their reasoning) are deliberately
   UNCHANGED -- out of scope, pending a separate accuracy-measured ticket.
2. `_JUDGE_MODEL_CANDIDATES[0]` is `qwen3.8:27b` (local), not the retired
   cloud model.
3. Availability is a real 1-token generation, not a tags listing -- a model
   that answers 410/404 on generation is unavailable EVEN IF `/api/tags`
   would have said otherwise.
4. The probe's budget is the SAME as the judge call's own timeout
   (`OllamaVisionJudge.timeout`, default `DEFAULT_JUDGE_TIMEOUT_SEC`), not a
   separate shorter one -- a cold load must fit inside it. An HTTP error
   status is definitive unavailability; a timeout is not (it only proves the
   probe's budget, whatever it is, was exceeded) and is reported distinctly.
5. When every candidate fails, the recorded reason lists EVERY candidate's
   failure, not just the last one.

Hermetic by construction: every HTTP call is stubbed.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx

from socr.core.config import EngineType, PipelineConfig
from socr.judge.ollama_judge import DEFAULT_JUDGE_TIMEOUT_SEC, OllamaVisionJudge, _post_generate
from socr.judge.table_rung_ollama import _build_payload
from socr.pipeline.orchestrator import UnifiedPipeline


def _pipeline(**overrides):
    pinned = {
        "primary_engine": EngineType.QWEN,
        "local_engine": EngineType.QWEN,
        "enabled_engines": [EngineType.QWEN],
    }
    pinned.update(overrides)
    return UnifiedPipeline(PipelineConfig(**pinned))


# ---------------------------------------------------------------------------
# 1. think: false on the PAGE judge only -- the table judge is untouched
# ---------------------------------------------------------------------------


def test_page_judge_generate_call_sends_think_false(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": '{"verdict": "PASS"}'}

    def _post(url, json=None, **kwargs):
        captured.update(json or {})
        return _Resp()

    monkeypatch.setattr(httpx, "post", _post)

    _post_generate("http://localhost:11434", "qwen3.8:27b", "prompt", "aGVsbG8=", timeout=5.0)

    assert captured.get("think") is False


def test_page_judge_probe_sends_think_false(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": ""}

    def _post(url, json=None, **kwargs):
        captured.update(json or {})
        return _Resp()

    monkeypatch.setattr(httpx, "post", _post)

    assert OllamaVisionJudge(model="qwen3.8:27b").is_available() is True
    assert captured.get("think") is False


def test_table_judge_chat_payload_does_not_send_think(monkeypatch):
    """Owner ruling (round 2): the table judge ladder / cell adjudicator use
    cloud thinking models (`glm-5.3-flash:cloud`, `kimi-k2.6:cloud`) whose
    accuracy was measured WITH reasoning on -- turning it off is a separate,
    unmeasured ticket. This guards against `think: false` creeping back in
    here alongside a future page-judge change."""
    payload = _build_payload("glm-5.3-flash:cloud", "prompt", "aGVsbG8=")
    assert "think" not in payload


# ---------------------------------------------------------------------------
# 2. Cold start: a slow-but-real load must not be mistaken for unavailability
# ---------------------------------------------------------------------------


def _cold_load_stub(monkeypatch, load_time_sec):
    """Simulate a model that answers successfully, but only if the caller's
    timeout budget was at least ``load_time_sec`` -- the shape of a cold
    model load, without an actual sleep."""

    def _post(url, json=None, timeout=None, **kwargs):
        if timeout is None or timeout < load_time_sec:
            raise httpx.ReadTimeout(f"did not answer within {timeout}s")

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"response": ""}

        return _Resp()

    monkeypatch.setattr(httpx, "post", _post)


def test_cold_load_within_budget_is_available(monkeypatch):
    """Measured: an unloaded qwen3.8:27b took ~46s to answer. A probe given
    the judge call's own (120s) budget must not treat that as unavailable."""
    _cold_load_stub(monkeypatch, load_time_sec=46.0)

    judge = OllamaVisionJudge(model="qwen3.8:27b", timeout=DEFAULT_JUDGE_TIMEOUT_SEC)
    assert judge.is_available() is True
    assert judge.unavailable_reason == ""


def test_cold_load_beyond_a_short_budget_times_out_not_available(monkeypatch):
    """The failure mode round 2 found: a probe with a budget shorter than the
    cold load reports unavailable -- but the reason must say TIMEOUT, not a
    fabricated HTTP status, since the daemon never actually answered no."""
    _cold_load_stub(monkeypatch, load_time_sec=46.0)

    judge = OllamaVisionJudge(model="qwen3.8:27b", timeout=10.0)
    assert judge.is_available() is False
    assert "timed out" in judge.unavailable_reason
    assert "10" in judge.unavailable_reason


def test_probe_uses_the_judge_calls_own_timeout_not_a_shorter_one(monkeypatch):
    """Difference pin: same cold-load shape, only the configured ``timeout``
    changes -- selection must differ, proving the probe shares that budget
    rather than a separate, hardcoded one."""
    _cold_load_stub(monkeypatch, load_time_sec=46.0)

    short = OllamaVisionJudge(model="qwen3.8:27b", timeout=10.0).is_available()
    long = OllamaVisionJudge(model="qwen3.8:27b", timeout=DEFAULT_JUDGE_TIMEOUT_SEC).is_available()

    assert short is False
    assert long is True
    assert short != long


def test_an_http_error_status_is_definitive_regardless_of_budget(monkeypatch):
    """A 410 must still mean unavailable even with the full judge-call budget
    -- only a TIMEOUT is inconclusive, never an HTTP error status."""

    def _post(url, json=None, timeout=None, **kwargs):
        request = httpx.Request("POST", "http://x/api/generate")
        response = httpx.Response(410, request=request, json={"error": "retired"})
        raise httpx.HTTPStatusError("410 error", request=request, response=response)

    monkeypatch.setattr(httpx, "post", _post)

    judge = OllamaVisionJudge(model="qwen3.5:cloud", timeout=DEFAULT_JUDGE_TIMEOUT_SEC)
    assert judge.is_available() is False
    assert "410" in judge.unavailable_reason
    assert "timed out" not in judge.unavailable_reason


# ---------------------------------------------------------------------------
# 3. A 410 on the default candidate falls through, and every reason is kept
# ---------------------------------------------------------------------------


def _stub_generate_with_status(monkeypatch, status_for_model):
    """POST /api/generate answers per-model with the given HTTP status (or 200
    if not named; ``"TIMEOUT"`` raises instead), carrying a retirement-shaped
    error body on failure."""

    class _Resp:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "http://x/api/generate")
                response = httpx.Response(
                    self.status_code,
                    request=request,
                    json={"error": f"{status_for_model.get('_model', '')} was retired at ..."},
                )
                raise httpx.HTTPStatusError(
                    f"{self.status_code} error", request=request, response=response
                )

        def json(self):
            return {"response": "{}"}

    def _post(url, json=None, timeout=None, **kwargs):
        model = (json or {}).get("model", "")
        status_for_model["_model"] = model
        outcome = status_for_model.get(model, 200)
        if outcome == "TIMEOUT":
            raise httpx.ReadTimeout(f"{model} did not answer within {timeout}s")
        return _Resp(outcome)

    monkeypatch.setattr(httpx, "post", _post)


def test_retired_default_falls_through_to_the_next_candidate(monkeypatch):
    from socr.pipeline.orchestrator import UnifiedPipeline as _UP

    status = {_UP.JUDGE_MODEL_DEFAULT: 410, "minicpm-v:8b": 200}
    _stub_generate_with_status(monkeypatch, status)

    pipe = _pipeline()
    chosen = pipe._resolve_judge_model()

    assert chosen == "minicpm-v:8b"


def test_every_candidates_reason_is_recorded_not_just_the_last(monkeypatch):
    from socr.pipeline.orchestrator import UnifiedPipeline as _UP

    status = {_UP.JUDGE_MODEL_DEFAULT: "TIMEOUT", "minicpm-v:8b": 404, "qwen3-vl:8b": 410}
    _stub_generate_with_status(monkeypatch, status)

    pipe = _pipeline()
    chosen = pipe._resolve_judge_model()

    assert chosen is None
    reason = pipe._judge_unavailable_reason
    assert _UP.JUDGE_MODEL_DEFAULT in reason and "timed out" in reason
    assert "minicpm-v:8b" in reason and "404" in reason
    assert "qwen3-vl:8b" in reason and "410" in reason


def test_degradation_audit_event_carries_every_candidates_reason(monkeypatch):
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState
    from socr.pipeline.orchestrator import UnifiedPipeline as _UP

    status = {_UP.JUDGE_MODEL_DEFAULT: "TIMEOUT", "minicpm-v:8b": 404, "qwen3-vl:8b": 410}
    _stub_generate_with_status(monkeypatch, status)

    pipe = _pipeline(quiet=True)

    from test_p35_cold_review_round2 import _build_fixture_pdf
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        pdf = _build_fixture_pdf(Path(tmp))
        state = DocumentState(DocumentHandle(pdf))
        pipe._build_page_judge(state)

    events = [e for e in state.events if e.kind == "judge_degraded_to_heuristic"]
    assert events, "expected a degradation event when every candidate fails"
    reason = events[0].data["unavailable_reason"]
    assert _UP.JUDGE_MODEL_DEFAULT in reason and "timed out" in reason
    assert "minicpm-v:8b" in reason and "404" in reason
    assert "qwen3-vl:8b" in reason and "410" in reason


# ---------------------------------------------------------------------------
# 4. Difference pin: same tags listing, generation OK vs 410 -> different pick
# ---------------------------------------------------------------------------


def test_tags_listing_does_not_override_a_failing_generation(monkeypatch):
    """A model /api/tags reports as pulled must still be rejected if the exact
    generation the judge would send fails -- the GH-903 defect verbatim."""
    from socr.pipeline.orchestrator import UnifiedPipeline as _UP

    def _tags_always_says_everything_is_pulled(*a, **k):
        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "models": [
                        {"name": _UP.JUDGE_MODEL_DEFAULT},
                        {"name": "minicpm-v:8b"},
                        {"name": "qwen3-vl:8b"},
                    ]
                }

        return _Resp()

    monkeypatch.setattr(httpx, "get", _tags_always_says_everything_is_pulled)

    # Same tags listing in both runs; only whether generation succeeds changes.
    status_retired = {_UP.JUDGE_MODEL_DEFAULT: 410, "minicpm-v:8b": 200}
    _stub_generate_with_status(monkeypatch, status_retired)
    retired_chosen = _pipeline()._resolve_judge_model()

    status_healthy = {_UP.JUDGE_MODEL_DEFAULT: 200, "minicpm-v:8b": 200}
    _stub_generate_with_status(monkeypatch, status_healthy)
    healthy_chosen = _pipeline()._resolve_judge_model()

    assert retired_chosen == "minicpm-v:8b"
    assert healthy_chosen == _UP.JUDGE_MODEL_DEFAULT
    assert retired_chosen != healthy_chosen


# ---------------------------------------------------------------------------
# 5. strict_local still forbids cloud candidates
# ---------------------------------------------------------------------------


def test_strict_local_forbids_an_explicit_cloud_override():
    pipe = _pipeline(strict_local=True, judge_model="qwen3.5:cloud")
    pipe._judge_model_cache = False
    with patch("socr.judge.ollama_judge.OllamaVisionJudge.is_available", return_value=True):
        chosen = pipe._resolve_judge_model()
    assert "cloud" not in (chosen or "").lower()


def test_strict_local_permits_the_local_default(monkeypatch):
    from socr.pipeline.orchestrator import UnifiedPipeline as _UP

    _stub_generate_with_status(monkeypatch, {_UP.JUDGE_MODEL_DEFAULT: 200})
    pipe = _pipeline(strict_local=True)
    assert pipe._resolve_judge_model() == _UP.JUDGE_MODEL_DEFAULT
