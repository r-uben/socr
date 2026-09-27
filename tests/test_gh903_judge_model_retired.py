"""GH-903: Ollama Cloud retired the default page judge (`qwen3.5:cloud`, 2026-09-25).

`POST /api/generate` 410'd on every call while `/api/tags` kept listing the
model, so the old tags-based availability probe kept selecting a judge that
could never answer. Three changes, pinned here:

1. Every judge request (page judge `_post_generate`/probe, table judge/cell
   adjudicator `_build_payload`) sends `"think": false` -- a thinking-model
   candidate (`qwen3.8:27b`, the new default) otherwise puts its verdict in
   `message.thinking`/`thinking` and leaves the field socr parses empty.
2. `_JUDGE_MODEL_CANDIDATES[0]` is `qwen3.8:27b` (local), not the retired
   cloud model.
3. Availability is a real 1-token generation, not a tags listing -- a model
   that answers 410/404 on generation is unavailable EVEN IF `/api/tags`
   would have said otherwise, and the ladder records why.

Hermetic by construction: every HTTP call is stubbed.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx

from socr.core.config import EngineType, PipelineConfig
from socr.judge.ollama_judge import OllamaVisionJudge, _post_generate
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
# 1. think: false on every judge request
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


def test_table_judge_chat_payload_sends_think_false():
    """Covers both the table judge rung 1 AND the cell adjudicator, which
    build their `/api/chat` body through this same function
    (`cell_transcribe.transcribe_cell` calls it via `table_rung_ollama`)."""
    payload = _build_payload("glm-5.3-flash:cloud", "prompt", "aGVsbG8=")
    assert payload["think"] is False


# ---------------------------------------------------------------------------
# 2. A 410 on the default candidate falls through, and the reason is recorded
# ---------------------------------------------------------------------------


def _stub_generate_with_status(monkeypatch, status_for_model):
    """POST /api/generate answers per-model with the given HTTP status (or 200
    if not named), carrying a retirement-shaped error body on failure."""

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

    def _post(url, json=None, **kwargs):
        model = (json or {}).get("model", "")
        status_for_model["_model"] = model
        return _Resp(status_for_model.get(model, 200))

    monkeypatch.setattr(httpx, "post", _post)


def test_retired_default_falls_through_to_the_next_candidate(monkeypatch):
    from socr.pipeline.orchestrator import UnifiedPipeline as _UP

    status = {_UP.JUDGE_MODEL_DEFAULT: 410, "minicpm-v:8b": 200}
    _stub_generate_with_status(monkeypatch, status)

    pipe = _pipeline()
    chosen = pipe._resolve_judge_model()

    assert chosen == "minicpm-v:8b"


def test_retired_default_reason_is_surfaced(monkeypatch):
    from socr.pipeline.orchestrator import UnifiedPipeline as _UP

    status = {_UP.JUDGE_MODEL_DEFAULT: 410, "minicpm-v:8b": 410, "qwen3-vl:8b": 410}
    _stub_generate_with_status(monkeypatch, status)

    pipe = _pipeline()
    chosen = pipe._resolve_judge_model()

    assert chosen is None
    assert "410" in pipe._judge_unavailable_reason
    assert "retired" in pipe._judge_unavailable_reason


def test_degradation_audit_event_carries_the_410_reason(monkeypatch):
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState
    from socr.pipeline.orchestrator import UnifiedPipeline as _UP

    status = {_UP.JUDGE_MODEL_DEFAULT: 410, "minicpm-v:8b": 410, "qwen3-vl:8b": 410}
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
    assert events, "expected a degradation event when every candidate 410s"
    assert "410" in events[0].data["unavailable_reason"]


# ---------------------------------------------------------------------------
# 3. Difference pin: same tags listing, generation OK vs 410 -> different pick
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
# 4. strict_local still forbids cloud candidates
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
