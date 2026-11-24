"""#1405: the figure-caption leg.

Four defects, one section each:

1. ``OllamaFigureEngine`` ignored ``OLLAMA_HOST``.
2. Its calls passed no context size, so the daemon default sized the model.
3. The paid caption fallback was outside ``--max-paid-pages`` / ``--cost-budget``
   and left nothing in the cost journal.
4. A configured vLLM vision server was never used for captions.

Hermetic: no Ollama, no vLLM server, no Gemini call. Every probe and every HTTP
call is replaced. Outcomes are pinned as DIFFERENCES between two runs that
change one thing, not as values observed on one machine.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from ocr_output_contract import assemble_pages
from PIL import Image

from socr.core import providers
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import EngineResult, FigureInfo
from socr.core.state import DocumentState
from socr.engines._figure_prompt import CAPTION_MARKER
from socr.engines.gemini_api import (
    FIGURE_NUM_CTX_DEFAULT,
    FIGURE_NUM_CTX_ENV,
    PAID_CAPTION_REFUSED_PREFIX,
    BudgetedPaidFigureEngine,
    GeminiAPIConfig,
    GeminiAPIEngine,
    LocalFirstFigureEngine,
    OllamaFigureEngine,
    paid_caption_refused_mark,
)
from socr.engines.vllm import VLLMConfig, VLLMEngine
from socr.figures import crop_descriptions
from socr.figures.extractor import ExtractedFigure, ExtractionResult
from socr.pipeline import orchestrator as orch

_PINNED_ENGINES = {
    "enabled_engines": [EngineType.QWEN],
    "fallback_chain": [],
    "primary_engine": EngineType.QWEN,
    "local_engine": EngineType.QWEN,
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OLLAMA_HOST", FIGURE_NUM_CTX_ENV):
        monkeypatch.delenv(name, raising=False)


def _image() -> Image.Image:
    return Image.new("RGB", (8, 8), "white")


def _pipeline(**overrides: object) -> orch.UnifiedPipeline:
    config = PipelineConfig(quiet=True, **_PINNED_ENGINES)
    for key, value in overrides.items():
        assert hasattr(config, key), f"PipelineConfig has no field {key!r}"
        setattr(config, key, value)
    return orch.UnifiedPipeline(config)


def _state(pages: int = 3) -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=Path("/nonexistent/doc.pdf"), page_count=pages)
    return DocumentState(handle=handle)


# ---------------------------------------------------------------------------
# 1 -- the Ollama host
# ---------------------------------------------------------------------------


def test_engine_host_follows_ollama_host(monkeypatch):
    default_host = OllamaFigureEngine().host
    monkeypatch.setenv("OLLAMA_HOST", "gpu-node:9999")
    assert OllamaFigureEngine().host == "http://gpu-node:9999"
    assert OllamaFigureEngine().host != default_host


def test_explicit_host_beats_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "gpu-node:9999")
    assert OllamaFigureEngine(host="http://other:1234").host == "http://other:1234"


def test_probe_and_call_go_to_the_resolved_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "gpu-node:9999")
    urls: list[str] = []

    def _get(url, **kw):
        urls.append(url)
        return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", url))

    def _post(url, **kw):
        urls.append(url)
        return httpx.Response(
            200, json={"message": {"content": "a chart"}}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "get", _get)
    monkeypatch.setattr(httpx, "post", _post)
    engine = OllamaFigureEngine()
    engine.is_available()
    engine.describe_figure(_image())
    assert urls and all(u.startswith("http://gpu-node:9999/") for u in urls), urls


# ---------------------------------------------------------------------------
# 2 -- the context window
# ---------------------------------------------------------------------------


def _captured_payloads(monkeypatch, engine: OllamaFigureEngine) -> list[dict]:
    payloads: list[dict] = []

    def _post(url, **kw):
        payloads.append(kw["json"])
        return httpx.Response(
            200, json={"message": {"content": "a chart"}}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", _post)
    engine.describe_figure(_image())
    engine.ask(_image(), "describe")
    return payloads


def test_both_ollama_calls_carry_the_engines_context_window(monkeypatch):
    engine = OllamaFigureEngine()
    payloads = _captured_payloads(monkeypatch, engine)
    assert len(payloads) == 2
    assert [p["options"]["num_ctx"] for p in payloads] == [FIGURE_NUM_CTX_DEFAULT] * 2


def test_context_window_follows_argument_then_environment(monkeypatch):
    assert OllamaFigureEngine().num_ctx == FIGURE_NUM_CTX_DEFAULT
    other = FIGURE_NUM_CTX_DEFAULT // 2
    monkeypatch.setenv(FIGURE_NUM_CTX_ENV, str(other))
    env_engine = OllamaFigureEngine()
    assert env_engine.num_ctx == other
    assert _captured_payloads(monkeypatch, env_engine)[0]["options"]["num_ctx"] == other
    assert OllamaFigureEngine(num_ctx=other // 2).num_ctx == other // 2


@pytest.mark.parametrize("bad", ["", "  ", "zero", "0", "-4096", "1.5"])
def test_a_malformed_override_falls_back_to_the_default(monkeypatch, bad):
    monkeypatch.setenv(FIGURE_NUM_CTX_ENV, bad)
    assert OllamaFigureEngine().num_ctx == FIGURE_NUM_CTX_DEFAULT


# ---------------------------------------------------------------------------
# 3a -- metering one paid call
# ---------------------------------------------------------------------------


def test_usage_price_is_the_page_price_on_the_measured_usage():
    """One formula: the page price is this function on the measured mean usage."""
    per_page = (
        providers.gemini_usage_cost_usd(
            providers._GEMINI_MEASURED_INPUT_TOKENS, providers._GEMINI_MEASURED_OUTPUT_TOKENS
        )
        / providers._GEMINI_MEASURED_PAGES
    )
    assert per_page == pytest.approx(providers.GEMINI_COST_PER_PAGE_USD)


def _gemini_engine(monkeypatch, response=None, exc: Exception | None = None) -> GeminiAPIEngine:
    engine = GeminiAPIEngine(GeminiAPIConfig(api_key="k", model="m"))
    engine._initialized = True

    class _Client:
        def post(self, url, json):
            if exc is not None:
                raise exc
            return response

        def close(self):
            pass

    monkeypatch.setattr(engine, "_get_client", lambda: _Client())
    return engine


def _ok(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body, request=httpx.Request("POST", "http://g"))


_TEXT = {"candidates": [{"content": {"parts": [{"text": "a line chart"}]}}]}


def test_a_metered_call_costs_its_reported_usage(monkeypatch):
    usage = {"promptTokenCount": 1000, "candidatesTokenCount": 200, "thoughtsTokenCount": 50}
    engine = _gemini_engine(monkeypatch, _ok({**_TEXT, "usageMetadata": usage}))
    engine.describe_figure(_image())
    assert engine.last_call_cost_usd == pytest.approx(providers.gemini_usage_cost_usd(1000, 250))
    assert engine.last_call_cost_usd > 0


def test_thinking_tokens_are_billed(monkeypatch):
    base = {"promptTokenCount": 1000, "candidatesTokenCount": 200}
    without = _gemini_engine(monkeypatch, _ok({**_TEXT, "usageMetadata": base}))
    without.describe_figure(_image())
    with_thoughts = _gemini_engine(
        monkeypatch, _ok({**_TEXT, "usageMetadata": {**base, "thoughtsTokenCount": 300}})
    )
    with_thoughts.describe_figure(_image())
    assert with_thoughts.last_call_cost_usd > without.last_call_cost_usd


def test_a_response_without_usage_is_unknown_spend_not_zero(monkeypatch):
    engine = _gemini_engine(monkeypatch, _ok(_TEXT))
    engine.describe_figure(_image())
    assert engine.last_call_cost_usd is None


def test_a_transport_error_after_sending_is_unknown_spend(monkeypatch):
    engine = _gemini_engine(monkeypatch, exc=httpx.ReadTimeout("slow"))
    engine.describe_figure(_image())
    assert engine.last_call_cost_usd is None


def test_a_refused_request_and_an_unsent_one_cost_nothing(monkeypatch):
    refused = _gemini_engine(
        monkeypatch, httpx.Response(429, json={}, request=httpx.Request("POST", "http://g"))
    )
    refused.describe_figure(_image())
    assert refused.last_call_cost_usd == 0.0

    unsent = GeminiAPIEngine(GeminiAPIConfig(api_key="k", model="m"))
    unsent.last_call_cost_usd = None  # stale value from an earlier call
    monkeypatch.setattr(unsent, "initialize", lambda: False)
    unsent.describe_figure(_image())
    assert unsent.last_call_cost_usd == 0.0


# ---------------------------------------------------------------------------
# 3b -- the gate around the paid engine
# ---------------------------------------------------------------------------


class _Paid:
    name = "gemini-api"

    def __init__(self, cost: float | None = 0.25, raises: bool = False) -> None:
        self.calls = 0
        self.last_call_cost_usd = cost
        self._raises = raises

    def describe_figure(self, image, figure_type="unknown", context=""):
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return FigureInfo(0, 0, figure_type, description=f"{CAPTION_MARKER} paid", engine=self.name)

    def close(self):
        pass


def test_a_refused_call_sends_nothing_and_ships_the_mark():
    inner, recorded = _Paid(), []
    gate = BudgetedPaidFigureEngine(
        inner, admit=lambda: "the cap is used up", record=recorded.append
    )
    info = gate.describe_figure(_image())
    assert inner.calls == 0
    assert recorded == []
    assert info.description == paid_caption_refused_mark("the cap is used up")
    assert info.description.startswith(PAID_CAPTION_REFUSED_PREFIX)


def test_an_admitted_call_runs_and_is_recorded_once():
    inner, recorded = _Paid(cost=0.25), []
    gate = BudgetedPaidFigureEngine(inner, admit=lambda: None, record=recorded.append)
    info = gate.describe_figure(_image())
    assert inner.calls == 1
    assert recorded == [0.25]
    assert info.description.endswith("paid")


def test_a_call_that_raises_is_still_recorded():
    inner, recorded = _Paid(cost=None, raises=True), []
    gate = BudgetedPaidFigureEngine(inner, admit=lambda: None, record=recorded.append)
    with pytest.raises(RuntimeError):
        gate.describe_figure(_image())
    assert recorded == [None]


# ---------------------------------------------------------------------------
# 3c -- the orchestrator's caps
# ---------------------------------------------------------------------------


def _all_refusal_reasons() -> list[str]:
    reasons = []
    circuit = _pipeline()
    circuit._gemini_quota_circuit_open = True
    reasons.append(circuit._paid_caption_refusal(_state()))
    reasons.append(_pipeline(max_paid_pages=0)._paid_caption_refusal(_state()))
    spent = _state()
    budget = _pipeline(cost_budget=0.5)
    budget._caption_call_page = 1
    budget._record_paid_caption(spent, 0.75)
    reasons.append(budget._paid_caption_refusal(spent))
    return reasons


def test_every_refusal_reason_survives_the_floor_pages_caption_check():
    """A mark the number-free check rejects would be blanked on a floor page."""
    reasons = _all_refusal_reasons()
    assert all(reasons) and len(set(reasons)) == len(reasons), reasons
    for reason in reasons:
        mark = paid_caption_refused_mark(reason)
        assert crop_descriptions.find_number_tokens(mark) == [], mark
        assert crop_descriptions.clean_description(mark) == mark


def test_the_paid_page_cap_counts_caption_calls():
    state = _state()
    capped = _pipeline(max_paid_pages=1)
    uncapped = _pipeline()
    capped._caption_call_page = uncapped._caption_call_page = 2

    assert capped._admit_paid_caption(state) is None
    assert capped._paid_gemini_pages_used == 1
    second = capped._admit_paid_caption(state)
    assert second, "a second paid caption passed a cap of one"
    assert capped._paid_gemini_pages_used == 1, "a refused call was counted as spent"

    other = _state()
    assert uncapped._admit_paid_caption(other) is None
    assert uncapped._admit_paid_caption(other) is None

    refused = [e for e in state.events if e.kind == "figure_caption_paid_refused"]
    assert len(refused) == 1 and refused[0].page_num == 2
    assert not [e for e in other.events if e.kind == "figure_caption_paid_refused"]


def test_the_quota_circuit_refuses_the_caption_call():
    open_circuit, closed = _pipeline(), _pipeline()
    open_circuit._gemini_quota_circuit_open = True
    assert open_circuit._paid_caption_refusal(_state())
    assert closed._paid_caption_refusal(_state()) is None


def test_a_recorded_call_reaches_the_document_total_and_its_page():
    state = _state()
    pipe = _pipeline()
    assert state.total_cost == 0
    pipe._caption_call_page = 2
    pipe._record_paid_caption(state, 0.25)
    pipe._record_paid_caption(state, 0.5)
    assert state.total_cost == pytest.approx(0.75)
    assert state.pages[2].page_cost_usd == pytest.approx(0.75)
    assert state.pages[1].page_cost_usd == 0
    assert orch.UnifiedPipeline.PAID_CAPTION_RUN_ENGINE in state.engines_used


def test_an_unmetered_call_makes_the_total_unknown_and_closes_the_budget():
    state = _state()
    pipe = _pipeline(cost_budget=10.0)
    pipe._caption_call_page = 1
    assert pipe._paid_caption_refusal(state) is None
    pipe._record_paid_caption(state, None)
    assert state.total_cost is None
    assert state.pages[1].page_cost_usd is None
    assert pipe._paid_caption_refusal(state)


def test_the_cost_budget_refuses_once_spend_reaches_it():
    state = _state()
    pipe = _pipeline(cost_budget=0.5)
    pipe._caption_call_page = 1
    pipe._record_paid_caption(state, 0.25)
    assert pipe._paid_caption_refusal(state) is None
    pipe._record_paid_caption(state, 0.25)
    assert pipe._paid_caption_refusal(state)

    unbudgeted = _pipeline()
    assert unbudgeted._paid_caption_refusal(state) is None


def _engine_with_failing_ollama(monkeypatch, pipe: orch.UnifiedPipeline, paid_calls: list[int]):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(OllamaFigureEngine, "is_available", lambda self: True)
    monkeypatch.setattr(
        OllamaFigureEngine,
        "describe_figure",
        lambda self, image, figure_type="unknown", context="": FigureInfo(
            0, 0, figure_type, description="Ollama figure error: ConnectError: down"
        ),
    )
    monkeypatch.setattr(GeminiAPIEngine, "initialize", lambda self: True)

    def _paid(self, image, figure_type="unknown", context=""):
        paid_calls.append(1)
        self.last_call_cost_usd = 0.25
        return FigureInfo(0, 0, figure_type, description=f"{CAPTION_MARKER} paid", engine=self.name)

    monkeypatch.setattr(GeminiAPIEngine, "describe_figure", _paid)
    return pipe._get_vision_engine()


@pytest.mark.parametrize("ollama_up", [True, False])
def test_end_to_end_the_fallback_obeys_the_cap_and_journals_its_cost(monkeypatch, ollama_up):
    """Same figure, same failing local engine; only ``max_paid_pages`` differs."""
    outcomes = {}
    for label, overrides in (("open", {}), ("capped", {"max_paid_pages": 0})):
        pipe, state, calls = _pipeline(**overrides), _state(), []
        engine = _engine_with_failing_ollama(monkeypatch, pipe, calls)
        if not ollama_up:
            monkeypatch.setattr(OllamaFigureEngine, "is_available", lambda self: False)
            engine = pipe._get_vision_engine()
        pipe._caption_state, pipe._caption_call_page = state, 3
        info = engine.describe_figure(_image())
        outcomes[label] = (
            len(calls),
            info.description,
            state.total_cost,
            state.pages[3].page_cost_usd,
        )

    assert outcomes["open"] == (1, f"{CAPTION_MARKER} paid", 0.25, 0.25)
    calls, description, total, page = outcomes["capped"]
    assert calls == 0
    assert description.startswith(PAID_CAPTION_REFUSED_PREFIX)
    assert (total, page) == (0, 0)


def test_strict_local_never_builds_the_paid_engine(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(OllamaFigureEngine, "is_available", lambda self: False)
    built: list[int] = []
    monkeypatch.setattr(GeminiAPIEngine, "initialize", lambda self: built.append(1) or True)

    assert isinstance(_pipeline()._get_vision_engine(), BudgetedPaidFigureEngine)
    assert built == [1]
    assert _pipeline(strict_local=True)._get_vision_engine() is None
    assert built == [1], "the paid engine was initialised under --strict-local"


def _run_figure_phase(monkeypatch, tmp_path: Path, tag: str, **overrides):
    """Drive the real caption loop over two figures (pages 1 and 3), Ollama down."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(OllamaFigureEngine, "is_available", lambda self: False)
    monkeypatch.setattr(GeminiAPIEngine, "initialize", lambda self: True)

    def _paid(self, image, figure_type="unknown", context=""):
        self.last_call_cost_usd = 0.25
        return FigureInfo(0, 0, figure_type, description=f"{CAPTION_MARKER} paid", engine=self.name)

    monkeypatch.setattr(GeminiAPIEngine, "describe_figure", _paid)

    out = tmp_path / tag
    figures = []
    for num, page in ((1, 1), (2, 3)):
        path = out / "doc" / "figures" / f"figure_{num}_page{page}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        figures.append(
            ExtractedFigure(figure_num=num, page_num=page, image=_image(), saved_path=str(path))
        )

    pipe = _pipeline(describe_figures=True, save_figures=True, **overrides)
    state = _state(3)
    result = EngineResult(document_path=state.handle.path, engine="test")
    with patch("socr.pipeline.orchestrator.FigureExtractor") as extractor:
        extractor.return_value.extract.return_value = ExtractionResult(
            figures=figures, cap_reached=False
        )
        text = pipe._describe_and_embed_figures(
            state, result, out, assemble_pages(["P1.", "P2.", "P3."])
        )
    return pipe, state, result, text


def test_the_caption_loop_charges_each_page_and_marks_the_refused_figure(monkeypatch, tmp_path):
    """Same document twice; only ``max_paid_pages`` differs."""
    _, open_state, open_result, open_text = _run_figure_phase(monkeypatch, tmp_path, "open")
    pipe, state, result, text = _run_figure_phase(monkeypatch, tmp_path, "capped", max_paid_pages=1)

    assert [f.description for f in open_result.figures] == [f"{CAPTION_MARKER} paid"] * 2
    assert open_state.total_cost == pytest.approx(0.5)
    assert open_state.pages[1].page_cost_usd == pytest.approx(0.25)
    assert open_state.pages[3].page_cost_usd == pytest.approx(0.25)
    assert PAID_CAPTION_REFUSED_PREFIX not in open_text

    first, second = result.figures
    assert first.description == f"{CAPTION_MARKER} paid"
    assert second.description.startswith(PAID_CAPTION_REFUSED_PREFIX)
    assert second.description in text, "the refusal is not visible in the document body"
    assert state.total_cost == pytest.approx(0.25)
    assert state.pages[1].page_cost_usd == pytest.approx(0.25)
    assert state.pages[3].page_cost_usd == 0
    refused = [e for e in state.events if e.kind == "figure_caption_paid_refused"]
    assert [e.page_num for e in refused] == [3]
    assert pipe._caption_state is None and pipe._caption_call_page == 0


# ---------------------------------------------------------------------------
# 4 -- the vLLM tier
# ---------------------------------------------------------------------------

_VLLM = {
    "qwen_backend": "vllm",
    "qwen_vllm_url": "http://node:8123/v1",
    "qwen_vllm_model": "Org/Model",
}


def _models_response(ids: list[str], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, json={"data": [{"id": i} for i in ids]}, request=httpx.Request("GET", "http://v")
    )


def test_serves_model_needs_the_exact_id(monkeypatch):
    engine = VLLMEngine(VLLMConfig(base_url="http://node:8123/v1", model="Org/Model"))
    seen: list[str] = []

    def _get(url, **kw):
        seen.append(url)
        return _models_response(["Org/Model"])

    monkeypatch.setattr(httpx, "get", _get)
    assert engine.serves_model() is True
    assert seen == ["http://node:8123/v1/models"]

    monkeypatch.setattr(httpx, "get", lambda url, **kw: _models_response(["Org/Model-Other"]))
    assert (
        VLLMEngine(VLLMConfig(base_url="http://node:8123/v1", model="Org/Model")).serves_model()
        is False
    )


@pytest.mark.parametrize(
    "get",
    [
        lambda url, **kw: _models_response(["Org/Model"], status=503),
        lambda url, **kw: (_ for _ in ()).throw(httpx.ConnectError("refused")),
        lambda url, **kw: httpx.Response(200, text="not json", request=httpx.Request("GET", url)),
    ],
)
def test_serves_model_is_false_on_any_probe_failure(monkeypatch, get):
    monkeypatch.setattr(httpx, "get", get)
    engine = VLLMEngine(VLLMConfig(base_url="http://node:8123/v1", model="Org/Model"))
    assert engine.serves_model() is False


def test_the_vllm_tier_is_used_only_when_the_backend_is_vllm(monkeypatch):
    monkeypatch.setattr(VLLMEngine, "serves_model", lambda self: True)
    monkeypatch.setattr(OllamaFigureEngine, "is_available", lambda self: True)

    with_vllm = _pipeline(**_VLLM)._get_vision_engine()
    assert isinstance(with_vllm, LocalFirstFigureEngine)
    assert isinstance(with_vllm._ollama, VLLMEngine)
    assert with_vllm._ollama.config.base_url == _VLLM["qwen_vllm_url"]
    assert with_vllm._ollama.config.model == _VLLM["qwen_vllm_model"]

    same_url_other_backend = _pipeline(**{**_VLLM, "qwen_backend": "auto"})._get_vision_engine()
    assert isinstance(same_url_other_backend._ollama, OllamaFigureEngine)


def test_an_unserved_vllm_model_falls_through_to_ollama(monkeypatch):
    monkeypatch.setattr(VLLMEngine, "serves_model", lambda self: False)
    monkeypatch.setattr(OllamaFigureEngine, "is_available", lambda self: True)
    engine = _pipeline(**_VLLM)._get_vision_engine()
    assert isinstance(engine._ollama, OllamaFigureEngine)


@pytest.mark.parametrize(
    "local_description, expect_paid",
    [
        (f"{CAPTION_MARKER} a line chart of two series", False),
        ("vLLM API error (404)", True),
        ("vLLM request timed out after 120.0s", True),
        ("Unable to generate meaningful description", True),
        ("", True),
    ],
)
def test_a_failed_vllm_caption_goes_to_the_gated_fallback(
    monkeypatch, local_description, expect_paid
):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(VLLMEngine, "serves_model", lambda self: True)
    monkeypatch.setattr(
        VLLMEngine,
        "describe_figure",
        lambda self, image, figure_type="unknown", context="": FigureInfo(
            0, 0, figure_type, description=local_description, engine="vllm"
        ),
    )
    monkeypatch.setattr(GeminiAPIEngine, "initialize", lambda self: True)
    paid: list[int] = []

    def _paid(self, image, figure_type="unknown", context=""):
        paid.append(1)
        self.last_call_cost_usd = 0.25
        return FigureInfo(0, 0, figure_type, description=f"{CAPTION_MARKER} paid", engine=self.name)

    monkeypatch.setattr(GeminiAPIEngine, "describe_figure", _paid)

    pipe, state = _pipeline(**_VLLM), _state()
    engine = pipe._get_vision_engine()
    pipe._caption_state, pipe._caption_call_page = state, 1
    info = engine.describe_figure(_image())
    assert bool(paid) is expect_paid
    assert (state.total_cost > 0) is expect_paid
    assert info.description == (f"{CAPTION_MARKER} paid" if expect_paid else local_description)


def test_the_caption_identity_names_the_vllm_model_only_while_it_serves(monkeypatch):
    monkeypatch.setattr(OllamaFigureEngine, "is_available", lambda self: True)
    monkeypatch.setattr(VLLMEngine, "serves_model", lambda self: True)
    serving = _pipeline(**_VLLM)._resolve_caption_engine_identity()
    monkeypatch.setattr(VLLMEngine, "serves_model", lambda self: False)
    not_serving = _pipeline(**_VLLM)._resolve_caption_engine_identity()
    monkeypatch.setattr(VLLMEngine, "serves_model", lambda self: True)
    other_backend = _pipeline(
        **{**_VLLM, "qwen_backend": "auto"}
    )._resolve_caption_engine_identity()

    assert serving == f"vllm:{_VLLM['qwen_vllm_model']}"
    assert not_serving == other_backend
    assert serving != not_serving


def test_the_identity_and_the_engine_agree(monkeypatch):
    """The fingerprint must name the engine that would actually caption."""
    monkeypatch.setattr(OllamaFigureEngine, "is_available", lambda self: True)
    for serves in (True, False):
        monkeypatch.setattr(VLLMEngine, "serves_model", lambda self, _s=serves: _s)
        pipe = _pipeline(**_VLLM)
        local = pipe._get_vision_engine()._ollama
        identity = pipe._resolve_caption_engine_identity()
        assert identity.startswith("vllm:") is isinstance(local, VLLMEngine)
