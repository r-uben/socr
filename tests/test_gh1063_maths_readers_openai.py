"""#1063: both maths readers reach an OpenAI-compatible (vLLM) server.

Hermetic: a real local HTTP server plays both the Ollama ``/api/generate`` API and
an OpenAI-compatible ``/v1/chat/completions`` API, so the transport itself is
exercised, not a patched reader. CI has no model; ``_available_engines_for_agentic``
is patched like every p4r lane test.

The pins are DIFFERENCES (CLAUDE.md, #257): the same page, Ollama fake versus
vLLM fake, must ship identical output; nothing here asserts a (status,
audit_passed) tuple measured on one machine.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from test_equation_lane_pipeline_p4r import (
    _EQ_SOURCE,
    _LATEX_CONTAINED,
    _make_pdf,
    _make_pipeline,
    _make_state,
    _page_view,
    _region,
)

pytestmark = pytest.mark.real_preflight

fitz = pytest.importorskip("fitz")

SERVED = "Qwen/Qwen3-VL-30B-A3B-Instruct"

_REAL_HTTPX_GET = httpx.get  # captured at import, before the autouse fixture fakes it


class _FakeServer:
    """Records every request; answers Ollama and OpenAI shapes with one LaTeX reply."""

    def __init__(
        self, latex: str = _LATEX_CONTAINED, finish_reason: str = "stop", status=200, raw=None
    ):
        self.raw = raw  # verbatim chat/completions body, for malformed-reply cases
        self.latex = latex
        self.finish_reason = finish_reason
        self.status = status
        self.requests: list[tuple[str, dict]] = []
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append((self.path, body))
                if outer.status != 200:
                    self.send_response(outer.status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if self.path.endswith("/chat/completions") and outer.raw is not None:
                    reply = outer.raw
                elif self.path.endswith("/chat/completions"):
                    reply = {
                        "choices": [
                            {
                                "message": {"content": outer.latex},
                                "finish_reason": outer.finish_reason,
                            }
                        ]
                    }
                else:
                    reply = {"response": outer.latex, "done": True}
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802 - the /models liveness listing
                data = json.dumps({"data": [{"id": SERVED}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):  # silence
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()

    def paths(self) -> list[str]:
        return [p for p, _ in self.requests]


@pytest.fixture
def server():
    made: list[_FakeServer] = []

    def _make(**kw):
        srv = _FakeServer(**kw)
        made.append(srv)
        return srv

    yield _make
    for srv in made:
        srv.close()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)


def _vllm_overrides(srv: _FakeServer) -> dict:
    return {
        "qwen_backend": "vllm",
        "qwen_vllm_url": f"{srv.url}/v1",
        "qwen_vllm_model": SERVED,
    }


def _ollama_overrides(srv: _FakeServer) -> dict:
    return {"ollama_host": srv.url}


# --- the P4-R equation region lane -----------------------------------------------------


def _run_region_lane(tmp_path: Path, **config):
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(tmp_path)
    pipeline = _make_pipeline(equation_region_lane=True, **config)
    state = _make_state(pdf)
    pipeline._last_assessment = state._last_assessment

    def _detect(page, page_num):
        from socr.math.detect_equations import EquationDetectionResult

        if page_num != 2:
            return EquationDetectionResult(page_num=page_num, regions=[], detection_time_s=0.0)
        return _region(page)

    def _routed(page_num, *args, **kwargs):
        out = PageOutput(
            page_num=page_num,
            text="| Row | A | B |\n| --- | --- | --- |\n| 0 | 0.10 | 0.20 |",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        return PageDecision(page_num=page_num, final_output=out, accepted=True)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (
        patch("socr.math.detect_equations.detect_display_equations", side_effect=_detect),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)
        pipeline._phase_assemble(state, out_dir)
    return state


def _kinds(state) -> set[str]:
    return {e.kind for e in state.events if e.page_num == 2}


def test_region_lane_reaches_the_openai_server_with_the_served_model(tmp_path, server):
    srv = server()
    state = _run_region_lane(tmp_path, **_vllm_overrides(srv))

    assert srv.paths() == ["/v1/chat/completions"]
    body = srv.requests[0][1]
    assert body["model"] == SERVED
    assert body["max_tokens"] > 1  # a real reply cap, not the canary's
    content = body["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    from socr.math.validate_latex import EQUATION_LATEX_PROMPT

    assert content[1]["text"] == EQUATION_LATEX_PROMPT
    assert "equation_region_reading_attached" in _kinds(state)


def test_region_lane_output_is_identical_on_ollama_and_vllm(tmp_path, server):
    """Pin the DIFFERENCE: same page, only the transport changes."""
    ollama_srv, vllm_srv = server(), server()
    ollama = _run_region_lane(tmp_path / "ollama", **_ollama_overrides(ollama_srv))
    vllm = _run_region_lane(tmp_path / "vllm", **_vllm_overrides(vllm_srv))

    assert ollama_srv.paths() == ["/api/generate"]
    assert vllm_srv.paths() == ["/v1/chat/completions"]
    assert _page_view(ollama, 2) == _page_view(vllm, 2)
    assert _kinds(ollama) == _kinds(vllm)
    assert "equation_region_reading_attached" in _kinds(vllm)


def test_region_lane_ollama_request_is_unchanged(tmp_path, server):
    srv = server()
    _run_region_lane(tmp_path, **_ollama_overrides(srv))
    path, body = srv.requests[0]
    assert path == "/api/generate"
    assert body["model"] == "qwen3-vl:30b-a3b-instruct"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0, "num_ctx": 8192}


def test_region_lane_pinned_model_is_the_one_sent(tmp_path, server):
    srv = server()
    overrides = _vllm_overrides(srv)
    overrides.update(qwen_model="my/pinned-model", qwen_model_pinned=True)
    _run_region_lane(tmp_path, **overrides)
    assert srv.requests[0][1]["model"] == "my/pinned-model"


def test_region_lane_other_lane_model_is_refused_not_replaced(tmp_path, server):
    srv = server()
    state = _run_region_lane(
        tmp_path, clean_equation_model="some-other:tag", **_vllm_overrides(srv)
    )
    assert srv.requests == []
    assert "equation_region_reading_unvalidated" in _kinds(state)


def test_region_lane_vllm_server_error_surfaces_and_retries_later(tmp_path, server):
    srv = server(status=500)
    state = _run_region_lane(tmp_path, **_vllm_overrides(srv))
    assert srv.paths() == ["/v1/chat/completions"]
    assert "equation_region_reading_unvalidated" in _kinds(state)
    assert state.pages[2].equation_lane_retry_pending is True


def test_region_lane_truncated_reply_is_not_attached(tmp_path, server):
    srv = server(finish_reason="length")
    state = _run_region_lane(tmp_path, **_vllm_overrides(srv))
    assert "equation_region_reading_attached" not in _kinds(state)
    assert "equation_region_reading_unvalidated" in _kinds(state)


def test_region_lane_without_a_url_or_model_is_still_refused(tmp_path, server):
    srv = server()
    state = _run_region_lane(
        tmp_path, qwen_backend="vllm", qwen_vllm_url=f"{srv.url}/v1", qwen_vllm_model=""
    )
    assert srv.requests == []
    assert "equation_region_reading_unvalidated" in _kinds(state)


def test_addressable_check_follows_the_backend(server):
    from socr.core.providers import PROFILE_QWEN_LOCAL

    srv = server()
    vllm = _make_pipeline(equation_region_lane=True, **_vllm_overrides(srv))
    assert vllm._equation_lane_backend_addressable(PROFILE_QWEN_LOCAL)
    gemini = _make_pipeline(equation_region_lane=True, qwen_backend="gemini")
    assert not gemini._equation_lane_backend_addressable(PROFILE_QWEN_LOCAL)
    blank = _make_pipeline(
        equation_region_lane=True, qwen_backend="vllm", qwen_vllm_model="", qwen_model=""
    )
    assert not blank._equation_lane_backend_addressable(PROFILE_QWEN_LOCAL)


# --- the corrupt-math recovery ---------------------------------------------------------


def _corrupt_page_run(tmp_path: Path, **config):
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState
    from socr.math import recover

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Prose before.", fontsize=11)
    page.insert_text((240, 200), _EQ_SOURCE, fontsize=11)
    pdf = tmp_path / "corrupt.pdf"
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc.save(str(pdf))
    doc.close()
    native = f"Prose before.\n\n{_EQ_SOURCE}\n"

    pipeline = _make_pipeline(equation_region_lane=False, recover_corrupt_math=True, **config)
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=1)
    state = DocumentState(handle=handle)
    state.pages[1] = PageState(
        page_num=1, is_born_digital=True, native_text=native, has_corrupt_math=True
    )
    group = recover._CorruptLineGroup(fitz.Rect(230, 185, 400, 215), [_EQ_SOURCE])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with patch.object(recover, "_recovery_groups", return_value=[group]):
        page_out = pipeline._recover_corrupt_math_page(state, 1, out_dir)
    return state, page_out


def _region_event(state):
    return next(e for e in state.events if e.kind == "corrupt_math_region_recovery")


def test_corrupt_math_reaches_the_openai_server(tmp_path, server):
    srv = server()
    state, page_out = _corrupt_page_run(tmp_path, **_vllm_overrides(srv))

    assert srv.paths() == ["/v1/chat/completions"]
    body = srv.requests[0][1]
    assert body["model"] == SERVED
    from socr.math.recover import CORRUPT_MATH_PROMPT

    assert body["messages"][0]["content"][1]["text"] == CORRUPT_MATH_PROMPT
    assert _LATEX_CONTAINED in page_out.text
    assert page_out.provider_model == SERVED
    assert page_out.provider_backend == "openai-compatible"
    assert _region_event(state).data["model_id"] == SERVED


def test_corrupt_math_page_output_is_identical_on_ollama_and_vllm(tmp_path, server):
    ollama_srv, vllm_srv = server(), server()
    _, ollama = _corrupt_page_run(tmp_path / "o", **_ollama_overrides(ollama_srv))
    _, vllm = _corrupt_page_run(tmp_path / "v", **_vllm_overrides(vllm_srv))

    assert ollama_srv.paths() == ["/api/generate"]
    assert (ollama.text, ollama.status, ollama.failure_mode, ollama.audit_passed) == (
        vllm.text,
        vllm.status,
        vllm.failure_mode,
        vllm.audit_passed,
    )
    assert "equations/" in vllm.text


def test_corrupt_math_ollama_request_is_unchanged(tmp_path, server):
    srv = server()
    _, page_out = _corrupt_page_run(tmp_path, **_ollama_overrides(srv))
    path, body = srv.requests[0]
    assert path == "/api/generate"
    assert body["model"] == "qwen3-vl:30b-a3b-instruct"
    assert body["options"] == {"temperature": 0, "num_ctx": 8192}
    assert page_out.provider_backend == "ollama-compatible"


def test_corrupt_math_server_down_surfaces_on_the_page(tmp_path, server):
    srv = server(status=500)
    state, page_out = _corrupt_page_run(tmp_path, **_vllm_overrides(srv))
    # Two attempts (the empty reply is retried once), then fail closed.
    assert srv.paths() == ["/v1/chat/completions"] * 2
    assert _region_event(state).data["unresolved_regions"] == 1
    assert "corrupt equation unresolved" in page_out.text


@pytest.mark.parametrize("backend", ["vllm", "ollama"])
def test_corrupt_math_strict_local_cloud_model_sends_nothing(tmp_path, server, backend):
    srv = server()
    overrides = _vllm_overrides(srv) if backend == "vllm" else _ollama_overrides(srv)
    state, page_out = _corrupt_page_run(
        tmp_path, strict_local=True, math_model="qwen3.5:cloud", **overrides
    )
    assert srv.requests == []
    assert "strict-local" in page_out.skip_reason


def test_corrupt_math_other_model_is_refused_not_replaced(tmp_path, server):
    srv = server()
    _, page_out = _corrupt_page_run(tmp_path, math_model="local-other:7b", **_vllm_overrides(srv))
    assert srv.requests == []
    assert "not served by the OpenAI-compatible server" in page_out.skip_reason


# --- the clean-equation (legacy sidecar) path shares the reader ------------------------


def _legacy_run(tmp_path: Path, **config):
    """Drive ``_attach_equation_latex_sidecars`` (production wiring) over one crop."""
    from socr.core.audit_log import AuditEvent
    from socr.core.result import PageOutput, PageStatus

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = _make_pdf(tmp_path)
    crop = tmp_path / "c.png"
    crop.write_bytes(fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), False).tobytes("png"))
    pipeline = _make_pipeline(equation_region_lane=False, **config)
    state = _make_state(pdf)
    state.events.append(
        AuditEvent(
            page_num=2,
            kind="equation_region_detected",
            engine="native+equations",
            detail="detected",
            data={"crop_path": str(crop), "source_text": _EQ_SOURCE},
        )
    )
    po = PageOutput(page_num=2, text="native", status=PageStatus.SUCCESS, engine="native")
    pipeline._attach_equation_latex_sidecars(state, [po], page_nums=[2])
    return state, po


def test_legacy_clean_equation_path_reaches_the_openai_server(tmp_path, server):
    srv = server()
    state, po = _legacy_run(tmp_path, **_vllm_overrides(srv))
    assert srv.paths() == ["/v1/chat/completions"]
    assert srv.requests[0][1]["model"] == SERVED
    assert _LATEX_CONTAINED in po.text


def test_legacy_clean_equation_other_model_is_recorded_not_dropped(tmp_path, server):
    srv = server()
    state, po = _legacy_run(tmp_path, clean_equation_model="local-other:7b", **_vllm_overrides(srv))
    assert srv.requests == []
    kinds = {e.kind for e in state.events if e.page_num == 2}
    assert "equation_sidecar_skipped_model_refused" in kinds
    assert state.pages[2].equation_sidecar_skipped is True


# --- P0-1: cloud policy applies to the model that will actually be called ---------------

_CLOUD_PINNED = {"qwen_model": "x:cloud", "qwen_model_pinned": True}
_POLICIES = {
    "strict-local": {"strict_local": True},
    "zero-cap": {"max_cost_per_page": 0.0, "max_cost_per_page_pinned": True},
}


@pytest.mark.parametrize("policy", list(_POLICIES))
def test_pinned_cloud_served_model_is_blocked_on_every_path(tmp_path, server, policy):
    from socr import preflight

    srv = server()
    cfg = {**_vllm_overrides(srv), **_CLOUD_PINNED, **_POLICIES[policy]}

    _, page_out = _corrupt_page_run(tmp_path / "c", **cfg)
    state, _po = _legacy_run(tmp_path / "l", **cfg)
    region = _run_region_lane(tmp_path / "r", **cfg)

    assert srv.requests == []
    assert "forbids remote model x:cloud" in page_out.skip_reason
    assert "equation_sidecar_skipped_model_refused" in {e.kind for e in state.events}
    assert "equation_region_reading_unvalidated" in _kinds(region)

    pipeline = _make_pipeline(equation_region_lane=True, recover_corrupt_math=True, **cfg)
    check = preflight.check_maths_readers(pipeline.config, pipeline)
    assert check.missing and "x:cloud" in check.detail
    assert srv.requests == []


# --- P0-2: the OpenAI path refuses prose and strips reasoning ---------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "I cannot read this image.",
        "I cannot read x_t.",
        "<think>the crop is blank</think>\nSorry, nothing to transcribe.",
        "<think>never closed",
        "<think>maybe x = 1 + y_2?</think>\nI cannot read this image.",
    ],
)
def test_prose_and_bare_reasoning_replies_are_not_candidates(tmp_path, server, reply):
    srv = server(latex=reply)
    state, page_out = _corrupt_page_run(tmp_path, **_vllm_overrides(srv))
    assert "corrupt equation unresolved" in page_out.text
    assert _region_event(state).data["unresolved_regions"] == 1
    assert _region_event(state).data["reader_failures"]


def test_reasoning_wrapped_latex_keeps_only_the_latex(tmp_path, server):
    srv = server(latex="<think>reading the crop</think>\n```latex\n" + _LATEX_CONTAINED + "\n```")
    _, page_out = _corrupt_page_run(tmp_path, **_vllm_overrides(srv))
    assert _LATEX_CONTAINED in page_out.text
    assert "think" not in page_out.text and "reading the crop" not in page_out.text


# --- P1-3: an explicit OpenAI backend with nothing configured never falls to Ollama ------


def test_unconfigured_openai_backend_is_refused_not_sent_to_ollama(tmp_path, server):
    from socr import preflight

    ollama = server()
    cfg = {"qwen_backend": "vllm", "qwen_vllm_url": "", "ollama_host": ollama.url}
    _, page_out = _corrupt_page_run(tmp_path / "c", **cfg)
    state, _ = _legacy_run(tmp_path / "l", **cfg)
    region = _run_region_lane(tmp_path / "r", **cfg)

    assert ollama.requests == []
    assert "not falling back to Ollama" in page_out.skip_reason
    assert "equation_sidecar_skipped_model_refused" in {e.kind for e in state.events}
    assert "equation_region_reading_unvalidated" in _kinds(region)
    pipeline = _make_pipeline(equation_region_lane=True, **cfg)
    assert preflight.check_maths_readers(pipeline.config, pipeline).missing


# --- P1-4: malformed replies stay inside the failure contract ---------------------------


@pytest.mark.parametrize(
    "raw",
    [
        {"choices": {"message": {"content": "x = 1"}}},
        {"choices": [{"message": "x = 1"}]},
        {"choices": [{"message": {"content": ["x = 1"]}}]},
        {"choices": []},
        ["not", "an", "object"],
    ],
    ids=["choices-dict", "message-str", "content-list", "no-choices", "array-body"],
)
def test_malformed_replies_fail_closed_with_a_recorded_reason(tmp_path, server, raw):
    srv = server(raw=raw)
    state, page_out = _corrupt_page_run(tmp_path, **_vllm_overrides(srv))
    assert "corrupt equation unresolved" in page_out.text
    assert _region_event(state).data["reader_failures"]
    assert "reader failures" in _region_event(state).detail


# --- P1-6: the preflight canary needs a parsable reply, not just a 2xx ------------------


@pytest.mark.parametrize("raw", [{"choices": []}, {}, {"choices": [{"message": "x"}]}])
def test_preflight_rejects_a_2xx_without_a_reply(server, raw):
    from socr import preflight

    srv = server(raw=raw)
    pipeline = _make_pipeline(equation_region_lane=True, **_vllm_overrides(srv))
    with patch.object(httpx, "get", _REAL_HTTPX_GET):
        check = preflight.check_maths_readers(pipeline.config, pipeline)
    assert check.missing and "no generation answered" in check.detail


# --- the real `auto` + VLLM_BASE_URL transport ------------------------------------------


def test_auto_backend_with_vllm_base_url_reaches_the_server(tmp_path, server, monkeypatch):
    srv = server()
    monkeypatch.setenv("VLLM_BASE_URL", f"{srv.url}/v1")
    state = _run_region_lane(tmp_path / "r")
    _, page_out = _corrupt_page_run(tmp_path / "c")
    assert srv.paths() == ["/v1/chat/completions"] * 2
    assert {r[1]["model"] for r in srv.requests} == {SERVED}
    assert "equation_region_reading_attached" in _kinds(state)
    assert page_out.provider_backend == "openai-compatible"


# --- preflight -------------------------------------------------------------------------


def test_preflight_vllm_reachable_is_present_and_unreachable_is_missing(server):
    from socr import preflight

    srv = server()
    pipeline = _make_pipeline(
        equation_region_lane=True, recover_corrupt_math=True, **_vllm_overrides(srv)
    )
    config = pipeline.config
    real_get = _REAL_HTTPX_GET
    with (
        patch.object(preflight, "probe_model_generation") as ollama_probe,
        patch.object(httpx, "get", real_get),  # conftest fakes "no daemon" for GET
    ):
        up = preflight.check_maths_readers(config, pipeline)
        ollama_probe.assert_not_called()
        assert up.needed and up.ok and SERVED in up.detail
        assert srv.requests[0][1]["model"] == SERVED

        srv.close()
        down = preflight.check_maths_readers(config, pipeline)
    assert down.missing and "no generation answered" in down.detail


def test_preflight_vllm_other_lane_model_is_missing_without_a_request(server):
    from socr import preflight

    srv = server()
    pipeline = _make_pipeline(
        equation_region_lane=True, math_model="local-other:7b", **_vllm_overrides(srv)
    )
    check = preflight.check_maths_readers(pipeline.config, pipeline)
    assert check.missing and "local-other:7b" in check.detail
    assert srv.requests == []


def test_preflight_ollama_is_unchanged(server):
    from socr import preflight

    pipeline = _make_pipeline(equation_region_lane=True)
    with patch.object(preflight, "probe_model_generation", return_value=(True, "")) as probe:
        check = preflight.check_maths_readers(pipeline.config, pipeline)
    assert check.ok and probe.called


# --- the prose rule: two consecutive plain words, outside \\text-style contents ----------


@pytest.mark.parametrize(
    "reply",
    [
        "0.25",
        "-1",
        "a/b",
        r"\alpha",
        "x_t",
        r"\log x",
        r"\text{for all } x",
        r"\operatorname{rank} A",
    ],
)
def test_marker_free_and_text_macro_readings_are_accepted(reply):
    from socr.math.recover import clean_openai_reply

    assert clean_openai_reply(reply) == reply


@pytest.mark.parametrize("reply", ["I cannot read x_t.", "the image is blank", "No equation here"])
def test_prose_is_rejected(reply):
    from socr.math.recover import MathsReplyError, clean_openai_reply

    with pytest.raises(MathsReplyError):
        clean_openai_reply(reply)


def test_marker_free_reading_is_attached_end_to_end(tmp_path, server):
    srv = server(latex="0.25")
    _, page_out = _corrupt_page_run(tmp_path, **_vllm_overrides(srv))
    assert "0.25" in page_out.text and "corrupt equation unresolved" not in page_out.text


# --- legacy cloud refusal is latched like the others -----------------------------------


@pytest.mark.parametrize("policy", list(_POLICIES))
def test_legacy_cloud_clean_equation_model_is_latched(tmp_path, server, policy):
    srv = server()
    state, _po = _legacy_run(
        tmp_path, clean_equation_model="x:cloud", **_POLICIES[policy], **_vllm_overrides(srv)
    )
    assert srv.requests == []
    assert "equation_sidecar_skipped_model_refused" in {e.kind for e in state.events}
    assert state.pages[2].equation_sidecar_skipped is True


# --- canary: content must be non-empty text --------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": "  "}}]},
    ],
    ids=["no-content", "null-content", "blank-content"],
)
def test_preflight_rejects_empty_content(server, raw):
    from socr import preflight

    srv = server(raw=raw)
    pipeline = _make_pipeline(equation_region_lane=True, **_vllm_overrides(srv))
    with patch.object(httpx, "get", _REAL_HTTPX_GET):
        assert preflight.check_maths_readers(pipeline.config, pipeline).missing


# --- the model and backend actually used are what is recorded --------------------------


def _fingerprint_extra(pipeline) -> dict:
    with (
        patch("ocr_output_contract.run_fingerprint", side_effect=lambda *a, extra, **k: extra),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        return pipeline._run_fingerprint()


def test_fingerprint_and_provenance_name_the_served_model_on_vllm_only(server):
    import types

    srv = server()
    vllm = _make_pipeline(
        equation_region_lane=True, recover_corrupt_math=True, **_vllm_overrides(srv)
    )
    ollama = _make_pipeline(equation_region_lane=True, recover_corrupt_math=True)
    v, o = _fingerprint_extra(vllm), _fingerprint_extra(ollama)
    assert v["math_model"] == v["clean_equation_model"] == SERVED
    assert v["maths_backend"] == "openai-compatible"
    assert o["math_model"] == o["clean_equation_model"] == "qwen3-vl:30b-a3b-instruct"
    assert "maths_backend" not in o

    state = types.SimpleNamespace(engine_runs=[types.SimpleNamespace(engine="native+math")])
    assert vllm._fingerprint_inputs(state)["native+math"][:2] == (SERVED, "openai-compatible")
    assert ollama._fingerprint_inputs(state)["native+math"][:2] == (
        "qwen3-vl:30b-a3b-instruct",
        "ollama-compatible",
    )


# --- round 4: escaped braces and contractions -------------------------------------------

_SNEAKY_PROSE = [
    r"\text{\{} I cannot read this image.",
    "I'm sorry.",
    "I can't read this",
    r"\text{unclosed I cannot read",
]


@pytest.mark.parametrize("reply", _SNEAKY_PROSE)
def test_sneaky_prose_is_rejected_end_to_end_on_the_legacy_path(tmp_path, server, reply):
    srv = server(latex=reply)
    state, po = _legacy_run(tmp_path, **_vllm_overrides(srv))
    assert srv.paths() == ["/v1/chat/completions"]
    kinds = {e.kind for e in state.events if e.page_num == 2}
    assert "equation_latex_accepted" not in kinds
    assert "cannot read" not in po.text and "sorry" not in po.text


@pytest.mark.parametrize("reply", _SNEAKY_PROSE)
def test_sneaky_prose_is_rejected_on_the_other_two_paths(tmp_path, server, reply):
    srv = server(latex=reply)
    _, page_out = _corrupt_page_run(tmp_path / "c", **_vllm_overrides(srv))
    region = _run_region_lane(tmp_path / "r", **_vllm_overrides(srv))
    assert "corrupt equation unresolved" in page_out.text
    assert "equation_region_reading_attached" not in _kinds(region)


@pytest.mark.parametrize(
    "reply", [r"\{x\}", r"\text{\{a\}}", r"\text{\{a\}} x", r"\text{if {a} b} x"]
)
def test_escaped_braces_and_nested_text_are_accepted(reply):
    from socr.math.recover import clean_openai_reply

    assert clean_openai_reply(reply) == reply


def test_escaped_brace_does_not_hide_prose_behind_a_later_closing_brace(tmp_path, server):
    """An escaped ``\\{`` is a character: counting it as nesting lets a later stray ``}``
    close the macro at the end of the reply and blank the whole sentence."""
    reply = r"\text{\{} I cannot read this image }"
    from socr.math.recover import MathsReplyError, clean_openai_reply

    with pytest.raises(MathsReplyError):
        clean_openai_reply(reply)
    srv = server(latex=reply)
    state, po = _legacy_run(tmp_path, **_vllm_overrides(srv))
    assert "equation_latex_accepted" not in {e.kind for e in state.events}


def test_unbalanced_text_macro_blanks_nothing():
    from socr.math.recover import _blank_text_macro_contents

    assert "I cannot read" in _blank_text_macro_contents(r"\text{unclosed I cannot read")
    assert "I cannot read" not in _blank_text_macro_contents(r"\text{closed I cannot read}")
