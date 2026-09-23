"""GH-873: the judge must be reachable on a box that serves the VLM through vLLM.

Before this, ``_JUDGE_MODEL_CANDIDATES`` was Ollama-only, so a machine with no
Ollama daemon -- the Bocconi HPC nodes, where vLLM already serves the vision
model inside the same job -- could never resolve a judge and every run degraded
to the heuristic judge with nothing in ``metadata.json`` recording it.

Every test here is hermetic: no Ollama, no network, no provider. The HTTP layer
is patched at ``socr.judge.vllm_judge.httpx``, and the pipeline is built with
``object.__new__`` so no constructor side effect can reach out.

The pins are differences, not absolute pipeline outcomes: what changes when the
vLLM pair is configured versus when it is not. An absolute outcome would be a
provider-dependent value that CI cannot reproduce.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from socr.core.config import PipelineConfig
from socr.judge import vllm_judge
from socr.judge.vllm_judge import VLLMVisionJudge, _image_media_type, _post_chat
from socr.pipeline.orchestrator import UnifiedPipeline

SERVED = "Qwen/Qwen3-VL-30B-A3B-Instruct"
URL = "http://127.0.0.1:8000/v1"


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _models_payload(*ids):
    return {"data": [{"id": i} for i in ids]}


# --------------------------------------------------------------------------
# is_available: the exact-id guard
# --------------------------------------------------------------------------


def test_unset_model_is_never_available(monkeypatch):
    """An empty model must not be guessed from whatever the server happens to serve.

    Guessing is exactly the #133 bug in the Ollama backend: availability
    reported true, then the call 404'd at judge time on a model nobody asked
    for. With no model there is nothing to match, so the answer is no.
    """
    called = []
    monkeypatch.setattr(
        vllm_judge.httpx,
        "get",
        lambda *a, **k: called.append(a) or _Resp(_models_payload(SERVED)),
    )
    assert VLLMVisionJudge(model="", base_url=URL).is_available() is False
    assert called == [], "an unset model must not even probe the server"


def test_available_only_on_an_exact_served_id(monkeypatch):
    """The guard: a family prefix must NOT satisfy a request for another model.

    Both halves matter. Without the negative case the check could be
    ``return True`` and still pass; without the positive case it could be
    ``return False``.
    """
    monkeypatch.setattr(vllm_judge.httpx, "get", lambda *a, **k: _Resp(_models_payload(SERVED)))

    assert VLLMVisionJudge(model=SERVED, base_url=URL).is_available() is True
    assert VLLMVisionJudge(model="Qwen/Qwen3-VL-8B-Instruct", base_url=URL).is_available() is False
    # A prefix of the served id is a different model, not the same one.
    assert VLLMVisionJudge(model="Qwen/Qwen3-VL-30B", base_url=URL).is_available() is False


def test_unreachable_server_is_unavailable_not_an_exception(monkeypatch):
    import httpx as real_httpx

    def _boom(*a, **k):
        raise real_httpx.ConnectError("refused")

    monkeypatch.setattr(vllm_judge.httpx, "get", _boom)
    assert VLLMVisionJudge(model=SERVED, base_url=URL).is_available() is False


def test_base_url_trailing_slash_does_not_double(monkeypatch):
    seen = {}

    def _capture(url, **kwargs):
        seen["url"] = url
        return _Resp(_models_payload(SERVED))

    monkeypatch.setattr(vllm_judge.httpx, "get", _capture)
    VLLMVisionJudge(model=SERVED, base_url=URL + "/").is_available()
    assert seen["url"] == f"{URL}/models"


# --------------------------------------------------------------------------
# the wire format
# --------------------------------------------------------------------------


def test_post_chat_sends_an_openai_vision_message(monkeypatch):
    seen = {}

    def _capture(url, json=None, timeout=None):  # noqa: A002 - httpx's own kwarg name
        seen["url"] = url
        seen["json"] = json
        return _Resp({"choices": [{"message": {"content": '{"verdict": "OK"}'}}]})

    monkeypatch.setattr(vllm_judge.httpx, "post", _capture)
    out = _post_chat(URL, SERVED, "PROMPT", "data:image/png;base64,AAA", 12.0)

    assert out == '{"verdict": "OK"}'
    assert seen["url"] == f"{URL}/chat/completions"
    body = seen["json"]
    assert body["model"] == SERVED
    assert body["temperature"] == 0, "judging must not sample"
    content = body["messages"][0]["content"]
    kinds = [part["type"] for part in content]
    assert kinds == ["text", "image_url"]
    assert content[0]["text"] == "PROMPT"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,AAA"


def test_post_chat_returns_empty_string_when_the_server_sends_no_choices(monkeypatch):
    """A malformed reply must not raise out of the killable child."""
    monkeypatch.setattr(vllm_judge.httpx, "post", lambda *a, **k: _Resp({"choices": []}))
    assert _post_chat(URL, SERVED, "P", "data:image/png;base64,AAA", 1.0) == ""

    monkeypatch.setattr(vllm_judge.httpx, "post", lambda *a, **k: _Resp({}))
    assert _post_chat(URL, SERVED, "P", "data:image/png;base64,AAA", 1.0) == ""


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("p.png", "image/png"),
        ("p.PNG", "image/png"),
        ("p.jpg", "image/jpeg"),
        ("p.jpeg", "image/jpeg"),
        ("p.webp", "image/webp"),
        ("p.bin", "image/png"),
    ],
)
def test_media_type_follows_the_suffix(name, expected):
    """A JPEG must not be announced as a PNG: some servers decode strictly."""
    assert _image_media_type(Path(name)) == expected


def test_judge_builds_a_data_uri_from_the_rendered_file(monkeypatch, tmp_path):
    """The bytes on disk reach the server, base64'd, under the right media type."""
    img = tmp_path / "page.jpg"
    img.write_bytes(b"\xff\xd8not-a-real-jpeg")

    captured = {}

    def _fake_run_killable(spec, timeout):
        captured["spec"] = spec
        captured["timeout"] = timeout
        return '{"verdict": "OK", "confidence": 1.0}'

    monkeypatch.setattr(vllm_judge, "run_killable", _fake_run_killable)
    judge = VLLMVisionJudge(model=SERVED, base_url=URL, timeout=7.0)
    judge.judge(img, "SOME OCR TEXT")

    spec = captured["spec"]
    assert spec.func == "socr.judge.vllm_judge:_post_chat", (
        "the model call must stay a top-level function so run_killable can kill it"
    )
    base_url, model, prompt, data_uri, timeout = spec.args
    assert (base_url, model) == (URL, SERVED)
    assert captured["timeout"] == 7.0 and timeout == 7.0
    assert data_uri.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(data_uri.split(",", 1)[1]) == img.read_bytes()
    assert "SOME OCR TEXT" in prompt


# --------------------------------------------------------------------------
# resolution and selection: the DIFFERENCE the config makes
# --------------------------------------------------------------------------


def _pipeline(config: PipelineConfig) -> UnifiedPipeline:
    """A pipeline with no constructor side effects, as the orchestrator's own
    comments say tests build one."""
    p = object.__new__(UnifiedPipeline)
    p.config = config
    return p


def test_resolver_returns_the_vllm_model_without_probing_ollama(monkeypatch):
    """The whole point: no Ollama daemon, and a judge model still resolves.

    ``OllamaVisionJudge`` is replaced with something that explodes on
    construction, so a resolver that still probed the candidate ladder could
    not pass this quietly.
    """
    import socr.judge.ollama_judge as oj

    class _Explode:
        def __init__(self, *a, **k):
            raise AssertionError("the Ollama ladder must not be probed when vLLM is named")

    monkeypatch.setattr(oj, "OllamaVisionJudge", _Explode)

    cfg = PipelineConfig(judge_vllm_url=URL, judge_vllm_model=SERVED)
    assert _pipeline(cfg)._resolve_judge_model() == SERVED


def test_the_vllm_pair_is_what_changes_the_resolution(monkeypatch):
    """Difference pin, parametrised over the one thing under test.

    Same process, same patched (absent) Ollama, changing only whether the pair
    is configured. Without it the resolver finds nothing; with it, the named
    model.
    """
    import socr.judge.ollama_judge as oj

    class _Absent:
        def __init__(self, *a, **k):
            pass

        def is_available(self):
            return False

    monkeypatch.setattr(oj, "OllamaVisionJudge", _Absent)

    without = _pipeline(PipelineConfig())._resolve_judge_model()
    with_pair = _pipeline(
        PipelineConfig(judge_vllm_url=URL, judge_vllm_model=SERVED)
    )._resolve_judge_model()

    assert without is None
    assert with_pair == SERVED
    assert without != with_pair


def test_half_a_pair_is_not_a_configuration(monkeypatch):
    """A url with no model (or the reverse) must not resolve to something.

    Otherwise a partially-set run would name a judge it cannot call, which is
    the provenance lie #133 fixed for the Ollama backend.
    """
    import socr.judge.ollama_judge as oj

    class _Absent:
        def __init__(self, *a, **k):
            pass

        def is_available(self):
            return False

    monkeypatch.setattr(oj, "OllamaVisionJudge", _Absent)

    assert _pipeline(PipelineConfig(judge_vllm_url=URL))._resolve_judge_model() is None
    assert _pipeline(PipelineConfig(judge_vllm_model=SERVED))._resolve_judge_model() is None


def test_strict_local_does_not_forbid_an_operator_named_server(monkeypatch):
    """--strict-local must not strip the judge on a box that has only vLLM.

    The locality rule is about cloud identities in the candidate ladder. A
    named endpoint is an operator instruction, like --qwen-vllm-url, and
    forbidding it would leave the HPC case with no judge again -- the exact
    condition this ticket exists to remove.
    """
    import socr.judge.ollama_judge as oj

    class _Absent:
        def __init__(self, *a, **k):
            pass

        def is_available(self):
            return False

    monkeypatch.setattr(oj, "OllamaVisionJudge", _Absent)
    cfg = PipelineConfig(judge_vllm_url=URL, judge_vllm_model=SERVED, strict_local=True)
    assert _pipeline(cfg)._resolve_judge_model() == SERVED


def test_cli_flags_reach_the_config():
    """The flags must actually land on the config, or everything above is inert."""
    from click.testing import CliRunner

    from socr.cli import cli

    result = CliRunner().invoke(cli, ["process", "--help"])
    assert result.exit_code == 0
    assert "--judge-vllm-url" in result.output
    assert "--judge-vllm-model" in result.output


def test_config_defaults_leave_the_pair_unset():
    cfg = PipelineConfig()
    assert cfg.judge_vllm_url == ""
    assert cfg.judge_vllm_model == ""
    assert json.dumps({"url": cfg.judge_vllm_url})  # serialisable, no sentinel object
