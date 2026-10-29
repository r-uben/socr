"""#1254: Ollama cloud models reached by API key, with no daemon.

A cloud-tagged model (``glm-5.3-flash:cloud``) goes to Ollama's direct cloud API under its
cloud name (``glm-5.3-flash``) with ``Authorization: Bearer <OLLAMA_API_KEY>`` when the key
is set. Without the key, and for every local model, the call is exactly what it was.
The key is sent to the cloud endpoint only, and never appears in an error.

Hermetic: every test sets or clears ``OLLAMA_API_KEY`` itself and fakes ``httpx``.
Each behaviour is pinned as a DIFFERENCE between the key-set and key-unset runs.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import socr
from socr.core.ollama_utils import (
    OLLAMA_API_KEY_ENV,
    OLLAMA_CLOUD_ENDPOINT,
    cloud_api_model,
    ollama_endpoint,
    route_ollama,
    urllib_auth_headers,
)
from socr.judge import table_rung_ollama
from socr.judge.table_rung_ollama import _post_chat as REAL_POST_CHAT
from socr.judge.table_rung_ollama import build_ollama_rung, ollama_rung_reachable

KEY = "test-key-1254-not-real"
LOCAL = "http://localhost:11434"
CLOUD_MODEL = "glm-5.3-flash:cloud"


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture
def real_post_chat(monkeypatch):
    """The real transport seam: the suite's autouse fixture replaces ``_post_chat`` with a
    no-daemon stub, and these tests are about what the real one sends (httpx is faked)."""
    monkeypatch.setattr(table_rung_ollama, "_post_chat", REAL_POST_CHAT)


@pytest.fixture(params=[False, True], ids=["no-key", "key"])
def key_set(request, monkeypatch) -> bool:
    if request.param:
        monkeypatch.setenv(OLLAMA_API_KEY_ENV, KEY)
    else:
        monkeypatch.delenv(OLLAMA_API_KEY_ENV, raising=False)
    return request.param


@pytest.mark.parametrize(
    ("model", "api"),
    [
        ("glm-5.3-flash:cloud", "glm-5.3-flash"),
        ("kimi-k2.6:cloud", "kimi-k2.6"),
        ("gpt-oss:120b-cloud", "gpt-oss:120b"),
        ("qwen3-vl:30b-a3b-instruct", None),
        ("glm-5.3-flash", None),
    ],
)
def test_cloud_api_model_strips_only_the_cloud_tag(model: str, api: str | None) -> None:
    assert cloud_api_model(model) == api


def test_only_a_cloud_model_with_a_key_moves_to_the_cloud(key_set: bool) -> None:
    moved = route_ollama(LOCAL, CLOUD_MODEL)
    assert moved == ((OLLAMA_CLOUD_ENDPOINT, "glm-5.3-flash") if key_set else (LOCAL, CLOUD_MODEL))
    # A local model never leaves the daemon, key or not.
    assert route_ollama(LOCAL, "qwen3-vl:30b-a3b-instruct") == (LOCAL, "qwen3-vl:30b-a3b-instruct")


@pytest.mark.parametrize(
    "host",
    [
        LOCAL,
        "http://ollama.com",
        "https://ollama.com.evil.example",
        "https://evil.example/https://ollama.com",
        "https://user:pw@gpu-box:11434",
    ],
)
def test_the_key_is_never_sent_to_another_host(monkeypatch, host: str) -> None:
    monkeypatch.setenv(OLLAMA_API_KEY_ENV, KEY)
    _url, extra = ollama_endpoint(host, "/api/chat")
    assert KEY not in repr(extra.get("headers", {}))
    assert KEY not in repr(urllib_auth_headers(host))


def test_the_cloud_endpoint_gets_the_bearer_header_only_with_a_key(key_set: bool) -> None:
    url, extra = ollama_endpoint(OLLAMA_CLOUD_ENDPOINT, "/api/chat", strip_slash=True)
    assert url == "https://ollama.com/api/chat"
    expected = {"Authorization": f"Bearer {KEY}"} if key_set else None
    assert extra.get("headers") == expected
    assert (urllib_auth_headers(OLLAMA_CLOUD_ENDPOINT) or None) == expected


def test_basic_auth_userinfo_is_unchanged(key_set: bool) -> None:
    """GH-976's remote-daemon login keeps working, with or without a cloud key."""
    url, extra = ollama_endpoint("https://user:pw@gpu-box:11434", "/api/tags")
    assert url == "https://gpu-box:11434/api/tags"
    assert isinstance(extra["auth"], httpx.BasicAuth)
    assert "headers" not in extra


def _chat_response(url: str, content: str = '{"verdict": "faithful"}') -> httpx.Response:
    return httpx.Response(
        200,
        json={"message": {"role": "assistant", "content": content}},
        request=httpx.Request("POST", url),
    )


def test_post_chat_sends_a_cloud_model_to_the_cloud_api(
    monkeypatch, key_set: bool, real_post_chat
) -> None:
    calls: list[dict] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return _chat_response(url)

    monkeypatch.setattr(httpx, "post", fake_post)
    table_rung_ollama._post_chat(LOCAL, {"model": CLOUD_MODEL, "messages": []}, 5.0)
    (call,) = calls
    if key_set:
        assert call["url"] == "https://ollama.com/api/chat"
        assert call["json"]["model"] == "glm-5.3-flash"
        assert call["headers"] == {"Authorization": f"Bearer {KEY}"}
    else:
        assert call["url"] == f"{LOCAL}/api/chat"
        assert call["json"]["model"] == CLOUD_MODEL
        assert "headers" not in call


def test_reachability_probes_the_route_the_call_will_take(monkeypatch, key_set: bool) -> None:
    """The cloud lists ``glm-5.3-flash``, the daemon ``glm-5.3-flash:cloud``: each route
    must find the model under its own name."""
    listed = {
        "https://ollama.com/api/tags": [{"name": "glm-5.3-flash"}],
        f"{LOCAL}/api/tags": [{"name": CLOUD_MODEL}],
    }
    seen: list[str] = []

    def fake_get(url, **kwargs):
        seen.append(url)
        return httpx.Response(200, json={"models": listed[url]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert ollama_rung_reachable(CLOUD_MODEL, LOCAL) is True
    assert seen == (["https://ollama.com/api/tags"] if key_set else [f"{LOCAL}/api/tags"])


def test_a_rejected_key_is_a_refusal_and_never_appears_in_the_error(
    monkeypatch, tmp_path, real_post_chat
) -> None:
    monkeypatch.setenv(OLLAMA_API_KEY_ENV, KEY)
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"\x89PNG\r\n\x1a\n")

    def fake_post(url, **kwargs):
        return httpx.Response(401, text="unauthorized", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    result = build_ollama_rung(CLOUD_MODEL, LOCAL, 5.0)(crop, "| a | b |", None)
    assert result.ok is False
    assert result.unavailable is True and result.refusal is True
    assert KEY not in result.error


def test_preflight_names_the_route(monkeypatch, key_set: bool) -> None:
    from socr.core.config import PipelineConfig
    from socr.preflight import table_judge_routes

    config = PipelineConfig()
    config.table_judge_rung1_host = LOCAL
    config.table_judge_rung1_model = CLOUD_MODEL
    routes = table_judge_routes(config)
    if key_set:
        assert "rung 1: glm-5.3-flash via the ollama cloud API" in routes
    else:
        assert f"rung 1: {CLOUD_MODEL} via the ollama daemon at {LOCAL}" in routes
    assert KEY not in routes
