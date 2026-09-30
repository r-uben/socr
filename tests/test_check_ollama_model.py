"""Unit tests for check_ollama_model (GH-910: HTTP ``/api/tags``, never the CLI).

Hermetic: no subprocess, no socket, no real HTTP. ``host_reachable`` and
``httpx.get`` are stubbed on the module under test.
"""

from __future__ import annotations

import os
import subprocess
import time

import httpx
import pytest

from socr.core import ollama_utils
from socr.core.ollama_utils import check_ollama_model as _check_ollama_model

_LISTED = [
    "qwen3-vl:30b-a3b-instruct",
    "qwen3-vl:8b",
    "deepseek-ocr:latest",
    "glm-ocr:latest",
]
_MODEL = "qwen3-vl:30b-a3b-instruct"


def _tags(names=_LISTED, key="name", status=200):
    return httpx.Response(
        status,
        json={"models": [{key: n} for n in names]},
        request=httpx.Request("GET", "http://x/api/tags"),
    )


@pytest.fixture
def http(monkeypatch):
    """Reachable host; ``http.calls`` records GETs; set ``http.result``."""

    class _H:
        calls: list = []
        result = None

    h = _H()
    h.calls = []
    h.result = _tags()

    def _get(url, **kw):
        h.calls.append((url, kw))
        if isinstance(h.result, BaseException):
            raise h.result
        return h.result

    monkeypatch.setattr(ollama_utils, "host_reachable", lambda host, *a, **k: True)
    monkeypatch.setattr(ollama_utils.httpx, "get", _get)
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.test:11434")
    return h


# (listed names, asked-for name, present?) -- the rules the ``ollama list``
# NAME-column compare had: exact string equality, tag included.
_MATCH_TABLE = [
    (_LISTED, "qwen3-vl:30b-a3b-instruct", True),
    (_LISTED, "qwen3-vl:8b", True),
    (_LISTED, "deepseek-ocr:latest", True),
    (_LISTED, "deepseek-ocr", False),  # no :latest defaulting
    (["deepseek-ocr:latest"], "deepseek-ocr", False),
    (["qwen3-vl"], "qwen3-vl:latest", False),
    (_LISTED, "qwen3-vl", False),  # no prefix match
    (_LISTED, "qwen3-vl:30b", False),  # a prefix of a listed tag is not it
    (["qwen3-vl:30b"], "qwen3-vl:30b-a3b-instruct", False),
    (_LISTED, "QWEN3-VL:8B", False),  # case-sensitive
    (_LISTED, "nonexistent:7b", False),
    ([], _MODEL, False),
]


@pytest.mark.parametrize("key", ["name", "model"])
@pytest.mark.parametrize(("listed", "asked", "present"), _MATCH_TABLE)
def test_matching_rules(http, listed, asked, present, key):
    http.result = _tags(listed, key=key)
    err = _check_ollama_model(asked)
    if present:
        assert err is None
    else:
        assert err is not None
        assert asked in err
        assert "ollama pull" in err


def test_requests_resolved_host_tags_endpoint(http):
    _check_ollama_model(_MODEL)
    assert [c[0] for c in http.calls] == ["http://ollama.test:11434/api/tags"]
    assert http.calls[0][1]["timeout"] == ollama_utils.TAGS_CHECK_TIMEOUT_SEC


def test_unreachable_makes_no_http_call(http, monkeypatch):
    monkeypatch.setattr(ollama_utils, "host_reachable", lambda host, *a, **k: False)
    err = _check_ollama_model(_MODEL)
    assert err is not None and "not running" in err.lower()
    assert http.calls == []


def test_http_error_status(http):
    http.result = _tags(status=500)
    err = _check_ollama_model(_MODEL)
    assert err is not None and "500" in err


def test_connection_error_after_reachable(http):
    http.result = httpx.ConnectError("boom")
    err = _check_ollama_model(_MODEL)
    assert err is not None and "not running" in err.lower()


def test_httpx_timeout(http):
    http.result = httpx.ReadTimeout("slow")
    err = _check_ollama_model(_MODEL)
    assert err is not None and "timeout" in err.lower()


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"nope": 1}', b'{"models": 3}'])
def test_malformed_json(http, body):
    http.result = httpx.Response(
        200, content=body, request=httpx.Request("GET", "http://x/api/tags")
    )
    err = _check_ollama_model(_MODEL)
    assert err is not None and "unreadable" in err


def test_trickling_peer_hits_total_deadline(http, monkeypatch):
    """httpx's timeout is per-read; the total deadline must still cut it off."""
    monkeypatch.setattr(ollama_utils, "TAGS_CHECK_TIMEOUT_SEC", 0.2)
    monkeypatch.setattr(ollama_utils.httpx, "get", lambda *a, **k: time.sleep(5))
    start = time.monotonic()
    err = _check_ollama_model(_MODEL)
    assert time.monotonic() - start < 2
    assert err is not None and "timeout" in err.lower()


def test_never_spawns_a_process(http, monkeypatch):
    """The CLI launches Ollama.app and steals focus; no exec of any kind."""

    def _boom(*a, **k):
        raise AssertionError("check_ollama_model must not spawn a process")

    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, _boom)
    monkeypatch.setattr("os.system", _boom)
    for name in ("posix_spawn", "posix_spawnp", "execv", "execvp", "execve", "fork"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, _boom)
    assert _check_ollama_model(_MODEL) is None
    http.result = _tags([])
    assert _check_ollama_model(_MODEL) is not None
    monkeypatch.setattr(ollama_utils, "host_reachable", lambda host, *a, **k: False)
    assert _check_ollama_model(_MODEL) is not None


def test_null_models_reads_as_not_found(http):
    """Older Ollama reports an empty store as ``{"models": null}``."""
    http.result = httpx.Response(
        200, content=b'{"models": null}', request=httpx.Request("GET", "http://x/api/tags")
    )
    err = _check_ollama_model(_MODEL)
    assert err is not None and "not found" in err
