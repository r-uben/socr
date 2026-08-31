"""TICKET-A2: CLI1 rung — the ollama-cloud table judge (GH-353).

Pins the `/api/chat` integration path chosen by the GH-356 bake-off
(`docs/log/2026-08-30_gh356-bakeoff.md`): base64 crop, `format="json"`,
`stream=False`, model/host/timeout from config (G1). Every test guards against
a real network call — an autouse fixture makes `httpx.post` raise if it is
ever reached directly, so a hole in `_post_chat` mocking fails loud instead of
hanging on a live daemon.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest

from socr.judge.table_rung_ollama import (
    _build_payload,
    _post_chat,
    _rung_id,
    build_ollama_rung,
)
from socr.judge.table_verdict import Finding, FindingCode, RungResult

PASS_JSON = '{"verdict": "PASS", "confidence": "high", "findings": []}'
FAIL_JSON = (
    '{"verdict": "FAIL", "confidence": "high", "findings": '
    '[{"code": "MISSING_VALUE", "where": "row 2, col B", "detail": "cell is blank"}]}'
)


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sentinel: any test that fails to stub `_post_chat` hits this instead
    of a live ollama daemon."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("network must not run in tests — mock _post_chat instead")

    monkeypatch.setattr(httpx, "post", _boom)


@pytest.fixture
def crop_path(tmp_path: Path) -> Path:
    path = tmp_path / "crop.png"
    path.write_bytes(b"not-a-real-png-but-bytes-are-enough-to-base64-encode")
    return path


class TestRungId:
    def test_rung_id_matches_the_documented_shape(self):
        assert _rung_id("glm-5.3-flash:cloud") == "ollama:glm-5.3-flash:cloud"


class TestPayload:
    def test_payload_carries_the_exact_bakeoff_contract(self):
        payload = _build_payload("glm-5.3-flash:cloud", "the prompt", "QkFTRTY0")
        assert payload["model"] == "glm-5.3-flash:cloud"
        assert payload["format"] == "json"
        assert payload["stream"] is False
        assert payload["messages"] == [
            {
                "role": "user",
                "content": "the prompt",
                "images": ["QkFTRTY0"],
            }
        ]


class TestBuildOllamaRung:
    def test_pass_verdict_round_trips(self, monkeypatch: pytest.MonkeyPatch, crop_path: Path):
        captured: dict[str, object] = {}

        def _fake_post_chat(host: str, payload: dict, timeout: float) -> str:
            captured["host"] = host
            captured["payload"] = payload
            captured["timeout"] = timeout
            return PASS_JSON

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _fake_post_chat)

        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a | b |\n| - | - |\n| 1 | 2 |", None)

        assert isinstance(result, RungResult)
        assert result.ok is True
        assert result.rung == "ollama:glm-5.3-flash:cloud"
        assert result.verdict is not None
        assert result.verdict.verdict == "PASS"
        assert result.verdict.findings == []

        # Exact outgoing JSON payload (model, format, images, stream=False).
        payload = captured["payload"]
        assert payload["model"] == "glm-5.3-flash:cloud"
        assert payload["format"] == "json"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1
        message = payload["messages"][0]
        assert message["role"] == "user"
        assert isinstance(message["images"], list) and len(message["images"]) == 1
        assert captured["timeout"] == 600.0

    def test_fail_verdict_round_trips(self, monkeypatch: pytest.MonkeyPatch, crop_path: Path):
        monkeypatch.setattr(
            "socr.judge.table_rung_ollama._post_chat",
            lambda host, payload, timeout: FAIL_JSON,
        )
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n|   |", None)

        assert result.ok is True
        assert result.verdict.verdict == "FAIL"
        assert result.verdict.findings[0].code is FindingCode.MISSING_VALUE

    def test_prior_findings_do_not_reach_the_prompt(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        """GH-359 ruling 4: even a caller that still passes findings must
        not leak them into the judge prompt."""
        captured: dict[str, object] = {}

        def _fake_post_chat(host: str, payload: dict, timeout: float) -> str:
            captured["prompt"] = payload["messages"][0]["content"]
            return PASS_JSON

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _fake_post_chat)

        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        prior = [Finding(code=FindingCode.WRONG_BINDING, where="row 3", detail="label shifted")]
        rung(crop_path, "| a |\n| - |\n| 1 |", prior)

        assert "row 3" not in captured["prompt"]
        assert "label shifted" not in captured["prompt"]
        assert "independently" in captured["prompt"].lower()

    def test_malformed_json_is_s1_failure_not_an_exception(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        monkeypatch.setattr(
            "socr.judge.table_rung_ollama._post_chat",
            lambda host, payload, timeout: "not json at all",
        )
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.verdict is None
        assert result.error

    def test_timeout_is_s1_failure_without_sleeping(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        def _raise_timeout(host: str, payload: dict, timeout: float) -> str:
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_timeout)

        slept = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.verdict is None
        assert "timed out" in result.error.lower() or "TimeoutException" in result.error
        assert slept == []

    def test_connection_error_is_s1_failure(self, monkeypatch: pytest.MonkeyPatch, crop_path: Path):
        def _raise_connect(host: str, payload: dict, timeout: float) -> str:
            raise httpx.ConnectError("no daemon")

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_connect)

        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.verdict is None

    def test_http_status_error_is_s1_failure(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        def _raise_status(host: str, payload: dict, timeout: float) -> str:
            request = httpx.Request("POST", "http://localhost:11434/api/chat")
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("404 not found", request=request, response=response)

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_status)

        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.verdict is None

    def test_missing_crop_file_is_s1_failure_not_an_exception(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The crop read (`Path.read_bytes`) sits inside the same try/except
        as the network call — a missing/unreadable crop must classify as ¬S1
        like any other rung failure, never propagate as OSError."""
        monkeypatch.setattr(
            "socr.judge.table_rung_ollama._post_chat",
            lambda host, payload, timeout: PASS_JSON,
        )
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        missing_crop = tmp_path / "does-not-exist.png"

        result = rung(missing_crop, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.verdict is None
        assert result.error

    def test_host_resolves_through_the_shared_gh222_resolver(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        monkeypatch.setenv("OLLAMA_HOST", "gpu-node:9999")
        captured: dict[str, object] = {}

        def _fake_post_chat(host: str, payload: dict, timeout: float) -> str:
            captured["host"] = host
            return PASS_JSON

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _fake_post_chat)

        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        rung(crop_path, "md", None)

        assert captured["host"] == "http://gpu-node:9999"

    def test_explicit_host_wins_over_env(self, monkeypatch: pytest.MonkeyPatch, crop_path: Path):
        monkeypatch.setenv("OLLAMA_HOST", "gpu-node:9999")
        captured: dict[str, object] = {}

        def _fake_post_chat(host: str, payload: dict, timeout: float) -> str:
            captured["host"] = host
            return PASS_JSON

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _fake_post_chat)

        rung = build_ollama_rung(
            "glm-5.3-flash:cloud", host="http://explicit-host:11434", timeout=600.0
        )
        rung(crop_path, "md", None)

        assert captured["host"] == "http://explicit-host:11434"


class TestNoRealPostChat:
    def test_real_post_chat_calls_httpx_post_and_hits_the_sentinel(self, crop_path: Path):
        """`_post_chat` itself is unmocked here — proves the autouse sentinel
        actually intercepts a real transport attempt rather than being a
        no-op guard that never fires."""
        with pytest.raises(AssertionError, match="network must not run"):
            _post_chat("http://localhost:11434", {"model": "x"}, 1.0)
