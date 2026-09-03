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
        assert result.unavailable is False

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
        assert result.unavailable is False

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
        assert "multiple tables may be visible" not in captured["prompt"].lower()

    def test_page_scope_context_reaches_the_outgoing_prompt(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        """GH-373: the real CLI1 rung calls build_table_judge_prompt with no
        scope argument. The gate's context manager is the wire that splices
        the page-scope fragment into the payload."""
        from socr.judge.table_prompt import table_judge_prompt_scope

        captured: dict[str, object] = {}

        def _fake_post_chat(host: str, payload: dict, timeout: float) -> str:
            captured["prompt"] = payload["messages"][0]["content"]
            return PASS_JSON

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _fake_post_chat)

        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        with table_judge_prompt_scope("page"):
            rung(crop_path, "| a |\n| - |\n| 1 |", None)

        flat = " ".join(captured["prompt"].split()).lower()
        assert "multiple tables may be visible" in flat
        assert "whose content matches the emitted markdown" in flat

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
        assert result.unavailable is False

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
        assert result.unavailable is True

    def test_connection_error_is_s1_failure(self, monkeypatch: pytest.MonkeyPatch, crop_path: Path):
        def _raise_connect(host: str, payload: dict, timeout: float) -> str:
            raise httpx.ConnectError("no daemon")

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_connect)

        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.verdict is None
        assert result.unavailable is True

    def test_http_status_error_is_s1_failure(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        def _raise_status(host: str, payload: dict, timeout: float) -> str:
            request = httpx.Request("POST", "http://localhost:11434/api/chat")
            response = httpx.Response(
                404, request=request, text='{"error":"model not found, try pulling it first"}'
            )
            raise httpx.HTTPStatusError("404 not found", request=request, response=response)

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_status)

        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.verdict is None
        assert result.unavailable is True

    def test_missing_crop_file_is_s1_failure_not_an_exception(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The crop read (`Path.read_bytes`) sits inside the preparation block
        before the network call — a missing/unreadable crop must classify as ¬S1
        like any other rung failure, never propagate as OSError, and not be
        marked unavailable."""
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
        assert result.unavailable is False

    # -----------------------------------------------------------------
    # P1 prep item 1 (docs/log/2026-09-02_gh359-ladder-terminals-design.md,
    # "Panel and synthesis"): split the ¬S1 causes above by whether the rung
    # was actually reachable. httpx transport/status failures mean the host
    # could not be reached or refused the call -- exactly the shape the
    # retry latch exists for. A malformed JSON answer or an unreadable local
    # crop are NOT rung outages: the daemon is up and answering, so a retry
    # right now would hit the identical failure, and latching on it would
    # never converge.
    #
    # ``RungResult.unavailable`` carries that bit. Cold review round 2
    # narrowed the status branch: a response means the daemon ANSWERED, so
    # only statuses describing the service as unusable (5xx, 429, 408, and
    # 404 for a model that is not pulled) are outages. A 400 on our own
    # payload is a defect in this code and must not latch.
    # -----------------------------------------------------------------

    def test_timeout_is_unavailable(self, monkeypatch: pytest.MonkeyPatch, crop_path: Path):
        def _raise_timeout(host: str, payload: dict, timeout: float) -> str:
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_timeout)
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.unavailable is True

    def test_connection_error_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        def _raise_connect(host: str, payload: dict, timeout: float) -> str:
            raise httpx.ConnectError("no daemon")

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_connect)
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.unavailable is True

    def test_http_status_error_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        def _raise_status(host: str, payload: dict, timeout: float) -> str:
            request = httpx.Request("POST", "http://localhost:11434/api/chat")
            response = httpx.Response(
                404, request=request, text='{"error":"model not found, try pulling it first"}'
            )
            raise httpx.HTTPStatusError("404 not found", request=request, response=response)

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_status)
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.unavailable is True

    def test_bad_request_status_is_not_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        """Cold review round 2, finding 2. A 400 means the daemon answered and
        rejected OUR payload -- a deterministic defect in this code, not an
        outage. Latching it would re-run the ladder on every resume to send the
        same malformed request again."""

        def _raise_status(host: str, payload: dict, timeout: float) -> str:
            request = httpx.Request("POST", "http://localhost:11434/api/chat")
            response = httpx.Response(400, request=request)
            raise httpx.HTTPStatusError("400 bad request", request=request, response=response)

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_status)
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.unavailable is False

    @pytest.mark.parametrize("status_code", [429, 500, 503])
    def test_service_states_are_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path, status_code: int
    ):
        """Rate limiting and server errors describe the SERVICE as unusable,
        which is exactly what the latch is for."""

        def _raise_status(host: str, payload: dict, timeout: float) -> str:
            request = httpx.Request("POST", "http://localhost:11434/api/chat")
            response = httpx.Response(status_code, request=request)
            raise httpx.HTTPStatusError(str(status_code), request=request, response=response)

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_status)
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.unavailable is True

    @pytest.mark.parametrize("status", [401, 403, 407])
    def test_credential_refusals_are_outages_and_trip_the_breaker(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path, status: int
    ):
        """Cold review round 3. Credentials, a revoked token or a proxy can be
        restored, and until they are the rung cannot succeed -- exactly the
        state the latch remembers. They are also REFUSALS: the same run's next
        table would be refused identically, so the gate stops asking."""

        def _raise_status(host: str, payload: dict, timeout: float) -> str:
            request = httpx.Request("POST", "http://localhost:11434/api/chat")
            response = httpx.Response(status, request=request)
            raise httpx.HTTPStatusError(str(status), request=request, response=response)

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_status)
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.unavailable is True
        assert result.refusal is True

    def test_a_route_404_is_a_defect_not_a_missing_model(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        """404 is the one ambiguous status. A body that does not say the model
        is missing means we asked for a route that does not exist -- our own
        defect, identical forever."""

        def _raise_status(host: str, payload: dict, timeout: float) -> str:
            request = httpx.Request("POST", "http://localhost:11434/api/chatt")
            response = httpx.Response(404, request=request, text="404 page not found")
            raise httpx.HTTPStatusError("404", request=request, response=response)

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise_status)
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.unavailable is False

    @pytest.mark.parametrize(
        "exc,unavailable",
        [
            (httpx.ConnectError("refused"), True),
            (httpx.ReadTimeout("slow"), True),
            # Client configuration and response-decoding problems are ours.
            (httpx.UnsupportedProtocol("gopher://x"), False),
            (httpx.DecodingError("bad gzip"), False),
            (httpx.TooManyRedirects("loop"), False),
        ],
        ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else str(v),
    )
    def test_transport_errors_use_the_shared_classifier(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path, exc, unavailable: bool
    ):
        def _raise(host: str, payload: dict, timeout: float) -> str:
            raise exc

        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _raise)
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.unavailable is unavailable

    def test_malformed_json_is_not_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, crop_path: Path
    ):
        """The daemon answered; the answer was junk. Content-shaped, not a
        rung outage -- must not latch a retry."""
        monkeypatch.setattr(
            "socr.judge.table_rung_ollama._post_chat",
            lambda host, payload, timeout: "not json at all",
        )
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        result = rung(crop_path, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.unavailable is False

    def test_unreadable_crop_is_not_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """A local preparation failure (missing/unreadable crop) is not the
        rung being down -- the daemon was never even contacted."""
        monkeypatch.setattr(
            "socr.judge.table_rung_ollama._post_chat",
            lambda host, payload, timeout: PASS_JSON,
        )
        rung = build_ollama_rung("glm-5.3-flash:cloud", host=None, timeout=600.0)
        missing_crop = tmp_path / "does-not-exist.png"

        result = rung(missing_crop, "| a |\n| - |\n| 1 |", None)

        assert result.ok is False
        assert result.unavailable is False

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


class TestOllamaRungReachable:
    """Cold review round 2, finding 1: the resume gate's reachability notion
    for rung 1 -- daemon up AND the judge model actually pulled.

    A bare ``/api/tags`` liveness ping was not enough: a healthy daemon that
    never pulled the model answers it and then fails every judge call.
    """

    def _tags(self, monkeypatch: pytest.MonkeyPatch, models: list[str] | None, *, fail=None):
        def _get(url: str, timeout: float = 5.0):
            if fail is not None:
                raise fail
            request = httpx.Request("GET", url)
            return httpx.Response(
                200, request=request, json={"models": [{"name": m} for m in (models or [])]}
            )

        monkeypatch.setattr("socr.judge.table_rung_ollama.httpx.get", _get)

    def test_model_listed_is_reachable(self, monkeypatch: pytest.MonkeyPatch):
        from socr.judge.table_rung_ollama import ollama_rung_reachable

        self._tags(monkeypatch, ["glm-5.3-flash:cloud", "other:latest"])
        assert ollama_rung_reachable("glm-5.3-flash:cloud", "http://localhost:11434") is True

    def test_daemon_up_but_model_not_pulled_is_not_reachable(self, monkeypatch: pytest.MonkeyPatch):
        from socr.judge.table_rung_ollama import ollama_rung_reachable

        self._tags(monkeypatch, ["something-else:latest"])
        assert ollama_rung_reachable("glm-5.3-flash:cloud", "http://localhost:11434") is False

    def test_implicit_latest_tag_matches(self, monkeypatch: pytest.MonkeyPatch):
        from socr.judge.table_rung_ollama import ollama_rung_reachable

        self._tags(monkeypatch, ["judge:latest"])
        assert ollama_rung_reachable("judge", "http://localhost:11434") is True

    def test_a_family_prefix_does_not_satisfy_an_exact_tag(self, monkeypatch: pytest.MonkeyPatch):
        """#133's rule, reused: availability means the pull, not the family."""
        from socr.judge.table_rung_ollama import ollama_rung_reachable

        self._tags(monkeypatch, ["qwen3-vl:30b-a3b-instruct"])
        assert ollama_rung_reachable("qwen3-vl:8b", "http://localhost:11434") is False

    def test_transport_failure_is_not_reachable(self, monkeypatch: pytest.MonkeyPatch):
        from socr.judge.table_rung_ollama import ollama_rung_reachable

        self._tags(monkeypatch, None, fail=httpx.ConnectError("refused"))
        assert ollama_rung_reachable("glm-5.3-flash:cloud", "http://localhost:11434") is False
