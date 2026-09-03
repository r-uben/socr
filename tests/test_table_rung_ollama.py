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


# ==========================================================================
# P1: the blind cell-transcription ADJUDICATOR, on this same transport.
#
# Cold review round 1, finding 4. The adjudicator used to be reached through
# ``cursor-agent -p`` with the crop named as ``@<path>`` inside the prompt
# text -- a transport nobody had proven. A live probe disproved it: on a
# synthetic crop whose target cell reads 0.058, ``cursor-agent -p --model
# kimi-k3-max`` answered 0.92 (it never received the image), while the same
# crop POSTed to the ollama daemon was read correctly by ``kimi-k2.6:cloud``
# (0.058) and by ``glm-5.3-flash:cloud`` (0.058). A blind reader that never
# saw the crop can still emit a schema-valid, guessable token and clear a
# table nobody looked at, so these tests exist to keep the PIXELS on the wire.
# ==========================================================================


class TestAdjudicatorIdentity:
    def test_the_identity_is_the_configured_model_never_a_constant(self):
        from socr.judge.table_rung_ollama import adjudicator_rung_id, make_ollama_cell_adjudicator
        from socr.judge.table_verdict import RUNG_KIND_CELL_ADJUDICATOR, rung_kind

        adj = make_ollama_cell_adjudicator("some-other-model:cloud", None, 5.0)
        assert adj.rung_id == adjudicator_rung_id("some-other-model:cloud")
        assert adj.executing == "some-other-model:cloud"
        # Its own KIND, distinct from the reader rung that shares the
        # transport -- otherwise the latch could not tell them apart.
        assert adj.rung_kind == RUNG_KIND_CELL_ADJUDICATOR
        assert rung_kind(adj.rung_id) == RUNG_KIND_CELL_ADJUDICATOR
        assert rung_kind(adj.rung_id) != rung_kind(_rung_id("some-other-model:cloud"))


class TestAdjudicatorPayload:
    def test_the_request_carries_the_crop_bytes_and_only_coordinates(self, crop_path: Path):
        import base64
        import json

        from socr.judge.table_rung_ollama import make_ollama_cell_adjudicator

        seen = {}

        def _fake_post(host, payload, timeout):
            seen["payload"] = payload
            return json.dumps({"R1C2": "11", "H1C1": "yr"})

        adj = make_ollama_cell_adjudicator("kimi-k2.6:cloud", "http://h:1", 7.0)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("socr.judge.table_rung_ollama._post_chat", _fake_post)
            result = adj(crop_path, ["R1C2", "H1C1"])

        assert result.ok is True
        assert result.tokens == {"R1C2": "11", "H1C1": "yr"}
        payload = seen["payload"]
        assert payload["model"] == "kimi-k2.6:cloud"
        assert payload["format"] == "json"
        assert payload["stream"] is False
        # The pixels, verbatim.
        assert payload["messages"][0]["images"] == [
            base64.b64encode(crop_path.read_bytes()).decode("ascii")
        ]
        # Coordinates, and nothing from THIS table that could be agreed with.
        # The spliced coordinate grammar carries a fixed worked example (round
        # 2, N1) that is identical for every table and corroborates nothing, so
        # it is excluded -- and asserted present, so it cannot be dropped to
        # make this pass.
        from socr.judge.table_verdict import load_cell_ref_grammar

        prompt = payload["messages"][0]["content"]
        grammar = load_cell_ref_grammar()
        assert grammar in prompt
        assert "R1C2" in prompt and "H1C1" in prompt
        assert "11" not in prompt.replace(grammar, "")

    def test_the_public_call_has_no_parameter_for_the_extraction(self):
        import inspect

        from socr.judge.table_rung_ollama import transcribe_cells_ollama

        params = set(inspect.signature(transcribe_cells_ollama).parameters)
        assert params == {"crop_path", "cell_refs", "model", "host", "timeout"}


class TestAdjudicatorClassification:
    """Typed unavailability, inherited from the reader rung verbatim."""

    def _adj(self):
        from socr.judge.table_rung_ollama import make_ollama_cell_adjudicator

        return make_ollama_cell_adjudicator("kimi-k2.6:cloud", "http://h:1", 5.0)

    def _run(self, crop_path, raiser_or_text):
        def _fake_post(host, payload, timeout):
            if isinstance(raiser_or_text, BaseException):
                raise raiser_or_text
            return raiser_or_text

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("socr.judge.table_rung_ollama._post_chat", _fake_post)
            return self._adj()(crop_path, ["R1C2"])

    def test_a_connection_error_is_an_outage(self, crop_path: Path):
        result = self._run(crop_path, httpx.ConnectError("no daemon"))
        assert result.ok is False
        assert result.unavailable is True
        assert result.refusal is False

    def test_a_quota_status_is_an_outage_AND_a_refusal(self, crop_path: Path):
        response = httpx.Response(429, request=httpx.Request("POST", "http://h:1/api/chat"))
        result = self._run(
            crop_path, httpx.HTTPStatusError("429", request=response.request, response=response)
        )
        assert result.unavailable is True
        assert result.refusal is True

    def test_a_malformed_body_is_a_defect_that_never_latches(self, crop_path: Path):
        result = self._run(crop_path, "this is not json")
        assert result.ok is False
        assert result.unavailable is False
        assert result.refusal is False

    def test_a_partial_answer_is_a_defect_not_a_partial_clearance(self, crop_path: Path):
        """Strict by construction: clearing a table on fewer cells than were
        asked about would answer a question nobody put."""
        import json

        def _fake_post(host, payload, timeout):
            return json.dumps({"R1C2": "11"})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("socr.judge.table_rung_ollama._post_chat", _fake_post)
            result = self._adj()(crop_path, ["R1C2", "R3C4"])
        assert result.ok is False
        assert result.unavailable is False

    def test_an_unrequested_extra_cell_is_a_defect(self, crop_path: Path):
        import json

        def _fake_post(host, payload, timeout):
            return json.dumps({"R1C2": "11", "R9C9": "guessed"})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("socr.judge.table_rung_ollama._post_chat", _fake_post)
            result = self._adj()(crop_path, ["R1C2"])
        assert result.ok is False

    def test_a_missing_crop_is_a_defect_and_makes_no_request(self, tmp_path: Path):
        called = []

        def _fake_post(host, payload, timeout):
            called.append(payload)
            return "{}"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("socr.judge.table_rung_ollama._post_chat", _fake_post)
            result = self._adj()(tmp_path / "nope.png", ["R1C2"])
        assert result.ok is False
        assert result.unavailable is False
        assert called == []


class TestAdjudicatorReachability:
    def test_the_probe_asks_about_the_adjudicators_own_model(self, monkeypatch):
        """Cold review round 1, finding 2: reader rung 1 being pulled says
        nothing about whether the adjudicator's model is."""
        from socr.judge.table_rung_ollama import adjudicator_rung_reachable

        asked = []

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"models": [{"name": "glm-5.3-flash:cloud"}]}

        def _get(url, timeout=None):
            asked.append(url)
            return _Resp()

        monkeypatch.setattr("socr.judge.table_rung_ollama.httpx.get", _get)
        assert adjudicator_rung_reachable("glm-5.3-flash:cloud", "http://h:1") is True
        assert adjudicator_rung_reachable("kimi-k2.6:cloud", "http://h:1") is False
        assert asked, "the probe must actually ask the daemon"


class TestUnreadableIsItsOwnWireState:
    """Cold review round 2, N2.

    Prompt rules 2 and 3 both used to say "return the empty string", so
    "I looked and the cell is blank" and "I could not read this cell at all"
    arrived as the same token. The guard has no way to tell those apart, so a
    NON-reading could withhold a rejected table against a non-empty
    extraction, or clear one against an empty extraction. The wire schema now
    carries the difference: a string is a reading, ``null`` is not.
    """

    def _adj(self):
        from socr.judge.table_rung_ollama import make_ollama_cell_adjudicator

        return make_ollama_cell_adjudicator("kimi-k2.6:cloud", "http://h:1", 5.0)

    def _answer(self, crop_path, body):
        def _fake_post(host, payload, timeout):
            return body

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("socr.judge.table_rung_ollama._post_chat", _fake_post)
            return self._adj()(crop_path, ["R1C1", "R1C2"])

    def test_null_is_a_successful_typed_non_reading_not_a_defect(self, crop_path: Path):
        import json

        result = self._answer(crop_path, json.dumps({"R1C1": None, "R1C2": "11"}))
        assert result.ok is True
        assert result.unreadable == ("R1C1",)
        # And it carries NO token, so nothing downstream can compare it.
        assert result.tokens == {"R1C2": "11"}

    def test_the_empty_string_stays_a_reading(self, crop_path: Path):
        import json

        result = self._answer(crop_path, json.dumps({"R1C1": "", "R1C2": "11"}))
        assert result.ok is True
        assert result.unreadable == ()
        assert result.tokens == {"R1C1": "", "R1C2": "11"}

    def test_a_null_still_has_to_be_a_requested_key(self, crop_path: Path):
        """Strictness is unchanged: ``null`` is a legal VALUE, not a licence to
        answer about cells nobody asked about or to omit ones they did."""
        import json

        assert self._answer(crop_path, json.dumps({"R1C1": None})).ok is False
        assert (
            self._answer(crop_path, json.dumps({"R1C1": None, "R1C2": "11", "R9C9": None})).ok
            is False
        )

    def test_a_non_string_non_null_value_is_still_a_defect(self, crop_path: Path):
        import json

        result = self._answer(crop_path, json.dumps({"R1C1": 11, "R1C2": "11"}))
        assert result.ok is False
        assert result.unavailable is False

    def test_the_prompt_asks_for_the_two_states_separately(self):
        from socr.judge.table_rung_ollama import build_blind_cell_prompt

        prompt = build_blind_cell_prompt(["R1C1"])
        assert "`null`" in prompt
        # The two rules must not resolve to the same answer any more.
        assert prompt.count("return the empty string for it") == 0


class TestAMalformedResponseBodyNeverEscapesAsAnException:
    """Cold review round 3, NEW C.

    ``_post_chat`` returned ``message.content`` untyped, and both callers reach
    for string methods on it. A daemon -- or a proxy in front of one --
    answering ``{"message": {"content": 1}}`` escaped as an ``AttributeError``
    out of two functions that both promise a typed failure result instead. The
    body is type-checked at the seam now, and the wrong shape is a DEFECT: it
    reproduces on the next call, so it must never latch.
    """

    class _Resp:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    def _post(self, monkeypatch, body):
        # The suite's autouse hermeticity fixture stubs ``_post_chat`` itself,
        # but the seam under test IS ``_post_chat``. ``_post_chat`` was bound
        # at this module's import, before any fixture ran, so restoring that
        # reference exercises the real body-shape validation while ``httpx``
        # stays firmly mocked.
        monkeypatch.setattr("socr.judge.table_rung_ollama._post_chat", _post_chat)
        monkeypatch.setattr(
            "socr.judge.table_rung_ollama.httpx.post", lambda *a, **k: self._Resp(body)
        )

    BAD_BODIES = [
        {"message": {"content": 1}},
        {"message": {"content": None}},
        {"message": {"content": ["a"]}},
        {"message": "not an object"},
        ["not an object at all"],
    ]

    @pytest.mark.parametrize("body", BAD_BODIES)
    def test_the_adjudicator_returns_a_deterministic_failure(
        self, monkeypatch, crop_path: Path, body
    ):
        from socr.judge.table_rung_ollama import make_ollama_cell_adjudicator

        self._post(monkeypatch, body)
        result = make_ollama_cell_adjudicator("m:cloud", "http://h:1", 5.0)(crop_path, ["R1C1"])

        assert result.ok is False
        assert result.unavailable is False
        assert result.refusal is False
        assert result.tokens == {} and result.unreadable == ()

    @pytest.mark.parametrize("body", BAD_BODIES)
    def test_the_reader_rung_returns_a_deterministic_failure(
        self, monkeypatch, crop_path: Path, body
    ):
        """The reader shares the seam and the same never-raises contract."""
        self._post(monkeypatch, body)
        result = build_ollama_rung("m:cloud", "http://h:1", 5.0)(crop_path, "| a |\n| - |\n", None)

        assert result.ok is False
        assert result.unavailable is False
        assert result.verdict is None

    def test_a_well_formed_body_still_works(self, monkeypatch, crop_path: Path):
        """The control: type-checking the body must not reject a real one."""
        self._post(monkeypatch, {"message": {"content": PASS_JSON}})
        result = build_ollama_rung("m:cloud", "http://h:1", 5.0)(crop_path, "| a |\n| - |\n", None)
        assert result.ok is True
