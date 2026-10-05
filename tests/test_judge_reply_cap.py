"""The page judge's reply is bounded, and a reply cut at the bound is no verdict.

Recorded 2026-10-04 on the Bocconi HPC (job 682725): with no ``max_tokens`` the
vLLM judge looped on one issue for 27,201 tokens until the context ran out
(``finish_reason: length``). The fixtures are real reply bodies:

- ``vllm_judge_truncated_raw.json``: that reply, with the issue strings replaced
  by ``<redacted>`` (the corpus is copyrighted). Envelope, usage and
  ``finish_reason`` are verbatim; the content keeps its recorded shape, a JSON
  object cut mid-string.
- ``ollama_judge_truncated_raw.json``: a local Ollama reply cut by a small
  ``num_predict`` (no page content), carrying ``done_reason: length``.
- ``vllm_judge_accept_raw.json``: a complete accept, for the control.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.core.result import JUDGE_OUTCOME_EXCEPTION, PageOutput, PageStatus
from socr.judge import ollama_judge, vllm_judge
from socr.judge.judge import (
    JUDGE_MAX_REPLY_TOKENS,
    JudgeReplyTruncatedError,
    is_page_judge_timeout,
)

FIXTURES = Path(__file__).parent / "fixtures"
URL = "http://127.0.0.1:8000/v1"
SERVED = "Qwen/Qwen3-VL-30B-A3B-Instruct"
ACCEPT_JSON = '{"faithful": true, "issues": [], "confidence": 0.98, "suggested_action": "accept"}'


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _capture_post(monkeypatch, module, payload):
    seen = {}

    def _post(url, **kwargs):
        seen["json"] = kwargs.get("json")
        return _Resp(payload)

    monkeypatch.setattr(module.httpx, "post", _post)
    return seen


# --------------------------------------------------------------------------
# the cap is sent
# --------------------------------------------------------------------------


def test_vllm_request_carries_the_reply_cap(monkeypatch):
    seen = _capture_post(monkeypatch, vllm_judge, _load("vllm_judge_accept_raw.json"))
    vllm_judge._post_chat(URL, SERVED, "P", "data:image/png;base64,AAA", 1.0)
    assert seen["json"]["max_tokens"] == JUDGE_MAX_REPLY_TOKENS


def test_ollama_request_carries_the_same_cap(monkeypatch):
    seen = _capture_post(
        monkeypatch, ollama_judge, {"response": ACCEPT_JSON, "done": True, "done_reason": "stop"}
    )
    ollama_judge._post_generate("http://localhost:11434", "m", "P", "AAA", 1.0)
    assert seen["json"]["options"]["num_predict"] == JUDGE_MAX_REPLY_TOKENS
    assert seen["json"]["options"]["temperature"] == 0


# --------------------------------------------------------------------------
# a reply cut at the cap is a failure, never a verdict
# --------------------------------------------------------------------------


def test_recorded_vllm_runaway_raises(monkeypatch):
    _capture_post(monkeypatch, vllm_judge, _load("vllm_judge_truncated_raw.json"))
    with pytest.raises(JudgeReplyTruncatedError):
        vllm_judge._post_chat(URL, SERVED, "P", "data:image/png;base64,AAA", 1.0)


def test_recorded_ollama_truncation_raises(monkeypatch):
    _capture_post(monkeypatch, ollama_judge, _load("ollama_judge_truncated_raw.json"))
    with pytest.raises(JudgeReplyTruncatedError):
        ollama_judge._post_generate("http://localhost:11434", "m", "P", "AAA", 1.0)


@pytest.mark.parametrize("backend", ["vllm", "ollama"])
def test_a_truncated_reply_that_happens_to_parse_still_raises(monkeypatch, backend):
    """The finish reason decides, not whether the fragment parses.

    ``parse_verdict`` would read ``faithful: true`` out of this content; the
    length flag says the model did not finish, so it must never be accepted.
    """
    if backend == "vllm":
        payload = _load("vllm_judge_truncated_raw.json")
        payload["choices"][0]["message"]["content"] = ACCEPT_JSON
        _capture_post(monkeypatch, vllm_judge, payload)
        call = lambda: vllm_judge._post_chat(URL, SERVED, "P", "data:,", 1.0)  # noqa: E731
    else:
        payload = _load("ollama_judge_truncated_raw.json")
        payload["response"] = ACCEPT_JSON
        _capture_post(monkeypatch, ollama_judge, payload)
        call = lambda: ollama_judge._post_generate("http://h", "m", "P", "AAA", 1.0)  # noqa: E731
    with pytest.raises(JudgeReplyTruncatedError):
        call()


def test_a_complete_recorded_reply_still_returns_its_content(monkeypatch):
    _capture_post(monkeypatch, vllm_judge, _load("vllm_judge_accept_raw.json"))
    raw = vllm_judge._post_chat(URL, SERVED, "P", "data:image/png;base64,AAA", 1.0)
    assert json.loads(raw)["faithful"] is True


def test_truncation_is_not_classified_as_a_timeout():
    """Only a timeout may license #713's credentialed stand-in."""
    assert is_page_judge_timeout(JudgeReplyTruncatedError("cap")) is False


def test_route_page_records_a_truncated_reply_as_an_unaccepted_judge_failure(monkeypatch, tmp_path):
    """End to end through VLMPageJudge -> VLLMVisionJudge -> _post_chat.

    ``run_killable`` is replaced by an in-process call so the patched
    ``httpx.post`` is the one that answers; everything above it is the shipped
    code path.
    """
    from socr.pipeline import agentic

    _capture_post(monkeypatch, vllm_judge, _load("vllm_judge_truncated_raw.json"))
    monkeypatch.setattr(
        vllm_judge,
        "run_killable",
        lambda spec, timeout: vllm_judge._post_chat(*spec.args),
    )
    image = tmp_path / "p1.png"
    image.write_bytes(b"\x89PNG\r\n")
    judge = agentic.VLMPageJudge(
        vllm_judge.VLLMVisionJudge(model=SERVED, base_url=URL), lambda _n: image
    )

    class _Prof:
        engine = type("E", (), {"value": "qwen"})()
        id = "qwen-vllm"
        model = SERVED
        backend = "vllm"
        cost_per_page_usd = 0.0
        timeout_sec = 5

    def _run(_prof, page_num):
        return PageOutput(page_num=page_num, text="Some text.", status=PageStatus.SUCCESS)

    decision = agentic.route_page(1, [_Prof()], _run, judge)
    attempt = decision.attempts[-1]
    assert attempt.accepted is False
    assert attempt.output.judge_outcome == JUDGE_OUTCOME_EXCEPTION
    assert "JudgeReplyTruncatedError" in attempt.reason or "cap" in attempt.reason
