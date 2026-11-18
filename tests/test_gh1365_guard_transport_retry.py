"""#1365: the blind-cell guard retries ONE transient transport failure.

Pins DIFFERENCES between arms, not absolute outcomes (CLAUDE.md hermeticity
rule): the retried arm must equal the arm that never saw the error, and the
double-failure arm must equal the pre-fix unavailable outcome.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from socr.core import ollama_utils
from socr.core.config import PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.state import DocumentState
from socr.judge import table_rung_ollama
from socr.judge.table_cell_guard import (
    GUARD_TRANSPORT_RETRIES,
    GuardDisposition,
    evaluate_cell_guard,
)
from socr.judge.table_rung_ollama import BlindCellResult, adjudicator_rung_id
from socr.judge.table_verdict import is_transient_transport_error
from socr.tables.binding import BindingEvidence

REFS = ["R1C2", "R2C3"]
TOKENS = {"R1C2": "100", "R2C3": "400"}
RUNG = adjudicator_rung_id("test-adjudicator:cloud")


def _state() -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=Path("/tmp/doc.pdf"), page_count=1)
    return DocumentState(handle=handle)


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return httpx.HTTPStatusError(
        f"status {code}", request=request, response=httpx.Response(code, request=request)
    )


def _decide(adjudicator, **config):
    return evaluate_cell_guard(
        state=_state(),
        page_num=1,
        crop_path=None,
        extraction_tokens=dict(TOKENS),
        requested_refs=REFS,
        geometry_evidence=BindingEvidence.ABSTAIN,
        adjudicator=adjudicator,
        config=PipelineConfig(**config),
    )


class _Script:
    """An adjudicator that plays a fixed list of results/exceptions, counting calls."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = 0

    def __call__(self, crop_path, refs):
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        if isinstance(step, BaseException):
            raise step
        return step


def _ok():
    return BlindCellResult(rung=RUNG, ok=True, tokens=dict(TOKENS))


def _transient():
    return BlindCellResult(
        rung=RUNG, ok=False, error="504", unavailable=True, transient_transport=True
    )


def _view(decision):
    """Everything an outcome comparison needs, minus the retry counter."""
    return (
        decision.disposition,
        decision.adjudicator_ran,
        decision.unavailable,
        decision.refusal,
        decision.detail,
    )


def test_a_transient_then_success_equals_the_arm_with_no_error():
    clean = _Script(_ok())
    retried = _Script(_transient(), _ok())
    clean_decision = _decide(clean)
    retried_decision = _decide(retried)
    assert _view(retried_decision) == _view(clean_decision)
    assert retried_decision.disposition is GuardDisposition.VERIFIED_BY_BLIND_CELL_TRANSCRIPTION
    assert (clean.calls, retried.calls) == (1, 2)
    assert (clean_decision.transport_retries, retried_decision.transport_retries) == (0, 1)


def test_a_raised_transient_exception_is_retried_too():
    script = _Script(httpx.ConnectError("refused"), _ok())
    assert _decide(script).disposition is GuardDisposition.VERIFIED_BY_BLIND_CELL_TRANSCRIPTION
    assert script.calls == 2


def test_b_both_fail_equals_todays_unavailable_outcome():
    both = _Script(_transient(), _transient())
    once = _Script(_transient())
    both_decision = _decide(both, table_judge_adjudicator_cost_per_call_usd=0.0)
    # Today's outcome: the same failure with no retry available (a priced call
    # never retries, so it is the exact pre-fix path).
    today = _decide(once, table_judge_adjudicator_cost_per_call_usd=0.5)
    assert _view(both_decision) == _view(today)
    assert both_decision.unavailable and both_decision.adjudicator_ran
    assert both_decision.disposition is GuardDisposition.NOT_CLEARED
    assert (both.calls, once.calls) == (2, 1)
    assert (both_decision.transport_retries, today.transport_retries) == (1, 0)


@pytest.mark.parametrize(
    "failure",
    [
        BlindCellResult(rung=RUNG, ok=False, error="401", unavailable=True, refusal=True),
        BlindCellResult(rung=RUNG, ok=False, error="400 bad request"),
        BlindCellResult(rung=RUNG, ok=False, error="unusable answer"),
        BlindCellResult(rung=RUNG, ok=True, tokens={"R1C2": "7", "R2C3": "400"}),
        _status_error(404),
        _status_error(429),
        ValueError("malformed"),
        ollama_utils.TotalDeadlineExceeded("pipeline deadline"),
    ],
)
def test_c_non_transient_failures_are_called_exactly_once(failure):
    script = _Script(failure)
    decision = _decide(script)
    assert script.calls == 1
    assert decision.transport_retries == 0


def test_d_never_more_than_two_calls():
    script = _Script(_transient())
    _decide(script)
    assert script.calls == 1 + GUARD_TRANSPORT_RETRIES == 2


def test_a_priced_adjudicator_is_never_called_twice():
    script = _Script(_transient(), _ok())
    decision = _decide(script, table_judge_adjudicator_cost_per_call_usd=0.25)
    assert script.calls == 1
    assert decision.unavailable


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_status_error(500), True),
        (_status_error(504), True),
        (_status_error(599), True),
        (_status_error(400), False),
        (_status_error(401), False),
        (_status_error(404), False),
        (_status_error(429), False),
        (httpx.ConnectTimeout("t"), True),
        (httpx.ReadTimeout("t"), True),
        (httpx.ConnectError("c"), True),
        (ConnectionError("c"), True),
        (TimeoutError("t"), True),
        (ollama_utils.TotalDeadlineExceeded("d"), False),
        (httpx.UnsupportedProtocol("p"), False),
        (httpx.DecodingError("d"), False),
        (ValueError("v"), False),
    ],
)
def test_transient_is_decided_from_type_and_status(exc, expected):
    assert is_transient_transport_error(exc) is expected


def test_transcriber_flags_a_504_and_not_a_400(monkeypatch, tmp_path):
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"png")
    flags = {}
    for code in (504, 400):

        def _post(host, payload, timeout, code=code):
            raise _status_error(code)

        monkeypatch.setattr(table_rung_ollama, "_post_chat", _post)
        result = table_rung_ollama.transcribe_cells_ollama(crop, REFS, "m:cloud", None, 5.0)
        flags[code] = result.transient_transport
        assert not result.ok
    assert flags == {504: True, 400: False}


def test_end_to_end_504_then_good_answer_clears(monkeypatch, tmp_path):
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"png")
    answers = [_status_error(504), '{"R1C2": "100", "R2C3": "400"}']
    calls = []

    def _post(host, payload, timeout):
        calls.append(1)
        step = answers.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(table_rung_ollama, "_post_chat", _post)
    adjudicator = table_rung_ollama.make_ollama_cell_adjudicator("m:cloud", None, 5.0)
    decision = evaluate_cell_guard(
        state=_state(),
        page_num=1,
        crop_path=crop,
        extraction_tokens=dict(TOKENS),
        requested_refs=REFS,
        geometry_evidence=BindingEvidence.ABSTAIN,
        adjudicator=adjudicator,
        config=PipelineConfig(),
    )
    assert decision.disposition is GuardDisposition.VERIFIED_BY_BLIND_CELL_TRANSCRIPTION
    assert len(calls) == 2
