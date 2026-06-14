"""Cost-aware agentic per-page router tests.

The router is pure given injected (run_provider, judge), so these use stubs —
no engines, no models. They pin the cost-effective behavior: cheapest-first,
escalate only on rejection, bound by max_attempts, keep best on total failure.
"""

from __future__ import annotations

import time

import pytest

from socr.core.config import EngineType
from socr.core.providers import provider_ladder
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import (
    DEFAULT_PROVIDER_TIMEOUTS,
    AcceptDecision,
    HeuristicPageJudge,
    VLMPageJudge,
    route_page,
)

LADDER = provider_ladder(
    {EngineType.GLM, EngineType.GEMINI, EngineType.MISTRAL},
    include_ineligible=True,
)
# -> [GLM(free), GEMINI(0.0002), MISTRAL(0.001)]


def _run(audit=None, conf=None, text=None):
    audit = audit or {}
    conf = conf or {}
    text = text or {}

    def run(engine: EngineType, page_num: int) -> PageOutput:
        return PageOutput(
            page_num=page_num,
            text=text.get(engine, f"ocr from {engine.value}"),
            status=PageStatus.SUCCESS,
            engine=engine.value,
            audit_passed=audit.get(engine, True),
            confidence=conf.get(engine, 0.5),
        )

    return run


class _StubJudge:
    """Accepts only outputs whose engine is in `accept`."""

    def __init__(self, accept):
        self.accept = set(accept)

    def assess(self, output, provider):
        return AcceptDecision(accept=provider.engine in self.accept, reason="stub")


def test_accepts_cheapest_no_escalation():
    d = route_page(1, LADDER, _run(), _StubJudge({EngineType.GLM}))
    assert d.accepted
    assert d.winning_engine == "glm"
    assert len(d.attempts) == 1
    assert d.escalations == 0
    assert d.total_cost_usd == 0.0


def test_escalates_to_next_when_cheap_rejected():
    d = route_page(2, LADDER, _run(), _StubJudge({EngineType.GEMINI}))
    assert d.accepted
    assert d.winning_engine == "gemini"
    assert [a.engine for a in d.attempts] == [EngineType.GLM, EngineType.GEMINI]
    assert d.escalations == 1
    assert d.total_cost_usd == pytest.approx(0.0002)  # free GLM + gemini


def test_climbs_full_ladder_then_accepts_priciest():
    d = route_page(3, LADDER, _run(), _StubJudge({EngineType.MISTRAL}))
    assert d.accepted and d.winning_engine == "mistral"
    assert d.escalations == 2
    assert d.total_cost_usd == pytest.approx(0.0012)


def test_all_rejected_keeps_best_audit_passed():
    # None accepted -> best-effort prefers the audit_passed attempt.
    judge = _StubJudge(set())
    run = _run(audit={EngineType.GLM: False, EngineType.GEMINI: True, EngineType.MISTRAL: False})
    d = route_page(4, LADDER, run, judge)
    assert not d.accepted
    assert len(d.attempts) == 3  # tried everything
    assert d.final_output.engine == "gemini"  # the only audit_passed one


def test_all_rejected_uses_confidence_then_words():
    judge = _StubJudge(set())
    run = _run(
        audit={e: False for e in (EngineType.GLM, EngineType.GEMINI, EngineType.MISTRAL)},
        conf={EngineType.GLM: 0.1, EngineType.GEMINI: 0.9, EngineType.MISTRAL: 0.3},
    )
    d = route_page(5, LADDER, run, judge)
    assert d.final_output.engine == "gemini"  # highest confidence


def test_max_attempts_bounds_cost():
    # Only the cheapest provider is tried even though it's rejected.
    d = route_page(6, LADDER, _run(), _StubJudge({EngineType.MISTRAL}), max_attempts=1)
    assert len(d.attempts) == 1
    assert d.attempts[0].engine == EngineType.GLM
    assert not d.accepted


def test_provider_exception_is_recorded_and_skipped():
    def run(engine, page_num):
        if engine == EngineType.GLM:
            raise RuntimeError("ollama down")
        return PageOutput(
            page_num=page_num,
            text="cloud ok",
            engine=engine.value,
            status=PageStatus.SUCCESS,
            audit_passed=True,
        )

    d = route_page(7, LADDER, run, _StubJudge({EngineType.GEMINI}))
    assert d.accepted and d.winning_engine == "gemini"
    assert d.attempts[0].engine == EngineType.GLM
    assert d.attempts[0].output.status == PageStatus.ERROR  # recorded the failure
    assert d.attempts[1].engine == EngineType.GEMINI


def test_empty_ladder_yields_error():
    d = route_page(8, [], _run(), _StubJudge(set()))
    assert not d.accepted
    assert d.final_output.status == PageStatus.ERROR
    assert d.attempts == []


# --- judge adapters -------------------------------------------------------


def test_heuristic_page_judge_accepts_good_rejects_empty():
    from socr.audit.heuristics import HeuristicsChecker

    judge = HeuristicPageJudge(HeuristicsChecker(min_word_count=10))
    good = PageOutput(page_num=1, text=" ".join(["word"] * 50), status=PageStatus.SUCCESS)
    empty = PageOutput(page_num=1, text="", status=PageStatus.ERROR)
    prof = LADDER[0]
    assert judge.assess(good, prof).accept
    assert not judge.assess(empty, prof).accept


def test_vlm_page_judge_uses_verdict():
    from socr.judge.judge import JudgeVerdict

    class _VJ:
        def __init__(self, faithful):
            self.faithful = faithful

        def judge(self, image_path, ocr_text):
            issues = [] if self.faithful else ["garbled"]
            return JudgeVerdict(faithful=self.faithful, confidence=0.8, issues=issues)

    rendered = []
    judge = VLMPageJudge(_VJ(True), render_image=lambda pn: rendered.append(pn) or f"/img/{pn}.png")
    out = PageOutput(page_num=3, text="some text", status=PageStatus.SUCCESS)
    dec = judge.assess(out, LADDER[0])
    assert dec.accept and rendered == [3]  # it rendered the right page

    judge_bad = VLMPageJudge(_VJ(False), render_image=lambda pn: f"/img/{pn}.png")
    assert not judge_bad.assess(out, LADDER[0]).accept


def test_provider_attempt_records_id_model_backend():
    d = route_page(1, LADDER, _run(), _StubJudge({EngineType.GLM}))
    assert d.accepted
    attempt = d.attempts[0]
    assert attempt.provider_id != ""
    assert attempt.model != ""
    assert attempt.backend != ""


# --- provider timeout (TICKET-C1) -----------------------------------------


def test_slow_provider_timeout_escalates():
    """A provider that exceeds its timeout must escalate, not hang the batch.

    Scenario: GLM is given a 50 ms timeout but sleeps 10 s.
    GEMINI is instant and accepted.  Expected:
      (1) GLM attempt recorded with ERROR status,
      (2) GEMINI was tried and accepted,
      (3) total elapsed < 2 s.
    """

    def run(engine: EngineType, page_num: int) -> PageOutput:
        if engine == EngineType.GLM:
            time.sleep(10)  # simulate a stall
        return PageOutput(
            page_num=page_num,
            text=f"ok from {engine.value}",
            status=PageStatus.SUCCESS,
            engine=engine.value,
            audit_passed=True,
        )

    t0 = time.monotonic()
    d = route_page(
        1,
        LADDER,
        run,
        _StubJudge({EngineType.GEMINI}),
        provider_timeout={EngineType.GLM: 0.05},
    )
    elapsed = time.monotonic() - t0

    # (1) GLM attempt must be recorded as ERROR
    glm_att = next(a for a in d.attempts if a.engine == EngineType.GLM)
    assert glm_att.output.status == PageStatus.ERROR

    # (2) GEMINI must have been tried and accepted
    assert d.accepted
    assert d.winning_engine == "gemini"

    # (3) wall-clock must be well under 2 s (GLM did NOT sleep 10 s)
    assert elapsed < 2.0, f"route_page took {elapsed:.2f}s — GLM stall not intercepted"


def test_no_timeout_runs_normally():
    """Backward compat: omitting provider_timeout must not change routing behaviour."""
    d = route_page(1, LADDER, _run(), _StubJudge({EngineType.GLM}))
    assert d.accepted
    assert d.winning_engine == "glm"
    assert len(d.attempts) == 1


def test_default_provider_timeouts_keys():
    """DEFAULT_PROVIDER_TIMEOUTS must cover the two primary agentic rungs.

    Tests that QWEN and GEMINI keys are present.  Deliberately does NOT assert
    specific float values because the calibration is tunable — the important
    invariant is that the named constant exists and has the right shape.
    """
    assert EngineType.QWEN in DEFAULT_PROVIDER_TIMEOUTS
    assert EngineType.GEMINI in DEFAULT_PROVIDER_TIMEOUTS
    # Both values must be positive finite floats
    for engine, seconds in DEFAULT_PROVIDER_TIMEOUTS.items():
        assert isinstance(seconds, float), f"{engine}: expected float, got {type(seconds)}"
        assert seconds > 0, f"{engine}: timeout must be positive"
