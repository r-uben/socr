"""GH-160: post-route table escalation must honour --max-cost-per-page and
--cost-budget, not just --strict-local.

``_resolve_table_escalation_provider`` picked its rung from ``available`` (only
tier-filtered), not from ``ladder`` (tier- AND cost-filtered), so a cap below
the escalation profile's price never suppressed it. Separately, nothing
checked the DOCUMENT's remaining ``--cost-budget`` before firing the call.

Hermetic: no provider, no network, no live model.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from test_p35_cold_review_round2 import _build_fixture_pdf  # noqa: E402

from socr.core.config import PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402


def _pipeline(**overrides) -> UnifiedPipeline:
    cfg = PipelineConfig(quiet=True, escalate_ambiguous_tables=True, **overrides)
    pipe = object.__new__(UnifiedPipeline)
    pipe.config = cfg
    return pipe


# ---------------------------------------------------------------------------
# 1 — provider selection must come from the cost-filtered ladder
# ---------------------------------------------------------------------------


def test_cap_below_escalation_price_excludes_it_at_the_real_call_site():
    """Pin the DIFFERENCE (CLAUDE.md) at ``_build_ladder_and_escalation_profile``,
    the exact helper ``_phase_agentic`` calls. Main's bug was the call site
    itself passing the merely tier-filtered ``available`` into
    ``_resolve_table_escalation_provider`` instead of the cost-filtered
    ``ladder`` -- a fact that a test of ``_resolve_table_escalation_provider``
    alone (unchanged internally) cannot detect.
    """
    available = [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]

    # Below Gemini's price: escalation must be suppressed.
    cap = PROFILE_GEMINI.cost_per_page_usd / 2
    capped = _pipeline(max_cost_per_page=cap)
    ladder, escalation = capped._build_ladder_and_escalation_profile(available)
    assert PROFILE_GEMINI not in ladder
    assert escalation is None

    # No cap: unchanged from today, Gemini is still the escalation rung.
    uncapped = _pipeline(max_cost_per_page=0.0)
    ladder, escalation = uncapped._build_ladder_and_escalation_profile(available)
    assert PROFILE_GEMINI in ladder
    assert escalation is PROFILE_GEMINI


# ---------------------------------------------------------------------------
# 2 — remaining --cost-budget gates the call before it fires
# ---------------------------------------------------------------------------


def _state_with_spend(tmp_path: Path, spent: float) -> DocumentState:
    pdf = _build_fixture_pdf(tmp_path)
    state = DocumentState(DocumentHandle(pdf))
    if spent:
        from socr.core.result import DocumentStatus, EngineResult

        state.record_engine_run(
            EngineResult(
                document_path=pdf,
                engine="qwen",
                status=DocumentStatus.SUCCESS,
                pages=[],
                pages_processed=1,
                cost=spent,
            ),
            page_nums=[1],
        )
    return state


def test_escalation_refused_when_it_would_exceed_remaining_budget(tmp_path: Path):
    state = _state_with_spend(tmp_path, spent=PROFILE_GEMINI.cost_per_page_usd)
    ps = state.pages[1]
    bo = PageOutput(page_num=1, text="native", status=PageStatus.SUCCESS, engine="qwen")
    ps.attempts.append(bo)
    ps.best_output = bo

    # Budget already exhausted by the prior spend: remaining < Gemini's price.
    pipe = _pipeline(
        cost_budget=PROFILE_GEMINI.cost_per_page_usd,  # all of it already spent
        table_judge_ladder=False,
    )

    def _run_provider_must_not_fire(profile, page_num):
        raise AssertionError("escalation call must not fire over remaining budget")

    degraded, out = pipe._escalate_table_page(
        state,
        1,
        ps,
        bo,
        PROFILE_GEMINI,
        _run_provider_must_not_fire,
        state.handle.path,
        needs_escalation=True,
    )
    assert degraded is False
    assert out is bo
    assert state.total_cost == PROFILE_GEMINI.cost_per_page_usd  # unchanged, no double spend


def test_escalation_fires_when_budget_permits_it(tmp_path: Path):
    state = _state_with_spend(tmp_path, spent=0.0)
    ps = state.pages[1]
    bo = PageOutput(page_num=1, text="native", status=PageStatus.SUCCESS, engine="qwen")
    ps.attempts.append(bo)
    ps.best_output = bo

    pipe = _pipeline(
        cost_budget=PROFILE_GEMINI.cost_per_page_usd * 10,
        table_judge_ladder=False,
    )

    def _run_provider(profile, page_num):
        return PageOutput(
            page_num=page_num, text="native", status=PageStatus.SUCCESS, engine=profile.engine.value
        )

    with patch("socr.tables.escalation_decision.decide_escalation") as mock_decide:
        mock_decide.return_value.accepted = False
        mock_decide.return_value.reason = "no better"
        mock_decide.return_value.gate = "control"
        mock_decide.return_value.delta = 0.0
        degraded, out = pipe._escalate_table_page(
            state,
            1,
            ps,
            bo,
            PROFILE_GEMINI,
            _run_provider,
            state.handle.path,
            needs_escalation=True,
        )
    assert degraded is False
    assert out is bo
    assert state.total_cost == PROFILE_GEMINI.cost_per_page_usd  # the call WAS paid for


# ---------------------------------------------------------------------------
# 3 — a call that was actually made must be billed, even on an early exit
#     (round-2 review finding: reviewer flagged this as GH-160's own third
#     acceptance criterion, not out of scope)
# ---------------------------------------------------------------------------


def test_escalation_timeout_still_bills_the_attempt(tmp_path: Path):
    state = _state_with_spend(tmp_path, spent=0.0)
    ps = state.pages[1]
    bo = PageOutput(page_num=1, text="native", status=PageStatus.SUCCESS, engine="qwen")
    ps.attempts.append(bo)
    ps.best_output = bo

    pipe = _pipeline(escalation_timeout_sec=0.05)

    def _run_provider_hangs(profile, page_num):
        time.sleep(0.5)
        return PageOutput(
            page_num=page_num, text="x", status=PageStatus.SUCCESS, engine=profile.engine.value
        )

    degraded, out = pipe._escalate_table_page(
        state,
        1,
        ps,
        bo,
        PROFILE_GEMINI,
        _run_provider_hangs,
        state.handle.path,
        needs_escalation=True,
    )
    assert degraded is True  # lane disabled for the rest of the document
    assert out is bo
    assert state.total_cost == PROFILE_GEMINI.cost_per_page_usd, (
        "a call that was actually launched is billable even though it timed out"
    )


def test_escalation_wrong_engine_still_bills_the_attempt(tmp_path: Path):
    state = _state_with_spend(tmp_path, spent=0.0)
    ps = state.pages[1]
    bo = PageOutput(page_num=1, text="native", status=PageStatus.SUCCESS, engine="qwen")
    ps.attempts.append(bo)
    ps.best_output = bo

    pipe = _pipeline()

    def _run_provider_wrong_engine(profile, page_num):
        # A failed engine call converted to native text by a DIFFERENT engine
        # -- the exact shape `_run_engine_on_pages` produces on failure.
        return PageOutput(
            page_num=page_num, text="fallback", status=PageStatus.SUCCESS, engine="qwen"
        )

    degraded, out = pipe._escalate_table_page(
        state,
        1,
        ps,
        bo,
        PROFILE_GEMINI,
        _run_provider_wrong_engine,
        state.handle.path,
        needs_escalation=True,
    )
    assert degraded is False
    assert out is bo
    assert state.total_cost == PROFILE_GEMINI.cost_per_page_usd, (
        "the call was made (and answered by the wrong engine) -- still billable"
    )


def test_escalation_empty_candidate_still_bills_the_attempt(tmp_path: Path):
    state = _state_with_spend(tmp_path, spent=0.0)
    ps = state.pages[1]
    bo = PageOutput(page_num=1, text="native", status=PageStatus.SUCCESS, engine="qwen")
    ps.attempts.append(bo)
    ps.best_output = bo

    pipe = _pipeline()

    def _run_provider_empty(profile, page_num):
        return PageOutput(
            page_num=page_num, text="", status=PageStatus.SUCCESS, engine=profile.engine.value
        )

    degraded, out = pipe._escalate_table_page(
        state,
        1,
        ps,
        bo,
        PROFILE_GEMINI,
        _run_provider_empty,
        state.handle.path,
        needs_escalation=True,
    )
    assert degraded is False
    assert out is bo
    assert state.total_cost == PROFILE_GEMINI.cost_per_page_usd, (
        "the call was made and answered with no usable text -- still billable"
    )
