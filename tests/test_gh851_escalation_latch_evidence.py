"""GH-851: the escalation lane is withheld on evidence of a WEDGE, not of one slow page.

Before: one page exceeding ``escalation_timeout_sec`` latched the lane off for every
later page of the document ("lane disabled for the rest of this document"). Measured
on the run corpus, the 32 timeouts were ordinary slow reads, not wedges.

After: the timed-out call's future is remembered. The next page that qualifies for
escalation looks at it: still outstanding means the provider is presumed wedged and
that page is withheld WITH a ``table_escalation_withheld`` event; finished means it was
only slow and the page is escalated normally.

The two end-to-end runs below differ in exactly one thing -- whether the abandoned
call has finished by the time page 2 qualifies -- so the pin is a DIFFERENCE, not an
absolute outcome measured on one machine (CLAUDE.md). Hermetic: the ladder, the judge
and the engine call are all patched; nothing needs ollama or a provider.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fitz")

from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.pipeline.agentic import AcceptDecision  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402
from test_gh855_scoring_independent_of_lane_health import (  # noqa: E402
    _MISSING_COLUMNS_CANDIDATE,
    _audit_events,
    _config,
    _grid_pdf,
    _route_fn,
)

_DEADLINE = 0.2


def _run(tmp_path: Path, *, slow_call_finishes_before_page_two: bool):
    """Two-page document; page 1's escalation call outlives the deadline.

    Returns ``(provider_calls, events)``. The only variable between the runs is
    whether page 1's abandoned call has completed when page 2 asks for escalation.
    """
    pdf = _grid_pdf(tmp_path / "doc.pdf", pages=2)
    cfg = _config()
    cfg.escalation_timeout_sec = _DEADLINE
    cfg.local_engine = PROFILE_QWEN_LOCAL.engine
    pipeline = UnifiedPipeline(cfg)

    release = threading.Event()
    page_one_returned = threading.Event()
    calls: list[int] = []

    def _engine(state, pages, fallback, engine, mode, **kwargs):
        page_num = pages[0]
        calls.append(page_num)
        if page_num == 1:
            release.wait(timeout=30)
            page_one_returned.set()
        return [
            PageOutput(
                page_num=page_num,
                text=_MISSING_COLUMNS_CANDIDATE,
                status=PageStatus.SUCCESS,
                engine=engine.value,
            )
        ]

    def _score(state, page_num, ps, bo):
        # Runs on the page-major loop thread right before escalation. This is the
        # one place the two runs differ: let page 1's slow call finish (or not)
        # before page 2 is judged.
        if page_num == 2 and slow_call_finishes_before_page_two:
            release.set()
            assert page_one_returned.wait(timeout=10)
            time.sleep(0.05)  # let the worker thread publish its Future result
        return True

    try:
        with (
            patch.object(
                pipeline,
                "_available_engines_for_agentic",
                return_value=[PROFILE_QWEN_LOCAL, PROFILE_GEMINI],
            ),
            patch.object(UnifiedPipeline, "_page_has_tables", return_value=False),
            patch.object(pipeline, "_surface_table_scoring", side_effect=_score),
            patch.object(pipeline, "_run_engine_on_pages", side_effect=_engine),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route_fn),
            patch(
                "socr.pipeline.agentic.HeuristicPageJudge.assess",
                return_value=AcceptDecision(accept=True, reason="heuristics passed"),
            ),
            patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
        ):
            pipeline.process(pdf, tmp_path / "out")
    finally:
        release.set()
    return calls, _audit_events(tmp_path / "out")


def _kinds(events, page_num):
    return [e["kind"] for e in events if e.get("page_num") == page_num]


def test_a_slow_page_does_not_remove_escalation_from_the_next_page(tmp_path: Path) -> None:
    calls, events = _run(tmp_path, slow_call_finishes_before_page_two=True)

    assert calls == [1, 2], "page 2 qualified and must get its escalation call"
    assert "table_escalation_timeout" in _kinds(events, 1)
    assert "table_escalation_withheld" not in _kinds(events, 2)
    detail = next(e["detail"] for e in events if e["kind"] == "table_escalation_timeout")
    assert "lane disabled" not in detail, "the event must not claim a latch that did not happen"


def test_a_still_outstanding_call_withholds_the_next_page_and_says_so(tmp_path: Path) -> None:
    calls, events = _run(tmp_path, slow_call_finishes_before_page_two=False)

    assert calls == [1], "a second call must not stack on an unresponsive provider"
    assert "table_escalation_timeout" in _kinds(events, 1)
    withheld = [e for e in events if e["kind"] == "table_escalation_withheld"]
    assert [e["page_num"] for e in withheld] == [2]
    assert "p1" in withheld[0]["detail"]


def test_the_two_runs_differ_only_in_whether_the_abandoned_call_finished(tmp_path: Path) -> None:
    """Pin the DIFFERENCE: same document, same timeout, same stubs."""
    slow_calls, slow_events = _run(tmp_path / "slow", slow_call_finishes_before_page_two=True)
    wedge_calls, wedge_events = _run(tmp_path / "wedge", slow_call_finishes_before_page_two=False)

    assert slow_calls != wedge_calls
    assert _kinds(slow_events, 1) == _kinds(wedge_events, 1)
    assert set(_kinds(wedge_events, 2)) - set(_kinds(slow_events, 2)) == {
        "table_escalation_withheld"
    }


def test_withheld_and_timeout_events_are_replayed_on_resume_and_distrust_the_page() -> None:
    from socr.core.manifest import CREDENTIAL_BLOCKING_EVENT_KINDS
    from socr.core.tables_trust import TABLE_DISTRUST_KINDS

    replayed = UnifiedPipeline.resume_restore_kinds()
    for kind in ("table_escalation_timeout", "table_escalation_withheld"):
        assert kind in replayed, f"{kind} is lost on resume"
        assert kind in TABLE_DISTRUST_KINDS
        assert kind in CREDENTIAL_BLOCKING_EVENT_KINDS


def test_a_real_wedge_still_withholds_every_later_qualifying_page(tmp_path: Path) -> None:
    """Direct, hermetic: a call that never returns keeps the lane closed per page."""
    from test_gh96_escalation_lane import _GEMINI, _out, _pipeline, _SHIFTED, _State, _PageState

    pdf = _grid_pdf(tmp_path / "doc.pdf", pages=3)
    pipe = _pipeline(escalation_timeout_sec=_DEADLINE)
    state = _State()
    release = threading.Event()
    abandoned: list = []
    calls: list[int] = []

    def hang(profile, page_num):
        calls.append(page_num)
        release.wait(timeout=30)
        return _out(_SHIFTED)

    try:
        results = [
            pipe._escalate_table_page(
                state,
                n,
                _PageState(),
                _out(_SHIFTED, engine="qwen"),
                _GEMINI,
                hang,
                pdf,
                needs_escalation=True,
                abandoned=abandoned,
            )[0]
            for n in (1, 2, 3)
        ]
    finally:
        release.set()

    assert results == [False, True, True]
    assert calls == [1]
    kinds = [(e.page_num, e.kind) for e in state.events]
    assert (2, "table_escalation_withheld") in kinds
    assert (3, "table_escalation_withheld") in kinds
