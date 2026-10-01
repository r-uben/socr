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

import concurrent.futures
import threading
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
    _page_sidecar,
    _config,
    _grid_pdf,
    _route_fn,
)

_DEADLINE = 0.2


def _run(root: Path, *, mode: str):
    """Two-page document; page 1's escalation call outlives the deadline.

    ``mode``: ``slow_done`` (page 1's abandoned call finishes before page 2
    qualifies), ``wedge`` (it never does), ``healthy`` (no call is slow). Runs
    share ``root/out``, so a second call with a different mode is a RESUME.

    Returns ``(provider_calls, events, process_result)``.
    """
    pdf = root / "doc.pdf"
    if not pdf.exists():  # a resume must see the SAME bytes (input checksum gate)
        _grid_pdf(pdf, pages=2)
    cfg = _config()
    cfg.escalation_timeout_sec = _DEADLINE
    cfg.local_engine = PROFILE_QWEN_LOCAL.engine
    pipeline = UnifiedPipeline(cfg)

    release = threading.Event()
    calls: list[int] = []
    escalation_futures: list[concurrent.futures.Future] = []
    real_submit = concurrent.futures.ThreadPoolExecutor.submit

    def _recording_submit(self, fn, *args, **kwargs):
        fut = real_submit(self, fn, *args, **kwargs)
        if getattr(fn, "__name__", "") == "run_provider":
            escalation_futures.append(fut)
        return fut

    def _engine(state, pages, fallback, engine, mode_, **kwargs):
        page_num = pages[0]
        calls.append(page_num)
        if page_num == 1 and mode != "healthy":
            release.wait(timeout=30)
        return [
            PageOutput(
                page_num=page_num,
                text=_MISSING_COLUMNS_CANDIDATE,
                status=PageStatus.SUCCESS,
                engine=engine.value,
            )
        ]

    def _score(state, page_num, ps, bo):
        # Runs on the page-major loop thread right before escalation. In
        # ``slow_done`` page 1's abandoned call is released and its Future is
        # awaited itself (no sleep): by the time page 2 asks, it is done.
        if page_num == 2 and mode == "slow_done":
            release.set()
            concurrent.futures.wait(escalation_futures[:1], timeout=10)
            assert escalation_futures[0].done()
        return True

    try:
        with (
            patch.object(concurrent.futures.ThreadPoolExecutor, "submit", _recording_submit),
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
            result = pipeline.process(pdf, root / "out")
    finally:
        release.set()
    return calls, _audit_events(root / "out"), result


def _kinds(events, page_num):
    return [e["kind"] for e in events if e.get("page_num") == page_num]


def test_a_slow_page_does_not_remove_escalation_from_the_next_page(tmp_path: Path) -> None:
    calls, events, _ = _run(tmp_path, mode="slow_done")

    assert calls == [1, 2], "page 2 qualified and must get its escalation call"
    assert "table_escalation_timeout" in _kinds(events, 1)
    assert "table_escalation_withheld" not in _kinds(events, 2)
    detail = next(e["detail"] for e in events if e["kind"] == "table_escalation_timeout")
    assert "lane disabled" not in detail, "the event must not claim a latch that did not happen"


def test_a_still_outstanding_call_withholds_the_next_page_and_says_so(tmp_path: Path) -> None:
    calls, events, _ = _run(tmp_path, mode="wedge")

    assert calls == [1], "a second call must not stack on an unresponsive provider"
    assert "table_escalation_timeout" in _kinds(events, 1)
    withheld = [e for e in events if e["kind"] == "table_escalation_withheld"]
    assert [e["page_num"] for e in withheld] == [2]
    assert "p1" in withheld[0]["detail"]


def test_the_two_runs_differ_only_in_whether_the_abandoned_call_finished(tmp_path: Path) -> None:
    """Pin the DIFFERENCE: same document, same timeout, same stubs."""
    slow_calls, slow_events, _ = _run(tmp_path / "slow", mode="slow_done")
    wedge_calls, wedge_events, _ = _run(tmp_path / "wedge", mode="wedge")

    assert slow_calls != wedge_calls
    assert _kinds(slow_events, 1) == _kinds(wedge_events, 1)
    assert set(_kinds(wedge_events, 2)) - set(_kinds(slow_events, 2)) == {
        "table_escalation_withheld"
    }


def test_withheld_page_is_demoted_at_page_and_document_status(tmp_path: Path) -> None:
    """Pin the DIFFERENCE between runs; the fixture is otherwise identical."""
    _, _, healthy = _run(tmp_path / "healthy", mode="healthy")
    _, _, slow = _run(tmp_path / "slow", mode="slow_done")
    _, _, wedge = _run(tmp_path / "wedge", mode="wedge")

    slow_p2 = _page_sidecar(tmp_path / "slow" / "out", 2)
    wedge_p2 = _page_sidecar(tmp_path / "wedge" / "out", 2)

    # Page 2: escalated (slow) versus withheld (wedge) -- only the wedge demotes it.
    assert slow_p2["status"] == PageStatus.SUCCESS.value
    assert wedge_p2["status"] != slow_p2["status"]
    assert wedge_p2["failure_mode"] == "table_unverified"
    assert wedge_p2["text"] == slow_p2["text"], "demotion must not discard the page text"

    # Document: a clean run is SUCCESS; one withheld page makes it not-SUCCESS, and
    # the withheld page is named in what the document reports.
    assert healthy.status != wedge.status
    assert "2" in (wedge.error or "") and "1, 2" in wedge.error
    assert "1, 2" not in (slow.error or "")


def test_a_wedged_run_is_retried_on_resume_once_the_provider_is_healthy(tmp_path: Path) -> None:
    wedge_calls, _, _ = _run(tmp_path, mode="wedge")
    assert wedge_calls == [1]

    resume_calls, resume_events, _ = _run(tmp_path, mode="healthy")

    assert 2 in resume_calls, "the withheld page must be reprocessed and escalated on resume"
    assert 1 in resume_calls, "the timed-out page must be reprocessed on resume"
    assert "table_escalation_withheld" not in _kinds(resume_events, 2)


def test_control_a_healthy_run_is_skipped_on_resume(tmp_path: Path) -> None:
    """Without the marker the same resume skips the page (the gate is the cause)."""
    first_calls, _, _ = _run(tmp_path, mode="healthy")
    assert first_calls == [1, 2]
    again_calls, _, _ = _run(tmp_path, mode="healthy")
    assert again_calls == []


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
