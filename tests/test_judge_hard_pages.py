"""Phase 3b — VLM judge on HARD pages (catch semantic table/math corruption)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline


def _handle(pages: int = 1) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        return DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=pages)


def _state_with_page(engine="gemini", has_tables=True, has_equations=False):
    state = DocumentState(handle=_handle(1))
    bo = PageOutput(
        page_num=1,
        text="rows of numbers",
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=True,
    )
    state.pages[1].attempts.append(bo)
    state.pages[1].best_output = bo
    return state, bo


def _assessment(has_tables=True, has_equations=False):
    return DocumentAssessment(
        path=Path("/tmp/fake.pdf"),
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text="",
                confidence=1.0,
                has_tables=has_tables,
                has_equations=has_equations,
            )
        ],
    )


class _FakeJudge:
    """Stand-in for VLMPageJudge: verdict driven by the AcceptDecision we return."""

    def __init__(self, decision):
        self._decision = decision

    def factory(self, _judge, _renderer):  # matches VLMPageJudge(judge, renderer)
        return self

    def assess(self, _output, _provider):  # matches PageJudge.assess(output, provider)
        return self._decision


def _run(monkeypatch, pipe, state, decision, model="mock"):
    from socr.pipeline import agentic

    fake = _FakeJudge(decision)
    monkeypatch.setattr(pipe, "_resolve_judge_model", lambda: model)
    monkeypatch.setattr(pipe, "_make_page_renderer", lambda s: lambda pn: Path("/tmp/x.png"))
    monkeypatch.setattr(agentic, "VLMPageJudge", fake.factory)
    pipe._phase_judge_hard_pages(state)


def test_rejected_hard_page_loses_best_output(monkeypatch):
    from socr.pipeline.agentic import AcceptDecision

    pipe = UnifiedPipeline(PipelineConfig(quiet=True))
    state, bo = _state_with_page()
    pipe._last_assessment = _assessment(has_tables=True)
    _run(monkeypatch, pipe, state, AcceptDecision(accept=False, reason="wrong digits"))
    assert state.pages[1].best_output is None  # -> needs_repair
    assert state.pages[1].needs_repair
    assert bo.failure_mode == FailureMode.AUDIT_FAILED


def test_accepted_hard_page_kept(monkeypatch):
    from socr.pipeline.agentic import AcceptDecision

    pipe = UnifiedPipeline(PipelineConfig(quiet=True))
    state, bo = _state_with_page()
    pipe._last_assessment = _assessment(has_tables=True)
    _run(monkeypatch, pipe, state, AcceptDecision(accept=True, reason="faithful"))
    assert state.pages[1].best_output is bo  # untouched


def test_easy_page_not_judged(monkeypatch):
    from socr.pipeline.agentic import AcceptDecision

    pipe = UnifiedPipeline(PipelineConfig(quiet=True))
    state, bo = _state_with_page()
    pipe._last_assessment = _assessment(has_tables=False, has_equations=False)  # not hard
    _run(monkeypatch, pipe, state, AcceptDecision(accept=False, reason="x"))
    assert state.pages[1].best_output is bo  # skipped (not a hard page)


def test_native_text_not_judged(monkeypatch):
    from socr.pipeline.agentic import AcceptDecision

    pipe = UnifiedPipeline(PipelineConfig(quiet=True))
    state, bo = _state_with_page(engine="native")
    pipe._last_assessment = _assessment(has_tables=True)
    _run(monkeypatch, pipe, state, AcceptDecision(accept=False, reason="x"))
    assert state.pages[1].best_output is bo  # native is char-exact; not judged


def test_noop_when_no_judge_model(monkeypatch):
    from socr.pipeline.agentic import AcceptDecision

    pipe = UnifiedPipeline(PipelineConfig(quiet=True))
    state, bo = _state_with_page()
    pipe._last_assessment = _assessment(has_tables=True)
    _run(monkeypatch, pipe, state, AcceptDecision(accept=False, reason="x"), model=None)
    assert state.pages[1].best_output is bo  # no vision judge -> no-op


def test_disabled_by_config():
    # Phase is gated in process(); the flag default is on, off-switch is respected.
    assert PipelineConfig().judge_hard_pages is True
    assert PipelineConfig(judge_hard_pages=False).judge_hard_pages is False
