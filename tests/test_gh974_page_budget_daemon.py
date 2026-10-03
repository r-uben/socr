"""GH-974: daemon deadline workers, and a total wall-clock budget per page ladder.

Part 1: an abandoned deadline worker must not hold the interpreter open. Measured
in a CHILD interpreter (the claim is about interpreter shutdown): the stdlib
``ThreadPoolExecutor`` control blocks until the hung call returns; the real
``route_page`` and ``_read_with_deadline`` sites exit promptly.

Part 2: the page budget. Fake slow rungs exhaust it; the page ends UNVERIFIED
within the budget plus one call; exactly one budget event; the next page runs
normally. Hermetic: no Ollama, no provider, no model.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

import socr
from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge.ladder_budget import (
    TABLE_LADDER_BUDGET_EXHAUSTED_KIND,
    PageLadderBudget,
)
from socr.judge.table_verdict import (
    TABLE_LADDER_ACCEPTED_KIND,
    TABLE_LADDER_UNVERIFIED_KIND,
    RungResult,
    TableJudgeVerdict,
)
from socr.pipeline import orchestrator
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.binding import BindingEvidence

SRC = Path(socr.__file__).resolve().parent

# --------------------------------------------------------------------------
# Part 1: daemon workers
# --------------------------------------------------------------------------

#: How long the abandoned call hangs in the child, and the bound a prompt exit
#: must beat. The control must take at least the hang; the fixed sites must not.
_HANG = 8.0
_PROMPT = _HANG / 2


def _child_lifetime(body: str) -> float:
    start = time.monotonic()
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        # the child must import THIS tree's socr, not the editable install's
        env={**os.environ, "PYTHONPATH": str(SRC.parent)},
        timeout=_HANG * 4,
        check=True,
    )
    return time.monotonic() - start


def test_control_threadpool_worker_blocks_exit():
    """The defect: a ThreadPoolExecutor worker abandoned with shutdown(wait=False)."""
    lifetime = _child_lifetime(f"""
        import concurrent.futures, time
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        f = ex.submit(time.sleep, {_HANG})
        try:
            f.result(timeout=0.2)
        except concurrent.futures.TimeoutError:
            ex.shutdown(wait=False)
        """)
    assert lifetime >= _HANG, lifetime


def test_route_page_abandoned_provider_does_not_block_exit():
    lifetime = _child_lifetime(f"""
        import time
        from socr.core.config import EngineType
        from socr.core.providers import PROFILE_QWEN_LOCAL
        from socr.pipeline.agentic import route_page

        class J:
            def assess(self, output, provider):
                raise AssertionError("never reached")

        d = route_page(
            1, [PROFILE_QWEN_LOCAL], lambda prof, n: time.sleep({_HANG}), J(),
            provider_timeout={{PROFILE_QWEN_LOCAL.engine: 0.2}},
        )
        """)
    assert lifetime < _PROMPT, lifetime


def test_crop_reader_abandoned_call_does_not_block_exit():
    lifetime = _child_lifetime(f"""
        import time
        from pathlib import Path
        from socr.tables.extract import TableCropExtractor, _CropTimeoutError

        class R:
            def read(self, p):
                time.sleep({_HANG})

        class Self:
            _reader = R()

        try:
            TableCropExtractor._read_with_deadline(Self(), Path("x.png"), 0.2, 1)
        except _CropTimeoutError:
            pass
        """)
    assert lifetime < _PROMPT, lifetime


@pytest.mark.parametrize(
    "rel", ["pipeline/orchestrator.py", "pipeline/agentic.py", "tables/extract.py"]
)
def test_no_threadpool_executor_is_constructed_at_a_deadline_site(rel):
    """Static guard for the sites a child process cannot cheaply drive
    (the escalation pool and ``_TimeoutJudge``): none may build a non-daemon pool."""
    code = [
        line
        for line in (SRC / rel).read_text().splitlines()
        if not line.lstrip().startswith("#") and re.search(r"ThreadPoolExecutor\(", line)
    ]
    assert code == []


# --------------------------------------------------------------------------
# Part 2: the page budget
# --------------------------------------------------------------------------

SLOW = 0.3  # seconds each fake rung call takes
BUDGET = 0.5  # two calls start inside it (t=0, t=0.3); a third (t=0.6) does not


def _ruled_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    cols = [100, 220, 300, 380]
    rows = [100 + i * 22 for i in range(4)]
    for _ in range(2):
        page = doc.new_page()
        for r, y in enumerate(rows):
            for c, x in enumerate(cols):
                page.insert_text((x + 4, y + 12), f"{r}{c}", fontsize=9)
        for yy in rows:
            page.draw_line((100, yy), (460, yy))
        for xx in cols + [460]:
            page.draw_line((xx, rows[0]), (xx, rows[-1]))
    path = tmp_path / "doc.pdf"
    doc.save(path)
    doc.close()
    return path


_TABLE_MD = (
    "| c0 | c1 | c2 | c3 |\n"
    "| --- | --- | --- | --- |\n"
    "| 10 | 11 | 12 | 13 |\n"
    "| 20 | 21 | 22 | 23 |\n"
    "| 30 | 31 | 32 | 33 |\n"
)


def _rung(name: str, model: str, *, sleep: float, passes: bool, calls: list):
    def _judge(crop_path, markdown, prior_findings):
        calls.append(name)
        time.sleep(sleep)
        if passes:
            return RungResult(
                rung=name,
                ok=True,
                verdict=TableJudgeVerdict(verdict="PASS", confidence="high", findings=[]),
            )
        return RungResult(rung=name, ok=False, error="slow peer gave no verdict")

    _judge.rung_kind = "ollama"
    _judge.rung_id = name
    _judge.executing = model
    return _judge


def _pipeline(tmp_path: Path, **overrides) -> tuple[UnifiedPipeline, DocumentState]:
    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        save_figures=False,
        write_manifest=False,
        table_judge_ladder=True,
        **{"quiet": True, **overrides},
    )
    pipeline = UnifiedPipeline(config)
    pipeline._binding_evidence_for_witness = lambda *a, **kw: (None, BindingEvidence.ABSTAIN)
    pipeline._build_table_cell_adjudicator = lambda: None
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=_ruled_pdf(tmp_path), page_count=2)
    return pipeline, DocumentState(handle=handle)


def _gate(pipeline, state, page_num, rungs) -> float:
    bo = PageOutput(
        page_num=page_num,
        text=_TABLE_MD,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    start = time.monotonic()
    pipeline._run_table_judge_gate(state, page_num, state.pages[page_num], bo, rungs)
    return time.monotonic() - start


def _kinds(state: DocumentState, page_num: int) -> list[str]:
    return [e.kind for e in state.events if e.page_num == page_num]


def test_budget_exhausted_page_is_unverified_one_event_next_page_normal(tmp_path):
    pipeline, state = _pipeline(tmp_path, table_judge_page_budget_sec=BUDGET)
    calls: list[str] = []
    slow = [_rung(f"r{i}", f"m{i}", sleep=SLOW, passes=False, calls=calls) for i in range(4)]

    elapsed = _gate(pipeline, state, 1, slow)

    assert calls == ["r0", "r1"], "calls starting past the budget must be skipped"
    assert elapsed < BUDGET + SLOW + 1.0, elapsed  # budget + one call + CI slack
    budget_events = [e for e in state.events if e.kind == TABLE_LADDER_BUDGET_EXHAUSTED_KIND]
    assert len(budget_events) == 1
    assert budget_events[0].page_num == 1
    assert f"{BUDGET:g}s" in budget_events[0].detail
    assert _kinds(state, 1).count(TABLE_LADDER_UNVERIFIED_KIND) == 1
    assert state.pages[1].table_ladder_disposition is not None

    # Page 2: a fresh budget, a healthy rung. The exhausted budget must not leak.
    fast = _rung("ok", "mfast", sleep=0.0, passes=True, calls=calls)
    _gate(pipeline, state, 2, [fast])
    assert _kinds(state, 2) == [TABLE_LADDER_ACCEPTED_KIND]
    assert pipeline._ladder_budget is None
    assert len([e for e in state.events if e.kind == TABLE_LADDER_BUDGET_EXHAUSTED_KIND]) == 1


def test_difference_pin_same_page_without_a_tight_budget_runs_every_rung(tmp_path):
    """Only the budget differs: the loose run calls all three rungs, no budget event."""
    pipeline, state = _pipeline(tmp_path, table_judge_page_budget_sec=60.0)
    calls: list[str] = []
    slow = [_rung(f"r{i}", f"m{i}", sleep=SLOW, passes=False, calls=calls) for i in range(3)]
    _gate(pipeline, state, 1, slow)
    assert calls == ["r0", "r1", "r2"]
    assert TABLE_LADDER_BUDGET_EXHAUSTED_KIND not in _kinds(state, 1)
    assert _kinds(state, 1).count(TABLE_LADDER_UNVERIFIED_KIND) == 1


def test_default_budget_is_timeout_times_stages(tmp_path):
    pipeline, _ = _pipeline(tmp_path, table_judge_timeout_sec=7.0)
    assert pipeline._page_ladder_budget_sec([object(), object()]) == 7.0 * 3
    override, _ = _pipeline(tmp_path, table_judge_page_budget_sec=1.5)
    assert override._page_ladder_budget_sec([object()]) == 1.5


def test_one_console_line_per_rung_call(tmp_path):
    pipeline, state = _pipeline(tmp_path, table_judge_page_budget_sec=60.0, quiet=False)
    calls: list[str] = []
    rungs = [_rung(f"r{i}", f"model{i}", sleep=0.0, passes=False, calls=calls) for i in range(2)]
    printed: list[str] = []
    with patch.object(orchestrator.console, "print", lambda msg, *a, **k: printed.append(msg)):
        _gate(pipeline, state, 1, rungs)
    lines = [m for m in printed if "table ladder" in m]
    assert len(lines) == 2 == len(calls)
    assert "r0" in lines[0] and "model0" in lines[0] and re.search(r"\d+\.\ds", lines[0])
    assert "r1" in lines[1] and "model1" in lines[1]


def test_quiet_prints_nothing(tmp_path):
    pipeline, state = _pipeline(tmp_path, table_judge_page_budget_sec=60.0, quiet=True)
    calls: list[str] = []
    printed: list[str] = []
    with patch.object(orchestrator.console, "print", lambda msg, *a, **k: printed.append(msg)):
        _gate(pipeline, state, 1, [_rung("r0", "m", sleep=0.0, passes=False, calls=calls)])
    assert printed == []


def test_adjudicator_and_cell_transcribe_respect_the_budget(tmp_path):
    clock = iter([0.0, 100.0, 100.0, 100.0, 100.0]).__next__
    budget = PageLadderBudget(1.0, clock=clock)
    ran: list[str] = []

    def adjudicator(crop, refs):
        ran.append("adjudicator")

    adjudicator.rung_id = "adjudicator:m"
    wrapped = budget.wrap_adjudicator(adjudicator)
    result = wrapped(None, ["R1C1"])
    assert ran == [] and result.ok is False and "1s" in result.error
    assert wrapped.rung_id == "adjudicator:m", "identity attributes must survive the wrap"
    assert budget.wrap_adjudicator(None) is None

    pipeline, _ = _pipeline(tmp_path)
    pipeline._ladder_budget = budget
    with patch("socr.judge.cell_transcribe.transcribe_cell", side_effect=AssertionError("called")):
        assert pipeline._transcribe_cell_token(Path("crop.png")) is None
    assert budget.skipped == 2


def test_gate_hands_the_guard_chain_a_budgeted_adjudicator(tmp_path):
    """The wiring, not just the wrapper: what ``evaluate_cell_guard`` receives."""
    pipeline, state = _pipeline(tmp_path, table_judge_page_budget_sec=60.0)
    seen: list = []

    def fake_adjudicator(crop, refs):
        raise AssertionError("not called here")

    fake_adjudicator.rung_id = "adjudicator:m"
    pipeline._build_table_cell_adjudicator = lambda: fake_adjudicator

    def spy(**kwargs):
        seen.append(kwargs["adjudicator"])
        raise RuntimeError("fail closed")  # the guard chain's own handler absorbs this

    rejecting = RungResult(
        rung="r0",
        ok=True,
        verdict=TableJudgeVerdict(verdict="FAIL", confidence="high", findings=[]),
    )

    def rung(crop, md, prior):
        return rejecting

    rung.rung_id = "r0"
    with patch.object(orchestrator, "evaluate_cell_guard", spy):
        _gate(pipeline, state, 1, [rung, rung])
    assert len(seen) == 1
    assert seen[0] is not fake_adjudicator and seen[0].__wrapped__ is fake_adjudicator
