"""GH-974 review round 1 (Astra): shared budget, guard-chain exhaustion, finalized
status through ``process()``, and the resume fingerprint. Hermetic: fake rungs,
no Ollama, no provider (``_available_engines_for_agentic`` and
``_resolve_judge_model`` patched wherever ``process()`` runs).

Every pin is a DIFFERENCE between two runs that change only the budget.
"""

from __future__ import annotations

import contextlib
import functools
import json
import time
from pathlib import Path
from unittest.mock import patch

import fitz

from socr.core.config import PipelineConfig  # noqa: F401
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge import ladder_budget
from socr.judge.ladder_budget import TABLE_LADDER_BUDGET_EXHAUSTED_KIND
from socr.judge.table_rung_ollama import BlindCellResult
from socr.judge.table_verdict import (
    TABLE_LADDER_UNVERIFIED_KIND,
    Finding,
    FindingCode,
    RungResult,
    TableJudgeVerdict,
)
from socr.pipeline.orchestrator import UnifiedPipeline
from test_gh974_page_budget_daemon import BUDGET, SLOW, _gate, _pipeline, _rung
from test_ladder_e2e import (
    CLEAN_MD,
    CLEAN_PAGE,
    SHIFT_CORRECT_MD,
    SHIFT_PAGE,
    SHIFT_SHIFTED_MD,
    _fixture_copy,
    _make_config,
    _pass_verdict,
    _process_and_capture,
    _route_page_per_page,
)


def _budget_events(state):
    return [e for e in state.events if e.kind == TABLE_LADDER_BUDGET_EXHAUSTED_KIND]


# -- multi-table: one budget per PAGE ------------------------------------------


def _two_table_pdf(tmp_path: Path) -> tuple[Path, str]:
    """One page, two stacked ruled grids with distinct cell text, plus its markdown."""
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 220, 300, 380, 460]
    md_parts = []
    for t, top in enumerate((100, 300)):
        rows = [top + i * 22 for i in range(4)]
        grid = [[f"{t}{r}{c}" for c in range(4)] for r in range(3)]
        for r in range(3):
            for c in range(4):
                page.insert_text((cols[c] + 4, rows[r] + 15), grid[r][c], fontsize=9)
        for yy in rows:
            page.draw_line((cols[0], yy), (cols[-1], yy))
        for xx in cols:
            page.draw_line((xx, rows[0]), (xx, rows[-1]))
        md_parts.append(
            "| "
            + " | ".join(grid[0])
            + " |\n| --- | --- | --- | --- |\n"
            + "".join("| " + " | ".join(row) + " |\n" for row in grid[1:])
        )
    path = tmp_path / "two.pdf"
    doc.save(path)
    doc.close()
    return path, "\n".join(md_parts)


class _VirtualClock:
    """A monotonic clock that only moves when a fake rung "runs" (GH-991).

    The budget logic compares elapsed time against a budget; real ``time.sleep``
    made that comparison depend on scheduler latency, so a loaded machine could
    push a third call inside (or out of) the budget. Virtual time makes the
    comparison exact: each rung call costs exactly ``SLOW`` virtual seconds.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def timed(self, rung):
        """``rung`` made to take ``SLOW`` virtual seconds instead of sleeping."""

        @functools.wraps(rung)
        def _run(crop_path, markdown, prior_findings):
            try:
                return rung(crop_path, markdown, prior_findings)
            finally:
                self.now += SLOW

        return _run


@contextlib.contextmanager
def _virtual_budget_clock():
    """Make every ``PageLadderBudget`` the orchestrator builds read a virtual clock."""
    clock = _VirtualClock()
    real = ladder_budget.PageLadderBudget
    with patch.object(
        ladder_budget,
        "PageLadderBudget",
        lambda *a, **kw: real(*a, clock=clock, **kw),
    ):
        yield clock


def _virtual_rungs(clock: _VirtualClock, calls: list[str], n: int):
    return [
        clock.timed(_rung(f"r{i}", f"m{i}", sleep=0.0, passes=False, calls=calls)) for i in range(n)
    ]


def test_budget_is_shared_across_the_tables_on_one_page(tmp_path):
    pdf, md = _two_table_pdf(tmp_path)

    def run(budget):
        pipeline, state = _pipeline(tmp_path, table_judge_page_budget_sec=budget)
        with patch.object(DocumentHandle, "__post_init__", lambda self: None):
            state.handle = DocumentHandle(path=pdf, page_count=1)
        calls: list[str] = []
        with _virtual_budget_clock() as clock:
            rungs = _virtual_rungs(clock, calls, 3)
            bo = PageOutput(page_num=1, text=md, status=PageStatus.SUCCESS, engine="qwen")
            pipeline._run_table_judge_gate(state, 1, state.pages[1], bo, rungs)
        return calls, state

    calls, state = run(BUDGET)
    unverified = [e for e in state.events if e.kind == TABLE_LADDER_UNVERIFIED_KIND]
    assert len(unverified) == 2, "both tables must be witnessed, or this pins nothing"
    # Table 1 spends the budget (r0, r1); table 2 gets no call of its own.
    assert calls == ["r0", "r1"]
    assert len(_budget_events(state)) == 1

    # Same page, same THREE rungs; only the budget is raised.
    loose_calls, loose_state = run(60.0)
    assert loose_calls == ["r0", "r1", "r2"] * 2
    assert not _budget_events(loose_state)


def test_same_page_only_the_budget_differs(tmp_path):
    """Replaces the old 4-versus-3 comparison: equal rung counts, one knob."""
    results = {}
    for budget in (BUDGET, 60.0):
        pipeline, state = _pipeline(tmp_path, table_judge_page_budget_sec=budget)
        calls: list[str] = []
        with _virtual_budget_clock() as clock:
            _gate(pipeline, state, 1, _virtual_rungs(clock, calls, 4))
        results[budget] = (calls, len(_budget_events(state)))
    assert results[BUDGET] == (["r0", "r1"], 1)
    assert results[60.0] == (["r0", "r1", "r2", "r3"], 0)


# -- adjudicator ---------------------------------------------------------------


def test_exhausted_adjudicator_leaves_the_page_unverified(tmp_path):
    """The reader REJECTS, so the guard chain would ask the adjudicator; the
    budget is spent by then."""

    def run(budget):
        pipeline, state = _pipeline(tmp_path, table_judge_page_budget_sec=budget)
        asked: list = []

        def adjudicator(crop, refs):
            asked.append(list(refs))
            return BlindCellResult(rung="adjudicator:m", ok=False, error="unscripted")

        adjudicator.rung_kind = "adjudicator"
        adjudicator.rung_id = "adjudicator:m"
        pipeline._build_table_cell_adjudicator = lambda: adjudicator
        verdict = TableJudgeVerdict(
            verdict="FAIL",
            confidence="high",
            findings=[Finding(code=FindingCode.WRONG_BINDING, where="R1C1", detail="x")],
        )

        def rung(crop, md, prior):
            time.sleep(SLOW)
            return RungResult(rung="r0", ok=True, verdict=verdict)

        rung.rung_id = "r0"
        _gate(pipeline, state, 1, [rung])
        return asked, state

    asked, state = run(SLOW / 3)
    assert asked == [], "an exhausted budget must not reach the adjudicator"
    assert state.pages[1].table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
    assert len(_budget_events(state)) == 1

    asked_loose, _ = run(60.0)
    assert asked_loose, "control: with budget the guard chain does ask the adjudicator"


# -- process(): finalized status and the transcriber ---------------------------


class _KeyedRung:
    """Delay and outcome chosen by the markdown handed in. The delay is VIRTUAL
    (GH-1034): it advances ``clock`` instead of sleeping, so the budget comparison
    cannot depend on scheduler latency."""

    def __init__(self, rung_id, script, clock):
        self.rung_id = rung_id
        self.executing = rung_id
        self._script = script
        self._clock = clock

    def __call__(self, crop_path, markdown, prior_findings):
        delay, passes = self._script.get(markdown.strip(), (0.0, True))
        self._clock.now += delay
        if passes:
            return RungResult(rung=self.rung_id, ok=True, verdict=_pass_verdict("high"))
        return RungResult(rung=self.rung_id, ok=False, error="slow peer gave no verdict")


def _process(tmp_path, name, text_by_page, make_rungs, **cfg):
    """``make_rungs(clock)`` builds the rungs against the run's virtual clock."""
    pdf = _fixture_copy(tmp_path, name)
    pipeline = UnifiedPipeline(_make_config(table_judge_ladder=True, **cfg))
    with contextlib.ExitStack() as stack:
        for p in (
            patch("socr.pipeline.orchestrator.route_page", _route_page_per_page(text_by_page)),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            # These scenarios pin the ladder's binding clamp on a row-shifted page. The PDF's own
            # text contradicts that page, so the #1022 native-contradiction withhold would end it
            # WITHHELD; it is stubbed here and pinned in test_native_contradiction.py.
            patch.object(pipeline, "_withhold_contradicted_unverified_tables", return_value=None),
            patch.object(pipeline, "_plan_native_table_first", return_value=None),
        ):
            stack.enter_context(p)
        clock = stack.enter_context(_virtual_budget_clock())
        stack.enter_context(
            patch.object(pipeline, "_build_table_judge_rungs", return_value=make_rungs(clock))
        )
        result, state = _process_and_capture(pipeline, pdf, tmp_path / f"{name}_out")
    return result, state, tmp_path / f"{name}_out"


def _winning(out_dir: Path, page_num: int) -> dict:
    path = next(out_dir.rglob(f"pages/{page_num:05d}.json"))
    return json.loads(path.read_text())["winning_output"]


def test_budget_exhaustion_demotes_page_and_document_through_process(tmp_path):
    """Page 1's first reader stalls past the budget, so the second is skipped; page 2
    is healthy and gets a fresh budget. Only the budget differs between the runs."""
    text = {CLEAN_PAGE: CLEAN_MD, SHIFT_PAGE: SHIFT_CORRECT_MD}

    def rungs(clock):
        return [
            _KeyedRung("r0", {CLEAN_MD.strip(): (SLOW, False)}, clock),
            _KeyedRung("r1", {}, clock),
        ]

    tight, tight_state, tight_out = _process(
        tmp_path, "tight", text, rungs, table_judge_page_budget_sec=SLOW / 3
    )
    loose, loose_state, loose_out = _process(
        tmp_path, "loose", text, rungs, table_judge_page_budget_sec=60.0
    )

    assert loose.status == DocumentStatus.SUCCESS
    assert tight.status == DocumentStatus.AUDIT_FAILED
    assert str(CLEAN_PAGE) in (tight.error or "")
    assert tight_state.pages[CLEAN_PAGE].table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
    assert tight_state.pages[SHIFT_PAGE].table_ladder_disposition is None
    assert _winning(tight_out, CLEAN_PAGE)["status"] == PageStatus.WARNING.value
    assert _winning(loose_out, CLEAN_PAGE)["status"] == PageStatus.SUCCESS.value
    assert _winning(tight_out, SHIFT_PAGE)["status"] == _winning(loose_out, SHIFT_PAGE)["status"]
    assert [e.page_num for e in _budget_events(tight_state)] == [CLEAN_PAGE]
    assert not _budget_events(loose_state)


def test_exhausted_cell_transcriber_cannot_lift_the_binding_clamp(tmp_path):
    """A shifted table gets a high PASS; only the transcriber could lift the clamp.
    With the budget spent it is not called and the table stays UNVERIFIED."""
    text = {CLEAN_PAGE: CLEAN_MD, SHIFT_PAGE: SHIFT_SHIFTED_MD}

    def run(name, budget):
        def rungs(clock):
            return [_KeyedRung("r0", {SHIFT_SHIFTED_MD.strip(): (SLOW, True)}, clock)]

        asked: list = []
        with patch(
            "socr.judge.cell_transcribe.transcribe_cell",
            side_effect=lambda *a, **k: asked.append(1),
        ):
            result, state, out = _process(
                tmp_path, name, text, rungs, table_judge_page_budget_sec=budget
            )
        return asked, result, state, out

    asked, result, state, out = run("tight", SLOW / 3)
    assert asked == [], "an exhausted budget must not reach the transcriber"
    assert state.pages[SHIFT_PAGE].table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
    assert result.status == DocumentStatus.AUDIT_FAILED
    assert _winning(out, SHIFT_PAGE)["status"] == PageStatus.WARNING.value
    assert _budget_events(state)

    asked_loose, _, loose_state, _ = run("loose", 60.0)
    assert asked_loose, "control: with budget the transcriber is reached"
    assert loose_state.pages[SHIFT_PAGE].table_ladder_disposition == FailureMode.TABLE_UNVERIFIED


# -- resume: an explicit budget is fingerprinted, the default is not -------------


def _seed_rejected(tmp_path, **cfg):
    from test_ladder_resume import _make_pipeline, _real_pdf, _seed_and_flush

    pdf = _real_pdf(tmp_path)
    out = tmp_path / "out"
    pipeline = _make_pipeline(**cfg)
    pipeline._scan_root = pdf.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    _seed_and_flush(pipeline, state, out, FailureMode.TABLE_REJECTED)
    return pdf, out, _make_pipeline


def _resume(make, pdf, out, **cfg):
    other = make(**cfg)
    other._scan_root = pdf.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    return other, other._load_terminal_page(state, 1, out)


def test_resume_with_changed_explicit_budget_reprocesses(tmp_path):
    pdf, out, make = _seed_rejected(tmp_path, table_judge_page_budget_sec=100.0)
    same, resumed = _resume(make, pdf, out, table_judge_page_budget_sec=100.0)
    assert resumed is not None, "control: the same explicit budget resumes"
    changed, resumed = _resume(make, pdf, out, table_judge_page_budget_sec=200.0)
    assert resumed is None
    assert changed._run_fingerprint() != same._run_fingerprint()


def test_default_budget_leaves_the_fingerprint_and_resume_unchanged(tmp_path):
    pdf, out, make = _seed_rejected(tmp_path)  # budget None
    _, resumed = _resume(make, pdf, out)
    assert resumed is not None
    # Unset adds no key, so a pre-GH-974 fingerprint is unchanged; setting it changes it.
    assert make()._run_fingerprint() == make(table_judge_page_budget_sec=None)._run_fingerprint()
    assert make()._run_fingerprint() != make(table_judge_page_budget_sec=5.0)._run_fingerprint()
    # Ladder off: the budget is inert and not fingerprinted.
    assert (
        make(table_judge_ladder=False)._run_fingerprint()
        == make(table_judge_ladder=False, table_judge_page_budget_sec=5.0)._run_fingerprint()
    )


def _fingerprint_extra(**cfg) -> dict:
    """The ``extra`` dict ``_run_fingerprint`` hands to the contract function."""
    import ocr_output_contract

    from test_ladder_resume import _make_pipeline

    seen: dict = {}

    def spy(*args, extra=None, **kwargs):
        seen.update(extra)
        return "fp"

    with patch.object(ocr_output_contract, "run_fingerprint", spy):
        _make_pipeline(**cfg)._run_fingerprint()
    return seen


def test_budget_key_is_present_only_when_explicit():
    """Absence IS the default, so a pre-GH-974 fingerprint (and its resumes) is intact."""
    assert "table_judge_page_budget_sec" not in _fingerprint_extra()
    assert _fingerprint_extra(table_judge_page_budget_sec=5.0)["table_judge_page_budget_sec"] == 5.0
    assert "table_judge_page_budget_sec" not in _fingerprint_extra(
        table_judge_ladder=False, table_judge_page_budget_sec=5.0
    )
