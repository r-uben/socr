"""#1361: a table the ladder WITHHOLDS on a free reading gets one paid reading, checked like a routed one.

Hermetic per CLAUDE.md: the engines, the agentic ladder, the page judge, the table gate's outcome
and the cell-binding / native-contradiction checks are pinned, and no claim is an absolute outcome
measured on this machine. Every claim is a DIFFERENCE between two arms run in one process: the
lane switched off (``_paid_reading_after_table_withhold`` replaced by the identity) against the
lane on, or two lane arms differing in exactly one input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_gh1185_text_layer_numbers_e2e import _doc_dir, _page, _prose, _rotated_pdf

import socr
from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.marks import Check, CheckResult
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core import verdict as verdict_module
from socr.core.verdict import (
    PAID_AFTER_DISPUTE_REASON,
    PAID_AFTER_TABLE_WITHHOLD_REASON,
    PERMANENTLY_UNRESOLVABLE_MODES,
)
from socr.pipeline import orchestrator as orch
from socr.pipeline.agentic import AcceptDecision
from socr.pipeline.orchestrator import PAID_AFTER_TABLE_WITHHOLD_KIND, UnifiedPipeline
from socr.tables.reconcile import find_table_blocks

#: The cell the free reader shifted. A table carrying it is a table the stub ladder withholds.
SHIFTED = "9.9"
HEADER = "| Country | Q1 | Q2 | Q3 | Q4 | Q5 |\n| --- | --- | --- | --- | --- | --- |\n"
GOOD_KOREA = "4.0 | 2.6 | 5.7 | -.8 | 4.0"
SHIFT_KOREA = f"2.6 | 5.7 | -.8 | 4.0 | {SHIFTED}"


#: The page's rows as prose, the one form the text-layer checks owe the reading (the same line the
#: #1185 fixture's reader ships), so the table is the only thing the two readers differ on.
_PROSE_ROWS = (
    "Korea grew 4.0 2.6 5.7 -.8 4.0. China grew 6.9 7.0 6.6 6.4 7.1. Mexico grew 2.3 1.1 2.5 2.4."
)


def _table(korea: str) -> str:
    """The page's rotated rows, as the PDF prints them (see ``_rotated_pdf``: three rows, six
    times), so the layer owes the reading every number and the table is all the readers differ on."""
    rows = [f"| Korea | {korea} |", "| China | 6.9 | 7.0 | 6.6 | 6.4 | 7.1 |"]
    rows.append("| Mexico | 2.3 | 1.1 | 2.5 | 2.4 |  |")
    first = HEADER + "\n".join(rows) + "\n"
    again = "\n".join([f"| Korea | {GOOD_KOREA} |", *rows[1:]]) + "\n"
    return first + again * 5


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


class _Reader:
    """Free reader (``qwen``) reads *free_korea*; the paid one (``gemini``) reads *paid_korea*."""

    def __init__(self, name: str, calls: list, free_korea: str, paid_korea: str) -> None:
        self.name = name
        self.calls = calls
        self.korea = free_korea if name == "qwen" else paid_korea

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        self.calls.append((self.name, tuple(page_nums)))
        if self.korea is None:  # a paid reader that answers with nothing
            return [
                PageOutput(page_num=n, text="", status=PageStatus.ERROR, engine=self.name)
                for n in page_nums
            ]
        return [
            PageOutput(
                page_num=n,
                text=f"{_prose(n)}. {_PROSE_ROWS}\n\n{_table(self.korea)}",
                status=PageStatus.SUCCESS,
                engine=self.name,
            )
            for n in page_nums
        ]


class _Judge:
    """The page judge: a rung is accepted unless a reason is given for its engine."""

    def __init__(self, free_reason: str = "", paid_reason: str = "") -> None:
        self.free_reason = free_reason
        self.paid_reason = paid_reason

    def assess(self, output, provider):
        reason = self.paid_reason if provider.engine.value == "gemini" else self.free_reason
        if reason:
            return AcceptDecision(accept=False, reason=reason)
        return AcceptDecision(accept=True, reason="faithful")


def _stub_gate(state, page_num, ps, bo, rungs) -> None:
    """The table ladder's outcome, pinned: a table carrying the shifted cell is withheld."""
    withheld = SHIFTED in (bo.text or "")
    for index in range(len(find_table_blocks(bo.text or ""))):
        state.events.append(
            AuditEvent(
                page_num=page_num,
                kind="table_ladder_withheld" if withheld else "table_ladder_accepted",
                engine=bo.engine or "",
                detail="stub ladder",
                data={"table_id": f"p{page_num}-t{index}"},
            )
        )
    if withheld:
        ps.table_ladder_disposition = FailureMode.TABLE_WITHHELD


def _run(
    out_dir: Path,
    pdf: Path,
    monkeypatch,
    *,
    lane: bool = True,
    free_reason: str = "",
    paid_reason: str = "",
    free_korea: str = SHIFT_KOREA,
    paid_korea: str | None = GOOD_KOREA,
    calls: list | None = None,
    reprocess: bool = False,
    strict_local: bool = False,
    cost_budget: float = 0.0,
    max_paid_pages: int | None = None,
):
    calls = calls if calls is not None else []
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda et: _Reader(et.value, calls, free_korea, paid_korea))
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
                native_first=True,
                write_manifest=True,
                reprocess=reprocess,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
                describe_figure_crops=False,
                strict_local=strict_local,
                cost_budget=cost_budget,
                max_paid_pages=max_paid_pages,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
        pipe._build_page_judge = lambda state: _Judge(free_reason, paid_reason)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        pipe._run_table_judge_gate = _stub_gate
        pipe._run_native_cell_binding_check = lambda *a, **k: None
        # The checks on the shipped reading that compare it with this fixture's synthetic rotated
        # layer are orthogonal to the lane and would mark BOTH arms; they are pinned off so the
        # lane's own clause is the only one a page carries.
        pipe._run_row_binding_check = lambda *a, **k: False
        for name in (
            "text_layer_numbers_clause",
            "text_layer_drops_clause",
            "text_layer_words_clause",
        ):
            m.setattr(verdict_module, name, lambda output, p: None)
        pipe._withhold_contradicted_unverified_tables = lambda *a, **k: None
        if not lane:
            pipe._paid_reading_after_table_withhold = lambda state, page_num, ps, bo, **k: bo
        return pipe.process(pdf, output_dir=out_dir)


def _paid_calls(calls: list) -> list:
    return [c for c in calls if c[0] == "gemini"]


def _clauses(sidecar: dict, check: str) -> list[dict]:
    return [c for c in sidecar["verdict"]["clauses"] if c["check"] == check]


def _events(out: Path) -> list[dict]:
    log = json.loads((_doc_dir(out) / "audit_log.json").read_text())
    return log if isinstance(log, list) else log.get("events", [])


def _lane_events(out: Path) -> list[dict]:
    return [e for e in _events(out) if e.get("kind") == PAID_AFTER_TABLE_WITHHOLD_KIND]


def _kinds(out: Path) -> list[str]:
    return [e["kind"] for e in _events(out)]


def _shipped(out: Path) -> tuple:
    """What the page ends as: status, mode, audit verdict, the body that ships, its text."""
    side, body = _page(out, 1)
    w = side["winning_output"]
    return side["status"], w["failure_mode"], side["audit_passed"], body, w["text"]


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    return _rotated_pdf(tmp_path / "doc.pdf", pages=1)


def test_the_paid_reading_ships_marked_and_the_arm_without_the_withhold_pays_nothing(
    tmp_path: Path, monkeypatch, pdf: Path
) -> None:
    """Two arms, one input: the free reader read the table shifted (withheld) or right."""
    held, held_calls = tmp_path / "held", []
    clean, clean_calls = tmp_path / "clean", []
    _run(held, pdf, monkeypatch, calls=held_calls)
    _run(clean, pdf, monkeypatch, calls=clean_calls, free_korea=GOOD_KOREA)
    assert len(_paid_calls(held_calls)) == 1
    assert _paid_calls(clean_calls) == []

    h_side, h_body = _page(held, 1)
    c_side, c_body = _page(clean, 1)
    w = h_side["winning_output"]
    assert w["engine"] == "gemini"
    assert w["paid_after_table_withhold"] is True
    assert "paid_after_witness_dispute" not in w
    assert h_side["status"] == "warning"
    # the mode is whichever clause names one first (here #1251's); it is one a resume restores
    assert FailureMode(w["failure_mode"]) in PERMANENTLY_UNRESOLVABLE_MODES
    [clause] = _clauses(h_side, Check.READING_DISPUTE.value)
    assert (clause["result"], clause["scope"], clause["reason"]) == (
        CheckResult.UNKNOWN.value,
        "page",
        PAID_AFTER_TABLE_WITHHOLD_REASON,
    )
    assert PAID_AFTER_TABLE_WITHHOLD_REASON != PAID_AFTER_DISPUTE_REASON
    assert "withheld by the table checks" in clause["reason"]
    # the text is the paid reading, and it is a table, not the failure marker
    assert GOOD_KOREA in h_body and SHIFTED not in h_body
    assert "unverifiable table" not in h_body
    # the control arm shipped the free reading with no clause and no flag
    assert c_side["winning_output"]["engine"] == "qwen"
    assert _clauses(c_side, Check.READING_DISPUTE.value) == []
    assert "paid_after_table_withhold" not in c_side["winning_output"]
    assert "reading_dispute" not in c_body


def test_the_lane_off_arm_ends_withheld_and_the_lane_on_arm_does_not(
    tmp_path: Path, monkeypatch, pdf: Path
) -> None:
    off, off_calls = tmp_path / "off", []
    on, on_calls = tmp_path / "on", []
    _run(off, pdf, monkeypatch, lane=False, calls=off_calls)
    _run(on, pdf, monkeypatch, lane=True, calls=on_calls)
    assert _paid_calls(off_calls) == []
    assert len(_paid_calls(on_calls)) == 1
    o_status, o_mode, o_audit, o_body, _ = _shipped(off)
    assert (o_status, o_mode, o_audit) == ("error", FailureMode.TABLE_WITHHELD.value, False)
    assert "unverifiable table" in o_body
    n_status, n_mode, _audit, n_body, _ = _shipped(on)
    assert (n_status, n_mode) != (o_status, o_mode)
    assert GOOD_KOREA in n_body


def test_the_audit_log_records_the_escalation_and_replaces_the_earlier_ladder_verdict(
    tmp_path: Path, monkeypatch, pdf: Path
) -> None:
    out = tmp_path / "out"
    _run(out, pdf, monkeypatch)
    [event] = _lane_events(out)
    assert event["data"]["outcome"] == "accepted"
    assert event["data"]["earlier_engine"] == "qwen"
    assert {s["kind"] for s in event["data"]["superseded"]} == {"table_ladder_withheld"}
    kinds = _kinds(out)
    # the free reading's withheld verdict is gone from the page's record; the paid one's stands
    assert "table_ladder_withheld" not in kinds
    assert kinds.count("table_ladder_accepted") == 1


@pytest.mark.parametrize(
    "arm",
    [
        pytest.param({"paid_reason": "judge: numbers do not match"}, id="page-judge"),
        pytest.param(
            {"paid_reason": "source_evidence_table: numeric tokens unsupported by page evidence"},
            id="witness",
        ),
        pytest.param({"paid_korea": SHIFT_KOREA}, id="own-table-ladder"),
    ],
)
def test_a_paid_reading_failing_any_check_leaves_the_page_as_the_lane_off_arm_ends_it(
    tmp_path: Path, monkeypatch, pdf: Path, arm: dict
) -> None:
    off, on, calls = tmp_path / "off", tmp_path / "on", []
    _run(off, pdf, monkeypatch, lane=False)
    _run(on, pdf, monkeypatch, lane=True, calls=calls, **arm)
    assert len(_paid_calls(calls)) == 1, "the paid reader was called, and refused"
    assert _shipped(on) == _shipped(off)
    # the page's ladder record is the earlier one, untouched; only the lane's event is new
    assert [k for k in _kinds(on) if k != PAID_AFTER_TABLE_WITHHOLD_KIND] == _kinds(off)
    [event] = _lane_events(on)
    assert event["data"]["outcome"].startswith("rejected_by_")
    side, _ = _page(on, 1)
    assert "paid_after_table_withhold" not in side["winning_output"]
    # a refused paid reading was still paid for
    assert side["page_cost_usd"] == PROFILE_GEMINI.cost_per_page_usd


def test_cost_accounting_equals_the_routed_paid_rung(
    tmp_path: Path, monkeypatch, pdf: Path
) -> None:
    """The routed arm: the judge refuses the free rung, so ``route_page`` pays the reader."""
    lane, routed = tmp_path / "lane", tmp_path / "routed"
    _run(lane, pdf, monkeypatch)
    _run(routed, pdf, monkeypatch, lane=False, free_reason="judge: refused")
    l_side, _ = _page(lane, 1)
    r_side, _ = _page(routed, 1)
    assert l_side["page_cost_usd"] == r_side["page_cost_usd"] == PROFILE_GEMINI.cost_per_page_usd
    assert l_side["cost_usd"] == r_side["cost_usd"]
    assert l_side["winning_output"]["cost_usd"] == r_side["winning_output"]["cost_usd"]


def test_the_budget_decides_exactly_as_the_routed_rung_does(
    tmp_path: Path, monkeypatch, pdf: Path
) -> None:
    price = PROFILE_GEMINI.cost_per_page_usd
    assert price > 0
    off, tight, roomy = tmp_path / "off", tmp_path / "tight", tmp_path / "roomy"
    tight_calls, roomy_calls = [], []
    _run(off, pdf, monkeypatch, lane=False)
    _run(tight, pdf, monkeypatch, calls=tight_calls, cost_budget=price / 2)
    _run(roomy, pdf, monkeypatch, calls=roomy_calls, cost_budget=price * 2)
    assert _paid_calls(tight_calls) == []
    assert len(_paid_calls(roomy_calls)) == 1
    assert _shipped(tight) == _shipped(off)
    assert _page(tight, 1)[0]["page_cost_usd"] == 0.0
    [event] = _lane_events(tight)
    assert event["data"]["outcome"] == "budget_exhausted"
    assert _shipped(roomy) != _shipped(off)


def test_the_paid_page_cap_blocks_the_lane_like_the_route(
    tmp_path: Path, monkeypatch, pdf: Path
) -> None:
    off, capped, calls = tmp_path / "off", tmp_path / "capped", []
    _run(off, pdf, monkeypatch, lane=False)
    _run(capped, pdf, monkeypatch, calls=calls, max_paid_pages=0)
    assert _paid_calls(calls) == []
    assert _shipped(capped) == _shipped(off)
    assert _page(capped, 1)[0]["page_cost_usd"] == 0.0


def test_strict_local_never_calls_the_paid_reader(tmp_path: Path, monkeypatch, pdf: Path) -> None:
    off, strict, calls = tmp_path / "off", tmp_path / "strict", []
    _run(off, pdf, monkeypatch, lane=False, strict_local=True)
    _run(strict, pdf, monkeypatch, calls=calls, strict_local=True)
    assert _paid_calls(calls) == []
    assert _shipped(strict) == _shipped(off)
    assert _lane_events(strict) == []


def test_a_page_a_paid_reader_already_read_gets_no_second_call(
    tmp_path: Path, monkeypatch, pdf: Path
) -> None:
    """The judge refuses the free rung, so ``route_page`` pays once; the paid reading's own table
    is then withheld (it carries the shifted cell). The lane must not pay again."""
    out, calls = tmp_path / "out", []
    _run(out, pdf, monkeypatch, free_reason="judge: refused", paid_korea=SHIFT_KOREA, calls=calls)
    assert len(_paid_calls(calls)) == 1
    assert _lane_events(out) == []
    assert "paid_after_table_withhold" not in _page(out, 1)[0]["winning_output"]


def _lane_call(ps, bo, route_calls: list):
    """Drive the lane's method directly on a page whose ladder ended WITHHELD."""
    from types import SimpleNamespace

    state = SimpleNamespace(
        events=[],
        total_cost=0.0,
        handle=SimpleNamespace(path=Path("doc.pdf")),
        record_engine_run=lambda *a, **k: None,
    )
    pipe = UnifiedPipeline(PipelineConfig(agentic=True, quiet=True, judge_backend="heuristic"))

    def route_provider(profile, page_num):
        route_calls.append(profile.engine.value)
        raise RuntimeError("stub reader: no answer")

    got = pipe._paid_reading_after_table_withhold(
        state,
        1,
        ps,
        bo,
        ladder=[PROFILE_QWEN_LOCAL, PROFILE_GEMINI],
        judge=_Judge(),
        route_provider=route_provider,
        provider_timeout=None,
        on_candidate=None,
        table_gate=lambda page_num, ps, out: None,
    )
    return got, state


def _withheld_page(*extra_attempts):
    from socr.core.state import PageState

    free = PageOutput(
        page_num=1,
        text=f"{_table(SHIFT_KOREA)}",
        status=PageStatus.SUCCESS,
        engine="qwen",
        provider_id=PROFILE_QWEN_LOCAL.id,
    )
    ps = PageState(page_num=1, is_born_digital=True)
    ps.attempts.extend([free, *extra_attempts])
    ps.best_output = free
    ps.table_ladder_disposition = FailureMode.TABLE_WITHHELD
    return ps, free


def test_a_page_with_any_paid_attempt_is_not_eligible_however_that_attempt_ended() -> None:
    """A paid attempt that timed out or raised costs nothing and holds no text, and is still the
    page's one paid call. Pinned as a difference: the same page without it IS eligible."""
    attempt = PageOutput(
        page_num=1, text="", status=PageStatus.ERROR, engine="gemini", provider_id=PROFILE_GEMINI.id
    )
    eligible_calls: list = []
    ps, free = _withheld_page()
    _lane_call(ps, free, eligible_calls)
    assert eligible_calls == [PROFILE_GEMINI.engine.value]

    blocked_calls: list = []
    ps, free = _withheld_page(attempt)
    got, state = _lane_call(ps, free, blocked_calls)
    assert blocked_calls == []
    assert got is free and state.events == []


def test_unknown_page_spend_fails_closed() -> None:
    calls: list = []
    ps, free = _withheld_page()
    ps.page_cost_usd = None
    got, _ = _lane_call(ps, free, calls)
    assert calls == [] and got is free


def test_a_resumed_page_restores_without_a_second_paid_call(
    tmp_path: Path, monkeypatch, pdf: Path
) -> None:
    out = tmp_path / "out"
    first: list = []
    _run(out, pdf, monkeypatch, calls=first)
    before = {p.name: p.read_bytes() for p in (_doc_dir(out) / "pages").glob("*")}
    second: list = []
    _run(out, pdf, monkeypatch, calls=second, reprocess=True)
    assert len(_paid_calls(first)) == 1
    assert second == []
    assert {p.name: p.read_bytes() for p in (_doc_dir(out) / "pages").glob("*")} == before
    # the record of the escalation survives the resume
    assert [e["data"]["outcome"] for e in _lane_events(out)] == ["accepted"]


def test_the_flag_survives_the_sidecar_round_trip() -> None:
    out = PageOutput(
        page_num=1,
        text="x",
        status=PageStatus.SUCCESS,
        engine="gemini",
        paid_after_table_withhold=True,
    )
    assert PageOutput.from_dict(out.to_dict()).paid_after_table_withhold is True
    plain = PageOutput(page_num=1, text="x", status=PageStatus.SUCCESS, engine="gemini")
    assert "paid_after_table_withhold" not in plain.to_dict()


def test_replay_does_not_block_a_recorded_paid_rung_that_followed_an_accepted_free_one(
    tmp_path: Path,
) -> None:
    """cascade_replay decides paid calls from the recorded journal. The lane's call follows an
    ACCEPTED free rung, which no block rule reads, so replay counts it under the old and the new
    rules alike: nothing in replay needs to know about the lane."""
    from socr.routing.cascade_replay import ReplayRules, replay_manifest

    manifest = {
        "entries": {
            "1": {
                "page_num": 1,
                "fingerprint": {"engine": "gemini"},
                "journal": [
                    {"engine": "qwen", "accepted": True, "reason": "faithful"},
                    {"engine": "gemini", "accepted": True, "reason": "faithful"},
                ],
            }
        }
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    for rules in (ReplayRules(), ReplayRules(block_paid_on_source_evidence=True)):
        report = replay_manifest(path, rules)
        assert (report.old_gemini_pages, report.new_gemini_pages) == (1, 1)


def test_the_clause_alone_names_the_mode_and_its_own_reason() -> None:
    """With no other clause on the page the mode is the lane's, and a resume restores it."""
    from test_gh1346_paid_after_dispute_marked import _p
    from socr.core.manifest import apply_verdict_marks

    out = PageOutput(
        page_num=1,
        text="Korea grew 4.0 2.6.",
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
        judge_accepted=True,
        paid_after_table_withhold=True,
    )
    shipped, verdict = apply_verdict_marks(out, _p(), 1)
    assert shipped.status is PageStatus.WARNING
    assert shipped.failure_mode is FailureMode.PAID_READING_AFTER_DISPUTE
    [clause] = [c for c in verdict.clauses if c.check is Check.READING_DISPUTE]
    assert clause.reason == PAID_AFTER_TABLE_WITHHOLD_REASON
    # the flag set for the OTHER cause keeps #1346's wording: the two reasons never mix
    other = PageOutput(
        **{**out.__dict__, "paid_after_table_withhold": False, "paid_after_witness_dispute": True}
    )
    _, other_verdict = apply_verdict_marks(other, _p(), 1)
    [other_clause] = [c for c in other_verdict.clauses if c.check is Check.READING_DISPUTE]
    assert other_clause.reason == PAID_AFTER_DISPUTE_REASON
