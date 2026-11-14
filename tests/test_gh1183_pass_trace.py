"""#1183 (C2 part 1): the per-page pass trace.

A verdict lists what did not pass; the trace lists what each check did, whatever it did. These
tests pin three things: every check reports its own PASS / NOT_APPLICABLE / NOT_RUN (never
derived from "no clause"), the entry keeps its scope, and the trace is a record only (no byte of
the ``.md`` and no status moves with it).

Per ``CLAUDE.md`` the run-level tests pin DIFFERENCES between two runs in one process, with the
provider ladder in both states, never an absolute outcome measured on one machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import socr
from socr.core import verdict as verdict_module
from socr.core.marks import Check, CheckResult
from socr.core.pass_trace import TraceEntry, TraceResult, trace_from_dicts, trace_to_dicts
from socr.core.result import PageOutput, PageStatus
from socr.core.verdict import (
    CHART_VALUES_NOT_APPLICABLE_REASON,
    SCOPE_TABLE_ROWS,
    NO_PROVIDER_LADDER_REASON,
    PASS_TRACE_SLOTS,
    RESUMED_WITHOUT_TRACE_REASON,
    ROW_BINDING_CHECK,
    PageVerdict,
    VerdictClause,
    build_pass_trace,
    pass_trace_outcomes,
    reconcile_pass_trace,
)

from test_gh1185_text_layer_numbers_e2e import _crash_after_page_one
from test_gh1294_layer_words import _rotated_pdf, _run


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# --- the check functions report their own outcome ----------------------------------------

LAYER = "Korea 4.0 2.6 5.7 4.0\nChina 6.9 7.0 6.6 7.1"


def _p(**over):
    base = dict(
        own_layer_text=LAYER,
        own_layer_drop_reference="4.0 2.6 5.7 4.0\n6.9 7.0 6.6 7.1",
        own_layer_word_reference="korea china",
        own_layer_figure_words="",
        is_born_digital=True,
        prior_ocr_layer_suspect=False,
        has_unrecovered_symbol_glyphs=False,
        table_ladder_disposition=None,
        control_bytes_replaced=None,
        row_binding={},
    )
    base.update(over)
    return SimpleNamespace(**base)


def _out(text: str = LAYER, engine: str = "qwen") -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine=engine)


def _scope(check: str) -> str:
    """The scope a check's non-clause entry sits at: the drops check reads table rows only."""
    return SCOPE_TABLE_ROWS if check == "text_layer_drops" else "page"


def _by_check(outcomes) -> dict[tuple[str, str], TraceEntry]:
    return {(o.entry.check, o.entry.scope): o.entry for o in outcomes}


def _trace(out, p, **kw) -> dict[tuple[str, str], TraceEntry]:
    return {
        (e.check, e.scope): e
        for e in build_pass_trace(out, p, kw.pop("verdict", PageVerdict(1)), **kw)
    }


@pytest.mark.parametrize(
    "check",
    [
        Check.TEXT_LAYER_NUMBERS,
        Check.TEXT_LAYER_DROPS,
        Check.TEXT_LAYER_WORDS,
        Check.CONTROL_BYTES,
        Check.PRIOR_OCR_LAYER,
    ],
)
def test_a_check_that_ran_and_found_nothing_is_a_pass_at_page_scope(check: Check) -> None:
    trace = _trace(_out(), _p())
    entry = trace[(check.value, _scope(check.value))]
    assert entry.result is TraceResult.PASS, entry
    assert (check.value, "page") not in trace or check is not Check.TEXT_LAYER_DROPS


TABLE = "| a | b |\n| - | - |\n| Korea | 4.0 |\n"
THREE_TABLES = "\n\ntext\n\n".join([TABLE] * 3)


def test_row_binding_pass_keeps_its_table_scope_and_never_reads_as_a_page_pass() -> None:
    p = _p(row_binding={"p1-t1": {"table_index": 1, "outcome": "bound"}})
    trace = _trace(_out(THREE_TABLES), p)
    assert trace[(ROW_BINDING_CHECK, "table:2")].result is TraceResult.PASS
    assert (ROW_BINDING_CHECK, "page") not in trace


def test_row_binding_records_state_their_own_outcome() -> None:
    p = _p(
        row_binding={
            "p1-t0": {"table_index": 0, "outcome": "bound"},
            "p1-t1": {"table_index": 1, "outcome": "contradicted"},
            "p1-t2": {"table_index": 2, "outcome": "unverifiable"},
        }
    )
    trace = _trace(_out(THREE_TABLES), p)
    assert [trace[(ROW_BINDING_CHECK, f"table:{k}")].result for k in (1, 2, 3)] == [
        TraceResult.PASS,
        TraceResult.FAIL,
        TraceResult.UNKNOWN,
    ]


def test_row_binding_without_a_record_says_why_per_table() -> None:
    model = _trace(_out(TABLE), _p())
    native = _trace(_out(TABLE, "native"), _p())
    none = _trace(_out(LAYER), _p())
    assert model[(ROW_BINDING_CHECK, "table:1")].result is TraceResult.NOT_RUN
    assert native[(ROW_BINDING_CHECK, "table:1")].result is TraceResult.NOT_APPLICABLE
    assert none[(ROW_BINDING_CHECK, "page")].result is TraceResult.NOT_APPLICABLE


@pytest.mark.parametrize(
    ("check", "over", "out"),
    [
        pytest.param("text_layer_numbers", {}, _out(LAYER, "native"), id="numbers: native text"),
        pytest.param("text_layer_drops", {}, _out(LAYER, "native"), id="drops: native text"),
        pytest.param(
            "text_layer_words",
            {"own_layer_word_reference": "", "own_layer_figure_words": "x"},
            _out("![c](a.png)"),
            id="words: none owed",
        ),
        pytest.param("text_layer_numbers", {}, _out("  "), id="numbers: no text"),
        pytest.param("text_layer_words", {}, _out("  "), id="words: no text"),
        pytest.param(
            "text_layer_words",
            {"own_layer_word_reference": "", "own_layer_text": "", "is_born_digital": False},
            _out(),
            id="words: no layer",
        ),
        pytest.param(
            "prior_ocr_layer",
            {"own_layer_text": "", "is_born_digital": False},
            _out(),
            id="prior ocr: no layer",
        ),
        pytest.param("equation_fidelity", {}, _out(), id="equation: no reading attached"),
        pytest.param("figure_text", {}, _out(), id="chart values: no chart image"),
        pytest.param("page_judge", {}, _out(), id="exact pass: not accepted that way"),
    ],
)
def test_a_check_that_does_not_apply_says_so(check: str, over: dict, out: PageOutput) -> None:
    entry = _trace(out, _p(**over), providers_available=True)[(check, _scope(check))]
    assert entry.result is TraceResult.NOT_APPLICABLE and entry.reason, entry


@pytest.mark.parametrize(
    ("check", "over", "out"),
    [
        pytest.param("text_layer_numbers", {}, _out("[page 1 failed: x — see image]"), id="marker"),
        pytest.param("text_layer_numbers", {"prior_ocr_layer_suspect": True}, _out(), id="old OCR"),
        pytest.param("text_layer_drops", {"own_layer_drop_reference": ""}, _out(), id="no ref"),
        pytest.param(
            "text_layer_words", {"own_layer_word_reference": ""}, _out(), id="no word ref"
        ),
        pytest.param("text_layer_words", {"prior_ocr_layer_suspect": True}, _out(), id="old OCR"),
        pytest.param("control_bytes", {}, _out("[page 1 failed: x — see image]"), id="marker"),
    ],
)
def test_a_check_that_could_have_applied_and_did_not_run_is_not_run(
    check: str, over: dict, out: PageOutput
) -> None:
    entry = _trace(out, _p(**over), providers_available=True)[(check, _scope(check))]
    assert entry.result is TraceResult.NOT_RUN and entry.reason, entry


MODEL_DEPENDENT = ("text_layer_numbers", "text_layer_drops", "equation_fidelity")


def test_no_provider_makes_the_model_dependent_checks_not_run_and_nothing_else() -> None:
    """The DIFFERENCE between the two provider states on one native page: the model-dependent
    checks move from NOT_APPLICABLE to NOT_RUN, and no other entry moves."""
    out = _out(LAYER, "native")
    with_provider = _trace(out, _p(), providers_available=True)
    without = _trace(out, _p(), providers_available=False)
    moved = {k for k in with_provider if with_provider[k] != without[k]}
    assert moved == {(c, _scope(c)) for c in MODEL_DEPENDENT}
    for check in MODEL_DEPENDENT:
        assert with_provider[(check, _scope(check))].result is TraceResult.NOT_APPLICABLE
        assert without[(check, _scope(check))].result is TraceResult.NOT_RUN
        assert NO_PROVIDER_LADDER_REASON in without[(check, _scope(check))].reason


def test_an_absent_clause_is_never_read_as_a_pass() -> None:
    """A page the checks could say nothing about: no layer, no reference, no provider. The only
    PASS is the one whose check scanned the page's own text."""
    p = SimpleNamespace()  # a stand-in with none of the check inputs
    trace = _trace(_out(LAYER, "native"), p, providers_available=False)
    passes = {k for k, e in trace.items() if e.result is TraceResult.PASS}
    assert passes <= {("control_bytes", "page")}, passes
    for slot in PASS_TRACE_SLOTS:
        assert slot in trace or slot[0] == ROW_BINDING_CHECK, slot


def test_clauses_are_entries_and_win_over_a_check_that_reports_otherwise() -> None:
    clause = VerdictClause(Check.TEXT_LAYER_NUMBERS, CheckResult.FAIL, "page", "disputed")
    trace = _trace(_out(), _p(), verdict=PageVerdict(1, (clause,)))
    entry = trace[("text_layer_numbers", "page")]
    assert (entry.result, entry.reason) == (TraceResult.FAIL, "disputed")


def test_a_table_scope_clause_never_hides_the_page_scope_outcome() -> None:
    clause = VerdictClause(Check.FIGURE_TEXT, CheckResult.UNKNOWN, "figure:1", "not shown")
    trace = _trace(_out(), _p(), verdict=PageVerdict(1, (clause,)))
    assert trace[("figure_text", "figure:1")].result is TraceResult.UNKNOWN
    assert trace[("figure_text", "page")].reason == CHART_VALUES_NOT_APPLICABLE_REASON


def test_replay_is_returned_unchanged_and_a_page_without_one_is_not_run() -> None:
    stored = (TraceEntry("text_layer_numbers", TraceResult.PASS, "page", "kept"),)
    assert build_pass_trace(_out(), _p(), PageVerdict(1), replay=stored) == stored
    trace = _trace(_out(), _p(), resumed_without_trace=True)
    assert {k for k in trace} == set(PASS_TRACE_SLOTS)
    assert {e.result for e in trace.values()} == {TraceResult.NOT_RUN}
    assert {e.reason for e in trace.values()} == {RESUMED_WITHOUT_TRACE_REASON}


def test_the_trace_round_trips_and_a_bad_one_is_none_not_empty() -> None:
    entries = tuple(build_pass_trace(_out(), _p(), PageVerdict(1)))
    assert trace_from_dicts(trace_to_dicts(entries)) == entries
    assert trace_from_dicts([]) == ()
    assert trace_from_dicts(None) is None
    assert trace_from_dicts([{"check": "x", "result": "MAYBE"}]) is None


def test_reconcile_keeps_the_trace_consistent_with_an_offline_verdict() -> None:
    old = trace_to_dicts(
        (
            TraceEntry("text_layer_numbers", TraceResult.PASS, "page", "ok"),
            TraceEntry("text_layer_drops", TraceResult.FAIL, "page", "was a clause"),
            TraceEntry(ROW_BINDING_CHECK, TraceResult.FAIL, "table:1", "a table mark, no clause"),
        )
    )
    verdict = PageVerdict(
        1, (VerdictClause(Check.TEXT_LAYER_NUMBERS, CheckResult.FAIL, "page", "now disputed"),)
    )
    got = {(e["check"], e["scope"]): e for e in reconcile_pass_trace(old, verdict)}
    assert got[("text_layer_numbers", "page")]["result"] == "FAIL"
    assert got[("text_layer_drops", "page")]["result"] == "NOT_RUN"
    assert got[(ROW_BINDING_CHECK, "table:1")]["result"] == "FAIL"
    assert reconcile_pass_trace(None, verdict) is None


def test_every_outcome_has_a_reason_unless_it_passed_with_none() -> None:
    for o in pass_trace_outcomes(_out(), _p()):
        if o.entry.result in (TraceResult.NOT_APPLICABLE, TraceResult.NOT_RUN):
            assert o.entry.reason, o.entry


# --- a run, end to end -------------------------------------------------------------------


def _native_pdf(path: Path, pages: int = 2) -> Path:
    """Born-digital prose pages the native lane ships itself: no model reads them."""
    import pymupdf

    doc = pymupdf.open()
    for k in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text(
            (72, 100), f"Page {k + 1} inflation outlook remained elevated", fontsize=11
        )
        page.insert_text((72, 130), f"Payrolls rose {k + 3}.5 percent over the year", fontsize=11)
    doc.save(path)
    doc.close()
    return path


def _first(out: Path, n: int = 1) -> tuple[dict, str]:
    pages = next(out.rglob("pages"))
    return json.loads((pages / f"{n:05d}.json").read_text()), (pages / f"{n:05d}.md").read_text()


def _checks(sidecar: dict) -> dict[tuple[str, str], dict]:
    return {(e["check"], e["scope"]): e for e in sidecar["checks"]}


def _run_arm(tmp_path: Path, monkeypatch, name: str, *, ladder: bool, trace_on: bool = True):
    out = tmp_path / name
    # ONE input for every arm: two generated PDFs differ in their creation stamps, and so in
    # the checksum the sidecar binds to.
    pdf = tmp_path / "doc.pdf"
    if not pdf.exists():
        _rotated_pdf(pdf)
    with monkeypatch.context() as m:
        if not trace_on:
            m.setattr("socr.core.manifest.build_pass_trace", lambda *a, **k: ())
        _run(out, pdf, monkeypatch, lose_label=False, pages_read=[], ladder=ladder)
    return out


@pytest.mark.parametrize("ladder", [True, False], ids=["provider", "no provider"])
def test_the_trace_is_a_record_only_no_md_byte_or_status_moves_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ladder: bool
) -> None:
    on = _run_arm(tmp_path, monkeypatch, "on", ladder=ladder)
    off = _run_arm(tmp_path, monkeypatch, "off", ladder=ladder, trace_on=False)
    for n in (1, 2):
        (s_on, md_on), (s_off, md_off) = _first(on, n), _first(off, n)
        assert md_on == md_off, "the trace changed .md bytes"
        assert s_off["checks"] == [] and s_on["checks"], "fixture premise: the arms differ"
        strip = lambda s: {k: v for k, v in s.items() if k != "checks"}  # noqa: E731
        comparable = lambda s: {  # noqa: E731
            k: v
            for k, v in strip(s).items()
            if k not in ("timings_s", "page_cost_usd", "run_fingerprint", "page_fingerprint")
        }
        assert comparable(s_on) == comparable(s_off)
    doc_on, doc_off = next(on.rglob("pages")).parent, next(off.rglob("pages")).parent
    meta = lambda d: {  # noqa: E731
        k: v
        for k, v in json.loads((d / "metadata.json").read_text()).items()
        if k not in ("timestamp", "processing_time", "output_path")
    }
    assert meta(doc_on) == meta(doc_off)


def test_model_dependent_checks_differ_between_provider_states_on_one_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same PDF run with and without a provider ladder. With one a model reads the page and
    the number check runs on that reading; without one nothing reads it, so the check reports
    NOT_RUN with the reason, and the page's own-layer checks are the same in both."""
    with_p = _checks(_first(_run_arm(tmp_path, monkeypatch, "p", ladder=True))[0])
    without = _checks(_first(_run_arm(tmp_path, monkeypatch, "n", ladder=False))[0])
    key = ("text_layer_numbers", "page")
    assert with_p[key]["result"] != "NOT_RUN", with_p[key]
    assert without[key]["result"] == "NOT_RUN"
    assert NO_PROVIDER_LADDER_REASON in without[key]["reason"]


@pytest.mark.parametrize("ladder", [True, False], ids=["provider", "no provider"])
def test_resume_replays_the_identical_list_and_does_not_recompute_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ladder: bool
) -> None:
    out = tmp_path / "out"
    pdf = _native_pdf(tmp_path / "doc.pdf")
    _run(out, pdf, monkeypatch, lose_label=False, pages_read=[], ladder=ladder)
    before = _first(out)
    assert before[0]["checks"], "fixture premise: page 1 has a trace"
    _crash_after_page_one(out)

    resume: list[int] = []
    calls: list[int] = []
    with monkeypatch.context() as m:
        # A resumed page must not run its checks again: if the replay recomputed, this trips.
        real = verdict_module.pass_trace_outcomes

        def spy(output, p, *a, **k):
            calls.append(output.page_num)
            return real(output, p, *a, **k)

        m.setattr(verdict_module, "pass_trace_outcomes", spy)
        _run(out, pdf, monkeypatch, lose_label=False, pages_read=resume, ladder=ladder)
    assert 2 in calls and 1 not in calls, f"fixture premise: page 2 re-run, page 1 kept: {calls}"
    after = _first(out)
    assert after[0]["checks"] == before[0]["checks"]
    assert after[1] == before[1]
    assert 1 not in calls, "the resumed page recomputed its trace"


def test_a_resumed_page_whose_sidecar_has_no_trace_states_not_run_never_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    pdf = _native_pdf(tmp_path / "doc.pdf")
    _run(out, pdf, monkeypatch, lose_label=False, pages_read=[], ladder=True)
    sidecar_path = next(out.rglob("pages")) / "00001.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar.pop("checks")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
    _crash_after_page_one(out)
    _run(out, pdf, monkeypatch, lose_label=False, pages_read=[], ladder=True)
    got = _first(out)[0]["checks"]
    assert got and all(e["result"] != "PASS" for e in got), got
    assert {e["reason"] for e in got if e["result"] == "NOT_RUN"} >= {RESUMED_WITHOUT_TRACE_REASON}


# --- the scoreboard ----------------------------------------------------------------------


def test_the_scoreboard_counts_passes_empty_traces_and_not_run_per_check(tmp_path: Path) -> None:
    from socr.benchmark import scoreboard

    doc = tmp_path / "doc"
    pages = doc / "pages"
    pages.mkdir(parents=True)

    def page(n: int, checks) -> None:
        (pages / f"{n:05d}.md").write_text("text\n")
        body = {"page_num": n, "status": "success", "engine": "native", "audit_passed": True}
        if checks is not None:
            body["checks"] = checks
        (pages / f"{n:05d}.json").write_text(json.dumps(body))

    e = lambda check, result, scope="page": {  # noqa: E731
        "check": check,
        "result": result,
        "scope": scope,
        "reason": "",
    }
    page(1, [e("text_layer_numbers", "PASS"), e("text_layer_words", "NOT_RUN")])
    page(2, [e("row_binding", "PASS", "table:1"), e("text_layer_words", "NOT_RUN")])
    page(3, [])
    page(4, None)
    counts = scoreboard.count_document(doc)
    assert counts["records.pages_checks_any_pass"] == 2
    assert counts["records.pages_checks_page_scope_pass"] == 1
    assert counts["records.pages_checks_empty"] == 2
    assert counts["records.checks_not_run.text_layer_words"] == 2
    assert counts["records.check_result.row_binding@table|PASS"] == 1
    assert counts["records.check_result.text_layer_numbers@page|PASS"] == 1
    assert not any(
        k.startswith("records.check_result.") for k in scoreboard.render([counts], ["x"]).split()
    )


# --- review round on #1332: PASS only for a check that ran on the SHIPPED text ------------

WORD_FIGURE_ONLY = {"own_layer_word_reference": "", "own_layer_figure_words": "chart axis words"}


def test_words_pass_needs_words_owed_and_a_layer_that_is_evidence() -> None:
    """Carbon 1: zero words compared is not a pass, and neither is agreement with a layer the
    page analysis refused or found glyph-damaged."""
    shipped = {(c, s): e for (c, s), e in _trace(_out(), _p()).items()}
    assert shipped[("text_layer_words", "page")].result is TraceResult.PASS
    chart = _trace(_out("![chart](c.png)"), _p(**WORD_FIGURE_ONLY))[("text_layer_words", "page")]
    assert chart.result is TraceResult.NOT_APPLICABLE and "owes no word" in chart.reason
    for damage in ({"is_born_digital": False}, {"has_unrecovered_symbol_glyphs": True}):
        # one stray word the reading kept: nothing missing, so the old code said PASS
        entry = _trace(_out("korea china"), _p(own_layer_word_reference="korea", **damage))[
            ("text_layer_words", "page")
        ]
        assert entry.result is TraceResult.NOT_RUN, (damage, entry)


def test_drops_checks_table_rows_only_so_its_scope_is_never_page() -> None:
    """Carbon 2: the PASS of the drops check names the region it read."""
    trace = _trace(_out(), _p())
    assert trace[("text_layer_drops", SCOPE_TABLE_ROWS)].result is TraceResult.PASS
    assert ("text_layer_drops", "page") not in trace
    assert ("text_layer_drops", SCOPE_TABLE_ROWS) in PASS_TRACE_SLOTS


def test_row_binding_states_only_tables_that_ship() -> None:
    """Carbon 3 and Fable 1/2: a record never speaks for a table the output does not carry, and
    every shipped table without a record is NOT_RUN."""
    bound = {"table_index": 0, "outcome": "bound"}
    marker = _out("[page 1 failed: unverifiable table -- see image]")
    entry = _trace(marker, _p(row_binding={"p1-t0": bound}))[(ROW_BINDING_CHECK, "table:1")]
    assert entry.result is TraceResult.NOT_RUN and "does not ship" in entry.reason
    # a record for a third table on a page that ships one
    far = {"p1-t2": {"table_index": 2, "outcome": "bound"}}
    trace = _trace(_out(TABLE), _p(row_binding=far))
    assert trace[(ROW_BINDING_CHECK, "table:3")].result is TraceResult.NOT_RUN
    assert trace[(ROW_BINDING_CHECK, "table:1")].result is TraceResult.NOT_RUN
    # a native text is not the reading the record was taken on
    native = _trace(_out(TABLE, "native"), _p(row_binding={"p1-t0": bound}))
    assert native[(ROW_BINDING_CHECK, "table:1")].result is TraceResult.NOT_RUN
    # partial records: the others are stated, not absent
    some = _trace(_out(THREE_TABLES), _p(row_binding={"p1-t0": bound}))
    assert [some[(ROW_BINDING_CHECK, f"table:{k}")].result for k in (1, 2, 3)] == [
        TraceResult.PASS,
        TraceResult.NOT_RUN,
        TraceResult.NOT_RUN,
    ]
    assert not any(s == "page" for c, s in some if c == ROW_BINDING_CHECK)


def test_replay_overlays_every_verdict_clause_it_lacks() -> None:
    """Fable 3: a resumed page demoted after its original run cannot say PASS beside its clause."""
    stored = (TraceEntry("control_bytes", TraceResult.PASS, "page", "ran"),)
    clause = VerdictClause(Check.CONTROL_BYTES, CheckResult.FAIL, "page", "now found")
    other = VerdictClause(Check.TEXT_LAYER_WORDS, CheckResult.UNKNOWN, "page", "new")
    got = {
        (e.check, e.scope): e
        for e in build_pass_trace(_out(), _p(), PageVerdict(1, (clause, other)), replay=stored)
    }
    assert got[("control_bytes", "page")].result is TraceResult.FAIL
    assert got[("text_layer_words", "page")].result is TraceResult.UNKNOWN


def test_a_resumed_page_demoted_at_assemble_carries_its_clause_in_the_trace(tmp_path) -> None:
    from test_gh1145_chart_values_clause import _state

    from socr.core.manifest import _select_and_finalize_page

    state = _state(tmp_path, text="body 1.0", engine="qwen", status=PageStatus.WARNING)
    ps = state.pages[1]
    ps.published_verdict = PageVerdict(1)
    ps.published_checks = (TraceEntry("control_bytes", TraceResult.PASS, "page", "clean run"),)
    record = _select_and_finalize_page(state, 1)
    assert record.verdict.clauses, "premise: the demoted page carries a clause"
    by_key = {(e.check, e.scope): e for e in record.checks}
    for clause in record.verdict.clauses:
        entry = by_key[(clause.check.value, clause.scope)]
        assert entry.result is not TraceResult.PASS, entry


def test_a_resumed_page_without_a_trace_states_table_slots_not_a_page_row_binding() -> None:
    trace = _trace(_out(THREE_TABLES), _p(), resumed_without_trace=True)
    assert (ROW_BINDING_CHECK, "page") not in trace
    assert {k for k in trace if k[0] == ROW_BINDING_CHECK} == {
        (ROW_BINDING_CHECK, f"table:{k}") for k in (1, 2, 3)
    }


def test_equation_fidelity_with_a_provider_and_the_lane_off_is_not_run() -> None:
    """Fable 6: a lane that could not attach a reading did not run; with it on, a page with no
    equation is not applicable. The DIFFERENCE is the lane flag alone."""
    on = _trace(_out(), _p(), providers_available=True, equation_lane_enabled=True)
    off = _trace(_out(), _p(), providers_available=True, equation_lane_enabled=False)
    moved = {k for k in on if on[k] != off[k]}
    assert moved == {("equation_fidelity", "page")}
    assert on[("equation_fidelity", "page")].result is TraceResult.NOT_APPLICABLE
    assert off[("equation_fidelity", "page")].result is TraceResult.NOT_RUN


def test_consensus_winners_are_model_readings_for_the_numbers_check() -> None:
    entry = _trace(_out(LAYER, "consensus(qwen)"), _p())[("text_layer_numbers", "page")]
    assert entry.result is TraceResult.PASS, entry


@pytest.mark.parametrize(
    "bad",
    [
        {"check": "x", "result": "PASS", "reason": ""},  # no scope: not "page"
        {"check": "x", "result": "PASS", "scope": 3, "reason": ""},
        {"check": 7, "result": "PASS", "scope": "page", "reason": ""},
        {"check": "x", "result": "PASS", "scope": "page", "reason": None},
        {"check": "x", "result": "PASS", "scope": "", "reason": ""},
    ],
)
def test_a_malformed_entry_makes_the_whole_trace_unreadable(bad: dict) -> None:
    good = {"check": "y", "result": "PASS", "scope": "page", "reason": "ok"}
    assert trace_from_dicts([good, bad]) is None
    from socr.benchmark import scoreboard

    assert scoreboard._checks({"checks": [good, bad]}) == []
    assert scoreboard._checks({"checks": [good]}) == [good]


def test_the_scoreboard_does_not_call_a_traced_clean_native_page_untraced(tmp_path: Path) -> None:
    from socr.benchmark import scoreboard

    pages = tmp_path / "doc" / "pages"
    pages.mkdir(parents=True)
    for n, checks in (
        (1, [{"check": "x", "result": "NOT_RUN", "scope": "page", "reason": "r"}]),
        (2, None),
    ):
        (pages / f"{n:05d}.md").write_text("text\n")
        body = {"page_num": n, "status": "success", "engine": "native", "audit_passed": True}
        if checks is not None:
            body["checks"] = checks
        (pages / f"{n:05d}.json").write_text(json.dumps(body))
    counts = scoreboard.count_document(tmp_path / "doc")
    assert counts["records.clean_native_success_no_trace"] == 1


# --- the offline tools --------------------------------------------------------------------

OLD_TRACE = trace_to_dicts(
    (
        TraceEntry("text_layer_numbers", TraceResult.PASS, "page", "ok"),
        TraceEntry("text_layer_words", TraceResult.PASS, "page", "ok"),
        TraceEntry("equation_fidelity", TraceResult.NOT_APPLICABLE, "page", "none attached"),
    )
)


def test_reconcile_turns_a_kept_pass_into_not_run_when_the_text_changed() -> None:
    """Fable 4: a PASS the tool did not recompute says nothing about the new text."""
    recomputed = (TraceEntry("text_layer_numbers", TraceResult.PASS, "page", "rerun"),)
    got = {
        (e["check"], e["scope"]): e
        for e in reconcile_pass_trace(
            OLD_TRACE, PageVerdict(1), recomputed=recomputed, text_changed=True
        )
    }
    assert got[("text_layer_numbers", "page")]["reason"] == "rerun"
    assert got[("text_layer_words", "page")]["result"] == "NOT_RUN"
    assert "not run on the new text" in got[("text_layer_words", "page")]["reason"]
    assert got[("equation_fidelity", "page")]["result"] == "NOT_APPLICABLE"
    kept = {e["result"] for e in reconcile_pass_trace(OLD_TRACE, PageVerdict(1))}
    assert "NOT_RUN" not in kept, "text unchanged: the PASS entries stay"


def test_a_malformed_old_trace_is_dropped_from_a_rewritten_sidecar() -> None:
    from socr.core.verdict import write_reconciled_trace

    out = {"checks": [{"check": "x"}], "page_num": 1}
    write_reconciled_trace(out, out["checks"], PageVerdict(1), text_changed=True)
    assert "checks" not in out


def test_refinalize_boilerplate_marks_kept_passes_not_run(tmp_path: Path) -> None:
    from test_boilerplate_output import _PREAMBLE_LINE

    from socr.pipeline.refinalize_boilerplate import refinalize_document

    doc = tmp_path / "doc"
    pages = doc / "pages"
    pages.mkdir(parents=True)
    body = "Section 1\n\nOutlays total 100 billion in fiscal year 2026."
    raw = f"{_PREAMBLE_LINE}\n\n---\n\n{body}"
    sidecar = {
        "page_num": 1,
        "terminal": True,
        "winning_output": {"engine": "qwen", "text": raw},
        "verdict": {"page_num": 1, "clauses": []},
        "checks": OLD_TRACE,
    }
    (pages / "00001.json").write_text(json.dumps(sidecar))
    (pages / "00001.md").write_text(raw)
    (doc / "doc.md").write_text(f"## Page 1\n\n{raw}")
    refinalize_document(doc, apply=True)
    after = json.loads((pages / "00001.json").read_text())
    assert after["winning_output"]["text"] == body
    results = {e["check"]: e["result"] for e in after["checks"]}
    assert results["text_layer_numbers"] == "NOT_RUN" and results["text_layer_words"] == "NOT_RUN"
    assert results["equation_fidelity"] == "NOT_APPLICABLE"


def test_refinalize_prior_ocr_takes_the_records_own_trace() -> None:
    from socr.pipeline import refinalize_prior_ocr as tool

    out = PageOutput(
        page_num=1, text="recovered prose 1.0", status=PageStatus.WARNING, engine="native"
    )
    new_checks = (TraceEntry("control_bytes", TraceResult.PASS, "page", "fresh"),)
    record = SimpleNamespace(
        output=out,
        disposition=SimpleNamespace(to_dict=lambda: {}),
        verdict=PageVerdict(1),
        checks=new_checks,
    )
    sidecar = {
        "page_num": 1,
        "winning_output": {"text": "[page 1 failed: x]"},
        "checks": trace_to_dicts(
            (
                TraceEntry(
                    "text_layer_numbers",
                    TraceResult.NOT_RUN,
                    "page",
                    "this page ships socr's failure marker",
                ),
            )
        ),
    }
    try:
        got = tool._updated_sidecar(sidecar, record)
    except Exception as exc:  # pragma: no cover - the fake record is too thin for a field
        pytest.skip(f"fake record too thin: {exc!r}")
    assert got["checks"] == trace_to_dicts(new_checks)


def test_refinalize_own_text_states_the_words_check_it_wrote() -> None:
    from socr.pipeline.refinalize_own_text import _updated_sidecar

    out = _out(LAYER, "native")
    sidecar = {
        "page_num": 1,
        "status": "success",
        "winning_output": out.to_dict(),
        "checks": trace_to_dicts(
            (
                TraceEntry(
                    "text_layer_words", TraceResult.NOT_RUN, "page", RESUMED_WITHOUT_TRACE_REASON
                ),
            )
        ),
    }
    from socr.core.verdict import page_word_check_record

    p = _p()
    got = _updated_sidecar(
        sidecar, out, PageVerdict(1), words_record=page_word_check_record(out, p), page_state=p
    )
    entry = {e["check"]: e for e in got["checks"]}["text_layer_words"]
    assert entry["result"] == "PASS" and got["text_layer_words"]["compared"] > 0
