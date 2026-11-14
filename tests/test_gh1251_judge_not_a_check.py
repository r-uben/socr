"""#1251 (C3): the page judge's acceptance is not a check.

A model-read page the page judge accepted stays SUCCESS only when the pass trace holds an
independent confirmation on the shipped text: the words check passed at page scope, the numbers
check passed (or does not apply), and no entry of any check is a FAIL or UNKNOWN. Otherwise the
page ships under ``page_judge=UNKNOWN`` (a WARNING) in a mode a resume restores instead of
re-reading. Every judge call leaves ``pages/NNNNN.judge-calls``.

Per ``CLAUDE.md`` the run-level tests pin a DIFFERENCE between two arms of one process (only the
independent evidence differs), with the provider ladder in both states, and never an absolute
outcome measured here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_gh1185_text_layer_numbers_e2e import _crash_after_page_one
from test_gh1294_layer_words import _Reader, _rotated_pdf, _run

import socr
from socr.core import verdict as verdict_module
from socr.core.marks import Check, CheckResult
from socr.core.page_credential import sha256_text
from socr.core.pass_trace import TraceEntry, TraceResult
from socr.core.result import (
    JUDGE_OUTCOME_COMPLETED,
    FailureMode,
    PageOutput,
    PageStatus,
    contract_status_for,
)
from socr.core.verdict import (
    JUDGE_ONLY_REASON,
    PERMANENTLY_UNRESOLVABLE_MODES,
    ROW_BINDING_CHECK,
    clause_is_reprocessable,
    independent_page_evidence,
    page_text_digest,
    silent_success_clauses,
)
from socr.judge.judge import (
    JUDGE_CALL_DEADLINE,
    JUDGE_CALL_ERROR,
    JUDGE_CALL_TIMEOUT,
    JUDGE_CALL_VERDICT,
    JudgeVerdict,
    judge_call_record,
    load_judge_prompt,
)


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# --- the predicate: independent evidence on the shipped text ------------------------------

TEXT = "Korea 4.0 2.6 5.7 4.0\nChina 6.9 7.0 6.6 7.1"


def _e(check: str, result: TraceResult, scope: str = "page", text: str = TEXT) -> TraceEntry:
    sha = page_text_digest(text) if result is TraceResult.PASS else ""
    return TraceEntry(check, result, scope, "", text_sha=sha)


WORDS_PASS = _e("text_layer_words", TraceResult.PASS)
NUMBERS_PASS = _e("text_layer_numbers", TraceResult.PASS)


def test_words_and_numbers_passes_at_page_scope_are_the_evidence() -> None:
    assert independent_page_evidence([WORDS_PASS, NUMBERS_PASS], TEXT)


def test_no_numbers_owed_counts_as_no_numbers_to_check() -> None:
    na = _e("text_layer_numbers", TraceResult.NOT_APPLICABLE)
    assert independent_page_evidence([WORDS_PASS, na], TEXT)


@pytest.mark.parametrize(
    "numbers", [TraceResult.NOT_RUN, TraceResult.UNKNOWN, TraceResult.FAIL], ids=lambda r: r.value
)
def test_numbers_that_did_not_pass_are_no_evidence(numbers: TraceResult) -> None:
    assert not independent_page_evidence([WORDS_PASS, _e("text_layer_numbers", numbers)], TEXT)


@pytest.mark.parametrize(
    "words",
    [TraceResult.NOT_RUN, TraceResult.NOT_APPLICABLE, TraceResult.UNKNOWN, TraceResult.FAIL],
    ids=lambda r: r.value,
)
def test_words_that_did_not_pass_are_no_evidence(words: TraceResult) -> None:
    assert not independent_page_evidence([_e("text_layer_words", words), NUMBERS_PASS], TEXT)


def test_a_missing_words_entry_is_no_evidence() -> None:
    assert not independent_page_evidence([NUMBERS_PASS], TEXT)


@pytest.mark.parametrize("result", [TraceResult.FAIL, TraceResult.UNKNOWN], ids=lambda r: r.value)
@pytest.mark.parametrize(
    ("check", "scope"),
    [("control_bytes", "page"), (ROW_BINDING_CHECK, "table:1"), ("figure_text", "page")],
)
def test_any_failure_or_unknown_entry_of_any_check_withdraws_the_evidence(
    check: str, scope: str, result: TraceResult
) -> None:
    other = TraceEntry(check, result, scope, "found")
    assert not independent_page_evidence([WORDS_PASS, NUMBERS_PASS, other], TEXT)


def test_a_table_scope_pass_is_not_a_page_pass() -> None:
    table = TraceEntry(
        "text_layer_words", TraceResult.PASS, "table:1", "", text_sha=page_text_digest(TEXT)
    )
    assert not independent_page_evidence([table, NUMBERS_PASS], TEXT)


def test_a_pass_computed_on_other_text_is_no_evidence_for_this_text() -> None:
    stale = _e("text_layer_words", TraceResult.PASS, text="an older reading")
    assert not independent_page_evidence([stale, NUMBERS_PASS], TEXT)
    assert not independent_page_evidence(
        [TraceEntry("text_layer_words", TraceResult.PASS, "page", ""), NUMBERS_PASS], TEXT
    )


# --- the clause on a model reading the judge accepted -------------------------------------

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
        hidden_footnote_words="",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _out(
    text: str = LAYER,
    *,
    engine: str = "qwen",
    accepted: bool | None = True,
    outcome: str = JUDGE_OUTCOME_COMPLETED,
    reason: str = "faithful",
) -> PageOutput:
    return PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine=engine,
        judge_outcome=outcome,
        judge_accepted=accepted,
        judge_reason=reason,
        audit_passed=True,
    )


def _judge_only(out: PageOutput, p) -> list:
    return [c for c in silent_success_clauses(out, p) if c.reason == JUDGE_ONLY_REASON]


def test_a_judge_approved_reading_the_layer_confirms_stays_clean() -> None:
    assert silent_success_clauses(_out(), _p()) == ()


def test_a_page_whose_layer_owes_no_words_has_only_the_approval() -> None:
    """Numbers confirm, no word is owed: the words check cannot pass, so nothing independent
    confirmed the reading and the approval stands alone."""
    [clause] = silent_success_clauses(_out(), _p(own_layer_word_reference=""))
    assert (clause.check, clause.result, clause.scope) == (
        Check.PAGE_JUDGE,
        CheckResult.UNKNOWN,
        "page",
    )
    assert clause.reason == JUDGE_ONLY_REASON


def test_a_page_with_no_layer_is_already_demoted_by_the_numbers_clause_and_gains_no_line() -> None:
    got = silent_success_clauses(_out(), _p(own_layer_text=""))
    assert [c.check for c in got] == [Check.TEXT_LAYER_NUMBERS]


def test_the_predicate_does_not_read_the_judges_reason() -> None:
    p = _p(own_layer_word_reference="")
    reasons = [
        "faithful",
        "",
        "native_table_verifier: x",
        "tolerated page-boundary text",
        "x" * 300,
    ]
    got = {r: [c.reason for c in silent_success_clauses(_out(reason=r), p)] for r in reasons}
    assert all(v == [JUDGE_ONLY_REASON] for v in got.values()), got


@pytest.mark.parametrize(
    "out",
    [
        pytest.param(_out(accepted=False), id="the judge refused"),
        pytest.param(_out(accepted=None), id="no acceptance recorded"),
        pytest.param(_out(outcome=""), id="no judge outcome"),
        pytest.param(_out(engine="native"), id="not a model reading"),
    ],
)
def test_only_a_judge_accepted_model_reading_is_in_scope(out: PageOutput) -> None:
    assert _judge_only(out, _p(own_layer_word_reference="")) == []


def test_a_cloud_reading_is_in_scope_and_the_wording_does_not_name_who_judged() -> None:
    clause = _judge_only(_out(engine="gemini"), _p(own_layer_word_reference=""))[0]
    assert "produced it" not in clause.reason and "model that" not in clause.reason
    assert "no independent check" in clause.reason


def test_a_page_another_check_already_demotes_gains_no_second_line() -> None:
    short = "Korea 4.0 2.6 5.7 4.0"
    got = silent_success_clauses(_out(short), _p())
    assert got and all(c.reason != JUDGE_ONLY_REASON for c in got)


def test_the_seam_demotes_records_the_mode_and_is_idempotent() -> None:
    from socr.core.manifest import apply_verdict_marks

    p = _p(own_layer_word_reference="")
    kept, kept_v = apply_verdict_marks(_out(), _p(), 1)
    demoted, v = apply_verdict_marks(_out(), p, 1)
    assert (kept.status, kept.failure_mode, kept_v.clauses) == (
        PageStatus.SUCCESS,
        FailureMode.NONE,
        (),
    )
    assert (demoted.status, demoted.failure_mode) == (
        PageStatus.WARNING,
        FailureMode.READING_UNCORROBORATED,
    )
    assert [c.reason for c in v.clauses] == [JUDGE_ONLY_REASON]
    assert "page_judge=UNKNOWN" in demoted.text
    assert demoted.text.endswith(LAYER), "body text is not touched: only a mark line is added"
    again, v2 = apply_verdict_marks(demoted, p, 1)
    assert (again.text, again.status, again.failure_mode, v2) == (
        demoted.text,
        demoted.status,
        demoted.failure_mode,
        v,
    )


def test_a_page_the_old_clause_names_is_still_reprocessable_and_this_one_is_not() -> None:
    from socr.core.verdict import EXACT_PASS_UNJUDGED_REASON, VerdictClause

    old = VerdictClause(Check.PAGE_JUDGE, CheckResult.UNKNOWN, "page", EXACT_PASS_UNJUDGED_REASON)
    new = VerdictClause(Check.PAGE_JUDGE, CheckResult.UNKNOWN, "page", JUDGE_ONLY_REASON)
    timeout = VerdictClause(Check.PAGE_JUDGE, CheckResult.UNKNOWN, "page", "judge did not finish")
    assert clause_is_reprocessable(old) and clause_is_reprocessable(timeout)
    assert not clause_is_reprocessable(new)
    assert FailureMode.READING_UNCORROBORATED in PERMANENTLY_UNRESOLVABLE_MODES
    assert FailureMode.MODEL_OUTPUT_UNJUDGED not in PERMANENTLY_UNRESOLVABLE_MODES


def test_the_resume_gate_admits_this_clause_and_still_refuses_a_resolvable_judge_clause() -> None:
    from socr.core.verdict import EXACT_PASS_UNJUDGED_REASON, PageVerdict, VerdictClause
    from socr.pipeline.orchestrator import _resume_verdict_is_permanently_unverifiable as gate

    def verdict(reason: str) -> dict:
        clause = VerdictClause(Check.PAGE_JUDGE, CheckResult.UNKNOWN, "page", reason)
        return PageVerdict(1, (clause,)).to_dict()

    assert gate(verdict(JUDGE_ONLY_REASON), FailureMode.READING_UNCORROBORATED.value)
    assert not gate(verdict(EXACT_PASS_UNJUDGED_REASON), FailureMode.MODEL_OUTPUT_UNJUDGED.value)
    assert not gate(verdict("judge did not finish"), FailureMode.PAGE_JUDGE_TIMEOUT.value)


# --- a run, end to end: only the independent evidence differs -----------------------------


def _without_words_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """The arm with the independent check removed. The judge, the reading and every other check
    are the same; only the words check reports NOT_RUN."""
    monkeypatch.setattr(
        verdict_module,
        "text_layer_words_outcome",
        lambda output, p: verdict_module._not_run(Check.TEXT_LAYER_WORDS, "removed in this arm"),
    )


def _page(out: Path, n: int = 1) -> tuple[dict, str]:
    pages = next(out.rglob("pages"))
    return json.loads((pages / f"{n:05d}.json").read_text()), (pages / f"{n:05d}.md").read_text()


def _arm(tmp_path: Path, monkeypatch, name: str, *, evidence: bool, ladder: bool):
    out = tmp_path / name
    pdf = tmp_path / "doc.pdf"
    if not pdf.exists():
        _rotated_pdf(pdf)
    read: list[int] = []
    with monkeypatch.context() as m:
        if not evidence:
            _without_words_evidence(m)
        result = _run(out, pdf, monkeypatch, lose_label=False, pages_read=read, ladder=ladder)
    sidecar, md = _page(out)
    doc_dir = next(out.rglob("pages")).parent
    return {
        "page": (sidecar["status"], sidecar["failure_mode"], sidecar["audit_passed"]),
        "clauses": [
            (c["check"], c["result"], c["reason"])
            for c in (sidecar.get("verdict") or {}).get("clauses") or []
        ],
        "mark": "page_judge=UNKNOWN" in md,
        "document": result.status.value,
        "metadata": json.loads((doc_dir / "metadata.json").read_text())["status"],
        "cli": contract_status_for(result).value,
        "judge_accepted": sidecar["winning_output"].get("judge_accepted"),
        "read": read,
    }


@pytest.mark.parametrize("ladder", [True, False], ids=["provider", "no provider"])
def test_removing_the_independent_check_changes_the_label_at_every_level_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ladder: bool
) -> None:
    with_ev = _arm(tmp_path, monkeypatch, "with", evidence=True, ladder=ladder)
    without = _arm(tmp_path, monkeypatch, "without", evidence=False, ladder=ladder)
    if not ladder:
        assert not with_ev["read"], "fixture premise: with no provider no model reads the page"
        assert with_ev == without, "no model reading, no judge acceptance: nothing differs"
        return
    assert 1 in with_ev["read"] and 1 in without["read"]
    assert with_ev["judge_accepted"] is True and without["judge_accepted"] is True
    differing = {k for k in with_ev if with_ev[k] != without[k]} - {"read"}
    assert differing == {"page", "clauses", "mark", "document", "metadata", "cli"}
    assert without["page"] == ("warning", "reading_uncorroborated", False)
    assert [c[2] for c in without["clauses"]] == [JUDGE_ONLY_REASON]
    assert with_ev["clauses"] == [] and with_ev["page"][0] == "success"
    assert (with_ev["document"], without["document"]) == ("success", "audit_failed")
    assert (with_ev["metadata"], without["metadata"]) == ("completed", "partial")
    assert (with_ev["cli"], without["cli"]) == ("completed", "partial")


@pytest.mark.parametrize("ladder", [True, False], ids=["provider", "no provider"])
def test_the_demoted_page_is_restored_on_resume_not_re_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ladder: bool
) -> None:
    """The same layer and config give the same absence of evidence, and a re-read is not
    deterministic: re-reading would only change shipped text. Both provider states."""
    out = tmp_path / "out"
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    with monkeypatch.context() as m:
        _without_words_evidence(m)
        _run(out, pdf, monkeypatch, lose_label=False, pages_read=[], ladder=ladder)
        before = _page(out)
        _crash_after_page_one(out)
        resume: list[int] = []
        _run(out, pdf, monkeypatch, lose_label=False, pages_read=resume, ladder=ladder)
    if not ladder:
        assert not resume and before[0]["failure_mode"] != "reading_uncorroborated"
        return
    assert before[0]["failure_mode"] == "reading_uncorroborated", "fixture premise"
    assert 2 in resume, "the resume did not reach the page loop, so the ledger was not tested"
    assert 1 not in resume, f"the demoted page was re-read on resume: {resume}"
    assert _page(out) == before


# --- the judge's calls leave a side file --------------------------------------------------


class _FakeVLM:
    """The model behind ``VLMPageJudge``: scripted replies, or an exception for a call."""

    model = "fake-judge-model"

    def __init__(self, script: list) -> None:
        self._script = list(script)
        self.calls = 0

    def judge(self, image_path, ocr_text) -> JudgeVerdict:
        step = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        if isinstance(step, BaseException):
            raise step
        return step


def _verdict(reply: str = '{"faithful": true}', faithful: bool = True) -> JudgeVerdict:
    return JudgeVerdict(
        faithful=faithful,
        issues=[] if faithful else ["a dropped row"],
        confidence=0.9,
        suggested_action="accept" if faithful else "escalate_engine",
        raw=reply,
    )


def _judged_run(out: Path, pdf: Path, monkeypatch, fake: _FakeVLM, pages_read: list[int]):
    """``_run`` with the real ``VLMPageJudge`` over *fake* instead of the fixed judge."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.pipeline.agentic import VLMPageJudge
    from socr.pipeline.orchestrator import UnifiedPipeline

    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Reader(False, pages_read))
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=True,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
                describe_figure_crops=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: VLMPageJudge(
            fake,
            lambda page_num: out,
            record_call=lambda rec: pipe._record_page_judge_call(state, rec),
        )
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        return pipe.process(pdf, output_dir=out)


def _calls(out: Path, n: int = 1):
    pages = next(out.rglob("pages"))
    path = pages / f"{n:05d}.judge-calls"
    return path, json.loads(path.read_text()) if path.exists() else None


def test_every_judge_call_leaves_its_prompt_digest_model_and_full_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = '{"faithful": true, "suggested_action": "accept", "confidence": 0.9, "issues": []}'
    out = tmp_path / "out"
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    fake = _FakeVLM([_verdict(reply)])
    read: list[int] = []
    _judged_run(out, pdf, monkeypatch, fake, read)
    assert fake.calls >= 2, "fixture premise: both pages were judged"
    for n in (1, 2):
        path, calls = _calls(out, n)
        sidecar, md = _page(out, n)
        assert isinstance(calls, list) and calls and all(c["outcome"] == "verdict" for c in calls)
        [first, *_] = calls
        assert first["reply"] == reply and first["reply_chars"] == len(reply)
        assert first["parsed"] == {
            "faithful": True,
            "issues": [],
            "confidence": 0.9,
            "suggested_action": "accept",
        }
        assert first["judge_model"] == "fake-judge-model"
        assert first["judge_backend"] == "_FakeVLM"
        import hashlib

        assert first["prompt_sha256"] == hashlib.sha256(load_judge_prompt().encode()).hexdigest()
        # The page ships the reading unchanged (the judge accepted it and the layer confirms it).
        assert (
            first["reading_sha256"]
            == hashlib.sha256(sidecar["winning_output"]["text"].encode()).hexdigest()
        )
        ref = sidecar["judge_calls"]
        assert ref == {
            "path": path.name,
            "sha256": sha256_text(path.read_text(encoding="utf-8")),
            "count": len(calls),
        }


def test_a_timeout_and_a_crash_are_recorded_as_such_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    fake = _FakeVLM([TimeoutError("took too long"), ValueError("bad reply"), _verdict()])
    _judged_run(out, pdf, monkeypatch, fake, [])
    _, first = _calls(out, 1)
    _, second = _calls(out, 2)
    assert [c["outcome"] for c in first] == [JUDGE_CALL_TIMEOUT]
    assert [c["outcome"] for c in second] == [JUDGE_CALL_ERROR]
    assert first[0]["error"] == {"type": "TimeoutError", "message": "took too long"}
    assert second[0]["error"] == {"type": "ValueError", "message": "bad reply"}
    assert "reply" not in first[0] and "parsed" not in second[0], "no verdict, no reply to keep"
    assert all(
        c["prompt_sha256"] and c["judge_model"] == "fake-judge-model" for c in first + second
    )


def test_the_side_file_and_its_reference_survive_a_resume_unchanged_and_are_not_fingerprinted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    fake = _FakeVLM([_verdict('{"faithful": true}')])
    _judged_run(out, pdf, monkeypatch, fake, [])
    path, before_calls = _calls(out, 1)
    before_bytes = path.read_bytes()
    before_sidecar, _ = _page(out)
    _crash_after_page_one(out)
    quiet = _FakeVLM([_verdict('{"faithful": false}')])
    resume: list[int] = []
    _judged_run(out, pdf, monkeypatch, quiet, resume)
    assert 2 in resume and 1 not in resume, "fixture premise: page 1 was kept, page 2 re-run"
    after_sidecar, _ = _page(out)
    assert path.read_bytes() == before_bytes, "a kept page's calls are preserved byte for byte"
    assert after_sidecar["judge_calls"] == before_sidecar["judge_calls"]
    assert after_sidecar["run_fingerprint"] == before_sidecar["run_fingerprint"]
    # Page 2 was run again, so its file holds THIS run's calls and no stale ones.
    _, page_two = _calls(out, 2)
    assert {c["reply"] for c in page_two} == {'{"faithful": false}'}


def test_a_page_no_model_judged_has_no_side_file_and_no_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    _run(
        out,
        _rotated_pdf(tmp_path / "doc.pdf"),
        monkeypatch,
        lose_label=False,
        pages_read=[],
        ladder=True,
    )
    path, calls = _calls(out, 1)
    sidecar, _ = _page(out)
    assert calls is None and not path.exists() and "judge_calls" not in sidecar


def test_an_altered_side_file_is_not_restored_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from socr.pipeline.orchestrator import UnifiedPipeline

    (tmp_path / "00001.judge-calls").write_text('[{"outcome": "verdict"}]', encoding="utf-8")
    ref = {"path": "00001.judge-calls", "sha256": "sha256:other", "count": 1}
    with caplog.at_level("WARNING"):
        got = UnifiedPipeline._load_page_judge_calls(tmp_path, 1, ref)
    assert got == [] and "could not be restored" in caplog.text
    good = {"path": "00001.judge-calls", "sha256": sha256_text('[{"outcome": "verdict"}]')}
    assert UnifiedPipeline._load_page_judge_calls(tmp_path, 1, good) == [{"outcome": "verdict"}]
    assert UnifiedPipeline._load_page_judge_calls(tmp_path, 2, good) == []


def test_the_deadline_hit_is_a_call_without_a_reply() -> None:
    record = judge_call_record(
        page_num=3, outcome=JUDGE_CALL_DEADLINE, judge=_FakeVLM([]), reading="text"
    )
    assert record["outcome"] == JUDGE_CALL_DEADLINE and "reply" not in record
    assert record["judge_model"] == "fake-judge-model" and record["page"] == 3
    assert JUDGE_CALL_VERDICT == "verdict"


def test_readers_of_the_pages_folder_do_not_take_the_side_file_for_a_sidecar(
    tmp_path: Path,
) -> None:
    from socr.benchmark import acceptance, scoreboard
    from socr.core.page_files import page_sidecar_paths

    doc = tmp_path / "doc"
    pages = doc / "pages"
    pages.mkdir(parents=True)
    (pages / "00001.json").write_text(
        json.dumps({"page_num": 1, "status": "success", "engine": "qwen"})
    )
    (pages / "00001.md").write_text("x")
    (pages / "00001.judge-calls").write_text("[]")
    assert [p.name for p in page_sidecar_paths(pages)] == ["00001.json"]
    counts = scoreboard.count_document(doc)
    assert counts["coverage.pages"] == 1 and not counts["coverage.unreadable_sidecars"]
    assert acceptance._terminal_pages(doc).noncanonical == []


def test_our_own_deadline_is_reported_once_and_the_inner_judges_own_timeout_is_not() -> None:
    import time

    from socr.judge.judge import PageJudgeTimeoutError
    from socr.pipeline.orchestrator import UnifiedPipeline

    class _Slow:
        def assess(self, output, provider):
            time.sleep(0.4)
            raise AssertionError("the deadline has long passed; the result is discarded")

    class _InnerTimeout:
        def assess(self, output, provider):
            raise TimeoutError("raised by the judge itself")

    seen: list[int] = []
    out = PageOutput(page_num=4, text="reading", status=PageStatus.SUCCESS, engine="qwen")
    slow = UnifiedPipeline._TimeoutJudge(
        _Slow(), 0.05, on_deadline=lambda o: seen.append(o.page_num)
    )
    with pytest.raises(PageJudgeTimeoutError):
        slow.assess(out, None)
    assert seen == [4]
    inner = UnifiedPipeline._TimeoutJudge(
        _InnerTimeout(), 5.0, on_deadline=lambda o: seen.append(-1)
    )
    with pytest.raises(PageJudgeTimeoutError):
        inner.assess(out, None)
    assert seen == [4], "an inner timeout is recorded by the inner judge, not as our deadline"
