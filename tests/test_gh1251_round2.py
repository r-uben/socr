"""#1251 (C3), review round on PR #1338: resume must not trust an empty verdict over the trace,
the predicate keys on the absence of evidence, the words PASS says how much it compared, and the
judge-calls file is durable (seeded on a re-judge and on a retry, kept when unreadable, raw replies
kept, no late record after our own deadline).

Run-level tests pin a DIFFERENCE between two arms of one process (CLAUDE.md), in both provider
states where the provider matters, and never an absolute outcome measured on this machine.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_gh1185_text_layer_numbers_e2e import _crash_after_page_one
from test_gh1251_judge_not_a_check import (
    _calls,
    _FakeVLM,
    _judged_run,
    _out,
    _p,
    _page,
    _verdict,
)
from test_gh1294_layer_words import _rotated_pdf, _run

import socr
from socr.core.marks import Check
from socr.core.pass_trace import TraceEntry, TraceResult
from socr.core.page_credential import sha256_text
from socr.core.result import FailureMode, PageStatus
from socr.core.state import PageState
from socr.core.verdict import (
    JUDGE_ONLY_REASON,
    PageVerdict,
    page_text_digest,
    resumed_verdict_with_judge_only_clause,
    silent_success_clauses,
)
from socr.judge.judge import (
    JUDGE_CALL_DEADLINE,
    JUDGE_CALL_ERROR,
    JudgeReplyUnparseableError,
    judge_call_record,
    page_judge_prompt_sha256,
    parse_verdict,
)
from socr.pipeline.orchestrator import UnifiedPipeline, _page_judge_prompt_digest


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# --- 1. resume must not trust an empty published verdict over the trace --------------------


def _words_entry(text: str) -> TraceEntry:
    return TraceEntry(
        "text_layer_words", TraceResult.PASS, "page", "", text_sha=page_text_digest(text)
    )


def _numbers_entry(text: str) -> TraceEntry:
    return TraceEntry(
        "text_layer_numbers", TraceResult.PASS, "page", "", text_sha=page_text_digest(text)
    )


def test_the_resumed_verdict_is_derived_from_the_trace_the_page_will_write() -> None:
    out = _out()
    empty = PageVerdict(1, ())
    good = (_words_entry(out.text), _numbers_entry(out.text))
    # evidence on this text: the empty published verdict stands
    assert resumed_verdict_with_judge_only_clause(empty, out, good) is empty
    # the same page, the same empty verdict, only the trace differs: no evidence, or a PASS
    # computed on other text, or no trace at all (a sidecar that recorded none)
    stale = (_words_entry("an older reading"), _numbers_entry("an older reading"))
    for trace in ((), stale, None, good[:1]):
        got = resumed_verdict_with_judge_only_clause(empty, out, trace)
        assert [c.reason for c in got.clauses] == [JUDGE_ONLY_REASON], trace


def test_other_published_clauses_are_never_discarded_and_scope_is_kept() -> None:
    from socr.core.marks import CheckResult
    from socr.core.verdict import VerdictClause

    out = _out()
    other = VerdictClause(Check.TABLE_LABEL, CheckResult.UNKNOWN, "page", "a published reason")
    carries = PageVerdict(1, (other,))
    assert resumed_verdict_with_judge_only_clause(carries, out, ()) is carries
    native = _out(engine="native")
    empty = PageVerdict(1, ())
    assert resumed_verdict_with_judge_only_clause(empty, native, ()) is empty
    not_success = _out()
    not_success = type(not_success)(**{**not_success.__dict__, "status": PageStatus.WARNING})
    assert resumed_verdict_with_judge_only_clause(empty, not_success, ()) is empty


def _tamper_page_one_trace(out: Path, *, drop_words: bool) -> None:
    """Edit page 1's sidecar the way a sidecar can be wrong: its words PASS names other text (or
    is missing), while its verdict stays EMPTY and its status SUCCESS."""
    path = next(out.rglob("pages/00001.json"))
    side = json.loads(path.read_text())
    assert side["verdict"]["clauses"] == [], "fixture premise: the page was clean"
    checks = side["checks"]
    for entry in checks:
        if entry["check"] == "text_layer_words" and entry["result"] == "PASS":
            if drop_words:
                entry["result"], entry["reason"] = "NOT_RUN", "removed by the test"
                entry.pop("text_sha", None)
            else:
                entry["text_sha"] = sha256_text("a different text")
    path.write_text(json.dumps(side, indent=2, ensure_ascii=False))


@pytest.mark.parametrize("drop_words", [False, True], ids=["stale digest", "no words pass"])
@pytest.mark.parametrize("ladder", [True, False], ids=["provider", "no provider"])
def test_a_resumed_success_page_with_no_verified_evidence_is_not_kept_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ladder: bool, drop_words: bool
) -> None:
    """Two arms of one resume, differing only in whether page 1's sidecar still holds its
    digest-verified words PASS. The page is kept (not re-read) in both; only the arm that lost
    the evidence is demoted, at every level, and the untouched arm resumes as it was."""
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    if not ladder:
        out = tmp_path / "none"
        _run(out, pdf, monkeypatch, lose_label=False, pages_read=[], ladder=False)
        side, _md = _page(out)
        assert side["winning_output"]["engine"] != "qwen", (
            "with no provider no model reads the page"
        )
        reasons = [c["reason"] for c in side["verdict"]["clauses"]]
        assert JUDGE_ONLY_REASON not in reasons, "nothing is a model reading: no such clause"
        return
    arms = {}
    for name, tamper in (("control", False), ("tampered", True)):
        out = tmp_path / name
        _run(out, pdf, monkeypatch, lose_label=False, pages_read=[], ladder=ladder)
        before = _page(out)
        if tamper:
            _tamper_page_one_trace(out, drop_words=drop_words)
        _crash_after_page_one(out)
        resume: list[int] = []
        _run(out, pdf, monkeypatch, lose_label=False, pages_read=resume, ladder=ladder)
        assert 2 in resume, "the resume did not reach the page loop, so the ledger was not tested"
        assert 1 not in resume, f"page 1 was re-read on resume: {resume}"
        after = _page(out)
        words = [e for e in after[0]["checks"] if e["check"] == "text_layer_words"]
        arms[name] = (before, after, [e["result"] for e in words])
    before, after, _ = arms["control"]
    assert (after[0]["status"], after[0]["verdict"]["clauses"]) == ("success", [])
    _before_t, after_t, words_t = arms["tampered"]
    assert after_t[0]["status"] == "warning"
    assert after_t[0]["failure_mode"] == FailureMode.READING_UNCORROBORATED.value
    assert [c["reason"] for c in after_t[0]["verdict"]["clauses"]] == [JUDGE_ONLY_REASON]
    assert "page_judge=UNKNOWN" in after_t[1]
    assert "PASS" not in words_t, "the trace written never says PASS without the digest"


# --- 2. the predicate keys on the absence of evidence --------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"outcome": "page_judge_completed", "accepted": True},
        {"outcome": "", "accepted": True},
        {"outcome": "", "accepted": None},
    ],
    ids=["completed", "empty outcome (crop-patched re-judge)", "no judge fields"],
)
def test_a_model_reading_needs_the_evidence_whatever_its_judge_fields_say(fields: dict) -> None:
    """The difference is only the layer's words: with them the page is clean, without them it is
    marked, for every judge-field state."""
    with_words = silent_success_clauses(_out(**fields), _p())
    without = silent_success_clauses(_out(**fields), _p(own_layer_word_reference=""))
    assert with_words == ()
    assert [c.reason for c in without] == [JUDGE_ONLY_REASON]


# --- 3. the words PASS says how much it compared; the scoreboard counts what C3 relies on --


def test_the_words_pass_names_how_many_words_it_compared_and_the_page_ships() -> None:
    from socr.core.verdict import text_layer_words_outcome

    out = _out(_out().text + "\nplus some extra words")
    entry = text_layer_words_outcome(out, _p()).entry
    assert entry.result is TraceResult.PASS
    assert "compared 2 word(s); the page ships 6" in entry.reason, entry.reason


def _sidecar_for(tmp: Path, name: str, *, words: str | None, compared: int, status="success"):
    """A one-page document folder with a model SUCCESS page and a hand-written trace."""
    doc = tmp / name
    pages = doc / "pages"
    pages.mkdir(parents=True)
    text = "Korea 4.0 2.6\nChina 6.9 7.0"
    checks = []
    if words:
        checks.append(
            {
                "check": "text_layer_words",
                "result": words,
                "scope": "page",
                "reason": "r",
                "text_sha": page_text_digest(text),
            }
        )
        checks.append(
            {
                "check": "text_layer_numbers",
                "result": "PASS",
                "scope": "page",
                "reason": "r",
                "text_sha": page_text_digest(text),
            }
        )
    (pages / "00001.json").write_text(
        json.dumps(
            {
                "page_num": 1,
                "status": status,
                "engine": "qwen",
                "terminal": True,
                "audit_passed": True,
                "winning_output": {"text": text, "engine": "qwen"},
                "checks": checks,
                "text_layer_words": {"compared": compared, "missing": 0},
            }
        )
    )
    (pages / "00001.md").write_text(text)
    return doc


def test_the_scoreboard_counts_model_success_without_evidence_and_weak_words_passes(
    tmp_path: Path,
) -> None:
    from socr.benchmark import scoreboard

    strong = scoreboard.count_document(_sidecar_for(tmp_path, "a", words="PASS", compared=200))
    weak = scoreboard.count_document(
        _sidecar_for(tmp_path, "b", words="PASS", compared=scoreboard.WEAK_WORDS_PASS_BELOW - 1)
    )
    bare = scoreboard.count_document(_sidecar_for(tmp_path, "c", words=None, compared=0))
    demoted = scoreboard.count_document(
        _sidecar_for(tmp_path, "d", words=None, compared=0, status="warning")
    )
    key_evidence = "records.model_success_without_verified_evidence"
    key_weak = "records.pages_weak_words_pass"
    assert (strong[key_evidence], strong[key_weak]) == (0, 0)
    assert (weak[key_evidence], weak[key_weak]) == (0, 1)
    assert (bare[key_evidence], bare[key_weak]) == (1, 0)
    assert demoted[key_evidence] == 0, "a page that is not SUCCESS is not the invariant's concern"


# --- 5. judge-calls durability --------------------------------------------------------------


def test_a_reply_that_does_not_parse_is_still_kept_in_full() -> None:
    reply = "I think the page is fine. " * 40  # long, not JSON
    with pytest.raises(JudgeReplyUnparseableError) as caught:
        parse_verdict(reply)
    assert caught.value.raw == reply and isinstance(caught.value, ValueError)
    record = judge_call_record(
        page_num=1, outcome=JUDGE_CALL_ERROR, judge=_FakeVLM([]), reading="r", error=caught.value
    )
    assert record["reply"] == reply and record["reply_chars"] == len(reply)
    assert record["error"]["type"] == "JudgeReplyUnparseableError"
    # a JSON value that is not an object is the same failure, with the reply kept
    with pytest.raises(JudgeReplyUnparseableError) as listed:
        parse_verdict("[1, 2]")
    assert listed.value.raw == "[1, 2]"


def test_the_vlm_judge_records_an_unparseable_reply_through_the_page_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    reply = "not json at all " * 30
    err = JudgeReplyUnparseableError("no JSON object found", raw=reply)
    _judged_run(out, pdf, monkeypatch, _FakeVLM([err]), [])
    _, calls = _calls(out, 1)
    assert [c["outcome"] for c in calls] == [JUDGE_CALL_ERROR]
    assert calls[0]["reply"] == reply


def test_a_failure_to_build_the_call_record_never_decides_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from socr.pipeline import agentic
    from socr.pipeline.agentic import VLMPageJudge

    def boom(**_kw):
        raise RuntimeError("record builder broke")

    monkeypatch.setattr("socr.judge.judge.judge_call_record", boom)
    seen: list[dict] = []
    out = _out()
    ok = VLMPageJudge(_FakeVLM([_verdict()]), lambda n: "img", record_call=seen.append)
    assert ok.assess(out, SimpleNamespace()).accept is True, "the verdict is returned"
    failing = VLMPageJudge(
        _FakeVLM([TimeoutError("slow")]), lambda n: "img", record_call=seen.append
    )
    with pytest.raises(TimeoutError, match="slow"):  # the judge's own exception, not the builder's
        failing.assess(out, SimpleNamespace())
    assert seen == [] and agentic is not None


def test_the_run_manifest_and_every_call_record_share_one_prompt_identity() -> None:
    assert _page_judge_prompt_digest() == page_judge_prompt_sha256()
    record = judge_call_record(page_num=1, outcome="verdict", judge=_FakeVLM([]), reading="x")
    assert record["prompt_sha256"] == _page_judge_prompt_digest()
    from socr.judge.judge import load_judge_prompt

    assert record["prompt_sha256"] == hashlib.sha256(load_judge_prompt().encode()).hexdigest()


def test_a_call_that_answers_after_our_deadline_is_not_recorded_the_deadline_row_stands() -> None:
    """Chosen design (documented in the log): once our deadline has hit, the worker is
    abandoned and its late answer is discarded, so it is not recorded either. Records made
    before the deadline stay; the deadline row is written atomically with the abandon flag."""
    from socr.judge.judge import PageJudgeTimeoutError
    from socr.pipeline.agentic import VLMPageJudge

    state = SimpleNamespace(pages={4: PageState(page_num=4)})
    record = lambda rec: UnifiedPipeline._record_page_judge_call(state, rec)  # noqa: E731

    class _SlowThenAnswers(_FakeVLM):
        def judge(self, image_path, ocr_text):
            time.sleep(0.5)
            return _verdict()

    leaf = VLMPageJudge(_SlowThenAnswers([]), lambda n: "img", record_call=record)
    guard = UnifiedPipeline._TimeoutJudge(
        leaf,
        0.05,
        on_deadline=lambda o: UnifiedPipeline._record_page_judge_call(
            state,
            judge_call_record(
                page_num=o.page_num, outcome=JUDGE_CALL_DEADLINE, judge=leaf._judge, reading=o.text
            ),
        ),
    )
    out = _out()
    out.page_num = 4
    with pytest.raises(PageJudgeTimeoutError):
        guard.assess(out, None)
    time.sleep(0.8)  # the abandoned worker answers now
    assert [c["outcome"] for c in state.pages[4].judge_calls] == [JUDGE_CALL_DEADLINE]
    # a call that answers inside the deadline is recorded, once
    fast = VLMPageJudge(_FakeVLM([_verdict()]), lambda n: "img", record_call=record)
    UnifiedPipeline._TimeoutJudge(fast, 5.0).assess(out, None)
    assert [c["outcome"] for c in state.pages[4].judge_calls] == [JUDGE_CALL_DEADLINE, "verdict"]


def test_a_retried_page_keeps_the_first_attempts_judge_calls_and_the_retry_follows() -> None:
    first_call = {"page": 1, "outcome": "timeout"}
    ps = PageState(page_num=1)
    ps.judge_calls.append(first_call)
    ps.judge_calls_unrestored = {"reason": "x", "ref": {"path": "00001.judge-calls"}}
    snapshot = PageState(page_num=1)  # the page before its first attempt: no calls
    state = SimpleNamespace(pages={1: ps}, events=[])
    first = SimpleNamespace(
        output=SimpleNamespace(
            status=PageStatus.ERROR, failure_mode=FailureMode.NONE, engine="qwen"
        ),
        disposition=SimpleNamespace(
            ending=SimpleNamespace(value="e"), primary_reason=SimpleNamespace(value="r")
        ),
    )
    UnifiedPipeline._reset_page_for_retry(state, 1, copy.deepcopy(snapshot), [], first, [])
    fresh = state.pages[1]
    assert fresh is not ps and fresh.judge_calls == [first_call]
    UnifiedPipeline._record_page_judge_call(state, {"page": 1, "outcome": "verdict"})
    assert [c["outcome"] for c in fresh.judge_calls] == ["timeout", "verdict"]
    assert fresh.judge_calls_unrestored == ps.judge_calls_unrestored


# --- the file survives a re-judge (seeded) and an unreadable one is kept ---------------------


@pytest.fixture
def hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The #1013 file's hermetic pins (CI has no provider and no judge model)."""
    from socr.core.providers import PROFILE_QWEN_LOCAL

    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


def _recording_harness(tmp_path: Path):
    """The #1013 harness with its stub judge wrapped so each of its calls is recorded on the page
    the way the VLM page judge records them."""
    from test_gh1013_rejudge_timed_out_candidate import _Harness

    from socr.judge.judge import JudgeVerdict, PageJudgeTimeoutError

    h = _Harness(tmp_path)

    class _Recording:
        model = "stub-judge"

        def __init__(self, inner, state) -> None:
            self._inner, self._state = inner, state

        def assess(self, output, provider):
            try:
                decision = self._inner.assess(output, provider)
            except PageJudgeTimeoutError as exc:
                rec = judge_call_record(
                    page_num=output.page_num,
                    outcome="timeout",
                    judge=self,
                    reading=output.text,
                    error=exc,
                )
                UnifiedPipeline._record_page_judge_call(self._state, rec)
                raise
            rec = judge_call_record(
                page_num=output.page_num,
                outcome="verdict",
                judge=self,
                reading=output.text,
                verdict=JudgeVerdict(faithful=True, raw="{}"),
            )
            UnifiedPipeline._record_page_judge_call(self._state, rec)
            return decision

    h._build_judge = lambda state: (
        setattr(state, "agentic_judge_model", h.judge_identity) or _Recording(h.judge, state)
    )
    return h


def _file(h) -> Path:
    return next((h.out / "doc" / "pages").glob("00001.judge-calls"))


def test_a_rejudge_extends_the_first_runs_calls_and_does_not_replace_them(
    tmp_path: Path, hermetic: None
) -> None:
    h = _recording_harness(tmp_path)
    h.judge.mode = "timeout"
    h.run()
    first = json.loads(_file(h).read_text())
    assert [c["outcome"] for c in first] == ["timeout"], "fixture premise: run 1 recorded"
    h.judge.mode = "accept"
    h.ocr_calls.clear()
    h.run()
    assert h.ocr_calls == [], "fixture premise: the kept candidate was re-judged, not re-read"
    calls = json.loads(_file(h).read_text())
    assert [c["outcome"] for c in calls] == ["timeout", "verdict"], calls
    assert calls[0] == first[0], "the first run's record is carried over untouched"
    ref = h.sidecar()["judge_calls"]
    assert ref["count"] == 2 and ref["sha256"] == sha256_text(_file(h).read_text())


@pytest.mark.parametrize("damage", ["altered", "missing"])
def test_an_unreadable_calls_file_is_kept_and_flagged_when_new_calls_arrive(
    tmp_path: Path, hermetic: None, damage: str
) -> None:
    h = _recording_harness(tmp_path)
    h.judge.mode = "timeout"
    h.run()
    original = _file(h).read_text()
    old_ref = h.sidecar()["judge_calls"]
    if damage == "altered":
        _file(h).write_text(original.replace("timeout", "verdict"), encoding="utf-8")
    else:
        _file(h).unlink()
    h.judge.mode = "accept"
    h.run()
    ref = h.sidecar()["judge_calls"]
    assert ref["count"] == 1 and "prior_unavailable" in ref
    prior = ref["prior_unavailable"]
    assert prior["sha256"] == old_ref["sha256"] and prior["reason"]
    if damage == "altered":
        aside = _file(h).with_name("00001.judge-calls.unrestored")
        assert prior["kept_as"] == aside.name
        assert aside.read_text() == original.replace("timeout", "verdict"), "never deleted"
    else:
        assert prior["kept_as"] is None
    # the next flush of the same page does not move the NEW file aside as if it were the lost one
    assert json.loads(_file(h).read_text())[0]["outcome"] == "verdict"


@pytest.mark.parametrize("damage", ["altered", "missing"])
def test_an_unreadable_calls_file_on_a_resumed_page_keeps_file_and_reference_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    out = tmp_path / "out"
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    _judged_run(out, pdf, monkeypatch, _FakeVLM([_verdict('{"faithful": true}')]), [])
    path, _calls_before = _calls(out, 1)
    before_ref = _page(out)[0]["judge_calls"]
    if damage == "altered":
        path.write_text("[]", encoding="utf-8")
    else:
        path.unlink()
    _crash_after_page_one(out)
    resume: list[int] = []
    _judged_run(out, pdf, monkeypatch, _FakeVLM([_verdict('{"faithful": false}')]), resume)
    assert 2 in resume and 1 not in resume, "fixture premise: page 1 was kept, page 2 re-run"
    ref = _page(out)[0]["judge_calls"]
    assert ref["unavailable"] and ref["sha256"] == before_ref["sha256"], ref
    if damage == "altered":
        assert path.read_text() == "[]", "an unreadable file is kept as it was, never deleted"
    else:
        assert not path.exists()


def test_a_judge_calls_file_is_never_deleted_for_a_page_with_an_unreadable_history(
    tmp_path: Path,
) -> None:
    """The writer: no calls in memory and an unreadable history flagged: nothing is unlinked."""
    path = tmp_path / "00001.judge-calls"
    path.write_text("corrupt", encoding="utf-8")
    ps = PageState(page_num=1)
    ps.judge_calls_unrestored = {"reason": "digest mismatch", "ref": {"path": path.name}}
    ref = UnifiedPipeline._write_page_judge_calls(ps, tmp_path, 1)
    assert path.read_text() == "corrupt"
    assert ref == {"path": path.name, "unavailable": "digest mismatch"}
    clean = PageState(page_num=1)
    assert UnifiedPipeline._write_page_judge_calls(clean, tmp_path, 1) is None
    assert not path.exists(), "a page with no history at all has no file (stale one removed)"


# --- 6. refinalize point-drops: the clause is derived after the repaired payloads exist -------


def test_point_drops_derives_the_clause_after_the_repaired_row_binding_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from test_gh1282_refinalize_point_drops import SLIPPED, _runs, _tool

    _on, off, pdf_dir = _runs(tmp_path, SLIPPED, 0)
    import socr.core.verdict as verdict_module

    real = verdict_module.with_judge_only_clause
    seen: list = []

    def spy(published, output, old_checks, event_kinds=frozenset(), **kw):
        seen.append(kw.get("row_binding_records"))
        return real(published, output, old_checks, event_kinds, **kw)

    monkeypatch.setattr(verdict_module, "with_judge_only_clause", spy)
    _tool(off, pdf_dir, "--apply", capsys=capsys)
    assert len(seen) == 1
    records = seen[0]
    assert records, (
        "the clause was derived with no row-binding records: the repaired ones come later"
    )
    repaired = [r for r in records.values() if r.get("repairs")]
    assert repaired, "the records handed over are the repaired payloads, not the old ones"
