"""#1251 (C3), final review round on PR #1338.

* a resumed page with the #1181 exception takes the #1181 clause, derived (event present) and the
  judge-only clause (event absent), never an empty verdict;
* a malformed judge reply field is an unparseable reply that keeps the reply, also across the
  killable child's error transport;
* a second unreadable calls history is never overwritten;
* the scoreboard counts what C3 relies on only from a verified trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_gh1181_exact_pass_unjudged_clause import EXACT_PASS, WITH_PROSE, _state
from test_gh1251_round2 import _out

from socr.core.manifest import _select_and_finalize_page
from socr.core.page_credential import sha256_text
from socr.core.result import PageStatus
from socr.core.state import PageState
from socr.core.verdict import (
    EXACT_PASS_UNJUDGED_REASON,
    JUDGE_ONLY_REASON,
    PageVerdict,
    resumed_verdict_with_judge_only_clause,
)
from socr.judge.judge import JudgeReplyUnparseableError, parse_verdict
from socr.pipeline.orchestrator import UnifiedPipeline


# --- 1. resume with the #1181 exception ------------------------------------------------------


def _resumed(tmp_path, kinds):
    """The seam on a TERMINAL RESUMED page: published verdict empty, no trace in the sidecar, the
    unjudged EXACT_PASS accept recorded in the page's judge reason."""
    state = _state(tmp_path, kinds=kinds, own_layer=None)
    ps = state.pages[1]
    ps.published_verdict = PageVerdict(1, ())
    ps.published_checks = ()
    return _select_and_finalize_page(state, 1)


@pytest.mark.parametrize("event", [True, False], ids=["event replayed", "event absent"])
def test_a_resumed_exact_pass_page_never_reads_success_with_no_clause(tmp_path, event) -> None:
    rec = _resumed(tmp_path, EXACT_PASS if event else ())
    reasons = [c.reason for c in rec.verdict.clauses]
    assert rec.output.status is PageStatus.WARNING
    assert reasons, "a SUCCESS-reading page with zero clauses and no evidence"
    # the arms differ only in the replayed event: which clause names the page differs with it
    assert (EXACT_PASS_UNJUDGED_REASON in reasons) is event
    assert (JUDGE_ONLY_REASON in reasons) is (not event)


def test_the_helper_derives_the_1181_clause_for_the_event_and_reason() -> None:
    from socr.core.verdict import EXACT_PASS_ACCEPT_REASON, NATIVE_TABLE_EXACT_PASS_KIND

    out = _out(reason=EXACT_PASS_ACCEPT_REASON)
    empty = PageVerdict(1, ())
    got = resumed_verdict_with_judge_only_clause(
        empty, out, (), frozenset({NATIVE_TABLE_EXACT_PASS_KIND})
    )
    assert [c.reason for c in got.clauses] == [EXACT_PASS_UNJUDGED_REASON]


# --- 2. malformed fields are an unparseable reply that keeps the reply ----------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"faithful": true, "issues": 7}',
        '{"faithful": true, "suggested_action": ["accept"]}',
    ],
)
def test_a_malformed_field_is_an_unparseable_reply_holding_the_reply(reply: str) -> None:
    with pytest.raises(JudgeReplyUnparseableError) as caught:
        parse_verdict(reply)
    assert caught.value.raw == reply


def test_the_killable_error_transport_keeps_the_raw_reply() -> None:
    from socr.core.killable import CallSpec, run_killable

    reply = '{"faithful": true, "issues": 7}'
    spec = CallSpec(func="socr.judge.judge:parse_verdict", args=(reply,))
    with pytest.raises(RuntimeError) as caught:
        run_killable(spec, timeout=60)
    assert getattr(caught.value, "raw", None) == reply


# --- 3. a second unreadable history is not overwritten --------------------------------------


def test_a_second_unreadable_history_takes_its_own_aside_name(tmp_path: Path) -> None:
    path = tmp_path / "00001.judge-calls"
    first = tmp_path / "00001.judge-calls.unrestored"
    first.write_text("first unreadable history", encoding="utf-8")
    path.write_text("second unreadable history", encoding="utf-8")
    ps = PageState(page_num=1)
    ps.judge_calls = [{"page": 1, "outcome": "verdict"}]
    ps.judge_calls_unrestored = {"reason": "digest mismatch", "ref": {"path": path.name}}
    ref = UnifiedPipeline._write_page_judge_calls(ps, tmp_path, 1)
    assert first.read_text() == "first unreadable history"
    kept = tmp_path / ref["prior_unavailable"]["kept_as"]
    assert kept != first and kept.read_text() == "second unreadable history"
    assert json.loads(path.read_text())[0]["outcome"] == "verdict"


# --- 4. the scoreboard counts only verified evidence ----------------------------------------


def _doc(tmp: Path, name: str, *, text, sha_of: str, compared: int, words="PASS") -> Path:
    from socr.core.verdict import page_text_digest

    doc = tmp / name
    pages = doc / "pages"
    pages.mkdir(parents=True)
    shipped = "Korea 4.0 2.6\nChina 6.9 7.0"
    checks = [
        {
            "check": check,
            "result": result,
            "scope": "page",
            "reason": "r",
            "text_sha": page_text_digest(sha_of),
        }
        for check, result in (("text_layer_words", words), ("text_layer_numbers", "PASS"))
    ]
    winning = {"engine": "qwen"} if text is None else {"text": text, "engine": "qwen"}
    (pages / "00001.json").write_text(
        json.dumps(
            {
                "page_num": 1,
                "status": "success",
                "engine": "qwen",
                "terminal": True,
                "audit_passed": True,
                "winning_output": winning,
                "checks": checks,
                "text_layer_words": {"compared": compared, "missing": 0},
            }
        )
    )
    (pages / "00001.md").write_text(shipped)
    return doc


def test_the_scoreboard_does_not_trust_missing_text_or_a_stale_pass(tmp_path: Path) -> None:
    from socr.benchmark import scoreboard

    shipped = "Korea 4.0 2.6\nChina 6.9 7.0"
    weak = scoreboard.WEAK_WORDS_PASS_BELOW - 1
    key_evidence = "records.model_success_without_verified_evidence"
    key_weak = "records.pages_weak_words_pass"
    good = scoreboard.count_document(
        _doc(tmp_path, "a", text=shipped, sha_of=shipped, compared=weak)
    )
    no_text = scoreboard.count_document(
        _doc(tmp_path, "b", text=None, sha_of="", compared=weak)  # PASS digest of an empty text
    )
    stale = scoreboard.count_document(
        _doc(tmp_path, "c", text=shipped, sha_of="an older reading", compared=weak)
    )
    assert (good[key_evidence], good[key_weak]) == (0, 1)
    assert no_text[key_evidence] == 1, "no winning text is no evidence, not an empty page"
    assert (stale[key_evidence], stale[key_weak]) == (1, 0), "a stale PASS is not a weak PASS"
