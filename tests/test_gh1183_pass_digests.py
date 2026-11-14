"""#1183 (C2 part 1, review round 2): a PASS names the text it was computed on.

Structural rule: a page-scope PASS carries ``text_sha`` and a ``row_binding`` ``table:N`` PASS
carries ``table_sha``. At every write of ``checks`` a PASS whose digest is not the digest of the
text being shipped now (or that has none) is NOT_RUN. Each test below is one input the round-2
reviews built (Fable A/B/C, Carbon residuals 1, 3, 5, the control-byte blocker and the
recomputed-finding should-fix).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from socr.core.marks import Check, CheckResult
from socr.core.pass_trace import TraceEntry, TraceResult, trace_from_dicts, trace_to_dicts
from socr.core.result import PageOutput, PageStatus
from socr.core.verdict import (
    CHECKED_TEXT_CHANGED_REASON,
    ROW_BINDING_CHECK,
    PageVerdict,
    VerdictClause,
    build_pass_trace,
    control_bytes_clause,
    control_bytes_outcome,
    page_text_digest,
    reconcile_pass_trace,
    rerun_text_dependent_checks,
    table_block_digest,
    table_block_digests,
    text_layer_words_outcome,
    verify_pass_digests,
    write_reconciled_trace,
)

LAYER = "Korea 4.0 2.6 5.7 4.0\nChina 6.9 7.0 6.6 7.1"
TABLE_A = "| a | b |\n| - | - |\n| Korea | 4.0 |\n"
TABLE_B = "| c | d |\n| - | - |\n| China | 6.9 |\n"
CAPTION = "| Figure 1 | caption |\n| - | - |\n| shown | here |\n"


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


def _out(text: str = LAYER, engine: str = "qwen") -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine=engine)


def _trace(out, p, verdict=None, **kw) -> dict[tuple[str, str], TraceEntry]:
    return {
        (e.check, e.scope): e for e in build_pass_trace(out, p, verdict or PageVerdict(1), **kw)
    }


def _passes(trace) -> set[tuple[str, str]]:
    return {k for k, e in trace.items() if e.result is TraceResult.PASS}


def _bound(text: str, index: int) -> dict:
    return {
        "table_index": index,
        "outcome": "bound",
        "table_sha": table_block_digest(text, index),
    }


# --- the digest rule itself ---------------------------------------------------------------


def test_every_fresh_pass_carries_the_digest_of_the_text_it_ran_on() -> None:
    text = TABLE_A + "\ntext\n\n" + TABLE_B
    p = _p(row_binding={"p1-t0": _bound(text, 0), "p1-t1": _bound(text, 1)})
    trace = _trace(_out(text), p)
    page_passes = [
        e for (c, s), e in trace.items() if e.result is TraceResult.PASS and c != ROW_BINDING_CHECK
    ]
    assert page_passes, "premise: the page has page-scope passes"
    assert {e.text_sha for e in page_passes} == {page_text_digest(text)}
    tables = table_block_digests(text)
    assert trace[(ROW_BINDING_CHECK, "table:1")].table_sha == tables[0]
    assert trace[(ROW_BINDING_CHECK, "table:2")].table_sha == tables[1]
    assert trace[(ROW_BINDING_CHECK, "table:1")].result is TraceResult.PASS
    assert tables[0] != tables[1]


def test_digests_round_trip_and_a_non_string_digest_makes_the_trace_unreadable() -> None:
    entry = TraceEntry(
        "x", TraceResult.PASS, "page", "ok", text_sha="sha256:a", table_sha="sha256:b"
    )
    assert trace_from_dicts(trace_to_dicts((entry,))) == (entry,)
    bad = {"check": "x", "result": "PASS", "scope": "page", "reason": "", "text_sha": 3}
    assert trace_from_dicts([bad]) is None
    plain = TraceEntry("x", TraceResult.NOT_RUN, "page", "r")
    assert set(plain.to_dict()) == {"check", "result", "scope", "reason"}


def test_a_pass_stays_only_for_the_text_it_was_computed_on_and_never_without_a_digest() -> None:
    text = "shipped"
    good = TraceEntry("a", TraceResult.PASS, "page", "", text_sha=page_text_digest(text))
    stale = TraceEntry("b", TraceResult.PASS, "page", "", text_sha=page_text_digest("older"))
    bare = TraceEntry("c", TraceResult.PASS, "page", "")
    finding = TraceEntry("d", TraceResult.FAIL, "page", "found")
    got = {e.check: e for e in verify_pass_digests([good, stale, bare, finding], text)}
    assert got["a"].result is TraceResult.PASS
    assert got["b"].result is TraceResult.NOT_RUN and got["b"].reason == CHECKED_TEXT_CHANGED_REASON
    assert got["c"].result is TraceResult.NOT_RUN
    assert got["d"].result is TraceResult.FAIL, "a finding is not a pass: it stays a finding"


# --- Fable A and Carbon residual 3: a table record is tied to its table's bytes -----------


def test_a_record_does_not_state_a_different_table_that_took_its_ordinal() -> None:
    """Fable A: t0 bound, t1 unverifiable; t0 left the page, so the shipped table 1 is the old
    t1. Ordinal 1 is not the bound table: no PASS."""
    ran_on = TABLE_A + "\ntext\n\n" + TABLE_B
    p = _p(
        row_binding={
            "p1-t0": _bound(ran_on, 0),
            "p1-t1": {"table_index": 1, "outcome": "unverifiable"},
        }
    )
    kept = _trace(_out(ran_on), p)
    assert kept[(ROW_BINDING_CHECK, "table:1")].result is TraceResult.PASS
    shipped = _trace(_out(TABLE_B), p)
    assert shipped[(ROW_BINDING_CHECK, "table:1")].result is not TraceResult.PASS


def test_a_table_whose_bytes_changed_in_place_is_not_a_pass() -> None:
    p = _p(row_binding={"p1-t0": _bound(TABLE_A, 0)})
    edited = TABLE_A.replace("4.0", "9.9")
    assert _trace(_out(TABLE_A), p)[(ROW_BINDING_CHECK, "table:1")].result is TraceResult.PASS
    assert _trace(_out(edited), p)[(ROW_BINDING_CHECK, "table:1")].result is not TraceResult.PASS


def test_a_caption_table_inserted_above_a_bound_table_gives_no_table_pass() -> None:
    """Carbon 3: the figure phase adds a caption table above the bound one."""
    p = _p(row_binding={"p1-t0": _bound(TABLE_A, 0)})
    shifted = _trace(_out(CAPTION + "\n" + TABLE_A), p)
    assert not {k for k in _passes(shifted) if k[0] == ROW_BINDING_CHECK}, shifted
    # the DIFFERENCE is the inserted table alone
    assert _trace(_out(TABLE_A), p)[(ROW_BINDING_CHECK, "table:1")].result is TraceResult.PASS


# --- Fable B, Carbon 5: a replayed PASS is for the replayed text --------------------------


def _first_run(text: str, p=None) -> tuple:
    return tuple(build_pass_trace(_out(text), p or _p(), PageVerdict(1)))


def test_replay_over_the_same_text_keeps_every_pass() -> None:
    stored = _first_run(LAYER)
    assert _passes({(e.check, e.scope): e for e in stored})
    assert build_pass_trace(_out(LAYER), _p(), PageVerdict(1), replay=stored) == stored


def test_a_resumed_page_that_assemble_demotes_to_a_marker_keeps_no_pass() -> None:
    stored = _first_run(LAYER)
    marker = "[page 1 failed: malformed table emission (x)]"
    clause = VerdictClause(Check.TABLE_STRUCTURE, CheckResult.FAIL, "page", "malformed")
    got = _trace(_out(marker), _p(), PageVerdict(1, (clause,)), replay=stored)
    assert not _passes(got), _passes(got)
    stale = got[("control_bytes", "page")]
    assert stale.result is TraceResult.NOT_RUN and stale.reason == CHECKED_TEXT_CHANGED_REASON


def test_a_resumed_page_whose_text_gains_a_figure_caption_keeps_no_pass() -> None:
    stored = _first_run(LAYER)
    captioned = LAYER + "\n\nFigure 1: a described figure"
    got = _trace(_out(captioned), _p(), replay=stored)
    assert not _passes(got)
    assert _passes(_trace(_out(LAYER), _p(), replay=stored)), "unchanged text keeps them"


def test_a_resumed_row_binding_pass_follows_its_table_across_a_caption_insert() -> None:
    p = _p(row_binding={"p1-t0": _bound(TABLE_A, 0)})
    stored = _first_run(TABLE_A, p)
    got = _trace(_out(CAPTION + "\n" + TABLE_A), _p(), replay=stored)
    assert not {k for k in _passes(got) if k[0] == ROW_BINDING_CHECK}


# --- Fable C and the recomputed-finding should-fix ----------------------------------------


def test_a_verdict_clause_removes_every_old_non_clause_entry_of_its_check_at_any_scope() -> None:
    old = trace_to_dicts(
        (
            TraceEntry(
                "text_layer_drops",
                TraceResult.PASS,
                "table_rows",
                "",
                text_sha=page_text_digest("t"),
            ),
        )
    )
    verdict = PageVerdict(
        1, (VerdictClause(Check.TEXT_LAYER_DROPS, CheckResult.FAIL, "page", "dropped"),)
    )
    got = reconcile_pass_trace(old, verdict, text="t")
    drops = [e for e in got if e["check"] == "text_layer_drops"]
    assert [(e["result"], e["scope"]) for e in drops] == [("FAIL", "page")]


def test_a_recomputed_finding_the_verdict_lacks_is_kept_not_dropped() -> None:
    text = "t"
    old = trace_to_dicts(
        (
            TraceEntry(
                "text_layer_words", TraceResult.PASS, "page", "", text_sha=page_text_digest(text)
            ),
        )
    )
    found = TraceEntry("text_layer_words", TraceResult.FAIL, "page", "words missing")
    got = reconcile_pass_trace(old, PageVerdict(1), recomputed=(found,), text=text)
    assert [(e["check"], e["result"]) for e in got] == [("text_layer_words", "FAIL")]


def test_reconcile_with_a_changed_text_turns_a_stale_pass_not_run_and_a_current_one_stays() -> None:
    old = trace_to_dicts(
        (
            TraceEntry(
                "text_layer_words", TraceResult.PASS, "page", "", text_sha=page_text_digest("old")
            ),
            TraceEntry(
                "prior_ocr_layer", TraceResult.PASS, "page", "", text_sha=page_text_digest("new")
            ),
        )
    )
    got = {e["check"]: e for e in reconcile_pass_trace(old, PageVerdict(1), text="new")}
    assert got["text_layer_words"]["result"] == "NOT_RUN"
    assert got["text_layer_words"]["reason"] == CHECKED_TEXT_CHANGED_REASON
    assert got["prior_ocr_layer"]["result"] == "PASS"


# --- Carbon: control bytes are read off the shipped text ----------------------------------


@pytest.mark.parametrize("engine", ["qwen", "native"])
def test_a_raw_control_byte_in_the_shipped_text_is_a_finding_not_a_pass(engine: str) -> None:
    out = _out("Korea \x01 4.0\nChina 6.9", engine)
    outcome = control_bytes_outcome(out, _p(control_bytes_replaced=None))
    assert outcome.entry.result is TraceResult.FAIL
    assert outcome.clause is not None and "U+0001" in outcome.clause.reason
    assert control_bytes_clause(out, _p()) is not None
    clean = control_bytes_outcome(_out("Korea 4.0\nChina 6.9", engine), _p())
    assert clean.entry.result is TraceResult.PASS


def test_point_drops_splicing_a_control_cell_cannot_stamp_a_control_byte_pass() -> None:
    """Carbon: the tool splices layer cells into the text without the guard; the trace it writes
    is read off those bytes."""
    spliced = _out("| a | b |\n| - | - |\n| Korea | 4.0\x01 |\n")
    old = trace_to_dicts(
        (TraceEntry("control_bytes", TraceResult.NOT_RUN, "page", "resumed without trace"),)
    )
    sidecar: dict = {"checks": old}
    write_reconciled_trace(
        sidecar,
        old,
        PageVerdict(1),
        recomputed=rerun_text_dependent_checks(spliced, _p(), old),
        text=spliced.text,
    )
    entry = {e["check"]: e for e in sidecar["checks"]}["control_bytes"]
    assert entry["result"] == "FAIL", entry


# --- Carbon residual 1: the stripped footnote words are not owed --------------------------


def test_words_pass_does_not_count_words_a_stripped_footnote_wrapper_owned() -> None:
    footnote = "note shown below"
    p = _p(
        own_layer_word_reference=footnote,
        own_layer_figure_words="",
        hidden_footnote_words=footnote,
    )
    shipped = _out("![Figure 1](fig.png)", engine="native")
    entry = text_layer_words_outcome(shipped, p).entry
    assert entry.result is not TraceResult.PASS, entry
    # the DIFFERENCE is the stripped wrapper alone
    owed = _p(
        own_layer_word_reference=footnote, own_layer_figure_words="", hidden_footnote_words=""
    )
    assert text_layer_words_outcome(shipped, owed).entry.result is not TraceResult.NOT_APPLICABLE


# --- through the finalize seam ------------------------------------------------------------


def test_the_seam_replays_passes_for_the_saved_text_and_not_for_a_text_assemble_rewrote(
    tmp_path,
) -> None:
    """A resumed page: the same saved text keeps its replayed passes; the identical page with a
    text assemble changed (a failure marker, a figure caption) keeps none. The DIFFERENCE is the
    shipped text alone."""
    from test_gh1145_chart_values_clause import _state

    from socr.core.manifest import _select_and_finalize_page

    body = "Korea 4.0 and China 6.9 rose over the year."

    def _resumed(saved):
        state = _state(tmp_path, text=body, engine="qwen", status=PageStatus.SUCCESS)
        first = _select_and_finalize_page(state, 1, saved_text=body)
        ps = state.pages[1]
        ps.published_verdict = first.verdict
        ps.published_checks = first.checks
        return first, _select_and_finalize_page(state, 1, saved_text=saved(first))

    first, same = _resumed(lambda first: first.output.text)
    assert any(e.result is TraceResult.PASS for e in first.checks), "premise"
    assert [e for e in same.checks if e.result is TraceResult.PASS] == [
        e for e in first.checks if e.result is TraceResult.PASS
    ]
    _, demoted = _resumed(lambda first: "[page 1 failed: malformed table emission (x)]")
    assert not [e for e in demoted.checks if e.result is TraceResult.PASS]
    _, captioned = _resumed(lambda first: first.output.text + "\n\nFigure 1: shown")
    assert not [e for e in captioned.checks if e.result is TraceResult.PASS]


def test_a_finding_the_verdict_lacks_is_kept_by_the_live_seam_as_by_the_offline_tools() -> None:
    """A check that found something the verdict does not carry (a published verdict, a patched
    check) keeps its slot as the finding it is; live and offline state the same trace."""
    p = _p(own_layer_word_reference="korea china imports")
    live = _trace(_out(LAYER), p, PageVerdict(1))
    assert live[("text_layer_words", "page")].result is TraceResult.FAIL
    offline = {
        (e["check"], e["scope"]): e
        for e in reconcile_pass_trace(
            trace_to_dicts(()),
            PageVerdict(1),
            recomputed=rerun_text_dependent_checks(_out(LAYER), p),
            text=LAYER,
        )
    }
    assert offline[("text_layer_words", "page")]["result"] == "FAIL"
