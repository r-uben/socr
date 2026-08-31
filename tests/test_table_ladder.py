"""GH-353 TICKET-A4: ladder state machine + page reducer transition table."""

from __future__ import annotations

from pathlib import Path

import pytest

from socr.judge.table_ladder import (
    PageLadderResult,
    TableLadderOutcome,
    TableLadderResult,
    reduce_page_ladder,
    run_table_ladder,
)
from socr.judge.table_verdict import Finding, FindingCode, RungResult, TableJudgeVerdict

CROP = Path("/tmp/crop-0.png")
MARKDOWN = "| a | b |\n|---|---|\n| 1 | 2 |"

FAIL_FINDINGS = [Finding(code=FindingCode.MISSING_VALUE, where="row 1", detail="12,450 missing")]


def _pass(rung: str, confidence: str) -> RungResult:
    return RungResult(
        rung=rung,
        ok=True,
        verdict=TableJudgeVerdict(verdict="PASS", confidence=confidence, findings=[]),
    )


def _fail(rung: str, findings: list[Finding] | None = None) -> RungResult:
    return RungResult(
        rung=rung,
        ok=True,
        verdict=TableJudgeVerdict(
            verdict="FAIL", confidence="low", findings=findings or FAIL_FINDINGS
        ),
    )


def _not_s1(rung: str, error: str = "timeout") -> RungResult:
    return RungResult(rung=rung, ok=False, verdict=None, error=error)


def _rung(result: RungResult, captured: list | None = None):
    """Build a rung callable that returns ``result`` and records the call."""

    def _callable(crop_path, markdown, prior_findings):
        if captured is not None:
            captured.append((crop_path, markdown, prior_findings))
        return result

    return _callable


def _rung_sequence(*results: RungResult, captured: list | None = None):
    """Build a rung callable returning successive results, one per call."""
    remaining = list(results)

    def _callable(crop_path, markdown, prior_findings):
        if captured is not None:
            captured.append((crop_path, markdown, prior_findings))
        return remaining.pop(0)

    return _callable


# --------------------------------------------------------------------------
# Single-rung outcomes
# --------------------------------------------------------------------------


def test_a_high_confidence_accepts_immediately_single_rung():
    calls: list = []
    rung1 = _rung(_pass("rung1", "high"), calls)

    result = run_table_ladder([rung1], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.ACCEPTED
    assert result.final_verdict.verdict == "PASS"
    assert len(result.rung_results) == 1
    assert len(calls) == 1
    # No prior findings on the first call.
    assert calls[0][2] is None


def test_a_high_confidence_accepts_without_calling_second_rung():
    calls: list = []
    rung1 = _rung(_pass("rung1", "high"), calls)
    rung2_calls: list = []
    rung2 = _rung(_pass("rung2", "high"), rung2_calls)

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.ACCEPTED
    assert len(result.rung_results) == 1
    assert rung2_calls == []


# --------------------------------------------------------------------------
# A-low confirmation transitions
# --------------------------------------------------------------------------


def test_a_low_then_a_confirms_and_accepts():
    calls: list = []
    rung1 = _rung_sequence(_pass("rung1", "low"), captured=calls)
    rung2 = _rung_sequence(_pass("rung2", "high"), captured=calls)

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.ACCEPTED
    assert len(result.rung_results) == 2
    # Confirm escalation carries no findings (nothing to complain about).
    assert calls[1][2] is None


def test_a_low_then_b_rejects():
    rung1 = _rung(_pass("rung1", "low"))
    rung2 = _rung(_fail("rung2"))

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.REJECTED
    assert result.final_verdict.verdict == "FAIL"
    assert len(result.rung_results) == 2


def test_a_low_then_not_s1_unverifies():
    """Symmetric pin: a real low-confidence PASS witness followed by a ¬S1 substitute
    still exhausts to UNVERIFIED — the failed last rung leaves no corroboration."""
    rung1 = _rung(_pass("rung1", "low"))
    rung2 = _rung(_not_s1("rung2"))

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.UNVERIFIED
    assert result.final_verdict is None
    assert len(result.rung_results) == 2


def test_lone_low_confidence_pass_with_no_corroboration_is_unverified():
    """GH-359 ruling 1: last-rung PASS+low with no preceding PASS is UNVERIFIED."""
    rung1 = _rung(_pass("rung1", "low"))

    result = run_table_ladder([rung1], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.UNVERIFIED
    assert result.final_verdict is None


def test_not_s1_then_low_confidence_pass_at_last_rung_is_unverified():
    """A substituted ¬S1 provides no corroboration: the following low-confidence PASS
    is still a lone witness and must not silently accept."""
    rung1 = _rung(_not_s1("rung1"))
    rung2 = _rung(_pass("rung2", "low"))

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.UNVERIFIED
    assert result.final_verdict is None
    assert len(result.rung_results) == 2


def test_low_confidence_pass_then_low_confidence_pass_accepts():
    """Two real low-confidence PASS witnesses in agreement are sufficient
    corroboration and accept."""
    rung1 = _rung(_pass("rung1", "low"))
    rung2 = _rung(_pass("rung2", "low"))

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.ACCEPTED
    assert result.final_verdict.confidence == "low"


# --------------------------------------------------------------------------
# B (FAIL / tiebreak) transitions
# --------------------------------------------------------------------------


def test_b_then_a_low_unverifies():
    """GH-359 rulings 1+2: CLI₁ FAIL + CLI₂ PASS+low is UNVERIFIED, not accept."""
    rung1 = _rung(_fail("rung1"))
    rung2 = _rung(_pass("rung2", "low"))

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.UNVERIFIED
    assert result.final_verdict is None


def test_not_a_table_fail_at_last_rung_is_rejected():
    """GH-359 ruling 7: NOT_A_TABLE is a content FAIL → REJECTED, not a reroute."""
    findings = [Finding(code=FindingCode.NOT_A_TABLE, where="crop", detail="this is a chart")]
    rung1 = _rung(_fail("rung1", findings))

    result = run_table_ladder([rung1], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.REJECTED
    assert result.final_verdict.findings[0].code is FindingCode.NOT_A_TABLE


def test_b_then_a_accepts():
    calls: list = []
    rung1 = _rung_sequence(_fail("rung1"), captured=calls)
    rung2 = _rung_sequence(_pass("rung2", "high"), captured=calls)

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.ACCEPTED
    assert result.final_verdict.verdict == "PASS"
    # GH-359 ruling 4: B-escalation does not carry findings.
    assert calls[1][2] is None


def test_b_then_b_rejects_without_forwarding_findings():
    calls: list = []
    rung1 = _rung_sequence(_fail("rung1", FAIL_FINDINGS), captured=calls)
    second_findings = [Finding(code=FindingCode.WRONG_BINDING, where="row 2", detail="shifted")]
    rung2 = _rung_sequence(_fail("rung2", second_findings), captured=calls)

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.REJECTED
    assert result.final_verdict.findings == second_findings
    # GH-359 ruling 4: crop + markdown, nothing else.
    assert calls[1][2] is None


def test_b_then_not_s1_unverifies():
    """GH-359 ruling 3: mixed B then C is UNVERIFIED, not REJECTED."""
    rung1 = _rung(_fail("rung1"))
    rung2 = _rung(_not_s1("rung2"))

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.UNVERIFIED
    assert result.final_verdict is None


# --------------------------------------------------------------------------
# C (¬S1 / substitute) transitions
# --------------------------------------------------------------------------


def test_c_then_a_accepts():
    calls: list = []
    rung1 = _rung_sequence(_not_s1("rung1"), captured=calls)
    rung2 = _rung_sequence(_pass("rung2", "high"), captured=calls)

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.ACCEPTED
    # Substitute escalation carries NO findings — fresh eyes.
    assert calls[1][2] is None


def test_c_then_b_rejects():
    rung1 = _rung(_not_s1("rung1"))
    rung2 = _rung(_fail("rung2"))

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.REJECTED
    assert result.final_verdict.verdict == "FAIL"


def test_c_then_not_s1_exhausts_to_unverified():
    rung1 = _rung(_not_s1("rung1", error="timeout"))
    rung2 = _rung(_not_s1("rung2", error="connection refused"))

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert result.outcome is TableLadderOutcome.UNVERIFIED
    assert result.final_verdict is None
    assert result.rung_results[0].error == "timeout"
    assert result.rung_results[1].error == "connection refused"


# --------------------------------------------------------------------------
# Contract checks
# --------------------------------------------------------------------------


def test_empty_rung_sequence_raises():
    with pytest.raises(ValueError):
        run_table_ladder([], CROP, MARKDOWN, table_id="t0")


def test_crop_path_and_markdown_are_passed_through_unchanged():
    calls: list = []
    rung1 = _rung(_pass("rung1", "high"), calls)

    run_table_ladder([rung1], CROP, MARKDOWN, table_id="t0")

    assert calls[0][0] == CROP
    assert calls[0][1] == MARKDOWN


def test_table_id_is_preserved_on_the_result():
    rung1 = _rung(_pass("rung1", "high"))

    result = run_table_ladder([rung1], CROP, MARKDOWN, table_id="table-3")

    assert result.table_id == "table-3"


def test_all_rung_results_kept_in_order_regardless_of_outcome():
    rung1 = _rung(_not_s1("rung1"))
    rung2 = _rung(_fail("rung2"))

    result = run_table_ladder([rung1, rung2], CROP, MARKDOWN, table_id="t0")

    assert [r.rung for r in result.rung_results] == ["rung1", "rung2"]


# --------------------------------------------------------------------------
# Page reducer
# --------------------------------------------------------------------------


def _table_result(outcome: TableLadderOutcome, table_id: str = "t") -> TableLadderResult:
    return TableLadderResult(table_id=table_id, outcome=outcome, rung_results=[])


def test_page_reducer_all_accepted_is_accepted():
    tables = [
        _table_result(TableLadderOutcome.ACCEPTED, "t0"),
        _table_result(TableLadderOutcome.ACCEPTED, "t1"),
    ]

    result = reduce_page_ladder(tables)

    assert isinstance(result, PageLadderResult)
    assert result.outcome is TableLadderOutcome.ACCEPTED
    assert result.table_results == tables


def test_page_reducer_any_rejected_wins_over_unverified():
    tables = [
        _table_result(TableLadderOutcome.UNVERIFIED, "t0"),
        _table_result(TableLadderOutcome.REJECTED, "t1"),
        _table_result(TableLadderOutcome.ACCEPTED, "t2"),
    ]

    result = reduce_page_ladder(tables)

    assert result.outcome is TableLadderOutcome.REJECTED
    # Every per-table result is kept, not just the deciding one.
    assert len(result.table_results) == 3


def test_page_reducer_one_pass_one_fail_is_rejected():
    """Mixed page: one table accepted, one table exhausted to REJECTED."""
    tables = [
        _table_result(TableLadderOutcome.ACCEPTED, "t0"),
        _table_result(TableLadderOutcome.REJECTED, "t1"),
    ]

    result = reduce_page_ladder(tables)

    assert result.outcome is TableLadderOutcome.REJECTED
    assert {t.outcome for t in result.table_results} == {
        TableLadderOutcome.ACCEPTED,
        TableLadderOutcome.REJECTED,
    }


def test_page_reducer_one_pass_one_not_s1_is_unverified():
    """Mixed page: one table accepted, one table exhausted to UNVERIFIED."""
    tables = [
        _table_result(TableLadderOutcome.ACCEPTED, "t0"),
        _table_result(TableLadderOutcome.UNVERIFIED, "t1"),
    ]

    result = reduce_page_ladder(tables)

    assert result.outcome is TableLadderOutcome.UNVERIFIED
    assert {t.outcome for t in result.table_results} == {
        TableLadderOutcome.ACCEPTED,
        TableLadderOutcome.UNVERIFIED,
    }


def test_page_reducer_no_tables_is_accepted():
    result = reduce_page_ladder([])

    assert result.outcome is TableLadderOutcome.ACCEPTED
    assert result.table_results == []


def test_page_reducer_end_to_end_from_real_ladder_runs():
    """Drive the reducer off actual run_table_ladder outputs, not fixtures."""
    accepted_rung = _rung(_pass("rung1", "high"))
    rejected_rung1 = _rung_sequence(_fail("r1"))
    rejected_rung2 = _rung_sequence(_fail("r2"))

    accepted = run_table_ladder([accepted_rung], CROP, MARKDOWN, table_id="t0")
    rejected = run_table_ladder([rejected_rung1, rejected_rung2], CROP, MARKDOWN, table_id="t1")

    result = reduce_page_ladder([accepted, rejected])

    assert result.outcome is TableLadderOutcome.REJECTED
    assert result.table_results[0].table_id == "t0"
    assert result.table_results[1].table_id == "t1"
