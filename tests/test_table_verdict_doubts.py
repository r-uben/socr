"""P1 (task t1): ``TableJudgeVerdict.doubts`` -- the value-free canonical
cell list a LOW-confidence PASS carries so the tiebreak chain (Q1) knows
which cells to check without the reader ever stating what it doubted the
CONTENT to be.

Rules pinned here (owner ruling, docs/log/2026-09-02_gh359-ladder-terminals-design.md,
"Owner rulings on the ladder flip", Q1(b)):

- A LOW-confidence PASS MUST carry at least one doubt (empty doubts on a low
  PASS is unusable -- the tiebreak chain would have nothing to check).
- A HIGH-confidence PASS MUST NOT carry doubts (nothing to check on a
  confident answer; doubts are explicitly a low-confidence signal).
- Every doubt uses the canonical ``RxCy`` / ``HxCy`` grammar -- never a
  value, never free text.
- FAIL verdicts are unaffected: ``findings`` stays the FAIL vocabulary,
  ``doubts`` stays empty for FAIL. The existing "findings empty iff PASS"
  rule is untouched.
- Old JSON (no ``doubts`` key at all) remains valid for HIGH PASS and FAIL --
  this is an additive, backward-compatible field. A LOW PASS with no key at
  all is the new-schema violation this ticket introduces: it must fail
  because there is nothing to check, not because the key was renamed.
"""

from __future__ import annotations

import json

import pytest

from socr.judge.table_verdict import (
    TableJudgeVerdict,
    TableVerdictParseError,
    parse_table_verdict,
)


def _payload(**overrides) -> dict:
    base = {"verdict": "PASS", "confidence": "high", "findings": []}
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Backward compatibility: existing JSON shapes are untouched
# --------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_high_confidence_pass_with_no_doubts_key_still_parses(self) -> None:
        verdict = parse_table_verdict(json.dumps(_payload()))
        assert verdict.verdict == "PASS"
        assert verdict.doubts == []

    def test_fail_with_no_doubts_key_still_parses(self) -> None:
        payload = _payload(
            verdict="FAIL",
            confidence="low",
            findings=[{"code": "MISSING_VALUE", "where": "row 1", "detail": "gone"}],
        )
        verdict = parse_table_verdict(json.dumps(payload))
        assert verdict.verdict == "FAIL"
        assert verdict.doubts == []


# --------------------------------------------------------------------------
# New schema: LOW PASS requires doubts
# --------------------------------------------------------------------------


class TestLowPassRequiresDoubts:
    def test_low_pass_with_no_doubts_key_is_s1_failure(self) -> None:
        payload = _payload(confidence="low")
        with pytest.raises(TableVerdictParseError):
            parse_table_verdict(json.dumps(payload))

    def test_low_pass_with_empty_doubts_list_is_s1_failure(self) -> None:
        payload = _payload(confidence="low", doubts=[])
        with pytest.raises(TableVerdictParseError):
            parse_table_verdict(json.dumps(payload))

    def test_low_pass_with_one_valid_doubt_parses(self) -> None:
        payload = _payload(confidence="low", doubts=["R2C3"])
        verdict = parse_table_verdict(json.dumps(payload))
        assert verdict.verdict == "PASS"
        assert verdict.confidence == "low"
        assert verdict.doubts == ["R2C3"]

    def test_low_pass_with_several_doubts_parses_in_order(self) -> None:
        payload = _payload(confidence="low", doubts=["R2C3", "H1C1", "R4C2"])
        verdict = parse_table_verdict(json.dumps(payload))
        assert verdict.doubts == ["R2C3", "H1C1", "R4C2"]

    @pytest.mark.parametrize(
        "bad_doubt",
        ["", "row 2 col 3", "the revenue figure looks off", "r2c3", "R2C3: 1,204"],
    )
    def test_low_pass_with_a_malformed_doubt_is_s1_failure(self, bad_doubt: str) -> None:
        payload = _payload(confidence="low", doubts=[bad_doubt])
        with pytest.raises(TableVerdictParseError):
            parse_table_verdict(json.dumps(payload))

    def test_low_pass_doubts_must_be_a_list(self) -> None:
        payload = _payload(confidence="low", doubts="R2C3")
        with pytest.raises(TableVerdictParseError):
            parse_table_verdict(json.dumps(payload))


# --------------------------------------------------------------------------
# New schema: HIGH PASS forbids doubts
# --------------------------------------------------------------------------


class TestHighPassForbidsDoubts:
    def test_high_pass_with_doubts_is_s1_failure(self) -> None:
        payload = _payload(confidence="high", doubts=["R2C3"])
        with pytest.raises(TableVerdictParseError):
            parse_table_verdict(json.dumps(payload))

    def test_high_pass_with_explicit_empty_doubts_list_is_fine(self) -> None:
        payload = _payload(confidence="high", doubts=[])
        verdict = parse_table_verdict(json.dumps(payload))
        assert verdict.doubts == []


# --------------------------------------------------------------------------
# FAIL verdicts never carry doubts
# --------------------------------------------------------------------------


class TestFailForbidsDoubts:
    def test_fail_with_doubts_is_s1_failure(self) -> None:
        payload = _payload(
            verdict="FAIL",
            confidence="low",
            findings=[{"code": "MISSING_VALUE", "where": "row 1", "detail": "gone"}],
            doubts=["R2C3"],
        )
        with pytest.raises(TableVerdictParseError):
            parse_table_verdict(json.dumps(payload))


# --------------------------------------------------------------------------
# Dataclass defaults
# --------------------------------------------------------------------------


def test_table_judge_verdict_doubts_defaults_to_empty_list() -> None:
    verdict = TableJudgeVerdict(verdict="PASS", confidence="high")
    assert verdict.doubts == []


def test_findings_empty_iff_pass_rule_is_unaffected_by_doubts() -> None:
    payload = _payload(confidence="low", doubts=["R2C3"], findings=[{"code": "MISSING_VALUE"}])
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict(json.dumps(payload))
