"""GH-353 TICKET-A1: verdict schema, rung contract, event kinds."""

from __future__ import annotations

import json

import pytest

from socr.judge.table_verdict import (
    TABLE_LADDER_ACCEPTED_KIND,
    TABLE_LADDER_EVENT_KINDS,
    TABLE_LADDER_REJECTED_KIND,
    TABLE_LADDER_UNVERIFIED_KIND,
    Finding,
    FindingCode,
    RungResult,
    TableJudgeVerdict,
    TableVerdictParseError,
    parse_table_verdict,
    rung_result_from_output,
)

PASS_JSON = json.dumps({"verdict": "PASS", "confidence": "high", "findings": []})

FAIL_JSON = json.dumps(
    {
        "verdict": "FAIL",
        "confidence": "low",
        "findings": [
            {
                "code": "MISSING_VALUE",
                "where": "row 3, col Revenue",
                "detail": "12,450 present in the crop, absent from the output",
            }
        ],
    }
)


def test_exact_json_pass_parses():
    verdict = parse_table_verdict(PASS_JSON)
    assert verdict.verdict == "PASS"
    assert verdict.confidence == "high"
    assert verdict.findings == []
    assert verdict.passed is True
    assert verdict.is_confident_pass is True
    assert verdict.raw == PASS_JSON


def test_exact_json_fail_parses():
    verdict = parse_table_verdict(FAIL_JSON)
    assert verdict.verdict == "FAIL"
    assert verdict.passed is False
    assert len(verdict.findings) == 1
    finding = verdict.findings[0]
    assert finding.code == FindingCode.MISSING_VALUE
    assert finding.where == "row 3, col Revenue"
    assert "12,450" in finding.detail


def test_fenced_json_is_accepted_not_s1_failure():
    """The gemini CLI fences routinely — fenced JSON must parse, not ¬S1."""
    fenced = f"```json\n{PASS_JSON}\n```"
    verdict = parse_table_verdict(fenced)
    assert verdict.verdict == "PASS"
    assert verdict.findings == []


def test_fenced_json_fail_is_accepted():
    fenced = f"```json\n{FAIL_JSON}\n```"
    verdict = parse_table_verdict(fenced)
    assert verdict.verdict == "FAIL"
    assert verdict.findings[0].code == FindingCode.MISSING_VALUE


def test_prose_wrapped_json_parses():
    prose = f"Here is my assessment of the table:\n\n{PASS_JSON}\n\nLet me know if you need more."
    verdict = parse_table_verdict(prose)
    assert verdict.verdict == "PASS"


def test_pass_requires_empty_findings():
    bad = json.dumps(
        {
            "verdict": "PASS",
            "confidence": "high",
            "findings": [{"code": "MISSING_VALUE", "where": "x", "detail": "y"}],
        }
    )
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict(bad)


def test_fail_requires_nonempty_findings():
    bad = json.dumps({"verdict": "FAIL", "confidence": "low", "findings": []})
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict(bad)


def test_missing_verdict_field_is_s1_failure():
    bad = json.dumps({"confidence": "high", "findings": []})
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict(bad)


def test_invalid_verdict_value_is_s1_failure():
    bad = json.dumps({"verdict": "MAYBE", "confidence": "high", "findings": []})
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict(bad)


def test_missing_confidence_field_is_s1_failure():
    bad = json.dumps({"verdict": "PASS", "findings": []})
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict(bad)


def test_unknown_finding_code_is_s1_failure():
    bad = json.dumps(
        {
            "verdict": "FAIL",
            "confidence": "low",
            "findings": [{"code": "TOTALLY_MADE_UP", "where": "x", "detail": "y"}],
        }
    )
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict(bad)


def test_findings_entry_not_an_object_is_s1_failure():
    bad = json.dumps({"verdict": "FAIL", "confidence": "low", "findings": ["MISSING_VALUE"]})
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict(bad)


def test_findings_not_a_list_is_s1_failure():
    bad = json.dumps({"verdict": "FAIL", "confidence": "low", "findings": "MISSING_VALUE"})
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict(bad)


def test_non_json_text_is_s1_failure():
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict("The table looks fine to me, no JSON here.")


def test_empty_output_is_s1_failure():
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict("")


def test_whitespace_only_output_is_s1_failure():
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict("   \n\t  ")


def test_json_array_instead_of_object_is_s1_failure():
    with pytest.raises(TableVerdictParseError):
        parse_table_verdict(json.dumps(["PASS"]))


@pytest.mark.parametrize("code", list(FindingCode))
def test_all_six_finding_codes_are_closed_and_parseable(code):
    payload = json.dumps(
        {
            "verdict": "FAIL",
            "confidence": "low",
            "findings": [{"code": code.value, "where": "somewhere", "detail": "some evidence"}],
        }
    )
    verdict = parse_table_verdict(payload)
    assert verdict.findings[0].code == code


def test_finding_code_enum_has_exactly_six_members():
    assert {c.value for c in FindingCode} == {
        "MISSING_VALUE",
        "FABRICATED_VALUE",
        "WRONG_BINDING",
        "HEADER_MANGLED",
        "STRUCTURE_MERGED",
        "NOT_A_TABLE",
    }


def test_rung_result_from_output_ok_on_valid_json():
    result = rung_result_from_output("ollama:glm-5.3-flash:cloud", PASS_JSON, latency_sec=1.5)
    assert isinstance(result, RungResult)
    assert result.rung == "ollama:glm-5.3-flash:cloud"
    assert result.ok is True
    assert result.verdict is not None
    assert result.verdict.verdict == "PASS"
    assert result.latency_sec == 1.5
    assert result.error == ""


def test_rung_result_from_output_not_ok_on_garbage():
    result = rung_result_from_output("gemini", "not json at all", latency_sec=0.2)
    assert result.ok is False
    assert result.verdict is None
    assert result.error != ""


def test_rung_result_from_output_not_ok_on_empty_string():
    result = rung_result_from_output("gemini", "", latency_sec=0.0)
    assert result.ok is False
    assert result.verdict is None


def test_rung_callable_protocol_shape():
    """A plain function matching the signature satisfies the protocol duck-type."""

    def fake_rung(crop_path, markdown, prior_findings):
        return RungResult(rung="fake", ok=True, verdict=parse_table_verdict(PASS_JSON))

    from pathlib import Path

    result = fake_rung(Path("/tmp/crop.png"), "| a |\n|---|\n| 1 |", None)
    assert result.ok is True
    assert result.rung == "fake"


def test_prior_findings_flow_into_tiebreak_call():
    prior = [Finding(code=FindingCode.MISSING_VALUE, where="row 1", detail="missing")]

    captured = {}

    def fake_rung(crop_path, markdown, prior_findings):
        captured["prior_findings"] = prior_findings
        return RungResult(rung="fake", ok=True, verdict=parse_table_verdict(FAIL_JSON))

    from pathlib import Path

    fake_rung(Path("/tmp/crop.png"), "markdown", prior)
    assert captured["prior_findings"] == prior


def test_table_judge_verdict_dataclass_defaults():
    verdict = TableJudgeVerdict(verdict="PASS", confidence="high")
    assert verdict.findings == []
    assert verdict.raw == ""


def test_audit_event_kinds_are_distinct_strings():
    kinds = {TABLE_LADDER_ACCEPTED_KIND, TABLE_LADDER_REJECTED_KIND, TABLE_LADDER_UNVERIFIED_KIND}
    assert len(kinds) == 3
    assert all(isinstance(k, str) for k in kinds)
    assert kinds == TABLE_LADDER_EVENT_KINDS
