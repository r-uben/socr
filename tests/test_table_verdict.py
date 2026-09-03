"""GH-353 TICKET-A1: verdict schema, rung contract, event kinds."""

from __future__ import annotations

import errno
import json
import subprocess

import httpx
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
    assert result.unavailable is False


def test_rung_result_from_output_not_ok_on_garbage():
    result = rung_result_from_output("gemini", "not json at all", latency_sec=0.2)
    assert result.ok is False
    assert result.verdict is None
    assert result.error != ""
    assert result.unavailable is False


def test_rung_result_from_output_not_ok_on_empty_string():
    result = rung_result_from_output("gemini", "", latency_sec=0.0)
    assert result.ok is False
    assert result.verdict is None
    assert result.unavailable is False


def test_rung_result_dataclass_defaults():
    res = RungResult(rung="test", ok=True)
    assert res.verdict is None
    assert res.latency_sec == 0.0
    assert res.error == ""
    assert res.unavailable is False


def test_rung_result_from_output_schema_invalid_is_not_unavailable():
    # Schema errors (missing field, bad enum) are answered-but-unusable S1 failures,
    # not transport/reachability failures -- unavailable must stay False.
    bad_schema = json.dumps({"verdict": "UNKNOWN", "confidence": "high", "findings": []})
    result = rung_result_from_output("gemini", bad_schema, latency_sec=0.1)
    assert result.ok is False
    assert result.verdict is None
    assert "missing/invalid 'verdict'" in result.error
    assert result.unavailable is False


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


# ---------------------------------------------------------------------------
# Cold review round 3, finding 2: ONE shared classification table, every row
# pinned. "Outage" means an external condition that can be restored WITHOUT
# changing this code or this call; anything the next identical call would hit
# again is a defect. Every row below still ends the table TABLE_UNVERIFIED --
# only the retry latch differs.
# ---------------------------------------------------------------------------


class TestSpawnErrnoClassification:
    @pytest.mark.parametrize(
        "err,outage",
        [
            (errno.ENOENT, True),  # binary absent -- install it
            (errno.ENOEXEC, True),  # not an executable image -- rebuild it
            (errno.EACCES, True),  # may not execute it -- chmod it
            (errno.EPERM, True),
            (errno.E2BIG, False),  # argv too large -- this call, every time
            (errno.EPIPE, False),
            (errno.EINVAL, False),
            (errno.EISDIR, False),
        ],
        ids=lambda v: errno.errorcode.get(v, str(v)) if isinstance(v, int) else str(v),
    )
    def test_spawn_errno_rows(self, err: int, outage: bool) -> None:
        from socr.judge.table_verdict import classify_spawn_oserror

        assert classify_spawn_oserror(OSError(err, "x")) is outage

    def test_an_errno_free_oserror_is_not_an_outage(self) -> None:
        from socr.judge.table_verdict import classify_spawn_oserror

        assert classify_spawn_oserror(OSError("no errno at all")) is False


class TestHttpStatusClassification:
    @pytest.mark.parametrize(
        "status,outage",
        [
            (401, True),  # credentials can be restored; until then it never works
            (403, True),
            (407, True),  # proxy auth
            (408, True),
            (429, True),
            (500, True),
            (502, True),
            (503, True),
            (599, True),  # top of the server-error range
            (400, False),  # our payload is wrong
            (405, False),
            (409, False),
            (422, False),
            (200, False),
            # Cold review round 4: ">= 500" also swept in codes no HTTP server
            # issues, which a broken proxy or a test double can produce.
            (600, False),
            (999, False),
        ],
    )
    def test_status_rows(self, status: int, outage: bool) -> None:
        from socr.judge.table_verdict import classify_http_status

        assert classify_http_status(status) is outage

    @pytest.mark.parametrize(
        "body,outage",
        [
            # The daemon's actual missing-model shapes.
            ('{"error":"model \'judge\' not found, try pulling it first"}', True),
            ('{"error":"model judge:latest not found"}', True),
            ("model not found, try pulling it first", True),
            # Cold review round 4: a wrong ROUTE whose path happens to contain
            # the word. A bare "model" substring read this as a missing model
            # and latched a defect that is identical forever.
            ('{"error":"route /api/model-info not found; use /api/chat"}', False),
            ("404 page not found", False),
            ("", False),
            # The word alone, with nothing saying anything was not found.
            ('{"error":"model parameter required"}', False),
        ],
    )
    def test_404_is_an_outage_only_when_the_body_says_the_model_is_missing(
        self, body: str, outage: bool
    ) -> None:
        """The one status that needs the body: ollama 404s both for a model
        that was never pulled (pull it and the identical call works) and for a
        route that does not exist (our own defect, forever)."""
        from socr.judge.table_verdict import classify_http_status

        assert classify_http_status(404, body) is outage

    def test_the_daemons_error_field_is_what_is_read(self) -> None:
        """Reading the documented ``error`` field rather than the whole payload
        keeps an unrelated string elsewhere in the document from deciding it."""
        from socr.judge.table_verdict import classify_http_status

        body = '{"error":"route not found","hint":"model \'x\' not found"}'
        assert classify_http_status(404, body) is False


class TestRefusalMarkers:
    @pytest.mark.parametrize(
        "text",
        [
            "Error: quota exceeded for this project",
            "RESOURCE_EXHAUSTED: rate limit reached",
            "429 Too Many Requests",
            "request was unauthorized",
            "authentication failed",
            "IneligibleTierError: migrate to Antigravity",
            "503 Service Unavailable, try again later",
            "connection refused",
        ],
    )
    def test_recognised_refusals(self, text: str) -> None:
        from socr.judge.table_verdict import output_reads_as_refusal

        assert output_reads_as_refusal(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "unknown flag: --nope",
            "usage: agy [options]",
            # Cold review round 4: a flag whose NAME contains a marker word.
            # The bare "quota" marker latched this deterministic usage error
            # and tripped the run breaker on it.
            "unknown option --quota-project; see usage",
            "error: --rate-limit-config must be a path",
            "unauthorized_client is not a valid value for --mode",
            # Cold review round 3: loose single-word markers used to make these
            # read as outages, which is the expensive direction of the mistake.
            "could not open /home/u/.config: permission denied",
            "please run 'agy login' first to configure this workspace",
            "error reading local file: connection.json is malformed",
            "",
        ],
    )
    def test_deterministic_local_errors_are_not_refusals(self, text: str) -> None:
        from socr.judge.table_verdict import output_reads_as_refusal

        assert output_reads_as_refusal(text) is False

    def test_classification_is_not_limited_to_the_audit_excerpt(self) -> None:
        """The 500-char excerpt is the AUDIT TRAIL's. Classifying on it threw
        away any refusal printed past that cutoff -- a false negative that
        settles the table permanently."""
        from socr.judge.table_verdict import CLASSIFY_CAPTURE_CHARS, output_reads_as_refusal

        assert CLASSIFY_CAPTURE_CHARS > 500
        assert output_reads_as_refusal("x" * 3000 + " quota exceeded") is True

    def test_both_streams_are_read(self) -> None:
        """A CLI may print its refusal on stdout; reading only stderr is the
        same false negative one step smaller."""
        from socr.judge.table_verdict import output_reads_as_refusal

        assert output_reads_as_refusal("", "quota exceeded") is True


class TestAvailabilityExceptions:
    @pytest.mark.parametrize(
        "exc,outage",
        [
            (httpx.ConnectError("refused"), True),
            (httpx.ReadTimeout("slow"), True),
            (httpx.PoolTimeout("busy"), True),
            (httpx.ProxyError("proxy"), True),
            (ConnectionError("reset"), True),
            (ConnectionResetError("reset"), True),
            (TimeoutError("waited"), True),
            (subprocess.TimeoutExpired(["agy"], 1.0), True),
            # Client configuration: the URL names a scheme httpx cannot speak.
            # Identical forever, so retrying is pure cost.
            (httpx.UnsupportedProtocol("gopher://"), False),
            (httpx.DecodingError("bad gzip"), False),
            (httpx.TooManyRedirects("loop"), False),
            (TypeError("unsupported operand"), False),
            (AssertionError("invariant"), False),
            (KeyError("verdict"), False),
            (ValueError("bad literal"), False),
            (RuntimeError("crashed"), False),
            # A local file problem is a deterministic local defect; spawn errors
            # go through classify_spawn_oserror instead, which reads the errno.
            (FileNotFoundError("crop.png"), False),
            (OSError("broken pipe"), False),
        ],
        ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else str(v),
    )
    def test_exception_rows(self, exc: BaseException, outage: bool) -> None:
        from socr.judge.table_verdict import is_availability_exception

        assert is_availability_exception(exc) is outage


class TestRungKind:
    @pytest.mark.parametrize(
        "rung_id,kind",
        [
            ("ollama:glm-5.3-flash:cloud", "ollama"),
            ("ollama", "ollama"),
            ("gemini", "gemini"),
            ("unknown", "unknown"),
            ("", ""),
        ],
    )
    def test_kind_rows(self, rung_id: str, kind: str) -> None:
        from socr.judge.table_verdict import rung_kind

        assert rung_kind(rung_id) == kind
