"""GH-353: the table-judge ladder's owned contract.

One place holds the verdict schema, the S1/S2 vocabulary, the rung calling
convention, and the audit-event kinds every other ladder module (A2/A3
rungs, A4 state machine, B0 witnesses, B1 gate, B2 trust) imports instead of
re-deriving.

Two successes per rung (design: ``docs/log/2026-08-30_table-judge-ladder.md``):

- **S1 — the judge answered.** The process ran, strict JSON parsed, in time.
  ¬S1 = timeout, crash, garbage output, quota exhaustion, unparseable JSON,
  or a schema violation (missing field / unknown finding code). ¬S1 is
  represented here by ``TableVerdictParseError`` / ``RungResult.ok is False``
  — never by a synthesized FAIL verdict, because a substitute rung must see
  fresh eyes, not a verdict nobody actually produced.
- **S2 — the judge approved.** ``TableJudgeVerdict.verdict == "PASS"``. Only
  meaningful when S1 holds.

Fenced JSON (```json ... ```) is NOT an S1 failure — the gemini CLI fences
routinely; ``parse_table_verdict`` strips it via the same balanced-brace
extraction ``socr.judge.judge`` already uses for the page-level judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from socr.judge.judge import _extract_json

# --------------------------------------------------------------------------
# Verdict schema
# --------------------------------------------------------------------------

VALID_VERDICTS = ("PASS", "FAIL")
VALID_CONFIDENCE = ("high", "low")


class FindingCode(str, Enum):
    """Closed enum — one value per measured failure family.

    Deliberately closed (not free text): a new failure family is a design
    decision (new code + prompt wording), not something a judge should be
    able to invent on the fly.
    """

    MISSING_VALUE = "MISSING_VALUE"
    FABRICATED_VALUE = "FABRICATED_VALUE"
    WRONG_BINDING = "WRONG_BINDING"
    HEADER_MANGLED = "HEADER_MANGLED"
    STRUCTURE_MERGED = "STRUCTURE_MERGED"
    NOT_A_TABLE = "NOT_A_TABLE"


@dataclass
class Finding:
    """One judge-reported defect.

    ``where`` is a cell/row/col reference in the OUTPUT's coordinates (the
    markdown the judge was shown, not the crop); ``detail`` is one evidence
    sentence.
    """

    code: FindingCode
    where: str = ""
    detail: str = ""


class TableVerdictParseError(ValueError):
    """Raised when a rung's raw output cannot be trusted as a verdict.

    This IS the S1-failure signal — every case that reaches here (empty
    output, non-JSON, missing required field, unknown finding code) means
    the judge did not answer, not that it answered FAIL.
    """


@dataclass
class TableJudgeVerdict:
    """A single-table faithfulness verdict from one ladder rung.

    ``findings`` is empty iff ``verdict == "PASS"`` — enforced by
    ``parse_table_verdict``, not merely documented, so a malformed PASS (or
    a FAIL with no evidence) is treated as ¬S1 rather than silently
    accepted.
    """

    verdict: str  # "PASS" | "FAIL"
    confidence: str  # "high" | "low"
    findings: list[Finding] = field(default_factory=list)
    raw: str = ""  # raw rung output, kept for the audit journal / debugging

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    @property
    def is_confident_pass(self) -> bool:
        return self.verdict == "PASS" and self.confidence == "high"


def parse_table_verdict(text: str) -> TableJudgeVerdict:
    """Strict-parse one rung's raw output into a ``TableJudgeVerdict``.

    Raises ``TableVerdictParseError`` for anything that is not a trustworthy
    verdict: empty output, non-JSON, a non-object JSON value, a missing or
    invalid ``verdict``/``confidence``, a malformed ``findings`` entry, an
    unknown finding ``code``, or a PASS/FAIL whose findings list disagrees
    with the empty-iff-PASS rule. Every one of these IS the S1 failure —
    callers must not fall back to treating it as FAIL.
    """
    if not text or not text.strip():
        raise TableVerdictParseError("empty rung output")

    try:
        data = _extract_json(text)
    except ValueError as exc:
        raise TableVerdictParseError(f"no JSON object found: {exc}") from exc

    if not isinstance(data, dict):
        raise TableVerdictParseError(f"parsed JSON is not an object: {text[:200]!r}")

    verdict = data.get("verdict")
    if verdict not in VALID_VERDICTS:
        raise TableVerdictParseError(f"missing/invalid 'verdict': {verdict!r}")

    confidence = data.get("confidence")
    if confidence not in VALID_CONFIDENCE:
        raise TableVerdictParseError(f"missing/invalid 'confidence': {confidence!r}")

    raw_findings = data.get("findings")
    if raw_findings is None:
        raw_findings = []
    if not isinstance(raw_findings, list):
        raise TableVerdictParseError(f"'findings' is not a list: {raw_findings!r}")

    findings: list[Finding] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            raise TableVerdictParseError(f"finding is not an object: {item!r}")
        code_raw = item.get("code")
        try:
            code = FindingCode(code_raw)
        except ValueError as exc:
            raise TableVerdictParseError(f"unknown finding code: {code_raw!r}") from exc
        findings.append(
            Finding(
                code=code,
                where=str(item.get("where", "")),
                detail=str(item.get("detail", "")),
            )
        )

    if verdict == "PASS" and findings:
        raise TableVerdictParseError("PASS verdict must carry empty findings")
    if verdict == "FAIL" and not findings:
        raise TableVerdictParseError("FAIL verdict must carry at least one finding")

    return TableJudgeVerdict(verdict=verdict, confidence=confidence, findings=findings, raw=text)


# --------------------------------------------------------------------------
# Rung contract
# --------------------------------------------------------------------------


@dataclass
class RungResult:
    """One rung's answer for one table: the S1 outcome plus the payload.

    ``ok`` is the S1 bit (the judge answered). ``verdict`` is populated iff
    ``ok`` is True. ``error`` carries the ¬S1 reason (parse failure,
    timeout, transport error) for the audit trail — never a fabricated
    verdict.
    """

    rung: str  # rung identifier, e.g. "ollama:glm-5.3-flash:cloud" or "gemini"
    ok: bool
    verdict: TableJudgeVerdict | None = None
    latency_sec: float = 0.0
    error: str = ""


class RungCallable(Protocol):
    """Anything that judges one table crop and returns a ``RungResult``.

    ``prior_findings`` remains on the signature for call-site compatibility.
    GH-359 ruling 4: the ladder always passes ``None``. Judge input is crop
    + markdown, nothing else — a B-escalation does not carry the complaint.
    """

    def __call__(
        self,
        crop_path: Path,
        markdown: str,
        prior_findings: list[Finding] | None,
    ) -> RungResult: ...


def rung_result_from_output(rung: str, text: str, latency_sec: float) -> RungResult:
    """Classify one rung's raw stdout/response body into a ``RungResult``.

    The shared S1 classification point for rung implementations (A2/A3):
    parse failure of any kind becomes ``ok=False`` with the reason preserved
    in ``error``, never a synthesized FAIL.
    """
    try:
        verdict = parse_table_verdict(text)
    except TableVerdictParseError as exc:
        return RungResult(
            rung=rung, ok=False, verdict=None, latency_sec=latency_sec, error=str(exc)
        )
    return RungResult(rung=rung, ok=True, verdict=verdict, latency_sec=latency_sec)


# --------------------------------------------------------------------------
# Audit-event kinds
# --------------------------------------------------------------------------
#
# Per-table events the gate (B1) emits and B2 registers in
# ``TABLE_DISTRUST_KINDS`` / ``RESOLVING_KINDS``. Three kinds — one per
# ladder outcome that a downstream consumer must be able to tell apart:
# accepted (resolving), rejected (content problem, distrust), unverified
# (infra problem, distrust). Intermediate per-rung steps (tiebreak,
# substitute) are not separate event kinds; they are ``RungResult`` values
# a caller can log, not terminal dispositions a consumer reads.

TABLE_LADDER_ACCEPTED_KIND = "table_ladder_accepted"
TABLE_LADDER_REJECTED_KIND = "table_ladder_rejected"
TABLE_LADDER_UNVERIFIED_KIND = "table_ladder_unverified"
#: GH-367: supporting evidence for a clamp lift/hold. Not a fourth
#: terminal — the three kinds above remain the only content outcomes.
#: Kept out of ``TABLE_LADDER_EVENT_KINDS`` so the GH-359 drift guard
#: (exactly those three terminals) stays about terminals.
TABLE_BINDING_ADJUDICATED_KIND = "table_binding_adjudicated"

#: All audit-event kinds this module defines, for callers that want to
#: sanity-check membership without importing each constant by name.
TABLE_LADDER_EVENT_KINDS: frozenset[str] = frozenset(
    {
        TABLE_LADDER_ACCEPTED_KIND,
        TABLE_LADDER_REJECTED_KIND,
        TABLE_LADDER_UNVERIFIED_KIND,
    }
)
