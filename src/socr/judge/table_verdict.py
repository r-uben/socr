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

import errno
import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import httpx

from socr.judge.judge import _extract_json

# --------------------------------------------------------------------------
# Cell reference grammar and resolver
# --------------------------------------------------------------------------

_CELL_REF_RE = re.compile(r"^([RH])([1-9]\d*)C([1-9]\d*)$")

#: The ONE definition of the coordinate contract, as policy data rather than
#: prose duplicated per prompt (cold review round 2, N1).
#:
#: It used to be written out twice, and the two copies disagreed about which
#: physical column ``C1`` is. Splicing one file into every prompt that mentions
#: a cell is what makes that class of drift impossible rather than merely
#: fixed. ``tests/test_cell_ref_grammar.py`` closes the loop by checking the
#: worked examples against ``resolve_cell_refs``.
#:
#: The RULE and the EXAMPLES are two files on purpose (cold review round 4).
#: A worked example has to show cells with contents in them, and the
#: blind-transcription prompt may not contain cell contents AT ALL: a
#: value-bearing example is an answer key handed to a reader whose only value
#: is that it has seen nothing but the image. The rule below carries no cell
#: contents and goes to both prompts; the examples go to the reader prompt
#: only, and even there every cell holds its own reference rather than a
#: plausible value.
_CELL_REF_GRAMMAR_PATH = Path(__file__).resolve().parent.parent / "prompts" / "cell_ref_grammar.md"
_CELL_REF_EXAMPLES_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "cell_ref_examples.md"
)


def load_cell_ref_examples() -> str:
    """The worked examples for the coordinate rule, verbatim.

    READER PROMPT ONLY. Never spliced into the blind-transcription prompt --
    see ``load_cell_ref_grammar``'s note and
    ``tests/test_cell_ref_grammar.py::TestTheBlindPromptCarriesNoAnswers``.
    """
    return _CELL_REF_EXAMPLES_PATH.read_text(encoding="utf-8").strip()


def load_cell_ref_grammar() -> str:
    """The canonical cell-reference RULE, verbatim. Safe for every prompt.

    Carries no cell contents of any kind, so it can be shown to a blind
    reader without telling it what anyone expects to be there.
    """
    return _CELL_REF_GRAMMAR_PATH.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class CellRef:
    """A canonical, value-free coordinate for a single table cell.

    1-indexed to match human/judge descriptions ("row 2, column 3").
    Can refer to a body cell (``R2C3``) or a header cell (``H1C2``).
    """

    row: int
    col: int
    is_header: bool = False

    def __str__(self) -> str:
        prefix = "H" if self.is_header else "R"
        return f"{prefix}{self.row}C{self.col}"


def parse_cell_ref(text: str) -> CellRef:
    """Parse a canonical cell reference string into a ``CellRef``.

    Canonical grammar:
      - Body cell: ``R<row>C<col>`` with 1-indexed integers (e.g. ``R2C3``).
      - Header cell: ``H<row>C<col>`` with 1-indexed integers (e.g. ``H1C2``).

    Raises ``ValueError`` on any non-canonical, malformed, or 0-indexed string.
    """
    if not isinstance(text, str):
        raise ValueError(f"cell ref must be a string: {text!r}")
    m = _CELL_REF_RE.match(text)
    if not m:
        raise ValueError(f"malformed cell reference: {text!r}")
    kind, row_str, col_str = m.groups()
    return CellRef(row=int(row_str), col=int(col_str), is_header=(kind == "H"))


def resolve_cell_refs(
    markdown: str,
    refs: Sequence[CellRef | str],
) -> dict[CellRef, str] | None:
    """Resolve a collection of cell references against an emitted markdown table.

    Uses ``socr.tables.binding.parse_grid`` to extract the table grid.
    If the markdown does not parse as a table grid, or if ANY reference is
    malformed, missing, out of range, or invalid, returns ``None`` (fails closed;
    the whole set is unresolved).

    Returns a mapping from ``CellRef`` to the raw cell token string in the grid.
    If ``refs`` is empty and the markdown contains a valid grid, returns ``{}``.
    """
    if not isinstance(markdown, str):
        return None
    from socr.tables.binding import parse_grid

    grid = parse_grid(markdown)
    if grid is None:
        return None
    if not refs:
        return {}

    resolved: dict[CellRef, str] = {}
    for item in refs:
        if isinstance(item, CellRef):
            ref = item
        elif isinstance(item, str):
            try:
                ref = parse_cell_ref(item)
            except ValueError:
                return None
        else:
            return None

        # Cold review round 3, NEW A: ``C<k>`` is the k-th PHYSICAL column,
        # counting from the leftmost, for header and body rows alike. Header
        # refs used to be offset by one, on the assumption that the first
        # column is always a name column that headings do not number. Nothing
        # in this repo detects such a column -- ``parse_grid`` accepts any
        # equal-width table of two or more columns, and a table whose first
        # column holds values is explicitly supported -- so on such a table the
        # prompts described a column that does not exist and every header
        # coordinate named the wrong physical cell. A correct blind reading of
        # one heading could then be compared against its neighbour and withhold
        # a correct table.
        row_cells = (
            grid.header_rows[ref.row - 1]
            if ref.is_header and 1 <= ref.row <= len(grid.header_rows)
            else grid.rows[ref.row - 1]
            if not ref.is_header and 1 <= ref.row <= len(grid.rows)
            else None
        )
        if row_cells is None:
            return None
        if not (1 <= ref.col <= len(row_cells)):
            return None
        resolved[ref] = row_cells[ref.col - 1]

    return resolved


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
    doubts: list[str] = field(default_factory=list)
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
    unknown finding ``code``, a PASS/FAIL whose findings list disagrees
    with the empty-iff-PASS rule, or invalid ``doubts`` for the verdict/confidence.
    Every one of these IS the S1 failure — callers must not fall back to treating
    it as FAIL.
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

    raw_doubts = data.get("doubts")
    doubts: list[str] = []
    if raw_doubts is not None:
        if not isinstance(raw_doubts, list):
            raise TableVerdictParseError(f"'doubts' is not a list: {raw_doubts!r}")
        if verdict == "PASS" and confidence == "high" and raw_doubts:
            raise TableVerdictParseError("high-confidence PASS must not carry doubts")
        if verdict == "FAIL" and raw_doubts:
            raise TableVerdictParseError("FAIL verdict must not carry doubts")
        if verdict == "PASS" and confidence == "low":
            if not raw_doubts:
                raise TableVerdictParseError("low-confidence PASS must carry at least one doubt")
            for item in raw_doubts:
                if not isinstance(item, str):
                    raise TableVerdictParseError(f"doubt is not a string: {item!r}")
                try:
                    parse_cell_ref(item)
                except ValueError as exc:
                    raise TableVerdictParseError(
                        f"malformed doubt cell reference: {item!r}"
                    ) from exc
                doubts.append(item)
    else:
        if verdict == "PASS" and confidence == "low":
            raise TableVerdictParseError("low-confidence PASS must carry at least one doubt")

    return TableJudgeVerdict(
        verdict=verdict,
        confidence=confidence,
        findings=findings,
        doubts=doubts,
        raw=text,
    )


# --------------------------------------------------------------------------
# Rung contract
# --------------------------------------------------------------------------


@dataclass
class RungResult:
    """One rung's answer for one table: the S1 outcome plus the payload.

    ``ok`` is the S1 bit (the judge answered). ``verdict`` is populated iff
    ``ok`` is True. ``error`` carries the ¬S1 reason (parse failure,
    timeout, transport error) for the audit trail — never a fabricated
    verdict. ``unavailable`` is True when the configured rung could not be
    attempted or communicated with due to transient/external availability.
    """

    rung: str  # rung identifier, e.g. "ollama:glm-5.3-flash:cloud" or "gemini"
    ok: bool
    verdict: TableJudgeVerdict | None = None
    latency_sec: float = 0.0
    error: str = ""
    unavailable: bool = False
    #: Cold review round 3. A recognised EXTERNAL refusal on the real call --
    #: quota, rate limit, revoked credentials, the service saying no. Always
    #: implies ``unavailable``; the extra bit is what lets the gate trip a
    #: per-run circuit breaker for this rung, so one refusal does not get
    #: re-paid by every remaining table in the run.
    refusal: bool = False


# --------------------------------------------------------------------------
# Availability classification -- ONE table, shared by both rungs, the ladder's
# per-rung guard and the gate's whole-ladder guard.
#
# Cold review rounds 2 and 3. Every branch below still ends the table
# TABLE_UNVERIFIED; the only thing this decides is whether the outcome carries
# the retry latch (and which rung it names). The two ways to get it wrong are
# not symmetric but both are real:
#
#   * a FALSE NEGATIVE permanently settles a table nobody ever judged -- the
#     rung comes back and the page is never re-read;
#   * a FALSE POSITIVE re-judges forever -- every resume re-pays timeout x
#     tables x rungs to reproduce a failure that is not going to change.
#
# So "outage" means specifically: an external condition that can plausibly be
# restored WITHOUT changing this code or this call. Anything the next
# identical call would hit again is a defect.
# --------------------------------------------------------------------------

#: Rung KINDS. The latch records which kind was unavailable, so the resume
#: gate can ask about that rung rather than about "some rung somewhere"
#: (cold review round 3, finding 1). ``RungResult.rung`` carries the kind as
#: its prefix: "gemini", or "ollama:<model>".
RUNG_KIND_OLLAMA = "ollama"
RUNG_KIND_GEMINI = "gemini"
#: P1: the blind cell-transcription ADJUDICATOR's kind. Deliberately NOT a
#: reader kind -- it is kept out of ``TABLE_JUDGE_RUNG_KINDS`` so that the
#: adjudicator being reachable can never satisfy "some reader can look at
#: this table". It IS latchable and probeable in its own right: an
#: adjudicator outage is a reason to retry the page later, just like a
#: reader outage, and recovery must be asked about THIS kind (cold review
#: round 1, finding 2) rather than widened to the readers.
#:
#: Named for the ROLE, not the vendor. The adjudicator rides the ollama
#: transport (cold review round 1, finding 4) but is a different model from
#: a different vendor than reader rung 1, so it needs an identity of its own:
#: ``"ollama"`` would make the two indistinguishable in the latch, and a
#: vendor name would have to be rewritten on the next model swap.
RUNG_KIND_CELL_ADJUDICATOR = "adjudicator"


def rung_kind(rung_id: str) -> str:
    """The rung KIND from a ``RungResult.rung`` identifier.

    ``"ollama:glm-5.3-flash:cloud"`` -> ``"ollama"``; ``"gemini"`` ->
    ``"gemini"``. An identifier from neither family (a synthesized
    ``"unknown"``, or an injected test rung) returns itself, which the gate
    treats as an unrecognised kind rather than silently mapping it onto a
    real rung's reachability.
    """
    return (rung_id or "").split(":", 1)[0]


#: Errnos that mean the CLI could not be spawned because of the ENVIRONMENT:
#: it is absent, it is not an executable image, or this process may not run
#: it. Installing, fixing or chmod-ing the binary makes the identical call
#: work, so these are outages. Every other spawn errno describes THIS call --
#: E2BIG (argv too large for the kernel) above all, which an oversized prompt
#: reproduces on every retry (cold review round 3).
_SPAWN_OUTAGE_ERRNOS: frozenset[int] = frozenset(
    {errno.ENOENT, errno.ENOEXEC, errno.EACCES, errno.EPERM}
)


def classify_spawn_oserror(exc: OSError) -> bool:
    """Whether an ``OSError`` raised while SPAWNING a CLI rung is an outage."""
    return exc.errno in _SPAWN_OUTAGE_ERRNOS


#: HTTP statuses that describe the SERVICE as unusable rather than the
#: request as wrong. 401/403/407 are included deliberately: credentials, a
#: revoked token or a proxy can be restored, and until they are the rung
#: cannot succeed -- which is exactly the state the latch exists to remember.
#: 408/429 are transient by definition.
_UNAVAILABLE_STATUS_CODES: frozenset[int] = frozenset({401, 403, 407, 408, 429})

#: Server-error range, 500-599 inclusive. Cold review round 4: ">= 500" also
#: swept in 600+, which is not a status code any HTTP server issues and which a
#: broken proxy or a test double can produce.
_SERVER_ERROR_FLOOR = 500
_SERVER_ERROR_CEILING = 599

#: 404 is ambiguous and is the one status that needs the body: ollama returns
#: it for a model that was never pulled (an outage -- pulling it fixes the
#: identical call, and ``ollama_rung_reachable`` refuses for the same reason)
#: and also for a route that does not exist (a defect in this code, forever).
#:
#: Cold review round 4: a bare ``"model"`` substring conflated the two. A
#: wrong-route body such as ``route /api/model-info not found`` contains the
#: word and was read as a missing model. Match the daemon's ACTUAL error shape
#: instead -- ollama emits ``model '<name>' not found`` (optionally followed by
#: a pull hint) -- and require whitespace after ``model``, which is what
#: separates it from ``model-info``.
_MODEL_MISSING_RE = re.compile(
    r"model\s+(?:[\"']?[\w:.\-/]+[\"']?\s+)?(?:was\s+)?not\s+found",
    re.IGNORECASE,
)
_MODEL_PULL_HINT_RE = re.compile(r"try\s+pulling|pull\s+the\s+model", re.IGNORECASE)


def _error_field(body: str) -> str:
    """The daemon's own ``error`` field, when the body is the JSON it documents.

    Ollama's API returns ``{"error": "..."}``. Reading the field rather than
    the whole document keeps an unrelated string elsewhere in the payload from
    deciding the classification. Falls back to the raw body when it is not that
    shape, so a plain-text error is still classified.
    """
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return body or ""
    if isinstance(parsed, dict):
        value = parsed.get("error")
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("message"), str):
            return value["message"]
        return ""
    return body or ""


def classify_http_status(status_code: int, body: str = "") -> bool:
    """Whether an HTTP status from a rung call is an outage."""
    if _SERVER_ERROR_FLOOR <= status_code <= _SERVER_ERROR_CEILING:
        return True
    if status_code in _UNAVAILABLE_STATUS_CODES:
        return True
    if status_code == 404:
        message = _error_field(body or "")
        return bool(_MODEL_MISSING_RE.search(message) or _MODEL_PULL_HINT_RE.search(message))
    return False


#: Phrases that identify an EXTERNAL refusal from a CLI that ran fine.
#:
#: Cold review round 4: these are WHOLE PHRASES the CLIs actually emit, matched
#: as phrases. The previous list carried the bare marker ``quota`` under
#: unrestricted substring matching, so ``unknown option --quota-project`` -- an
#: ordinary deterministic usage error -- classified as an external refusal,
#: latched, and tripped the run breaker. A single word is never enough
#: evidence that a service refused us; the sentence it appears in is.
_REFUSAL_PHRASES: tuple[str, ...] = (
    "quota exceeded",
    "quota exhausted",
    "insufficient quota",
    "out of quota",
    "exceeded your current quota",
    "quota limit reached",
    "rate limit exceeded",
    "rate limit reached",
    "rate_limit_exceeded",
    "resource exhausted",
    "resource_exhausted",
    "too many requests",
    "unauthenticated",
    "unauthorized",
    "authentication failed",
    "invalid api key",
    "api key expired",
    "token expired",
    # The exact error class the gemini CLI raised when Google retired the free
    # individual tier (docs/log/2026-08-30_gh353-ticket-a3.md). Spelled in full:
    # the non-word guards mean a prefix would not match the real message.
    "ineligibletiererror",
    "service unavailable",
    "temporarily unavailable",
    "try again later",
    "connection refused",
    "connection reset",
    "could not connect",
    "network is unreachable",
    "timed out",
)

#: Each phrase is matched with non-word guards on both ends, so a phrase is
#: never recognised inside a longer token (``--quota-project``,
#: ``unauthorized_client``).
_REFUSAL_RE = re.compile(
    "|".join(rf"(?<!\w){re.escape(phrase)}(?!\w)" for phrase in _REFUSAL_PHRASES),
    re.IGNORECASE,
)

#: How much captured output is scanned for a refusal phrase. Classification
#: must NOT be limited to the short excerpt kept for the audit trail -- a quota
#: message printed after that cutoff was being discarded before it was ever
#: classified. Bounded so a runaway CLI cannot make this scan unbounded work.
CLASSIFY_CAPTURE_CHARS = 20_000


def output_reads_as_refusal(*streams: str) -> bool:
    """Whether any captured CLI stream names an external refusal.

    Both stdout and stderr are scanned: a CLI is free to print its refusal on
    either, and reading only one is a false negative that settles the table
    forever.
    """
    return any(
        _REFUSAL_RE.search((stream or "")[:CLASSIFY_CAPTURE_CHARS]) is not None
        for stream in streams
    )


#: Exception types that mean the rung could not be REACHED. A rung is
#: contractually non-raising, so anything escaping one is unexpected -- but
#: the cause still separates a transient outage from a software defect.
#:
#: ``httpx.UnsupportedProtocol`` is carved out of ``TransportError``: it means
#: the URL we built names a scheme httpx cannot speak, which is our own
#: configuration and will fail identically forever (cold review round 3).
#:
#: Bare ``OSError`` is NOT in the set -- a missing or unreadable local crop is
#: a deterministic local defect, and spawn errors go through
#: ``classify_spawn_oserror`` instead, which reads the errno.
#: ``AssertionError``, ``TypeError``, ``KeyError``, ``ValueError``,
#: ``httpx.DecodingError`` and ``httpx.TooManyRedirects`` are defects.
_AVAILABILITY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    ConnectionError,
    TimeoutError,
    subprocess.TimeoutExpired,
)
_AVAILABILITY_EXCEPTION_CARVE_OUTS: tuple[type[BaseException], ...] = (httpx.UnsupportedProtocol,)


def is_availability_exception(exc: BaseException) -> bool:
    """Whether ``exc`` means the rung was unreachable, not that it misbehaved.

    The single reference used by every place that has to turn an exception
    into ``RungResult.unavailable`` -- the ladder's per-rung guard and the
    gate's whole-ladder guard -- so the latch cannot mean one thing in one
    file and another thing in the next.
    """
    if isinstance(exc, _AVAILABILITY_EXCEPTION_CARVE_OUTS):
        return False
    if isinstance(exc, OSError) and not isinstance(exc, (ConnectionError, TimeoutError)):
        return False
    return isinstance(exc, _AVAILABILITY_EXCEPTIONS)


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
    in ``error``, never a synthesized FAIL. ``unavailable`` remains False
    for all output parsing outcomes (valid, empty, malformed, or schema-invalid).
    """
    try:
        verdict = parse_table_verdict(text)
    except TableVerdictParseError as exc:
        return RungResult(
            rung=rung,
            ok=False,
            verdict=None,
            latency_sec=latency_sec,
            error=str(exc),
            unavailable=False,
        )
    return RungResult(
        rung=rung, ok=True, verdict=verdict, latency_sec=latency_sec, unavailable=False
    )


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
#: P1 (owner ruling Q2): the FOURTH terminal. A table two readers rejected
#: and neither guard could clear ships NO table bytes -- the region is
#: replaced by the failed-table marker plus the page image. Distinct from
#: ``table_ladder_rejected`` on purpose: rejected shipped the text demoted
#: under a warning, withheld ships none of it, and conflating them would
#: make every historical rejected page look like a withhold on replay.
TABLE_LADDER_WITHHELD_KIND = "table_ladder_withheld"
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
        # P1: widened DELIBERATELY from GH-359's three terminals to four. The
        # drift guard exists so a terminal is never added by accident; this one
        # is added by owner ruling Q2 and is recorded in
        # docs/log/2026-09-03_p1-ladder-flip.md.
        TABLE_LADDER_WITHHELD_KIND,
    }
)

#: The two acceptance reasons the guard chain can produce, recorded on the
#: ``table_ladder_accepted`` event's ``data["reason"]``. An ordinary ladder
#: acceptance carries no reason value at all -- these name the two ways the
#: READERS were overruled, which is a different fact and must be legible as one.
REASON_VERIFIED_BY_GEOMETRY = "verified_by_geometry"
REASON_VERIFIED_BY_BLIND_CELL_TRANSCRIPTION = "verified_by_blind_cell_transcription"
