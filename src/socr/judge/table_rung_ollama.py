"""GH-353 TICKET-A2: CLI1 rung — the ollama-cloud table judge, and (P1) the
blind cell-transcription adjudicator that rides the same transport.

Builds an `A1` `RungCallable` bound to one ollama model/host, POSTing
`/api/chat` (NOT `/api/generate` — the GH-356 bake-off's integration path;
`ollama run --images` does not exist on ollama 0.32.15) with the table crop
as a base64 image, `format="json"`, `stream=False`.

The network call is isolated behind `_post_chat`, the module's one transport
seam: tests monkeypatch THIS function, never `httpx` globally, so the exact
outgoing JSON payload (built by `_build_payload`, kept separate so it can be
asserted on without a real request) stays independently verifiable.

Every failure mode this module can produce — a missing/unreadable crop file,
timeout, connection error, HTTP error status, or a verdict that fails strict
parsing — becomes `RungResult(ok=False)` (¬S1, `socr.judge.table_verdict`).
It never raises and never synthesizes a FAIL verdict; a substitute rung must
see fresh eyes, not a verdict nobody actually produced.
"""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import httpx

from socr.judge.table_prompt import build_table_judge_prompt
from socr.judge.table_verdict import (
    RUNG_KIND_CELL_ADJUDICATOR,
    RUNG_KIND_OLLAMA,
    Finding,
    RungResult,
    classify_http_status,
    is_availability_exception,
    output_reads_as_refusal,
    rung_result_from_output,
)
from socr.tables.extract import resolve_ollama_host

#: Statuses where the service itself refused us, as opposed to being down.
#: These trip the gate's per-run circuit breaker: the next table in the same
#: run will be refused identically, so paying for it is pure waste.
_REFUSAL_STATUS_CODES: frozenset[int] = frozenset({401, 403, 407, 429})


def _response_text(response: httpx.Response) -> str:
    """The response body as text, or "" if it cannot be read.

    Only ever used to CLASSIFY, so a body that cannot be decoded must not
    raise out of the classification path.
    """
    try:
        return response.text or ""
    except Exception:  # a body we cannot read simply carries no evidence
        return ""


def _with_implicit_tag(name: str) -> str:
    """Ollama reports and accepts a bare name as ``name:latest``."""
    return name if ":" in name else f"{name}:latest"


def ollama_rung_reachable(model: str, host: str | None, timeout: float = 5.0) -> bool:
    """Whether rung 1 could be attempted right now: daemon up AND model pulled.

    Cold review round 2, finding 1. The resume gate and this rung must share
    ONE notion of reachability, or the gate reopens a document that the rung
    will immediately re-latch. A bare ``/api/tags`` liveness ping is not that
    notion: a healthy daemon that has never pulled the judge model answers it
    happily and then 404s on every judge call, forever.

    Matched on the full ``name:tag`` for the same reason ``OllamaVisionJudge.
    is_available`` does (#133): a prefix match let an installed
    ``qwen3-vl:30b-a3b-instruct`` satisfy a request for ``qwen3-vl:8b``.

    Cheap by construction -- one GET, no generation, no model load.
    """
    resolved = resolve_ollama_host(host)
    try:
        resp = httpx.get(f"{resolved.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
        names = {_with_implicit_tag(m.get("name", "")) for m in resp.json().get("models", [])}
    except (httpx.HTTPError, OSError, ValueError) as exc:
        logger.debug("table judge rung 1 unreachable at %s: %s", resolved, exc)
        return False
    return _with_implicit_tag(model) in names


#: Rung identifier prefix (`RungResult.rung`, e.g. "ollama:glm-5.3-flash:cloud"),
#: matching the example in `socr.judge.table_verdict.RungResult`'s docstring.
_RUNG_PREFIX = RUNG_KIND_OLLAMA

logger = logging.getLogger(__name__)


def _rung_id(model: str) -> str:
    return f"{_RUNG_PREFIX}:{model}"


def _build_payload(model: str, prompt: str, image_b64: str) -> dict[str, Any]:
    """The exact `/api/chat` request body — kept separate from the network
    call so tests can assert on it without touching `httpx`."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "format": "json",
        "stream": False,
    }


def _post_chat(host: str, payload: dict[str, Any], timeout: float) -> str:
    """The sole network seam. Returns the assistant message content verbatim.

    Any transport failure (timeout, connection error, non-2xx status) raises
    `httpx.HTTPError` — the caller classifies it as ¬S1; this function never
    swallows it into an empty string, which would be indistinguishable from
    a genuinely empty judge response.

    Cold review round 3, NEW C: the body is TYPE-CHECKED here, at the trust
    boundary. Every caller's contract is "never raises", and both of them
    reach for string methods on what comes back; a daemon (or a proxy in
    front of one) answering ``{"message": {"content": 1}}`` would otherwise
    escape as an ``AttributeError`` from inside a function that promises a
    typed failure result. A body of the wrong shape is a DEFECT, so it is
    raised as the ``ValueError`` both callers already classify as one rather
    than being coerced into a string that would then fail to parse for a
    misleading reason.
    """
    resp = httpx.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        raise ValueError(f"ollama response is not a JSON object: {type(body).__name__}")
    message = body.get("message", {})
    if not isinstance(message, dict):
        raise ValueError(f"ollama response 'message' is not an object: {type(message).__name__}")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ValueError(
            f"ollama response 'message.content' is not a string: {type(content).__name__}"
        )
    return content


def build_ollama_rung(model: str, host: str | None, timeout: float):
    """Build an A1 `RungCallable` bound to one ollama model/host/timeout.

    `host` resolves through `resolve_ollama_host` (explicit value, then
    `OLLAMA_HOST`, then localhost) exactly once, at construction time — the
    same rule every other ollama backend caller in this repo follows (GH-222).
    Constructed once by the gate (B1) and injected, so tests never need a
    live daemon.
    """
    resolved_host = resolve_ollama_host(host)
    rung = _rung_id(model)

    def _judge(
        crop_path: Path,
        markdown: str,
        prior_findings: list[Finding] | None,
    ) -> RungResult:
        start = time.monotonic()
        try:
            prompt = build_table_judge_prompt(
                markdown,
                [
                    {"code": finding.code.value, "where": finding.where, "detail": finding.detail}
                    for finding in (prior_findings or [])
                ],
            )
            image_b64 = base64.b64encode(Path(crop_path).read_bytes()).decode("ascii")
            payload = _build_payload(model, prompt, image_b64)
        except OSError as exc:
            latency_sec = time.monotonic() - start
            return RungResult(
                rung=rung,
                ok=False,
                latency_sec=latency_sec,
                error=f"{type(exc).__name__}: {exc}",
                unavailable=False,
            )

        try:
            text = _post_chat(resolved_host, payload, timeout)
        except httpx.HTTPStatusError as exc:
            # The daemon ANSWERED, so the status is the evidence. Classified by
            # the shared table (``classify_http_status``), which reads the body
            # for the one ambiguous status, 404.
            latency_sec = time.monotonic() - start
            body = _response_text(exc.response)
            unavailable = classify_http_status(exc.response.status_code, body)
            return RungResult(
                rung=rung,
                ok=False,
                latency_sec=latency_sec,
                error=f"{type(exc).__name__}: {exc}",
                unavailable=unavailable,
                # A refusal the SERVICE issued (credentials, quota, rate limit)
                # rather than the service being down: worth a per-run breaker.
                refusal=unavailable
                and (
                    exc.response.status_code in _REFUSAL_STATUS_CODES
                    or output_reads_as_refusal(body)
                ),
            )
        except (httpx.HTTPError, OSError) as exc:
            # No response at all. ``is_availability_exception`` is the same
            # reference the ladder guard uses, so a DecodingError or a
            # TooManyRedirects is a defect here too, not an outage.
            latency_sec = time.monotonic() - start
            return RungResult(
                rung=rung,
                ok=False,
                latency_sec=latency_sec,
                error=f"{type(exc).__name__}: {exc}",
                unavailable=is_availability_exception(exc),
            )
        except ValueError as exc:
            # A 200 whose BODY is the wrong shape (round 3, NEW C). The rung's
            # contract is ¬S1, never an exception, and this reproduces on every
            # call, so it is a defect and does not latch.
            latency_sec = time.monotonic() - start
            return RungResult(
                rung=rung,
                ok=False,
                latency_sec=latency_sec,
                error=f"{type(exc).__name__}: {exc}",
                unavailable=False,
            )
        latency_sec = time.monotonic() - start
        return rung_result_from_output(rung, text, latency_sec)

    # Cold review round 5: the callable advertises its own identity. The gate
    # receives opaque callables, so without this the only way back to "which
    # rung is this" was the position of its RESULT in a list -- which stopped
    # being the configured ladder position once the breaker began filtering,
    # and which cannot identify a rung that was rebuilt for the next document
    # in the same batch.
    _judge.rung_kind = RUNG_KIND_OLLAMA
    _judge.rung_id = rung
    _judge.executing = model
    return _judge


# ----------------------------------------------------------------------
# P1 (owner rulings Q1/Q2): the blind cell-transcription ADJUDICATOR.
#
# Third vendor, and deliberately NOT a third reader rung. ``run_table_ladder``
# never calls this: it produces no PASS/FAIL verdict, carries no confidence,
# and cannot accept or reject a table on its own. It answers one question --
# "what token is printed in these cells?" -- from the crop alone, and the gate
# compares its answer to the extraction.
#
# **Blindness is the whole point.** The public API takes a crop path and a
# list of canonical cell references and nothing else -- no emitted markdown,
# no reader verdict, no finding detail, no extraction token. If it saw the
# value it is meant to corroborate, its agreement would be worth nothing.
#
# **Why it lives HERE, on the ollama transport (cold review round 1,
# finding 4).** The first build reached Kimi through ``cursor-agent -p`` with
# the crop named as ``@<path>`` inside the prompt text. That transport was
# never proven, and a live probe disproved it: on a synthetic crop whose cell
# next to "Gamma" reads ``0.058``, ``cursor-agent -p --model kimi-k3-max``
# answered ``0.92`` -- it never saw the image. The same crop POSTed to the
# local ollama daemon (``images`` on the request body) was read correctly by
# ``kimi-k2.6:cloud`` (``0.058``) and by ``glm-5.3-flash:cloud`` (``0.058``).
# A text-only handoff returning schema-valid tokens is the WORST failure
# shape here, because a guessable token (``0``, ``N/A``) can normalize equal
# to the extraction and clear a table nobody looked at. So the adjudicator
# uses the transport that is proven to carry pixels -- the same one reader
# rung 1 uses -- with a different vendor's model for independence, and
# inherits this module's identity tagging, typed unavailability
# classification, health probe and refusal signal unchanged.
# ----------------------------------------------------------------------

#: The adjudicator's prompt. Values never appear in it; see the file.
_CELLS_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "table_cells_transcribe.md"
)


def load_table_cells_transcribe_prompt() -> str:
    return _CELLS_PROMPT_PATH.read_text(encoding="utf-8")


@dataclass
class BlindCellResult:
    """One adjudicator call's answer for one table.

    Deliberately NOT a ``RungResult``. A ``RungResult`` with ``ok=True`` and
    ``verdict=None`` would violate that type's own contract (ok implies a
    verdict), and every consumer of ``RungResult`` reads it as a reader's
    opinion. This carries tokens, not an opinion.

    ``unavailable`` / ``refusal`` mean exactly what they mean on
    ``RungResult``: the call could not be made or was externally refused, so
    the page latches and a later run retries. ``ok=False`` with
    ``unavailable=False`` is a deterministic defect that must never latch.
    """

    rung: str
    ok: bool
    tokens: dict[str, str] = dataclass_field(default_factory=dict)
    #: References the blind reader reported it could NOT read (wire ``null``),
    #: as distinct from cells it read and found blank (wire ``""``). Cold
    #: review round 2, N2: one token for both states meant "I did not look"
    #: was indistinguishable from "I looked and it was empty", so a
    #: no-reading could WITHHELD a rejected table against a non-empty
    #: extraction, or CLEAR one against an empty extraction. Neither is a
    #: reading, so neither may do either. These refs carry no token at all.
    unreadable: tuple[str, ...] = ()
    latency_sec: float = 0.0
    error: str = ""
    unavailable: bool = False
    refusal: bool = False


def adjudicator_rung_id(model: str) -> str:
    """The adjudicator's identity: kind plus the CONFIGURED model.

    Cold review round 1, finding 5: never a module constant. The journal must
    name what actually ran, or a non-default model produces a truthful
    fingerprint over a false provider record.
    """
    return f"{RUNG_KIND_CELL_ADJUDICATOR}:{model}"


def adjudicator_rung_reachable(model: str, host: str | None, timeout: float = 5.0) -> bool:
    """Whether the adjudicator could be attempted right now.

    Its own probe, on its own model (cold review round 1, finding 2): daemon
    up AND *this* model pulled, matched on the full ``name:tag``. A reader
    model being present says nothing about the adjudicator's model, so the
    two questions never share an answer.
    """
    return ollama_rung_reachable(model, host, timeout=timeout)


#: The construction boundary between the two parts of the blind prompt.
#:
#: Cold review round 6. The blind prompt is POLICY plus a generated REQUEST
#: LIST, and the invariant that keeps it blind is structural rather than
#: lexical: the policy may contain no concrete coordinate at all, and concrete
#: coordinates may appear only in the request list, which carries coordinates
#: and nothing else. A binding of a coordinate to a value -- in ANY spelling,
#: numeric or not, prose or JSON or a markdown row -- has to name a concrete
#: coordinate, so it cannot exist without breaking the first half of that.
#:
#: Rounds 4 and 5 each closed one spelling of the same leak and each left the
#: next one open, because the guards recognised syntax. This heading is what
#: lets a guard stop recognising syntax: it is the seam the prompt AS SENT can
#: be split at, so the property can be asserted on what actually went out.
REQUEST_LIST_HEADING = "Cells to transcribe:"


def build_blind_cell_prompt_parts(cell_refs: Sequence[str]) -> tuple[str, str]:
    """The blind prompt as its two parts: ``(policy, request_list)``.

    ``policy`` is the template plus the spliced coordinate rule -- fixed text,
    identical for every table, and containing no concrete coordinate.
    ``request_list`` is generated from ``cell_refs`` and contains the canonical
    references and nothing else: no value, no markdown, no finding detail.
    """
    from socr.judge.table_verdict import load_cell_ref_grammar

    policy = load_table_cells_transcribe_prompt().replace(
        "{{CELL_REF_GRAMMAR}}", load_cell_ref_grammar()
    )
    wanted = ", ".join(str(ref) for ref in cell_refs)
    return policy, f"{REQUEST_LIST_HEADING} {wanted}\n"


def build_blind_cell_prompt(cell_refs: Sequence[str]) -> str:
    """The policy prompt plus the requested references. Values never appear.

    The only caller-supplied content is the reference list -- canonical
    ``R<row>C<col>`` / ``H<row>C<col>`` strings, which carry coordinates and
    nothing else.
    """
    policy, request_list = build_blind_cell_prompt_parts(cell_refs)
    return f"{policy}\n\n{request_list}"


def split_blind_cell_prompt(prompt: str) -> tuple[str, str]:
    """Split a built prompt back into ``(policy, request_list)``.

    Lets a guard assert the structural invariant on the prompt AS SENT --
    taken off the wire payload -- rather than on what a builder claims it
    produced. Raises ``ValueError`` unless the boundary occurs exactly once,
    because a prompt with two request lists, or none, is not a prompt whose
    parts anyone can reason about.
    """
    count = prompt.count(REQUEST_LIST_HEADING)
    if count != 1:
        raise ValueError(f"expected exactly one {REQUEST_LIST_HEADING!r} boundary, found {count}")
    policy, request_list = prompt.split(REQUEST_LIST_HEADING, 1)
    return policy, REQUEST_LIST_HEADING + request_list


def build_cells_payload(model: str, prompt: str, image_b64: str) -> dict[str, Any]:
    """The exact ``/api/chat`` body for one blind transcription.

    Separate from the network call for the same reason ``_build_payload`` is:
    the presence of the image bytes on the wire is the one property this
    whole design rests on, and a test must be able to assert it without a
    live daemon.
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "format": "json",
        "stream": False,
    }


def parse_blind_cell_output(
    text: str, cell_refs: Sequence[str]
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Strict-parse the adjudicator's body into readings and non-readings.

    Returns ``(tokens, unreadable)``. A JSON string value -- INCLUDING the
    empty string -- is a READING: the model looked at that cell and reports
    what was printed there, possibly nothing. A JSON ``null`` is the typed
    NON-reading the prompt asks for when the cell is cut off, obscured or
    illegible (cold review round 2, N2); it carries no token and can neither
    clear a table nor count as disagreement.

    Raises ``ValueError`` for every untrustworthy shape: empty output, no
    JSON object, a non-object value, a value that is neither a string nor
    ``null``, a missing requested key, or an extra key nobody asked for.
    Every one of those is a DEFECT -- a body we cannot read is never softened
    into a partial answer, because a partial answer would clear a table on
    fewer cells than the guard asked about.
    """
    from socr.judge.judge import _extract_json

    if not text or not text.strip():
        raise ValueError("empty adjudicator output")
    data = _extract_json(text)
    if not isinstance(data, dict):
        raise ValueError("adjudicator output is not a JSON object")

    wanted = {str(ref) for ref in cell_refs}
    got = set(data.keys())
    if got != wanted:
        missing = sorted(wanted - got)
        extra = sorted(got - wanted)
        raise ValueError(f"cell key mismatch (missing={missing}, unrequested={extra})")

    tokens: dict[str, str] = {}
    unreadable: list[str] = []
    for key, value in data.items():
        if value is None:
            unreadable.append(key)
            continue
        if not isinstance(value, str):
            raise ValueError(f"value for {key!r} is neither a string nor null: {value!r}")
        tokens[key] = value
    return tokens, tuple(sorted(unreadable))


def transcribe_cells_ollama(
    crop_path: Path | None,
    cell_refs: Sequence[str],
    model: str,
    host: str | None,
    timeout: float,
) -> BlindCellResult:
    """Transcribe exactly ``cell_refs`` from ``crop_path``. Never raises.

    The signature is the blindness contract: a crop, a list of coordinates,
    and transport settings. There is no parameter through which the emitted
    markdown, a reader verdict, a finding or an extraction token could reach
    the model.
    """
    rung = adjudicator_rung_id(model)
    resolved_host = resolve_ollama_host(host)
    start = time.monotonic()
    try:
        if crop_path is None:
            raise OSError("no crop image for the adjudicated table")
        image_b64 = base64.b64encode(Path(crop_path).read_bytes()).decode("ascii")
        payload = build_cells_payload(model, build_blind_cell_prompt(cell_refs), image_b64)
    except OSError as exc:
        # A crop we cannot read is a DEFECT of this call, not an outage: the
        # same missing file reproduces on every rerun.
        return BlindCellResult(
            rung=rung,
            ok=False,
            latency_sec=time.monotonic() - start,
            error=f"{type(exc).__name__}: {exc}",
            unavailable=False,
        )

    try:
        text = _post_chat(resolved_host, payload, timeout)
    except httpx.HTTPStatusError as exc:
        latency_sec = time.monotonic() - start
        body = _response_text(exc.response)
        unavailable = classify_http_status(exc.response.status_code, body)
        return BlindCellResult(
            rung=rung,
            ok=False,
            latency_sec=latency_sec,
            error=f"{type(exc).__name__}: {exc}",
            unavailable=unavailable,
            refusal=unavailable
            and (
                exc.response.status_code in _REFUSAL_STATUS_CODES or output_reads_as_refusal(body)
            ),
        )
    except (httpx.HTTPError, OSError) as exc:
        latency_sec = time.monotonic() - start
        return BlindCellResult(
            rung=rung,
            ok=False,
            latency_sec=latency_sec,
            error=f"{type(exc).__name__}: {exc}",
            unavailable=is_availability_exception(exc),
        )
    except ValueError as exc:
        # A body of the wrong SHAPE (round 3, NEW C). Deterministic: the same
        # daemon answers the same way on the next call, so it never latches.
        latency_sec = time.monotonic() - start
        return BlindCellResult(
            rung=rung,
            ok=False,
            latency_sec=latency_sec,
            error=f"adjudicator ({model}) unusable response: {exc}",
            unavailable=False,
        )
    latency_sec = time.monotonic() - start

    try:
        tokens, unreadable = parse_blind_cell_output(text, cell_refs)
    except (ValueError, TypeError) as exc:
        return BlindCellResult(
            rung=rung,
            ok=False,
            latency_sec=latency_sec,
            error=f"adjudicator ({model}) unusable answer: {exc}",
            unavailable=False,
        )
    return BlindCellResult(
        rung=rung, ok=True, tokens=tokens, unreadable=unreadable, latency_sec=latency_sec
    )


def make_ollama_cell_adjudicator(model: str, host: str | None, timeout: float):
    """Bind the transport into the callable the guard service injects.

    Shape: ``(crop_path, cell_refs) -> BlindCellResult``. Advertises its kind
    and its EXECUTING identity the same way the reader rungs do, so the audit
    trail, the metering and the retry latch name what actually ran rather
    than a default constant.
    """

    def _adjudicator(crop_path: Path | None, cell_refs: Sequence[str]) -> BlindCellResult:
        return transcribe_cells_ollama(crop_path, cell_refs, model, host, timeout)

    _adjudicator.rung_kind = RUNG_KIND_CELL_ADJUDICATOR
    _adjudicator.rung_id = adjudicator_rung_id(model)
    _adjudicator.executing = model
    return _adjudicator
