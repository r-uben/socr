"""GH-353 TICKET-A2: CLI1 rung — the ollama-cloud table judge.

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
from pathlib import Path
from typing import Any

import httpx

from socr.judge.table_prompt import build_table_judge_prompt
from socr.judge.table_verdict import (
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
    """
    resp = httpx.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


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
