"""GH-353 TICKET-A2: CLI1 rung — the ollama-cloud table judge.

Builds an `A1` `RungCallable` bound to one ollama model/host, POSTing
`/api/chat` (NOT `/api/generate` — the GH-356 bake-off's integration path;
`ollama run --images` does not exist on ollama 0.32.15) with the table crop
as a base64 image, `format="json"`, `stream=False`.

The network call is isolated behind `_post_chat`, the module's one transport
seam: tests monkeypatch THIS function, never `httpx` globally, so the exact
outgoing JSON payload (built by `_build_payload`, kept separate so it can be
asserted on without a real request) stays independently verifiable.

Every failure mode this module can produce — timeout, connection error, HTTP
error status, or a verdict that fails strict parsing — becomes
`RungResult(ok=False)` (¬S1, `socr.judge.table_verdict`). It never raises and
never synthesizes a FAIL verdict; a substitute rung must see fresh eyes, not
a verdict nobody actually produced.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import httpx

from socr.judge.table_prompt import build_table_judge_prompt
from socr.judge.table_verdict import Finding, RungResult, rung_result_from_output
from socr.tables.extract import resolve_ollama_host

#: Rung identifier prefix (`RungResult.rung`, e.g. "ollama:glm-5.3-flash:cloud"),
#: matching the example in `socr.judge.table_verdict.RungResult`'s docstring.
_RUNG_PREFIX = "ollama"


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
        prompt = build_table_judge_prompt(
            markdown,
            [
                {"code": finding.code.value, "where": finding.where, "detail": finding.detail}
                for finding in (prior_findings or [])
            ],
        )
        image_b64 = base64.b64encode(Path(crop_path).read_bytes()).decode("ascii")
        payload = _build_payload(model, prompt, image_b64)

        start = time.monotonic()
        try:
            text = _post_chat(resolved_host, payload, timeout)
        except httpx.HTTPError as exc:
            latency_sec = time.monotonic() - start
            return RungResult(
                rung=rung,
                ok=False,
                latency_sec=latency_sec,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_sec = time.monotonic() - start
        return rung_result_from_output(rung, text, latency_sec)

    return _judge
