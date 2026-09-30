"""Lightweight Ollama helpers — no engine-framework dependencies."""

from __future__ import annotations

import socket
import threading
import time
from urllib.parse import urlsplit

import httpx


#: GH-910: total wall-clock budget for the ``/api/tags`` listing. The retired
#: ``ollama list`` subprocess used 10s; kept so a slow-but-alive daemon is
#: treated as before. It is a TOTAL deadline (see ``_get_tags``), not httpx's
#: per-read timeout.
TAGS_CHECK_TIMEOUT_SEC = 10.0

_UNREACHABLE_MSG = "Ollama is not running or not installed"


def _get_tags(host: str, timeout: float) -> httpx.Response | None:
    """``GET {host}/api/tags`` under a TOTAL deadline; ``None`` when it expires.

    httpx's ``timeout=`` is per-read inactivity, so a peer trickling bytes never
    trips it. The request runs in a daemon thread joined with *timeout* (the
    approach ``_resolve_within`` uses): an overrun is abandoned, not waited on,
    and cannot keep the process alive. No process is spawned. Transport errors
    propagate as ``httpx.HTTPError`` / ``OSError``.
    """
    box: list = []

    def _work() -> None:
        try:
            box.append(httpx.get(f"{host}/api/tags", timeout=timeout))
        except BaseException as exc:  # handed to the caller, re-raised there
            box.append(exc)

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    thread.join(timeout)
    if not box:
        return None
    if isinstance(box[0], BaseException):
        raise box[0]
    return box[0]


def check_ollama_model(model_name: str) -> str | None:
    """Return an error string if the model is not PULLED, or None if it is.

    GH-910: reads ``GET {OLLAMA_HOST}/api/tags`` over HTTP, never the ``ollama``
    CLI -- on macOS that binary launches Ollama.app and steals window focus
    whenever the server is unreachable. Matching is an EXACT compare of
    *model_name* against each listed ``name`` (or ``model``), tag included, as
    the ``ollama list`` NAME column was: no ``:latest`` defaulting, no prefix.

    A LISTING, not proof the model can generate. Ollama Cloud kept listing
    ``qwen3.5:cloud`` after retiring it (every call 410 Gone, GH-903/GH-905),
    so a ``:cloud`` tag must never be gated on this alone -- use
    :func:`probe_model_generation`.
    """
    from socr.tables.extract import resolve_ollama_host

    host = resolve_ollama_host()
    if not host_reachable(host):
        return _UNREACHABLE_MSG
    try:
        resp = _get_tags(host, TAGS_CHECK_TIMEOUT_SEC)
        if resp is None:
            return "Ollama did not respond (timeout)"
        if resp.status_code != 200:
            return f"{_UNREACHABLE_MSG} (HTTP {resp.status_code} from /api/tags)"
        models = resp.json()["models"]
        names: set[str] = set()
        for entry in models:
            for key in ("name", "model"):
                value = entry.get(key)
                if isinstance(value, str):
                    names.add(value)
    except httpx.TimeoutException:
        return "Ollama did not respond (timeout)"
    except (httpx.HTTPError, OSError):
        return _UNREACHABLE_MSG
    except (ValueError, KeyError, TypeError, AttributeError):
        return f"{_UNREACHABLE_MSG} (unreadable /api/tags response)"
    if model_name not in names:
        return f"Ollama model '{model_name}' not found. Pull it with: ollama pull {model_name}"
    return None


#: GH-903 round 4 (CI slowdown, cubic P2); moved here by GH-905 so the qwen
#: cloud-rung probe shares it: a per-candidate ``run_killable``
#: spawn is a real ``multiprocessing.spawn`` -- tens of milliseconds even to
#: fail fast -- and CI (no Ollama daemon at all) pays that on EVERY candidate
#: in the ladder, on every run. A plain, short-timeout connect is enough to
#: tell "nothing is listening here" apart from "something is, slowly", and
#: unlike the generation probe's own budget (which must accommodate a cold
#: MODEL load, ~46s measured), an unreachable HOST does not get any more
#: reachable the longer you wait -- so this budget is small and fixed, not
#: derived from ``self.timeout``. This is NOT a model-availability claim: the
#: daemon can be up with the wrong model pulled, or none at all -- the check
#: below only ever short-circuits to unavailable, never to available, and a
#: reachable host still gets the full killable generation probe.
CONNECT_PROBE_TIMEOUT_SEC = 1.0

#: GH-903 / GH-905: probes send ``think: false``. A thinking model otherwise
#: puts its answer in ``thinking`` and leaves ``response`` empty, so a probe
#: (one token) could pass or fail for reasons unrelated to availability.
PROBE_THINK = False

#: Wall-clock budget for a generation probe. A cold-loaded model was measured
#: (GH-903, 2026-09-26, unloaded ``qwen3.8:27b`` on the owner's Mac) to take
#: ~46s to answer a 1-token generation; a shorter budget mistook that load for
#: unavailability. Shared by the page-judge probe and the cloud-rung probe.
DEFAULT_PROBE_TIMEOUT_SEC = 120.0

#: Minimal generation: `num_predict=1` bounds the token count so probing a
#: model that turns out to be slow to answer costs one token, not a full
#: judge-sized response.
_PROBE_OPTIONS = {"num_predict": 1}
_PROBE_PROMPT = "hi"


def probe_failure_reason(exc: Exception) -> str:
    """A short, human-readable reason a judge candidate's probe answered "no".

    Only reached for a DEFINITIVE answer -- the daemon actually responded
    (an HTTP error status, e.g. 410 retired / 404 never pulled) or refused
    the connection outright. A timeout is never classified here: it is not
    proof of unavailability (the probe's budget may simply have been too
    short for a candidate that is cold-loading, measured ~46s for an unloaded
    ``qwen3.8:27b``), and `the caller` reports it distinctly, from the
    ``run_killable`` boundary's own ``TimeoutError`` (see below), before this
    function is ever called.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        body_detail = ""
        try:
            body = exc.response.json()
            if isinstance(body, dict):
                body_detail = str(body.get("error", ""))
        except ValueError:
            body_detail = exc.response.text
        status = exc.response.status_code
        return f"HTTP {status}" + (f": {body_detail}" if body_detail else "")
    return f"{type(exc).__name__}: {exc}"


def host_reachable(host: str, timeout: float = CONNECT_PROBE_TIMEOUT_SEC) -> bool:
    """Cheap "is anything listening at all" check -- never spawns a process,
    never sends an HTTP request, never generates a token (GH-903 round 4,
    cubic P2).

    A raw TCP connect, deliberately -- NOT an ``httpx`` request. An HTTP
    round trip has to read a response, and a peer that trickles the BODY
    (this module's own killable-boundary tests use exactly such a server)
    would defeat an ``httpx`` timeout the same way it defeats
    ``probe_generate``'s, hanging this "cheap" check indefinitely with
    nothing bounding it (unlike the generation probe, this check does not run
    behind ``run_killable``, on purpose -- it exists to AVOID that spawn). A
    bare socket connect only waits on the TCP handshake, which a trickling
    peer cannot stall -- the handshake either completes or the OS refuses it,
    both fast. Any failure to connect (refused, DNS failure, this timeout)
    means "no daemon here": that is the one case a caller may treat
    as definitive without ever calling ``probe_generate``.

    A host string that does not parse (a malformed ``OLLAMA_HOST``, e.g. a
    non-numeric port -- ``resolve_ollama_host`` returns such values
    unchanged) is also "no daemon here": ``urlsplit`` / ``.port`` raise
    ``ValueError`` on it, and that must degrade the judge, not abort the run.
    """
    try:
        parts = urlsplit(host)
        hostname = parts.hostname
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return False
    if not hostname:
        return False
    # GH-905 (cubic P2): ``socket.create_connection(..., timeout=)`` bounds each
    # connect, NOT the DNS lookup it performs first, so a slow resolver could
    # block this pre-check past its budget. One TOTAL deadline covers resolve +
    # connect: the lookup runs in a daemon thread joined with the budget (a
    # stuck resolver is abandoned, not waited on, and cannot keep the process
    # alive), and the connect gets only what is left. No process is spawned.
    deadline = time.monotonic() + timeout
    infos = _resolve_within(hostname, port, timeout)
    if not infos:
        return False
    for family, socktype, proto, _canon, sockaddr in infos:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        # Socket CONSTRUCTION can fail too (descriptor exhaustion, an address
        # family the host doesn't support); that must degrade to "unreachable",
        # not raise out of routing (GH-905 round 4, cubic P2).
        try:
            sock = socket.socket(family, socktype, proto)
        except OSError:
            continue
        try:
            sock.settimeout(remaining)
            sock.connect(sockaddr)
            return True
        except OSError:
            continue
        finally:
            sock.close()
    return False


def _resolve_within(hostname: str, port: int, timeout: float) -> list | None:
    """``getaddrinfo`` bounded by *timeout* seconds; ``None`` on failure or expiry.

    A lookup that outlives *timeout* leaves its daemon thread running until the
    system resolver itself gives up. This is deliberate and bounded: availability
    is resolved once per candidate model per run and then memoized (the judge
    ladder and the pinned cloud rung), so a stuck resolver costs at most a
    handful of short-lived threads per run, never one per page, and a daemon
    thread cannot keep the process alive (GH-905 round 4, cubic P2).
    """
    box: list = []

    def _work() -> None:
        try:
            box.append(socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM))
        except OSError:
            box.append(None)

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    thread.join(timeout)
    return box[0] if box else None


def probe_generate(host: str, model: str, timeout: float) -> dict[str, object]:
    """Top-level, picklable probe body run through ``run_killable`` (GH-903
    round 3, P2-b). Reached only via ``probe_model_generation``; callers never
    invoke it directly.

    ``httpx``'s ``timeout=`` is a per-READ inactivity timeout, not a total
    wall-clock deadline (the same gap #172 closed for ``judge()`` itself): a
    peer that keeps the connection open and trickles a byte before every read
    interval never trips it. ``run_killable`` is what actually bounds this
    call now, by killing the child's process group past ``timeout`` -- so
    THIS function must not classify a timeout itself; it returns a plain,
    picklable outcome for anything it CAN classify (an HTTP status, a refused
    connection), and re-raises a timeout so ``run_killable``'s own
    reclassification (``KillableTimeoutError``, a ``TimeoutError`` subclass)
    is what the parent sees -- exactly the same path ``judge()`` already
    relies on for ``is_page_judge_timeout``.

    A caught exception is returned, not raised, for every non-timeout case:
    ``run_killable`` collapses ANY child exception that crosses the pipe into
    a generic ``RuntimeError`` carrying only the original type name and
    message (it cannot safely pickle arbitrary exception instances, e.g. an
    ``httpx.HTTPStatusError`` holding a live ``Response``), which would lose
    the response body ``probe_failure_reason`` needs. Classifying HERE, then
    crossing the pipe as a plain dict, keeps that detail.
    """
    try:
        resp = httpx.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": _PROBE_PROMPT,
                "stream": False,
                "think": PROBE_THINK,
                "options": _PROBE_OPTIONS,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return {"available": True, "reason": ""}
    except httpx.TimeoutException:
        raise
    except (httpx.HTTPError, OSError) as exc:
        return {"available": False, "reason": probe_failure_reason(exc)}


def probe_model_generation(
    host: str,
    model: str,
    timeout: float,
    *,
    reachable=None,
    runner=None,
) -> tuple[bool, str]:
    """Whether a real 1-token generation on THIS EXACT *model* succeeds.

    GH-905 (shared with GH-903's page-judge probe). Returns
    ``(available, reason)``; *reason* is "" when available. A listing
    (``ollama list`` / ``/api/tags``) is NOT this check: Ollama Cloud retired
    ``qwen3.5:cloud`` on 2026-09-25 and kept listing it, while every generation
    returned 410 Gone.

    Order: (1) ``host_reachable`` -- a spawn-free TCP connect; unreachable is
    definitive. (2) ``probe_generate`` run through ``run_killable`` with
    *timeout* as the wall-clock deadline, so a peer that trickles bytes cannot
    wedge the caller. A timeout is reported as such (inconclusive, not
    "retired"); an HTTP error status is definitive.

    *reachable* / *runner* default to :func:`host_reachable` /
    :func:`socr.core.killable.run_killable`; callers pass their own module-level
    names so tests that patch those names on the caller keep working.
    """
    from socr.core.killable import CallSpec, run_killable

    reachable = reachable or host_reachable
    runner = runner or run_killable
    if not reachable(host):
        return False, f"ollama host unreachable: {host}"
    spec = CallSpec(
        func="socr.core.ollama_utils:probe_generate",
        args=(host, model, timeout),
    )
    try:
        outcome = runner(spec, timeout=timeout)
    except TimeoutError:
        return False, f"timed out after {timeout:.0f}s"
    if outcome["available"]:
        return True, ""
    return False, str(outcome["reason"])
