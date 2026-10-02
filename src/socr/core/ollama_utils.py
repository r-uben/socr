"""Lightweight Ollama helpers — no engine-framework dependencies."""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

#: GH-910: total wall-clock budget for the ``/api/tags`` listing. The retired
#: ``ollama list`` subprocess used 10s; kept so a slow-but-alive daemon is
#: treated as before. It is a TOTAL deadline (see ``_get_tags``), not httpx's
#: per-read timeout.
TAGS_CHECK_TIMEOUT_SEC = 10.0

_UNREACHABLE_MSG = "Ollama is not running or not installed"
_TIMEOUT_MSG = "Ollama did not respond (timeout)"


class TotalDeadlineExceeded(httpx.ReadTimeout, TimeoutError):
    """A call outlived its TOTAL wall-clock deadline (GH-968).

    Subclasses BOTH ``httpx.ReadTimeout`` (so every httpx caller's existing
    ``except httpx.TimeoutException`` / ``httpx.HTTPError`` classification keeps
    working: a rung maps it to ``RungResult(ok=False)``) and the builtin
    ``TimeoutError`` (so the urllib callers' ``except (URLError, TimeoutError,
    OSError)`` also catches it). No caller needed to change.
    """


def call_with_total_deadline(fn: Callable[[], Any], timeout: float, *, label: str = "") -> Any:
    """Run *fn* under a TOTAL wall-clock deadline of *timeout* seconds (GH-968).

    httpx/urllib timeouts are per-read/per-socket-op inactivity limits: a peer
    that trickles a byte (or a keepalive) every few seconds never trips them and
    the calling thread hangs indefinitely. *fn* runs in a daemon thread joined
    with *timeout*; an overrun is abandoned, not waited on, and cannot keep the
    process alive. Returns *fn*'s value; re-raises what *fn* raised; on overrun
    raises :class:`TotalDeadlineExceeded` naming *label* so the failure that
    surfaces (``RungResult.error``, a logged warning) identifies the call.
    """
    finished, value = _call_within(fn, timeout)
    if not finished:
        what = label or "Ollama call"
        logger.warning("%s exceeded its total deadline of %ss; abandoned", what, timeout)
        raise TotalDeadlineExceeded(f"{what} exceeded total deadline of {timeout}s")
    if isinstance(value, BaseException):
        raise value
    return value


def _call_within(fn: Callable[[], object], timeout: float) -> tuple[bool, object]:
    """Run *fn* in a daemon thread joined with *timeout*: ``(finished, value_or_exc)``.

    An overrun is abandoned, not waited on, and cannot keep the process alive.
    Anything *fn* raises comes back as the value (an exception instance) for the
    caller to re-raise or map; ``finished`` is False when the deadline expired.
    Prefer :func:`call_with_total_deadline`; this is its primitive.
    """
    box: list[object] = []

    def _work() -> None:
        try:
            box.append(fn())
        except BaseException as exc:  # handed to the caller
            box.append(exc)

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    thread.join(timeout)
    if not box:
        return False, None
    return True, box[0]


def _get_tags(host: str, timeout: float) -> httpx.Response | None:
    """``GET {host}/api/tags`` under a TOTAL deadline; ``None`` when it expires.

    httpx's ``timeout=`` is per-read inactivity, so a peer trickling bytes never
    trips it. The request runs in a daemon thread joined with *timeout* (the
    approach ``_resolve_within`` uses): an overrun is abandoned, not waited on,
    and cannot keep the process alive. No process is spawned. Transport errors
    propagate as ``httpx.HTTPError`` / ``OSError``.
    """
    finished, value = _call_within(lambda: httpx.get(f"{host}/api/tags", timeout=timeout), timeout)
    if not finished:
        return None
    if isinstance(value, BaseException):
        raise value
    return value  # type: ignore[return-value]


def _listed_model_names(resp: httpx.Response) -> set[str]:
    """Every ``name`` / ``model`` string listed by an ``/api/tags`` response."""
    # ``null`` is how older Ollama reports an empty store: nothing pulled,
    # so "not found", not "unreadable".
    models = resp.json()["models"]
    if models is None:
        models = []
    if not isinstance(models, list):
        raise TypeError(f"/api/tags 'models' is {type(models).__name__}, not a list")
    names: set[str] = set()
    for entry in models:
        for key in ("name", "model"):
            value = entry.get(key)
            if isinstance(value, str):
                names.add(value)
    return names


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
            return _TIMEOUT_MSG
        if resp.status_code != 200:
            return f"{_UNREACHABLE_MSG} (HTTP {resp.status_code} from /api/tags)"
        names = _listed_model_names(resp)
    except httpx.TimeoutException:
        return _TIMEOUT_MSG
    except (httpx.HTTPError, OSError):
        return _UNREACHABLE_MSG
    except (ValueError, KeyError, TypeError, AttributeError):
        return f"{_UNREACHABLE_MSG} (unreadable /api/tags response)"
    if model_name not in names:
        return f"Ollama model '{model_name}' not found. Pull it with: ollama pull {model_name}"
    return None


#: TCP-connect budget for ``host_reachable``: small and fixed, since an
#: unreachable host does not become reachable by waiting. It only ever
#: short-circuits to unavailable and never proves a model is available.
CONNECT_PROBE_TIMEOUT_SEC = 1.0

#: The ``think`` flag every judge/probe generation sends. A thinking model
#: (e.g. ``qwen3.8:27b``) with ``think`` unset and ``format=json`` puts its
#: answer in ``thinking`` and leaves ``response`` empty, which reads as "no JSON
#: object found" or a timeout rather than a real answer. ``False`` was measured
#: (2026-09-26, GH-903) to return correct JSON in ~8s warm and is harmless on
#: non-thinking models (``qwen3-vl:30b-a3b-instruct``), so it is sent
#: unconditionally rather than per-model. The probe MUST send the same flag as
#: the real page-judge call (``_post_generate``) or a thinking candidate could
#: pass the probe and still fail every judge call. Only the PAGE judge and the
#: probes send it: the table judge ladder and the cell adjudicator use cloud
#: thinking models whose measured accuracy depends on their reasoning traces
#: (see ``table_rung_ollama.py`` and the GH-903 log).
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
    """A short, human-readable reason a probe answered "no".

    Only for a DEFINITIVE answer: an HTTP error status (410 retired, 404 never
    pulled) or a refused connection. A timeout is never classified here (it is
    not proof of unavailability); ``probe_model_generation`` reports it from
    ``run_killable``'s own ``TimeoutError`` before this is called.
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
    """Cheap "is anything listening at all" check: a raw TCP connect.

    Never spawns a process, sends an HTTP request or generates a token. Not an
    ``httpx`` request on purpose: a peer that trickles the response body defeats
    an httpx timeout, while a bare connect only waits on the TCP handshake. This
    check does not run behind ``run_killable`` (it exists to avoid that spawn).
    Any failure to connect (refused, DNS failure, timeout) means "no daemon
    here", which a caller may treat as definitive.

    A host string that does not parse (a malformed ``OLLAMA_HOST``, e.g. a
    non-numeric port) is also "no daemon here" and degrades, never aborts.
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


def _resolve_within(hostname: str, port: int, timeout: float) -> list[tuple] | None:
    """``getaddrinfo`` bounded by *timeout* seconds; ``None`` on failure or expiry.

    A lookup that outlives *timeout* leaves its daemon thread running until the
    system resolver itself gives up. This is deliberate and bounded: availability
    is resolved once per candidate model per run and then memoized (the judge
    ladder and the pinned cloud rung), so a stuck resolver costs at most a
    handful of short-lived threads per run, never one per page, and a daemon
    thread cannot keep the process alive (GH-905 round 4, cubic P2).
    """

    def _lookup() -> list[tuple] | None:
        try:
            return socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except OSError:
            return None

    finished, value = _call_within(_lookup, timeout)
    if not finished or isinstance(value, BaseException):
        return None
    return value  # type: ignore[return-value]


def probe_generate(host: str, model: str, timeout: float) -> dict[str, object]:
    """Top-level, picklable probe body run through ``run_killable``.

    Reached only via ``probe_model_generation``. ``httpx``'s ``timeout=`` is a
    per-read inactivity timeout, so ``run_killable`` is what bounds the call;
    a timeout is therefore re-raised for ``run_killable`` to reclassify.
    Every other failure is returned as a plain dict, not raised: ``run_killable``
    collapses any child exception into a bare ``RuntimeError`` and would lose
    the response body ``probe_failure_reason`` needs.
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
    reachable: Callable[[str], bool] | None = None,
    runner: Callable[..., dict] | None = None,
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
