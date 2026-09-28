"""Local-VLM judge backend (Ollama).

Realizes the "near-zero marginal cost, headless, no ToS risk" path from the
design review: the judge runs as a local vision model on the same machine as the
OCR (e.g. Qwen2-VL on the Bocconi A100/H100 nodes), reachable via the Ollama
HTTP API. No subscription CLI, no metered tokens.

Kept separate from ``judge.py`` so the verdict parsing / scoring logic stays
importable and testable without Ollama installed.
"""

from __future__ import annotations

import base64
import socket
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from socr.core.killable import CallSpec, run_killable
from socr.judge.judge import JudgeVerdict, load_judge_prompt, parse_verdict

DEFAULT_MODEL = "qwen2-vl:7b"
DEFAULT_HOST = "http://localhost:11434"

#: GH-903 round 4: the page judge used to ignore ``OLLAMA_HOST`` entirely --
#: every construction site left ``host`` unset, so it always hit the bare
#: ``DEFAULT_HOST`` literal above regardless of where the daemon (or a test's
#: fake one) actually was, unlike every other Ollama call site in this repo
#: (``socr.tables.extract.resolve_ollama_host``, used by the table judge and
#: the engines). ``OllamaVisionJudge.__init__`` now resolves through the same
#: helper, so a deployment that has pointed its Ollama client at a remote or
#: non-default host via the env var socr's other subsystems already honor is
#: not silently disagreed with here, and so the reachability tests below can
#: actually exercise "no daemon" by setting that variable rather than hoping
#: nothing is listening on localhost.
#: GH-903 round 4 (CI slowdown, cubic P2): a per-candidate ``run_killable``
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

#: The judge call's own wall-clock budget (``OllamaVisionJudge.timeout``, and
#: the default the availability probe below now shares -- GH-903 round 2). A
#: cold-loaded ``qwen3.8:27b`` (the new default candidate; unloaded, i.e. not
#: resident in GPU memory when the probe runs) was measured on the owner's
#: Mac to take ~46s to answer a 1-token generation. The probe's FIRST cut
#: (a separate, shorter budget) treated that slow-but-real load as
#: unavailability, memoized ``None`` for the whole run (and a whole ``socr
#: batch``), and degraded every page to the heuristic judge. The probe must
#: get the same budget the real judge call gets, or it will keep being wrong
#: about exactly the case (a cold model) it exists to tell apart from a
#: genuinely retired/missing one.
DEFAULT_JUDGE_TIMEOUT_SEC = 120.0

#: GH-903: the judge model ladder includes thinking models (e.g. `qwen3.8:27b`).
#: With `think` unset and `format=json`, a thinking model puts its answer in
#: `thinking` and leaves `response` empty, which socr then reports as "no JSON
#: object found in judge output" or a timeout waiting on a stream that never
#: emits the final answer. `"think": false` was measured (2026-09-26) to
#: return correct JSON in ~8s warm, and was verified harmless (HTTP 200, normal
#: output) on non-thinking models too (`qwen3-vl:30b-a3b-instruct`), so it is
#: sent unconditionally rather than branching on a per-model "is this a
#: thinking model" table that would need updating every time a new candidate
#: is added. Only the PAGE judge sends this (GH-903 round 2): the table judge
#: ladder and the cell-transcription adjudicator use cloud thinking models
#: (`glm-5.3-flash:cloud`, `kimi-k2.6:cloud`) with measured accuracy behind
#: their reasoning traces, and turning that off is an unmeasured change this
#: ticket does not make -- see `table_rung_ollama.py` and the GH-903 log.
_THINK = False

#: Minimal generation: `num_predict=1` bounds the token count so probing a
#: model that turns out to be slow to answer costs one token, not a full
#: judge-sized response.
_PROBE_OPTIONS = {"num_predict": 1}
_PROBE_PROMPT = "hi"


def _post_generate(host: str, model: str, prompt: str, image_b64: str, timeout: float) -> str:
    """The ONLY part of ``judge()`` that crosses the killable boundary (GH-172).

    Deliberately a top-level function taking plain picklable args -- never a
    bound method -- so it can be run inside a fresh ``spawn``-ed child via
    ``CallSpec``. The ``httpx`` client timeout below is kept as
    defence-in-depth (panel ruling: legal, never the close); the caller's
    ``run_killable`` wall-clock deadline is what actually bounds a peer that
    keeps the response stream open and trickles bytes, which defeats this
    per-chunk read timeout (measured in ``docs/log/2026-09-17_172-design.md``).
    """
    resp = httpx.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0},  # judging should be as stable as we can make it
            "format": "json",
            "think": _THINK,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _probe_failure_reason(exc: Exception) -> str:
    """A short, human-readable reason a judge candidate's probe answered "no".

    Only reached for a DEFINITIVE answer -- the daemon actually responded
    (an HTTP error status, e.g. 410 retired / 404 never pulled) or refused
    the connection outright. A timeout is never classified here: it is not
    proof of unavailability (the probe's budget may simply have been too
    short for a candidate that is cold-loading, measured ~46s for an unloaded
    ``qwen3.8:27b``), and ``is_available`` reports it distinctly, from the
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


def _host_reachable(host: str, timeout: float = CONNECT_PROBE_TIMEOUT_SEC) -> bool:
    """Cheap "is anything listening at all" check -- never spawns a process,
    never sends an HTTP request, never generates a token (GH-903 round 4,
    cubic P2).

    A raw TCP connect, deliberately -- NOT an ``httpx`` request. An HTTP
    round trip has to read a response, and a peer that trickles the BODY
    (this module's own killable-boundary tests use exactly such a server)
    would defeat an ``httpx`` timeout the same way it defeats
    ``_probe_generate``'s, hanging this "cheap" check indefinitely with
    nothing bounding it (unlike the generation probe, this check does not run
    behind ``run_killable``, on purpose -- it exists to AVOID that spawn). A
    bare socket connect only waits on the TCP handshake, which a trickling
    peer cannot stall -- the handshake either completes or the OS refuses it,
    both fast. Any failure to connect (refused, DNS failure, this timeout)
    means "no daemon here": that is the one case ``is_available`` may treat
    as definitive without ever calling ``_probe_generate``.

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
    try:
        with socket.create_connection((hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_generate(host: str, model: str, timeout: float) -> dict[str, object]:
    """Top-level, picklable probe body run through ``run_killable`` (GH-903
    round 3, P2-b) -- never called directly by ``is_available``.

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
    the response body ``_probe_failure_reason`` needs. Classifying HERE, then
    crossing the pipe as a plain dict, keeps that detail.
    """
    try:
        resp = httpx.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": _PROBE_PROMPT,
                "stream": False,
                "think": _THINK,
                "options": _PROBE_OPTIONS,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return {"available": True, "reason": ""}
    except httpx.TimeoutException:
        raise
    except (httpx.HTTPError, OSError) as exc:
        return {"available": False, "reason": _probe_failure_reason(exc)}


class OllamaVisionJudge:
    """Judge backed by a local Ollama vision model."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        timeout: float = DEFAULT_JUDGE_TIMEOUT_SEC,
    ) -> None:
        from socr.tables.extract import resolve_ollama_host

        self.model = model
        # GH-903 round 4: resolve through the SAME helper the table judge and
        # engines use (explicit arg, then ``OLLAMA_HOST``, then
        # ``DEFAULT_HOST``) -- an unset ``host`` used to always mean the bare
        # ``DEFAULT_HOST`` literal, ignoring a deployment that has already
        # pointed its Ollama client elsewhere via the env var.
        self.host = resolve_ollama_host(host).rstrip("/")
        self.timeout = timeout
        self._prompt = load_judge_prompt()
        #: Set by ``is_available()`` -- "" when it returned True, otherwise a
        #: short reason (GH-903) a caller can surface at run level (e.g. the
        #: 410/retired message), without changing ``is_available``'s bool
        #: contract that every existing caller relies on.
        self.unavailable_reason: str = ""

    def is_available(self) -> bool:
        """True iff a real 1-token generation on THIS EXACT model succeeds.

        GH-903: a ``/api/tags`` listing is not this check -- Ollama Cloud
        retired ``qwen3.5:cloud`` on 2026-09-25, but ``/api/tags`` kept
        listing it (retirement is a generation-time 410, not a catalogue
        change), so the old tags-based probe kept selecting a judge that
        raised on every call. Only a real generation observes what a judge
        call will actually see. ``think: false`` matches ``_post_generate``:
        a thinking-model candidate (e.g. ``qwen3.8:27b``) would otherwise put
        its answer in ``thinking`` and leave ``response`` empty, which reads
        as an empty/unparseable body rather than as "unavailable" -- so this
        probe must send the same flag the real judge call does, or a thinking
        candidate could pass the probe and still fail every judge call.

        GH-903 round 2: the probe's budget is ``self.timeout`` -- the SAME
        one the judge call itself gets -- not a shorter, separate one. A cold
        model load (measured ~46s for an unloaded ``qwen3.8:27b``) is real
        latency, not unavailability; a probe with its own tighter budget
        mistook that load for a 404 and memoized ``None`` for the whole run.

        GH-903 round 3 (P2-b): that budget is enforced by ``run_killable``,
        the SAME killable-process boundary ``judge()`` uses (GH-172) -- not
        by ``httpx``'s own ``timeout=`` alone, which only bounds inactivity
        between reads. A peer that trickles a byte before every read interval
        never trips that, and would otherwise wedge resolution (and the whole
        per-page loop that consults it) indefinitely. A timeout here -- from
        ``run_killable``'s own deadline, or the child's own httpx timeout,
        which ``run_killable`` reclassifies identically -- is reported
        distinctly from an HTTP error status (``_probe_failure_reason``): the
        latter is definitive, the former only proves this call needed longer
        than ``timeout``.

        GH-903 round 4 (cubic P2): a cheap reachability pre-check
        (``_host_reachable``) runs FIRST, with no spawn -- an unreachable
        host is definitive and does not need the killable generation probe
        at all. This is what keeps a host with no Ollama daemon (CI) from
        paying a real ``multiprocessing.spawn`` per candidate in the ladder.
        """
        self.unavailable_reason = ""
        if not _host_reachable(self.host):
            self.unavailable_reason = f"ollama host unreachable: {self.host}"
            return False
        spec = CallSpec(
            func="socr.judge.ollama_judge:_probe_generate",
            args=(self.host, self.model, self.timeout),
        )
        try:
            outcome = run_killable(spec, timeout=self.timeout)
        except TimeoutError:
            self.unavailable_reason = f"timed out after {self.timeout:.0f}s"
            return False
        if outcome["available"]:
            return True
        self.unavailable_reason = str(outcome["reason"])
        return False

    def judge(self, image_path: Path, ocr_text: str) -> JudgeVerdict:
        """Judge one page. The model call runs behind a killable process boundary.

        GH-172: a wedged Ollama connection used to block this thread's HTTP
        call indefinitely -- ``httpx``'s read timeout is per-chunk, not
        per-request, so a peer that keeps trickling bytes into an open
        response never trips it, and the ``ThreadPoolExecutor`` deadline that
        used to wrap ``assess()`` (``_TimeoutJudge`` in the orchestrator)
        could only ABANDON that thread, not stop it -- the interpreter still
        joined it at exit. ``run_killable`` runs ``_post_generate`` in its own
        killable child process instead, so this call now returns (with
        ``KillableTimeoutError``, a ``TimeoutError`` subclass the existing
        ``is_page_judge_timeout``/``judge_outcome`` machinery already
        classifies) within ``self.timeout`` regardless of what the peer does.
        """
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        prompt = f"{self._prompt}\n\n---\nCANDIDATE TRANSCRIPTION:\n\n{ocr_text}"
        spec = CallSpec(
            func="socr.judge.ollama_judge:_post_generate",
            args=(self.host, self.model, prompt, image_b64, self.timeout),
        )
        raw = run_killable(spec, timeout=self.timeout)
        return parse_verdict(raw)
