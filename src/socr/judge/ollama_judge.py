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
from pathlib import Path

import httpx

from socr.core.killable import CallSpec, run_killable
from socr.judge.judge import JudgeVerdict, load_judge_prompt, parse_verdict

DEFAULT_MODEL = "qwen2-vl:7b"
DEFAULT_HOST = "http://localhost:11434"

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


def _probe_failure_reason(exc: Exception, timeout: float) -> str:
    """A short, human-readable reason a judge candidate failed its probe.

    Two different kinds of evidence, on purpose (GH-903 round 2):

    - An HTTP error status (``httpx.HTTPStatusError``, e.g. 410 retired, 404
      never pulled) is DEFINITIVE: the daemon answered and said no, so this
      candidate is unavailable regardless of how long the probe waited.
    - A timeout is NOT proof of unavailability -- it only proves the probe's
      own budget was too short for a candidate that may simply be cold-
      loading (measured ~46s for an unloaded ``qwen3.8:27b``). It is reported
      distinctly (``"timed out after Ns"``) so a run that degrades to
      heuristics can tell "this model is gone" apart from "this model needed
      longer than ``timeout`` to warm up".
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
    if isinstance(exc, httpx.TimeoutException):
        return f"timed out after {timeout:.0f}s"
    return f"{type(exc).__name__}: {exc}"


class OllamaVisionJudge:
    """Judge backed by a local Ollama vision model."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_JUDGE_TIMEOUT_SEC,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
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

        GH-903 round 2: the probe uses ``self.timeout`` -- the SAME budget
        the judge call itself gets -- not a shorter, separate one. A cold
        model load (measured ~46s for an unloaded ``qwen3.8:27b``) is real
        latency, not unavailability; a probe with its own tighter budget
        mistook that load for a 404 and memoized ``None`` for the whole run.
        A timeout is therefore reported distinctly from an HTTP error status
        (see ``_probe_failure_reason``): the latter is definitive, the former
        only proves this call needed longer than ``timeout``.
        """
        self.unavailable_reason = ""
        try:
            resp = httpx.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": _PROBE_PROMPT,
                    "stream": False,
                    "think": _THINK,
                    "options": _PROBE_OPTIONS,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except (httpx.HTTPError, OSError) as exc:
            self.unavailable_reason = _probe_failure_reason(exc, self.timeout)
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
