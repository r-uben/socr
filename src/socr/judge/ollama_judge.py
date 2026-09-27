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

#: GH-903: the judge model ladder includes thinking models (e.g. `qwen3.8:27b`).
#: With `think` unset and `format=json`, a thinking model puts its answer in
#: `thinking` and leaves `response` empty, which socr then reports as "no JSON
#: object found in judge output" or a 120s timeout waiting on a stream that
#: never emits the final answer. `"think": false` was measured (2026-09-26) to
#: return correct JSON in ~8s warm, and was verified harmless (HTTP 200, normal
#: output) on non-thinking models too (`qwen3-vl:30b-a3b-instruct`,
#: `glm-5.3-flash:cloud`), so it is sent unconditionally rather than branching
#: on a per-model "is this a thinking model" table that would need updating
#: every time a new candidate is added.
_THINK = False

#: GH-903: the availability probe used to be a `/api/tags` listing, which lied
#: about a model Ollama Cloud had retired (410 Gone on generation, but the tag
#: still listed). A real 1-token generation is the only check that observes
#: what a judge call will actually see. Bounded to the same order as
#: `socr.core.ollama_utils.check_ollama_model`'s subprocess timeout (10s): this
#: is a probe, not the judge call itself (`timeout`, default 120s), and a
#: candidate that cannot answer this fast must not block the ladder while the
#: real judge call further down still has its own generous budget.
PROBE_TIMEOUT_SEC = 10.0

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
    """A short, human-readable reason a judge candidate failed its probe.

    Distinguishes a model Ollama Cloud retired (410, still listed in
    ``/api/tags``) from a model that was simply never pulled (404) or a
    daemon that is not reachable at all -- the run-level surface (GH-903)
    needs to say WHY, not just that the ladder fell through.
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


class OllamaVisionJudge:
    """Judge backed by a local Ollama vision model."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
        probe_timeout: float = PROBE_TIMEOUT_SEC,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.probe_timeout = probe_timeout
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
                timeout=self.probe_timeout,
            )
            resp.raise_for_status()
            return True
        except (httpx.HTTPError, OSError) as exc:
            self.unavailable_reason = _probe_failure_reason(exc)
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
