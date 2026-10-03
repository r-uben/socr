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

from socr.core.ollama_utils import raise_for_status_redacted
from socr.core.killable import CallSpec, run_killable
from socr.core.ollama_utils import (  # noqa: F401 -- CONNECT_PROBE_TIMEOUT_SEC re-exported
    CONNECT_PROBE_TIMEOUT_SEC,
    DEFAULT_PROBE_TIMEOUT_SEC,
    PROBE_THINK,
    probe_model_generation,
)
from socr.core.ollama_utils import host_reachable as _host_reachable
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
DEFAULT_JUDGE_TIMEOUT_SEC = DEFAULT_PROBE_TIMEOUT_SEC


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
            "think": PROBE_THINK,
        },
        timeout=timeout,
    )
    raise_for_status_redacted(resp)
    return resp.json().get("response", "")


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

        Delegates to ``probe_model_generation`` (a listing is not proof; see
        there) with ``self.timeout`` as the budget, and sets
        ``unavailable_reason``. ``_host_reachable`` / ``run_killable`` are
        passed from this module so tests can patch them here.
        """
        available, self.unavailable_reason = probe_model_generation(
            self.host,
            self.model,
            self.timeout,
            reachable=_host_reachable,
            runner=run_killable,
        )
        return available

    def judge(self, image_path: Path, ocr_text: str) -> JudgeVerdict:
        """Judge one page. The model call runs behind a killable process boundary.

        ``run_killable`` runs ``_post_generate`` in a child process, so this
        returns (raising ``KillableTimeoutError``, a ``TimeoutError`` subclass
        that ``is_page_judge_timeout`` classifies) within ``self.timeout``
        whatever the peer does (GH-172).
        """
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        prompt = f"{self._prompt}\n\n---\nCANDIDATE TRANSCRIPTION:\n\n{ocr_text}"
        spec = CallSpec(
            func="socr.judge.ollama_judge:_post_generate",
            args=(self.host, self.model, prompt, image_b64, self.timeout),
        )
        raw = run_killable(spec, timeout=self.timeout)
        return parse_verdict(raw)
