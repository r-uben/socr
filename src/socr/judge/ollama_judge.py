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
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def _with_implicit_tag(name: str) -> str:
    """Ollama resolves an untagged model reference to ``:latest``."""
    return name if ":" in name else f"{name}:latest"


class OllamaVisionJudge:
    """Judge backed by a local Ollama vision model."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._prompt = load_judge_prompt()

    def is_available(self) -> bool:
        """True if the Ollama server is up and THIS EXACT model is pulled.

        Matched on the full ``name:tag``. A prefix match on the name alone (the
        historical behaviour) let an installed ``qwen3-vl:30b-a3b-instruct``
        satisfy a request for ``qwen3-vl:8b``: the model reported as available,
        then ``judge()`` 404'd at judge time on a model that was never pulled
        (#133). Availability must mean the pull, not the family.
        """
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            resp.raise_for_status()
            names = {_with_implicit_tag(m.get("name", "")) for m in resp.json().get("models", [])}
            return _with_implicit_tag(self.model) in names
        except (httpx.HTTPError, ValueError):
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
