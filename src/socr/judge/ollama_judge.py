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

from socr.judge.judge import JudgeVerdict, load_judge_prompt, parse_verdict

DEFAULT_MODEL = "qwen2-vl:7b"
DEFAULT_HOST = "http://localhost:11434"


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
        """True if the Ollama server is up and the model is pulled."""
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
            return any(self.model.split(":")[0] in n for n in names)
        except (httpx.HTTPError, ValueError):
            return False

    def judge(self, image_path: Path, ocr_text: str) -> JudgeVerdict:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        prompt = f"{self._prompt}\n\n---\nCANDIDATE TRANSCRIPTION:\n\n{ocr_text}"
        resp = httpx.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0},  # judging should be as stable as we can make it
                "format": "json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return parse_verdict(resp.json().get("response", ""))
