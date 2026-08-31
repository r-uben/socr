"""GH-367: constrained cell-raster transcription.

A transcriber is not a judge. It is shown one native-word bbox crop and
must return a token string. It never sees markdown, the native string,
findings, or a PASS/FAIL schema. Arithmetic in ``tables.adjudication``
decides whether that token disproves a ``bind()`` contradiction.

Transport reuses the table-judge ollama chat seam (``_post_chat`` /
``_build_payload``) so tests can monkeypatch one function. Every failure
— missing crop, timeout, unparseable JSON, missing ``token`` — returns
None (not a disproof). Never raises.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from socr.judge.judge import _extract_json
from socr.judge.table_rung_ollama import _build_payload, _post_chat
from socr.tables.extract import resolve_ollama_host

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "cell_transcribe.md"


def load_cell_transcribe_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def parse_transcribe_output(text: str) -> str | None:
    """Return the token string, or None if the output is not usable.

    Empty token after strip is None — the prompt uses {\"token\":\"\"}
    for unreadable crops, which is absence of evidence, not a token.
    """
    if not text or not text.strip():
        return None
    try:
        data = _extract_json(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    token = data.get("token")
    if not isinstance(token, str):
        return None
    token = token.strip()
    return token or None


def transcribe_cell(
    crop_path: Path,
    *,
    model: str,
    host: str | None,
    timeout: float,
) -> str | None:
    """POST the crop to ollama /api/chat; return a token or None. Never raises."""
    import base64

    try:
        image_b64 = base64.b64encode(Path(crop_path).read_bytes()).decode("ascii")
    except OSError as exc:
        logger.warning("cell transcribe: unreadable crop %s (%s)", crop_path, exc)
        return None
    payload = _build_payload(model, load_cell_transcribe_prompt(), image_b64)
    try:
        raw = _post_chat(resolve_ollama_host(host), payload, timeout)
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("cell transcribe: %s (%s: %s)", crop_path, type(exc).__name__, exc)
        return None
    return parse_transcribe_output(raw)
