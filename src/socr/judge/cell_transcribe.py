"""GH-367: constrained cell-raster transcription.

A transcriber is not a judge. It is shown one native-word bbox crop and
must return a token string. It never sees markdown, the native string,
findings, or a PASS/FAIL schema. Arithmetic in ``tables.adjudication``
decides whether that token disproves a ``bind()`` contradiction.

Transport reuses the table-judge ollama chat seam, resolved THROUGH
``table_rung_ollama`` rather than copied into this module's namespace, so
monkeypatching that shared seam actually reaches this caller (GH-388 review):
a copied reference would let a test believe it had stubbed the transport while
this function still opened a socket to a real ollama daemon. Every failure
— missing crop, timeout, unparseable JSON, missing ``token`` — returns
None (not a disproof). Never raises.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from socr.judge.judge import _extract_json
from socr.judge import table_rung_ollama as _table_rung_ollama
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
    # GH-388 review (cubic P2): the failure boundary must cover EVERYTHING the
    # docstring promises, not just transport. Prompt construction can raise
    # OSError on an unreadable policy file; ``resp.json()`` inside the transport
    # raises ValueError on a 200 body that is not JSON, and attribute access on
    # an unexpected shape raises AttributeError/KeyError/TypeError. Any of those
    # escaping breaks the "Never raises" contract and, worse, would abort the
    # gate instead of simply not being a disproof. Not-a-disproof is the correct
    # outcome for every one of them.
    try:
        payload = _table_rung_ollama._build_payload(model, load_cell_transcribe_prompt(), image_b64)
        raw = _table_rung_ollama._post_chat(resolve_ollama_host(host), payload, timeout)
        return parse_transcribe_output(raw)
    except (httpx.HTTPError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        logger.warning("cell transcribe: %s (%s: %s)", crop_path, type(exc).__name__, exc)
        return None
