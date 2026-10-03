"""Typesafe (Jev) confirmation for vision-model tables on unsure native pages.

When the PDF text layer cannot exact-pass a model-emitted markdown table (rotated /
lane-refused word lists, whole-page OCR, etc.), socr may keep the table only if
Typesafe answers yes to a single question: does the markdown table match the page
image?

If stored words already exact-pass the table, the native verifier ships without
calling Typesafe. Missing API key or a non-yes answer fails closed.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Protocol

import httpx

from socr.tables.native_first import upright_words_for_page
from socr.tables.native_verifier import VerifierState, _verify_from_words, verify_native_table
from socr.tables.reconcile import find_table_blocks

logger = logging.getLogger(__name__)

TYPESAFE_SYSTEMONE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_DEFAULT_MODEL = "jev"
TYPESAFE_TIMEOUT_SEC = 120.0

TABLE_MATCH_QUESTION = (
    "Does this markdown table match the table shown in the page image? Answer yes or no only."
)

_VISION_ENGINES_SKIP = frozenset({"native", "chart_asset", ""})

_YES_RE = re.compile(r"^\s*yes\b", re.IGNORECASE)


class HttpPostFn(Protocol):
    def __call__(self, url: str, *, json: dict, headers: dict, timeout: float) -> object: ...


def typesafe_api_key() -> str | None:
    key = (os.environ.get("TYPESAFE_API_KEY") or "").strip()
    return key or None


def stored_words_match_table(page, markdown: str) -> bool:
    """True when the page's stored word geometry exact-passes *markdown*."""
    verdict = verify_native_table(page, markdown or "")
    return verdict.state == VerifierState.EXACT_PASS and not verdict.hard_fail


def upright_word_check_unsure(page, markdown: str) -> bool:
    """True when upright-frame words do not exact-pass the table."""
    words, _rotation = upright_words_for_page(page)
    if not words:
        try:
            words = list(page.get_text("words") or [])
        except Exception:
            words = []
    if not words:
        return True
    verdict = _verify_from_words(words, markdown or "", scope_label="typesafe-upright")
    if verdict.hard_fail:
        return True
    return verdict.state != VerifierState.EXACT_PASS


def is_vision_model_table_output(engine: str | None) -> bool:
    name = (engine or "").strip()
    if name in _VISION_ENGINES_SKIP:
        return False
    if name.startswith("native"):
        return False
    return True


def needs_typesafe_confirmation(
    page,
    markdown: str,
    *,
    vision_model_output: bool,
) -> bool:
    """Whether a kept vision-model table must be confirmed by Typesafe."""
    if not vision_model_output:
        return False
    if not find_table_blocks(markdown or ""):
        return False
    if stored_words_match_table(page, markdown):
        return False
    # Whole-page model output, or upright geometry cannot exact-pass (Fed p14 class).
    if upright_word_check_unsure(page, markdown):
        return True
    return vision_model_output


def _table_markdown_for_gate(markdown: str) -> str:
    blocks = find_table_blocks(markdown or "")
    if not blocks:
        return ""
    lines = (markdown or "").splitlines()
    chunks: list[str] = []
    for block in blocks:
        chunks.append("\n".join(lines[block.start : block.end + 1]))
    return "\n\n".join(chunks).strip()


def _page_png_base64(page, *, dpi: int = 150) -> str:
    pix = page.get_pixmap(dpi=dpi)
    return base64.standard_b64encode(pix.tobytes("png")).decode("ascii")


def _parse_yes_from_response(payload: object) -> bool | None:
    """Return True/False when a yes/no is explicit; None when unparseable."""
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, str):
        text = payload.strip()
        if _YES_RE.match(text):
            return True
        if re.match(r"^\s*no\b", text, re.IGNORECASE):
            return False
        return None
    if isinstance(payload, dict):
        for key in ("answer", "response", "text", "result", "verdict"):
            if key in payload:
                parsed = _parse_yes_from_response(payload[key])
                if parsed is not None:
                    return parsed
        for key in ("answers", "results"):
            if key in payload:
                items = payload[key]
                if isinstance(items, list) and items:
                    parsed = _parse_yes_from_response(items[0])
                    if parsed is not None:
                        return parsed
        return None
    if isinstance(payload, list) and payload:
        return _parse_yes_from_response(payload[0])
    return None


def typesafe_table_matches_page(
    page,
    markdown: str,
    *,
    model: str = TYPESAFE_DEFAULT_MODEL,
    api_key: str | None = None,
    post_fn: HttpPostFn | None = None,
    timeout_sec: float = TYPESAFE_TIMEOUT_SEC,
) -> bool:
    """Ask Typesafe whether *markdown* matches the rasterized *page*.

    Returns True only on an explicit yes. Every other outcome (no, HTTP error,
    missing key, unparseable body) is False (fail closed).
    """
    table_md = _table_markdown_for_gate(markdown)
    if not table_md:
        return False
    key = (api_key or typesafe_api_key() or "").strip()
    if not key:
        logger.info("typesafe: no API key; failing closed")
        return False
    poster = post_fn or httpx.post
    body = {
        "model": model,
        "state": {
            "image": _page_png_base64(page),
            "markdown_table": table_md,
        },
        "questions": [TABLE_MATCH_QUESTION],
    }
    headers = {"Authorization": f"Bearer {key}"}
    try:
        resp = poster(
            TYPESAFE_SYSTEMONE_URL,
            json=body,
            headers=headers,
            timeout=timeout_sec,
        )
        if hasattr(resp, "raise_for_status"):
            resp.raise_for_status()
        data = resp.json() if hasattr(resp, "json") else resp
    except Exception as exc:
        logger.warning("typesafe: request failed (%s); failing closed", exc)
        return False
    parsed = _parse_yes_from_response(data)
    return parsed is True


@dataclass(frozen=True)
class TypesafeGate:
    """Injectable Typesafe confirmation (tests swap ``post_fn`` / ``api_key``)."""

    model: str = TYPESAFE_DEFAULT_MODEL
    api_key: str | None = None
    post_fn: HttpPostFn | None = None
    timeout_sec: float = TYPESAFE_TIMEOUT_SEC

    def confirm(self, page, markdown: str) -> bool:
        return typesafe_table_matches_page(
            page,
            markdown,
            model=self.model,
            api_key=self.api_key,
            post_fn=self.post_fn,
            timeout_sec=self.timeout_sec,
        )
