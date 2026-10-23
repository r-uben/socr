"""Detect and strip model OCR preamble loops with no document content.

Some VLMs answer with a short English lead-in (often mentioning "transcription"
and "Markdown") repeated with ``---`` separators and no page text. Those readings
must fail audit and escalate; a reading that only prefixes real content with the
same lead-in should ship with the prefix removed.
"""

from __future__ import annotations

import re

#: Standalone horizontal-rule lines (model filler, not GFM setext headings).
_SEPARATOR_LINE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def _normalize_block(text: str) -> str:
    return " ".join(text.split())


def _is_preamble_line(line: str) -> bool:
    """True for model lead-in lines at the start of OCR output."""
    stripped = line.strip()
    if not stripped:
        return True
    if _SEPARATOR_LINE_RE.match(stripped):
        return True
    lower = stripped.lower()
    if lower.startswith(("the following", "here is", "here's", "here are", "below is")):
        if "transcription" in lower or "transcribed" in lower:
            return True
    if lower.startswith("transcription") and ("markdown" in lower or "image" in lower):
        return True
    return False


def _is_preamble_block(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return True
    return all(_is_preamble_line(ln) for ln in lines)


def _split_blocks(text: str) -> list[str]:
    """Split on blank lines and standalone ``---`` rules."""
    if not text or not text.strip():
        return []
    chunks = re.split(r"\n\s*---\s*\n|\n\n+", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def strip_leading_ocr_boilerplate(text: str) -> str:
    """Remove exact preamble-shaped lines and leading separators at the start only."""
    if not text:
        return text
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if _is_preamble_line(stripped):
            i += 1
            continue
        break
    body = "\n".join(lines[i:])
    return body.lstrip("\n") if body else ""


def collapse_exact_repeated_blocks(text: str) -> str:
    """Drop consecutive duplicate blocks; collapse an all-identical repetition."""
    blocks = _split_blocks(text)
    if not blocks:
        return ""
    if all(_is_preamble_block(b) for b in blocks):
        return ""
    normed = [_normalize_block(b) for b in blocks]
    if len(set(normed)) == 1 and _is_preamble_block(blocks[0]):
        return ""
    deduped: list[str] = []
    prev: str | None = None
    for block, key in zip(blocks, normed):
        if key == prev:
            continue
        deduped.append(block)
        prev = key
    return "\n\n".join(deduped)


def substantive_ocr_text(text: str) -> str:
    """Text that should count as page content after boilerplate removal."""
    from socr.core.blank_page import blank_page_marker

    if not text or not text.strip():
        return ""
    if text.strip() == blank_page_marker().strip():
        return text.strip()
    stripped = strip_leading_ocr_boilerplate(text)
    collapsed = collapse_exact_repeated_blocks(stripped)
    return collapsed.strip()


def is_boilerplate_only_output(text: str) -> bool:
    """True when *text* has no document content after boilerplate normalization."""
    return not substantive_ocr_text(text)


def leading_preamble_to_strip(text: str) -> bool:
    """True when the raw text begins with removable preamble (but may have body)."""
    if not text:
        return False
    stripped = strip_leading_ocr_boilerplate(text)
    return stripped != text.strip() and bool(stripped.strip())
