"""Output guards for the Gemma cloud OCR rung."""

from __future__ import annotations

import re

from socr.core.result import FailureMode, PageOutput, PageStatus

# Approximate chars per token for post-hoc truncation when the CLI has no cap flag.
_CHARS_PER_TOKEN = 4

_IMAGE_URL_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)]+)\)", re.IGNORECASE)
_EQUATION_RE = re.compile(
    r"\$\$[\s\S]+?\$\$|\$[^$\n]+\$|\\\(|\\\[|```(?:math|latex)",
    re.IGNORECASE,
)


def max_output_chars(max_tokens: int) -> int:
    return max(256, int(max_tokens) * _CHARS_PER_TOKEN)


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    limit = max_output_chars(max_tokens)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[truncated: gemma output token cap]"


def has_repeated_text_loop(text: str, *, min_repeats: int = 3) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < min_repeats:
        return False
    run = 1
    prev = lines[0]
    for ln in lines[1:]:
        if ln == prev:
            run += 1
            if run >= min_repeats:
                return True
        else:
            prev = ln
            run = 1
    return False


def has_image_url_loop(text: str, *, min_repeats: int = 3) -> bool:
    urls = _IMAGE_URL_RE.findall(text)
    if len(urls) < min_repeats:
        return False
    from collections import Counter

    return any(c >= min_repeats for c in Counter(urls).values())


def flag_equations_unverified(text: str) -> tuple[str, list[str]]:
    if not _EQUATION_RE.search(text):
        return text, []
    note = "gemma equations unverified — verify against source image before citing"
    if note in text:
        return text, [note]
    return f"{text.rstrip()}\n\n> [{note}]\n", [note]


def apply_gemma_output_guards(
    text: str,
    *,
    page_num: int,
    max_output_tokens: int,
) -> PageOutput:
    """Validate Gemma OCR text; return SUCCESS or ERROR ``PageOutput``."""
    cleaned = truncate_to_token_budget(text, max_output_tokens)
    if has_image_url_loop(cleaned):
        return PageOutput(
            page_num=page_num,
            text="",
            status=PageStatus.ERROR,
            engine="gemma",
            failure_mode=FailureMode.GARBAGE,
            error="gemma output rejected: repeated image URL loop",
            audit_passed=False,
        )
    if has_repeated_text_loop(cleaned):
        return PageOutput(
            page_num=page_num,
            text="",
            status=PageStatus.ERROR,
            engine="gemma",
            failure_mode=FailureMode.GARBAGE,
            error="gemma output rejected: repeated text loop",
            audit_passed=False,
        )
    flagged_text, notes = flag_equations_unverified(cleaned)
    return PageOutput(
        page_num=page_num,
        text=flagged_text,
        status=PageStatus.SUCCESS,
        engine="gemma",
        audit_passed=True,
        audit_notes=notes,
    )
