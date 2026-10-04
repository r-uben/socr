"""Number-free descriptions for genuine figure crops.

This is optional enrichment for a citation corpus, so the one rule is: a description
must never carry a value that could be wrong. The model is asked for kind of figure,
axes, series and what is compared; a mechanical validator then rejects any answer
containing a digit or a spelled-out number, retries once, and otherwise DROPS the
description. A dropped description is a missing description, never a wrong one.

Only crops are described. "Crop" is decided from the ASSET KIND (the filename the
pipeline itself gave the asset), never from pixel size: page-sized images
(``chart_page_N``, ``scanned_figure_page``, ``failed_table``, ``page_image``) are
rendered whole pages, and a vision model shown a text page invents a figure for it
(measured: 3 of 12 shipped descriptions on a mixed sample, 0 of 11 on crops only).

Everything here is pure (no I/O, no model); the orchestrator injects the model call.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from socr.engines._figure_prompt import CAPTION_MARKER

# Same wording as the legacy caption marker, with "no values" added: the marker is what
# tells a reader (and a downstream parser) that this prose is not source text.
DESCRIPTION_MARKER = CAPTION_MARKER[:-1] + ", no values]"
DESCRIPTION_PREFIX = f"> *Figure description {DESCRIPTION_MARKER}:*"

# Asset kinds that are genuine figure crops. An allow-list, so an asset kind added later
# is NOT described until someone decides it is a crop.
_CROP_ASSET_RE = re.compile(r"^(?:chart_region_p\d+_\d+|figure_\d+_page\d+)\.png$")

PROMPT = (
    "Write a short alt-text description of this figure from an academic paper, in at most "
    "three sentences. State only: the kind of figure (for example line chart, bar chart, "
    "scatter plot, histogram, table, diagram, photograph); what the axes or dimensions "
    "represent; the names of the series, groups or categories that you can read in the "
    "legend or labels; the frequency or sample if it is named in words; and what is being "
    "compared.\n"
    "Hard rules: write NO digits and NO numbers, neither as numerals nor as words "
    "(no one, two, first, second, once, twice, half, Roman numerals, or percent signs). Do not "
    "state any value, percentage, year, date, count, tick label, rank, magnitude or size of "
    "an effect, and do not say how much anything changes. If a label contains a number, "
    "leave that part out. If you cannot read a label clearly, leave it out instead of "
    "guessing. If the image is not a figure, say so in one sentence. Output only the "
    "description."
)
RETRY_SUFFIX = (
    "\n\nYour previous answer contained numbers or digits ({bad}). Rewrite the description "
    "with no digits and no number words at all."
)

# Identity of the wording the model was given; part of the cache key and the run
# fingerprint, so changing the prompt can never reuse descriptions made under another.
PROMPT_VERSION = hashlib.sha256((PROMPT + RETRY_SUFFIX).encode()).hexdigest()[:12]

# English number words, rejected as whole words (optional plural "s"). The corpus is
# English; other languages are out of scope. "last" is allowed: it carries no value.
# Cardinals including "one" (a harmless pronoun is dropped too: a dropped description is a
# missing description, a shipped count could be wrong), ordinals, multipliers, fractions.
_CARDINALS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    "eighty|ninety|hundred|thousand|million|billion|trillion|dozen|percent|per cent"
)
_ORDINALS = (
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|"
    "thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|"
    "twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|"
    "hundredth|thousandth|millionth|billionth|firstly|secondly|thirdly"
)
_MULTIPLIERS = "once|twice|thrice|double|triple|quadruple|half|halves|quarter|quarters"
_NUMBER_WORD_RE = re.compile(
    rf"\b(?:{_CARDINALS}|{_ORDINALS}|{_MULTIPLIERS})(?:s|ths?)?\b", re.IGNORECASE
)
_SPELLED_RE = _NUMBER_WORD_RE  # name kept for readers of the earlier version

# Standalone uppercase Roman numerals. "I" is the pronoun and is allowed. Single letters
# L, C, D, M are common panel and variable labels ("Panel C"), so only V and X count among
# the single letters; every valid multi-letter numeral (II, III, IV, VI, XII ...) counts.
_ROMAN_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[IVXLCDM]+(?![A-Za-z0-9])")
_ROMAN_VALID_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_VALUE_SYMBOLS = "%\u2030\u2031\u2152\u2189"  # % per-mille per-ten-thousand and two fractions


def _roman_numerals(text: str) -> list[str]:
    out = []
    for m in _ROMAN_TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok == "I" or not _ROMAN_VALID_RE.match(tok):
            continue
        if len(tok) == 1 and tok not in ("V", "X"):
            continue
        out.append(tok)
    return out


def is_crop_asset(ref_target: str) -> bool:
    """True iff *ref_target* names a genuine figure crop (by asset kind, not size)."""
    return bool(_CROP_ASSET_RE.match(PurePosixPath(ref_target.split("?", 1)[0]).name))


def find_number_tokens(text: str) -> list[str]:
    """Every digit-like character and spelled-out number in *text* (empty = clean)."""
    found = [
        ch
        for ch in text
        if ch.isnumeric() or unicodedata.category(ch).startswith("N") or ch in _VALUE_SYMBOLS
    ]
    found += [m.group(0) for m in _NUMBER_WORD_RE.finditer(text)]
    found += _roman_numerals(text)
    return found


def clean_description(raw: str | None) -> str | None:
    """One markdown-safe line, or ``None`` if the text cannot ship as a blockquote."""
    if not raw:
        return None
    text = " ".join(raw.split())
    if not text:
        return None
    # Must not be able to open a table, a code fence, math, an image, a link, raw HTML (and
    # so an HTML comment that would hide later content) or a second quote level.
    if any(tok in text for tok in ("|", "`", "![", "](", "\\", "<", ">", "$")):
        return None
    return text


@dataclass(frozen=True)
class DescriptionOutcome:
    """Result for one figure. ``text`` is set iff the description may ship."""

    text: str | None
    retried: bool
    status: str  # "described" | "dropped"
    reason: str  # "" when described; else why it was dropped
    violations: tuple[str, ...] = ()


def describe_number_free(ask: Callable[[str], str | None]) -> DescriptionOutcome:
    """Run the prompt, validate, retry ONCE, otherwise drop.

    ``ask(prompt)`` returns the model's raw answer or ``None`` when the model is
    unreachable / errored (never raises into the pipeline). The validator is the only
    gate: nothing containing a digit or spelled number is ever returned as ``text``.
    """
    first = ask(PROMPT)
    if first is None:
        return DescriptionOutcome(None, False, "dropped", "model_unavailable")
    cleaned = clean_description(first)
    bad = find_number_tokens(first) if cleaned is None else find_number_tokens(cleaned)
    if cleaned is not None and not bad:
        return DescriptionOutcome(cleaned, False, "described", "")

    named = ", ".join(sorted(set(bad)))[:80] or "markup characters"
    second = ask(PROMPT + RETRY_SUFFIX.format(bad=named))
    if second is None:
        return DescriptionOutcome(None, True, "dropped", "model_unavailable", tuple(bad))
    cleaned2 = clean_description(second)
    bad2 = find_number_tokens(second) if cleaned2 is None else find_number_tokens(cleaned2)
    if cleaned2 is not None and not bad2:
        return DescriptionOutcome(cleaned2, True, "described", "")
    reason = "unsafe_markup" if cleaned2 is None and not bad2 else "number_in_description"
    return DescriptionOutcome(None, True, "dropped", reason, tuple(bad2))


_REF_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

# Page-sized means the crop's bbox covers this fraction of its page or more. Derived from
# measurement, not guessed: over 1183 extracted-figure records in the archive re-OCR, with page
# size read from each source PDF, the genuine crops I inspected reach at most 0.675 of the page
# (a full-width table image), while every whole-page image I inspected (text pages stored as
# ``figure_N_pageP``: 11 of them) covers at least 0.941. The cut sits in that empty gap, at
# its rounded midpoint. A crop with no recorded bbox is never described.
PAGE_SIZED_BBOX_FRACTION = 0.8


def bbox_page_fraction(
    bbox: tuple[float, float, float, float] | list[float] | None,
    page_width: float | None,
    page_height: float | None,
) -> float | None:
    """Fraction of the page a crop's bbox covers, or ``None`` if it cannot be known."""
    if not bbox or not page_width or not page_height or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return area / (page_width * page_height)


def is_page_sized(fraction: float | None) -> bool:
    """True when the crop must not be described: page-sized, or its size is unknown."""
    return fraction is None or fraction >= PAGE_SIZED_BBOX_FRACTION


def has_crop_ref(text: str) -> bool:
    """True iff *text* references at least one crop asset (cheap pre-check)."""
    return any(is_crop_asset(m.group(1)) for m in _REF_RE.finditer(text))


def is_inside(base: Path, target: Path) -> bool:
    """True iff *target* resolves inside *base* (symlinks and ``..`` resolved)."""
    try:
        target.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return False
    return True


def safe_asset_path(doc_dir: Path, target: str) -> Path | None:
    """The file *target* names, only if it is inside ``doc_dir/figures``; else ``None``.

    Rejects absolute paths, any ``..`` segment, URLs and anything whose resolved location
    (symlinks included) leaves this document's figures directory.
    """
    if (
        not target
        or "://" in target
        or target.startswith(("/", "\\"))
        or ":" in target.split("/")[0]
    ):
        return None
    if ".." in PurePosixPath(target.replace("\\", "/")).parts:
        return None
    figures = doc_dir / "figures"
    path = doc_dir / target
    return path if is_inside(figures, path) else None


_FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
_MATH_FENCE_RE = re.compile(r"^\s*(\$\$|\\\[|\\\]|\\begin\{|\\end\{)")


def _tableish(line: str) -> bool:
    """A GFM or borderless table row: a leading pipe, or two or more pipes anywhere."""
    st = line.strip()
    return st.startswith("|") or st.count("|") >= 2


def insert_descriptions(
    body: str,
    describe_ref: Callable[[str], str | None],
) -> str:
    """Insert ``> *Figure description ...* text`` directly under each crop image ref.

    ``describe_ref(target)`` is called once per crop ref that is not already followed by a
    description and returns the validated text, or ``None`` to leave the ref alone.
    Idempotent (a ref already followed by a description line is skipped, so a resumed
    page is not re-described), and byte-preserving when nothing is inserted: the input
    string is returned unchanged.

    A ref is SKIPPED, never described, when inserting beside it could change how other content
    renders: inside a fenced or indented code block, a display-math block, an HTML comment, or
    a table (a row with a pipe, bordered or not, or a line directly continuing one: GFM turns
    that line into a row).
    """
    if "![" not in body:
        return body
    lines = body.split("\n")
    out: list[str] = []
    fence = ""  # the opening fence marker while inside a fenced block
    in_math = False
    in_comment = False
    changed = False
    prev_tableish = False
    prev_blank = True
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        blank = not line.strip()

        skip = False
        fm = _FENCE_RE.match(line)
        if fence:
            skip = True
            if fm and fm.group(1)[0] == fence[0] and len(fm.group(1)) >= len(fence):
                fence = ""
        elif fm:
            fence = fm.group(1)
            skip = True
        if not skip and _MATH_FENCE_RE.match(line):
            # ``$$ ... $$`` on one line is complete; a lone opener/closer toggles the block.
            if not (
                line.strip().startswith("$$")
                and line.strip().endswith("$$")
                and len(line.strip()) > 2
            ):
                in_math = not in_math
            skip = True
        if in_math:
            skip = True
        if in_comment or "<!--" in line:
            skip = True
            if "<!--" in line and "-->" not in line.split("<!--", 1)[1]:
                in_comment = True
            if "-->" in line:
                in_comment = False
        tableish = _tableish(line)
        indented_code = prev_blank and (line.startswith("    ") or line.startswith("\t"))
        if tableish or (prev_tableish and not blank) or indented_code:
            skip = True
        prev_tableish = tableish or (prev_tableish and not blank)
        prev_blank = blank
        if skip:
            continue

        targets = [m.group(1) for m in _REF_RE.finditer(line)]
        crops = [t for t in targets if is_crop_asset(t)]
        if not crops:
            continue
        # Already described (resume / second pass)? Look past blank lines.
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and lines[j].startswith(DESCRIPTION_PREFIX):
            continue
        texts = [t for t in (describe_ref(c) for c in crops) if t]
        if not texts:
            continue
        for text in texts:
            out.extend(["", f"{DESCRIPTION_PREFIX} {text}"])
        # Keep a blank line between the description and whatever follows, and make sure the
        # next line cannot lazily continue the blockquote.
        if i < len(lines) and lines[i].strip():
            out.append("")
        changed = True
    return "\n".join(out) if changed else body
