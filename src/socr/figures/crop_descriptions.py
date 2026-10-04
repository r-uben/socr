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
from pathlib import PurePosixPath

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
    "Hard rules: write NO digits and NO numbers, neither as numerals nor as words. Do not "
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

# Spelled-out numbers. "one" is deliberately absent: it is overwhelmingly a pronoun
# ("one of the series"), and a count of one cannot be told from it mechanically.
_SPELLED = (
    "zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    "eighty|ninety|hundred|thousand|million|billion|trillion|dozen|double|triple|half|"
    "quarter|percent"
)
_SPELLED_RE = re.compile(rf"\b(?:{_SPELLED})s?\b", re.IGNORECASE)


def is_crop_asset(ref_target: str) -> bool:
    """True iff *ref_target* names a genuine figure crop (by asset kind, not size)."""
    return bool(_CROP_ASSET_RE.match(PurePosixPath(ref_target.split("?", 1)[0]).name))


def find_number_tokens(text: str) -> list[str]:
    """Every digit-like character and spelled-out number in *text* (empty = clean)."""
    found = [ch for ch in text if ch.isnumeric() or unicodedata.category(ch).startswith("N")]
    found += [m.group(0) for m in _SPELLED_RE.finditer(text)]
    return found


def clean_description(raw: str | None) -> str | None:
    """One markdown-safe line, or ``None`` if the text cannot ship as a blockquote."""
    if not raw:
        return None
    text = " ".join(raw.split())
    if not text:
        return None
    # Must not be able to open a table, a code fence, an image or a second quote level.
    if any(tok in text for tok in ("|", "`", "![", "](", "\\")):
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
    """
    if "![" not in body:
        return body
    lines = body.split("\n")
    out: list[str] = []
    in_fence = False
    fence = ""
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in ("```", "~~~"):
            if not in_fence:
                in_fence, fence = True, marker
            elif marker == fence:
                in_fence = False
        out.append(line)
        i += 1
        if in_fence or stripped.startswith("|"):
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
        # Keep a blank line between the description and whatever follows.
        if i < len(lines) and lines[i].strip():
            out.append("")
        changed = True
    return "\n".join(out) if changed else body
