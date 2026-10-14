"""#1050 step 2: read the words a PDF draws as shapes, and place them honestly.

Step 1 (``socr.core.text_coverage``) finds text-like ink with no visible
text-layer words over it, groups it into regions and saves each region as a
PNG. This module turns one such crop into text.

Four rules the owner set, encoded here:

* **The page never loses its native text.** Nothing in this module edits the
  native prose; it only inserts blocks.
* **A model reading never silently replaces anything.** Every recovered block
  sits in a blockquote under a visible mark, so it can never be read -- or
  grepped -- as native prose.
* **Nothing is confirmed.** Owner ruling, 2026-10-14. Confirmation by token
  containment kept passing wrong readings through three review rounds: a
  label/value swap (content is an unordered set), dash-led numbers that fall
  out of every bucket, and glued tokens (``pval=0.05`` vs ``pval=0.99``). So
  every reading ships under the SAME ``OCR, not verified`` mark, beside its
  picture, and the page mark never implies a check happened.
* **Every recovered region keeps its picture**, so a reader can always settle
  the words against the image. A region with no reading at all is exactly step
  1: picture and mark, no text.

Pure: no model client, no I/O. The orchestrator injects ``ask`` (the same
``OllamaFigureEngine``-backed callable ``socr.figures.crop_descriptions`` uses).
"""

from __future__ import annotations

import re

PROMPT = (
    "Transcribe the text in this image exactly as it is printed. Keep the reading order "
    "and keep each printed line on its own line.\n"
    "Hard rules: copy the words and numbers verbatim. Do not translate, summarise, "
    "explain, complete, correct or comment on anything. Do not describe the image. Do not "
    "add a heading, a caption or any word that is not printed in the image. If part of the "
    "image is unreadable, leave it out rather than guessing. If the image contains no "
    "readable text, reply with exactly NO TEXT.\n"
    "Output only the transcription."
)

#: The model's own "nothing here" answer.
EMPTY_REPLY = "NO TEXT"

#: Marks. Blockquotes, like ``UNVERIFIED_TABLE_MARK`` and the figure-description
#: marker, so they read as an aside in rendered Markdown and a consumer can
#: count them with one grep. The in-figure wording is fixed by the 2026-10-09
#: design page (``docs/plans/routing/README.md``, Target 1).
#: Owner ruling 2026-10-14: there is ONE mark, and it says the reading was not
#: verified. Three review rounds found token-containment "confirmation" passing
#: wrong readings -- a label/value swap, dash-led numbers, glued tokens -- so
#: nothing ships as confirmed and the page mark never implies a check happened.
UNVERIFIED = "OCR, not verified"
FIGURE_TEXT_PREFIX = f"Text in figure ({UNVERIFIED})"
PAGE_TEXT_PREFIX = f"Text drawn as shapes ({UNVERIFIED})"
RECOVERED_MARK_RE = re.compile(
    rf"^> (?:{re.escape(FIGURE_TEXT_PREFIX)}|{re.escape(PAGE_TEXT_PREFIX)}):",
    re.MULTILINE,
)
#: Any mark that would claim a reading was checked. Nothing may ever emit one.
CONFIRMED_STYLE_RE = re.compile(r"\(OCR\)|unconfirmed|recovered by OCR", re.IGNORECASE)

# Exact lead-ins the model uses to introduce a transcription. These are STRIPPED,
# never used to drop a reading (round 3, Fable #6): the old rule discarded any
# answer opening "I ", "It appears" or "Here is", and a figure that genuinely
# begins "It appears in Table 3" or a label starting with "I" is ordinary page
# text. Dropping a real reading on a regex is silent content loss; shipping it
# under the same "(OCR, not verified)" mark is not.
_LEAD_IN_RE = re.compile(
    r"^\s*(?:here(?:'s| is| are)(?: the)?[^:\n]{0,40}:"
    r"|the (?:text|transcription)(?: in the image)?(?: is| reads)?:"
    r"|transcription:)\s*",
    re.IGNORECASE,
)

# A reply that is a REFUSAL, not a transcription at all. Only these drop the
# reading, because there is nothing in them to ship.
_REFUSAL_RE = re.compile(
    r"^\s*(?:i'm sorry\b|i am sorry\b|sorry[,.]|i can(?:'t|not)\b|"
    r"i'm unable\b|i am unable\b|as an ai\b|unfortunately,? i\b)",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^\s*```[^\n]*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def clean_reading(raw: str | None) -> str | None:
    """Strip fences and reject non-transcriptions. ``None`` means "no reading"."""
    if not raw:
        return None
    text = raw.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text or text.strip().upper() == EMPTY_REPLY:
        return None
    if _REFUSAL_RE.match(text):
        return None
    # A lead-in is stripped and the rest kept: the words after it are the
    # transcription, and throwing them away loses page content.
    text = _LEAD_IN_RE.sub("", text, count=1).strip()
    if not text:
        return None
    # ``text`` is already stripped, so only per-line trailing space is left.
    return "\n".join(ln.rstrip() for ln in text.splitlines()) or None


def recovered_block(text: str, *, in_figure: bool) -> str:
    """The Markdown block a reading ships as. There is only one form.

    A blockquote so the mark and the recovered words cannot be mistaken for
    native prose. The one-line form matches the design page's example; a
    multi-line reading keeps its printed line breaks under the same mark.
    """
    prefix = FIGURE_TEXT_PREFIX if in_figure else PAGE_TEXT_PREFIX
    lines = text.splitlines()
    if len(lines) == 1:
        return f"> {prefix}: {lines[0]}"
    body = "\n".join(f"> {ln}" if ln.strip() else ">" for ln in lines)
    return f"> {prefix}:\n>\n{body}"


def insert_after_anchor(body: str, anchor: str, block: str) -> tuple[str, bool]:
    """Insert *block* as its own paragraph after the line *anchor* ends on.

    *anchor* is the tail of the native text block that precedes the region on
    the page, so the recovered words land at their reading-order position
    (2026-10-09 ruling). The match is whitespace-tolerant because the native
    extractor rewraps and rejoins lines; it is anchored on the LINE the anchor
    ends on, not on a blank-line paragraph, because a shipped native page
    separates its printed lines with single newlines and can be one "paragraph"
    end to end.

    An empty anchor means nothing precedes the region: the block goes first. A
    ``False`` second element means the anchor was not found, and the caller must
    fall back to the page tail -- never guess a position.
    """
    if not block:
        return body, False
    if _already_present(body, block):
        return body, True
    if not anchor.strip():
        # Item D: nothing precedes the region, so the block goes first -- but
        # AFTER any part already placed there, or a second region above all
        # native text ships before the first and reverses page order.
        cut = _after_placed_parts(body, 0)
        if cut <= 0:
            return (block + "\n\n" + body if body else block), True
        return (body[:cut].rstrip("\n") + "\n\n" + block + "\n\n" + body[cut:].lstrip("\n")), True
    # Lookarounds, not \b: without them a short anchor ("is a") matches inside
    # a native word ("this apple"), but \b itself fails whenever the anchor
    # ends on punctuation -- and a sentence tail usually does.
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(w) for w in anchor.split()) + r"(?!\w)"
    # Matches inside a blockquote are skipped: an earlier region's recovered
    # block is already in the body and must never anchor the next one. An
    # anchor that still matches more than once does not determine a position
    # (a running header repeats), so it falls back rather than guessing.
    found = [m for m in re.finditer(pattern, body) if not _in_quote(body, m.start())]
    if len(found) != 1:
        return body, False
    match = found[0]
    cut = _after_placed_parts(body, body.find("\n", match.end()))
    if cut == -1:
        return (body.rstrip() + "\n\n" + block), True
    tail = body[cut:]
    rest = tail.lstrip("\n")
    # Keep whatever separation already followed that line, but never less than a
    # blank line: the block must read as its own paragraph, not as the next one.
    after = "\n" * max(2, len(tail) - len(rest))
    return (body[:cut] + "\n\n" + block + after + rest), True


def _already_present(body: str, block: str) -> bool:
    """True iff *body* already contains *block* as whole paragraphs.

    Two failures this has to avoid at once:

    * plain ``block in body`` drops a second region whose reading is a PREFIX
      of the first's ("… Total 100" inside "… Total 100 USD"): the caller reads
      the True as "placed" and skips that region's picture too;
    * matching a single paragraph misses a block that spans several, which the
      real piece does (a reading plus its picture). Applying twice then
      inserted it twice (round 3, Argon #8).

    So the comparison is over a RUN of paragraphs, each matched whole.
    """
    wanted = [para.strip() for para in block.split("\n\n") if para.strip()]
    if not wanted:
        return False
    have = [para.strip() for para in body.split("\n\n") if para.strip()]
    span = len(wanted)
    return any(have[i : i + span] == wanted for i in range(len(have) - span + 1))


#: An image ref step 2 placed, as ``text_coverage.uncovered_text_image_ref``
#: writes it. A placed PART is a recovered block OR its picture: item E found
#: that skipping only blockquotes left two regions as B1, B2, pic2, pic1,
#: detaching each picture from the words it belongs to.
_PLACED_IMAGE_RE = re.compile(r"^!\[Uncovered text page \d+ region \d+\]\(")


def _is_placed_part(paragraph: str) -> bool:
    """True when *paragraph* is a block or picture step 2 already placed."""
    head = paragraph.lstrip()
    return head.startswith(">") or bool(_PLACED_IMAGE_RE.match(head))


def _after_placed_parts(body: str, cut: int) -> int:
    """Move *cut* past any parts step 2 already placed at that point.

    Two regions under one anchor would otherwise both land at the same cut, and
    the second would be inserted BEFORE the first, reversing their reading
    order (item D for an empty anchor, item E for a shared one) and splitting a
    block from its own picture.
    """
    while cut != -1:
        rest = body[cut:].lstrip("\n")
        if not _is_placed_part(rest):
            return cut
        advanced = body.find("\n", cut + (len(body[cut:]) - len(rest)))
        while advanced != -1 and body[advanced + 1 :].lstrip(" ").startswith(">"):
            advanced = body.find("\n", advanced + 1)
        if advanced == -1:
            return -1
        cut = advanced
    return cut


def _in_quote(body: str, pos: int) -> bool:
    """True when *pos* sits on a Markdown blockquote line (a recovered block)."""
    line_start = body.rfind("\n", 0, pos) + 1
    return body[line_start:pos].lstrip().startswith(">")
