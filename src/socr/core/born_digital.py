"""Born-digital PDF detection and native text extraction.

Detects pages with genuine native text layers (born-digital) vs scanned pages
that may have low-quality baked-in OCR. Born-digital pages can skip OCR entirely
and use the extracted text directly.

Heuristics for distinguishing born-digital from baked-in OCR:
  - Text density: born-digital pages have consistent, dense text relative to
    page area; garbage OCR layers tend to be sparse or garbled.
  - Character quality: genuine text has normal word-length distributions and
    low ratios of non-ASCII/garbage characters.
  - Font consistency: born-digital PDFs embed proper fonts; scanned PDFs with
    OCR layers often have no real font info or use a single "invisible" font.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from socr.core.glyph_recovery import GlyphRepairReport
from socr.core.pdf import apply_glyph_recovery, open_pdf

logger = logging.getLogger(__name__)

# Font families used exclusively for typeset mathematics in LaTeX/LuaTeX/XeTeX.
# These appear in basefont names regardless of whether PyMuPDF's text extraction
# still looks like math — which it often doesn't (subscripts flattened, Greek
# dropped, reading order broken).  A single match means the page has real math.
#
# Uses re.search (not re.match) to handle subset-prefixed names like
# "ABCDEF+CMMI10".
_MATH_FONT_RE = re.compile(
    r"(?i)(CMMI|CMSY|CMEX|MSAM|MSBM|"  # Computer Modern + AMS math
    r"STIXMath|STIXSize|STIXNonUnicode|XITSMath|LatinModernMath|LMMath|"  # STIX/OpenType math
    r"AsanaMath|LibertinusMath|CambriaMath|NewCMMath|"  # other OTF math families
    r"Euler|rsfs)"  # Euler script, RSFS (calligraphic)
)

# Math typeset with a broken font/ToUnicode map decodes operators and delimiters
# as Latin-1 letters and fraction glyphs: '=' -> '¼', '(' -> 'ð', ')' -> 'Þ',
# '+' -> 'þ', '-' -> a C0 control char. These glyphs do not occur in clean
# English or math prose, so even a few per line are a near-perfect signal of
# corrupted math — which the prose-oriented `_encoding_corruption_ratio`
# (mid-caps / slash-digits / run-on words) misses entirely. Greek (σ, ρ) and real
# typography (en-dash, U+2212 minus, ﬁ ligature, curly quotes) are deliberately
# NOT in this set: they are legitimate and must never trip math-corruption.
_MATH_MOJIBAKE_CHARS = frozenset("ðÞþÐ¼½¾")
# C0 control characters (excluding tab/newline/CR) — '-' and other operators
# decode to these on this font class.
_C0_CONTROL_CHARS = frozenset(chr(c) for c in range(0x20) if c not in (0x09, 0x0A, 0x0D))
_MATH_CORRUPTION_GLYPHS = _MATH_MOJIBAKE_CHARS | _C0_CONTROL_CHARS


def line_has_corrupt_math(line: str) -> bool:
    """True if a line contains font-map mojibake standing in for math symbols."""
    return any(ch in _MATH_CORRUPTION_GLYPHS for ch in line)


#: Left-to-right text. Exempt from relegation on every page — see
#: ``block_is_page_furniture``.
_HORIZONTAL = (1.0, 0.0)

#: Decimal places used when comparing PyMuPDF line directions. They are unit
#: vectors, so three places distinguish the axis-aligned cases (1,0) / (0,1) /
#: (0,-1) while absorbing float noise in text set at a slight angle.
_DIR_PRECISION = 3


def dominant_text_direction(blocks: list) -> tuple[float, float]:
    """The page's prevailing writing direction, from its own line dirs.

    Derived from the page, never assumed to be horizontal: a rotated scan or a
    CJK vertical layout has a different prevailing direction, and hard-coding
    ``(1, 0)`` would classify every line on such a page as furniture.
    """
    tally: dict[tuple[float, float], int] = {}
    for block in blocks:
        for line in block.get("lines", []) or []:
            d = line.get("dir", (1.0, 0.0))
            key = (round(float(d[0]), _DIR_PRECISION), round(float(d[1]), _DIR_PRECISION))
            tally[key] = tally.get(key, 0) + 1
    if not tally:
        return (1.0, 0.0)
    return max(tally.items(), key=lambda kv: kv[1])[0]


def block_is_page_furniture(block: dict, dominant: tuple[float, float]) -> bool:
    """True when a block runs against the page's prevailing text direction.

    Publisher stamps ride the page margin rotated 90 degrees — "Downloaded from
    https://academic.oup.com/..." on every page of a JSTOR/OUP download. Spliced
    into prose they wreck its reading order.

    Callers RELEGATE such blocks to the end of the page; they must not delete
    them. Rotated text is not proof of worthlessness — a chart's axis labels and a
    rotated table header are also counter-directional, and are content. Direction
    is evidence about layout only (#145 review).

    Direction is the discriminator because it is a fact about the block rather
    than a guess about its words: matching on "Downloaded from" would be a
    publisher-specific string list that rots, while a stamp set at 90 degrees to
    the body text is structurally distinguishable on any page, in any language.

    A block with no directional lines is NOT furniture — absence of evidence must
    not delete text (#145).

    HORIZONTAL TEXT IS NEVER FURNITURE, even when it is the minority direction on
    its page. A page whose table is set landscape has a ROTATED dominant
    direction, and without this exemption the rule deletes the one horizontal
    block on it — which on the reference document is the running head carrying
    the page number, and a page number is citable content. Suppressing a stray
    publisher stamp is worth little; deleting a page number is precisely the loss
    this module exists to prevent, so the asymmetry is deliberate.
    """
    dirs = [
        (
            round(float(line.get("dir", dominant)[0]), _DIR_PRECISION),
            round(float(line.get("dir", dominant)[1]), _DIR_PRECISION),
        )
        for line in block.get("lines", []) or []
    ]
    if not dirs:
        return False
    return all(d != dominant and d != _HORIZONTAL for d in dirs)


def upright_rotation_degrees(direction: tuple[float, float]) -> int:
    """Degrees to rotate a page so its dominant text reads horizontally.

    DERIVED from the page's own writing direction, never guessed. ``direction``
    is the unit vector ``dominant_text_direction`` returns, so the angle that
    restores it to ``(1, 0)`` is simply its negated bearing. A guess is not an
    acceptable substitute: on the reference document ``(0, -1)`` needs 90 and
    ``prerotate(270)`` yields an upside-down page, which is no better for a model
    than the sideways original.

    Returns 0 for horizontal text and for the all-zero vector, matching
    :func:`text_direction_is_rotated`: absence of directional evidence must not
    cause a rotation (#145).

    Snapped to a right angle. Text set at a slight skew shares this code path,
    and rotating by an arbitrary angle would resample the glyphs for no gain --
    the model tolerates a few degrees of skew, and only the quadrant matters.
    """
    import math

    dx, dy = float(direction[0]), float(direction[1])
    if dx == 0.0 and dy == 0.0:
        return 0
    bearing = math.degrees(math.atan2(dy, dx))
    return int(round(-bearing / 90.0) * 90) % 360


def upright_rotation_for(page, clip=None) -> int:
    """Rotation in degrees making ``page``'s text read horizontally.

    With ``clip=None``, inspects the whole page's text direction. With a clip
    (a fitz.Rect or tuple bbox), inspects text direction within that region.

    When a clip is provided:
    - If the clip contains text lines, returns the rotation for the clip's
      dominant direction, ignoring the page-wide direction.
    - If the clip contains no text lines, falls back to the page-level rotation.
    - If clipped inspection raises an exception, returns 0 immediately (fail-open)
      rather than falling back to the page-derived rotation.

    Returns 0 whenever the page/clip is uninspectable, has no directional
    evidence, or cannot be inspected -- rendering unrotated is the status quo, so
    a failure here must never be worse than not having tried (fail-open by design).
    """
    try:
        if clip is None:
            blocks = page.get_text("dict").get("blocks", [])
        else:
            blocks = page.get_text("dict", clip=clip).get("blocks", [])
    except Exception:
        if clip is not None:
            return 0
        else:
            return 0

    if clip is not None:
        has_clipped_line = False
        for block in blocks:
            for line in block.get("lines", []) or []:
                text = "".join(span.get("text", "") for span in line.get("spans", []) or [])
                if text.strip():
                    has_clipped_line = True
                    break
            if has_clipped_line:
                break

        if not has_clipped_line:
            try:
                blocks = page.get_text("dict").get("blocks", [])
            except Exception:
                return 0

    if not blocks:
        return 0

    return upright_rotation_degrees(dominant_text_direction(blocks))


def text_direction_is_rotated(direction: tuple[float, float]) -> bool:
    """True when a page's dominant text direction runs off the horizontal axis.

    ``abs(dy) >= abs(dx)`` is an axis comparison — which axis the text run lies
    closer to — not a numeric threshold; it reuses the same unit-vector line
    directions as ``dominant_text_direction`` and introduces no new constant.

    Ties (45 degrees, where ``abs(dx) == abs(dy)``) fail closed to rotated: the
    y-clustering rowizer that assembles reading order from line positions is
    provably wrong at 45 degrees, so a tied direction must not be treated as
    safely horizontal.

    The exact all-zero vector ``(0.0, 0.0)`` stays horizontal, per the
    absence-of-evidence precedent (#145, see ``block_is_page_furniture`` above:
    "A block with no directional lines is NOT furniture — absence of evidence
    must not delete text"). A page with no directional evidence at all must not
    be routed as rotated on the strength of that absence.
    """
    if direction == (0.0, 0.0):
        return False
    return abs(direction[1]) >= abs(direction[0])


def rotated_text_is_shredded(blocks: list, direction: tuple[float, float]) -> bool:
    """True when a rotated page's extracted lines are pieces of ONE text run.

    #263. On a rotated page the extractor sometimes breaks a single caption
    into one "line" per glyph run and emits them in y-order, i.e. reversed.
    The reference page (Kaminska-Mumtaz-Sustek p38) ships 177 characters over
    47 lines, 32 of them two characters or fewer::

        MC / O / F / round / a / ields / y / n / i / anges / h / c / requency

    Read off that page's own ``get_text("dict")``, every one of those lines
    carries ``dir=(0.0, -1.0)`` and ``bbox`` x-extent ``491.8 .. 503.7`` --
    the SAME column -- with y-extents that butt directly against each other
    (``94.5..113.7``, ``113.7..122.8``, ``122.5..130.1``, ...). They are not
    lines. They are one line the extractor cut up, and the cuts are visible in
    the geometry: two genuine lines of rotated text sit side by side in
    *different* columns and each spans the column's whole length, whereas two
    pieces of one run share a column and are adjacent along it.

    So the predicate compares each break against the page's own type size:

    * two lines are considered at all only when their extents PERPENDICULAR to
      the text direction overlap -- they sit on the same baseline column;
    * the break between them is **spurious** when their separation ALONG the
      text direction is non-negative (they do not overlap, so they are not
      stacked lines) and no larger than the thinner line's own perpendicular
      extent, which is that line's glyph height on this page. A gap smaller
      than one glyph height on a shared baseline is an inter-word space, not a
      line break;
    * every other pair is a genuine break.

    The verdict is ``spurious > genuine``: a comparison between two counts the
    page itself produces. There is no threshold to tune, and nothing here is
    calibrated against a corpus.

    A fraction-of-short-lines test was measured and REJECTED. Over the 11
    rotated born-digital pages in the reference corpus it cannot separate the
    defect from a chart: Pflueger-Rinaldi p37 extracts 67 short lines out of 95
    and is perfectly sound -- they are axis tick labels (``0 5 10 15 20``) --
    while its caption reads back verbatim. This predicate fires on 1 of those
    11 pages, the one known-damaged page, and on none of the other 10.

    ``blocks`` is ``page.get_text("dict")["blocks"]``; ``direction`` is the
    page's dominant text direction. Fewer than two comparable lines means no
    break to judge, and the answer is False -- the same absence-of-evidence
    rule the rest of this module follows (#145).
    """
    along = 1 if abs(direction[1]) >= abs(direction[0]) else 0
    perp = 1 - along
    spans: list[tuple[float, float, float, float]] = []
    for block in blocks or []:
        for line in block.get("lines", []) or []:
            bbox = line.get("bbox")
            if not bbox:
                continue
            text = "".join(span.get("text", "") for span in line.get("spans", []) or [])
            if not text.strip():
                continue
            line_dir = (
                round(float(line.get("dir", direction)[0]), _DIR_PRECISION),
                round(float(line.get("dir", direction)[1]), _DIR_PRECISION),
            )
            if line_dir != direction:
                continue
            spans.append((bbox[perp], bbox[perp + 2], bbox[along], bbox[along + 2]))
    if len(spans) < 2:
        return False
    # Column first, then position down the column: consecutive entries are the
    # pairs a reader would see as adjacent.
    spans.sort(key=lambda s: ((s[0] + s[1]) / 2.0, s[2]))
    spurious = genuine = 0
    for (p0, p1, a0, a1), (q0, q1, b0, b1) in zip(spans, spans[1:]):
        if min(p1, q1) - max(p0, q0) <= 0:
            genuine += 1  # different columns: a real line break
            continue
        separation = max(b0 - a1, a0 - b1)
        glyph_height = min(p1 - p0, q1 - q0)
        if 0 <= separation <= glyph_height:
            spurious += 1
        else:
            genuine += 1
    return spurious > genuine


#: Word tokens used to decide whether a region's markdown already contains a
#: line. Letters and numbers only: punctuation and the markdown table scaffolding
#: (pipes, dashes) must not count as content.
_WORD_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+(?:\.\d+)?")

#: Fraction of a text block that must lie inside a table region before the
#: region is treated as already representing it. Set at the midpoint because that
#: is what "the region contains this block" means — most of the block is inside
#: it. Not tuned: anything above 0.5 is a block the region genuinely covers, and
#: anything below is a block that merely brushes the region's edge. The old
#: behaviour was effectively a threshold of zero, which deleted a paragraph over
#: a single point of contact (#145).
_REGION_COVERAGE_DROP = 0.5


def _region_token_index(regions) -> Counter:
    """Multiset of word tokens across every region's markdown replacement."""
    idx: Counter = Counter()
    for _rect, md in regions:
        idx.update(_WORD_TOKEN_RE.findall(md or ""))
    return idx


def _line_is_in_region_text(line: dict, region_tokens: Counter) -> bool:
    """True when a region's markdown already contains this line's words.

    The test for "already represented", replacing the geometric one. A line whose
    every token appears in the replacement text is safe to drop; a line carrying
    even one token the replacement lacks is content that would be lost, and is
    kept — duplicated at worst.

    A line with no word tokens (pure punctuation, a rule) is droppable: it
    carries nothing to lose.
    """
    text = "".join(s.get("text", "") for s in line.get("spans", []) or [])
    tokens = _WORD_TOKEN_RE.findall(text)
    if not tokens:
        return True
    return all(region_tokens.get(t, 0) > 0 for t in tokens)


def _rect_coverage(inner, outer) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer`` (0.0-1.0)."""
    area = inner.get_area()
    if area <= 0:
        return 0.0
    overlap = inner & outer
    if not overlap.is_valid or overlap.is_empty:
        return 0.0
    return overlap.get_area() / area


#: Fraction of a span that must lie inside a link rectangle before the span is
#: treated as carrying that link. Same midpoint reasoning as
#: ``_REGION_COVERAGE_DROP`` above: most of the span is inside the rectangle, so
#: the link genuinely covers it. A link rect is drawn around its anchor text by
#: the producing application, so partial overlap is the ordinary case at the
#: edges of a wrapped line, not evidence of a different link.
_LINK_COVERAGE_MIN = 0.5


def _uri_links(page) -> list[tuple[object, str, str, tuple[int, ...]]]:
    """External URI links on a page, as ``(rect, uri, anchor_text)``.

    GH-127: ``page.get_links()`` was never called, so every hyperlink in the text
    layer was dropped — 1,103 of them across half the sampled corpus. For a
    citation corpus a DOI present in the source and absent from the output is
    outright content loss, not formatting loss.

    ``anchor_text`` is resolved from the page's WORDS, not its spans. PyMuPDF
    routinely returns a whole uniform-font line as a single span, so a link drawn
    around a phrase inside that line ("Smith 2020" in "See Smith 2020 for
    details.") covers only a fraction of the span and matches nothing at span
    granularity. Words carry their own boxes, so the anchor is recoverable
    exactly; wrapping then happens by substring, which leaves every original
    character untouched and inserts only the markdown brackets.

    Internal links (``LINK_GOTO`` page jumps, named destinations) are excluded:
    they address a position in the PDF, not a resource, and have no meaning once
    the page is markdown. Guarded like every other page access in this module —
    a damaged link table must degrade to "no links", never raise out of
    extraction (#145's absence-of-evidence precedent).
    """
    try:
        raw = page.get_links() or []
    except Exception:
        return []
    if not raw:
        return []
    try:
        words = page.get_text("words") or []
    except Exception:
        words = []

    out: list[tuple[object, str, str, tuple[int, ...]]] = []
    for link in raw:
        # One damaged entry must not cost the page every other link. The whole
        # per-link body is inside the boundary -- URI, rectangle AND word coverage
        # -- because a malformed word tuple raised from the coverage scan, outside
        # the narrower guard this replaces (GH-127 review). Extraction degrades to
        # fewer links, never to an exception.
        try:
            uri = (link.get("uri") or "").strip()
            if not uri:
                continue
            rect = fitz.Rect(link["from"])
            if rect.is_empty or not rect.is_valid:
                continue
            covered = [
                (i, w[4])
                for i, w in enumerate(words)
                if _rect_coverage(fitz.Rect(w[0], w[1], w[2], w[3]), rect) >= _LINK_COVERAGE_MIN
            ]
        except Exception:
            continue
        anchor = " ".join(t for _i, t in covered).strip()
        idxs = [i for i, _t in covered]
        out.append((rect, uri, anchor, tuple(idxs)))
    return out


def _word_char_spans(text: str, words: list) -> dict[int, tuple[int, int]]:
    """Map each word index to its ``(start, end)`` offsets in the flat page text.

    GH-127 review. Anchors used to be located with ``text.find(anchor)``, which
    binds by CONTENT: a paper citing "Smith 2020" in the body and again in the
    bibliography -- where the DOI actually lives -- had the URI attached to the
    in-text mention instead. A wrong URI on the wrong citation is worse than a
    dropped link, and this is a citation corpus.

    Words come back from ``get_text("words")`` in reading order, which is the order
    ``get_text("text")`` lays them out, so a single left-to-right cursor pass
    recovers each occurrence's own offsets. A word that cannot be located from the
    cursor onward (hyphenation, a ligature the two extractors spell differently) is
    skipped rather than guessed at, and its link simply resolves no span.
    """
    spans: dict[int, tuple[int, int]] = {}
    cursor = 0
    for i, w in enumerate(words):
        token = w[4]
        if not token:
            continue
        idx = text.find(token, cursor)
        if idx < 0:
            continue
        spans[i] = (idx, idx + len(token))
        cursor = idx + len(token)
    return spans


def _apply_links_to_flat_text(text: str, links: list, words: list | None = None) -> str:
    """Wrap resolved link anchors inside an already-flattened page string.

    GH-127: ``extract_structured`` short-circuits a page with no tables to raw
    ``get_text("text")`` and never walks the span dict -- which is EVERY prose
    page, i.e. the majority of the corpus. Routing those pages through the dict
    walk instead would change their text for reasons unrelated to links (line
    stripping, furniture relegation) and churn every golden fragment, so links
    are applied to the flat string instead.

    Anchors are located by WORD POSITION, not by ``text.find(anchor)``. Content
    matching bound every link to the first textual occurrence, so a "Smith 2020"
    appearing in both the body and the bibliography put the bibliography's DOI on
    the in-text mention (GH-127 review). It also produced malformed nested output
    when one anchor was a substring of another already-wrapped one:
    ``[Smith [2020](b) for details](a)``.

    Replacements are applied right-to-left so earlier offsets stay valid, and any
    link whose span overlaps one already applied is skipped rather than nested.
    Every original character survives; only the brackets are inserted.
    """
    if not links or not text:
        return text

    char_spans = _word_char_spans(text, words or [])

    resolved: list[tuple[int, int, str, str]] = []
    for link in links:
        uri = link[1]
        anchor = link[2]
        idxs = link[3] if len(link) > 3 else ()
        if not anchor or anchor == uri:
            continue
        bounds = [char_spans[i] for i in idxs if i in char_spans]
        if bounds:
            start_c = min(b[0] for b in bounds)
            end_c = max(b[1] for b in bounds)
        else:
            # No word geometry (a damaged text layer, or words the two extractors
            # spell differently). Fall back to the first textual occurrence -- the
            # old behaviour, now only for links that would otherwise be lost.
            idx = text.find(anchor)
            if idx < 0:
                continue
            start_c, end_c = idx, idx + len(anchor)
        resolved.append((start_c, end_c, uri, text[start_c:end_c]))

    # Left-to-right so the FIRST link wins a contested span, then applied in
    # reverse so each splice leaves the earlier offsets untouched.
    resolved.sort(key=lambda r: (r[0], r[1]))
    kept: list[tuple[int, int, str, str]] = []
    last_end = -1
    for start_c, end_c, uri, anchor in resolved:
        if start_c < last_end:
            continue  # overlaps an already-accepted anchor; never nest
        kept.append((start_c, end_c, uri, anchor))
        last_end = end_c

    for start_c, end_c, uri, anchor in reversed(kept):
        text = text[:start_c] + _emit_run(anchor, uri) + text[end_c:]
    return text


def _emit_run(text: str, uri: str) -> str:
    """Wrap ``text`` as a markdown link. Whitespace stays outside the brackets."""
    inner = text.strip()
    if not uri or not inner:
        return text
    # A visible URL that links to itself needs no anchor -- "[http://x](http://x)"
    # is noise, and the address is already readable in the text.
    if inner == uri:
        return text
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    return f"{lead}[{inner}]({uri}){trail}"


def _line_text(spans, links: list[tuple[object, str, str]]) -> str:
    """Join a line's spans, wrapping link anchors in markdown links.

    With no links on the page this is byte-identical to the previous
    ``"".join(...)`` — the golden fragment tests depend on that.

    Anchors are wrapped by SUBSTRING so the span's own characters and spacing
    survive verbatim; only the brackets are inserted. A link whose anchor text
    could not be resolved from the words falls back to wrapping any span the
    rectangle genuinely covers, so the URI is still recovered.
    """
    if not links:
        return "".join(s.get("text", "") for s in spans)

    parts: list[str] = []
    for span in spans:
        text = span.get("text", "")
        if not text.strip():
            parts.append(text)
            continue
        try:
            span_rect = fitz.Rect(span["bbox"])
        except Exception:
            parts.append(text)
            continue

        # Every intersecting link, not just the first: a uniform-font line is one
        # span, so a references line can carry several DOIs and `break` dropped all
        # but one (GH-127 review). Applied right-to-left so earlier offsets survive,
        # and an anchor overlapping one already wrapped here is skipped, never
        # nested.
        hits: list[tuple[int, int, str, str]] = []
        for link in links:
            rect, uri, anchor = link[0], link[1], link[2]
            if not span_rect.intersects(rect):
                continue
            if not anchor or anchor not in text:
                # No whole-span fallback. Wrapping the span on rectangle coverage
                # alone marked an ENTIRE line for a phrase-sized link, because a
                # uniform-font line is a single span -- the behaviour this
                # anchor-resolution replaced (GH-127 review).
                continue
            hits.append((text.index(anchor), text.index(anchor) + len(anchor), uri, anchor))

        hits.sort(key=lambda h: (h[0], h[1]))
        kept: list[tuple[int, int, str, str]] = []
        last_end = -1
        for start_c, end_c, uri, anchor in hits:
            if start_c < last_end:
                continue
            kept.append((start_c, end_c, uri, anchor))
            last_end = end_c

        for start_c, end_c, uri, anchor in reversed(kept):
            text = text[:start_c] + _emit_run(anchor, uri) + text[end_c:]
        parts.append(text)
    return "".join(parts)


# A slash that BEGINS a token and is immediately followed by a digit: "(/997)",
# "/55-84", "pp. /23". This is the eaten-leading-digit signature — a stroke glyph
# decoded as '/' where a digit belongs, so "(1997)" ships as "(/997)".
#
# Deliberately narrower than the ratio's `/\d|\d/|\(/`. That broader pattern is
# correct for a DENSITY score, where a few false hits wash out, but it fires on
# text this corpus is full of — "1/2", "2019/20", "12/31/2024", "9/11",
# "example.com/2024" — and so cannot drive an absolute-count gate. Requiring the
# slash to start the token excludes every one of those (the slash there is
# preceded by a digit or letter) while still catching both the eaten year and the
# eaten page range.
_DIGIT_CORRUPTION_RE = re.compile(r"(?<![0-9A-Za-z])/[0-9]")


def count_digit_corruption(text: str) -> int:
    """Number of eaten-leading-digit occurrences in ``text``.

    An ABSOLUTE count, never a ratio: the corpus invariant is per-token ("a wrong
    number is worse than a missing one"), so page length must not dilute it. Two
    destroyed citation years are two destroyed citation years whether the page
    carries 500 words or 5,000 (#136).
    """
    return len(_DIGIT_CORRUPTION_RE.findall(text or ""))


# Semantically-empty characters that publishers inject into the text layer for
# justification / line-breaking. They carry no content but corrupt downstream use:
# the soft hyphen splits words so search misses them ("hetero\xadgeneous" != the
# query), and zero-width spaces wedge themselves around sub/superscripts so math
# reads as "c​i,t". Native text bypasses ``OutputNormalizer`` (it is set
# straight from the PDF text layer), so it must be cleaned HERE, at the extraction
# boundary, or the junk ships unaltered.
_ZERO_WIDTH_STRIP = frozenset({chr(0x200B), chr(0x00AD)})  # ZWSP, soft hyphen
# Non-breaking / figure / thin / hair / narrow spaces. Collapsed to U+0020 (not
# stripped): they ARE spaces, just typographic variants that break search and
# token splitting. Plain ASCII space is the portable form for Markdown prose.
_EXOTIC_SPACES = frozenset(
    chr(c) for c in (0x00A0, 0x2002, 0x2003, 0x2007, 0x2008, 0x2009, 0x200A, 0x202F)
)


def clean_native_text(text: str) -> tuple[str, int, int]:
    """Strip semantically-empty invisibles and normalize exotic spaces.

    Returns ``(cleaned, n_stripped, n_spaces_normalized)``. Lossless for content:
    only removes zero-width chars (ZWSP, soft hyphen) and collapses typographic
    spaces to U+0020. Does NOT touch Private Use Area glyphs — those are a content
    signal (see :func:`count_pua_chars`) and are flagged, never silently altered.
    Deliberately not NFKC: that would fold math alphanumerics (𝒯, superscripts)
    into ASCII and create a NEW silent math corruption.
    """
    stripped = 0
    spaced = 0
    out: list[str] = []
    for ch in text:
        if ch in _ZERO_WIDTH_STRIP:
            stripped += 1
        elif ch in _EXOTIC_SPACES:
            spaced += 1
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out), stripped, spaced


def count_pua_chars(text: str) -> int:
    """Count Unicode Private Use Area codepoints in a text layer.

    A born-digital text layer should never contain PUA characters: they are
    font-private glyphs (math symbols the font maps without a real ToUnicode
    entry — e.g. UniMath U+F766 = script-T, STIXNonUnicode U+E14B). They render
    correctly ONLY inside the embedding font and become boxes or vanish anywhere
    else, so their presence is a direct fingerprint of weak/broken ToUnicode and
    hence silent math-glyph loss. Covers the BMP PUA and both supplementary PUA-A/B
    planes.
    """
    return sum(
        1
        for ch in text
        if 0xE000 <= ord(ch) <= 0xF8FF
        or 0xF0000 <= ord(ch) <= 0xFFFFD
        or 0x100000 <= ord(ch) <= 0x10FFFD
    )


#: Glyphs a PDF uses to draw an unordered list marker. Their presence in the text
#: layer means the page *has* a list; flat text can only render them as literal
#: characters mid-paragraph, which is the GH-127 symptom.
_LIST_MARKER_GLYPHS = frozenset("•‣◦▪●⁃⁌⁍")

#: A markdown list item or ATX heading in the emitted text. If the emitted text
#: already carries these, nothing was lost and no signal fires.
_MD_LIST_RE = re.compile(r"(?m)^\s*[-*+]\s+")
_MD_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")


def detect_native_structure_loss(page: fitz.Page, native_text: str) -> dict[str, int]:
    """Count structural cues in the PDF that the emitted flat text cannot represent.

    GH-127. The native lane ships ``page.get_text("text")``, which is a linear dump:
    bullet glyphs become literal characters inside a paragraph, headings become
    ordinary lines, and a paragraph number set in the margin lands inside the
    sentence beside it. Measured on the EFO-Nov-2022 corpus, the native lane emitted
    zero headings and zero list items across all 14 pages it handled, while the VLM
    lane emitted 46 headings and 54 list items across 51 pages.

    This function does not fix any of that. It reports what was lost so the loss
    stops being invisible -- a page can then still ship, but not silently.

    Every comparison is derived from the page's own measured geometry (its modal body
    font size, its modal body left edge), never a tuned constant, because point sizes
    and margins are document-specific.

    Returns a dict of cue counts, whose names state their epistemic strength:
    ``unrepresented_lists`` and ``unrepresented_headings`` are verified losses -- the
    cue is present in the PDF *and* absent from the emitted text. ``marginal_number_cues``
    is weaker: it proves only that the page sets numbers in its margin, not that they
    were mangled. Do not report it as a loss count.

    All zero means the flat dump was an adequate representation of this page's
    structure, not that the text is correct.
    """
    cues = {"unrepresented_lists": 0, "unrepresented_headings": 0, "marginal_number_cues": 0}
    try:
        blocks = page.get_text("dict").get("blocks") or []
    except Exception:  # noqa: BLE001 - a detector must never break extraction
        return cues

    spans: list[tuple[float, float, float, str]] = []  # (size, x0, x1, text)
    for block in blocks:
        for line in block.get("lines") or []:
            for span in line.get("spans") or []:
                text = str(span.get("text") or "")
                if not text.strip():
                    continue
                bbox = span.get("bbox") or (0.0, 0.0, 0.0, 0.0)
                spans.append((float(span.get("size") or 0.0), float(bbox[0]), float(bbox[2]), text))

    if not spans:
        return cues

    # Body size, weighted by how much text is set in it. Footnotes are many short
    # spans; weighting by character count keeps them from winning the vote.
    weight_by_size: Counter[float] = Counter()
    for size, _x0, _x1, text in spans:
        weight_by_size[round(size, 1)] += len(text.strip())
    body_size = weight_by_size.most_common(1)[0][0]

    left_weight: Counter[float] = Counter(
        round(x0, 1) for size, x0, _x1, _t in spans if round(size, 1) == body_size
    )
    if not left_weight:
        return cues
    body_left = left_weight.most_common(1)[0][0]

    # 1 and 2. Count the SHORTFALL, not presence-or-absence. An earlier revision
    # suppressed every source cue as soon as the output contained one markdown list
    # or one heading, so a page with three source bullets and one emitted list item
    # reported zero loss for the two that vanished -- the partial case, which is the
    # common one, was invisible.
    source_lists = sum(
        1 for _s, _x0, _x1, text in spans if text.strip() and text.strip()[0] in _LIST_MARKER_GLYPHS
    )
    cues["unrepresented_lists"] = max(0, source_lists - len(_MD_LIST_RE.findall(native_text)))

    # Heading spans are counted per span, so a title broken across lines counts once
    # per line; the emitted side is counted the same way for the comparison to hold.
    source_headings = sum(1 for size, _x0, _x1, _t in spans if round(size, 1) > body_size)
    cues["unrepresented_headings"] = max(
        0, source_headings - len(_MD_HEADING_RE.findall(native_text))
    )

    # 3. A digits-only span sitting entirely left of the body column: a marginal
    #    paragraph number. Flat text drops it into the prose, creating a number that
    #    was never in that sentence -- the worst outcome for a citation corpus.
    cues["marginal_number_cues"] = sum(
        1
        for size, _x0, x1, text in spans
        if text.strip().isdigit() and x1 < body_left and round(size, 1) == body_size
    )

    return cues


@dataclass
class PageAssessment:
    """Per-page born-digital assessment."""

    page_num: int  # 1-indexed
    is_born_digital: bool
    native_text: str
    confidence: float  # 0.0 to 1.0
    char_count: int = 0
    word_count: int = 0
    font_count: int = 0
    has_images: bool = False
    has_tables: bool = False  # page contains table-like structures
    has_figures: bool = False  # page contains embedded images (alias for has_images)
    has_equations: bool = False  # page contains math/equations
    needs_ocr_enhancement: bool = False  # native layer has a known deficiency
    has_corrupt_math: bool = False  # font-map mojibake in math (needs region OCR -> LaTeX)
    has_unmapped_math_glyphs: bool = False  # PUA glyphs (weak ToUnicode) -> silent math-glyph loss
    #: #217: the page's symbol font had no ToUnicode map and at least one glyph
    #: it draws is not in the verified recovery table, so those characters are
    #: still whatever the extractor produced. Set for every page of an affected
    #: document -- the missing map is a document-level font property, so a page
    #: cannot be clean while its font is not. A log line alone reaches nothing
    #: that ships, which is the same gap #136 was filed for.
    has_unrecovered_symbol_glyphs: bool = False
    has_unverifiable_table_region: bool = False  # TR-3: per-region geometry hard-fail
    #: GH-195: one record per text-strategy grid rejected because a lane boundary
    #: split a native numeric token. The word-geometry fallback that replaced it
    #: is lossless, so this is a VISIBILITY signal, not a defect flag — the page
    #: is not demoted on it. Empty on a page where every grid rendered cleanly.
    text_grid_rejections: list[dict] = field(default_factory=list)
    #: #136: mid-band encoding corruption of the COSMETIC class (lost spaces, fused
    #: words) — the page is trustworthy for content but its text layer is suspect.
    #: Propagated to PageState so the agentic native lane can emit a durable audit
    #: event; the historical ``notes`` entry alone reached nothing that ships.
    has_encoding_hygiene_suspect: bool = False
    #: GH-147 A2: set ONLY by the refusal branch in ``_assess_page_signals`` when
    #: the native table lane is actually refused (rotated text direction + table
    #: detected on a born-digital page). ``has_tables and text_is_rotated`` alone
    #: over-fires: ``has_tables`` is stamped before the early non-born-digital
    #: returns, so a rotated *scanned/garbled* page with a ruled table would also
    #: match, even though the refusal branch never ran and there is no native
    #: text to have retained. Audit code must key off this flag, not re-derive
    #: "refusal happened" from conditions that merely correlate with it.
    native_table_lane_refused: bool = False
    #: #263: this page's dominant text direction is rotated AND its native
    #: layer is confetti -- one glyph run per extracted line, all in one
    #: column (see ``rotated_text_is_shredded``). Set ONLY in the no-table
    #: branch of ``_assess_page_signals``, never re-derived downstream, the
    #: same rule ``native_table_lane_refused`` above states. The rotated
    #: has_tables case is not covered here because it is already refused
    #: unconditionally by the GH-147 branch, and because a rotated TABLE's
    #: cells legitimately tile along the reading axis, which is the geometry
    #: the shred predicate reads as damage.
    native_rotated_text_shredded: bool = False
    #: Backward-compatible aggregate set ONLY inside the non-rotated,
    #: has_tables branch of ``_assess_page_signals`` (the structured-extraction
    #: branch, not the refusal branch, and never for non-table pages). It
    #: aggregates raw emission defects (GH-226), raw content defects (GH-190),
    #: and parsed shape defects (GH-151 TICKET-B1). The parsed-shape term is
    #: ``ragged`` or ``detached_label_rows`` only (never ``defective`` as a
    #: whole, never ``orphan_rows`` alone — see the design note at
    #: ``docs/log/2026-08-13_gh151-b1-design.md``). Audit code MUST key off
    #: this flag and MUST NEVER re-derive it downstream — the same GH-147 A2
    #: rule as ``native_table_lane_refused`` above.
    native_table_structure_defective: bool = False
    #: GH-226 exact raw-emission defect code. This remains separate from the
    #: aggregate above, so a raw content or parsed shape defect cannot be
    #: misreported as emission provenance.
    native_table_emission_defect: str = ""
    #: GH-200: header-attribution HARD verdict on the native markdown (only
    #: checked when the aggregate above did NOT already fire -- the emission,
    #: content, and shape terms are cheaper and take priority in cost order).
    #: True iff ``header_attribution.header_attribution`` found
    #: a data lane whose native header words are absent from the emitted
    #: header row entirely (destroyed, not merely misplaced -- see
    #: ``header_attribution``'s module docstring for SOFT vs HARD). SOFT and
    #: UNVERIFIABLE verdicts never set this flag. Rides the exact same
    #: propagation as ``native_table_structure_defective``: set once here,
    #: never re-derived downstream, deliberately absent from ``needs_repair``.
    native_table_header_unattributed: bool = False
    #: The page's prevailing text-line direction, stamped once by ``_assess_page``
    #: from ``dominant_text_direction()``. The ``_HORIZONTAL`` default conflates
    #: "genuinely horizontal" with "no directional evidence" (an empty tally from
    #: ``dominant_text_direction()``) — consumers must NOT read this field's
    #: default as proof that directional text evidence existed.
    dominant_text_direction: tuple[float, float] = _HORIZONTAL
    notes: list[str] = field(default_factory=list)

    @property
    def text_is_rotated(self) -> bool:
        return text_direction_is_rotated(self.dominant_text_direction)


@dataclass
class DocumentAssessment:
    """Document-level born-digital assessment."""

    path: Path
    pages: list[PageAssessment]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def born_digital_count(self) -> int:
        return sum(1 for p in self.pages if p.is_born_digital)

    @property
    def scanned_count(self) -> int:
        return self.page_count - self.born_digital_count

    @property
    def is_fully_born_digital(self) -> bool:
        return all(p.is_born_digital for p in self.pages)

    @property
    def is_fully_scanned(self) -> bool:
        return not any(p.is_born_digital for p in self.pages)

    @property
    def is_mixed(self) -> bool:
        return not self.is_fully_born_digital and not self.is_fully_scanned

    def born_digital_pages(self) -> list[int]:
        """Return 1-indexed page numbers of born-digital pages."""
        return [p.page_num for p in self.pages if p.is_born_digital]

    def scanned_pages(self) -> list[int]:
        """Return 1-indexed page numbers of scanned pages."""
        return [p.page_num for p in self.pages if not p.is_born_digital]


def _is_lane_stacked(table: object) -> bool:
    """True when a ``find_tables()`` result looks lane-stacked (whitespace-gutter collapse).

    A whitespace-gutter table (no ruling lines, CE forecaster-grid style) may be
    returned by ``find_tables()`` with multiple values newline-stacked inside a
    single cell — the "lines" strategy collapses the whitespace-separated columns.
    Detecting this lets ``extract_structured`` route the region to the
    word-geometry rowizer instead of the passthrough ``_table_to_markdown`` path.

    The check is purely structural: any cell containing an embedded newline
    signals that the extractor concatenated what should be separate column values.
    Never raises.
    """
    try:
        rows = table.extract()
    except Exception:
        return False
    for row in rows or []:
        for cell in row or []:
            if cell and isinstance(cell, str) and "\n" in cell:
                return True
    return False


class BornDigitalDetector:
    """Detect born-digital pages and extract native text from PDFs.

    A page is considered born-digital if it has a text layer that looks like
    genuine authored text rather than a baked-in OCR layer. The detector uses
    multiple signals: text density, character quality, font diversity, and
    the presence/absence of embedded images.
    """

    # Minimum characters for a page to be considered as having meaningful text.
    # A typical academic paper page has 2000-4000 chars. Very short text layers
    # are likely artifacts or watermarks rather than genuine content.
    MIN_CHARS_FOR_TEXT_LAYER = 50

    # Minimum words per page for born-digital classification when text quality is
    # *unknown* (i.e., before quality checks run).  Pages with fewer words than
    # this AND poor quality signals are classified as scanned.
    #
    # For *clean* text (no CID artifacts, low garbage ratio, normal word lengths,
    # at least one proper font), this threshold is NOT enforced — a sparse born-
    # digital page (chapter divider, figure caption, section heading) should be
    # classified as born-digital even with fewer words.  Only the absolute floor
    # MIN_WORDS_SPARSE is enforced for clean text.
    #
    # Basis: 15 was historically calibrated for detecting baked-in OCR layers
    # (garbage OCR tends to produce many single-char "words", so a low absolute
    # count was a useful proxy).  That signal is now captured directly by the
    # quality checks (garbage ratio, word-length distribution, CID artifacts).
    MIN_WORDS_PER_PAGE = 15

    # Absolute minimum words to trust as a real native text layer, even when
    # all quality checks pass.  Below this, the text is too thin to extract
    # meaningful content (single page numbers, watermarks, stray characters).
    #
    # Basis: a figure caption or section heading is "Figure 1." or "Chapter 3"
    # at minimum — at least 2 tokens.  We use 3 to require at minimum a brief
    # phrase (number + noun + adjective) rather than a lone label or watermark.
    # Three words is a loose floor; the quality checks do the real discrimination.
    MIN_WORDS_SPARSE = 3

    # Fraction of page area covered by embedded raster images above which a page
    # is considered image-dominant.  Used to gate the clean-short-text exception
    # (GH-35-FU, consilium decision id 20260615T104828Z-1577):
    #
    # A scanned page + baked-in OCR layer looks indistinguishable from a
    # born-digital figure page with a caption when the text layer is short and
    # clean.  But on a scanned page the raster image covers the ENTIRE page,
    # while on a genuine born-digital figure page the chart/figure occupies a
    # substantial but clearly sub-total fraction (surrounding white-space, caption
    # area, header region all reduce image coverage well below 1.0).
    #
    # The panel converged on ~0.90–0.95.  We use 0.90 as the threshold:
    #   - A full-page scan rendered as a single raster hits ≥ 0.98 (image fills
    #     essentially the whole printable area after margins).
    #   - A born-digital figure page with a chart in the body and a caption at
    #     the top/bottom typically reaches 0.50–0.80 coverage, well below 0.90.
    #   - The margin of 0.10 absorbs scanner over-crop and image-rect rounding
    #     without encroaching on the legitimate figure-page population.
    #
    # Over-correcting (routing to OCR) is the safer direction here: an extra OCR
    # call on a true born-digital page is suboptimal but recoverable; silently
    # skipping OCR on a real scan causes permanent content loss.
    RASTER_DOMINANCE_RATIO = 0.90

    # Maximum ratio of garbage/non-printable characters. Born-digital text is
    # clean; OCR layers on scanned PDFs often contain (cid:XX) references,
    # replacement chars, and control characters.
    MAX_GARBAGE_RATIO = 0.05

    # Maximum ratio of characters that are just spaces. Baked-in OCR on scanned
    # pages often produces text with excessive spacing (one char per glyph with
    # spaces between).
    MAX_SPACE_RATIO = 0.60

    # Minimum average word length. Garbage OCR produces many single-char "words".
    MIN_AVG_WORD_LENGTH = 2.5

    # Maximum average word length. Garbled text can fuse characters into long
    # non-word strings.
    MAX_AVG_WORD_LENGTH = 20.0

    # Encoding-corruption ratio: a born-digital PDF can have a broken font/ToUnicode
    # map that yields VALID characters in wrong positions ("Journal"->"Joumal",
    # "1997"->"(/997)") — invisible to the garbage-char check. Signals: mid-word
    # capitals ("ofFinancial"), slash-for-digit ("(/997)"), run-on words. Calibrated
    # on real pages: clean prose ~0.000, a header-only glitch (Fama-French) ~0.019,
    # pervasive data corruption ~0.095.
    # FLAG threshold: record the page as suspect (visibility) but still trust native.
    ENCODING_CORRUPTION_FLAG = 0.01
    # ESCALATE threshold: native is untrustworthy page-wide -> route to OCR (read the
    # image, not the broken encoding). Set well above the header-only case so a clean
    # body is never sent to a slower/lossier VLM over a few bad header tokens.
    MAX_ENCODING_CORRUPTION = 0.05
    # #136: eaten-leading-digit occurrences that route the page to OCR, as an
    # ABSOLUTE count rather than a density. One is enough: a single "(/997)" is a
    # wrong publication year, which the corpus invariant ranks as worse than a
    # missing one, and no amount of surrounding clean prose makes it less wrong.
    MAX_DIGIT_CORRUPTION_HITS = 1

    def __init__(
        self,
        min_chars: int | None = None,
        min_words: int | None = None,
        max_garbage_ratio: float | None = None,
    ) -> None:
        if min_chars is not None:
            self.MIN_CHARS_FOR_TEXT_LAYER = min_chars
        if min_words is not None:
            self.MIN_WORDS_PER_PAGE = min_words
        if max_garbage_ratio is not None:
            self.MAX_GARBAGE_RATIO = max_garbage_ratio
        #: Report from the most recent symbol-font recovery (#217). ``None``
        #: until a document has been assessed.
        self.last_glyph_repair: GlyphRepairReport | None = None

    def detect(self, pdf_path: Path | str) -> DocumentAssessment:
        """Analyze all pages of a PDF for born-digital content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            DocumentAssessment with per-page results.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages: list[PageAssessment] = []
        # repair=False: the report is needed here, so recovery is applied
        # explicitly below rather than silently inside the open.
        with open_pdf(pdf_path, repair=False) as doc:
            self._recover_symbol_fonts(doc, pdf_path)
            for page_idx in range(len(doc)):
                assessment = self._assess_page(doc[page_idx], page_idx + 1)
                pages.append(assessment)

            self._mark_unrecovered_glyphs(pages)

        return DocumentAssessment(path=pdf_path, pages=pages)

    def detect_page(self, pdf_path: Path | str, page_num: int) -> PageAssessment:
        """Assess a single page (1-indexed).

        Args:
            pdf_path: Path to the PDF file.
            page_num: 1-indexed page number.

        Returns:
            PageAssessment for the requested page.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # repair=False: the report is needed here, so recovery is applied
        # explicitly below rather than silently inside the open.
        with open_pdf(pdf_path, repair=False) as doc:
            if page_num < 1 or page_num > len(doc):
                raise ValueError(f"Page {page_num} out of range (document has {len(doc)} pages)")
            self._recover_symbol_fonts(doc, pdf_path)
            assessment = self._assess_page(doc[page_num - 1], page_num)
            self._mark_unrecovered_glyphs([assessment])
            return assessment

    def _mark_unrecovered_glyphs(self, pages: list[PageAssessment]) -> None:
        """Propagate the recovery report onto the pages so it can reach output.

        The report is document-level because the missing ToUnicode map is a
        property of the font, not of any one page: a document is either clean or
        comprehensively affected.
        """
        report = self.last_glyph_repair
        if report is None or not report.needs_attention:
            return
        for page in pages:
            page.has_unrecovered_symbol_glyphs = True

    def _recover_symbol_fonts(self, doc: fitz.Document, pdf_path: Path) -> None:
        """Rebuild missing ToUnicode maps before any text is read (#217).

        Must run before the first ``get_text`` on *doc*: an embedded symbol font
        with no ToUnicode makes the text layer hand back the raw byte, so a minus
        sign extracts as ``2`` and ``-0.12`` ships as ``20.12``. Repairing the
        document here fixes every reader downstream at once rather than each
        call site patching text after the fact.

        Never raises: a document that cannot be repaired must still be assessed
        exactly as it was before this existed.
        """
        try:
            report = apply_glyph_recovery(doc, pdf_path)
        except Exception as exc:  # pragma: no cover - defensive
            # Cleared, not left stale: a reused detector would otherwise
            # attribute the previous document's report to this one.
            self.last_glyph_repair = None
            logger.warning("[glyph] %s: recovery failed (%s)", pdf_path.name, exc)
            return

        self.last_glyph_repair = report
        if report.repaired:
            logger.info(
                "[glyph] %s: rebuilt ToUnicode for %d font(s), %d glyph(s)",
                pdf_path.name,
                len(report.repaired_fonts),
                report.mapped_glyph_count,
            )
        # Checked independently of ``repaired``: a document where NOTHING was
        # recoverable attaches no CMap at all, and that is the worst case, not
        # a quiet one.
        if report.needs_attention:
            # Surfaced, not swallowed: these characters are still whatever the
            # extractor produced, so the page is not trustworthy.
            logger.warning(
                "[glyph] %s: %d drawn glyph(s) have no verified mapping and were "
                "left unchanged: %s",
                pdf_path.name,
                len(report.unmapped_glyphs),
                ", ".join(report.unmapped_glyphs),
            )

    @staticmethod
    def _text_blocks(page: fitz.Page) -> list:
        """``page.get_text("dict")["blocks"]``, guarded like every other read.

        Same fallback contract as ``_assess_page`` below: a page whose text
        layer cannot be walked yields no blocks rather than raising out of the
        detector.
        """
        try:
            return page.get_text("dict").get("blocks", [])
        except Exception:
            return []

    def _assess_page(self, page: fitz.Page, page_num: int) -> PageAssessment:
        """Assess a page and stamp its dominant text direction onto the result.

        ``_assess_page_signals`` has multiple early returns (encoding failures,
        image-only pages, etc.), so the direction is computed once here and
        stamped onto whichever ``PageAssessment`` comes back — a single stamping
        point that survives every early return of the signals body, rather than
        threading the computation through 11 separate return sites.

        The extraction is guarded because this call now runs *before*
        ``_assess_page_signals``, whose body already tolerates damaged pages
        (encoding failures, unreadable text layers). Unguarded, a page that used
        to degrade into a low-confidence assessment would instead raise out of
        the detector. Every other ``get_text`` call in this module is wrapped the
        same way. The fallback is ``_HORIZONTAL``, matching the
        absence-of-evidence precedent (#145): no directional evidence must not
        route a page as rotated.
        """
        try:
            blocks = page.get_text("dict").get("blocks", [])
        except Exception:
            blocks = []
        direction = dominant_text_direction(blocks)
        assessment = self._assess_page_signals(page, page_num, direction)
        assessment.dominant_text_direction = direction
        if text_direction_is_rotated(direction):
            assessment.notes.append(f"rotated text direction {direction}")
        return assessment

    def _assess_page_signals(
        self, page: fitz.Page, page_num: int, direction: tuple[float, float]
    ) -> PageAssessment:
        """Assess whether a single page is born-digital.

        Uses multiple signals to distinguish genuine born-digital text from
        low-quality baked-in OCR layers.
        """
        notes: list[str] = []

        # Extract raw text from the PDF text layer, then clean it at the boundary:
        # native text never passes through OutputNormalizer, so publisher-injected
        # zero-width spaces / soft hyphens / exotic spaces must be removed here or
        # they ship verbatim (broken search, unreadable math). PUA glyphs survive
        # the clean and are detected separately as a math-glyph-loss signal.
        raw_text = page.get_text("text")
        raw_text, _zw_stripped, _spaces_normalized = clean_native_text(raw_text)
        pua_count = count_pua_chars(raw_text)
        has_unmapped_math_glyphs = pua_count > 0
        char_count = len(raw_text)
        words = raw_text.split()
        word_count = len(words)

        # Count distinct fonts used on the page
        font_count = self._count_fonts(page)

        # Check for embedded images (raster content)
        has_images = self._has_images(page)

        # Detect structured content types
        has_tables = self._detect_tables(page)
        has_figures = has_images  # figures = embedded raster images
        has_corrupt_math = self._detect_corrupt_math(raw_text)
        has_equations = (
            self._detect_math_fonts(page) or self._detect_equations(raw_text) or has_corrupt_math
        )

        # --- Decision logic ---

        # No text layer at all: definitely scanned.  This is a hard gate — fewer
        # than MIN_CHARS_FOR_TEXT_LAYER characters means there is not enough
        # content to be useful even if the few chars are perfectly clean.
        if char_count < self.MIN_CHARS_FOR_TEXT_LAYER:
            notes.append(
                f"insufficient text layer ({char_count} chars < {self.MIN_CHARS_FOR_TEXT_LAYER})"
            )
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.95,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # Compute text quality signals before the word-count check so that
        # "short but clean" (sparse born-digital pages: figure captions, chapter
        # headings, section dividers) can be rescued from the word-count gate.
        # All quality signals below are still checked regardless of word count.
        garbage_ratio = self._garbage_ratio(raw_text)
        space_ratio = raw_text.count(" ") / max(len(raw_text), 1)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        has_cid = bool(re.search(r"\(cid:\d+\)", raw_text))

        # A text layer is "clean" when ALL of the following hold:
        #   - no CID font-mapping artifacts (definitive sign of broken font map)
        #   - low garbage-character ratio (baked-in OCR on scans tends to be noisy)
        #   - normal average word length (garbled text fuses or fragments tokens)
        #   - at least one embedded font (pages with no fonts have no real text layer)
        #
        # Clean short text on a page with proper fonts is a genuine native layer
        # (figure caption, section heading, sparse table).  Such pages must NOT be
        # classified as scanned solely because they have fewer than MIN_WORDS_PER_PAGE
        # words — that would route them to OCR and potentially degrade the output.
        text_layer_is_clean = (
            not has_cid
            and garbage_ratio <= self.MAX_GARBAGE_RATIO
            and self.MIN_AVG_WORD_LENGTH <= avg_word_len <= self.MAX_AVG_WORD_LENGTH
            and font_count > 0
        )

        # Word-count gate.  Enforced unconditionally only when the text layer is
        # dirty (quality signals indicate baked-in OCR or no real layer).  When the
        # text is clean, only the absolute floor MIN_WORDS_SPARSE is applied — this
        # catches lone watermarks or page-number stubs that would extract as a single
        # token, not sparse-but-real content.
        if word_count < self.MIN_WORDS_PER_PAGE:
            if not text_layer_is_clean or word_count < self.MIN_WORDS_SPARSE:
                notes.append(
                    f"too few words ({word_count} < {self.MIN_WORDS_PER_PAGE})"
                    + ("" if text_layer_is_clean else "; dirty text layer")
                )
                return PageAssessment(
                    page_num=page_num,
                    is_born_digital=False,
                    native_text="",
                    confidence=0.85,
                    char_count=char_count,
                    word_count=word_count,
                    font_count=font_count,
                    has_images=has_images,
                    has_tables=has_tables,
                    has_figures=has_figures,
                    has_equations=has_equations,
                    notes=notes,
                )

            # Clean short text — but gate the pass-through by raster coverage.
            # A scanned page with a baked-in OCR caption is INDISTINGUISHABLE from
            # a born-digital figure page via text quality alone.  However, a scan
            # always fills nearly the full page with a raster image, while a genuine
            # born-digital figure page has the chart occupying a sub-total fraction
            # of the page area (caption, header, and white-space reduce coverage).
            # If the page is image-dominant (raster coverage >= RASTER_DOMINANCE_RATIO),
            # route to OCR to prevent permanent content loss on real scans.
            # Non-image-dominant sparse-but-clean pages still classify born-digital.
            # GH-35-FU | consilium decision id 20260615T104828Z-1577
            raster_cov = self._raster_coverage(page)
            if raster_cov >= self.RASTER_DOMINANCE_RATIO:
                notes.append(
                    f"sparse native layer ({word_count} words); image-dominant page "
                    f"({raster_cov:.1%} raster coverage >= {self.RASTER_DOMINANCE_RATIO:.0%} "
                    "threshold) — routing to OCR to avoid baked-in-OCR false-positive (GH-35-FU)"
                )
                return PageAssessment(
                    page_num=page_num,
                    is_born_digital=False,
                    native_text="",
                    confidence=0.85,
                    char_count=char_count,
                    word_count=word_count,
                    font_count=font_count,
                    has_images=has_images,
                    has_tables=has_tables,
                    has_figures=has_figures,
                    has_equations=has_equations,
                    notes=notes,
                )

            # Clean short text on a non-image-dominant page: pass through to
            # born-digital path with a note so the audit log can surface it.
            notes.append(
                f"sparse native layer ({word_count} words); clean text, "
                f"raster coverage {raster_cov:.1%} < {self.RASTER_DOMINANCE_RATIO:.0%}, "
                "classifying as born-digital"
            )

        # CID artifacts: definitive sign of broken font mapping on scanned PDF
        if has_cid:
            notes.append("CID font mapping artifacts detected")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.95,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # High garbage ratio: likely baked-in OCR with garbled output
        if garbage_ratio > self.MAX_GARBAGE_RATIO:
            notes.append(f"high garbage ratio ({garbage_ratio:.1%})")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.80,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # Excessive spacing: baked-in OCR often spaces out individual chars
        if space_ratio > self.MAX_SPACE_RATIO:
            notes.append(f"excessive spacing ({space_ratio:.1%})")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.75,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # Abnormal word lengths: garbled text fuses or fragments words
        if avg_word_len < self.MIN_AVG_WORD_LENGTH:
            notes.append(f"avg word length too short ({avg_word_len:.1f})")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.70,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        if avg_word_len > self.MAX_AVG_WORD_LENGTH:
            notes.append(f"avg word length too long ({avg_word_len:.1f})")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.70,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # Encoding corruption: a broken font/ToUnicode map yields valid characters
        # in wrong positions, invisible to the garbage-char check. Pervasive
        # corruption means the text layer can't be trusted page-wide -> read the
        # image instead. (Mild, header-only glitches are flagged below, not escalated.)
        encoding_corruption = self._encoding_corruption_ratio(raw_text)
        # #136: eaten-leading-digit corruption is a WRONG NUMBER, not a hygiene
        # defect, so it is gated on absolute count and routed to OCR at the first
        # occurrence — the same remedy as pervasive corruption below, for the same
        # reason: OCR RECOVERS the true digit instead of merely confessing that one
        # was lost. Ratio gating cannot serve this class; the identical damage
        # scores 3% on a references page and 0.75% on a long prose page, and the
        # second ships under the flag floor with no signal at all.
        #
        # A false positive costs one extra OCR pass on a page (spend); a false
        # negative ships a wrong citation year as SUCCESS. For a citation corpus
        # that asymmetry justifies erring hot.
        digit_corruption = count_digit_corruption(raw_text)
        if digit_corruption >= self.MAX_DIGIT_CORRUPTION_HITS:
            notes.append(
                f"text-layer digit corruption ({digit_corruption} eaten-digit "
                "occurrence(s), e.g. '(/997)' for '(1997)') -> OCR"
            )
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.75,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )
        if encoding_corruption > self.MAX_ENCODING_CORRUPTION:
            notes.append(f"text-layer encoding corrupted ({encoding_corruption:.1%}) -> OCR")
            return PageAssessment(
                page_num=page_num,
                is_born_digital=False,
                native_text="",
                confidence=0.75,
                char_count=char_count,
                word_count=word_count,
                font_count=font_count,
                has_images=has_images,
                has_tables=has_tables,
                has_figures=has_figures,
                has_equations=has_equations,
                notes=notes,
            )

        # --- All checks passed: page is born-digital ---

        # Compute confidence based on signal strength
        confidence = self._compute_confidence(
            char_count=char_count,
            word_count=word_count,
            garbage_ratio=garbage_ratio,
            space_ratio=space_ratio,
            avg_word_len=avg_word_len,
            font_count=font_count,
            has_images=has_images,
        )

        # Complex content is metadata, not a routing decision. Clean native
        # tables are extracted from the PDF text layer, figures are handled by
        # the figure pass, and clean equations should not force whole-page OCR.
        # Only evidence that the native layer itself is deficient should request
        # enhancement.
        has_complex_content = has_tables or has_figures or has_equations
        needs_ocr_enhancement = has_corrupt_math

        # Flag mild encoding corruption (e.g. a broken header font) for visibility
        # without escalating: the body is still trustworthy, but the page is marked
        # suspect so it is never silently relied on.
        #
        # #136: "marked suspect" used to mean ONLY this note, and nothing in the
        # pipeline reads PageAssessment.notes — so the page shipped SUCCESS with no
        # surface anywhere. The flag below is what carries the mark into PageState,
        # the sidecar and the audit log. The digit class never reaches here; it was
        # routed to OCR above.
        encoding_hygiene_suspect = encoding_corruption > self.ENCODING_CORRUPTION_FLAG
        if encoding_hygiene_suspect:
            notes.append(f"text-layer encoding suspect ({encoding_corruption:.1%})")

        # TR-3: reset the per-extraction unverifiable flag so _assess_page can
        # read it after calling extract_structured.  Initialise to False so pages
        # without tables (which never call _verify_regions) are not flagged.
        self._last_extraction_had_unverifiable: bool = False

        # GH-195: same side-channel shape as the TR-3 flag above — reset per
        # page so a rejection on page 7 is never attributed to page 8.
        self._last_extraction_grid_rejections: list[dict] = []

        native_table_lane_refused = False
        native_rotated_text_shredded = False
        native_table_structure_defective = False
        native_table_emission_defect = ""
        native_table_header_unattributed = False
        if has_tables:
            if text_direction_is_rotated(direction):
                # GH-147: the rowizer clusters rows by y; on a page whose text
                # runs at 90 degrees the rows run along x, so extract_structured
                # would emit a transposed grid as trusted native text. Refuse the
                # native table lane and retain the prose instead — the page is
                # still routed to OCR (needs_ocr_enhancement below) rather than
                # shipping a table nobody asked to transpose.
                native_text = raw_text.strip()
                notes.append(
                    "born-digital: native table reconstruction refused "
                    f"(dominant text direction {direction} is rotated); prose retained, "
                    "page routed to OCR"
                )
                needs_ocr_enhancement = True
                native_table_lane_refused = True
            else:
                # Use structured extraction that renders tables as markdown. Clean it
                # too: extract_structured builds its own text from the layer, so it
                # carries the same invisibles as raw_text and bypasses the boundary
                # clean above.
                native_text, _, _ = clean_native_text(self.extract_structured(page))
                notes.append("born-digital: structured extraction (tables detected)")

                # Compute the three terms of the backward-compatible native
                # table aggregate independently: GH-226 raw emission, GH-190
                # raw content, and GH-151 TICKET-B1 parsed shape. The raw terms
                # must be checked before parsing can discard their evidence.
                # Set at the moment of the evidence,
                # never re-derived downstream (the same rule GH-147 A2 states
                # for native_table_lane_refused above). The parsed-grid term
                # remains narrowed to ragged / detached_label_rows only --
                # orphan_rows alone is
                # excluded because a blank-label row with values is often a
                # legitimate standard-error / t-statistic continuation row
                # (fires on 27/29 real table blocks if included unnarrowed).
                # ``structural_gate_fires`` is the single source of truth for
                # this predicate -- tests import the same function.
                from socr.tables import structure_check

                reports = structure_check.check_markdown(native_text)
                native_table_emission_defect = structure_check.table_emission_defect(native_text)
                native_table_content_defect = structure_check.table_content_defect(native_text)
                native_table_structure_defective = bool(
                    native_table_emission_defect
                    or native_table_content_defect
                    or structure_check.structural_gate_fires(reports)
                )

                # GH-200: header-attribution term, disjunctive with the
                # grid-shape check above -- TR-3's numeric multiset and the
                # grid-shape gate are BOTH blind to a header band that is
                # destroyed/detached while every numeral stays correct (the
                # 2026-08-15 hand judgement: 4/4 damaged pages). Only the HARD
                # verdict is a defect; SOFT and UNVERIFIABLE never gate here.
                # Rides the exact same propagation as
                # ``native_table_structure_defective`` above (state.py
                # ``apply_born_digital``); deliberately absent from
                # ``needs_repair`` for the same --native-only reason.
                if not native_table_structure_defective:
                    words = page.get_text("words")
                    native_table_header_unattributed = (
                        structure_check.table_output_defect(native_text, words)
                        == structure_check.DEFECT_HEADER_UNATTRIBUTED
                    )

        else:
            native_text = raw_text.strip()
            notes.append("born-digital: clean text layer detected")

            # #263: until now the rotation refusal above was reachable ONLY
            # through ``if has_tables:``, so a rotated page with no detected
            # table shipped its native layer untouched -- and on a rotated
            # FIGURE page that layer is character-level confetti under a clean
            # SUCCESS. Rotation is now consulted on this branch too.
            #
            # The condition is rotation AND shredding, not rotation alone: a
            # rotated page whose text is re-laid cleanly extracts
            # byte-for-byte (measured, and pinned by
            # ``test_landscape_refusal_a2_gh147.py``), and refusing it would
            # route a sound free page to a paid VLM. ``rotated_text_is_shredded``
            # decides that on this page's own geometry -- see its docstring for
            # why a short-line fraction was measured and rejected.
            if text_direction_is_rotated(direction) and rotated_text_is_shredded(
                self._text_blocks(page), direction
            ):
                notes.append(
                    "born-digital: native text layer refused "
                    f"(dominant text direction {direction} is rotated and the extracted "
                    "lines are pieces of one text run); page routed to OCR"
                )
                needs_ocr_enhancement = True
                native_rotated_text_shredded = True

        if _zw_stripped or _spaces_normalized:
            notes.append(
                f"native layer cleaned: stripped {_zw_stripped} zero-width char(s), "
                f"normalized {_spaces_normalized} exotic space(s)"
            )

        # TR-3: read the per-region geometry hard-fail flag written by
        # extract_structured → _verify_regions during table extraction above.
        has_unverifiable_table_region = self._last_extraction_had_unverifiable
        # GH-195: text-strategy grids rejected for numeric-token destruction.
        text_grid_rejections = list(self._last_extraction_grid_rejections)

        if has_complex_content:
            content_types = []
            if has_tables:
                content_types.append("tables")
            if has_figures:
                content_types.append("figures")
            if has_equations:
                content_types.append("equations")
            notes.append(f"complex content detected ({', '.join(content_types)})")

        if has_corrupt_math:
            notes.append(
                f"corrupt math text layer ({self._count_math_corruption(raw_text)} mojibake "
                "glyphs); equation regions need image-OCR -> LaTeX"
            )

        # Unmapped math glyphs (PUA): the native prose is trustworthy, but math
        # symbols are font-private and lost outside the embedding font. We do NOT
        # set needs_ocr_enhancement here — that would route the whole page to OCR,
        # the lane empirically shown to make these pages worse (it falls back to the
        # same broken native layer). The flag surfaces via the audit log (and, when
        # equation recovery is enabled, the region-crop lane), never silently.
        if has_unmapped_math_glyphs:
            notes.append(
                f"unmapped math glyphs ({pua_count} private-use codepoint(s)); native prose "
                "trusted, math symbols are font-private (weak ToUnicode) -> need region OCR"
            )

        return PageAssessment(
            page_num=page_num,
            is_born_digital=True,
            native_text=native_text,
            confidence=confidence,
            char_count=char_count,
            word_count=word_count,
            font_count=font_count,
            has_images=has_images,
            has_tables=has_tables,
            has_figures=has_figures,
            has_equations=has_equations,
            needs_ocr_enhancement=needs_ocr_enhancement,
            has_corrupt_math=has_corrupt_math,
            has_unmapped_math_glyphs=has_unmapped_math_glyphs,
            has_unverifiable_table_region=has_unverifiable_table_region,
            text_grid_rejections=text_grid_rejections,
            has_encoding_hygiene_suspect=encoding_hygiene_suspect,
            native_table_lane_refused=native_table_lane_refused,
            native_rotated_text_shredded=native_rotated_text_shredded,
            native_table_structure_defective=native_table_structure_defective,
            native_table_emission_defect=native_table_emission_defect,
            native_table_header_unattributed=native_table_header_unattributed,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Content type detection
    # ------------------------------------------------------------------

    def _detect_tables(self, page: fitz.Page) -> bool:
        """Check if the page contains table-like structures.

        Two passes:
        1. PyMuPDF's built-in table detector (catches ruled/bordered tables).
        2. Lane-cooccupancy gate (``has_numeric_columns``) — catches borderless
           regression tables common in academic PDFs.  Replaces the old
           single-token-line-ratio heuristic (``_detect_columnar_numbers``)
           that false-fired on chart-axis labels and CE front-matter where many
           lines each carry a single x/y tick value.

        The lane-cooccupancy gate requires numeric tokens to form a genuine
        row × column grid (multiple distinct x-lanes co-occupied per data row),
        which chart axes never satisfy: tick labels run down a single x position,
        not across several independent lanes simultaneously.
        """
        try:
            tables = page.find_tables()
            if len(tables.tables) > 0:
                return True
        except Exception:
            pass

        from socr.tables.reconstruct import has_numeric_columns

        return has_numeric_columns(page)

    @staticmethod
    def _detect_math_fonts(page: fitz.Page) -> bool:
        """Detect mathematical content via PDF font metadata (source-side).

        PyMuPDF's text extraction silently mangles math typeset with Computer
        Modern, STIX, or similar math fonts: subscripts flatten, Greek letters
        drop, reading order breaks around equations.  The *extracted text*
        looks like normal prose to any string-based checker — so we must read
        the font names embedded in the PDF before extraction ever runs.

        A single match anywhere on the page is sufficient: even one inline math
        span (e.g., ``β̂`` mid-sentence) is enough to corrupt that passage.

        Uses ``page.get_fonts()`` only — it lists all fonts used on the page
        and is O(font_count), far cheaper than a full dict extraction.  Font
        subsets are handled transparently because we use ``re.search`` rather
        than ``re.match``, so ``ABCDEF+CMMI10`` still matches ``CMMI``.
        """
        try:
            for font in page.get_fonts():
                # font tuple: (xref, ext, type, basefont, name, encoding, ...)
                basefont = font[3] if len(font) > 3 else ""
                if basefont and _MATH_FONT_RE.search(basefont):
                    return True
        except Exception:
            pass
        return False

    # Minimum corrupt-math glyphs on a page before it is flagged. A couple of
    # stray control chars can occur incidentally; systematic math corruption
    # produces dozens (observed 5-145 per affected page on the test corpus).
    MIN_MATH_CORRUPTION_GLYPHS = 4

    @staticmethod
    def _count_math_corruption(text: str) -> int:
        """Count font-map mojibake glyphs standing in for math symbols."""
        return sum(1 for ch in text if ch in _MATH_CORRUPTION_GLYPHS)

    def _detect_corrupt_math(self, text: str) -> bool:
        """True if the page's math is font-map corrupted (operators/delimiters
        decoded as Latin-1 letters / fraction glyphs / control chars).

        Distinct from `_encoding_corruption_ratio`, which targets *prose*
        corruption (mid-caps, slash-digits, run-ons) and is blind to these math
        substitutions. The text layer is unrecoverable here, so a flagged page
        routes its equation regions to image-OCR -> LaTeX rather than being
        trusted or whole-page OCR'd.
        """
        return self._count_math_corruption(text) >= self.MIN_MATH_CORRUPTION_GLYPHS

    def _detect_equations(self, text: str) -> bool:
        """Check if text contains raw LaTeX math markup.

        This is a secondary fallback for PDFs that embed literal LaTeX strings
        (rare in typeset documents, but possible in some preprint formats).
        Font-based detection (_detect_math_fonts) is the primary signal and
        runs first.

        Inline dollar signs are common in non-math contexts (currency), so we
        require paired delimiters or explicit LaTeX commands.
        """
        if not text:
            return False

        # LaTeX math commands (high confidence)
        latex_commands = re.compile(
            r"\\(?:frac|sum|int|prod|lim|infty|partial|nabla|alpha|beta|gamma"
            r"|delta|epsilon|theta|lambda|sigma|omega|begin\{(?:equation|align"
            r"|gather|math|displaymath)\})"
        )
        if latex_commands.search(text):
            return True

        # Display math delimiters: $$ ... $$ or \[ ... \]
        if re.search(r"\$\$.+?\$\$", text, re.DOTALL):
            return True
        if re.search(r"\\\[.+?\\\]", text, re.DOTALL):
            return True

        return False

    # ------------------------------------------------------------------
    # Structured text extraction
    # ------------------------------------------------------------------

    def extract_structured(self, page: fitz.Page) -> str:
        """Extract text with tables rendered as markdown.

        For pages with tables, replaces table regions with markdown table
        representations while keeping surrounding prose as plain text.
        For pages without tables, returns plain text (same as get_text()).

        GH-127: hyperlinks in the text layer are emitted as markdown links.
        Resolved once per page and applied to prose and page furniture alike --
        a journal footer is exactly where a DOI lives. A page with no links
        produces byte-identical output to before.

        NOTE: the ``find_tables`` failure path below returns raw ``get_text``
        and therefore still drops links. That is the pre-existing degraded
        path for a damaged page; recovering links there needs the dict walk
        this function does after it, and is deliberately out of scope here.
        """
        try:
            tables_result = page.find_tables()
        except Exception:
            return page.get_text("text").strip()

        # GH-127: resolved once, not per line -- get_links() parses the page's
        # link table on every call.
        _links = _uri_links(page)

        # Collect table bounding boxes and their markdown representations.
        # For each table returned by find_tables(), detect lane-stacking: a
        # whitespace-gutter table (no ruling lines, CE-style) may come back with
        # embedded newlines inside cells because the "lines" strategy collapses
        # whitespace-separated columns into one cell.  Such a region is
        # lane-stacked — route it to the word-geometry rowizer rather than the
        # passthrough _table_to_markdown path that preserves the embedded newlines.
        table_regions: list[tuple[fitz.Rect, str]] = []
        lane_stacked_regions: list[tuple[fitz.Rect, str]] = []
        for table in tables_result.tables:
            if _is_lane_stacked(table):
                # Route to word-geometry rowizer scoped to this region's words.
                # A multi-schema merge is TR-2's job: if rowize_from_word_list
                # cannot produce a valid single-schema grid from these words it
                # returns [] and we fall through to the whole-page fallbacks.
                from socr.tables.reconstruct import rowize_from_word_list

                bbox = fitz.Rect(table.bbox)
                region_words = [
                    w for w in page.get_text("words") if bbox.contains(fitz.Point(w[0], w[1]))
                ]
                rowized = rowize_from_word_list(region_words)
                for rect, md in rowized:
                    lane_stacked_regions.append((rect, md))
            else:
                md = self._table_to_markdown(table)
                if md:
                    table_regions.append((fitz.Rect(table.bbox), md))

        # Merge: lane-stacked replacements take priority; non-stacked tables keep
        # their passthrough markdown.  The rowizer may produce more or fewer
        # regions than the original find_tables() result — that is expected.
        table_regions.extend(lane_stacked_regions)

        # Born-digital booktabs tables (top/mid/bottom rules only) make the
        # default lines strategy return nothing, so the table would otherwise be
        # dumped as a flat token stream. Recover the grid from text alignment
        # (char-exact native values, no model). reconstruct_table_regions self-
        # gates on numeric-column structure, so it is safe to call on any page.
        if not table_regions:
            from socr.tables.reconstruct import reconstruct_table_regions

            # GH-195: collect the text-strategy grid rejections so the page can
            # SURFACE them. The fallback rowizer is proven lossless, so the page
            # is not demoted — but "this page's layout is adversarial to
            # find_tables(strategy='text') and a grid had to be rejected and
            # rebuilt" is operationally different from "everything rendered
            # cleanly first time", and until now only a log line said so.
            _rejections: list[dict] = []
            table_regions = reconstruct_table_regions(page, rejections=_rejections)
            if _rejections:
                self._last_extraction_grid_rejections = list(_rejections)

        # Word-geometry rowizer fallback (TR-1/TR-2): when find_tables() returns
        # nothing AND the text-strategy reconstruct also fails (e.g. a multi-region
        # page where the text strategy over-merges chart + table + prose into one
        # spanning grid that fails _looks_tabular), try segmenting the page by
        # vertical gaps derived from the page's own row-height distribution and
        # rowizing each segment independently.  This is the correct path for
        # CE-style whitespace-gutter tables on pages with multiple content regions.
        #
        # TR-2 chart-clip: use the chart-aware variant so chart-drawing clusters
        # are excluded from the rowizer and returned as image-ref placeholders.
        # The chart's tick-label rows (all single-column) previously diluted the
        # historical table's data_row_frac below _looks_tabular's 0.5 threshold,
        # causing it to fall back to flat text.  Clipping the chart words first
        # fixes this without any magic threshold.
        if not table_regions:
            from socr.tables.reconstruct import rowize_from_words_chart_aware

            page_num = getattr(page, "number", 0) + 1
            table_regions = rowize_from_words_chart_aware(page, page_num=page_num)

        if not table_regions:
            # GH-127: this is the prose-page path -- no dict walk happens here,
            # so links are applied to the flat string (see the helper's note).
            try:
                _flat_words = page.get_text("words") or []
            except Exception:
                _flat_words = []
            return _apply_links_to_flat_text(page.get_text("text").strip(), _links, _flat_words)

        # Sort regions top-to-bottom by their y0 coordinate (reading order).
        # Column-aware ordering (x-band then y) would be needed for multi-column
        # layouts; for single-column pages (CE p.4 style) y0 alone is correct.
        # This is documented as a known limitation in the design note §1b.
        table_regions.sort(key=lambda tr: tr[0].y0)

        # TR-2/TR-3 per-region verifier scoping: verify each table region
        # independently against its own native numeric lanes.  The whole-page
        # verifier (verify_native_table) combines lanes from ALL tables on the
        # page, which causes false geometry_impossible_collapse hard-fails when
        # two tables with different schemas sit on the same page (e.g. a 4-col
        # forecaster grid + a 3-col historical table → 7 combined lanes but each
        # table only has 3-4).  Calling verify_native_table_region per region
        # fixes this by clipping get_text("words") to the region bbox first.
        # TR-3: _verify_regions now returns True when any region hard-fails so
        # _assess_page can flag the page for D3 fail-closed routing.
        had_unverifiable = self._verify_regions(page, table_regions)
        # Store on the instance so _assess_page can read it after the call to
        # extract_structured — the PageAssessment is built there, not here.
        self._last_extraction_had_unverifiable = had_unverifiable

        # TR-2 token-coverage post-check: every native numeric token must land in
        # exactly one region (no orphaned / double-counted token).  This is a
        # deterministic safety net — not a gate that suppresses output, just a
        # debug log so operators can trace lost tokens without re-running.
        self._check_token_coverage(page, table_regions)

        # Collect all region bboxes for the overlap-suppress check below.
        all_region_rects = [r for r, _ in table_regions]

        # Build output by interleaving prose text and region content.
        # Use text blocks from get_text("dict") to get position-aware text.
        try:
            page_dict = page.get_text("dict")
        except Exception:
            return page.get_text("text").strip()

        blocks = page_dict.get("blocks", [])
        output_parts: list[str] = []
        relegated_parts: list[str] = []
        region_idx = 0
        dominant = dominant_text_direction(blocks)

        for block in blocks:
            # Skip image blocks (type 1)
            if block.get("type", 0) == 1:
                continue

            # Step 0: text running against the page's prevailing direction is
            # RELEGATED to the end of the page, not deleted.
            #
            # The problem being solved is reading order, not the existence of the
            # text: a publisher stamp spliced into the middle of a paragraph
            # corrupts the prose around it. Relegation fixes that completely.
            #
            # Deleting it would ALSO discard legitimately rotated content — a
            # chart's axis labels, a rotated table header, a marginal note — and
            # this module must not delete text it cannot prove is worthless.
            # Direction is evidence about layout, never about value (#145 review).
            if block_is_page_furniture(block, dominant):
                for line in block.get("lines", []) or []:
                    # GH-127: furniture carries links too -- a journal footer is
                    # exactly where a DOI lives, so it must not lose them.
                    text = _line_text(line.get("spans", []) or [], _links)
                    if text.strip():
                        relegated_parts.append(text.strip())
                continue

            block_rect = fitz.Rect(block["bbox"])

            # Check if we need to insert a region before this block
            while region_idx < len(table_regions):
                region_rect, region_content = table_regions[region_idx]
                if region_rect.y0 <= block_rect.y0:
                    output_parts.append(f"\n{region_content}\n")
                    region_idx += 1
                else:
                    break

            # How much of this block does a table region already represent?
            #
            # This was `any(block_rect.intersects(r))` — true on ONE POINT of
            # contact, dropping the whole block. A table region routinely ends a
            # few points below its last rule, clipping the top line of the notes
            # paragraph beneath it; the entire paragraph then vanished from a page
            # that still shipped SUCCESS. On the reference page the notes block was
            # 23% inside the region and 77% outside, taking with it the sample
            # sizes and "robust standard errors are in parentheses" (#145).
            #
            # Coverage is measured PER REGION and the content test is run against
            # the tokens of the regions this block actually overlaps. A combined
            # index across every region on the page would let a line be deleted
            # because its words appear in a DIFFERENT table — which does not make
            # this region's replacement contain it.
            covering = [
                (r, md)
                for (r, md) in table_regions
                if _rect_coverage(block_rect, r) >= _REGION_COVERAGE_DROP
            ]

            lines = block.get("lines", []) or []
            if covering:
                # Mostly inside a region — but geometry alone must NOT decide. A
                # region's rectangle routinely spans text its markdown never
                # captured: the table's own title and column headers sit inside the
                # bbox, and an over-long region swallows the first lines of the
                # notes paragraph beneath the table. Dropping on position deleted
                # all of those.
                #
                # So drop a line only when the covering region's markdown
                # DEMONSTRABLY already contains it. Absence of the text in the
                # replacement is proof that dropping would lose it (#145).
                #
                # A block less than half inside any region keeps every line
                # untouched: the region does not represent it, so nothing about it
                # is a candidate for suppression.
                covering_tokens = _region_token_index(covering)
                lines = [ln for ln in lines if not _line_is_in_region_text(ln, covering_tokens)]

            for line in lines:
                spans = line.get("spans", [])
                line_text = _line_text(spans, _links)
                if line_text.strip():
                    output_parts.append(line_text.strip())

        # Append any remaining regions that come after all text blocks
        while region_idx < len(table_regions):
            _, region_content = table_regions[region_idx]
            output_parts.append(f"\n{region_content}\n")
            region_idx += 1

        # Counter-directional text last, so it never interrupts the reading order
        # it was removed from — and never disappears either.
        output_parts.extend(relegated_parts)

        return "\n".join(output_parts).strip()

    def _verify_regions(
        self,
        page: fitz.Page,
        regions: list[tuple[fitz.Rect, str]],
    ) -> bool:
        """TR-2/TR-3 per-region native verifier.

        For each table region, clips get_text("words") to the region bbox and
        runs the two-tier geometry check against the region's own markdown text.
        This avoids false hard-fails that arise when whole-page lane counting
        merges columns from multiple tables with different schemas.

        Only regions whose text contains a markdown table separator are checked
        (image-asset placeholders like chart refs have no numeric lanes).

        Returns
        -------
        bool
            True if ANY table region produced a geometry hard-fail
            (``geometry_impossible_collapse``).  The caller stores this flag on
            the PageAssessment so the D3 fail-closed floor in
            ``_winning_page_output`` can route an unverifiable page to the
            image-asset lane instead of silently shipping the collapsed native.
        """
        from socr.tables.native_verifier import verify_native_table_region

        any_hard_fail = False
        for region_rect, region_text in regions:
            # Skip non-table regions (chart image refs, prose blocks)
            if "| --- |" not in region_text and "| --- " not in region_text:
                continue
            try:
                vr = verify_native_table_region(page, region_text, region_rect)
            except Exception as exc:
                logger.debug("per-region verifier failed for region %s: %s", region_rect, exc)
                continue

            if vr.hard_fail:
                any_hard_fail = True
                logger.warning(
                    "per-region verifier: geometry_impossible_collapse in region y=%.0f..%.0f — %s",
                    region_rect.y0,
                    region_rect.y1,
                    vr.reason,
                )
            elif vr.warn:
                logger.debug(
                    "per-region verifier: warn for region y=%.0f..%.0f — %s",
                    region_rect.y0,
                    region_rect.y1,
                    vr.reason,
                )
        return any_hard_fail

    def _check_token_coverage(
        self,
        page: fitz.Page,
        regions: list[tuple[fitz.Rect, str]],
    ) -> None:
        """TR-2 token-coverage post-check (diagnostic, not a gate).

        Verifies that every native numeric token on the page falls inside at
        most one table/chart region.  Orphaned tokens (in no region) and
        double-counted tokens (in two or more overlapping regions) are both
        logged at DEBUG level so operators can trace lost or duplicated values
        without re-running the pipeline.

        This is a deterministic safety net — it never suppresses output.  A
        future escalation path (TR-5 VLM confirm/split) can use these counts
        to decide when geometry-led segmentation has failed.
        """
        from socr.core.table_grid import NUM_TOKEN_RE, NUMERIC_RE

        try:
            words = page.get_text("words")
        except Exception:
            return

        # Collect all region bboxes.
        region_rects = [r for r, _ in regions]

        orphaned: list[str] = []
        double_counted: list[str] = []

        for w in words:
            x0, y0, _, _, text = w[0], w[1], w[2], w[3], w[4]
            if not (NUM_TOKEN_RE.match(text) and NUMERIC_RE.search(text)):
                continue

            hits = sum(1 for r in region_rects if r.x0 <= x0 <= r.x1 and r.y0 <= y0 <= r.y1)
            if hits == 0:
                orphaned.append(f"{text}@({x0:.0f},{y0:.0f})")
            elif hits > 1:
                double_counted.append(f"{text}@({x0:.0f},{y0:.0f})")

        if orphaned:
            logger.debug(
                "token-coverage: %d orphaned numeric token(s) not in any region: %s",
                len(orphaned),
                orphaned[:10],
            )
        if double_counted:
            logger.debug(
                "token-coverage: %d double-counted numeric token(s) in >1 region: %s",
                len(double_counted),
                double_counted[:10],
            )

    def _table_to_markdown(self, table: object) -> str:
        """Convert a PyMuPDF Table object to a markdown table string.

        Args:
            table: A table object from page.find_tables() with an
                   extract() method returning a list of rows (lists of cells).

        Returns:
            Markdown table string, or empty string if table is empty/invalid.
        """
        try:
            rows = table.extract()
        except Exception:
            return ""

        if not rows:
            return ""

        # Clean cell values: replace None with empty string, strip whitespace
        cleaned: list[list[str]] = []
        for row in rows:
            cleaned.append([(cell.strip() if isinstance(cell, str) else "") for cell in row])

        if not cleaned:
            return ""

        # Build markdown table
        col_count = max(len(row) for row in cleaned)
        # Pad rows to uniform column count
        for row in cleaned:
            while len(row) < col_count:
                row.append("")

        lines: list[str] = []

        # Header row
        header = cleaned[0]
        lines.append("| " + " | ".join(header) + " |")

        # Separator
        lines.append("| " + " | ".join("---" for _ in header) + " |")

        # Data rows
        for row in cleaned[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Low-level page analysis helpers
    # ------------------------------------------------------------------

    def _count_fonts(self, page: fitz.Page) -> int:
        """Count distinct fonts used on a page.

        Born-digital pages typically use 2-6 fonts (body, bold, italic, math).
        Scanned pages with OCR layers often use 0 or 1 font.
        """
        fonts = page.get_fonts()
        # Each font entry is (xref, ext, type, basefont, name, encoding)
        unique_names = {f[3] for f in fonts if f[3]}
        return len(unique_names)

    def _has_images(self, page: fitz.Page) -> bool:
        """Check if the page has embedded raster images.

        Scanned pages are essentially large images. Born-digital pages may
        have images too (figures), but combined with a rich text layer.
        """
        images = page.get_images()
        return len(images) > 0

    @staticmethod
    def _raster_coverage(page: fitz.Page) -> float:
        """Fraction of page area covered by embedded raster images (0.0 – 1.0).

        Uses ``page.get_image_info()`` which returns per-image bounding boxes in
        PDF user-space coordinates (same coordinate system as ``page.rect``).

        Area is computed as the SUM of individual image bbox areas, clamped to
        the page area, rather than a union bounding box.  Union would overcount
        for non-overlapping images (e.g. two small charts placed side-by-side
        would give a bounding box that covers the gap between them).  Summing
        individual areas slightly overcounts for overlapping images (rare in
        practice), but never exceeds 1.0 after clamping — still safe for a
        dominance threshold comparison.

        Returns 0.0 on any exception so that a buggy or unavailable API never
        wrongly gates an otherwise clean classification.
        """
        try:
            page_area = page.rect.get_area()
            if page_area <= 0:
                return 0.0
            image_area_sum = sum(
                fitz.Rect(info["bbox"]).get_area()
                for info in page.get_image_info()
                if info.get("bbox")
            )
            return min(image_area_sum / page_area, 1.0)
        except Exception:
            return 0.0

    def _encoding_corruption_ratio(self, text: str) -> float:
        """Fraction of word tokens that show font/ToUnicode encoding corruption.

        Catches the failure mode the garbage-char check misses: VALID characters
        in wrong positions from a broken font map. Three structural signals, none
        of which clean prose produces in quantity:

        - mid-word capital ("ofFinancial", "FrenchfJoumal") — a capital after a
          lowercase inside a token, from dropped inter-word spaces;
        - slash-for-digit ("(/997)", "/53") — a stroke glyph decoded as '/';
        - run-on tokens (>= 16 chars) — fused words.

        Returns 0.0 for fewer than 20 alpha tokens (too little to judge).
        """
        tokens = text.split()
        alpha = [t for t in tokens if any(c.isalpha() for c in t)]
        if len(alpha) < 20:
            return 0.0
        midcap = sum(1 for t in alpha if re.search(r"[a-z][A-Z]", t))
        slash_digit = sum(1 for t in tokens if re.search(r"/\d|\d/|\(/", t))
        run_on = sum(1 for t in alpha if len(t) >= 16)
        return (midcap + slash_digit + run_on) / len(alpha)

    def _garbage_ratio(self, text: str) -> float:
        """Ratio of garbage characters to total characters.

        Garbage = control chars, replacement chars, private-use-area chars,
        and other non-printable characters that shouldn't appear in real text.
        """
        if not text:
            return 0.0

        garbage_count = 0
        for ch in text:
            cp = ord(ch)
            # Control chars (except newline, tab, carriage return)
            if cp < 0x20 and cp not in (0x09, 0x0A, 0x0D):
                garbage_count += 1
            # Replacement character
            elif cp == 0xFFFD:
                garbage_count += 1
            # Private use area
            elif 0xE000 <= cp <= 0xF8FF:
                garbage_count += 1
            # Surrogates (should not appear in valid text)
            elif 0xD800 <= cp <= 0xDFFF:
                garbage_count += 1

        return garbage_count / len(text)

    def _compute_confidence(
        self,
        char_count: int,
        word_count: int,
        garbage_ratio: float,
        space_ratio: float,
        avg_word_len: float,
        font_count: int,
        has_images: bool,
    ) -> float:
        """Compute confidence score for born-digital classification.

        Starts at a base confidence and adjusts based on signal strength.
        More text, more fonts, lower garbage = higher confidence.
        """
        confidence = 0.80

        # More text = more confident
        if word_count > 100:
            confidence += 0.05
        if word_count > 300:
            confidence += 0.05

        # Very clean text = more confident
        if garbage_ratio < 0.01:
            confidence += 0.03

        # Multiple fonts = clearly authored, not OCR
        if font_count >= 2:
            confidence += 0.03
        if font_count >= 4:
            confidence += 0.02

        # Normal word lengths
        if 3.5 <= avg_word_len <= 7.0:
            confidence += 0.02

        return min(confidence, 1.0)
