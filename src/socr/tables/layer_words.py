"""#1294: which words of a page's own text layer is the shipped page owed?

The 2026-11-11 audit found SUCCESS pages that had lost words their PDF prints ("Core PCE
inflation" shipped as "PCE inflation", "seasonally adjusted annual rates" and "projected"
gone), and no check compared shipped words with the layer: ``text_layer_numbers`` and
``text_layer_drops`` compare numbers, on model readings only.

This module is the words side. :func:`word_reference_from_page` reads the page once, at
page analysis, and keeps every word a reader of the page is owed, minus two classes that a
correct page legitimately does not carry as text (each measured on the Fed gold pages,
2026-11-12, see ``docs/log/2026-11-12_1294-text-layer-words.md``):

* **Repeated text at the page edge** -- a word at a repeated slot (the document's running
  heads, release stamps, classification banners; ``born_digital._repeated_text_slots``) that
  lies above every lettered body word, below every one, or in a block set against the
  page's writing direction. Repeated text INSIDE the body is kept: a table heading repeated
  on every page ("percentage point contribution to growth") is content, and excusing all
  repeated text hid real losses.
* **Words inside a chart or image box** are kept apart (:attr:`WordReference.figure_text`):
  they are owed only when the page does not ship the figure as a picture.

:func:`missing_words` compares a shipped page with the reference. It is a multiset
difference over words of at least :data:`MIN_WORD_LETTERS` letters, with three rescues for
differences that are not losses: a line-break hyphen (the layer's ``Home-``/``builders`` is
the shipped ``Homebuilders``), a join of consecutive shipped words (the layer's fused
``onthe``, the page's split ``head winds``), and a ligature whose second glyph the layer
dropped (``infation`` for ``inflation``). Every rescue is word-aligned: a word is never found
inside another one. A rescue can hide a loss, never invent one.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

#: The shortest word compared. Shorter words ("of", "the", "and", "in") occur many times
#: on a page, so a multiset deficit in them says nothing about which sentence lost one,
#: and the audit's text-layer counts (fable, 2026-11-11) used this same definition, so the
#: scoreboard's ``textlayer.words_missing`` and this check count the same words.
MIN_WORD_LETTERS = 4

#: Unicode's Mathematical Alphanumeric Symbols (math italic ``𝑦``, ``𝑟``): a formula's
#: variables, which a layer prints as letters and a reading writes as LaTeX. Not words.
_MATH_LETTERS = "\U0001d400-\U0001d7ff"
_WORD_RE = re.compile(rf"[^\W\d_{_MATH_LETTERS}]{{{MIN_WORD_LETTERS},}}")
_ANY_WORD_RE = re.compile(r"[^\W\d_]+")
_LETTER_RE = re.compile(r"[^\W\d_]")
#: A hyphen between two letters inside a shipped word ("longer-term"): the layer may print
#: it at a line break, where :func:`word_reference_from_page` joins the two halves.
_INWORD_HYPHEN_RE = re.compile(r"([^\W\d_])-([^\W\d_])")
#: A line-break hyphen the shipped page kept as printed ("infla-" / "tion"): the native lane
#: keeps a split it has no witness for (#1074), and the reference joined it.
_BREAK_HYPHEN_RE = re.compile(r"([^\W\d_])[-\u00ad][ \t]*\n[ \t]*([^\W\d_])")
#: Line-end hyphens: ASCII hyphen-minus and the soft hyphen.
_LINE_END_HYPHENS = ("-", "­")
#: Typographic ligatures a damaged layer drops or splits (``fi`` printed as one glyph).
_LIGATURES = ("ffi", "ffl", "fi", "fl", "ff")


@dataclass(frozen=True)
class WordReference:
    """The words a shipped page owes its own text layer, space-separated."""

    #: words outside every chart and image box, in reading order
    text: str
    #: words inside a chart or image box: owed only when no figure picture ships
    figure_text: str = ""


#: The soft hyphen: an invisible break hint a layer may print inside a word
#: ("guaran\u00adteed"). It is not a character of the word.
_SOFT_HYPHEN = "\u00ad"


def word_tokens(text: str) -> Counter:
    """The compared words of *text*, case-folded, as a multiset."""
    return Counter(word_tokens_in_order(text))


def _ligature_variants(word: str) -> set[str]:
    """*word* with each ligature collapsed to its ``f`` (``inflation`` -> ``infation``): the
    form a layer that dropped the ligature's second glyph prints. Not the form with the whole
    ligature deleted, which would rescue unrelated words (``iation``)."""
    return {word.replace(lig, "f") for lig in _LIGATURES if lig in word}


def _joins(parts: list[str], targets: set[str]) -> Counter:
    """How often each of *targets* is the join of two or more consecutive *parts*: the layer
    fused words the page ships apart ("onthe"), or the page split a word ("head winds").
    Word-aligned on both ends, so a word is never found inside another one ("core" in
    "underscore")."""
    found: Counter = Counter()
    if not targets:
        return found
    longest = max(len(t) for t in targets)
    for i, first in enumerate(parts):
        joined = first
        for nxt in parts[i + 1 :]:
            joined += nxt
            if len(joined) > longest:
                break
            if joined in targets:
                found[joined] += 1
    return found


def missing_words(
    reference: WordReference, shipped: str, *, figure_shipped: bool, excused: str = ""
) -> list[str]:
    """The reference's words *shipped* does not carry, in reference order, with repeats.

    *shipped* is the page's own reading (socr's marks and notes already removed by the
    caller). *figure_shipped* is whether the page ships a picture, which carries the words
    inside chart and image boxes. *excused* is text the page deliberately does not ship although
    its layer prints it (#1306: a hidden footnote wrapper stripped because the printed copy of the
    note is on the page); its words are subtracted from what is owed, one for one, so removing the
    duplicate causes no missing word and a word printed only there stays owed.
    """
    owed = word_tokens(reference.text)
    if not figure_shipped:
        owed += word_tokens(reference.figure_text)
    owed -= word_tokens(excused)
    joined = _INWORD_HYPHEN_RE.sub(r"\1\2", _BREAK_HYPHEN_RE.sub(r"\1\2", shipped or ""))
    have = word_tokens(shipped) | word_tokens(joined)
    ligatures = set().union(*(_ligature_variants(w) for w in have)) if have else set()
    deficit = owed - have
    parts = [w.casefold() for w in _ANY_WORD_RE.findall((shipped or "").replace(_SOFT_HYPHEN, ""))]
    joins = _joins(parts, set(deficit))
    short: Counter = Counter()
    for word, n in deficit.items():
        if have.get(word, 0) == 0 and word in ligatures:
            continue
        n -= min(n, joins.get(word, 0))
        if n:
            short[word] = n
    out = []
    order = reference.text.casefold().split() + reference.figure_text.casefold().split()
    for word in order:
        if short.get(word, 0) > 0:
            out.append(word)
            short[word] -= 1
    return out


def _centre_in(word: Sequence, boxes: Sequence[Sequence[float]]) -> bool:
    cx, cy = (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def edge_furniture(words: Sequence[Sequence], is_furniture, sideways_blocks: set[int]) -> set[int]:
    """Indexes of *words* that are repeated text at the page edge.

    A word qualifies when ``is_furniture(word)`` and it lies above every lettered body word,
    below every one, or in a block of *sideways_blocks*. A word with no letter (a page
    number) is not body: a folio above a running head must not pull the head into the body.
    """
    body = [w for w in words if not is_furniture(w) and _LETTER_RE.search(str(w[4]))]
    top = min((w[1] for w in body), default=float("inf"))
    bottom = max((w[3] for w in body), default=float("-inf"))
    return {
        i
        for i, w in enumerate(words)
        if is_furniture(w) and (w[3] <= top or w[1] >= bottom or w[5] in sideways_blocks)
    }


def reference_from_words(
    words: Sequence[Sequence],
    is_furniture,
    *,
    sideways_blocks: set[int] = frozenset(),
    figure_boxes: Sequence[Sequence[float]] = (),
) -> WordReference:
    """The pure core of :func:`word_reference_from_page`, over ``get_text("words")`` tuples.

    Words are taken in PyMuPDF's order (block, line, word). A line-end hyphen followed by a
    next word on another line that starts with a lower-case letter is a hyphenation break:
    the two halves are one word. The next line may be another block: some generators write
    every line as its own block (Fed Beige Books, 2024-2026).
    """
    edge = edge_furniture(words, is_furniture, set(sideways_blocks))
    kept = [(w, _centre_in(w, figure_boxes)) for i, w in enumerate(words) if i not in edge]
    body: list[str] = []
    figure: list[str] = []
    i = 0
    while i < len(kept):
        word, in_figure = kept[i]
        text = str(word[4])
        nxt = kept[i + 1][0] if i + 1 < len(kept) else None
        if (
            nxt is not None
            and text.endswith(_LINE_END_HYPHENS)
            and (nxt[5], nxt[6]) != (word[5], word[6])
            and str(nxt[4])[:1].islower()
        ):
            text = text.rstrip("".join(_LINE_END_HYPHENS)) + str(nxt[4])
            i += 1
        (figure if in_figure else body).extend(word_tokens_in_order(text))
        i += 1
    return WordReference(" ".join(body), " ".join(figure))


def word_tokens_in_order(text: str) -> list[str]:
    """The compared words of *text*, case-folded, in order. A soft hyphen inside a word is
    dropped (a line-end one is handled by :func:`reference_from_words` first)."""
    text = (text or "").replace(_SOFT_HYPHEN + "\n", _SOFT_HYPHEN).replace(_SOFT_HYPHEN, "")
    return [w.casefold() for w in _WORD_RE.findall(text)]


def word_reference_from_page(page, *, repeated_slots, folio_slots) -> WordReference:
    """The reference for :func:`missing_words`, read from a PyMuPDF page.

    ``repeated_slots`` and ``folio_slots`` are ``BornDigitalDetector._repeated_text_slots``
    and ``_folio_slots`` of the page's document. Chart and image boxes are the ones
    ``layer_body.drop_reference_from_page`` excludes (``chart_region_bboxes`` and
    ``get_image_info``), in the same unrotated coordinates as the words. Raises on a
    PyMuPDF failure; the caller must treat that as "not checked".
    """
    from socr.core.born_digital import block_is_page_furniture, dominant_text_direction
    from socr.tables.layer_body import slot_predicate
    from socr.tables.reconstruct import chart_region_bboxes

    words = page.get_text("words")
    blocks = page.get_text("dict").get("blocks", [])
    dominant = dominant_text_direction(blocks)
    sideways = {
        int(b.get("number", n))
        for n, b in enumerate(blocks)
        if b.get("type", 0) == 0 and block_is_page_furniture(b, dominant)
    }
    boxes = [tuple(r) for r in chart_region_bboxes(page)]
    boxes += [tuple(i["bbox"]) for i in page.get_image_info()]
    return reference_from_words(
        words,
        slot_predicate(repeated_slots, folio_slots),
        sideways_blocks=sideways,
        figure_boxes=boxes,
    )
