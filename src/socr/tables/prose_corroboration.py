"""#1043: corroborate a model reading's PROSE against the page's own text layer.

A page whose every model reading was rejected only because of a table used to ship as
the whole-page fail-closed floor, so its correct prose was lost as text. This module is
the mechanical test that lets that prose ship.

The first version scored a bag of bigrams and a bag-of-words recall; a review ran
counterexamples against it and it shipped dropped sentences, changed decimals, an inserted
minus sign, flipped meanings and a mutated table row. A bag cannot satisfy "a wrong number
is worse than a missing one", so the guard is now an ORDERED, STRICT ALIGNMENT:

* the reading's prose (everything outside its markdown table blocks) is aligned, in order,
  against the layer's tokens with ``difflib.SequenceMatcher``;
* EVERY non-equal step must be explainable as OCR noise ONLY (``NOISE_CLASS`` below).
  Any inserted, deleted or replaced word, number, sign or negation refuses the page;
* numbers are whole tokens, sign and decimal included, so ``4.2`` is not ``4.8`` and is not
  ``-4.2``. They are compared against the layer's own bands, never against the table;
* the reading must COVER the layer end to end. A layer band the reading does not carry is a
  refusal, unless that band is the withheld table itself (see ``_table_band_flags``).

Noise class (deliberately tiny, applied identically to both sides):

1. case, surrounding punctuation and markdown decoration, Unicode compatibility forms
   (ligatures, fullwidth forms, quote variants), soft hyphens and zero-width characters,
   and the Unicode minus sign read as ``-``;
2. a purely alphabetic run split or joined differently (line-break hyphenation, a word
   cut in two by the layer): the runs are equal once non-alphanumerics are removed.

A digit-bearing token has NO noise: it must be equal after class 1.

Known limits, stated rather than hidden:

* COMMON MODE. A line missing from BOTH the layer and the reading is invisible to a
  comparison between the two. socr has no measure of how complete an invisible layer is,
  and one witness cannot close this; the page ships WARNING with the tables withheld, so
  it is never a clean page.
* A layer band is treated as the withheld table when every alphabetic word on it occurs in
  the reading's own table block (numerals are free: the table was rejected, its numbers
  may be wrong or missing), within a budget of the table's own token count. A dropped
  prose line built ONLY from the table's own words would hide there.
* An invisible OCR layer is noisy, so a strict alignment refuses many pages whose prose
  is in fact right. That is the intended direction.

The earlier cutoffs (0.90 similarity, derived from same-document negative controls) are
gone with the bag-of-words score. ``PROSE_CORROBORATION_MIN_TOKENS`` stays: four of 3,363
scored negative controls (3,444 built) were identical short running captions that score a
perfect 1.0 under ANY measure, every one with 16 or fewer prose tokens, so a prose region
shorter than that cannot discriminate (``docs/log/2026-10-08_table-only-floor-prose.md``).
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

#: Fewest prose tokens a reading may carry and still be corroborated. See module docstring.
PROSE_CORROBORATION_MIN_TOKENS = 17

#: What may differ between the reading and the layer without refusing the page. Documentation
#: of the two classes implemented by ``_normalise`` and ``_alpha_join_equal``.
NOISE_CLASS = (
    "case, surrounding punctuation, markdown decoration, Unicode compatibility forms, "
    "soft hyphen / zero-width, unicode minus; and an alphabetic run split or joined "
    "differently (hyphenation, line-break joins)"
)

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MARKER_RE = re.compile(r"\[page \d+[^\]]*\]|\[socr:[^\]]*\]")

#: The rejection reasons socr's table gates author, by prefix. A reading refused with one
#: of these was refused for its table, by construction.
TABLE_GATE_PREFIXES = ("table_structure_failed", "source_evidence_table", "native_table_verifier")

#: A free-text judge clause that names a table structure. Deliberately the weak link of
#: the table-only test (a model chose the words): the ordered alignment, not this
#: pattern, is what stops a wrong page from shipping.
_TABLE_CLAUSE_RE = re.compile(
    r"\btable|\bcolumns?\b|\brows?\b|\bgrid\b|\bheaders?\b", re.IGNORECASE
)

#: A clause that ALSO names something that is not a table makes the rejection mixed, whatever
#: else it says ("the table is malformed and the figure axis labels are missing"). The page
#: then floors: the judge refused more than the table.
_NON_TABLE_CLAUSE_RE = re.compile(
    r"\bfigures?\b|\baxis\b|\baxes\b|\bcaptions?\b|\bequations?\b|\bfootnotes?\b"
    r"|\bparagraphs?\b|\btext\b|\blegends?\b",
    re.IGNORECASE,
)


def rejection_is_table_only(reason: str) -> bool:
    """Whether a rejection reason names only table defects.

    A gate reason (``table_structure_failed: ...``) is table-only by construction. Free
    judge text is split into clauses and EVERY clause must name a table structure and
    nothing else: one clause about a missing paragraph, a figure or an axis makes the
    rejection mixed. An empty reason is unknown, never table-only.
    """
    reason = (reason or "").strip()
    if not reason:
        return False
    if reason.startswith(TABLE_GATE_PREFIXES):
        return True
    clauses = [c.strip() for c in re.split(r"[;\n]", reason) if c.strip()]
    return bool(clauses) and all(
        _TABLE_CLAUSE_RE.search(c) and not _NON_TABLE_CLAUSE_RE.search(c) for c in clauses
    )


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

#: Characters stripped from both ends of a token (noise class 1): quotes, brackets, emphasis,
#: heading and code marks, escapes and clause punctuation. NOT stripped: ``.`` inside or
#: leading a number, ``%``, ``/``, ``$``, ``+`` and ``-`` (they are part of a number).
_EDGE = "\"'“”‘’„‚«»()[]{}<>*_#`\\|:;,!?~"
_STANDALONE_DASHES = frozenset({"-", "–", "—", "‒"})
_DASH_MINUS_RE = re.compile(r"^[–—‒](?=\.?\d)")


def _has_digit(token: str) -> bool:
    return any(ch.isdigit() or ch.isnumeric() for ch in token)


def _normalise(raw: str) -> str:
    """One whitespace-delimited word under noise class 1; ``""`` when nothing is left."""
    t = unicodedata.normalize("NFKC", raw).casefold()
    t = t.replace("­", "").replace("​", "").replace("−", "-")
    t = t.strip(_EDGE).rstrip(".").strip(_EDGE)
    t = _DASH_MINUS_RE.sub("-", t)
    return "" if t in _STANDALONE_DASHES else t


def _tokens(text: str) -> list[str]:
    return [t for t in (_normalise(w) for w in text.split()) if t]


def _alnum(span: list[str]) -> str:
    return "".join(ch for ch in "".join(span) if ch.isalnum())


def _alpha_join_equal(reading: list[str], layer: list[str]) -> bool:
    """Noise class 2: two alphabetic runs equal once split/join and hyphens are ignored."""
    if any(_has_digit(t) for t in reading) or any(_has_digit(t) for t in layer):
        return False
    joined = _alnum(reading)
    return bool(joined) and joined == _alnum(layer)


def strip_image_refs(text: str) -> str:
    """*text* without markdown image references (model-authored paths are never trusted)."""
    return _IMAGE_RE.sub("", text or "")


def prose_of(reading_text: str) -> tuple[str, list[str], int]:
    """The reading's prose, its table lines, and its table-block count.

    Markdown table blocks, image references, HTML comments and socr markers are removed from
    the prose; the removed table lines are returned for the table vocabulary.
    """
    from socr.tables.reconcile import find_table_blocks

    blocks = find_table_blocks(reading_text)
    lines = reading_text.splitlines()
    drop: set[int] = set()
    for block in blocks:
        drop.update(range(block.start, block.end + 1))
    prose = "\n".join(ln for i, ln in enumerate(lines) if i not in drop)
    for pattern in (_IMAGE_RE, _COMMENT_RE, _MARKER_RE):
        prose = pattern.sub(" ", prose)
    return prose, [lines[i] for i in sorted(drop)], len(blocks)


# ---------------------------------------------------------------------------
# The withheld table, located in the layer
# ---------------------------------------------------------------------------


def _layer_bands(native_words: list) -> list[list[str]]:
    """The layer's baseline bands, top of page to bottom, each as normalised tokens."""
    from socr.tables.row_corroboration import partition_prose_bands

    bands: list[list[str]] = []
    for _is_prose, band in partition_prose_bands(list(native_words)):
        toks = [t for t in (_normalise(str(w[4])) for w in band) if t]
        if toks:
            bands.append(toks)
    return bands


def _table_band_flags(bands: list[list[str]], table_tokens: list[str]) -> list[bool]:
    """Which layer bands are the withheld table itself.

    A band is the table when every ALPHABETIC word on it occurs in the reading's own table
    block. Numerals are free: this table was rejected, so its numbers may be wrong or
    missing, and the point is only to know which layer bands the prose must NOT carry.
    The flagged tokens may not exceed the table's own token count (``_budget``): a table
    cannot account for more layer than it has cells.
    """
    vocab = {t for t in table_tokens if not _has_digit(t)}
    budget = len(table_tokens)
    flags: list[bool] = []
    for band in bands:
        is_table = all(_has_digit(t) or t in vocab for t in band)
        if is_table and len(band) <= budget:
            budget -= len(band)
            flags.append(True)
        else:
            flags.append(False)
    return flags


# ---------------------------------------------------------------------------
# The ordered strict alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProseCorroboration:
    """Outcome of aligning a reading's prose, in order, against a page's word layer."""

    #: Prose tokens in the reading (outside its table blocks).
    prose_tokens: int
    #: Alignment steps that are not equal and are NOT explained by ``NOISE_CLASS``.
    #: Any one refuses the page.
    mismatches: int
    #: Steps that differ but are explained by noise class 2 (hyphenation / split / join).
    noise_edits: int
    #: Layer tokens skipped because they belong to the withheld table's bands.
    skipped_layer_tokens: int
    table_blocks: int
    #: ``False`` when the page has no layer words to compare against.
    has_layer: bool

    @property
    def passed(self) -> bool:
        return (
            self.has_layer
            and self.table_blocks > 0
            and self.mismatches == 0
            and self.prose_tokens >= PROSE_CORROBORATION_MIN_TOKENS
        )


def corroborate_prose(reading_text: str, native_words: list) -> ProseCorroboration:
    """Align *reading_text*'s prose against *native_words* (``page.get_text("words")``)."""
    prose, table_lines, blocks = prose_of(reading_text or "")
    reading = _tokens(prose)
    bands = _layer_bands(native_words) if native_words else []
    if not bands:
        return ProseCorroboration(len(reading), 0, 0, 0, blocks, False)

    flags = _table_band_flags(bands, _tokens("\n".join(table_lines)))
    layer: list[str] = []
    layer_is_table: list[bool] = []
    for band, flagged in zip(bands, flags, strict=True):
        layer.extend(band)
        layer_is_table.extend([flagged] * len(band))

    matcher = difflib.SequenceMatcher(None, reading, layer, autojunk=False)
    mismatches = noise_edits = skipped = 0
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            # A reading's PROSE that reproduces a table band is a table row shipped outside
            # the withheld region, un-withheld and unverified: refuse it.
            if any(layer_is_table[j1:j2]):
                mismatches += 1
            continue
        span_layer = [(layer[j], layer_is_table[j]) for j in range(j1, j2)]
        kept = [tok for tok, is_table in span_layer if not is_table]
        skipped += len(span_layer) - len(kept)
        # difflib's vocabulary is relative to the first sequence (the reading): "delete" is
        # reading-only text, "insert" is layer-only text.
        if op == "delete":
            mismatches += 1  # the reading says something the layer does not
        elif op == "insert":
            if kept:
                mismatches += 1  # a layer line the reading does not cover
        elif _alpha_join_equal(reading[i1:i2], kept):
            noise_edits += 1
        else:
            mismatches += 1
    return ProseCorroboration(len(reading), mismatches, noise_edits, skipped, blocks, True)
