"""#1043: corroborate a model reading's PROSE against the page's own text layer.

A page whose every model reading was rejected only because of a table used to ship as
the whole-page fail-closed floor, so its correct prose was lost as text. This module is
the mechanical test that lets that prose ship: the reading's words OUTSIDE its markdown
table blocks must reproduce the page's own word layer (the invisible OCR layer of a scan,
the native text of a born-digital page).

It is not the vocabulary-overlap guard #652 deleted in round 10. That guard asked whether
a model's words overlapped a withheld table's vocabulary, which a table's own labels
satisfy. This one compares by ORDER (bigrams), requires the layer to be COVERED by the
reading (so a reading that is another page, or a fragment of this one, fails), and
refuses any numeral the layer does not print.

Measured (``docs/log/2026-10-08_table-only-floor-prose.md``; counts only, the corpus is
copyrighted). The negative control is a reading of ANOTHER page of the same document
scored against this page's layer: the hardest wrong reading, since journal, vocabulary
and running header are shared.

* ``PROSE_CORROBORATION_MIN_TOKENS``: 4 of 3,363 scored negative controls (3,444 built) score a perfect
  1.0, every one with 16 or fewer prose tokens (identical short running captions). A
  prose region that short cannot discriminate, so the floor is that count plus one.
* ``PROSE_CORROBORATION_MIN``: among the negatives at or above that token floor the
  maximum score is 0.892; the cutoff is the next tenth above it.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

#: Fewest prose tokens a reading may carry and still be corroborated. See module docstring.
PROSE_CORROBORATION_MIN_TOKENS = 17

#: Lowest corroboration score a reading may carry. See module docstring.
PROSE_CORROBORATION_MIN = 0.90

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MARKER_RE = re.compile(r"\[page \d+[^\]]*\]")

#: The rejection reasons socr's table gates author, by prefix. A reading refused with one
#: of these was refused for its table, by construction.
TABLE_GATE_PREFIXES = ("table_structure_failed", "source_evidence_table", "native_table_verifier")

#: A free-text judge clause that names a table structure. Deliberately the weak link of
#: the table-only test (a model chose the words): the corroboration score, not this
#: pattern, is what stops a wrong page from shipping.
_TABLE_CLAUSE_RE = re.compile(
    r"\btable|\bcolumns?\b|\brows?\b|\bgrid\b|\bheaders?\b", re.IGNORECASE
)


def rejection_is_table_only(reason: str) -> bool:
    """Whether a rejection reason names only table defects.

    A gate reason (``table_structure_failed: ...``) is table-only by construction. Free
    judge text is split into clauses and EVERY clause must name a table structure: one
    clause about a missing paragraph, a figure or an axis makes the rejection mixed.
    An empty reason is unknown, never table-only.
    """
    reason = (reason or "").strip()
    if not reason:
        return False
    if reason.startswith(TABLE_GATE_PREFIXES):
        return True
    clauses = [c.strip() for c in re.split(r"[;\n]", reason) if c.strip()]
    return bool(clauses) and all(_TABLE_CLAUSE_RE.search(c) for c in clauses)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


def _bigrams(tokens: list[str]) -> Counter:
    return Counter(zip(tokens, tokens[1:], strict=False))


def _has_digit(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def prose_of(reading_text: str) -> tuple[str, int]:
    """*reading_text* with its markdown table blocks, image refs and socr markers removed.

    Returns ``(prose, table_block_count)``.
    """
    from socr.tables.reconcile import find_table_blocks

    blocks = find_table_blocks(reading_text)
    drop: set[int] = set()
    for block in blocks:
        drop.update(range(block.start, block.end + 1))
    kept = [line for i, line in enumerate(reading_text.splitlines()) if i not in drop]
    prose = "\n".join(kept)
    for pattern in (_IMAGE_RE, _COMMENT_RE, _MARKER_RE):
        prose = pattern.sub(" ", prose)
    return prose, len(blocks)


@dataclass(frozen=True)
class ProseCorroboration:
    """Outcome of comparing a reading's prose to a page's word layer."""

    #: ``min(precision, recall)``; ``None`` when either side has nothing to compare.
    score: float | None
    #: Fraction of the reading's prose bigrams the layer also prints, in order.
    precision: float | None
    #: Fraction of the layer's own prose-band words the reading (prose plus table) covers.
    recall: float | None
    #: Numeral-bearing prose tokens the layer does not print. Any one refuses the page.
    unmatched_numerals: int
    prose_tokens: int
    table_blocks: int

    @property
    def passed(self) -> bool:
        return (
            self.table_blocks > 0
            and self.score is not None
            and self.score >= PROSE_CORROBORATION_MIN
            and self.unmatched_numerals == 0
            and self.prose_tokens >= PROSE_CORROBORATION_MIN_TOKENS
        )


def corroborate_prose(reading_text: str, native_words: list) -> ProseCorroboration:
    """Score *reading_text*'s prose against *native_words* (``page.get_text("words")``)."""
    from socr.tables.row_corroboration import partition_prose_bands

    prose, blocks = prose_of(reading_text or "")
    prose_tokens = _tokens(prose)
    layer_tokens = _tokens(" ".join(str(w[4]) for w in native_words or []))
    layer = Counter(layer_tokens)

    prose_counts = Counter(prose_tokens)
    table_counts = Counter(_tokens("\n".join(_table_lines(reading_text or ""))))
    prose_bigrams = _bigrams(prose_tokens)
    n_bigrams = sum(prose_bigrams.values())
    precision = (
        sum((prose_bigrams & _bigrams(layer_tokens)).values()) / n_bigrams if n_bigrams else None
    )

    layer_prose: Counter = Counter()
    if native_words:
        for is_prose, band in partition_prose_bands(list(native_words)):
            if is_prose:
                layer_prose.update(_tokens(" ".join(str(w[4]) for w in band)))
    n_layer_prose = sum(layer_prose.values())
    recall = (
        sum((layer_prose & (prose_counts + table_counts)).values()) / n_layer_prose
        if n_layer_prose
        else None
    )

    unmatched = sum(
        ((Counter({t: c for t, c in prose_counts.items() if _has_digit(t)})) - layer).values()
    )
    score = min(precision, recall) if precision is not None and recall is not None else None
    return ProseCorroboration(
        score=score,
        precision=precision,
        recall=recall,
        unmatched_numerals=unmatched,
        prose_tokens=len(prose_tokens),
        table_blocks=blocks,
    )


def _table_lines(reading_text: str) -> list[str]:
    from socr.tables.reconcile import find_table_blocks

    lines = reading_text.splitlines()
    out: list[str] = []
    for block in find_table_blocks(reading_text):
        out.extend(lines[block.start : block.end + 1])
    return out
