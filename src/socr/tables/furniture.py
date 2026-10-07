"""Page furniture: what a document prints at the same place on several pages.

#988 M2. Report-style PDFs (Coca-Cola's sustainability reports, 2018-2021)
print a website navigation bar across the top of every page: a menu line, a
section sub-menu line, and drawn rules under them. Gemini transcribes the bar
as a Markdown table, and the table gate then judges it as one: an empty table
(``table_content_empty``), a ragged one (``grid_shape``), and the bar's rules
become the real table's header cut (``header_unattributed``). Those verdicts
rejected complete readings of the table below the bar.

The bar is identified by repetition, not by position on the page: a template
element is printed at identical coordinates on other pages of the same
document, while a page's own content is not. Nothing here edits shipped text;
callers only leave furniture out of what the gate checks.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from socr.tables.locate import _horizontal_rules
from socr.tables.reconcile import _is_table_line

#: Decimal places kept when comparing positions across pages (``round`` ndigits).
#: A layout template places its elements at the same coordinates on every page;
#: whole points absorb float noise in the extraction and nothing else.
_POSITION_DECIMALS = 0

#: A word is furniture when its text is printed at the same position on at
#: least this many pages. Two is the least that shows a repeat. A section
#: sub-menu is repeated only on its own section's pages, so a share of the
#: document would miss it (on 2021 p74, the "Data Appendix" sub-menu).
_MIN_WORD_REPEAT_PAGES = 2

#: A drawn rule is furniture when it is drawn on MORE than this share of the
#: pages (and on at least ``_MIN_WORD_REPEAT_PAGES`` of them). Dropping a rule
#: changes the header cut of every table on the page, so rules need the
#: stronger evidence: a continued table repeats its rules on a few pages, the
#: page template on most. Measured on the four Coca-Cola reports: header-cut
#: HARD verdicts on cached answers fall from 23 to 1.
_RULE_FURNITURE_SHARE = 0.5

#: Words are compared by their letter-and-digit chunks. The model and the text
#: layer break a menu item differently ("Portfolio/Reducing" written, "Portfolio/"
#: and "Reducing" printed), and punctuation (``&``, ``|``, ``[Home]``) carries no
#: identity of its own.
_CHUNK_RE = re.compile(r"\w+")

_PositionKey = tuple[str, float, float]
_RuleKey = tuple[float, float, float]


def _chunks(text: str) -> list[str]:
    return _CHUNK_RE.findall(text.casefold())


def _word_key(word: Sequence) -> _PositionKey:
    return (
        str(word[4]).casefold(),
        round(float(word[0]), _POSITION_DECIMALS),
        round(float(word[1]), _POSITION_DECIMALS),
    )


def _rule_key(rule: tuple[float, float, float]) -> _RuleKey:
    y, x0, x1 = (round(float(v), _POSITION_DECIMALS) for v in rule)
    return (y, x0, x1)


@dataclass(frozen=True)
class DocumentFurniture:
    """Word positions and drawn rules a document repeats across its pages."""

    word_positions: frozenset[_PositionKey] = frozenset()
    rules: frozenset[_RuleKey] = frozenset()

    def page_words(self, words: Iterable[Sequence] | None) -> frozenset[str]:
        """Chunks of the words this page prints at a furniture position."""
        if not words or not self.word_positions:
            return frozenset()
        return frozenset(
            chunk
            for key in map(_word_key, words)
            if key in self.word_positions
            for chunk in _chunks(key[0])
        )

    def keep_rules(
        self, rules: list[tuple[float, float, float]] | None
    ) -> list[tuple[float, float, float]] | None:
        """*rules* without the document's furniture rules (``None`` passes through)."""
        if rules is None or not self.rules:
            return rules
        return [r for r in rules if _rule_key(r) not in self.rules]


def document_furniture(doc) -> DocumentFurniture:
    """Scan every page of *doc* (a ``fitz.Document``) for repeated words and rules."""
    return furniture_from_pages((page.get_text("words"), _horizontal_rules(page)) for page in doc)


def furniture_from_pages(
    pages: Iterable[tuple[Iterable[Sequence], Iterable[tuple[float, float, float]]]],
) -> DocumentFurniture:
    """Repeated words and rules over ``(words, rules)`` per page, in any order."""
    word_pages: Counter[_PositionKey] = Counter()
    rule_pages: Counter[_RuleKey] = Counter()
    n_pages = 0
    for words, rules in pages:
        n_pages += 1
        word_pages.update({_word_key(w) for w in words})
        rule_pages.update({_rule_key(r) for r in rules})
    return DocumentFurniture(
        word_positions=frozenset(k for k, n in word_pages.items() if n >= _MIN_WORD_REPEAT_PAGES),
        rules=frozenset(
            k
            for k, n in rule_pages.items()
            if n >= _MIN_WORD_REPEAT_PAGES and n > n_pages * _RULE_FURNITURE_SHARE
        ),
    )


def _run_is_furniture(run: list[str], furniture_words: frozenset[str]) -> bool:
    chunks = [chunk for line in run for chunk in _chunks(line)]
    return bool(chunks) and all(chunk in furniture_words for chunk in chunks)


def strip_furniture_runs(markdown: str, furniture_words: frozenset[str]) -> str:
    """*markdown* without the pipe runs that transcribe page furniture.

    A run is furniture when every word in it is one the page prints at a
    furniture position. Each such run is replaced by one blank line, so the
    text around it does not join up. When every pipe run is furniture the text
    is returned unchanged: the answer then has no table of its own, and the
    gate must judge what it does have.
    """
    if not markdown or not furniture_words:
        return markdown
    lines = markdown.splitlines()
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if _is_table_line(lines[i]):
            j = i
            while j < len(lines) and _is_table_line(lines[j]):
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    furniture = [(i, j) for i, j in runs if _run_is_furniture(lines[i:j], furniture_words)]
    if not furniture or len(furniture) == len(runs):
        return markdown
    out: list[str] = []
    prev = 0
    for i, j in furniture:
        out.extend(lines[prev:i])
        out.append("")
        prev = j
    out.extend(lines[prev:])
    return "\n".join(out)
