"""Page furniture: words a document prints at the same place on several pages.

#988 M2. Report-style PDFs (Coca-Cola's sustainability reports, 2018-2021)
print a website navigation bar across the top of every page: a menu line, a
section sub-menu line, and drawn rules under them. Gemini transcribes the bar
as a Markdown table, and the table gate then judges it as one: an empty table
(``table_content_empty``), a ragged one (``grid_shape``), and the sub-menu
between the bar's two rules is owed to the real table's header
(``header_unattributed``). Those verdicts rejected complete readings of the
table below the bar.

The bar is identified by repetition, not by position on the page: a template
element is printed at identical coordinates on other pages of the same
document, while a page's own content is not. Only words are compared. Drawn
rules are not: a table layout repeated on most pages repeats its rules too,
and dropping them switched the header cut off on such documents (PR #1042
review).

Repetition alone does not separate the bar from a table header that a layout
repeats on every page. Where else the words are printed does: a menu is also
printed on pages with no table (a section's overview), a table's header only
above its table (PR #1042 review).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from socr.tables.header_repair import (
    _MIN_DATA_NUMERIC_CELLS,
    _all_rows_by_y,
    _row_numeric_multiset,
)
from socr.tables.native_verifier import is_numeric_token
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

#: Text is split into pieces on whitespace, Markdown pipes and emphasis.
_PIECE_SPLIT_RE = re.compile(r"[\s|*]+")

#: A non-numeric piece is compared by its letter-and-digit chunks. The model and
#: the text layer break a menu item differently ("Portfolio/Reducing" written,
#: "Portfolio/" and "Reducing" printed), and punctuation (``&``, ``[Home]``)
#: carries no identity of its own. A numeric piece is compared whole, so a table
#: value ``0.32`` is never matched by a page number ``32`` and a ``0``.
_CHUNK_RE = re.compile(r"\w+")

#: Audit-event kind the table gate records when it removes furniture runs from the
#: text it ships; the removed runs are in ``data["runs"]``.
TABLE_FURNITURE_REMOVED_KIND = "table_furniture_removed"

_PositionKey = tuple[str, float, float]


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for piece in _PIECE_SPLIT_RE.split(text.casefold()):
        if not piece:
            continue
        if is_numeric_token(piece):
            out.append(piece)
        else:
            out.extend(_CHUNK_RE.findall(piece))
    return out


def _word_key(word: Sequence) -> _PositionKey:
    return (
        str(word[4]).casefold(),
        round(float(word[0]), _POSITION_DECIMALS),
        round(float(word[1]), _POSITION_DECIMALS),
    )


def _has_data_row(words: list[Sequence]) -> bool:
    """Whether the page prints a row the header cut would take as table data."""
    return any(
        len(_row_numeric_multiset(row)) >= _MIN_DATA_NUMERIC_CELLS
        for row in _all_rows_by_y(words).values()
    )


@dataclass(frozen=True)
class DocumentFurniture:
    """Word positions a document repeats across its pages.

    ``off_table_positions`` are the furniture positions also printed on a page
    with no data row, where the header cut finds no table.
    """

    word_positions: frozenset[_PositionKey] = frozenset()
    off_table_positions: frozenset[_PositionKey] = frozenset()

    def is_furniture_word(self, word: Sequence) -> bool:
        """Whether *word* (a ``get_text("words")`` tuple) sits at a furniture position."""
        return _word_key(word) in self.word_positions

    def is_menu_band(self, words: Sequence[Sequence]) -> bool:
        """Whether *words*, one band of a page, are a menu rather than a table's header.

        Every word must be furniture, and a word with a letter or digit must also
        be printed, at the same place, on a page with no table. Not every word:
        a menu sets its current item apart, which moves the words of that item
        on its own pages (Coca-Cola 2021 p74: 20 of the band's 23 words are also
        printed at that place on the p65 overview, all but the current item's
        "Greenhouse", "&" and "Waste").
        """
        return (
            bool(words)
            and all(self.is_furniture_word(w) for w in words)
            and any(
                _CHUNK_RE.search(str(w[4])) and _word_key(w) in self.off_table_positions
                for w in words
            )
        )

    def page_words(self, words: Iterable[Sequence] | None) -> frozenset[str]:
        """Tokens of the words this page prints at a furniture position."""
        if not words or not self.word_positions:
            return frozenset()
        return frozenset(
            token
            for key in map(_word_key, words)
            if key in self.word_positions
            for token in _tokens(key[0])
        )


def document_furniture(doc) -> DocumentFurniture:
    """Scan every page of *doc* (a ``fitz.Document``) for repeated words."""
    return furniture_from_pages(page.get_text("words") for page in doc)


def furniture_from_pages(pages: Iterable[Iterable[Sequence]]) -> DocumentFurniture:
    """Repeated word positions over each page's words, pages in any order."""
    word_pages: Counter[_PositionKey] = Counter()
    off_table: set[_PositionKey] = set()
    for page_words in pages:
        words = list(page_words)
        keys = {_word_key(w) for w in words}
        word_pages.update(keys)
        if not _has_data_row(words):
            off_table |= keys
    repeated = frozenset(k for k, n in word_pages.items() if n >= _MIN_WORD_REPEAT_PAGES)
    return DocumentFurniture(word_positions=repeated, off_table_positions=repeated & off_table)


def _run_is_furniture(run: list[str], furniture_words: frozenset[str]) -> bool:
    tokens = [token for line in run for token in _tokens(line)]
    return bool(tokens) and all(token in furniture_words for token in tokens)


def _run_has_numeric_cell(run: list[str]) -> bool:
    return any(
        is_numeric_token(cell) for line in run for cell in line.strip().strip("|").split("|")
    )


def strip_furniture_runs(
    markdown: str, furniture_words: frozenset[str], *, keep_numbered: bool = False
) -> tuple[str, list[str]]:
    """*markdown* without the pipe runs that transcribe page furniture, and those runs.

    A run is furniture when every token in it is one the page prints at a
    furniture position. Each such run is replaced by one blank line, so the
    text around it does not join up. When every pipe run is furniture nothing
    is removed: the answer then has no table of its own, and the gate must
    judge what it does have.

    ``keep_numbered`` keeps the furniture runs with a cell that is a number. The
    gate checks the answer without every furniture run, but removes from the
    text it ships only those with no numeric cell (PR #1042 review): a data
    table repeated at the same place on two pages must never leave the output.
    A number inside a text cell does not keep a run: Coca-Cola's sub-menus
    print "2020 Sustainability Goals", and keeping those runs left a header-only
    table that the manifest backstop refused on 5 complete answers.
    """
    if not markdown or not furniture_words:
        return markdown, []
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
        return markdown, []
    if keep_numbered:
        furniture = [(i, j) for i, j in furniture if not _run_has_numeric_cell(lines[i:j])]
        if not furniture:
            return markdown, []
    out: list[str] = []
    prev = 0
    for i, j in furniture:
        out.extend(lines[prev:i])
        out.append("")
        prev = j
    out.extend(lines[prev:])
    text = "\n".join(out) + ("\n" if markdown.endswith("\n") else "")
    return text, ["\n".join(lines[i:j]) for i, j in furniture]
