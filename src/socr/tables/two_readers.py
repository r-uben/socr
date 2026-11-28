"""#1409 step 3: do two independent readings of one page image agree on everything?

Owner ruling (2026-11-26, approved for build and test on a branch): a scanned page with no
trustworthy text layer may ship as text, unmarked, when two independent readings of the page
image agree on everything. This module is that test. It is pure: no PDF, no model, no I/O.
The caller (``pipeline/refinalize_two_readers.py``) decides which pages it may be applied to
and what ships.

Two readings can agree and both be wrong. Nothing here can see that; the evidence the rule
rests on is a hand check of agreed readings (24 table pages, 3,112 cells, 0 errors).

The test: ALL of these, each named in the verdict when it fails
----------------------------------------------------------------
1. **Usable.** Each reading is non-empty, was not cut off (``finish_reason`` is not
   ``length``; no code fence left open) and has no unterminated table or furniture tag.
2. **Furniture.** A reader may tag running heads, feet and page numbers
   (``<page_header>``, ``<page_footer>``, ``<page_number>``); another writes them as plain
   lines. Every tagged block of EITHER reading is taken out of BOTH: where it is tagged it
   is cut with its tags; in the other reading it is cut where whole lines at the top or the
   bottom of the text (outside tables), taken together, hold exactly the block's words in
   order. So ``11/13/90 -2-`` tagged as one block is the two lines ``11/13/90`` and ``-2-``
   of the other reading. No line is furniture by its position, length or shape: only a
   reader's own tag makes it so, and only an exact word match removes it from the other.
   A tagged block the other reading does not hold at an edge is a failure
   (:data:`FURNITURE_UNMATCHED`): one reading saw page furniture the other did not write
   there, and the text that ships must not lack what a reader saw.
3. **Numbers.** With furniture out, the two bodies are cut into rows the same way (a
   ``<tr>`` of an HTML table, a markdown pipe row, a line) and compared with
   ``second_reader``'s rules in both directions: every number-bearing row of each is found
   in the other (inside one row, or as whole rows end to end, or, for running text, across
   lines that are not table rows; a table row's ends are never crossed, on either side),
   and every value occurs the same number of times. No chart fallback and no chart-label
   excuse: a row only one reader wrote is a difference.
4. **Words.** The word tokens of the two bodies are the same multiset, and outside tables
   they are the same sequence. Normalisation, on both alike (:func:`word_tokens`): entities
   unescaped; code fences and image placeholders dropped; ``\\frac{a}{b}`` is ``a/b``, a
   ``^{x}`` or ``<sup>``/``<sub>`` or superscript digit is a token of its own; ``$`` dropped
   and ``\\%`` is ``%``; other tags stripped; NFKC (and the fraction slash it writes for a
   glyph such as ``\u00bd`` is ``/``, so the glyph is the ``1/2`` of ``\\frac``); curly quotes
   straightened; dashes are
   spaces; a hyphen at a line break before a lowercase letter joins its word; markdown
   ``* _ # | > [ ]`` and dot leaders stripped; lowercase; a token is letters and digits with
   ``. , ' / -`` inside it.
5. **Nothing a reader is known to get wrong.** Neither reading has ``~`` directly before a
   digit (a known reader defect where the page prints a minus), nor an image placeholder
   (a picture stands where text or numbers may be). The readings hold the same number of
   tables (:data:`TABLE_WITHOUT_COUNTERPART` otherwise): a reader that tabulates a chart
   has a table the other has not.
6. **Tables cell for cell.** The n-th table of one reading has the same grid as the n-th of
   the other: the same rows, each with the same cells in the same columns, a cell being
   its words. The grid is the one socr ships (``core.html_tables``: ``colspan`` and
   ``rowspan`` expanded), with trailing empty cells and empty rows left out.
   Measured on the 1990s pilot (687 pages): of 38 table pages passing rules 1 to 5, 15
   differ here, and every one looked at was a real difference of structure (two rows
   merged into one cell, a header one column off for want of a ``colspan``). Rules 3 and 4
   cannot see those: row keys laid end to end and word multisets are equal.

What this cannot see
--------------------
* an error both readers make;
* a table column both readers shifted the same way;
* punctuation, capitalisation, emphasis and line breaks (the word normalisation folds them),
  inside table cells too.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from socr.core.html_tables import html_table_grid
from socr.tables.second_reader import (
    _FENCE_RE,
    _FURNITURE_RE,
    _HTML_TAG_RE,
    _TABLE_RE,
    TILDE,
    _counts,
    _found,
    _unclosed,
    _unfenced,
    reader_rows,
    row_numbers,
)

#: ``finish_reason`` of an output the server cut at its token budget.
FINISH_REASON_CUT_OFF = "length"

SIDE_A = "a"
SIDE_B = "b"

UNUSABLE_EMPTY = "reading_empty"
UNUSABLE_CUT_OFF = "reading_cut_off"
UNUSABLE_UNCLOSED = "reading_has_an_unclosed_block"
FURNITURE_UNMATCHED = "furniture_one_reader_tagged_is_not_at_the_edge_of_the_other"
NUMBERS_DIFFER = "numbers_differ"
WORDS_DIFFER = "words_differ"
WORD_ORDER_DIFFERS = "word_order_outside_tables_differs"
TILDE_BEFORE_DIGIT = "tilde_before_a_digit"
IMAGE_PLACEHOLDER = "image_placeholder"
TABLE_WITHOUT_COUNTERPART = "table_without_counterpart"
TABLE_GRIDS_DIFFER = "table_grids_differ"

RULE = "two_reader_agreement"
#: ``engine`` of a page shipped by this rule. Not a reading engine of socr's: no layer check
#: and no judge clause of a run applies to it.
ENGINE = "two-readers"
#: Recorded with every agreement, so a reader of the record knows what "agree" meant.
RULE_VERSION = 1


@dataclass(frozen=True)
class Failure:
    """One condition that did not hold."""

    #: one of the module's reason constants
    reason: str
    #: which reading it is about (``a`` / ``b``), or ``""`` for a difference between them
    side: str = ""
    #: a short, quotable detail
    detail: str = ""


@dataclass(frozen=True)
class TwoReaderVerdict:
    #: every condition held
    agree: bool
    failures: tuple[Failure, ...] = ()
    #: number-bearing rows and word tokens compared (of reading A, furniture out)
    number_rows: int = 0
    words: int = 0
    tables: int = 0
    #: furniture blocks taken out of both readings
    furniture_blocks: int = 0
    #: rule 6 held (kept apart so the cost of that rule can be measured)
    same_grids: bool = False

    def reasons(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f.reason for f in self.failures))

    def to_dict(self) -> dict:
        return {
            "rule": RULE,
            "rule_version": RULE_VERSION,
            "agree": self.agree,
            "failures": [
                {"reason": f.reason, "side": f.side, "detail": f.detail} for f in self.failures
            ],
            "number_rows": self.number_rows,
            "words": self.words,
            "tables": self.tables,
            "furniture_blocks": self.furniture_blocks,
            "same_grids": self.same_grids,
        }


# --- words ------------------------------------------------------------------------------

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\b[^>]*>", re.I)
_FRACTION_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_CARET_RE = re.compile(r"\^\{([^{}]*)\}|\^(\w)")
_SUP_SUB_RE = re.compile(r"</?su[pb]>", re.I)
_SUPERSCRIPTS = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
_SUBSCRIPTS = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
_SUPERSCRIPT_RUN_RE = re.compile(f"[{_SUPERSCRIPTS}{_SUBSCRIPTS}]+")
_CURLY = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
_SLASHES = str.maketrans({"\u2044": "/", "\u2215": "/"})
_DASHES_RE = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]|--+")
#: a hyphen at a line break before a lowercase letter: one word broken over two lines
_LINE_BREAK_HYPHEN_RE = re.compile(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=[a-z])")
_DOT_LEADER_RE = re.compile(r"(?:\.[ \t]*){3,}")
_MARKDOWN_RE = re.compile(r"[*_#|>\[\]`]")
_WORD_RE = re.compile(r"[^\W_]+(?:[.,'/-][^\W_]+)*")
_TILDE_DIGIT_RE = re.compile(rf"{re.escape(TILDE)}(?=\.?\d)")


def word_tokens(text: str) -> tuple[str, ...]:
    """The normalised word tokens of *text*, in order. The same function for both readings."""
    text = "\n".join(line for line in text.split("\n") if not _FENCE_RE.match(line))
    text = _IMAGE_RE.sub(" ", text)
    text = _SUP_SUB_RE.sub(" ", text)
    # Strip allowlisted HTML/reader tags BEFORE unescaping entities so inequalities
    # (``p < 0.05 and z > 1.96``, ``&lt;...&gt;``) and literal tokens like ``<1>``/``<PR>``
    # survive intact.
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _FRACTION_RE.sub(lambda m: f" {m[1].strip()}/{m[2].strip()} ", text)
    text = _CARET_RE.sub(lambda m: f" {m[1] if m[1] is not None else m[2]} ", text)
    text = text.replace("\\%", "%").replace("$", "")
    text = _SUPERSCRIPT_RUN_RE.sub(lambda m: f" {m[0]} ", text)
    # NFKC writes a vulgar-fraction glyph with U+2044: the same slash as ``\frac`` gets
    text = unicodedata.normalize("NFKC", text).translate(_CURLY).translate(_SLASHES)
    text = _LINE_BREAK_HYPHEN_RE.sub("", text)
    text = _DASHES_RE.sub(" ", text)
    text = _DOT_LEADER_RE.sub(" ", text)
    text = _MARKDOWN_RE.sub(" ", text)
    return tuple(_WORD_RE.findall(text.lower()))


# --- furniture --------------------------------------------------------------------------

_PIPE_ROW_RE = re.compile(r"^\s*\|")


def _cut_tagged(text: str) -> tuple[str, list[tuple[str, ...]]]:
    """*text* without its tagged furniture, and each tagged block's words."""
    blocks: list[tuple[str, ...]] = []

    def take(match: re.Match) -> str:
        words = word_tokens(match[2])
        if words:
            blocks.append(words)
        return "\n"

    return _FURNITURE_RE.sub(take, text), blocks


def _edge_lines(text: str) -> tuple[list[str], list[int], list[int]]:
    """The lines of *text*, and the indices of the word-bearing lines that lie before its
    first table and after its last one (top block, bottom block)."""
    lines = text.split("\n")
    spans = [(m.start(), m.end()) for m in _TABLE_RE.finditer(text)]
    first = spans[0][0] if spans else len(text)
    last = spans[-1][1] if spans else 0
    top: list[int] = []
    bottom: list[int] = []
    offset = 0
    for index, line in enumerate(lines):
        start, end = offset, offset + len(line)
        offset = end + 1
        if _PIPE_ROW_RE.match(line) or not word_tokens(line):
            continue
        if end <= first:
            top.append(index)
        if start >= last:
            bottom.append(index)
    return lines, top, bottom


def _cut_at_edges(text: str, blocks: list[tuple[str, ...]]) -> tuple[str, list[tuple[str, ...]]]:
    """*text* without the lines at its top and bottom that hold a block's words exactly, and
    the blocks that were not found there. Each block is used once; cutting is repeated, so a
    head written as two blocks in one reading and two lines in the other is found."""
    left = list(blocks)
    progress = True
    while progress and left:
        progress = False
        lines, top, bottom = _edge_lines(text)
        for block in left:
            cut = _run_holding(lines, top, block, from_top=True)
            if cut is None:
                cut = _run_holding(lines, bottom, block, from_top=False)
            if cut is None:
                continue
            text = "\n".join(line for index, line in enumerate(lines) if index not in cut)
            left.remove(block)
            progress = True
            break
    return text, left


def _run_holding(
    lines: list[str], edge: list[int], block: tuple[str, ...], *, from_top: bool
) -> set[int] | None:
    """The outermost word-bearing lines of *edge* that together are exactly *block*."""
    order = edge if from_top else edge[::-1]
    taken: list[int] = []
    words: tuple[str, ...] = ()
    for index in order:
        taken.append(index)
        line = word_tokens(lines[index])
        words = words + line if from_top else line + words
        if words == block:
            return set(taken)
        if len(words) >= len(block):
            return None
    return None


# --- tables -----------------------------------------------------------------------------


def _tables(text: str) -> list[str]:
    return [m[1] for m in _TABLE_RE.finditer(text)]


def _outside_tables(text: str) -> str:
    out = _TABLE_RE.sub("\n", text)
    return "\n".join(line for line in out.split("\n") if not _PIPE_ROW_RE.match(line))


def _grid(table: str) -> tuple[tuple[tuple[tuple[str, ...], tuple[str, ...]], ...], ...]:
    """The table as socr ships it (spans expanded), each cell both its normalised words
    AND its canonical signed/bracketed numbers (``row_numbers``), so cross-row sign or
    bracket swaps (``-1.0`` vs ``1.0``, ``(0.4)`` vs ``0.4``) can never pass as equal.
    Trailing empty cells and empty rows are left out (readers pad a ragged table differently)."""
    rows = []
    for row in html_table_grid(table):
        cells = [(word_tokens(cell), row_numbers(cell)) for cell in row]
        while cells and not cells[-1][0] and not cells[-1][1]:
            cells.pop()
        if cells:
            rows.append(tuple(cells))
    return tuple(rows)


# --- the test ---------------------------------------------------------------------------


def _usable(text: str, finish_reason: str | None, side: str) -> tuple[str, Failure | None]:
    """``(the reading without its code fence, None)``, or ``("", why it cannot be used)``."""
    if not (text or "").strip():
        return "", Failure(UNUSABLE_EMPTY, side)
    body, open_fence = _unfenced(text)
    if finish_reason == FINISH_REASON_CUT_OFF or open_fence:
        return "", Failure(UNUSABLE_CUT_OFF, side)
    if _unclosed(body):
        return "", Failure(UNUSABLE_UNCLOSED, side)
    return body, None


def _quote(words: tuple[str, ...] | list[str]) -> str:
    return " ".join(words)


def _number_failures(a: str, b: str) -> list[Failure]:
    rows_a, _ = reader_rows(a)
    rows_b, _ = reader_rows(b)
    out = [
        Failure(NUMBERS_DIFFER, side, f"row with the numbers {_quote(row.key)} is not in the other")
        for side, rows, other in ((SIDE_A, rows_a, rows_b), (SIDE_B, rows_b, rows_a))
        for row in rows
        if not _found(row, other)
    ]
    count_a, count_b = _counts(rows_a), _counts(rows_b)
    out.extend(
        Failure(
            NUMBERS_DIFFER,
            detail=f"{value} occurs {count_a[value]} time(s) in a and {count_b[value]} in b",
        )
        for value in sorted(set(count_a) | set(count_b))
        if count_a[value] != count_b[value]
    )
    return out


def _word_failures(a: str, b: str) -> list[Failure]:
    words_a, words_b = Counter(word_tokens(a)), Counter(word_tokens(b))
    out: list[Failure] = []
    if words_a != words_b:
        only_a, only_b = words_a - words_b, words_b - words_a
        out.append(
            Failure(
                WORDS_DIFFER,
                detail=f"only in a: {_quote(sorted(only_a.elements()))} / "
                f"only in b: {_quote(sorted(only_b.elements()))}",
            )
        )
    order_a, order_b = word_tokens(_outside_tables(a)), word_tokens(_outside_tables(b))
    if order_a != order_b:
        at = next(
            (i for i, (x, y) in enumerate(zip(order_a, order_b, strict=False)) if x != y),
            min(len(order_a), len(order_b)),
        )
        out.append(
            Failure(
                WORD_ORDER_DIFFERS,
                detail=f"at word {at + 1}: a has {_quote(order_a[at : at + 1])!r}, "
                f"b has {_quote(order_b[at : at + 1])!r}",
            )
        )
    return out


def _table_failures(a: str, b: str) -> list[Failure]:
    tables_a, tables_b = _tables(a), _tables(b)
    if len(tables_a) != len(tables_b):
        return [
            Failure(
                TABLE_WITHOUT_COUNTERPART,
                detail=f"a has {len(tables_a)} table(s), b has {len(tables_b)}",
            )
        ]
    out: list[Failure] = []
    for n, (x, y) in enumerate(zip(tables_a, tables_b, strict=True), start=1):
        grid_x, grid_y = _grid(x), _grid(y)
        if grid_x != grid_y:
            at = next(
                (i for i, (r, q) in enumerate(zip(grid_x, grid_y, strict=False)) if r != q),
                min(len(grid_x), len(grid_y)),
            )
            out.append(Failure(TABLE_GRIDS_DIFFER, detail=f"table {n}, row {at + 1}"))
    return out


def unfenced(reading: str) -> str:
    """A reader's raw output without the code fence some readers wrap it in."""
    return _unfenced(reading)[0]


def bodies(a: str, b: str) -> tuple[str, str, int, list[Failure]]:
    """The two usable, unfenced readings with their furniture out (rule 2), how many blocks
    were cut, and a failure for each tagged block the other reading does not hold."""
    body_a, blocks_a = _cut_tagged(a)
    body_b, blocks_b = _cut_tagged(b)
    body_b, missing_from_b = _cut_at_edges(body_b, blocks_a)
    body_a, missing_from_a = _cut_at_edges(body_a, blocks_b)
    failures = [
        Failure(FURNITURE_UNMATCHED, side, _quote(block))
        for side, blocks in ((SIDE_B, missing_from_b), (SIDE_A, missing_from_a))
        for block in blocks
    ]
    return body_a, body_b, len(blocks_a) + len(blocks_b), failures


def compare_two_readings(
    a: str,
    b: str,
    *,
    finish_reason_a: str | None = None,
    finish_reason_b: str | None = None,
) -> TwoReaderVerdict:
    """Do the raw outputs *a* and *b* of two readers of one page image agree on everything?

    ``finish_reason_*``: what each reader's server reported; ``None`` is "not recorded",
    which is not "cut off" (the structural checks still apply).
    """
    text_a, bad_a = _usable(a, finish_reason_a, SIDE_A)
    text_b, bad_b = _usable(b, finish_reason_b, SIDE_B)
    unusable = [f for f in (bad_a, bad_b) if f is not None]
    if unusable:
        return TwoReaderVerdict(False, tuple(unusable))
    failures: list[Failure] = []
    for side, text in ((SIDE_A, text_a), (SIDE_B, text_b)):
        if _TILDE_DIGIT_RE.search(text):
            failures.append(Failure(TILDE_BEFORE_DIGIT, side))
        if _IMAGE_RE.search(text):
            failures.append(Failure(IMAGE_PLACEHOLDER, side))
    body_a, body_b, cut, unmatched = bodies(text_a, text_b)
    failures.extend(unmatched)
    failures.extend(_number_failures(body_a, body_b))
    failures.extend(_word_failures(body_a, body_b))
    table_failures = _table_failures(body_a, body_b)
    failures.extend(table_failures)
    tables_a = _tables(body_a)
    return TwoReaderVerdict(
        agree=not failures,
        failures=tuple(failures),
        number_rows=len(reader_rows(body_a)[0]),
        words=len(word_tokens(body_a)),
        tables=len(tables_a),
        furniture_blocks=cut,
        same_grids=not table_failures,
    )
