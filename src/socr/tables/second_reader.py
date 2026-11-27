"""#1409: do the numbers a page ships agree with an independent second reading of its image?

A hand check of 150 unmarked pages found wrong numbers no existing check caught: minus
signs the text layer set on lines of their own (#1411), stacked fractions shipped as lone
digits (#1412), a fraction shipped twice, and hidden PDF text shipped as if printed. A
second reader that looks only at the page IMAGE read all of them correctly. This module
compares the two texts. It is pure: no PDF, no model, no I/O. The caller supplies both.

What the verdict licenses
-------------------------
``DISAGREE`` and ``UNKNOWN`` may ADD a warning mark. ``AGREE`` licenses nothing: this
check never lifts a mark and never changes shipped text, and nothing the second reader
wrote is ever shipped.

Why not the existing tokeniser
------------------------------
``tables/reading_agreement.numeric_tokens`` (the prior-OCR-layer agreement rule, the
closest analogue) splits on whitespace and keeps a token only when the whole of it is a
number. Measured on this module's cases, it drops a range (``1.7–1.8``, ``2022–24``)
and a tilde-signed number (``~.5``) whole, does not fold U+2010 (the hyphen #1411 ships),
and tokenises its two sides differently on purpose (a layer against a model reading). Here
both sides are readings, so ONE scanner is applied to both: whatever it makes of a string,
it makes the same of that string on the other side, and only a difference in the text can
make a difference in the tokens. The canonical form of a number is the repo's
(``row_binding_check._canon_number``).

Numbers
-------
A number is a maximal digit literal: digits, thousands commas in groups of three, one
decimal point followed by digits; or a point followed by digits. It is found wherever it
stands, also run into a word (``Q1``, ``2012H1``, ``by1.2percent``). Normalised, on both
sides alike, and nothing else is:

* a point directly after a letter, a closing bracket, a percent or a closing quote is a
  full stop, so ``baseline.13`` holds the callout ``13``, not ``.13``;
* a sign: a hyphen or dash (:data:`MINUS_CHARS`) directly before the number, itself not
  directly after a word character, a point, a comma, a closing bracket or a percent (so the
  dash of ``5-8``, ``1.7–1.8`` and ``Apr-09`` is a separator, on both sides). A dash set
  apart from its number by a space is NOT joined: the faults this check exists for are
  signs that drifted away from their numbers;
* ``~`` directly before a number stays a tilde (:data:`TILDE`): the second reader
  sometimes writes it where the page prints a minus. That is a reader defect and must
  read as a disagreement, never as a minus;
* thousands commas are dropped; ``.5`` and ``0.5`` are one value, as they already are for
  ``_canon_number`` (``-.7`` against ``-7`` stays a difference). No float cast: ``2.10``
  is not ``2.1``;
* a fraction is ONE token ``a/b`` whether written ``\\frac{a}{b}``,
  ``<sup>a</sup>/<sub>b</sub>``, ``<sup>a</sup>/b``, a vulgar-fraction glyph or ``a/b``.
  A stacked fraction shipped as lone digits (``1 1 2`` for one and a half) is three
  integers and differs;
* a number directly inside its own brackets, ``(0.4)``, keeps them (a bracket can be a
  negative sign or a standard error);
* a superscript digit is a separate number (a footnote callout), never a further digit of
  the number it follows.

Rows, and the matching rule
---------------------------
A ROW is one unit of the text as its writer segmented it: a non-blank line of shipped
markdown (a pipe-table row is a line); on the reader side a ``<tr>`` of an HTML table, or
a non-blank line outside tables. A row's KEY is its numbers in order; rows without numbers
take no part. Reader text inside ``<page_header>`` / ``<page_footer>`` is FURNITURE.

The two readers segment differently: shipped text is often one flat line carrying several
table rows, a reader paragraph may be several shipped lines. So a row X is FOUND on the
other side when either

(a) its key is a contiguous run inside the key of ONE row of the other side, or
(b) its key equals the keys of whole rows of the other side laid end to end. The rows need
    not be neighbours, nor in order: a paragraph the reader gives as one line is shipped as
    two lines with another block between them wherever it runs across a column break
    (measured: the Monetary Policy Report's forecast-uncertainty box), and a row may be
    used twice (the count rule below is what bounds repetition); or
(c) its key is a contiguous run inside a STRETCH of the other side: consecutive rows, in
    their order, none of which its writer marked as a table row (a reader ``<tr>``, a
    shipped pipe row). Running text is wrapped differently by the two readers (shipped
    footnotes run on across each other's ends, measured on the 2025 Monetary Policy
    Report), so a line may start in one paragraph of the other side and end in the next.
    A table row's ends are never crossed: a ``<tr>`` must be found by (a) or (b), and no
    row is found by running over the end of a ``<tr>``.

Both directions are checked: every shipped row must be found among the reader's body
rows, or by the same rules among the rows of one furniture block (socr ships some running
heads and drops others, so reader furniture may account for a shipped line but is never
REQUIRED of the shipped text: no threshold decides which shipped line "is" furniture).
Every reader body row must be found among the shipped rows.

Then the COUNTS: for each value, the shipped text may print it no fewer times than the
reader's body and no more than body plus furniture. This is what sees a row shipped twice
or dropped beside an identical one, which the row rule (no consumption) cannot.

Charts
------
The second reader INVENTS data tables from charts, with values that change run to run.
Estimated chart values are forbidden in the corpus, so such a table must neither ship nor,
alone, mark a page. The intended guard is to mask the chart boxes in the image before the
reader sees it. For readings made on unmasked images this module applies a fallback:
when the shipped page embeds a picture (``chart_picture``; detected from the shipped
markdown when not given), a reader HTML table with a row that is not found is taken to be
a chart's: all its rows, and the line directly above it (the caption the reader writes
over a table it derived), are recorded as excluded instead of as differences, and their
numbers are not required by the count rule. The fallback gives up, on such pages only: a
real table row the shipped text dropped whole, a value the shipped text prints fewer
times than the page within such a table, and a wrong number in the line above it.

The shipped side of a chart is the other half. socr ships a chart as a picture plus the
labels printed on it (axis ticks, dates), which a reader that tabulates the chart, or sees
it masked, does not write as rows. ``chart_label_text`` is the text the page's own text
layer holds INSIDE the chart boxes (the caller cuts it from the PDF with the boxes the
page sidecar records). A shipped row that is not found is recorded as excluded when every
number of it is among those labels, counted; and the count rule lets the shipped text
print a value that many times more. These are numbers printed on the page, never
estimates. Without ``chart_label_text`` (a scan has no layer) every shipped row must be
found. What this gives up, on pages with chart labels only: a wrong shipped row all of
whose numbers also occur as chart labels.

What this cannot see
--------------------
* a value moved between rows whose keys both stay findable: two one-number rows that
  swapped values, or a value moved across a row boundary when the neighbouring rows are
  one row on the other side (labels are not compared: glued words, lost ligatures and
  hyphenation make them noisy, and a row is not bound to its label here);
* a wrong number both readers agree on, and anything that is not a number (words, glued
  words, labels, order of paragraphs);
* a hidden-text line whose numbers all occur, in order, in a reader row or furniture block;
* a detached sign that both sides lost;
* decimal commas and space-grouped thousands (not used in the Fed corpus this was built on).

``UNKNOWN`` means the reader's output could not be used: empty, cut off (the caller's
``reader_truncated``, an unclosed code fence, table or furniture tag), or no number on
either side.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from socr.core.marks import SOCR_MARKER_RE
from socr.tables.row_binding_check import _canon_number

#: Characters that print as a minus sign in front of a number: ASCII hyphen, U+2010 hyphen
#: (what #1411's text layers hold), U+2011 non-breaking hyphen, U+2012 figure dash, U+2013
#: en dash, U+2014 em dash, U+2212 minus, U+FE63 small hyphen-minus, U+FF0D fullwidth.
MINUS_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2212\ufe63\uff0d"
#: Kept as written in front of a number; never a minus (a known second-reader defect).
TILDE = "~"

SIDE_SHIPPED = "shipped"
SIDE_READER = "reader"

DIFF_ROW_NOT_FOUND = "row_not_found"
DIFF_COUNT = "count"

UNKNOWN_EMPTY = "reader_output_empty"
UNKNOWN_TRUNCATED = "reader_output_truncated"
UNKNOWN_UNCLOSED = "reader_output_has_an_unclosed_block"
UNKNOWN_NO_NUMBERS = "no_numbers_on_either_side"

NORMALISATION = (
    "one_scanner_for_both_sides",
    "point_after_a_word_is_a_full_stop",
    "minus_variants_folded_only_directly_before_a_number",
    "tilde_before_a_number_kept_as_tilde",
    "thousands_commas_dropped",
    "leading_zero_before_the_point_implied",
    "fraction_notations_folded_to_one_token",
    "own_brackets_kept",
    "superscript_digits_are_separate_numbers",
    "reader_furniture_may_account_for_a_shipped_row_and_is_never_required",
    "row_found_inside_one_row_or_as_whole_rows_end_to_end",
    "running_text_found_across_rows_that_are_not_table_rows",
    "shipped_rows_made_of_chart_box_layer_numbers_excluded",
    "page_counts_bounded_by_reader_body_and_furniture",
)


class Agreement(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Difference:
    """One row or value the two readings do not share."""

    #: which side holds what the other lacks
    side: str
    kind: str
    #: the row as written (``row_not_found``) or the value (``count``)
    text: str
    #: the row's key; for ``count`` the single value
    numbers: tuple[str, ...]
    #: ``count`` only: occurrences in the shipped text and in the reader's body
    shipped_count: int = 0
    reader_count: int = 0


@dataclass(frozen=True)
class SecondReaderVerdict:
    outcome: Agreement
    #: why the outcome is UNKNOWN; empty otherwise
    unknown_reason: str = ""
    differences: tuple[Difference, ...] = ()
    #: number-bearing rows compared on each side (reader: body rows only)
    shipped_rows: int = 0
    reader_rows: int = 0
    #: rows of reader tables left out as chart-derived (the fallback), with the line above each
    excluded_reader_rows: tuple[str, ...] = ()
    #: shipped rows left out because every number of them is a label printed in a chart box
    excluded_shipped_rows: tuple[str, ...] = ()
    chart_fallback: bool = False
    normalisation: tuple[str, ...] = field(default=NORMALISATION)

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome.value,
            "unknown_reason": self.unknown_reason,
            "differences": [
                {
                    "side": d.side,
                    "kind": d.kind,
                    "text": d.text,
                    "numbers": list(d.numbers),
                    "shipped_count": d.shipped_count,
                    "reader_count": d.reader_count,
                }
                for d in self.differences
            ],
            "shipped_rows": self.shipped_rows,
            "reader_rows": self.reader_rows,
            "excluded_reader_rows": list(self.excluded_reader_rows),
            "excluded_shipped_rows": list(self.excluded_shipped_rows),
            "chart_fallback": self.chart_fallback,
            "normalisation": list(self.normalisation),
        }


# --- numbers ----------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
#: Allowlisted tag-shaped HTML and reader layout tags only (matching ``core/html_tables.py``).
#: Never uses catch-all ``<[^>]+>``, which deletes inequalities (``p < 0.05 and z > 1.96``)
#: and literal tokens such as ``<1>`` or ``<PR>``.
_ALLOWED_HTML_TAGS = (
    "b|strong|i|em|u|s|strike|span|font|small|big|mark|del|ins|tt|code|a|sup|sub|br"
    "|html|head|body|table|thead|tbody|tfoot|tr|td|th|caption|col|colgroup"
    "|p|div|center|hr|img|ul|ol|li|h1|h2|h3|h4|h5|h6|pre|blockquote"
    "|section|article|header|footer|figure|figcaption"
    "|page_header|page_footer|page_number"
)
_HTML_TAG_RE = re.compile(rf"</?(?:{_ALLOWED_HTML_TAGS})(?:\s[^<>\n]*)?/?>", re.I)
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_FRACTION_SLASHES = "/\u2044\u2215"
_SLASH = f"[{re.escape(_FRACTION_SLASHES)}]"
#: ``<sup>1</sup>/<sub>2</sub>`` and the reader's shorter ``<sup>1</sup>/2``
_HTML_FRACTION_RE = re.compile(
    rf"<sup>\s*(\d+)\s*</sup>\s*{_SLASH}\s*(?:<sub>\s*(\d+)\s*</sub>|(\d+)(?![\d.,]))", re.I
)
_LATEX_FRACTION_RE = re.compile(r"\\[dt]?frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}")
_SUPERSCRIPT_DIGITS = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
_SUBSCRIPT_DIGITS = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
_GLYPH_FRACTION_RE = re.compile(rf"([{_SUPERSCRIPT_DIGITS}]+){_SLASH}([{_SUBSCRIPT_DIGITS}]+)")
_SUPERSCRIPT_RUN_RE = re.compile(rf"[{_SUPERSCRIPT_DIGITS}{_SUBSCRIPT_DIGITS}]+")
_HTML_SUPERSCRIPT_RE = re.compile(r"<su[pb]>(.*?)</su[pb]>", re.I | re.S)
#: ``1/2`` written flat. Not part of a longer slashed run (a date ``6/10/2015``) and not a
#: slash between words (``Q4/Q4``).
_PLAIN_FRACTION_RE = re.compile(rf"(?<![\w.,/])(\d+){_SLASH}(\d+)(?![\w/]|[.,]\d)")
#: Private-use brackets around a fraction's parts while the text is scanned, so the scanner
#: reads a fraction as one token whatever notation it came from.
_FRAC_OPEN, _FRAC_MID, _FRAC_CLOSE = "\ue000", "\ue001", "\ue002"

_SIGN = f"[{re.escape(MINUS_CHARS)}{re.escape(TILDE)}]"
#: A point opens a number (``.5``) only where a number can start. Directly after a letter,
#: a closing bracket, a percent or a closing quote it is a full stop, and the digits after it
#: are a separate integer: a footnote callout run on by the text layer (``baseline.13``).
_POINT_START = r"(?<![^\W\d_])(?<![)\]%\"'\u201d\u2019])\.\d+"
_BODY = rf"(?:\d{{1,3}}(?:,\d{{3}})+(?!\d)|\d+)(?:\.\d+)?|{_POINT_START}"
_TOKEN_RE = re.compile(
    rf"{_FRAC_OPEN}(?P<num>\d+){_FRAC_MID}(?P<den>\d+){_FRAC_CLOSE}"
    rf"|(?P<open>\()?(?P<sign>(?<![\w.,)\]%]){_SIGN})?\$?(?P<body>{_BODY})(?P<close>%?\))?"
)


def _fraction(numerator: str, denominator: str) -> str:
    return f" {_FRAC_OPEN}{numerator}{_FRAC_MID}{denominator}{_FRAC_CLOSE} "


def _fold_notation(text: str) -> str:
    """``text`` with every fraction notation in the scanner's one form, markup removed."""
    text = _IMAGE_RE.sub(" ", _COMMENT_RE.sub(" ", text))
    text = _HTML_FRACTION_RE.sub(lambda m: _fraction(m[1], m[2] or m[3]), text)
    # a superscript or subscript is its own token, never more digits of its neighbour
    text = _HTML_SUPERSCRIPT_RE.sub(lambda m: f" {m[1]} ", text)
    # Strip allowlisted HTML/reader tags BEFORE unescaping entities so literal
    # ``&lt;...&gt;`` inequalities and tokens like ``<1>``/``<PR>`` survive intact.
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _LATEX_FRACTION_RE.sub(lambda m: _fraction(m[1], m[2]), text)
    text = _GLYPH_FRACTION_RE.sub(
        lambda m: _fraction(
            unicodedata.normalize("NFKC", m[1]), unicodedata.normalize("NFKC", m[2])
        ),
        text,
    )
    out = []
    for char in text:
        # a vulgar-fraction glyph: Unicode decomposes it to ``numerator U+2044 denominator``
        if unicodedata.decomposition(char).startswith("<fraction>"):
            parts = re.split(_SLASH, unicodedata.normalize("NFKC", char))
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                out.append(_fraction(parts[0], parts[1]))
                continue
        out.append(char)
    text = "".join(out)
    text = _SUPERSCRIPT_RUN_RE.sub(lambda m: f" {unicodedata.normalize('NFKC', m[0])} ", text)
    return _PLAIN_FRACTION_RE.sub(lambda m: _fraction(m[1], m[2]), text)


def row_numbers(text: str) -> tuple[str, ...]:
    """The numbers of one row, canonical, in order. The same function for both sides."""
    out: list[str] = []
    for match in _TOKEN_RE.finditer(_fold_notation(text)):
        if match["num"] is not None:
            out.append(f"{int(match['num'])}/{int(match['den'])}")
            continue
        sign = match["sign"] or ""
        body = match["body"].replace(",", "")
        value = _canon_number(body) or body
        if sign == TILDE:
            value = TILDE + value
        elif sign:
            value = "-" + value
        if match["open"] and match["close"]:
            value = f"({value})"
        out.append(value)
    return tuple(out)


# --- rows -------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    text: str
    key: tuple[str, ...]
    table: bool = False
    #: reader side: which HTML table the row belongs to, or is the caption line of
    group: int | None = None


_FENCE_RE = re.compile(r"^\s*```[A-Za-z]*\s*$")
_FURNITURE_RE = re.compile(r"<(page_header|page_footer|page_number)>(.*?)</\1>", re.S | re.I)
_FURNITURE_OPEN_RE = re.compile(r"<(?:page_header|page_footer)>", re.I)
_FURNITURE_CLOSE_RE = re.compile(r"</(?:page_header|page_footer)>", re.I)
_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.S | re.I)
_TABLE_OPEN_RE = re.compile(r"<table[\s>]", re.I)
_TABLE_CLOSE_RE = re.compile(r"</table>", re.I)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_ROW_OPEN_RE = re.compile(r"<tr[\s>]", re.I)
_ROW_CLOSE_RE = re.compile(r"</tr>", re.I)
_CELL_END_RE = re.compile(r"</t[dh]>", re.I)
#: a line socr authored, not the page: a mark in a blockquote, or a ``[socr: …]`` note
_OWN_LINE_RE = re.compile(r"^\s*>\s*\[")
_PICTURE_RE = _IMAGE_RE


def _row(text: str, *, table: bool = False, group: int | None = None) -> _Row | None:
    key = row_numbers(text)
    if not key:
        return None
    return _Row(" ".join(_HTML_TAG_RE.sub(" ", text).split()), key, table, group)


def _line_rows(text: str) -> list[_Row]:
    """One row per number-bearing line; a markdown pipe row is a table row."""
    return [
        row
        for line in text.split("\n")
        if (row := _row(line, table=line.lstrip().startswith("|"))) is not None
    ]


def shipped_rows(markdown: str) -> list[_Row]:
    """The number-bearing lines of the shipped page, socr's own lines left out."""
    lines = [
        SOCR_MARKER_RE.sub(" ", line)
        for line in (markdown or "").split("\n")
        if not _OWN_LINE_RE.match(line)
    ]
    return _line_rows("\n".join(lines))


def _unfenced(reader: str) -> tuple[str, bool]:
    """The reader's output without its code fence, and "the fence was left open"."""
    lines = reader.strip().split("\n")
    fences = [i for i, line in enumerate(lines) if _FENCE_RE.match(line)]
    if not fences:
        return reader, False
    if fences[0] == 0 and len(fences) % 2:
        return "\n".join(lines[1:]), True
    return "\n".join(line for i, line in enumerate(lines) if i not in fences), False


def _unclosed(text: str) -> bool:
    return (
        len(_TABLE_OPEN_RE.findall(text)) != len(_TABLE_CLOSE_RE.findall(text))
        or len(_ROW_OPEN_RE.findall(text)) != len(_ROW_CLOSE_RE.findall(text))
        or len(_FURNITURE_OPEN_RE.findall(text)) != len(_FURNITURE_CLOSE_RE.findall(text))
    )


def reader_rows(text: str) -> tuple[list[_Row], list[list[_Row]]]:
    """``(body rows, furniture blocks)`` of an unfenced reader output."""
    furniture: list[list[_Row]] = []

    def take_furniture(match: re.Match) -> str:
        rows = _line_rows(match[2])
        if rows:
            furniture.append(rows)
        return "\n"

    text = _FURNITURE_RE.sub(take_furniture, text)
    body: list[_Row] = []
    cursor = 0
    for group, table in enumerate(_TABLE_RE.finditer(text)):
        before = text[cursor : table.start()]
        lines = before.rstrip().split("\n")
        body.extend(_line_rows("\n".join(lines[:-1])))
        caption = _row(lines[-1], table=lines[-1].lstrip().startswith("|"), group=group)
        if caption is not None:
            body.append(caption)
        for tr in _ROW_RE.finditer(table[1]):
            cells = _CELL_END_RE.sub(" | ", _BR_RE.sub(" ", tr[1])).replace("\n", " ")
            row = _row(cells, table=True, group=group)
            if row is not None:
                body.append(row)
        cursor = table.end()
    body.extend(_line_rows(text[cursor:]))
    return body, furniture


# --- matching ---------------------------------------------------------------------------


def _inside_one(key: tuple[str, ...], rows: list[_Row]) -> bool:
    n = len(key)
    return any(row.key[i : i + n] == key for row in rows for i in range(len(row.key) - n + 1))


def _whole_rows(key: tuple[str, ...], rows: list[_Row]) -> bool:
    """Is ``key`` whole rows of ``rows`` laid end to end, in any order?"""
    keys = {row.key for row in rows}
    reachable = [True] + [False] * len(key)
    for start in range(len(key)):
        if reachable[start]:
            for other in keys:
                if key[start : start + len(other)] == other:
                    reachable[start + len(other)] = True
    return reachable[len(key)]


def _stretches(rows: list[_Row]) -> list[_Row]:
    """Each maximal run of consecutive non-table rows as one row."""
    out: list[_Row] = []
    run: list[_Row] = []
    for row in [*rows, None]:
        if row is not None and not row.table:
            run.append(row)
            continue
        if run:
            out.append(_Row("", tuple(value for r in run for value in r.key)))
            run = []
    return out


def _found(row: _Row, rows: list[_Row]) -> bool:
    """Rules (a), (b) and, for a row that is not a table row, (c)."""
    if _inside_one(row.key, rows) or _whole_rows(row.key, rows):
        return True
    return not row.table and _inside_one(row.key, _stretches(rows))


def _counts(rows: list[_Row]) -> Counter:
    return Counter(value for row in rows for value in row.key)


def compare_readings(
    shipped: str,
    reader: str,
    *,
    chart_picture: bool | None = None,
    chart_label_text: str | None = None,
    reader_truncated: bool = False,
) -> SecondReaderVerdict:
    """Compare the numbers of the ``shipped`` page markdown with the second ``reader``'s raw
    output for the same page image.

    ``chart_picture``: the reader may have seen a chart (the image was not masked) and the
    page ships a picture, so the chart fallback applies. ``None`` detects it from an embedded
    picture in ``shipped``; pass ``False`` when the chart boxes were masked in the image.
    ``chart_label_text``: what the page's text layer holds inside its chart boxes.
    ``reader_truncated``: the caller knows the output was cut off (``finish_reason`` length).
    """
    if not (reader or "").strip():
        return SecondReaderVerdict(Agreement.UNKNOWN, unknown_reason=UNKNOWN_EMPTY)
    text, open_fence = _unfenced(reader)
    if reader_truncated or open_fence:
        return SecondReaderVerdict(Agreement.UNKNOWN, unknown_reason=UNKNOWN_TRUNCATED)
    if _unclosed(text):
        return SecondReaderVerdict(Agreement.UNKNOWN, unknown_reason=UNKNOWN_UNCLOSED)
    ours = shipped_rows(shipped)
    body, furniture = reader_rows(text)
    if not ours and not body:
        return SecondReaderVerdict(Agreement.UNKNOWN, unknown_reason=UNKNOWN_NO_NUMBERS)
    fallback = bool(_PICTURE_RE.search(shipped or "")) if chart_picture is None else chart_picture

    labels = Counter(row_numbers(chart_label_text or ""))
    differences: list[Difference] = []
    labelled: list[_Row] = []
    for row in ours:
        if _found(row, body) or any(_found(row, block) for block in furniture):
            continue
        if labels and not Counter(row.key) - labels:
            labelled.append(row)
            continue
        differences.append(Difference(SIDE_SHIPPED, DIFF_ROW_NOT_FOUND, row.text, row.key))
    unfound = [row for row in body if not _found(row, ours)]
    charts = {row.group for row in unfound if fallback and row.table and row.group is not None}
    excluded = [row for row in body if row.group in charts]
    for row in unfound:
        if row.group not in charts:
            differences.append(Difference(SIDE_READER, DIFF_ROW_NOT_FOUND, row.text, row.key))

    have = _counts(ours)
    floor = _counts(body) - _counts(excluded)
    ceiling = _counts(body) + _counts([row for block in furniture for row in block]) + labels
    for value in sorted(set(have) | set(floor)):
        if have[value] < floor[value]:
            side = SIDE_READER
        elif have[value] > ceiling[value]:
            side = SIDE_SHIPPED
        else:
            continue
        differences.append(
            Difference(side, DIFF_COUNT, value, (value,), have[value], _counts(body)[value])
        )
    return SecondReaderVerdict(
        Agreement.DISAGREE if differences else Agreement.AGREE,
        differences=tuple(differences),
        shipped_rows=len(ours),
        reader_rows=len(body),
        excluded_reader_rows=tuple(row.text for row in excluded),
        excluded_shipped_rows=tuple(row.text for row in labelled),
        chart_fallback=fallback,
    )


def describe_differences(verdict: SecondReaderVerdict) -> str:
    """The differences as one mark reason, every one named.

    Joined with `` / `` because the mark grammar turns ``;`` into ``,`` (``clean_reason``),
    and written without square brackets, which it turns into round ones.
    """
    parts = []
    for d in verdict.differences:
        numbers = " ".join(d.numbers)
        if d.kind == DIFF_COUNT:
            parts.append(
                f"{d.text} is shipped {d.shipped_count} time(s), "
                f"the second reading has it {d.reader_count} time(s)"
            )
        elif d.side == SIDE_SHIPPED:
            parts.append(f"shipped row with the numbers {numbers} is not in the second reading")
        else:
            parts.append(
                f"second reading's row with the numbers {numbers} is not in the shipped text"
            )
    return " / ".join(parts)
