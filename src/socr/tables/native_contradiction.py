"""A shipped table that the PDF's own text layer contradicts.

A table the ladder could not verify ships as text with a flag. Most of those
tables are fine. Some are not, and for some of them the PDF itself says so: the
text layer prints a minus the table dropped, prints a row's values under a
different row label than the grid binds them to, or does not print a number the
grid shows. This module finds exactly those, mechanically, from the page's own
words. A table with such a contradiction is withheld instead of shipped as
"unverified" (see ``orchestrator._withhold_contradicted_unverified_tables``).

Evidence rule. A check speaks only when the native text layer is usable for the
table; otherwise it abstains (``NO_EVIDENCE``) and the table keeps its current
disposition. Abstention is never read as agreement and never as contradiction.
The text layer is unusable when

* most of its characters are invisible (render mode 3): an OCR layer under a
  raster, whose digits are the OCR engine's reading and not the printer's; or
* more than ``EXTRA_NUMBERS_MAX_SHARE`` (row_corroboration's existing tolerance)
  of the table's numbers are absent from it: it does not carry this table.

Checks (each a ``Contradiction`` kind):

``sign``         The table prints a number unsigned that the page prints with a
                 minus, or the reverse. The page's minus takes four shapes, all
                 read at character level because a word-level extraction loses
                 the first two: an undecodable control character before the
                 digits (#990), a ``2`` in another font at the digits' size (#913),
                 a minus glyph abutting the digits, and the ``- 0.48`` sign-space
                 form (#930). Compared by absolute value as a multiset: a value is
                 contradicted when the table has more unsigned copies than the page
                 has unsigned copies AND the page has more signed copies than the
                 table (and symmetrically for an invented minus). A range
                 (``1990-2000``) and a digit inside a label (``SIEM50``) are not
                 signed numbers.
``row_shift``    A table row's numeric run sits, whole, on exactly one printed
                 line of the table's region; that line prints a row label, and the
                 grid binds the run to a different label (or to no label) while
                 binding that printed label to other values. Needs the table's
                 region (``locate_tables``) and upright text; abstains otherwise.
``number_absent``  A number the table shows that the region's text layer does not
                 print at all (``row_corroboration``'s ``extra_numbers``). Fires
                 only while the layer demonstrably carries the table (see above),
                 so one wrong cell in a table whose other numbers all match is
                 the case it exists for.

Not covered, by design: header attribution, shredded captions, phantom tables
and row-count errors. The existing helpers cannot test them without also
flagging tables a reader would call correct (measured, see the decision log).

No new thresholds: every tolerance here is an existing named quantity
(``EXTRA_NUMBERS_MAX_SHARE``, ``MINUS_AS_DIGIT_SIZE_TOLERANCE_PT``,
``_SAME_TEXT_DIRECTION_TOL_RAD``). The only cut introduced is a majority, which
is what "the layer's text is mostly invisible" means.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

from socr.core.glyph_recovery import MINUS_AS_DIGIT_SIZE_TOLERANCE_PT
from socr.tables.native_verifier import (
    _normalize_cell,
    _normalize_numeric_token,
    is_numeric_token,
)
from socr.tables.row_corroboration import (
    EXTRA_NUMBERS_MAX_SHARE,
    cluster_band_words,
    corroborate_rows,
    table_blocks,
    words_in_region,
)

logger = logging.getLogger(__name__)

#: Contradiction kinds, recorded on the event and the page.
SIGN = "sign"
ROW_SHIFT = "row_shift"
NUMBER_ABSENT = "number_absent"
CONTRADICTION_KINDS: tuple[str, ...] = (SIGN, ROW_SHIFT, NUMBER_ABSENT)

#: Outcome of one check.
CONTRADICTED = "contradicted"
CLEAR = "clear"
NO_EVIDENCE = "no_evidence"

#: Glyphs that print a minus: U+2212, en dash, em dash, hyphen-minus (escaped, it is
#: a character-class member).
_MINUS_CLASS = "−–—\\-"
#: An undecodable glyph (#990): every C0 control except tab, newline, carriage return.
_CONTROL_CLASS = "\x00-\x08\x0b\x0c\x0e-\x1f"
#: A printed number: optional opening bracket, thousands groups or a bare run, an
#: optional decimal part, or a leading-point decimal.
_NUMBER = r"\(?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
#: A sign directly before a number (one optional space: the ``- 0.48`` form, #930). The
#: sign must not follow a word character or a closing bracket (``1990-2000``), and the
#: number must not follow a word character (``SIEM50``).
_PAGE_NUMBER_RE = re.compile(
    rf"(?:(?<![\w)\]])(?P<sg>[{_MINUS_CLASS}{_CONTROL_CLASS}])[  ]?)?"
    rf"(?<![\w])(?P<n>{_NUMBER})"
)
_CELL_NUMBER_RE = re.compile(
    rf"(?:(?<![\w)\]])(?P<sg>[{_MINUS_CLASS}])[  ]?)?(?<![\w])(?P<n>{_NUMBER}\)?)"
)
_WORD_RE = re.compile(r"[a-z]+")
#: A word of one letter is a footnote mark or a variable, not a row label.
_MIN_LABEL_WORD_LEN = 2

#: How a page prints a sign.
_NEG, _POS = "neg", "pos"


@dataclass(frozen=True)
class Contradiction:
    """One mechanical disagreement between a table and the page's own text."""

    kind: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail}


# -- evidence usability -------------------------------------------------------------


def text_layer_is_visible(page) -> bool:
    """False when most of the page's text is invisible (render mode 3).

    That is an OCR layer sitting under a raster: its digits are an OCR engine's
    reading, so agreement or disagreement with it proves nothing about the print.
    A page with no text at all has no layer to read either.
    """
    invisible = total = 0
    for trace in page.get_texttrace():
        count = len(trace["chars"])
        total += count
        if trace["type"] == 3:
            invisible += count
    return total > 0 and invisible * 2 <= total


def _abs_value(raw: str) -> str | None:
    raw = raw.strip().lstrip("(").rstrip(")")
    if not is_numeric_token(raw):
        return None
    return _normalize_numeric_token(raw)


# -- sign ---------------------------------------------------------------------------


def native_signed_numbers(page) -> list[tuple[str, str]]:
    """``(absolute value, sign)`` for every number the page prints, at character level.

    ``sign`` is ``"neg"`` for a minus glyph, an undecodable control character or a
    mis-extracted ``2`` standing for a minus (#990, #913), else ``"pos"``.
    """
    found: list[tuple[str, str]] = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", ()):
            spans = line["spans"]
            # #913: a one-character ``2`` in another font at the digits' size is a
            # minus drawn from a symbol font. Read it as the control character it
            # stands for, so the number after it is not parsed as "2.029".
            swapped = {}
            for a, b in zip(spans, spans[1:]):
                if (
                    len(a["chars"]) == 1
                    and a["chars"][0]["c"] == "2"
                    and a["font"] != b["font"]
                    and abs(a["size"] - b["size"]) <= MINUS_AS_DIGIT_SIZE_TOLERANCE_PT
                ):
                    after = "".join(c["c"] for c in b["chars"]).lstrip()
                    if after[:1].isdigit() or (after[:1] == "." and after[1:2].isdigit()):
                        swapped[id(a["chars"][0])] = "\x01"
            text = "".join(swapped.get(id(c), c["c"]) for s in spans for c in s["chars"])
            for match in _PAGE_NUMBER_RE.finditer(text):
                value = _abs_value(match.group("n"))
                if value is not None:
                    found.append((value, _NEG if match.group("sg") else _POS))
    return found


def table_signed_numbers(markdown: str) -> list[tuple[str, str, int]]:
    """``(absolute value, sign, block index)`` for every number a table prints."""
    found: list[tuple[str, str, int]] = []
    for block_index, rows in enumerate(table_blocks(markdown)):
        for row in rows:
            for cell in row:
                for match in _CELL_NUMBER_RE.finditer(_normalize_cell(cell)):
                    value = _abs_value(match.group("n"))
                    if value is not None:
                        found.append((value, _NEG if match.group("sg") else _POS, block_index))
    return found


def sign_contradictions(
    native: list[tuple[str, str]],
    page_table_numbers: list[tuple[str, str, int]],
    target_numbers: list[tuple[str, str, int]],
) -> tuple[str, list[Contradiction]]:
    """Signs the table prints against signs the page prints.

    *page_table_numbers* are every table number on the page (so a sibling table's
    copy of a value is not mistaken for this table's); *target_numbers* are the
    numbers of the table(s) being judged. Returns the outcome and the findings that
    concern the targets.
    """
    if not page_table_numbers:
        return NO_EVIDENCE, []
    native_total: Counter = Counter(v for v, _ in native)
    table_total: Counter = Counter(v for v, *_ in page_table_numbers)
    absent = sum(max(0, table_total[v] - native_total[v]) for v in table_total)
    if not native_total or absent / sum(table_total.values()) > EXTRA_NUMBERS_MAX_SHARE:
        return NO_EVIDENCE, []
    native_neg = Counter(v for v, s in native if s == _NEG)
    native_pos = Counter(v for v, s in native if s == _POS)
    table_neg = Counter(v for v, s, _ in page_table_numbers if s == _NEG)
    table_pos = Counter(v for v, s, _ in page_table_numbers if s == _POS)
    target = {(v, s) for v, s, _ in target_numbers}
    found: list[Contradiction] = []
    for value in sorted(table_total):
        dropped = min(table_pos[value] - native_pos[value], native_neg[value] - table_neg[value])
        if dropped > 0 and (value, _POS) in target:
            found.append(Contradiction(SIGN, f"{value}: the page prints a minus the table dropped"))
        invented = min(table_neg[value] - native_neg[value], native_pos[value] - table_pos[value])
        if invented > 0 and (value, _NEG) in target:
            found.append(
                Contradiction(SIGN, f"{value}: the table prints a minus the page does not")
            )
    return (CONTRADICTED if found else CLEAR), found


# -- row shift ----------------------------------------------------------------------


_SIGNED_WORD_RE = re.compile(rf"[{_MINUS_CLASS}]?(?P<n>{_NUMBER}\)?)")


def _printed_abs_value(word: str) -> str | None:
    """The unsigned value of one printed word, or None when it is not a number.

    Unsigned because a row's cells are compared by value (``_row_values``), and a
    row with a negative cell must still be found on its printed line.
    """
    match = _SIGNED_WORD_RE.fullmatch(word.strip("*"))
    return _abs_value(match.group("n")) if match else None


def _label_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= _MIN_LABEL_WORD_LEN}


def _row_values(cells: list[str]) -> list[str]:
    values: list[str] = []
    for cell in cells:
        for match in _CELL_NUMBER_RE.finditer(_normalize_cell(cell)):
            value = _abs_value(match.group("n"))
            if value is not None:
                values.append(value)
    return values


def row_shift_contradictions(words: list, markdown: str, region) -> tuple[str, list[Contradiction]]:
    """Row values the grid binds to a different label than the page prints them under."""
    region_words = words_in_region(words, region)
    printed: list[tuple[tuple[str, ...], set[str]]] = []
    for band in cluster_band_words(region_words):
        numbers: list[str] = []
        label: set[str] = set()
        seen_number = False
        for word in sorted(band, key=lambda w: w[0]):
            value = _printed_abs_value(word[4])
            if value is not None:
                numbers.append(value)
                seen_number = True
            elif not seen_number:
                label |= _label_words(word[4])
        printed.append((tuple(numbers), label))
    if not any(numbers for numbers, _ in printed):
        return NO_EVIDENCE, []

    checked = 0
    found: list[Contradiction] = []
    for rows in table_blocks(markdown):
        stubs = [_label_words(row[0]) if row else set() for row in rows]
        for index, row in enumerate(rows):
            values = _row_values(row[1:])
            if len(values) < 2:
                continue
            want = tuple(values)
            lines = [
                i
                for i, (numbers, _) in enumerate(printed)
                if any(
                    numbers[s : s + len(want)] == want for s in range(len(numbers) - len(want) + 1)
                )
            ]
            if len(lines) != 1:
                continue
            checked += 1
            label = printed[lines[0]][1]
            if not label or label & stubs[index]:
                continue
            if any(label & stubs[j] for j in range(len(rows)) if j != index):
                found.append(
                    Contradiction(
                        ROW_SHIFT,
                        f"row {index}: its values are printed on the line labelled "
                        f"{sorted(label)[:3]}, which the grid binds to other values",
                    )
                )
    if checked == 0:
        return NO_EVIDENCE, []
    return (CONTRADICTED if found else CLEAR), found


def page_is_upright(page, region) -> bool:
    """Whether every text line inside *region* runs left to right."""
    from socr.tables.ship_gate import _SAME_TEXT_DIRECTION_TOL_RAD, _angle_between

    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", ()):
            x0, y0, x1, y1 = line["bbox"]
            if region is not None and not (
                region[0] <= (x0 + x1) / 2 <= region[2] and region[1] <= (y0 + y1) / 2 <= region[3]
            ):
                continue
            if _angle_between(tuple(line["dir"]), (1.0, 0.0)) > _SAME_TEXT_DIRECTION_TOL_RAD:
                return False
    return True


# -- absent number ------------------------------------------------------------------


def number_absent_contradictions(
    words: list, markdown: str, region
) -> tuple[str, list[Contradiction]]:
    """Numbers the table prints that the region's text layer does not."""
    result = corroborate_rows(words, markdown, region)
    if result.candidate_numbers == 0 or result.native_numeric_rows == 0:
        return NO_EVIDENCE, []
    if result.extra_share is not None and result.extra_share > EXTRA_NUMBERS_MAX_SHARE:
        return NO_EVIDENCE, []
    if not result.extra_numbers:
        return CLEAR, []
    shown = ", ".join(sorted(set(result.extra_numbers))[:5])
    return CONTRADICTED, [
        Contradiction(
            NUMBER_ABSENT,
            f"{len(result.extra_numbers)} of {result.candidate_numbers} numbers are not printed "
            f"on the page ({shown})",
        )
    ]


# -- entry point --------------------------------------------------------------------


def contradictions_for_tables(
    page,
    page_markdown: str,
    target_markdowns: list[str],
    regions: list | None = None,
) -> list[list[Contradiction]]:
    """Findings per target table, in order. Never raises: a failure abstains.

    *page_markdown* is the page's whole shipped text (its other tables count for the
    sign multiset). *regions* is one ``(x0, y0, x1, y1)`` per target, from the
    table locator; ``None`` (or a ``None`` entry) means the table could not be placed,
    which abstains the region checks and leaves the page-level sign check.
    """
    none: list[list[Contradiction]] = [[] for _ in target_markdowns]
    try:
        if not text_layer_is_visible(page):
            return none
        words = page.get_text("words")
        native = native_signed_numbers(page)
        page_numbers = table_signed_numbers(page_markdown)
        out: list[list[Contradiction]] = []
        for index, markdown in enumerate(target_markdowns):
            region = regions[index] if regions and index < len(regions) else None
            found: list[Contradiction] = []
            _, sign = sign_contradictions(native, page_numbers, table_signed_numbers(markdown))
            found += sign
            if region is not None:
                if page_is_upright(page, region):
                    found += row_shift_contradictions(words, markdown, region)[1]
                found += number_absent_contradictions(words, markdown, region)[1]
            out.append(found)
        return out
    except Exception as exc:  # the check is advisory; doubt keeps today's behaviour
        logger.warning(
            "native contradiction check failed (%s: %s); abstaining", type(exc).__name__, exc
        )
        return none
