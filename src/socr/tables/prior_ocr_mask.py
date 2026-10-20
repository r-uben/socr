"""#1156 Part B: an earlier OCR layer's prose, with every number in it withheld.

The owner's ruling (2026-10-17, temporary until a second reader can confirm these pages):
a scan whose text layer is an EARLIER OCR run, and whose model reading was rejected or
never made, keeps that layer's prose under its mark, every number in it replaced by a
visible token, and the original -- numbers included -- retained in the sidecar, unshipped.

The transform runs ONCE, on the retained ORIGINAL layer, before socr adds any line of its
own (a mark, a fence, an image reference). That is what makes it idempotent across a
re-finalisation and a resume: a second pass over the same page reads the same source and
computes the same bytes. Nothing here may ever be applied to text socr has finalised.

Ported from the abandoned first attempt (``fix/issue-1156`` at 35a00024, whose review
rounds built the number grammar below). Its order-sensitive comparator is NOT ported:
nothing in production calls it until a second reader exists (#1208), and that branch keeps
it.
"""

from __future__ import annotations

import re
import sys

from socr.tables.native_verifier import _normalize_numeric_token


#: Every character Unicode calls a number beyond the ASCII digits -- ``Nd`` (other
#: scripts' digits, fullwidth digits), ``No`` (vulgar fractions, superscripts, circled
#: numbers) and ``Nl`` (letter numbers: precomposed Roman numerals) -- derived from the
#: Unicode database at import rather than listed by hand. A hand list missed 1,702 such
#: characters (Argon, Part B diff review); ``\\d`` alone is ``Nd`` only and shipped the
#: half of "3 1/2" written as one glyph (Argon, stack review). Such a character may end a
#: digit run ("3\u00bd" is one number) or stand alone, and a sign in front of it belongs
#: to it.
def _non_ascii_numeral_class() -> str:
    import unicodedata

    spans: list[list[int]] = []
    for cp in range(0x80, sys.maxunicode + 1):
        if unicodedata.category(chr(cp)) in ("Nd", "No", "Nl"):
            if spans and spans[-1][1] == cp - 1:
                spans[-1][1] = cp
            else:
                spans.append([cp, cp])
    body = "".join(
        re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}"
        for a, b in spans
    )
    return f"[{body}]"


_VULGAR = _non_ascii_numeral_class()

#: A minus written as an HTML entity belongs to the number, not to the prose. A sign
#: counts only when a numeral does not precede it, so the dash of a range ("12-13")
#: stays on the page instead of being eaten as the second number's minus.
_SIGN = r"(?:[-+−–—]|&minus;|&#8722;|&#x2212;)?"

#: Numerals that can be an OPERAND of a range, so a dash after one is a dash and not a
#: sign. The ASCII digits only: a superscript footnote marker is not an operand, and
#: treating it as one swallowed the sign after it (``¹-0.021`` read as
#: ``¹+0.021``).
_RANGE_OPERAND = "0-9"

#: A number's body: numerals, with ``,`` admitted only as a thousands separator (exactly
#: three digits after it, so "June 18,1957" is two numbers) and ``.`` as a decimal point.
#: The integer part may be EMPTY ("-.021", "(.023)"), and a vulgar fraction or an entity
#: may close a run, so "3½" and "3&#189;" are each ONE number.
_BODY = (
    rf"(?:[{_RANGE_OPERAND}]+(?:(?:,[0-9]{{3}}(?![0-9]))|(?:[. ][{_RANGE_OPERAND}]+))*"
    rf"(?:{_VULGAR}|&[#a-zA-Z0-9]+;)?"
    rf"|\.[{_RANGE_OPERAND}]+"
    rf"|{_VULGAR})"
)

#: A signed number comes first, so an entity that is a minus ("&minus;9.9") is that
#: number's sign; the bare-entity alternative is last, for one that carries no sign. The
#: entity alternative sits outside the lookbehind, so "3&#189;" is one number and not
#: "3" followed by "189".
_NUMBER_RE = re.compile(
    rf"(?:(?<![{_RANGE_OPERAND}]){_SIGN}(?:{_BODY}|&[#a-zA-Z0-9]+;)|&[#a-zA-Z0-9]+;)"
)

#: What a withheld number becomes when the page image ships beside the text.
NUMBER_WITHHELD = "[number withheld, see image]"
#: ... and when it does not (figure saving off, or the render failed). Deliberately
#: digit-free: the token sits inside the layer's own lines, where any digit would read as
#: part of the page (Argon, design review). The page's mark already names the page.
NUMBER_WITHHELD_NO_IMAGE = "[number withheld, see PDF page]"


def _number_matches(text: str):
    """Every number in *text*, in reading order, as ``(match, normalised value)``.

    The SINGLE enumeration: :func:`numeric_tokens_in_order` and :func:`mask_numbers` both
    walk it, so position *k* means the same number to anything that confirms a position
    and to the masker.
    """
    if not text:
        return
    for match in _NUMBER_RE.finditer(text):
        value = _normalize_numeric_token(match.group(0))
        # An entity that decodes to no numeral at all (a stray "&amp;") is not a number.
        if not value or not any(ch.isdigit() or ch.isnumeric() for ch in value):
            continue
        # A currency symbol or a significance mark may sit between the bracket and the
        # number on either side: "($400)" and "(0.023*)" are bracketed values.
        before = text[: match.start()].rstrip().rstrip("*%$£€¥†‡§")
        after = text[match.end() :].lstrip().lstrip("*%$†‡§")
        if before.endswith("(") and after.startswith(")"):
            value = f"({value})"
        yield match, value


def numeric_tokens_in_order(text: str) -> tuple[str, ...]:
    """The numbers in *text*, normalised, in the order they are written."""
    return tuple(value for _match, value in _number_matches(text))


def mask_numbers(
    text: str, confirmed: frozenset[int] = frozenset(), *, token: str = NUMBER_WITHHELD
) -> tuple[str, int]:
    """Replace every number in *text* with *token*, except the positions in *confirmed*.

    ``confirmed`` holds POSITIONS in :func:`numeric_tokens_in_order`'s order, never
    values: by value, a confirmed ``400`` would uncover the ``400`` inside ``(400)``, a
    different quantity. Every number run is masked -- a date, a range, a footnote callout
    -- because nothing here can tell which of them a reader would cite. One token per
    number, never collapsed. Returns the masked text and how many numbers were withheld.
    """
    out: list[str] = []
    cursor = 0
    withheld = 0
    for index, (match, _value) in enumerate(_number_matches(text)):
        out.append(text[cursor : match.start()])
        if index in confirmed:
            out.append(match.group(0))
        else:
            withheld += 1
            out.append(token)
        cursor = match.end()
    out.append(text[cursor:])
    return "".join(out), withheld


def withhold_shreds(text: str) -> tuple[str, int]:
    """Drop the lines of an earlier OCR layer that are fragments, not text.

    A SHRED is a line whose content is a single letter: what a scanner's OCR makes of a
    rotated axis label, ``P``, ``r``, ``i``, ``c``, ``e``, one letter per line (#1027's
    "known garbage"). Measured over 311 prior-OCR pages, page-level statistics form a
    smooth gradient with no trough, so a page-level cut would be a guessed number; the
    line-level distinction is a difference in kind and needs none. A lone digit is not a
    shred -- it can be a table value -- and neither is a word, however short.
    """
    kept: list[str] = []
    dropped = 0
    for line in (text or "").splitlines():
        stripped = line.strip()
        if len(stripped) == 1 and stripped.isalpha():
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept).strip(), dropped


#: A word: two letters or more in a row. What "the layer reads" means below.
_WORD_RE = re.compile(r"[^\W\d_]{2,}")


def independently_confirmed_positions(record: dict | None, layer: str) -> frozenset[int]:
    """THE SEAM for #1208: number positions a second, independent reader confirmed.

    Nothing in production writes such a record yet, so today this is always empty and
    every number is withheld. The contract a writer must meet, each clause fail-closed:

    * ``provenance == INDEPENDENT_READER`` -- a reader that is neither the layer itself
      nor the same model's judge. The layer is never evidence for itself;
    * ``agreed is True`` over the WHOLE page. A partial agreement unmasks nothing: two
      readings that disagree somewhere can match small integers at misaligned positions
      by coincidence (Argon, design review);
    * ``compared`` equals the number count of *layer*, so the record was computed
      against this exact text and its positions index it;
    * ``confirmed`` names EVERY position. A whole-page agreement confirms them all; a
      record that agrees yet confirms a subset is self-contradictory, and honouring it
      could uncover the "3" of a "3 1/2" whose fraction the reader did not confirm.
    """
    if not isinstance(record, dict):
        return frozenset()
    if record.get("provenance") != INDEPENDENT_READER or record.get("agreed") is not True:
        return frozenset()
    count = len(numeric_tokens_in_order(layer))
    if count == 0 or record.get("compared") != count:
        return frozenset()
    confirmed = record.get("confirmed") or ()
    if any(not isinstance(i, int) or isinstance(i, bool) for i in confirmed):
        return frozenset()
    every = frozenset(range(count))
    return every if frozenset(confirmed) == every else frozenset()


#: The only provenance :func:`independently_confirmed_positions` accepts.
INDEPENDENT_READER = "independent_reader"


def masked_layer_text(
    layer: str, confirmed: frozenset[int] = frozenset(), *, token: str = NUMBER_WITHHELD
) -> tuple[str, int, int]:
    """The layer as it ships: numbers masked, THEN shreds withheld.

    Masking first, on the raw layer, so a confirmed position indexes the same text the
    record was computed on; dropping a shred line afterwards cannot shift it, and no
    masked token is a single letter (Argon, design review).

    Returns ``(text, numbers_withheld, shreds_withheld)``. ``text`` is EMPTY when no word
    survives -- only masked numbers, punctuation or nothing at all -- and the caller then
    ships its fail-closed marker, which is #1027's own case: a page number alone is not
    prose to keep.
    """
    masked, withheld = mask_numbers(layer or "", confirmed, token=token)
    kept, shreds = withhold_shreds(masked)
    if not _WORD_RE.search(kept.replace(token, "")):
        return "", withheld, shreds
    return kept, withheld, shreds
