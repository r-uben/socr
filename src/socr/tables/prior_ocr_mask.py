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

import html.entities
import re
import sys

from socr.tables.native_verifier import _normalize_numeric_token


def _numeral_class(categories: tuple[str, ...]) -> str:
    """A regex character class of every code point in the given Unicode categories,
    derived from the Unicode database at import rather than listed by hand: a hand list
    missed 1,702 numerals (Argon, Part B diff review)."""
    import unicodedata

    spans: list[list[int]] = []
    for cp in range(sys.maxunicode + 1):
        if unicodedata.category(chr(cp)) in categories:
            if spans and spans[-1][1] == cp - 1:
                spans[-1][1] = cp
            else:
                spans.append([cp, cp])
    body = "".join(
        re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}"
        for a, b in spans
    )
    return f"[{body}]"


#: Every numeral that is not a decimal digit: ``No`` (vulgar fractions, superscripts,
#: circled numbers) and ``Nl`` (letter numbers, the precomposed Roman numerals). Decimal
#: digits of every script (``Nd``) are ``\d``. Such a character may close a digit run
#: ("3½" is one number), and a run of them is one number ("ⅩⅡ").
_OTHER_NUMERAL = _numeral_class(("No", "Nl"))

#: Named HTML entities that write a numeral ("&frac12;", "&sup2;"), from the HTML5 entity
#: table: an entity may close or form a number only when it decodes to one. Matching any
#: entity there swallowed "&ndash;" into "3&ndash;5" (Argon, Part B diff review).
_NUMERIC_ENTITY = "&(?:{})".format(
    "|".join(
        sorted(
            (
                re.escape(name)
                for name, value in html.entities.html5.items()
                if name.endswith(";") and value and all(ch.isnumeric() for ch in value)
            ),
            key=len,
            reverse=True,
        )
    )
)

#: A sign belongs to the number it precedes, unless a decimal digit precedes it: then it
#: is the dash of a range ("12-13") and stays on the page. ``&minus;`` is the one entity
#: spelling of a sign left after character references are decoded.
_SIGN = r"(?:[-+−–—]|&minus;)?"

#: A number's tail: a numeral that is not a decimal digit, or an entity that writes one.
_TAIL = rf"(?:{_OTHER_NUMERAL}|{_NUMERIC_ENTITY})"

#: A number's body. ``,`` joins only a thousands group (exactly three digits after it, so
#: "June 18,1957" is two numbers); ``.`` is a decimal point. The integer part may be EMPTY
#: ("-.021", "(.023)"), and either form may end in a tail: "3½" and ".5½" are
#: each ONE number -- without the tail on the leading-dot form, the "½" of ".5½"
#: shipped unmasked (Argon, Part B diff review). A run of tails stands alone.
_BODY = rf"(?:\d+(?:,\d{{3}}(?!\d)|[. ]\d+)*{_TAIL}?|\.\d+{_TAIL}?|{_TAIL}+)"

_NUMBER_RE = re.compile(rf"(?<!\d){_SIGN}{_BODY}")

#: The five characters that are markup in Markdown and HTML. A character reference to
#: one of them stays an entity (named, so it carries no digit); decoding it would turn a
#: layer's "&#60;script&#62;" into raw HTML on the page.
_MARKUP_ENTITIES = {"<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&apos;"}
#: The closing ";" is optional: an HTML parser decodes "&#xBD" as "\u00bd" without it, and
#: those hex digits are letters, so neither the grammar nor the invariant would see the
#: numeral it writes (Argon, round 2 re-review).
_CHAR_REF_RE = re.compile(r"&#(?:[xX]([0-9a-fA-F]+)|([0-9]+));?")


def _decode_character_references(text: str) -> str:
    """Write each numeric character reference as the character it names.

    "&#8212;" is an em dash, not the number 8212; left as written, its digits shipped
    (Argon, Part B diff review). Masking runs on the decoded text, so a reference that
    names a numeral ("&#189;") is masked as that numeral.
    """

    def one(match: re.Match) -> str:
        try:
            ch = chr(int(match.group(1), 16) if match.group(1) else int(match.group(2)))
        except (ValueError, OverflowError):
            return match.group(0)
        return _MARKUP_ENTITIES.get(ch, ch)

    return _CHAR_REF_RE.sub(one, text or "")


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
    return tuple(value for _match, value in _number_matches(_decode_character_references(text)))


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
    text = _decode_character_references(text)
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
    # A stacked column is a table: the layer's own evidence, asked of the text that would
    # ship. #1155 lets a row label with its values ship under a mark, and never a column
    # dump (Argon P0-2, reproduced: a three-column table under the raster reached this
    # ending as "Loans / 12.4 / 13.1 / Deposits / ...").
    if holds_a_stacked_column(kept, token):
        return "", withheld, shreds
    # The positive invariant: whatever the grammar above missed, no numeral ships. The
    # grammar is a list of the forms a number takes, and a list is what leaked three rounds
    # running; this asks the opposite question of the text that would ship, and a page
    # that fails it keeps its floor (round 2, orchestrator).
    #
    # Asked of the layer with EVERY number withheld, confirmed or not, so a confirmed
    # position never switches the check off for the rest of the page: what remains after
    # the full mask is what the grammar missed (Fable, round 2 P2-3).
    fully_masked, _all = mask_numbers(layer or "", frozenset(), token=token)
    if carries_a_numeral(withhold_shreds(fully_masked)[0], token):
        return "", withheld, shreds
    return kept, withheld, shreds


#: How many value lines page furniture fills at the top or bottom edge of a page: a date
#: over a folio ("8/23/66" over "-55-"). Measured on the Fed calibration run's 390 floor
#: pages: the leading run of such lines is 0, 1 or 2 on every page, and the trailing run
#: is 0, 1 or 2 -- or 4 and more, and every one of those is a column of values (no page
#: has 3). A longer edge run is a column (Argon, round 2: labels first, then their values
#: stacked below the last worded line).
_EDGE_FURNITURE_LINES = 2


def holds_a_stacked_column(text: str, token: str = NUMBER_WITHHELD) -> bool:
    """Whether the layer holds values stacked as a column: a table or a chart whose rows
    were lost.

    A line with no word (only numbers, masked or not, and punctuation) is a value. Alone
    between worded lines it is furniture -- a folio, or a folio split over two lines
    ("IV -" over "14"); two back to back there are a column. At the top or bottom edge
    the page's furniture sits, a date over a folio, so an edge run is a column only when
    it is longer than :data:`_EDGE_FURNITURE_LINES`. Measured on the 390 floor pages of
    the Fed calibration run: 44 hold an interior column and every one is a table or a
    chart axis; the smallest are two values ("2.0" over "1.6").
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]

    def wordless(line: str) -> bool:
        return not _WORD_RE.search(line.replace(token, ""))

    worded = [i for i, ln in enumerate(lines) if not wordless(ln)]
    if not worded:
        return False
    if worded[0] > _EDGE_FURNITURE_LINES or len(lines) - 1 - worded[-1] > _EDGE_FURNITURE_LINES:
        return True
    run = 0
    for line in lines[worded[0] + 1 : worded[-1]]:
        run = run + 1 if wordless(line) else 0
        if run >= 2:
            return True
    return False


def carries_a_numeral(text: str, token: str = NUMBER_WITHHELD) -> bool:
    """Whether any character of *text*, outside the withheld-number tokens, is a numeral:
    Unicode ``Nd``, ``No`` or ``Nl``, or any character Unicode gives a numeric value --
    the CJK numerals are letters (``Lo``) by category (Argon, round 2 re-review). The
    grammar masks only the first kind; a page carrying the second keeps its floor."""
    import unicodedata

    return any(
        ch.isnumeric() or unicodedata.category(ch) in ("Nd", "No", "Nl")
        for ch in (text or "").replace(token, "")
    )
