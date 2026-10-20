"""#1185: do the numbers a reading ships occur in the page's own characters?

A model reading of a page can be judged "faithful" and still ship ``-8`` where the page
prints ``-.8`` (2018 Tealbook p125, the case this module exists for). The page's own text
layer, when it is the document's characters and not an earlier OCR run (#1156), is an
independent witness of every number printed on it. This module compares the two; the
caller decides what the verdict licenses, from ``provenance``.

Pure: no PDF access, no model. The caller supplies the reference text(s).

What "agree" means
------------------
Every number the reading ships occurs in the page's own characters AT LEAST as often:
a multiset over the page, not a per-cell binding. A shipped number the layer has fewer
copies of is DISPUTED and named, so a mark can quote it. Its layer counterpart, when an
order alignment pairs it one-for-one, is named beside it ("text layer says -.8, reading
says -8"); otherwise it is named alone.

Why a page multiset, and what it cannot see:

* The text layer's ORDER is not the reading's: two-column pages, column-major table
  extraction and rotated tables all stream numbers in a different sequence. An earlier
  draft (commit 0d045283) compared position by position, where one inserted number turns
  every later position into a dispute. Order is used here only to NAME a counterpart.
* It is blind to a value moved to another cell or row of the same page (same multiset),
  and to a wrong number that happens to equal another number printed on the page. Binding
  numbers to cells is ``tables/native_cell_check.py``'s job (#1206) where it can locate the
  table; it cannot on a rotated page, which is the motivating case.
* Numbers the layer has and the reading does not (``reference_only``) are recorded, never
  decisive. A reading that DROPS numbers is therefore not caught here: that is issue #1218.
  Measured for #1185 on run 23319732's qwen-CHECKED pages, with socr's furniture detection
  removing page numbers and running heads: 21% of comparable pages lose a decimal, and in a
  random 25 of them only 2-3 were real losses -- the rest were formulas typed as text in
  the layer, DOIs, section numbers and footnote callouts.

The reference is ONE extraction of the page's characters. A union with socr's structured
extraction (``native_text``) was measured and rejected: over run 23319732's 3,071
qwen-CHECKED pages it cleared 5 more pages than the cleaned layer alone (284 against 289
disputed), and a per-token union can manufacture agreement where the two extractions
attach a sign or a bracket differently (Argon, plan review): ``-0.8`` in one and a
detached ``0.8`` in the other would pass a reading that dropped the minus.

Normalisation, applied to both sides alike (recorded on the record):

1. socr's shared numeric token predicate and normaliser
   (``native_verifier.is_numeric_token`` / ``_normalize_numeric_token``): Unicode minus
   and dashes fold to ``-``, ``-.8`` and ``-0.8`` are one value, thousands separators,
   ``%`` and ``$`` are dropped. No float cast: ``2.10`` and ``2.1`` differ;
2. brackets are part of the token: ``(0.014)`` is a standard error, never ``0.014``;
3. a LaTeX fraction contributes its parts; other LaTeX presentation (``$``, braces,
   commands, ``\\%``) is removed, so ``$(-2.946)$`` is the number ``(-2.946)``;
4. a footnote callout after a closing bracket is split off: ``(2020).12`` is ``(2020)``
   and ``12``;
5. a comma is a thousands separator only before exactly three digits, so ``(0,1)`` is an
   interval and no number;
6. a sentence's full stop and unbalanced brackets around a number are punctuation.

Reading side only (``reading_numbers``): mathematics other than a lone number is dropped.
Checking formulas is the equation lane's job, and math fonts are where text layers garble
(PUA glyphs, brackets drawn as ``p``/``q``); comparing them here measured as noise.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

from socr.tables.native_verifier import _normalize_numeric_token, is_numeric_token

#: Where the reference came from. The caller decides what each licenses: only
#: ``OWN_CHARACTERS`` and ``PRODUCED_READING`` (an independent reader, #1208) may support
#: CHECKED; agreement with a ``PRIOR_OCR`` layer never does.
PROVENANCE_OWN_CHARACTERS = "own_characters"
PROVENANCE_PRIOR_OCR = "prior_ocr"
PROVENANCE_PRODUCED_READING = "produced_reading"
PROVENANCES = (PROVENANCE_OWN_CHARACTERS, PROVENANCE_PRIOR_OCR, PROVENANCE_PRODUCED_READING)

NORMALISATION = (
    "native_verifier_token_normalisation",
    "brackets_kept_as_part_of_the_token",
    "latex_fraction_expanded_to_parts",
    "latex_presentation_removed",
    "footnote_callout_split_after_bracket",
    "comma_is_thousands_separator_only_before_three_digits",
    "sentence_punctuation_and_unbalanced_brackets_stripped",
    "significance_stars_and_bracket_spacing_removed",
    "reading_equation_tags_and_superscripts_dropped",
    "reference_control_characters_and_word_digit_joins_split",
    "running_text_brackets_and_year_callouts_matched_to_unused_layer_numbers",
    "reading_mathematics_outside_tables_dropped_unless_a_lone_number",
    "page_multiset_order_names_counterparts_only",
)

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_FRACTION_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
_LATEX_PUNCT_RE = re.compile(r"\\[,;:! ]|[${}]")
_MATH_RE = re.compile(
    r"\$\$.+?\$\$|\$[^$\n]+?\$|\\\(.+?\\\)|\\\[.+?\\\]|\\begin\{(\w+\*?)\}.+?\\end\{\1\}", re.S
)
#: what marks a number as a REFERENCE to something rather than a value: an equation tag,
#: a superscript (a footnote callout ``${}^{21}$``, an exponent). Reading side only.
#: A power (``10^{15}``, ``2^8``) is a different number from its base, and a subscript
#: (``\gamma_4``) is a name: both go whole. The base is a WHOLE integer -- not preceded
#: by a digit or a point -- so a starred cell ``-0.04^{***}`` loses its stars, never its
#: last digit (a unit test caught the ``4`` being eaten as a power base).
_REFERENCE_MARK_RE = re.compile(
    r"\\tag\*?\{[^{}]*\}|\\textsuperscript\{[^{}]*\}"
    r"|(?<![\d.])\d+\^\{\s*-?\d+\s*\}|(?<![\d.])\d+\^-?\d+"
    r"|\^\{[^{}]*\}|\^\w+|_\{[^{}]*\}|_\w+"
)
#: Reference side only: control characters (a damaged layer's undecodable glyphs) and the
#: places a layer runs a word into a number (``Sect.3.1``, ``a10%increase``, ``(×200)``).
#: Splitting there can only ADD reference tokens, which never creates a dispute.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\u0300-\u036f\u20d0-\u20ff]")
#: a minus the layer set apart from its number (``− 0.077``): joined, so a reading that
#: DROPPED that minus is disputed rather than matched by the bare ``0.077``
#: The typographic minus (U+2212) is a sign wherever it stands; an ASCII hyphen set apart
#: from a number is usually a range (``5 - 8``), so it joins only where a cell starts: at a
#: line start or after an opening bracket (Argon, diff review).
_DETACHED_MINUS_RE = re.compile(
    r"(?m)()(\u2212)[ \t]+(?=\.?\d)|(^[ \t]*|[(\[][ \t]*)(-)[ \t]+(?=\.?\d)"
)
_WORD_DIGIT_RE = re.compile(
    r"(?<=[^\W\d_])\.?(?=[\d(\[$])|(?<=[^\W\d_][-\u2212])(?=\d)|(?<=[\d%)\]])[.:;]?(?=[^\W\d_])|(?<=[^\d\s]),(?=\d)"
    r"|,(?=[^\W\d_])|(?<=[)\]])(?=\d)|(?<=%)\.(?=\d)"
    r"|[\u00d7=<>\u2243\u2248\u2264\u2265]"
)
#: A dollar sign in front of an amount (``-$8 billion``, ``$1.4trn``) is currency, and must
#: not open a maths span that swallows the next one (Argon, diff review).
_CURRENCY_RE = re.compile(r"(?<![\\$])\$(?=[-\u2212]?\d[\d,.]*(?:[A-Za-z%]|\s+[A-Za-z]))")
_MATH_DELIMITERS = ("\\(", "\\)", "\\[", "\\]")
#: significance stars, the way a table cell writes them in or out of maths: ``^{***}``,
#: ``^*``, ``\ast``. Presentation, not part of the number (Argon, plan review).
_STARS_RE = re.compile(r"\^\{\s*(?:\*|\\ast\s*)+\}|\^(?:\*|\\ast)+|\\ast|\u2217+")
#: a bracket set apart from its number by spaces, as some layers extract ``( 0.04 )``
_BRACKET_SPACE_RE = re.compile(r"([(\[])\s+(?=[-\u2212.\d])|(?<=[\d%])\s+([)\]])")
_SPLIT_RE = re.compile(r"[\s|]+")
_CALLOUT_RE = re.compile(r"(?<=[)\]])([.,;:])(?=\d)")
_BAD_COMMA_RE = re.compile(r"\d,(?!\d{3}(?:\D|$))")
_EDGE = "*_`\"'\u201c\u201d\u2018\u2019,;:!?"
_OPEN = {"(": ")", "[": "]"}
_CLOSE = {")": "(", "]": "["}


def _clean(raw: str) -> tuple[str, str]:
    """``raw`` without surrounding punctuation, plus the bracket pair it was written in."""
    token, previous = raw, None
    while token != previous:
        previous = token
        token = token.strip(_EDGE)
        if token.endswith(".") and not token.endswith(".."):
            token = token[:-1]
        if token[:1] in _OPEN and token.count(token[0]) > token.count(_OPEN[token[0]]):
            token = token[1:]
        if token[-1:] in _CLOSE and token.count(token[-1]) > token.count(_CLOSE[token[-1]]):
            token = token[:-1]
    pair = token[0] + token[-1] if len(token) > 2 else ""
    if pair in ("()", "[]"):
        return token[1:-1], pair
    return token, ""


def _strip_latex(text: str) -> str:
    text = text.replace("\\%", "%").replace("\\$", " ")
    return _LATEX_PUNCT_RE.sub("", _LATEX_COMMAND_RE.sub(" ", text))


def _lone_number_or_nothing(match: re.Match) -> str:
    inner = _strip_latex(_STARS_RE.sub("", match.group(0)))
    for delimiter in _MATH_DELIMITERS:
        inner = inner.replace(delimiter, "")
    parts = inner.split()
    if len(parts) == 1:
        core, _ = _clean(parts[0])
        if core and is_numeric_token(core):
            return f" {parts[0]} "
    return " "


def _is_table_row(line: str) -> bool:
    """A markdown pipe row, or a line carrying HTML table cells."""
    stripped = line.lstrip()
    return stripped.startswith("|") or "<td" in stripped or "<th" in stripped


def _drop_mathematics_outside_tables(text: str) -> str:
    """Drop mathematics from each run of non-table lines, as a whole: a display block
    (``$$ ... $$``) spans lines, so a line-by-line pass would never see both delimiters."""
    out: list[str] = []
    prose: list[str] = []
    for line in text.split("\n"):
        if _is_table_row(line):
            if prose:
                out.append(_MATH_RE.sub(_lone_number_or_nothing, "\n".join(prose)))
                prose = []
            out.append(line)
        else:
            prose.append(line)
    if prose:
        out.append(_MATH_RE.sub(_lone_number_or_nothing, "\n".join(prose)))
    return "\n".join(out)


def numeric_tokens(text: str, *, reading: bool = False) -> tuple[tuple[str, str], ...]:
    """``(normalised, as written)`` for every number in ``text``, in order.

    ``reading=True`` applies the reading-side rules (mathematics outside tables and
    reference marks dropped); otherwise the reference-side splitting applies.
    """
    if not text:
        return ()
    if reading:
        text = _REFERENCE_MARK_RE.sub(" ", text)
    else:
        text = _WORD_DIGIT_RE.sub(" ", _CONTROL_RE.sub(" ", text))
        text = _DETACHED_MINUS_RE.sub(
            lambda m: (m.group(1) or m.group(3) or "") + (m.group(2) or m.group(4)), text
        )
    text = _COMMENT_RE.sub(" ", text)
    text = _IMAGE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _TAG_RE.sub(" ", text)
    text = _BRACKET_SPACE_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _FRACTION_RE.sub(lambda m: f" {m.group(1)} {m.group(2)} ", text)
    if reading:
        text = _CURRENCY_RE.sub("", text)
        # A table row keeps its mathematics: a cell is a number however it is typeset
        # (``$-0.04^{***}$``), and table cells are where a wrong number is cited.
        text = _drop_mathematics_outside_tables(text)
    text = _STARS_RE.sub("", text)
    text = _CALLOUT_RE.sub(r"\1 ", _strip_latex(text))
    out: list[tuple[str, str]] = []
    for raw in _SPLIT_RE.split(text):
        if not raw:
            continue
        core, pair = _clean(raw)
        if not core or _BAD_COMMA_RE.search(core) or not is_numeric_token(core):
            continue
        value = _normalize_numeric_token(core)
        if not value:
            continue
        if pair:
            out.append((f"{pair[0]}{value}{pair[1]}", f"{pair[0]}{core}{pair[1]}"))
        else:
            out.append((value, core))
    return tuple(out)


@dataclass(frozen=True)
class AgreementRecord:
    """What the comparison found. Persisted, so a resume and the meter can read it."""

    provenance: str
    #: How many numbers the reading ships. 0 means the agreement is vacuous.
    compared: int
    #: One entry per shipped number the reference does not hold: ``(reference, reading)``
    #: as written, the reference side ``None`` when no counterpart could be paired.
    disputed: tuple[tuple[str | None, str], ...] = ()
    #: How many of the reference's numbers the reading does not ship. Recorded only.
    reference_only: int = 0
    normalisation: tuple[str, ...] = field(default=NORMALISATION)

    @property
    def agreed(self) -> bool:
        return self.compared > 0 and not self.disputed

    def to_dict(self) -> dict:
        return {
            "provenance": self.provenance,
            "agreed": self.agreed,
            "compared": self.compared,
            "disputed": [list(item) for item in self.disputed],
            "reference_only": self.reference_only,
            "normalisation": list(self.normalisation),
        }


def _reading_tokens(reading: str) -> list[tuple[str, str, bool]]:
    """The reading's numbers in order, each tagged "written in a table row"."""
    out: list[tuple[str, str, bool]] = []
    block: list[str] = []
    in_table = False
    for line in (reading or "").split("\n") + [None]:
        row = line is not None and _is_table_row(line)
        if line is None or row != in_table:
            if block:
                tokens = numeric_tokens("\n".join(block), reading=True)
                out.extend((value, written, in_table) for value, written in tokens)
            block, in_table = [], row
        if line is not None:
            block.append(line)
    return out


#: A year in running text with its footnote callout run on by the layer: ``2009.9`` for
#: ``2009.`` + callout ``9`` (Nakamura-Steinsson p15). Callouts are one or two digits.
_YEAR_RE = re.compile(r"19\d\d|20[0-3]\d")
_CALLOUT_SUFFIX_RE = re.compile(r"\.?\d{1,2}")


def _same_number_in_prose(reading: str, layer: str) -> bool:
    """Do these two tokens print the same number, as running text (never a table cell)?

    Two relations, both measured on run 23319732's disputed pages and both about how a
    layer or a reading WRITES a number in a sentence, not about its value:

    * brackets on one side only: a citation ``(2017)`` against a layer's ``2017``, a list
      ``1.`` against ``(1)``, an equation reference ``(14)`` whose brackets a math font
      drew as ``ð``/``Þ``. In a table a bracket separates a standard error from a
      coefficient, so table cells never take this relation;
    * a year with its footnote callout run on: ``2009`` against ``2009.9`` or ``20055``.
    """
    bare = reading.strip("()[]")
    # Integers only: on a decimal a bracket is a standard error, never punctuation.
    if bare.lstrip("-").isdigit() and bare == layer.strip("()[]"):
        return True
    return (
        bool(_YEAR_RE.fullmatch(bare))
        and layer.startswith(bare)
        and bool(_CALLOUT_SUFFIX_RE.fullmatch(layer[len(bare) :]))
    )


def compare_numbers(reference: str, reading: str, *, provenance: str) -> AgreementRecord:
    """Compare the numbers ``reading`` ships with those in ``reference``.

    ``reference`` is the page's text in stream order (its order only NAMES counterparts).
    ``reading`` is the text under test with the caller's own lines already removed.
    """
    if provenance not in PROVENANCES:
        raise ValueError(f"unknown provenance: {provenance!r}")
    order = numeric_tokens(reference)
    tagged = _reading_tokens(reading)
    reading_tokens = [(value, written) for value, written, _ in tagged]
    held = Counter(value for value, _ in order)
    shipped = Counter(value for value, _ in reading_tokens)
    surplus = shipped - held
    unused = held - shipped  # layer numbers no shipped number claimed

    a = [value for value, _ in order]
    b = [value for value, _ in reading_tokens]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    counterpart: dict[int, int] = {}
    matched: set[int] = set()
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            matched.update(range(j1, j2))
        elif op == "replace" and i2 - i1 == j2 - j1:
            counterpart.update({j1 + k: i1 + k for k in range(i2 - i1)})
    # Name each surplus occurrence, preferring an occurrence the alignment did not match.
    left = Counter(surplus)
    named: list[tuple[int, str | None, str]] = []
    for j in sorted(range(len(reading_tokens)), key=lambda j: (j in matched, j)):
        value, written = reading_tokens[j]
        if left[value] <= 0:
            continue
        left[value] -= 1
        if not tagged[j][2]:
            # Running text: the number may be printed in the layer in another writing.
            # Only an UNUSED layer number can stand for it, so nothing is counted twice.
            twin = next(
                (u for u in unused if unused[u] > 0 and _same_number_in_prose(value, u)), None
            )
            if twin is not None:
                unused[twin] -= 1
                continue
        i = counterpart.get(j)
        named.append((j, order[i][1] if i is not None else None, written))
    return AgreementRecord(
        provenance=provenance,
        compared=len(reading_tokens),
        disputed=tuple((ref, read) for _, ref, read in sorted(named)),
        reference_only=sum(unused.values()),
    )


def describe_disputes(record: AgreementRecord) -> str:
    """The disputed numbers as one mark reason: every one is named, none is dropped.

    Joined with `` / `` because the mark grammar turns ``;`` into ``,`` (``clean_reason``),
    which would run one dispute into the next.
    """
    parts = [
        f"text layer says {ref}, reading says {read}"
        if ref is not None
        else f"reading says {read}, not in the text layer"
        for ref, read in record.disputed
    ]
    return " / ".join(parts)
