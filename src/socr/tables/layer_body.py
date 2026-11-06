"""#1218: which numbers of a page's own text layer is the reading owed?

A model reading can drop numbers the page prints. Counting layer-minus-reading over every
layer number is mostly noise (running heads, footnote callouts, formulas typed as text,
chart labels), so the layer's numbers are first selected by GEOMETRY, independent of any
reading. Only the numeric TAIL of a LABEL ROW is owed to the reading: a line that is a
label (words, one with >= 2 letters) followed by at least ``LABEL_ROW_MIN_NUMBERS`` words
that are all numbers, with no math operator on the line and not a constant-step axis (a
run of calendar years, or a tail at a chart or image box).

The pure core takes ``get_text("words")`` tuples; :func:`drop_reference_from_page` is the
one adapter that reads a PyMuPDF page. Ported from the precision team's
``scratch/army/precision/scripts/frame2.py`` (line building, rotation, overprint dedupe,
height ratio); the document's repeated word slots
(``born_digital._repeated_text_slots`` + ``_folio_slots``) replace its corpus-level band.

Words are removed before the row split when they are furniture (a repeated slot or a page
number), a superscript (PyMuPDF's span flag, else the word's height against its line), or
inside a chart or image box that no detected table overlaps.

Residual, not handled: a callout glued into a number by the layer (``1431`` for ``143`` +
superscript ``1``) is one word at one height; it needs character boxes.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from socr.tables.reading_agreement import numeric_tokens

FURNITURE = "FURNITURE"
FOOTNOTE = "FOOTNOTE"
MARGIN = "MARGIN"
BODY = "BODY"
CLASSES = (BODY, MARGIN, FOOTNOTE, FURNITURE)

#: frame2.py ``MIN_CHARS_FOR_ORIENTATION``: a word shorter than this says nothing about
#: which way the text runs ("1", "-" are as tall as wide in either orientation).
MIN_CHARS_FOR_ORIENTATION = 4

#: frame2.py ``ROTATED_SHARE``: a page is rotated when more than half of its multi-character
#: words are taller than wide.
ROTATED_SHARE = 0.5

#: frame2.py ``MIN_DIRECTION_VOTES``: how many vertically-adjacent word pairs must agree
#: before the reading direction of a rotated page (top-to-bottom or bottom-to-top) is trusted.
MIN_DIRECTION_VOTES = 3

#: frame2.py ``LINE_CENTRE_TOL``: a word joins the current line when its centre is within
#: this fraction of the smaller of the two word heights from the line's mean centre.
LINE_CENTRE_TOL = 0.5

#: frame2.py ``DUP_OVERPRINT_TOL_HEIGHTS``: a word printed twice at nearly the same place
#: (fake bold, overprint) is one word; "nearly" is this fraction of its height.
DUP_OVERPRINT_TOL_HEIGHTS = 0.25

#: frame2.py ``FOOT_RATIO``: a word box height below this fraction of the line median is a
#: superscript or footnote mark (decides only when no span flags are supplied; recorded on
#: every number either way so the two signals can be compared).
FOOT_RATIO = 0.75

#: frame2.py ``MIN_WORDS_FOR_RATIO``: a line median height means nothing with fewer words.
MIN_WORDS_FOR_RATIO = 3

#: frame2.py ``AXIS_MIN_RUN``: two numbers always make a "constant step"; three show it is
#: constant.
AXIS_MIN_RUN = 3

#: Relative tolerance of "the same step" between consecutive numbers (frame2.py ``1e-9``):
#: float noise only.
_STEP_REL_TOL = 1e-9

#: Decimal places a slot position is rounded to: ``born_digital._repeated_text_slots`` rounds
#: word origins to 0.1 pt, and the predicate must key a word the same way.
_SLOT_DECIMALS = 1

#: U+2212 only: a typographic minus a layer sets apart from its number is a sign; an ASCII
#: hyphen apart from a number is usually a range (reading_agreement ``_DETACHED_MINUS_RE``).
_DETACHED_MINUS = "−"

#: PyMuPDF span flag bit 0: superscript.
SPAN_SUPERSCRIPT_FLAG = 1

#: #1218 v3: a LABEL ROW is a line with a label (words, at least one with >= 2 letters)
#: followed by a tail of at least this many words that are all numbers. Only tail numbers
#: are owed to the reading. Derived, not chosen: the smallest tail length at which NO line
#: of the Fed native-transcript success pages (pure prose) qualifies as a label row.
#: Measured 2026-11-06 over 5,423 pages of 61 transcript documents
#: (``scratch/army/1218/data/summary.txt``, "G TABLE"): lines qualifying with tail >= 1: 1,085,
#: >= 2: 94, >= 3: 0 (and 0 for every larger tail). So 3.
LABEL_ROW_MIN_NUMBERS = 3

#: Characters that make a line a formula, never a label row.
LABEL_ROW_OPERATORS = "=\u2248\u2264\u2265+\u2211\u222b"

#: significance stars and daggers glued to a number (``0.5**``, ``12\u2020``)
_STARS_RE = re.compile(r"[*\u2217\u2020\u2021\u00a7]+")
_TAIL_EDGE = ",;"
_LETTERS_RE = re.compile(r"[^\W\d_]")
_TWO_LETTERS_RE = re.compile(r"[^\W\d_]{2}")

#: A decimal point the layer did not extract (a vector glyph, or ``:`` for ``.``) leaves a
#: number as two words: ``0`` ``004`` for ``0.004``, ``-24`` ``9*`` for ``-24.9*``. Inside a
#: label row's tail, such a pair is the SAME number when its gap is below this fraction of
#: the tail's LARGEST gap AND at least ``DECIMAL_FRAGMENT_MIN_PAIRS`` pairs of the tail show
#: it (the cells of a table column repeat the pattern).
#:
#: Derived (#1218, ``scratch/army/1218/data/fragment_hist.txt``): over 10,642 integer+digits
#: pairs in the label-row tails of the calibration run (23319732) and the Fed run (aebc444c),
#: the evidence that a pair is ONE number is that its joined decimal occurs in the model's
#: reading (171 pairs). That support is 97 of 288 pairs below a ratio of 0.06, 14 of 436 in
#: 0.06-0.10 and zero from 0.10 to 0.54; 0.06 is the cut that predicts it best (max F1). The
#: pair counts themselves have no empty valley (closely set integer cells are common), so the
#: support, not the counts, sets the cut. Two pairs: at the cut, tails with exactly one close
#: pair were supported 1 of 42 times, tails with two or more 96 of 246.
#: A wrong join usually adds a false drop (the merged decimal is absent); it can hide a real one when
#: the reading ships the merged decimal. #1185 may dispute that decimal, but not if the page prints
#: the same value elsewhere or the reading writes it inside mathematics that check excludes.
DECIMAL_GAP_FRACTION = 0.06
DECIMAL_FRAGMENT_MIN_PAIRS = 2

#: A calendar year, with the footnote callout digit a layer glues on (``2008`` + ``1`` is
#: ``20081``): a constant-step run of these is a time axis or a header of years, never data.
_CALENDAR_YEAR_RE = re.compile(r"(?:19|20)\d\d[1-9]?")
_FRAGMENT_LEFT_RE = re.compile(r"^[-\u2212]?\d+$")
_FRAGMENT_RIGHT_RE = re.compile(r"^:?(\d+)([*\u2217\u2020\u2021\u00a7]*)$")
#: ``0,332``: a thousands group never starts with 0, so a leading ``0,`` is a decimal comma
#: (Andresson-Kilman p24). Only after a zero integer part: ``1,332`` stays 1332.
_ZERO_COMMA_RE = re.compile(r"^([-\u2212]?0),(\d+)(?=[*\u2217\u2020\u2021\u00a7]*$)")

Predicate = Callable[[Sequence], bool]


@dataclass(frozen=True)
class LayerNumber:
    """One numeric token of the page's own text layer, with its geometric class."""

    written: str
    value: str
    cls: str
    line_index: int | None
    line_text: str
    #: id of the label row whose numeric tail holds this number (page-unique), or None;
    #: only tail numbers are owed to the reading.
    row: int | None = None
    row_label: str = ""


@dataclass(frozen=True)
class _Word:
    """A word in line coordinates: ``u`` along the reading direction, ``v`` across it."""

    u0: float
    u1: float
    v0: float
    v1: float
    text: str
    furniture: bool
    superscript: bool | None
    #: the word's own box in page coordinates (unrotated), for box-proximity tests
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def slot_predicate(repeated_slots, folio_slots) -> Predicate:
    """Furniture test over ``BornDigitalDetector._repeated_text_slots`` / ``_folio_slots``.

    A word is furniture when its (text, x0, y0) slot is repeated, or its (x0, y0) slot is a
    folio slot and it is a number (a folio's digits change on every page).
    """

    def is_furniture(word: Sequence) -> bool:
        text = str(word[4])
        x, y = round(word[0], _SLOT_DECIMALS), round(word[1], _SLOT_DECIMALS)
        if (text.casefold(), x, y) in repeated_slots:
            return True
        return text.strip().isdigit() and (x, y) in folio_slots

    return is_furniture


def superscript_boxes(page_dict: dict) -> list[tuple[float, float, float, float]]:
    """Boxes of the spans PyMuPDF flags superscript, from ``page.get_text("dict")``."""
    boxes = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if int(span.get("flags", 0)) & SPAN_SUPERSCRIPT_FLAG:
                    boxes.append(tuple(span["bbox"]))
    return boxes


def _in_any(word: Sequence, boxes: Sequence[Sequence[float]]) -> bool:
    cx, cy = (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def _orient_share(words: Sequence[Sequence]) -> tuple[float, int]:
    multi = [w for w in words if len(str(w[4])) >= MIN_CHARS_FOR_ORIENTATION]
    if not multi:
        return 0.0, 0
    return sum((w[3] - w[1]) > (w[2] - w[0]) for w in multi) / len(multi), len(multi)


def _reading_sign(words: Sequence[Sequence]) -> int | None:
    """+1 when a rotated page's lines read top-to-bottom, -1 bottom-to-top, None if unclear."""
    votes: Counter = Counter()
    rot = [
        w
        for w in words
        if (w[3] - w[1]) > (w[2] - w[0]) and len(str(w[4])) >= MIN_CHARS_FOR_ORIENTATION
    ]
    for a, b in zip(rot, rot[1:]):
        ax, bx = (a[0] + a[2]) / 2, (b[0] + b[2]) / 2
        if a[0] <= bx <= a[2] or b[0] <= ax <= b[2]:
            dy = (b[1] + b[3]) / 2 - (a[1] + a[3]) / 2
            if dy:
                votes[1 if dy > 0 else -1] += 1
    if sum(votes.values()) < MIN_DIRECTION_VOTES:
        return None
    return 1 if votes[1] >= votes[-1] else -1


def _sign_from_rotation(rotation: int) -> int:
    """Reading direction in unrotated coordinates implied by the page's /Rotate.

    /Rotate 90 displays unrotated (x, y) at (H - y, x): a line that reads left to right on
    screen has its screen x rising, so its unrotated y FALLS: it runs up the unrotated page
    (-1). /Rotate 270 is the reverse (+1). frame2.py's default for "no evidence" (-1) stays
    for the other values.
    """
    return 1 if rotation % 360 == 270 else -1


def _dedupe_overprint(words: Sequence[Sequence]) -> list[Sequence]:
    kept: list[Sequence] = []
    for w in words:
        tol = DUP_OVERPRINT_TOL_HEIGHTS * (w[3] - w[1])
        if not any(
            k[4] == w[4] and abs(k[0] - w[0]) <= tol and abs(k[1] - w[1]) <= tol for k in kept
        ):
            kept.append(w)
    return kept


def _to_uv(w: Sequence, rotated: bool, sign: int, furniture: bool, sup: bool | None) -> _Word:
    x0, y0, x1, y1, text = w[:5]
    box = (x0, y0, x1, y1)
    if not rotated:
        return _Word(x0, x1, y0, y1, str(text), furniture, sup, box)
    if sign > 0:
        return _Word(y0, y1, x0, x1, str(text), furniture, sup, box)
    return _Word(-y1, -y0, x0, x1, str(text), furniture, sup, box)


def _build_lines(words: list[_Word]) -> list[list[_Word]]:
    """Group words into lines by centre across the reading direction; words sorted along it."""
    order = sorted(words, key=lambda w: (w.v0 + w.v1) / 2)
    lines: list[dict] = []
    for w in order:
        c = (w.v0 + w.v1) / 2
        h = w.v1 - w.v0
        if lines:
            ln = lines[-1]
            mean_c = sum(ln["cs"]) / len(ln["cs"])
            median_h = statistics.median(ln["hs"])
            if abs(c - mean_c) <= LINE_CENTRE_TOL * min(median_h, h):
                ln["words"].append(w)
                ln["cs"].append(c)
                ln["hs"].append(h)
                continue
        lines.append({"words": [w], "cs": [c], "hs": [h]})
    return [sorted(ln["words"], key=lambda w: w.u0) for ln in lines]


def _repair_decimal_comma(text: str) -> str:
    return _ZERO_COMMA_RE.sub(r"\1.\2", text)


def _merge_fragments(
    tail: list[tuple[str, _Word]],
) -> tuple[list[tuple[str, _Word]], dict[int, str], set[int]]:
    """Join decimal fragments in ``tail``: (items, {left word id: merged text}, consumed ids)."""
    gaps = [b[1].u0 - a[1].u1 for a, b in zip(tail, tail[1:])]
    if len(gaps) < 2:
        return tail, {}, set()
    cut = DECIMAL_GAP_FRACTION * max(gaps)
    pairs = [
        k
        for k, gap in enumerate(gaps)
        if 0 <= gap < cut
        and _FRAGMENT_LEFT_RE.match(tail[k][0])
        and _FRAGMENT_RIGHT_RE.match(tail[k + 1][0])
    ]
    # a word cannot be the right half of one pair and the left half of the next: walking left
    # to right, a candidate that overlaps the pair just KEPT is dropped (one that overlaps a
    # candidate which itself lost the conflict is not)
    kept_pairs: list[int] = []
    for k in pairs:
        if not kept_pairs or kept_pairs[-1] != k - 1:
            kept_pairs.append(k)
    pairs = kept_pairs
    if len(pairs) < DECIMAL_FRAGMENT_MIN_PAIRS:
        return tail, {}, set()
    merged: dict[int, str] = {}
    consumed: set[int] = set()
    items: list[tuple[str, _Word]] = []
    skip = {k + 1 for k in pairs}
    for k, (text, w) in enumerate(tail):
        if k in skip:
            continue
        if k in pairs:
            right = _FRAGMENT_RIGHT_RE.match(tail[k + 1][0])
            text = f"{text}.{right.group(1)}{right.group(2)}"
            merged[id(w)] = text
            consumed.add(id(tail[k + 1][1]))
        items.append((text, w))
    return items, merged, consumed


def _tail_number(text: str) -> bool:
    """A word that is one number, optionally starred and followed by ``,`` or ``;``."""
    core = _STARS_RE.sub("", text).strip(_TAIL_EDGE)
    return bool(core) and not _LETTERS_RE.search(core) and len(numeric_tokens(core)) == 1


def _tail_is_axis(tail: Sequence[str]) -> bool:
    """Absolute values with a constant nonzero step (signs are often missing from a layer)."""
    if len(tail) < AXIS_MIN_RUN:
        return False
    try:
        v = [
            abs(float(_STARS_RE.sub("", t).strip(_TAIL_EDGE + "()[]$%").replace(",", "")))
            for t in tail
        ]
    except ValueError:
        return False
    d = v[1] - v[0]
    return d != 0 and all(
        abs((b - a) - d) <= _STEP_REL_TOL * max(1.0, abs(d)) for a, b in zip(v, v[1:])
    )


def _near_boxes(
    words: Sequence[_Word],
    chart_boxes: Sequence[Sequence[float]],
    image_boxes: Sequence[Sequence[float]] = (),
) -> bool:
    """Does any of ``words`` touch a chart box, or an image box grown by one text height?

    ``chart_boxes`` (``chart_region_bboxes``) arrive ALREADY grown by the page's own row gap
    so that tick labels fall inside them: they are used as given. An image box is the raster's
    raw extent, its tick labels sit just outside it, so it is grown by the tail's own median
    text height (relative to the type size, not an absolute length). Growing a chart box too
    would count a data row near a chart as the chart's axis.
    """
    if not chart_boxes and not image_boxes:
        return False
    margin = statistics.median(w.v1 - w.v0 for w in words)
    grown = [*((b, 0.0) for b in chart_boxes), *((b, margin) for b in image_boxes)]
    return any(
        w.box[0] <= b[2] + m
        and w.box[2] >= b[0] - m
        and w.box[1] <= b[3] + m
        and w.box[3] >= b[1] - m
        for w in words
        for b, m in grown
    )


def _is_axis_not_data(
    tail: Sequence[str], words: Sequence[_Word], chart_boxes, image_boxes=()
) -> bool:
    """A constant-step tail that is a time axis or header, or a plot's tick labels.

    ``China 10 20 30`` is a data row whose values happen to step evenly (Astra, PR #1277),
    so a constant step alone excludes nothing. The step must come with evidence: every value
    is a calendar year (a time axis or a column header of years), or the tail sits at a chart
    or image box (:func:`_near_boxes`).
    """
    if not _tail_is_axis(tail):
        return False
    years = all(_CALENDAR_YEAR_RE.fullmatch(_STARS_RE.sub("", t).strip(_TAIL_EDGE)) for t in tail)
    return years or _near_boxes(words, chart_boxes, image_boxes)


def _label_row(
    words: Sequence[_Word],
    min_numbers: int,
    axis_boxes: Sequence[Sequence[float]] = (),
    image_boxes: Sequence[Sequence[float]] = (),
) -> tuple[str, set[int], dict[int, str], set[int]] | None:
    """(label, tail word ids, tail texts by word id, consumed ids) if ``words`` form a row.

    The tail texts are the repaired ones (decimal fragments joined, ``0,332`` as ``0.332``);
    consumed words are the right halves of joined fragments.

    ``words`` are the line's words with furniture and superscripts already removed. A
    typographic minus the layer set apart from its number joins it.
    """
    items: list[tuple[str, _Word]] = []
    for w in words:
        if items and items[-1][0] == _DETACHED_MINUS:
            items[-1] = (_DETACHED_MINUS + w.text, w)
        else:
            items.append((w.text, w))
    if any(ch in text for text, _ in items for ch in LABEL_ROW_OPERATORS):
        return None
    split = len(items)
    while split > 0 and _tail_number(items[split - 1][0]):
        split -= 1
    label, tail = items[:split], items[split:]
    tail, merged, consumed = _merge_fragments(tail)
    tail = [(_repair_decimal_comma(text), w) for text, w in tail]
    if len(tail) < min_numbers or not label:
        return None
    if not any(_TWO_LETTERS_RE.search(text) for text, _ in label):
        return None
    if _is_axis_not_data([text for text, _ in tail], [w for _, w in tail], axis_boxes, image_boxes):
        return None
    return (
        " ".join(text for text, _ in label),
        {id(w) for _, w in tail},
        {id(w): text for text, w in tail},
        consumed,
    )


def classify_layer_numbers(
    words: Sequence[Sequence],
    page_rect: Sequence[float],
    is_furniture: Predicate | None,
    *,
    rotation: int = 0,
    superscripts: Sequence[Sequence[float]] | None = None,
    min_numbers: int = LABEL_ROW_MIN_NUMBERS,
    excluded_boxes: Sequence[Sequence[float]] = (),
    image_boxes: Sequence[Sequence[float]] = (),
) -> list[LayerNumber]:
    """Every numeric token of ``words`` with its class and run; lines in reading order.

    ``words`` are ``page.get_text("words")`` tuples ``(x0, y0, x1, y1, text, ...)``.
    ``page_rect`` is accepted for symmetry with the page and is not used (no band by
    position). ``excluded_boxes`` are chart-region boxes and ``image_boxes`` raster image boxes,
    in the words' own coordinates; words centred in either are ignored. Only an image box is
    grown when deciding whether a tail is its axis (see :func:`_near_boxes`). ``rotation`` is the page's /Rotate; it decides the reading direction of a
    sideways page only where the words themselves do not. ``superscripts`` are the boxes
    from :func:`superscript_boxes`; when given, FOOTNOTE follows the span flag, otherwise
    the height ratio.
    """
    # Words inside a chart region or a raster image are chart or figure labels, not the
    # page's body: removed before anything else. Boxes share the words' coordinate convention
    # (``get_text("words")``, unrotated), as do ``chart_region_bboxes`` and ``get_image_info``.
    out_boxes = [*excluded_boxes, *image_boxes]
    kept = [w for w in words if str(w[4]).strip() and not (out_boxes and _in_any(w, out_boxes))]
    flags = {id(w): bool(is_furniture is not None and is_furniture(w)) for w in kept}
    sups = {id(w): (_in_any(w, superscripts) if superscripts is not None else None) for w in kept}
    kept = _dedupe_overprint(kept)
    share, n_multi = _orient_share(kept)
    rotated = (n_multi > 0 and share > ROTATED_SHARE) or (n_multi == 0 and rotation % 180 == 90)
    sign = 1
    if rotated:
        sign = _reading_sign(kept) or _sign_from_rotation(rotation)
    cross: list[Sequence] = []
    if rotated:
        # Upright words on a sideways page (page labels like "I-31") are not on its line grid.
        cross = [
            w
            for w in kept
            if len(str(w[4])) >= MIN_CHARS_FOR_ORIENTATION and (w[2] - w[0]) > (w[3] - w[1])
        ]
        cross_ids = {id(w) for w in cross}
        kept = [w for w in kept if id(w) not in cross_ids]
    lines = _build_lines([_to_uv(w, rotated, sign, flags[id(w)], sups[id(w)]) for w in kept])
    entries: list[dict] = []
    row_id = -1
    for i, line in enumerate(lines):
        text = " ".join(w.text for w in line)
        median_h = statistics.median(w.v1 - w.v0 for w in line)
        owed = [
            w
            for w in line
            if not w.furniture
            and not (
                w.superscript
                if w.superscript is not None
                else len(line) >= MIN_WORDS_FOR_RATIO and (w.v1 - w.v0) / median_h < FOOT_RATIO
            )
        ]
        row = _label_row(owed, min_numbers, excluded_boxes, image_boxes)
        if row is not None:
            row_id += 1
        carry = ""
        for w in line:
            in_tail = row is not None and id(w) in row[1]
            if row is not None and id(w) in row[3]:
                continue  # right half of a joined decimal fragment
            toks = numeric_tokens(row[2][id(w)] if in_tail else carry + w.text)
            carry = w.text if w.text == _DETACHED_MINUS else ""
            if not toks:
                continue
            ratio = (w.v1 - w.v0) / median_h if len(line) >= MIN_WORDS_FOR_RATIO else 1.0
            sup = w.superscript if w.superscript is not None else ratio < FOOT_RATIO
            if w.furniture:
                cls = FURNITURE
            elif sup:
                cls = FOOTNOTE
            else:
                cls = BODY
            for value, written in toks:
                entries.append(
                    {
                        "word": w,
                        "line": i,
                        "written": written,
                        "value": value,
                        "cls": cls,
                        "text": text,
                        "row": row_id if in_tail else None,
                        "row_label": row[0] if in_tail else "",
                    }
                )

    out = [
        LayerNumber(
            e["written"], e["value"], e["cls"], e["line"], e["text"], e["row"], e["row_label"]
        )
        for e in entries
    ]
    for w in cross:
        for value, written in numeric_tokens(str(w[4])):
            out.append(LayerNumber(written, value, MARGIN, None, str(w[4])))
    return out


def candidate_numbers(numbers: Sequence[LayerNumber]) -> list[LayerNumber]:
    """The label-row tail numbers, in the order :func:`body_text` writes them."""
    sel = [n for n in numbers if n.row is not None and n.line_index is not None]
    return [sel[p] for p in sorted(range(len(sel)), key=lambda p: (sel[p].line_index, p))]


def body_text(numbers: Sequence[LayerNumber]) -> str:
    """The label-row tail numbers' written forms: spaces within a line, newlines between."""
    lines: dict[int, list[str]] = {}
    for n in numbers:
        if n.row is not None and n.line_index is not None:
            lines.setdefault(n.line_index, []).append(n.written)
    return "\n".join(" ".join(lines[k]) for k in sorted(lines))


@dataclass(frozen=True)
class DropReference:
    """What a model reading owes the page, from the page's own text layer."""

    #: the label-row tail numbers, written forms, one text line per layer line
    text: str


def drop_reference_from_page(page, *, repeated_slots, folio_slots) -> DropReference:
    """The reference for ``reading_agreement.dropped_numbers``, read from a PyMuPDF page.

    ``repeated_slots`` and ``folio_slots`` are ``BornDigitalDetector._repeated_text_slots``
    and ``_folio_slots`` of the page's document.

    Coordinate conventions. ``get_text("words")``, ``get_text("dict")`` span boxes,
    ``get_image_info()`` bboxes and ``chart_region_bboxes`` (drawings) are all UNROTATED
    page coordinates, so they are compared without conversion.

    Every word inside a chart box or an image box is excluded: its numbers are chart or
    figure labels (eye-adjudicated sample, #1218: every Fed false alarm was a chart page).
    A table that ``chart_region_bboxes`` mistakes for a chart (#1268) is therefore NOT
    checked here -- a missed check, never a false mark. Keeping boxes that overlap a
    ``find_tables()`` table was measured and rejected: on the 102-page sample it recovered no
    real drop and re-admitted three chart false alarms. Raises on a PyMuPDF failure; the caller must treat that
    as "no evidence", never as "nothing dropped".
    """
    from socr.tables.reconstruct import chart_region_bboxes

    words = page.get_text("words")
    excluded = [tuple(r) for r in chart_region_bboxes(page)]
    images = [tuple(i["bbox"]) for i in page.get_image_info()]
    is_furniture = slot_predicate(repeated_slots, folio_slots)
    numbers = classify_layer_numbers(
        words,
        tuple(page.rect),
        is_furniture,
        rotation=int(page.rotation or 0),
        superscripts=superscript_boxes(page.get_text("dict")),
        excluded_boxes=excluded,
        image_boxes=images,
    )
    return DropReference(body_text(numbers))
