"""Native-layer table row extraction for born-digital pages.

Recovers ``(hierarchy path, ordered values)`` for each row of a table directly
from the PDF's own text layer, with no model involved. Two consumers rely on it:
the GH-96 exactness metric (as ground truth) and the GH-96 escalation canary (as
the oracle a candidate's numbers are checked against).

Lives in ``tables`` rather than ``benchmark`` because it is a native-layer table
parser used by the production pipeline; the benchmark package re-exports it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from socr.benchmark.scorer import BenchmarkScorer
from socr.tables.native_verifier import strip_presentation

# A row whose label is only a hierarchy marker ("of which:") introduces children of
# the preceding labelled row; it carries no values of its own.
_MARKER_RE = re.compile(r"^\s*of\s+which\s*:?\s*$", re.IGNORECASE)

_FOOTNOTE_SUFFIX_RE = re.compile(r"(?<=[a-z\)])[0-9]{1,2}$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _is_value(cell: str) -> bool:
    """True when *cell* is a table value.

    Emphasis is stripped first (GH-103's lesson, in a second tokenizer):
    ``BenchmarkScorer._is_numeric_cell`` anchors its regex and does not tolerate
    markdown, so a bold cell like ``**43.2**`` reads as non-numeric. Engines emit
    section-total rows bold, so leaving this unstripped silently discards every
    parent row — it under-scored a candidate by 39 points before this was found.
    """
    return bool(cell) and BenchmarkScorer._is_numeric_cell(strip_presentation(cell))


def normalize_label(label: str) -> str:
    """Fold a row label to a comparison key.

    Emphasis, footnote superscripts and punctuation are presentation; the words are
    the identity.
    """
    text = label.replace("**", "").replace("*", "").replace("__", "").strip()
    text = text.lower()
    text = _FOOTNOTE_SUFFIX_RE.sub("", text)
    return _NON_ALNUM_RE.sub("", text)


@dataclass(frozen=True)
class LabeledRow:
    """One data row: its hierarchy path and its ordered values.

    ``path`` is ``(parent, ..., label)``. The last element is the row's own label;
    everything before it is the enclosing hierarchy, used for reporting so a failure
    names the block it occurred in.
    """

    path: tuple[str, ...]
    values: tuple[str, ...]

    @property
    def label(self) -> str:
        return self.path[-1] if self.path else ""

    @property
    def key(self) -> str:
        return normalize_label(self.label)

    @property
    def display(self) -> str:
        return " / ".join(self.path)


def native_rows_from_page(page) -> list[LabeledRow]:
    """Ground-truth rows from a born-digital page's native text layer.

    Scoped to the page's located table regions (``tables.locate.locate_tables``) —
    without that scoping the parser happily reads prose sentences that merely
    contain numbers ("rising to £61.7 billion (2.1 per cent of GDP)") as table rows.

    Within a region, words are grouped into their true text lines (fitz block/line
    indices rather than a y tolerance), the label is the leading non-numeric run and
    the values are the numeric tokens in x order. Hierarchy comes from x-indentation
    of the label, which is the only model-free parent signal available: a child row
    is indented relative to its parent.

    Note ``native_verifier._rows_by_y`` is deliberately NOT reused here — it keeps
    numeric tokens only, because it exists for lane counting.
    """
    from socr.tables.locate import locate_tables

    try:
        boxes = locate_tables(page)
    except Exception:  # locate_tables never raises, but stay defensive
        boxes = []
    if not boxes:
        return []

    rows: list[LabeledRow] = []
    for box in boxes:
        rows.extend(_rows_in_region(page, box.bbox))
    return rows


def _rows_in_region(page, bbox) -> list[LabeledRow]:
    """Labelled rows from the words falling inside one table bbox.

    Two things this must get right, both learned from the OBR reference page:

    1. **Row grouping is by vertical overlap, not by fitz block/line.** A PDF table
       routinely emits the row label and its numbers as separate text objects, so
       block/line indices split one visual row in two. Words sharing a visual row
       have overlapping ``[y0, y1]`` intervals; words on different rows do not. That
       is parameter-free — no y tolerance to tune.

    2. **The label/value boundary is a column position, not "the first number".**
       "Growth Plan after 17 October reversals" contains a numeral inside the label.
       The value columns are found by clustering the x-positions of numeric tokens
       across the whole region (reusing the native verifier's lane clusterer), and
       anything left of the first value column is label text.
    """
    x0_lim, y0_lim, x1_lim, y1_lim = bbox
    words = [
        (x0, y0, x1, y1, word)
        for x0, y0, x1, y1, word, *_ in page.get_text("words")
        if x0 >= x0_lim and x1 <= x1_lim and y0 >= y0_lim and y1 <= y1_lim
    ]
    if not words:
        return []

    # (1) cluster words into visual rows by y-interval overlap
    words.sort(key=lambda w: (w[1], w[0]))
    bands: list[list[tuple[float, float, float, float, str]]] = []
    band_y1 = None
    for word in words:
        if band_y1 is None or word[1] >= band_y1:
            bands.append([word])
            band_y1 = word[3]
        else:
            bands[-1].append(word)
            band_y1 = max(band_y1, word[3])

    # (2) split each row at its last non-numeric word.
    #
    # Neither "everything before the first number" nor an x-clustered column
    # boundary survives real data: the first rule truncates
    # "Growth Plan after 17 October reversals" at the 17, and the second is
    # chicken-and-egg (that stray 17 forms its own x-cluster and drags the boundary
    # left). A table row is a label followed by values, so the last non-numeric word
    # ends the label. Parameter-free, and correct for both cases above.
    raw: list[tuple[float, str, list[tuple[float, str]]]] = []
    for band in bands:
        ordered = sorted(band, key=lambda w: w[0])
        last_text = -1
        for idx, word in enumerate(ordered):
            if not _is_value(word[4]):
                last_text = idx
        if last_text < 0:
            continue  # numbers only: an orphan row, no label to key on
        label = " ".join(w[4].strip() for w in ordered[: last_text + 1]).strip()
        values = [(w[0], w[4].strip()) for w in ordered[last_text + 1 :] if _is_value(w[4])]
        indent = ordered[0][0]
        if label and values and not _MARKER_RE.match(label):
            raw.append((indent, label, values))

    if not raw:
        return []

    raw = _drop_footnote_markers(raw, _superscript_tokens(page, bbox))

    # Parent = the nearest preceding row at a strictly smaller indent.
    rows: list[LabeledRow] = []
    stack: list[tuple[float, str]] = []
    for indent, label, values in raw:
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = tuple(lbl for _i, lbl in stack) + (label,)
        rows.append(LabeledRow(path=path, values=tuple(values)))
        stack.append((indent, label))
    return rows


def _superscript_tokens(page, bbox) -> set[tuple[int, str]]:
    """Locate footnote superscripts inside *bbox*, keyed by (rounded x, text).

    A footnote marker is a separate word, so "Memo: UK oil and gas revenues5"
    leaves a bare ``5`` between the label and the first real value. The
    label/value split cannot see it — the token is numeric and follows the last
    non-numeric word — so it inflates the row's width and injects a spurious
    value into the ground truth.

    Font size separates them, but only *within a row*. Comparing against the whole
    region's modal size misfires badly: an entire "Memo:" row is often set smaller
    than the table body, and region-wide sizing then flags all of its real values
    as markers. A superscript is small relative to the text it is attached to, so
    the comparison is per-row.

    Deliberately not a width heuristic — trimming rows merely wider than the modal
    row silently deletes real leading values from legitimately wide rows.
    """
    x0_lim, y0_lim, x1_lim, y1_lim = bbox
    by_line: dict[tuple[int, int], list[tuple[float, float, str]]] = {}
    for bi, block in enumerate(page.get_text("dict").get("blocks", [])):
        for li, line in enumerate(block.get("lines", [])):
            for span in line.get("spans", []):
                sx0, sy0, sx1, sy1 = span.get("bbox", (0, 0, 0, 0))
                if sx0 < x0_lim or sx1 > x1_lim or sy0 < y0_lim or sy1 > y1_lim:
                    continue
                text = (span.get("text") or "").strip()
                if text:
                    by_line.setdefault((bi, li), []).append(
                        (round(span.get("size", 0.0), 1), sx0, text)
                    )

    markers: set[tuple[int, str]] = set()
    for spans in by_line.values():
        sizes = [s for s, _x, _t in spans]
        row_size = max(set(sizes), key=sizes.count)
        for size, x, text in spans:
            if size < row_size and _is_value(text):
                markers.add((round(x), text))
    return markers


def _drop_footnote_markers(
    raw: list[tuple[float, str, list[tuple[float, str]]]],
    superscripts: set[tuple[int, str]],
) -> list[tuple[float, str, list[str]]]:
    """Remove superscript markers that the label/value split counted as values."""
    return [
        (indent, label, [v for x, v in values if (round(x), v) not in superscripts])
        for indent, label, values in raw
    ]
