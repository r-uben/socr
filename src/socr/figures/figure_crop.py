"""#1276: where a figure-presence crop is cut.

The box ``chart_region_bboxes`` returns is built to EXCLUDE a chart's labels from a table
rowizer, not to frame a picture: it runs to the page's left edge (so a two-column page's
neighbouring prose falls inside it), ends one row below the drawing (so a ``NOTE:`` / ``SOURCE:``
line and a second row of axis labels are cut), starts one row above it (so the figure's title is
cut), and a multi-panel figure is several boxes. A hand check of 128 such crops on the 156-doc
run found 121 partial and 37 carrying the neighbouring column's prose.

This module cuts from where the chart is DRAWN (the drawing cluster) and grows the rectangle
only by text the PDF's own layer says belongs to the figure:

* sideways: numeric-only blocks beside the drawing (tick labels), reached without crossing
  any non-numeric word. Never the page edge, never another column's prose;
* upward: the title block directly above in the drawing's own column, then further label
  blocks, while the gap stays within the page's body line pitch;
* downward: axis labels, then ``NOTE:`` / ``SOURCE:`` blocks, under the same gap rule;
* never into another chart's drawing, and a block already taken by the region above is not
  taken again by the region below;
* regions that share a title block (panels of one figure), or whose rectangles overlap, are cut
  as one crop.

Every distance is derived from the page: the body line pitch (median step between consecutive
lines of a block) and the median word height. A block is a "label" rather than a paragraph by
the one named constant below.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from socr.figures.unshipped_figures import _TICK_NUMBER_RE

#: A title, an axis-label row or a note wraps at most this many lines before it is a paragraph
#: of body prose. Blocks that open with a caption lead word (below) are exempt: a ``SOURCE:``
#: note legitimately runs long.
MAX_LABEL_LINES = 2

#: Words that open a figure's TITLE block (it sits above the drawing; below it, the same words
#: open the NEXT figure) and its NOTE block (below the drawing). English, as the corpus is.
_TITLE_LEAD_RE = re.compile(r"^\s*(figure|fig\.|chart|exhibit|panel|\d+\.)(?=\W|$)", re.IGNORECASE)
_NOTE_LEAD_RE = re.compile(r"^\s*(note|notes|source|sources)(?=\W|$)", re.IGNORECASE)


@dataclass(frozen=True)
class CropGroup:
    """One crop: the chart regions it covers (1-based detector ordinals) and its rectangle."""

    indices: tuple[int, ...]
    rect: tuple[float, float, float, float]

    @property
    def index(self) -> int:
        """The ordinal that names the crop file: the first region it covers."""
        return self.indices[0]


@dataclass
class _Block:
    ident: int
    x0: float
    y0: float
    x1: float
    y1: float
    lines: int
    text: str
    numeric: bool


def _blocks(words: list) -> list[_Block]:
    by_block: dict[int, list] = {}
    for w in words:
        by_block.setdefault(int(w[5]), []).append(w)
    out: list[_Block] = []
    for ident, ws in by_block.items():
        ws = sorted(ws, key=lambda w: (int(w[6]), w[0]))
        tokens = [str(w[4]) for w in ws]
        out.append(
            _Block(
                ident=ident,
                x0=min(w[0] for w in ws),
                y0=min(w[1] for w in ws),
                x1=max(w[2] for w in ws),
                y1=max(w[3] for w in ws),
                lines=len({int(w[6]) for w in ws}),
                text=" ".join(tokens),
                numeric=all(_TICK_NUMBER_RE.match(t) for t in tokens),
            )
        )
    return out


def line_pitch(words: list) -> float:
    """The page's body line pitch: the median step between consecutive lines of one block,
    else the median word height. ``0.0`` when the page has no words."""
    steps: list[float] = []
    by_block: dict[int, dict[int, float]] = {}
    for w in words:
        by_block.setdefault(int(w[5]), {}).setdefault(int(w[6]), float(w[1]))
    for lines in by_block.values():
        ys = [lines[k] for k in sorted(lines)]
        steps += [b - a for a, b in zip(ys, ys[1:]) if b > a]
    if steps:
        return statistics.median(steps)
    heights = [float(w[3]) - float(w[1]) for w in words if w[3] > w[1]]
    return statistics.median(heights) if heights else 0.0


def _x_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def _is_label(block: _Block, *, above: bool) -> bool:
    """A block that belongs to the figure on the side of it the walk is on. A title lead below
    the drawing is the next figure's title; a note lead above it is the previous figure's note."""
    lead = _TITLE_LEAD_RE if above else _NOTE_LEAD_RE
    other = _NOTE_LEAD_RE if above else _TITLE_LEAD_RE
    if lead.match(block.text):
        return True
    return block.lines <= MAX_LABEL_LINES and not other.match(block.text)


def _grow(rect: list[float], block: _Block) -> None:
    rect[0] = min(rect[0], block.x0)
    rect[1] = min(rect[1], block.y0)
    rect[2] = max(rect[2], block.x1)
    rect[3] = max(rect[3], block.y1)


def _region_rect(
    core: tuple[float, float, float, float],
    blocks: list[_Block],
    pitch: float,
    others: list[tuple[float, float, float, float]],
    taken_below: set[int],
) -> tuple[list[float], set[int], set[int]]:
    """The grown rectangle of one region, the blocks taken ABOVE it, and those taken BELOW."""
    rect = [float(core[0]), float(core[1]), float(core[2]), float(core[3])]

    def touches_other(block: _Block) -> bool:
        return any(
            _x_overlap(block.x0, block.x1, o[0], o[2])
            and _x_overlap(block.y0, block.y1, o[1], o[3])
            for o in others
        )

    # Sideways: numeric tick labels in the drawing's own height band, nearest first, reached
    # without crossing a non-numeric word (the neighbouring column's prose stops the walk).
    band = [
        c for c in blocks if c.numeric and _x_overlap(c.y0, c.y1, rect[1] - pitch, rect[3] + pitch)
    ]
    changed = True
    while changed:
        changed = False
        for c in band:
            if touches_other(c) or _x_overlap(c.x0, c.x1, rect[0], rect[2]):
                continue
            gap = rect[0] - c.x1 if c.x1 <= rect[0] else c.x0 - rect[2]
            if gap > pitch:
                continue
            _grow(rect, c)
            changed = True
    # Tick labels may stick out past the drawing's top and bottom by half a label.
    for c in blocks:
        if (
            c.numeric
            and _x_overlap(c.x0, c.x1, rect[0], rect[2])
            and _x_overlap(c.y0, c.y1, rect[1] - pitch, rect[3] + pitch)
            and not touches_other(c)
        ):
            _grow(rect, c)

    above: set[int] = set()
    below: set[int] = set()
    # Downward.
    cursor = rect[3]
    while True:
        cands = [
            c
            for c in blocks
            if c.ident not in below
            and c.y1 > cursor
            and _x_overlap(c.x0, c.x1, rect[0], rect[2])
            and not touches_other(c)
        ]
        if not cands:
            break
        nxt = min(cands, key=lambda c: c.y0)
        if nxt.y0 - cursor > pitch or not _is_label(nxt, above=False):
            break
        below.add(nxt.ident)
        _grow(rect, nxt)
        cursor = rect[3]
    # Upward.
    cursor = rect[1]
    while True:
        cands = [
            c
            for c in blocks
            if c.ident not in above
            and c.ident not in taken_below
            and c.y0 < cursor
            and _x_overlap(c.x0, c.x1, rect[0], rect[2])
            and not touches_other(c)
        ]
        if not cands:
            break
        nxt = max(cands, key=lambda c: c.y1)
        if cursor - nxt.y1 > pitch or not _is_label(nxt, above=True):
            break
        above.add(nxt.ident)
        _grow(rect, nxt)
        cursor = rect[1]
    return rect, above, below


def crop_groups(
    words: list, cores: list[tuple[float, float, float, float]], page_rect
) -> list[CropGroup]:
    """One crop rectangle per figure, from the drawing clusters *cores* (detector order).

    Returns groups in detector order of their first region. *page_rect* clamps the result.
    With no words on the page there is nothing to grow by and each core is its own crop.
    """
    n = len(cores)
    if n == 0:
        return []
    pitch = line_pitch(words)
    blocks = _blocks(words) if pitch > 0 else []
    heights = [float(w[3]) - float(w[1]) for w in words if w[3] > w[1]]
    pad = max(0.0, (pitch - statistics.median(heights)) / 2.0) if heights and pitch else 0.0

    order = sorted(range(n), key=lambda i: (cores[i][1], cores[i][0]))
    rects: dict[int, list[float]] = {}
    above: dict[int, set[int]] = {}
    taken_below: set[int] = set()
    for i in order:
        others = [cores[j] for j in range(n) if j != i]
        rect, up, down = _region_rect(cores[i], blocks, pitch, others, taken_below)
        rects[i], above[i] = rect, up
        taken_below |= down

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            shares_title = bool(above[i] & above[j])
            a, b = rects[i], rects[j]
            overlaps = _x_overlap(a[0], a[2], b[0], b[2]) and _x_overlap(a[1], a[3], b[1], b[3])
            if shares_title or overlaps:
                parent[find(j)] = find(i)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: list[CropGroup] = []
    for members in groups.values():
        x0 = min(rects[i][0] for i in members) - pad
        y0 = min(rects[i][1] for i in members) - pad
        x1 = max(rects[i][2] for i in members) + pad
        y1 = max(rects[i][3] for i in members) + pad
        out.append(
            CropGroup(
                indices=tuple(sorted(i + 1 for i in members)),
                rect=(
                    max(float(page_rect.x0), x0),
                    max(float(page_rect.y0), y0),
                    min(float(page_rect.x1), x1),
                    min(float(page_rect.y1), y1),
                ),
            )
        )
    return sorted(out, key=lambda g: g.indices[0])


def expand_covered(covered: set[int], groups: list[CropGroup]) -> set[int]:
    """A group's crop is named by its first region, so the crop covers every member."""
    out = set(covered)
    for g in groups:
        if g.index in covered:
            out.update(g.indices)
    return out
