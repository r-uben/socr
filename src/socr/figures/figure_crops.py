"""#1053: per-figure crops for the chart-asset lane (first slice: raster, single column).

The lane used to ship a page's flat native text and append ONE PNG of the whole page. On a page
that is prose plus a figure, that PNG repeats every paragraph and the figure sits below text it was
printed above. This module plans the alternative: cut each drawn raster figure out on its own,
place the crop at its y-position between the paragraphs (the same ``(rect, content)`` regions the
table interleaver takes), and say, for every native word, who owns it.

Ownership. Every word of the page is owned exactly once: by a figure box (its centre lies inside),
by a caption (it sits in a block that opens with a "Figure N" label), or by the prose. The owner
map decides which figures are unread; it never removes a word. A region is its placeholder only
and the interleaver runs with ``suppress_represented=False``, so every native word ships exactly
once, in stream order: words inside a box land right after its crop, and a line straddling a box
edge stays whole.

Unread words. A box with no native word inside it may hold words only in pixels. Nothing in this
slice reads them, so the page is flagged (``FailureMode.FIGURE_WORDS_UNREAD``), not trusted and not
re-routed: the page keeps its exact native prose and the crop is where those pixels live.

Pages this module declines (``plan_figure_page`` returns ``None``) keep today's whole-page route:
no qualifying raster figure, an unknown placement, a vector chart, a rotated page, or a layout that
is not clearly one column.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Owner labels of the word-to-owner map. A figure's owner is ``figure:<index>`` (1-based).
OWNER_PROSE = "prose"
OWNER_CAPTION = "caption"
OWNER_FIGURE_PREFIX = "figure:"


@dataclass(frozen=True)
class FigurePlan:
    """The crop plan of one page: boxes, per-word owners, and the contents per box."""

    #: ``(x0, y0, x1, y1)`` per figure, top to bottom; the 1-based position is the figure index.
    boxes: tuple[tuple[float, float, float, float], ...]
    #: One owner per word of ``page.get_text("words")``, in that order.
    owners: tuple[str, ...]
    #: Figure index -> how many native words lie inside its box.
    figure_word_counts: dict[int, int] = field(default_factory=dict)

    @property
    def unread_figures(self) -> list[int]:
        """Indices of the figures with no native word inside (their words are unread)."""
        return [i for i in range(1, len(self.boxes) + 1) if not self.figure_word_counts.get(i)]

    def owner_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for owner in self.owners:
            counts[owner] = counts.get(owner, 0) + 1
        return counts


#: Two boxes are "beside" each other when they overlap in y by more than this fraction of the
#: shorter one: more than half of it is level with the other. The midpoint, as in the
#: interleaver's ``_REGION_COVERAGE_DROP`` (a block half inside a region is genuinely inside it);
#: not tuned.
BESIDE_OVERLAP_MIN = 0.5


def _beside(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """True when *a* and *b* sit side by side: disjoint in x, level in y (``BESIDE_OVERLAP_MIN``)."""
    if a[2] > b[0] and b[2] > a[0]:
        return False
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    shorter = min(a[3] - a[1], b[3] - b[1])
    return shorter > 0 and overlap / shorter > BESIDE_OVERLAP_MIN


def is_single_column(page, boxes) -> bool:
    """True when nothing on *page* sits beside anything else.

    Read from the page's own text blocks and the figure boxes, no tuned constant: a two-column
    page has blocks level with blocks in the other column, a figure with text wrapped around it
    has a block level with the box. The interleaver orders regions by y alone, which is only right
    when this holds. A right-aligned page number level with a heading also fails it, and that
    fails closed: the page keeps the whole-page route.
    """
    items = [
        tuple(float(v) for v in b[:4])
        for b in page.get_text("blocks")
        if len(b) > 6 and b[6] == 0 and str(b[4]).strip()
    ]
    items.extend(tuple(box) for box in boxes)
    return not any(_beside(a, b) for i, a in enumerate(items) for b in items[i + 1 :])


def _inside(word, box) -> bool:
    cx, cy = (word[0] + word[2]) / 2.0, (word[1] + word[3]) / 2.0
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def word_owners(page, words, boxes) -> list[str]:
    """The owner of each word in *words* (``page.get_text("words")`` tuples), exactly one each."""
    from socr.figures.scanned_figures import _CAPTION_LINE_RE

    caption_blocks: set[int] = set()
    for block in page.get_text("blocks"):
        if len(block) > 6 and block[6] == 0 and _CAPTION_LINE_RE.match(str(block[4]).lstrip()):
            caption_blocks.add(int(block[5]))

    owners: list[str] = []
    for w in words:
        index = next((i for i, box in enumerate(boxes, start=1) if _inside(w, box)), 0)
        if index:
            owners.append(f"{OWNER_FIGURE_PREFIX}{index}")
        elif int(w[5]) in caption_blocks:
            owners.append(OWNER_CAPTION)
        else:
            owners.append(OWNER_PROSE)
    return owners


def is_in_reading_order(page, boxes) -> bool:
    """True when the interleaver, walking the blocks in stream order, places every crop correctly.

    The interleaver emits a region just before the first block whose y0 is at or below the
    region's y0, so a page whose stream writes a block below a box before a block above it would
    ship crop, below, above, and a block that starts above a box but has a line at or below its top is
    emitted whole before the crop. Blocks it relegates as furniture are not walked and are skipped
    here the same way.
    """
    from socr.core.born_digital import block_is_page_furniture, dominant_text_direction

    blocks = page.get_text("dict").get("blocks", [])
    dominant = dominant_text_direction(blocks)
    walked = [
        b for b in blocks if b.get("type", 0) == 0 and not block_is_page_furniture(b, dominant)
    ]
    for box in boxes:
        seen_below = False
        for block in walked:
            if block["bbox"][1] >= box[1]:
                seen_below = True
            elif seen_below:
                return False
            elif any(ln["bbox"][1] >= box[1] for ln in block.get("lines", []) or []):
                # The block starts above the box and has a line at or below its top: the crop
                # would be emitted after the whole block, leaving that line above the figure.
                return False
    return True


def plan_figure_page(page) -> FigurePlan | None:
    """Plan the crop route for *page*, or ``None`` when it keeps the whole-page route."""
    from socr.figures.extractor import _vector_chart_found, figure_boxes
    from socr.tables.reconstruct import chart_region_bboxes

    try:
        if page.rotation:
            return None
        boxes = figure_boxes(page)
        if not boxes:
            return None
        if _vector_chart_found(page) or chart_region_bboxes(page):
            return None
        if not is_single_column(page, boxes) or not is_in_reading_order(page, boxes):
            return None
        words = page.get_text("words") or []
    except Exception as exc:  # noqa: BLE001 - unknown geometry keeps the whole-page route
        logger.debug("figure crops: planning failed, whole-page route: %s", exc)
        return None
    owners = word_owners(page, words, boxes)
    return FigurePlan(
        boxes=tuple(boxes),
        owners=tuple(owners),
        figure_word_counts={
            i: owners.count(f"{OWNER_FIGURE_PREFIX}{i}") for i in range(1, len(boxes) + 1)
        },
    )
