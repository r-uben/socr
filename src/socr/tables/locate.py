"""Locate tables on a born-digital page with a *precise, croppable* bounding box.

Two detectors, both validated against synthetic econ/finance table styles:

1. **Ruled tables** — ``page.find_tables()`` (default "lines" strategy) returns a
   tight bbox whenever the table has drawn cell borders.

2. **Booktabs tables** — the econ-paper norm: only top / under-header / bottom
   horizontal rules, no verticals. ``find_tables`` "lines" misses these entirely,
   and its "text" strategy returns a bbox that swallows surrounding prose (on a
   realistic mixed page it spanned the section title, a paragraph, the caption and
   the table). But the horizontal rules are real drawing vectors: reading their
   y-coordinates from ``page.get_drawings()`` gives a tight vertical band, and the
   rule endpoints give the horizontal extent. That band is a precise, crop-safe
   bbox.

Fully borderless (whitespace-only) tables have no geometric anchor and are out of
scope; the whole-page hard-page judge still covers them.

All bboxes are PDF points (72-unit space, origin top-left, y-down), the same
space ``find_tables`` and ``get_drawings`` report in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# A drawn line counts as a table rule only if it is near-horizontal and spans a
# meaningful width — short ticks, underlines and inline rules are excluded. The
# width floor is in PDF points (~1 inch); derived from the geometry of the line,
# not a tuned percentage.
_RULE_FLATNESS_PT = 1.0  # |y0 - y1| below this => horizontal
_RULE_MIN_WIDTH_PT = 72.0  # >= ~1 inch wide to be a table rule, not an underline
# Two rules belong to the same table if they share most of their horizontal span
# and are not separated by a large vertical gap. Both are geometric, not tuned.
_RULE_X_OVERLAP = 0.6  # fraction of shared width to be the "same" table column band
_BOX_IOU_DEDUP = 0.5  # ruled/booktabs boxes overlapping this much are the same table


@dataclass(frozen=True)
class TableBox:
    """A located table region.

    ``bbox`` is ``(x0, y0, x1, y1)`` in PDF points. ``source`` records which
    detector found it ("ruled" | "booktabs") for logging and tests.
    """

    bbox: tuple[float, float, float, float]
    source: str

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def locate_tables(page) -> list[TableBox]:
    """Return precise, croppable table boxes on ``page``, top-to-bottom.

    Never raises: any detector failure is logged and yields no box, so a bad page
    degrades to "no dual-pass" rather than dropping the page.
    """
    boxes: list[TableBox] = []
    boxes.extend(_ruled_tables(page))
    boxes.extend(_booktabs_tables(page))

    # Scanned pages expose no vectors, so the detectors above find nothing. Fall
    # back to detecting rules in the rendered raster. Gated on the scanned
    # signature (a page image + no vector drawings) so we never run CV on the
    # born-digital common case, where the vector path already answered.
    if not boxes and _is_scanned(page):
        from socr.tables.image_locate import locate_tables_image

        boxes.extend(locate_tables_image(page))

    boxes = _dedup(boxes)
    boxes.sort(key=lambda b: (b.bbox[1], b.bbox[0]))  # reading order
    return boxes


def _is_scanned(page) -> bool:
    """Scanned-page signature: a raster image present and no vector drawings.

    Born-digital pages (even table-less prose) carry vector drawings; a scan is a
    single embedded image with an empty drawing list.
    """
    try:
        return bool(page.get_images()) and not page.get_drawings()
    except Exception:  # pragma: no cover - defensive
        return False


def _ruled_tables(page) -> list[TableBox]:
    """Fully ruled tables via PyMuPDF's default (lines) strategy."""
    try:
        result = page.find_tables()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("find_tables failed: %s", exc)
        return []
    out: list[TableBox] = []
    for table in getattr(result, "tables", []):
        bbox = getattr(table, "bbox", None)
        if bbox and _plausible(bbox):
            out.append(TableBox(bbox=tuple(float(v) for v in bbox), source="ruled"))
    return out


def _booktabs_tables(page) -> list[TableBox]:
    """Tables localised from their horizontal rules (top/mid/bottom).

    Groups long horizontal drawn lines into vertically-contiguous bands that share
    a horizontal span, then bounds each group by ``(min x, top y, max x, bottom y)``.
    A single isolated rule is not a table (needs >= 2 rules to bound a band).
    """
    rules = _horizontal_rules(page)
    return bands_from_rules(rules, page.rect, source="booktabs")


def bands_from_rules(
    rules: list[tuple[float, float, float]], page_rect, source: str
) -> list[TableBox]:
    """Group horizontal rules ``(y, x0, x1)`` (PDF points) into table bands.

    Shared by the vector booktabs detector and the image-raster detector. Rules
    that horizontally overlap are stacked into one band, bounded by their extent
    and clamped to the page. A band needs >= 2 rules (one rule cannot bound a
    table).

    NOTE (known limitation): rules alone cannot tell one tall table from two
    stacked ones. Real pages confirmed both shapes with similar inter-rule gaps
    (a 238pt gap inside a single full-page table vs a 160pt gap between two small
    tables), so a gap-split heuristic would mis-handle one to fix the other.
    Multi-table pages therefore over-merge into one band -> an imprecise crop ->
    the reconciler's column-count check flags rather than patches. Failing to
    flag-only is safe; it just forfeits the benefit. Precise multi-table
    splitting needs content-aware region detection (future work).
    """
    if len(rules) < 2:
        return []

    rules = sorted(rules)
    groups: list[list[tuple[float, float, float]]] = []
    for rule in rules:  # (y, x0, x1)
        placed = False
        for group in groups:
            if _x_overlaps(rule, group):
                group.append(rule)
                placed = True
                break
        if not placed:
            groups.append([rule])

    out: list[TableBox] = []
    for group in groups:
        if len(group) < 2:
            continue
        ys = [r[0] for r in group]
        x0 = max(page_rect.x0, min(r[1] for r in group))
        x1 = min(page_rect.x1, max(r[2] for r in group))
        bbox = (x0, min(ys), x1, max(ys))
        if _plausible(bbox):
            out.append(TableBox(bbox=bbox, source=source))
    return out


def _horizontal_rules(page) -> list[tuple[float, float, float]]:
    """Extract long horizontal drawn lines as ``(y, x0, x1)``, sorted by y.

    Reads both explicit line items ("l") and thin filled/stroked rectangles
    ("re") that LaTeX/booktabs sometimes emit for rules.
    """
    try:
        drawings = page.get_drawings()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("get_drawings failed: %s", exc)
        return []

    # Reject rules outside the visible page: real papers carry page-frame lines
    # and crop marks drawn in the margin (observed at y < 0 and y > page height on
    # dense Fama-French pages). Including them stretches a table band to the whole
    # page. Clamp x to the page so a rule bleeding into the margin still anchors
    # the band sanely.
    page_rect = page.rect
    rules: list[tuple[float, float, float]] = []
    for d in drawings:
        for item in d.get("items", []):
            kind = item[0]
            if kind == "l":  # line: ("l", p0, p1)
                (px0, py0), (px1, py1) = item[1], item[2]
                if abs(py0 - py1) <= _RULE_FLATNESS_PT and abs(px1 - px0) >= _RULE_MIN_WIDTH_PT:
                    rules.append((min(py0, py1), min(px0, px1), max(px0, px1)))
            elif kind == "re":  # rectangle: ("re", Rect, ...)
                rect = item[1]
                w = abs(rect.x1 - rect.x0)
                h = abs(rect.y1 - rect.y0)
                if h <= _RULE_FLATNESS_PT and w >= _RULE_MIN_WIDTH_PT:
                    y = (rect.y0 + rect.y1) / 2.0
                    rules.append((y, min(rect.x0, rect.x1), max(rect.x0, rect.x1)))
    rules = [
        (y, max(page_rect.x0, x0), min(page_rect.x1, x1))
        for (y, x0, x1) in rules
        if page_rect.y0 <= y <= page_rect.y1
    ]
    rules.sort()
    return rules


def _x_overlaps(rule: tuple[float, float, float], group: list[tuple[float, float, float]]) -> bool:
    """True if ``rule`` shares >= _RULE_X_OVERLAP of its width with the group's span."""
    g_x0 = min(r[1] for r in group)
    g_x1 = max(r[2] for r in group)
    _, r_x0, r_x1 = rule
    inter = max(0.0, min(r_x1, g_x1) - max(r_x0, g_x0))
    width = min(r_x1 - r_x0, g_x1 - g_x0)
    return width > 0 and inter / width >= _RULE_X_OVERLAP


def _plausible(bbox) -> bool:
    """Reject degenerate boxes (zero area, inverted coords)."""
    x0, y0, x1, y1 = bbox
    return x1 - x0 > 1.0 and y1 - y0 > 1.0


def _dedup(boxes: list[TableBox]) -> list[TableBox]:
    """Drop a box that substantially overlaps an already-kept one.

    A ruled detection and a booktabs detection can fire on the same table; keep
    the larger (it bounds the full region), preferring ``ruled`` on a tie since
    its bbox comes straight from cell geometry.
    """
    kept: list[TableBox] = []
    for box in sorted(boxes, key=lambda b: b.area, reverse=True):
        if any(_iou(box.bbox, k.bbox) >= _BOX_IOU_DEDUP for k in kept):
            continue
        kept.append(box)
    return kept


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
