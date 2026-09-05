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


# ---------------------------------------------------------------------------
# VI-C2a: row-band / column-edge / ordinal-origin helpers.
#
# Pure functions over ``(page, region)`` — no binder state, no candidate
# markdown. ``region`` is a witness table's own bbox, the same tuple
# ``TableBox.bbox`` already carries. They give the verifier an address for a
# disputed cell that neither the native binder nor the candidate's own
# structure gets to nominate. Design:
# docs/plans/verifier-independence/logs/2026-09-05_C1-design.md, §(a).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowBand:
    """One printed table row's vertical extent inside a witness region."""

    y0: float
    y1: float
    source: str  # "rule" | "line"


def row_bands_from_rules(
    rules: list[tuple[float, float, float]], region: tuple[float, float, float, float]
) -> list[RowBand]:
    """Pair consecutive horizontal rules inside ``region`` into row bands.

    Only trustworthy where a table draws a rule between every printed row.
    A booktabs table's top/mid/bottom rules still pair into bands here, but
    they span many printed rows each; :func:`row_bands` only accepts this
    result where it corresponds 1:1 with the line-derived bands.
    """
    x0, y0, x1, y1 = region
    ys = sorted({r[0] for r in rules if y0 <= r[0] <= y1 and _rule_overlaps_span(r, x0, x1)})
    return [RowBand(y0=a, y1=b, source="rule") for a, b in zip(ys, ys[1:])]


def row_bands_from_lines(page, region: tuple[float, float, float, float]) -> list[RowBand]:
    """Group PDF text lines inside ``region`` into row bands.

    One printed row = one band. Two lines join only when their boxes overlap
    in y by more than the overlap any two adjacent printed rows share —
    adjacent meaning consecutive unique-baseline groups at the region's
    line pitch (the modal unique-baseline gap). Baseline-distance clustering
    is transitive at a 9.5 pt pitch on 10 pt type and collapses six printed
    rows into three (or one). No text layer, or too few lines to establish
    a pitch, returns no bands: that is an abstain input, not a guess.
    """
    lines = _text_lines_in_region(page, region)
    if not lines:
        return []
    groups = _group_lines_by_baseline(lines)
    return [
        RowBand(
            y0=min(ln["bbox"][1] for ln in group),
            y1=max(ln["bbox"][3] for ln in group),
            source="line",
        )
        for group in groups
    ]


def row_bands(page, region: tuple[float, float, float, float]) -> list[RowBand]:
    """Row bands for ``region``: rules where they address one printed row
    each, lines otherwise (C1 §(f) decision 1 — rules-else-lines).

    A booktabs table's top/mid/bottom rules pair into a handful of bands
    that each span many printed rows — not "per-row rules". The
    distinguishing test is page-derived correspondence, not a rule-count
    threshold: rule-derived bands are trusted only when there are exactly as
    many of them as line-derived bands, and each one contains its
    line-derived counterpart.
    """
    line_bands = row_bands_from_lines(page, region)
    if not line_bands:
        return []  # no text layer: abstain input, never a guess
    rule_bands = row_bands_from_rules(_horizontal_rules(page), region)
    if len(rule_bands) == len(line_bands) and all(
        rb.y0 <= lb.y0 and rb.y1 >= lb.y1 for rb, lb in zip(rule_bands, line_bands)
    ):
        return rule_bands
    return line_bands


def ordinal_origin(page, region: tuple[float, float, float, float]) -> float | None:
    """The y of the second horizontal-rule group inside ``region`` — the
    header rule ordinals are counted from (C1 §(a)).

    Two consecutive rules are one drawn border (booktabs' doubled
    ``\\toprule``) iff no text-line baseline in the region lies between
    them *and* their gap is smaller than the smallest text-line height in
    the region — a gap no printed line could fit in. Otherwise they are
    distinct. Both conditions are page-derived; neither needs a gap
    distribution. No text lines in the region means the conditions cannot
    be evaluated, so this returns ``None`` (no certified origin → abstain),
    never a guess. ``None`` also when fewer than two rules, or fewer than
    two groups, exist.
    """
    x0, y0, x1, y1 = region
    rules = sorted(
        r for r in _horizontal_rules(page) if y0 <= r[0] <= y1 and _rule_overlaps_span(r, x0, x1)
    )
    if len(rules) < 2:
        return None
    lines = _text_lines_in_region(page, region)
    if not lines:
        return None
    groups: list[list[tuple[float, float, float]]] = [[rules[0]]]
    for rule in rules[1:]:
        prev = groups[-1][-1]
        if _rules_are_one_border(prev[0], rule[0], lines):
            groups[-1].append(rule)
        else:
            groups.append([rule])
    if len(groups) < 2:
        return None
    second_group = groups[1]
    return sum(r[0] for r in second_group) / len(second_group)


def label_column_edge(page, region: tuple[float, float, float, float]) -> float | None:
    """The label column's right edge ``R`` inside ``region`` (C1 §(a)).

    ``R`` starts at the leftmost ``x0`` among every non-leftmost line of any
    printed row, then shrinks to a whitespace edge: while some line
    straddles it, ``R`` moves left to that line's own ``x0``. Fixed point
    observed in <=2 passes on the corpus this was measured against — an
    observation, not a limit. ``None`` on a one-column region (every row has
    exactly one line), when no candidate lies right of the region's own
    left edge, or when ``R`` collapses onto the leftmost text (a wrapped
    label's own ``x0`` is not a column edge).
    """
    lines = _text_lines_in_region(page, region)
    if not lines:
        return None
    candidates: list[float] = []
    for group in _group_lines_by_baseline(lines):
        ordered = sorted(group, key=lambda ln: ln["x0"])
        candidates.extend(ln["x0"] for ln in ordered[1:])
    if not candidates:
        return None
    edge = min(candidates)
    while True:
        straddling = [ln for ln in lines if ln["x0"] < edge < ln["x1"]]
        if not straddling:
            break
        narrower = min(ln["x0"] for ln in straddling)
        if narrower >= edge:
            break
        edge = narrower
    # R is a column edge only if it sits strictly right of the region's
    # left bound *and* of the leftmost text — a wrapped label's own x0
    # is not a column edge (it is the label column collapsing onto the
    # region's left content edge, the same outcome as R == region.x0
    # when the label is flush with the box).
    content_x0 = min(ln["x0"] for ln in lines)
    if edge <= region[0] or edge <= content_x0:
        return None
    return edge


def band_index_for(bands: list[RowBand], y_mid: float) -> int | None:
    """Index of the unique band whose ``[y0, y1]`` contains ``y_mid``.

    ``None`` when no band contains the point, or when more than one does
    (overlapping bands: abstain, do not pick the first).
    """
    hits = [idx for idx, band in enumerate(bands) if band.y0 <= y_mid <= band.y1]
    return hits[0] if len(hits) == 1 else None


def _text_lines_in_region(page, region: tuple[float, float, float, float]) -> list[dict]:
    """PDF text lines inside ``region``, each as a dict with ``baseline``
    (first span's ``origin`` y), ``size`` (dominant span size), and ``bbox``.
    """
    try:
        text_dict = page.get_text("dict", clip=tuple(region))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("get_text(dict) failed: %s", exc)
        return []
    lines: list[dict] = []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            bbox = line["bbox"]
            lines.append(
                {
                    "baseline": spans[0]["origin"][1],
                    "size": max(s["size"] for s in spans),
                    "bbox": bbox,
                    "x0": bbox[0],
                    "x1": bbox[2],
                }
            )
    return lines


def _group_lines_by_baseline(lines: list[dict]) -> list[list[dict]]:
    """Group text-line dicts into printed rows by vertical-extent overlap.

    Two lines belong to one band only if their boxes overlap in y by more
    than the overlap any two *adjacent printed rows* share. Adjacent rows
    are consecutive unique-baseline groups whose baseline gap equals the
    region's line pitch — the modal unique-baseline-to-baseline distance
    (on a tie, the larger gap: the row step, not a wrap/subscript). Fewer
    than three lines cannot establish a pitch: return no groups (abstain
    input, never a guess).
    """
    if len(lines) < 3:
        return []
    ordered = sorted(lines, key=lambda ln: ln["baseline"])
    by_base: dict[float, list[dict]] = {}
    for ln in ordered:
        by_base.setdefault(ln["baseline"], []).append(ln)
    baselines = sorted(by_base)
    if len(baselines) == 1:
        return [ordered]
    gaps = [b - a for a, b in zip(baselines, baselines[1:])]
    pitch = _modal_value(gaps)

    def _union_box(key: float) -> tuple[float, float, float, float]:
        group = by_base[key]
        return (
            min(ln["bbox"][0] for ln in group),
            min(ln["bbox"][1] for ln in group),
            max(ln["bbox"][2] for ln in group),
            max(ln["bbox"][3] for ln in group),
        )

    adjacent_overlap = 0.0
    for a, b, gap in zip(baselines, baselines[1:], gaps):
        if gap == pitch:
            adjacent_overlap = max(adjacent_overlap, _y_overlap(_union_box(a), _union_box(b)))
    groups: list[list[dict]] = [[ordered[0]]]
    for cur in ordered[1:]:
        band_box = (
            min(ln["bbox"][0] for ln in groups[-1]),
            min(ln["bbox"][1] for ln in groups[-1]),
            max(ln["bbox"][2] for ln in groups[-1]),
            max(ln["bbox"][3] for ln in groups[-1]),
        )
        if _y_overlap(band_box, cur["bbox"]) > adjacent_overlap:
            groups[-1].append(cur)
        else:
            groups.append([cur])
    return groups


def _modal_value(values: list[float]) -> float:
    """The most frequent value in *values*; on a tie, the larger one."""
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    return max(v for v, n in counts.items() if n == best)


def _y_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Shared vertical span of two ``(x0, y0, x1, y1)`` boxes, or 0."""
    return max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _rules_are_one_border(y_a: float, y_b: float, lines: list[dict]) -> bool:
    """True iff *y_a*..*y_b* is one drawn border, not two distinct rules.

    No text-line baseline between them, and the gap is smaller than the
    smallest text-line height in the region (a gap no printed line could
    fit in). *lines* must be non-empty — the caller abstains otherwise.
    """
    min_height = min(ln["bbox"][3] - ln["bbox"][1] for ln in lines)
    if min_height <= 0:
        return False
    if any(y_a < ln["baseline"] < y_b for ln in lines):
        return False
    return (y_b - y_a) < min_height


def _rule_overlaps_span(rule: tuple, x0: float, x1: float) -> bool:
    """True if ``rule``'s ``(x0, x1)`` shares >= ``_RULE_X_OVERLAP`` of its
    width with the ``(x0, x1)`` span.
    """
    rx0, rx1 = rule[1], rule[2]
    inter = max(0.0, min(rx1, x1) - max(rx0, x0))
    width = min(rx1 - rx0, x1 - x0)
    return width > 0 and inter / width >= _RULE_X_OVERLAP
