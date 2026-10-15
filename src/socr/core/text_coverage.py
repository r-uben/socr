"""#1050 step 1: model-free ink-vs-visible-text-layer coverage on trusted-native pages.

Hand-check precision on silent-SUCCESS pages: 53/60 (88%) had some real
uncovered text; 237/295 regions (80%) were citable. Astra's region precision
was 41%; Wonder's success-stratum region precision was 80–94% but flagged
more pages (131 vs Fable's 106) with more furniture. Fable's cuts come from
600 corpus pages / 1.19 M covered glyphs (seed 20261013).

No model call. No OCR of crops. A page is ``passed``, ``flagged`` or
``unknown``. Ink is local-contrast in either polarity (dark-on-light
or light-on-dark). Each polarity is envelope-filtered on its own, then
the regions are unioned: a page-global mid-grey threshold hid black
lettering on a mid or dark fill. There is no repeat-ink suppression:
a logo flagged by mistake is an unneeded warning; a label suppressed
by mistake is silent loss. A row is a word if it meets Astra's
page-local ``min_run_components`` *or* Fable's corpus median word width.
"""

from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass, field

import numpy as np

from socr.tables.image_locate import DETECT_DPI

logger = logging.getLogger(__name__)

# ``DETECT_DPI`` is the raster ``image_locate`` and ``blank_page`` already use
# to find ink. The #1099 visible-char rule (span alpha ≠ 0) is applied in
# ``visible_char_boxes``; private names are not imported across packages.
#
# ``image_locate._MIN_FILL_FRAC`` / ``_MAX_THICK_PT`` are the same numbers
# (0.55, 2.0 pt). Copied: ``test_package_layering`` forbids importing private
# names across ``socr.*`` packages.
_MIN_FILL_FRAC = 0.55
_MAX_THICK_PT = 2.0

SCALE = DETECT_DPI / 72.0

# image_locate's adaptive binarisation (block px, offset in grey levels).
ADAPT_BLOCK = 15
ADAPT_C = 2

# image_locate's rule thickness, in px at DETECT_DPI.
MAX_THICK_PX = max(2.0, _MAX_THICK_PT * SCALE)

# Majority: more than half a component's pixels lie under a visible char box.
# Design choice, not a measured optimum: a component is covered iff the
# visible-layer paint owns most of its pixels.
COVER_MAJORITY = 0.5

# Width-CV is unstable on fewer members; same cut Fable used on the same rows.
# Design choice: CV is undefined / noisy below this count.
ROW_CV_MIN_MEMBERS = 4

# Page-local glyph-height median is used only when the page has this many
# covered components. Below that, the corpus median (GLYPH_H_P50_PX). Design
# floor, not a measured percentile: a median of a handful of leftovers is
# not a page scale.
H_MED_MIN_COVERED = 20

# ---------------------------------------------------------------------------
# Corpus-derived cuts (Fable calib, 600 pages, seed 20261013, 1,190,972 glyphs,
# 48,804 text rows). Named for the percentile they are.
# ---------------------------------------------------------------------------
GLYPH_H_P02_PX = 2.0  # glyph height p2 of covered glyphs
GLYPH_H_P50_PX = 12.0  # median covered-glyph height
GLYPH_H_P98_PX = 19.0  # glyph height p98
GLYPH_W_P02_PX = 3.0
WORD_W_P50_PX = 48.0  # median native word width
WORD_W_P98_PX = 135.0  # a residual blob may be a blurred word
# Valley of the covered-glyph height histogram between the punctuation mode
# (3 px) and the letter mode (12 px).
LETTER_H_PX = 6.0
ROW_FILL_P01 = 0.327  # median fill of real text rows, p1
ROW_FILL_P99 = 0.917  # p99; solids sit at 1.0. ``_MIN_FILL_FRAC`` (0.55) is
# image_locate's solid-rule floor and sits inside this envelope.
ROW_H_RATIO_P99 = 1.438  # row-box height / tallest glyph
ROW_CV_W_P01 = 0.120  # width coefficient of variation, p1

VERDICT_PASSED = "passed"
VERDICT_FLAGGED = "flagged"
VERDICT_UNKNOWN = "unknown"

UNCOVERED_TEXT_NOTE = "uncovered text — see image"
UNCOVERED_TEXT_NOTE_NO_IMAGE = "uncovered text — image unavailable"


def uncovered_text_marker(page_num: int) -> str:
    """Visible mark, same shape as #1048's ``[page N failed: … — see image]``."""
    return f"[page {page_num}: {UNCOVERED_TEXT_NOTE}]"


def uncovered_text_marker_no_image(page_num: int) -> str:
    """Fallback mark when apply never saved a region crop."""
    return f"[page {page_num}: {UNCOVERED_TEXT_NOTE_NO_IMAGE}]"


def uncovered_text_image_ref(page_num: int, region_i: int) -> str:
    return (
        f"![Uncovered text page {page_num} region {region_i}]"
        f"(figures/uncovered_text_p{page_num}_r{region_i}.png)"
    )


def uncovered_text_png_name(page_num: int, region_i: int) -> str:
    return f"uncovered_text_p{page_num}_r{region_i}.png"


@dataclass
class CoverageRegion:
    pdf: tuple[float, float, float, float]
    px: tuple[int, int, int, int]
    glyphs: int
    band: str
    kind: str
    #: #1050 step 2: how many pre-merge regions this one was built from.
    merged_from: int = 1
    #: #1050 step 2 (owner ruling 2026-10-14): the box of the FIGURE these
    #: regions sit in, when they sit in one. The crop is then the whole figure,
    #: not the ink: a lone "Quarters" label cut out of its chart cannot be
    #: checked by eye. ``None`` for a region that is not in a figure.
    figure_pdf: tuple[float, float, float, float] | None = None


@dataclass
class CoveragePage:
    verdict: str
    regions: list[CoverageRegion] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0

    @property
    def kept(self) -> list[CoverageRegion]:
        return list(self.regions)


def _render_gray(page):
    import cv2
    import fitz

    mat = fitz.Matrix(SCALE, SCALE)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else arr[:, :, 0]
    return gray, mat


def _polarity_masks(gray):
    """Dark-on-light and light-on-dark ink, same ADAPT_BLOCK / ADAPT_C.

    Each mask is envelope-filtered on its own. OR-ing the pixels first
    thickens dark glyphs with a light halo and clears real rows; a
    page-global mid-grey switch hid black lettering on a mid or dark fill.
    """
    import cv2

    inv = cv2.bitwise_not(gray)
    dark = cv2.adaptiveThreshold(
        inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ADAPT_BLOCK, -ADAPT_C
    )
    light = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ADAPT_BLOCK, -ADAPT_C
    )
    return dark > 0, light > 0


def _to_pix(rect, page, mat):
    import fitz

    r = (fitz.Rect(rect) * page.rotation_matrix) * mat
    return int(np.floor(r.x0)), int(np.floor(r.y0)), int(np.ceil(r.x1)), int(np.ceil(r.y1))


def _from_pix(x0, y0, x1, y1, page, mat):
    import fitz

    r = fitz.Rect(x0, y0, x1, y1) * ~mat * page.derotation_matrix
    r.normalize()
    return (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))


def visible_char_boxes(page) -> tuple[list, list]:
    """(visible, invisible) char rects. #1099: a span is visible iff alpha ≠ 0."""
    vis, inv = [], []
    raw = page.get_text("rawdict") or {}
    for block in raw.get("blocks", ()):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", ()):
            for span in line.get("spans", ()):
                if not isinstance(span, dict):
                    continue
                target = vis if span.get("alpha", 255) != 0 else inv
                for ch in span.get("chars", ()):
                    c = ch.get("c")
                    bbox = ch.get("bbox")
                    if not c or c.isspace() or not bbox or len(bbox) < 4:
                        continue
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    target.append(bbox)
    return vis, inv


def _paint(shape, rects, page, mat):
    m = np.zeros(shape, dtype=bool)
    h, w = shape
    for r in rects:
        x0, y0, x1, y1 = _to_pix(r, page, mat)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def _link_rows(gx0, gy0, gw, gh, gap):
    gx1, gy1 = gx0 + gw, gy0 + gh
    order = np.argsort(gx0)
    parent = list(range(len(gx0)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ii in range(len(order)):
        a = order[ii]
        for jj in range(ii + 1, len(order)):
            b = order[jj]
            if gx0[b] - gx1[a] > gap:
                break
            ov = min(gy1[a], gy1[b]) - max(gy0[a], gy0[b])
            if ov >= 0.5 * min(gh[a], gh[b]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    rows: dict[int, list[int]] = collections.defaultdict(list)
    for k in range(len(gx0)):
        rows[find(k)].append(k)
    return [np.array(m) for m in rows.values()]


def _row_shape(m, gx0, gy0, gw, gh, fill):
    bx0, by0 = gx0[m].min(), gy0[m].min()
    bx1, by1 = (gx0[m] + gw[m]).max(), (gy0[m] + gh[m]).max()
    max_h = gh[m].max()
    return (
        float(np.median(fill[m])),
        float((by1 - by0) / max(1, max_h)),
        int(bx1 - bx0),
        int(max_h),
        (int(bx0), int(by0), int(bx1), int(by1)),
    )


def _locate_band(page, mat, px) -> str:
    try:
        text_rects = [
            _to_pix(b[:4], page, mat)
            for b in page.get_text("blocks")
            if len(b) > 6 and b[6] == 0 and str(b[4]).strip()
        ]
    except Exception:
        text_rects = []
    top = min((r[1] for r in text_rects), default=None)
    bot = max((r[3] for r in text_rects), default=None)
    cy = (px[1] + px[3]) / 2
    if top is not None and cy < top:
        return "above_text"
    if bot is not None and cy > bot:
        return "below_text"
    return "body"


def _page_min_run(page, mat, labels, covered_set) -> int | None:
    """Astra's same-page floor: ceil(median native-word component count).

    A native word's component count is how many *covered* connected components
    have their box overlapping that word. Derived on this page, not a global
    pixel width. ``None`` when the page has no scored word (fall back to
    Fable's corpus ``WORD_W_P50_PX``).
    """
    words = page.get_text("words") or []
    if not words or not covered_set:
        return None
    h, w = labels.shape
    counts: list[int] = []
    for wd in words:
        if len(wd) < 5 or not str(wd[4]).strip():
            continue
        x0, y0, x1, y1 = _to_pix(wd[:4], page, mat)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        n = sum(1 for i in np.unique(labels[y0:y1, x0:x1]) if int(i) in covered_set)
        if n:
            counts.append(n)
    if not counts:
        return None
    return int(math.ceil(float(np.median(counts))))


def _accept_runs(gx0, gy0, gw, gh, gfill, h_med, min_run, *, vertical: bool) -> list:
    """Envelope-filter linked runs. *vertical* swaps axes so a column is a word."""
    if vertical:
        linked = _link_rows(gy0, gx0, gh, gw, h_med)
    else:
        linked = _link_rows(gx0, gy0, gw, gh, h_med)
    out = []
    for m in linked:
        if len(m) < 2:
            continue
        if vertical:
            mf, hr, width, max_h, box = _row_shape(m, gy0, gx0, gh, gw, gfill)
            box = (box[1], box[0], box[3], box[2])
            sizes = gh[m]
        else:
            mf, hr, width, max_h, box = _row_shape(m, gx0, gy0, gw, gh, gfill)
            sizes = gw[m]
        # Keep a row if it looks like a word either way: Astra's
        # page-local min_run (narrow figure labels; Golosov) OR
        # Fable's corpus median word width (wide short runs;
        # Zheng / Andrade). Replacing the width cut with min_run
        # alone dropped those two Fable T pages.
        if min_run is not None:
            if len(m) < min_run and width < WORD_W_P50_PX:
                continue
        elif width < WORD_W_P50_PX:
            continue
        if max_h < LETTER_H_PX:
            continue
        if not (ROW_FILL_P01 <= mf <= ROW_FILL_P99):
            continue
        if hr > ROW_H_RATIO_P99:
            continue
        if len(m) >= ROW_CV_MIN_MEMBERS:
            cv = float(np.std(sizes) / max(1e-9, np.mean(sizes)))
            if cv < ROW_CV_W_P01:
                continue
        out.append([*box, len(m)])
    return out


def _locate_kind(page, mat, px) -> str:
    try:
        rasters = []
        for info in page.get_image_info():
            if info.get("bbox"):
                rasters.append(_to_pix(info["bbox"], page, mat))
    except Exception:
        return "unknown"
    if rasters:
        best = 0.0
        ax0, ay0, ax1, ay1 = px
        area = max(1, (ax1 - ax0) * (ay1 - ay0))
        for bx0, by0, bx1, by1 in rasters:
            ix = max(0, min(ax1, bx1) - max(ax0, bx0))
            iy = max(0, min(ay1, by1) - max(ay0, by0))
            best = max(best, ix * iy / area)
        if best >= COVER_MAJORITY:
            return "raster"
    return "vector_or_bare"


def _regions_from_ink(page, mat, ink, layer) -> list[CoverageRegion]:
    """Envelope-filter one polarity's ink into coverage regions."""
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    hit = np.bincount(labels[layer], minlength=n).astype(np.float64)
    tot = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(tot > 0, hit / tot, 0.0)
    x, y, w, h, area = (stats[:, i] for i in range(5))
    idx = np.arange(1, n)
    resid_idx = idx[frac[idx] < COVER_MAJORITY]
    covered_idx = idx[frac[idx] >= COVER_MAJORITY]
    h_med = (
        float(np.median(h[covered_idx]))
        if len(covered_idx) >= H_MED_MIN_COVERED
        else GLYPH_H_P50_PX
    )
    rh, rw = h[resid_idx], w[resid_idx]
    # page.rotation swaps pixmap axes. Filter on the orientation-
    # invariant pair so a 16 pt Helvetica-Oblique raster that flags
    # at 0° also flags at 90/270. Raw rh/rw clears the rotated case.
    short = np.minimum(rh, rw)
    long = np.maximum(rh, rw)
    glyph_like = (
        (short >= GLYPH_H_P02_PX)
        & (short <= GLYPH_H_P98_PX)
        & (long >= GLYPH_W_P02_PX)
        & (long <= WORD_W_P98_PX)
    )
    thin = short <= MAX_THICK_PX
    long_ = long > GLYPH_H_P98_PX
    glyph_like &= ~(thin & long_)
    # Solids that fill more of their box than ``_MIN_FILL_FRAC`` and are
    # thicker than a rule are still glyphs when they sit in the height band;
    # the fill envelope below drops dashed-line rows, not this cut.
    _ = _MIN_FILL_FRAC
    g = resid_idx[glyph_like]
    min_run = _page_min_run(page, mat, labels, set(int(i) for i in covered_idx))
    regions: list[CoverageRegion] = []
    if len(g) < 2:
        return regions
    gx0, gy0, gw, gh = x[g], y[g], w[g], h[g]
    gfill = area[g] / np.maximum(1, gw * gh)
    row_boxes = []
    row_boxes.extend(_accept_runs(gx0, gy0, gw, gh, gfill, h_med, min_run, vertical=False))
    # page.rotation 90/270 turns a horizontal label into a column in
    # the pixmap. Vertical linking on *upright* pages treats rules
    # and spines as words (production silent flags 101 → 1096).
    # Upright y-axis labels stay a follow-up (Ben–Gunn–Khan p7).
    if getattr(page, "rotation", 0) % 360 in (90, 270):
        row_boxes.extend(_accept_runs(gx0, gy0, gw, gh, gfill, h_med, min_run, vertical=True))
    rparent = list(range(len(row_boxes)))

    def rfind(a):
        while rparent[a] != a:
            rparent[a] = rparent[rparent[a]]
            a = rparent[a]
        return a

    for i in range(len(row_boxes)):
        for j in range(i + 1, len(row_boxes)):
            a, b = row_boxes[i], row_boxes[j]
            vgap = max(a[1], b[1]) - min(a[3], b[3])
            xov = min(a[2], b[2]) - max(a[0], b[0])
            if vgap <= h_med and xov > 0:
                ra, rb = rfind(i), rfind(j)
                if ra != rb:
                    rparent[rb] = ra
    blocks: dict[int, list] = collections.defaultdict(list)
    for i in range(len(row_boxes)):
        blocks[rfind(i)].append(row_boxes[i])
    for members in blocks.values():
        bx0 = min(r[0] for r in members)
        by0 = min(r[1] for r in members)
        bx1 = max(r[2] for r in members)
        by1 = max(r[3] for r in members)
        px = (int(bx0), int(by0), int(bx1), int(by1))
        pdf = _from_pix(*px, page, mat)
        regions.append(
            CoverageRegion(
                pdf=pdf,
                px=px,
                glyphs=int(sum(r[4] for r in members)),
                band=_locate_band(page, mat, px),
                kind=_locate_kind(page, mat, px),
            )
        )
    return regions


def _page_fraction(pdf_box, page) -> float | None:
    """Fraction of the page *pdf_box* covers, in PDF points."""
    try:
        rect = page.rect
        page_area = float(rect.width) * float(rect.height)
    except Exception:
        return None
    if page_area <= 0:
        return None
    x0, y0, x1, y1 = pdf_box
    return (max(0.0, x1 - x0) * max(0.0, y1 - y0)) / page_area


def _drawn_marks_px(page, mat) -> list[tuple[int, int, int, int]]:
    """Every drawn path's box on the page, in pixmap coordinates."""
    out: list[tuple[int, int, int, int]] = []
    try:
        drawings = page.get_drawings()
    except Exception as exc:
        logger.debug("text coverage: get_drawings failed (%s)", exc)
        return out
    for d in drawings:
        rect = d.get("rect")
        if rect is None or rect.is_empty:
            continue
        out.append(_to_pix(rect, page, mat))
    return out


def _cluster_own_extent(box, marks) -> tuple[int, int, int, int] | None:
    """Shrink *box* to the union of the drawn marks inside it.

    #1050 step 2 round 2, item A. ``chart_region_bboxes`` expands a cluster to
    the page's LEFT EDGE and by a row margin, so on a two-column page a
    right-hand chart's rect reaches across the left column. Cropping that, and
    reading it, duplicates the left column's prose. The figure's own extent is
    what it actually drew.
    """
    inside = [m for m in marks if _inside(m, box)]
    if not inside:
        return None
    return (
        min(m[0] for m in inside),
        min(m[1] for m in inside),
        max(m[2] for m in inside),
        max(m[3] for m in inside),
    )


def _native_prose_inside(page, mat, box, marks) -> bool:
    """True when a native word sits in *box* but on none of its drawn marks.

    Item A, second half. A framed callout is a drawn frame around ORDINARY
    PROSE; a chart's own labels sit on its marks (or are drawn, and so have no
    native word at all). A word floating inside the box and touching nothing
    drawn is the page's text, and grouping it into a figure crop would ship it
    twice.
    """
    try:
        words = page.get_text("words") or []
    except Exception as exc:
        logger.debug("text coverage: words unavailable for figure test (%s)", exc)
        return True
    from socr.tables.source_evidence import plain_text_tokens

    for wd in words:
        if len(wd) < 5 or not str(wd[4]).strip():
            continue
        # Only a CONTENT word counts, by socr's existing definition (the table
        # witness's content-token rule). Measured on Hansen p23: the only
        # native words floating inside that chart are the panel labels "(A)"
        # and "(B)", which are figure furniture, not prose. A framed callout's
        # body is a sentence, and every word of it is a content token.
        if not plain_text_tokens(str(wd[4]))[1]:
            continue
        wx0, wy0, wx1, wy1 = _to_pix(wd[:4], page, mat)
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        if not (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]):
            continue
        # A FRAME is hollow, but its box covers everything it encloses, so
        # every word inside a framed callout would look like it sits "on a
        # mark". A mark that contains the whole cluster is the frame; it is
        # skipped. Exact containment, not a coverage fraction.
        if not any(
            m[0] <= cx <= m[2] and m[1] <= cy <= m[3] and not _inside(box, m) for m in marks
        ):
            return True
    return False


def figure_boxes_px(page, mat, region_px=()) -> list[tuple[int, int, int, int]]:
    """The page's FIGURE boxes, in pixmap coordinates (#1050 step 2).

    Two halves, both from detectors socr already ships:

    * **Raster** (``figures.extractor.figure_boxes``): a placed image IS a
      figure by definition and its bbox IS the picture. Nothing further is
      asked of it -- a raster bar chart has no vector marks to find, and
      requiring them shipped a label strip without its bars (round 2, item B).
    * **Vector** (``tables.reconstruct.chart_region_bboxes``): used only to
      LOCATE a cluster. Its rect is expanded to the page's left edge and by a
      row margin, so the figure's own extent is recomputed from the marks
      inside it (item A).

    A vector cluster must then look like a chart rather than like drawn text or
    a framed callout: it must carry marks away from the uncovered ink, and no
    native word may float inside it touching nothing drawn.
    """
    boxes: list[tuple[int, int, int, int]] = []
    try:
        from socr.figures.extractor import figure_boxes

        for bbox in figure_boxes(page) or ():
            boxes.append(_to_pix(bbox, page, mat))
    except Exception as exc:
        logger.debug("text coverage: raster figure boxes unavailable (%s)", exc)
    marks = _drawn_marks_px(page, mat)
    try:
        from socr.tables.reconstruct import chart_region_bboxes

        for rect in chart_region_bboxes(page) or ():
            candidate = _to_pix((rect.x0, rect.y0, rect.x1, rect.y1), page, mat)
            own = _cluster_own_extent(candidate, marks)
            if own is None:
                continue
            if not _cluster_has_marks_beyond_the_ink(own, marks, region_px):
                continue
            if _native_prose_inside(page, mat, own, marks):
                continue
            boxes.append(own)
    except Exception as exc:
        logger.debug("text coverage: vector chart boxes unavailable (%s)", exc)
    return boxes


def _cluster_has_marks_beyond_the_ink(box, marks, region_px) -> bool:
    """True when *box* contains drawn paths that are NOT the uncovered text itself.

    ``chart_region_bboxes`` clusters any drawing, so it cannot tell a chart
    from a box of VECTOR-DRAWN TEXT: measured 2026-10-14, it returns one
    cluster for Hansen p23's chart AND for Schoenegger p10's prompt box, which
    the owner was explicit about ("we should not read that box as an image...
    it's just text"). ``has_chart_marks`` is False on both.

    The separation needs no new detector and no threshold, because the ink is
    already segmented: a chart has drawn marks (axes, plot lines, ticks)
    OUTSIDE its uncovered-text regions, while a box of drawn text has nothing
    in it but the letters. A rule-thin mark counts only when it SPANS the
    cluster (item B): a gridline is a mark, a page rule is not.
    """
    inside_regions = [r for r in region_px if _inside(r, box)]
    for px in marks:
        if not _inside(px, box):
            continue
        # A rule is not a plot mark (``MAX_THICK_PX`` is image_locate's own rule
        # thickness, already used by the ink filter). Tested on the SHORT side,
        # so a data polyline -- which rises and falls -- is never rule-thin;
        # only a perfectly flat line is, and that is an axis. A spans-the-box
        # exemption was tried and reverted: a framed text box's divider rule
        # spans it too, and it flipped Schoenegger p10 to "figure".
        if min(px[2] - px[0], px[3] - px[1]) <= MAX_THICK_PX:
            continue
        if not any(
            px[0] < r[2] and px[2] > r[0] and px[1] < r[3] and px[3] > r[1] for r in inside_regions
        ):
            return True
    return False


def _container_boxes(page, mat) -> list[tuple[int, int, int, int]]:
    """Drawn rectangles and raster placements, in pixmap coordinates.

    #1050 step 2. These are the page's own statements about what belongs
    together: a framed prompt box is one drawn rectangle (Schoenegger p10),
    a pasted newspaper clipping is one placed raster (Acosta p21). Nothing
    here is a threshold -- a container is whatever the PDF drew.
    """
    from socr.figures.crop_descriptions import is_page_sized

    boxes: list[tuple[int, int, int, int]] = []
    try:
        for d in page.get_drawings():
            rect = d.get("rect")
            if rect is None or rect.is_empty:
                continue
            # A page-sized DRAWN rect is page furniture -- a border, a
            # background fill -- and contains every region by construction, so
            # it would merge unrelated ink across the prose between it. A
            # page-sized RASTER is not furniture: it is one placed image, and
            # everything inside it came from that one object (the Acosta
            # clipping's scan is placed over 85% of the page). Hence the
            # asymmetry: drawn containers must be smaller than the page,
            # rasters need not be.
            if is_page_sized(_page_fraction((rect.x0, rect.y0, rect.x1, rect.y1), page)):
                continue
            boxes.append(_to_pix(rect, page, mat))
    except Exception as exc:  # a malformed content stream must not fail the page
        logger.debug("text coverage: get_drawings failed (%s)", exc)
    try:
        for info in page.get_image_info():
            bbox = info.get("bbox")
            if bbox:
                boxes.append(_to_pix(bbox, page, mat))
    except Exception as exc:
        logger.debug("text coverage: get_image_info failed (%s)", exc)
    return boxes


def _inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _swallows_native_text(box, native_px) -> bool:
    """True when *box* overlaps a VISIBLE native character box.

    #1050 step 2, item B. Two uncovered bands that sit above and below a native
    line merge into a crop containing that line. The model then reads it, and
    the page ships that line twice -- once as native prose, once as a recovered
    block. The merge is refused instead, and the members stay separate: a few
    extra crops cost nothing, a duplicated line is wrong text.
    """
    x0, y0, x1, y1 = box
    for cx0, cy0, cx1, cy1 in native_px:
        if cx0 < x1 and cx1 > x0 and cy0 < y1 and cy1 > y0:
            return True
    return False


def merge_regions_by_figure(page, mat, regions: list[CoverageRegion]) -> list[CoverageRegion]:
    """Owner ruling 2026-10-14: regions inside ONE figure are one region.

    Before this, a chart's labels each became their own crop -- 23 on Hansen
    et al. 2018 p23, 18 on Bugel et al. p23, almost all single labels such as
    "Quarters" or "Top 1% share". A label cut out of its chart cannot be
    checked by eye, which defeats keeping the picture at all.

    The merged region's crop is the WHOLE FIGURE (``figure_pdf``), not the ink
    union, and the model reads that one crop instead of one per label. The
    native-text guard that applies to other containers is deliberately NOT
    applied here: a chart's own axis labels have a text layer, and excluding
    every figure that contains one would leave nothing.
    """
    if not regions:
        return regions
    boxes = figure_boxes_px(page, mat, [r.px for r in regions])
    if not boxes:
        return regions
    claimed: set[int] = set()
    merged: list[CoverageRegion] = []
    for box in boxes:
        members = [i for i, reg in enumerate(regions) if i not in claimed and _inside(reg.px, box)]
        if not members:
            continue
        claimed.update(members)
        px = (
            min(regions[i].px[0] for i in members),
            min(regions[i].px[1] for i in members),
            max(regions[i].px[2] for i in members),
            max(regions[i].px[3] for i in members),
        )
        merged.append(
            CoverageRegion(
                pdf=_from_pix(*px, page, mat),
                px=px,
                glyphs=sum(regions[i].glyphs for i in members),
                band=_locate_band(page, mat, px),
                kind=_locate_kind(page, mat, px),
                merged_from=sum(regions[i].merged_from for i in members),
                figure_pdf=_from_pix(*box, page, mat),
            )
        )
    if not claimed:
        return regions
    merged.extend(regions[i] for i in range(len(regions)) if i not in claimed)
    merged.sort(key=lambda r: (r.px[1], r.px[0]))
    return merged


def merge_regions_by_container(
    page, mat, regions: list[CoverageRegion], native_px=()
) -> list[CoverageRegion]:
    """#1050 step 2: one coherent uncovered block is ONE region.

    Step 1 cut Schoenegger p10's framed prompt box into 4 crops and Acosta
    p21's pasted clipping into 7, because it links rows only when their
    vertical gap is under a glyph height: a paragraph break or a column gap
    inside one block starts a new region. Grouping is taken from the page's
    own geometry instead of from a new gap threshold: two regions that lie
    inside the SAME drawn rectangle or the SAME placed raster are one block.

    The merged box is the union of the member regions, never the container.
    Acosta's clipping raster is placed over 85% of the page because the scan
    carries wide white margins; its ink is 5%. Cropping the container would
    hand the model the whole page, native prose included.

    Three refusals, each falling back to the separate member regions:

    * the union would be page-sized (``PAGE_SIZED_BBOX_FRACTION``, measured for
      crop descriptions), so no merge can produce a whole-page crop;
    * the container is a page-sized DRAWN rect -- furniture, not a box (handled
      in ``_container_boxes``; a page-sized raster IS still a container);
    * the union would overlap a visible native character box, which would hand
      the model a line the page already ships and have it ship twice.
    """
    if len(regions) < 2:
        return regions
    from socr.figures.crop_descriptions import is_page_sized

    containers = _container_boxes(page, mat)
    if not containers:
        return regions
    parent = list(range(len(regions)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for box in containers:
        members = [i for i, reg in enumerate(regions) if _inside(reg.px, box)]
        if len(members) < 2:
            continue
        ux0 = min(regions[i].px[0] for i in members)
        uy0 = min(regions[i].px[1] for i in members)
        ux1 = max(regions[i].px[2] for i in members)
        uy1 = max(regions[i].px[3] for i in members)
        union_pdf = _from_pix(ux0, uy0, ux1, uy1, page, mat)
        if is_page_sized(_page_fraction(union_pdf, page)):
            continue
        if _swallows_native_text((ux0, uy0, ux1, uy1), native_px):
            continue
        root = find(members[0])
        for i in members[1:]:
            other = find(i)
            if other != root:
                parent[other] = root
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(len(regions)):
        groups[find(i)].append(i)
    merged: list[CoverageRegion] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(regions[members[0]])
            continue
        px = (
            min(regions[i].px[0] for i in members),
            min(regions[i].px[1] for i in members),
            max(regions[i].px[2] for i in members),
            max(regions[i].px[3] for i in members),
        )
        pdf = _from_pix(*px, page, mat)
        merged.append(
            CoverageRegion(
                pdf=pdf,
                px=px,
                glyphs=sum(regions[i].glyphs for i in members),
                band=_locate_band(page, mat, px),
                kind=_locate_kind(page, mat, px),
                merged_from=sum(regions[i].merged_from for i in members),
            )
        )
    merged.sort(key=lambda r: (r.px[1], r.px[0]))
    return merged


#: #1050 step 2: how many trailing words of the preceding native block are used
#: to find the insertion point in the shipped page body. A MATCH WINDOW, not a
#: quality threshold: long enough that the snippet is unlikely to occur twice on
#: one page, short enough to survive the native extractor's paragraph rejoining.
#: A snippet that is not found makes the caller fall back, never guess.
ANCHOR_TAIL_WORDS = 8


def preceding_text_snippet(page, region_pdf, *, tail_words: int = ANCHOR_TAIL_WORDS) -> str:
    """Tail of the last native text block that ends above *region_pdf*.

    #1050 step 2 places a recovered block at its reading-order position
    (2026-10-09 ruling). The page's own text blocks say what precedes the
    region; the caller locates this snippet in the shipped body. ``""`` means
    nothing precedes it (the region is above all native text).
    """
    try:
        blocks = [
            b for b in page.get_text("blocks") if len(b) > 6 and b[6] == 0 and str(b[4]).strip()
        ]
    except Exception as exc:
        logger.debug("text coverage: anchor blocks unavailable (%s)", exc)
        return ""
    x0, top, x1 = region_pdf[0], region_pdf[1], region_pdf[2]
    # Item K: on a two-column page every right-column block also "ends above" a
    # left-column region, and the lowest of them wins -- so the left column's
    # words were anchored to the right column's text. A block only precedes the
    # region if it shares some of the region's horizontal span. No threshold:
    # any overlap at all counts, and a region that overlaps nothing above it
    # has no anchor and goes to the page tail rather than to a guess.
    above = [b for b in blocks if b[3] <= top and b[0] < x1 and b[2] > x0]
    if not above:
        return ""
    best = max(above, key=lambda b: (b[3], b[1]))
    words = str(best[4]).split()
    return " ".join(words[-tail_words:])


def analyze_page(page, *, group: bool = True) -> CoveragePage:
    """Ink-vs-layer check on one page. Never raises; failures are ``unknown``.

    ``group=False`` returns step 1's regions untouched. Grouping exists to make
    ONE crop readable; with no reader there is nothing to read, and step 2 must
    then be byte-identical to main (round 2, item E).
    """
    import time

    t0 = time.perf_counter()
    try:
        gray, mat = _render_gray(page)
        vis, _inv = visible_char_boxes(page)
        dark, light = _polarity_masks(gray)
        layer = _paint(dark.shape, vis, page, mat)
        regions: list[CoverageRegion] = []
        seen: set[tuple[float, float, float, float]] = set()
        for ink in (dark, light):
            for reg in _regions_from_ink(page, mat, ink, layer):
                if reg.pdf in seen:
                    continue
                seen.add(reg.pdf)
                regions.append(reg)
        if group:
            native_px = [_to_pix(b, page, mat) for b in vis]
            # Figures first: a figure is the strongest statement the page makes
            # about what belongs together, and its crop is the whole figure.
            regions = merge_regions_by_figure(page, mat, regions)
            regions = merge_regions_by_container(page, mat, regions, native_px)
        return CoveragePage(
            verdict=VERDICT_FLAGGED if regions else VERDICT_PASSED,
            regions=regions,
            seconds=round(time.perf_counter() - t0, 3),
        )
    except Exception as exc:
        logger.debug("text coverage: page failed closed (%s)", exc)
        return CoveragePage(
            verdict=VERDICT_UNKNOWN,
            error=f"{type(exc).__name__}: {exc}",
            seconds=round(time.perf_counter() - t0, 3),
        )


def coverage_for_document(doc, *, group: bool = True) -> list[CoveragePage]:
    """Per-page verdicts for a whole PDF. No furniture suppression."""
    return [analyze_page(doc[i], group=group) for i in range(len(doc))]


def _fill_with_local_background(arr, paint, protect):
    """Replace the pixels in *paint* with the colour that surrounds them.

    Round 5, item 1. Painting covered glyphs WHITE was itself a signal: on a
    grey or dark fill -- a chart exported as a raster with native labels over
    it -- the model was handed fat white words and read them, so the mask
    created the duplication it exists to prevent. Measured with Fable's metric
    (contrast pixels inside the native char boxes, against the fill): white
    painting left 395% on a 0.6 grey and 389% on a dark fill.

    Each covered component is filled with the MEDIAN of the pixels in a ring
    around it -- a per-component estimate, so a gradient is followed rather
    than flattened. The mask is first dilated by ``MAX_THICK_PX``
    (image_locate's rule thickness, already used by the ink filter, so no new
    number) because a glyph's anti-aliased halo lies just outside the component
    and reads as the glyph's own edge if it is left behind.
    """
    import cv2

    radius = max(1, int(round(MAX_THICK_PX)))
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    solid = cv2.dilate(paint.astype(np.uint8), kernel).astype(bool) & ~protect
    solid |= paint
    out = arr.copy()
    count, labels, stats, _c = cv2.connectedComponentsWithStats(
        solid.astype(np.uint8), connectivity=8
    )
    height, width = solid.shape
    for i in range(1, count):
        x, y, w, h = (int(stats[i, j]) for j in range(4))
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1, y1 = min(width, x + w + radius), min(height, y + h + radius)
        window = out[y0:y1, x0:x1]
        ring = ~solid[y0:y1, x0:x1]
        if not ring.any():
            continue
        out[y0:y1, x0:x1][labels[y0:y1, x0:x1] == i] = np.median(window[ring], axis=0)
    return out


def native_ink_masked(page, mat, arr):
    """*arr* with the page's NATIVE INK painted out, component by component.

    Round 4, items 1 and 2. Two things this must get right, and the first
    version got both wrong:

    * **Every char the shipped prose is built from**, visible AND invisible. An
      alpha-0 OCR layer over a raster is not drawn, but ``get_text`` returns it
      and the native text ships it, so leaving it unpainted duplicated the
      sentence (recovery ON 2, OFF 1).
    * **The ink, not the box.** A char box spans ascender to descender, so
      painting boxes erased uncovered ink just below a baseline -- a drawn
      "0.058" came back as "U.UD& arawn pelow". What is painted is what step 1
      already classes as COVERED: a connected ink component whose pixels lie
      mostly under a native char box (``COVER_MAJORITY``). Uncovered ink -- the
      very thing recovery exists to read -- is never painted.

    Falls back to the unmasked array only for an empty page; any failure is the
    caller's to handle, because reading an unmasked crop reopens duplication.
    """
    import cv2

    out = arr.copy()
    visible, invisible = visible_char_boxes(page)
    native_boxes = [*visible, *invisible]
    if not native_boxes:
        return out
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY) if out.shape[2] >= 3 else out[:, :, 0]
    layer = _paint(gray.shape, native_boxes, page, mat)
    paint = np.zeros(gray.shape, dtype=bool)
    protect = np.zeros(gray.shape, dtype=bool)
    for ink in _polarity_masks(gray):
        count, labels, _stats, _c = cv2.connectedComponentsWithStats(
            ink.astype(np.uint8), connectivity=8
        )
        if count < 2:
            continue
        hit = np.bincount(labels[layer], minlength=count).astype(np.float64)
        total = np.bincount(labels.ravel(), minlength=count).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(total > 0, hit / total, 0.0)
        covered = np.zeros(count, dtype=bool)
        covered[1:] = frac[1:] >= COVER_MAJORITY
        paint |= covered[labels]
        uncovered = np.zeros(count, dtype=bool)
        uncovered[1:] = frac[1:] < COVER_MAJORITY
        protect |= uncovered[labels]
    if paint.any():
        out = _fill_with_local_background(out, paint, protect & ~paint)
    return out


def crop_box(region: CoverageRegion) -> tuple[float, float, float, float]:
    """The box to CROP for *region*: the whole figure when it is in one.

    One place decides this, because the saved picture and the model's crop
    must come from the same box: a reader checking words against a different
    crop than the model read is checking the wrong thing.
    """
    return region.figure_pdf or region.pdf


def render_region_png(
    page, pdf_rect: tuple[float, float, float, float], dest, *, mask_native: bool = False
) -> bool:
    """Write a DETECT_DPI crop of *pdf_rect* to *dest*.

    Uses the same pixmap transform as ``analyze_page`` (scale, then
    ``page.rotation`` via ``_to_pix``). Clipping unrotated coords on a
    rotated render produced an all-white 180° crop.

    ``mask_native=True`` paints out every VISIBLE native character box first
    (#1050 step 2, round 3). This decouples what the model READS from what the
    reader SEES: the model is handed a crop containing only ink the text layer
    does not already carry, so it cannot repeat a sentence the page already
    ships, whatever the figure detection decided. The picture that SHIPS is
    rendered without the mask. The whole duplication class -- a raster placed
    under prose, a callout with an icon, a two-column chart box -- is closed
    here rather than by chasing shapes.
    """
    import cv2
    import fitz

    mat = fitz.Matrix(SCALE, SCALE)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if mask_native:
        arr = native_ink_masked(page, mat, arr)
    x0, y0, x1, y1 = _to_pix(pdf_rect, page, mat)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(pix.width, x1), min(pix.height, y1)
    crop = arr[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else arr
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = bool(cv2.imwrite(str(dest), cv2.cvtColor(crop[:, :, :3], cv2.COLOR_RGB2BGR)))
    if not ok:
        raise OSError(f"cv2.imwrite returned False for {dest}")
    return True
