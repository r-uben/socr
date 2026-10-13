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


def _ink_mask(gray):
    """Union of both polarities, for callers that only need a pixel mask."""
    dark, light = _polarity_masks(gray)
    return dark | light


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


def analyze_page(page) -> CoveragePage:
    """Ink-vs-layer check on one page. Never raises; failures are ``unknown``."""
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


def coverage_for_document(doc) -> list[CoveragePage]:
    """Per-page verdicts for a whole PDF. No furniture suppression."""
    return [analyze_page(doc[i]) for i in range(len(doc))]


def render_region_png(page, pdf_rect: tuple[float, float, float, float], dest) -> bool:
    """Write a DETECT_DPI crop of *pdf_rect* to *dest*.

    Uses the same pixmap transform as ``analyze_page`` (scale, then
    ``page.rotation`` via ``_to_pix``). Clipping unrotated coords on a
    rotated render produced an all-white 180° crop.
    """
    import cv2
    import fitz

    mat = fitz.Matrix(SCALE, SCALE)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    x0, y0, x1, y1 = _to_pix(pdf_rect, page, mat)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(pix.width, x1), min(pix.height, y1)
    crop = arr[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else arr
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = bool(cv2.imwrite(str(dest), cv2.cvtColor(crop[:, :, :3], cv2.COLOR_RGB2BGR)))
    if not ok:
        raise OSError(f"cv2.imwrite returned False for {dest}")
    return True
