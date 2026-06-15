"""Deterministic display-equation region detection for born-digital pages.

GH-36a: model-free foundation phase.  Detects display-equation regions using
PyMuPDF geometry — NO whole-page OCR, NO model call.  The results feed the
crop-PNG pipeline (also GH-36a) and eventually the engine/validation layer
(GH-36b, pending consilium decision 20260615T210537Z-6621).

Detection strategy (conservative — a missed equation is acceptable at this
phase; the goal is precise bounding boxes, not recall maximisation):

  1. **Math-font lines** — a text line whose spans are predominantly in a
     recognised math font (CMMI/CMSY/STIX/…) is an equation line.  Reuses
     ``_MATH_FONT_RE`` from ``born_digital``.  Same source as the page-level
     ``has_equations`` signal already computed by ``BornDigitalDetector``.

  2. **Display-equation geometry** — a line (or group of lines) qualifies as
     a *display* equation if it is horizontally centred on the page.  Inline
     equations (mid-sentence math) are excluded because they are embedded in
     prose lines that also contain non-math text.

  3. **Equation numbering** — a flush-right token matching a parenthesised
     number pattern on the same line as math is a strong confirmer
     (``(3.4)``, ``(A.1)``).  A math-font line with a number is promoted even
     if the centering is only approximate.

All tolerance thresholds are **named constants** with documented basis below.
None are bare literals.

Named constants and their basis
--------------------------------
``DISPLAY_MIN_MATH_SPAN_RATIO``
    Minimum fraction of a line's spans (by character count) that must come
    from a recognised math font for the line to count as a *math-font line*.
    Basis: typical displayed equations are typeset entirely in math fonts;
    inline math mid-sentence shares the line with roman text and may fall
    below this ratio.  0.50 keeps inline math out while admitting display
    equations (including those with short prose labels on the same line such
    as "Let  …  =").  Conservative: a missed equation is preferable to a
    false-positive prose crop.

``DISPLAY_CENTER_TOLERANCE_PT``
    Maximum deviation (in points) of a candidate line's horizontal midpoint
    from the page text block's horizontal midpoint before the line is
    considered off-centre.  Basis: typical A4/letter page text block is
    ~420 pt wide; a tolerance of 20 pt (~5 % of that) admits equations
    centred within a column margin without also admitting left-aligned prose.
    Equations with numbering flush-right may skew the apparent centre of the
    *typeset* region — the tolerance is applied to the *text line bbox*, not
    the visual equation, so narrow-but-centred equation lines pass easily.

``DISPLAY_MIN_HEIGHT_PT``
    Minimum bounding-box height (in points) for a candidate display-equation
    region.  Basis: a single-line display equation at 10-12 pt sits in a bbox
    roughly 10-14 pt tall; sub/superscript extents add 2-4 pt.  The threshold
    of 8 pt rejects stray single-pixel artefacts while admitting any real
    typeset line.  After merging adjacent lines a multi-line equation will
    naturally exceed this by a wide margin.

``DISPLAY_MERGE_GAP_PT``
    Vertical gap between consecutive equation lines below which they are merged
    into one region.  Mirrors ``_REGION_MERGE_GAP_PT`` from ``recover.py``
    (6.0 pt) exactly, so multi-line display blocks stay as one crop.

``DISPLAY_PAD_PT``
    Padding added around a merged region before rendering the crop.  Mirrors
    ``_REGION_PAD_PT`` from ``recover.py`` (4.0 pt) so sub/superscripts at
    the edge are not clipped.

``DISPLAY_EQ_NUM_RE``
    Regex matching an equation number in the parenthesised-number convention
    common in academic papers: ``(3)``, ``(3.4)``, ``(A.1)``, ``(3.4.1)``.
    A match anywhere on a candidate line promotes it even if centering is
    approximate.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Detection constants ─────────────────────────────────────────────────────

# Minimum fraction of a line's character count that must come from math-font
# spans for the line to be treated as a math-font line.  See module docstring.
DISPLAY_MIN_MATH_SPAN_RATIO: float = 0.50

# Maximum deviation (points) of a line's horizontal midpoint from the page
# text-block midpoint for the line to be considered "centred".
DISPLAY_CENTER_TOLERANCE_PT: float = 20.0

# Minimum bounding-box height (points) of a merged equation region.
DISPLAY_MIN_HEIGHT_PT: float = 8.0

# Vertical gap (points) between consecutive equation lines that triggers a
# merge (same semantics as ``_REGION_MERGE_GAP_PT`` in ``recover.py``).
DISPLAY_MERGE_GAP_PT: float = 6.0

# Horizontal padding added to every merged region before crop rendering
# (same semantics as ``_REGION_PAD_PT`` in ``recover.py``).
DISPLAY_PAD_PT: float = 4.0

# Equation number pattern: (3), (3.4), (A.1), (3.4.1) — typical in papers.
DISPLAY_EQ_NUM_RE: re.Pattern = re.compile(r"\(\s*[A-Za-z]?\d+(?:\.\d+)*\s*\)")

# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class EquationRegion:
    """One detected display-equation region on a page.

    All coordinates are in PDF points (same coordinate space as fitz.Rect).
    ``bbox`` is the padded crop rectangle; ``source_bbox`` is the tight
    merged line rectangle before padding.
    ``has_eq_number`` is True when an equation-number pattern was detected
    on one of the constituent lines (strong confirmation signal).
    """

    page_num: int  # 1-indexed
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) padded
    source_bbox: tuple[float, float, float, float]  # tight merged rect
    has_eq_number: bool = False
    crop_path: str | None = None  # filled in after crop-PNG is saved


@dataclass
class EquationDetectionResult:
    """Returned by ``detect_display_equations``.

    ``regions`` may be empty on a page without detected display equations.
    ``detection_time_s`` is the wall-clock time for the geometry pass only
    (model-free); this feeds the throughput harness (GH-36a AC5).
    """

    page_num: int
    regions: list[EquationRegion] = field(default_factory=list)
    detection_time_s: float = 0.0


# ── Core detection ─────────────────────────────────────────────────────────


def detect_display_equations(
    page,
    page_num: int,
    *,
    min_math_span_ratio: float = DISPLAY_MIN_MATH_SPAN_RATIO,
    center_tolerance_pt: float = DISPLAY_CENTER_TOLERANCE_PT,
    min_height_pt: float = DISPLAY_MIN_HEIGHT_PT,
    merge_gap_pt: float = DISPLAY_MERGE_GAP_PT,
    pad_pt: float = DISPLAY_PAD_PT,
) -> EquationDetectionResult:
    """Detect display-equation regions on a born-digital PDF page.

    Uses only PyMuPDF geometry (``get_text("dict")`` + ``get_fonts()``).
    Never raises; returns an empty result on any unexpected error.

    Parameters
    ----------
    page:
        A ``fitz.Page`` object (from an open ``fitz.Document``).
    page_num:
        1-indexed page number (for provenance records).
    min_math_span_ratio, center_tolerance_pt, min_height_pt, merge_gap_pt, pad_pt:
        Override the named module constants for callers that need different
        sensitivity (e.g. per-document calibration or tests).

    Returns
    -------
    EquationDetectionResult
        Detected regions with padded bboxes.  ``detection_time_s`` is the
        wall-clock time of the geometry pass only (no model call).
    """
    t0 = time.perf_counter()

    try:
        result = _detect(
            page,
            page_num,
            min_math_span_ratio=min_math_span_ratio,
            center_tolerance_pt=center_tolerance_pt,
            min_height_pt=min_height_pt,
            merge_gap_pt=merge_gap_pt,
            pad_pt=pad_pt,
        )
    except Exception as exc:
        logger.warning("equation detection failed on page %d: %s", page_num, exc)
        result = EquationDetectionResult(page_num=page_num)

    result.detection_time_s = time.perf_counter() - t0
    return result


def _detect(
    page,
    page_num: int,
    *,
    min_math_span_ratio: float,
    center_tolerance_pt: float,
    min_height_pt: float,
    merge_gap_pt: float,
    pad_pt: float,
) -> EquationDetectionResult:
    """Inner (non-defensive) implementation of detect_display_equations."""
    from socr.core.born_digital import _MATH_FONT_RE

    # --- Step 0: collect page-level font names for quick span-level lookup ---
    # page.get_fonts() returns (xref, ext, type, basefont, name, encoding, ...)
    math_font_names: set[str] = set()
    try:
        for font in page.get_fonts():
            basefont = font[3] if len(font) > 3 else ""
            name = font[4] if len(font) > 4 else ""
            for candidate in (basefont, name):
                if candidate and _MATH_FONT_RE.search(candidate):
                    # Normalise: subset prefix like "ABCDEF+" is stripped by
                    # PyMuPDF already for the basefont; keep both basefont and
                    # name for belt-and-suspenders matching below.
                    math_font_names.add(candidate)
    except Exception:
        pass  # no math fonts detectable; proceed (will find no lines)

    if not math_font_names:
        # Fast exit: no math fonts on this page → no display equations.
        return EquationDetectionResult(page_num=page_num)

    # --- Step 1: find page geometry for centering check ---
    page_rect = page.rect  # fitz.Rect
    page_width = page_rect.width

    try:
        full_dict = page.get_text("dict")
    except Exception:
        return EquationDetectionResult(page_num=page_num)

    blocks = full_dict.get("blocks", [])

    # Display equations are centred relative to the page's *type area* (text
    # column), not the full page width.  We approximate the type-area midpoint
    # as the median of prose-block horizontal midpoints; if no prose blocks are
    # found we fall back to the page-width half.  Using the page half directly
    # is robust even when all blocks are math blocks (centred on the page).
    #
    # Rationale: the text-column union approach (min/max of all block x extents)
    # is misleading because prose blocks that span the full column width pull
    # the computed midpoint toward the page centre *anyway*, while narrower
    # equation blocks (shorter text) would appear off-centre relative to the
    # wider prose union.  The page-half is a simpler, equally stable reference.
    text_mid_x = page_width / 2.0

    # --- Step 2: collect candidate math-font lines ---
    # A line is a candidate if a sufficient fraction of its character content
    # comes from spans whose font is a recognised math font.
    candidates: list[tuple[float, float, float, float, bool]] = []
    # Each entry: (x0, y0, x1, y1, has_eq_number)

    for block in blocks:
        if block.get("type", 0) == 1:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            total_chars = 0
            math_chars = 0
            has_eq_num = False

            for span in spans:
                text = span.get("text", "")
                font = span.get("font", "")
                char_count = len(text)
                total_chars += char_count

                # Check if this span's font is a math font.
                # Span-level font names are usually just the basefont/name
                # pair; try matching them against the page-level math set or
                # directly against the regex (some embeddings differ slightly).
                is_math_span = font in math_font_names or (
                    bool(font) and _MATH_FONT_RE.search(font) is not None
                )
                if is_math_span:
                    math_chars += char_count

                # Equation number check: look for (3), (3.4), (A.1) on any span.
                if DISPLAY_EQ_NUM_RE.search(text):
                    has_eq_num = True

            if total_chars == 0:
                continue
            math_ratio = math_chars / total_chars
            if math_ratio < min_math_span_ratio:
                continue

            # --- Step 3: centering check ---
            lx0, ly0, lx1, ly1 = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
            line_mid_x = (lx0 + lx1) / 2.0
            offset = abs(line_mid_x - text_mid_x)

            # Promote if centred OR if it has an equation number (which is a
            # reliable display-equation signal even when the line is slightly
            # asymmetric due to the flush-right number itself).
            if offset <= center_tolerance_pt or has_eq_num:
                candidates.append((lx0, ly0, lx1, ly1, has_eq_num))

    if not candidates:
        return EquationDetectionResult(page_num=page_num)

    # --- Step 4: sort by vertical position and merge adjacent lines ---
    candidates.sort(key=lambda c: c[1])  # sort by y0

    merged: list[tuple[float, float, float, float, bool]] = []
    cur_x0, cur_y0, cur_x1, cur_y1, cur_eq = candidates[0]

    for nx0, ny0, nx1, ny1, neq in candidates[1:]:
        if ny0 - cur_y1 <= merge_gap_pt:
            # Merge: extend bounding box.
            cur_x0 = min(cur_x0, nx0)
            cur_x1 = max(cur_x1, nx1)
            cur_y1 = max(cur_y1, ny1)
            cur_eq = cur_eq or neq
        else:
            merged.append((cur_x0, cur_y0, cur_x1, cur_y1, cur_eq))
            cur_x0, cur_y0, cur_x1, cur_y1, cur_eq = nx0, ny0, nx1, ny1, neq
    merged.append((cur_x0, cur_y0, cur_x1, cur_y1, cur_eq))

    # --- Step 5: apply height filter and padding ---
    regions: list[EquationRegion] = []
    for rx0, ry0, rx1, ry1, req in merged:
        height = ry1 - ry0
        if height < min_height_pt:
            continue

        # Padded crop rectangle (clipped to page bounds)
        px0 = max(page_rect.x0, rx0 - pad_pt)
        py0 = max(page_rect.y0, ry0 - pad_pt)
        px1 = min(page_rect.x1, rx1 + pad_pt)
        py1 = min(page_rect.y1, ry1 + pad_pt)

        regions.append(
            EquationRegion(
                page_num=page_num,
                bbox=(px0, py0, px1, py1),
                source_bbox=(rx0, ry0, rx1, ry1),
                has_eq_number=req,
            )
        )

    return EquationDetectionResult(page_num=page_num, regions=regions)


# ── Crop-PNG storage ────────────────────────────────────────────────────────


def save_equation_crops(
    regions: list[EquationRegion],
    page,
    save_dir: Path,
    dpi: int = 300,
) -> None:
    """Render and save each detected equation region as a crop PNG.

    Naming convention: ``equation_{n}_page{N}.png`` — mirrors figure naming
    (``figure_{fig_num}_page{page_num}.png`` in ``FigureExtractor._save``).
    ``save_dir`` is created with ``parents=True`` if it does not exist.
    Fills in ``region.crop_path`` for each successfully saved crop.
    Never raises; failed crops leave ``crop_path`` as None.
    """
    try:
        import fitz
    except ImportError:
        logger.warning("fitz not available; cannot save equation crops")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0

    for n, region in enumerate(regions, start=1):
        px0, py0, px1, py1 = region.bbox
        clip = fitz.Rect(px0, py0, px1, py1)
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
            path = save_dir / f"equation_{n}_page{region.page_num}.png"
            pix.save(str(path))
            region.crop_path = str(path)
        except Exception as exc:
            logger.warning(
                "failed to save equation crop %d on page %d: %s",
                n,
                region.page_num,
                exc,
            )
