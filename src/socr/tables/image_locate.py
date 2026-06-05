"""Image-based table localization for *scanned* pages.

The vector detectors (``find_tables`` lines + booktabs rule-band) both read
``page.get_drawings()`` — which is empty on a true scan, where the page is a
single raster image and the table rules are pixels, not vectors. Confirmed across
the owner's whole pre-1996 set: no scanned page exposes a single vector rule.

This module finds horizontal rules directly in the rendered raster via
connected-component analysis. The hard part the vector case never had is telling a
drawn rule from a justified-prose text row (both span the page width). Three
discriminators separate them, validated on real pages (a dense table page -> its
rules; three real scanned prose pages -> zero false positives):

- **thin**: a rule is a few pixels tall; a text row is x-height tall.
- **solid**: a rule is one continuous ink run; a text row is many glyph blobs.
- **wide**: a rule spans most of the page width.

Detected rules are converted to PDF points and handed to the shared
``bands_from_rules`` grouping, so downstream (crop -> re-read -> reconcile) is
identical to the vector path.
"""

from __future__ import annotations

import logging

from socr.tables.locate import TableBox, bands_from_rules

logger = logging.getLogger(__name__)

# Detection raster resolution. Enough to resolve thin rules without large images;
# the crop pass later re-renders at its own (higher) DPI.
DETECT_DPI = 150
# Rule geometry. Width/fill are DPI-independent ratios; thickness scales with DPI
# (a typeset rule is ~0.5-1pt; at 150 DPI that is ~1-2px, so allow a little slack
# for scan blur). Derived from the rule's physical size, not tuned to a corpus.
_MIN_WIDTH_FRAC = 0.40   # span >= 40% of page width
_MIN_FILL_FRAC = 0.55    # >= 55% of the bbox width is continuous ink (solid)
_MAX_THICK_PT = 2.0      # <= 2pt tall -> thin; excludes text rows


def locate_tables_image(page, dpi: int = DETECT_DPI) -> list[TableBox]:
    """Locate ruled/booktabs tables on a scanned page from its raster.

    Never raises: any failure (missing cv2, render error) logs and yields no box,
    so a bad page degrades to "no dual-pass" rather than dropping the page.
    """
    rules = _raster_rules(page, dpi)
    return bands_from_rules(rules, page.rect, source="image")


def _raster_rules(page, dpi: int) -> list[tuple[float, float, float]]:
    """Detect horizontal rules in the page raster, returned as ``(y, x0, x1)`` in
    PDF points (so they share the vector path's coordinate space)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("image table detection needs opencv-python-headless + numpy")
        return []

    try:
        import fitz

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else arr[:, :, 0]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("raster render failed for image table detection (%s)", exc)
        return []

    h, w = gray.shape
    # Ink -> white on black, robust to scan shading.
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2
    )
    # Isolate horizontal structure: opening with a long thin horizontal kernel
    # survives only where continuous horizontal ink ran (rules), not glyph rows.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 8), 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    scale = 72.0 / dpi               # px -> PDF points
    max_thick_px = max(2.0, _MAX_THICK_PT * dpi / 72.0)
    rules: list[tuple[float, float, float]] = []
    for i in range(1, n_labels):  # 0 is background
        x, y, cw, ch, area = stats[i]
        if cw < w * _MIN_WIDTH_FRAC:    # wide
            continue
        if ch > max_thick_px:           # thin
            continue
        if area < cw * _MIN_FILL_FRAC:  # solid
            continue
        ymid = (y + ch / 2.0) * scale
        rules.append((ymid, x * scale, (x + cw) * scale))
    return rules
