"""#1100: a page with no text layer and no ink never reaches a model.

Ink is reused from existing machinery, not a new threshold: a native word,
or a colour-channel below 255 in a render at the same ``DETECT_DPI``
``tables.image_locate`` uses to find ink on scans. Greyscale is not used:
near-white RGB is ink. A single faint mark is ink. A white-only drawing
is not (Dougherty 2013 p3). Fail closed: any lookup or render failure is
not blank, so the page may still reach a model.
"""

from __future__ import annotations

import logging

from socr.tables.image_locate import DETECT_DPI

logger = logging.getLogger(__name__)

BLANK_PAGE_NOTE = "blank page"


def blank_page_marker() -> str:
    """House marker for an empty page: ``[socr: blank page]``.

    Chosen over an HTML comment because ``socr_marker`` / ``SOCR_MARKER_RE``
    are the one recognizer for spans socr authors. The note is a receipt, not
    document content, and it is not a ``[page N failed: …]`` marker, so the
    page can ship SUCCESS and be skipped on resume.
    """
    from socr.core.manifest import socr_marker

    return socr_marker(BLANK_PAGE_NOTE)


def is_blank_page(page) -> bool:
    """True when *page* has no text layer and no ink.

    Rule: any native word, or any colour-channel below 255 at ``DETECT_DPI``,
    is ink. Near-white RGB (255, 255, 250) is ink. A white-only drawing
    (Dougherty 2013 p3) is not. Never raises.
    """
    try:
        if page.get_text("words"):
            return False
        # Drawings/images are not ink by themselves: Dougherty 2013 p3 has one
        # drawing and zero dark pixels (a mark that does not render). Ink is
        # a non-white pixel at DETECT_DPI, the same raster ``image_locate`` uses.
        return not _has_render_ink(page)
    except Exception as exc:  # noqa: BLE001 - unknown geometry is not blank
        logger.debug("blank page: lookup failed, not blank: %s", exc)
        return False


def _has_render_ink(page) -> bool:
    """True when a ``DETECT_DPI`` colour render has any non-white channel.

    Greyscale collapses near-white RGB ``(255, 255, 250)`` to 255. This render
    is RGB so each channel is judged on its own. Transparency composites onto
    white: unpainted stays ``(255, 255, 255)``; a painted translucent mark
    becomes a non-white RGB value. If an alpha plane is present, fully
    transparent pixels are unpainted (not ink) and any painted non-opaque-white
    pixel is ink. Fail closed at the caller.
    """
    import fitz

    mat = fitz.Matrix(DETECT_DPI / 72.0, DETECT_DPI / 72.0)
    # #1100-r2: colour samples, not greyscale — (255,255,250) is ink
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    samples = pix.samples
    n = pix.n
    if n >= 4:
        # RGBA (or more): alpha 0 is unpainted, not ink.
        return any(
            samples[i + 3] > 0
            and (
                samples[i] < 255
                or samples[i + 1] < 255
                or samples[i + 2] < 255
                or samples[i + 3] < 255
            )
            for i in range(0, len(samples), n)
        )
    return any(sample < 255 for sample in samples)
