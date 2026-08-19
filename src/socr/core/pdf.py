"""One way to open a PDF for reading, so every reader sees the same text (#244).

## Why this exists

``repair_symbol_font_text`` (#217) rebuilds the ``/ToUnicode`` map that some
publisher PDFs omit, **on the ``fitz.Document`` object it is given**. When that
object is closed the repair goes with it.

socr opens the same file in many places. Wiring the repair into only some of them
is worse than wiring it into none: the native lane reports ``−0.12`` while table
extraction, opening its own handle, still reads ``20.12`` from the same page.
Machinery that compares the two lanes then sees a disagreement that is an artefact
of *where the file was opened*, and can distrust the corrected output because the
uncorrected reader disagrees with it.

So: open through :func:`open_pdf` whenever the document will be read as text.

## Why not simply repair on every open

The repair walks the font list of every page. On a clean 411-page document that is
~200 ms — negligible once, ruinous inside a per-page loop that reopens the file.

A document's font inventory does not change while socr runs, so a file found to
need no repair is remembered by identity (path, size, mtime) and later opens skip
the scan. The cache holds only the *negative* result, which is the common case; a
file that does need repair is repaired every time, because the repair applies to a
fresh Document object each time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz

from socr.core.glyph_recovery import GlyphRepairReport, repair_symbol_font_text

logger = logging.getLogger(__name__)

#: Files observed to need no glyph recovery, keyed by (path, size, mtime_ns).
#: Negative results only — see the module docstring.
_NO_REPAIR_NEEDED: set[tuple[str, int, int]] = set()


def _identity(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_size, stat.st_mtime_ns)


def reset_repair_cache() -> None:
    """Forget which files were found clean. For tests, and after rewriting a PDF."""
    _NO_REPAIR_NEEDED.clear()


def apply_glyph_recovery(doc: fitz.Document, path: Path | str) -> GlyphRepairReport:
    """Repair *doc* in memory, skipping the scan for files already found clean.

    Never raises: a document that cannot be repaired must still be readable
    exactly as it was before this existed.
    """
    path = Path(path)
    identity = _identity(path)
    if identity is not None and identity in _NO_REPAIR_NEEDED:
        return GlyphRepairReport()

    try:
        report = repair_symbol_font_text(doc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[glyph] %s: recovery failed (%s)", path.name, exc)
        return GlyphRepairReport()

    if identity is not None and not report.repaired and not report.needs_attention:
        _NO_REPAIR_NEEDED.add(identity)
    return report


def open_pdf(path: Path | str, *, repair: bool = True) -> fitz.Document:
    """Open *path* for reading, with symbol-font glyph recovery applied.

    Pass ``repair=False`` only when the document is not read as text — page
    counting, rasterising to an image, splitting pages — where the recovery
    would cost a font scan and change nothing.

    Returns a ``fitz.Document``; the caller owns closing it, exactly as with
    ``fitz.open``.
    """
    doc = fitz.open(path)
    if repair:
        apply_glyph_recovery(doc, path)
    return doc
