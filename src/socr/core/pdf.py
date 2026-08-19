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

from socr.core.glyph_recovery import (
    GlyphRepairReport,
    repair_symbol_font_text,
    replay_cmaps,
)

logger = logging.getLogger(__name__)

#: Files observed to need no glyph recovery, keyed by (path, size, mtime_ns).
_NO_REPAIR_NEEDED: set[tuple[str, int, int]] = set()

#: Files that DO need recovery, keyed the same way, holding the derived plan.
#: Deriving a plan walks every page's text; replaying it is a few object writes.
#: Caching only the negative result meant an affected document paid the full walk
#: on every open, and the agentic loop opens each document several times per page
#: — one 60 s scan became hours (#246).
_REPAIR_PLAN: dict[tuple[str, int, int], GlyphRepairReport] = {}


def _identity(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_size, stat.st_mtime_ns)


def reset_repair_cache() -> None:
    """Forget which files were found clean. For tests, and after rewriting a PDF."""
    _NO_REPAIR_NEEDED.clear()
    _REPAIR_PLAN.clear()


def apply_glyph_recovery(doc: fitz.Document, path: Path | str) -> GlyphRepairReport:
    """Repair *doc* in memory, skipping the scan for files already found clean.

    Never raises: a document that cannot be repaired must still be readable
    exactly as it was before this existed.
    """
    path = Path(path)
    identity = _identity(path)
    if identity is not None:
        if identity in _NO_REPAIR_NEEDED:
            return GlyphRepairReport()
        cached = _REPAIR_PLAN.get(identity)
        if cached is not None:
            # Same bytes, so the same object xrefs and the same plan. Replay it
            # rather than re-deriving: this is the whole point of the cache.
            replay_cmaps(doc, cached.cmaps)
            return cached

    try:
        report = repair_symbol_font_text(doc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[glyph] %s: recovery failed (%s)", path.name, exc)
        return GlyphRepairReport()

    if identity is not None:
        if report.repaired:
            _REPAIR_PLAN[identity] = report
        elif not report.needs_attention:
            _NO_REPAIR_NEEDED.add(identity)
        # Neither repaired nor cacheable-clean: the document has drawn glyphs
        # this module cannot recover. Deliberately not cached — the scan is the
        # only thing that produces that warning, and the case is now rare enough
        # (an affected font whose glyphs are genuinely unknown) that paying for
        # it is preferable to remembering a stale verdict.
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
