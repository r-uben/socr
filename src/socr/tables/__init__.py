"""Dual-pass table extraction.

The hard-page judge (Phase 3b) *detects* corrupted tables by looking at the whole
page. This package *extracts* them precisely: locate each table's bounding box,
crop it, re-read the crop with a table-specialised VLM pass, and reconcile that
reading against the whole-page OCR. Disagreement between the two passes is a
built-in corruption signal; the crop pass (higher effective resolution,
table-focused) is treated as authoritative and patched back into the page.

Scope (v1): tables whose geometry gives a *precise* bounding box — fully ruled
tables (PyMuPDF ``find_tables`` lines strategy) and booktabs-style tables (top /
mid / bottom horizontal rules, localised from ``page.get_drawings()``). Fully
borderless whitespace-only tables have no geometric anchor and are left to the
whole-page judge; see ``locate.py`` for why ``find_tables`` text-strategy boxes
are too over-inclusive to crop safely.
"""

from socr.tables.locate import TableBox, locate_tables
from socr.tables.reconcile import TableReconcileResult, reconcile_page_tables

__all__ = [
    "TableBox",
    "locate_tables",
    "TableReconcileResult",
    "reconcile_page_tables",
]
