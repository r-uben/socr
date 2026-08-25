"""GH-96 hierarchy-aware cell exactness — re-export of the canonical module.

The implementation lives in ``socr.core.table_grid`` so production table code
(``escalation_decision``) can score a page without importing the benchmark
harness (#175). This module keeps the historical import path working
(orchestrator, tests).
"""

from socr.core.table_grid import (
    CellMiss,
    ExactnessReport,
    markdown_rows,
    score_page,
    score_rows,
)
from socr.tables.native_rows import (
    MARKER_RE as _MARKER_RE,
    LabeledRow,
    is_value as _is_value,
    native_rows_from_page,
    normalize_label,
    rows_establish_grid,
    superscript_tokens as _superscript_tokens,
)

__all__ = [
    "CellMiss",
    "ExactnessReport",
    "LabeledRow",
    "markdown_rows",
    "native_rows_from_page",
    "normalize_label",
    "rows_establish_grid",
    "score_page",
    "score_rows",
    "_MARKER_RE",
    "_is_value",
    "_superscript_tokens",
]
