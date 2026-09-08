"""Ditto-mark detection (#625): keep verbatim, surface -- never fill down.

Corpus finding: a repeated-value table column is sometimes printed with a
ditto mark (``"``, ``''``, ``”``, ``″``, ``〃``) standing for "same as the row
above" instead of the value itself. socr's transcription of that mark is
faithful -- the page really prints it -- so this is not content loss in the
usual sense: no number is wrong or missing. It is a third state, present,
faithful, and semantically inert to a downstream parser.

Owner ruling (#625, 2026-09-08): option 3. Keep the mark verbatim; surface the
fact. Resolving a ditto mark into the value it stands for (option 2) was
explicitly rejected -- a fill-down that guesses wrong (a mark that refers two
rows up, or across a block boundary) manufactures a number, which is the
failure mode this corpus most wants to avoid. This module therefore only
detects and reports; it never rewrites a cell.

Pure and text-only, like ``socr.math.accounting`` -- it opens no PDF and calls
no provider, because its caller runs inside repeated page finalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from socr.tables.reconcile import find_table_blocks

#: Audit-event kind naming one table/column pair carrying a detected ditto
#: mark on a page's shipped table. One event per (page, table_id,
#: column_index); see ``socr.core.tables_trust.TABLE_DISTRUST_KINDS``.
DITTO_UNRESOLVED_KIND = "table_ditto_unresolved"

#: Ditto marks the owner ruling names verbatim, in the order given in the
#: issue. Each stands alone as a cell's ENTIRE content -- a mark embedded in
#: a longer string (e.g. a legitimate quoted value like ``"n/a"``) is not a
#: ditto cell, it is a quoted string, and must not be flagged.
DITTO_MARKS: tuple[str, ...] = ('"', "''", "”", "″", "〃")
_DITTO_CELL_RE = re.compile("^(?:" + "|".join(re.escape(m) for m in DITTO_MARKS) + ")$")


def _is_ditto_cell(cell: str) -> bool:
    return bool(_DITTO_CELL_RE.match(cell.strip()))


@dataclass(frozen=True)
class DittoColumn:
    """One table/column pair carrying at least one ditto-mark cell."""

    table_id: str
    column_index: int
    ditto_cells: int

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "column_index": self.column_index,
            "ditto_cells": self.ditto_cells,
        }


def detect_ditto_columns(markdown: str, page_num: int) -> list[DittoColumn]:
    """Scan every table block in ``markdown`` for cells that are ditto marks.

    Table ids follow the same ``p{page_num}-t{idx}`` convention as
    ``socr.tables.witness`` (``idx`` here is the table's ordinal position
    among ``find_table_blocks``' findings on this page's SHIPPED text, since
    this scan is text-level and has no region witness to key off of -- the
    same relationship ``_apply_unresolved_math_guard`` has to the corrupt-math
    lane's own region-keyed evidence).

    The header row (grid row 0) is excluded: a ditto mark stands for "same as
    the row above", and a header has no row above it. A column is reported
    once, with the COUNT of ditto cells found in its body, if it carries at
    least one; cells are read only, never rewritten -- this is detection, not
    resolution.
    """
    if not markdown:
        return []
    columns: list[DittoColumn] = []
    for idx, block in enumerate(find_table_blocks(markdown)):
        grid = block.grid
        if len(grid) < 2:
            continue
        body = grid[1:]
        n_cols = max((len(row) for row in body), default=0)
        table_id = f"p{page_num}-t{idx}"
        for col in range(n_cols):
            count = sum(1 for row in body if col < len(row) and _is_ditto_cell(row[col]))
            if count:
                columns.append(DittoColumn(table_id=table_id, column_index=col, ditto_cells=count))
    return columns
