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

from socr.tables.reconcile import _markdown_content_lines, find_table_blocks

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

    ``markdown`` is provenance-stripped via ``_markdown_content_lines`` before
    ``find_table_blocks`` ever sees it (GH-687). A ditto mark inside a fenced
    code sample or an HTML comment is not a reading of the page -- it is the
    model showing what a grid looks like, or echoing dead text -- and must not
    manufacture the #625 distrust signal on an otherwise clean page. This
    scan parses the actual grid via ``find_table_blocks`` rather than reading
    raw rows for a formatting defect, so it follows the same provenance
    contract as ``_has_table_grid`` (:580) -- NOT
    ``_strip_emission_literal_blocks``, which its own docstring names
    "emission-only" and which the three emission-defect predicates
    (``table_emission_defect``, ``table_content_defect``,
    ``raw_table_block_lines``) use to blank raw-HTML ``<pre>``/``<script>``/
    ``<style>``/``<textarea>`` blocks before reading rows for a SHAPE defect
    in the shipped text itself. Ditto detection asks a different question --
    "does this parsed grid carry a distrust cell" -- the same question the
    grid-existence predicates ask, and none of those apply the literal-block
    strip either.

    The header row (grid row 0) is excluded: a ditto mark stands for "same as
    the row above", and a header has no row above it. A column is reported
    once, with the COUNT of ditto cells found in its body, if it carries at
    least one; cells are read only, never rewritten -- this is detection, not
    resolution.
    """
    if not markdown:
        return []
    stripped = "\n".join(_markdown_content_lines(markdown))
    columns: list[DittoColumn] = []
    for idx, block in enumerate(find_table_blocks(stripped)):
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
