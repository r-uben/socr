"""Pure diagnostic grid-shape report for markdown table blocks.

This module is a pure diagnostic: it reports the structural shape of a parsed
table grid, and never mutates, repairs, or routes anything. Given a grid, it
answers "is this shape suspicious?" and nothing more — callers decide what, if
anything, to do about a defective report.

Density (``empty_cell_ratio`` / the footprint-density derivation described on
:class:`GridStructureReport`) is diagnostic only. It can never contribute a
finding or make a grid ``defective`` — sparse tables are common and legitimate
(blank cells in a regression table, missing observations), so no threshold on
emptiness is applied anywhere in this module. The only two findings that can
ever fire are width raggedness and same-row orphan labels.

``check_grid``'s input contract: a separator-free ``Sequence[Sequence[str]]``.
Callers are expected to have already stripped markdown separator rows (the
``---|---`` line) before calling — ``check_markdown`` does this via
``reconcile.find_table_blocks``, which in turn uses ``reconcile._parse_grid``.
``check_grid`` itself performs no markdown parsing and never drops rows that
look like separators; see ``test_separator_free_contract`` for the pin.

Inherited blind spot (documented here, not fixed — handed to TICKET-B1):
``reconcile._parse_grid`` treats an all-blank pipe row (e.g. ``"|  |  |  |"``)
as a separator row, because its separator test is
``all(_SEP_CELL.match(c.strip()) for c in cells if c.strip())``, which is
vacuously true when every cell is blank (the generator yields nothing, and
``all()`` of an empty iterable is ``True``). Such rows are silently dropped
before ``check_grid`` ever sees them. Verified against the real parser:
``_parse_grid(["|  |  |  |"]) == []``, and
``"| a | b |\\n| --- | --- |\\n|  |  |\\n| c | d |"`` parses to
``[["a", "b"], ["c", "d"]]`` — the all-blank row in the middle vanishes rather
than surfacing as, say, an orphan or an empty body row. This module does not
and cannot see the missing row, so it cannot flag it; a fix belongs in
``reconcile._parse_grid``, not here.

Row indices are zero-based into the parsed, separator-free grid. Row 0 is
always the header row.

Cell blankness is whitespace-only: a cell counts as empty iff
``not cell.strip()``. This is deliberately **not**
``native_verifier.strip_presentation``, which also strips standalone currency
symbols (so a cell containing only ``"$"`` would be treated as empty there).
Using that stripping here would wrongly count a ``"$"``-only cell as blank.
``native_verifier.py`` is owned by a different ticket this wave and is not
imported by this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from socr.tables.reconcile import find_table_blocks

FINDING_RAGGED = "ragged"
FINDING_ORPHAN_ROWS = "orphan_rows"


@dataclass(frozen=True)
class GridStructureReport:
    """Threshold-free structural report for one parsed table grid.

    ``empty_cell_ratio`` is the only derived ratio this report exposes. A
    caller that wants density measured against the grid's *rectangular
    footprint* (rows x max row width) rather than against the actual cell
    count can derive it from the two denominators already on this report:
    ``(empty_cells + (footprint_cells - total_cells)) / footprint_cells``
    when ``footprint_cells > 0`` — the off-grid "missing" cells of a ragged
    grid count as empty for that purpose. (Worked example, p26 fixture:
    28/56 by footprint vs the actual 21/49 by real cell count.)
    """

    row_widths: tuple[int, ...]
    ragged: bool
    orphan_rows: tuple[int, ...]  # body rows only (indices >= 1)
    empty_cells: int  # whitespace-empty among actual cells
    total_cells: int  # sum of row lengths
    footprint_cells: int  # len(grid) * max(row_widths), 0 when grid has no rows
    findings: tuple[str, ...]

    @property
    def empty_cell_ratio(self) -> float:
        """Fraction of actual cells that are whitespace-empty; never divides by zero."""
        return self.empty_cells / self.total_cells if self.total_cells else 0.0

    @property
    def defective(self) -> bool:
        """At least one threshold-free structural finding fired.

        A fact about the evidence, explicitly not a routing decision.
        """
        return bool(self.findings)


def check_grid(grid: Sequence[Sequence[str]]) -> GridStructureReport:
    """Report the structural shape of a separator-free grid. Pure; no mutation.

    ``grid`` must already have markdown separator rows removed by the caller.
    """
    row_widths = tuple(len(row) for row in grid)
    ragged = len(set(row_widths)) > 1

    orphan_rows: list[int] = []
    for i, row in enumerate(grid):
        if i == 0:
            continue  # header row is never an orphan
        if len(row) == 0:
            continue  # defensively non-orphan
        label = row[0]
        if label.strip():
            continue  # labelled row, not an orphan
        if any(cell.strip() for cell in row[1:]):
            orphan_rows.append(i)

    empty_cells = sum(1 for row in grid for cell in row if not cell.strip())
    total_cells = sum(row_widths)
    footprint_cells = len(grid) * max(row_widths) if row_widths else 0

    findings: list[str] = []
    if ragged:
        findings.append(FINDING_RAGGED)
    if orphan_rows:
        findings.append(FINDING_ORPHAN_ROWS)

    return GridStructureReport(
        row_widths=row_widths,
        ragged=ragged,
        orphan_rows=tuple(orphan_rows),
        empty_cells=empty_cells,
        total_cells=total_cells,
        footprint_cells=footprint_cells,
        findings=tuple(findings),
    )


def check_markdown(page_md: str) -> list[GridStructureReport]:
    """Report structure for every table block on a page, in document order.

    Thin wrapper only: delegates parsing entirely to
    ``reconcile.find_table_blocks`` (no second markdown parser). Inherits that
    parser's documented blind spot: an all-blank pipe row is treated as a
    separator and silently dropped before it ever reaches ``check_grid`` — see
    the module docstring. That gap is left for TICKET-B1 to address.
    """
    return [check_grid(block.grid) for block in find_table_blocks(page_md)]
