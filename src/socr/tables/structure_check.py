"""Pure diagnostic grid-shape report for markdown table blocks.

This module is a pure diagnostic: it reports the structural shape of a parsed
table grid, and never mutates, repairs, or routes anything. Given a grid, it
answers "is this shape suspicious?" and nothing more — callers decide what, if
anything, to do about a defective report.

Density (``empty_cell_ratio`` / the footprint-density derivation described on
:class:`GridStructureReport`) is diagnostic only. It can never contribute a
finding or make a grid ``defective`` — sparse tables are common and legitimate
(blank cells in a regression table, missing observations), so no threshold on
emptiness is applied anywhere in this module. Density never votes. Three
findings can fire: width raggedness, same-row orphan labels (diagnostic
only — never gates, see TICKET-B1), and detached label/values row pairs
(``FINDING_DETACHED_LABEL``, a row-pair adjacency invariant).

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

GH-151 TICKET-B1 adds a third finding, ``FINDING_DETACHED_LABEL``: a
row-pair adjacency invariant, decidable from exactly one adjacent pair of
body rows, that never references a third row or any cross-row signature
(no modal vote, no majority, no numeric constant beyond the pair itself).
It exists because ``FINDING_ORPHAN_ROWS`` alone cannot distinguish a
legitimate standard-error / t-statistic continuation row (blank label,
populated values, following a row that ALSO has values) from a genuinely
severed label row (e.g. an ``R2`` label whose values were pushed onto the
next row by a superscript). Density still never votes.

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
FINDING_DETACHED_LABEL = "detached_label"


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
    #: 0-based index of the LEFT row of each firing label/values split pair
    #: (body rows only). See ``FINDING_DETACHED_LABEL`` / GH-151 TICKET-B1.
    detached_label_rows: tuple[int, ...]
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


def _is_canonical_column_band(value_cells: Sequence[str]) -> bool:
    """Whether the non-blank value cells read exactly ``(1), (2), ..., (k)``.

    A group-heading row (label, no values) immediately above a ``(1)...(n)``
    column-number band is a legitimate table layout, not a split row — the
    heading has no values because it is a heading, and the numbered row has
    no label because it is naming columns, not observations. Excluded here so
    ``FINDING_DETACHED_LABEL`` does not fire on it. This is the ONLY exclusion
    the predicate admits; it is local to the pair (no threshold, no reference
    to any other row).
    """
    nonblank = [cell.strip() for cell in value_cells if cell.strip()]
    if not nonblank:
        return False
    return nonblank == [f"({i})" for i in range(1, len(nonblank) + 1)]


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

    # GH-151 TICKET-B1: detached-label row pairs. A row-pair adjacency
    # invariant only — never a modal/majority signature, never a numeric
    # threshold. For each adjacent body-row pair (i, i+1) with i >= 1: row i
    # carries a label and zero values, row i+1 carries values and no label.
    # That shape is exactly a physically split row (e.g. an "R2" label whose
    # values landed on the next line). The one exclusion is the canonical
    # "(1) (2) ... (k)" column-number band, which is a legitimate heading
    # pattern, not a split row.
    detached_label_rows: list[int] = []
    for i in range(1, len(grid) - 1):
        row = grid[i]
        follower = grid[i + 1]
        if not row or not follower:
            continue
        if not row[0].strip():
            continue  # left row has no label -> not a detached label
        if any(cell.strip() for cell in row[1:]):
            continue  # left row has values -> not label-only
        if follower[0].strip():
            continue  # follower has its own label -> not a split continuation
        if not any(cell.strip() for cell in follower[1:]):
            continue  # follower has no values either -> nothing to attach
        if _is_canonical_column_band(follower[1:]):
            continue  # group heading above a (1)...(n) band: legitimate
        detached_label_rows.append(i)

    empty_cells = sum(1 for row in grid for cell in row if not cell.strip())
    total_cells = sum(row_widths)
    footprint_cells = len(grid) * max(row_widths) if row_widths else 0

    findings: list[str] = []
    if ragged:
        findings.append(FINDING_RAGGED)
    if orphan_rows:
        findings.append(FINDING_ORPHAN_ROWS)
    if detached_label_rows:
        findings.append(FINDING_DETACHED_LABEL)

    return GridStructureReport(
        row_widths=row_widths,
        ragged=ragged,
        orphan_rows=tuple(orphan_rows),
        detached_label_rows=tuple(detached_label_rows),
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
