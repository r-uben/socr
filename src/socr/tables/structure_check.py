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


def structural_gate_fires(reports: Sequence[GridStructureReport]) -> bool:
    """Whether the GH-151 TICKET-B1 structural gate fires over a page's reports.

    The single source of truth for the gate predicate: ``ragged`` or
    ``detached_label_rows`` on any block, and nothing else -- never
    ``GridStructureReport.defective`` as a whole (which also counts
    ``orphan_rows``, deliberately excluded per the ticket's narrowing
    decision; see ``check_grid``'s ``FINDING_ORPHAN_ROWS`` comment). Shared
    by ``born_digital.py`` (the production caller) and the negative-control
    tests, so a test asserting "the gate does not fire" is provably
    asserting the same condition production branches on, not a
    lookalike copy of it.
    """
    return any(r.ragged or r.detached_label_rows for r in reports)


# GH-200: the escalation-gate defect codes. A disjunction, evaluated in cost
# order -- the grid-shape term is string-only and free; the header term needs
# native word geometry and runs only when the shape term did not already fire.
DEFECT_NONE = ""
DEFECT_GRID_SHAPE = "grid_shape"
DEFECT_HEADER_UNATTRIBUTED = "header_unattributed"


def table_output_defect(
    output_md: str,
    words: list | None,
    rules: list[tuple[float, float, float]] | None = None,
) -> str:
    """Whether *output_md* (the text about to ship) has a structural defect.

    GH-200, extended by GH-212. A disjunction in cost order:

    1. ``structural_gate_fires`` on the emitted grid (B1's own predicate,
       ragged OR detached_label_rows, unchanged -- see ``structural_gate_fires``
       docstring). String-only, no I/O.
    2. ``header_cut.header_cut_verdict`` on each emitted table block. Runs only
       when the shape term did not already fire, and only when the caller
       supplied both native words and the page's drawn horizontal rules.

    **On term 2 and the four reverts that precede it.** Earlier implementations
    of a header-attribution disjunct (GH-151 T3's token-pattern rule,
    ``062bdef``'s year-band rule, the positional rule, and the normalized
    comparison) each failed in one of two directions: abstaining on the
    hand-judged 4-of-4 header-loss case, or returning HARD on *byte-perfect
    correct* tables carrying significance-star rows and ``n.a.`` cells, both
    ubiquitous in this corpus. A false HARD is not a missed catch; it REJECTS
    correct output and can drive it to the fail-closed marker.

    All four recovered header ROLE from token content. ``header_cut`` does not:
    it cuts the page at the drawn rule above the numeric anchor and owes only
    what lies above it, so star and ``n.a.`` rows are excluded geometrically
    rather than by vocabulary, and its tokens are never inspected. Every step
    abstains rather than guessing. See
    ``docs/log/2026-08-19_212-header-attribution-design.md``.

    ``header_attribution`` and ``table_header_verdicts`` below are retained
    unchanged: they still produce the advisory SOFT signal and the abstain rate
    that ``header_cut`` deliberately does not report.

    **rules=None abstains.** A caller with no ``fitz`` page (the verifier
    exception path, and ``born_digital``'s native-lane check, which flags the
    native text layer rather than a model-emitted table) passes nothing and gets
    the shape term alone. That is the intended behaviour for those callers, not
    an oversight.

    Deliberately NOT nested with TR-3 (``native_verifier.verify_native_table``)
    -- TR-3 is evaluated by the caller as a separate disjunct. Measured in
    ``docs/log/2026-08-14_gh151-b1-predicate-design.md``: the two signals do
    not subsume each other (TR-3 misses 31/66 shape-gate pages; the shape gate
    misses 27 pages TR-3 catches). Pure; no mutation, no model calls -- the
    caller precomputes ``rules``, so this never touches a page object.
    """
    reports = check_markdown(output_md)
    if structural_gate_fires(reports):
        return DEFECT_GRID_SHAPE

    if words and rules:
        from socr.tables.header_attribution import HeaderVerdict
        from socr.tables.header_cut import header_cut_verdict

        for block in find_table_blocks(output_md):
            if header_cut_verdict(block.grid, words, rules) is HeaderVerdict.HARD:
                return DEFECT_HEADER_UNATTRIBUTED

    return DEFECT_NONE


def table_header_verdicts(output_md: str, words: list | None) -> list:
    """Header-attribution verdict for every table block on *output_md*.

    Exposed separately from ``table_output_defect`` so a caller can COUNT the
    ``UNVERIFIABLE`` abstain rate (the hand-judgement/panel-mandated
    surfacing: the abstain rate, not the precision, is the number a header
    notation gap would show up in -- see ``header_attribution``'s module
    docstring on ``_NUM_TOKEN_RE``'s leading-decimal blind spot). Pure; no
    I/O, no event emission -- callers decide what, if anything, to record.
    """
    if not words:
        return []
    from socr.tables.header_attribution import header_attribution

    return [header_attribution(block.grid, words) for block in find_table_blocks(output_md)]
