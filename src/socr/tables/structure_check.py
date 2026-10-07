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

Inherited parser blind spot (documented here, deliberately unchanged):
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
and cannot see the missing row in its parsed-grid diagnostics. GH-190 closes
the shipping consequence of this blind spot by having ``table_output_defect``
inspect raw rows before parsing. ``_parse_grid`` itself and reconciliation diffs
remain blind to the dropped row and are deliberately unchanged; a parser-level
fix is outside this gate.

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

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from socr.tables.reconcile import (
    TABLE_CONTENT_EMPTY,
    TABLE_EMISSION_LATEX_LEAK,
    TABLE_EMISSION_WIDTH_MISMATCH,
    find_table_blocks,
    raw_table_block_lines,
    table_content_defect,
    table_emission_defect,
)

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
    ``reconcile.find_table_blocks`` (no second markdown parser). It inherits
    the parser's documented blind spot: an all-blank pipe row is treated as a
    separator and silently dropped before it ever reaches ``check_grid`` — see
    the module docstring. The shipping gate closes the consequence through its
    separate raw-row ``table_content_defect`` term; this parsed-grid path is
    deliberately unchanged.
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


# GH-200: the escalation-gate defect codes. A disjunction, evaluated in
# precedence order: emission defects, raw-row content defects, parsed
# grid-shape defects, then the header term, which needs native word geometry.
DEFECT_NONE = ""
DEFECT_GRID_SHAPE = "grid_shape"
DEFECT_HEADER_UNATTRIBUTED = "header_unattributed"
DEFECT_TABLE_TRUNCATED = "table_truncated"
DEFECT_TABLE_LATEX_LEAK = TABLE_EMISSION_LATEX_LEAK
DEFECT_TABLE_WIDTH_MISMATCH = TABLE_EMISSION_WIDTH_MISMATCH
DEFECT_TABLE_CONTENT_EMPTY = TABLE_CONTENT_EMPTY


def _final_row_truncated(block_lines: Sequence[str]) -> bool:
    """TICKET-A2 (#645) term (a): does this table block's last row break the
    block's own established leading/trailing-pipe style?

    Reads ``block_lines`` (header, separator, body -- ``raw_table_block_lines``'s
    own contract, ORIGINAL text, before ``_parse_grid`` can reshape or drop a
    malformed row). A row's "style" here is only whether it ends with ``|`` --
    a model that writes every row bordered on both sides and then stops mid
    number, e.g. the census fixture's ``| 2019 | 364.2 | 7,05`` with no
    closing pipe, breaks that style on exactly its last row. Needs at least
    two body rows to call: one to establish the style, one (the last) to
    compare against it -- a single-body-row block or an empty block abstains
    (returns False), since there is nothing to establish a style from.

    Deliberately conservative: a candidate whose rows are ALL unterminated
    (no trailing-pipe convention at all -- some models never close the right
    border) is not truncated by this term. Only a MIXED block -- terminated
    rows, then an unterminated final one -- is, since that mix is the actual
    signature of a row cut off mid-emission rather than a candidate's own
    consistent formatting choice.
    """
    body = [ln for ln in block_lines[2:] if ln.strip()]
    if len(body) < 2:
        return False
    trailing = [ln.strip().endswith("|") for ln in body]
    *earlier, last = trailing
    return bool(earlier) and all(earlier) and not last


#: TICKET-A2 (#645): ``table_shaped_native_row_count`` counts every native
#: band matching the candidate's own row width, INCLUDING a candidate's own
#: printed header/legend row when that row happens to be numeric-shaped
#: (e.g. year column headers) and slips past ``is_column_index_row``'s
#: narrower sequential-digit convention -- ``numeric_body_rows`` strips the
#: analogous row from the CANDIDATE side, so an otherwise-complete candidate
#: can look exactly one native row short with no truncation involved. One
#: such stray band is indistinguishable from a real header row; only a
#: shortfall exceeding it is treated as evidence of a dropped DATA row.
_STRAY_HEADER_BAND_ALLOWANCE = 1


#: TICKET (#703) round 2: how many recurring numeric column lanes the NATIVE
#: page must show before term (b)'s row-shape reconciliation says anything.
#: The reconciliation compares row WIDTHS; a single recurring lane gives every
#: native band width 1, which is the vacuous case #703 was filed for, so two is
#: the smallest arity at which a native row has a shape to reconcile at all --
#: an arity floor, not a fitted threshold. Deliberately WEAKER than the
#: detector's own ``reconstruct._MIN_LANES_PER_ROW`` (3): this is an ABSTENTION
#: gate on a content-loss guard, so a wrong "no lanes here" silently disarms A2
#: while a wrong "lanes here" only leaves A2 armed as it was before #703.
#: "Recurring" is not redefined -- ``has_recurring_numeric_columns`` keeps the
#: existing GH-248 rule that a lane counts only if it appears on at least
#: ``_MIN_TABLE_ROWS`` bands.
_MIN_RECONCILABLE_LANES = 2


def _native_page_has_column_lanes(words: list) -> bool:
    """TICKET (#703) round 2: does the NATIVE page show recurring numeric
    column lanes -- evidence that survives a truncated candidate?

    Round 1 gated term (b) on the candidate's own row widths, which is
    unsound: the dominance disappears together with the missing rows. A
    numeric table whose surviving rows happen to be sparse (two one-number
    rows retained, eighteen dense rows dropped) then reads as a text table and
    the guard A2 exists for switches itself off -- measured, the truncated
    reading wins selection over the complete one. Native geometry cannot be
    truncated by a model, so the eligibility question is asked there instead.

    The discriminator between "a table" and "prose that mentions figures" is
    not how many numerals a band carries -- on the BoE #703 page the native
    table-shaped count is 19/4/2 bands at widths 1/2/3, table-specific at no
    width -- but ALIGNMENT: a table's numerals recur in shared x-lanes down
    the page, prose figures scatter. Measured on that page: 16 x0-lanes, only
    2 of them recurring, and NO band populating two recurring lanes at once.
    The ECB bulletin p2/p3 truncation fixtures show 11 recurring lanes and 3
    such bands; a mixed dense/sparse numeric table shows 2 lanes and 18.

    Reuses ``reconstruct.has_recurring_numeric_columns`` (GH-248's lane-reuse
    rule, both x0 and x1 anchors per GH-349) rather than a second
    implementation of "column lane".

    Round 3 asks that helper for ``seeded_lanes`` rather than its default
    adjacency clustering. The detector's greedy chaining is safe where its
    answer is used positively but not here: one unrelated numeral printed
    between two real columns (a footnote value at x=18 between columns at
    x=12 and x=24, inside the 6pt tolerance of both) chains them into a single
    lane, this gate returns False, term (b) abstains, and the truncated
    candidate wins selection over the complete one. Recurrence-seeded lanes
    cannot be bridged by a position that occurs once, so the one-off bridge no
    longer flips this verdict.

    **What the verdict does and does not mean.** This is a bounded detector
    over TOKEN POSITIONS, not a semantic table detector. A positive verdict
    says extracted numeric tokens recur in shared x-lanes; it does not
    establish that those lanes are the cells of one table, and two token lanes
    are not proof of two physical columns. Two measured limitations:

    * A space-grouped number can be extracted as two tokens. Four lines of
      ``1 234`` printed at 6pt come back from PyMuPDF as two words per line
      (x=50.000 and x=55.004), which register as two recurring token lanes and
      return True. The adjacency clustering this gate replaced does the same on
      that fixture (its x1 anchor separates prefix from suffix), so this is the
      detector's standing limitation, not the seeding's. Signs and brackets are
      not affected: ``-0.5`` and ``(12)`` each extract as one token.
    * The question is asked of the WHOLE page, so bands that belong to no
      table -- numbered source citations, marker at one x and year at another
      -- can arm term (b) for a candidate whose own table is elsewhere.

    A negative verdict is correspondingly bounded: it says the page shows no
    recurring numeric token lanes at this arity, which is why term (b) then
    abstains rather than acquits.
    """
    if not words:
        return False
    from socr.tables.reconstruct import has_recurring_numeric_columns

    return has_recurring_numeric_columns(words, _MIN_RECONCILABLE_LANES, seeded_lanes=True)


#: #988: a footnote marker a model writes after a number as markup rather than
#: as a plain digit -- ``$^3$``, ``$^{2, 3}$`` or ``<sup>3</sup>``. Unicode
#: superscript digits (``³``) are mapped by ``_SUPERSCRIPT_DIGITS``.
_MARKUP_SUPERSCRIPT_RE = re.compile(r"\$\^\{?([\d,\s]+)\}?\$|<sup>([\d,\s]+)</sup>")
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

#: #988: what splits a candidate's text into tokens for the supply side of
#: term (b): whitespace, cell delimiters, Markdown emphasis asterisks and
#: ``<br>`` (a two-line cell is printed as two native lines). The asterisks
#: matter for chart labels written as ``- **18**: 0%``: unsplit, ``**18**:``
#: is not a numeric token and the native ``18`` reads as never written.
_CANDIDATE_TOKEN_SPLIT_RE = re.compile(r"[\s|*]+|<br\s*/?>")


def _glued_marker(match: re.Match) -> str:
    """The digits of a ``_MARKUP_SUPERSCRIPT_RE`` match, without the markup and
    without the separators of a multi-marker group (``$^{2, 3}$`` -> ``23``)."""
    return re.sub(r"[,\s]", "", match.group(1) or match.group(2))


def _candidate_numeric_supply(markdown: str) -> Counter:
    """#988: every genuine numeric token the candidate WROTE, anywhere in its
    text (table cells, ``<br>`` second lines, header lines, footnotes, footer),
    as a multiset.

    Uses ``row_corroboration._is_genuine_numeric``, the predicate
    ``baseline_bands`` applies to the native words, so both sides of term (b)
    normalise a token the same way. Before tokenising, a footnote marker the
    model wrote after a number is glued to it the way PyMuPDF's
    ``get_text("words")`` glues a printed superscript to the value it follows
    (measured on Coca-Cola 2019 p66: the 6 pt span ``0.32`` and the 4 pt
    superscript ``3`` come back as the one word ``0.323``). Without this the
    model's ``0.32³`` can never account for the native ``0.323``. Gluing is the
    conservative direction: when PyMuPDF does NOT glue a marker, the band stays
    unaccounted, which is the behaviour before #988.
    """
    from socr.tables.row_corroboration import _is_genuine_numeric

    text = (markdown or "").translate(_SUPERSCRIPT_DIGITS)
    text = _MARKUP_SUPERSCRIPT_RE.sub(_glued_marker, text)
    supply: Counter = Counter()
    for token in _CANDIDATE_TOKEN_SPLIT_RE.split(text):
        if not token:
            continue
        is_numeric, normalized = _is_genuine_numeric(token)
        if is_numeric:
            supply[normalized] += 1
    return supply


def _truncated_row_shortfall(words: list | None, markdown: str) -> bool:
    """TICKET-A2 (#645) term (b): the candidate's numeric body-row count falls
    short of the native table-shaped row count by more than A1b's own
    row-count allowance permits -- and by more than one stray header/legend
    band could explain (see ``_STRAY_HEADER_BAND_ALLOWANCE``) -- AND, since
    #988, so does the count of native bands whose numbers the candidate wrote.

    **#988: a row-count shortfall must be confirmed by content.** The row
    count compares the candidate's numeric BODY rows with every table-shaped
    band on the WHOLE page. The two sides count different things. The
    native side includes every printed line with
    ``row_shape_min`` numerals: the year caption over the columns, footnote
    lines, the page footer, prose with figures, chart labels. The candidate
    side counts only body rows, with header lines, ``<br>`` second lines and
    blank-stub rows excluded. On a page with one-value rows
    ``row_shape_min`` is 1, so every numbered line qualifies. Measured on the
    Coca-Cola 2018-2021 sustainability reports (cluster job 687398, 73
    rejected pages): the term fired on 58 of 73 cached answers, 32 of the 40
    ``table_truncated`` rejections were complete and correct readings, and no
    answer had lost a row. 2021 p74 (Gemini, every number matching the text
    layer) read 17 rows against 24 bands. The 7 surplus bands were the year
    caption, a lone superscript, two ``<br>`` second lines, two footnote
    lines and the footer.

    So the row count still runs first, unchanged, over the same bands
    (``row_corroboration.is_table_shaped_band``, the predicate
    ``table_shaped_native_row_count`` uses). When it falls short, the same
    inequality is applied a second time to the number of bands the candidate
    ACCOUNTS for, and the term fires only if that falls short too. A band is
    accounted for iff its numeric tokens can be drawn from the multiset of
    numeric tokens the candidate wrote anywhere in its text
    (``_candidate_numeric_supply``). The draw CONSUMES the tokens, so one
    written value accounts for one band. Requiring both counts means the term
    never refuses a candidate the row count alone accepts. A formatting
    difference between the two sides (``10,234`` against a printed
    ``10 234``, a footnote marker the model dropped) can only leave a band
    unaccounted on a page the row count already doubts. Region scoping of the
    native count (the family #988 rounds 1-4 tried and dropped) is not used:
    measured again on this corpus, a table-span count is blind to a cut tail.

    Measured on the 56 Coca-Cola answers the audit judged complete and
    correct (every value checked against the page image), with
    ``table_truncated`` alone (both terms). The term now fires on 6 of the
    56 (47 before #988). On synthetic cuts of the same answers (the largest
    table; 50 answers have one to cut):

    * last 10%, 25% or 50% of rows removed (a model that stopped): caught
      43, 41 and 44 times of 50 (before: 44, 42, 44);
    * the same shares deleted from the middle: caught 26, 41 and 44 times
      (before: 45, 46, 44, while also firing on 47 of the 56 uncut answers).

    Known limits: a dropped row is missed when its numbers also appear
    elsewhere in the candidate (``0%``, a repeated total) or when it is the
    one row the allowance absorbs. A candidate that restates a dropped row's
    numbers in prose is credited. The final-row term (a), unchanged, still
    catches a reply cut mid-row.

    Reuses ``row_corroboration.ROW_CORROBORATION_MIN`` (36/39) rather than a
    second named allowance, already measured and owned by A1b/A1a. Abstains
    (returns False) with no ``words`` -- exception-path callers of
    ``table_output_defect`` supply none, matching every other geometry-needing
    term in this module.

    Also abstains when the native page shows no recurring numeric column
    lanes (``_native_page_has_column_lanes``, #703): the reconciliation
    compares NUMERIC row shapes, and on a text table (prose cells, zero or one
    number each) the candidate-derived ``row_shape_min`` collapses to 1, at
    which point every prose line carrying a figure counts as a native table
    row and a complete candidate reads as a massive shortfall. Such candidates
    are left to term (a) and to the ladder verdict.

    ``row_shape_min`` is still the candidate's own minimum, deliberately. The
    lane gate has already established that this page HAS column structure, so
    counting native bands with one or more numerals is no longer "every prose
    line on a prose page" -- and keeping the minimum is what catches the
    sparse-prefix truncation the lane gate was added for (2 surviving rows of
    width 1 against 20 native bands). Substituting a lane-derived width would
    change the native counts A2 measured on the ECB fixtures for no gain on
    any page measured here.
    """
    if not words:
        return False
    if not _native_page_has_column_lanes(words):
        return False
    from socr.tables.row_corroboration import (
        ROW_CORROBORATION_MIN,
        baseline_bands,
        is_table_shaped_band,
        numeric_body_rows,
        table_blocks,
    )

    candidate_rows = [
        row for rows in table_blocks(markdown) for row in numeric_body_rows(rows) if row
    ]
    if not candidate_rows:
        return False

    row_shape_min = min(len(row) for row in candidate_rows)
    bands = [
        band.tokens
        for band in baseline_bands(words)
        if is_table_shaped_band(band.tokens, row_shape_min)
    ]
    if not _falls_short(len(candidate_rows), len(bands), ROW_CORROBORATION_MIN):
        return False

    # #988: the row count says rows are missing; confirm it by content.
    supply = _candidate_numeric_supply(markdown)
    accounted = 0
    for tokens in bands:
        needed = Counter(tokens)
        if not needed - supply:
            supply -= needed
            accounted += 1
    return _falls_short(accounted, len(bands), ROW_CORROBORATION_MIN)


def _falls_short(count: int, native_rows: int, corroboration_min: float) -> bool:
    """Term (b)'s inequality: *count* undercuts *native_rows* by more than one
    stray band (``_STRAY_HEADER_BAND_ALLOWANCE``) AND by more than A1b's
    row-count allowance (*corroboration_min*, ``ROW_CORROBORATION_MIN``)."""
    if native_rows <= 0:
        return False
    if count >= native_rows - _STRAY_HEADER_BAND_ALLOWANCE:
        return False
    return count < math.ceil(native_rows * corroboration_min)


def table_truncated(output_md: str, words: list | None) -> bool:
    """TICKET-A2 (#645): whether *output_md* looks like it was cut off
    mid-emission rather than being a complete (if otherwise defective)
    reading of the page.

    Two independent terms, either firing truncates the WHOLE candidate --
    partial content anywhere on the page is not a complete reading of it:

    (a) ``_final_row_truncated`` on any table block's RAW lines (before
        ``_parse_grid`` can reshape or drop the very row this is looking
        for) -- a final row breaking the block's own established
        leading/trailing-pipe style.
    (b) ``_truncated_row_shortfall`` -- the candidate's own numeric body
        rows undercount the native table-shaped row count by more than
        A1b's row-count allowance, i.e. whole rows are simply missing from
        the end (or middle) of the emission, AND (#988) the bands whose
        numbers the candidate wrote undercount it too.
    """
    for block_lines in raw_table_block_lines(output_md):
        if _final_row_truncated(block_lines):
            return True
    return _truncated_row_shortfall(words, output_md)


def table_output_defect(
    output_md: str,
    words: list | None,
    rules: list[tuple[float, float, float]] | None = None,
) -> str:
    """Whether *output_md* (the text about to ship) has a structural defect.

    GH-200, extended by GH-190, GH-212, and GH-226. A disjunction in stable
    precedence order:

    1. ``table_emission_defect`` on raw Markdown rows, before the delimiter is
       discarded: residual LaTeX table structure or a delimiter disagreeing
       with an otherwise rectangular header/body grid. String-only, no I/O.
    2. ``table_content_defect`` on raw Markdown rows, before ``_parse_grid``
       can erase all-blank rows. GH-190 closes the shipping consequence of
       that parser blind spot by inspecting raw rows; ``_parse_grid`` itself
       and reconciliation diffs remain blind and deliberately unchanged.
    3. ``table_truncated`` (TICKET-A2, #645), also on raw rows before
       ``_parse_grid`` can reshape or drop the row this term is looking for: a
       final row breaking its block's own leading/trailing-pipe style, or
       a numeric-row count undercutting the native table-shaped row count by
       more than A1b's row-count allowance permits, confirmed (#988) by the
       count of bands whose numbers the candidate wrote. Needs ``words`` for its
       second half; abstains on that half without it, same as term 4.
    4. ``structural_gate_fires`` on the emitted grid (B1's own predicate,
       ragged OR detached_label_rows, unchanged -- see ``structural_gate_fires``
       docstring). String-only, no I/O.
    5. ``header_cut.header_cut_verdict`` on each emitted table block. Runs only
       when the shape and preceding terms did not already fire, and only when
       the caller supplied both native words and the page's drawn horizontal
       rules.

    **On term 5 and the four reverts that precede it.** Earlier implementations
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

    **rules=None abstains.** Exception-path callers execute
    ``table_output_defect`` with no page geometry and receive the raw emission,
    raw content, and parsed shape terms. ``born_digital`` computes those three
    terms directly for its aggregate, then invokes ``table_output_defect``
    separately only for the header-attribution term. That division is intended,
    not an oversight.

    Deliberately NOT nested with TR-3 (``native_verifier.verify_native_table``)
    -- TR-3 is evaluated by the caller as a separate disjunct. Measured in
    ``docs/log/2026-08-14_gh151-b1-predicate-design.md``: the two signals do
    not subsume each other (TR-3 misses 31/66 shape-gate pages; the shape gate
    misses 27 pages TR-3 catches). Pure; no mutation, no model calls -- the
    caller precomputes ``rules``, so this never touches a page object.
    """
    emission_defect = table_emission_defect(output_md)
    if emission_defect:
        return emission_defect

    content_defect = table_content_defect(output_md)
    if content_defect:
        return content_defect

    if table_truncated(output_md, words):
        return DEFECT_TABLE_TRUNCATED

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
