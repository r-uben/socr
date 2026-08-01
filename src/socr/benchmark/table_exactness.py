"""GH-96: hierarchy-aware cell exactness for tables.

``BenchmarkScorer.score_table_cells`` compares numeric cells *positionally* and,
when the predicted grid's shape differs from the ground truth's, degrades to
multiset recall of the GT's numeric values. That fallback is blind to the failure
mode this module exists to measure.

On a hierarchical fiscal table the local VLM produces a *permutation*: every digit
is present, in the right column, in the right order, but the label-to-row binding
slides by one inside each nested block::

                                  native      emitted
    September energy package       43.2       (empty)
      Energy price guarantee       24.8         43.2
      Energy bill relief scheme    18.4         24.8
      (orphan row, no label)          -         18.4

Because no value is lost, multiset recall is 100%. Measured on OBR EFO November
2022 page 13, ``score_table_cells`` reports **100.0%** for output that is **32.5%**
correct by label, and gives the identical 100.0% to a near-perfect transcription of
the same page. It cannot separate them.

This module keys cells to *rows identified by label*, matched in document order so
that a label reused under two different parents (``Other measures`` appears twice in
that table) cannot be credited twice against the same source row.

Deliberately model-free: on a born-digital page the native text layer supplies the
true label order and every digit at zero cost.

## #123 TICKET-B2: column position, once rows are matched

Matching a row by label says nothing about whether its *values* landed in the right
column. ``markdown_rows`` used to filter empty cells out of a row before scoring, and
``native_rows_from_page`` read values in x-order with no record of which column each
came from — so ``['-1.0', '1.0', '28.1', '']`` and ``['2.5', '0.5', '', '13.8']`` both
reduced to three values and both matched a three-value ground truth. A value in the
wrong column scored as correct.

The fix keeps ``LabeledRow.values`` compacted on both sides (unchanged: it is what
keeps ``cells`` — ground-truth only — a stable denominator) and adds two *parallel*
position fields: ``lanes`` on the ground-truth side (an anonymous, page-wide column
index assigned by clustering native value positions) and ``columns`` on the markdown
side (each value's own position in its emitted row, before compaction). Scoring then
looks for **one global, monotone, injective map** from lanes to columns — chosen to
maximise total cell agreement across every matched row on the page — rather than
comparing raw ordinal positions. "Global" and "monotone" both matter: a *local*,
per-row map would let column identity drift row to row, and a non-monotone map could
explain a genuine column *transposition* as if it were correct.

**Uniform-shift limitation.** A whole-page shift by exactly one lane in the same
direction, applied identically to incumbent and candidate, is invisible to this
metric in the specific case where every shifted value happens to still land on some
lane the map can use — column identity here is positional and page-relative, not
semantic (no header is read), so the metric cannot distinguish "the right value in
the right column" from "the right value in a column that coincidentally lines up".

**Map-search freedom (do not dismiss this).** The map is fitted to the *prediction*
being scored, so a wider emitted table has strictly more admissible monotone maps to
choose from (``C(M, L)`` of them, at ``L`` lanes and ``M`` columns) — freedom that
grows with ``M``. The mechanism a wider output could exploit is low-entropy, repeated
cell values (``0.0``, ``-``, blanks): more columns give the map more paths to route a
lane through one of them and harvest a coincidental match. This was weighed against
two corrections — penalising unmapped candidate columns, and scoring both sides under
the incumbent's map — and both were rejected: the first makes strictness a function of
the *prediction's* shape rather than the ground truth's, and the second is
structurally anti-improvement (a candidate that correctly finds a stub column the
incumbent missed shifts everything after it by one column and would score near zero
*for being right*). The freedom is accepted and shipped; ``lane_count``,
``column_count``, ``lane_to_column`` and ``unmapped_nonempty_columns`` on
``ExactnessReport`` exist so a suspicious win is auditable, never so it is penalised.

## #123 TICKET-C1: unexplained lanes are content loss, not diagnostics

``unmapped_nonempty_columns`` is the emitted side of the gap; ``unexplained_lanes``
is its mirror on the ground-truth side — lanes that carry a value in a matched row
but that the best map never gave a column. That is not a shape mismatch to audit
quietly; it is native-layer content with nowhere to land in the output, which is
exactly what the repo's no-silent-content-loss rule exists to catch. Unlike the raw
``native_lanes``/``output_cols`` gap logged by ``native_verifier`` (frequently
benign — spanning or paired headers routinely produce a wide gap on a page that is
fine), "does at least one lane go unexplained" is threshold-free: zero versus
non-zero, not a tuned cutoff. ``escalation_decision`` therefore treats it as a
lexicographically prior signal to ``exact`` (see that module), and the pipeline
surfaces it as a named audit kind via ``tables_trust`` rather than only a diagnostic
field nobody reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from socr.benchmark.scorer import BenchmarkScorer
from socr.tables.native_rows import (  # noqa: F401  (re-exported for the metric's callers)
    _MARKER_RE,
    LabeledRow,
    _is_value,
    _superscript_tokens,
    native_rows_from_page,
    normalize_label,
    rows_establish_grid,
)
from socr.tables.native_verifier import strip_presentation


@dataclass
class CellMiss:
    """One disagreement, attributed to a named row."""

    path: tuple[str, ...]
    column: int
    expected: str
    got: str | None  # None when the row was never found in the prediction


@dataclass
class ExactnessReport:
    """Hierarchy-aware exactness for one page."""

    gt_rows: int = 0
    cells: int = 0
    exact: int = 0
    rows_not_found: int = 0
    orphan_rows: int = 0  # prediction rows carrying values but no label
    labelled_but_empty: int = 0  # prediction rows with a label and no values
    duplicate_rows: int = 0  # byte-identical repeated prediction rows
    lane_ambiguous_cells: int = 0  # GT values with no page-wide lane support (#123 B2)
    # Diagnostics only (#123 TICKET-B2) — NEVER used to penalise a score, only to make
    # a suspicious win auditable. See the module docstring's "map-search freedom".
    lane_count: int = 0  # L: distinct ground-truth lanes on the page
    column_count: int = 0  # M: distinct emitted markdown columns on the page
    lane_to_column: dict[int, int] = field(default_factory=dict)  # the chosen map
    unmapped_nonempty_columns: int = 0  # emitted columns with values the map didn't use
    # #123 TICKET-C1: the mirror image of unmapped_nonempty_columns. A ground-truth
    # lane that carries a value in a matched row but received no column under the
    # best map — content the native layer has and the emitted table has nowhere to
    # put. Unlike the raw lane/column gap this is threshold-free: zero versus
    # non-zero is a fact about the data, not a tuned cutoff.
    unexplained_lanes: int = 0
    misses: list[CellMiss] = field(default_factory=list)
    ceiling_note: str = ""

    @property
    def pct(self) -> float | None:
        """Exact cells as a percentage, or None when there is nothing to score."""
        if not self.cells:
            return None
        return round(100.0 * self.exact / self.cells, 1)

    @property
    def scorable(self) -> bool:
        """False when the ground truth is too degenerate to draw a conclusion from.

        A tie between two engines on a page where this is False means the parser hit
        its ceiling, NOT that both engines failed.
        """
        return not self.ceiling_note


def _split_label_and_values(cells: list[str]) -> tuple[str, list[str]]:
    """Row label (column 0) and its ordered value cells."""
    if not cells:
        return "", []
    label = cells[0].replace("**", "").strip()
    return label, [strip_presentation(c) for c in cells[1:]]


def markdown_rows(markdown: str) -> tuple[list[LabeledRow], ExactnessReport]:
    """Parse a markdown table into labelled rows, plus shape diagnostics.

    The returned report carries only the prediction-side counters
    (``orphan_rows``, ``labelled_but_empty``, ``duplicate_rows``); scoring fills in
    the rest.
    """
    grid = BenchmarkScorer._markdown_table_cells(markdown)
    diag = ExactnessReport()

    seen: dict[tuple[str, ...], int] = {}
    rows: list[LabeledRow] = []
    parent: str | None = None
    last_label: str | None = None

    for cells in grid:
        label, raw_values = _split_label_and_values(cells)
        columns = [i for i, c in enumerate(raw_values) if _is_value(c)]
        values = [raw_values[i] for i in columns]

        signature = tuple(cells)
        seen[signature] = seen.get(signature, 0) + 1

        if _MARKER_RE.match(label):
            # The marker itself is structural: subsequent rows are children of the
            # last labelled row.
            parent = last_label
            continue

        if not label:
            if values:
                diag.orphan_rows += 1
            continue

        if not values:
            diag.labelled_but_empty += 1
            last_label = label
            # A labelled row with no values is a candidate parent, but only the
            # explicit marker actually opens a child block.
            continue

        path = (parent, label) if parent else (label,)
        rows.append(
            LabeledRow(
                path=tuple(p for p in path if p),
                values=tuple(values),
                columns=tuple(columns),
            )
        )
        last_label = label

    diag.duplicate_rows = sum(n - 1 for n in seen.values() if n > 1)
    return rows, diag


def _best_lane_column_map(
    matched: list[tuple[LabeledRow, LabeledRow]], lane_count: int, column_count: int
) -> dict[int, int]:
    """The single monotone, injective lane -> column map maximising total agreement.

    #123 TICKET-B2. A Needleman-Wunsch-shaped DP over an ``L`` x ``M`` grid: lane
    ``i`` may map to column ``j`` (score = how many matched rows agree there), skip a
    lane (it maps nowhere), or skip a column (nothing maps to it) — never both a skip
    and a map at once, and never non-monotonically (lane order must match column
    order, so a genuine transposition cannot be explained away). ``O(L*M)``, pure
    Python, negligible next to the OCR call this gates.

    Ties are broken deterministically — map over skip-column over skip-lane — so a
    tied score always yields the same map; only ``misses`` attribution could differ
    between two equally-scoring maps, never ``exact``.
    """
    if lane_count == 0 or column_count == 0:
        return {}

    score = [[0] * column_count for _ in range(lane_count)]
    for want, got in matched:
        got_by_column = dict(zip(got.columns, got.values))
        for lane, expected in zip(want.lanes, want.values):
            if not (0 <= lane < lane_count):
                continue
            for column, actual in got_by_column.items():
                if column >= column_count:
                    continue
                if BenchmarkScorer._norm_cell(actual) == BenchmarkScorer._norm_cell(expected):
                    score[lane][column] += 1

    dp = [[0] * (column_count + 1) for _ in range(lane_count + 1)]
    choice = [[""] * (column_count + 1) for _ in range(lane_count + 1)]
    for i in range(lane_count + 1):
        for j in range(column_count + 1):
            if i == 0 and j == 0:
                continue
            best, best_choice = -1, ""
            if i > 0 and j > 0:
                candidate = dp[i - 1][j - 1] + score[i - 1][j - 1]
                if candidate > best:
                    best, best_choice = candidate, "map"
            if j > 0 and dp[i][j - 1] > best:
                best, best_choice = dp[i][j - 1], "skip_column"
            if i > 0 and dp[i - 1][j] > best:
                best, best_choice = dp[i - 1][j], "skip_lane"
            dp[i][j] = best
            choice[i][j] = best_choice

    mapping: dict[int, int] = {}
    i, j = lane_count, column_count
    while i > 0 or j > 0:
        move = choice[i][j]
        if move == "map":
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif move == "skip_column":
            j -= 1
        else:
            i -= 1
    return mapping


def score_rows(gt: list[LabeledRow], predicted: list[LabeledRow]) -> ExactnessReport:
    """Compare predicted rows against ground truth, keyed by label in document order.

    Matching is greedy and in-order: the n-th ground-truth occurrence of a label is
    matched to the n-th predicted occurrence. That is what stops a label reused under
    two parents from being credited twice against the same source row - the collision
    that made an earlier scratch scorer understate a near-perfect page by 6 cells.

    #123 TICKET-B2: once rows are matched by label, each cell is compared through the
    page's single global lane->column map (see the module docstring), not by raw
    ordinal position. A ``LabeledRow`` built without position data (``lanes``/
    ``columns`` both the default ``()``) - as most of this module's own unit tests
    build them - falls back to the pre-B2 ordinal comparison for that row pair, so
    hand-built fixtures keep their original meaning.
    """
    report = ExactnessReport(gt_rows=len(gt))

    pending: dict[str, list[LabeledRow]] = {}
    for row in predicted:
        pending.setdefault(row.key, []).append(row)

    matched: list[tuple[LabeledRow, LabeledRow]] = []
    not_found: list[LabeledRow] = []
    for want in gt:
        report.cells += len(want.values)
        queue = pending.get(want.key)
        if not queue:
            not_found.append(want)
            continue
        matched.append((want, queue.pop(0)))

    report.rows_not_found = len(not_found)
    for want in not_found:
        for column, expected in enumerate(want.values):
            report.misses.append(
                CellMiss(path=want.path, column=column, expected=expected, got=None)
            )

    lane_count = (
        max((lane for want, _got in matched for lane in want.lanes if lane >= 0), default=-1) + 1
    )
    column_count = max((c for row in predicted for c in row.columns), default=-1) + 1
    lane_to_column = _best_lane_column_map(matched, lane_count, column_count)
    report.lane_count = lane_count
    report.column_count = column_count
    report.lane_to_column = dict(lane_to_column)

    used_columns = set(lane_to_column.values())
    all_columns_with_values = {c for row in predicted for c in row.columns}
    report.unmapped_nonempty_columns = len(all_columns_with_values - used_columns)

    # #123 TICKET-C1: lanes that carry a value in a matched row but the map never
    # used. A lane the map maps to a wrong column still counts as "explained" here
    # (the miss is a value error, not a missing home); this only counts lanes with
    # nowhere to go at all.
    matched_lanes_with_values = {lane for want, _got in matched for lane in want.lanes if lane >= 0}
    report.unexplained_lanes = len(matched_lanes_with_values - set(lane_to_column))

    for want, got in matched:
        if not want.lanes or not got.columns:
            # No position data on one side (a hand-built fixture that never set
            # lanes/columns): degrade to the pre-B2 ordinal comparison rather than
            # treating every value as lane-ambiguous.
            for column, expected in enumerate(want.values):
                actual = got.values[column] if column < len(got.values) else ""
                if BenchmarkScorer._norm_cell(actual) == BenchmarkScorer._norm_cell(expected):
                    report.exact += 1
                else:
                    report.misses.append(
                        CellMiss(path=want.path, column=column, expected=expected, got=actual)
                    )
            continue

        got_by_column = dict(zip(got.columns, got.values))
        row_mapped_columns = {
            lane_to_column[lane] for lane in want.lanes if lane >= 0 and lane in lane_to_column
        }
        row_unmapped_values = [v for c, v in got_by_column.items() if c not in row_mapped_columns]

        for column, expected in enumerate(want.values):
            lane = want.lanes[column] if column < len(want.lanes) else -1

            if lane < 0:
                # Lane-ambiguous (#123 TICKET-B2 design note §4): excluded from the
                # alignment - it must not consume a lane slot in the injective map,
                # the "stray numeral drags the boundary" failure in miniature.
                # Credited only on presence among the row's OWN unmapped columns,
                # never guessed positionally.
                report.lane_ambiguous_cells += 1
                if any(
                    BenchmarkScorer._norm_cell(v) == BenchmarkScorer._norm_cell(expected)
                    for v in row_unmapped_values
                ):
                    report.exact += 1
                else:
                    report.misses.append(
                        CellMiss(path=want.path, column=column, expected=expected, got=None)
                    )
                continue

            mapped_column = lane_to_column.get(lane)
            actual = got_by_column.get(mapped_column) if mapped_column is not None else None
            if actual is not None and BenchmarkScorer._norm_cell(
                actual
            ) == BenchmarkScorer._norm_cell(expected):
                report.exact += 1
            else:
                report.misses.append(
                    CellMiss(path=want.path, column=column, expected=expected, got=actual)
                )

    if len(gt) < 2:
        report.ceiling_note = f"ground truth parsed only {len(gt)} row(s); not scorable"
    elif report.rows_not_found == len(gt) and predicted:
        # The prediction HAS rows, they just don't match any ground-truth label.
        # That is as likely to be a parser failure as an engine failure, so the
        # score is not evidence either way.
        #
        # Deliberately NOT flagged when the prediction is empty: an engine that
        # emitted no table at all has genuinely failed, and a 0% there is a real
        # measurement. Conflating the two would have hidden the biggest wins in the
        # GH-96 calibration (pages 46, 48, 55 emitted nothing).
        report.ceiling_note = (
            "no ground-truth row label matched the prediction; "
            "the parser, not the engine, may be at fault"
        )
    return report


def score_page(page, markdown: str) -> ExactnessReport:
    """Score a page's emitted markdown against its own native text layer.

    #123 TICKET-B1: the native row parser reads chart axis labels and legends as if they
    were table rows (OBR reference page 54: prose plus two fan charts, no table,
    scored 0.0% against five fabricated ground-truth rows). Before scoring,
    ``gt`` must clear the same GH-113 grid predicate the escalation trigger
    uses. Below it, this returns with ``gt_rows=0`` and ``cells=0`` — not a
    fabricated-row count — so a caller thresholding on ``gt_rows`` (as
    ``escalation_decision`` does) also treats the page as having no usable
    ground truth, rather than only ``.scorable`` catching it.
    """
    gt = native_rows_from_page(page)
    if not rows_establish_grid(gt):
        return ExactnessReport(
            ceiling_note=(
                f"native text layer parsed {len(gt)} row(s) that do not form a "
                "grid (need at least two value columns shared by at least two "
                "rows); ground truth does not establish a table, not scorable"
            )
        )
    predicted, diag = markdown_rows(markdown)
    report = score_rows(gt, predicted)
    report.orphan_rows = diag.orphan_rows
    report.labelled_but_empty = diag.labelled_but_empty
    report.duplicate_rows = diag.duplicate_rows
    return report
