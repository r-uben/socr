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
        values = [c for c in raw_values if _is_value(c)]

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
        rows.append(LabeledRow(path=tuple(p for p in path if p), values=tuple(values)))
        last_label = label

    diag.duplicate_rows = sum(n - 1 for n in seen.values() if n > 1)
    return rows, diag


def score_rows(gt: list[LabeledRow], predicted: list[LabeledRow]) -> ExactnessReport:
    """Compare predicted rows against ground truth, keyed by label in document order.

    Matching is greedy and in-order: the n-th ground-truth occurrence of a label is
    matched to the n-th predicted occurrence. That is what stops a label reused under
    two parents from being credited twice against the same source row - the collision
    that made an earlier scratch scorer understate a near-perfect page by 6 cells.
    """
    report = ExactnessReport(gt_rows=len(gt))

    pending: dict[str, list[LabeledRow]] = {}
    for row in predicted:
        pending.setdefault(row.key, []).append(row)

    for want in gt:
        report.cells += len(want.values)
        queue = pending.get(want.key)
        if not queue:
            report.rows_not_found += 1
            for column, expected in enumerate(want.values):
                report.misses.append(
                    CellMiss(path=want.path, column=column, expected=expected, got=None)
                )
            continue

        got = queue.pop(0)
        for column, expected in enumerate(want.values):
            actual = got.values[column] if column < len(got.values) else ""
            if BenchmarkScorer._norm_cell(actual) == BenchmarkScorer._norm_cell(expected):
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
    """Score a page's emitted markdown against its own native text layer."""
    gt = native_rows_from_page(page)
    predicted, diag = markdown_rows(markdown)
    report = score_rows(gt, predicted)
    report.orphan_rows = diag.orphan_rows
    report.labelled_but_empty = diag.labelled_but_empty
    report.duplicate_rows = diag.duplicate_rows
    return report
