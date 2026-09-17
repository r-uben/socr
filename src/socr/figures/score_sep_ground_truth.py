"""#747: score the SEP dot-plot corpus's reader and model readings against the
Fed's own published per-bin counts.

This is a developer instrument, not a pipeline stage or a CI gate -- see the
module docstring of ``socr.figures.sep_ground_truth`` for why a ground-truth
table cannot be a test fixture (it is fetched over the network, cached, and
the fetch itself can only run where a caller has network access) and
``docs/log/`` for why no corpus score is asserted anywhere in the suite: the
scores are a measurement, not yet a known constant, and CI has neither the
network nor a provider to reproduce them.

Every path is a caller argument, per this repo's rule against baking data into
code. Typical use::

    uv run socr-score-sep-ground-truth \\
        ~/Data/socr/sep-dotplots/in \\
        ~/Data/socr/sep-dotplots/ground-truth \\
        --model-dir ~/Data/socr/sep-dotplots/out

The reader side always runs fresh, against this tree's own
``read_chart_page`` -- a prior pipeline run's own on-disk reader output is not
an acceptable substitute (see the decision log this ticket adds), because a
stale run predates fixes this tree already carries. The model side, by
contrast, is read from a prior run's ``<doc>/<doc>.md`` model-authored grids:
a model's numbers come from a crop image largely independent of this tree's
reader geometry, so a stale run's model column is still informative even
where its reader column is not. Passing the same ``--model-dir`` a stale run
used is therefore intentional, not an oversight; pass a fresh one instead if a
fresh multi-engine run exists.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from socr.core.pdf import open_pdf
from socr.figures.chart_data import find_filled_grids
from socr.figures.chart_reader import INTEGER, PRESENT, read_chart_page
from socr.figures.chart_reconcile import _bin_key, _series_key
from socr.figures.sep_ground_truth import GroundTruthUnavailable, ReleaseTable, cached_release_table
from socr.tables.reconstruct import chart_region_bboxes

#: The corpus's own naming: ``sep-<release date>-p<page>.pdf``. The release
#: date is the ground-truth key; nothing about a filename is otherwise
#: interpreted.
_CORPUS_NAME_RE = re.compile(r"sep-(?P<date>\d{8})-p\d+")

#: A model-authored grid's panel identity is not carried in `FilledGrid`
#: itself (it names its columns and rows, not the page section it came from):
#: the corpus's own model output puts one Markdown heading directly above
#: each grid, naming the year (or "Longer run") panel it renders. This is a
#: property of THIS corpus's rendering, not a contract `chart_data` makes, so
#: it lives here rather than in `chart_data`.
_PANEL_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
#: The corpus's own rendering is not consistent about *how* it marks a panel
#: heading: some documents write it as a bold-only line (``**2020**``)
#: instead of a Markdown ``#`` heading. Measured on `sep-20201216-p09`, whose
#: first grid was silently attributed to the page's own `## Page 1` heading
#: (the nearest ``#`` heading above it) until this was added -- 24 cells
#: reported `no_ground_truth` for a harness reason, not a model one.
_BOLD_HEADING_RE = re.compile(r"^\*\*(.+?)\*\*\s*$")

#: Per-cell classification, in the order a caller should read them: an
#: unambiguous match, a count that is simply wrong, a paired swap between
#: adjacent bins (a dot in the wrong place, not a miscount -- no total can
#: catch it), a nonzero reading where the Fed shows nothing, and the two
#: bookkeeping cases (nothing was read; the reading names a bin ground truth
#: does not).
EXACT = "exact"
WRONG_COUNT = "wrong_count"
WRONG_BIN = "wrong_bin"
FABRICATED = "fabricated"
MISSING = "missing"
NO_GROUND_TRUTH = "no_ground_truth"


@dataclass(frozen=True)
class CellScore:
    doc: str
    panel: str
    series: str
    bin_label: str
    truth: int | None
    value: int | None
    outcome: str


def _release_date(pdf_path: Path) -> str:
    m = _CORPUS_NAME_RE.search(pdf_path.stem)
    if not m:
        raise ValueError(f"{pdf_path.name!r} does not match sep-<date>-p<page>.pdf")
    return m.group("date")


def _classify_series(
    doc: str,
    panel: str,
    series: str,
    bin_order: list[str],
    truth: dict[str, int],
    values: dict[str, int],
) -> list[CellScore]:
    """Score one (panel, series) column, bin by bin, against *truth*.

    *bin_order* is every bin key either side names, truth's own table order
    first (so a wrong-bin pair is checked against its true neighbour) and any
    bin only the reading names appended after. A "wrong bin" pair is detected
    before the per-cell fallback classification runs, because a swap between
    adjacent bins reads -- in isolation -- as one fabrication and one
    undercount, and scoring each cell alone would hide the one thing that
    makes it a swap rather than two independent errors.
    """
    diffs = {b: (values[b] - truth[b]) for b in bin_order if b in values and b in truth}
    wrong_bin_bins: set[str] = set()
    for a, b in zip(bin_order, bin_order[1:]):
        da, db = diffs.get(a), diffs.get(b)
        if da is not None and db is not None and da != 0 and da == -db:
            wrong_bin_bins.add(a)
            wrong_bin_bins.add(b)

    out = []
    for b in bin_order:
        t = truth.get(b)
        v = values.get(b)
        if v is None:
            outcome = MISSING
        elif t is None:
            outcome = NO_GROUND_TRUTH
        elif v == t:
            outcome = EXACT
        elif b in wrong_bin_bins:
            outcome = WRONG_BIN
        elif t == 0 and v > 0:
            outcome = FABRICATED
        else:
            outcome = WRONG_COUNT
        out.append(CellScore(doc, panel, series, b, t, v, outcome))
    return out


#: One panel's readings, keyed by series/header then bin. Kept as a list of
#: ``(panel_label, by_series)`` pairs rather than a ``dict[str, ...]`` keyed by
#: label: `PanelReading.label` and a model grid's heading text are corpus
#: content, not an identity guarantee -- two panels can genuinely share a
#: label (a layout change, a new corpus with repeated years, a duplicated
#: heading). A dict keyed by label silently collapses that collision, the
#: later panel overwriting the earlier one with no trace it happened (#752:
#: measured on a real release, three of four panels on one page vanished this
#: way -- 120 cells missing from the report, not even as `no_ground_truth`).
#: A list preserves every panel this function found, in the order it found
#: them, so nothing a caller iterates over is ever discarded for sharing a
#: label with something else on the same page.
PanelReadings = list[tuple[str, dict[str, dict[str, int]]]]


def _reader_readings(pdf_path: Path) -> PanelReadings:
    """A ``(panel_label, {series_label: {bin_key: count}})`` pair per panel on
    one corpus page, read fresh -- never from a prior run's on-disk output.
    See the module docstring for why a stale run is not an acceptable
    substitute here, and :data:`PanelReadings` for why this is a list rather
    than a dict keyed by label."""
    doc = open_pdf(pdf_path)
    try:
        page = doc[0]
        boxes = chart_region_bboxes(page)
        reading = read_chart_page(page, boxes, page_num=1, source_checksum=pdf_path.name)
    finally:
        doc.close()
    out: PanelReadings = []
    for panel in reading.panels.values():
        by_series: dict[str, dict[str, int]] = {}
        for series in panel.series:
            if series.presence != PRESENT:
                continue
            counts = {
                "|".join(_bin_key(c.bin_label)): c.count
                for c in series.cells
                if c.status == INTEGER and c.count is not None
            }
            by_series[series.name] = counts
        out.append((panel.label, by_series))
    return out


def _model_readings(model_doc_dir: Path, doc_stem: str) -> PanelReadings:
    """Same shape as :func:`_reader_readings`, from a prior run's model grids."""
    md_path = model_doc_dir / f"{doc_stem}.md"
    if not md_path.exists():
        return []
    text = md_path.read_text(encoding="utf-8", errors="replace")
    # `chart_data._parse_grids` indexes ``start``/``end`` into `text.split("\n")`,
    # not `text.splitlines()` -- the two differ on `\r\n` input, and matching a
    # grid's line index against the wrong split silently mislabels every panel.
    lines = text.split("\n")
    heading_at: dict[int, str] = {}
    current = ""
    for i, line in enumerate(lines):
        m = _PANEL_HEADING_RE.match(line) or _BOLD_HEADING_RE.match(line)
        if m:
            current = m.group(1)
        heading_at[i] = current
    out: PanelReadings = []
    for grid in find_filled_grids(text):
        panel_label = heading_at.get(grid.start, "")
        if not panel_label:
            continue
        if _is_transposed(grid.rows, grid.data_headers):
            # Measured on `sep-20251210-p09`: this corpus's model sometimes
            # writes bin ranges as the COLUMN headers and one generic row
            # label ("Number of Participants") instead of a bin-labelled row
            # per projection month. That row names no projection month a
            # ground-truth column could match, so it is skipped rather than
            # scored under a "series" named for a bin range -- a silent
            # transpose would misclassify every cell as `no_ground_truth`
            # for the wrong reason (an unmatched series, not a genuinely
            # absent one).
            continue
        by_series: dict[str, dict[str, int]] = {}
        for row_label, cells in grid.rows:
            bin_key = "|".join(_bin_key(row_label))
            for header, raw in zip(grid.data_headers, cells):
                value = _as_int(raw)
                if value is None:
                    continue
                by_series.setdefault(header, {})[bin_key] = value
        # One entry per grid, even when `panel_label` repeats a label already
        # appended above -- see `PanelReadings`. Two grids under the same
        # heading are two distinct series dicts here, never merged into one.
        out.append((panel_label, by_series))
    return out


def _is_transposed(rows: tuple[tuple[str, tuple[str, ...]], ...], data_headers: list[str]) -> bool:
    """A grid with bins along the header row instead of the label column.

    Every other grid in this corpus keys each row by a bin and each column by
    a projection month; this shape (one row, a generic label, and every
    column itself a bin range) is its transpose. Detected by content, not by
    position, using the same `_bin_key` atom count `chart_reconcile` already
    uses to tell a bin label from a categorical one (2 atoms vs. 1 or 0).
    """
    if len(rows) != 1:
        return False
    row_label, _cells = rows[0]
    return len(_bin_key(row_label)) != 2 and all(len(_bin_key(h)) == 2 for h in data_headers)


def _as_int(text: str) -> int | None:
    stripped = text.strip()
    return int(stripped) if re.fullmatch(r"-?\d+", stripped) else None


def _truth_for_panel(rt: ReleaseTable, panel_label: str) -> dict[str, dict[str, int]] | None:
    panel = rt.panel(panel_label)
    if panel is None:
        return None
    return {
        series_label: {"|".join(_bin_key(c.bin_label)): c.count for c in cells}
        for series_label, cells in panel.columns.items()
    }


def _score_side(
    doc: str,
    side_readings: PanelReadings,
    rt: ReleaseTable,
) -> list[CellScore]:
    out: list[CellScore] = []
    for panel_label, by_series in side_readings:
        truth_by_series = _truth_for_panel(rt, panel_label)
        if truth_by_series is None:
            for series_label, values in by_series.items():
                out.extend(
                    CellScore(doc, panel_label, series_label, b, None, v, NO_GROUND_TRUTH)
                    for b, v in values.items()
                )
            continue
        norm_truth = {_series_key(s): (s, cells) for s, cells in truth_by_series.items()}
        for series_label, values in by_series.items():
            match = norm_truth.get(_series_key(series_label))
            if match is None:
                out.extend(
                    CellScore(doc, panel_label, series_label, b, None, v, NO_GROUND_TRUTH)
                    for b, v in values.items()
                )
                continue
            truth_label, truth = match
            bin_order = list(truth.keys()) + [b for b in values if b not in truth]
            out.extend(_classify_series(doc, panel_label, truth_label, bin_order, truth, values))
    return out


def _run(corpus_dir: Path, cache_dir: Path, model_dir: Path | None) -> dict:
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs under {corpus_dir}")
    reader_scores: list[CellScore] = []
    model_scores: list[CellScore] = []
    skipped: list[str] = []
    for pdf_path in pdfs:
        doc_stem = pdf_path.stem
        release_date = _release_date(pdf_path)
        try:
            rt = cached_release_table(release_date, cache_dir)
        except GroundTruthUnavailable as exc:
            skipped.append(f"{doc_stem}: {exc}")
            continue
        reader_scores.extend(_score_side(doc_stem, _reader_readings(pdf_path), rt))
        if model_dir is not None:
            model_scores.extend(
                _score_side(doc_stem, _model_readings(model_dir / doc_stem, doc_stem), rt)
            )
    return {
        "documents_scored": len(pdfs) - len(skipped),
        "documents_skipped": skipped,
        "reader": _summarize(reader_scores),
        "model": _summarize(model_scores) if model_dir is not None else None,
    }


def _summarize(scores: list[CellScore]) -> dict:
    counts = Counter(s.outcome for s in scores)
    mistakes = [s for s in scores if s.outcome not in (EXACT, NO_GROUND_TRUTH)]
    return {
        "cells": len(scores),
        "by_outcome": dict(counts),
        "mistakes": [
            {
                "doc": s.doc,
                "panel": s.panel,
                "series": s.series,
                "bin": s.bin_label,
                "truth": s.truth,
                "value": s.value,
                "outcome": s.outcome,
            }
            for s in mistakes
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus_dir", type=Path, help="a directory of sep-<date>-p<page>.pdf files")
    parser.add_argument(
        "cache_dir", type=Path, help="where fetched Fed ground-truth pages are cached"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="a prior pipeline run's per-document output directories, for model-vs-truth scoring",
    )
    parser.add_argument("--json", type=Path, help="write the full report here as well")
    args = parser.parse_args()
    report = _run(args.corpus_dir, args.cache_dir, args.model_dir)
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - module-execution entry point
    raise SystemExit(main())
