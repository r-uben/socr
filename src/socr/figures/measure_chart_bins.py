"""Measure the Stage 1 bin-row selection over a corpus of single-page PDFs.

This is the instrument behind every number in the #735 round-6 notes
(``docs/log/2026-09-12_735-sep-reader.md``), committed so those numbers can be
reproduced from the tree rather than rebuilt by hand. It is a developer tool,
not part of the pipeline: it drives ``read_chart_page`` directly, on CPU, and
writes nothing but its own report.

Every corpus path is an argument -- nothing about a machine's layout is baked
in. Typical use::

    uv run socr-measure-chart-bins \\
        ~/Data/socr/sep-dotplots/in \\
        ~/Data/socr/sep-dotplots/in-minutes \\
        ~/Data/socr/fixtures/dotplot

Each argument is a directory of PDFs or a single PDF. The report gives, per
corpus and in total, the counts the round-6 rule was chosen against: how many
rows the bars corroborate, how often the best score ties, whether the attested
row is the topmost row inside the axis' span, and whether any winner moves when a
bar is required to lie inside the bin it covers. Two scorings are reported side by side, because the candidate
rules the round-6 notes rejected were measured against the ROUND 5 scoring
(coverage of exactly one token centre) while the shipped reader also requires
containment -- the same corpus gives different counts under the two, and saying
which is meant is half of what these numbers are for.

Round 7 replaced the round-6 rival rule with a numeric-label gate, so the report
also counts how many rows below each axis carry only numbers, how many panels
have no such row at all (the gate's refusals), and whether the attested row is
one of them. On both Fed corpora and the reference those are 0 refusals and
every attested row numeric, which is the "costs nothing measured" claim in
``docs/log/2026-09-12_735-sep-reader.md``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from socr.core.pdf import open_pdf
from socr.figures.chart_reader import (
    _alnum_tokens,
    _attesting_bars,
    _bin_edges,
    _numeric_row,
    _resting_bars,
    read_chart_page,
)
from socr.tables.reconstruct import chart_region_bboxes


def _pdfs(target: Path) -> list[Path]:
    return [target] if target.is_file() else sorted(target.glob("*.pdf"))


def _covering_score(tokens, bars) -> int:
    """Round 5's corroboration: bars covering exactly one of the row's centres."""
    return sum(1 for b in bars if sum(1 for cx, _t in tokens if b.x0 <= cx <= b.x1) == 1)


class Measurement:
    """Counts accumulated over every ``read_bins`` call of a corpus."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.clearances: list[float] = []
        self.non_numeric_winners: list[str] = []

    def observe(self, frame, rows, marks, residual) -> None:
        below = [r for r in rows if r.y0 > frame.baseline]
        plural = [r for r in below if len(_alnum_tokens(r)) >= 2]
        bars = _resting_bars(frame, residual, marks)
        tokens = [_alnum_tokens(r) for r in plural]
        covering = [_covering_score(t, bars) for t in tokens]
        contained = [len(_attesting_bars(r, bars)) for r in plural]
        c = self.counts
        c["calls"] += 1
        c["rows_all_numeric"] += sum(1 for r in plural if _numeric_row(r))
        c["calls_with_no_numeric_row"] += not any(_numeric_row(r) for r in plural)
        c["corroborated_2_or_more_covering"] += sum(1 for s in covering if s) >= 2
        c["corroborated_2_or_more_contained"] += sum(1 for s in contained if s) >= 2
        best = max(covering, default=0)
        if not best:
            c["no_row_corroborated"] += 1
            return
        c["best_ties"] += covering.count(best) > 1
        c["winner_attested_by_one_bar"] += best == 1
        w = covering.index(best)
        win = tokens[w]
        in_span = [
            i for i, t in enumerate(tokens) if all(frame.x0 <= cx <= frame.x1 for cx, _t in t)
        ]
        c["winner_topmost_in_span"] += bool(in_span) and w == in_span[0]
        c["winner_has_most_tokens"] += len(win) >= max(
            len(t) for t, s in zip(tokens, covering, strict=True) if s
        )
        c["winner_tokens_equal_bar_count"] += len(win) == len(bars)
        c["plural_in_span_rows_above_winner"] += sum(1 for i in in_span if i < w)
        best_contained = max(contained, default=0)
        c["winner_moves_under_containment"] += (
            not best_contained or contained.index(best_contained) != w
        )
        edges = _bin_edges([cx for cx, _t in win])
        for bar in bars:
            covered = [i for i, (cx, _t) in enumerate(win) if bar.x0 <= cx <= bar.x1]
            if len(covered) == 1:
                i = covered[0]
                self.clearances.append(min(bar.x0 - edges[i], edges[i + 1] - bar.x1))
        c["winner_is_numeric"] += _numeric_row(plural[w])
        if not _numeric_row(plural[w]):
            self.non_numeric_winners.append(" ".join(text for _cx, text in win)[:80])

    def report(self) -> dict:
        out = dict(self.counts)
        out["minimum_clearance_points"] = min(self.clearances, default=None)
        out["non_numeric_winners"] = self.non_numeric_winners
        return out


def _run(targets: list[Path]) -> dict:
    total = Measurement()
    per_corpus: dict[str, dict] = {}
    absorbed = Counter()
    for target in targets:
        corpus = Measurement()
        files = _pdfs(target)
        if not files:
            raise SystemExit(f"no PDFs under {target}")
        for path in files:
            doc = open_pdf(path)
            for page in doc:
                boxes = chart_region_bboxes(page)
                if not boxes:
                    continue
                _instrument(page, boxes, [corpus, total], absorbed)
            doc.close()
        # Keyed by the path as given rather than its basename: two corpora
        # both called "in" under different parents would otherwise
        # overwrite each other's report while the totals kept both.
        per_corpus[str(target)] = {"files": len(files), **corpus.report()}
    return {
        "per_corpus": per_corpus,
        "total": total.report(),
        "second_lines_absorbed": absorbed["absorbed"],
        "second_lines_absorbed_with_a_multi_token_column": absorbed["multi_token_column"],
    }


def _instrument(page, boxes, measurements: list[Measurement], absorbed: Counter) -> None:
    """Run one page, observing every bin-row selection and every absorbed line."""
    from socr.figures import chart_reader as cr

    read_bins, aligned = cr.read_bins, cr._aligned

    def watched_bins(frame, rows, marks, residual):
        for m in measurements:
            m.observe(frame, rows, marks, residual)
        return read_bins(frame, rows, marks, residual)

    def watched_aligned(*args, **kwargs):
        # Signature-agnostic on purpose. This wrapper took (row, centres) and
        # broke the whole tool the moment ``_aligned`` gained its axis-span
        # argument: every corpus page raised TypeError from inside read_bins,
        # while the tool's own --help still exited 0 (#735 round 8 review).
        out = aligned(*args, **kwargs)
        if out is not None:
            absorbed["absorbed"] += 1
            absorbed["multi_token_column"] += any(" " in fragment for fragment in out)
        return out

    cr.read_bins, cr._aligned = watched_bins, watched_aligned
    try:
        read_chart_page(page, boxes, page_num=1, source_checksum="measurement")
    finally:
        cr.read_bins, cr._aligned = read_bins, aligned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", nargs="+", type=Path, help="a directory of PDFs, or one PDF")
    parser.add_argument("--json", type=Path, help="write the full report here as well")
    args = parser.parse_args()
    report = _run(args.corpus)
    text = json.dumps(report, indent=2, default=float)
    if args.json:
        args.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - module-execution entry point
    # Without this, ``python -m socr.figures.measure_chart_bins <dirs>`` exits 0
    # having printed nothing, which is indistinguishable from a corpus with no
    # chart pages. A silent success is how the first numeric gate got through.
    raise SystemExit(main())
