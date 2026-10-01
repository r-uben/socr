"""GH-916: evidence for ``ship_gate._PANEL_GAP_ROWS``.

The native-first ship gate extends a table's vertical span past its first/last
paired row through full-width rows no further than ``_PANEL_GAP_ROWS`` row pitches
away (``ship_gate.extended_span``). This tool measures, on a corpus of native-first
SHIP pages, what that bound should be:

(a) the distribution of gaps (in the block's own row pitches) between consecutive
    core paired rows, and which pages sit in its tail;
(b) for each candidate bound, whether the gate still reaches the omitted rows of
    known dropped-panel pages (``--known DOC_PREFIX:PAGE``);
(c) for each candidate bound, the pages that fire ``data_row_missing`` ONLY because
    of the extension (the rows are outside the core span), i.e. the pages where a
    larger bound pulls in more numeric lines. Inspect those against the page image.

Inputs are the two page sets the gate was measured on: a rotated-pages index
(``[{"doc","page"}]``, upright re-read via ``attempt_rotated_native_table``) and a
census of upright native-first SHIP pages (one JSON object per line with ``doc``,
``page``, ``upright``). Counts and basenames only; no page text is written.

    uv run socr-measure-ship-gate-gaps --pdf-dir ~/papers/pdf \
        --rotated-index gh902/q917/index.json --census upright-census/census.jsonl \
        --known 2017__fama__ap.pdf:398 --known lopez_lira_tang_zhu:32
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import statistics
from pathlib import Path

DEFAULT_BOUNDS = ("0", "1", "2", "3", "4", "5", "6", "8", "10", "none")


def _bound(text: str) -> float | None:
    return None if text == "none" else float(text)


def _page_inputs(args):
    """Yield ``(set_name, doc, page, words, markdown)`` for every page to measure."""
    import pymupdf

    from socr.tables.native_first import attempt_rotated_native_table

    pdf_dir = Path(args.pdf_dir).expanduser()
    if args.rotated_index:
        for item in json.loads(Path(args.rotated_index).expanduser().read_text()):
            with pymupdf.open(pdf_dir / item["doc"]) as doc:
                attempt = attempt_rotated_native_table(doc.load_page(item["page"] - 1))
            if attempt is not None:
                yield "rotated", item["doc"], item["page"], attempt.words, attempt.markdown
    if args.census:
        from socr.core.config import PipelineConfig
        from socr.core.document import DocumentHandle
        from socr.core.state import DocumentState
        from socr.pipeline.orchestrator import UnifiedPipeline

        recs = [
            json.loads(line)
            for line in Path(args.census).expanduser().read_text().splitlines()
            if line.strip()
        ]
        pages: dict[tuple[str, int], None] = {}
        for rec in recs:
            if rec.get("upright"):
                pages[(rec["doc"], rec["page"])] = None
        by_doc: dict[str, list[int]] = collections.defaultdict(list)
        for doc_name, page in pages:
            by_doc[doc_name].append(page)
        config = PipelineConfig()
        config.quiet = True
        config.judge_backend = "heuristic"
        pipe = UnifiedPipeline(config)
        pipe._resolve_judge_model = lambda: ""
        pipe._available_engines_for_agentic = lambda: []
        for doc_name, page_nums in by_doc.items():
            state = DocumentState(handle=DocumentHandle(path=pdf_dir / doc_name))
            pipe._phase_analyze(state)
            with pymupdf.open(pdf_dir / doc_name) as doc:
                for page in page_nums:
                    words = list(doc.load_page(page - 1).get_text("words"))
                    yield "upright", doc_name, page, words, state.pages[page].native_text or ""


def measure(inputs, bounds, known):
    from socr.tables import ship_gate as g

    gaps: list[tuple[str, str, int, float]] = []
    fires: dict[str, dict[tuple[str, int], int]] = {b: {} for b in bounds}
    for set_name, doc, page, words, markdown in inputs:
        blocks = g._output_blocks(markdown)
        if not blocks or not words:
            continue
        src = g._source_rows(words)
        anchors = g._Anchors(blocks, src)
        for found in anchors.per_block:
            geo = g._table_geometry(found, src)
            if not geo:
                continue
            ys = sorted(geo[1])
            diffs = [b - a for a, b in zip(ys, ys[1:])]
            if len(diffs) < 2:
                continue
            pitch = statistics.median(diffs)
            if pitch > 0:
                gaps += [(set_name, doc, page, round(d / pitch, 2)) for d in diffs]
        for b in bounds:
            found_faults = g.data_row_missing_faults(blocks, anchors, src, _bound(b))
            if found_faults:
                fires[b][(doc, page)] = len(found_faults)
    return gaps, fires


def report(gaps, fires, bounds, known) -> dict:
    vals = sorted(v for *_rest, v in gaps)
    out: dict = {"n_gaps": len(vals)}
    quant = {}
    for q in (0.5, 0.9, 0.95, 0.99, 1.0):
        quant[str(q)] = vals[min(len(vals) - 1, int(q * len(vals)))] if vals else None
    out["quantiles"] = quant
    p95 = quant["0.95"] or 0
    tail = collections.Counter((doc, page) for _s, doc, page, v in gaps if v > p95)
    out["tail_above_p95_pages"] = [
        {"doc": d, "page": p, "gaps_above_p95": n} for (d, p), n in tail.most_common()
    ]

    def reaches(bound: str, spec: str) -> bool:
        needle, _, page = spec.rpartition(":")
        return any(needle in doc and int(page) == pg for (doc, pg) in fires[bound])

    out["known_dropped_panel_pages_fire"] = {
        b: {spec: reaches(b, spec) for spec in known} for b in bounds
    }
    out["pages_firing_data_row_missing"] = {b: len(fires[b]) for b in bounds}
    base = bounds[0]
    out["pages_firing_only_with_larger_bound"] = {
        b: sorted([f"{d}:{p}" for (d, p) in fires[b] if (d, p) not in fires[base]])
        for b in bounds[1:]
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdf-dir", default="~/papers/pdf")
    parser.add_argument("--rotated-index")
    parser.add_argument("--census")
    parser.add_argument("--bounds", default=",".join(DEFAULT_BOUNDS))
    parser.add_argument("--known", action="append", default=[], help="DOC_SUBSTRING:PAGE")
    parser.add_argument("--out", help="write the full JSON report here")
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    bounds = args.bounds.split(",")
    gaps, fires = measure(_page_inputs(args), bounds, args.known)
    result = report(gaps, fires, bounds, args.known)
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).expanduser().write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
