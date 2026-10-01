"""GH-916: evidence for ``ship_gate._PANEL_GAP_ROWS``.

The native-first ship gate extends a table's vertical span past its first/last
paired row through full-width rows no further than ``_PANEL_GAP_ROWS`` row pitches
away (``ship_gate.extended_span``). This tool measures, on a corpus of native-first
SHIP pages, what that bound should be:

(a) the distribution of gaps (in the block's own row pitches) between consecutive
    core paired rows, and which pages sit in its tail;
(b) for each candidate bound, whether each SPECIFIC omitted source row of the known
    dropped-row pages lies inside the covered span (``--known DOC:PAGE:Y,Y``, the y
    keys of the rows the grid omits);
(c) for each candidate bound, every source row newly covered beyond the bound-0
    baseline (page, y, row index), on every page including those that already fire,
    and the increment over the previous bound; rows not in ``--known`` are the
    candidates for false extension: inspect them against the page image.
Bound 0 is a true no-extension baseline (the core span only).

Inputs are the two page sets the gate was measured on: a rotated-pages index
(``[{"doc","page"}]``, upright re-read via ``attempt_rotated_native_table``) and a
census of upright native-first SHIP pages (one JSON object per line with ``doc``,
``page``, ``upright``). Counts and basenames only; no page text is written.

    uv run socr-measure-ship-gate-gaps --pdf-dir ~/papers/pdf \
        --rotated-index gh902/q917/index.json --census upright-census/census.jsonl \
        --known 2017__fama__ap.pdf:398:310,321,343,354
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


def _included_rows(g, blocks, pairs, src, bound):
    """Source rows (y keys) that a candidate-row check would cover at *bound*.

    A row is covered when it lies in the extended span of some block, is not itself a
    paired row, and occupies >= 2 of that block's lanes (what ``data_row_missing``
    treats as a candidate). Bound ``"0"`` is the unextended baseline: the core span.
    """
    anchor_ys = {y for found in pairs for _i, y in found}
    covered: set[int] = set()
    for lanes, _core, y_lo, y_hi in g.table_spans(blocks, pairs, src, _bound(bound)):
        for y, words in src.items():
            if y in anchor_ys or not (y_lo <= y <= y_hi):
                continue
            hit = {g._lane_of(w[0], lanes) for w in g._numeric_words(words)} - {None}
            if len(hit) >= 2:
                covered.add(y)
    return covered


def measure(inputs, bounds):
    from socr.tables import ship_gate as g

    gaps: list[tuple[str, str, int, float]] = []
    covered: dict[str, dict[tuple[str, int], list[int]]] = {b: {} for b in bounds}
    fired: dict[str, dict[tuple[str, int], list[int]]] = {b: {} for b in bounds}
    order: dict[tuple[str, int], list[int]] = {}
    inputs = list(inputs)
    # Results are keyed by (doc, page). A page present in both the rotated and the
    # upright set would silently overwrite one measurement, so refuse it (cubic P2, #920).
    seen: dict[tuple[str, int], str] = {}
    dupes = []
    for set_name, doc, page, _words, _md in inputs:
        if (doc, page) in seen:
            dupes.append(f"{doc}:{page} ({seen[(doc, page)]} and {set_name})")
        seen.setdefault((doc, page), set_name)
    if dupes:
        raise ValueError("pages appear in more than one input set: " + "; ".join(dupes))
    for set_name, doc, page, words, markdown in inputs:
        blocks = g._output_blocks(markdown)
        if not blocks or not words:
            continue
        src = g._source_rows(words)
        order[(doc, page)] = sorted(src)
        pairs = g._unique_pairs(blocks, src)
        for found in pairs:
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
            rows = _included_rows(g, blocks, pairs, src, b)
            if rows:
                covered[b][(doc, page)] = sorted(rows)
            faults = g.data_row_missing_faults(blocks, pairs, src, _bound(b))
            ys = [int(f["detail"].split("y=")[1].split()[0]) for f in faults]
            if ys:
                fired[b][(doc, page)] = sorted(ys)
    return gaps, covered, fired, order


def _parse_known(specs):
    """``DOC_SUBSTRING:PAGE:Y,Y`` -> ``[(substring, page, [y, ...])]``."""
    out = []
    for spec in specs:
        needle, page, ys = spec.rsplit(":", 2)
        out.append((needle, int(page), [int(y) for y in ys.split(",") if y]))
    return out


def report(gaps, covered, fired, order, bounds, known) -> dict:
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
    parsed = _parse_known(known)

    def is_known(doc: str, page: int, y: int) -> bool:
        return any(n in doc and pg == page and y in ys for n, pg, ys in parsed)

    # (b) ROW level: is each specific omitted source row inside a covered span?
    out["known_omitted_rows_reached"] = {
        b: {
            f"{n}:{pg}:{y}": any(
                n in doc and pg == page and y in rows for (doc, page), rows in covered[b].items()
            )
            for n, pg, ys in parsed
            for y in ys
        }
        for b in bounds
    }
    # (c) every row covered at a bound but not at the no-extension baseline, on ANY
    # page (including pages that already fire at the baseline), and the increment
    # over the previous bound. A row is "known" only if listed in --known.
    base = "0"
    if base != bounds[0]:
        raise ValueError("the first bound must be the zero baseline")
    newly: dict[str, dict] = {}
    previous = covered[base]
    for b in bounds[1:]:
        rows = []
        increment = []
        for key, ys in covered[b].items():
            base_rows = set(covered[base].get(key, []))
            prev_rows = set(previous.get(key, []))
            for y in ys:
                tag = f"{key[0]}:{key[1]}:y={y}:row#{order[key].index(y)}"
                if y not in base_rows:
                    rows.append((tag, is_known(key[0], key[1], y)))
                    if y not in prev_rows:
                        increment.append((tag, is_known(key[0], key[1], y)))
        newly[b] = {
            "newly_covered_vs_baseline": len(rows),
            "of_which_known_omitted": sum(1 for _t, k in rows if k),
            "not_known_omitted": [t for t, k in rows if not k],
            "increment_over_previous_bound": [t for t, _k in increment],
        }
        previous = covered[b]
    out["extension_by_bound"] = newly
    # Rows that FIRE (are reported missing) at a bound but not at the baseline: the
    # ones that matter. A fired row not in --known is a candidate false extension.
    firing: dict[str, dict] = {}
    for b in bounds[1:]:
        rows = []
        for key, ys in fired[b].items():
            base_rows = set(fired[base].get(key, []))
            rows += [
                (f"{key[0]}:{key[1]}:y={y}:row#{order[key].index(y)}", is_known(key[0], key[1], y))
                for y in ys
                if y not in base_rows
            ]
        firing[b] = {
            "newly_firing": len(rows),
            "of_which_known_omitted": sum(1 for _t, k in rows if k),
            "not_known_omitted": [t for t, k in rows if not k],
        }
    out["newly_firing_by_bound"] = firing
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdf-dir", default="~/papers/pdf")
    parser.add_argument("--rotated-index")
    parser.add_argument("--census")
    parser.add_argument("--bounds", default=",".join(DEFAULT_BOUNDS))
    parser.add_argument(
        "--known",
        action="append",
        default=[],
        help="DOC_SUBSTRING:PAGE:Y,Y (source row y keys omitted from the grid)",
    )
    parser.add_argument("--out", help="write the full JSON report here")
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    bounds = args.bounds.split(",")
    # The zero baseline is always computed, first, whatever the caller asked for.
    bounds = ["0"] + [b for b in bounds if b != "0"]
    gaps, covered, fired, order = measure(_page_inputs(args), bounds)
    result = report(gaps, covered, fired, order, bounds, args.known)
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).expanduser().write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
