#!/usr/bin/env python3
"""Offline native-cell check over stored model markdown + source PDFs.

Shepherd can run this on the 585-page overnight census (or any directory layout):

  ~/venvs/socr/bin/python tools/native_cell_census.py \\
    --pages-root /path/to/paged \\
    --doc-sha 4169f9722680aac1

Each page directory is expected to contain ``socr_input.pdf`` (single-page PDF) and
``page.md`` (model markdown). Prints per-page table outcomes and aggregate counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socr.core.born_digital import flatten_page_spans  # noqa: E402
from socr.tables.locate import locate_tables  # noqa: E402
from socr.tables.native_cell_check import check_table_cell_binding  # noqa: E402
from socr.tables.native_contradiction import pair_regions  # noqa: E402
from socr.tables.reconcile import find_table_blocks  # noqa: E402


def _scan_page(page_dir: Path) -> list[dict]:
    pdf = page_dir / "socr_input.pdf"
    md_path = page_dir / "page.md"
    if not pdf.is_file() or not md_path.is_file():
        return []
    text = md_path.read_text(encoding="utf-8")
    blocks = find_table_blocks(text)
    if not blocks:
        return [{"table_id": "none", "outcome": "skip", "detail": "no_table_blocks"}]
    doc = fitz.open(pdf)
    page = doc[0]
    words = page.get_text("words")
    spans = flatten_page_spans(page)
    located = locate_tables(page)
    raw = [tuple(b.bbox) for b in located]
    mds = [b.markdown for b in blocks]
    paired = pair_regions(page, mds, raw)
    out: list[dict] = []
    for i, block in enumerate(blocks):
        pick = paired[i] if i < len(paired) else None
        if pick is None:
            out.append(
                {
                    "table_id": block.table_id or f"t{i}",
                    "outcome": "ambiguous",
                    "detail": "pairing inconclusive",
                }
            )
            continue
        peers = [tuple(p) for j, p in enumerate(paired) if j != i and p is not None]
        res = check_table_cell_binding(
            table_id=block.table_id or f"t{i}",
            words=words,
            markdown=block.markdown,
            region_bbox=pick,
            spans=spans,
            peer_region_bboxes=peers or None,
            locator_boxes=raw,
            page=page,
        )
        matched = len(res.binding.matched_cells) if res.binding else 0
        out.append(
            {
                "table_id": block.table_id or f"t{i}",
                "outcome": res.outcome.value,
                "matched": matched,
                "detail": res.detail,
            }
        )
    doc.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-root", type=Path, required=True)
    parser.add_argument(
        "--doc-sha", type=str, default=None, help="Optional subdirectory under pages-root"
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    root = args.pages_root
    if args.doc_sha:
        root = root / args.doc_sha
    page_dirs = sorted(p for p in root.glob("p*") if p.is_dir())
    if not page_dirs:
        print(f"No page directories under {root}", file=sys.stderr)
        return 1

    agg: Counter[str] = Counter()
    report: list[dict] = []
    for pd in page_dirs:
        tables = _scan_page(pd)
        for t in tables:
            agg[t["outcome"]] += 1
        report.append({"page": pd.name, "tables": tables})

    print(f"Pages scanned: {len(page_dirs)}")
    print("Aggregate outcomes:", dict(agg))
    for entry in report:
        flags = [f"{t['table_id']}:{t['outcome']}" for t in entry["tables"]]
        if any(o != "verified" and o != "skip" for o in (t["outcome"] for t in entry["tables"])):
            print(f"  {entry['page']}: {', '.join(flags)}")

    if args.json_out:
        args.json_out.write_text(json.dumps({"pages": report, "aggregate": dict(agg)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
