#!/usr/bin/env python3
"""Offline native-cell check over stored model markdown + source PDFs.

Shepherd can run this on the 585-page overnight census (or any directory layout):

  ~/venvs/socr/bin/python tools/native_cell_census.py \\
    --pages-root /path/to/paged \\
    --doc-sha 4169f9722680aac1

Each page directory is expected to contain ``socr_input.pdf`` (single-page PDF) and
``page.md`` (model markdown). Emits per-page table outcomes with ``paired``, ``matched``,
and a short ``reason`` suitable for aggregation.
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
from socr.tables.native_cell_check import (  # noqa: E402
    NativeCellOutcome,
    check_table_cell_binding,
)
from socr.tables.native_contradiction import pair_regions  # noqa: E402
from socr.tables.reconcile import find_table_blocks  # noqa: E402


def _reason_from_result(
    *,
    outcome: str,
    paired: bool,
    detail: str,
    matched: int,
    binding,
) -> str:
    if not paired:
        return "pairing_inconclusive"
    if outcome == NativeCellOutcome.VERIFIED.value:
        return "verified"
    if outcome == NativeCellOutcome.CONTRADICTED.value:
        if binding and binding.contradicted_cells:
            return "contradicted_cells"
        return "row_anchor_mismatch"
    if "latex_tabular_unparsed" in detail:
        return "latex_unparsed"
    if "bind failed" in detail:
        return "bind_failed"
    if binding is None:
        return detail or "no_binding"
    if binding.column_binding_unverifiable and matched == 0:
        return "column_lanes_unverifiable"
    if binding.model_unbound:
        return f"model_unbound={len(binding.model_unbound)}"
    if binding.native_unbound:
        return f"native_unbound={len(binding.native_unbound)}"
    if binding.row_binding_unverifiable:
        return "row_binding_unverifiable"
    if matched == 0:
        return "zero_matched"
    return "fallback_not_met"


def _scan_page(page_dir: Path) -> list[dict]:
    pdf = page_dir / "socr_input.pdf"
    md_path = page_dir / "page.md"
    if not pdf.is_file() or not md_path.is_file():
        return []
    text = md_path.read_text(encoding="utf-8")
    blocks = find_table_blocks(text)
    if not blocks:
        return [
            {
                "table_id": "none",
                "outcome": "skip",
                "paired": False,
                "matched": 0,
                "reason": "no_table_blocks",
                "detail": "no_table_blocks",
            }
        ]
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
        is_paired = pick is not None
        if not is_paired:
            out.append(
                {
                    "table_id": block.table_id or f"t{i}",
                    "outcome": NativeCellOutcome.AMBIGUOUS.value,
                    "paired": False,
                    "matched": 0,
                    "reason": "pairing_inconclusive",
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
        reason = _reason_from_result(
            outcome=res.outcome.value,
            paired=True,
            detail=res.detail,
            matched=matched,
            binding=res.binding,
        )
        out.append(
            {
                "table_id": block.table_id or f"t{i}",
                "outcome": res.outcome.value,
                "paired": True,
                "matched": matched,
                "reason": reason,
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
    parser.add_argument(
        "--page-glob",
        type=str,
        default="p*",
        help="Glob under pages-root for page directories (default: p*)",
    )
    args = parser.parse_args()

    root = args.pages_root
    if args.doc_sha:
        root = root / args.doc_sha
    page_dirs = sorted(p for p in root.glob(args.page_glob) if p.is_dir())
    if not page_dirs:
        print(f"No page directories under {root}", file=sys.stderr)
        return 1

    agg: Counter[str] = Counter()
    report: list[dict] = []
    for pd in page_dirs:
        tables = _scan_page(pd)
        page_summary = {
            "page": pd.name,
            "tables": tables,
            "outcome": "skip",
            "matched": 0,
            "paired": bool(tables),
            "reason": "no_tables",
        }
        if tables:
            non_skip = [t for t in tables if t["outcome"] != "skip"]
            if non_skip:
                worst = max(
                    non_skip,
                    key=lambda t: (
                        2
                        if t["outcome"] == NativeCellOutcome.CONTRADICTED.value
                        else 1
                        if t["outcome"] == NativeCellOutcome.AMBIGUOUS.value
                        else 0
                    ),
                )
                page_summary.update(
                    {
                        "outcome": worst["outcome"],
                        "matched": sum(t.get("matched", 0) for t in non_skip),
                        "paired": all(t.get("paired", False) for t in non_skip),
                        "reason": worst.get("reason", ""),
                    }
                )
        for t in tables:
            agg[t["outcome"]] += 1
        report.append(page_summary)

    print(f"Pages scanned: {len(page_dirs)}")
    print("Aggregate table outcomes:", dict(agg))
    for entry in report:
        if entry["outcome"] not in ("verified", "skip"):
            flags = [
                f"{t['table_id']}:{t['outcome']}(m={t.get('matched',0)})"
                for t in entry["tables"]
            ]
            print(
                f"  {entry['page']}: outcome={entry['outcome']} "
                f"matched={entry['matched']} reason={entry['reason']} :: {', '.join(flags)}"
            )

    payload = {"pages": report, "aggregate": dict(agg)}
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
