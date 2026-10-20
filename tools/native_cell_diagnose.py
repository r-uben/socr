#!/usr/bin/env python3
"""Diagnose why a golden table is not VERIFIED (matched cells, blockers, paths)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socr.core.born_digital import flatten_page_spans  # noqa: E402
from socr.tables.locate import locate_tables  # noqa: E402
from socr.tables.native_cell_check import (  # noqa: E402
    NativeCellOutcome,
    _fallback_verified_allowed,
    _header_year_mismatch,
    _model_covers_native_data_rows,
    _numeric_unbound_blocks_verify,
    _row_anchor_scan,
    _row_label_contradictions_are_soft,
    check_table_cell_binding,
    classify_binding_for_cell_check,
)
from socr.tables.native_contradiction import (  # noqa: E402
    extend_check_region_for_trailing_words,
    pair_regions,
    refine_check_region,
)


def diagnose(
    pdf: Path,
    md_path: Path,
    *,
    peer_md: Path | None = None,
) -> None:
    doc = fitz.open(pdf)
    page = doc[0]
    raw = [tuple(b.bbox) for b in locate_tables(page)]
    md = md_path.read_text(encoding="utf-8")
    mds = [md]
    if peer_md:
        mds.insert(0, peer_md.read_text(encoding="utf-8"))
    pr = pair_regions(page, mds, raw)
    if peer_md and "tableau2" in md_path.name:
        pick, peers = pr[-1], [pr[0]]
    elif peer_md:
        pick, peers = pr[1], [pr[0]]
    else:
        pick, peers = pr[0], None
    if pick and peers:
        region = extend_check_region_for_trailing_words(
            page, refine_check_region(pick, peers), md
        )
    else:
        region = pick
    res = check_table_cell_binding(
        table_id="diag",
        words=page.get_text("words"),
        markdown=md,
        region_bbox=pick,
        spans=flatten_page_spans(page),
        peer_region_bboxes=peers,
        locator_boxes=raw,
        page=page,
    )
    b = res.binding
    print(f"outcome={res.outcome.value} detail={res.detail}")
    if not b:
        doc.close()
        return
    print(
        f"structural={b.structural_agreement} fully_checked={b.fully_checked} "
        f"col_unv={b.column_binding_unverifiable} row_unv={b.row_binding_unverifiable} "
        f"matched={len(b.matched_cells)} ambiguous={b.ambiguous_count}"
    )
    print(f"header_year_mismatch={_header_year_mismatch(b, md)}")
    print(f"row_label_contr={len(b.row_label_contradictions)} soft={_row_label_contradictions_are_soft(b)}")
    print(f"contr_cells={len(b.contradicted_cells)} native_unbound={len(b.native_unbound)} model_unbound={len(b.model_unbound)}")
    mm, cmp, per_row = _row_anchor_scan(b, md)
    hard = [m for m in mm if m[2] not in ("ambiguous_label",) and m[1] != "ambiguous_locale"]
    print(f"row_scan comparisons={cmp} hard_mismatches={hard[:8]} ambiguous_locale={sum(1 for m in mm if m[1]=='ambiguous_locale')}")
    print(
        f"numeric_unbound_blocks={_numeric_unbound_blocks_verify(b, region)} "
        f"covers_native={_model_covers_native_data_rows(b, md)} "
        f"fallback_ok={_fallback_verified_allowed(b, md, region, comparisons_per_row=per_row)}"
    )
    if b.contradicted_cells[:3]:
        for c in b.contradicted_cells[:5]:
            print(f"  CONTR cell {c.row_path[-1][:40] if c.row_path else ''} native={c.native_token} model={c.model_token}")
    nu = [c for c in b.native_unbound if c.token and c.col_path]
    if nu[:5]:
        print("  native_unbound sample:", [(c.token, c.col_path) for c in nu[:5]])
    doc.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("markdown")
    p.add_argument("--peer-md", type=Path, default=None)
    args = p.parse_args()
    diagnose(Path(args.pdf), Path(args.markdown), peer_md=args.peer_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
