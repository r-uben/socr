#!/usr/bin/env python3
"""Evaluate native cell binding on golden table fixtures (+ mutants).

Usage:
  ~/venvs/socr/bin/python tools/native_cell_golden_eval.py
  ~/venvs/socr/bin/python tools/native_cell_golden_eval.py --compare-ref origin/main
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "resf_native_cell"

sys.path.insert(0, str(ROOT / "src"))

from socr.core.born_digital import flatten_page_spans  # noqa: E402
from socr.tables.locate import locate_tables  # noqa: E402
from socr.tables.native_cell_check import (  # noqa: E402
    NativeCellOutcome,
    check_table_cell_binding,
)
from socr.tables.native_contradiction import pair_regions  # noqa: E402


@dataclass(frozen=True)
class GoldenCase:
    tag: str
    pdf: Path
    md: Path
    page_index: int = 0
    peer_md: Path | None = None
    peer_table_index: int = 0
    """When set, use this ``locate_tables`` index instead of ``pair_regions`` alone."""
    locator_table_index: int | None = None


GOLDENS: list[GoldenCase] = [
    GoldenCase("RESF p59 T1", FIX / "resf_p59.pdf", FIX / "p59_tableau1.md"),
    GoldenCase(
        "RESF p59 T2",
        FIX / "resf_p59.pdf",
        FIX / "p59_tableau2.md",
        peer_md=FIX / "p59_tableau1.md",
        peer_table_index=0,
    ),
    GoldenCase("RESF p67 T8", FIX / "resf_p67.pdf", FIX / "p67_tableau8.md"),
    GoldenCase("PLFSS p38 T18", FIX / "plfss_p38.pdf", FIX / "p38_tableau18.md"),
    GoldenCase("PLFSS p44 full", FIX / "plfss_p44.pdf", FIX / "p44_native_draft.md"),
    GoldenCase("PLFSS p65 full", FIX / "plfss_p65.pdf", FIX / "p65_full.md"),
    GoldenCase(
        "RESF p60 T2bis",
        FIX / "resf_p60.pdf",
        FIX / "p60_tableau2bis.md",
        locator_table_index=1,
    ),
    GoldenCase(
        "RESF p60 T3",
        FIX / "resf_p60.pdf",
        FIX / "p60_tableau3.md",
        locator_table_index=2,
    ),
    GoldenCase("RESF p63 T7", FIX / "resf_p63.pdf", FIX / "p63_tableau7.md"),
    GoldenCase(
        "RESF p42 T1",
        FIX / "resf_p42.pdf",
        FIX / "p42_tableau1_native.md",
        locator_table_index=0,
    ),
    GoldenCase(
        "RESF p42 T2",
        FIX / "resf_p42.pdf",
        FIX / "p42_tableau2_native.md",
        locator_table_index=1,
    ),
    GoldenCase("RESF p61 T1", FIX / "resf_p61.pdf", FIX / "p61_tableau_native.md"),
    GoldenCase(
        "PLFSS p39 T2",
        FIX / "plfss_p39.pdf",
        FIX / "p39_tableau2_native.md",
        locator_table_index=1,
    ),
    GoldenCase("PLFSS p40 T1", FIX / "plfss_p40.pdf", FIX / "p40_tableau_native.md"),
    GoldenCase("PLFSS p41 T1", FIX / "plfss_p41.pdf", FIX / "p41_tableau_native.md"),
    GoldenCase("PLFSS p42 T1", FIX / "plfss_p42.pdf", FIX / "p42_plfss_native.md"),
]


def _parse_md_rows(md: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln for ln in md.splitlines() if ln.startswith("|") and "---" not in ln]
    if not lines:
        return [], []
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    rows: list[list[str]] = []
    for ln in lines[2:]:
        rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return header, rows


def _to_md(header: list[str], rows: list[list[str]]) -> str:
    out = "| " + " | ".join(header) + " |\n| " + " | ".join("---" for _ in header) + " |\n"
    for r in rows:
        out += "| " + " | ".join(r) + " |\n"
    return out


def _mutants(header: list[str], rows: list[list[str]]) -> dict[str, str]:
    variants: dict[str, str] = {}
    if not rows:
        return variants
    r0 = copy.deepcopy(rows)
    if len(r0[0]) > 1 and r0[0][1].strip() not in ("", "-"):
        r0[0][1] = "9" + r0[0][1].lstrip("-+")
        variants["mut_number"] = _to_md(header, r0)
    r1 = copy.deepcopy(rows)
    for ri, row in enumerate(r1):
        for ci, cell in enumerate(row[1:], start=1):
            if cell.strip().startswith("-"):
                row[ci] = cell.replace("-", "", 1)
                variants["mut_sign"] = _to_md(header, r1)
                break
        if "mut_sign" in variants:
            break
    if len(rows) > 1:
        r2 = copy.deepcopy(rows)
        r2[0][0], r2[1][0] = rows[1][0], rows[0][0]
        variants["mut_swap_labels"] = _to_md(header, r2)
    r3 = copy.deepcopy(rows)
    if len(r3) > 1:
        del r3[0]
        variants["mut_drop_row"] = _to_md(header, r3)
    if header and len(header) > 2:
        r4 = copy.deepcopy(rows)
        h4 = header[:]
        h4[1], h4[2] = header[2], header[1]
        for row in r4:
            if len(row) > 2:
                row[1], row[2] = row[2], row[1]
        variants["mut_swap_years"] = _to_md(h4, r4)
    if header:
        h5 = header + ["extra"]
        r5 = [row + ["1"] for row in rows]
        variants["mut_add_col"] = _to_md(h5, r5)
    r6 = copy.deepcopy(rows)
    r6.append([""] + ["0"] * (len(header) - 1))
    variants["mut_add_row"] = _to_md(header, r6)
    return variants


def _resolve_region(case: GoldenCase, page, md: str, raw: list[tuple]) -> tuple:
    if case.locator_table_index is not None and case.locator_table_index < len(raw):
        return raw[case.locator_table_index]
    mds = [md]
    if case.peer_md is not None:
        mds.insert(case.peer_table_index, case.peer_md.read_text(encoding="utf-8"))
    pr = pair_regions(page, mds, raw)
    if case.peer_md is None:
        return pr[0]
    if "tableau2" in case.md.name:
        return pr[-1]
    return pr[1] if len(pr) > 1 else pr[0]


def _run_case(case: GoldenCase, md: str) -> dict:
    doc = fitz.open(case.pdf)
    page = doc[case.page_index]
    raw = [tuple(b.bbox) for b in locate_tables(page)]
    pick = _resolve_region(case, page, md, raw)
    peers = None
    if case.peer_md is not None:
        mds = [case.peer_md.read_text(encoding="utf-8"), md]
        pr = pair_regions(page, mds, raw)
        peers = [pr[0]] if pr[0] is not None else None
    res = check_table_cell_binding(
        table_id=case.tag,
        words=page.get_text("words"),
        markdown=md,
        region_bbox=pick,
        spans=flatten_page_spans(page),
        peer_region_bboxes=peers,
        locator_boxes=raw,
        page=page,
    )
    doc.close()
    matched = len(res.binding.matched_cells) if res.binding else 0
    return {
        "outcome": res.outcome.value,
        "matched": matched,
        "detail": res.detail,
    }


def _eval_all() -> tuple[list[dict], dict[str, int]]:
    rows_out: list[dict] = []
    counts: dict[str, int] = {}
    for case in GOLDENS:
        md = case.md.read_text(encoding="utf-8")
        header, body = _parse_md_rows(md)
        correct = _run_case(case, md)
        counts[correct["outcome"]] = counts.get(correct["outcome"], 0) + 1
        muts = _mutants(header, body)
        mut_results = {name: _run_case(case, m) for name, m in muts.items()}
        rows_out.append(
            {
                "page": case.tag,
                "correct": correct,
                "mutants": mut_results,
            }
        )
    return rows_out, counts


def _run_eval_subprocess() -> tuple[list[dict], dict[str, int]]:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "native_cell_golden_eval.py"), "--json"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    start = proc.stdout.find("{")
    data = json.loads(proc.stdout[start:])
    return data["goldens"], data["aggregate_correct"]


def _eval_at_git_ref(ref: str) -> tuple[list[dict], dict[str, int]]:
    """Check out table modules at *ref* in a subprocess, then restore."""
    rel_paths = [
        ROOT / "src/socr/tables/native_cell_check.py",
        ROOT / "src/socr/tables/binding.py",
    ]
    backups = {p: p.read_bytes() for p in rel_paths}
    try:
        for p in rel_paths:
            rel = p.relative_to(ROOT).as_posix()
            p.write_bytes(
                subprocess.run(
                    ["git", "show", f"{ref}:{rel}"],
                    check=True,
                    capture_output=True,
                    cwd=ROOT,
                ).stdout
            )
        return _run_eval_subprocess()
    finally:
        for p, data in backups.items():
            p.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-ref",
        metavar="GIT_REF",
        default=None,
        help="Compare baseline results at GIT_REF (e.g. origin/main)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.compare_ref:
        before_rows, before_counts = _eval_at_git_ref(args.compare_ref)
        # Restore current modules
        for mod in ("socr.tables.binding", "socr.tables.native_cell_check"):
            if mod in sys.modules:
                del sys.modules[mod]
        sys.path[:] = [p for p in sys.path if "golden_ref_" not in p]
        after_rows, after_counts = _eval_all()
        if args.json:
            print(
                json.dumps(
                    {
                        "ref": args.compare_ref,
                        "before": {"goldens": before_rows, "aggregate": before_counts},
                        "after": {"goldens": after_rows, "aggregate": after_counts},
                    },
                    indent=2,
                )
            )
            return 0
        print(f"Compare ref {args.compare_ref} -> working tree")
        print(f"{'Page':<22} {'Before':<22} {'After':<22}")
        print("-" * 70)
        for b, a in zip(before_rows, after_rows, strict=True):
            bc = b["correct"]
            ac = a["correct"]
            mark = " *" if bc != ac else ""
            print(
                f"{b['page']:<22} {bc['outcome']}/{bc['matched']:<16} "
                f"{ac['outcome']}/{ac['matched']:<16}{mark}"
            )
        print("-" * 70)
        print("Before:", before_counts, "After:", after_counts)
        return 0

    rows_out, counts = _eval_all()

    if args.json:
        print(json.dumps({"goldens": rows_out, "aggregate_correct": counts}, indent=2))
        return 0

    print(f"{'Page':<22} {'Correct':<12} {'Matched':<8} Mutants (must not VERIFY)")
    print("-" * 90)
    for row in rows_out:
        c = row["correct"]
        bad = [
            f"{k}:{v['outcome'][0].upper()}"
            for k, v in row["mutants"].items()
            if v["outcome"] == "verified"
        ]
        mut_s = ", ".join(f"{k}:{v['outcome'][:1].upper()}" for k, v in row["mutants"].items())
        print(f"{row['page']:<22} {c['outcome']:<12} {c['matched']:<8} {mut_s}")
        if bad:
            print(f"  *** MUTANT VERIFY: {bad}")
    print("-" * 90)
    print(
        "Aggregate on correct goldens:",
        f"V={counts.get('verified', 0)}",
        f"A={counts.get('ambiguous', 0)}",
        f"C={counts.get('contradicted', 0)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
