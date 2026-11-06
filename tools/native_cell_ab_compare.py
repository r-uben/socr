#!/usr/bin/env python3
"""Compare native cell outcomes with binding/check paths toggled.

Shows where coalescing, year-lane calibration, or LaTeX handling change results.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

import socr.tables.binding as binding_mod  # noqa: E402
import socr.tables.native_cell_check as ncc_mod  # noqa: E402
from socr.core.born_digital import flatten_page_spans  # noqa: E402
from socr.tables.locate import locate_tables  # noqa: E402
from socr.tables.native_cell_check import check_table_cell_binding  # noqa: E402
from socr.tables.native_contradiction import pair_regions  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from native_cell_golden_eval import GOLDEN_SETS, GoldenCase, _run_case  # noqa: E402


@dataclass(frozen=True)
class Toggle:
    name: str
    apply: Callable[[], Callable[[], None]]


def _patch(module, name: str, replacement):
    original = getattr(module, name)

    def enter():
        setattr(module, name, replacement)
        return lambda: setattr(module, name, original)

    return enter


def _run_with_toggles(
    case: GoldenCase, md: str, toggles: list[Callable[[], Callable[[], None]]]
) -> dict:
    restores: list[Callable[[], None]] = []
    for enter in toggles:
        restores.append(enter())
    try:
        return _run_case(case, md)
    finally:
        for restore in reversed(restores):
            restore()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        choices=sorted(GOLDEN_SETS),
        default="track2",
        help="Golden set (default track2 = comparable baseline pages)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    goldens = GOLDEN_SETS[args.set]

    toggles = [
        Toggle(
            "no_coalesce",
            _patch(binding_mod, "_coalesce_numeric_word_fragments", lambda w: w),
        ),
        Toggle(
            "no_year_lane",
            _patch(binding_mod, "_year_header_lane_centers", lambda w: None),
        ),
        Toggle(
            "no_latex_convert",
            _patch(
                ncc_mod,
                "_prepare_markdown_for_cell_check",
                lambda md: (md, None),
            ),
        ),
    ]

    rows: list[dict] = []
    for case in goldens:
        md = case.md.read_text(encoding="utf-8")
        baseline = _run_case(case, md)
        row = {"page": case.tag, "baseline": baseline, "toggles": {}}
        for t in toggles:
            row["toggles"][t.name] = _run_with_toggles(case, md, [t.apply])
        changed = [
            name
            for name, res in row["toggles"].items()
            if res["outcome"] != baseline["outcome"] or res["matched"] != baseline["matched"]
        ]
        row["changed_by"] = changed
        rows.append(row)

    if args.json:
        import json

        print(json.dumps(rows, indent=2))
        return 0

    print(f"{'Page':<24} {'Base':<12} {'Mtch':<6} Changed by toggle(s)")
    print("-" * 88)
    for row in rows:
        b = row["baseline"]
        ch = ", ".join(row["changed_by"]) if row["changed_by"] else "-"
        print(f"{row['page']:<24} {b['outcome']:<12} {b['matched']:<6} {ch}")
        for name in row["changed_by"]:
            t = row["toggles"][name]
            print(f"  {name}: outcome={t['outcome']} matched={t['matched']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
