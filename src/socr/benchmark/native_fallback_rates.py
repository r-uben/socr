"""P6 stage-D input: how often would each NATIVE_FALLBACK trigger fire, by trigger?

No model calls. Runs the born-digital detector over a set of PDFs and, for every
born-digital page with native text, records which of the analysis-time
demotion triggers behind the ``NATIVE_FALLBACK`` ending fired
(``src/socr/core/manifest.py``, the ``native_is_fallback`` / ``native_demoted``
conjunction). This is the per-trigger enumeration the P6 panel required before
the fourth ending can be assigned trigger-by-trigger to native prose or the
floor (``docs/log/2026-09-02_p6-selector-collapse-design.md`` §8 Q1).

Two of the production triggers are not observable at analysis time and are
reported as such: ``chart_asset_render_failed`` (a runtime render failure) and
``native_table_structure_failed`` (set by the verifier on the routed path).
``p.attempts`` gating (the page reached the ladder) is likewise runtime; the
counts below are the upper bound of pages that CAN demote, not the shipped rate.

Content-free by construction: counts and basenames only.

Usage::

    PYTHONPATH=src ~/venvs/socr/bin/python -m socr.benchmark.native_fallback_rates <pdf>...
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from socr.core.born_digital import BornDigitalDetector
from socr.core.pdf import open_pdf


@dataclass
class Row:
    doc: str
    page: int
    needs_ocr_enhancement: bool
    corrupt_math: bool
    unverifiable_table_region: bool
    table_structure_defective: bool
    table_header_unattributed: bool
    text_grid_rejected: bool
    has_tables: bool


@dataclass
class Tally:
    pages: int = 0
    born_digital_with_text: int = 0
    rows: list[Row] = field(default_factory=list)


def measure(pdfs: list[Path]) -> Tally:
    det = BornDigitalDetector()
    tally = Tally()
    for pdf in pdfs:
        try:
            with open_pdf(pdf) as _doc:  # existence / readability, via the shared seam
                pass
            assessment = det.detect(pdf)
        except Exception as exc:  # noqa: BLE001 - one corrupt PDF must not end the run
            print(f"skip {pdf.name}: {exc!r}", file=sys.stderr)
            continue
        for pa in assessment.pages:
            tally.pages += 1
            if not (pa.is_born_digital and pa.native_text):
                continue
            tally.born_digital_with_text += 1
            tally.rows.append(
                Row(
                    doc=pdf.name,
                    page=pa.page_num,
                    needs_ocr_enhancement=pa.needs_ocr_enhancement,
                    corrupt_math=pa.has_corrupt_math,
                    unverifiable_table_region=pa.has_unverifiable_table_region,
                    table_structure_defective=pa.native_table_structure_defective,
                    table_header_unattributed=pa.native_table_header_unattributed,
                    text_grid_rejected=bool(pa.text_grid_rejections),
                    has_tables=pa.has_tables,
                )
            )
    return tally


def _pct(n: int, d: int) -> str:
    return f"{n} ({100.0 * n / d:.1f}%)" if d else f"{n} (n/a)"


def report(tally: Tally) -> str:
    rows = tally.rows
    base = len(rows)
    triggers = {
        "needs_ocr_enhancement (of which corrupt math)": lambda r: r.needs_ocr_enhancement,
        "  corrupt math alone": lambda r: r.corrupt_math,
        "unverifiable table region (TR-3 geometry hard-fail)": lambda r: (
            r.unverifiable_table_region
        ),
        "native table structure defective (GH-151 B1)": lambda r: r.table_structure_defective,
        "native table header unattributed": lambda r: r.table_header_unattributed,
        "text-strategy grid rejected (GH-195)": lambda r: r.text_grid_rejected,
        "ANY analysis-time trigger": lambda r: (
            r.needs_ocr_enhancement
            or r.unverifiable_table_region
            or r.table_structure_defective
            or r.table_header_unattributed
            or r.text_grid_rejected
        ),
        "any trigger on a page WITHOUT a table signal": lambda r: (
            not r.has_tables
            and (
                r.needs_ocr_enhancement
                or r.unverifiable_table_region
                or r.table_structure_defective
                or r.table_header_unattributed
                or r.text_grid_rejected
            )
        ),
    }
    lines = [
        "# NATIVE_FALLBACK trigger rates (analysis-time, no model calls)",
        "",
        f"- pages: {tally.pages}; born-digital with native text: {base}",
        "- not observable here (runtime): chart_asset_render_failed,"
        " native_table_structure_failed, the p.attempts gate",
        "",
        "| trigger | pages | share of born-digital pages |",
        "|---|---|---|",
    ]
    for name, pred in triggers.items():
        n = sum(1 for r in rows if pred(r))
        lines.append(f"| {name} | {n} | {_pct(n, base)} |")
    per_doc: Counter[str] = Counter()
    for r in rows:
        if triggers["ANY analysis-time trigger"](r):
            per_doc[r.doc] += 1
    lines += [
        "",
        f"- documents with at least one triggered page: {len(per_doc)}",
        f"- max triggered pages in one document: {max(per_doc.values()) if per_doc else 0}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    pdfs = sorted(Path(a) for a in args if a.lower().endswith(".pdf"))
    if not pdfs:
        print(__doc__, file=sys.stderr)
        return 2
    print(report(measure(pdfs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
