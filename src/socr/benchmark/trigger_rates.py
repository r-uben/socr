"""P4-M: how many free-lane pages would each candidate equation trigger move?

No model calls. Runs the born-digital detector over a set of PDFs and, for every
page that takes the free native lane today (born-digital, native text, no OCR
enhancement, no table signal), records which equation-detector term fired and how
much text sits in math-font spans. The output is the table the P4 trigger is
chosen from (``docs/log/2026-09-02_p4-structure-lane-design.md``, section 7).

Content-free by construction: only counts and file basenames are printed, never
page text (the corpus is copyrighted; see the 2026-08-22 measurement convention).

Usage::

    PYTHONPATH=src ~/venvs/socr/bin/python -m socr.benchmark.trigger_rates <pdf>...
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from socr.core.born_digital import _MATH_FONT_RE, BornDigitalDetector

#: Buckets for the math-font character count, chosen to expose the shape of the
#: distribution, not to define a threshold. The threshold, if any, is picked from
#: the printed table with a documented reason.
_BUCKETS: tuple[tuple[int, int | None], ...] = (
    (0, 0),
    (1, 10),
    (11, 50),
    (51, 200),
    (201, None),
)


@dataclass
class PageRow:
    doc: str
    page: int
    free_lane: bool
    chart_asset_lane: bool
    font_term: bool
    regex_term: bool
    corrupt_term: bool
    hygiene_term: bool
    math_font_chars: int


@dataclass
class Tally:
    pages: int = 0
    born_digital: int = 0
    table_pages: int = 0
    free_lane: int = 0
    chart_asset: int = 0
    rows: list[PageRow] = field(default_factory=list)


def _math_font_chars(page: fitz.Page) -> int:
    """Characters rendered in a math-font span on the page."""
    n = 0
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if _MATH_FONT_RE.search(span.get("font", "") or ""):
                    n += len(span.get("text", ""))
    return n


def measure(pdfs: list[Path]) -> Tally:
    det = BornDigitalDetector()
    tally = Tally()
    for pdf in pdfs:
        try:
            assessment = det.detect(pdf)
        except Exception as exc:  # noqa: BLE001 - a corrupt PDF must not end the run
            print(f"skip {pdf.name}: {exc!r}", file=sys.stderr)
            continue
        with fitz.open(str(pdf)) as doc:
            for pa in assessment.pages:
                tally.pages += 1
                if not pa.is_born_digital:
                    continue
                tally.born_digital += 1
                if pa.has_tables:
                    tally.table_pages += 1
                # Mirrors ``_is_trusted_native_without_ocr`` with native_first on.
                free = bool(pa.native_text) and not pa.needs_ocr_enhancement and not pa.has_tables
                if not free:
                    continue
                tally.free_lane += 1
                # A raster on a free-lane page routes to the chart-asset lane
                # (still no model): counted, not excluded.
                chart = pa.has_figures
                if chart:
                    tally.chart_asset += 1
                page = doc[pa.page_num - 1]
                raw_text = page.get_text("text")
                tally.rows.append(
                    PageRow(
                        doc=pdf.name,
                        page=pa.page_num,
                        free_lane=True,
                        chart_asset_lane=chart,
                        font_term=BornDigitalDetector._detect_math_fonts(page),
                        regex_term=det._detect_equations(raw_text),
                        corrupt_term=pa.has_corrupt_math,
                        hygiene_term=(
                            pa.has_unmapped_math_glyphs or pa.has_unrecovered_symbol_glyphs
                        ),
                        math_font_chars=_math_font_chars(page),
                    )
                )
    return tally


def _pct(n: int, d: int) -> str:
    return f"{n} ({100.0 * n / d:.1f}%)" if d else f"{n} (n/a)"


def report(tally: Tally) -> str:
    rows = tally.rows
    free = len(rows)
    lines = [
        "# P4-M trigger rates (no model calls)",
        "",
        f"- pages: {tally.pages}; born-digital: {tally.born_digital}; "
        f"table pages (already routed): {tally.table_pages}",
        f"- free-lane pages today: {free} of {tally.pages} "
        f"({100.0 * free / tally.pages:.1f}% of all pages)"
        if tally.pages
        else "- no pages",
        f"- of which chart-asset lane (raster present, still no model): {tally.chart_asset}",
        "",
        "## Free-lane pages each candidate trigger would move to the ladder",
        "",
        "| trigger | pages moved | share of free lane |",
        "|---|---|---|",
    ]
    triggers = {
        "has_equations as detected (font OR regex OR corrupt)": lambda r: (
            r.font_term or r.regex_term or r.corrupt_term
        ),
        "font term only": lambda r: r.font_term,
        "regex term only": lambda r: r.regex_term,
        "corrupt-math term only": lambda r: r.corrupt_term,
        "hygiene flags only (unmapped / unrecovered glyphs)": lambda r: r.hygiene_term,
        "regex OR corrupt OR hygiene (no font term)": lambda r: (
            r.regex_term or r.corrupt_term or r.hygiene_term
        ),
    }
    for name, pred in triggers.items():
        n = sum(1 for r in rows if pred(r))
        lines.append(f"| {name} | {n} | {_pct(n, free)} |")
    lines += [
        "",
        "## Math-font characters per free-lane page (distribution, not a threshold)",
        "",
        "| bucket | pages | share of free lane |",
        "|---|---|---|",
    ]
    for lo, hi in _BUCKETS:
        n = sum(
            1 for r in rows if r.math_font_chars >= lo and (hi is None or r.math_font_chars <= hi)
        )
        label = f"{lo}" if hi == lo else (f"{lo}-{hi}" if hi is not None else f">{lo - 1}")
        lines.append(f"| {label} | {n} | {_pct(n, free)} |")
    lines += ["", "## Per-document free-lane pages moved by `has_equations` as detected", ""]
    per_doc: Counter[str] = Counter()
    per_doc_free: Counter[str] = Counter()
    for r in rows:
        per_doc_free[r.doc] += 1
        if r.font_term or r.regex_term or r.corrupt_term:
            per_doc[r.doc] += 1
    lines += ["| document | free-lane pages | moved |", "|---|---|---|"]
    for doc in sorted(per_doc_free):
        lines.append(f"| {doc} | {per_doc_free[doc]} | {per_doc[doc]} |")
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
