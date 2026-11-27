"""#1409 step 1: run the second-reader comparator over the 79 hand-checked pages.

Offline: the second reader's stored readings, the shipped page text, the page sidecars and
the input PDFs. No model, no network. Run from the repo root, on stdin:

    PYTHONPATH=src ~/venvs/socr/bin/python - [--no-labels] [--rows N] \\
        < docs/log/2026-11-26_1409_second_reader_eval.py

``--no-labels`` withholds the chart-box layer text (the fallback alone); ``--rows`` sets how
many differing rows are printed per page.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import fitz

from socr.tables.second_reader import compare_readings

WORK = Path("/Volumes/Main/Library/socr-work")
READINGS = WORK / "ocria-test/hc79"
SHIPPED = WORK / "fullrun/audit/handcheck150"
RUN = WORK / "fullrun/out"
PDFS = Path("/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs")


def chart_label_text(doc: str, page: int) -> tuple[str | None, int, int]:
    """``(layer text inside the page's recorded figure boxes, boxes, pictures shipped)``."""
    sidecar = json.loads((RUN / doc / doc / "pages" / f"{page:05d}.json").read_text())
    refs = sidecar.get("figure_refs") or []
    boxes = [ref["bbox"] for ref in refs if ref.get("bbox")]
    if not boxes:
        return None, 0, len(refs)
    # The recorded boxes are in the page's displayed (rotated) frame; words are not. A box is
    # tight around the drawing, so an axis label often straddles its edge: a word that
    # TOUCHES a box counts.
    with fitz.open(PDFS / f"{doc}.pdf") as pdf:
        sheet = pdf[page - 1]
        rects = [fitz.Rect(b) for b in boxes]
        inside = [
            str(w[4])
            for w in sheet.get_text("words")
            if any((fitz.Rect(w[:4]) * sheet.rotation_matrix).intersects(r) for r in rects)
        ]
    return " ".join(inside), len(boxes), len(refs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--rows", type=int, default=0)
    args = parser.parse_args()
    tally: Counter = Counter()
    for entry in json.loads((READINGS / "manifest.json").read_text()):
        page_id, doc, page = entry["id"], entry["doc"], entry["page"]
        shipped = (SHIPPED / f"{page_id}.md").read_text()
        reading = (READINGS / "out" / f"hc{page_id}.md").read_text()
        labels, boxes, refs = chart_label_text(doc, page)
        verdict = compare_readings(
            shipped, reading, chart_label_text=None if args.no_labels else labels
        )
        tally[verdict.outcome.value] += 1
        print(
            f"{page_id} {verdict.outcome.value:8} shipped_rows={verdict.shipped_rows} "
            f"reader_rows={verdict.reader_rows} differences={len(verdict.differences)} "
            f"excluded_reader={len(verdict.excluded_reader_rows)} "
            f"excluded_shipped={len(verdict.excluded_shipped_rows)} "
            f"picture={int(verdict.chart_fallback)} boxes={boxes}/{refs} "
            f"label_words={len((labels or '').split())}"
        )
        for d in verdict.differences[: args.rows]:
            print(
                f"    {d.side} {d.kind} shipped={d.shipped_count} reader={d.reader_count} "
                f"{' '.join(d.numbers)[:90]} | {d.text[:110]}"
            )
    print(dict(tally))


main()
