#!/usr/bin/env python3
"""How often does a LARGE raster put a born-digital page in the chart lane?

GH-511. #510 closed the small half of #167: a raster must reach
``CHART_MIN_CLUSTER_AREA`` in PLACED area before its page routes to the
chart-asset lane. The large half is open, and cannot be closed by geometry -- a
page-sized photograph and a page-sized raster chart share placement, aspect
ratio and relation to the surrounding text.

#511 says the honest first step is a measurement, not an implementation: if a
large raster on a born-digital page is RARE, the current fail-toward-chart
behaviour is the right trade and the ticket closes with the number recorded.

This measures the population, not the classification. It answers "how many pages
could be affected at all", which is the bound. Deciding chart-vs-photo needs
something that reads the image (#511 lists the candidates); counting how often
that decision would even arise does not.

Reported per page, so a paper with one offending page is not hidden behind one
with forty. Content-free per ``2026-08-22_binding-oracle-corpus-measurement.md``:
counts and basenames only, never page content.

The population alone does not decide #511, so the same sweep also draws a
reproducible SAMPLE: ``--sample N --out DIR`` crops the offending raster from N
of the hit pages and writes them as PNGs for a human (or a VLM) to classify.
The draw is ordered by ``sha256(paper|page)``, so which pages land in the sample
is fixed by the corpus, not chosen by whoever ran it.

Usage:
    large-raster-frequency.py <dir containing the corpus PDFs> [--limit N]
                              [--sample N --out DIR]
"""

from __future__ import annotations

import hashlib
import logging
import sys
from collections import Counter
from pathlib import Path

import fitz

from socr.figures.extractor import CHART_MIN_CLUSTER_AREA, has_chart_marks

from socr.core.born_digital import BornDigitalDetector


def _largest_raster_area(page) -> float:
    """Largest PLACED area of any embedded raster, in pt^2. 0.0 when none."""
    largest = 0.0
    for image in page.get_images():
        try:
            rects = page.get_image_rects(image[0])
        except Exception:  # noqa: BLE001 - unmeasurable counts as none here
            continue
        for rect in rects or ():
            largest = max(largest, abs(rect.width) * abs(rect.height))
    return largest


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: large-raster-frequency.py <pdf dir> [--limit N]")
    pdf_dir = Path(sys.argv[1])
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    sample = 0
    if "--sample" in sys.argv:
        sample = int(sys.argv[sys.argv.index("--sample") + 1])
    sample_dir = None
    if "--out" in sys.argv:
        sample_dir = Path(sys.argv[sys.argv.index("--out") + 1])
    if sample and sample_dir is None:
        raise SystemExit("--sample needs --out DIR")

    # The detector logs per-region table diagnostics at WARNING. Useful in a
    # pipeline run, deafening across 277 papers -- and it buried the traceback
    # that ended the first sweep.
    logging.disable(logging.CRITICAL)

    detector = BornDigitalDetector()
    pdfs = sorted(pdf_dir.glob("*.pdf"))[:limit]

    tally = Counter()
    papers_with_large: set[str] = set()
    per_paper: Counter = Counter()
    sample_pool: list[tuple[str, int, float]] = []

    for path in pdfs:
        try:
            doc = fitz.open(path)
        except Exception:  # noqa: BLE001 - an unreadable file is not a measurement
            tally["unreadable_pdf"] += 1
            continue
        try:
            for index in range(doc.page_count):
                try:
                    page = doc[index]
                except Exception:  # noqa: BLE001
                    # A malformed page tree killed the first full sweep at paper
                    # ~200. One bad page must cost one page, not the corpus.
                    tally["unreadable_page"] += 1
                    continue
                # The FULL assessment, despite the cost.
                #
                # A char-count proxy was tried first and was wrong in exactly
                # the way that matters: a scanned book with an OCR text layer
                # passes it, and those pages are full-page rasters -- so the
                # proxy reported 99.75% and the number was obviously nonsense.
                # I had reasoned it would over-count the DENOMINATOR and give a
                # lower bound; it inflates the numerator too, because the pages
                # it wrongly admits are precisely the ones carrying a big image.
                # Recorded because the wrong version looks perfectly reasonable.
                assessment = detector._assess_page(page, index + 1)
                if not assessment.is_born_digital:
                    tally["scanned_page"] += 1
                    continue
                tally["born_digital_page"] += 1

                area = _largest_raster_area(page)
                if area <= 0.0:
                    continue
                tally["born_digital_page_with_raster"] += 1
                if area < CHART_MIN_CLUSTER_AREA:
                    tally["raster_below_the_gate"] += 1
                    continue

                tally["raster_at_or_above_the_gate"] += 1
                papers_with_large.add(path.name)
                per_paper[path.name] += 1

                # Confirms the gate and the predicate agree, and nothing more.
                #
                # This was written to ask "would VECTOR evidence route the page
                # anyway?", and it CANNOT: has_chart_marks takes the raster fast
                # path, so a page that just cleared the area gate answers True
                # because of that raster. The two counts are equal by
                # construction, not by measurement. Kept as the agreement check
                # it actually is -- a divergence would mean the gate here and
                # the production predicate had drifted apart.
                if has_chart_marks(page):
                    tally["predicate_agrees"] += 1

                sample_pool.append((path.name, index + 1, area))
        finally:
            doc.close()
        tally["paper"] += 1

    print(f"corpus: {tally['paper']} paper(s) read, {tally['unreadable_pdf']} unreadable\n")
    for key in (
        "born_digital_page",
        "scanned_page",
        "born_digital_page_with_raster",
        "raster_below_the_gate",
        "raster_at_or_above_the_gate",
        "predicate_agrees",
    ):
        print(f"  {key:34s} {tally[key]:6d}")

    bd = tally["born_digital_page"] or 1
    print(
        f"\n  large raster on a born-digital page: "
        f"{tally['raster_at_or_above_the_gate'] / bd:.2%} of born-digital pages, "
        f"across {len(papers_with_large)} of {tally['paper']} papers"
    )
    if per_paper:
        print("\n  papers with the most such pages:")
        for name, count in per_paper.most_common(10):
            print(f"    {count:4d}  {name[:60]}")

    if sample and sample_dir is not None:
        _write_sample(pdf_dir, sample_pool, sample, sample_dir)
    return 0


def _write_sample(
    pdf_dir: Path,
    pool: list[tuple[str, int, float]],
    want: int,
    out_dir: Path,
) -> None:
    """Crop the offending raster from ``want`` hit pages, deterministically.

    Ordering by a hash of the page's identity keeps the draw out of the hands of
    whoever ran the sweep: the same corpus yields the same sample, and a sample
    that happens to flatter a conclusion cannot have been picked for it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = sorted(pool, key=lambda h: hashlib.sha256(f"{h[0]}|{h[1]}".encode()).hexdigest())
    print(f"\n  sample of {min(want, len(pool))} drawn from {len(pool)} hit page(s) -> {out_dir}")
    for n, (name, page_num, _area) in enumerate(pool[:want], 1):
        doc = fitz.open(pdf_dir / name)
        try:
            page = doc[page_num - 1]
            clip = None
            best = 0.0
            for image in page.get_images():
                try:
                    rects = page.get_image_rects(image[0])
                except Exception:  # noqa: BLE001
                    continue
                for rect in rects or ():
                    area = abs(rect.width) * abs(rect.height)
                    if area > best:
                        best, clip = area, rect
            page.get_pixmap(matrix=fitz.Matrix(1.1, 1.1), clip=clip).save(out_dir / f"{n:02d}.png")
            print(f"    {n:02d}  {name[:48]:48s} p{page_num}")
        finally:
            doc.close()


if __name__ == "__main__":
    raise SystemExit(main())
