#!/usr/bin/env python3
"""Which table pages of the lane-comparison manifest carry an eligible row.

GH-338. The 2026-08-30 measurement scored 13 rows out of the manifest's 15
``kind: table`` pages and listed neither the 13 nor the 2 it dropped. The
selection script was never committed, so the original 13 cannot be RECOVERED --
only recomputed. This is that recomputation, and it is labelled as one: a
difference between its output and the original run's is a fact about two
implementations of the same written rule, not proof of either.

The rule, quoted from the log:

    a y-band carrying >= 3 numeric tokens with >= 2 neighbouring numeric bands
    within twice the page's own median band pitch

Everything measurable is taken from socr rather than reimplemented, so the
numeric-token predicate and the banding are the ones production uses:
``_is_numeric_word`` and ``round(y0)`` grouping from ``socr.tables.reconstruct``.
No page content is printed -- only (paper, page) identifiers and counts, per the
content-free convention of ``2026-08-22_binding-oracle-corpus-measurement.md``.

Usage:
    row-eligibility.py <manifest.json> <dir containing the corpus PDFs>
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import fitz

from socr.tables.reconstruct import _is_numeric_word

MIN_NUMERIC_TOKENS = 3  # ">= 3 numeric tokens", quoted above
MIN_NEIGHBOURS = 2  # ">= 2 neighbouring numeric bands"
NEIGHBOUR_PITCH_MULT = 2.0  # "within twice the page's own median band pitch"


def numeric_bands(page) -> tuple[list[int], float | None]:
    """Return (y of each band with enough numeric tokens, median band pitch)."""
    bands: dict[int, list] = defaultdict(list)
    for w in page.get_text("words"):
        bands[round(w[1])].append(w)

    ys = sorted(bands)
    if len(ys) < 2:
        return [], None
    pitch = statistics.median(ys[i + 1] - ys[i] for i in range(len(ys) - 1))

    numeric = [
        y for y in ys if sum(1 for w in bands[y] if _is_numeric_word(w)) >= MIN_NUMERIC_TOKENS
    ]
    return numeric, pitch


def eligible_rows(page) -> int:
    """How many bands on this page satisfy the whole rule."""
    numeric, pitch = numeric_bands(page)
    if pitch is None or not numeric:
        return 0
    reach = NEIGHBOUR_PITCH_MULT * pitch
    return sum(
        1
        for y in numeric
        if sum(1 for other in numeric if other != y and abs(other - y) <= reach) >= MIN_NEIGHBOURS
    )


def _key(name: str) -> str:
    """Filename identity, insensitive to underscore runs and case."""
    return re.sub(r"_+", "_", name).lower()


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: row-eligibility.py <manifest.json> <pdf dir>")
    manifest = json.loads(Path(sys.argv[1]).read_text())
    pdf_dir = Path(sys.argv[2])

    # The manifest records basenames only (the corpus is copyrighted and this
    # repo is public), and at least one has been renamed since -- a doubled
    # underscore collapsed to one. Resolve on a normalised name so a cosmetic
    # rename does not silently turn a scored page into "pdf not found", which
    # would look exactly like a dropped page.
    #
    # GH-501: an alias must never be AMBIGUOUS. Two corpus files can normalise
    # to the same key, and a dict comprehension would silently keep the last
    # one -- scoring one paper while printing another paper's identifier, which
    # is the one failure this whole script exists to rule out. Exact basename
    # always wins; a normalised key claimed by more than one file is refused.
    by_exact = {p.name: p for p in pdf_dir.glob("*.pdf")}
    by_key: dict[str, Path] = {}
    ambiguous: dict[str, list[str]] = defaultdict(list)
    for path in sorted(by_exact.values()):
        ambiguous[_key(path.name)].append(path.name)
    for key, names in ambiguous.items():
        if len(names) == 1:
            by_key[key] = pdf_dir / names[0]

    rows = []
    unresolved: list[tuple[str, list[str]]] = []
    for entry in manifest:
        table_pages = [p["page"] for p in entry["pages"] if p["kind"] == "table"]
        if not table_pages:
            continue
        pdf = by_exact.get(entry["pdf"]) or by_key.get(_key(entry["pdf"]))
        if pdf is None:
            unresolved.append((entry["pdf"], ambiguous.get(_key(entry["pdf"]), [])))
            rows.extend((entry["pdf"], page_num, None) for page_num in table_pages)
            continue
        doc = fitz.open(pdf)
        try:
            for page_num in table_pages:
                rows.append((entry["pdf"], page_num, eligible_rows(doc[page_num - 1])))
        finally:
            doc.close()

    eligible = [r for r in rows if r[2]]
    print(f"{len(rows)} table pages; {len(eligible)} carry at least one eligible row\n")
    for name, page_num, count in rows:
        mark = "--" if count is None else ("ELIGIBLE" if count else "dropped ")
        shown = "UNRESOLVED" if count is None else f"{count} eligible band(s)"
        print(f"  {mark}  {name[:44]:44s} p{page_num:<4d} {shown}")

    # GH-501: a partial corpus must not read as an eligibility result. An
    # unresolved paper prints in the same table as a page with zero eligible
    # bands, and the two mean opposite things -- "not measured" versus
    # "measured and dropped". Exit non-zero so a caller cannot mistake one for
    # the other, and say which papers and why.
    if unresolved:
        print(f"\n{len(unresolved)} paper(s) could not be resolved in {pdf_dir}:")
        for name, collisions in unresolved:
            why = (
                f"normalised name claimed by {len(collisions)} files: {collisions}"
                if collisions
                else "no file with this basename, and no unambiguous normalised match"
            )
            print(f"  {name}\n    {why}")
        print("\nThe counts above are INCOMPLETE; this is not an eligibility result.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
