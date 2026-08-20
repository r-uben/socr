#!/usr/bin/env python3
"""Build the page manifest for the lane comparison — the selection half.

Committed because the record claims the page choice was deterministic rather than
by eye. That claim was previously unverifiable: the runner takes the manifest as
input and contains no selection code, so nothing in the repo showed how pages
were picked. This is that code.

Selection is by signal density, not by looking at the pages:
  table    — parenthesised standard errors, the shape of a regression table
  equation — math glyph and LaTeX-fragment density
  figure   — an embedded raster with little text

Up to four pages per document, tables first, then equations, then figures.
"""

import glob
import json
import os
import re
import sys

import fitz

SE = re.compile(r"\(\d\.\d{2,3}\)")
MATH = re.compile(r"[∫∑∏√≤≥≈±∈∂]|\\frac|\\sum")
ORDER = {"table": 0, "equation": 1, "figure": 2}


def classify(page) -> tuple[str | None, int, int, int]:
    text = page.get_text("text")
    se = len(SE.findall(text))
    math = len(MATH.findall(text))
    imgs = len(page.get_images())
    if se >= 8:
        return "table", se, math, imgs
    if math >= 6:
        return "equation", se, math, imgs
    if imgs >= 1 and len(text) < 1200:
        return "figure", se, math, imgs
    return None, se, math, imgs


def select(pdf_glob: str, max_docs: int = 9, per_doc: int = 4) -> list[dict]:
    out = []
    for path in sorted(glob.glob(pdf_glob)):
        if os.path.getsize(path) < 50_000:
            continue
        try:
            doc = fitz.open(path)
        except Exception:  # noqa: BLE001
            continue
        if doc.page_count > 80:
            doc.close()
            continue
        pages = []
        for i in range(min(doc.page_count, 45)):
            kind, se, math, imgs = classify(doc[i])
            if kind:
                pages.append({"page": i + 1, "kind": kind, "se": se, "math": math, "imgs": imgs})
        doc.close()
        pages.sort(key=lambda p: (ORDER[p["kind"]], -p["se"], -p["math"]))
        if pages:
            # basename only: the corpus is copyrighted and this repo is public,
            # so its location on disk is not committed.
            out.append(
                {
                    "pdf": os.path.basename(path),
                    "name": os.path.basename(path)[:60],
                    "pages": pages[:per_doc],
                }
            )
    return out[:max_docs]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: select.py '<glob of pdfs>' [out.json]")
    sel = select(sys.argv[1])
    dest = sys.argv[2] if len(sys.argv) > 2 else "manifest.json"
    json.dump(sel, open(dest, "w"), indent=1)
    print(f"{sum(len(s['pages']) for s in sel)} pages across {len(sel)} documents -> {dest}")
