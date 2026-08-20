#!/usr/bin/env python3
"""Run every selected page through BOTH lanes and keep both answers.

One socr run per document with --no-native-first, so the model is forced on every
page. socr caches each engine's attempt, so the native answer and the model answer
for the same page both survive even when one of them is discarded — which is the
whole point: #259 is about a correct model answer being thrown away.

Per-document subprocess timeout, because socr can hang before it ever reaches the
model (observed: 12 minutes elapsed, 1.96s CPU, no output directory).
"""

import base64
import glob
import json
import os
import re
import subprocess
import sys
import time

import fitz

SP = "/private/tmp/claude-501/-Users-rubenffuertes-repos-tools-socr/bae3cd33-3698-4e2f-999a-b6ed84289e24/scratchpad"
CAMP = f"{SP}/campaign"
NUM = re.compile(r"-?\d+\.\d+")
PER_PAGE_BUDGET_S = 1080  # measured: ~14 min/page on dense tables (dual-pass + table rereads)


def run_doc(entry, idx):
    name = entry["name"].replace(".pdf", "")
    slug = f"doc{idx:02d}"
    pages = [p["page"] for p in entry["pages"]]
    work = f"{CAMP}/{slug}"
    os.makedirs(work, exist_ok=True)

    src = fitz.open(entry["pdf"])
    ex = fitz.open()
    for pg in pages:
        ex.insert_pdf(src, from_page=pg - 1, to_page=pg - 1)
    ex_path = f"{work}/excerpt.pdf"
    ex.save(ex_path)
    ex.close()

    budget = PER_PAGE_BUDGET_S * len(pages)
    t0 = time.time()
    status = "ok"
    try:
        subprocess.run(
            [
                os.path.expanduser("~/venvs/socr/bin/socr"),
                "process",
                ex_path,
                "--agentic",
                "--no-native-first",
                "-o",
                f"{work}/out",
            ],
            cwd=work,
            timeout=budget,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "PYTHONPATH": f"{SP}/ciwt/src"},
        )
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    except Exception as exc:  # noqa: BLE001
        status = f"ERROR {exc}"
    elapsed = round(time.time() - t0, 1)

    # Mine every cached attempt: engine -> text, per page.
    records = []
    outdir = glob.glob(f"{work}/out/*/")
    if outdir:
        base = outdir[0]
        # cache entries carry page_num — map per (page, engine), never document-wide
        by_page = {}
        for f in glob.glob(f"{base}cache/*/*.json"):
            try:
                d = json.load(open(f))
            except Exception:  # noqa: BLE001
                continue
            txt = d.get("text") or ""
            pn = d.get("page_num")
            if txt and pn is not None:
                by_page.setdefault(pn, {})[d.get("engine") or "?"] = txt
        for n, pg in enumerate(pages, start=1):
            pj = f"{base}pages/{n:05d}.json"
            shipped = f"{base}pages/{n:05d}.md"
            rec = {
                "doc": name,
                "slug": slug,
                "source_page": pg,
                "kind": next(p["kind"] for p in entry["pages"] if p["page"] == pg),
                "run_status": status,
                "elapsed_s": elapsed,
            }
            if os.path.exists(pj):
                d = json.load(open(pj))
                rec["shipped_engine"] = d.get("engine")
                rec["page_status"] = d.get("status")
                rec["flags"] = {k: v for k, v in d.items() if k.startswith("native_table") and v}
            if os.path.exists(shipped):
                rec["shipped_text"] = open(shipped).read()
            # page image
            pix = fitz.open(ex_path)[n - 1].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            rec["image_b64"] = base64.b64encode(pix.tobytes("jpeg", jpg_quality=68)).decode()
            records.append(rec)
        for n, r in enumerate(records, start=1):
            r["candidates"] = by_page.get(n, {})
            r["decimals"] = {e: len(NUM.findall(t)) for e, t in r["candidates"].items()}
    src.close()
    return records, status, elapsed


def main():
    sel = json.load(open(f"{CAMP}/manifest.json"))
    all_recs = []
    for i, entry in enumerate(sel, start=1):
        recs, status, elapsed = run_doc(entry, i)
        all_recs += recs
        print(
            f"[{i}/{len(sel)}] {entry['name'][:44]:44s} {status:8s} {elapsed:7.1f}s  pages={len(recs)}",
            flush=True,
        )
        json.dump(all_recs, open(f"{CAMP}/records.json", "w"))
    print(f"DONE: {len(all_recs)} page records -> {CAMP}/records.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
