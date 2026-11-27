"""#1257 replay: the row-binding check before and after plan B1, on the cached 156-doc run.

No OCR.  The cached page markdown (``pages/NNNNN.md``), the source PDF page and the detected table
boxes are FROZEN: both versions of the check read the same bytes, so any difference is the binding
logic and none of it is model noise.  The old check is the file at commit ``BASELINE`` (main before
this change), loaded from git.

Reports, with their denominators:

* every table with a row-binding record in the run (539 pages) and its outcome before/after;
* the gold-clean subset (pages any auditor seat marked clean, by page identity: 33 pages, 39 tables);
* the 84 contradicted tables, replayed from the withheld markdown kept in their sidecars;
* the gold misalignment pages (``row_or_column_misalignment``), before/after;
* the pages that lose their row-binding mark, cross-tabulated with the gold verdicts.

Run from the repo root::

    PYTHONPATH=src ~/venvs/socr/bin/python docs/log/2026-11-14_1257-replay.py

Paths are the constants below; nothing is read from the command line.
"""

from __future__ import annotations

import collections
import glob
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

#: The cached run: one directory per worker, ``out/w*/<document>/`` inside each.
RUN = "/Volumes/Main/Library/socr-work/fed_rerun_51270eb5_words/unpacked"
#: The source PDFs of that run, one ``<document>.pdf`` each.
PDFS = "/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs"
#: The auditors' gold pages (build_gold.py of the audit), one JSON line per verdict.
GOLD = "/Volumes/Main/Library/Databases/socr/gold/fed-2026-11-11.jsonl"
#: main before this change: the check as it was.
BASELINE = "b2d6454e"
BASELINE_FILE = "src/socr/tables/row_binding_check.py"
#: Worker processes.
WORKERS = 8
#: Where the per-table replay is written.
OUT = Path(tempfile.gettempdir()) / "socr-1257-replay.jsonl"
BINDING_DEFECT = "row_or_column_misalignment"


_BASELINE_MODULE = None


def load_baseline():
    global _BASELINE_MODULE
    if _BASELINE_MODULE is None:
        _BASELINE_MODULE = _load_baseline()
    return _BASELINE_MODULE


def _load_baseline():
    text = subprocess.run(
        ["git", "show", f"{BASELINE}:{BASELINE_FILE}"], capture_output=True, text=True, check=True
    ).stdout
    path = Path(tempfile.mkdtemp()) / "baseline_row_binding_check.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("baseline_row_binding_check", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["baseline_row_binding_check"] = module
    spec.loader.exec_module(module)
    return module


def index_pages() -> list[dict]:
    gold = collections.defaultdict(list)
    for line in open(GOLD, encoding="utf-8"):
        d = json.loads(line)
        gold[(d["doc"], d["page"])].append(d)
    out = []
    for f in sorted(glob.glob(RUN + "/*/out/w*/*/pages/*.json")):
        d = json.load(open(f, encoding="utf-8"))
        if not d.get("row_binding"):
            continue
        doc, page = f.split("/")[-3], d["page_num"]
        out.append(
            dict(
                doc=doc,
                page=page,
                sidecar=f,
                status=d["status"],
                mark_only=bool(d.get("row_binding_mark_only")),
                outs=[v["outcome"] for v in d["row_binding"].values()],
                gold=gold.get((doc, page), []),
            )
        )
    return out


def replay_page(item: dict) -> dict:
    import fitz

    from socr.tables import row_binding_check as new
    from socr.tables.locate import locate_tables
    from socr.tables.reconcile import find_table_blocks

    old = load_baseline()
    md = Path(item["sidecar"][:-5] + ".md").read_text(encoding="utf-8")
    page = fitz.open(f"{PDFS}/{item['doc']}.pdf")[item["page"] - 1]
    regions = [tuple(b.bbox) for b in locate_tables(page)] or None
    res_old, _ = old.check_page_row_binding(page, md, regions)
    res_new, _ = new.check_page_row_binding(page, md, regions)
    tables = []
    for k, (a, b) in enumerate(zip(res_old, res_new)):
        tables.append(
            dict(
                idx=k,
                old=a.outcome.value,
                new=b.outcome.value,
                new_evidence=b.evidence_counts(),
                old_rows=[(e.line_index, e.verdict.value, e.reason) for e in a.rows],
                new_rows=[(e.line_index, e.verdict.value, e.reason) for e in b.rows],
            )
        )
    out = dict(
        doc=item["doc"],
        page=item["page"],
        md_sha=hashlib.sha256(md.encode()).hexdigest(),
        n_blocks=len(find_table_blocks(md)),
        n_boxes=len(regions or []),
        tables=tables,
    )
    # the contradicted tables of this page: replayed from the markdown the sidecar kept
    sc = json.load(open(item["sidecar"], encoding="utf-8"))
    words, (boxes, union), _ = new._page_frame(page, regions)
    n_tables = sc.get("row_binding_table_count") or 0
    contradicted = []
    for rec in sc["row_binding"].values():
        if rec["outcome"] != "contradicted":
            continue
        region = boxes[rec["table_index"]] if len(boxes) == n_tables else union
        a = old.check_row_binding(words, region, rec["withheld_markdown"])
        b = new.check_row_binding(words, region, rec["withheld_markdown"])
        conv = ("displaced", "label_lost", "emptied")
        contradicted.append(
            dict(
                tid=rec["table_id"],
                old=a.outcome.value,
                new=b.outcome.value,
                same_rows=[e.line_index for e in a.rows if e.verdict.value in conv]
                == [e.line_index for e in b.rows if e.verdict.value in conv],
            )
        )
    out["contradicted"] = contradicted
    return out


def main() -> None:
    items = index_pages()
    by_key = {(i["doc"], i["page"]): i for i in items}
    with Pool(WORKERS) as pool, open(OUT, "w", encoding="utf-8") as f:
        results = []
        for r in pool.imap(replay_page, items, chunksize=4):
            f.write(json.dumps(r) + "\n")
            results.append(r)

    def clean(i):
        return any(g["content"] == "clean" for g in i["gold"])

    def counts(select):
        o, n = collections.Counter(), collections.Counter()
        for r in results:
            if select(by_key[(r["doc"], r["page"])]):
                for t in r["tables"]:
                    o[t["old"]] += 1
                    n[t["new"]] += 1
        return dict(o), dict(n)

    print(f"replayed pages: {len(results)}  (frozen markdown, PDF page and boxes)")
    print("tables, all           old / new:", *counts(lambda i: True))
    print("tables, gold-clean    old / new:", *counts(clean))
    pages_clean = sum(1 for i in items if clean(i) and "unverifiable" in i["outs"])
    print(f"gold-clean pages with an unverifiable table: {pages_clean}")

    con = [c for r in results for c in r["contradicted"]]
    print(
        f"contradicted tables replayed: {len(con)}; still contradicted: "
        f"{sum(1 for c in con if c['new'] == 'contradicted')}; same convicted rows: "
        f"{sum(1 for c in con if c['same_rows'])}"
    )

    mis = {
        k for k, i in by_key.items() if any(BINDING_DEFECT in g["defect_types"] for g in i["gold"])
    }
    gold_all = collections.defaultdict(list)
    for line in open(GOLD, encoding="utf-8"):
        d = json.loads(line)
        if BINDING_DEFECT in d["defect_types"]:
            gold_all[(d["doc"], d["page"])].append(d)
    print(
        f"gold misalignment pages (any seat/run): {len(gold_all)}; with a row-binding record: {len(mis)}"
    )
    for r in results:
        if (r["doc"], r["page"]) in gold_all:
            print("   ", r["doc"][-34:], r["page"], [(t["old"], t["new"]) for t in r["tables"]])

    newly = [
        (r, t) for r in results for t in r["tables"] if t["old"] != "bound" and t["new"] == "bound"
    ]
    kinds = collections.Counter()
    for _, t in newly:
        kinds.update(t["new_evidence"])
    print(f"tables newly BOUND: {len(newly)}; rows bound by new evidence: {dict(kinds)}")

    freed = [
        (r, by_key[(r["doc"], r["page"])])
        for r in results
        if any(t["old"] == "unverifiable" for t in r["tables"])
        and not any(t["new"] == "unverifiable" for t in r["tables"])
        and not any(t["old"] == "contradicted" for t in r["tables"])
    ]
    cat = collections.Counter()
    for r, i in freed:
        same = [g for g in i["gold"] if g["run"] == "51270eb5" and g["md_sha256"] == r["md_sha"]]
        if any(g["content"] == "defective" for g in same):
            c = "gold-defective on this exact markdown"
        elif any(g["content"] == "clean" for g in same):
            c = "gold-clean on this exact markdown"
        elif any(g["content"] == "defective" for g in i["gold"]):
            c = "gold-defective on the page (another reading of it)"
        elif any(g["content"] == "clean" for g in i["gold"]):
            c = "gold-clean on the page (another reading of it)"
        else:
            c = "no gold verdict"
        cat[(c, "mark only" if i["mark_only"] else "other marks too")] += 1
    print(f"pages that lose the row-binding mark: {len(freed)}")
    for k, v in sorted(cat.items()):
        print(f"    {v:3d}  {k[0]}  [{k[1]}]")
    print(f"per-table replay written to {OUT}")


if __name__ == "__main__":
    main()
