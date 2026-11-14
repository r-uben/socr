"""#1276 measurement on the cached 156-doc run. No OCR: reads sidecars and the PDFs only.

Reports (1) the native lane: pages whose chart box and regional tick evidence fire, with each
crop cut by ``figure_crop`` and rendered under CROPS; (2) the model lane: pages whose reading
carries more ``<!-- Image`` placeholders than figure images ship, where an image counts only
if its file exists under the document's own directory (the disk check, as the pipeline does),
placeholders in fenced code, inline code and table cells are not counted, and the link-only
count is reported beside it so the effect of the disk check is visible.

Run from the repo root::

    PYTHONPATH=src ~/venvs/socr/bin/python docs/log/2026-11-13_1276-measure.py

Paths are the constants below; nothing is read from the command line.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

#: The cached run: one directory per worker, ``out/w*/<document>/`` inside each.
RUN = "/Volumes/Main/Library/socr-work/fed_rerun_51270eb5_words/unpacked"
#: The source PDFs of that run, one ``<document>.pdf`` each.
PDFS = "/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs"
#: Where the measurement writes: crops, one JSON line per firing region, the placeholder table.
OUT_DIR = Path("/Users/rubenffuertes/repos/tools/socr/scratch/army/phaseA")
CROPS = OUT_DIR / "a3-crops-v2"
OUT = str(OUT_DIR / "a3-firings-v2.jsonl")
PLACEHOLDERS_OUT = str(OUT_DIR / "a3-placeholders-v3.json")
#: Worker processes.
WORKERS = 8


def doc_pages(doc_dir):
    for sc in sorted(glob.glob(os.path.join(doc_dir, "pages", "*.json"))):
        yield sc


def work(doc_dir):
    import fitz

    from socr.core.born_digital import BornDigitalDetector
    from socr.core.manifest import is_page_failed_marker
    from socr.figures import figure_presence as fp
    from socr.figures.unshipped_figures import tick_run
    from socr.tables.reconstruct import chart_region_clusters
    from socr.figures import figure_crop
    from socr.figures.chart_regions import render_chart_region_crops
    import fitz as _fitz

    stem = os.path.basename(doc_dir.rstrip("/"))
    pdf = os.path.join(PDFS, stem + ".pdf")
    rows = []
    stats_pages = []
    stats = Counter()
    placeholder_rows = []
    det = BornDigitalDetector()
    doc = fitz.open(pdf) if os.path.exists(pdf) else None
    for sc in doc_pages(doc_dir):
        j = json.load(open(sc))
        pn = int(j["page_num"])
        body = (j.get("winning_output") or {}).get("text", "") or ""
        engine = str(j.get("engine") or "")
        status = j.get("status")
        # qwen side
        if engine and not engine.startswith("native") and engine != "chart_asset":
            P = fp.placeholder_count(body)
            if P:
                on_disk = fp.shipped_image_count(body, Path(doc_dir))
                link_only = fp.shipped_image_count(body)
                has_tables = None
                if doc is not None:
                    try:
                        has_tables = bool(det._detect_tables(doc[pn - 1]))
                    except Exception:
                        has_tables = None
                placeholder_rows.append(
                    {
                        "doc": stem,
                        "page": pn,
                        "status": status,
                        "placeholders": P,
                        "images": on_disk,
                        "images_link_only": link_only,
                        "has_tables": has_tables,
                        "marked_by_rule": P > on_disk,
                    }
                )
            continue
        if not engine.startswith("native") or engine == "chart_asset":
            continue
        stats["native_pages"] += 1
        stats[f"native_{status}"] += 1
        if doc is None:
            stats["no_pdf"] += 1
            continue
        if j.get("chart_floor_body") or is_page_failed_marker(body):
            stats["skip_floor"] += 1
            continue
        if re.search(
            r"!\[[^\]]*\]\([^)]*(invisible_scan|prior_ocr_layer|scanned_figure|shredded)", body
        ):
            stats["skip_own_page_image"] += 1
            continue
        try:
            page = doc[pn - 1]
            clusters = chart_region_clusters(page)
            boxes = [b for _c, b in clusters]
        except Exception:
            stats["box_error"] += 1
            continue
        if not boxes:
            stats["no_box_pages"] += 1
            continue
        stats["native_with_box"] += 1
        try:
            has_tables = bool(det._detect_tables(page))
        except Exception:
            has_tables = False
        if has_tables:
            stats["skip_has_tables_gh189"] += 1
            continue
        words = list(page.get_text("words") or [])
        ass = fp.assess_regions(words, boxes, fp.own_region_refs(body, pn), [], False)
        stats["checked_pages"] += 1
        fires = [a for a in ass if a.fires]
        body_tick = tick_run(body)
        for a in ass:
            if a.ticks >= 3 and not a.table_region and a.tick_lines < fp.MIN_TICK_LINES:
                stats["regions_dropped_by_one_row_rule"] += 1
        if fires:
            stats["firing_pages"] += 1
            stats[f"firing_{status}"] += 1
            groups = figure_crop.crop_groups(
                list(words), [tuple(c) for c, _b in clusters], page.rect
            )
            fired = {a.index for a in fires}
            for g in groups:
                if not fired & set(g.indices):
                    continue
                crop_boxes = list(boxes)
                crop_boxes[g.index - 1] = _fitz.Rect(*g.rect)
                short = stem.replace("fed-meetings-", "")
                for asset in render_chart_region_crops(
                    pdf,
                    pn,
                    crop_boxes,
                    CROPS / short,
                    indices=[g.index],
                    filename=fp.crop_filename,
                ):
                    stats["crops_rendered"] += int(bool(asset.rendered))
                a = next(x for x in fires if x.index in g.indices)
                stats["crops"] += 1
                stats["crops_merged_panels"] += int(len(g.indices) > 1)
                rows.append(
                    {
                        "doc": stem,
                        "page": pn,
                        "region": g.index,
                        "regions": list(g.indices),
                        "ticks": a.ticks,
                        "tick_lines": a.tick_lines,
                        "box": [round(v, 1) for v in g.rect],
                        "old_box": [
                            round(boxes[g.index - 1].x0, 1),
                            round(boxes[g.index - 1].y0, 1),
                            round(boxes[g.index - 1].x1, 1),
                            round(boxes[g.index - 1].y1, 1),
                        ],
                        "status": status,
                        "engine": engine,
                        "body_tick_run": body_tick,
                        "regions_on_page": len(boxes),
                    }
                )
        else:
            stats["pass_pages"] += 1
            if all(a.covered for a in ass):
                stats["pass_all_shipped"] += 1
            else:
                stats["pass_no_evidence"] += 1
                if body_tick >= 3:
                    stats["pass_but_body_tick_ge3"] += 1
                    stats_pages.append(
                        {
                            "doc": stem,
                            "page": pn,
                            "status": status,
                            "regions": [a.to_dict() for a in ass],
                        }
                    )
                if any(a.table_region for a in ass):
                    stats["pass_table_region_excluded"] += 1
    if stats_pages:
        stats["_pass_tick_pages"] = stats_pages
    return rows, dict(stats), placeholder_rows


def main():
    docs = sorted(glob.glob(os.path.join(RUN, "*/out/w*/*/")))
    docs = [d for d in docs if os.path.isdir(os.path.join(d, "pages"))]
    print(len(docs), "docs", file=sys.stderr)
    rows, stats, ph, extra_pages = [], Counter(), [], []
    with Pool(WORKERS) as pool:
        for r, s, p in pool.imap_unordered(work, docs):
            rows += r
            extra_pages += s.pop("_pass_tick_pages", [])
            stats.update(s)
            ph += p
    rows.sort(key=lambda r: (r["doc"], r["page"], r["region"]))
    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(dict(stats), indent=1, sort_keys=True))
    print("pass pages whose whole body has a tick run:", json.dumps(extra_pages))
    print("firing regions", len(rows), "pages", len({(r["doc"], r["page"]) for r in rows}))
    marked = [p for p in ph if p["marked_by_rule"]]
    print("placeholder pages", len(ph), "marked by rule", len(marked))
    print(
        "  unmatched placeholders",
        sum(p["placeholders"] - p["images"] for p in marked),
        "| marked by link count only",
        sum(1 for p in ph if p["placeholders"] > p["images_link_only"]),
        "| table-signal pages among marked (judged after GH-189)",
        sum(1 for p in marked if p["has_tables"]),
    )
    print("  by status", dict(Counter(p["status"] for p in ph)))
    json.dump(ph, open(PLACEHOLDERS_OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
