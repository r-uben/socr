"""#1294 measurement: apply the words clause, with socr's own page analysis, to every page of
two finished runs, and score it against the auditors' gold pages. No OCR, no model.

Run (paths are the 2026-11-11 audit's; adjust):

    PYTHONPATH=docs/log ~/venvs/socr/bin/python -c \
      "import importlib; m = importlib.import_module('2026-11-12_1294-measure'); m.main('out.jsonl')"

then ``m.report('out.jsonl')``. Results: ``2026-11-12_1294-text-layer-words.md``.
"""

import collections
import hashlib
import json
from pathlib import Path

from socr.benchmark import scoreboard as sb

PDFS = Path("/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs")
RUNS = {
    "aebc444c": Path("/Volumes/Main/Library/socr-work/fed_aebc444c/unpacked"),
    "51270eb5": Path("/Volumes/Main/Library/socr-work/fed_rerun_51270eb5/unpacked"),
}
GOLD = "/Volumes/Main/Library/Databases/socr/gold/fed-2026-11-11.jsonl"


def _gold():
    g = collections.defaultdict(list)
    for l in open(GOLD):
        e = json.loads(l)
        g[(e["doc"], int(e["page"]), e["md_sha256"])].append(e)
    return g


def job(args):
    doc, dirs = args
    from socr.core.born_digital import BornDigitalDetector
    from socr.core.result import FailureMode, PageOutput
    from socr.core.state import PageState
    from socr.core.verdict import page_word_check_record, text_layer_words_clause

    gold = _gold()
    pdf = PDFS / f"{doc}.pdf"
    if not pdf.exists():
        return []
    try:
        analysis = BornDigitalDetector().detect(pdf)
    except Exception as exc:
        return [{"doc": doc, "error": repr(exc)}]
    out = []
    for run, doc_dir in dirs.items():
        for sc in sorted((doc_dir / "pages").glob("*.json")):
            side = sb._read_json(sc)
            md = sc.with_suffix(".md")
            if side is None or not md.exists():
                continue
            n = int(side.get("page_num") or int(sc.stem))
            if n - 1 >= len(analysis.pages):
                continue
            pa = analysis.pages[n - 1]
            p = PageState(page_num=n)
            for f in (
                "own_layer_word_reference",
                "own_layer_figure_words",
                "is_born_digital",
                "has_unrecovered_symbol_glyphs",
            ):
                setattr(p, f, getattr(pa, f))
            p.prior_ocr_layer_suspect = bool(side.get("prior_ocr_layer_suspect")) or bool(
                pa.invisible_text_over_raster
                or pa.invisible_text_scan_failed
                or pa.prior_ocr_layer
                or pa.prior_ocr_layer_scan_failed
            )
            disp = side.get("table_ladder_disposition")
            try:
                p.table_ladder_disposition = FailureMode(disp) if disp else None
            except ValueError:
                p.table_ladder_disposition = None
            output = PageOutput.from_dict(side.get("winning_output") or {})
            clause = text_layer_words_clause(output, p)
            rec = page_word_check_record(output, p)
            raw = md.read_bytes()
            g = gold.get((doc, n, hashlib.sha256(raw).hexdigest()), [])
            out.append(
                {
                    "run": run,
                    "doc": doc,
                    "page": n,
                    "status": side.get("status"),
                    "engine": side.get("engine"),
                    "marked": sb._is_marked(raw.decode(errors="replace")),
                    "applies": rec is not None,
                    "compared": (rec or {}).get("compared", 0),
                    "missing": (rec or {}).get("missing", 0),
                    "result": clause.result.value if clause else "",
                    "reason": clause.reason[:300] if clause else "",
                    "gold": sorted({e["content"] for e in g}),
                    "defects": sorted({d for e in g for d in e.get("defect_types") or []}),
                }
            )
    return out


def main(out_path, workers=6):
    from concurrent.futures import ProcessPoolExecutor

    docs = json.load(
        open(
            "/Volumes/Main/Library/Databases/socr/audits/2026-11-11_fed_independent/opus/rerun-doc-index.json"
        )
    )
    found = {r: sb.find_documents(root) for r, root in RUNS.items()}
    jobs = [(d, {r: found[r][d] for r in RUNS if d in found[r]}) for d in docs]
    n = 0
    with open(out_path, "w") as f, ProcessPoolExecutor(workers) as pool:
        for res in pool.map(job, jobs):
            for r in res:
                f.write(json.dumps(r) + "\n")
                n += 1
    print(n, "rows")


LOSS = {
    "lost_content",
    "lost_text",
    "lost_qualifier",
    "splice_math_prose_deletion",
    "dropped_ligature_glyphs",
    "lost_table",
    "dropped_redaction_notice",
}


def cls(r):
    if "defective" in r["gold"] and set(r["defects"]) & LOSS:
        return "loss"
    if "defective" in r["gold"]:
        return "other_defect"
    if r["gold"] == ["clean"]:
        return "clean"


def report(path, prev=None):
    rows = [json.loads(l) for l in open(path)]
    g = [r for r in rows if r["gold"] and r["status"] == "success" and not r["marked"]]
    for k in ["loss", "other_defect", "clean"]:
        s = [r for r in g if cls(r) == k]
        print(f"{k:13} n={len(s):3} flagged={sum(bool(r['result']) for r in s):3}")
    for run in ["aebc444c", "51270eb5"]:
        sil = [r for r in rows if r["run"] == run and r["status"] == "success" and not r["marked"]]
        print(
            run,
            "silent",
            len(sil),
            "flagged",
            sum(bool(r["result"]) for r in sil),
            dict(collections.Counter(r["engine"] for r in sil if r["result"])),
        )
    if prev:
        old = {(r["run"], r["doc"], r["page"]): r for r in map(json.loads, open(prev))}
        new_loss = [
            r
            for r in g
            if cls(r) == "loss"
            and r["result"]
            and not old[(r["run"], r["doc"], r["page"])]["result"]
        ]
        print("loss pages newly caught vs prev:", len(new_loss))
        for r in new_loss[:8]:
            print("  ", r["doc"][13:45], r["page"], r["reason"].split(": ", 1)[-1][:80])
        cl = [r for r in g if cls(r) == "clean" and r["result"]]
        for r in cl:
            print(
                "  CLEAN",
                r["run"][:4],
                r["doc"][13:45],
                r["page"],
                r["engine"],
                r["reason"].split(": ", 1)[-1][:80],
            )
