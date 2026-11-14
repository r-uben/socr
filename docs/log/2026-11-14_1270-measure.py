"""#1270 B4 measurement on the cached 156-doc run 51270eb5. Reads sidecars + PDFs. No OCR."""

import glob, json, os, re, sys, collections

sys.path.insert(0, "/Users/rubenffuertes/.herdr/worktrees/socr/fix-issue-1270-b4/src")
import fitz
from socr.core.born_digital import BornDigitalDetector, count_pua_chars
from socr.core.verdict import NO_NUMBER_REASON, DAMAGED_LAYER_AGREES_REASON, DAMAGED_LAYER_PREFIX

ROOT = "/Volumes/Main/Library/socr-work/fed_rerun_51270eb5_words/unpacked"
PDFS = "/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs"
GOLD = "/Volumes/Main/Library/Databases/socr/gold/fed-2026-11-11.jsonl"
gold = collections.defaultdict(set)
for l in open(GOLD):
    r = json.loads(l)
    if r["run"] == "51270eb5":
        gold[(r["doc"], r["page"])].add(r["verdict"])
clean_keys = {
    k
    for k, v in gold.items()
    if v & {"clean", "still_clean"}
    and not v & {"defective", "finding", "still_silent", "changed_still_wrong"}
}
print("gold-clean (doc,page) keys on 51270eb5:", len(clean_keys), file=sys.stderr)
pages = []
for f in glob.glob(ROOT + "/*/out/*/*/pages/*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    stem = f.split("/")[-3]
    pn = int(d["page_num"])
    v = d.get("verdict")
    clauses = v.get("clauses", []) if isinstance(v, dict) else []
    pages.append(
        dict(
            f=f,
            stem=stem,
            page=pn,
            status=d.get("status"),
            mode=d.get("failure_mode"),
            engine=d.get("engine"),
            clauses=clauses,
            words=d.get("text_layer_words"),
            cov=d.get("text_coverage"),
            events=d.get("audit_events") or [],
            terminal=d.get("terminal"),
            flag_glyph=d.get("has_unmapped_math_glyphs"),
            prior=d.get("prior_ocr_layer_suspect"),
        )
    )
print("pages", len(pages), file=sys.stderr)
json.dump(pages, open("/private/tmp/claude-501/b4/pages.json", "w"))
