"""#1270 B4 classes 1-3 under the release rule: what the cached run's pages do.

Reads the sidecars of run 51270eb5 and the source PDFs; no OCR. For each page that carried a
class 1, 2 or 3 mark before this change, it runs the CURRENT verdict code on the page's shipped
text and a page state rebuilt from the detector, and says whether the mark was removed
(``released``), reworded, or kept. Gold (the owner's hand grades, ``content`` clean or defective)
is joined on (doc, page).

    measure_release <run_root> <pdf_dir> <gold.jsonl> <out.json>

``run_root`` holds ``*/out/*/<stem>/pages/NNNNN.json`` (the unpacked run).
"""

import collections
import glob
from dataclasses import replace
import json
import os
import sys
from types import SimpleNamespace

from socr.core.born_digital import BornDigitalDetector
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.verdict import (
    BULLET_GLYPHS_REASON,
    SHAPE_REFUSED_LAYER_AGREES_REASON,
    SHAPE_REFUSED_LAYER_PREFIX,
    derive_clauses,
    private_use_bullets_released,
)
from socr.pipeline.refinalize_own_text import _page_state

PUA_REASON = "private-use codepoints in this page's native text layer are unrecovered mathematics"
NO_NUMBER = "the reading ships no number to check against the page's text layer"
AGREES_PREFIX = "the reading's numbers agree with this page's text layer, but"

run_root, pdf_dir, gold_path, out_path = sys.argv[1:5]

gold = collections.defaultdict(set)
for line in open(gold_path):
    row = json.loads(line)
    if row["run"] == "51270eb5":
        gold[(row["doc"], row["page"])].add(row["content"])


def gold_label(stem, page):
    got = gold.get((stem, page), set())
    if "defective" in got:
        return "defective"
    if got == {"clean"}:
        return "clean"
    return "other" if got else "ungraded"


def cls_of(clause):
    check, reason = clause["check"], clause["reason"]
    if check == "native_glyphs" and reason == PUA_REASON:
        return 1
    if check == "text_layer_numbers" and reason == NO_NUMBER:
        return 2
    if check == "text_layer_numbers" and reason.startswith(AGREES_PREFIX):
        return 3
    return None


detector = BornDigitalDetector()
rows = []
for path in glob.glob(f"{run_root}/*/out/*/*/pages/*.json"):
    try:
        side = json.load(open(path))
    except Exception:
        continue
    verdict = side.get("verdict")
    clauses = verdict.get("clauses", []) if isinstance(verdict, dict) else []
    classes = sorted({c for c in map(cls_of, clauses) if c})
    if not classes or side.get("terminal") is not True:
        continue
    stem = path.split("/")[-3]
    page = int(side["page_num"])
    pdf = os.path.join(pdf_dir, stem + ".pdf")
    try:
        pa = detector.detect_page(pdf, page)
    except Exception as exc:
        rows.append(dict(stem=stem, page=page, classes=classes, error=str(exc)))
        continue
    wo = side["winning_output"]
    p = _page_state(side, pa)
    p.text_coverage_verdict = side.get("text_coverage", "") or ""
    p.has_unmapped_math_glyphs = bool(pa.has_unmapped_math_glyphs)
    p.private_use_bullets_only = bool(pa.private_use_bullets_only)
    p.has_figures = pa.has_figures
    disposition = side.get("table_ladder_disposition")
    p.table_ladder_disposition = FailureMode(disposition) if disposition else None
    kinds = frozenset(e["kind"] for e in side.get("audit_events") or [])
    out = PageOutput(
        page_num=page,
        text=wo["text"],
        status=PageStatus(side["status"]),
        engine=side.get("engine") or "qwen",
        failure_mode=FailureMode(side.get("failure_mode") or "none"),
    )
    before = [(c["check"], c["reason"]) for c in clauses]
    released1 = 1 in classes and private_use_bullets_released(out, p, kinds)
    # After: a page whose private-use mark is lifted ships SUCCESS (the guard no longer demotes).
    shown = replace(out, status=PageStatus.SUCCESS) if released1 else out
    after_clauses = derive_clauses(shown, p, None, kinds)
    after = [(c.check.value, c.reason) for c in after_clauses]
    per_class = {}
    for k in classes:
        mine = [b for b in before if cls_of({"check": b[0], "reason": b[1]}) == k]
        if k == 1:
            gone = not any(a[0] == "native_glyphs" for a in after)
            reworded = any(a[1] == BULLET_GLYPHS_REASON for a in after)
        elif k == 2:
            gone = not any(a[0] == "text_layer_numbers" for a in after)
            reworded = False
        else:
            gone = not any(a[0] == "text_layer_numbers" for a in after)
            reworded = any(
                a[1] == SHAPE_REFUSED_LAYER_AGREES_REASON
                or a[1].startswith(SHAPE_REFUSED_LAYER_PREFIX)
                for a in after
            )
        per_class[k] = "released" if gone else ("reworded" if reworded else "kept")
        assert mine
    rows.append(
        dict(
            stem=stem,
            page=page,
            classes=classes,
            per_class=per_class,
            n_before=len(before),
            n_after=len(after),
            last_mark_removed=bool(before) and not after,
            gold=gold_label(stem, page),
            engine=side.get("engine"),
        )
    )

json.dump(rows, open(out_path, "w"), indent=1)
ok = [r for r in rows if "error" not in r]
print("candidate pages", len(rows), "errors", len(rows) - len(ok))
for k in (1, 2, 3):
    mine = [r for r in ok if k in r["per_class"]]
    tally = collections.Counter(r["per_class"][k] for r in mine)
    lost = [r for r in mine if r["last_mark_removed"] and r["per_class"][k] == "released"]
    print(
        f"class {k}: marked pages {len(mine)}; {dict(tally)}; pages that lose EVERY mark {len(lost)};"
        f" gold of those {dict(collections.Counter(r['gold'] for r in lost))}"
    )
    print(f"   gold over all marked pages {dict(collections.Counter(r['gold'] for r in mine))}")
for r in ok:
    if r["last_mark_removed"] and r["gold"] == "defective":
        print("DEFECTIVE GOLD PAGE RELEASED:", r["stem"], r["page"], r["classes"])
p28 = [
    r for r in ok if "tealbook-partb" in r["stem"] and "2018-05-02" in r["stem"] and r["page"] == 28
]
print("tealbook-partb 2018-05-02 p28:", p28)
