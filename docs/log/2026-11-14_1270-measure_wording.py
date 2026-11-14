"""#1270 B4 classes 1 and 3: which marks are reworded, and does any page change?

Reads the sidecars of run 51270eb5 (no OCR) and the source PDFs of its FOMC-meeting documents
(the only ones whose path follows from the stem). For every page that carried a class 1
(``native_glyphs`` "unrecovered mathematics") or class 3 ("agrees ... confirms nothing") mark, it
runs ``derive_clauses`` on the shipped text with a page state rebuilt from the detector and writes
the clauses. Run it twice, once with PYTHONPATH on ``main``'s src and once on this branch's, and
compare with ``--compare``: a page whose (check, result, scope) set differs would change status.

    measure_wording <run_root> <fed_root> <out.json>
    measure_wording --compare <main.json> <branch.json>
"""

import collections
import glob
import json
import os
import re
import sys
from types import SimpleNamespace

PUA_REASON = "private-use codepoints in this page's native text layer are unrecovered mathematics"
AGREES_PREFIX = "the reading's numbers agree with this page's text layer, but"


def cls_of(clause):
    if clause["check"] == "native_glyphs" and clause["reason"] == PUA_REASON:
        return 1
    if clause["check"] == "text_layer_numbers" and clause["reason"].startswith(AGREES_PREFIX):
        return 3
    return None


def pdf_of(fed_root, stem):
    m = re.fullmatch(r"fed-meetings-(\d{4})-(\d{4}-\d{2})-(\d{4}-\d{2}-\d{2})-(.+)", stem)
    if not m:
        return None
    year, month, day, doc = m.groups()
    for name in (doc, doc.replace("-", "_")):
        path = os.path.join(fed_root, "meetings", year, month, day, "raw", name + ".pdf")
        if os.path.exists(path):
            return path
    return None


def measure(run_root, fed_root, out_path):
    from socr.core.born_digital import BornDigitalDetector
    from socr.core.result import FailureMode, PageOutput, PageStatus
    from socr.core.verdict import derive_clauses
    from socr.pipeline.refinalize_own_text import _page_state

    detector = BornDigitalDetector()
    rows, skipped = [], collections.Counter()
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
        stem, page = path.split("/")[-3], int(side["page_num"])
        pdf = pdf_of(fed_root, stem)
        if pdf is None:
            skipped["no pdf path from stem"] += 1
            continue
        try:
            pa = detector.detect_page(pdf, page)
        except Exception:
            skipped["detector error"] += 1
            continue
        p = _page_state(side, pa)
        p.has_unmapped_math_glyphs = bool(pa.has_unmapped_math_glyphs)
        p.private_use_bullets_only = bool(getattr(pa, "private_use_bullets_only", False))
        disposition = side.get("table_ladder_disposition")
        p.table_ladder_disposition = FailureMode(disposition) if disposition else None
        kinds = frozenset(e["kind"] for e in side.get("audit_events") or [])
        wo = side["winning_output"]
        out = PageOutput(
            page_num=page,
            text=wo["text"],
            status=PageStatus(side["status"]),
            engine=side.get("engine") or "qwen",
            failure_mode=FailureMode(side.get("failure_mode") or "none"),
        )
        after = [
            (c.check.value, c.result.value, c.scope, c.reason)
            for c in derive_clauses(out, p, None, kinds)
        ]
        rows.append(dict(stem=stem, page=page, classes=classes, clauses=after))
    json.dump(dict(rows=rows, skipped=dict(skipped)), open(out_path, "w"), indent=1)
    print("measured", len(rows), "skipped", dict(skipped))


def compare(main_path, branch_path):
    a = {(r["stem"], r["page"]): r for r in json.load(open(main_path))["rows"]}
    b = {(r["stem"], r["page"]): r for r in json.load(open(branch_path))["rows"]}
    assert a.keys() == b.keys(), "the two runs measured different pages"
    key = lambda cs: sorted(c[:3] for c in cs)
    changed = [k for k in a if key(a[k]["clauses"]) != key(b[k]["clauses"])]
    reworded = collections.Counter()
    for k in a:
        for cls in a[k]["classes"]:
            diff = {tuple(c) for c in b[k]["clauses"]} - {tuple(c) for c in a[k]["clauses"]}
            reworded[cls] += 1 if diff else 0
    per_class = collections.Counter(c for r in a.values() for c in r["classes"])
    print("pages measured", len(a), "marked per class", dict(per_class))
    print("marks reworded per class", dict(reworded))
    print("pages whose (check, result, scope) set differs from main:", len(changed))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(*sys.argv[2:4])
    else:
        measure(*sys.argv[1:4])
