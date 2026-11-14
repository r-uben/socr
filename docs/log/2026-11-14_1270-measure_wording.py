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


def sidecars(run_root):
    """One sidecar per (document stem, page): a document retried into several run folders is
    counted once (the newest file wins, so the choice does not depend on glob order)."""
    newest = {}
    for path in glob.glob(f"{run_root}/*/out/*/*/pages/*.json"):
        stem, name = path.split("/")[-3], os.path.basename(path)
        key = (stem, name)
        if key not in newest or os.path.getmtime(path) > os.path.getmtime(newest[key]):
            newest[key] = path
    return [newest[k] for k in sorted(newest)]


def measure(run_root, fed_root, out_path):
    from socr.core.born_digital import BornDigitalDetector
    from socr.core.manifest import apply_verdict_marks
    from socr.core.result import FailureMode, PageOutput
    from socr.pipeline.refinalize_own_text import _page_state

    detector = BornDigitalDetector()
    rows, skipped = [], collections.Counter()
    for path in sidecars(run_root):
        try:
            side = json.load(open(path))
        except Exception:
            continue
        verdict = side.get("verdict")
        clauses = verdict.get("clauses", []) if isinstance(verdict, dict) else []
        if not any(map(cls_of, clauses)) or side.get("terminal") is not True:
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
        out = PageOutput.from_dict({"page_num": page, **side["winning_output"]})
        # The status is what THIS arm's seam computes from the page, never the sidecar's.
        final, new_verdict = apply_verdict_marks(out, p, page, event_kinds=kinds)
        after = [(c.check.value, c.result.value, c.scope, c.reason) for c in new_verdict.clauses]
        rows.append(
            dict(
                stem=stem,
                page=page,
                status=final.status.value,
                failure_mode=final.failure_mode.value,
                clauses=after,
            )
        )
    json.dump(dict(rows=rows, skipped=dict(skipped)), open(out_path, "w"), indent=1)
    print("measured", len(rows), "skipped", dict(skipped))


def compare(main_path, branch_path):
    a = {(r["stem"], r["page"]): r for r in json.load(open(main_path))["rows"]}
    b = {(r["stem"], r["page"]): r for r in json.load(open(branch_path))["rows"]}
    assert a.keys() == b.keys(), "the two runs measured different pages"
    key = lambda cs: sorted(c[:3] for c in cs)
    differs = [
        k
        for k in a
        if key(a[k]["clauses"]) != key(b[k]["clauses"])
        or (a[k]["status"], a[k]["failure_mode"]) != (b[k]["status"], b[k]["failure_mode"])
    ]
    marked, reworded = collections.Counter(), collections.Counter()
    for k in a:
        branch_by_key = collections.defaultdict(set)
        for c in b[k]["clauses"]:
            branch_by_key[tuple(c[:3])].add(c[3])
        for clause in a[k]["clauses"]:
            cls = cls_of(dict(check=clause[0], reason=clause[3]))
            if not cls:
                continue
            marked[cls] += 1
            # Clause level: this clause's reason is not among the branch's for the same
            # (check, result, scope).
            if clause[3] not in branch_by_key[tuple(clause[:3])]:
                reworded[cls] += 1
    print("pages measured", len(a), "class clauses on main", dict(marked))
    print("clauses reworded per class", dict(reworded))
    print("pages whose status, failure mode or (check, result, scope) set differs:", len(differs))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(*sys.argv[2:4])
    else:
        measure(*sys.argv[1:4])
