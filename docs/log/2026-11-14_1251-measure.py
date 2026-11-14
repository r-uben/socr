"""#1251 (C3) measurement on the cached 156-document run: which qwen SUCCESS pages keep SUCCESS
under the judge-is-not-a-check rule, and what the demotions cost on the auditors' gold pages.
No OCR, no model, no write.

    PYTHONPATH=src ~/venvs/socr/bin/python docs/log/2026-11-14_1251-measure.py run out.json [--words]
    PYTHONPATH=src ~/venvs/socr/bin/python docs/log/2026-11-14_1251-measure.py report out.json

Limitation, stated once: the cached sidecars carry no pass trace and no own-layer text. The
predicate is therefore recomputed the way ``socr-refinalize-own-text`` does it: socr's page
analysis of each PDF (current code) gives the layer inputs, and the CURRENT clause functions run
on the sidecar's ``winning_output``. A page's outcome is thus "what this code would label it",
which includes the effect of every check added since the run (#1185 numbers, #1294 words): the
output separates pages those older checks already demote from pages only this ticket demotes.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

from socr.benchmark import scoreboard as sb

PDFS = Path("/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs")
#: ``fed_rerun_51270eb5`` is the run the auditors graded (the gold rows join on its ``.md`` bytes);
#: ``fed_rerun_51270eb5_words`` is the same run re-finalized with #1294's words clause, the 12,413
#: sidecars the C3 brief counted (43 qwen SUCCESS pages). Pass ``--words`` for the second.
RUN = Path("/Volumes/Main/Library/socr-work/fed_rerun_51270eb5/unpacked")
RUN_WORDS = Path("/Volumes/Main/Library/socr-work/fed_rerun_51270eb5_words/unpacked")
RUN_ID = "51270eb5"
GOLD = Path("/Volumes/Main/Library/Databases/socr/gold/fed-2026-11-11.jsonl")


def _gold() -> dict:
    g: dict = collections.defaultdict(list)
    for line in GOLD.open():
        e = json.loads(line)
        if e["run"] == RUN_ID:
            g[(e["doc"], int(e["page"]), e["md_sha256"])].append(e)
    return g


def _job(args):
    doc, doc_dir = args
    from socr.core.born_digital import BornDigitalDetector
    from socr.core.result import PageOutput
    from socr.core.verdict import JUDGE_ONLY_REASON, page_word_check_record, silent_success_clauses
    from socr.pipeline.refinalize_own_text import _page_state

    gold = _gold()
    rows = []
    cand = []
    for sc in sb.page_sidecar_paths(doc_dir / "pages"):
        side = sb._read_json(sc)
        if side is None:
            continue
        out = PageOutput.from_dict(side.get("winning_output") or {})
        md = sc.with_suffix(".md")
        raw = md.read_bytes() if md.exists() else b""
        g = gold.get(
            (doc, int(side.get("page_num") or int(sc.stem)), hashlib.sha256(raw).hexdigest()), []
        )
        base = {
            "doc": doc,
            "page": int(side.get("page_num") or int(sc.stem)),
            "status": side.get("status"),
            "engine": side.get("engine"),
            "gold": sorted({e["content"] for e in g}),
            "defects": sorted({d for e in g for d in e.get("defect_types") or []}),
            "judge_accepted": out.judge_accepted,
            "judge_outcome": out.judge_outcome,
            "judge_reason_faithful": (out.judge_reason or "") == "faithful",
        }
        rows.append((sc, side, out, base))
        if side.get("status") == "success" and base["engine"] not in (
            "native",
            "chart_asset",
            "blank",
        ):
            cand.append(base["page"])
    pdf = PDFS / f"{doc}.pdf"
    analysis = None
    if cand and pdf.exists():
        try:
            analysis = BornDigitalDetector().detect(pdf)
        except Exception as exc:  # recorded, never skipped silently
            analysis = exc
    res = []
    for sc, side, out, base in rows:
        n = base["page"]
        r = dict(base)
        r["label_before"] = base["status"]
        r["after"] = None
        if base["status"] == "success" and base["engine"] not in ("native", "chart_asset", "blank"):
            if analysis is None or isinstance(analysis, Exception) or n - 1 >= len(analysis.pages):
                r["after"] = "unmeasured: no page analysis"
            else:
                p = _page_state(side, analysis.pages[n - 1])
                kinds = frozenset(
                    str(e.get("kind"))
                    for e in side.get("audit_events") or []
                    if isinstance(e, dict)
                )
                clauses = silent_success_clauses(out, p, kinds)
                words = page_word_check_record(out, p)
                if words is not None:
                    r["words_compared"] = words["compared"]
                if not clauses:
                    r["after"] = "success"
                elif all(c.reason == JUDGE_ONLY_REASON for c in clauses):
                    r["after"] = "demoted_by_c3"
                else:
                    r["after"] = "demoted_by_other_checks"
                    r["other"] = sorted({c.check.value for c in clauses})
        res.append(r)
    meta = sb._read_json(doc_dir / "metadata.json") or {}
    return doc, meta.get("status"), res


def main(out_path: str, words_run: bool = False, workers: int = 6) -> None:
    from concurrent.futures import ProcessPoolExecutor

    docs = sb.find_documents(RUN_WORDS if words_run else RUN)
    rows, doc_status = [], {}
    with ProcessPoolExecutor(workers) as pool:
        for doc, status, res in pool.map(_job, sorted(docs.items())):
            doc_status[doc] = status
            rows.extend(res)
    Path(out_path).write_text(json.dumps({"docs": doc_status, "pages": rows}))
    print(len(rows), "pages", len(doc_status), "documents")


def report(out_path: str) -> None:
    data = json.loads(Path(out_path).read_text())
    rows = data["pages"]
    qsucc = [r for r in rows if r["status"] == "success" and r["engine"] == "qwen"]
    print("qwen SUCCESS pages:", len(qsucc), collections.Counter(r["after"] for r in qsucc))
    print(
        "  judge accepted (typed):",
        sum(
            1
            for r in qsucc
            if r["judge_accepted"] is True and r["judge_outcome"] == "page_judge_completed"
        ),
    )
    gold_clean = [r for r in rows if r["gold"] == ["clean"]]
    print(
        "gold-clean pages present:",
        len(gold_clean),
        "-> D2 bound (1 in 100):",
        len(gold_clean) // 100,
    )
    for label in ("demoted_by_c3", "demoted_by_other_checks", "success"):
        sub = [r for r in qsucc if r["after"] == label]
        clean = [r for r in sub if r["gold"] == ["clean"]]
        defective = [r for r in sub if "defective" in r["gold"]]
        print(
            f"  {label}: {len(sub)} pages; gold-clean {len(clean)};"
            f" gold-defective {len(defective)};"
            f" no gold {sum(1 for r in sub if not r['gold'])}"
        )
    print(
        "  gold-defective among ALL qwen SUCCESS:",
        sum(1 for r in qsucc if "defective" in r["gold"]),
        "; marked by C3 only:",
        sum(1 for r in qsucc if "defective" in r["gold"] and r["after"] == "demoted_by_c3"),
        "; by other current checks:",
        sum(
            1 for r in qsucc if "defective" in r["gold"] and r["after"] == "demoted_by_other_checks"
        ),
        "; still SUCCESS:",
        sum(1 for r in qsucc if "defective" in r["gold"] and r["after"] == "success"),
    )
    by_doc: dict = collections.defaultdict(list)
    for r in rows:
        by_doc[r["doc"]].append(r)
    flips = {"c3_only": 0, "any_current": 0}
    for doc, rs in by_doc.items():
        if any(r["status"] != "success" for r in rs):
            continue  # already non-clean: no flip (C1 derives partial from any non-success page)
        if any(r["after"] == "demoted_by_c3" for r in rs):
            flips["c3_only"] += 1
        if any(str(r["after"]).startswith("demoted") for r in rs):
            flips["any_current"] += 1
    print("documents with every page SUCCESS now that a demotion would flip:", flips)
    print(
        "metadata.status of those documents:",
        collections.Counter(
            data["docs"][d]
            for d, rs in by_doc.items()
            if all(r["status"] == "success" for r in rs)
            and any(r["after"] == "demoted_by_c3" for r in rs)
        ),
    )
    nonfaithful = [r for r in qsucc if not r["judge_reason_faithful"]]
    print("qwen SUCCESS with a reason other than 'faithful':", len(nonfaithful))
    compared = sorted(r["words_compared"] for r in qsucc if "words_compared" in r)
    if compared:
        deciles = [compared[min(len(compared) - 1, len(compared) * k // 10)] for k in range(11)]
        print("words compared per qwen SUCCESS page (n, deciles 0..10):", len(compared), deciles)
        weak = sb.WEAK_WORDS_PASS_BELOW
        print(
            f"  below WEAK_WORDS_PASS_BELOW={weak}:",
            sum(1 for n in compared if n < weak),
            "; gold-defective among them:",
            sum(
                1
                for r in qsucc
                if "defective" in r["gold"] and r.get("words_compared", weak) < weak
            ),
        )


def _cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="measure the cached run and write the rows to OUT")
    run.add_argument("out")
    run.add_argument("--words", action="store_true", help="the words-refinalized run")
    rep = sub.add_parser("report", help="print the report of rows written by `run`")
    rep.add_argument("out")
    args = parser.parse_args(argv)
    if args.command == "report":
        report(args.out)
    else:
        main(args.out, words_run=args.words)


if __name__ == "__main__":
    _cli(sys.argv[1:])
