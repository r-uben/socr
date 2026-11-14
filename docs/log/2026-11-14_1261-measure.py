"""#1261 measurement on the cached 156-doc run. No OCR: reads sidecars and the PDFs only.

For every ERROR page whose disposition is ``fail_closed_marker`` it rebuilds the page the way
the pipeline holds it at the floor (``BornDigitalDetector.detect_page`` -> ``apply_born_digital``,
the detector count and boxes of the run's own sidecar, ``native_words`` from the PDF) and runs the
born-digital geometry floor TWICE in one process, changing only whether the native-order fallback
runs after the reading-order gate refuses (``_native_order_prose_body`` patched to None for the "old" arm). That is
the pinned DIFFERENCE, not a value measured on one machine. Each arm records the reason the floor
refused, so the 682 complete-box refusals are traced before any gate is relaxed.

Outputs (JSON lines, one per replayed page) and a summary:
  * refusal reasons of the old arm, split by the sidecar's box-inventory class;
  * ``textlayer.words_missing`` on ERROR pages, old and new (the scoreboard's own definition),
    as the run's shipped total minus the per-page change on the replayed pages;
  * the row-level leak diff: every value-bearing line (a line with a digit) the new arm ships that
    the old arm did not, with where its words sit relative to the detected boxes.

Run from the repo root::

    PYTHONPATH=src ~/venvs/socr/bin/python docs/log/2026-11-14_1261-measure.py

Paths are the constants below; nothing is read from the command line.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

#: The cached run: one directory per worker, ``out/w*/<document>/`` inside each.
RUN = "/Volumes/Main/Library/socr-work/fed_rerun_51270eb5_words/unpacked"
#: The source PDFs of that run, one ``<document>.pdf`` each.
PDFS = "/Volumes/Main/Library/Databases/socr/inputs/fed/pdfs"
#: Where the measurement writes.
OUT_DIR = Path("/Users/rubenffuertes/repos/tools/socr/scratch/army/phaseBC")
OUT = OUT_DIR / "b2-replay.jsonl"
SUMMARY = OUT_DIR / "b2-summary.json"
#: Worker processes.
WORKERS = 8
#: The old run's level the done bar names (``textlayer.words_missing.error``).
DONE_BAR_WORDS = 94_011


def _sidecars():
    for sc in sorted(glob.glob(os.path.join(RUN, "*", "out", "w*", "*", "pages", "*.json"))):
        yield sc


def _inventory_class(j):
    """The sidecar's own box-inventory class (the brief's split)."""
    count = j.get("detected_table_count") or 0
    boxes = j.get("detected_table_bboxes") or []
    if count == 0 and not boxes:
        return "no_detection"
    if len(boxes) < count:
        return "partial_boxes" if boxes else "count_without_boxes"
    if len(boxes) > count:
        return "more_boxes_than_count"
    return "complete_boxes"


def _value_lines(text):
    """Lines of the shipped body that carry a digit, banner/marker/image lines excluded."""
    from socr.benchmark.scoreboard import _shipped_for_comparison

    out = []
    for line in _shipped_for_comparison(text).splitlines():
        s = line.strip()
        if not s or s.startswith("[page "):
            continue
        if any(ch.isdigit() for ch in s):
            out.append(s)
    return out


def _line_geometry(words, boxes, line):
    """Where the words of *line* sit: tokens found in the page words, inside a box or not."""
    toks = [t for t in line.replace("\\", "").split() if t]
    inside = 0
    found = 0
    for t in toks:
        for w in words:
            if w[4] == t:
                found += 1
                cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
                if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes):
                    inside += 1
                break
    return {"tokens": len(toks), "found": found, "inside_box": inside}


def _canonical_dirs():
    """One document directory per stem. A retried document leaves a second folder with the
    same stem (``w3/<stem>`` and ``w7/<stem>``); both replayed would double the page and
    recovered-word totals and give indistinguishable ``(doc, page)`` rows. The folder with the
    most page sidecars wins (ties: the last by path); the rest are counted, not replayed."""
    by_stem: dict[str, list[str]] = {}
    for sc in _sidecars():
        d = os.path.dirname(os.path.dirname(sc))
        if d not in by_stem.setdefault(os.path.basename(d), []):
            by_stem[os.path.basename(d)].append(d)
    chosen, duplicates = [], 0
    for _stem, dirs in sorted(by_stem.items()):
        dirs.sort(key=lambda d: (len(glob.glob(os.path.join(d, "pages", "*.json"))), d))
        chosen.append(dirs[-1])
        duplicates += len(dirs) - 1
    return chosen, duplicates


def work(doc_dir):
    import fitz

    from socr.core.pdf import open_pdf
    from socr.benchmark.scoreboard import _shipped_for_comparison, _words
    from socr.core import manifest
    from socr.core.born_digital import BornDigitalDetector
    from socr.core.state import DocumentState, PageState

    stem = os.path.basename(doc_dir.rstrip("/"))
    pdf = os.path.join(PDFS, stem + ".pdf")
    rows, stats = [], Counter()
    det = BornDigitalDetector()
    doc = None
    native_doc = None
    if os.path.exists(pdf):
        # ``doc`` (raw) is the scoreboard's text-layer count; ``native_doc`` is what
        # production reads ``native_words`` from (``open_pdf``: symbol-font glyph recovery).
        try:
            doc = fitz.open(pdf)
            native_doc = open_pdf(pdf)
        except Exception:  # noqa: BLE001 - a corrupt PDF costs its pages, not the run
            doc = native_doc = None
            stats["pdf_open_error"] += 1
    for sc in sorted(glob.glob(os.path.join(doc_dir, "pages", "*.json"))):
        j = json.load(open(sc))
        if j.get("status") != "error":
            continue
        pn = int(j["page_num"])
        md_path = sc[:-5] + ".md"
        md = Path(md_path).read_text(errors="replace") if os.path.exists(md_path) else ""
        try:
            layer_words = _words(doc[pn - 1].get_text()) if doc is not None else Counter()
        except Exception as exc:  # noqa: BLE001 - recorded per page below
            stats["error_pages_unreadable"] += 1
            if (j.get("disposition") or {}).get("ending") == "fail_closed_marker":
                rows.append({"doc": stem, "page": pn, "replay_error": f"layer read: {exc!r}"})
            continue
        shipped_missing = sum((layer_words - _words(_shipped_for_comparison(md))).values())
        stats["error_pages"] += 1
        stats["error_words_missing_shipped"] += shipped_missing
        if (j.get("disposition") or {}).get("ending") != "fail_closed_marker":
            continue
        stats["floor_pages"] += 1
        if doc is None:
            stats["no_pdf"] += 1
            rows.append({"doc": stem, "page": pn, "replay_error": "no readable PDF"})
            continue
        try:
            pa = det.detect_page(pdf, pn)
            holder = type("H", (), {})()
            holder.pages = {pn: PageState(page_num=pn)}
            holder.native_vocabulary = None
            DocumentState.apply_born_digital(holder, type("A", (), {"pages": [pa]})())
            p = holder.pages[pn]
        except Exception as exc:  # noqa: BLE001 - a measurement records, never hides
            stats["replay_error"] += 1
            rows.append({"doc": stem, "page": pn, "replay_error": repr(exc)})
            continue
        if not p.is_born_digital:
            stats["not_born_digital"] += 1
            continue
        # The run's own detector evidence is the truth the floor saw.
        p.detected_table_count = int(j.get("detected_table_count") or 0)
        p.detected_table_bboxes = [tuple(b) for b in (j.get("detected_table_bboxes") or [])]
        p.d3_floor_png_ref = j.get("d3_floor_png_ref") or ""
        try:
            words = native_doc[pn - 1].get_text("words")
        except Exception as exc:  # noqa: BLE001 - a measurement records, never hides
            stats["replay_error"] += 1
            rows.append({"doc": stem, "page": pn, "replay_error": f"native words: {exc!r}"})
            continue
        p.native_words = words
        stats["replayed"] += 1

        def arm(patch_partial):
            original = manifest._native_order_prose_body
            if not patch_partial:
                manifest._native_order_prose_body = lambda *_a, **_k: None
            refusals: list[str] = []
            try:
                if manifest.prior_ocr_layer_found(p):
                    refusals.append("prior_ocr")
                    return (
                        manifest.table_floor_text_for_source(
                            p, pn, "", prose_recovery=False, refusals=[]
                        ),
                        refusals,
                    )
                text = manifest.table_floor_text_for_source(
                    p,
                    pn,
                    p.native_text or "",
                    prose_recovery=True,
                    geometry_floor=True,
                    chart_floor_offered=False,
                    refusals=refusals,
                )
                return text, refusals
            finally:
                manifest._native_order_prose_body = original

        old_text, old_ref = arm(False)
        new_text, new_ref = arm(True)
        shipped_floor = (j.get("winning_output") or {}).get("text", "") or ""
        cls = _inventory_class(j)
        miss_old = sum((layer_words - _words(_shipped_for_comparison(old_text))).values())
        miss_new = sum((layer_words - _words(_shipped_for_comparison(new_text))).values())
        # Multiset, not set: a value line the new arm ships twice (or once more than the old
        # arm) is a new line each time.
        old_vals = Counter(_value_lines(old_text))
        new_vals = []
        for ln in _value_lines(new_text):
            if old_vals[ln]:
                old_vals[ln] -= 1
            else:
                new_vals.append(ln)
        boxes = [tuple(b) for b in (j.get("detected_table_bboxes") or [])]
        rows.append(
            {
                "doc": stem,
                "page": pn,
                "class": cls,
                "count": p.detected_table_count,
                "boxes": len(boxes),
                "reproduces_run": old_text == shipped_floor,
                "old_reason": old_ref[-1] if old_ref else "",
                "new_reason": new_ref[-1] if new_ref else "",
                "changed": old_text != new_text,
                "words_missing_old": miss_old,
                "words_missing_new": miss_new,
                "new_value_lines": [
                    {"line": ln, **_line_geometry(words, boxes, ln)} for ln in new_vals
                ],
                "new_text": new_text if old_text != new_text else "",
            }
        )
    if doc is not None:
        doc.close()
    if native_doc is not None:
        native_doc.close()
    return rows, stats


def main():
    docs, duplicates = _canonical_dirs()
    rows, stats = [], Counter()
    stats["duplicate_doc_dirs_skipped"] = duplicates
    with Pool(WORKERS) as pool:
        for r, s in pool.imap_unordered(work, docs):
            rows.extend(r)
            stats.update(s)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        for r in sorted(rows, key=lambda r: (r["doc"], r["page"])):
            fh.write(json.dumps(r) + "\n")
    replayed = [r for r in rows if "replay_error" not in r]
    # Totals are over pages whose OLD arm reproduces the run's own floor bytes: a page whose old
    # arm differs from the cached run is not the pinned old-versus-new change. They are counted
    # and reported separately, never added in.
    ok = [r for r in replayed if r["reproduces_run"]]
    unreproduced = [r for r in replayed if not r["reproduces_run"]]
    by_class = Counter(r["class"] for r in ok)
    old_reasons = Counter((r["class"], r["old_reason"] or "RECOVERED") for r in ok)
    new_reasons = Counter((r["class"], r["new_reason"] or "RECOVERED") for r in ok)
    delta = sum(r["words_missing_old"] - r["words_missing_new"] for r in ok)
    half = [
        (r["doc"], r["page"], v["line"])
        for r in ok
        for v in r["new_value_lines"]
        if 2 * sum(any(ch.isdigit() for ch in t) for t in v["line"].split())
        >= len(v["line"].split())
    ]
    summary = {
        # Lines the new arm ships that the old arm did not and whose tokens are at least half
        # numerals: the candidate loose table values. Listed in full for hand review.
        "new_lines_half_numeric": half,
        "stats": dict(stats),
        "replayed": len(replayed),
        "replay_reproduces_run": len(ok),
        "replay_errors": len(rows) - len(replayed),
        "non_reproducing_rows_excluded": len(unreproduced),
        "non_reproducing_pages_changed": sum(r["changed"] for r in unreproduced),
        "by_class": dict(by_class),
        "old_reasons": {f"{k[0]}:{k[1]}": v for k, v in sorted(old_reasons.items())},
        "new_reasons": {f"{k[0]}:{k[1]}": v for k, v in sorted(new_reasons.items())},
        "pages_changed": sum(r["changed"] for r in ok),
        "words_recovered": delta,
        "words_missing_error_shipped": stats["error_words_missing_shipped"],
        "words_missing_error_new": stats["error_words_missing_shipped"] - delta,
        "done_bar_words": DONE_BAR_WORDS,
        "new_value_lines": sum(len(r["new_value_lines"]) for r in ok),
        "pages_with_new_value_lines": sum(bool(r["new_value_lines"]) for r in ok),
    }
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
