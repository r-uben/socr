"""GH-1291: the audit scoreboard — measure any socr run folder with one fixed set of counts.

The 2026-11-11 independent audit of the Fed run judged socr by counts that lived in
four auditors' scripts. This tool computes those counts from the shipped files alone,
so a fix can show that its target count moved and that no other count got worse.

A run folder is anything holding ``<doc>/pages/NNNNN.{md,json}`` document folders at
any depth (a Slurm task tree ``<task>/out/w<N>/<doc>/`` or a single output folder).

    uv run socr-scoreboard RUN_OLD RUN_NEW --pdf-dir PDFS [--docs FILE] [--json OUT]

With two runs, only the documents both runs hold are counted (or ``--docs``), and the
table shows old, new and the difference. ``--pdf-dir`` adds the text-layer group, which
opens every page of every PDF and is the slow part.

Every count is a number of pages unless its name says otherwise. The groups:
``coverage``, ``labels``, ``tells`` (silent defects visible in the shipped bytes of a
SUCCESS page), ``events`` (pages per audit-event kind), ``textlayer`` and ``records``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

# In-text marks a reader can see: a ``> [`` line, a failed-page stub, a withheld number.
_MARK_RE = re.compile(r"^> \[|\[page \d+ failed|\[number withheld", re.M)
# Control bytes other than tab/newline/carriage return, and the Unicode private-use area.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f-]")
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
# A mixed number rewritten as a power: 2¾ shipped as 2^{3/4}.
_POWER_FRACTION_RE = re.compile(r"\b\d+\^\{\d+/\d+\}")
# Text-layer comparison tokens (the definitions fable's v1 diff used).
_NUM_RE = re.compile(r"(?<![\w.])[-−–]?(?:\d[\d,]*\.?\d*|\.\d+)")
_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_BANNER_RE = re.compile(r"^> \[.*$", re.M)
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)

GROUPS = ("coverage", "labels", "tells", "events", "textlayer", "records", "gold")


def find_documents(run: Path) -> dict[str, Path]:
    """Map document name to its folder for every ``<doc>/pages/`` under ``run``.

    A document found twice keeps the folder with more page sidecars; ties keep the one
    that has ``metadata.json``.
    """
    found: dict[str, Path] = {}
    for pages in sorted(run.rglob("pages")):
        if not pages.is_dir():
            continue
        doc_dir = pages.parent
        name = doc_dir.name
        prev = found.get(name)
        if prev is None or _doc_rank(doc_dir) > _doc_rank(prev):
            found[name] = doc_dir
    return found


def _doc_rank(doc_dir: Path) -> tuple[int, int]:
    return (
        len(list((doc_dir / "pages").glob("*.json"))),
        int((doc_dir / "metadata.json").exists()),
    )


def _is_marked(text: str) -> bool:
    return bool(_MARK_RE.search(text))


def _clauses(sidecar: dict) -> list[dict]:
    verdict = sidecar.get("verdict") or {}
    return [c for c in verdict.get("clauses") or [] if isinstance(c, dict)]


def _missing_links(text: str, doc_dir: Path) -> int:
    missing = 0
    for target in _LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "#")):
            continue
        if not target.lower().endswith(_IMAGE_EXT):
            continue
        if not (doc_dir / target).exists() and not (doc_dir / "pages" / target).exists():
            missing += 1
    return missing


def _has_existing_image(text: str, doc_dir: Path) -> bool:
    for target in _LINK_RE.findall(text):
        if target.lower().endswith(_IMAGE_EXT) and (
            (doc_dir / target).exists() or (doc_dir / "pages" / target).exists()
        ):
            return True
    return False


def count_document(doc_dir: Path) -> collections.Counter:
    """All file-only counts for one document folder."""
    c: collections.Counter = collections.Counter()
    c["coverage.documents"] += 1
    status_by_page: dict[int, str] = {}
    page_statuses: collections.Counter = collections.Counter()
    for sc_path in sorted((doc_dir / "pages").glob("*.json")):
        sidecar = json.loads(sc_path.read_text())
        md_path = sc_path.with_suffix(".md")
        text = md_path.read_text(errors="replace") if md_path.exists() else ""
        page = int(sidecar.get("page_num") or int(sc_path.stem))
        status = str(sidecar.get("status") or "missing")
        engine = str(sidecar.get("engine") or "missing")
        passed = sidecar.get("audit_passed")
        mode = str(sidecar.get("failure_mode") or "none")
        clauses = _clauses(sidecar)
        events = {
            str(e.get("kind")) for e in sidecar.get("audit_events") or [] if isinstance(e, dict)
        }
        marked = _is_marked(text)
        status_by_page[page] = status
        page_statuses[status] += 1

        c["coverage.pages"] += 1
        if not md_path.exists():
            c["coverage.pages_without_md"] += 1
        if sidecar.get("terminal") is False:
            c["coverage.pages_not_terminal"] += 1

        c[f"labels.status.{status}"] += 1
        c[f"labels.status_engine.{status}|{engine}"] += 1
        if status != "success" and not marked:
            c["labels.non_success_unmarked"] += 1
        if status == "success" and marked:
            c["labels.success_marked"] += 1
        if status == "warning" and passed is True:
            c["labels.warning_audit_passed_true"] += 1
            if any(str(cl.get("result")).upper() == "FAIL" for cl in clauses):
                c["labels.warning_audit_passed_true_with_fail_clause"] += 1
        if status == "warning" and mode in ("none", ""):
            c["labels.warning_failure_mode_none"] += 1
        if "text_layer_numbers=FAIL" in text and passed is True:
            c["labels.text_layer_numbers_fail_audit_passed_true"] += 1
        if status == "error" and mode == "hallucination":
            c["labels.error_hallucination"] += 1

        if status == "success":
            if _CTRL_RE.search(text):
                c["tells.success_control_bytes"] += 1
            if "<!-- Image" in text:
                c["tells.success_image_comment"] += 1
            if "\\begin{tabular}" in text:
                c["tells.success_raw_latex_tabular"] += 1
        if _missing_links(text, doc_dir):
            c["tells.pages_with_image_link_to_missing_file"] += 1
        if "see image" in text and not _has_existing_image(text, doc_dir):
            c["tells.see_image_without_image"] += 1
        if engine == "native+math":
            c["tells.native_math_pages"] += 1
            if _POWER_FRACTION_RE.search(text):
                c["tells.native_math_power_fraction"] += 1

        for kind in events:
            c[f"events.{kind}"] += 1

        attempts = sidecar.get("attempts_summary") or sidecar.get("attempts") or []
        if len(attempts) > 1:
            c["records.pages_more_than_one_attempt"] += 1
        if status == "success" and engine == "native" and not events and not clauses:
            c["records.clean_native_success_no_trace"] += 1
        if any(str(cl.get("check")) == "unknown" for cl in clauses):
            c["records.pages_clause_check_unknown"] += 1
        digest = str(sidecar.get("socr_source_digest") or "missing")
        c[f"records.source_digest.{digest[:12]}"] += 1

    meta_path = doc_dir / "metadata.json"
    if not meta_path.exists():
        c["coverage.documents_unfinished"] += 1
    else:
        meta = json.loads(meta_path.read_text())
        doc_status = str(meta.get("status") or "missing")
        c[f"coverage.document_status.{doc_status}"] += 1
        if doc_status == "completed" and (page_statuses["warning"] or page_statuses["error"]):
            c["labels.completed_documents_with_non_success_pages"] += 1
        if doc_status == "completed" and page_statuses["error"]:
            c["labels.completed_documents_with_error_pages"] += 1
        if not meta.get("processing_time"):
            c["records.documents_processing_time_zero"] += 1
        if not any(k in meta for k in ("commit", "git_sha", "socr_commit", "socr_source_digest")):
            c["records.documents_without_code_version"] += 1
    if not ((doc_dir / "manifest.json").exists() and (doc_dir / "audit_log.json").exists()):
        c["coverage.documents_torn"] += 1

    trust_path = doc_dir / "tables_trust.json"
    if trust_path.exists():
        trust = json.loads(trust_path.read_text())
        pages = trust.get("pages") if isinstance(trust, dict) else None
        for key, entry in (pages or {}).items():
            if isinstance(entry, dict) and entry.get("flagged"):
                c["labels.tables_trust_flagged"] += 1
                if status_by_page.get(int(key)) == "success":
                    c["labels.tables_trust_flagged_on_success"] += 1
    return c


def _norm_number(token: str) -> str:
    token = token.replace("−", "-").replace("–", "-").replace(",", "")
    return token.lstrip("-").rstrip(".")


def _numbers(text: str) -> collections.Counter:
    return collections.Counter(
        _norm_number(t) for t in _NUM_RE.findall(text) if any(ch.isdigit() for ch in t)
    )


def _words(text: str) -> collections.Counter:
    return collections.Counter(w.lower() for w in _WORD_RE.findall(text))


def _shipped_for_comparison(text: str) -> str:
    text = _BANNER_RE.sub("", text)
    text = _IMAGE_MD_RE.sub("", text)
    return _COMMENT_RE.sub("", text)


def count_text_layer(args: tuple[Path, Path]) -> collections.Counter:
    """Text-layer words and numbers the shipped pages of one document do not carry.

    Multiset differences per page; a lead count, not a verdict (a damaged text layer
    or a correctly withheld region also shows up here).
    """
    from socr.core.pdf import open_pdf  # noqa: PLC0415 — only this group needs PyMuPDF

    doc_dir, pdf_path = args
    c: collections.Counter = collections.Counter()
    if not pdf_path.exists():
        c["textlayer.documents_without_pdf"] += 1
        return c
    with open_pdf(pdf_path) as pdf:
        for sc_path in sorted((doc_dir / "pages").glob("*.json")):
            md_path = sc_path.with_suffix(".md")
            if not md_path.exists():
                continue
            sidecar = json.loads(sc_path.read_text())
            status = str(sidecar.get("status") or "missing")
            page = int(sidecar.get("page_num") or int(sc_path.stem))
            if page - 1 >= len(pdf):
                c["textlayer.pages_beyond_pdf"] += 1
                continue
            layer = pdf[page - 1].get_text()
            shipped = _shipped_for_comparison(md_path.read_text(errors="replace"))
            lw, sw = _words(layer), _words(shipped)
            ln, sn = _numbers(layer), _numbers(shipped)
            missing_w = sum((lw - sw).values())
            missing_n = ln - sn
            extra_n = sn - ln
            c["textlayer.words"] += sum(lw.values())
            c["textlayer.words_missing"] += missing_w
            c[f"textlayer.words_missing.{status}"] += missing_w
            c["textlayer.numbers"] += sum(ln.values())
            c["textlayer.numbers_missing"] += sum(missing_n.values())
            c[f"textlayer.numbers_missing.{status}"] += sum(missing_n.values())
            if missing_w or missing_n:
                c["textlayer.pages_with_anything_missing"] += 1
            dropped = 0
            for token, n in missing_n.items():
                if token.startswith(".") and token[1:]:
                    dropped += min(n, extra_n.get(token[1:], 0))
            if dropped:
                c["textlayer.pages_dropped_decimal_point"] += 1
                c["textlayer.dropped_decimal_point_values"] += dropped
    return c


def load_gold(path: Path) -> dict[tuple[str, int], list[dict]]:
    """Gold verdicts keyed by (doc, page); one entry per seat and judged page version."""
    gold: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            gold[(entry["doc"], int(entry["page"]))].append(entry)
    return gold


def score_gold(
    found: dict[str, Path], gold: dict[tuple[str, int], list[dict]]
) -> tuple[collections.Counter, list[dict]]:
    """Carry each gold verdict to this run where the page ships byte-identical markdown.

    A page whose bytes match no judged version goes to the review queue: its verdict is
    unknown until someone reads it against the PDF. ``defective_silent`` is the count that
    matters most: a page judged wrong that ships as unmarked SUCCESS.
    """
    c: collections.Counter = collections.Counter()
    queue: list[dict] = []
    for (doc, page), entries in sorted(gold.items()):
        doc_dir = found.get(doc)
        md_path = doc_dir / "pages" / f"{page:05d}.md" if doc_dir else None
        if md_path is None or not md_path.exists():
            c["gold.pages_not_in_run"] += 1
            continue
        c["gold.pages"] += 1
        data = md_path.read_bytes()
        sidecar = json.loads(md_path.with_suffix(".json").read_text())
        status = str(sidecar.get("status") or "missing")
        silent_label = status == "success" and not _is_marked(data.decode(errors="replace"))
        digest = hashlib.sha256(data).hexdigest()
        contents = {e["content"] for e in entries if e["md_sha256"] == digest}
        if not contents:
            c["gold.changed_needs_review"] += 1
            c[f"gold.changed_needs_review.{'unmarked_success' if silent_label else 'marked'}"] += 1
            queue.append(
                {"doc": doc, "page": page, "status": status, "engine": sidecar.get("engine")}
            )
            continue
        if "defective" in contents and "clean" in contents:
            c["gold.carried.seats_disagree"] += 1
        if "defective" in contents:
            c[
                "gold.carried.defective_silent" if silent_label else "gold.carried.defective_marked"
            ] += 1
        elif "withheld_or_marked" in contents:
            c["gold.carried.withheld_or_marked"] += 1
        else:
            c["gold.carried.clean"] += 1
    return c, queue


def score_run(
    run: Path,
    docs: set[str] | None,
    pdf_dir: Path | None,
    workers: int,
    gold: dict[tuple[str, int], list[dict]] | None = None,
    queue_out: Path | None = None,
) -> collections.Counter:
    found = find_documents(run)
    if docs is not None:
        absent = sorted(docs - found.keys())
        found = {d: p for d, p in found.items() if d in docs}
    else:
        absent = []
    total: collections.Counter = collections.Counter()
    total["coverage.documents_requested_absent"] = len(absent)
    # Reading sidecars is I/O-bound (run folders often sit on an external or network drive).
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(count_document, found.values()):
            total.update(part)
    digests = [k for k in total if k.startswith("records.source_digest.")]
    total["records.distinct_source_digests"] = len(digests)
    if pdf_dir is not None:
        jobs = [(p, pdf_dir / f"{d}.pdf") for d, p in sorted(found.items())]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(count_text_layer, jobs, chunksize=4):
                total.update(part)
    if gold is not None:
        part, queue = score_gold(found, gold)
        total.update(part)
        if queue_out is not None:
            queue_out.write_text("".join(json.dumps(q) + "\n" for q in queue))
    return total


def _group_order(key: str) -> tuple[int, str]:
    group = key.split(".", 1)[0]
    return (GROUPS.index(group) if group in GROUPS else len(GROUPS), key)


def render(results: list[collections.Counter], labels: list[str]) -> str:
    keys = sorted({k for r in results for k in r}, key=_group_order)
    width = max(len(k) for k in keys) if keys else 10
    head = f"{'count':<{width}}  " + "  ".join(f"{lab:>12}" for lab in labels)
    if len(results) == 2:
        head += f"  {'delta':>10}"
    lines = [head, "-" * len(head)]
    group = None
    for key in keys:
        g = key.split(".", 1)[0]
        if g != group:
            if group is not None:
                lines.append("")
            group = g
        vals = [r.get(key, 0) for r in results]
        row = f"{key:<{width}}  " + "  ".join(f"{v:>12,}" for v in vals)
        if len(results) == 2:
            d = vals[1] - vals[0]
            row += f"  {d:>+10,}" if d else f"  {'':>10}"
        lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="socr-scoreboard", description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="one or two run folders")
    ap.add_argument(
        "--pdf-dir", type=Path, help="source PDFs named <doc>.pdf; adds the text-layer group"
    )
    ap.add_argument(
        "--docs", type=Path, help="file with one document name per line; count only these"
    )
    ap.add_argument("--json", type=Path, help="write every count as JSON here")
    ap.add_argument("--gold", type=Path, help="gold-page verdicts (JSONL); adds the gold group")
    ap.add_argument(
        "--gold-queue",
        type=Path,
        help="write each run's changed gold pages (needing a fresh read) to <this>.runN.jsonl",
    )
    ap.add_argument("--workers", type=int, default=8, help="processes for the text-layer group")
    ns = ap.parse_args(argv)
    if len(ns.runs) > 2:
        ap.error("give one or two run folders")

    docs: set[str] | None = None
    if ns.docs:
        docs = {ln.strip() for ln in ns.docs.read_text().splitlines() if ln.strip()}
    elif len(ns.runs) == 2:
        docs = set(find_documents(ns.runs[0])) & set(find_documents(ns.runs[1]))

    gold = load_gold(ns.gold) if ns.gold else None
    results = [
        score_run(
            r,
            docs,
            ns.pdf_dir,
            ns.workers,
            gold,
            ns.gold_queue.with_suffix(f".run{i + 1}.jsonl") if ns.gold_queue else None,
        )
        for i, r in enumerate(ns.runs)
    ]
    labels = [f"run{i + 1}" for i in range(len(ns.runs))]
    for lab, run in zip(labels, ns.runs):
        print(f"{lab}: {run}")
    print(render(results, labels))
    if ns.json:
        ns.json.write_text(
            json.dumps(
                {str(r): dict(sorted(c.items())) for r, c in zip(ns.runs, results)}, indent=1
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
