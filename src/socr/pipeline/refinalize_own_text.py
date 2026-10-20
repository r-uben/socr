"""#1185: check a FINISHED run's model readings against their pages' own text layers.

    socr-refinalize-own-text <out_dir> --pdf-dir <dir> [--dry-run]

A run made before #1185 counted a model reading CHECKED when the page judge accepted it,
whatever its numbers: 2018 Tealbook p125 shipped ``-8`` for a printed ``-.8`` as
"faithful". This tool applies the ``text_layer_numbers`` verdict clause
(``core.verdict.text_layer_numbers_clause``) to every model-read page of a finished run,
with NO new OCR and no model: the page's own characters come from socr's page analysis
of the run's input PDF (``BornDigitalDetector``, CPU only), accepted only for the very file
the run read (same checksum, same page count).

A page the clause speaks about is finalized through the SAME seam a run uses
(``core.manifest.apply_verdict_marks``) with its published verdict extended by the new
clause, so its mark, status and failure mode are what a run of this code writes. Then:

* ``pages/NNN.md`` and the page's section of the document ``.md`` (anchored on its
  ``## Page N`` heading, asserted to occur exactly once);
* the page sidecar's record-derived keys (status, failure mode, verdict, winning output,
  page fingerprint, socr version); everything else as the run wrote it. What the run had
  shipped is recorded in the document's ``refinalized_own_text.json``, NOT in the sidecar:
  a resume re-writes a kept page's sidecar from its state, and a key only this tool writes
  would not survive it, so the page would not be byte-identical across the resume;
* the page's ``manifest.json`` entry ``blob_ref``, its new text stored in the document's
  content-addressed ``cache/`` so ``replay`` rebuilds the same document;
* ``metadata.json``'s ``tables`` counts, recomputed from the sidecars (a demoted page's
  tables are no longer ``verified_text``), here and in the run's root index;
* the document's ``status``, here and in the root index: ``completed`` becomes ``partial``
  when a page comes out FAIL, as a run of this code records it (its document rule counts
  those pages). UNKNOWN stays at the page, as it does in a run.

Idempotent: a page whose verdict already carries the clause is left alone, and the counts
are recomputed for every document, so a run interrupted anywhere is completed by running
the tool again. Write order: fragments, document ``.md``, manifest and cache, sidecars,
then the counts.

Not done here, and why: a resume cannot do this (the run fingerprint carries a digest of
socr's source, so a resume on new code re-OCRs every page).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

#: The per-document record of what this tool changed and what the run had shipped.
LOG_NAME = "refinalized_own_text.json"

#: Why a page the CURRENT analysis finds an earlier OCR layer on, which the run did not
#: flag, is UNKNOWN here rather than left alone: in a run #1156's clause marks such a page
#: in the same pass, but this tool applies only #1185's, so abstaining would leave it an
#: unmarked SUCCESS (Fable, PR #1219 review P1-B: 22 of the calibration copy's 43).
PRIOR_OCR_UNMARKED_REASON = (
    "this page's text layer looks like an earlier OCR run's reading, not its own text, "
    "and the run did not mark it (#1156, socr-refinalize-prior-ocr), so the reading's "
    "numbers are not checked"
)

#: ``PageState`` fields the clause reads, copied from the page analysis.
_ANALYSIS_FIELDS = (
    "own_layer_text",
    "is_born_digital",
    "control_byte_digit_hits",
    "control_byte_scan_failed",
    "minus_as_digit_hits",
    "minus_as_digit_scan_failed",
    "has_unrecovered_symbol_glyphs",
    "has_encoding_hygiene_suspect",
)


@dataclass(frozen=True)
class PageChange:
    doc: str
    page_num: int
    before: str
    after: str
    result: str


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, data) -> None:
    """In the writers' own format: two-space indent, UTF-8, and the trailing newline
    exactly when the file being replaced had one."""
    newline = "\n" if path.is_file() and path.read_bytes().endswith(b"\n") else ""
    _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + newline)


def _analyse_pdf(pdf_dir: Path, meta: dict):
    """``(the document's page analysis, None)``, or ``(None, why not)``."""
    from ocr_output_contract import safe_checksum

    from socr.core.born_digital import BornDigitalDetector

    pdf = pdf_dir / str(meta.get("key") or "")
    if not pdf.is_file():
        return None, "PDF not found under --pdf-dir"
    if safe_checksum(pdf) != str(meta.get("checksum") or ""):
        return None, "PDF under --pdf-dir is not the run's input (checksum)"
    assessment = BornDigitalDetector().detect(pdf)
    if len(assessment.pages) != int(meta.get("pages") or 0):
        return None, "PDF under --pdf-dir is not the run's input (page count)"
    return assessment, None


def _analysis_finds_prior_ocr(pa) -> bool:
    """#1156's union, from the current page analysis."""
    return bool(
        pa.invisible_text_over_raster
        or pa.invisible_text_scan_failed
        or pa.prior_ocr_layer
        or pa.prior_ocr_layer_scan_failed
    )


def _page_state(sidecar: dict, pa):
    """The page state the clause reads: the analysis's layer and flags, and #1156's
    prior-OCR suspicion from EITHER the run's record or the analysis (the union a run
    computes), so a page the run flagged is never compared with its earlier-OCR layer."""
    from socr.core.state import PageState

    p = PageState(page_num=int(sidecar["page_num"]))
    for name in _ANALYSIS_FIELDS:
        setattr(p, name, getattr(pa, name))
    p.prior_ocr_layer_suspect = bool(
        sidecar.get("prior_ocr_layer_suspect")
    ) or _analysis_finds_prior_ocr(pa)
    return p


def refinalize_page(sidecar: dict, pa, event_kinds: frozenset[str] = frozenset()):
    """``(new winning output, new verdict)`` for this page, or ``None`` to leave it.

    *event_kinds* are the kinds of the page's events in the run's ``audit_log.json``: with
    the run's own prior-OCR flag they decide where the clause sits in the verdict."""
    from socr.core.manifest import apply_verdict_marks
    from socr.core.marks import Check
    from socr.core.result import PageOutput
    from socr.core.state import PageState
    from socr.core.verdict import (
        PageVerdict,
        text_layer_numbers_clause,
        with_text_layer_numbers_clause,
    )

    if sidecar.get("terminal") is not True:
        return None
    published = PageVerdict.from_dict(sidecar.get("verdict") or {})
    if any(c.check is Check.TEXT_LAYER_NUMBERS for c in published.clauses):
        return None  # already re-finalized, or written by a run of this code
    output = PageOutput.from_dict(sidecar.get("winning_output") or {})
    p = _page_state(sidecar, pa)
    if p.prior_ocr_layer_suspect and not sidecar.get("prior_ocr_layer_suspect"):
        # Only the analysis finds the earlier OCR layer, so no clause of the run marks the
        # page: the clause's no-own-text UNKNOWN (the same guards: a failed marker or an
        # empty reading still gets none), with the reason that says why.
        p.prior_ocr_layer_suspect = False
        p.own_layer_text = ""
        clause = text_layer_numbers_clause(output, p)
        if clause is not None:
            clause = replace(clause, reason=PRIOR_OCR_UNMARKED_REASON)
    else:
        clause = text_layer_numbers_clause(output, p)
    if clause is None:
        return None
    as_run = slot_reference_state(sidecar, output.page_num)
    extended = with_text_layer_numbers_clause(
        replace(published, page_num=output.page_num), clause, output, as_run, event_kinds
    )
    return apply_verdict_marks(output, p, output.page_num, published=extended)


def slot_reference_state(sidecar: dict, page_num: int):
    """The stand-in ``PageState`` ``with_text_layer_numbers_clause`` counts clauses on.

    It must carry every field a silent-success clause AHEAD of this one reads, because the
    slot is a COUNT: a field left out makes the count short and puts this clause above a mark
    the writing run had put above it.

    A named function rather than four lines inline, so the invariant can be tested instead of
    grepped for (#1187 merge: the mutation that emptied the restore survived a source-text
    assertion).
    """
    from socr.core.state import PageState

    p = PageState(page_num=page_num)
    p.prior_ocr_layer_suspect = bool(sidecar.get("prior_ocr_layer_suspect"))
    # #1187: a figure the page names and does not show is a silent-success clause ahead of
    # this one. Both keys are persisted by the live run.
    unlocated = sidecar.get("unshipped_figure_unlocated")
    p.unshipped_figure_unlocated = (
        [str(c) for c in unlocated] if isinstance(unlocated, list) else []
    )
    kind = sidecar.get("unshipped_figure_kind")
    p.unshipped_figure_kind = kind if isinstance(kind, str) else ""
    return p


def _page_event_kinds(doc_dir: Path) -> dict[int, frozenset[str]]:
    """Each page's event kinds from the run's ``audit_log.json``. A run writes that file only
    when the document has events (``_write_audit_log``), so a finished document without one
    had none: every page's kinds are empty, not unknown (Argon, round-2 review: skipping such
    documents left their pages unchecked, a regression on round 1)."""
    path = doc_dir / "audit_log.json"
    if not path.is_file():
        return {}
    kinds: dict[int, set[str]] = {}
    for event in _read_json(path).get("events") or []:
        if isinstance(event.get("page_num"), int):
            kinds.setdefault(event["page_num"], set()).add(str(event.get("kind") or ""))
    return {n: frozenset(k) for n, k in kinds.items()}


def _updated_sidecar(sidecar: dict, output, verdict) -> dict:
    from socr.core.manifest import is_page_failed_marker
    from socr.pipeline.orchestrator import _page_blob_key, _socr_source_digest, _socr_version

    winning = output.to_dict()
    out = dict(sidecar)
    out["status"] = winning.get("status", "missing")
    out["failure_mode"] = winning.get("failure_mode", "none")
    out["audit_passed"] = winning.get("audit_passed", False)
    out["verdict"] = verdict.to_dict()
    out["winning_output"] = winning
    out["page_fingerprint"] = (
        _page_blob_key(winning) if not is_page_failed_marker(winning.get("text", "")) else ""
    )
    out["socr_version"] = _socr_version()
    out["socr_source_digest"] = _socr_source_digest()
    if _has_disagreement(out):
        # The page's copy of the event a run of this code records for it (at assemble,
        # after the lanes' own events, so last).
        from socr.core.verdict import NUMBERS_DISAGREE_EVENT_DETAIL, NUMBERS_DISAGREE_KIND

        events = [
            e for e in out.get("audit_events") or [] if e.get("kind") != NUMBERS_DISAGREE_KIND
        ]
        events.append(
            {
                "data": {},
                "detail": NUMBERS_DISAGREE_EVENT_DETAIL,
                "engine": winning.get("engine") or "",
                "kind": NUMBERS_DISAGREE_KIND,
            }
        )
        out["audit_events"] = events
    return out


def _swap_once(haystack: str, old: str, new: str, where: Path) -> str:
    """Replace the one copy of *old*, or accept that an interrupted run already did."""
    # The already-done check comes FIRST: when *new* extends *old* (a mark added under an
    # existing one), *old* still occurs once after the replace, and a rerun would add the
    # mark again (Argon, diff review).
    if haystack.count(new) == 1:
        return haystack
    if haystack.count(old) == 1:
        return haystack.replace(old, new)
    raise RuntimeError(f"{where}: the page's section is not exactly once in it")


def _recount_tables(doc_dir: Path, meta_path: Path, meta: dict) -> None:
    """``metadata.json``'s ``tables`` block, recomputed from the sidecars as they now are,
    in the document's record and in the run's root index."""
    from socr.core.table_counts import count_from_sidecars

    if "tables" not in meta:
        return
    counts = count_from_sidecars(doc_dir)
    if counts is None:
        return
    tables = counts.to_dict()
    if meta.get("tables") != tables:
        meta["tables"] = tables
        _write_json(meta_path, meta)
    root_path = doc_dir.parent / "metadata.json"
    if root_path.is_file():
        root = _read_json(root_path)
        entry = (root.get("files") or {}).get(meta.get("key"))
        if isinstance(entry, dict) and "tables" in entry and entry["tables"] != tables:
            entry["tables"] = tables
            _write_json(root_path, root)


def _has_disagreement(sidecar: dict) -> bool:
    """The page's published verdict carries this clause's FAIL."""
    from socr.core.marks import Check
    from socr.core.verdict import CheckResult

    return any(
        c.get("check") == Check.TEXT_LAYER_NUMBERS.value
        and c.get("result") == CheckResult.FAIL.value
        for c in (sidecar.get("verdict") or {}).get("clauses") or []
    )


def _record_disagreement(
    doc_dir: Path, meta_path: Path, meta: dict, pages: list[tuple[int, str]]
) -> None:
    """What a run of this code records for a document with FAIL *pages* (``(page, engine)``):
    a clean status becomes ``partial`` (a worse one is left as the run wrote it), the
    ``error`` note names the pages, here and in the root index, and ``audit_log.json``
    holds one event per page. Each step checks before writing, so a rerun adds nothing."""
    from ocr_output_contract import Status

    from socr.core.verdict import (
        NUMBERS_DISAGREE_EVENT_DETAIL,
        NUMBERS_DISAGREE_KIND,
        numbers_disagree_note,
    )

    note = numbers_disagree_note([n for n, _engine in pages])

    def recorded(record: dict) -> bool:
        changed = False
        if record.get("status") == Status.COMPLETED.value:
            record["status"] = Status.PARTIAL.value
            changed = True
        error = record.get("error") or ""
        if note not in error:
            record["error"] = f"{error}; {note}" if error else note
            changed = True
        return changed

    if recorded(meta):
        _write_json(meta_path, meta)
    root_path = doc_dir.parent / "metadata.json"
    if root_path.is_file():
        root = _read_json(root_path)
        entry = (root.get("files") or {}).get(meta.get("key"))
        if isinstance(entry, dict) and recorded(entry):
            _write_json(root_path, root)
    audit_path = doc_dir / "audit_log.json"
    # A run with no events wrote no audit log; this document now has events, so it gets the
    # log a run of this code writes (``RunAudit.to_dict``'s shape, named after the PDF).
    audit = (
        _read_json(audit_path)
        if audit_path.is_file()
        else {
            "pdf_filename": Path(str(meta.get("key") or "")).name,
            "event_count": 0,
            "counts": {},
            "events": [],
        }
    )
    events = audit.get("events") or []
    have = {e.get("page_num") for e in events if e.get("kind") == NUMBERS_DISAGREE_KIND}
    added = False
    for page_num, engine in pages:
        if page_num in have:
            continue
        event = {
            "page_num": page_num,
            "kind": NUMBERS_DISAGREE_KIND,
            "engine": engine,
            "detail": NUMBERS_DISAGREE_EVENT_DETAIL,
            "data": {},
        }
        # After the page's last event, so the log still reads page by page.
        at = sum(1 for e in events if (e.get("page_num") or 0) <= page_num)
        events.insert(at, event)
        added = True
    if added:
        counts: dict[str, int] = {}
        for e in events:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        audit.update(event_count=len(events), counts=counts, events=events)
        _write_json(audit_path, audit)


def refinalize_document(
    doc_dir: Path, pdf_dir: Path, *, dry_run: bool = False, skipped: Counter | None = None
) -> list[PageChange]:
    """Re-finalize one document's model-read pages. Nothing is written until every page of
    the document has been decided, so an inconsistency leaves the document untouched."""
    from socr.core.marks import Check
    from socr.core.verdict import model_reading_engines

    skipped = Counter() if skipped is None else skipped
    pages_dir = doc_dir / "pages"
    meta_path = doc_dir / "metadata.json"
    if not pages_dir.is_dir():
        return []
    sidecar_paths = sorted(pages_dir.glob("[0-9]*.json"))
    candidates = []
    for path in sidecar_paths:
        sidecar = _read_json(path)
        engine = (sidecar.get("winning_output") or {}).get("engine") or ""
        if engine not in model_reading_engines():
            continue
        if sidecar.get("terminal") is not True:
            # A crash-recovery copy, not a finished page: the run's next pass rewrites it.
            skipped["page not finished (terminal is not true)"] += 1
            continue
        candidates.append((path, sidecar))
    if not candidates:
        return []
    if not meta_path.is_file():
        skipped["document has no metadata.json (run did not finish it)"] += len(candidates)
        return []
    meta = _read_json(meta_path)
    md_paths = sorted(doc_dir.glob("*.md"))
    if len(md_paths) != 1:
        skipped["document has no single .md"] += len(candidates)
        return []
    event_kinds = _page_event_kinds(doc_dir)
    assessment, why_not = _analyse_pdf(pdf_dir, meta)
    if assessment is None:
        skipped[why_not] += len(candidates)
        return []
    doc_md = md_paths[0].read_text(encoding="utf-8")
    changes: list[PageChange] = []
    writes: list[tuple[Path, str, Path, dict]] = []
    for path, sidecar in candidates:
        page_num = int(sidecar["page_num"])
        done = refinalize_page(
            sidecar, assessment.pages[page_num - 1], event_kinds.get(page_num, frozenset())
        )
        if done is None:
            continue
        output, verdict = done
        fragment_path = path.with_suffix(".md")
        old_text = (sidecar.get("winning_output") or {}).get("text") or ""
        old_fragment = fragment_path.read_text(encoding="utf-8")
        if old_fragment not in (old_text, output.text):
            raise RuntimeError(f"{fragment_path}: fragment is not its sidecar's text")
        heading = f"## Page {page_num}\n\n"
        doc_md = _swap_once(doc_md, heading + old_text, heading + output.text, md_paths[0])
        new_sidecar = _updated_sidecar(sidecar, output, verdict)
        writes.append((fragment_path, output.text, path, new_sidecar))
        changes.append(
            PageChange(
                doc=doc_dir.name,
                page_num=page_num,
                before=str(sidecar.get("status")),
                after=new_sidecar["status"],
                result=next(
                    c.result.value for c in verdict.clauses if c.check is Check.TEXT_LAYER_NUMBERS
                ),
            )
        )
    manifest_path = doc_dir / "manifest.json"
    manifest = _read_json(manifest_path) if writes and manifest_path.is_file() else None
    if manifest is not None:
        # Checked BEFORE the first write: a manifest without a page's entry used to raise
        # after the fragments were rewritten and before the sidecars, and every rerun
        # raised at the same place, leaving the two out of step (Argon, PR #1219 review).
        missing = [
            str(data["page_num"])
            for *_rest, data in writes
            if str(data["page_num"]) not in (manifest.get("entries") or {})
        ]
        if missing:
            raise RuntimeError(f"{manifest_path}: no entry for page(s) {', '.join(missing)}")
    if dry_run:
        return changes
    if writes:
        for fragment_path, text, _p, _s in writes:
            _write_text(fragment_path, text)
        _write_text(md_paths[0], doc_md)
        if manifest is not None:
            from socr.core.cache import BlobStore
            from socr.core.result import PageOutput

            store = BlobStore(doc_dir / "cache")
            for _f, _t, _p, data in writes:
                entry = manifest["entries"][str(data["page_num"])]
                entry["blob_ref"] = store.put_page(PageOutput.from_dict(data["winning_output"]))
            _write_json(manifest_path, manifest)
        log_path = doc_dir / LOG_NAME
        log = _read_json(log_path) if log_path.is_file() else {"tool": "socr-refinalize-own-text"}
        pages = log.setdefault("pages", {})
        for change, (_f, _t, _p, data) in zip(changes, writes):
            pages[str(change.page_num)] = {
                "before": {"status": change.before},
                "after": {"status": change.after, "text_layer_numbers": change.result},
                "socr_source_digest": data["socr_source_digest"],
            }
        _write_json(log_path, log)
        for _f, _t, path, data in writes:
            _write_json(path, data)
    # From the sidecars as they now are, not from this pass's changes, so a run interrupted
    # between the sidecars and this line is completed by the next one.
    disagreeing = [
        (int(sc["page_num"]), (sc.get("winning_output") or {}).get("engine") or "")
        for sc in map(_read_json, sidecar_paths)
        if _has_disagreement(sc)
    ]
    if disagreeing:
        _record_disagreement(doc_dir, meta_path, meta, disagreeing)
    _recount_tables(doc_dir, meta_path, meta)
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="socr-refinalize-own-text", description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--pdf-dir", type=Path, required=True, help="the run's input PDFs")
    parser.add_argument("--dry-run", action="store_true", help="count, write nothing")
    args = parser.parse_args(argv)
    counts: Counter = Counter()
    skipped: Counter = Counter()
    failed: list[tuple[Path, str]] = []
    # Every document directory with pages, with or without its metadata.json, so a document
    # the run did not finish is counted rather than passed over (Argon, PR #1219 review).
    for pages_dir in sorted(p for p in args.out_dir.rglob("pages") if p.is_dir()):
        doc_dir = pages_dir.parent
        try:
            changes = refinalize_document(
                doc_dir, args.pdf_dir, dry_run=args.dry_run, skipped=skipped
            )
        except Exception as exc:  # noqa: BLE001 -- reported per document, never swallowed
            failed.append((doc_dir, f"{type(exc).__name__}: {exc}"))
            continue
        for change in changes:
            counts[(change.before, change.after, change.result)] += 1
    for (before, after, result), n in sorted(counts.items()):
        print(f"{n:6d}  {before} -> {after}  (text_layer_numbers={result})")
    print(f"{sum(counts.values()):6d}  pages re-finalized")
    for reason, n in sorted(skipped.items()):
        print(f"{n:6d}  not checked: {reason}")
    for doc_dir, why in failed:
        print(f"FAILED  {doc_dir}: {why}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
