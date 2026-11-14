"""#1282: restore dropped decimal points in a FINISHED run, from the pages' own text layers.

    socr-refinalize-point-drops <out_dir> --pdf-dir <dir> [--apply]

A model that reads ``-.7`` as ``-7`` ships a wrong number under a mark.  Since #1282 a run
restores such a point before it finalizes the page (``UnifiedPipeline._repair_dropped_points``).
This tool does the same to a run made before, with NO new OCR and no model: the page's own
characters come from the run's input PDF, accepted only for the very file the run read (same
checksum, same page count; ``refinalize_own_text._analyse_pdf``).

DRY-RUN BY DEFAULT: without ``--apply`` it prints exactly what ``--apply`` would change and writes
nothing.  The rule is ``tables.row_binding_check.propose_page_point_repairs`` (a cell is changed
only when one text-layer line prints the row's exact label and the same values, differing by
dropped points only), under the same page predicate as the row-binding check
(``layer_can_judge_tables``).  A CONTRADICTED table is never repaired.

A page is repaired through the seam a run uses (``core.manifest.apply_verdict_marks``): the
repaired text replaces the reading's, the ``text_layer_numbers`` and ``text_layer_drops`` clauses
are RECOMPUTED on it (unlike ``refinalize_own_text``, which leaves a page that already carries
one alone), the stale text-layer marks are removed and the new ones written, and the page's
``row_binding`` record and events are rewritten from the check on the repaired text.  Then:
``pages/NNN.md`` and the document ``.md`` section, the manifest ``blob_ref`` and cache, the
sidecar (record-derived keys), the document ``audit_log.json`` (a ``table_point_repaired`` event
per repaired table; the text-layer events kept, shrunk or removed), the document notes naming
text-layer pages, ``metadata.json``'s table counts, and ``refinalized_point_drops.json`` with
every old and new cell.

A page is LEFT ALONE, and NAMED in the output and in the log (``left_alone``), when a repair was
proposed and applying it would lift a mark (the page would lose every clause it carries).  A repair
lifts no mark: a table with a repaired row is recorded UNVERIFIABLE even when every row now binds,
because the table ladder judged the unrepaired bytes; this is what the live step records too, so the
tool's bytes are the live run's bytes.

The document ``status`` is never moved back (a ``partial`` stays ``partial``): a stale ``partial``
is conservative, and a document can be ``partial`` for reasons this tool cannot see.

Idempotent: a repaired page has nothing left to repair, and a run interrupted anywhere is
completed by running the tool again.  Write order: fragments, document ``.md``, manifest and
cache, document audit log, sidecars, then notes and counts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from socr.core.verdict import pass_trace_outcomes, published_audit_passed, reconcile_pass_trace

#: The per-document record of what this tool changed.
LOG_NAME = "refinalized_point_drops.json"


@dataclass(frozen=True)
class PageChange:
    doc: str
    page_num: int
    #: ``(table_id, label, old, new)`` per repaired cell.
    cells: tuple[tuple[str, str, str, str], ...]


@dataclass
class _Done:
    """Everything one page's repair produces, before anything is written."""

    page_num: int
    output: object  # PageOutput
    verdict: object  # PageVerdict
    sidecar: dict
    change: PageChange
    repair_events: list[dict]
    payloads: dict[str, dict]
    old_text: str
    dropped_data: dict


def _text_layer_mark_re(page_num: int) -> re.Pattern[str]:
    from socr.core.marks import Check

    checks = "|".join((Check.TEXT_LAYER_NUMBERS.value, Check.TEXT_LAYER_DROPS.value))
    return re.compile(rf"^> \[page {page_num} unchecked: (?:{checks})=")


def without_text_layer_marks(text: str, page_num: int) -> str:
    """*text* without its ``text_layer_numbers`` / ``text_layer_drops`` page marks (the leading
    banner block only; every other banner stays, in its place)."""
    from socr.core.manifest import without_page_top_banners

    rx = _text_layer_mark_re(page_num)
    return without_page_top_banners(text, lambda banner: bool(rx.match(banner)))


def _page_state_for(sidecar: dict, pa):
    from socr.pipeline.refinalize_own_text import _page_state

    p = _page_state(sidecar, pa)
    p.refused_for_spacing_alone = bool(getattr(pa, "refused_for_spacing_alone", False))
    return p


def repair_page(
    sidecar: dict,
    pa,
    page,
    event_kinds: frozenset[str],
    skipped: Counter,
    named: dict[int, str] | None = None,
):
    """The :class:`_Done` of one page, or ``None`` (with the reason counted in *skipped*).  A page
    for which a repair was PROPOSED and then not applied is also NAMED in *named* (page -> why)."""
    from socr.core.manifest import apply_verdict_marks, is_page_failed_marker
    from socr.core.marks import Check
    from socr.core.result import PageOutput
    from socr.core.verdict import (
        PageVerdict,
        in_fresh_run_order,
        numbers_dropped_event_data,
        page_dropped_numbers,
        text_layer_drops_clause,
        text_layer_numbers_clause,
        with_text_layer_numbers_clause,
    )
    from socr.judge.table_verdict import TABLE_POINT_REPAIRED_KIND
    from socr.pipeline.orchestrator import _page_blob_key, _socr_source_digest, _socr_version
    from socr.pipeline.refinalize_own_text import slot_reference_state
    from socr.tables.locate import locate_tables
    from socr.tables.row_binding_check import (
        check_page_row_binding,
        effective_outcome,
        layer_can_judge_tables,
        propose_page_point_repairs,
        repair_event_detail,
        repair_row_dict,
        row_evidence_dicts,
        table_payload,
    )

    page_num = int(sidecar["page_num"])
    output = PageOutput.from_dict(sidecar.get("winning_output") or {})
    if output.table_floor_body or is_page_failed_marker(output.text or ""):
        skipped["page ships a floor or a failure marker, not the model's table"] += 1
        return None
    p = _page_state_for(sidecar, pa)
    if not layer_can_judge_tables(p, output, page):
        skipped[
            "the page's text layer may not judge its tables (not born-digital, or an earlier OCR)"
        ] += 1
        return None
    regions = [tuple(b.bbox) for b in locate_tables(page)] or None
    new_text, repaired, _refused = propose_page_point_repairs(page, output.text, regions)
    if not repaired or new_text == output.text:
        skipped["nothing to repair"] += 1
        return None
    new_results, _rot = check_page_row_binding(page, new_text, regions)

    # The verdict: the two text-layer clauses recomputed on the repaired reading.
    published = PageVerdict.from_dict(sidecar.get("verdict") or {})
    output2 = replace(output, text=without_text_layer_marks(new_text, page_num))
    as_run = slot_reference_state(sidecar, page_num)
    text_checks = (Check.TEXT_LAYER_NUMBERS, Check.TEXT_LAYER_DROPS)
    extended = replace(
        published,
        page_num=page_num,
        clauses=tuple(c for c in published.clauses if c.check not in text_checks),
    )
    numbers = text_layer_numbers_clause(output2, p)
    drops = text_layer_drops_clause(output2, p)
    if numbers is not None:
        extended = with_text_layer_numbers_clause(extended, numbers, output2, as_run, event_kinds)
    if drops is not None:
        extended = replace(extended, clauses=tuple(in_fresh_run_order((*extended.clauses, drops))))
    if published.clauses and not extended.clauses:
        why = "the page would lose every clause it carries (a lifted mark; #1282 lifts none)"
        skipped[why] += 1
        if named is not None:
            named[page_num] = why
        return None
    final, verdict = apply_verdict_marks(output2, p, page_num, published=extended)

    # The sidecar: record-derived keys, as ``refinalize_own_text._updated_sidecar``.
    winning = final.to_dict()
    new_sidecar = dict(sidecar)
    new_sidecar["status"] = winning.get("status", "missing")
    new_sidecar["failure_mode"] = winning.get("failure_mode", "none")
    new_sidecar["audit_passed"] = published_audit_passed(
        new_sidecar["status"], winning.get("audit_passed", False), verdict
    )
    new_sidecar["verdict"] = verdict.to_dict()
    new_sidecar["winning_output"] = winning
    new_sidecar["page_fingerprint"] = _page_blob_key(winning)
    new_sidecar["socr_source_digest"] = _socr_source_digest()
    new_sidecar["socr_version"] = _socr_version()

    # The row-binding record and events, from the check on the repaired text.
    payloads: dict[str, dict] = {}
    records = dict(sidecar.get("row_binding") or {})
    repairs_by_table = {i: reps for i, reps in repaired}
    from socr.tables.reconcile import find_table_blocks

    blocks = find_table_blocks(new_text)
    lines = new_text.splitlines()
    cells: list[tuple[str, str, str, str]] = []
    repair_events: list[dict] = []
    for idx, result in enumerate(new_results):
        table_id = f"p{page_num}-t{idx}"
        reps = repairs_by_table.get(idx)
        rows_data = None
        if reps:
            start = blocks[idx].start
            rows_data = [repair_row_dict(r, lines[start + r.md_line]) for r in reps]
            cells.extend((table_id, r.label, r.old, r.new) for r in reps)
            repair_events.append(
                {
                    "kind": TABLE_POINT_REPAIRED_KIND,
                    "engine": output.engine or "",
                    "detail": repair_event_detail(table_id, rows_data),
                    "data": {"table_id": table_id, "rows": rows_data},
                }
            )
        if table_id in records:
            # as the live step records it: a table with a repaired row is never BOUND (no lift)
            payload = table_payload(
                page_num,
                idx,
                effective_outcome(result.outcome.value, bool(rows_data)),
                row_evidence_dicts(result),
            )
            if "withheld_markdown" in records[table_id]:
                payload["withheld_markdown"] = records[table_id]["withheld_markdown"]
            if rows_data:
                payload["repairs"] = rows_data
            payloads[table_id] = payload
    if payloads:
        new_sidecar["row_binding"] = {**records, **payloads}
    # #1183: keep the pass trace consistent with the verdict and the row-binding records this tool
    # wrote (no invention): the check ran here, so its table entries are the records' own.
    # It also ran the two number checks again on the repaired text, so their outcomes are its own.
    _rerun = {Check.TEXT_LAYER_NUMBERS.value, Check.TEXT_LAYER_DROPS.value}
    _checks = reconcile_pass_trace(
        sidecar.get("checks"),
        verdict,
        new_sidecar.get("row_binding") if payloads else None,
        recomputed=tuple(o.entry for o in pass_trace_outcomes(final, p) if o.entry.check in _rerun),
    )
    if _checks is not None:
        new_sidecar["checks"] = _checks

    # The page's events: text-layer events kept, shrunk or removed; row-binding events rewritten;
    # the repair events added where a run records them (after the page's other events).
    still_numbers = any(c.check is Check.TEXT_LAYER_NUMBERS for c in verdict.clauses)
    still_drops = any(c.check is Check.TEXT_LAYER_DROPS for c in verdict.clauses)
    dropped_data = numbers_dropped_event_data(page_num, page_dropped_numbers(final, p))
    new_sidecar["audit_events"] = rewrite_page_events(
        list(sidecar.get("audit_events") or []),
        page_num,
        keep_numbers=still_numbers,
        keep_drops=still_drops,
        dropped_data=dropped_data,
        payloads=payloads,
        repair_events=repair_events,
    )
    return _Done(
        page_num=page_num,
        output=final,
        verdict=verdict,
        sidecar=new_sidecar,
        change=PageChange("", page_num, tuple(cells)),
        repair_events=repair_events,
        payloads=payloads,
        old_text=output.text or "",
        dropped_data=dropped_data,
    )


def rewrite_page_events(
    events: list[dict],
    page_num: int,
    *,
    keep_numbers: bool,
    keep_drops: bool,
    dropped_data: dict,
    payloads: dict[str, dict],
    repair_events: list[dict],
) -> list[dict]:
    """*events* (one page's, or a document log's with other pages' mixed in) after the repair.

    Only events of *page_num* (or without a ``page_num``: a sidecar's own list) are touched.
    The text-layer events of a clause the page no longer carries are removed; a ``drops`` event
    that remains gets the recomputed list.  A row-binding event takes its table's new payload.
    ``table_point_repaired`` events replace any earlier ones of the same table and sit after the
    page's last event (the order a run records them in: once the page's final output is known).
    Idempotent."""
    from socr.core.verdict import NUMBERS_DISAGREE_KIND, NUMBERS_DROPPED_KIND
    from socr.judge.table_verdict import ROW_BINDING_EVENT_KINDS, TABLE_POINT_REPAIRED_KIND

    def mine(e: dict) -> bool:
        return e.get("page_num", page_num) == page_num

    repaired_tables = {e["data"]["table_id"] for e in repair_events}
    out: list[dict] = []
    for e in events:
        if not mine(e):
            out.append(e)
            continue
        kind = e.get("kind")
        if kind == NUMBERS_DISAGREE_KIND and not keep_numbers:
            continue
        if kind == NUMBERS_DROPPED_KIND:
            if not keep_drops:
                continue
            e = {**e, "data": dropped_data}
        if kind in ROW_BINDING_EVENT_KINDS:
            payload = payloads.get((e.get("data") or {}).get("table_id"))
            if payload is not None:
                e = {**e, "data": payload}
        if (
            kind == TABLE_POINT_REPAIRED_KIND
            and (e.get("data") or {}).get("table_id") in repaired_tables
        ):
            continue
        out.append(e)
    mine_at = [n for n, e in enumerate(out) if mine(e)]
    at = (mine_at[-1] + 1) if mine_at else len(out)
    return [*out[:at], *repair_events, *out[at:]]


def _replace_note(error: str, old: str | None, new: str | None) -> str:
    """*error* with the note *old* (``None``: none was there) replaced by *new* (``None``:
    removed), in its place, notes being joined by ``"; "``."""
    if old is None or old not in error:
        if new is None or new in error:
            return error
        return f"{error}; {new}" if error else new
    if new is not None:
        return error.replace(old, new, 1)
    for gap in (old + "; ", "; " + old, old):
        if gap in error:
            return error.replace(gap, "", 1)
    return error


def _text_layer_pages(sidecar_paths, check) -> list[int]:
    from socr.pipeline.refinalize_own_text import _has_fail, _read_json

    return [int(sc["page_num"]) for sc in map(_read_json, sidecar_paths) if _has_fail(sc, check)]


def refinalize_document(
    doc_dir: Path, pdf_dir: Path, *, apply: bool = False, skipped: Counter | None = None
) -> list[PageChange]:
    """Repair one document's model-read pages.  Nothing is written until every page of the
    document has been decided, so an inconsistency leaves the document untouched."""
    from socr.core.pdf import open_pdf
    from socr.core.verdict import engine_is_model_reading
    from socr.pipeline.refinalize_own_text import (
        _analyse_pdf,
        _page_event_kinds,
        _read_json,
        _recount_tables,
        _swap_once,
        _write_json,
        _write_text,
    )

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
        if not engine_is_model_reading(engine):
            skipped["page engine is not a model reading"] += 1
            continue
        if sidecar.get("terminal") is not True:
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
    assessment, why_not = _analyse_pdf(pdf_dir, meta)
    if assessment is None:
        skipped[why_not] += len(candidates)
        return []
    event_kinds = _page_event_kinds(doc_dir)
    doc_md = md_paths[0].read_text(encoding="utf-8")
    done: list[tuple[Path, _Done]] = []
    named: dict[int, str] = {}
    with open_pdf(pdf_dir / str(meta.get("key") or ""), repair=True) as pdf:
        for path, sidecar in candidates:
            page_num = int(sidecar["page_num"])
            d = repair_page(
                sidecar,
                assessment.pages[page_num - 1],
                pdf[page_num - 1],
                event_kinds.get(page_num, frozenset()),
                skipped,
                named,
            )
            if d is None:
                continue
            fragment_path = path.with_suffix(".md")
            old_fragment = fragment_path.read_text(encoding="utf-8")
            if old_fragment not in (d.old_text, d.output.text):
                raise RuntimeError(f"{fragment_path}: fragment is not its sidecar's text")
            heading = f"## Page {page_num}\n\n"
            doc_md = _swap_once(doc_md, heading + d.old_text, heading + d.output.text, md_paths[0])
            done.append((path, d))
    changes = [replace(d.change, doc=doc_dir.name) for _p, d in done]
    for page_num, why in sorted(named.items()):
        skipped[f"{doc_dir.name} p{page_num} left alone: {why}"] += 1
    if apply and named and not done:
        _record_left_alone(doc_dir, named)
    if not done or not apply:
        return changes

    manifest_path = doc_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else None
    if manifest is not None:
        missing = [
            str(d.page_num)
            for _p, d in done
            if str(d.page_num) not in (manifest.get("entries") or {})
        ]
        if missing:
            raise RuntimeError(f"{manifest_path}: no entry for page(s) {', '.join(missing)}")

    from socr.core.marks import Check

    old_disagree = _text_layer_pages(sidecar_paths, Check.TEXT_LAYER_NUMBERS)
    old_drops = _text_layer_pages(sidecar_paths, Check.TEXT_LAYER_DROPS)
    for path, d in done:
        _write_text(path.with_suffix(".md"), d.output.text)
    _write_text(md_paths[0], doc_md)
    if manifest is not None:
        from socr.core.cache import BlobStore
        from socr.core.result import PageOutput

        store = BlobStore(doc_dir / "cache")
        for _p, d in done:
            entry = manifest["entries"][str(d.page_num)]
            entry["blob_ref"] = store.put_page(PageOutput.from_dict(d.sidecar["winning_output"]))
        _write_json(manifest_path, manifest)
    log_path = doc_dir / LOG_NAME
    log = _read_json(log_path) if log_path.is_file() else {"tool": "socr-refinalize-point-drops"}
    pages = log.setdefault("pages", {})
    for _p, d in done:
        pages[str(d.page_num)] = {
            "before": {
                "status": str(
                    next(sc for pp, sc in candidates if int(sc["page_num"]) == d.page_num).get(
                        "status"
                    )
                )
            },
            "after": {"status": d.sidecar["status"]},
            "cells": [
                {"table_id": t, "label": label, "old": old, "new": new}
                for t, label, old, new in d.change.cells
            ],
            "socr_source_digest": d.sidecar["socr_source_digest"],
        }
    if named:
        log["left_alone"] = {str(n): why for n, why in sorted(named.items())}
    _write_json(log_path, log)
    _rewrite_document_audit_log(doc_dir, done)
    for path, d in done:
        _write_json(path, d.sidecar)
    new_disagree = _text_layer_pages(sidecar_paths, Check.TEXT_LAYER_NUMBERS)
    new_drops = _text_layer_pages(sidecar_paths, Check.TEXT_LAYER_DROPS)
    _rewrite_document_notes(
        doc_dir, meta_path, meta, old_disagree, new_disagree, old_drops, new_drops
    )
    _recount_tables(doc_dir, meta_path, meta)
    return changes


def _record_left_alone(doc_dir: Path, named: dict[int, str]) -> None:
    """The pages a repair was proposed for and not applied, in the document's log."""
    from socr.pipeline.refinalize_own_text import _read_json, _write_json

    log_path = doc_dir / LOG_NAME
    log = _read_json(log_path) if log_path.is_file() else {"tool": "socr-refinalize-point-drops"}
    log["left_alone"] = {str(n): why for n, why in sorted(named.items())}
    _write_json(log_path, log)


def _rewrite_document_audit_log(doc_dir: Path, done: list[tuple[Path, _Done]]) -> None:
    """The document's ``audit_log.json`` after the repairs (a document with no log gets none: it
    had no events, and a repair here only adds ``table_point_repaired`` ones, so it gets the log a
    run of this code writes)."""
    from socr.core.marks import Check
    from socr.pipeline.refinalize_own_text import _read_json, _write_json

    path = doc_dir / "audit_log.json"
    audit = (
        _read_json(path)
        if path.is_file()
        else {"pdf_filename": doc_dir.name, "event_count": 0, "counts": {}, "events": []}
    )
    events = audit.get("events") or []
    before = json.dumps(events)
    for _p, d in done:
        verdict = d.verdict
        keep_numbers = any(c.check is Check.TEXT_LAYER_NUMBERS for c in verdict.clauses)
        keep_drops = any(c.check is Check.TEXT_LAYER_DROPS for c in verdict.clauses)
        events = rewrite_page_events(
            events,
            d.page_num,
            keep_numbers=keep_numbers,
            keep_drops=keep_drops,
            dropped_data=d.dropped_data,
            payloads=d.payloads,
            # (a document log's events carry their page; a page sidecar's do not)
            repair_events=[{"page_num": d.page_num, **e} for e in d.repair_events],
        )
    if json.dumps(events) != before:
        counts: dict[str, int] = {}
        for e in events:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        audit.update(event_count=len(events), counts=counts, events=events)
        _write_json(path, audit)


def _rewrite_document_notes(
    doc_dir: Path,
    meta_path: Path,
    meta: dict,
    old_disagree: list[int],
    new_disagree: list[int],
    old_drops: list[int],
    new_drops: list[int],
) -> None:
    """The ``error`` note naming text-layer pages, here and in the root index: the pages that
    still fail are named, a clause no page fails any more loses its note.  ``status`` is not
    touched (see the module docstring)."""
    from socr.core.verdict import numbers_disagree_note, numbers_dropped_note
    from socr.pipeline.refinalize_own_text import _read_json, _write_json

    def rewritten(record: dict) -> bool:
        error = record.get("error") or ""
        out = _replace_note(
            error,
            numbers_disagree_note(old_disagree) if old_disagree else None,
            numbers_disagree_note(new_disagree) if new_disagree else None,
        )
        out = _replace_note(
            out,
            numbers_dropped_note(old_drops) if old_drops else None,
            numbers_dropped_note(new_drops) if new_drops else None,
        )
        if out == error:
            return False
        if out:
            record["error"] = out
        else:
            record.pop("error", None)
        return True

    if rewritten(meta):
        _write_json(meta_path, meta)
    root_path = doc_dir.parent / "metadata.json"
    if root_path.is_file():
        root = _read_json(root_path)
        entry = (root.get("files") or {}).get(meta.get("key"))
        if isinstance(entry, dict) and rewritten(entry):
            _write_json(root_path, root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="socr-refinalize-point-drops", description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--pdf-dir", type=Path, required=True, help="the run's input PDFs")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the changes (default: print them, write nothing)",
    )
    args = parser.parse_args(argv)
    cells = 0
    pages = 0
    skipped: Counter = Counter()
    failed: list[tuple[Path, str]] = []
    for pages_dir in sorted(p for p in args.out_dir.rglob("pages") if p.is_dir()):
        doc_dir = pages_dir.parent
        try:
            changes = refinalize_document(doc_dir, args.pdf_dir, apply=args.apply, skipped=skipped)
        except Exception as exc:  # noqa: BLE001 -- reported per document, never swallowed
            failed.append((doc_dir, f"{type(exc).__name__}: {exc}"))
            continue
        for change in changes:
            pages += 1
            for table_id, label, old, new in change.cells:
                cells += 1
                print(f"{change.doc} p{change.page_num} {table_id} {label!r}: {old} -> {new}")
    verb = "repaired" if args.apply else "would repair"
    print(f"{pages:6d}  pages {verb}, {cells} cell(s)")
    if not args.apply:
        print("        (dry run: nothing written; pass --apply)")
    for reason, n in sorted(skipped.items()):
        print(f"{n:6d}  not repaired: {reason}")
    for doc_dir, why in failed:
        print(f"FAILED  {doc_dir}: {why}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
