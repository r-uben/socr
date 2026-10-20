"""#1156 Part B: re-finalize a FINISHED run's earlier-OCR floor pages from their sidecars.

A run made before Part B shipped two floors on a scan whose text layer is an earlier OCR
run and whose reading was rejected (or never made): Part A's ("earlier OCR, not re-read")
and #1027's ("invisible OCR layer unread"). Each ships a marker and the page image, and
retains the layer -- numbers included -- in the page sidecar as ``prior_ocr_layer_text``.

This tool recovers that work with NO new OCR and no GPU: for each such page it rebuilds
the page state the floor was selected from, runs socr's real ``_select_and_finalize_page``
on it (``finalized_page_record``), and, when that now selects the masked-layer ending, rewrites

* ``pages/NNN.md`` and the page's copy inside the document ``.md`` (asserted to occur
  exactly once, so nothing else in the document can change);
* the record-derived keys of ``pages/NNN.json`` -- everything else, including the
  retained layer itself, is left as the run wrote it;
* the page's ``manifest.json`` entry (its disposition, and its ``blob_ref`` to the new
  text, stored in the document's content-addressed ``cache/`` so ``replay`` rebuilds the
  same document);
* the document's audit log and ``metadata.json`` ``error``: the floor's events and the
  page's entry in the "produced no usable output" list go, the event a fresh run writes
  for such a page comes in.

A resume cannot do this: the run fingerprint carries a digest of socr's source, so any
code change makes a resume re-OCR every page.

Run it as ``socr-refinalize-prior-ocr <out_dir> [--dry-run]``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

#: The floors this tool may replace. Both are reachable only on a POSITIVE prior-OCR
#: finding, so the recorded ending is itself the evidence the page qualifies.
_FLOOR_REASONS = ("prior_ocr_layer_unread", "invisible_scan_unread")

#: Events the replaced floor wrote, which a fresh run of the masked ending does not.
_FLOOR_EVENT_KINDS = frozenset({"page_failed", "invisible_scan_unread"})

#: The image stems each floor renders, and the state field each one is read from.
_PNG_FIELDS = (
    ("invisible_scan_page", "invisible_scan_png_ref"),
    ("prior_ocr_layer_page", "prior_ocr_page_png_ref"),
)

#: The judge outcome of the one attempt an EXACT case carries. A fresh run's events for
#: the page depend on its attempt history (a judge timeout takes #1004's event instead of
#: ``native_fallback``; "best" among several rejected attempts is not in the sidecar), so
#: the tool re-finalizes only the case it can reproduce exactly -- one attempt, judged to
#: completion -- and counts every other page as needing a re-run. All 390 floor pages of
#: the Fed calibration run are such cases.
_EXACT_JUDGE_OUTCOMES = frozenset({"page_judge_completed"})

#: Sidecar keys that are ``PageState`` fields but not inputs to finalisation.
_NOT_RESTORED = frozenset({"page_num", "timings_s", "page_cost_usd"})

_FAILED_LIST_RE = re.compile(r"^page\(s\) ([0-9, ]+) (?P<note>.*)$")


@dataclass(frozen=True)
class PageChange:
    doc: str
    page_num: int
    before: tuple[str, str]
    after: tuple[str, str]
    numbers_withheld: int


def _rebuild_state(sidecar: dict, page_count: int):
    """The page state the floor was selected from, as far as the sidecar records it."""
    from socr.core.audit_log import AuditEvent
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    page_num = int(sidecar["page_num"])
    state = DocumentState(handle=DocumentHandle(path=Path("refinalize.pdf"), page_count=page_count))
    p = PageState(page_num=page_num)
    p.is_born_digital = True
    p.native_text = sidecar.get("prior_ocr_layer_text") or ""
    p.prior_ocr_layer_suspect = True
    p.text_coverage_verdict = sidecar.get("text_coverage") or ""
    old_text = (sidecar.get("winning_output") or {}).get("text") or ""
    png = next((ln for ln in old_text.splitlines() if ln.lstrip().startswith("![")), "")
    flag_from_png = ""
    for stem, field_name in _PNG_FIELDS:
        if png and f"/{stem}_p" in png and not getattr(p, field_name, ""):
            setattr(p, field_name, png.strip())
            flag_from_png = stem
    reason = (sidecar.get("disposition") or {}).get("primary_reason")
    if flag_from_png == "invisible_scan_page" or (
        not flag_from_png and reason == "invisible_scan_unread"
    ):
        p.invisible_text_over_raster = True
    else:
        p.prior_ocr_layer = True
    # Every other flag the sidecar persists goes back as the run left it, so the guards
    # and verdict clauses that read them see what a fresh finalisation would (Argon, diff
    # review). Only values of the field's own type: a sidecar is parsed JSON.
    import dataclasses

    defaults = PageState(page_num=page_num)
    for f in dataclasses.fields(PageState):
        if f.name in _NOT_RESTORED or f.name not in sidecar:
            continue
        value, default = sidecar[f.name], getattr(defaults, f.name)
        if default is None or isinstance(value, type(default)):
            setattr(p, f.name, value)
    p.native_text = sidecar.get("prior_ocr_layer_text") or ""
    p.prior_ocr_layer_suspect = True
    state.pages[page_num] = p
    for ev in sidecar.get("audit_events") or ():
        state.events.append(
            AuditEvent(
                page_num=page_num,
                kind=ev.get("kind", ""),
                engine=ev.get("engine", ""),
                detail=ev.get("detail", ""),
                data=ev.get("data") or {},
            )
        )
    return state, page_num


def is_exact_case(sidecar: dict) -> bool:
    """A terminal page with exactly one attempt, judged to completion."""
    attempts = sidecar.get("attempts_summary") or []
    return (
        sidecar.get("terminal") is True
        and len(attempts) == 1
        and attempts[0].get("judge_outcome") in _EXACT_JUDGE_OUTCOMES
    )


def is_floor_page(sidecar: dict) -> bool:
    return (
        (sidecar.get("disposition") or {}).get("primary_reason") in _FLOOR_REASONS
        and bool(sidecar.get("prior_ocr_layer_suspect"))
        and bool(sidecar.get("prior_ocr_layer_text"))
    )


def refinalize_sidecar(sidecar: dict, page_count: int):
    """The finalized record the masked ending gives this page, or ``None`` to leave it."""
    from socr.core.manifest import SelectionProvenance, finalized_page_record

    if not is_floor_page(sidecar) or not is_exact_case(sidecar):
        return None
    state, page_num = _rebuild_state(sidecar, page_count)
    record = finalized_page_record(state, page_num)
    if record.selection_provenance is not SelectionProvenance.PRIOR_OCR_LAYER_MASKED:
        return None
    return record


def _updated_sidecar(sidecar: dict, record) -> dict:
    from socr.core.manifest import is_page_failed_marker
    from socr.pipeline.orchestrator import (
        NATIVE_FALLBACK_EVENT_DETAIL,
        PRIOR_OCR_RETAINED_EVENT_DETAIL,
        _page_blob_key,
        _socr_source_digest,
        _socr_version,
    )

    winning = record.output.to_dict()
    out = dict(sidecar)
    out["disposition"] = record.disposition.to_dict()
    out["status"] = winning.get("status", "missing")
    out["failure_mode"] = winning.get("failure_mode", "none")
    out["audit_passed"] = winning.get("audit_passed", False)
    out["engine"] = winning.get("engine", "")
    out["provider"] = winning.get("provider_id", "")
    if record.verdict is not None:
        out["verdict"] = record.verdict.to_dict()
    out["winning_output"] = winning
    out["page_fingerprint"] = (
        _page_blob_key(winning) if not is_page_failed_marker(winning.get("text", "")) else ""
    )
    events = [
        e for e in sidecar.get("audit_events") or () if e.get("kind") not in _FLOOR_EVENT_KINDS
    ]
    # The two events a fresh run writes for this ending, in the order it writes them: the
    # page needed a reading, one ran and did not pass, and the page shipped its own
    # (masked) layer; and that layer is an earlier OCR run. The second names the engine
    # of the page's best attempt, which on an exact case is its one attempt.
    attempt_engine = (sidecar.get("attempts_summary") or [{}])[0].get("engine") or "native"
    if sidecar.get("needs_ocr_enhancement"):
        events.append(
            {
                "kind": "native_fallback",
                "engine": "native",
                "detail": NATIVE_FALLBACK_EVENT_DETAIL,
                "data": {},
            }
        )
    events.append(
        {
            "kind": "prior_ocr_layer_retained",
            "engine": attempt_engine,
            "detail": PRIOR_OCR_RETAINED_EVENT_DETAIL,
            "data": {},
        }
    )
    out["audit_events"] = events
    # Which code wrote these bytes, and what the run itself had shipped.
    out["refinalized_from"] = {
        "tool": "socr-refinalize-prior-ocr",
        "disposition": sidecar.get("disposition"),
        "socr_version": sidecar.get("socr_version"),
        "socr_source_digest": sidecar.get("socr_source_digest"),
    }
    out["socr_version"] = _socr_version()
    out["socr_source_digest"] = _socr_source_digest()
    return out


def _drop_from_error(error: str | None, pages: set[int]) -> str | None:
    """Remove *pages* from the leading "page(s) ... produced no usable output" clause."""
    from socr.core.result import LOST_CONTENT_NOTE

    if not error:
        return error
    head, sep, tail = error.partition("; ")
    match = _FAILED_LIST_RE.match(head)
    if not match or not match.group("note").startswith(LOST_CONTENT_NOTE):
        return error
    remaining = [n for n in (int(x) for x in match.group(1).split(", ")) if n not in pages]
    if remaining:
        head = f"page(s) {', '.join(str(n) for n in remaining)} {match.group('note')}"
        return head + sep + tail
    return tail or None


def _write_json(path: Path, data) -> None:
    """Atomically, in the writers' own format: two-space indent, UTF-8, and the trailing
    newline exactly when the file being replaced had one (``manifest.json`` does)."""
    newline = "\n" if path.is_file() and path.read_bytes().endswith(b"\n") else ""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + newline)
    tmp.replace(path)


def refinalize_document(
    doc_dir: Path, *, dry_run: bool = False, skipped: Counter | None = None
) -> list[PageChange]:
    """Re-finalize one document's floor pages; *skipped* counts the floor pages left alone,
    by reason."""
    skipped = Counter() if skipped is None else skipped
    pages_dir = doc_dir / "pages"
    meta_path = doc_dir / "metadata.json"
    if not pages_dir.is_dir() or not meta_path.is_file():
        return []
    meta = json.loads(meta_path.read_text())
    page_count = int(meta.get("pages") or 0)
    md_paths = sorted(p for p in doc_dir.glob("*.md"))
    if len(md_paths) != 1:
        # A document whose every page was floored has no ``.md`` at all; writing one is
        # the assembler's job, not this tool's. Left as it is, and said so.
        print(f"skipped {doc_dir}: {len(md_paths)} document .md files", file=sys.stderr)
        for sidecar_path in pages_dir.glob("*.json"):
            if is_floor_page(json.loads(sidecar_path.read_text())):
                skipped["document has no .md (every page floored)"] += 1
        return []
    doc_md = md_paths[0].read_text()
    changes: list[PageChange] = []
    writes: list[tuple[Path, object]] = []
    for sidecar_path in sorted(pages_dir.glob("*.json")):
        sidecar = json.loads(sidecar_path.read_text())
        if not is_floor_page(sidecar):
            continue
        if not is_exact_case(sidecar):
            skipped["attempt history not reproducible (re-run the page)"] += 1
            continue
        record = refinalize_sidecar(sidecar, page_count)
        if record is None:
            skipped["the floor is still the right ending (table seen, or no word reads)"] += 1
            continue
        fragment_path = sidecar_path.with_suffix(".md")
        old_fragment = fragment_path.read_text()
        new_text = record.output.text
        # The fragment is the winning text plus whatever line ending the writer added.
        old_text = (sidecar.get("winning_output") or {}).get("text") or ""
        if not old_fragment.startswith(old_text):
            raise RuntimeError(f"{fragment_path}: fragment does not carry its sidecar's text")
        new_fragment = new_text + old_fragment[len(old_text) :]
        if doc_md.count(old_text) != 1:
            raise RuntimeError(f"{fragment_path}: page text is not exactly once in the document")
        doc_md = doc_md.replace(old_text, new_text)
        new_sidecar = _updated_sidecar(sidecar, record)
        writes.append((fragment_path, new_fragment))
        writes.append((sidecar_path, new_sidecar))
        changes.append(
            PageChange(
                doc=doc_dir.name,
                page_num=int(sidecar["page_num"]),
                before=(
                    (sidecar.get("disposition") or {}).get("primary_reason", ""),
                    sidecar.get("status", ""),
                ),
                after=(record.disposition.primary_reason.value, new_sidecar["status"]),
                numbers_withheld=record.output.numbers_withheld,
            )
        )
    if not changes or dry_run:
        return changes
    changed = {c.page_num for c in changes}
    manifest_path = doc_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest is not None:
        from socr.core.cache import BlobStore
        from socr.core.result import PageOutput

        store = BlobStore(doc_dir / "cache")
        for path, data in writes:
            if isinstance(data, dict):
                entry = manifest["entries"][str(data["page_num"])]
                entry["blob_ref"] = store.put_page(PageOutput.from_dict(data["winning_output"]))
                entry["disposition"] = data["disposition"]
    # A document with no error carries no ``error`` key at all; a null would differ.
    error = _drop_from_error(meta.get("error"), changed)
    if error is None:
        meta.pop("error", None)
    else:
        meta["error"] = error
    audit_path = doc_dir / "audit_log.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    if audit is not None:
        events = [
            e
            for e in audit.get("events") or ()
            if not (e.get("page_num") in changed and e.get("kind") in _FLOOR_EVENT_KINDS)
        ]
        for path, data in writes:
            if isinstance(data, dict):
                for e in data["audit_events"]:
                    if e["kind"] in ("native_fallback", "prior_ocr_layer_retained"):
                        events.append({"page_num": data["page_num"], **e})
        audit["events"] = events
        audit["event_count"] = len(events)
        audit["counts"] = dict(Counter(e.get("kind") for e in events))
    for path, data in writes:
        if isinstance(data, str):
            path.write_text(data)
        else:
            _write_json(path, data)
    md_paths[0].write_text(doc_md)
    _write_json(meta_path, meta)
    # The run's root index mirrors each document's record (``RootIndex``), keyed by the
    # document's input-relative path.
    root_path = doc_dir.parent / "metadata.json"
    if root_path.is_file():
        root = json.loads(root_path.read_text())
        entry = (root.get("files") or {}).get(meta.get("key"))
        if entry is not None:
            if meta.get("error") is None:
                entry.pop("error", None)
            else:
                entry["error"] = meta["error"]
            _write_json(root_path, root)
    if audit is not None:
        _write_json(audit_path, audit)
    if manifest is not None:
        _write_json(manifest_path, manifest)
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="socr-refinalize-prior-ocr", description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="count, write nothing")
    args = parser.parse_args(argv)
    counts: Counter = Counter()
    skipped: Counter = Counter()
    withheld = 0
    for meta_path in sorted(args.out_dir.rglob("metadata.json")):
        if not (meta_path.parent / "pages").is_dir():
            continue
        for change in refinalize_document(meta_path.parent, dry_run=args.dry_run, skipped=skipped):
            counts[(change.before, change.after)] += 1
            withheld += change.numbers_withheld
    for (before, after), n in sorted(counts.items()):
        print(f"{n:6d}  {before[0]}/{before[1]} -> {after[0]}/{after[1]}")
    print(f"{sum(counts.values()):6d}  pages re-finalized; {withheld} numbers withheld")
    for reason, n in sorted(skipped.items()):
        print(f"{n:6d}  left on the floor: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
