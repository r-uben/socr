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
#: Events a fresh run of the masked ending writes for the page.
_ENDING_EVENT_KINDS = frozenset({"native_fallback", "prior_ocr_layer_retained"})

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
#: ``prior_ocr_table_seen`` is the live gate's MEMO, kept so one run cannot flip a page
#: between two finalisations. Restored here it would answer the gate before any signal is
#: read (Argon, round 2 re-review), so the tool recomputes the answer from every signal.
_NOT_RESTORED = frozenset({"page_num", "timings_s", "page_cost_usd", "prior_ocr_table_seen"})

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
    from socr.core.result import FailureMode
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
        if value is not None and "FailureMode" in str(f.type):
            # The one enum-typed field (``table_ladder_disposition``). Restored as the
            # enum, as resume restores it: the ladder guard compares by identity, and a
            # plain string silently never matched (Argon P1-1, reproduced).
            try:
                value = FailureMode(value)
            except ValueError:
                continue
        if default is None or isinstance(value, type(default)):
            setattr(p, f.name, value)
    p.native_text = sidecar.get("prior_ocr_layer_text") or ""
    p.prior_ocr_layer_suspect = True
    # The attempt record, which says of each reading whether it held a table: the gate's
    # one signal that lives in the readings, which the sidecar does not keep.
    p.attempts_summary_restored = [dict(row) for row in sidecar.get("attempts_summary") or []]
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


#: The native-table distrust flags. On a page with any of them a flagged model table can be
#: kept, and the structure-class lookahead then declines -- the one way a table page could
#: have reached a floor this tool replaces (see ``is_exact_case``).
_TABLE_DISTRUST_FLAGS = (
    "native_table_structure_failed",
    "native_table_unverifiable",
    "native_table_structure_defective",
    "native_table_header_unattributed",
)


#: The page assessment's table findings that the masked ending's gate reads. Sidecars
#: written since #1156 round 2 carry them; an older sidecar does not, and they are then
#: recomputed from the PDF by socr's own page analysis -- no OCR, no model -- or the page
#: is left on its floor. Measured on the Fed calibration run: 4 of the 345 pages the tool
#: would otherwise mask carry ``possible_table_structure_not_reconstructed`` (a two-column
#: label and value layout), which only the PDF can show.
_TABLE_STANDING_KEYS = (
    "has_tables",
    "possible_table_structure_not_reconstructed",
    "table_not_reconstructed",
)


def _with_table_standing(sidecar: dict, assessment) -> dict | None:
    """*sidecar* with the table findings filled in, from itself or from *assessment*
    (the document's page analysis), or ``None`` when neither has them."""
    if all(key in sidecar for key in _TABLE_STANDING_KEYS):
        return sidecar
    if assessment is None:
        return None
    pa = assessment.pages[int(sidecar["page_num"]) - 1]
    out = dict(sidecar)
    for key in _TABLE_STANDING_KEYS:
        out[key] = bool(getattr(pa, key, False))
    # Sparse, as a run writes it (Argon, round 2 re-review).
    count = int(getattr(pa, "detected_table_count", 0) or 0)
    if count > 0:
        out["detected_table_count"] = count
    return out


def _analyse_pdf(pdf_dir: Path | None, meta: dict, expected_checksum: str):
    """``(the document's page analysis, None)``, or ``(None, why not)``.

    The analysis decides whether a table page ships its rows, so it is accepted only for
    the very file the run read: same checksum, same page count (Fable / Argon, round 2).
    Model-free: socr's own born-digital page analysis, no OCR.
    """
    if pdf_dir is None:
        return None, "table findings not in the sidecar, and no --pdf-dir"
    pdf = pdf_dir / str(meta.get("key") or "")
    if not pdf.is_file():
        return None, "PDF not found under --pdf-dir"
    from ocr_output_contract import safe_checksum

    if not expected_checksum or safe_checksum(pdf) != expected_checksum:
        return None, "PDF under --pdf-dir is not the run's input (checksum)"
    from socr.core.born_digital import BornDigitalDetector

    assessment = BornDigitalDetector().detect(pdf)
    if len(assessment.pages) != int(meta.get("pages") or 0):
        return None, "PDF under --pdf-dir is not the run's input (page count)"
    return assessment, None


def _cached_readings(doc_dir: Path) -> dict[int, list[dict]]:
    """Every model reading the run kept in the document's content-addressed cache, by page.

    The cache stores each attempt's whole output, rejected ones included; the sidecar keeps
    only a summary. Measured on the Fed calibration run: each of its 390 floor pages has its
    one rejected reading there, and 23 of those readings hold a table.
    """
    out: dict[int, list[dict]] = {}
    for blob in sorted((doc_dir / "cache").glob("*/*.json")):
        try:
            data = _read_json(blob)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("page_num"), int) and "text" in data:
            out.setdefault(data["page_num"], []).append(data)
    return out


def _judged(sidecar: dict) -> bool:
    rows = sidecar.get("attempts_summary") or []
    return bool(rows) and all("held_a_table" in row for row in rows)


def _with_readings_judged(sidecar: dict, cached: dict[int, list[dict]]) -> dict | None:
    """*sidecar* with ``held_a_table`` on its attempt record, from itself (a run since #1156
    round 3 writes it) or from the rejected reading in the cache; ``None`` when neither
    can say. On an exact case the record holds one model attempt, so the reading is the
    page's one rejected cached output of that engine."""
    from socr.core.manifest import reading_holds_a_table
    from socr.core.result import PageOutput

    rows = sidecar.get("attempts_summary") or []
    if _judged(sidecar):
        return sidecar
    if len(rows) != 1:
        return None
    engine = rows[0].get("engine")
    found = [
        r
        for r in cached.get(int(sidecar["page_num"]), [])
        if r.get("engine") == engine and r.get("audit_passed") is False
    ]
    if len(found) != 1:
        return None
    out = dict(sidecar)
    out["attempts_summary"] = [
        {**rows[0], "held_a_table": reading_holds_a_table(PageOutput.from_dict(found[0]))}
    ]
    return out


def is_exact_case(sidecar: dict) -> bool:
    """A terminal page with exactly one MODEL attempt, judged to completion, whose
    structure-class standing is known.

    A sidecar written before #1156 round 2 does not carry ``has_tables`` (Argon P0-1), and
    it is not inferred from anything else: the page qualifies only when the live run's own
    selection proves it was not a table page. With a model attempt and no native-table
    distrust flag, ``_reaches_structure_class_branch`` returns True for every table page,
    so a table page could never have ended on either floor; ending there is the proof.
    A sidecar that does carry it is restored like every other flag, and the table gate
    reads it -- so the proof is required either way, which only ever keeps a page on its
    floor.

    The proof is NOT sufficient on its own, and nothing may treat it as such (Fable,
    round 2 P2-1): the lookahead returns False for a page that is not born-digital (a
    #1203 page), whatever its tables, so such a table page reaches a floor too. That is
    why the table findings (``_with_table_standing``) are required on every page, from the
    sidecar or from the run's own PDF, and the page is skipped when neither has them.
    """
    attempts = sidecar.get("attempts_summary") or []
    if not (
        sidecar.get("terminal") is True
        and len(attempts) == 1
        and attempts[0].get("judge_outcome") in _EXACT_JUDGE_OUTCOMES
        and not str(attempts[0].get("engine") or "").startswith("native")
    ):
        return False
    return not any(sidecar.get(flag) for flag in _TABLE_DISTRUST_FLAGS)


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
    # The table gate's answer, as a fresh run persists it: the masked ending is reached
    # only when the gate said "no table" (Argon, round 2 re-review).
    out["prior_ocr_table_seen"] = False
    out["page_fingerprint"] = (
        _page_blob_key(winning) if not is_page_failed_marker(winning.get("text", "")) else ""
    )
    # The floor's events go, and so does any older copy of the two this ending writes:
    # one event of each kind per page, as a fresh run writes it.
    events = [
        e
        for e in sidecar.get("audit_events") or ()
        if e.get("kind") not in _FLOOR_EVENT_KINDS | _ENDING_EVENT_KINDS
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


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    """Atomically, UTF-8 whatever the platform's default: the layers carry "−", "½"
    and accented names (Fable P2-3)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, data) -> None:
    """Atomically, in the writers' own format: two-space indent, UTF-8, and the trailing
    newline exactly when the file being replaced had one (``manifest.json`` does)."""
    newline = "\n" if path.is_file() and path.read_bytes().endswith(b"\n") else ""
    _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + newline)


def _swap_once(haystack: str, old: str, new: str, where: Path) -> str:
    """Replace the one copy of *old* with *new*, or accept that an interrupted run already
    did: every step of the rewrite must be repeatable, because the page sidecar is written
    LAST and a page whose sidecar still names the floor is redone in full (Fable P2-4)."""
    if haystack.count(old) == 1:
        return haystack.replace(old, new)
    if old not in haystack and haystack.count(new) == 1:
        return haystack
    raise RuntimeError(f"{where}: the page's text is not exactly once in it")


def refinalize_document(
    doc_dir: Path,
    *,
    dry_run: bool = False,
    skipped: Counter | None = None,
    pdf_dir: Path | None = None,
) -> list[PageChange]:
    """Re-finalize one document's floor pages; *skipped* counts the floor pages left alone,
    by reason.

    Write order: page fragments, the document ``.md``, the manifest and its cache, the
    metadata, the root index, the audit log -- and the page sidecars LAST. Each step is
    repeatable, so a run interrupted anywhere is completed by running the tool again: the
    sidecar still names the floor, and the page is redone.
    """
    skipped = Counter() if skipped is None else skipped
    pages_dir = doc_dir / "pages"
    meta_path = doc_dir / "metadata.json"
    if not pages_dir.is_dir() or not meta_path.is_file():
        return []
    meta = _read_json(meta_path)
    page_count = int(meta.get("pages") or 0)
    md_paths = sorted(p for p in doc_dir.glob("*.md"))
    if len(md_paths) != 1:
        # A document whose every page was floored has no ``.md`` at all; writing one is
        # the assembler's job, not this tool's. Left as it is, and counted.
        for sidecar_path in pages_dir.glob("*.json"):
            if is_floor_page(_read_json(sidecar_path)):
                skipped["document has no .md (every page floored)"] += 1
        return []
    doc_md = md_paths[0].read_text(encoding="utf-8")
    assessment: list = []  # the document's page analysis, run once and only if needed

    def page_analysis(expected_checksum: str):
        if not assessment:
            assessment.append(_analyse_pdf(pdf_dir, meta, expected_checksum))
        return assessment[0]

    cache: list = []  # the document's cached readings, read once and only if needed

    def cached_readings():
        if not cache:
            cache.append(_cached_readings(doc_dir))
        return cache[0]

    changes: list[PageChange] = []
    fragments: list[tuple[Path, str]] = []
    sidecars: list[tuple[Path, dict]] = []
    for sidecar_path in sorted(pages_dir.glob("*.json")):
        sidecar = _read_json(sidecar_path)
        if not is_floor_page(sidecar):
            continue
        if not is_exact_case(sidecar):
            skipped["attempt history or table standing not reproducible (re-run the page)"] += 1
            continue
        standing = _with_table_standing(sidecar, None)
        if standing is None:
            analysis, why_not = page_analysis(
                str(sidecar.get("input_checksum") or meta.get("checksum") or "")
            )
            if analysis is None:
                skipped[why_not] += 1
                continue
            standing = _with_table_standing(sidecar, analysis)
        recomputed = standing is not sidecar
        judged = _with_readings_judged(standing, {} if _judged(standing) else cached_readings())
        if judged is None:
            skipped["the rejected reading is not in the run's cache"] += 1
            continue
        from_cache = judged is not standing
        sidecar = judged
        record = refinalize_sidecar(sidecar, page_count)
        if record is None:
            skipped["the floor is still the right ending (a table, or no prose)"] += 1
            continue
        fragment_path = sidecar_path.with_suffix(".md")
        old_fragment = fragment_path.read_text(encoding="utf-8")
        new_text = record.output.text
        old_text = (sidecar.get("winning_output") or {}).get("text") or ""
        if old_fragment.startswith(old_text):
            new_fragment = new_text + old_fragment[len(old_text) :]
        elif old_fragment.startswith(new_text):
            new_fragment = old_fragment  # an interrupted run already wrote it
        else:
            raise RuntimeError(f"{fragment_path}: fragment does not carry its sidecar's text")
        doc_md = _swap_once(doc_md, old_text, new_text, md_paths[0])
        new_sidecar = _updated_sidecar(sidecar, record)
        if recomputed:
            new_sidecar["refinalized_from"]["table_findings"] = "recomputed from the PDF"
        if from_cache:
            new_sidecar["refinalized_from"]["readings"] = "judged from the run's cache"
        fragments.append((fragment_path, new_fragment))
        sidecars.append((sidecar_path, new_sidecar))
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
    for path, text in fragments:
        _write_text(path, text)
    _write_text(md_paths[0], doc_md)
    manifest_path = doc_dir / "manifest.json"
    if manifest_path.is_file():
        from socr.core.cache import BlobStore
        from socr.core.result import PageOutput

        manifest = _read_json(manifest_path)
        store = BlobStore(doc_dir / "cache")
        for _path, data in sidecars:
            entry = manifest["entries"][str(data["page_num"])]
            entry["blob_ref"] = store.put_page(PageOutput.from_dict(data["winning_output"]))
            entry["disposition"] = data["disposition"]
        _write_json(manifest_path, manifest)
    # A document with no error carries no ``error`` key at all; a null would differ.
    error = _drop_from_error(meta.get("error"), changed)
    if error is None:
        meta.pop("error", None)
    else:
        meta["error"] = error
    _write_json(meta_path, meta)
    # The run's root index mirrors each document's record (``RootIndex``), keyed by the
    # document's input-relative path.
    root_path = doc_dir.parent / "metadata.json"
    if root_path.is_file():
        root = _read_json(root_path)
        entry = (root.get("files") or {}).get(meta.get("key"))
        if entry is not None:
            if meta.get("error") is None:
                entry.pop("error", None)
            else:
                entry["error"] = meta["error"]
            _write_json(root_path, root)
    audit_path = doc_dir / "audit_log.json"
    if audit_path.is_file():
        audit = _read_json(audit_path)
        events = [
            e
            for e in audit.get("events") or ()
            if not (e.get("page_num") in changed and e.get("kind") in _FLOOR_EVENT_KINDS)
        ]
        for _path, data in sidecars:
            for e in data["audit_events"]:
                if e["kind"] in _ENDING_EVENT_KINDS:
                    # One event of each kind per page, as a fresh run writes it: an older
                    # copy (another engine, another detail) is replaced, not kept beside it.
                    events = [
                        old
                        for old in events
                        if not (
                            old.get("page_num") == data["page_num"] and old.get("kind") == e["kind"]
                        )
                    ]
                    events.append({"page_num": data["page_num"], **e})
        audit["events"] = events
        audit["event_count"] = len(events)
        audit["counts"] = dict(Counter(e.get("kind") for e in events))
        _write_json(audit_path, audit)
    for path, data in sidecars:
        _write_json(path, data)
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="socr-refinalize-prior-ocr", description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="count, write nothing")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="the run's input PDFs, for sidecars older than #1156 round 2: their table "
        "findings are recomputed by socr's page analysis (no OCR, no model)",
    )
    args = parser.parse_args(argv)
    counts: Counter = Counter()
    skipped: Counter = Counter()
    failed: list[tuple[Path, str]] = []
    withheld = 0
    for meta_path in sorted(args.out_dir.rglob("metadata.json")):
        if not (meta_path.parent / "pages").is_dir():
            continue
        # One inconsistent document is reported and the sweep goes on (Argon P1-1, Fable
        # P2-3): its pages are untouched, because nothing is written until every page of
        # the document has been checked.
        try:
            changes = refinalize_document(
                meta_path.parent, dry_run=args.dry_run, skipped=skipped, pdf_dir=args.pdf_dir
            )
        except Exception as exc:  # noqa: BLE001 -- reported per document, never swallowed
            failed.append((meta_path.parent, f"{type(exc).__name__}: {exc}"))
            continue
        for change in changes:
            counts[(change.before, change.after)] += 1
            withheld += change.numbers_withheld
    for (before, after), n in sorted(counts.items()):
        print(f"{n:6d}  {before[0]}/{before[1]} -> {after[0]}/{after[1]}")
    print(f"{sum(counts.values()):6d}  pages re-finalized; {withheld} numbers withheld")
    for reason, n in sorted(skipped.items()):
        print(f"{n:6d}  left on the floor: {reason}")
    for doc_dir, why in failed:
        print(f"FAILED  {doc_dir}: {why}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
