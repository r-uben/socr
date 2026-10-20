"""#1187: give a FINISHED run's pages the figure they name and do not show. No new OCR.

The Fed run is on socr 6ebb603e and is done. A resume cannot carry this fix: the run
fingerprint carries a digest of socr's source, so any code change makes a resume re-OCR every
page. This tool recovers the work instead -- model-free, GPU-free, from the page sidecars plus
the run's own input PDFs.

The RULE is not here. ``figures.unshipped_figures.decide`` holds it, and the live lane in
``_phase_agentic`` calls the same function: this tool supplies the body from a sidecar and the
PDF from ``--pdf-dir``, and then applies the same two production finalisation steps, in the
same order, through ``manifest.refinalize_unshipped_figure_page`` -- the public seam
``_select_and_finalize_page`` itself goes through -- so the bytes it writes are the bytes a
live run would.

It rewrites, in this order: the page fragments, the document ``.md``, the manifest and its
content-addressed cache, ``metadata.json`` and the root index, the audit log -- and the page
sidecars LAST. Every step is repeatable, so a run interrupted anywhere is completed by running
the tool again: the sidecar does not yet record the lane, and the page is redone. (Crops are the
one thing written during the scan rather than at the end; a crop whose page is then not written
is an orphan PNG that the re-run overwrites with the same bytes under the same name.)

Idempotent by construction: a page whose sidecar records this lane is skipped, the guard abstains
on a body that already shows a picture, and the document note is REPLACED rather than appended.
A second run changes 0.

Run it as ``socr-refinalize-unshipped-figures <out_dir> --pdf-dir <dir> [--dry-run]``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from socr.figures.unshipped_figures import (
    EVENT_KINDS,
    MARKED_KINDS,
    NOT_FOUND_KIND,
    NOTE_PREFIX,
    RENDER_FAILED_KIND,
    UNLOCATED_KIND,
    crop_filename,
    decide,
    document_note,
    locate_figure_regions,
)

#: Sidecar keys this lane writes. A page carrying ANY of them has been answered, whatever the
#: answer was, and is never answered twice. ``unshipped_figure_captions`` is in the list and is
#: load-bearing: a page a LIVE run abstained on carries only that one, and re-examining it here
#: would run the locator again on a page already settled.
_LANE_KEYS = (
    "unshipped_figure_refs",
    "unshipped_figure_captions",
    "unshipped_figure_unlocated",
    "unshipped_figure_kind",
)

#: ``PageState`` fields the two production functions read on this path, restored from the
#: sidecar so they see what a live run would. Anything else stays at its default: this lane
#: neither selects nor re-judges, so no other field can change its answer.
_RESTORED = (
    "scanned_figure_png_ref",
    "prior_ocr_page_png_ref",
    "rotated_shred_png_ref",
    "d3_floor_png_ref",
)

#: This lane's own clause in ``metadata.json``'s ``error``, for replacing rather than
#: appending. ``[^;]*`` is exact, not approximate: ``document_note`` carries no ``;`` of its
#: own, by construction and by its own comment, so this can never swallow another lane's note.
_NOTE_RE = re.compile(re.escape(NOTE_PREFIX) + r"[^;]*")


@dataclass(frozen=True)
class PageChange:
    doc: str
    page_num: int
    kind: str
    crops: int
    captions: int

    @property
    def outcome(self) -> str:
        """What this page got. Per KIND, because "marked" covers three different things and a
        partially-shipped page both ships a crop AND is marked."""
        return {
            RENDER_FAILED_KIND: "marked: a region was located and no crop could be saved",
            UNLOCATED_KIND: "marked: no figure region could be located",
        }.get(self.kind, "crop shipped AND marked: the crops are not proven complete")

    @property
    def marked(self) -> bool:
        return self.kind in MARKED_KINDS


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    """Atomically, UTF-8 whatever the platform's default: captions carry "−", "½" and accented
    names."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, data) -> None:
    """Atomically, in the writers' own format: two-space indent, UTF-8, and the trailing newline
    exactly when the file being replaced had one (``manifest.json`` does)."""
    newline = "\n" if path.is_file() and path.read_bytes().endswith(b"\n") else ""
    _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + newline)


def _swap_once(haystack: str, old: str, new: str, where: Path) -> str:
    """Replace the one copy of *old* with *new*, or accept that an interrupted run already did.

    **``new`` is checked FIRST**, and that order is the whole correctness of this function: this
    lane only ever ADDS to a page, so ``old`` is a substring of ``new``. Checking ``old`` first
    found its copy INSIDE the ``new`` an interrupted run had already written and appended the
    block a second time (Argon, diff review).
    """
    if old == new:
        return haystack
    if haystack.count(new) == 1:
        return haystack
    if haystack.count(old) == 1:
        return haystack.replace(old, new)
    raise RuntimeError(f"{where}: the page's text is not exactly once in it")


def _merge_fragment(old_fragment: str, old_text: str, new_text: str, where: Path) -> str:
    """The page fragment with its body replaced, repeatably.

    Same ordering trap as :func:`_swap_once`, and the same fix: in round 1 ``new_text`` started
    with ``old_text`` on the branch that appends a crop, so testing ``old_text`` first
    re-appended an interrupted run's block to itself.

    Round 2 made that shape unreachable rather than merely guarded: every page this lane acts
    on is now MARKED, and writing the mark rewrites the page's leading banner block, so the new
    body no longer begins with the old one (probed on a page another lane had already marked:
    ``new.startswith(old)`` and ``old.startswith(new)`` are both False). The order is kept
    anyway, because it is the safe one and a future change that stopped marking would restore
    the ambiguity -- but the mutation that swapped it is retired from the harness rather than
    pinned by a test that could not fail. ``_swap_once`` below is NOT in that position: the
    document ``.md`` holds the page's text inside a larger file, so its own ordering guard is
    still live and still mutated.
    """
    if old_fragment.startswith(new_text):
        return old_fragment  # an interrupted run already wrote it
    if old_fragment.startswith(old_text):
        return new_text + old_fragment[len(old_text) :]
    raise RuntimeError(f"{where}: fragment does not carry its sidecar's text")


def _already_answered(sidecar: dict) -> bool:
    return any(sidecar.get(key) for key in _LANE_KEYS)


def _page_state(sidecar: dict):
    """The page state the two production functions read, as far as the sidecar records it."""
    from socr.core.state import PageState

    p = PageState(page_num=int(sidecar["page_num"]))
    for name in _RESTORED:
        value = sidecar.get(name)
        if isinstance(value, str):
            setattr(p, name, value)
    return p


def _resolve_pdf(pdf_dir: Path | None, meta: dict) -> tuple[Path | None, str | None]:
    """``(the run's own input PDF, None)``, or ``(None, why not)``.

    Accepted only for the very file the run read -- same checksum -- because this tool renders a
    crop of a page and binds it to that page's text. A different edition of the same paper would
    put a picture of one figure under another's caption, which is worse than the caption with
    nothing under it.
    """
    if pdf_dir is None:
        return None, "no --pdf-dir"
    key = str(meta.get("key") or "")
    pdf = pdf_dir / key
    if not pdf.is_file():
        matches = [c for c in pdf_dir.rglob(Path(key).name) if c.is_file()]
        if len(matches) != 1:
            return None, "PDF not found under --pdf-dir"
        pdf = matches[0]
    from ocr_output_contract import safe_checksum

    expected = str(meta.get("checksum") or "")
    if not expected or safe_checksum(pdf) != expected:
        return None, "PDF under --pdf-dir is not the run's input (checksum)"
    return pdf, None


def _published_verdict(sidecar: dict, page_num: int, new_records: tuple[dict, ...]):
    """The verdict the page must now publish: what the run recorded, plus this lane's.

    Built here rather than recomputed from the page state, and deliberately: a sidecar's verdict
    carries clauses whose inputs are not serialized (``table_label_unverified``, and slice 3's
    escalation predicate, which reads ``p.attempts``), so a recomputed verdict would come back
    cleaner than the page left -- the rule ``build_page_verdict`` already states for a resumed
    page. Merging keeps those and adds this lane's.

    De-duplicated on the WHOLE clause, not on ``(check, scope)``: two clauses can legitimately
    share a check and a region ordinal and say different things, and keying on the pair dropped
    the second (Argon, diff review). Whole-clause equality is what makes a second tool run a
    no-op without discarding anything.
    """
    from socr.core.verdict import PageVerdict, VerdictClause

    # #1187 (merge with #1185): this lane's clauses go AHEAD of a ``text_layer_numbers``
    # clause, because ``verdict._clauses_ahead_of_text_layer_numbers`` puts them there on a
    # fresh run. Appending them would order the two marks one way when this tool runs after
    # ``socr-refinalize-own-text`` and the other way round when it runs before -- and neither
    # would match a fresh run on a page both lanes touch.
    from socr.core.marks import Check as _Check

    recorded_all = (sidecar.get("verdict") or {}).get("clauses") or []
    _numbers = _Check.TEXT_LAYER_NUMBERS.value
    ahead = [r for r in recorded_all if r.get("check") != _numbers]
    behind = [r for r in recorded_all if r.get("check") == _numbers]
    recorded = ahead
    clauses: list = []
    seen: set[tuple[str, str, str, str]] = set()
    for record in [*recorded, *new_records, *behind]:
        try:
            clause = VerdictClause.from_dict(record)
        except Exception:  # noqa: BLE001 -- an unreadable record states nothing
            continue
        key = (clause.check.value, clause.result.value, clause.scope, clause.reason)
        if key in seen:
            continue
        seen.add(key)
        clauses.append(clause)
    return PageVerdict(page_num=page_num, clauses=tuple(clauses))


def refinalize_page(
    sidecar: dict, pdf: Path, figures_dir: Path, max_per_page: int, *, render: bool = True
):
    """``(the finalized output, the verdict, the decision)``, or ``None`` when this page is not
    one of this lane's.

    ``render=False`` (the ``--dry-run`` path) writes NO PNG. A dry run that rendered would leave
    the crops behind on a run that reported changing nothing, which is the one thing a dry run
    must not do. It then reports what the lane WOULD do -- one ref per located region -- and the
    only thing it cannot foresee is a render that fails, which moves a page to the mark branch.
    """
    from socr.core.manifest import (
        page_ships_its_own_page_image,
        refinalize_unshipped_figure_page,
    )
    from socr.core.result import PageOutput
    from socr.figures.chart_regions import render_chart_region_crops
    from socr.figures.unshipped_figures import body_shows_a_picture, clause_records

    if _already_answered(sidecar):
        return None
    winning = sidecar.get("winning_output")
    if not isinstance(winning, dict):
        return None
    output = PageOutput.from_dict(winning)
    body = output.text or ""
    if not body.strip() or body_shows_a_picture(body):
        return None
    page_num = int(sidecar["page_num"])
    p = _page_state(sidecar)
    if page_ships_its_own_page_image(p, body):
        return None

    def _render(bboxes) -> list[str]:
        if not render:
            return [
                f"{figures_dir.name}/{crop_filename(page_num, i)}"
                for i in range(1, len(bboxes) + 1)
            ]
        return [
            asset.rel_path
            for asset in render_chart_region_crops(
                pdf, page_num, bboxes, figures_dir, filename=crop_filename
            )
            if asset.rendered and asset.rel_path
        ]

    decision = decide(body, lambda: locate_figure_regions(pdf, page_num, max_per_page), _render)
    if decision is None or decision.kind == NOT_FOUND_KIND:
        return None
    for name, value in decision.state_fields.items():
        setattr(p, name, value)
    output, verdict = refinalize_unshipped_figure_page(
        output,
        p,
        page_num,
        published=_published_verdict(
            sidecar, page_num, clause_records(page_num, decision.unlocated, decision.kind)
        ),
    )
    return output, verdict, decision


def _updated_sidecar(sidecar: dict, output, verdict, decision) -> dict:
    from socr.core.manifest import is_page_failed_marker
    from socr.pipeline.orchestrator import _page_blob_key, _socr_source_digest, _socr_version

    winning = output.to_dict()
    out = dict(sidecar)
    out.update(decision.state_fields)
    out["status"] = winning.get("status", "missing")
    out["failure_mode"] = winning.get("failure_mode", "none")
    out["audit_passed"] = winning.get("audit_passed", False)
    out["verdict"] = verdict.to_dict()
    out["winning_output"] = winning
    out["page_fingerprint"] = (
        _page_blob_key(winning) if not is_page_failed_marker(winning.get("text", "")) else ""
    )
    events = [e for e in sidecar.get("audit_events") or () if e.get("kind") not in EVENT_KINDS]
    events.append(_event_for(int(sidecar["page_num"]), decision))
    out["audit_events"] = events
    out["refinalized_from"] = {
        "tool": "socr-refinalize-unshipped-figures",
        "status": sidecar.get("status"),
        "socr_version": sidecar.get("socr_version"),
        "socr_source_digest": sidecar.get("socr_source_digest"),
    }
    out["socr_version"] = _socr_version()
    out["socr_source_digest"] = _socr_source_digest()
    return out


def _event_for(page_num: int, decision) -> dict:
    """One audit event, in the shape ``_flush_page_sidecar`` writes.

    No ``page_num`` key: the live writer omits it from a page's own ``audit_events`` (it is
    implied by the file), and adding one made a resume rewrite the sidecar -- byte drift the
    re-finalize contract forbids (Argon P1-3c, reproduced). The DOCUMENT audit log does carry
    ``page_num``, and gets it added where the events are merged there.
    """
    return {
        "kind": decision.kind,
        "engine": "",
        "detail": decision.detail,
        "data": {**decision.event_data(), "refinalized": True},
    }


def _answered_sidecars(pages_dir: Path, pending: dict[int, dict]) -> list[dict]:
    """Every sidecar this lane has answered: what is on disk, plus what is about to be.

    The document-level records (the note, the audit log) are built from THIS, never from one
    run's changes. A run interrupted part way through a document is redone by running the tool
    again, and the pages it already answered are skipped -- so a note built from "this run's
    changes" would name only the remainder (Argon, diff review). ``pending`` is needed as well
    as the disk, because the sidecars are written LAST and are the commit point.
    """
    out: list[dict] = []
    for path in sorted(pages_dir.glob("*.json")):
        try:
            data = _read_json(path)
        except (OSError, ValueError):
            continue
        data = pending.get(int(data["page_num"]), data)
        if _already_answered(data):
            out.append(data)
    return out


def _lane_pages(answered: list[dict]) -> tuple[list[int], list[int]]:
    """``(pages that shipped a crop, pages that are marked)``."""
    shipped = [int(d["page_num"]) for d in answered if d.get("unshipped_figure_refs")]
    marked = [int(d["page_num"]) for d in answered if d.get("unshipped_figure_unlocated")]
    return shipped, marked


def refinalize_document(
    doc_dir: Path,
    *,
    dry_run: bool = False,
    skipped: Counter | None = None,
    pdf_dir: Path | None = None,
    max_per_page: int = 3,
) -> list[PageChange]:
    """Re-finalize one document's unshipped-figure pages; *skipped* counts the rest, by reason.

    Nothing but the crops is written until every page of the document has been processed.
    """
    skipped = Counter() if skipped is None else skipped
    pages_dir = doc_dir / "pages"
    meta_path = doc_dir / "metadata.json"
    if not pages_dir.is_dir() or not meta_path.is_file():
        return []
    md_paths = sorted(doc_dir.glob("*.md"))
    if len(md_paths) != 1:
        return []
    meta = _read_json(meta_path)
    pdf, why_not = _resolve_pdf(pdf_dir, meta)
    if pdf is None:
        skipped[why_not or "no PDF"] += 1
        return []
    from ocr_output_contract import figures_dir_for

    figures_dir = figures_dir_for(doc_dir)
    doc_md = md_paths[0].read_text(encoding="utf-8")
    changes: list[PageChange] = []
    fragments: list[tuple[Path, str]] = []
    sidecars: list[tuple[Path, dict]] = []
    for sidecar_path in sorted(pages_dir.glob("*.json")):
        sidecar = _read_json(sidecar_path)
        result = refinalize_page(sidecar, pdf, figures_dir, max_per_page, render=not dry_run)
        if result is None:
            continue
        output, verdict, decision = result
        fragment_path = sidecar_path.with_suffix(".md")
        if not fragment_path.is_file():
            skipped["page fragment missing"] += 1
            continue
        old_text = (sidecar.get("winning_output") or {}).get("text") or ""
        new_text = output.text or ""
        new_fragment = _merge_fragment(
            fragment_path.read_text(encoding="utf-8"), old_text, new_text, fragment_path
        )
        doc_md = _swap_once(doc_md, old_text, new_text, md_paths[0])
        fragments.append((fragment_path, new_fragment))
        sidecars.append((sidecar_path, _updated_sidecar(sidecar, output, verdict, decision)))
        changes.append(
            PageChange(
                doc=doc_dir.name,
                page_num=int(sidecar["page_num"]),
                kind=decision.kind,
                crops=len(decision.refs),
                captions=len(decision.captions),
            )
        )
    if not changes or dry_run:
        return changes
    # Write order: fragments, the document ``.md``, the manifest and its cache, the metadata
    # and the root index, the audit log -- and the page SIDECARS LAST. The sidecar is this
    # lane's commit point (``_already_answered``), so a run interrupted at any earlier step
    # leaves the page unanswered and the next run redoes it in full. Every earlier step is
    # therefore written to be repeatable: ``_merge_fragment`` and ``_swap_once`` accept the
    # text an interrupted run already wrote, and the note REPLACES its own earlier copy.
    for path, text in fragments:
        _write_text(path, text)
    _write_text(md_paths[0], doc_md)
    # The document-level records are rebuilt from EVERY answered sidecar -- those on disk and
    # those about to be written -- so they say the same thing whether this run did all the
    # pages or only the ones an interrupted run had left.
    answered = _answered_sidecars(pages_dir, {int(d["page_num"]): d for _p, d in sidecars})
    manifest_path = doc_dir / "manifest.json"
    if manifest_path.is_file():
        from socr.core.cache import BlobStore
        from socr.core.result import PageOutput

        from socr.core.manifest import page_asset_refs

        manifest = _read_json(manifest_path)
        store = BlobStore(doc_dir / "cache")
        for _path, data in sidecars:
            entry = (manifest.get("entries") or {}).get(str(data["page_num"]))
            if entry is None:
                continue
            page = PageOutput.from_dict(data["winning_output"])
            entry["blob_ref"] = store.put_page(page)
            # #1187 round 2 (Argon P1-3a): a page this tool gave a PNG had no asset record,
            # while pages another lane gave one did. ``manifest.stale_assets`` therefore never
            # checked the new crops, and the next resume would add them -- drift the tool
            # introduced. Recorded with the SAME function ``build_manifest`` uses.
            refs = page_asset_refs(page.text or "", doc_dir)
            if refs:
                entry["assets"] = [
                    {"logical_path": r.logical_path, "content_hash": r.content_hash} for r in refs
                ]
        _write_json(manifest_path, manifest)
    note = document_note(*_lane_pages(answered))
    error = _NOTE_RE.sub("", str(meta.get("error") or "")).strip().strip(";").strip()
    meta["error"] = f"{error}; {note}" if error and note else (note or error) or None
    if meta["error"] is None:
        meta.pop("error", None)
    _write_json(meta_path, meta)
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
        kept = [e for e in audit.get("events") or () if e.get("kind") not in EVENT_KINDS]
        for data in answered:
            # The document log keys every event by page; the per-page sidecar does not (see
            # ``_event_for``). The key is added HERE, where the two shapes meet.
            kept.extend(
                {"page_num": int(data["page_num"]), **e}
                for e in data.get("audit_events") or ()
                if e.get("kind") in EVENT_KINDS
            )
        audit["events"] = kept
        audit["event_count"] = len(kept)
        audit["counts"] = dict(Counter(e.get("kind") for e in kept))
        _write_json(audit_path, audit)
    for path, data in sidecars:
        _write_json(path, data)
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="socr-refinalize-unshipped-figures", description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--pdf-dir", type=Path, default=None, help="the run's input PDFs")
    parser.add_argument("--dry-run", action="store_true", help="count, write nothing")
    parser.add_argument(
        "--max-per-page",
        type=int,
        default=3,
        help="crops per page; socr's own figures_max_per_page default",
    )
    args = parser.parse_args(argv)
    counts: Counter = Counter()
    skipped: Counter = Counter()
    failed: list[tuple[Path, str]] = []
    crops = 0
    for meta_path in sorted(args.out_dir.rglob("metadata.json")):
        if not (meta_path.parent / "pages").is_dir():
            continue
        try:
            changes = refinalize_document(
                meta_path.parent,
                dry_run=args.dry_run,
                skipped=skipped,
                pdf_dir=args.pdf_dir,
                max_per_page=args.max_per_page,
            )
        except Exception as exc:  # noqa: BLE001 -- reported per document, never swallowed
            failed.append((meta_path.parent, f"{type(exc).__name__}: {exc}"))
            continue
        for change in changes:
            counts[change.outcome] += 1
            crops += change.crops
    for outcome, n in sorted(counts.items()):
        print(f"{n:6d}  {outcome}")
    print(f"{sum(counts.values()):6d}  pages re-finalized; {crops} crop(s) rendered")
    for reason, n in sorted(skipped.items()):
        print(f"{n:6d}  left alone: {reason}")
    for doc_dir, why in failed:
        print(f"FAILED  {doc_dir}: {why}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
