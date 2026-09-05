"""TICKET-A1 (verifier-independence): replay `bind()` against frozen corpus output.

## Why this exists

`tables/binding.py:bind()` can only be exercised through a live pipeline
run today, so a binder change (A2's row-label repair) can only be measured
by re-running the free local OCR model end to end. This module replays the
mechanical binding check offline: it takes a page a run already scored,
re-derives the candidate table markdown and the native word layer, and
re-runs the CURRENT tree's `bind()` against them — without touching an
engine, a judge, or a network call.

## What "replay" means here

Nothing about a run is deterministic to reproduce except the two inputs
`bind()` actually consumes: the native PDF word layer (fixed — the PDF is
frozen) and the model's candidate table markdown (fixed — it was already
emitted and saved). Re-running `bind()` on the SAME two inputs with a
DIFFERENT tree is exactly the A2 experiment this harness exists for.

## Recovering the two inputs from a frozen corpus

A corpus directory (e.g. `~/Data/socr/ladder-run2-2026-09-04`) has this
layout, produced by a `run.sh` that called `socr` per document:

    in/<slug>.pdf                                   -- the frozen source PDF
    out/<slug>/<slug>/pages/NNNNN.json               -- one sidecar per page

`page_num` inside a sidecar is 1-indexed **into `in/<slug>.pdf`** (the
per-corpus extracted subset PDF), not the original document's page number —
confirmed by cross-checking `doc02`'s `manifest.json` page list length
against `in/doc02.pdf`'s page count.

**The model candidate markdown is NOT the per-table `witness.markdown`
`bind()` was actually called with** -- that block-scoped string is never
persisted. What is USUALLY persisted is `winning_output.text`, the model's
full emitted page markdown. `tables/witness.py:prepare_table_witnesses`
derives the SAME per-table blocks and per-table located boxes from a full
page's markdown, keyed as `p{page_num}-t{idx}` -- the same `table_id`
scheme `binding_adjudication` is keyed by. So replay re-derives
`witness.markdown` and `witness.box.bbox` from `winning_output.text` with
the current tree's witness/locate code, then calls `bind()` exactly as
`_binding_evidence_for_witness` does.

**`winning_output.text` is sometimes NOT the text `bind()` saw.** When a
table later fails closed, `manifest.py` overwrites the shipped text with
the D3 fail-closed marker (`f"[page {page_num} failed: unverifiable table
— see image]"`, `disposition.ending == "fail_closed_marker"`) -- this
happened to `doc01` page 2 in the frozen run-2 corpus. The marker carries
no table block at all, so the sidecar's OWN `winning_output.text` cannot
reproduce the binding.

`_select_candidate_for_table` recovers it from the page's own
`cache/*.json` route/extract entries, by **provenance only, never by
running `bind()`**: it collects EVERY `table_binding_adjudicated` audit
event recorded for this exact `table_id` (`judge.table_verdict
.TABLE_BINDING_ADJUDICATED_KIND`; each carries a winning `engine`) --
distinct engine VALUES across those events, not just the first event, are
what "provenance" means here -- then looks for the cache entries on this
`page_num` whose own `engine` field matches. Scoring a candidate by how
well it reproduces the recorded contradiction was tried and rejected in
review: that would let `bind()` choose its own input, so a change to
`bind()` under test (A2's row-label repair) could silently swap which
candidate is treated as ground truth out from under the regression it is
supposed to be measured against. Exactly one provenance engine AND exactly
one cache candidate for it -> used, noted. Two or more `table_binding
_adjudicated` events naming DIFFERENT engines (provenance itself
ambiguous), or zero/more-than-one distinct cache candidate for the single
agreed engine -> the row is `unreplayable` (`ReplayRow.unreplayable`) and
`bind()` is never called for it.

A non-empty note from `replay_table` (table_id missing among this
tree's witnesses, witness not LOCATED, or the page has no native words)
is the same outcome: `replay_page` marks the row `unreplayable` with
that note, the same path provenance ambiguity already uses. An empty
fresh side is NEVER compared against the frozen record as a binder
delta — that would report every recorded item as cleared.

**The persisted `binding_adjudication[<table_id]].items[]` records do NOT
carry `native_bbox`** (`adjudication.ContradictionItem.to_record` omits
it) -- so replay can only compare `(kind, native_token, model_token)` as a
multiset, duplicate counts preserved. That is exactly what
TICKET-A1 asks for; it is also why label accuracy / crop coverage (which
need a bbox) can only be computed from a FRESH re-bind, never from the
frozen record alone.

## What this module does NOT do

No hand-read label file exists yet (TICKET-A1b writes
`~/Data/socr/ladder-run2-2026-09-04/labels.json`, outside git). Label
accuracy and crop coverage are reported as unavailable until that file is
supplied via `--labels`. Crop rendering reuses
`orchestrator.UnifiedPipeline._render_adjudication_crop` UNMODIFIED, called
unbound (the method never reads `self`) so this module never imports a
whole pipeline config.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from socr.core.pdf import open_pdf
from socr.judge.table_verdict import TABLE_BINDING_ADJUDICATED_KIND
from socr.tables.adjudication import ContradictionItem, items_from_binding
from socr.tables.binding import bind
from socr.tables.witness import TableWitness, WitnessStatus, prepare_table_witnesses

#: (kind, native_token, model_token) — the only fields the persisted sidecar
#: record carries (see module docstring). Duplicate counts preserved.
ReplayItemKey = tuple[str, str, str]


def _item_key(item: ContradictionItem) -> ReplayItemKey:
    return (item.kind, item.native_token, item.model_token)


def _recorded_item_key(record: dict) -> ReplayItemKey:
    return (
        str(record.get("kind", "")),
        str(record.get("native_token", "")),
        str(record.get("model_token", "")),
    )


@dataclass(frozen=True)
class ReplayRow:
    """One table's fresh-vs-frozen comparison."""

    doc_slug: str
    page_num: int
    table_id: str
    recorded_status: str
    recorded_item_count: int
    fresh_item_count: int
    multiset_match: bool
    #: Present in the fresh bind() but not in the frozen record.
    added: tuple[ReplayItemKey, ...]
    #: Present in the frozen record but not in the fresh bind().
    removed: tuple[ReplayItemKey, ...]
    #: Derived from the GH-359 ruling 5 clamp rule alone (lifted -> not
    #: clamped, held -> clamped UNVERIFIED). Not the full page disposition —
    #: the guard chain (P1) can additionally exempt a table this module
    #: cannot see from the sidecar's `binding_adjudication` alone.
    final_disposition: str
    label_accuracy: str
    crop_coverage: str
    #: True iff ``bind()`` did not produce a comparable fresh side —
    #: provenance could not identify a single candidate, or ``replay_table``
    #: returned a non-empty note (table_id missing among witnesses, witness
    #: not LOCATED, no native words). Every fresh_* field is a placeholder,
    #: not a result; added/removed stay empty (never a binder delta).
    unreplayable: bool = False
    note: str = ""


@dataclass(frozen=True)
class PageRecord:
    """One frozen sidecar with a non-empty ``binding_adjudication``."""

    doc_slug: str
    page_num: int
    sidecar_path: Path
    pdf_path: Path
    cache_dir: Path
    model_markdown: str
    is_fail_closed_marker: bool
    #: table_id -> the DISTINCT ``engine`` values across every
    #: ``table_binding_adjudicated`` audit event recorded for that exact
    #: table_id (empty frozenset when no such event exists). The only
    #: provenance signal candidate selection uses -- more than one distinct
    #: value means provenance itself is ambiguous, not just the cache side.
    provenance_engines_by_table: dict = field(default_factory=dict)
    binding_adjudication: dict = field(default_factory=dict)


def discover_pages(corpus_dir: Path) -> list[PageRecord]:
    """Find every frozen page sidecar under *corpus_dir* that recorded a
    binding adjudication. Never opens a PDF or calls ``bind()`` — pure
    discovery, so a missing PDF fails per-row, not here."""
    records: list[PageRecord] = []
    out_dir = corpus_dir / "out"
    if not out_dir.is_dir():
        return records
    for sidecar_path in sorted(out_dir.glob("*/*/pages/*.json")):
        doc_slug = sidecar_path.parents[2].name
        data = json.loads(sidecar_path.read_text())
        adjudication = data.get("binding_adjudication") or {}
        if not adjudication:
            continue
        winning = data.get("winning_output") or {}
        model_markdown = winning.get("text", "")
        page_num = int(data.get("page_num") or 0)
        pdf_path = corpus_dir / "in" / f"{doc_slug}.pdf"
        records.append(
            PageRecord(
                doc_slug=doc_slug,
                page_num=page_num,
                sidecar_path=sidecar_path,
                pdf_path=pdf_path,
                cache_dir=sidecar_path.parents[1] / "cache",
                model_markdown=model_markdown,
                is_fail_closed_marker=_is_fail_closed_marker(model_markdown, page_num),
                provenance_engines_by_table=_provenance_engines_by_table(data, adjudication),
                binding_adjudication=adjudication,
            )
        )
    return records


def _provenance_engines_by_table(sidecar: dict, adjudication: dict) -> dict:
    """table_id -> the frozenset of DISTINCT ``engine`` values across every
    ``table_binding_adjudicated`` audit event recorded for that exact
    table_id. Every matching event is collected, not just the first: two
    events naming different engines means provenance ITSELF is ambiguous
    for that table, which is a different (and prior) question from whether
    the cache side later resolves to one candidate — repeating the SAME
    engine collapses to one element, which is not ambiguous."""
    by_table: dict = {table_id: set() for table_id in adjudication}
    for event in sidecar.get("audit_events") or []:
        if not isinstance(event, dict) or event.get("kind") != TABLE_BINDING_ADJUDICATED_KIND:
            continue
        table_id = (event.get("data") or {}).get("table_id")
        engine = event.get("engine") or None
        if table_id in by_table and engine:
            by_table[table_id].add(engine)
    return {table_id: frozenset(engines) for table_id, engines in by_table.items()}


def _is_fail_closed_marker(text: str, page_num: int) -> bool:
    """True iff *text* is the D3 fail-closed placeholder
    (``manifest.py:937,1291,1365``) rather than emitted table markdown —
    the case where ``winning_output.text`` cannot reproduce a binding."""
    return f"[page {page_num} failed: unverifiable table" in text


def _cache_candidate_texts(cache_dir: Path, page_num: int) -> list[tuple[str, str, str]]:
    """``(source_label, engine, text)`` for every cached route/extract
    attempt on *page_num* whose own text is not itself a fail-closed
    marker, sorted by cache filename for determinism. Route/extract cache
    entries are keyed by content hash, not page, so every ``*.json`` under
    *cache_dir* is opened and filtered by its own ``page_num`` field; never
    raises on a malformed cache entry."""
    if not cache_dir.is_dir():
        return []
    candidates: list[tuple[str, str, str]] = []
    for cache_path in sorted(cache_dir.glob("*/*.json")):
        try:
            entry = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(entry, dict) or entry.get("page_num") != page_num:
            continue
        text = entry.get("text")
        if not text or _is_fail_closed_marker(text, page_num):
            continue
        candidates.append(
            (f"cache/{cache_path.parent.name}/{cache_path.name}", entry.get("engine") or "", text)
        )
    return candidates


def _select_candidate_for_table(record: PageRecord, table_id: str) -> tuple[str | None, str]:
    """The model markdown to replay *table_id* against, chosen BY
    PROVENANCE ONLY -- never by calling ``bind()`` (see module docstring).

    Returns ``(text, note)``. ``text`` is ``None`` iff the table is
    unreplayable (no provenance engine recorded, or the cache side is
    empty / ambiguous for that engine) -- callers MUST NOT call ``bind()``
    in that case.
    """
    if not record.is_fail_closed_marker:
        return record.model_markdown, ""

    engines = record.provenance_engines_by_table.get(table_id, frozenset())
    if len(engines) == 0:
        return (
            None,
            "winning_output.text is the D3 fail-closed marker and no "
            f"table_binding_adjudicated audit event recorded a winning "
            f"engine for {table_id!r}; unreplayable",
        )
    if len(engines) > 1:
        return (
            None,
            "winning_output.text is the D3 fail-closed marker and "
            f"table_binding_adjudicated audit events for {table_id!r} name "
            f"conflicting engines {sorted(engines)!r}; provenance itself is "
            f"ambiguous; unreplayable",
        )
    (engine,) = engines

    candidates = _cache_candidate_texts(record.cache_dir, record.page_num)
    matches = [(source, text) for source, cand_engine, text in candidates if cand_engine == engine]
    distinct_texts = {text for _source, text in matches}

    if len(distinct_texts) == 0:
        return (
            None,
            "winning_output.text is the D3 fail-closed marker; no cache "
            f"candidate on this page has provenance engine {engine!r}; unreplayable",
        )
    if len(distinct_texts) > 1:
        sources = ", ".join(source for source, _text in matches)
        return (
            None,
            f"winning_output.text is the D3 fail-closed marker; "
            f"{len(distinct_texts)} distinct cache candidates share "
            f"provenance engine {engine!r} ({sources}); ambiguous, unreplayable",
        )
    source, text = matches[0]
    return (
        text,
        f"winning_output.text is the D3 fail-closed marker; substituted "
        f"{source} by provenance (engine={engine!r})",
    )


def _witness_for_table(witnesses: list[TableWitness], table_id: str) -> TableWitness | None:
    for witness in witnesses:
        if witness.table_id == table_id:
            return witness
    return None


def replay_table(
    pdf_path: Path,
    page_num: int,
    model_markdown: str,
    table_id: str,
) -> tuple[tuple[ContradictionItem, ...], str]:
    """Re-derive the table's witness (block markdown + located box) from
    *model_markdown* with the CURRENT tree, then re-run ``bind()`` against
    the native words on *page_num* of *pdf_path*.

    Returns ``(items, note)``. ``note`` is empty on a clean replay and
    otherwise names why nothing could be bound (no such table_id, table not
    LOCATED this tree, or the page has no native words) — never raises.
    Callers MUST treat a non-empty note as unreplayable: never compare
    the empty ``items`` against the frozen record as a binder delta.
    """
    with open_pdf(pdf_path) as doc:
        words = doc[page_num - 1].get_text("words")

    with prepare_table_witnesses(pdf_path, page_num, model_markdown) as witnesses:
        witness = _witness_for_table(witnesses, table_id)
        if witness is None:
            return (), f"table_id {table_id!r} not found among this tree's witnesses"
        if witness.status is not WitnessStatus.LOCATED or witness.box is None:
            return (), f"witness status {witness.status.value} (no located box this tree)"
        if not words:
            return (), "no native words on this page"
        binding_result = bind(words, witness.markdown, region=witness.box.bbox)
    return items_from_binding(binding_result), ""


def _final_disposition(status: str) -> str:
    if status == "lifted":
        return "ACCEPTED (binding lifted — GH-359 ruling 5 clamp released)"
    return "UNVERIFIED (binding held — GH-359 ruling 5 clamp applies)"


def _unreplayable_row(
    record: PageRecord,
    table_id: str,
    recorded: dict,
    recorded_item_count: int,
    note: str,
    label_accuracy: str,
    crop_coverage: str,
) -> ReplayRow:
    return ReplayRow(
        doc_slug=record.doc_slug,
        page_num=record.page_num,
        table_id=table_id,
        recorded_status=str(recorded.get("status", "")),
        recorded_item_count=recorded_item_count,
        fresh_item_count=0,
        multiset_match=False,
        added=(),
        removed=(),
        final_disposition=_final_disposition(str(recorded.get("status", ""))),
        label_accuracy=label_accuracy,
        crop_coverage=crop_coverage,
        unreplayable=True,
        note=note,
    )


def replay_page(record: PageRecord, labels: dict | None) -> list[ReplayRow]:
    rows: list[ReplayRow] = []
    for table_id, recorded in sorted(record.binding_adjudication.items()):
        recorded_items = recorded.get("items") or []
        recorded_counter = Counter(_recorded_item_key(r) for r in recorded_items)

        candidate_markdown, note = _select_candidate_for_table(record, table_id)

        label_accuracy = "n/a (no --labels file supplied; TICKET-A1b writes it)"
        crop_coverage = "n/a (no --labels file supplied; TICKET-A1b writes it)"
        if labels is not None:
            key = f"{record.doc_slug}:{table_id}"
            if key not in labels:
                label_accuracy = "n/a (no hand-read label for this table)"
                crop_coverage = "n/a (no hand-read label for this table)"
            else:
                label_accuracy = "see TICKET-A1b autopsy log (not scored here)"
                crop_coverage = "see TICKET-A1b autopsy log (not scored here)"

        if candidate_markdown is None:
            # Provenance could not identify a single candidate -- bind() is
            # NEVER called for this row (see module docstring: candidate
            # selection must not depend on a binding result).
            rows.append(
                _unreplayable_row(
                    record,
                    table_id,
                    recorded,
                    len(recorded_items),
                    note,
                    label_accuracy,
                    crop_coverage,
                )
            )
            continue

        fresh_items, bind_note = replay_table(
            record.pdf_path, record.page_num, candidate_markdown, table_id
        )
        note = " | ".join(part for part in (note, bind_note) if part)
        if bind_note:
            rows.append(
                _unreplayable_row(
                    record,
                    table_id,
                    recorded,
                    len(recorded_items),
                    note,
                    label_accuracy,
                    crop_coverage,
                )
            )
            continue
        fresh_counter = Counter(_item_key(item) for item in fresh_items)

        added = tuple(sorted((fresh_counter - recorded_counter).elements()))
        removed = tuple(sorted((recorded_counter - fresh_counter).elements()))

        rows.append(
            ReplayRow(
                doc_slug=record.doc_slug,
                page_num=record.page_num,
                table_id=table_id,
                recorded_status=str(recorded.get("status", "")),
                recorded_item_count=len(recorded_items),
                fresh_item_count=len(fresh_items),
                multiset_match=not added and not removed,
                added=added,
                removed=removed,
                final_disposition=_final_disposition(str(recorded.get("status", ""))),
                label_accuracy=label_accuracy,
                crop_coverage=crop_coverage,
                note=note,
            )
        )
    return rows


def replay_corpus(corpus_dir: Path, labels: dict | None = None) -> list[ReplayRow]:
    rows: list[ReplayRow] = []
    for record in discover_pages(corpus_dir):
        rows.extend(replay_page(record, labels))
    return rows


def format_report(rows: list[ReplayRow]) -> str:
    lines = []
    header = (
        f"{'doc':6} {'page':4} {'table_id':9} {'recorded':9} {'rec#':4} "
        f"{'fresh#':6} {'match':12}  disposition"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        match_col = "UNREPLAYABLE" if row.unreplayable else ("YES" if row.multiset_match else "NO")
        fresh_col = "-" if row.unreplayable else str(row.fresh_item_count)
        lines.append(
            f"{row.doc_slug:6} {row.page_num:<4} {row.table_id:9} "
            f"{row.recorded_status:9} {row.recorded_item_count:<4} "
            f"{fresh_col:<6} {match_col:12}  {row.final_disposition}"
        )
        if row.note:
            lines.append(f"       note: {row.note}")
        if row.added:
            lines.append(f"       + fresh-only: {list(row.added)}")
        if row.removed:
            lines.append(f"       - frozen-only: {list(row.removed)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="socr-replay-binding",
        description=(
            "Replay bind() against a frozen corpus of page sidecars: for "
            "each recorded binding_adjudication, re-derive the table "
            "witness from the recorded model markdown and re-run bind() "
            "on the frozen PDF with the current tree, then compare the "
            "resulting contradiction multiset against what was recorded."
        ),
    )
    parser.add_argument("corpus_dir", type=Path, help="e.g. ~/Data/socr/ladder-run2-2026-09-04")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Hand-read label file written by TICKET-A1b (optional).",
    )
    args = parser.parse_args(argv)

    labels = None
    if args.labels is not None:
        labels = json.loads(args.labels.read_text())

    rows = replay_corpus(args.corpus_dir.expanduser(), labels)
    print(format_report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
