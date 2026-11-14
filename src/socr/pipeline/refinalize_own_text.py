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

#1218 adds a second clause the same way, ``text_layer_drops``
(``core.verdict.text_layer_drops_clause``), keyed on its own check: a page whose reading
omits numbers the layer prints in table rows gets it, its own mode, event and document note.

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
from socr.core.verdict import published_audit_passed

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
    # #1218: the drop clause's reference, from the same analysis a run reads it from.
    "own_layer_drop_reference",
    # #1294: the words clause's reference and the page's figure words, likewise.
    "own_layer_word_reference",
    "own_layer_figure_words",
    "is_born_digital",
    "control_byte_digit_hits",
    "control_byte_scan_failed",
    "minus_as_digit_hits",
    "minus_as_digit_scan_failed",
    "has_unrecovered_symbol_glyphs",
    "has_encoding_hygiene_suspect",
)


_DROP_FIELDS = ("own_layer_drop_reference", "own_layer_word_reference", "own_layer_figure_words")


@dataclass(frozen=True)
class PageChange:
    doc: str
    page_num: int
    before: str
    after: str
    result: str
    #: #1218: the ``text_layer_drops`` result this pass added ("" when it added none).
    drops: str = ""
    #: #1294: the ``text_layer_words`` result this pass added ("" when it added none).
    words: str = ""


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
        if name in _DROP_FIELDS:
            # Empty is "not checked", never "nothing dropped": an analysis without the
            # reference (a stand-in, or one older than #1218) simply gets no drop clause.
            setattr(p, name, getattr(pa, name, "") or "")
        else:
            setattr(p, name, getattr(pa, name))
    # #1306: the footnote wrapper strip reads the painted glyphs; None (not read) keeps wrappers.
    p.painted_layer_text = getattr(pa, "painted_layer_text", None)
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
        in_fresh_run_order,
        text_layer_drops_clause,
        text_layer_numbers_clause,
        text_layer_words_clause,
        with_text_layer_numbers_clause,
    )

    if sidecar.get("terminal") is not True:
        return None
    published = PageVerdict.from_dict(sidecar.get("verdict") or {})
    output = PageOutput.from_dict(sidecar.get("winning_output") or {})
    p = _page_state(sidecar, pa)
    # #1218: the drop clause is its own check, so it is applied (and its idempotence keyed)
    # on its own: a page already carrying a ``text_layer_drops`` clause is left alone. A
    # page of a run of this code that the clause does not fire on recomputes to None every
    # time, so nothing changes. Taken BEFORE the prior-OCR branch below mutates ``p``: the
    # clause already emits nothing for ``prior_ocr_layer_suspect`` (#1156: that layer never
    # votes), whether the run or only the analysis found it.
    drops = None
    if not any(c.check is Check.TEXT_LAYER_DROPS for c in published.clauses):
        drops = text_layer_drops_clause(output, p)
    # #1294: the words clause, the same way (its own check, keyed on its own idempotence).
    words = None
    if not any(c.check is Check.TEXT_LAYER_WORDS for c in published.clauses):
        words = text_layer_words_clause(output, p)
    clause = None
    if not any(c.check is Check.TEXT_LAYER_NUMBERS for c in published.clauses):
        # (a page carrying the clause: already re-finalized, or written by a run of this code)
        if p.prior_ocr_layer_suspect and not sidecar.get("prior_ocr_layer_suspect"):
            # Only the analysis finds the earlier OCR layer, so no clause of the run marks
            # the page: the clause's no-own-text UNKNOWN (the same guards: a failed marker
            # or an empty reading still gets none), with the reason that says why.
            p.prior_ocr_layer_suspect = False
            p.own_layer_text = ""
            clause = text_layer_numbers_clause(output, p)
            if clause is not None:
                clause = replace(clause, reason=PRIOR_OCR_UNMARKED_REASON)
        else:
            clause = text_layer_numbers_clause(output, p)
    if clause is None and drops is None and words is None:
        return None
    as_run = slot_reference_state(sidecar, output.page_num)
    extended = replace(published, page_num=output.page_num)
    if clause is not None:
        extended = with_text_layer_numbers_clause(extended, clause, output, as_run, event_kinds)
    if drops is not None:
        # By the fresh-run order, which puts it last of the silent-success clauses; a count
        # of the clauses ahead would be a second definition of that order.
        extended = replace(extended, clauses=tuple(in_fresh_run_order((*extended.clauses, drops))))
    if words is not None:
        extended = replace(extended, clauses=tuple(in_fresh_run_order((*extended.clauses, words))))
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


def _insert_event(events: list[dict], event: dict, follows) -> list[dict]:
    """*events* with *event* where a run records it: before the first event *follows* says a
    run records after it, else last. ``audit_log``'s sort cannot do this: unlisted kinds all
    share one rank, so it keeps their order of arrival."""
    at = next((n for n, e in enumerate(events) if follows(e)), len(events))
    return [*events[:at], event, *events[at:]]


def _persisted_dropped(events: list[dict]) -> list[str] | None:
    """The numbers of the first ``numbers_dropped`` event that lists any, else ``None``."""
    from socr.core.verdict import NUMBERS_DROPPED_KIND

    for e in events:
        dropped = (e.get("data") or {}).get("dropped")
        if e.get("kind") == NUMBERS_DROPPED_KIND and isinstance(dropped, list) and dropped:
            return list(dropped)
    return None


def _drops_clause_count(sidecar: dict) -> int | None:
    from socr.core.marks import Check
    from socr.core.verdict import drops_count_from_reason

    for c in (sidecar.get("verdict") or {}).get("clauses") or []:
        if c.get("check") == Check.TEXT_LAYER_DROPS.value:
            return drops_count_from_reason(c.get("reason") or "")
    return None


def resolve_dropped(sidecar: dict, recomputed, document_data: dict | None = None):
    """``(numbers, "")`` for the page's ``numbers_dropped`` event, or ``(None, why not)``.

    A persisted list is NEVER replaced: the first of (1) the sidecar's own event, (2) the
    document's audit event for the page, (3) a recomputation from the analysis. (2) and (3)
    are accepted only when their count equals the count the page's own clause states, so a
    reassessment that failed (an empty reference reads as ``()``) or came out different can
    neither erase nor rewrite evidence. Otherwise nothing is written and the reason is
    returned, for the caller to surface."""
    persisted = _persisted_dropped(list(sidecar.get("audit_events") or []))
    if persisted is not None:
        return persisted, ""
    stated = _drops_clause_count(sidecar)
    if stated is None:
        return None, "the page's clause states no count to hold a list to"
    doc = (document_data or {}).get("dropped")
    if isinstance(doc, list) and len(doc) == stated:
        return list(doc), ""
    recomputed = list(recomputed or ())
    if not recomputed:
        return None, f"the clause states {stated} number(s) and the reassessment found none"
    if len(recomputed) != stated:
        return (
            None,
            f"the clause states {stated} number(s), the reassessment found {len(recomputed)}",
        )
    return recomputed, ""


def _reposition(events: list[dict], kind: str, make, follows) -> list[dict]:
    """*events* with the one event of *kind* at the position a run gives it: the existing
    event itself when there is one (its data untouched), else ``make()`` (``None``: none)."""
    existing = next((e for e in events if e.get("kind") == kind), None)
    event = existing if existing is not None else make()
    if event is None:
        return events
    return _insert_event([e for e in events if e.get("kind") != kind], event, follows)


def _with_number_events(
    sidecar: dict, engine: str, recomputed, document_data=None, missing_words=None
):
    """``(audit_events, why a dropped list could not be written or "")`` for the page sidecar.

    #1185's and #1218's events, each once, at the position a run gives it
    (``verdict.event_follows_numbers_*``) -- the EXISTING ones too, so an artifact the earlier
    tool wrote in another order converges, and the sidecar and the document cannot disagree.
    The dropped numbers are the event's data (see :func:`resolve_dropped`)."""
    from socr.core.verdict import (
        NUMBERS_DISAGREE_EVENT_DETAIL,
        NUMBERS_DISAGREE_KIND,
        NUMBERS_DROPPED_EVENT_DETAIL,
        NUMBERS_DROPPED_KIND,
        event_follows_numbers_disagree,
        event_follows_numbers_dropped,
        numbers_dropped_event_data,
    )

    events = list(sidecar.get("audit_events") or [])
    why = ""
    if _has_disagreement(sidecar):
        events = _reposition(
            events,
            NUMBERS_DISAGREE_KIND,
            lambda: {
                "data": {},
                "detail": NUMBERS_DISAGREE_EVENT_DETAIL,
                "engine": engine,
                "kind": NUMBERS_DISAGREE_KIND,
            },
            event_follows_numbers_disagree,
        )
    if _has_drops(sidecar):
        dropped, why = resolve_dropped(sidecar, recomputed, document_data)
        events = _reposition(
            events,
            NUMBERS_DROPPED_KIND,
            lambda: (
                None
                if dropped is None
                else {
                    "data": numbers_dropped_event_data(int(sidecar["page_num"]), dropped),
                    "detail": NUMBERS_DROPPED_EVENT_DETAIL,
                    "engine": engine,
                    "kind": NUMBERS_DROPPED_KIND,
                }
            ),
            event_follows_numbers_dropped,
        )
        # an existing event with no list is completed, never one that has a list replaced
        if dropped is not None and _persisted_dropped(events) is None:
            events = [
                {**e, "data": numbers_dropped_event_data(int(sidecar["page_num"]), dropped)}
                if e.get("kind") == NUMBERS_DROPPED_KIND
                else e
                for e in events
            ]
    if _has_words(sidecar):
        # #1294: the words event, after #1218's; its data are the page's missing words.
        from socr.core.verdict import (
            WORDS_MISSING_EVENT_DETAIL,
            WORDS_MISSING_KIND,
            event_follows_words_missing,
            words_missing_event_data,
        )

        events = _reposition(
            events,
            WORDS_MISSING_KIND,
            lambda: (
                None
                if not missing_words
                else {
                    "data": words_missing_event_data(int(sidecar["page_num"]), missing_words),
                    "detail": WORDS_MISSING_EVENT_DETAIL,
                    "engine": engine,
                    "kind": WORDS_MISSING_KIND,
                }
            ),
            event_follows_words_missing,
        )
    return events, why


def _repaired_number_events(sidecar: dict, recomputed, document_data=None, missing_words=None):
    """``(sidecar with its number events whole and in a run's order or None if it is, why a
    list could not be written or "")``, for a page that already carries the clauses.

    Such a page is not re-finalized, so an interrupted or earlier-tool pass is repaired
    here, from :func:`resolve_dropped`'s evidence only."""
    if sidecar.get("terminal") is not True or not (
        _has_drops(sidecar) or _has_disagreement(sidecar) or _has_words(sidecar)
    ):
        return None, ""
    engine = (sidecar.get("winning_output") or {}).get("engine") or ""
    events, why = _with_number_events(sidecar, engine, recomputed, document_data, missing_words)
    if events == (sidecar.get("audit_events") or []):
        return None, why
    return {**sidecar, "audit_events": events}, why


def _updated_sidecar(
    sidecar: dict,
    output,
    verdict,
    dropped=(),
    document_data=None,
    why_sink=None,
    missing_words=None,
    words_record=None,
) -> dict:
    from socr.core.manifest import is_page_failed_marker
    from socr.pipeline.orchestrator import _page_blob_key, _socr_source_digest, _socr_version

    winning = output.to_dict()
    out = dict(sidecar)
    out["status"] = winning.get("status", "missing")
    out["failure_mode"] = winning.get("failure_mode", "none")
    out["audit_passed"] = published_audit_passed(
        out["status"], winning.get("audit_passed", False), verdict
    )
    out["verdict"] = verdict.to_dict()
    out["winning_output"] = winning
    out["page_fingerprint"] = (
        _page_blob_key(winning) if not is_page_failed_marker(winning.get("text", "")) else ""
    )
    out["socr_version"] = _socr_version()
    out["socr_source_digest"] = _socr_source_digest()
    out["audit_events"], why = _with_number_events(
        out, winning.get("engine") or "", dropped, document_data, missing_words
    )
    # #1294: the words check's record, PASS or not, as a run writes it.
    if words_record is not None:
        out["text_layer_words"] = words_record
    if why and why_sink is not None:
        why_sink[int(sidecar["page_num"])] = why
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


def _has_fail(sidecar: dict, check) -> bool:
    from socr.core.verdict import CheckResult

    return any(
        c.get("check") == check.value and c.get("result") == CheckResult.FAIL.value
        for c in (sidecar.get("verdict") or {}).get("clauses") or []
    )


def _has_disagreement(sidecar: dict) -> bool:
    """The page's published verdict carries this clause's FAIL."""
    from socr.core.marks import Check

    return _has_fail(sidecar, Check.TEXT_LAYER_NUMBERS)


def _has_words(sidecar: dict) -> bool:
    """#1294: the page's published verdict carries a ``text_layer_words`` FAIL."""
    from socr.core.marks import Check

    return _has_fail(sidecar, Check.TEXT_LAYER_WORDS)


def _has_drops(sidecar: dict) -> bool:
    """#1218: the page's published verdict carries a ``text_layer_drops`` FAIL."""
    from socr.core.marks import Check

    return _has_fail(sidecar, Check.TEXT_LAYER_DROPS)


def _place_note(error: str, note: str, later_notes: tuple[str, ...]) -> str:
    """*error* with *note* once, where a run puts it: before the first of the notes a run
    appends after it (*later_notes*, by prefix), else last. A note already present is moved
    there, so an artifact the earlier tool wrote in another order converges."""
    if note in error:
        if error == note:
            return error
        for old in (note + "; ", "; " + note):
            if old in error:
                error = error.replace(old, "", 1)
                break
    if not error:
        return note
    if error.startswith(later_notes):
        return f"{note}; {error}"
    at = min((i for i in (error.find("; " + p) for p in later_notes) if i >= 0), default=-1)
    if at >= 0:
        return f"{error[:at]}; {note}{error[at:]}"
    return f"{error}; {note}"


def _record_disagreement(
    doc_dir: Path,
    meta_path: Path,
    meta: dict,
    pages: list[tuple[int, str]],
    *,
    drops: bool = False,
    words: bool = False,
    data_by_page: dict[int, dict] | None = None,
) -> None:
    """What a run of this code records for a document with FAIL *pages* (``(page, engine)``):
    a clean status becomes ``partial`` (a worse one is left as the run wrote it), the
    ``error`` note names the pages, here and in the root index, and ``audit_log.json``
    holds one event per page. Each step checks before writing, so a rerun adds nothing."""
    from ocr_output_contract import Status

    from socr.core.audit_log import event_rank
    from socr.core.verdict import (
        INVALID_FINAL_EMISSION_NOTE_PREFIX,
        NUMBERS_DISAGREE_EVENT_DETAIL,
        NUMBERS_DISAGREE_KIND,
        NUMBERS_DROPPED_EVENT_DETAIL,
        NUMBERS_DROPPED_KIND,
        NUMBERS_DROPPED_NOTE_PREFIX,
        WORDS_MISSING_NOTE_PREFIX,
        event_follows_numbers_disagree,
        event_follows_numbers_dropped,
        numbers_disagree_note,
        numbers_dropped_note,
    )

    # ``words`` (#1294): the same record for the pages that ship without layer words.
    if words:
        from socr.core.verdict import (
            WORDS_MISSING_EVENT_DETAIL,
            WORDS_MISSING_KIND,
            event_follows_words_missing,
            words_missing_note,
        )

        kind, detail = WORDS_MISSING_KIND, WORDS_MISSING_EVENT_DETAIL
        note = words_missing_note([n for n, _engine in pages])
        follows, later_notes = event_follows_words_missing, (INVALID_FINAL_EMISSION_NOTE_PREFIX,)
    # ``drops`` (#1218): the same document-level record, with the wording of an OMISSION.
    elif drops:
        kind, detail = NUMBERS_DROPPED_KIND, NUMBERS_DROPPED_EVENT_DETAIL
        note = numbers_dropped_note([n for n, _engine in pages])
        follows = event_follows_numbers_dropped
        # #1294: a run appends its words note after the drops note.
        later_notes = (WORDS_MISSING_NOTE_PREFIX, INVALID_FINAL_EMISSION_NOTE_PREFIX)
    else:
        kind, detail = NUMBERS_DISAGREE_KIND, NUMBERS_DISAGREE_EVENT_DETAIL
        note = numbers_disagree_note([n for n, _engine in pages])
        follows = event_follows_numbers_disagree
        later_notes = (
            NUMBERS_DROPPED_NOTE_PREFIX,
            WORDS_MISSING_NOTE_PREFIX,
            INVALID_FINAL_EMISSION_NOTE_PREFIX,
        )

    def recorded(record: dict) -> bool:
        changed = False
        if record.get("status") == Status.COMPLETED.value:
            record["status"] = Status.PARTIAL.value
            changed = True
        error = record.get("error") or ""
        placed = _place_note(error, note, later_notes)
        if placed != error:
            record["error"] = placed
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
    before = json.dumps(events)
    for page_num, engine in pages:
        data = (data_by_page or {}).get(page_num) or {}
        held = next(
            (
                n
                for n, e in enumerate(events)
                if e.get("kind") == kind and e.get("page_num") == page_num
            ),
            None,
        )
        if held is not None:
            # The existing event, moved to where a run has it. Its numbers are never
            # replaced; an event with none (an interrupted pass) is completed.
            event = events.pop(held)
            listed = "missing" if words else "dropped"
            if (drops or words) and data and not (event.get("data") or {}).get(listed):
                event["data"] = data
        elif (drops or words) and not data:
            continue  # no persisted or confirmed list to write (surfaced by the caller)
        else:
            event = {
                "page_num": page_num,
                "kind": kind,
                "engine": engine,
                "detail": detail,
                "data": data,
            }
        # Where a run's sorted log has it: among the page's unranked kinds, before the first
        # one a run records after it; else after the page's last event.
        page_at = [i for i, e in enumerate(events) if e.get("page_num") == page_num]
        end = (
            (page_at[-1] + 1)
            if page_at
            else sum(1 for e in events if (e.get("page_num") or 0) <= page_num)
        )
        at = next(
            (i for i in page_at if event_rank(events[i]["kind"]) == 9 and follows(events[i])),
            end,
        )
        events.insert(at, event)
    if json.dumps(events) != before:
        counts: dict[str, int] = {}
        for e in events:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        audit.update(event_count=len(events), counts=counts, events=events)
        _write_json(audit_path, audit)


def _document_drops_data(doc_dir: Path) -> dict[int, dict]:
    """Each page's ``numbers_dropped`` event data in the document's ``audit_log.json``."""
    from socr.core.verdict import NUMBERS_DROPPED_KIND

    path = doc_dir / "audit_log.json"
    if not path.is_file():
        return {}
    return {
        e["page_num"]: e.get("data") or {}
        for e in _read_json(path).get("events") or []
        if e.get("kind") == NUMBERS_DROPPED_KIND and isinstance(e.get("page_num"), int)
    }


def _surface_unwritten(
    doc_dir: Path, sidecar_paths, why_by_page: dict, skipped: Counter, *, write: bool
) -> None:
    """Say which pages carry the ``text_layer_drops`` clause with no list of numbers behind it.

    From the sidecars as they now are. Never silent: each page is counted under the tool's
    "not checked" lines (printed by ``main``) and, when writing, recorded in the document's
    ``refinalized_own_text.json`` under ``unwritten_drops_events`` (removed again when a later
    pass writes them). Nothing about the page is changed."""
    unwritten = {}
    for path in sidecar_paths:
        sidecar = _read_json(path)
        if (
            _has_drops(sidecar)
            and _persisted_dropped(list(sidecar.get("audit_events") or [])) is None
        ):
            n = int(sidecar["page_num"])
            unwritten[str(n)] = why_by_page.get(n, "no persisted list and none could be confirmed")
    for n, why in unwritten.items():
        skipped[f"page {n}: numbers_dropped event has no list of numbers ({why})"] += 1
    log_path = doc_dir / LOG_NAME
    if not write or not (unwritten or log_path.is_file()):
        return
    log = _read_json(log_path) if log_path.is_file() else {"tool": "socr-refinalize-own-text"}
    if unwritten:
        log["unwritten_drops_events"] = unwritten
    elif "unwritten_drops_events" in log:
        del log["unwritten_drops_events"]
    elif log_path.is_file():
        return
    _write_json(log_path, log)


def refinalize_document(
    doc_dir: Path, pdf_dir: Path, *, dry_run: bool = False, skipped: Counter | None = None
) -> list[PageChange]:
    """Re-finalize one document's model-read pages. Nothing is written until every page of
    the document has been decided, so an inconsistency leaves the document untouched."""
    from socr.core.marks import Check
    from socr.core.result import PageOutput
    from socr.core.verdict import (
        engine_is_model_reading,
        page_dropped_numbers,
        page_missing_words,
        page_word_check_record,
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
    repairs: list[tuple[Path, dict]] = []
    # What the document's own audit log already lists for a page, and why a list could not
    # be written for a page (both feed :func:`resolve_dropped`; neither is ever overwritten).
    held_by_document = _document_drops_data(doc_dir)
    why_by_page: dict[int, str] = {}
    for path, sidecar in candidates:
        page_num = int(sidecar["page_num"])
        done = refinalize_page(
            sidecar, assessment.pages[page_num - 1], event_kinds.get(page_num, frozenset())
        )
        if done is None:
            # A page that already carries the clause: make its events whole and in a run's
            # order (see ``_repaired_number_events``). No page text changes.
            shipped = PageOutput.from_dict(sidecar.get("winning_output") or {})
            state = _page_state(sidecar, assessment.pages[page_num - 1])
            whole, why = _repaired_number_events(
                sidecar,
                page_dropped_numbers(shipped, state),
                held_by_document.get(page_num),
                page_missing_words(shipped, state),
            )
            if why:
                why_by_page[page_num] = why
            # #1294: a page the words check passes (or one an older run wrote) gets the
            # check's record, as a run of this code writes it. Never replaced once there.
            record = page_word_check_record(shipped, state)
            if record is not None and "text_layer_words" not in sidecar:
                whole = {**(whole if whole is not None else sidecar), "text_layer_words": record}
            # #1244 C1: re-derive the PUBLISHED audit_passed here too. A page an earlier pass
            # (or a run from before the derivation) demoted keeps a stale ``true`` otherwise,
            # because this path never goes through ``_updated_sidecar``.
            if sidecar.get("terminal") is True:
                published = published_audit_passed(
                    sidecar.get("status"),
                    (sidecar.get("winning_output") or {}).get("audit_passed", False),
                    sidecar.get("verdict"),
                )
                current = whole if whole is not None else sidecar
                if current.get("audit_passed") != published:
                    whole = {**current, "audit_passed": published}
            if whole is not None:
                repairs.append((path, whole))
            continue
        output, verdict = done
        fragment_path = path.with_suffix(".md")
        old_text = (sidecar.get("winning_output") or {}).get("text") or ""
        old_fragment = fragment_path.read_text(encoding="utf-8")
        if old_fragment not in (old_text, output.text):
            raise RuntimeError(f"{fragment_path}: fragment is not its sidecar's text")
        heading = f"## Page {page_num}\n\n"
        doc_md = _swap_once(doc_md, heading + old_text, heading + output.text, md_paths[0])
        state = _page_state(sidecar, assessment.pages[page_num - 1])
        new_sidecar = _updated_sidecar(
            sidecar,
            output,
            verdict,
            page_dropped_numbers(output, state),
            held_by_document.get(page_num),
            why_by_page,
            missing_words=page_missing_words(output, state),
            words_record=page_word_check_record(output, state),
        )
        writes.append((fragment_path, output.text, path, new_sidecar))
        changes.append(
            PageChange(
                doc=doc_dir.name,
                page_num=page_num,
                before=str(sidecar.get("status")),
                after=new_sidecar["status"],
                result=next(
                    (
                        c.result.value
                        for c in verdict.clauses
                        if c.check is Check.TEXT_LAYER_NUMBERS
                    ),
                    "",
                ),
                drops=next(
                    (c.result.value for c in verdict.clauses if c.check is Check.TEXT_LAYER_DROPS),
                    "",
                ),
                words=next(
                    (c.result.value for c in verdict.clauses if c.check is Check.TEXT_LAYER_WORDS),
                    "",
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
        _surface_unwritten(doc_dir, [p for p, _s in candidates], why_by_page, skipped, write=False)
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
            # An earlier pass (#1185's) may already have an entry for this page: this one
            # adds to it. Its `before` is the status the page had BEFORE any pass, and its
            # numbers result is the one its own clause produced; both are kept.
            earlier = pages.get(str(change.page_num)) or {}
            kept_after = earlier.get("after") or {}
            pages[str(change.page_num)] = {
                "before": earlier.get("before") or {"status": change.before},
                "after": {
                    "status": change.after,
                    "text_layer_numbers": (
                        kept_after["text_layer_numbers"]
                        if "text_layer_numbers" in kept_after
                        else change.result
                    ),
                    **({"text_layer_drops": change.drops} if change.drops else {}),
                    **({"text_layer_words": change.words} if change.words else {}),
                },
                "socr_source_digest": data["socr_source_digest"],
            }
        _write_json(log_path, log)
        for _f, _t, path, data in writes:
            _write_json(path, data)
    for path, data in repairs:
        _write_json(path, data)
    _surface_unwritten(doc_dir, [p for p, _s in candidates], why_by_page, skipped, write=True)
    # From the sidecars as they now are, not from this pass's changes, so a run interrupted
    # between the sidecars and this line is completed by the next one.
    disagreeing = [
        (int(sc["page_num"]), (sc.get("winning_output") or {}).get("engine") or "")
        for sc in map(_read_json, sidecar_paths)
        if _has_disagreement(sc)
    ]
    if disagreeing:
        _record_disagreement(doc_dir, meta_path, meta, disagreeing)
    # #1218: after #1185's, the order a run records them in.
    dropping = [
        (int(sc["page_num"]), (sc.get("winning_output") or {}).get("engine") or "")
        for sc in map(_read_json, sidecar_paths)
        if _has_drops(sc)
    ]
    if dropping:
        # The numbers come from the page sidecars' own events (written with the clause), so
        # an interrupted run's later pass writes the same data a run does.
        from socr.core.verdict import NUMBERS_DROPPED_KIND

        data_by_page = {
            int(sc["page_num"]): e.get("data") or {}
            for sc in map(_read_json, sidecar_paths)
            if _has_drops(sc)
            for e in sc.get("audit_events") or []
            if e.get("kind") == NUMBERS_DROPPED_KIND
        }
        _record_disagreement(
            doc_dir, meta_path, meta, dropping, drops=True, data_by_page=data_by_page
        )
    # #1294: after #1218's, the order a run records them in.
    missing = [
        (int(sc["page_num"]), (sc.get("winning_output") or {}).get("engine") or "")
        for sc in map(_read_json, sidecar_paths)
        if _has_words(sc)
    ]
    if missing:
        from socr.core.verdict import WORDS_MISSING_KIND

        words_by_page = {
            int(sc["page_num"]): e.get("data") or {}
            for sc in map(_read_json, sidecar_paths)
            if _has_words(sc)
            for e in sc.get("audit_events") or []
            if e.get("kind") == WORDS_MISSING_KIND
        }
        _record_disagreement(
            doc_dir, meta_path, meta, missing, words=True, data_by_page=words_by_page
        )
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
            counts[(change.before, change.after, change.result, change.drops, change.words)] += 1
    for (before, after, result, drops, words), n in sorted(counts.items()):
        extra = f", text_layer_drops={drops}" if drops else ""
        extra += f", text_layer_words={words}" if words else ""
        print(f"{n:6d}  {before} -> {after}  (text_layer_numbers={result}{extra})")
    print(f"{sum(counts.values()):6d}  pages re-finalized")
    for reason, n in sorted(skipped.items()):
        print(f"{n:6d}  not checked: {reason}")
    for doc_dir, why in failed:
        print(f"FAILED  {doc_dir}: {why}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
