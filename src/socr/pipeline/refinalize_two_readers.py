"""#1409 step 3: ship a scanned page as text when two independent readings of its image agree.

    socr-refinalize-two-readers <out_dir> --pdf-dir <dir>
        --readings-dir <A> --readings-dir-b <B> [--apply]

Owner ruling (2026-11-26, approved for build and test on a branch): a scanned page with no
trustworthy text layer may ship as TEXT, unmarked, when two independent readings of the page
image agree on everything (``tables.two_readers``). Charts still ship as pictures and never
as numbers. Any disagreement leaves the page exactly as the run left it.

DRY-RUN BY DEFAULT: without ``--apply`` it prints what ``--apply`` would lift and writes
nothing. This tool calls no reader: both readings come from disk, in the format of
``refinalize_second_reader.reading_record``, one directory per reader
(``<dir>/<document directory name>/<NNNNN>.json``).

Which pages
-----------
ALL of:

* finished (``terminal``), and NOT a clean page: its status is not SUCCESS, or it carries a
  verdict clause, or its text carries a mark. A page that is SUCCESS without a mark is
  never touched;
* no text layer of its own: the layer has no words, or it is a suspected earlier OCR run
  (``refinalize_second_reader._has_own_layer``, the predicate of that tool's
  ``not_applicable`` path). A born-digital page is never touched;
* no chart or figure recorded by the run: none of :data:`FIGURE_KEYS` is set in the
  sidecar, and the shipped text embeds no picture other than a floor's own whole-page
  picture (:data:`FLOOR_PICTURE_STEMS`). A page with a recorded chart stays as it is;
* both readings are at hand, each for this page and this input file, from two DIFFERENT
  reader models.

What ships
----------
Reading A (``--readings-dir``), converted by the step socr already applies to a model
reading: ``core.html_tables.clean_residual_html`` (HTML tables to markdown tables through
the span-aware grid, tags stripped, entities decoded) and the generic
``OutputNormalizer``. The shipped text is a function of the stored reading alone. Reading B
confirms and is never shipped. The conversion is checked, not trusted: the page is left
alone when the converted text does not hold exactly the reading's words and numbers, or
still holds a tag (:func:`shipped_text`). Page furniture ships as reading A wrote it, as
lines of text, which is what socr ships for a model reading today; it was compared too
(``two_readers``' rule 2 removes a furniture block from the comparison only on an exact
word match with the other reading).

No provenance line is written on the page. The precedent is #1381 (a scan's earlier OCR
layer confirmed by an exact fresh reading): such a page ships unmarked, and the confirmation
is recorded in the sidecar, the pass trace and the document's audit log. This tool does the
same: the sidecar key ``two_reader_agreement`` (rule, both reader models, both prompt
hashes, both reading digests), a ``two_reader_agreement`` PASS entry in ``checks`` stamped
with the digest of the shipped text, an audit event, ``refinalized_from`` (what the run had
shipped), and the two readings themselves under ``<doc_dir>/two_readers/{a,b}/``.

What a lifted page becomes
--------------------------
SUCCESS, engine ``two-readers``, no clause, through the seam a run uses
(``core.manifest.apply_verdict_marks``). Written in this order: the two readings, the page
fragment and the page's section of the document ``.md``, the manifest entry (``blob_ref``,
disposition, and its picture dropped from ``assets`` when the page no longer shows one) and
cache, ``refinalized_two_readers.json`` (every compared page, before and after, with the
reasons of a page that did not agree), ``metadata.json`` and the root index, the audit log,
and the sidecars LAST. A page whose sidecar still names the old ending is redone in full,
so a run interrupted anywhere is completed by running the tool again.

The document: as ``socr-refinalize-prior-ocr`` does when it replaces a floor. A lifted page
leaves the "produced no usable output" list of ``error``, the floor's ``page_failed`` /
``invisible_scan_unread`` events go, and the table counts are recomputed. The document
``status`` is NEVER moved (that tool moves none): a ``partial`` stays ``partial`` even when
every page is now SUCCESS. The note naming the pages that "did not finish with status
success and a passing audit" is rewritten for the pages that still have not (and removed
when none is left). Other notes that name the page by number are left as the run wrote
them. The floor's picture file stays on disk.

Idempotent: a lifted page is a clean SUCCESS page and is no longer selected.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from socr.core.page_files import page_sidecar_paths
from socr.tables.second_reader import row_numbers
from socr.tables.two_readers import (
    ENGINE,
    RULE,
    RULE_VERSION,
    TwoReaderVerdict,
    compare_two_readings,
    unfenced,
    word_tokens,
)

TOOL = "socr-refinalize-two-readers"
LOG_NAME = "refinalized_two_readers.json"
#: Where a document keeps the two readings a lifted page rests on.
READINGS_DIR = "two_readers"
#: The trace entry and audit event of a page confirmed by this rule.
AGREEMENT_KIND = RULE
AGREEMENT_REASON = (
    "two independent readings of the page image agree on every number, word and table cell "
    "(two readings can agree and both be wrong)"
)
AGREEMENT_EVENT_DETAIL = (
    "this scanned page has no text layer of its own; two independent readings of the page "
    "image agree on every number, word and table cell, so the page ships the first reading "
    "as text, confirmed by reader agreement and by nothing else (#1409)"
)

#: Sidecar keys by which a run records a chart or figure on the page.
FIGURE_KEYS = (
    "figure_refs",
    "unshipped_figure_refs",
    "unshipped_figure_kind",
    "unshipped_figure_unlocated",
    "unshipped_figure_captions",
    "scanned_figure_png_ref",
    "chart_floor_body",
    "chart_asset_detection_failed",
    "chart_asset_render_failed",
    "chart_region_inventory_failed",
    "chart_region_placement_unresolved",
    "chart_region_render_failed",
)
#: File-name stems of the whole-page picture a fail-closed floor ships in place of the text.
#: Such a picture is the floor itself, not a figure of the page.
FLOOR_PICTURE_STEMS = ("failed_table_p", "invisible_scan_page_p", "prior_ocr_layer_page_p")
#: Events a replaced floor wrote (``refinalize_prior_ocr._FLOOR_EVENT_KINDS``).
_FLOOR_EVENT_KINDS = frozenset({"page_failed", "invisible_scan_unread"})
#: Additional replaced-reading events that describe the superseded OCR attempt, never the
#: newly shipped two-reader text.
_SUPERSEDED_READING_EVENT_KINDS = frozenset(
    {
        "table_binding_adjudicated",
        "table_binding_boundary_resolved",
        "native_cell_check_verified",
        "native_table_verifier_exact_pass",
        "table_point_repaired",
        "table_blank_lines_dropped",
        "table_furniture_removed",
        "table_spacer_rows_dropped",
        "table_wrapped_label_merged",
        "table_ladder_budget_exhausted",
        "paid_reading_after_table_withhold",
        "paid_reading_lifted_floor",
        "source_evidence_table_corroborated",
        "source_evidence_table_extra_resolution",
        "numbers_disagree_with_second_reader",
        "rejudge_accepted",
        "rejudge_error",
        "rejudge_rejected",
        "rejudge_timeout",
        "native_table_model_retry",
        "native_ship_gate_deferred",
        "rotated_native_table_quarantined",
        "judge_wedged_circuit_open",
        "provider_empty_response",
        "page_retry",
        "text_ink_unread",
        "control_bytes_replaced",
        "hidden_footnote_stripped",
        "native_math_unrecovered",
        "native_math_font_unrecovered",
        "native_unrecovered_symbol_glyphs",
        "native_encoding_hygiene_suspect",
        "possible_table_structure_not_reconstructed",
    }
)
#: Standard sidecar keys belonging to the replaced failed reading, reset to clean defaults
#: when lifting a page so resume never rejects or re-demotes the lifted output.
_RESET_SIDECAR_DEFAULTS: dict[str, object] = {
    "needs_ocr_enhancement": False,
    "native_table_structure_failed": False,
    "native_rotated_text_shredded": False,
    "rotated_shred_png_ref": "",
    "native_table_structure_defective": False,
    "native_table_emission_defect": "",
    "native_table_content_defect": "",
    "native_table_header_unattributed": False,
    "native_table_unverifiable": False,
    "native_table_model_retry": "",
    "native_table_unverifiable_ordinals": [],
    "native_table_region_count": 0,
    "native_table_region_identities": [],
    "d3_floor_png_ref": "",
    "scanned_table_evidence_failed": False,
    "equation_lane_retry_pending": False,
    "equation_sidecar_skipped": False,
    "fabricated_image_refs": 0,
    "judge_rejected": False,
    "structure_class_model_kept": False,
    "table_ladder_disposition": None,
    "table_ladder_superseded": None,
    "table_ladder_incomplete": False,
    "binding_adjudication": {},
    "native_cell_binding": {},
}
#: Sparse sidecar keys belonging to the replaced failed reading, popped on lift.
_POP_SIDECAR_KEYS: tuple[str, ...] = (
    "table_judge_retry_pending",
    "table_judge_retry_rungs",
    "table_escalation_retry_pending",
    "not_processed_after_halt",
    "row_binding",
    "row_binding_contradicted_tables",
    "row_binding_unverifiable_tables",
    "row_binding_table_count",
    "row_binding_mark_only",
    "unchecked_table_kind",
    "unchecked_table_values",
    "unchecked_table_value_list",
    "withheld_table_text",
    "chart_floor_abstention",
    "text_coverage",
    "text_ink_fully_accounted",
    "text_hidden_layer_ink_accounted",
    "text_layer_words",
    "prior_ocr_layer_agreement",
    "table_witness_corroboration",
    "table_ditto_columns",
    "table_label_unverified",
    "has_unmapped_math_glyphs",
    "private_use_bullets_only",
    "math_recovery_evidence",
    "equation_region_evidence",
)

_PICTURE_RE = re.compile(r"!\[[^\]]*\]\(([^)]*)\)")
_FURNITURE_BLOCK_TAG_RE = re.compile(r"</?(?:page_header|page_footer)(?:\s[^<>\n]*)?>", re.I)
_PAGE_NUMBER_TAG_RE = re.compile(r"</?page_number(?:\s[^<>\n]*)?>", re.I)
_TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9_]*(?:\s[^<>\n]*)?/?>")


def _replaced_event_kinds() -> frozenset[str]:
    from socr.core.tables_trust import RESOLVING_KINDS, TABLE_DISTRUST_KINDS

    return (
        _FLOOR_EVENT_KINDS
        | TABLE_DISTRUST_KINDS
        | RESOLVING_KINDS
        | _SUPERSEDED_READING_EVENT_KINDS
        | {AGREEMENT_KIND}
    )


OUTCOME_LIFTED = "lifted"
OUTCOME_DISAGREE = "disagree"
OUTCOME_CONVERSION = "conversion_refused"


@dataclass(frozen=True)
class PageResult:
    doc: str
    page_num: int
    #: ``lifted`` / ``disagree`` / ``conversion_refused``
    outcome: str
    before: tuple[str, str] = ("", "")
    reasons: tuple[str, ...] = ()


@dataclass
class _Compared:
    path: Path
    sidecar: dict
    result: PageResult
    record: dict
    readings: tuple[dict, dict]
    new_text: str = ""
    new_sidecar: dict | None = None


# --- selection --------------------------------------------------------------------------


def is_clean(sidecar: dict) -> bool:
    """SUCCESS, no verdict clause, no mark line: a page this tool never touches."""
    from socr.core.marks import has_any_mark_line
    from socr.core.result import PageStatus

    winning = sidecar.get("winning_output") or {}
    success = PageStatus.SUCCESS.value
    return (
        sidecar.get("status") == success
        and winning.get("status") == success
        and not (sidecar.get("verdict") or {}).get("clauses")
        and not has_any_mark_line(winning.get("text") or "", int(sidecar["page_num"]))
    )


def records_a_figure(sidecar: dict) -> bool:
    """The run recorded a chart or figure on this page, or its text embeds a picture that is
    not a floor's whole-page picture."""
    if any(sidecar.get(key) for key in FIGURE_KEYS):
        return True
    text = (sidecar.get("winning_output") or {}).get("text") or ""
    return any(
        not Path(target).name.startswith(FLOOR_PICTURE_STEMS)
        for target in _PICTURE_RE.findall(text)
    )


def selects(sidecar: dict) -> str:
    """Why the sidecar alone rules this page out, or ``""`` (the layer is asked afterwards)."""
    if sidecar.get("terminal") is not True:
        return "page not finished (terminal is not true)"
    if is_clean(sidecar):
        return "page is SUCCESS without a mark"
    if records_a_figure(sidecar):
        return "the run recorded a chart or figure on the page"
    return ""


# --- what ships -------------------------------------------------------------------------


def shipped_text(reading: str) -> tuple[str, str]:
    """``(the page markdown reading A ships as, "")``, or ``("", why it is refused)``.

    The conversion is socr's own for a model reading, plus unwrapping the three reader
    furniture tags (``<page_header>``, ``<page_footer>``, ``<page_number>``) that rule 2
    already validates so readers emitting tagged furniture ship as plain lines of text.
    It must lose nothing: the result holds exactly the reading's words and numbers, in
    order, and no unconverted markup tag."""
    from socr.core.html_tables import clean_residual_html
    from socr.core.normalizer import OutputNormalizer

    raw = unfenced(reading)
    unwrapped = _PAGE_NUMBER_TAG_RE.sub("", raw)
    unwrapped = _FURNITURE_BLOCK_TAG_RE.sub("\n\n", unwrapped)
    text = OutputNormalizer().normalize(clean_residual_html(unwrapped))
    if _TAG_RE.search(text):
        return "", "the converted reading still holds markup socr has no conversion for"
    if word_tokens(text) != word_tokens(raw):
        return "", "the conversion to page markdown would change the reading's words"
    if row_numbers(text) != row_numbers(raw):
        return "", "the conversion to page markdown would change the reading's numbers"
    return text, ""


# --- one page ---------------------------------------------------------------------------


def _reader(reading: dict) -> dict:
    from socr.pipeline.refinalize_second_reader import _sha256

    return {
        "reader_model": reading.get("reader_model"),
        "prompt_sha256": reading.get("prompt_sha256"),
        "render": reading.get("render"),
        "finish_reason": reading.get("finish_reason"),
        "text_sha256": _sha256(reading["text"]),
    }


def compare_page(doc: str, path: Path, sidecar: dict, a: dict, b: dict) -> _Compared:
    """Compare one selected page's two readings and, on agreement, build what it becomes."""
    from socr.core.manifest import (
        PageDisposition,
        PageEnding,
        PagePrimaryReason,
        apply_verdict_marks,
    )
    from socr.core.pass_trace import TraceEntry, TraceResult, trace_from_dicts, trace_to_dicts
    from socr.core.result import FailureMode, PageOutput, PageStatus
    from socr.core.state import PageState
    from socr.core.verdict import (
        PageVerdict,
        published_audit_passed,
        reconcile_pass_trace,
        stamp_text_digest,
    )
    from socr.pipeline.orchestrator import _page_blob_key, _socr_source_digest, _socr_version
    from socr.pipeline.refinalize_second_reader import _sha256

    page_num = int(sidecar["page_num"])
    disposition = sidecar.get("disposition") or {}
    before = (str(disposition.get("primary_reason") or ""), str(sidecar.get("status") or ""))
    verdict: TwoReaderVerdict = compare_two_readings(
        a["text"],
        b["text"],
        finish_reason_a=a.get("finish_reason"),
        finish_reason_b=b.get("finish_reason"),
    )
    old_text = (sidecar.get("winning_output") or {}).get("text") or ""
    record = {
        "outcome": OUTCOME_DISAGREE,
        "before": {
            "status": before[1],
            "failure_mode": sidecar.get("failure_mode"),
            "disposition": sidecar.get("disposition"),
            "text_sha256": _sha256(old_text),
        },
        "readers": {"a": _reader(a), "b": _reader(b)},
        "comparison": verdict.to_dict(),
    }
    if not verdict.agree:
        result = PageResult(doc, page_num, OUTCOME_DISAGREE, before, verdict.reasons())
        return _Compared(path, sidecar, result, record, (a, b))
    new_text, refused = shipped_text(a["text"])
    if refused:
        record["outcome"] = OUTCOME_CONVERSION
        record["conversion"] = refused
        result = PageResult(doc, page_num, OUTCOME_CONVERSION, before, (refused,))
        return _Compared(path, sidecar, result, record, (a, b))

    output = PageOutput(
        page_num=page_num,
        text=new_text,
        status=PageStatus.SUCCESS,
        failure_mode=FailureMode.NONE,
        engine=ENGINE,
        audit_passed=True,
    )
    final, page_verdict = apply_verdict_marks(
        output, PageState(page_num=page_num), page_num, published=PageVerdict(page_num=page_num)
    )
    if page_verdict.clauses or final.status is not PageStatus.SUCCESS or final.text != new_text:
        raise RuntimeError(f"{path}: the verdict seam did not publish the agreed text clean")
    winning = final.to_dict()
    agreement = {
        "rule": RULE,
        "rule_version": RULE_VERSION,
        "shipped_reading": "a",
        "readers": record["readers"],
        "number_rows": verdict.number_rows,
        "words": verdict.words,
        "tables": verdict.tables,
        "shipped_text_sha256": _sha256(new_text),
    }
    new_disposition = PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.ACCEPTED_OUTPUT
    ).to_dict()
    out = dict(sidecar)
    # Clear stale failed-run sidecar fields from the replaced reading so resume
    # cannot reject or re-demote this lifted page on old table/judge/floor state.
    for key, default in _RESET_SIDECAR_DEFAULTS.items():
        if key in out:
            out[key] = (
                list(default)
                if isinstance(default, list)
                else (dict(default) if isinstance(default, dict) else default)
            )
    for key in _POP_SIDECAR_KEYS:
        out.pop(key, None)
    out["disposition"] = new_disposition
    out["status"] = winning.get("status", "missing")
    out["failure_mode"] = winning.get("failure_mode", "none")
    out["audit_passed"] = published_audit_passed(out["status"], True, page_verdict)
    out["engine"] = ENGINE
    out["provider"] = ""
    out["verdict"] = page_verdict.to_dict()
    # The trace: nothing the run checked was checked on this text, so its PASSes and its
    # clauses become NOT_RUN (``reconcile_pass_trace``), and the one check that did run on
    # it, this rule, is stated with the text's digest. No trace is invented for a sidecar
    # that had none: the entry then stands alone.
    kept = reconcile_pass_trace(sidecar.get("checks"), page_verdict, text=new_text) or []
    entry = stamp_text_digest(
        TraceEntry(AGREEMENT_KIND, TraceResult.PASS, reason=AGREEMENT_REASON), new_text
    )
    entries = [e for e in (trace_from_dicts(kept) or ()) if e.check != AGREEMENT_KIND]
    out["checks"] = trace_to_dicts(sorted([*entries, entry], key=lambda e: (e.check, e.scope)))
    out["winning_output"] = winning
    out["page_fingerprint"] = _page_blob_key(winning)
    drop_kinds = _replaced_event_kinds()
    out["audit_events"] = [
        *[e for e in sidecar.get("audit_events") or () if e.get("kind") not in drop_kinds],
        {
            "kind": AGREEMENT_KIND,
            "engine": ENGINE,
            "detail": AGREEMENT_EVENT_DETAIL,
            "data": agreement,
        },
    ]
    out[AGREEMENT_KIND] = agreement
    out["refinalized_from"] = {
        "tool": TOOL,
        "disposition": sidecar.get("disposition"),
        "status": sidecar.get("status"),
        "failure_mode": sidecar.get("failure_mode"),
        "engine": sidecar.get("engine"),
        "verdict": sidecar.get("verdict"),
        "text_sha256": _sha256(old_text),
        "socr_version": sidecar.get("socr_version"),
        "socr_source_digest": sidecar.get("socr_source_digest"),
    }
    out["socr_version"] = _socr_version()
    out["socr_source_digest"] = _socr_source_digest()
    record["outcome"] = OUTCOME_LIFTED
    record["after"] = {
        "status": out["status"],
        "failure_mode": out["failure_mode"],
        "disposition": new_disposition,
        "text_sha256": agreement["shipped_text_sha256"],
    }
    record["socr_source_digest"] = out["socr_source_digest"]
    result = PageResult(doc, page_num, OUTCOME_LIFTED, before)
    return _Compared(path, sidecar, result, record, (a, b), new_text, out)


# --- one document -----------------------------------------------------------------------


def _reading(root: Path, doc_name: str, path: Path, page_num: int, checksum: str):
    """``(reading, "")`` from *root*, or ``(None, why not)``."""
    from socr.pipeline.refinalize_second_reader import _usable, reading_path

    source = reading_path(root, doc_name, path)
    if not source.is_file():
        return None, "no reading at hand"
    try:
        reading = json.loads(source.read_text(encoding="utf-8"))
    except ValueError:
        reading = None
    why = _usable(reading, page_num, checksum)
    return (None, why) if why else (reading, "")


def refinalize_document(
    doc_dir: Path,
    pdf_dir: Path,
    readings_a: Path,
    readings_b: Path,
    *,
    apply: bool = False,
    skipped: Counter | None = None,
) -> list[PageResult]:
    """Compare one document's selected pages. Nothing is written until every page of the
    document has been decided, so an inconsistency leaves the document as it was."""
    from socr.core.pdf import open_pdf
    from socr.pipeline.refinalize_own_text import (
        _analyse_pdf,
        _read_json,
        _recount_tables,
        _swap_once,
        _write_json,
        _write_text,
    )
    from socr.pipeline.refinalize_prior_ocr import _drop_from_error
    from socr.pipeline.refinalize_second_reader import _has_own_layer

    skipped = Counter() if skipped is None else skipped
    pages_dir = doc_dir / "pages"
    meta_path = doc_dir / "metadata.json"
    if not pages_dir.is_dir():
        return []
    candidates = []
    for path in page_sidecar_paths(pages_dir, "[0-9]*.json"):
        sidecar = _read_json(path)
        why = selects(sidecar)
        if why:
            skipped[f"not selected: {why}"] += 1
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
    checksum = str(meta.get("checksum") or "")
    wanted = []
    for path, sidecar in candidates:
        page_num = int(sidecar["page_num"])
        a, why_a = _reading(readings_a, doc_dir.name, path, page_num, checksum)
        b, why_b = _reading(readings_b, doc_dir.name, path, page_num, checksum)
        if a is None or b is None:
            skipped[f"not compared: reading A: {why_a or 'ok'}; reading B: {why_b or 'ok'}"] += 1
            continue
        if not a.get("reader_model") or a.get("reader_model") == b.get("reader_model"):
            skipped["not compared: the two readings are not from two different reader models"] += 1
            continue
        wanted.append((path, sidecar, a, b))
    if not wanted:
        return []
    assessment, why_not = _analyse_pdf(pdf_dir, meta)
    if assessment is None:
        skipped[why_not] += len(wanted)
        return []

    doc_md = md_paths[0].read_text(encoding="utf-8")
    compared: list[_Compared] = []
    with open_pdf(pdf_dir / str(meta.get("key") or ""), repair=True) as pdf:
        for path, sidecar, a, b in wanted:
            page_num = int(sidecar["page_num"])
            if _has_own_layer(sidecar, assessment.pages[page_num - 1], pdf[page_num - 1]):
                skipped["not selected: the page has a text layer of its own"] += 1
                continue
            c = compare_page(doc_dir.name, path, sidecar, a, b)
            if c.new_sidecar is not None:
                fragment_path = path.with_suffix(".md")
                old_text = (sidecar.get("winning_output") or {}).get("text") or ""
                if fragment_path.read_text(encoding="utf-8") not in (old_text, c.new_text):
                    raise RuntimeError(f"{fragment_path}: fragment is not its sidecar's text")
                heading = f"## Page {page_num}\n\n"
                doc_md = _swap_once(doc_md, heading + old_text, heading + c.new_text, md_paths[0])
            compared.append(c)
    results = [c.result for c in compared]
    if not apply or not compared:
        return results

    lifted = [c for c in compared if c.new_sidecar is not None]
    manifest_path = doc_dir / "manifest.json"
    manifest = _read_json(manifest_path) if lifted and manifest_path.is_file() else None
    if manifest is not None:
        missing = [
            str(c.result.page_num)
            for c in lifted
            if str(c.result.page_num) not in (manifest.get("entries") or {})
        ]
        if missing:
            raise RuntimeError(f"{manifest_path}: no entry for page(s) {', '.join(missing)}")

    for c in lifted:
        for side, reading in zip(("a", "b"), c.readings, strict=True):
            own = doc_dir / READINGS_DIR / side / c.path.name
            own.parent.mkdir(parents=True, exist_ok=True)
            _write_json(own, reading)
    for c in lifted:
        _write_text(c.path.with_suffix(".md"), c.new_text)
    if lifted:
        _write_text(md_paths[0], doc_md)
    if manifest is not None:
        from socr.core.cache import BlobStore
        from socr.core.result import PageOutput

        store = BlobStore(doc_dir / "cache")
        for c in lifted:
            entry = manifest["entries"][str(c.result.page_num)]
            entry["blob_ref"] = store.put_page(
                PageOutput.from_dict(c.new_sidecar["winning_output"])
            )
            entry["disposition"] = c.new_sidecar["disposition"]
            # The page shows no picture any more; a recorded asset it does not reference
            # would be reported stale for ever. The file itself stays on disk.
            kept = [
                asset
                for asset in entry.get("assets") or []
                if str(asset.get("logical_path") or "") in c.new_text
            ]
            if kept:
                entry["assets"] = kept
            else:
                entry.pop("assets", None)
        _write_json(manifest_path, manifest)
    log_path = doc_dir / LOG_NAME
    log = _read_json(log_path) if log_path.is_file() else {"tool": TOOL}
    pages = dict(log.get("pages") or {})
    for c in compared:
        pages[str(c.result.page_num)] = c.record
    log["pages"] = {n: pages[n] for n in sorted(pages, key=int)}
    if not log_path.is_file() or _read_json(log_path) != log:
        _write_json(log_path, log)
    if lifted:
        from socr.core.verdict import pages_not_success_note
        from socr.pipeline.refinalize_point_drops import _replace_note

        changed = {c.result.page_num for c in lifted}
        error = _drop_from_error(meta.get("error"), changed)
        # #1244's note, for the pages that are still not SUCCESS with a passing audit.
        unfinished = sorted(
            int(sc["page_num"])
            for sc in map(_read_json, page_sidecar_paths(pages_dir, "[0-9]*.json"))
            if sc.get("terminal") is True
            and not (sc.get("status") == "success" and sc.get("audit_passed") is True)
        )
        still = [n for n in unfinished if n not in changed]
        # Only a note the run wrote is rewritten: none is ever added.
        if error and unfinished and pages_not_success_note(unfinished) in error:
            error = (
                _replace_note(
                    error,
                    pages_not_success_note(unfinished),
                    pages_not_success_note(still) if still else None,
                )
                or None
            )
        audit_events = _rewrite_audit_log(doc_dir, meta, lifted)
        trust_changed, new_trust_note = _rewrite_tables_trust(doc_dir, meta, audit_events)
        if trust_changed and error:
            from socr.core.tables_trust import TRUST_NOTE_PREFIX

            parts = [p for p in error.split("; ") if p]
            replaced_trust = False
            rebuilt: list[str] = []
            for part in parts:
                if part.startswith(TRUST_NOTE_PREFIX):
                    replaced_trust = True
                    if new_trust_note:
                        rebuilt.append(new_trust_note)
                else:
                    rebuilt.append(part)
            if replaced_trust:
                error = "; ".join(rebuilt) or None
        if error != meta.get("error"):
            if error is None:
                meta.pop("error", None)
            else:
                meta["error"] = error
            _write_json(meta_path, meta)
            root_path = doc_dir.parent / "metadata.json"
            if root_path.is_file():
                root = _read_json(root_path)
                entry = (root.get("files") or {}).get(meta.get("key"))
                if isinstance(entry, dict):
                    if error is None:
                        entry.pop("error", None)
                    else:
                        entry["error"] = error
                    _write_json(root_path, root)
    for c in lifted:
        _write_json(c.path, c.new_sidecar)
    if lifted:
        _recount_tables(doc_dir, meta_path, meta)
    return results


def _rewrite_audit_log(doc_dir: Path, meta: dict, lifted: list[_Compared]) -> list[dict]:
    """The document's audit log: the lifted pages' replaced-reading events go, one agreement
    event per lifted page comes in (replaced in place on a rerun). Returns the final events."""
    from socr.pipeline.refinalize_own_text import _read_json, _write_json

    path = doc_dir / "audit_log.json"
    audit = (
        _read_json(path)
        if path.is_file()
        else {
            "pdf_filename": Path(str(meta.get("key") or "")).name,
            "event_count": 0,
            "counts": {},
            "events": [],
        }
    )
    changed = {c.result.page_num for c in lifted}
    drop_kinds = _replaced_event_kinds() - {AGREEMENT_KIND}
    events = [
        e
        for e in audit.get("events") or ()
        if not (e.get("page_num") in changed and e.get("kind") in drop_kinds)
    ]
    for c in lifted:
        page_num = c.result.page_num
        own = next(e for e in c.new_sidecar["audit_events"] if e["kind"] == AGREEMENT_KIND)
        new_event = {"page_num": page_num, **own}
        at = [
            i
            for i, e in enumerate(events)
            if e.get("page_num") == page_num and e.get("kind") == AGREEMENT_KIND
        ]
        if at:
            events[at[0]] = new_event
            events = [e for i, e in enumerate(events) if i not in at[1:]]
            continue
        mine = [i for i, e in enumerate(events) if e.get("page_num") == page_num]
        end = (
            mine[-1] + 1 if mine else sum(1 for e in events if (e.get("page_num") or 0) <= page_num)
        )
        events.insert(end, new_event)
    if events != (audit.get("events") or []):
        audit["events"] = events
        audit["event_count"] = len(events)
        audit["counts"] = dict(Counter(e.get("kind") for e in events))
        _write_json(path, audit)
    return events


def _rewrite_tables_trust(
    doc_dir: Path, meta: dict, audit_events: list[dict]
) -> tuple[bool, str | None]:
    """Recompute ``tables_trust.json`` from the updated document events.

    Returns ``(changed_existing_trust, new_trust_note)``: removes stale ``tables_trust.json``
    when lifting retired every untrusted table, or rewrites it with only the still-untrusted
    pages. Leaves documents that had no ``tables_trust.json`` untouched if still clean.
    """
    from socr.core.audit_log import AuditEvent
    from socr.core.tables_trust import build_tables_trust, trust_note

    trust_path = doc_dir / "tables_trust.json"
    had_trust = trust_path.is_file()
    old_bytes = trust_path.read_bytes() if had_trust else None
    event_objs = [
        AuditEvent(
            page_num=int(e.get("page_num") or 0),
            kind=str(e.get("kind") or ""),
            engine=str(e.get("engine") or ""),
            detail=str(e.get("detail") or ""),
            data=dict(e.get("data") or {}),
        )
        for e in audit_events
        if isinstance(e, dict)
    ]
    trust = build_tables_trust(Path(str(meta.get("key") or "")).name, event_objs)
    if not trust.pages and not trust.advisory_native_cell_pages:
        if had_trust:
            trust_path.unlink()
            return True, None
        return False, None
    trust.save(trust_path)
    new_bytes = trust_path.read_bytes()
    return (had_trust and new_bytes != old_bytes), trust_note(trust)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=TOOL, description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--pdf-dir", type=Path, required=True, help="the run's input PDFs")
    parser.add_argument(
        "--readings-dir",
        type=Path,
        required=True,
        help="reader A, whose reading ships: <dir>/<document directory name>/<NNNNN>.json",
    )
    parser.add_argument(
        "--readings-dir-b", type=Path, required=True, help="reader B, which confirms"
    )
    parser.add_argument(
        "--apply", action="store_true", help="write (default: print, write nothing)"
    )
    args = parser.parse_args(argv)
    counts: Counter = Counter()
    reasons: Counter = Counter()
    skipped: Counter = Counter()
    failed: list[tuple[Path, str]] = []
    for pages_dir in sorted(p for p in args.out_dir.rglob("pages") if p.is_dir()):
        doc_dir = pages_dir.parent
        try:
            results = refinalize_document(
                doc_dir,
                args.pdf_dir,
                args.readings_dir,
                args.readings_dir_b,
                apply=args.apply,
                skipped=skipped,
            )
        except Exception as exc:  # noqa: BLE001 -- reported per document, never swallowed
            failed.append((doc_dir, f"{type(exc).__name__}: {exc}"))
            continue
        for result in results:
            counts[(result.outcome, *result.before)] += 1
            for reason in result.reasons:
                reasons[reason] += 1
            if result.outcome == OUTCOME_LIFTED:
                print(f"{result.doc} p{result.page_num}: {'/'.join(result.before)} -> lifted")
    verb = "lifted" if args.apply else "would be lifted"
    total = sum(n for (outcome, *_rest), n in counts.items() if outcome == OUTCOME_LIFTED)
    print(f"{total:6d}  pages {verb}")
    for (outcome, reason, status), n in sorted(counts.items()):
        print(f"{n:6d}  compared: {outcome}  (was {reason}/{status})")
    for reason, n in sorted(reasons.items()):
        print(f"{n:6d}  did not agree: {reason}")
    if not args.apply:
        print("        (dry run: nothing written; pass --apply)")
    for reason, n in sorted(skipped.items()):
        print(f"{n:6d}  {reason}")
    for doc_dir, why in failed:
        print(f"FAILED  {doc_dir}: {why}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
