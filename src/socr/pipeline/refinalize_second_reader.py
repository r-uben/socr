"""#1409: mark the unmarked pages of a FINISHED run whose numbers a second reader disputes.

    socr-refinalize-second-reader <out_dir> --pdf-dir <dir> [--apply]
        [--readings-dir <dir>]
        [--reader-url <url> --reader-model <id> --reader-prompt-file <file>]

A hand check of 150 unmarked pages found wrong numbers no check of socr's caught (#1411 lost
minus signs, #1412 broken fractions, a doubled fraction, hidden PDF text). A second reader
that looks only at the page IMAGE read them correctly. This tool compares each selected
page's shipped numbers with that reader's reading (``tables.second_reader``) and, on a
difference, puts a ``second_reader_numbers`` mark on the page.

It only ever ADDS that one clause. It never lifts a mark, never changes a page's shipped
text below the mark, and never ships anything the second reader wrote.

DRY-RUN BY DEFAULT: without ``--apply`` it prints what ``--apply`` would mark, writes
nothing and calls no reader. A page with no reading at hand is counted, not read.

Which pages
-----------
A finished page (``terminal``) that is SUCCESS, carries no verdict clause and no mark line,
and ships at least one number. A page that already carries any mark is never touched,
whatever a reading says.

Readings
--------
ONE on-disk format (:func:`reading_record`), one JSON file per page, named like the page's
sidecar: ``<dir>/<document directory name>/<NNNNN>.json``. The tool keeps its own under the
document (``<doc_dir>/second_reader/``, :data:`READINGS_DIR`), so a rerun calls no reader;
``--readings-dir`` imports readings made elsewhere (a cluster batch job writes the same
files) and, with ``--apply``, copies the ones it uses into the document. A reading is used
only when it names this page and the run's input file (``input_checksum``); the document's
own copy wins over an imported one.

A reading is made here only with ``--apply`` and all three ``--reader-*`` arguments, through
``engines.vllm.VLLMEngine.read_page``. The prompt is the READER's own and is not part of socr:
it is read from ``--reader-prompt-file`` and its SHA-256 is recorded in each reading.

Charts
------
The second reader tabulates charts with invented values. A page that ships a picture is
compared with ``chart_label_text``: the words of the page's own text layer that touch a
figure box the page sidecar records (``figure_refs[].bbox``, in the page's displayed frame,
so words are rotated into it). See ``tables.second_reader`` for what that rule gives up.
A picture page with no text layer of its own (a scan, or a layer that is an earlier OCR
run) has no such labels: it is NOT compared, and is recorded ``not_applicable``. That is
not a pass. Chart boxes are not masked in the image in this version.

What a marked page becomes
--------------------------
Through the seam a run uses (``core.manifest.apply_verdict_marks``): DISAGREE is a FAIL
clause naming every differing row, UNKNOWN (the reading was empty or cut off) an UNKNOWN
clause; the page becomes WARNING with ``numbers_disagree_with_second_reader`` or
``second_reader_unusable``. Then, in this order: the readings used, ``pages/NNN.md`` and
the page's section of the document ``.md``, the manifest ``blob_ref`` and cache,
``refinalized_second_reader.json`` (every compared page with its outcome, AGREE included),
the document ``audit_log.json`` (one event per FAIL page), the sidecars, then the document
note and status (``completed`` becomes ``partial``, here and in the root index) and the
table counts. A page that AGREES is not written at all: its bytes are the run's.

Idempotent: a marked page is no longer selected, and an AGREE page is compared again from
its stored reading with the same result, so a second ``--apply`` changes nothing and calls
no reader. A run interrupted anywhere is completed by running the tool again.

A resume of the run does not replace this tool and does not undo it on the same code: the
run fingerprint carries a digest of socr's source, so a resume on OTHER code re-reads every
page (the marks are then lost with the pages, and this tool must be run again).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from socr.core.page_files import page_sidecar_paths
from socr.tables.second_reader import (
    UNKNOWN_EMPTY,
    UNKNOWN_NO_NUMBERS,
    UNKNOWN_TRUNCATED,
    UNKNOWN_UNCLOSED,
    Agreement,
    SecondReaderVerdict,
    compare_readings,
    describe_differences,
    shipped_rows,
)

TOOL = "socr-refinalize-second-reader"
#: The per-document record of every page this tool compared.
LOG_NAME = "refinalized_second_reader.json"
#: Where a document keeps the readings this tool used, one file per page.
READINGS_DIR = "second_reader"
#: Version of the reading file format (:func:`reading_record`).
READING_FORMAT = 1

#: The page image the reader is given: the setting the 79-page evaluation used (300 DPI PNG,
#: ``docs/log/2026-11-26_1409_second_reader_eval.py``). Recorded in every reading.
READER_RENDER_DPI = 300
#: The reader's output budget and temperature in that evaluation (the model card's values).
READER_MAX_TOKENS = 24576
READER_TEMPERATURE = 0.0
#: The request timeout the evaluation's readings were made with, in seconds.
READER_TIMEOUT_S = 3600.0
#: ``finish_reason`` of an output the server cut at the token budget.
FINISH_REASON_CUT_OFF = "length"

#: The audit event a document records for each page the clause FAILs.
SECOND_READER_DISAGREE_KIND = "numbers_disagree_with_second_reader"
SECOND_READER_DISAGREE_EVENT_DETAIL = (
    "the numbers this page ships differ from a second, independent reading of the page "
    "image; the rows are named in the page's mark, nothing was corrected, and the page is "
    "never counted CHECKED (#1409)"
)
NOTE_PREFIX = "shipped numbers differ from a second reading of the page image on page(s) "

OUTCOME_NOT_APPLICABLE = "not_applicable"
NOT_APPLICABLE_SCAN_PICTURE = (
    "the page ships a picture and has no text layer of its own to tell the chart's labels "
    "from its body, so it was not compared"
)

_UNKNOWN_REASONS = {
    UNKNOWN_EMPTY: "a second reading of the page image came back empty, so the page's "
    "numbers were not compared",
    UNKNOWN_TRUNCATED: "a second reading of the page image was cut off, so the page's "
    "numbers were not compared",
    UNKNOWN_UNCLOSED: "a second reading of the page image ends inside a table or a header, "
    "so the page's numbers were not compared",
    UNKNOWN_NO_NUMBERS: "neither the page nor a second reading of its image holds a number",
}

#: ``(png bytes) -> (raw reader text, finish_reason or None)``
ReadPage = Callable[[bytes], "tuple[str, str | None]"]


def disagree_note(pages: list[int]) -> str:
    """The document's ``error`` note naming the pages this clause FAILs."""
    return NOTE_PREFIX + ", ".join(str(n) for n in pages) + " (named on each page)"


# --- readings ---------------------------------------------------------------------------


def reading_record(
    *,
    page_num: int,
    input_checksum: str,
    text: str,
    finish_reason: str | None,
    reader_model: str,
    prompt_sha256: str | None,
    render_dpi: int | None,
) -> dict:
    """One page's second reading, as it is stored. The ONE format: this tool's own readings,
    and readings imported with ``--readings-dir``, are this dict as JSON.

    ``input_checksum`` is the run's input file (``metadata.json``'s ``checksum``).
    ``finish_reason``, ``prompt_sha256`` and ``render_dpi`` are ``None`` when whoever made the
    reading did not record them: unknown, never assumed."""
    return {
        "format": READING_FORMAT,
        "page_num": page_num,
        "input_checksum": input_checksum,
        "reader_model": reader_model,
        "prompt_sha256": prompt_sha256,
        "render": {"dpi": render_dpi, "image_format": "png", "chart_boxes_masked": False},
        "finish_reason": finish_reason,
        "text": text,
    }


def reading_path(readings_root: Path, doc_name: str, sidecar_path: Path) -> Path:
    """Where the reading of the page with sidecar *sidecar_path* lives under *readings_root*."""
    return readings_root / doc_name / sidecar_path.name


def _usable(reading, page_num: int, input_checksum: str) -> str:
    """Why *reading* cannot be used for this page, or ``""``."""
    if not isinstance(reading, dict) or reading.get("format") != READING_FORMAT:
        return "the reading file is not in this tool's format"
    if reading.get("page_num") != page_num:
        return "the reading file is another page's"
    if reading.get("input_checksum") != input_checksum:
        return "the reading was made from another input file (checksum)"
    if not isinstance(reading.get("text"), str):
        return "the reading file holds no text"
    return ""


def render_page_png(page) -> bytes:
    """The page image the reader is given (:data:`READER_RENDER_DPI`)."""
    import fitz

    scale = READER_RENDER_DPI / 72
    return page.get_pixmap(matrix=fitz.Matrix(scale, scale)).tobytes("png")


def vllm_reader(base_url: str, model: str, prompt: str) -> ReadPage:
    """A :data:`ReadPage` over an OpenAI-compatible endpoint (``engines.vllm``)."""
    from socr.engines.vllm import VLLMConfig, VLLMEngine

    engine = VLLMEngine(VLLMConfig(base_url=base_url, model=model, timeout=READER_TIMEOUT_S))

    def read(png: bytes) -> tuple[str, str | None]:
        return engine.read_page(
            png, prompt, max_tokens=READER_MAX_TOKENS, temperature=READER_TEMPERATURE
        )

    return read


# --- one page ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageResult:
    doc: str
    page_num: int
    #: ``agree`` / ``disagree`` / ``unknown`` / ``not_applicable``
    outcome: str
    #: the mark line the page gains ("" when it gains none)
    mark: str = ""
    reason: str = ""


@dataclass
class _Compared:
    """Everything one page's comparison produces, before anything is written."""

    path: Path
    sidecar: dict
    result: PageResult
    reading: dict
    #: the reading has to be written into the document (it was imported or just made)
    store_reading: bool
    record: dict
    #: set for a page that gains the clause
    output: object | None = None  # PageOutput
    new_sidecar: dict | None = None
    old_text: str = ""


def selects(sidecar: dict) -> str:
    """Why this page is not compared, or ``""`` when it is selected."""
    from socr.core.marks import has_any_mark_line
    from socr.core.result import PageStatus

    if sidecar.get("terminal") is not True:
        return "page not finished (terminal is not true)"
    winning = sidecar.get("winning_output") or {}
    if sidecar.get("status") != PageStatus.SUCCESS.value or (
        winning.get("status") != PageStatus.SUCCESS.value
    ):
        return "page is not SUCCESS"
    if (sidecar.get("verdict") or {}).get("clauses"):
        return "page already carries a verdict clause"
    text = winning.get("text") or ""
    if has_any_mark_line(text, int(sidecar["page_num"])):
        return "page already carries a mark"
    if not shipped_rows(text):
        return "page ships no number"
    return ""


def figure_boxes(sidecar: dict) -> list[list[float]]:
    """The boxes of the pictures this page ships, as its sidecar records them."""
    return [
        [float(v) for v in ref["bbox"]]
        for ref in sidecar.get("figure_refs") or []
        if isinstance(ref, dict) and isinstance(ref.get("bbox"), list) and len(ref["bbox"]) == 4
    ]


def chart_label_text(page, boxes: list[list[float]]) -> str:
    """The words of the page's text layer that touch one of *boxes*.

    The boxes are in the page's displayed frame and ``get_text("words")`` is not, so each word
    is rotated into it. A box is tight around the drawing and an axis label often straddles
    its edge: a word that TOUCHES a box counts."""
    import fitz

    rects = [fitz.Rect(box) for box in boxes]

    def displayed(word) -> fitz.Rect:
        return (fitz.Rect(word[:4]) * page.rotation_matrix).normalize()

    return " ".join(
        str(word[4])
        for word in page.get_text("words")
        if any(displayed(word).intersects(rect) for rect in rects)
    )


def _has_own_layer(sidecar: dict, pa, page) -> bool:
    """Does the page have words of its own (not an earlier OCR run's) in its text layer?"""
    from socr.pipeline.refinalize_own_text import _analysis_finds_prior_ocr

    if sidecar.get("prior_ocr_layer_suspect") or _analysis_finds_prior_ocr(pa):
        return False
    return any(str(word[4]).strip() for word in page.get_text("words"))


def clause_for(verdict: SecondReaderVerdict):
    """The ``second_reader_numbers`` clause of a comparison, ``None`` when it agrees."""
    from socr.core.marks import SCOPE_PAGE, Check, CheckResult
    from socr.core.verdict import VerdictClause

    if verdict.outcome is Agreement.AGREE:
        return None
    if verdict.outcome is Agreement.DISAGREE:
        return VerdictClause(
            Check.SECOND_READER_NUMBERS,
            CheckResult.FAIL,
            SCOPE_PAGE,
            describe_differences(verdict),
        )
    return VerdictClause(
        Check.SECOND_READER_NUMBERS,
        CheckResult.UNKNOWN,
        SCOPE_PAGE,
        _UNKNOWN_REASONS[verdict.unknown_reason],
    )


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def compare_page(
    doc: str,
    path: Path,
    sidecar: dict,
    pa,
    page,
    reading: dict,
    store_reading: bool,
) -> _Compared:
    """Compare one selected page with its reading and, on a difference, finalize it."""
    from socr.core.manifest import apply_verdict_marks, is_page_failed_marker
    from socr.core.marks import leading_marks
    from socr.core.result import PageOutput
    from socr.core.verdict import (
        PageVerdict,
        in_fresh_run_order,
        published_audit_passed,
        rerun_text_dependent_checks,
        write_reconciled_trace,
    )
    from socr.pipeline.orchestrator import _page_blob_key, _socr_source_digest, _socr_version
    from socr.pipeline.refinalize_own_text import _page_state

    page_num = int(sidecar["page_num"])
    output = PageOutput.from_dict(sidecar.get("winning_output") or {})
    old_text = output.text or ""
    boxes = figure_boxes(sidecar)
    ships_picture = bool(boxes) or "![" in old_text
    read = {
        "reader_model": reading.get("reader_model"),
        "prompt_sha256": reading.get("prompt_sha256"),
        "render": reading.get("render"),
        "finish_reason": reading.get("finish_reason"),
        "text_sha256": _sha256(reading["text"]),
    }
    if ships_picture and not _has_own_layer(sidecar, pa, page):
        result = PageResult(
            doc, page_num, OUTCOME_NOT_APPLICABLE, reason=NOT_APPLICABLE_SCAN_PICTURE
        )
        record = {"outcome": result.outcome, "reason": result.reason, "reading": read}
        return _Compared(path, sidecar, result, reading, store_reading, record)
    labels = chart_label_text(page, boxes) if boxes else None
    verdict = compare_readings(
        old_text,
        reading["text"],
        chart_label_text=labels,
        reader_truncated=reading.get("finish_reason") == FINISH_REASON_CUT_OFF,
    )
    record = {
        "outcome": verdict.outcome.value,
        "reading": read,
        "shipped_text_sha256": _sha256(old_text),
        "comparison": verdict.to_dict(),
        "chart_label_words": len(labels.split()) if labels is not None else None,
    }
    clause = clause_for(verdict)
    if clause is None:
        result = PageResult(doc, page_num, verdict.outcome.value)
        return _Compared(path, sidecar, result, reading, store_reading, record)

    published = PageVerdict.from_dict(sidecar.get("verdict") or {})
    extended = replace(
        published,
        page_num=page_num,
        clauses=tuple(in_fresh_run_order((*published.clauses, clause))),
    )
    p = _page_state(sidecar, pa)
    final, new_verdict = apply_verdict_marks(output, p, page_num, published=extended)
    # The two things this tool must never do, held here by construction and not by review:
    # the page keeps every clause it had and gains exactly this one, and the shipped text
    # is the run's text under new leading mark lines.
    if tuple(new_verdict.clauses) != tuple(extended.clauses):
        raise RuntimeError(f"{path}: the verdict seam did not publish exactly the added clause")
    new_text = final.text or ""
    if not new_text.endswith(old_text.lstrip("\n")) or new_text == old_text:
        raise RuntimeError(f"{path}: marking the page would change its shipped text")
    marks = [m for m in leading_marks(new_text, page_num) if m not in old_text]
    winning = final.to_dict()
    new_sidecar = dict(sidecar)
    new_sidecar["status"] = winning.get("status", "missing")
    new_sidecar["failure_mode"] = winning.get("failure_mode", "none")
    new_sidecar["audit_passed"] = published_audit_passed(
        new_sidecar["status"], winning.get("audit_passed", False), new_verdict
    )
    new_sidecar["verdict"] = new_verdict.to_dict()
    write_reconciled_trace(
        new_sidecar,
        sidecar.get("checks"),
        new_verdict,
        recomputed=rerun_text_dependent_checks(final, p, sidecar.get("checks")),
        text=new_text,
    )
    new_sidecar["winning_output"] = winning
    new_sidecar["page_fingerprint"] = (
        _page_blob_key(winning) if not is_page_failed_marker(new_text) else ""
    )
    new_sidecar["socr_version"] = _socr_version()
    new_sidecar["socr_source_digest"] = _socr_source_digest()
    if verdict.outcome is Agreement.DISAGREE:
        new_sidecar["audit_events"] = [
            *[
                e
                for e in sidecar.get("audit_events") or []
                if e.get("kind") != SECOND_READER_DISAGREE_KIND
            ],
            _event(output.engine or "", record),
        ]
    record["before"] = {"status": str(sidecar.get("status"))}
    record["after"] = {"status": new_sidecar["status"], "failure_mode": new_sidecar["failure_mode"]}
    record["socr_source_digest"] = new_sidecar["socr_source_digest"]
    result = PageResult(
        doc, page_num, verdict.outcome.value, mark="\n".join(marks), reason=clause.reason
    )
    return _Compared(
        path, sidecar, result, reading, store_reading, record, final, new_sidecar, old_text
    )


def _event(engine: str, record: dict) -> dict:
    return {
        "kind": SECOND_READER_DISAGREE_KIND,
        "engine": engine,
        "detail": SECOND_READER_DISAGREE_EVENT_DETAIL,
        "data": {
            "differences": record["comparison"]["differences"],
            "reader_model": record["reading"]["reader_model"],
        },
    }


# --- one document -----------------------------------------------------------------------


def _reading_for(
    doc_dir: Path,
    path: Path,
    page_num: int,
    checksum: str,
    readings_dir: Path | None,
    skipped: Counter,
):
    """``(reading, it must be stored in the document)`` from the document's own readings,
    then *readings_dir*; ``(None, False)`` when neither holds a usable one."""
    from socr.pipeline.refinalize_own_text import _read_json

    own = doc_dir / READINGS_DIR / path.name
    sources = [(own, False)]
    if readings_dir is not None:
        sources.append((reading_path(readings_dir, doc_dir.name, path), True))
    for source, store in sources:
        if not source.is_file():
            continue
        try:
            reading = _read_json(source)
        except ValueError:
            reading = None
        why = _usable(reading, page_num, checksum)
        if not why:
            return reading, store
        skipped[f"reading not used: {why}"] += 1
    return None, False


def refinalize_document(
    doc_dir: Path,
    pdf_dir: Path,
    *,
    apply: bool = False,
    readings_dir: Path | None = None,
    read_page: ReadPage | None = None,
    reader_model: str = "",
    prompt_sha256: str | None = None,
    skipped: Counter | None = None,
) -> list[PageResult]:
    """Compare one document's selected pages. Nothing of the run's is written until every
    page of the document has been decided, so an inconsistency leaves the document as it was.
    A reading just MADE is stored at once (it cost a model call and changes nothing else)."""
    from socr.core.pdf import open_pdf
    from socr.pipeline.refinalize_own_text import (
        _analyse_pdf,
        _read_json,
        _swap_once,
        _write_json,
        _write_text,
    )

    skipped = Counter() if skipped is None else skipped
    pages_dir = doc_dir / "pages"
    meta_path = doc_dir / "metadata.json"
    if not pages_dir.is_dir():
        return []
    sidecar_paths = page_sidecar_paths(pages_dir, "[0-9]*.json")
    candidates = []
    for path in sidecar_paths:
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
    # Decide which pages have a reading before the (slow) page analysis of the PDF.
    wanted = []
    for path, sidecar in candidates:
        page_num = int(sidecar["page_num"])
        reading, store = _reading_for(doc_dir, path, page_num, checksum, readings_dir, skipped)
        if reading is None and not (apply and read_page is not None):
            skipped[
                "no reading at hand (nothing compared; a reader is called only with --apply)"
            ] += 1
            continue
        wanted.append((path, sidecar, reading, store))
    if not wanted:
        return []
    assessment, why_not = _analyse_pdf(pdf_dir, meta)
    if assessment is None:
        skipped[why_not] += len(wanted)
        return []

    doc_md = md_paths[0].read_text(encoding="utf-8")
    compared: list[_Compared] = []
    with open_pdf(pdf_dir / str(meta.get("key") or ""), repair=True) as pdf:
        for path, sidecar, reading, store in wanted:
            page_num = int(sidecar["page_num"])
            page = pdf[page_num - 1]
            if reading is None:
                text, finish_reason = read_page(render_page_png(page))
                reading = reading_record(
                    page_num=page_num,
                    input_checksum=checksum,
                    text=text,
                    finish_reason=finish_reason,
                    reader_model=reader_model,
                    prompt_sha256=prompt_sha256,
                    render_dpi=READER_RENDER_DPI,
                )
                own = doc_dir / READINGS_DIR / path.name
                own.parent.mkdir(parents=True, exist_ok=True)
                _write_json(own, reading)
                store = False
            c = compare_page(
                doc_dir.name, path, sidecar, assessment.pages[page_num - 1], page, reading, store
            )
            if c.output is not None:
                fragment_path = path.with_suffix(".md")
                if fragment_path.read_text(encoding="utf-8") != c.old_text:
                    raise RuntimeError(f"{fragment_path}: fragment is not its sidecar's text")
                heading = f"## Page {page_num}\n\n"
                doc_md = _swap_once(
                    doc_md, heading + c.old_text, heading + c.output.text, md_paths[0]
                )
            compared.append(c)
    results = [c.result for c in compared]
    if not apply:
        return results

    done = [c for c in compared if c.output is not None]
    manifest_path = doc_dir / "manifest.json"
    manifest = _read_json(manifest_path) if done and manifest_path.is_file() else None
    if manifest is not None:
        missing = [
            str(c.result.page_num)
            for c in done
            if str(c.result.page_num) not in (manifest.get("entries") or {})
        ]
        if missing:
            raise RuntimeError(f"{manifest_path}: no entry for page(s) {', '.join(missing)}")

    for c in compared:
        if c.store_reading:
            own = doc_dir / READINGS_DIR / c.path.name
            own.parent.mkdir(parents=True, exist_ok=True)
            _write_json(own, c.reading)
    for c in done:
        _write_text(c.path.with_suffix(".md"), c.output.text)
    if done:
        _write_text(md_paths[0], doc_md)
    if manifest is not None:
        from socr.core.cache import BlobStore
        from socr.core.result import PageOutput

        store_ = BlobStore(doc_dir / "cache")
        for c in done:
            entry = manifest["entries"][str(c.result.page_num)]
            entry["blob_ref"] = store_.put_page(
                PageOutput.from_dict(c.new_sidecar["winning_output"])
            )
        _write_json(manifest_path, manifest)
    log_path = doc_dir / LOG_NAME
    log = _read_json(log_path) if log_path.is_file() else {"tool": TOOL}
    pages = log.setdefault("pages", {})
    for c in compared:
        pages[str(c.result.page_num)] = c.record
    log["pages"] = {n: pages[n] for n in sorted(pages, key=int)}
    if not log_path.is_file() or _read_json(log_path) != log:
        _write_json(log_path, log)
    for c in done:
        _write_json(c.path, c.new_sidecar)
    # From the sidecars as they now are, so a pass interrupted before this line is completed
    # by the next one (whose own ``done`` is then empty).
    _record_document(doc_dir, meta_path, meta, sidecar_paths)
    return results


def _fail_pages(sidecar_paths: list[Path]) -> list[tuple[int, dict]]:
    from socr.core.marks import Check
    from socr.pipeline.refinalize_own_text import _has_fail, _read_json

    return [
        (int(sc["page_num"]), sc)
        for sc in map(_read_json, sidecar_paths)
        if _has_fail(sc, Check.SECOND_READER_NUMBERS)
    ]


def _record_document(doc_dir: Path, meta_path: Path, meta: dict, sidecar_paths: list[Path]) -> None:
    """What the document records for its FAIL pages: one audit event each, the ``error`` note
    naming them and ``completed`` -> ``partial``, here and in the root index; then the #1244
    completion rule and the table counts, as the other re-finalize tools apply them. Each
    step checks before writing, so a rerun adds nothing. A document with no page carrying
    this tool's clause is not touched."""
    from ocr_output_contract import Status

    from socr.core.marks import Check
    from socr.pipeline.refinalize_own_text import (
        _read_json,
        _record_unfinished_pages,
        _recount_tables,
        _write_json,
    )

    failing = _fail_pages(sidecar_paths)
    marked = failing or any(
        c.get("check") == Check.SECOND_READER_NUMBERS.value
        for sc in map(_read_json, sidecar_paths)
        for c in (sc.get("verdict") or {}).get("clauses") or []
    )
    if not marked:
        return
    if failing:
        note = disagree_note([n for n, _sc in failing])

        def recorded(record: dict) -> bool:
            changed = False
            if record.get("status") == Status.COMPLETED.value:
                record["status"] = Status.PARTIAL.value
                changed = True
            error = record.get("error") or ""
            kept = [part for part in error.split("; ") if part and not part.startswith(NOTE_PREFIX)]
            placed = "; ".join([*kept, note])
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
        events = list(audit.get("events") or [])
        before = json.dumps(events)
        for page_num, sidecar in failing:
            own = next(
                (
                    e
                    for e in sidecar.get("audit_events") or []
                    if e.get("kind") == SECOND_READER_DISAGREE_KIND
                ),
                None,
            )
            if own is None or any(
                e.get("kind") == SECOND_READER_DISAGREE_KIND and e.get("page_num") == page_num
                for e in events
            ):
                continue
            # After the page's last event; else where the page falls in page order.
            at = [i for i, e in enumerate(events) if e.get("page_num") == page_num]
            end = (
                at[-1] + 1 if at else sum(1 for e in events if (e.get("page_num") or 0) <= page_num)
            )
            events.insert(end, {"page_num": page_num, **own})
        if json.dumps(events) != before:
            counts: dict[str, int] = {}
            for e in events:
                counts[e["kind"]] = counts.get(e["kind"], 0) + 1
            audit.update(event_count=len(events), counts=counts, events=events)
            _write_json(audit_path, audit)
    _record_unfinished_pages(doc_dir, meta_path, meta, sidecar_paths)
    _recount_tables(doc_dir, meta_path, meta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=TOOL, description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--pdf-dir", type=Path, required=True, help="the run's input PDFs")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the marks (default: print them, write nothing, call no reader)",
    )
    parser.add_argument(
        "--readings-dir",
        type=Path,
        help="readings made elsewhere: <dir>/<document directory name>/<NNNNN>.json",
    )
    parser.add_argument("--reader-url", help="OpenAI-compatible endpoint of the second reader")
    parser.add_argument("--reader-model", help="the model id the endpoint serves")
    parser.add_argument(
        "--reader-prompt-file", type=Path, help="the reader's own prompt (not part of socr)"
    )
    args = parser.parse_args(argv)
    reader_args = (args.reader_url, args.reader_model, args.reader_prompt_file)
    if any(reader_args) and not all(reader_args):
        parser.error("--reader-url, --reader-model and --reader-prompt-file go together")
    read_page = None
    prompt_sha = None
    if all(reader_args) and args.apply:
        prompt = args.reader_prompt_file.read_text(encoding="utf-8")
        prompt_sha = _sha256(prompt)
        read_page = vllm_reader(args.reader_url, args.reader_model, prompt)

    counts: Counter = Counter()
    skipped: Counter = Counter()
    failed: list[tuple[Path, str]] = []
    for pages_dir in sorted(p for p in args.out_dir.rglob("pages") if p.is_dir()):
        doc_dir = pages_dir.parent
        try:
            results = refinalize_document(
                doc_dir,
                args.pdf_dir,
                apply=args.apply,
                readings_dir=args.readings_dir,
                read_page=read_page,
                reader_model=args.reader_model or "",
                prompt_sha256=prompt_sha,
                skipped=skipped,
            )
        except Exception as exc:  # noqa: BLE001 -- reported per document, never swallowed
            failed.append((doc_dir, f"{type(exc).__name__}: {exc}"))
            continue
        for result in results:
            counts[result.outcome] += 1
            if result.mark:
                print(f"{result.doc} p{result.page_num} {result.outcome}:\n    {result.mark}")
            elif result.outcome == OUTCOME_NOT_APPLICABLE:
                print(f"{result.doc} p{result.page_num} {result.outcome}: {result.reason}")
    verb = "marked" if args.apply else "would be marked"
    marked = counts[Agreement.DISAGREE.value] + counts[Agreement.UNKNOWN.value]
    print(f"{marked:6d}  pages {verb}")
    for outcome, n in sorted(counts.items()):
        print(f"{n:6d}  compared: {outcome}")
    if not args.apply:
        print("        (dry run: nothing written, no reader called; pass --apply)")
    for reason, n in sorted(skipped.items()):
        print(f"{n:6d}  {reason}")
    for doc_dir, why in failed:
        print(f"FAILED  {doc_dir}: {why}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
