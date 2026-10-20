"""#1185 round 2: ``socr-refinalize-own-text`` decisions that the end-to-end run cannot reach.

Each test is a DIFFERENCE on one input: the run's record versus the current page analysis,
an appended clause versus one placed where a fresh run puts it, a finished page versus a
crash-recovery copy.
"""

from __future__ import annotations

import json

import pytest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import socr
from socr.core import verdict as verdict_module
from socr.core.manifest import apply_verdict_marks
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.pipeline import refinalize_own_text as tool

_LAYER = "Korea 4.0 2.6 5.7 -.8 4.0"
_READING = "Korea 4.0 2.6 5.7 -8 4.0"


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _analysis(*, prior_ocr: bool, layer: str = _LAYER, born_digital: bool = True):
    return SimpleNamespace(
        own_layer_text=layer,
        is_born_digital=born_digital,
        control_byte_digit_hits=0,
        control_byte_scan_failed=False,
        minus_as_digit_hits=0,
        minus_as_digit_scan_failed=False,
        has_unrecovered_symbol_glyphs=False,
        has_encoding_hygiene_suspect=False,
        invisible_text_over_raster=False,
        invisible_text_scan_failed=False,
        prior_ocr_layer=prior_ocr,
        prior_ocr_layer_scan_failed=False,
    )


def _sidecar(output: PageOutput, verdict, **extra) -> dict:
    return {
        "page_num": output.page_num,
        "terminal": True,
        "status": output.status.value,
        "winning_output": output.to_dict(),
        "verdict": verdict.to_dict(),
        **extra,
    }


def _old_run(output: PageOutput, p) -> tuple[PageOutput, object]:
    """What a run made before #1185 published for this page."""
    with mock.patch.object(verdict_module, "text_layer_numbers_clause", lambda o, p: None):
        return apply_verdict_marks(output, p, output.page_num)


def _success(text: str = _READING) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")


def test_a_page_only_the_current_analysis_calls_earlier_ocr_is_marked_not_left_checked() -> None:
    """Fable P1-B: the run did not flag the page, the analysis does. Abstaining (as a run
    may, because #1156's clause fires in the same pass) would leave it unmarked SUCCESS,
    since this tool applies only #1185's clause. The same page the run DID flag is left to
    the run's own mark."""
    old_out, old_v = _old_run(_success(), SimpleNamespace(prior_ocr_layer_suspect=False))
    assert old_out.status is PageStatus.SUCCESS and not old_v.clauses, "fixture premise"

    unflagged = tool.refinalize_page(_sidecar(old_out, old_v), _analysis(prior_ocr=True))
    flagged = tool.refinalize_page(
        _sidecar(old_out, old_v, prior_ocr_layer_suspect=True), _analysis(prior_ocr=True)
    )
    assert flagged is None
    out, verdict = unflagged
    [clause] = verdict.clauses
    assert (clause.check, clause.result) == (
        verdict_module.Check.TEXT_LAYER_NUMBERS,
        verdict_module.CheckResult.UNKNOWN,
    )
    assert clause.reason == tool.PRIOR_OCR_UNMARKED_REASON
    assert out.status is PageStatus.WARNING
    assert out.text.startswith("> [page 1 unchecked: text_layer_numbers=UNKNOWN")


def test_the_clause_takes_the_place_a_fresh_run_gives_it() -> None:
    """Argon 7.1: on a page already WARNING with another clause, a fresh run derives the
    silent-success clauses first; appending put this one last, so the sidecar's verdict
    differed from a fresh run's."""
    p = SimpleNamespace(page_num=1, own_layer_text=_LAYER, is_born_digital=True)
    warned = PageOutput(
        page_num=1,
        text=_READING,
        status=PageStatus.WARNING,
        engine="qwen",
        failure_mode=FailureMode.FIGURE_WORDS_UNREAD,
    )
    fresh_out, fresh_v = apply_verdict_marks(warned, p, 1)
    old_out, old_v = _old_run(warned, p)
    assert len(old_v.clauses) == 1, "fixture premise: the page carries another clause"

    out, verdict = tool.refinalize_page(_sidecar(old_out, old_v), _analysis(prior_ocr=False))
    assert [c.check for c in fresh_v.clauses][0] is verdict_module.Check.TEXT_LAYER_NUMBERS
    assert verdict == fresh_v
    assert out.text == fresh_out.text


def _doc(tmp_path: Path, sidecars: list[dict], *, metadata: bool = True, audit: bool = True):
    doc = tmp_path / "doc"
    (doc / "pages").mkdir(parents=True)
    for sc in sidecars:
        (doc / "pages" / f"{sc['page_num']:05d}.json").write_text(json.dumps(sc))
    if metadata:
        (doc / "metadata.json").write_text(json.dumps({"key": "doc.pdf", "pages": 9}))
    if audit:
        (doc / "audit_log.json").write_text(json.dumps({"events": []}))
    (doc / "doc.md").write_text("")
    return doc


def test_pages_the_tool_cannot_check_are_counted_not_dropped(tmp_path: Path) -> None:
    """Fable P2-6 / Argon 7.3: an unfinished page and a document without metadata.json are
    each counted in ``skipped``; the analysis is never reached. (Round 3: a document without
    audit_log.json is NOT skipped -- a run writes none when it has no events; see below.)"""
    old_out, old_v = _old_run(_success(), SimpleNamespace())
    done = _sidecar(old_out, old_v)
    unfinished = {**_sidecar(old_out, old_v), "page_num": 2, "terminal": False}
    native = {
        **done,
        "page_num": 3,
        "winning_output": {**done["winning_output"], "engine": "native"},
    }

    cases = {
        "page not finished": _doc(tmp_path / "a", [done, unfinished, native]),
        "no metadata.json": _doc(tmp_path / "b", [done], metadata=False),
    }
    with mock.patch.object(tool, "_analyse_pdf", return_value=(None, "no PDF here")):
        for name, doc in cases.items():
            skipped: Counter = Counter()
            assert tool.refinalize_document(doc, tmp_path, skipped=skipped) == []
            counted = {reason: n for reason, n in skipped.items() if reason != "no PDF here"}
            assert sum(counted.values()) == 1, (name, skipped)


def test_a_manifest_without_the_page_leaves_the_document_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    """Argon 7.2: the manifest is checked before the first write, so a document whose
    manifest lacks the page raises with every file as it was, and a rerun cannot find the
    fragments rewritten and the sidecars not."""
    import pytest

    from socr.core.born_digital import DocumentAssessment

    old_out, old_v = _old_run(_success(), SimpleNamespace())
    doc = _doc(tmp_path, [_sidecar(old_out, old_v)])
    (doc / "pages" / "00001.md").write_text(old_out.text)
    (doc / "doc.md").write_text(f"## Page 1\n\n{old_out.text}")
    (doc / "manifest.json").write_text(json.dumps({"entries": {}}))
    assessment = DocumentAssessment(path=tmp_path / "doc.pdf", pages=[_analysis(prior_ocr=False)])
    monkeypatch.setattr(tool, "_analyse_pdf", lambda pdf_dir, meta: (assessment, None))

    before = {p: p.read_bytes() for p in doc.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="no entry for page"):
        tool.refinalize_document(doc, tmp_path)
    assert {p: p.read_bytes() for p in doc.rglob("*") if p.is_file()} == before
    # Premise: with the entry present, the same document IS rewritten.
    (doc / "manifest.json").write_text(json.dumps({"entries": {"1": {}}}))
    assert tool.refinalize_document(doc, tmp_path)


def _finished_doc(tmp_path: Path, output: PageOutput, verdict, *, audit: bool) -> Path:
    """A finished one-page document as a run of the old code leaves it."""
    doc = _doc(tmp_path, [_sidecar(output, verdict)], audit=audit)
    (doc / "metadata.json").write_text(
        json.dumps({"key": "dir/doc.pdf", "pages": 1, "status": "completed"})
    )
    (doc / "pages" / "00001.md").write_text(output.text)
    (doc / "doc.md").write_text(f"## Page 1\n\n{output.text}")
    return doc


def _with_analysis(monkeypatch, tmp_path: Path) -> None:
    from socr.core.born_digital import DocumentAssessment

    assessment = DocumentAssessment(path=tmp_path / "doc.pdf", pages=[_analysis(prior_ocr=False)])
    monkeypatch.setattr(tool, "_analyse_pdf", lambda pdf_dir, meta: (assessment, None))


def test_a_finished_document_without_an_audit_log_is_checked_and_given_one(
    tmp_path: Path, monkeypatch
) -> None:
    """Argon, round-2 review: a run writes ``audit_log.json`` only when the document has
    events, so skipping documents without one left their disputed pages unmarked SUCCESS.
    Difference on one document state: with or without the (empty) log, the page comes out
    the same, and the log is created holding the one event."""
    old_out, old_v = _old_run(_success(), SimpleNamespace())
    _with_analysis(monkeypatch, tmp_path)
    results = {}
    for audit in (True, False):
        doc = _finished_doc(tmp_path / f"audit{audit}", old_out, old_v, audit=audit)
        [change] = tool.refinalize_document(doc, tmp_path)
        results[audit] = (
            change.after,
            change.result,
            json.loads((doc / "metadata.json").read_text())["status"],
        )
        log = json.loads((doc / "audit_log.json").read_text())
        assert log["counts"] == {"numbers_disagree_with_text_layer": 1}, log
        if not audit:
            assert log["pdf_filename"] == "doc.pdf" and log["event_count"] == 1
    assert results[True] == results[False] == ("warning", "FAIL", "partial")


def test_the_recorded_result_is_this_clauses_on_a_page_with_another_clause(
    tmp_path: Path, monkeypatch
) -> None:
    """Argon, round-2 review: once the clause is placed before a status clause, the LAST
    clause is not this one; the change, the tool's log and its summary must read this one."""
    warned = PageOutput(
        page_num=1,
        text=_READING,
        status=PageStatus.WARNING,
        engine="qwen",
        failure_mode=FailureMode.FIGURE_WORDS_UNREAD,
    )
    p = SimpleNamespace(page_num=1, own_layer_text=_LAYER, is_born_digital=True)
    old_out, old_v = _old_run(warned, p)
    _with_analysis(monkeypatch, tmp_path)
    doc = _finished_doc(tmp_path, old_out, old_v, audit=True)
    [change] = tool.refinalize_document(doc, tmp_path)
    after = json.loads((doc / "pages" / "00001.json").read_text())["verdict"]["clauses"]
    assert [c["check"] for c in after][-1] != "text_layer_numbers", "fixture premise"
    assert change.result == "FAIL"
    log = json.loads((doc / tool.LOG_NAME).read_text())
    assert log["pages"]["1"]["after"]["text_layer_numbers"] == "FAIL"


_TABLE = "| Country | a | b |\n| --- | --- | --- |\n| Korea | 4.0 | -8 |\n| China | 6.9 | 7.0 |\n"
_TABLE_LAYER = "Country a b Korea 4.0 -.8 China 6.9 7.0"


def _pdf(path: Path, pages: int = 1, text: str = _TABLE_LAYER) -> Path:
    import fitz

    doc = fitz.open()
    for _ in range(pages):
        doc.new_page().insert_text((72, 72), text, fontsize=9)
    doc.save(path)
    doc.close()
    return path


def _doc_for_pdf(tmp_path: Path, pdf: Path, *, pages: int | None = None) -> Path:
    """A finished one-page document whose metadata names *pdf* (its checksum, its pages)."""
    from ocr_output_contract import safe_checksum

    old_out, old_v = _old_run(_success(_TABLE), SimpleNamespace())
    doc = _finished_doc(tmp_path, old_out, old_v, audit=True)
    meta = json.loads((doc / "metadata.json").read_text())
    meta.update(key=pdf.name, checksum=safe_checksum(pdf), pages=pages or 1)
    (doc / "metadata.json").write_text(json.dumps(meta))
    return doc


def test_a_pdf_that_is_not_the_runs_input_changes_nothing(tmp_path: Path) -> None:
    """socr bot, round-2 review: the checksum and page-count guards were unpinned. A wrong
    file under --pdf-dir would demote faithful pages or let a wrong one stand. Difference on
    one document: its own PDF re-finalizes the page; another file of the same name, or the
    run's file with another page count on record, leaves every byte and counts the reason."""
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    own = _pdf(tmp_path / "own.pdf")
    other = _pdf(tmp_path / "other.pdf", text="Country a b Korea 4.0 -8 China 6.9 7.0 x")
    cases = {
        "own": (_doc_for_pdf(tmp_path / "a", own), own),
        "checksum": (_doc_for_pdf(tmp_path / "b", own), other),
        "page count": (_doc_for_pdf(tmp_path / "c", own, pages=2), own),
    }
    for name, (doc, served) in cases.items():
        target = pdfs / name.replace(" ", "_")
        target.mkdir()
        meta = json.loads((doc / "metadata.json").read_text())
        (target / meta["key"]).write_bytes(served.read_bytes())
        before = {p: p.read_bytes() for p in doc.rglob("*") if p.is_file()}
        skipped: Counter = Counter()
        changes = tool.refinalize_document(doc, target, skipped=skipped)
        if name == "own":
            assert [c.page_num for c in changes] == [1], changes  # the run's file is used
            continue
        assert changes == [] and any(name in reason for reason in skipped), (name, skipped)
        assert {p: p.read_bytes() for p in doc.rglob("*") if p.is_file()} == before, name


def test_a_rerun_after_an_interrupted_run_marks_the_page_once(tmp_path: Path, monkeypatch) -> None:
    """socr bot, round-2 review (Argon's interrupt case): interrupted after the document .md
    was written, a rerun must find the page's section already done, not add the mark again."""
    old_out, old_v = _old_run(_success(), SimpleNamespace())
    _with_analysis(monkeypatch, tmp_path)
    doc = _finished_doc(tmp_path, old_out, old_v, audit=True)
    (doc / "manifest.json").write_text(json.dumps({"entries": {"1": {}}}))
    real = tool._write_json

    def crash_on_manifest(path: Path, data) -> None:
        if path.name == "manifest.json":
            raise OSError("interrupted")
        real(path, data)

    with monkeypatch.context() as m:
        m.setattr(tool, "_write_json", crash_on_manifest)
        with pytest.raises(OSError):
            tool.refinalize_document(doc, tmp_path)
    assert "text_layer_numbers=FAIL" in (doc / "doc.md").read_text(), "premise: .md written"
    tool.refinalize_document(doc, tmp_path)
    assert (doc / "doc.md").read_text().count("text_layer_numbers=FAIL") == 1
    assert (doc / "pages" / "00001.md").read_text().count("text_layer_numbers=FAIL") == 1


def test_the_table_counts_follow_the_demoted_page(tmp_path: Path, monkeypatch) -> None:
    """socr bot, round-2 review: ``_recount_tables`` was unpinned. A table on a page the clause
    demotes is no longer ``verified_text``; the document's block must say so."""
    from socr.core.born_digital import DocumentAssessment
    from socr.core.table_counts import count_from_sidecars

    old_out, old_v = _old_run(_success(_TABLE), SimpleNamespace())
    doc = _finished_doc(tmp_path, old_out, old_v, audit=True)
    before = count_from_sidecars(doc).to_dict()
    meta = json.loads((doc / "metadata.json").read_text())
    (doc / "metadata.json").write_text(json.dumps({**meta, "tables": before}))
    assert before["verified_text"] == 1, "premise: the old run counted the table verified"
    page = _analysis(prior_ocr=False, layer=_TABLE_LAYER)
    assessment = DocumentAssessment(path=tmp_path / "doc.pdf", pages=[page])
    monkeypatch.setattr(tool, "_analyse_pdf", lambda pdf_dir, meta: (assessment, None))
    [change] = tool.refinalize_document(doc, tmp_path)
    assert change.result == "FAIL"
    after = json.loads((doc / "metadata.json").read_text())["tables"]
    assert after["verified_text"] == 0 and after == count_from_sidecars(doc).to_dict()
