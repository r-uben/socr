"""#1251 (C3), the offline half: the refinalize tools apply the page_judge clause exactly as a
live run does, so "re-finalize an old run" still equals "a run of this code".

An old run here is a run of the code before #1251 (its clause function is patched out). Hermetic
by the gh1185 harness: engine registry, ladder, page judge and judge model are pinned, and every
claim is a DIFFERENCE between arms in one process.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from test_gh1185_text_layer_numbers_e2e import (
    _ROWS,
    _comparable,
    _doc_dir,
    _page,
    _rotated_pdf,
    _run,
)

from socr.core import verdict as verdict_module
from socr.core.marks import Check, CheckResult
from socr.core.pass_trace import TraceEntry, TraceResult
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.verdict import (
    JUDGE_ONLY_REASON,
    PageVerdict,
    VerdictClause,
    page_text_digest,
    with_judge_only_clause,
)
from socr.pipeline.refinalize_own_text import refinalize_document


def _rows_only_pdf(path: Path) -> Path:
    """The rotated rows alone: the layer owes the reading no word, so nothing independent
    confirms a judge-approved reading of it."""
    import fitz

    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page(width=612, height=792)
        for i, row in enumerate(_ROWS * 6):
            page.insert_text((100 + i * 20, 700), row, fontsize=9, rotate=90)
    doc.save(path)
    doc.close()
    return path


def _levels(out: Path) -> tuple[dict, dict]:
    doc = json.loads((_doc_dir(out) / "metadata.json").read_text())
    [entry] = json.loads((out / "metadata.json").read_text())["files"].values()
    return doc, entry


def test_refinalize_turns_an_old_judge_only_page_into_what_this_code_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _rows_only_pdf(tmp_path / "doc.pdf")
    fresh, old = tmp_path / "fresh", tmp_path / "old"
    _run(fresh, pdf, monkeypatch, korea_q4="-.8", pages_read=[])
    with monkeypatch.context() as m:
        m.setattr(verdict_module, "_judge_only_clauses", lambda output, p, event_kinds: ())
        _run(old, pdf, monkeypatch, korea_q4="-.8", pages_read=[])

    # Fixture premises: the code under test marks the page, the old code did not.
    assert _page(fresh)[0]["failure_mode"] == FailureMode.READING_UNCORROBORATED.value
    assert _page(old)[0]["status"] == "success" and _page(old)[0]["verdict"]["clauses"] == []

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    shutil.copy(pdf, pdf_dir / pdf.name)
    changes = refinalize_document(_doc_dir(old), pdf_dir)
    assert {(c.page_num, c.before, c.after) for c in changes} == {
        (1, "success", "warning"),
        (2, "success", "warning"),
    }

    for n in (1, 2):
        (re_side, re_body), (fr_side, fr_body) = _page(old, n), _page(fresh, n)
        assert re_body == fr_body
        assert _comparable(re_side) == _comparable(fr_side)
        [clause] = re_side["verdict"]["clauses"]
        assert (clause["check"], clause["result"], clause["reason"]) == (
            "page_judge",
            "UNKNOWN",
            JUDGE_ONLY_REASON,
        )
    doc_md = lambda out: next(p for p in _doc_dir(out).glob("*.md")).read_text()  # noqa: E731
    assert doc_md(old) == doc_md(fresh)
    # The demotion reaches the document and the root index, as in the live run.
    keys = ("status", "error")
    for got, want in zip(_levels(old), _levels(fresh)):
        assert {k: got.get(k) for k in keys} == {k: want.get(k) for k in keys}
    assert _levels(old)[0]["status"] != "completed"

    # Idempotent.
    snapshot = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    assert refinalize_document(_doc_dir(old), pdf_dir) == []
    assert {p: p.read_bytes() for p in old.rglob("*") if p.is_file()} == snapshot


def test_a_page_with_an_independent_check_is_left_alone_by_the_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other arm of the difference: the same old run, but the layer has words the reading
    carries (the gh1185 fixture), so the words check passes and the tool changes nothing."""
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    old = tmp_path / "old"
    with monkeypatch.context() as m:
        m.setattr(verdict_module, "_judge_only_clauses", lambda output, p, event_kinds: ())
        _run(old, pdf, monkeypatch, korea_q4="-.8", pages_read=[])
    assert _page(old)[0]["status"] == "success"
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    shutil.copy(pdf, pdf_dir / pdf.name)
    snapshot = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    assert refinalize_document(_doc_dir(old), pdf_dir) == []
    assert {p: p.read_bytes() for p in old.rglob("*") if p.is_file()} == snapshot


# --- the helper, on its own ---------------------------------------------------------------


def _output(**kw) -> PageOutput:
    from socr.core.result import JUDGE_OUTCOME_COMPLETED

    base = dict(
        page_num=1,
        text="reading 4.0",
        status=PageStatus.SUCCESS,
        engine="qwen",
        judge_outcome=JUDGE_OUTCOME_COMPLETED,
        judge_accepted=True,
    )
    base.update(kw)
    return PageOutput(**base)


_JUDGE = VerdictClause(Check.PAGE_JUDGE, CheckResult.UNKNOWN, "page", JUDGE_ONLY_REASON)
_OTHER = VerdictClause(Check.TEXT_LAYER_WORDS, CheckResult.FAIL, "page", "1 word(s) missing")


def _pass(check: Check, text: str) -> dict:
    return TraceEntry(
        check.value, TraceResult.PASS, "page", "x", text_sha=page_text_digest(text)
    ).to_dict()


def test_helper_adds_the_clause_when_no_independent_check_ran() -> None:
    out = with_judge_only_clause(PageVerdict(page_num=1), _output(), [], frozenset())
    assert out.clauses == (_JUDGE,)


def test_helper_adds_nothing_when_words_and_numbers_passed_on_the_shipped_text() -> None:
    out = _output()
    trace = [_pass(Check.TEXT_LAYER_WORDS, out.text), _pass(Check.TEXT_LAYER_NUMBERS, out.text)]
    assert with_judge_only_clause(PageVerdict(page_num=1), out, trace) == PageVerdict(page_num=1)


def test_helper_ignores_a_pass_computed_on_other_text() -> None:
    out = _output()
    trace = [_pass(Check.TEXT_LAYER_WORDS, "other"), _pass(Check.TEXT_LAYER_NUMBERS, out.text)]
    assert with_judge_only_clause(PageVerdict(page_num=1), out, trace).clauses == (_JUDGE,)


def test_helper_uses_the_tools_own_recomputed_checks_when_the_sidecar_has_no_trace() -> None:
    out = _output()
    recomputed = (
        TraceEntry(Check.TEXT_LAYER_WORDS.value, TraceResult.PASS, "page", "x"),
        TraceEntry(Check.TEXT_LAYER_NUMBERS.value, TraceResult.PASS, "page", "x"),
    )
    got = with_judge_only_clause(PageVerdict(page_num=1), out, None, recomputed=recomputed)
    assert got == PageVerdict(page_num=1)


def test_helper_drops_the_clause_when_the_page_gains_another() -> None:
    got = with_judge_only_clause(
        PageVerdict(page_num=1, clauses=(_OTHER, _JUDGE)),
        _output(status=PageStatus.WARNING, failure_mode=FailureMode.READING_UNCORROBORATED),
        [],
    )
    assert got.clauses == (_OTHER,)


@pytest.mark.parametrize(
    "kw",
    [
        {"judge_accepted": False},
        {"judge_accepted": None},
        {"engine": "native"},
        {"judge_outcome": "page_judge_verifier_error"},
        {"status": PageStatus.WARNING},
    ],
)
def test_helper_is_scoped_to_an_accepted_model_reading_still_success(kw) -> None:
    assert with_judge_only_clause(PageVerdict(page_num=1), _output(**kw), []) == PageVerdict(
        page_num=1
    )
