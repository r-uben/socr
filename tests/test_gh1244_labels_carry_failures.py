"""#1244 / #1204 (fix plan C1): document and page labels carry failures up.

Two published labels read as clean when they were not:

* a document was ``completed`` while holding WARNING / ERROR pages (11 of 26 on the 156-doc
  Fed run), because the document gate was an allowlist of known defect buckets;
* a page's published ``audit_passed`` was the internal winner-selection flag (#252), so
  1,087 pages carrying a FAIL clause read ``true``.

The fix derives both: a document is clean only if EVERY page has a final record with status
SUCCESS (a missing record is incomplete), and the sidecar's top-level ``audit_passed`` is
derived at record time. The internal acceptance stays ``winning_output.audit_passed`` and is
what resume and selection read.

Hermetic per ``CLAUDE.md``: the engine registry, the ladder, the judge and the judge model are
pinned, ``primary_engine`` / ``local_engine`` / ``enabled_engines`` are pinned, and the
end-to-end claims are DIFFERENCES between a run with the fix and the same run with the fix
switched off, in both provider states, never an absolute outcome measured on this machine.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import socr
from socr.core.manifest import FinalizedPageRecord
from socr.core.marks import Check, CheckResult
from socr.core.result import DocumentStatus, PageOutput, PageStatus
from socr.core.verdict import PageVerdict, VerdictClause, published_audit_passed
from socr.pipeline import orchestrator as orch
from socr.pipeline.orchestrator import UnifiedPipeline, _pages_not_success

from test_resume_figure_embed_idempotent import (
    _CountingEngine,
    _crash_after_page_one,
    _FixedJudge,
    _prior_ocr_pdf,
)


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# The published label, as a function
# ---------------------------------------------------------------------------


def _verdict(*results: CheckResult) -> PageVerdict:
    checks = (Check.NATIVE_COVERAGE, Check.TEXT_LAYER_WORDS, Check.PRIOR_OCR_LAYER)
    return PageVerdict(
        page_num=1,
        clauses=tuple(VerdictClause(checks[i], r) for i, r in enumerate(results)),
    )


@pytest.mark.parametrize(
    ("status", "accepted", "results", "expected"),
    [
        # WARNING with UNKNOWN-only clauses: false by status, not by clause.
        ("warning", True, (CheckResult.UNKNOWN,), False),
        # SUCCESS with a FAIL clause: the 1,087-page lie.
        ("success", True, (CheckResult.FAIL,), False),
        ("success", True, (CheckResult.UNKNOWN, CheckResult.FAIL), False),
        # SUCCESS carrying only UNKNOWN does not block (owner ruling, DECISIONS C1).
        ("success", True, (CheckResult.UNKNOWN,), True),
        ("success", True, (), True),
        # Internal-false SUCCESS: a rejected reading never publishes true.
        ("success", False, (), False),
        ("success", None, (), False),
        ("error", True, (), False),
        ("warning", True, (), False),
    ],
)
def test_published_audit_passed_is_derived(status, accepted, results, expected) -> None:
    assert published_audit_passed(status, accepted, _verdict(*results)) is expected
    # The dict form (what a refinalizer reads back from a sidecar) agrees.
    assert published_audit_passed(status, accepted, _verdict(*results).to_dict()) is expected


def test_published_audit_passed_without_a_verdict_follows_status_and_acceptance() -> None:
    assert published_audit_passed("success", True, None) is True
    assert published_audit_passed(PageStatus.SUCCESS, True, None) is True
    assert published_audit_passed("warning", True, None) is False


# ---------------------------------------------------------------------------
# The document invariant, as a function
# ---------------------------------------------------------------------------


def _record(page_num: int, status: PageStatus) -> FinalizedPageRecord:
    return FinalizedPageRecord(
        output=PageOutput(page_num=page_num, text="x", status=status),
        disposition=None,  # type: ignore[arg-type]
        selection_provenance=None,  # type: ignore[arg-type]
    )


def test_pages_not_success_counts_missing_records_as_incomplete() -> None:
    ok = PageStatus.SUCCESS
    assert _pages_not_success([_record(1, ok), _record(2, ok)], 2) == []
    # The vacuous shapes: all() over nothing, or over a subset, is True. Neither may complete.
    assert _pages_not_success([], 2) == [1, 2]
    assert _pages_not_success([_record(1, ok)], 3) == [2, 3]
    assert _pages_not_success([_record(2, ok)], 2) == [1]


@pytest.mark.parametrize(
    "status", [PageStatus.WARNING, PageStatus.ERROR, PageStatus.SKIPPED, PageStatus.PENDING]
)
def test_pages_not_success_every_other_status_is_incomplete(status) -> None:
    assert _pages_not_success([_record(1, PageStatus.SUCCESS), _record(2, status)], 2) == [2]


# ---------------------------------------------------------------------------
# End to end: the fix on and off, in both provider states
# ---------------------------------------------------------------------------


def _fix_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-C1 behaviour: publish the internal flag, gate documents on buckets only."""
    monkeypatch.setattr(orch, "published_audit_passed", lambda status, accepted, verdict: accepted)
    monkeypatch.setattr(orch, "_pages_not_success", lambda records, page_count: [])


def _demote(monkeypatch: pytest.MonkeyPatch, *, page: int, when: str, how: str = "status") -> None:
    """Demote ``page`` in the ``pre`` records, the ``final`` ones, or both.

    ``how`` is ``status`` (WARNING), ``fail_clause`` (status stays SUCCESS, the verdict gains a
    FAIL clause), ``internal`` (status stays SUCCESS, the internal acceptance is False) or
    ``recovered`` (WARNING plus the typed recovered-prose flag).

    The text is untouched: only the label inputs the record carries change, which is exactly
    what a late guard (figure phase, final-body guard, chart-region pass) does.
    """
    import socr.core.manifest as manifest

    real = manifest.finalized_page_records

    def patched(state, saved_body=None):
        records = real(state, saved_body)
        is_final = saved_body is not None
        if when == "both" or (when == "final" and is_final) or (when == "pre" and not is_final):
            out = []
            for rec in records:
                if rec.output.page_num == page:
                    demoted = copy.copy(rec.output)
                    verdict = rec.verdict
                    if how in ("status", "recovered"):
                        demoted.status = PageStatus.WARNING
                    if how == "recovered":
                        demoted.scanned_prose_recovered = True
                    if how == "internal":
                        demoted.audit_passed = False
                    if how == "fail_clause":
                        verdict = PageVerdict(
                            page_num=page,
                            clauses=(VerdictClause(Check.NATIVE_COVERAGE, CheckResult.FAIL),),
                        )
                    rec = FinalizedPageRecord(
                        output=demoted,
                        disposition=rec.disposition,
                        selection_provenance=rec.selection_provenance,
                        verdict=verdict,
                    )
                out.append(rec)
            return out
        return records

    monkeypatch.setattr(manifest, "finalized_page_records", patched)


def _split_one_page_too_many_in_assemble(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ONLY ``_phase_assemble``'s final-body split yield page_count + 1 bodies."""
    import sys

    import ocr_output_contract as contract

    real = contract.split_native_pages

    def patched(text):
        bodies = real(text)
        if sys._getframe(1).f_code.co_name == "_phase_assemble":
            return [*bodies, "an extra body"]
        return bodies

    monkeypatch.setattr(contract, "split_native_pages", patched)


def _process(
    out: Path,
    pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: bool,
    accept: bool = True,
    pages_read: list[int] | None = None,
    fix: bool = True,
    demote: tuple[int, str] | tuple[int, str, str] | None = None,
    split_extra_page: bool = False,
):
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL

    with monkeypatch.context() as m:
        if not fix:
            _fix_off(m)
        if demote is not None:
            _demote(
                m, page=demote[0], when=demote[1], how=demote[2] if len(demote) > 2 else "status"
            )
        if split_extra_page:
            _split_one_page_too_many_in_assemble(m)
        m.setattr(
            orch,
            "get_engine",
            lambda engine_type: _CountingEngine(pages_read if pages_read is not None else []),
        )
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
                describe_figure_crops=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if provider else []
        pipe._build_page_judge = lambda state: _FixedJudge(accept)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        return pipe.process(pdf, output_dir=out)


def _prose_pdf(path: Path, *, pages: int = 2) -> Path:
    """Born-digital prose only: no figure, no table, nothing for a defect bucket to name."""
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=612, height=792)
        for row in range(6):
            page.insert_text(
                (72, 120 + row * 24),
                f"Page {i + 1} ordinary visible prose line {row} of a born-digital page.",
                fontsize=10,
            )
    doc.save(path)
    doc.close()
    return path


def _sidecar(out: Path, page: int) -> dict:
    return json.loads(next(iter(out.rglob(f"pages/{page:05d}.json"))).read_text())


def _doc_md(out: Path) -> str:
    return next(p for p in sorted(out.rglob("*.md")) if "pages" not in p.parts).read_text()


def _metadata(out: Path) -> dict:
    files = json.loads(next(iter(out.rglob("metadata.json"))).read_text())["files"]
    return next(iter(files.values()))


def _selection(sidecar: dict) -> dict:
    """What winner selection produced: everything the labels must NOT move."""
    wo = sidecar["winning_output"]
    return {
        "status": sidecar["status"],
        "failure_mode": sidecar["failure_mode"],
        "engine": sidecar["engine"],
        "text": wo["text"],
        "internal_audit_passed": wo["audit_passed"],
    }


@pytest.mark.parametrize("provider", [True, False])
@pytest.mark.parametrize("when", ["pre", "final", "both"])
def test_a_demoted_page_stops_the_document_reading_completed(
    tmp_path, monkeypatch, provider, when
) -> None:
    """Late demotion reaches the document status, the metadata, the error text and the CLI result.

    Arms: baseline (no demotion), demotion with the fix OFF, demotion with the fix ON. What is
    pinned is the difference: the fix-off arm cannot see the demotion at the document level (the
    hole), the fix-on arm can, and the text is byte-identical across all three.
    """
    pdf = _prose_pdf(tmp_path / "doc.pdf")
    base = _process(tmp_path / "base", pdf, monkeypatch, provider=provider)
    off = _process(
        tmp_path / "off", pdf, monkeypatch, provider=provider, fix=False, demote=(2, when)
    )
    on = _process(tmp_path / "on", pdf, monkeypatch, provider=provider, demote=(2, when))

    assert on.status is not DocumentStatus.SUCCESS
    assert _metadata(tmp_path / "on")["status"] != DocumentStatus.SUCCESS.value
    assert "page(s) 2 did not finish with status success" in (on.error or "")
    assert on.audit_passed is False
    # Without the fix the demotion is invisible at the document level: the fix-off arm equals
    # the undemoted baseline, whatever that is, and the fix-on arm differs from it.
    assert off.status is base.status
    assert on.status is not base.status
    # Labels only: the shipped text and winner selection of every page are identical.
    assert _doc_md(tmp_path / "on") == _doc_md(tmp_path / "off") == _doc_md(tmp_path / "base")
    for page in (1, 2):
        assert (
            _selection(_sidecar(tmp_path / "on", page))["text"]
            == (_selection(_sidecar(tmp_path / "off", page))["text"])
        )
        assert (
            _sidecar(tmp_path / "on", page)["winning_output"]["audit_passed"]
            == _sidecar(tmp_path / "off", page)["winning_output"]["audit_passed"]
        )


@pytest.mark.parametrize("provider", [True, False])
def test_the_fix_changes_no_selection_on_a_document_with_no_demotion(
    tmp_path, monkeypatch, provider
) -> None:
    """Scope guard, as a difference: switching the fix on moves labels and nothing else."""
    pdf = _prose_pdf(tmp_path / "doc.pdf")
    off = _process(tmp_path / "off", pdf, monkeypatch, provider=provider, fix=False)
    on = _process(tmp_path / "on", pdf, monkeypatch, provider=provider)
    assert _doc_md(tmp_path / "on") == _doc_md(tmp_path / "off")
    for page in (1, 2):
        assert _selection(_sidecar(tmp_path / "on", page)) == _selection(
            _sidecar(tmp_path / "off", page)
        )
    assert on.status is off.status


@pytest.mark.parametrize("provider", [True, False])
@pytest.mark.parametrize("accept", [True, False])
def test_published_label_differs_from_the_internal_acceptance(
    tmp_path, monkeypatch, provider, accept
) -> None:
    """A page that is not SUCCESS never publishes true; its internal acceptance is untouched."""
    pdf = _prior_ocr_pdf(tmp_path / "doc.pdf")
    off = tmp_path / "off"
    on = tmp_path / "on"
    _process(off, pdf, monkeypatch, provider=provider, accept=accept, fix=False)
    _process(on, pdf, monkeypatch, provider=provider, accept=accept)
    for page in (1, 2):
        a, b = _sidecar(off, page), _sidecar(on, page)
        assert _selection(a) == _selection(b), "the fix moved winner selection"
        assert b["winning_output"]["audit_passed"] == a["winning_output"]["audit_passed"]
        clauses = [c["result"] for c in (b.get("verdict") or {}).get("clauses", [])]
        expected = b["status"] == "success" and "FAIL" not in clauses
        assert b["audit_passed"] is (expected and b["winning_output"]["audit_passed"] is True)
        if b["status"] != "success":
            assert b["audit_passed"] is False
    assert _doc_md(on) == _doc_md(off)


@pytest.mark.parametrize("accept", [True, False])
def test_resume_eligibility_is_unchanged_by_the_label(tmp_path, monkeypatch, accept) -> None:
    """Resume reads the internal acceptance, so relabelling a page changes no skip decision.

    The accepted arm is the explicit exception (an accepted prior-OCR page is skip-and-kept,
    #1192); the rejected arm reprocesses. Both are compared across fix on/off rather than
    pinned, and the restored sidecar is identical to the one the first run wrote.
    """
    reads: dict[bool, list[int]] = {}
    sidecars: dict[bool, tuple[dict, dict]] = {}
    for fix in (True, False):
        pdf = _prior_ocr_pdf(tmp_path / f"{fix}.pdf")
        out = tmp_path / f"out-{fix}"
        _process(out, pdf, monkeypatch, provider=True, accept=accept, fix=fix)
        before = _sidecar(out, 1)
        _crash_after_page_one(out)
        resumed: list[int] = []
        _process(out, pdf, monkeypatch, provider=True, accept=accept, fix=fix, pages_read=resumed)
        reads[fix] = resumed
        sidecars[fix] = (before, _sidecar(out, 1))
    assert reads[True] == reads[False]
    if accept:
        assert 1 not in reads[True]
        before, after = sidecars[True]
        assert after == before
        assert after["winning_output"]["audit_passed"] is True
        assert after["audit_passed"] is False
    else:
        assert 1 in reads[True]


# ---------------------------------------------------------------------------
# The invariant uses the PUBLISHED label; a body that does not split fails closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", [True, False])
@pytest.mark.parametrize("how", ["fail_clause", "internal"])
@pytest.mark.parametrize("when", ["pre", "final"])
def test_a_success_page_that_publishes_false_blocks_completed(
    tmp_path, monkeypatch, provider, how, when
) -> None:
    """Status stays SUCCESS on the page, but its published audit_passed is false (a FAIL clause,
    or an internal acceptance that is not True): the document is not completed, and the named
    reason says page 2. Difference against the same run with the fix off and with no demotion."""
    pdf = _prose_pdf(tmp_path / "doc.pdf")
    base = _process(tmp_path / "base", pdf, monkeypatch, provider=provider)
    off = _process(
        tmp_path / "off", pdf, monkeypatch, provider=provider, fix=False, demote=(2, when, how)
    )
    on = _process(tmp_path / "on", pdf, monkeypatch, provider=provider, demote=(2, when, how))

    assert _sidecar(tmp_path / "on", 2)["status"] == "success"
    if when == "final":
        # The sidecar is written from the final records, so only there does it publish false.
        assert _sidecar(tmp_path / "on", 2)["audit_passed"] is False
    assert off.status is base.status
    assert on.status is not base.status
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert "page(s) 2 did not finish with status success" in (on.error or "")
    assert "page(s) 1" not in (on.error or "")
    assert _metadata(tmp_path / "on")["status"] != DocumentStatus.SUCCESS.value
    assert _doc_md(tmp_path / "on") == _doc_md(tmp_path / "off")


@pytest.mark.parametrize("provider", [True, False])
def test_a_body_that_does_not_split_fails_closed(tmp_path, monkeypatch, provider) -> None:
    """The final split yields page_count + 1 bodies: no final record can be read, so every page
    is without a final record and the document is not completed. The fix-off arm keeps the
    baseline status."""
    pdf = _prose_pdf(tmp_path / "doc.pdf")
    base = _process(tmp_path / "base", pdf, monkeypatch, provider=provider)
    off = _process(
        tmp_path / "off", pdf, monkeypatch, provider=provider, fix=False, split_extra_page=True
    )
    on = _process(tmp_path / "on", pdf, monkeypatch, provider=provider, split_extra_page=True)

    assert off.status is base.status
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert on.status is not base.status
    assert "page(s) 1, 2 did not finish with status success" in (on.error or "")
    assert on.audit_passed is False
    assert _metadata(tmp_path / "on")["status"] != DocumentStatus.SUCCESS.value


@pytest.mark.parametrize("provider", [True, False])
def test_a_recovered_prose_document_names_its_own_bucket_not_the_generic_reason(
    tmp_path, monkeypatch, provider
) -> None:
    """A document with a recovered-prose page is demoted by that bucket, which names itself in
    ``error``; the generic invariant note is only for documents no bucket names."""
    pdf = _prose_pdf(tmp_path / "doc.pdf")
    on = _process(
        tmp_path / "on", pdf, monkeypatch, provider=provider, demote=(2, "pre", "recovered")
    )
    assert on.status is DocumentStatus.AUDIT_FAILED
    assert "page(s) 2:" in (on.error or "")
    assert "see scanned_prose_recovered in the page sidecar" in (on.error or "")
    assert "did not finish with status success" not in (on.error or "")


# ---------------------------------------------------------------------------
# The control helper itself: it must fail on an unrelated demotion
# ---------------------------------------------------------------------------


def _result(status: DocumentStatus, error: str | None):
    from types import SimpleNamespace

    return SimpleNamespace(status=status, error=error)


_NOTE = "page(s) 2 did not finish with status success and a passing audit"


def test_the_control_helper_accepts_only_the_designed_unchecked_demotion() -> None:
    from c1_support import assert_clean_or_only_unchecked as check

    check(_result(DocumentStatus.SUCCESS, None), pages=(1, 2))
    check(_result(DocumentStatus.AUDIT_FAILED, _NOTE), pages=(1, 2))
    check(
        _result(DocumentStatus.AUDIT_FAILED, f"{_NOTE}; page(s) 2: an own note"),
        pages=(2,),
        other_notes=("an own note",),
    )
    # A page outside the fixture's design, a note it does not carry, a failed document, and an
    # AUDIT_FAILED with no C1 note are each an unrelated demotion.
    with pytest.raises(AssertionError):
        check(_result(DocumentStatus.AUDIT_FAILED, _NOTE), pages=(1,))
    with pytest.raises(AssertionError):
        check(
            _result(DocumentStatus.AUDIT_FAILED, f"{_NOTE}; page(s) 2: withheld table"),
            pages=(2,),
        )
    with pytest.raises(AssertionError):
        check(_result(DocumentStatus.ERROR, _NOTE), pages=(2,))
    with pytest.raises(AssertionError):
        check(_result(DocumentStatus.AUDIT_FAILED, "page(s) 2: withheld table"), pages=(2,))


@pytest.mark.parametrize("provider", [True, False])
def test_a_body_that_does_not_split_leaves_no_page_skippable_on_resume(
    tmp_path, monkeypatch, provider
) -> None:
    """The fail-closed document must not leave terminal SUCCESS sidecars behind: a rerun would
    skip pages whose final body this run could not confirm. Difference on the first run only:
    the same document with a body that splits resumes its pages; the one that does not splits
    reprocesses every page (spy on ``_load_terminal_page``)."""
    pdf = _prior_ocr_pdf(tmp_path / "doc.pdf")
    resumed: dict[bool, list[int]] = {}
    for split_fails in (False, True):
        out = tmp_path / f"out-{split_fails}"
        _process(out, pdf, monkeypatch, provider=provider, split_extra_page=split_fails)
        for sidecar_path in out.rglob("pages/*.json"):
            assert json.loads(sidecar_path.read_text())["terminal"] is not split_fails
        # The document-level gate would skip a finished document whole; remove the saved .md
        # so the per-page ledger is the gate under test, as ``_crash_after_page_one`` does.
        for md in out.rglob("*.md"):
            if "pages" not in md.parts:
                md.unlink()
        seen: list[int] = []
        real = UnifiedPipeline._load_terminal_page

        def spy(self, state, page_num, output_dir, _seen=seen, _real=real):
            loaded = _real(self, state, page_num, output_dir)
            if loaded is not None:
                _seen.append(page_num)
            return loaded

        with monkeypatch.context() as m:
            m.setattr(UnifiedPipeline, "_load_terminal_page", spy)
            _process(out, pdf, monkeypatch, provider=provider)
        resumed[split_fails] = sorted(seen)
    if provider:
        # With no provider the ladder is empty and no page is skippable in either arm, so the
        # premise holds only here; the claim below holds in both states.
        assert resumed[False], "fixture premise: a body that splits resumes its pages"
    assert resumed[True] == []
