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


def _demote(monkeypatch: pytest.MonkeyPatch, *, page: int, when: str) -> None:
    """Demote ``page`` to WARNING in the ``pre`` records, the ``final`` ones, or both.

    The text is untouched: only the status the record carries changes, which is exactly what a
    late guard (figure phase, final-body guard, chart-region pass) does.
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
                    demoted.status = PageStatus.WARNING
                    rec = FinalizedPageRecord(
                        output=demoted,
                        disposition=rec.disposition,
                        selection_provenance=rec.selection_provenance,
                        verdict=rec.verdict,
                    )
                out.append(rec)
            return out
        return records

    monkeypatch.setattr(manifest, "finalized_page_records", patched)


def _process(
    out: Path,
    pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: bool,
    accept: bool = True,
    pages_read: list[int] | None = None,
    fix: bool = True,
    demote: tuple[int, str] | None = None,
):
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL

    with monkeypatch.context() as m:
        if not fix:
            _fix_off(m)
        if demote is not None:
            _demote(m, page=demote[0], when=demote[1])
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
    if base.status is DocumentStatus.SUCCESS:
        # Without the fix the demotion is invisible at the document level.
        assert off.status is DocumentStatus.SUCCESS
        assert on.status is not off.status
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
    if off.status is DocumentStatus.SUCCESS:
        assert on.status is DocumentStatus.SUCCESS


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
