"""#1129: a page the judge rejected must not ship plain SUCCESS unmarked.

Hermetic: patch ``_available_engines_for_agentic`` and ``_resolve_judge_model`` on
every ``process()`` test. Pins DIFFERENCES, not absolute provider-dependent tuples.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import (
    PAGE_UNCHECKED_MARK_RE,
    UNACCEPTED_OUTPUT_MARK_RE,
    SelectionProvenance,
    _finalize_best_output_unverified,
    _select_and_finalize_page,
    _select_page_output_tagged,
    _whole_doc_page_texts,
    unaccepted_output_mark,
)
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import (
    JUDGE_OUTCOME_COMPLETED,
    JUDGE_OUTCOME_EXCEPTION,
    JUDGE_OUTCOME_TIMEOUT,
    REJECTION_JUDGE_ONLY,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState
from socr.pipeline.agentic import AcceptDecision, NativeTableVerifierJudge
from socr.pipeline.orchestrator import UnifiedPipeline

_REJECT_REASON = "judge rejected: cuts off mid-sentence"
_BODY = (
    "This paragraph is long enough to survive the heuristic judge's word-count gate "
    "when we need a non-empty rejected reading on a scanned page without tables."
)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert (
        Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")
    ), socr.__file__


def _rejected_best_output(*, outcome: str = JUDGE_OUTCOME_COMPLETED) -> PageOutput:
    return PageOutput(
        page_num=1,
        text=_BODY,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=False,
        judge_reason=_REJECT_REASON,
        judge_outcome=outcome,
    )


def test_demotion_differs_from_audit_passed_true_on_same_bytes() -> None:
    kept = _rejected_best_output()
    clean = PageOutput(
        page_num=1,
        text=_BODY,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        judge_reason="judge accepted",
        judge_outcome=JUDGE_OUTCOME_COMPLETED,
    )
    demoted = _finalize_best_output_unverified(kept)
    untouched = _finalize_best_output_unverified(clean)
    assert demoted.status is PageStatus.WARNING
    assert untouched.status is PageStatus.SUCCESS
    assert demoted.audit_passed is False
    assert untouched.audit_passed is True
    assert demoted.failure_mode is FailureMode.AUDIT_FAILED
    assert untouched.failure_mode is FailureMode.NONE


def test_judge_timeout_differs_from_completed_rejection() -> None:
    completed = _finalize_best_output_unverified(_rejected_best_output())
    timed_out = _finalize_best_output_unverified(
        _rejected_best_output(outcome=JUDGE_OUTCOME_TIMEOUT)
    )
    assert completed.failure_mode is FailureMode.AUDIT_FAILED
    assert timed_out.failure_mode is FailureMode.PAGE_JUDGE_TIMEOUT


def test_finalize_writes_mark_as_first_line(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "scan")
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.native_text = ""
    p.best_output = _rejected_best_output()
    record = _select_and_finalize_page(state, 1)
    first = (record.output.text or "").splitlines()[0]
    assert UNACCEPTED_OUTPUT_MARK_RE.match(first)
    assert _REJECT_REASON in first
    assert record.output.status is PageStatus.WARNING


def _scanned_assessment(pdf: Path) -> DocumentAssessment:
    return DocumentAssessment(
        path=pdf,
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=False,
                native_text="",
                confidence=0.2,
                needs_ocr_enhancement=True,
                has_tables=False,
                word_count=0,
            )
        ],
    )


def _run_scanned_process(tmp_path: Path, *, assess_side_effect) -> tuple:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    # Ink on the scan so #1100 does not short-circuit to the blank-page lane.
    page.insert_text((72, 72), "scanned page ink for routing")
    doc.save(pdf)
    doc.close()
    out_dir = tmp_path / "out"
    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        dual_pass_tables=False,
        escalate_ambiguous_tables=False,
        table_judge_ladder=False,
    )
    pipeline = UnifiedPipeline(config)
    pipeline.bd_detector = MagicMock()
    pipeline.bd_detector.detect.return_value = _scanned_assessment(pdf)

    def _stub_engine(self, state, page_nums, enhancement_pages, engine_type, label, **kwargs):
        return [
            PageOutput(
                page_num=page_nums[0],
                text=_BODY,
                status=PageStatus.SUCCESS,
                engine="qwen",
                confidence=0.9,
            )
        ]

    with (
        patch.object(pipeline, "_is_agentic_trusted_native", return_value=False),
        patch.object(pipeline, "_probe_backend_idle", return_value=True),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
        patch("socr.pipeline.orchestrator.probe_openai_server_idle", return_value=True),
        patch.object(
            UnifiedPipeline, "_run_engine_on_pages", autospec=True, side_effect=_stub_engine
        ),
        patch.object(
            NativeTableVerifierJudge,
            "assess",
            autospec=True,
            side_effect=assess_side_effect,
        ),
    ):
        return pipeline.process(pdf, out_dir), out_dir, pipeline, pdf


def test_process_judge_rejects_every_rung_ships_warning_and_mark(tmp_path: Path) -> None:
    calls: list[tuple] = []

    def _reject(self, output, provider):
        calls.append(("reject", provider))
        return AcceptDecision(accept=False, reason=_REJECT_REASON)

    result, out_dir, _pipeline, _pdf = _run_scanned_process(tmp_path, assess_side_effect=_reject)
    assert calls and calls[0][0] == "reject"
    frag = out_dir / "doc" / "pages" / "00001.md"
    assert frag.exists(), result.status
    body = frag.read_text(encoding="utf-8")
    assert UNACCEPTED_OUTPUT_MARK_RE.match(body.splitlines()[0])
    sidecar = json.loads((out_dir / "doc" / "pages" / "00001.json").read_text())
    assert sidecar["status"] == PageStatus.WARNING.value
    assert sidecar["winning_output"]["audit_passed"] is False
    assert sidecar["winning_output"]["status"] == PageStatus.WARNING.value


def test_process_judge_raises_every_rung_ships_warning_and_mark(tmp_path: Path) -> None:
    calls: list[str] = []

    def _raise(self, output, provider):
        calls.append("raise")
        raise RuntimeError("judge transport failed")

    result, out_dir, _pipeline, _pdf = _run_scanned_process(tmp_path, assess_side_effect=_raise)
    assert calls == ["raise"]
    frag = out_dir / "doc" / "pages" / "00001.md"
    assert frag.exists(), result.status
    body = frag.read_text(encoding="utf-8")
    assert UNACCEPTED_OUTPUT_MARK_RE.match(body.splitlines()[0])
    sidecar = json.loads((out_dir / "doc" / "pages" / "00001.json").read_text())
    assert sidecar["status"] == PageStatus.WARNING.value
    assert sidecar["winning_output"]["status"] != PageStatus.SUCCESS.value


def test_resume_never_treats_demoted_page_as_clean_success(tmp_path: Path) -> None:
    def _reject(self, output, provider):
        return AcceptDecision(accept=False, reason=_REJECT_REASON)

    def _accept(self, output, provider):
        return AcceptDecision(accept=True, reason="accepted")

    _result, out_dir, pipeline, pdf = _run_scanned_process(tmp_path, assess_side_effect=_reject)
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    pipeline._scan_root = pdf.parent
    assert pipeline._load_terminal_page(state, 1, out_dir) is None

    accept_root = tmp_path / "accept"
    accept_root.mkdir()
    _accept_result, accept_out, accept_pipeline, accept_pdf = _run_scanned_process(
        accept_root, assess_side_effect=_accept
    )
    accept_state = DocumentState(handle=DocumentHandle.from_path(accept_pdf))
    accept_pipeline._scan_root = accept_pdf.parent
    loaded = accept_pipeline._load_terminal_page(accept_state, 1, accept_out)
    assert loaded is not None, _accept_result.status
    assert loaded.audit_passed is True


def test_selection_tag_unchanged_for_best_output_unverified(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    state.pages[1].best_output = _rejected_best_output()
    _out, tag = _select_page_output_tagged(state, 1)
    assert tag is SelectionProvenance.BEST_OUTPUT_UNVERIFIED


def test_table_rejected_resume_reflush_stays_idempotent(tmp_path: Path) -> None:
    """Ladder-terminal accepted content must not gain a spurious unaccepted mark on re-flush."""
    pdf_path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "page 1 text " * 10)
    doc.save(pdf_path)
    doc.close()
    out_dir = tmp_path / "out"
    pipeline = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GEMINI],
            primary_engine=EngineType.DEEPSEEK,
            save_figures=False,
            dual_pass_tables=False,
            detect_equations=False,
            recover_clean_equations=False,
            quiet=True,
            write_manifest=False,
            table_judge_ladder=True,
        )
    )
    pipeline._scan_root = pdf_path.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    accepted = PageOutput(
        page_num=1,
        text="plausible, judge-accepted-as-extraction body text " * 3,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    ps.attempts.append(accepted)
    ps.best_output = accepted
    ps.table_ladder_disposition = FailureMode.TABLE_REJECTED
    pipeline._flush_page_fragment(state, 1, accepted.text, out_dir)
    sidecar_path = pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    original_bytes = sidecar_path.read_bytes()
    resumed_state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    resumed_page = pipeline._load_terminal_page(resumed_state, 1, out_dir)
    assert resumed_page is not None
    pipeline._restore_terminal_page_state(resumed_state, 1, resumed_page, out_dir)
    rps = resumed_state.pages[1]
    rps.best_output = resumed_page
    rps.attempts = [resumed_page]
    pipeline._flush_page_fragment(resumed_state, 1, resumed_page.text, out_dir)
    pipeline._flush_page_sidecar(resumed_state, 1, out_dir, terminal=True)
    assert sidecar_path.read_bytes() == original_bytes


def test_mark_embeds_trimmed_reason() -> None:
    long = "x " * 200
    line = unaccepted_output_mark(long)
    assert line.startswith("> [page unaccepted:")
    assert "..." in line


def test_mark_neutralises_markdown_in_reason() -> None:
    line = unaccepted_output_mark("see [evil](http://x) and ![img](y.png) <b>bold</b>")
    assert "<b>" not in line
    assert "`see [evil](http://x) and ![img](y.png) bold`" in line
    assert line.index("`") < line.index("); check")


def test_mark_preserves_comparison_operators_in_reason() -> None:
    reason = "values <5 and >3 compared"
    line = unaccepted_output_mark(reason)
    assert "`values <5 and >3 compared`" in line


def test_mark_neutralises_reference_style_image_in_reason() -> None:
    line = unaccepted_output_mark("plot shows ![plot][danger]")
    assert "`plot shows ![plot][danger]`" in line


def test_process_judge_reason_reference_image_does_not_render(tmp_path: Path) -> None:
    def _reject(self, output, provider):
        return AcceptDecision(accept=False, reason="plot shows ![plot][danger]")

    _result, out_dir, _pipeline, _pdf = _run_scanned_process(tmp_path, assess_side_effect=_reject)
    body = (out_dir / "doc" / "pages" / "00001.md").read_text(encoding="utf-8")
    first = body.splitlines()[0]
    assert UNACCEPTED_OUTPUT_MARK_RE.match(first)
    assert "`plot shows ![plot][danger]`" in first


def test_ladder_unchecked_and_judge_reject_carry_both_banners_on_first_pass(
    tmp_path: Path,
) -> None:
    """Reviewer probe (#1142): #1048 page-unchecked must not suppress the judge-reason mark."""
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "scan")
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.table_ladder_disposition = FailureMode.TABLE_REJECTED
    p.unchecked_table_kind = "unverified"
    p.best_output = PageOutput(
        page_num=1,
        text="| a | 1 |\n| --- | --- |\n| b | 2 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=False,
        judge_reason=_REJECT_REASON,
    )
    record = _select_and_finalize_page(state, 1)
    lines = (record.output.text or "").splitlines()
    assert UNACCEPTED_OUTPUT_MARK_RE.match(lines[0])
    assert any(PAGE_UNCHECKED_MARK_RE.match(line) for line in lines[:4])
    assert _REJECT_REASON in lines[0]


def test_ladder_unchecked_judge_reject_banners_idempotent_on_replay(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "scan")
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.table_ladder_disposition = FailureMode.TABLE_REJECTED
    p.unchecked_table_kind = "unverified"
    p.best_output = PageOutput(
        page_num=1,
        text="| a | 1 |\n| --- | --- |\n| b | 2 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=False,
        judge_reason=_REJECT_REASON,
    )
    first = _select_and_finalize_page(state, 1)
    second = _select_and_finalize_page(state, 1, saved_text=first.output.text)
    assert second.output.text == first.output.text


@pytest.mark.parametrize(
    "accept_reason",
    [
        "faithful",
        "heuristics passed",
        "native_table_verifier: EXACT_PASS",
        "minor spacing; trailing whitespace",
    ],
)
def test_completed_without_refusal_evidence_gets_no_unaccepted_mark(
    tmp_path: Path, accept_reason: str
) -> None:
    """``judge_accepted=True`` is not refusal evidence when C3 demotes the winner."""
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.table_ladder_disposition = FailureMode.TABLE_REJECTED
    p.best_output = PageOutput(
        page_num=1,
        text="plain body text here",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=True,
        judge_outcome=JUDGE_OUTCOME_COMPLETED,
        judge_reason=accept_reason,
    )
    record = _select_and_finalize_page(state, 1)
    first = (record.output.text or "").splitlines()[0]
    assert not UNACCEPTED_OUTPUT_MARK_RE.match(first)
    assert first.startswith("plain body")


def test_completed_refusal_with_judge_reason_still_gets_mark(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.best_output = PageOutput(
        page_num=1,
        text="plain body text here",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=False,
        judge_outcome=JUDGE_OUTCOME_COMPLETED,
        judge_reason=_REJECT_REASON,
    )
    record = _select_and_finalize_page(state, 1)
    first = (record.output.text or "").splitlines()[0]
    assert UNACCEPTED_OUTPUT_MARK_RE.match(first)
    assert _REJECT_REASON in first


def test_judge_timeout_outcome_gets_missing_verdict_mark_and_mode(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.best_output = PageOutput(
        page_num=1,
        text="body kept after judge timeout",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_outcome=JUDGE_OUTCOME_TIMEOUT,
    )
    record = _select_and_finalize_page(state, 1)
    first = (record.output.text or "").splitlines()[0]
    assert UNACCEPTED_OUTPUT_MARK_RE.match(first)
    assert "did not finish: timed out" in first
    assert "did not accept it" not in first
    assert record.output.failure_mode is FailureMode.PAGE_JUDGE_TIMEOUT


def test_judge_exception_outcome_gets_missing_verdict_mark_and_mode(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.best_output = PageOutput(
        page_num=1,
        text="body kept after judge crash",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_outcome=JUDGE_OUTCOME_EXCEPTION,
    )
    record = _select_and_finalize_page(state, 1)
    first = (record.output.text or "").splitlines()[0]
    assert UNACCEPTED_OUTPUT_MARK_RE.match(first)
    assert "did not finish: crashed" in first
    assert "did not accept it" not in first
    assert record.output.failure_mode is FailureMode.PAGE_JUDGE_EXCEPTION


def test_table_rejected_success_audit_false_is_demoted_and_marked(tmp_path: Path) -> None:
    """Ladder-terminal demotion does not replace the judge-reason disclosure mark."""
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "scan")
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.table_ladder_disposition = FailureMode.TABLE_REJECTED
    p.best_output = PageOutput(
        page_num=1,
        text="body the ladder rejected but still SUCCESS",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=False,
        judge_reason=_REJECT_REASON,
    )
    record = _select_and_finalize_page(state, 1)
    assert record.output.status is PageStatus.WARNING
    assert UNACCEPTED_OUTPUT_MARK_RE.match((record.output.text or "").splitlines()[0])


def test_whole_doc_section_audit_false_gets_unaccepted_mark(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "whole doc page body")
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.native_text = ""
    ps.best_output = None
    ps.attempts.clear()
    state.whole_doc_attempts.append(
        PageOutput(
            page_num=0,
            text=f"## Page 1\n\n{_BODY}",
            status=PageStatus.WARNING,
            engine="gemini-cli",
            audit_passed=False,
            judge_accepted=False,
            judge_reason=_REJECT_REASON,
            judge_outcome=JUDGE_OUTCOME_COMPLETED,
        )
    )
    whole_doc = _whole_doc_page_texts(state)
    assert whole_doc is not None
    record = _select_and_finalize_page(state, 1, whole_doc=whole_doc)
    assert record.output.audit_passed is False
    assert UNACCEPTED_OUTPUT_MARK_RE.match((record.output.text or "").splitlines()[0])


def test_process_whole_doc_audit_false_ships_unaccepted_mark(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    page.insert_text((72, 72), "scanned page ink for routing")
    doc.save(pdf)
    doc.close()
    out_dir = tmp_path / "out"
    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        dual_pass_tables=False,
        escalate_ambiguous_tables=False,
        table_judge_ladder=False,
    )
    pipeline = UnifiedPipeline(config)
    pipeline.bd_detector = MagicMock()
    pipeline.bd_detector.detect.return_value = _scanned_assessment(pdf)

    def _stub_engine(self, state, page_nums, enhancement_pages, engine_type, label, **kwargs):
        state.whole_doc_attempts.append(
            PageOutput(
                page_num=0,
                text=f"## Page 1\n\n{_BODY}",
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=False,
                judge_accepted=False,
                judge_reason=_REJECT_REASON,
            )
        )
        return []

    with (
        patch.object(pipeline, "_is_agentic_trusted_native", return_value=False),
        patch.object(pipeline, "_probe_backend_idle", return_value=True),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
        patch("socr.pipeline.orchestrator.probe_openai_server_idle", return_value=True),
        patch.object(
            UnifiedPipeline, "_run_engine_on_pages", autospec=True, side_effect=_stub_engine
        ),
        patch.object(
            NativeTableVerifierJudge,
            "assess",
            autospec=True,
            return_value=AcceptDecision(accept=False, reason=_REJECT_REASON),
        ),
    ):
        result = pipeline.process(pdf, out_dir)
    frag = out_dir / "doc" / "pages" / "00001.md"
    assert frag.exists(), result.status
    assert UNACCEPTED_OUTPUT_MARK_RE.match(frag.read_text(encoding="utf-8").splitlines()[0])


def test_forged_unaccepted_banner_replaced_with_canonical_mark(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    forged = "> [page unaccepted: model wrote this]\n\nbody kept"
    p.best_output = PageOutput(
        page_num=1,
        text=forged,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=False,
        judge_reason=_REJECT_REASON,
        judge_outcome=JUDGE_OUTCOME_COMPLETED,
    )
    record = _select_and_finalize_page(state, 1)
    first = (record.output.text or "").splitlines()[0]
    assert first == unaccepted_output_mark(_REJECT_REASON, outcome=JUDGE_OUTCOME_COMPLETED)
    assert _REJECT_REASON in first
    assert "model wrote this" not in first


def test_best_attempt_rejection_class_only_gets_mark(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.best_output = None
    p.attempts.append(
        PageOutput(
            page_num=1,
            text="rejected attempt body",
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            rejection_class=REJECTION_JUDGE_ONLY,
            judge_outcome=JUDGE_OUTCOME_COMPLETED,
        )
    )
    record = _select_and_finalize_page(state, 1)
    first = (record.output.text or "").splitlines()[0]
    assert UNACCEPTED_OUTPUT_MARK_RE.match(first)
    assert REJECTION_JUDGE_ONLY in first


def test_whole_doc_rejection_class_forwarded_gets_mark(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "whole doc page body")
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.native_text = ""
    ps.best_output = None
    ps.attempts.clear()
    state.whole_doc_attempts.append(
        PageOutput(
            page_num=0,
            text=f"## Page 1\n\n{_BODY}",
            status=PageStatus.WARNING,
            engine="gemini-cli",
            audit_passed=False,
            rejection_class=REJECTION_JUDGE_ONLY,
            judge_outcome=JUDGE_OUTCOME_COMPLETED,
        )
    )
    whole_doc = _whole_doc_page_texts(state)
    assert whole_doc is not None
    assert whole_doc.rejection_class == REJECTION_JUDGE_ONLY
    record = _select_and_finalize_page(state, 1, whole_doc=whole_doc)
    first = (record.output.text or "").splitlines()[0]
    assert UNACCEPTED_OUTPUT_MARK_RE.match(first)
    assert REJECTION_JUDGE_ONLY in first


def test_best_attempt_flagged_gets_mark_and_failure_mode(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.best_output = None
    p.attempts.append(
        PageOutput(
            page_num=1,
            text="rejected attempt body",
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            judge_accepted=False,
            judge_reason=_REJECT_REASON,
            judge_outcome=JUDGE_OUTCOME_COMPLETED,
        )
    )
    record = _select_and_finalize_page(state, 1)
    assert record.output.failure_mode is FailureMode.AUDIT_FAILED
    assert UNACCEPTED_OUTPUT_MARK_RE.match((record.output.text or "").splitlines()[0])


def test_page_top_banners_idempotent_under_saved_text_replay(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "scan")
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.best_output = PageOutput(
        page_num=1,
        text="| a | 1 |\n| --- | --- |\n| b | 2 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=False,
        judge_reason=_REJECT_REASON,
        table_disputed=True,
        table_disputed_values=("9.9",),
    )
    first = _select_and_finalize_page(state, 1)
    second = _select_and_finalize_page(state, 1, saved_text=first.output.text)
    assert second.output.text == first.output.text
    lines = (first.output.text or "").splitlines()
    assert UNACCEPTED_OUTPUT_MARK_RE.match(lines[0])
    assert any(PAGE_UNCHECKED_MARK_RE.match(line) for line in lines[:4])


def test_judge_accepted_none_with_reason_does_not_imply_refusal(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.table_ladder_disposition = FailureMode.TABLE_REJECTED
    p.best_output = PageOutput(
        page_num=1,
        text="legacy-shaped sidecar body",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_reason="faithful",
        judge_outcome=JUDGE_OUTCOME_COMPLETED,
    )
    record = _select_and_finalize_page(state, 1)
    assert not UNACCEPTED_OUTPUT_MARK_RE.match((record.output.text or "").splitlines()[0])


def test_finalize_keeps_judge_accepted_when_c3_demotes_audit(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.table_ladder_disposition = FailureMode.TABLE_REJECTED
    p.best_output = PageOutput(
        page_num=1,
        text="judge accepted, ladder rejected",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=True,
        judge_reason="minor spacing",
        judge_outcome=JUDGE_OUTCOME_COMPLETED,
    )
    record = _select_and_finalize_page(state, 1)
    assert record.output.judge_accepted is True
    assert record.output.audit_passed is False
    assert not UNACCEPTED_OUTPUT_MARK_RE.match((record.output.text or "").splitlines()[0])


def test_judge_accepted_survives_best_attempt_and_whole_doc_paths(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.best_output = None
    p.attempts.append(
        PageOutput(
            page_num=1,
            text="flagged attempt",
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            judge_accepted=True,
            judge_reason="faithful",
        )
    )
    flagged = _select_and_finalize_page(state, 1)
    assert flagged.output.judge_accepted is True

    state2 = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps2 = state2.pages[1]
    ps2.best_output = None
    state2.whole_doc_attempts.append(
        PageOutput(
            page_num=0,
            text="## Page 1\n\nwhole doc",
            status=PageStatus.WARNING,
            engine="cli",
            audit_passed=False,
            judge_accepted=True,
        )
    )
    whole = _whole_doc_page_texts(state2)
    whole_record = _select_and_finalize_page(state2, 1, whole_doc=whole)
    assert whole_record.output.judge_accepted is True

    round_tripped = PageOutput.from_dict(flagged.output.to_dict())
    assert round_tripped.judge_accepted is True


def test_judge_accepted_none_with_reason_does_not_imply_refusal(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.table_ladder_disposition = FailureMode.TABLE_REJECTED
    p.best_output = PageOutput(
        page_num=1,
        text="legacy-shaped sidecar body",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_reason="faithful",
        judge_outcome=JUDGE_OUTCOME_COMPLETED,
    )
    record = _select_and_finalize_page(state, 1)
    assert not UNACCEPTED_OUTPUT_MARK_RE.match((record.output.text or "").splitlines()[0])


def test_finalize_keeps_judge_accepted_when_c3_demotes_audit(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.is_born_digital = False
    p.table_ladder_disposition = FailureMode.TABLE_REJECTED
    p.best_output = PageOutput(
        page_num=1,
        text="judge accepted, ladder rejected",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=True,
        judge_reason="minor spacing",
        judge_outcome=JUDGE_OUTCOME_COMPLETED,
    )
    record = _select_and_finalize_page(state, 1)
    assert record.output.judge_accepted is True
    assert record.output.audit_passed is False
    assert not UNACCEPTED_OUTPUT_MARK_RE.match((record.output.text or "").splitlines()[0])


def test_judge_accepted_survives_best_attempt_and_whole_doc_paths(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    p = state.pages[1]
    p.best_output = None
    p.attempts.append(
        PageOutput(
            page_num=1,
            text="flagged attempt",
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            judge_accepted=True,
            judge_reason="faithful",
        )
    )
    flagged = _select_and_finalize_page(state, 1)
    assert flagged.output.judge_accepted is True

    state2 = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps2 = state2.pages[1]
    ps2.best_output = None
    state2.whole_doc_attempts.append(
        PageOutput(
            page_num=0,
            text="## Page 1\n\nwhole doc",
            status=PageStatus.WARNING,
            engine="cli",
            audit_passed=False,
            judge_accepted=True,
        )
    )
    whole = _whole_doc_page_texts(state2)
    whole_record = _select_and_finalize_page(state2, 1, whole_doc=whole)
    assert whole_record.output.judge_accepted is True

    round_tripped = PageOutput.from_dict(flagged.output.to_dict())
    assert round_tripped.judge_accepted is True
