"""GH-161: the resume ledger must not treat a judge-rejected page as terminal.

`_load_terminal_page` grants a resume skip on `winning_output.status == SUCCESS`
alone.  A SCANNED page whose every OCR attempt was rejected by the judge keeps
provider status SUCCESS while carrying `audit_passed=False` (agentic best-effort:
`att.output.audit_passed = att.accepted` in `_phase_agentic`), and
`_winning_page_output` returns that rejected attempt verbatim for a scanned page
(no born-digital native fallback exists to demote it to WARNING).  The sidecar
therefore records status="success" next to audit_passed=false, and the gate skips
the page on resume — restoring text every judge rejected.

The born-digital sibling of this shape is already covered by
``test_flagged_native_fallback_sidecar_records_winner_not_attempt``: there the
native fallback demotes the winner to WARNING, so the status check catches it.
On a scanned page nothing demotes the status, and only `audit_passed` remains.

Hermetic: drives `_flush_page_*` / `_load_terminal_page` directly — no provider
ladder, no `_phase_agentic`, so no `_available_engines_for_agentic` patch needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

from gh1185_agreement_spy import assert_agreed_on, spy_agreement


def _make_pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
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
        )
    )


def _real_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "page 1 text " * 10)
    doc.save(str(path))
    doc.close()
    return path


def test_legacy_success_audit_false_sidecar_is_not_resume_skippable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GH-161 / #1129 r2: old sidecars stayed SUCCESS with audit_passed=false.

    The WARNING status gate alone does not cover them; ``audit_passed is True`` must
    still refuse the skip. Prove the audit gate is load-bearing by flushing without
    the #1129 demotion/mark guards.
    """
    from socr.core import manifest as manifest_mod

    calls = spy_agreement(monkeypatch)  # #1185: must be installed before the code under test runs

    monkeypatch.setattr(manifest_mod, "_finalize_best_output_unverified", lambda output: output)
    monkeypatch.setattr(
        manifest_mod,
        "_apply_unaccepted_output_mark",
        lambda output, *args, **kwargs: output,
    )

    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent

    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.native_text = ""
    # #1185 (approved fixture change): control page given an own text layer and a number, so it is a real CHECKED reading
    ps.own_layer_text = "Goldman Sachs 2.3 1.9 judge-rejected body text"

    rejected = PageOutput(
        page_num=1,
        # #1185 (approved fixture change): the reading ships the same number as its own layer
        text="Goldman Sachs 2.3 plausible but judge-rejected body text without demotion guards",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=False,
    )
    ps.attempts.append(rejected)
    ps.best_output = rejected

    pipeline._flush_page_fragment(state, 1, rejected.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    sidecar = next(out_dir.rglob("pages/00001.json"))
    meta = json.loads(sidecar.read_text())
    assert meta["winning_output"]["status"] == PageStatus.SUCCESS.value
    assert meta["winning_output"]["audit_passed"] is False
    # #1185: the reading was compared against its own layer and agreed (the check ran).
    assert_agreed_on(calls, "Goldman Sachs 2.3 plausible")

    resumed = pipeline._load_terminal_page(state, 1, out_dir)
    assert resumed is None
    kinds = [ev.kind for ev in state.events if ev.page_num == 1]
    assert "resume_ledger_audit_reject" in kinds
    pipeline._load_terminal_page(state, 1, out_dir)
    after = [ev.kind for ev in state.events if ev.page_num == 1]
    assert after.count("resume_ledger_audit_reject") == 1


def test_judge_rejected_success_page_is_not_resume_skippable(tmp_path: Path) -> None:
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent

    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    # A SCANNED page: no native layer, so no fallback exists to demote the winner.
    ps.is_born_digital = False
    ps.native_text = ""

    # Every OCR attempt was rejected by the judge.  Extraction SUCCEEDED as an
    # operation (status=SUCCESS), the CONTENT was rejected (audit_passed=False) —
    # exactly what agentic best-effort keeps when nothing is accepted.
    rejected = PageOutput(
        page_num=1,
        text="| Goldman Sachs | 2.3 | 1.9 |\n\nplausible but judge-rejected body text",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        judge_accepted=False,
        judge_reason="table values disagree with the page image",
    )
    ps.attempts.append(rejected)
    ps.best_output = rejected

    pipeline._flush_page_fragment(state, 1, rejected.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    sidecar = next(out_dir.rglob("pages/00001.json"))
    meta = json.loads(sidecar.read_text())

    # Preconditions: terminal sidecar with a judge-rejected winner.  #1129 demotes
    # the finalized winner to WARNING (never SUCCESS + audit_passed=False).
    assert meta["terminal"] is True, meta
    assert meta["run_fingerprint"] == pipeline._run_fingerprint(), meta
    assert meta["winning_output"]["audit_passed"] is False, meta["winning_output"]
    assert meta["winning_output"]["status"] == PageStatus.WARNING.value, meta["winning_output"]
    assert meta["winning_output"]["text"].splitlines()[0].startswith("> [page unaccepted:")

    resumed = pipeline._load_terminal_page(state, 1, out_dir)
    assert resumed is None, (
        f"A judge-rejected best-effort page must not be restored on resume: {resumed!r}"
    )
    assert not [ev for ev in state.events if ev.kind == "resume_ledger_audit_reject"]


def test_error_status_audit_false_emits_no_resume_ledger_event(tmp_path: Path) -> None:
    """Cascade-halt ERROR pages must not gain ``resume_ledger_audit_reject`` (main behaviour)."""
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.native_text = ""
    errored = PageOutput(
        page_num=1,
        text="partial body before cascade halt",
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
        error="VLM timeout",
    )
    ps.attempts.append(errored)
    ps.best_output = errored
    pipeline._flush_page_fragment(state, 1, errored.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    resumed = pipeline._load_terminal_page(state, 1, out_dir)
    assert resumed is None
    assert not [ev for ev in state.events if ev.kind == "resume_ledger_audit_reject"]


def test_audit_passed_success_page_is_still_resume_skippable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reverse regression: the gate must still SKIP a genuinely clean page.

    A fix that simply stopped granting skips would satisfy the test above and
    silently destroy resume — every page re-OCR'd on every re-run.  Same setup,
    same sidecar shape, only the audit verdict differs.
    """
    calls = spy_agreement(monkeypatch)  # #1185: must be installed before the code under test runs
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent

    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.native_text = ""
    # #1185 (approved fixture change): control page given an own text layer and a number, so it is a real CHECKED reading
    ps.own_layer_text = "Goldman Sachs 2.3 1.9 accepted body text"

    accepted = PageOutput(
        page_num=1,
        text="| Goldman Sachs | 2.3 | 1.9 |\n\naccepted body text",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    ps.attempts.append(accepted)
    ps.best_output = accepted

    pipeline._flush_page_fragment(state, 1, accepted.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    sidecar = next(out_dir.rglob("pages/00001.json"))
    meta = json.loads(sidecar.read_text())
    assert meta["winning_output"]["status"] == PageStatus.SUCCESS.value, meta["winning_output"]
    assert meta["winning_output"]["audit_passed"] is True, meta["winning_output"]
    assert meta["winning_output"]["engine"] == "qwen"  # #1185: still a model reading
    # #1185: the control reading was compared against its own layer and agreed.
    assert_agreed_on(calls, "Goldman Sachs | 2.3")

    resumed = pipeline._load_terminal_page(state, 1, out_dir)
    assert resumed is not None, (
        "A clean page (status=SUCCESS, audit_passed=True) must still be skippable "
        "on resume — the gate has stopped skipping anything and resume is dead"
    )
    assert resumed.text == accepted.text
    assert not [ev for ev in state.events if ev.kind == "resume_ledger_audit_reject"], (
        "A clean page must not be reported as judge-rejected"
    )


def test_native_fallback_warning_audit_false_emits_no_resume_ledger_event(
    tmp_path: Path,
) -> None:
    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.native_text = "native fallback body"
    fallback = PageOutput(
        page_num=1,
        text="native fallback body",
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=False,
        failure_mode=FailureMode.NONE,
    )
    ps.attempts.append(fallback)
    ps.best_output = fallback
    pipeline._flush_page_fragment(state, 1, fallback.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    pipeline._load_terminal_page(state, 1, out_dir)
    assert not [ev for ev in state.events if ev.kind == "resume_ledger_audit_reject"]


def test_judge_accepted_table_unverified_warning_emits_no_resume_ledger_event(
    tmp_path: Path,
) -> None:
    from socr.core.manifest import finalized_page_record
    from socr.core.result import JUDGE_OUTCOME_COMPLETED

    pdf_path = _real_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.table_ladder_disposition = FailureMode.TABLE_UNVERIFIED
    ps.best_output = PageOutput(
        page_num=1,
        text="judge accepted then ladder demoted",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        judge_accepted=True,
        judge_reason="faithful",
        judge_outcome=JUDGE_OUTCOME_COMPLETED,
    )
    ps.attempts.append(ps.best_output)
    record = finalized_page_record(state, 1)
    demoted = record.output
    assert demoted.status is PageStatus.WARNING
    assert demoted.audit_passed is False
    pipeline._flush_page_fragment(state, 1, demoted.text or "", out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True, record=record)
    assert pipeline._load_terminal_page(state, 1, out_dir) is None
    assert not [ev for ev in state.events if ev.kind == "resume_ledger_audit_reject"]
