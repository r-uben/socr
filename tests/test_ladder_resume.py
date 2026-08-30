"""TICKET-D1b (GH-353): table judge ladder resume skip policy.

``_load_terminal_page`` requires ``status == SUCCESS`` and ``audit_passed is
True`` to grant a resume skip. C3's ``_apply_ladder_disposition_guard``
demotes a page carrying a GH-353 ladder terminal to ``status=WARNING,
audit_passed=False`` in its finalized ``winning_output`` -- so, before this
ticket, BOTH ladder terminals (REJECTED and UNVERIFIED) fell through the
existing gate and were reprocessed on every resume.

D1b adds ONE deliberate exception: a page whose sidecar
``table_ladder_disposition`` is ``FailureMode.TABLE_REJECTED`` now skips the
SUCCESS/audit_passed checks and IS skip-and-kept -- REJECTED is a
corroborated CONTENT verdict (both ladder rungs looked and said no), final
for the same input+config, not an infra doubt. UNVERIFIED gets NO such
exception (the ladder ran out of witnesses/rungs without an answer -- an
infra-shaped doubt) and keeps falling through to reprocess.

Hermetic: drives ``_flush_page_fragment`` / ``_flush_page_sidecar`` /
``_load_terminal_page`` directly, mirroring
``tests/test_gh161_resume_ledger_audit_gate.py``'s pattern -- no provider
ladder, no ``_phase_agentic``, so no ``_available_engines_for_agentic``
patch is needed. No rung/witness machinery is reachable from this surface.
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


def _make_pipeline(**overrides) -> UnifiedPipeline:
    kwargs = dict(
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
    kwargs.update(overrides)
    return UnifiedPipeline(PipelineConfig(**kwargs))


def _real_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "page 1 text " * 10)
    doc.save(str(path))
    doc.close()
    return path


def _seed_and_flush(
    pipeline: UnifiedPipeline,
    state: DocumentState,
    out_dir: Path,
    disposition: FailureMode | None,
) -> Path:
    """Mirror what B1's gate leaves behind for a demoted page: an ACCEPTED
    ``audit_passed=True`` attempt whose PageState disposition is set to
    ``disposition`` -- ``_apply_ladder_disposition_guard`` (C3) then demotes
    the FINALIZED ``winning_output`` in the sidecar to ``status=WARNING,
    audit_passed=False`` exactly as it does in production, so this fixture
    reproduces the real shape the gate must decide on, not a synthetic one.
    """
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
    ps.table_ladder_disposition = disposition

    pipeline._flush_page_fragment(state, 1, accepted.text, out_dir)
    sidecar_path = pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    return sidecar_path


class TestRejectedSkipsAndKeeps:
    def test_rejected_page_matching_fingerprint_is_skipped(self, tmp_path: Path) -> None:
        pdf_path = _real_pdf(tmp_path)
        out_dir = tmp_path / "out"
        pipeline = _make_pipeline()
        pipeline._scan_root = pdf_path.parent
        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))

        sidecar_path = _seed_and_flush(pipeline, state, out_dir, FailureMode.TABLE_REJECTED)
        meta = json.loads(sidecar_path.read_text())

        # Precondition: C3's guard really did demote the finalized winner --
        # if this fails, the rest of the test would be vacuous (the SUCCESS/
        # audit_passed checks would already grant the skip on their own).
        assert meta["table_ladder_disposition"] == FailureMode.TABLE_REJECTED.value, meta
        assert meta["winning_output"]["status"] == PageStatus.WARNING.value, meta["winning_output"]
        assert meta["winning_output"]["audit_passed"] is False, meta["winning_output"]

        resumed = pipeline._load_terminal_page(state, 1, out_dir)
        assert resumed is not None, "REJECTED page with a matching fingerprint must skip-and-keep"
        assert resumed.text.strip() == meta["winning_output"]["text"].strip()

    def test_sidecar_bytes_unchanged_after_resumed_run_reflushes(self, tmp_path: Path) -> None:
        """A REJECTED page that skips must reproduce byte-identical sidecar
        bytes when the resumed run re-flushes it (nothing about the skip
        path should perturb the persisted disposition/events)."""
        pdf_path = _real_pdf(tmp_path)
        out_dir = tmp_path / "out"
        pipeline = _make_pipeline()
        pipeline._scan_root = pdf_path.parent
        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))

        sidecar_path = _seed_and_flush(pipeline, state, out_dir, FailureMode.TABLE_REJECTED)
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

    def test_rejected_page_with_changed_rung_identity_is_reprocessed(self, tmp_path: Path) -> None:
        """The run fingerprint already binds B1's ladder extras (rung
        identities among them, per ``_run_fingerprint``'s ``extra`` dict) --
        a changed rung2 binary must invalidate the ledger entry before the
        REJECTED disposition is ever consulted, exactly like any other
        fingerprint mismatch."""
        pdf_path = _real_pdf(tmp_path)
        out_dir = tmp_path / "out"
        pipeline = _make_pipeline(table_judge_rung2_binary="agy")
        pipeline._scan_root = pdf_path.parent
        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))

        _seed_and_flush(pipeline, state, out_dir, FailureMode.TABLE_REJECTED)

        changed_pipeline = _make_pipeline(table_judge_rung2_binary="some-other-binary")
        changed_pipeline._scan_root = pdf_path.parent
        assert changed_pipeline._run_fingerprint() != pipeline._run_fingerprint()

        resumed_state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        resumed = changed_pipeline._load_terminal_page(resumed_state, 1, out_dir)
        assert resumed is None, (
            "A changed rung2 binary must reprocess a REJECTED page, not reuse the stale ledger"
        )


class TestUnverifiedAlwaysReprocesses:
    def test_unverified_page_matching_fingerprint_is_reprocessed(self, tmp_path: Path) -> None:
        pdf_path = _real_pdf(tmp_path)
        out_dir = tmp_path / "out"
        pipeline = _make_pipeline()
        pipeline._scan_root = pdf_path.parent
        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))

        sidecar_path = _seed_and_flush(pipeline, state, out_dir, FailureMode.TABLE_UNVERIFIED)
        meta = json.loads(sidecar_path.read_text())

        # Same demoted shape as REJECTED -- UNVERIFIED is also a ladder terminal
        # that C3's guard demotes, so this precondition rules out the
        # SUCCESS/audit_passed checks vacuously passing on their own.
        assert meta["table_ladder_disposition"] == FailureMode.TABLE_UNVERIFIED.value, meta
        assert meta["winning_output"]["status"] == PageStatus.WARNING.value, meta["winning_output"]
        assert meta["winning_output"]["audit_passed"] is False, meta["winning_output"]

        resumed = pipeline._load_terminal_page(state, 1, out_dir)
        assert resumed is None, "UNVERIFIED must never skip -- it exists only because infra failed"


class TestBaselineUntouched:
    def test_no_ladder_disposition_behaves_as_before(self, tmp_path: Path) -> None:
        """Control: a page with no ladder record at all (flag off / no
        tables / ladder ACCEPTED) keeps status=SUCCESS, audit_passed=True and
        must still skip through the ORIGINAL checks -- proves this ticket did
        not disturb the pre-existing gate for the non-ladder majority case."""
        pdf_path = _real_pdf(tmp_path)
        out_dir = tmp_path / "out"
        pipeline = _make_pipeline()
        pipeline._scan_root = pdf_path.parent
        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))

        sidecar_path = _seed_and_flush(pipeline, state, out_dir, None)
        meta = json.loads(sidecar_path.read_text())
        assert meta["table_ladder_disposition"] is None
        assert meta["winning_output"]["status"] == PageStatus.SUCCESS.value
        assert meta["winning_output"]["audit_passed"] is True

        resumed = pipeline._load_terminal_page(state, 1, out_dir)
        assert resumed is not None
