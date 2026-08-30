"""TICKET-C2 (GH-353): table judge ladder terminals surface at every level.

No-silent-loss guard: a page carrying ``FailureMode.TABLE_REJECTED`` /
``FailureMode.TABLE_UNVERIFIED`` (the two ladder terminals, B1's job to set —
this ticket injects them directly at the ``_phase_assemble`` seam, mirroring
the chart-lane test pattern in ``test_chart_lane.py``) must flip:

1. ``pages_ok`` -> document status ``AUDIT_FAILED`` (never a silent SUCCESS).
2. ``metadata.json`` -> ``Status.PARTIAL`` with an error note naming the mode.
3. ``_print_summary`` CLI output -> both terminal names visible.

Keyed off ``best_output.failure_mode``, NOT ``.status`` -- a page can
legitimately arrive as SUCCESS/audit_passed=False (GH-161), so a test that
only varied ``.status`` would not catch a regression that reads the wrong
field (the C3 reviewer note this ticket was built against).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline


def _make_pipeline(**overrides) -> UnifiedPipeline:
    cfg = PipelineConfig(
        primary_engine=EngineType.DEEPSEEK,
        enabled_engines=list(EngineType),
        agentic=False,
        quiet=True,
        native_first=True,
        **overrides,
    )
    return UnifiedPipeline(cfg)


def _make_state(tmp_path: Path, page_count: int = 1) -> DocumentState:
    pdf = tmp_path / "doc.pdf"
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=page_count)
    state = DocumentState(handle=handle)
    for pn in range(1, page_count + 1):
        ps = state.pages[pn]
        ps.is_born_digital = True
        ps.native_text = f"page {pn} prose"
    return state


def _inject_terminal(
    state: DocumentState,
    page_num: int,
    failure_mode: FailureMode,
    *,
    page_status: PageStatus = PageStatus.WARNING,
    audit_passed: bool = False,
) -> None:
    """Mirrors the chart-lane injection pattern: set attempts + best_output directly.

    ``page_status``/``audit_passed`` are overridable so a test can reproduce
    the GH-161 shape (status=SUCCESS, audit_passed=False) and prove
    aggregation still keys off ``failure_mode``.
    """
    out = PageOutput(
        page_num=page_num,
        text=f"page {page_num} kept text",
        status=page_status,
        engine="qwen",
        audit_passed=audit_passed,
        failure_mode=failure_mode,
        cost_usd=0.0,
    )
    ps = state.pages[page_num]
    ps.attempts.append(out)
    ps.best_output = out


class TestPagesOkAndDocumentStatus:
    def test_table_rejected_flips_document_status(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        state = _make_state(tmp_path)
        _inject_terminal(state, 1, FailureMode.TABLE_REJECTED)

        result = pipeline._phase_assemble(state, tmp_path)

        assert result.status == DocumentStatus.AUDIT_FAILED

    def test_table_unverified_flips_document_status(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        state = _make_state(tmp_path)
        _inject_terminal(state, 1, FailureMode.TABLE_UNVERIFIED)

        result = pipeline._phase_assemble(state, tmp_path)

        assert result.status == DocumentStatus.AUDIT_FAILED

    def test_gh161_shape_still_flips_status_when_keyed_off_failure_mode(
        self, tmp_path: Path
    ) -> None:
        """GH-161: status=SUCCESS + audit_passed=False must not slip through.

        The manifest can legitimately ship this combination (C3's guard only
        backfills ``failure_mode``, never rewrites ``.status``). Aggregation
        must still catch it because it reads ``failure_mode``, not
        ``.status``.
        """
        pipeline = _make_pipeline()
        state = _make_state(tmp_path)
        _inject_terminal(
            state,
            1,
            FailureMode.TABLE_REJECTED,
            page_status=PageStatus.SUCCESS,
            audit_passed=False,
        )

        result = pipeline._phase_assemble(state, tmp_path)

        assert result.status == DocumentStatus.AUDIT_FAILED

    def test_clean_document_is_unaffected(self, tmp_path: Path) -> None:
        """Control: a document with no ladder terminal ships clean SUCCESS."""
        pipeline = _make_pipeline()
        state = _make_state(tmp_path)
        clean = PageOutput(
            page_num=1,
            text="page 1 prose",
            status=PageStatus.SUCCESS,
            engine="native",
            audit_passed=True,
            cost_usd=0.0,
        )
        state.pages[1].attempts.append(clean)
        state.pages[1].best_output = clean

        result = pipeline._phase_assemble(state, tmp_path)

        assert result.status == DocumentStatus.SUCCESS
        assert not (result.error and "table_rejected" in result.error)
        assert not (result.error and "table_unverified" in result.error)


class TestMetadataNote:
    def test_rejected_note_reaches_metadata_json(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        state = _make_state(tmp_path)
        _inject_terminal(state, 1, FailureMode.TABLE_REJECTED)

        pipeline._phase_assemble(state, tmp_path)

        doc_dir = tmp_path / "doc"
        meta = json.loads((doc_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["status"] == "partial"
        assert "table_rejected" in meta["error"]

    def test_unverified_note_reaches_metadata_json(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        state = _make_state(tmp_path)
        _inject_terminal(state, 1, FailureMode.TABLE_UNVERIFIED)

        pipeline._phase_assemble(state, tmp_path)

        doc_dir = tmp_path / "doc"
        meta = json.loads((doc_dir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["status"] == "partial"
        assert "table_unverified" in meta["error"]


class TestCliSummary:
    def test_print_summary_names_both_terminals(self, tmp_path: Path, capsys) -> None:
        pipeline = _make_pipeline()
        state = _make_state(tmp_path, page_count=2)
        _inject_terminal(state, 1, FailureMode.TABLE_REJECTED)
        _inject_terminal(state, 2, FailureMode.TABLE_UNVERIFIED)

        result = pipeline._phase_assemble(state, tmp_path)
        pipeline._print_summary(result, state)

        captured = capsys.readouterr()
        assert "table_rejected" in captured.out
        assert "table_unverified" in captured.out


class TestPagesOkBucketsAreSeparate:
    def test_rejected_and_unverified_pages_both_counted(self, tmp_path: Path) -> None:
        """Regression guard for the aggregation predicate itself.

        A document with one REJECTED and one UNVERIFIED page must name both
        page numbers in the note (one bucket per disposition, matching the
        rest of ``_phase_assemble``'s pairing pattern).
        """
        pipeline = _make_pipeline()
        state = _make_state(tmp_path, page_count=2)
        _inject_terminal(state, 1, FailureMode.TABLE_REJECTED)
        _inject_terminal(state, 2, FailureMode.TABLE_UNVERIFIED)

        result = pipeline._phase_assemble(state, tmp_path)

        assert result.status == DocumentStatus.AUDIT_FAILED
        assert result.error is not None
        assert "1" in result.error
        assert "2" in result.error
        assert "table_rejected" in result.error
        assert "table_unverified" in result.error
