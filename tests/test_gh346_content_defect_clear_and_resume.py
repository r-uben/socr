"""GH-346: the GH-303 content term must clear on recovery and survive resume.

``native_table_content_defect`` was added in GH-303 as a sibling of
``native_table_emission_defect``, but neither the recovery clear path nor the
page sidecar was updated. So a page whose empty native table was recovered by an
accepted escalation kept empty-table provenance the clear path claimed to have
released, and a resumed run dropped the field entirely.

Both halves are pinned as DIFFERENCES: the same page, with and without the
production line, so neither can pass for an unrelated reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.config import PipelineConfig
from socr.core.state import DocumentHandle, DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

fitz = pytest.importorskip("fitz")

_EMISSION = "table_width_mismatch"
_CONTENT = "table_content_empty"


def _state() -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=Path("/tmp/gh346.pdf"), page_count=1)
    return DocumentState(handle=handle)


def _flagged_page(state: DocumentState):
    ps = state.pages[1]
    ps.native_table_structure_failed = True
    ps.native_table_structure_defective = True
    ps.native_table_emission_defect = _EMISSION
    ps.native_table_content_defect = _CONTENT
    return ps


class TestRecoveryClearsTheContentTerm:
    def test_both_defect_terms_are_released_together(self) -> None:
        """The bug: emission cleared, content left behind. They are siblings and
        the clear path claims to release the page's fail-closed provenance."""
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True))
        state = _state()
        ps = _flagged_page(state)

        from socr.core.providers import PROFILE_QWEN_LOCAL

        pipeline._clear_fail_closed_flags(state, 1, ps, PROFILE_QWEN_LOCAL)

        assert ps.native_table_emission_defect == ""
        assert ps.native_table_content_defect == "", (
            "the content term survived a recovery that released its sibling"
        )

    def test_a_page_with_no_flags_is_untouched(self) -> None:
        """Inertness: the clear path must not invent an event for a clean page,
        or the pin above could pass by clearing everything unconditionally."""
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True))
        state = _state()

        from socr.core.providers import PROFILE_QWEN_LOCAL

        pipeline._clear_fail_closed_flags(state, 1, state.pages[1], PROFILE_QWEN_LOCAL)

        assert not [e for e in state.events if e.kind == "table_escalation_recovered_fail_closed"]


class TestTheSidecarRoundTripsTheContentTerm:
    def test_the_term_survives_a_write_and_restore(self, tmp_path: Path) -> None:
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True))
        pipeline._scan_root = tmp_path
        state = _state()
        _flagged_page(state)

        sidecar = pipeline._flush_page_sidecar(state, 1, tmp_path, terminal=False)
        meta = json.loads(sidecar.read_text(encoding="utf-8"))

        assert meta.get("native_table_content_defect") == _CONTENT, (
            "the sidecar never recorded the content term, so a resume cannot restore it"
        )

        restored = _state()
        from socr.core.result import PageOutput, PageStatus

        reconstructed = PageOutput(
            page_num=1,
            text="body",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        pipeline._restore_terminal_page_state(restored, 1, reconstructed, tmp_path)
        assert restored.pages[1].native_table_content_defect == _CONTENT

    def test_emission_and_content_round_trip_the_same_way(self, tmp_path: Path) -> None:
        """Difference pin against a half-fix: whatever happens to emission must
        happen to content, since GH-303 made them siblings."""
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True))
        pipeline._scan_root = tmp_path
        state = _state()
        _flagged_page(state)

        sidecar = pipeline._flush_page_sidecar(state, 1, tmp_path, terminal=False)
        meta = json.loads(sidecar.read_text(encoding="utf-8"))

        assert ("native_table_emission_defect" in meta) == ("native_table_content_defect" in meta)
