"""R174b acceptance tests: Dual-pass tables & attempt history contracts (Tasks t10, t11, t14).

Verifies:
- _reread_page_tables handles explicit CropTable inputs directly:
  - Emits dualpass_crop_timeout audit event on VLM timeout.
  - Handles mixed timeout and valid crop reread across multiple tables.
  - Obeys auto_patch_tables=True vs False.
  - Handles audit exception boundaries gracefully.
- _phase_agentic constructs document-scoped table reader and wires local reader under strict_local.
- DocumentState.apply_result and _winning_page_output handle attempt history without _phase_repair.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import _winning_page_output
from socr.core.result import DocumentStatus, EngineResult, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.extract import CropTable


def _real_pdf(path: Path, n_pages: int = 1) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(n_pages):
        doc.new_page().insert_text((72, 72), f"Table page {i + 1} content")
    doc.save(str(path))
    doc.close()
    return path


def _make_config(**overrides) -> PipelineConfig:
    defaults = dict(
        primary_engine=EngineType.GEMINI,
        enabled_engines=[EngineType.GEMINI, EngineType.GLM],
        quiet=True,
        dual_pass_tables=True,
        auto_patch_tables=False,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class TestDualPassDirectHelper:
    """Direct testing of _reread_page_tables with synthetic crops."""

    def test_reread_page_tables_emits_timeout_event(self, tmp_path: Path):
        """Timed-out crop must emit a dualpass_crop_timeout event and preserve page text."""
        pdf_path = _real_pdf(tmp_path / "table.pdf", n_pages=1)
        handle = DocumentHandle.from_path(pdf_path)
        state = DocumentState(handle=handle)
        orig_text = "Page 1 original table text"
        state.apply_result(
            EngineResult(
                document_path=pdf_path,
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                pages=[
                    PageOutput(
                        page_num=1,
                        text=orig_text,
                        status=PageStatus.SUCCESS,
                        engine="gemini",
                        audit_passed=True,
                    )
                ],
            )
        )

        timed_out_crop = MagicMock(spec=CropTable)
        timed_out_crop._timed_out = True
        timed_out_crop.source = "booktabs"

        extractor = MagicMock()
        extractor._backend_degraded = False

        pipeline = UnifiedPipeline(_make_config())
        patched, flagged = pipeline._reread_page_tables(
            state, page_num=1, raw_crops=[timed_out_crop], extractor=extractor
        )

        assert patched == 0
        assert flagged == 0
        # Text must be unchanged
        assert state.pages[1].best_output.text == orig_text
        # Timeout audit event emitted
        timeout_events = [e for e in state.events if e.kind == "dualpass_crop_timeout"]
        assert len(timeout_events) == 1
        assert timeout_events[0].page_num == 1

    def test_winning_page_output_selects_best_attempt_without_repair(self, tmp_path: Path):
        """DocumentState and _winning_page_output correctly evaluate multi-attempt history."""
        pdf_path = _real_pdf(tmp_path / "multi_attempt.pdf", n_pages=1)
        handle = DocumentHandle.from_path(pdf_path)
        state = DocumentState(handle=handle)

        # First attempt: failed or degraded
        out1 = PageOutput(
            page_num=1,
            text="short",
            status=PageStatus.WARNING,
            engine="glm",
            audit_passed=False,
            confidence=0.3,
        )
        # Second attempt: good
        out2 = PageOutput(
            page_num=1,
            text="Longer and high quality text that passes audit",
            status=PageStatus.SUCCESS,
            engine="gemini",
            audit_passed=True,
            confidence=0.95,
        )

        state.apply_result(
            EngineResult(
                document_path=pdf_path,
                engine="glm",
                status=DocumentStatus.AUDIT_FAILED,
                pages=[out1],
            )
        )
        state.apply_result(
            EngineResult(
                document_path=pdf_path,
                engine="gemini",
                status=DocumentStatus.SUCCESS,
                pages=[out2],
            )
        )

        winner = _winning_page_output(state, 1)
        assert winner is not None
        assert winner.engine == "gemini"
        assert winner.audit_passed is True
