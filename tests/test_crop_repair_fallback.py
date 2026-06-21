"""GH-56 general table crop-repair fallback (issue #56).

Hermetic: stubs the crop VLM reader; no ollama/GPU. Verifies that structurally
broken tables are patched from a mocked crop reading when verification improves,
and rejected when it does not — even with auto_patch_tables=False.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.crop_repair import (
    crop_patch_improves_verification,
    page_needs_crop_repair_fallback,
    snapshot_structural_defects,
)
from socr.tables.extract import CropTable, TableCropExtractor
from socr.tables.locate import locate_tables

# ---------------------------------------------------------------------------
# Fixtures — broken tables (header collapse) + crop readings
# ---------------------------------------------------------------------------

_BROKEN_TEXT_TABLE_MD = """## Survey

| Topic | All responses combined |
| --- | --- |
| Quality | High | Medium | Low |
| Price | Cheap | Moderate | Expensive |
"""

_FIXED_TEXT_TABLE_MD = (
    "| Topic | High | Medium | Low |\n"
    "| --- | --- | --- | --- |\n"
    "| Quality | High | Medium | Low |\n"
    "| Price | Cheap | Moderate | Expensive |"
)

_BROKEN_PLAIN_HEADER_MD = """## Results

| Metric | Jan Feb Mar Apr |
| --- | --- |
| Revenue | 100 | 110 | 120 | 130 |
| Costs | 80 | 85 | 90 | 95 |
"""

_FIXED_PLAIN_HEADER_MD = (
    "| Metric | Jan | Feb | Mar | Apr |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| Revenue | 100 | 110 | 120 | 130 |\n"
    "| Costs | 80 | 85 | 90 | 95 |"
)

# Numeric typo table (structurally sound — crop-repair fallback must NOT fire).
_PAGE_MD_TYPO = """## Results

Prose before.

| var | (1) | (2) |
| --- | --- | --- |
| educ | 0.082 | 0.077 |
| se | (0.009) | (0.0l0) |

Prose after.
"""

_CROP_MD_TYPO = (
    "| var | (1) | (2) |\n| --- | --- | --- |\n| educ | 0.082 | 0.077 |\n| se | (0.009) | (0.010) |"
)

# Crop disagrees on a cell but keeps the collapsed header — no structural improvement.
_STILL_BROKEN_CROP_MD = (
    "| Metric | Jan Feb Mar Apr |\n"
    "| --- | --- |\n"
    "| Revenue | 999 | 110 | 120 | 130 |\n"
    "| Costs | 80 | 85 | 90 | 95 |"
)


def _assessment(has_tables: bool = True) -> DocumentAssessment:
    return DocumentAssessment(
        path=Path("/tmp/fake.pdf"),
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=False,
                native_text="",
                confidence=1.0,
                has_tables=has_tables,
                has_equations=False,
            )
        ],
    )


def _state_with_table_page(pdf: Path, page_text: str, *, engine: str = "qwen") -> tuple:
    state = DocumentState(handle=DocumentHandle(path=pdf))
    bo = PageOutput(
        page_num=1,
        text=page_text,
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=True,
    )
    state.pages[1].attempts.append(bo)
    state.pages[1].best_output = bo
    state.pages[1].has_tables = True
    return state, bo


def _build_booktabs_typo_page(tmp_path: Path) -> Path:
    """Ruled regression table page (mirrors test_dual_pass_tables layout)."""
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 220, 300, 380]
    data = [
        ["", "(1)", "(2)", "(3)"],
        ["log wage", "0.043", "0.051", "0.039"],
        ["", "(0.014)", "(0.016)", "(0.013)"],
        ["educ", "0.082", "0.077", "0.080"],
        ["N", "1,204", "1,204", "1,204"],
        ["R2", "0.31", "0.34", "0.36"],
    ]
    rows = [100 + i * 22 for i in range(len(data))]
    for r, row in enumerate(data):
        for c, cell in enumerate(row):
            page.insert_text((cols[c] + 4, rows[r] + 12), cell, fontsize=9)
    for yy in [rows[0] - 4, rows[1] - 2, rows[-1] + 12]:
        page.draw_line((100, yy), (460, yy))
    pdf = tmp_path / "typo.pdf"
    doc.save(pdf)
    doc.close()
    return pdf


def _build_ruled_table_page(tmp_path: Path) -> Path:
    """Minimal ruled table page so locate_tables() returns one box."""
    doc = fitz.open()
    page = doc.new_page()
    cols = [80, 180, 280, 380, 480]
    rows_y = [120, 150, 180]
    for y, label in zip(rows_y, ["Metric", "Revenue", "Costs"]):
        page.insert_text((cols[0] + 4, y), label, fontsize=9)
        for x, val in zip(cols[1:], ["100", "110", "120", "130"]):
            page.insert_text((x + 4, y), val, fontsize=9)
    for yy in [rows_y[0] - 4, rows_y[1] - 2, rows_y[-1] + 12]:
        page.draw_line((cols[0], yy), (520, yy))
    pdf = tmp_path / "doc.pdf"
    doc.save(pdf)
    doc.close()
    return pdf


class _StubReader:
    def __init__(self, value: str) -> None:
        self.value = value
        self.calls = 0
        self.model = "stub"

    def read(self, _image_path: Path) -> str:
        self.calls += 1
        return self.value


def _wire_reader(monkeypatch, reader: _StubReader) -> None:
    import socr.tables.extract as extract_mod

    monkeypatch.setattr(extract_mod, "OllamaTableReader", lambda *a, **k: reader)


def _raw_crop(markdown: str, source: str = "booktabs") -> list:
    return [CropTable(markdown=markdown, source=source, bbox=(0.0, 0.0, 1.0, 1.0))]


# ---------------------------------------------------------------------------
# Unit tests — crop_repair helpers
# ---------------------------------------------------------------------------


class TestCropRepairHelpers:
    def test_text_table_detected_as_needing_fallback(self) -> None:
        assert page_needs_crop_repair_fallback(_BROKEN_TEXT_TABLE_MD)

    def test_fixed_text_table_does_not_need_fallback(self) -> None:
        assert not page_needs_crop_repair_fallback(_FIXED_TEXT_TABLE_MD)

    def test_improvement_gate_accepts_header_fix(self) -> None:
        before = snapshot_structural_defects(_BROKEN_TEXT_TABLE_MD)
        after = snapshot_structural_defects(_FIXED_TEXT_TABLE_MD)
        assert before.header_collapsed
        assert not after.header_collapsed
        assert crop_patch_improves_verification(_BROKEN_TEXT_TABLE_MD, _FIXED_TEXT_TABLE_MD)

    def test_improvement_gate_rejects_unchanged_structure(self) -> None:
        assert not crop_patch_improves_verification(
            _BROKEN_PLAIN_HEADER_MD,
            _BROKEN_PLAIN_HEADER_MD.replace("Revenue", "Revenue"),  # noop
        )


# ---------------------------------------------------------------------------
# Integration — _reread_page_tables
# ---------------------------------------------------------------------------


class TestCropRepairFallbackReread:
    def test_broken_text_table_patched_when_verification_improves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf = _build_ruled_table_page(tmp_path)
        pipe = UnifiedPipeline(
            PipelineConfig(quiet=True, auto_patch_tables=False, strict_local=True)
        )
        state, bo = _state_with_table_page(pdf, _BROKEN_TEXT_TABLE_MD)
        reader = _StubReader(_FIXED_TEXT_TABLE_MD)
        extractor = TableCropExtractor(reader)

        patched_delta, flagged_delta = pipe._reread_page_tables(
            state, 1, _raw_crop(_FIXED_TEXT_TABLE_MD), extractor
        )

        assert patched_delta == 1
        assert "High | Medium | Low" in bo.text
        assert "All responses combined" not in bo.text
        assert any("crop-repair fallback" in n for n in bo.audit_notes)
        assert any(e.kind == "dualpass_patched" for e in state.events)

    def test_broken_plain_header_table_patched_when_verification_improves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf = _build_ruled_table_page(tmp_path)
        pipe = UnifiedPipeline(
            PipelineConfig(quiet=True, auto_patch_tables=False, strict_local=True)
        )
        state, bo = _state_with_table_page(pdf, _BROKEN_PLAIN_HEADER_MD)
        extractor = TableCropExtractor(_StubReader(_FIXED_PLAIN_HEADER_MD))

        patched_delta, _ = pipe._reread_page_tables(
            state, 1, _raw_crop(_FIXED_PLAIN_HEADER_MD), extractor
        )

        assert patched_delta == 1
        assert "| Jan | Feb | Mar | Apr |" in bo.text
        assert "Jan Feb Mar Apr" not in bo.text.split("| --- |")[0]

    def test_patch_rejected_when_crop_does_not_improve_verification(
        self,
        tmp_path: Path,
    ) -> None:
        pdf = _build_ruled_table_page(tmp_path)
        pipe = UnifiedPipeline(PipelineConfig(quiet=True, auto_patch_tables=False))
        state, bo = _state_with_table_page(pdf, _BROKEN_PLAIN_HEADER_MD)
        extractor = TableCropExtractor(_StubReader(_STILL_BROKEN_CROP_MD))

        patched_delta, flagged_delta = pipe._reread_page_tables(
            state, 1, _raw_crop(_STILL_BROKEN_CROP_MD), extractor
        )

        assert patched_delta == 0
        assert flagged_delta >= 1
        assert bo.text == _BROKEN_PLAIN_HEADER_MD
        assert any("crop-repair fallback declined" in n for n in bo.audit_notes)
        assert not any(e.kind == "dualpass_patched" for e in state.events)

    def test_timeout_blocks_crop_repair_fallback(self, tmp_path: Path) -> None:
        pdf = _build_ruled_table_page(tmp_path)
        pipe = UnifiedPipeline(PipelineConfig(quiet=True, auto_patch_tables=False))
        state, bo = _state_with_table_page(pdf, _BROKEN_TEXT_TABLE_MD)

        sentinel = CropTable(markdown="", source="booktabs", bbox=(0.0, 0.0, 1.0, 1.0))
        sentinel._timed_out = True  # type: ignore[attr-defined]
        good = CropTable(
            markdown=_FIXED_TEXT_TABLE_MD, source="booktabs", bbox=(0.0, 0.0, 1.0, 1.0)
        )
        extractor = TableCropExtractor(_StubReader(_FIXED_TEXT_TABLE_MD))

        pipe._reread_page_tables(state, 1, [sentinel, good], extractor)

        assert bo.text == _BROKEN_TEXT_TABLE_MD
        assert any(e.kind == "dualpass_crop_timeout" for e in state.events)

    def test_non_broken_table_stays_flag_only_without_auto_patch(
        self,
        tmp_path: Path,
    ) -> None:
        """Numeric typo only — not structural; must not trigger crop-repair fallback."""
        pdf = _build_booktabs_typo_page(tmp_path)

        pipe = UnifiedPipeline(PipelineConfig(quiet=True, auto_patch_tables=False))
        state, bo = _state_with_table_page(pdf, _PAGE_MD_TYPO)
        extractor = TableCropExtractor(_StubReader(_CROP_MD_TYPO))

        with fitz.open(pdf) as fitz_doc:
            boxes = locate_tables(fitz_doc[0])
        raw_crops = extractor.extract(pdf, 1, boxes)
        patched_delta, _ = pipe._reread_page_tables(state, 1, raw_crops, extractor)

        assert patched_delta == 0
        assert bo.text == _PAGE_MD_TYPO
        assert not any("crop-repair fallback" in n for n in bo.audit_notes)


class TestStrictLocalCropModel:
    def test_strict_local_uses_local_instruct_vlm(self) -> None:
        pipe = UnifiedPipeline(
            PipelineConfig(
                quiet=True,
                strict_local=True,
                judge_backend="heuristic",
                auto_patch_tables=False,
            )
        )
        # Even when cloud judge is "available", crop model must stay local.
        with patch.object(pipe, "_resolve_judge_model", return_value="qwen3.5:cloud"):
            assert pipe._resolve_crop_vlm_model() == PROFILE_QWEN_LOCAL.model

    def test_phase_dual_pass_wires_local_reader_under_strict_local(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdf = _build_ruled_table_page(tmp_path)
        constructed: list[str] = []

        import socr.tables.extract as extract_mod

        class _CapturingReader:
            timeout = 120.0

            def __init__(self, *, model: str, **kwargs) -> None:
                constructed.append(model)

            def read(self, _p: Path) -> str:
                return _FIXED_PLAIN_HEADER_MD

        monkeypatch.setattr(extract_mod, "OllamaTableReader", _CapturingReader)
        monkeypatch.setattr(
            UnifiedPipeline,
            "_resolve_judge_model",
            lambda self: "qwen3.5:cloud",
        )

        pipe = UnifiedPipeline(
            PipelineConfig(
                quiet=True,
                strict_local=True,
                auto_patch_tables=False,
                judge_backend="heuristic",
            )
        )
        pipe._last_assessment = _assessment()
        state, _bo = _state_with_table_page(pdf, _BROKEN_PLAIN_HEADER_MD)
        pipe._phase_dual_pass_tables(state)

        assert constructed == [PROFILE_QWEN_LOCAL.model]
