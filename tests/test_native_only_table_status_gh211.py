"""GH-211: --native-only surfaces unverifiable native tables without rerouting."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode, PageStatus
from socr.pipeline.orchestrator import UnifiedPipeline


class _FixedDetector:
    def __init__(self, assessment: DocumentAssessment) -> None:
        self._assessment = assessment

    def detect(self, path: Path) -> DocumentAssessment:
        assert path == self._assessment.path
        return self._assessment


def _native_table_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 70), "Estimate Standard error", fontsize=9)
    for row in range(10):
        y = 90 + row * 16
        page.insert_text((60, y), f"Coefficient {row}", fontsize=9)
        page.insert_text((260, y), f"{row}.12", fontsize=9)
    page.draw_line(fitz.Point(50, 60), fitz.Point(360, 60))
    page.draw_line(fitz.Point(50, 260), fitz.Point(360, 260))
    doc.save(path)
    doc.close()


def _run_native_only(tmp_path: Path, *, agentic: bool, unverifiable: bool):
    pdf_path = tmp_path / "paper.pdf"
    _native_table_pdf(pdf_path)
    native_text = "| Coefficient | Estimate |\n| --- | --- |\n| A | 0.12 |"
    assessment = DocumentAssessment(
        path=pdf_path,
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text=native_text,
                confidence=1.0,
                has_tables=True,
                has_unverifiable_table_region=unverifiable,
            )
        ],
    )
    pipeline = UnifiedPipeline(
        PipelineConfig(
            agentic=agentic,
            native_first=True,
            native_only=True,
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            tiered=False,
            dual_pass_tables=False,
            detect_equations=False,
            save_figures=False,
            write_manifest=True,
            quiet=False,
        )
    )
    pipeline.bd_detector = _FixedDetector(assessment)
    pipeline._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
    pipeline._surface_table_scoring = lambda *args, **kwargs: None

    def _unexpected_ocr(*args, **kwargs):
        raise AssertionError("--native-only table page must not enter the OCR ladder")

    pipeline._run_engine_on_pages = _unexpected_ocr
    captured = {}
    assemble = pipeline._phase_assemble

    def _capture_state(state, output_dir):
        captured["state"] = state
        return assemble(state, output_dir)

    pipeline._phase_assemble = _capture_state
    output_dir = tmp_path / "out"
    result = pipeline.process(pdf_path, output_dir)
    return result, captured["state"], output_dir, native_text


def _only(path: Path, pattern: str) -> Path:
    matches = list(path.rglob(pattern))
    assert len(matches) == 1, matches
    return matches[0]


@pytest.mark.parametrize("agentic", [False, True])
def test_native_only_unverifiable_table_is_flagged_without_rerouting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], agentic: bool
) -> None:
    capsys.readouterr()
    result, state, output_dir, native_text = _run_native_only(
        tmp_path, agentic=agentic, unverifiable=True
    )

    page = state.pages[1]
    assert len(page.attempts) == 1
    assert page.attempts[0].text == native_text
    assert page.attempts[0].status is PageStatus.WARNING
    assert page.attempts[0].audit_passed is False
    assert page.attempts[0].failure_mode is FailureMode.NATIVE_TABLE_STRUCTURE_FAILED
    assert result.status is DocumentStatus.AUDIT_FAILED
    assert result.cost == 0.0

    events = [event for event in state.events if event.kind == "table_structure_failed"]
    assert len(events) == 1
    assert events[0].page_num == 1

    # GH-211 MAJOR-2: OCR was NEVER attempted on this page (--native-only
    # disables the ladder). The generic "native_fallback" event lies here
    # ("OCR tried and never passed"); it must not be recorded for a page
    # whose only defect is native-table distrust under --native-only.
    fallback_events = [event for event in state.events if event.kind == "native_fallback"]
    assert fallback_events == [], (
        "native_fallback claims OCR was tried and failed, but --native-only "
        "never runs the OCR ladder -- this page was never attempted"
    )
    distrust_events = [
        event for event in state.events if event.kind == "native_only_table_distrusted"
    ]
    assert len(distrust_events) == 1
    assert distrust_events[0].page_num == 1
    assert "OCR never attempted" in distrust_events[0].detail

    sidecar = json.loads(_only(output_dir, "pages/00001.json").read_text(encoding="utf-8"))
    assert sidecar["native_table_unverifiable"] is True
    assert sidecar["status"] == PageStatus.WARNING.value
    assert sidecar["winning_output"]["audit_passed"] is False
    assert (
        sidecar["winning_output"]["failure_mode"] == FailureMode.NATIVE_TABLE_STRUCTURE_FAILED.value
    )

    audit = json.loads(_only(output_dir, "audit_log.json").read_text(encoding="utf-8"))
    assert audit["counts"]["table_structure_failed"] == 1

    trust = json.loads(_only(output_dir, "tables_trust.json").read_text(encoding="utf-8"))
    assert trust["untrusted_pages"] == [1]
    # GH-205 added an unconditional ``table_region_geometry_hard_fail`` detection
    # event for every TR-3 geometry hard-fail, so this page now carries a SECOND,
    # true reason alongside the demotion event. Nothing here was relaxed: the page
    # is still the only untrusted one, still WARNING, still audit_passed=False, and
    # ``table_structure_failed`` is still present. Only the exhaustive list grew.
    assert trust["pages"]["1"]["reasons"] == [
        "table_region_geometry_hard_fail",
        "table_structure_failed",
    ]

    doc_metadata = json.loads(
        _only(output_dir / "paper", "metadata.json").read_text(encoding="utf-8")
    )
    assert doc_metadata["status"] == "partial"
    assert "untrusted tables" in doc_metadata["error"]

    cli_output = capsys.readouterr().out
    assert "audit_failed" in cli_output
    assert "untrusted tables" in cli_output


@pytest.mark.parametrize("agentic", [False, True])
def test_native_only_verified_table_remains_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], agentic: bool
) -> None:
    capsys.readouterr()
    result, state, output_dir, native_text = _run_native_only(
        tmp_path, agentic=agentic, unverifiable=False
    )

    page = state.pages[1]
    assert len(page.attempts) == 1
    assert page.attempts[0].text == native_text
    assert page.attempts[0].status is PageStatus.SUCCESS
    assert page.attempts[0].audit_passed is True
    assert page.attempts[0].failure_mode is FailureMode.NONE
    assert result.status is DocumentStatus.SUCCESS
    assert result.cost == 0.0
    assert not [event for event in state.events if event.kind == "table_structure_failed"]
    assert not list(output_dir.rglob("tables_trust.json"))

    sidecar = json.loads(_only(output_dir, "pages/00001.json").read_text(encoding="utf-8"))
    assert sidecar["native_table_unverifiable"] is False
    assert sidecar["status"] == PageStatus.SUCCESS.value
    assert sidecar["winning_output"]["audit_passed"] is True

    cli_output = capsys.readouterr().out
    assert "audit_failed" not in cli_output
    assert "untrusted tables" not in cli_output
