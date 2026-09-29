"""Upright retry for GH-147 refused born-digital table pages.

After analyze refuses the sideways rowizer output, agentic native-table-first
re-rowizes upright and runs ``plan_native_table``. An exact pass ships the
grid without ``route_page``. A fail-closed refusal keeps the D3 floor without
a whole-page read. ``DEFER`` alone falls through to the existing OCR route.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from socr.core.born_digital import BornDigitalDetector
from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.native_first import (
    REFUSE,
    SHIP,
    NativeTableFirstWork,
    NativeTablePlan,
    attempt_rotated_native_table,
)


def _rotated_dense_forecast_pdf(path: Path) -> None:
    """The PP-6 grid drawn at 90 degrees with a ruled table signal."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (72, 50),
        "Table 1. GDP growth forecasts across baseline and shock scenarios.",
        fontsize=10,
        fontname="helv",
        rotate=90,
    )
    col_xs = [90.0, 180.0, 270.0, 360.0, 450.0]
    headers = ["Variable", "b", "s", "h", "q"]
    for ci, hdr in enumerate(headers):
        page.insert_text((col_xs[ci], 80), hdr, fontsize=9, fontname="helv", rotate=90)
    rows = [
        ["GDP", "0.253", "0.179", "0.211", "0.301"],
        ["CPI", "0.144", "0.135", "0.290", "0.188"],
        ["IP", "0.041", "0.050", "0.154", "0.099"],
        ["UR", "0.082", "0.321", "0.144", "0.211"],
        ["CB", "0.180", "0.171", "0.365", "0.244"],
        ["TR", "0.310", "0.220", "0.410", "0.188"],
    ]
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            page.insert_text(
                (col_xs[ci], 100 + ri * 22), cell, fontsize=9, fontname="helv", rotate=90
            )
    x0, y0, width, height = 70, 70, 400, 180
    for r in range(9):
        page.draw_line((x0, y0 + r * 20), (x0 + width, y0 + r * 20))
    for c in range(6):
        page.draw_line((x0 + c * 70, y0), (x0 + c * 70, y0 + height))
    doc.save(str(path))
    doc.close()


def _config() -> PipelineConfig:
    return PipelineConfig(
        agentic=True,
        native_first=True,
        native_only=False,
        primary_engine=EngineType.QWEN,
        enabled_engines=[EngineType.QWEN],
        tiered=False,
        dual_pass_tables=False,
        detect_equations=False,
        save_figures=False,
        quiet=True,
        table_judge_ladder=False,
    )


class TestAttemptRotatedNativeTable:
    def test_upright_rowizer_exact_passes(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "rotated.pdf"
        _rotated_dense_forecast_pdf(pdf_path)
        assessment = BornDigitalDetector().detect(pdf_path).pages[0]
        assert assessment.native_table_lane_refused is True
        doc = fitz.open(pdf_path)
        attempt = attempt_rotated_native_table(doc[0])
        doc.close()
        assert attempt is not None
        assert attempt.plan.action == SHIP
        assert "0.253" in attempt.markdown

    @pytest.mark.skipif(
        not Path(
            "/cursor/stores/bc-f87a255a-cbe0-40f3-b811-f505faf8233d/media/"
            "fed-minutes-example/fomcminutes20190619.pdf"
        ).is_file(),
        reason="FOMC fixture not present in the project store",
    )
    def test_fomc_page_14_does_not_exact_pass_upright(self) -> None:
        """Real rotated table: upright arm must not invent a shippable grid."""
        pdf = Path(
            "/cursor/stores/bc-f87a255a-cbe0-40f3-b811-f505faf8233d/media/"
            "fed-minutes-example/fomcminutes20190619.pdf"
        )
        doc = fitz.open(pdf)
        attempt = attempt_rotated_native_table(doc[13])
        doc.close()
        assert attempt is not None
        assert attempt.plan.action != SHIP


class TestAgenticRotatedNativeTableFirst:
    def test_upright_exact_pass_skips_route_page(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "rotated.pdf"
        _rotated_dense_forecast_pdf(pdf_path)
        pipeline = UnifiedPipeline(_config())
        route_calls: list[int] = []

        def _route(*_args, **_kwargs):
            route_calls.append(1)
            raise AssertionError("whole-page route_page must not run after upright exact pass")

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
        ):
            result = pipeline.process(pdf_path, tmp_path / "out")
        assert route_calls == []
        assert result.status == DocumentStatus.SUCCESS
        assert "0.253" in (result.markdown or "")
        sidecar = json.loads(
            next((tmp_path / "out").rglob("pages/00001.json")).read_text(encoding="utf-8")
        )
        kinds = [ev["kind"] for ev in sidecar["audit_events"]]
        assert "landscape_page_refused" in kinds
        assert "native_table_exact_pass" in kinds

    def test_upright_refuse_keeps_floor_without_route_page(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "rotated.pdf"
        _rotated_dense_forecast_pdf(pdf_path)
        pipeline = UnifiedPipeline(_config())
        route_calls: list[int] = []

        def _route(*_args, **_kwargs):
            route_calls.append(1)
            raise AssertionError("fail-closed refusal must not call route_page")

        refuse_work = NativeTableFirstWork(
            NativeTablePlan(REFUSE, reason="upright_row_count"),
        )

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_plan_native_table_first", return_value=refuse_work),
        ):
            result = pipeline.process(pdf_path, tmp_path / "out")
        assert route_calls == []
        fragment = next((tmp_path / "out").rglob("pages/00001.md")).read_text(encoding="utf-8")
        assert "unverifiable table" in fragment
        assert result.status != DocumentStatus.SUCCESS
