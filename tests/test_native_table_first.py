"""Native grid is the first reader of a born-digital table page.

An exact pass of the existing deterministic checks ships the structured
native table and does not call whole-page ``route_page``. A multiset
mismatch sends ``_transcribe_cell_token`` only the failing cells. A cell
that cannot be resolved keeps the D3 failure marker instead of the bad
number. Lane-count ambiguity with a clean value guard is not a named cell
and stays on the existing whole-page route.
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
    CELLS,
    DEFER,
    REFUSE,
    SHIP,
    plan_native_table,
    splice_cell_tokens,
    transcription_matches_native,
)


def _dense_forecast_pdf(path: Path) -> None:
    """The PP-6 multi-column grid. Measured to ``has_tables`` and EXACT_PASS."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (72, 50),
        "Table 1. GDP growth forecasts across baseline and shock scenarios.",
        fontsize=10,
        fontname="helv",
    )
    col_xs = [90.0, 180.0, 270.0, 360.0, 450.0]
    headers = ["Variable", "b", "s", "h", "q"]
    for ci, hdr in enumerate(headers):
        page.insert_text((col_xs[ci], 80), hdr, fontsize=9, fontname="helv")
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
            page.insert_text((col_xs[ci], 100 + ri * 22), cell, fontsize=9, fontname="helv")
    doc.save(str(path))
    doc.close()


def _page_words_and_text(pdf_path: Path) -> tuple[list, str]:
    assessment = BornDigitalDetector().detect(pdf_path).pages[0]
    assert assessment.has_tables
    doc = fitz.open(pdf_path)
    try:
        words = doc[0].get_text("words")
    finally:
        doc.close()
    return words, assessment.native_text or ""


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
        # The table-judge ladder stays default-on in production. These tests
        # pin it off so a missing cloud rung cannot rewrite a routing
        # assertion into an UNVERIFIED document. The ladder's own call site
        # is not what this change moves.
        table_judge_ladder=False,
    )


def _run(pdf_path: Path, tmp_path: Path, *, transcribe=None, native_text=None):
    pipeline = UnifiedPipeline(_config())
    if native_text is not None:
        real = BornDigitalDetector()

        class _Rewrite(BornDigitalDetector):
            def detect(self, path):
                assessment = real.detect(path)
                assessment.pages[0].native_text = native_text
                return assessment

        pipeline.bd_detector = _Rewrite()
    route_calls: list[int] = []

    def _route(*_args, **_kwargs):
        route_calls.append(1)
        raise AssertionError("whole-page route_page must not run for this page")

    transcribe_calls: list[tuple] = []

    def _transcribe(crop_path: Path):
        transcribe_calls.append((crop_path,))
        if transcribe is None:
            raise AssertionError("cell transcriber must not run on an exact pass")
        return transcribe(crop_path)

    with (
        patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_transcribe_cell_token", side_effect=_transcribe),
    ):
        result = pipeline.process(pdf_path, tmp_path / "out")
    return result, route_calls, transcribe_calls


class TestPlanNativeTable:
    def test_exact_pass_ships_without_cells(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "forecast.pdf"
        _dense_forecast_pdf(pdf_path)
        words, text = _page_words_and_text(pdf_path)
        plan = plan_native_table(words, text)
        assert plan.action == SHIP
        assert plan.cells == ()

    def test_one_wrong_cell_is_named(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "forecast.pdf"
        _dense_forecast_pdf(pdf_path)
        words, text = _page_words_and_text(pdf_path)
        bad = text.replace("0.253", "9.999", 1)
        plan = plan_native_table(words, bad)
        assert plan.action == CELLS
        assert len(plan.cells) == 1
        cell = plan.cells[0]
        assert cell.grid_token == "9.999"
        assert cell.native_token == "0.253"
        assert cell.bbox[2] > cell.bbox[0]
        assert cell.bbox[3] > cell.bbox[1]

    def test_structure_defect_refuses_even_when_numbers_match(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "forecast.pdf"
        _dense_forecast_pdf(pdf_path)
        words, text = _page_words_and_text(pdf_path)
        plan = plan_native_table(words, text, structure_defective=True)
        assert plan.action == REFUSE
        assert plan.cells == ()

    def test_numeric_orphan_refuses(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "forecast.pdf"
        _dense_forecast_pdf(pdf_path)
        words, text = _page_words_and_text(pdf_path)
        plan = plan_native_table(words, text, orphan_words=["0.999"])
        assert plan.action == REFUSE

    def test_no_words_defers_to_the_existing_route(self) -> None:
        markdown = "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
        plan = plan_native_table([], markdown)
        assert plan.action == DEFER
        assert plan.cells == ()

    def test_splice_replaces_only_the_named_cell(self) -> None:
        markdown = "| GDP | 9.999 | 0.179 |\n| --- | --- | --- |\n| CPI | 0.144 | 0.135 |\n"
        # The data row is the line the planner saw. Header is not a data row;
        # this fixture only checks the splice, so the separator is enough to
        # look like a row the caller already identified.
        row = "| GDP | 9.999 | 0.179 |"
        spliced = splice_cell_tokens(markdown, [(row, 1, "9.999", "0.253")])
        assert spliced is not None
        assert "9.999" not in spliced
        assert "0.253" in spliced
        assert "0.179" in spliced
        assert "0.144" in spliced

    def test_transcription_must_match_the_text_layer_not_the_grid(self) -> None:
        assert transcription_matches_native("0.253", "0.253", "9.999")
        assert not transcription_matches_native("9.999", "0.253", "9.999")
        assert not transcription_matches_native("1.000", "0.253", "9.999")
        assert not transcription_matches_native("  ", "0.253", "9.999")


class TestAgenticNativeTableFirst:
    def test_exact_pass_ships_native_grid_and_skips_route_page(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "forecast.pdf"
        _dense_forecast_pdf(pdf_path)
        result, route_calls, transcribe_calls = _run(pdf_path, tmp_path)
        assert route_calls == []
        assert transcribe_calls == []
        assert result.status == DocumentStatus.SUCCESS
        assert "0.253" in (result.markdown or "")
        assert "9.999" not in (result.markdown or "")
        assert "| GDP |" in (result.markdown or "")

    def test_failing_cell_is_transcribed_and_not_the_whole_page(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "forecast.pdf"
        _dense_forecast_pdf(pdf_path)
        _words, text = _page_words_and_text(pdf_path)
        bad = text.replace("0.253", "9.999", 1)

        def _ok(_crop: Path) -> str:
            return "0.253"

        result, route_calls, transcribe_calls = _run(
            pdf_path, tmp_path, transcribe=_ok, native_text=bad
        )
        assert route_calls == []
        assert len(transcribe_calls) == 1
        assert result.status == DocumentStatus.SUCCESS
        body = result.markdown or ""
        assert "0.253" in body
        assert "9.999" not in body

    def test_unresolved_cell_ships_the_failure_marker(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "forecast.pdf"
        _dense_forecast_pdf(pdf_path)
        _words, text = _page_words_and_text(pdf_path)
        bad = text.replace("0.253", "9.999", 1)

        def _none(_crop: Path) -> None:
            return None

        result, route_calls, transcribe_calls = _run(
            pdf_path, tmp_path, transcribe=_none, native_text=bad
        )
        assert route_calls == []
        assert len(transcribe_calls) == 1
        # A document whose only page is a failure marker has an empty canonical
        # body: the marker is honesty, not content. It ships on the page fragment,
        # which is the same surface a multi-page D3 floor uses.
        fragment = next((tmp_path / "out").rglob("pages/00001.md")).read_text(encoding="utf-8")
        assert "unverifiable table" in fragment
        assert "9.999" not in fragment
        assert "9.999" not in (result.markdown or "")
        sidecar = json.loads(
            next((tmp_path / "out").rglob("pages/00001.json")).read_text(encoding="utf-8")
        )
        assert sidecar["status"] == "error"
        assert sidecar["native_table_structure_failed"] is True
        assert sidecar["native_table_unverifiable"] is True
        assert result.status != DocumentStatus.SUCCESS
        assert result.error

    def test_two_runs_are_byte_identical(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "forecast.pdf"
        _dense_forecast_pdf(pdf_path)
        first, _, _ = _run(pdf_path, tmp_path / "a")
        second, _, _ = _run(pdf_path, tmp_path / "b")
        assert first.markdown == second.markdown
        assert first.markdown
