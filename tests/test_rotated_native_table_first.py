"""Upright retry for GH-147 refused born-digital table pages.

After analyze refuses the sideways rowizer output, agentic native-table-first
re-rowizes upright and runs ``plan_native_table``. Only an exact pass (SHIP)
ships the grid without ``route_page``. Any other outcome — no grid, read error,
REFUSE, DEFER, or CELLS — leaves the page on the existing whole-page route.
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
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import PageDecision, ProviderAttempt
from socr.tables.native_first import (
    REFUSE,
    SHIP,
    RotatedNativeTableAttempt,
    NativeTablePlan,
    attempt_rotated_native_table,
)


def _place(u: float, v: float, rotation: int, width: float, height: float) -> tuple[float, float]:
    """Map an upright-frame point (u right, v down) onto a page drawn at *rotation*.

    GH-902: fitz ``rotate=90`` text reads bottom-to-top, so the upright top edge
    lands on the LEFT of the page and the upright left edge at the BOTTOM;
    ``rotate=270`` is the mirror image. The earlier fixture laid the grid out
    180 degrees off this mapping, which is exactly what the wrong rowizer sign
    undid, so the fixture and the bug agreed and nothing failed.
    """
    if rotation == 0:
        return u, v
    if rotation == 90:
        return v, height - u
    if rotation == 270:
        return width - v, u
    raise ValueError(rotation)


_FORECAST_HEADERS = ["Variable", "b", "s", "h", "q"]
_FORECAST_ROWS = [
    ["GDP", "0.253", "0.179", "0.211", "0.301"],
    ["CPI", "0.144", "0.135", "0.290", "0.188"],
    ["IP", "0.041", "0.050", "0.154", "0.099"],
    ["UR", "0.082", "0.321", "0.144", "0.211"],
    ["CB", "0.180", "0.171", "0.365", "0.244"],
    ["TR", "0.310", "0.220", "0.410", "0.188"],
]


def _forecast_pdf(path: Path, rotation: int = 90) -> None:
    """The PP-6 grid with a ruled table signal, drawn at *rotation* (0, 90, 270)."""
    doc = fitz.open()
    width, height = 612, 792
    page = doc.new_page(width=width, height=height)

    def text(u: float, v: float, s: str, size: float) -> None:
        page.insert_text(
            _place(u, v, rotation, width, height),
            s,
            fontsize=size,
            fontname="helv",
            rotate=rotation,
        )

    text(72, 50, "Table 1. GDP growth forecasts across baseline and shock scenarios.", 10)
    text(72, 400, "* Forecasts are annualized percent changes.", 9)
    col_xs = [90.0, 180.0, 270.0, 360.0, 450.0]
    for ci, hdr in enumerate(_FORECAST_HEADERS):
        text(col_xs[ci], 80, hdr, 9)
    for ri, row in enumerate(_FORECAST_ROWS):
        for ci, cell in enumerate(row):
            text(col_xs[ci], 100 + ri * 22, cell, 9)
    x0, y0, tw, th = 70, 70, 400, 180
    for r in range(9):
        page.draw_line(
            _place(x0, y0 + r * 20, rotation, width, height),
            _place(x0 + tw, y0 + r * 20, rotation, width, height),
        )
    for c in range(6):
        page.draw_line(
            _place(x0 + c * 70, y0, rotation, width, height),
            _place(x0 + c * 70, y0 + th, rotation, width, height),
        )
    doc.save(str(path))
    doc.close()


def _rotated_dense_forecast_pdf(path: Path) -> None:
    _forecast_pdf(path, 90)


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
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            result = pipeline.process(pdf_path, tmp_path / "out")
        assert route_calls == []
        assert result.status == DocumentStatus.SUCCESS
        body = result.markdown or ""
        assert "0.253" in body
        assert "Table 1." in body
        assert "GDP growth forecasts across baseline and shock scenarios." in body or (
            "Table 1. GD" in body
        )
        assert "Forecasts are annualized percent changes." in body
        sidecar = json.loads(
            next((tmp_path / "out").rglob("pages/00001.json")).read_text(encoding="utf-8")
        )
        kinds = [ev["kind"] for ev in sidecar["audit_events"]]
        assert "landscape_page_refused" in kinds
        assert "native_table_exact_pass" in kinds

    def test_upright_failed_check_calls_route_page(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "rotated.pdf"
        _rotated_dense_forecast_pdf(pdf_path)
        pipeline = UnifiedPipeline(_config())
        route_calls: list[int] = []

        def _route(page_num, ladder, run_provider, judge, **kwargs):
            route_calls.append(page_num)
            rejected = PageOutput(
                page_num=page_num,
                text="model table",
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            )
            prof = ladder[0]
            att = ProviderAttempt(
                engine=prof.engine,
                output=rejected,
                cost_usd=0.0,
                accepted=True,
                reason="test",
                provider_id=prof.id,
                model=prof.model,
                backend=prof.backend,
            )
            return PageDecision(
                page_num=page_num, final_output=rejected, attempts=[att], accepted=True
            )

        refuse_attempt = RotatedNativeTableAttempt(
            plan=NativeTablePlan(REFUSE, reason="row_count"),
            markdown="| a | b |\n| --- | --- |\n| 1 | 2 |",
            words=[],
            structure_defective=False,
            header_unattributed=False,
        )

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch(
                "socr.tables.native_first.attempt_rotated_native_table",
                return_value=refuse_attempt,
            ),
        ):
            result = pipeline.process(pdf_path, tmp_path / "out")
        assert route_calls == [1]
        sidecar = json.loads(
            next((tmp_path / "out").rglob("pages/00001.json")).read_text(encoding="utf-8")
        )
        kinds = [ev["kind"] for ev in sidecar["audit_events"]]
        # The event lives in the sidecar's audit_events, not in the markdown:
        # a REFUSE plan must not record an exact-pass claim.
        assert "native_table_exact_pass" not in kinds
        assert "native_table_cell_unresolved" not in kinds

    @pytest.mark.skipif(
        not Path(
            "/cursor/stores/bc-f87a255a-cbe0-40f3-b811-f505faf8233d/media/"
            "fed-minutes-example/fomcminutes20190619.pdf"
        ).is_file(),
        reason="FOMC fixture not present in the project store",
    )
    def test_fomc_page_14_reaches_route_page(self, tmp_path: Path) -> None:
        pdf = Path(
            "/cursor/stores/bc-f87a255a-cbe0-40f3-b811-f505faf8233d/media/"
            "fed-minutes-example/fomcminutes20190619.pdf"
        )
        single = tmp_path / "fomc-p14.pdf"
        src = fitz.open(pdf)
        doc = fitz.open()
        doc.insert_pdf(src, from_page=13, to_page=13)
        doc.save(single)
        doc.close()
        src.close()

        pipeline = UnifiedPipeline(_config())
        route_calls: list[int] = []

        def _route(page_num, ladder, run_provider, judge, **kwargs):
            route_calls.append(page_num)
            rejected = PageOutput(
                page_num=page_num,
                text="",
                status=PageStatus.ERROR,
                engine="qwen",
                audit_passed=False,
            )
            prof = ladder[0]
            att = ProviderAttempt(
                engine=prof.engine,
                output=rejected,
                cost_usd=0.0,
                accepted=False,
                reason="test",
                provider_id=prof.id,
                model=prof.model,
                backend=prof.backend,
            )
            return PageDecision(
                page_num=page_num, final_output=rejected, attempts=[att], accepted=False
            )

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            pipeline.process(single, tmp_path / "out")
        assert route_calls == [1]


# ---------------------------------------------------------------------------
# GH-902: the upright correction must be applied with the right sign
# ---------------------------------------------------------------------------

_LABEL_ORDER = [_FORECAST_HEADERS[0]] + [r[0] for r in _FORECAST_ROWS]


def _first_column(markdown: str) -> list[str]:
    cells = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("| ---"):
            continue
        # The caption's words share the first column in this fixture, so the label
        # is the leftmost non-empty cell; the column holding it is checked below.
        row = [c.strip() for c in stripped.strip("|").split("|")]
        cells.append(next((c for c in row if c), ""))
    # Caption / footnote rows may share the grid; the labels' ORDER is the pin.
    return [c for c in cells if c in _LABEL_ORDER]


def _rowized_markdown(pdf_path: Path, *, rotation_sign: int = 1) -> str:
    from socr.core.born_digital import upright_rotation_for
    from socr.tables.reconstruct import rowize_from_word_list

    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        regions = rowize_from_word_list(
            page.get_text("words"),
            rotation=rotation_sign * upright_rotation_for(page),
            page_rect=page.rect,
        )
    finally:
        doc.close()
    return "\n\n".join(md for _rect, md in regions)


@pytest.mark.parametrize("rotation", [90, 270])
class TestRotationSign:
    def test_rowized_grid_equals_the_upright_twin_in_reading_order(
        self, tmp_path: Path, rotation: int
    ) -> None:
        upright_pdf = tmp_path / "upright.pdf"
        rotated_pdf = tmp_path / "rotated.pdf"
        _forecast_pdf(upright_pdf, 0)
        _forecast_pdf(rotated_pdf, rotation)

        fixed = _rowized_markdown(rotated_pdf)
        upright = _rowized_markdown(upright_pdf)
        assert upright.strip(), "the upright twin must rowize, or the pin is vacuous"
        assert _first_column(upright) == _LABEL_ORDER
        assert _first_column(fixed) == _LABEL_ORDER
        assert fixed == upright
        # Difference pin: the previous sign reads the same page 180 degrees flipped.
        old_sign = _rowized_markdown(rotated_pdf, rotation_sign=-1)
        assert old_sign != upright
        assert _first_column(old_sign) != _LABEL_ORDER

    def test_attempt_matches_the_upright_twin_and_ships_in_reading_order(
        self, tmp_path: Path, rotation: int
    ) -> None:
        from socr.tables.reconstruct import rowize_from_words

        upright_pdf = tmp_path / "upright.pdf"
        rotated_pdf = tmp_path / "rotated.pdf"
        _forecast_pdf(upright_pdf, 0)
        _forecast_pdf(rotated_pdf, rotation)

        doc = fitz.open(upright_pdf)
        upright_md = "\n\n".join(md for _r, md in rowize_from_words(doc[0]) if md.strip())
        doc.close()
        doc = fitz.open(rotated_pdf)
        attempt = attempt_rotated_native_table(doc[0])
        doc.close()

        assert attempt is not None
        assert attempt.markdown == upright_md
        assert attempt.plan.action == SHIP
        assert _first_column(attempt.markdown) == _LABEL_ORDER

    def test_the_witness_words_read_in_the_upright_twins_order(
        self, tmp_path: Path, rotation: int
    ) -> None:
        """The comparison words must be upright on their own, not just agree with the grid.

        ``plan_native_table`` cannot catch a flipped witness: its verifier pairs rows
        by number multiset, so a grid and witness flipped TOGETHER (or a grid flipped
        against a correct witness) both still exact-pass (measured: reversing rows,
        cells, or both leaves SHIP). The witness's own geometry is therefore the only
        place the sign is observable, so it is pinned directly against the upright twin.
        """
        from socr.tables.native_first import upright_words_for_page

        upright_pdf = tmp_path / "upright.pdf"
        rotated_pdf = tmp_path / "rotated.pdf"
        _forecast_pdf(upright_pdf, 0)
        _forecast_pdf(rotated_pdf, rotation)

        def reading_order(pdf: Path) -> list[str]:
            doc = fitz.open(pdf)
            try:
                words, _ = upright_words_for_page(doc[0])
            finally:
                doc.close()
            return [w[4] for w in sorted(words, key=lambda w: (round(w[1]), w[0]))]

        expected = reading_order(upright_pdf)
        assert expected, "the upright twin must have words, or the pin is vacuous"
        assert reading_order(rotated_pdf) == expected

    def test_region_rect_encloses_the_table_in_page_coordinates(
        self, tmp_path: Path, rotation: int
    ) -> None:
        """The output rect is rotated back to the page frame with the opposite sign."""
        from socr.tables.reconstruct import rowize_from_words

        rotated_pdf = tmp_path / "rotated.pdf"
        _forecast_pdf(rotated_pdf, rotation)
        doc = fitz.open(rotated_pdf)
        try:
            page = doc[0]
            regions = rowize_from_words(page)
            body = {c for row in _FORECAST_ROWS for c in row}
            cells = [w for w in page.get_text("words") if w[4] in body]
        finally:
            doc.close()
        assert regions and cells
        rect = regions[0][0]
        for x0, y0, x1, y1, *_ in cells:
            assert rect.x0 - 1 <= x0 and x1 <= rect.x1 + 1
            assert rect.y0 - 1 <= y0 and y1 <= rect.y1 + 1
