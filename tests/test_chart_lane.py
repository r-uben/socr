"""PP-7 chart-asset routing lane tests.

Covers:
- has_chart_marks(): cluster-first vector detection, raster fallback, logging.
- _is_chart_asset_page(): predicate fires on chart pages, not on tables/clean prose.
- Agentic-loop routing: chart lane produces native-prose + PNG ref + audit event.
- Force-PNG: PNG saved even when --save-figures is off; render failure is fail-closed.
- Non-regression: tables still route to ladder; clean prose still ships native.
- Fixture variety: monochrome academic-line-plot (documented expected behaviour).
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")


# ---------------------------------------------------------------------------
# PDF fixtures (pure PyMuPDF — no embedded rasters unless stated)
# ---------------------------------------------------------------------------


def _make_vector_chart_pdf(tmp_path: Path) -> Path:
    """A pure-vector chart page: coloured fills + thick line strokes, no embedded images.

    The page carries a small prose line (native text) plus vector chart shapes.
    This is the canonical CE-dashboard fixture for the chart-asset lane.
    Zero embedded raster images — get_images() must return an empty list so the
    test exercises the vector-detection path, not the raster fast-path.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    # Native text (sparse — this is what would become word-salad without the lane).
    page.insert_text((72, 40), "GDP Growth Forecast 2025", fontsize=14)

    # Coloured bar-chart bars (filled rectangles — qualify as data marks).
    red = (0.9, 0.1, 0.1)
    blue = (0.1, 0.1, 0.9)
    green = (0.1, 0.8, 0.1)
    for i, (col, x) in enumerate(zip([red, blue, green, red, blue], [100, 180, 260, 340, 420])):
        page.draw_rect(
            fitz.Rect(x, 500 - i * 40, x + 60, 680),
            color=col,
            fill=col,
            width=1,
        )

    # A series line (thick, coloured).
    page.draw_line(
        fitz.Point(100, 300),
        fitz.Point(480, 300),
        color=(0.8, 0.2, 0.0),
        width=3,
    )

    pdf_path = tmp_path / "vector_chart.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_decorated_prose_pdf(tmp_path: Path) -> Path:
    """A prose page with a few decorative horizontal rules (thin, neutral/black).

    Must NOT trigger the chart lane — scattered thin rules are not a chart cluster.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    # Long prose text.
    for i in range(20):
        page.insert_text(
            (72, 80 + i * 22),
            "This is a regular paragraph of born-digital prose text. " * 3,
            fontsize=10,
        )

    # Three thin black decorative horizontal rules — scattered, small area.
    for y in [150, 400, 650]:
        page.draw_line(
            fitz.Point(72, y),
            fitz.Point(540, y),
            color=(0, 0, 0),
            width=0.5,
        )

    pdf_path = tmp_path / "decorated_prose.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_raster_chart_pdf(tmp_path: Path) -> Path:
    """A page with an embedded raster image (PNG chart) — zero vector drawings."""
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (400, 300), color=(30, 100, 200))
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(100, 200, 500, 600), stream=img_bytes)

    pdf_path = tmp_path / "raster_chart.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_prose_only_pdf(tmp_path: Path) -> Path:
    """A clean prose page with dense multi-word text, no drawings, no images."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(40):
        page.insert_text(
            (72, 72 + i * 18),
            f"Line {i + 1}: Lorem ipsum dolor sit amet consectetur adipiscing elit. "
            "Sed ut perspiciatis unde omnis iste natus error.",
            fontsize=10,
        )
    pdf_path = tmp_path / "prose_only.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_monochrome_lineplot_pdf(tmp_path: Path) -> Path:
    """A B&W academic-style line-plot: thin black/gray strokes only, no coloured fills.

    This is a known edge case: monochrome line plots do NOT carry coloured fills or
    thick non-neutral strokes, so _has_vector_data_marks() returns False and
    has_chart_marks() returns False (vector path).  The page has no embedded raster
    either, so the raster fast-path also misses.

    Documented expected behaviour: has_chart_marks() returns False for this fixture.
    This is an accepted false-negative — monochrome academic plots without colour
    are indistinguishable from plain ruling at the deterministic level.  A future
    VLM pass could catch them, but that is out of scope for PP-7.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text((72, 40), "Figure 1: Output Gap Dynamics", fontsize=11)

    # Axes (thin, neutral black).
    black = (0, 0, 0)
    page.draw_line(fitz.Point(100, 600), fitz.Point(100, 200), color=black, width=0.75)
    page.draw_line(fitz.Point(100, 600), fitz.Point(500, 600), color=black, width=0.75)

    # Series lines (thin, gray/dark — neutral colors, thin strokes).
    gray = (0.3, 0.3, 0.3)
    dkgray = (0.15, 0.15, 0.15)
    points_a = [(100 + i * 40, 400 - i * 15) for i in range(11)]
    points_b = [(100 + i * 40, 450 - i * 8) for i in range(11)]
    for pts, col in [(points_a, black), (points_b, gray), (points_b, dkgray)]:
        for k in range(len(pts) - 1):
            page.draw_line(
                fitz.Point(*pts[k]),
                fitz.Point(*pts[k + 1]),
                color=col,
                width=0.75,
            )

    # Tick marks (short, thin).
    for i in range(11):
        x = 100 + i * 40
        page.draw_line(fitz.Point(x, 600), fitz.Point(x, 605), color=black, width=0.5)

    pdf_path = tmp_path / "monochrome_lineplot.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_dense_table_pdf(tmp_path: Path) -> Path:
    """A page dominated by a table grid (horizontal + vertical ruling lines)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    # Horizontal lines.
    for row in range(10):
        y = 100 + row * 40
        page.draw_line(fitz.Point(72, y), fitz.Point(540, y), color=(0, 0, 0), width=0.5)

    # Vertical lines.
    for col in range(7):
        x = 72 + col * 68
        page.draw_line(fitz.Point(x, 100), fitz.Point(x, 460), color=(0, 0, 0), width=0.5)

    # Numbers in cells.
    for row in range(8):
        for col in range(6):
            page.insert_text(
                (80 + col * 68, 120 + row * 40),
                f"{row * 10 + col:.2f}",
                fontsize=9,
            )

    pdf_path = tmp_path / "dense_table.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_framed_spike_pdf(tmp_path: Path, *, frame: str = "rect", n_spikes: int = 8) -> Path:
    """A monochrome spike plot: an enclosing frame plus thin neutral spikes.

    Models the GH-150 A1 motivating page (Heston p10): the frame and every
    spike are thin, neutral-coloured strokes, so ``_has_vector_data_marks``
    returns False and only ``_has_framed_data_cluster`` can admit the cluster.

    ``frame`` selects the enclosure variant:
    - ``"rect"``: one axis-aligned frame rectangle (the genuine-chart case);
    - ``"diagonal"``: a single corner-to-corner diagonal stroke whose bounding
      box spans the cluster (a non-frame that must NOT qualify);
    - ``"double"``: the frame rectangle drawn twice (fill+stroke-style
      duplicate that must not count itself as an interior mark).
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "Figure 2: Return Spikes by Interval", fontsize=11)

    black = (0, 0, 0)
    frame_rect = fitz.Rect(100, 100, 500, 500)
    if frame == "diagonal":
        page.draw_line(fitz.Point(100, 100), fitz.Point(500, 500), color=black, width=0.75)
    else:
        page.draw_rect(frame_rect, color=black, width=0.75)
        if frame == "double":
            page.draw_rect(frame_rect, color=black, width=0.75)

    # Thin neutral spikes inside the frame (each its own drawing, tiny bbox).
    gray = (0.25, 0.25, 0.25)
    for i in range(n_spikes):
        x = 130 + i * 40
        page.draw_line(fitz.Point(x, 480), fitz.Point(x, 380 - (i % 4) * 20), color=gray, width=0.6)

    pdf_path = tmp_path / f"framed_spikes_{frame}.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# has_chart_marks() unit tests
# ---------------------------------------------------------------------------


class TestHasChartMarks:
    """Unit tests for has_chart_marks() — model-free, deterministic."""

    def _open_page(self, pdf_path: Path):
        """Return the first page of a PDF (caller must close the doc)."""
        import fitz as _fitz

        doc = _fitz.open(pdf_path)
        return doc, doc[0]

    def test_vector_chart_detected(self, tmp_path: Path) -> None:
        """A pure-vector coloured chart page is detected (zero embedded raster)."""
        from socr.figures.extractor import has_chart_marks

        pdf = _make_vector_chart_pdf(tmp_path)
        doc, page = self._open_page(pdf)
        try:
            # Confirm it IS zero-raster (otherwise we'd test the raster fast-path).
            assert len(page.get_images()) == 0, (
                "Fixture must have zero embedded raster images for this test to be "
                "load-bearing (otherwise the raster fast-path fires, not vector detection)"
            )
            assert has_chart_marks(page) is True
        finally:
            doc.close()

    def test_raster_chart_detected(self, tmp_path: Path) -> None:
        """A page with an embedded raster image is detected via the raster fast-path."""
        from socr.figures.extractor import has_chart_marks

        pdf = _make_raster_chart_pdf(tmp_path)
        doc, page = self._open_page(pdf)
        try:
            assert len(page.get_images()) >= 1, "Fixture must have an embedded raster image"
            assert has_chart_marks(page) is True
        finally:
            doc.close()

    def test_decorated_prose_not_detected(self, tmp_path: Path) -> None:
        """Decorative horizontal rules on prose pages do NOT trigger the chart lane.

        Cluster-first: scattered thin neutral rules do not form a coherent cluster
        with data marks, so has_chart_marks() returns False.
        """
        from socr.figures.extractor import has_chart_marks

        pdf = _make_decorated_prose_pdf(tmp_path)
        doc, page = self._open_page(pdf)
        try:
            result = has_chart_marks(page)
            # Thin neutral rules must not false-trigger the chart lane.
            assert result is False, (
                "Decorative-rule prose page false-triggered the chart lane; "
                "cluster-first filter should have rejected it"
            )
        finally:
            doc.close()

    def test_prose_only_not_detected(self, tmp_path: Path) -> None:
        """A clean prose page with no drawings is not detected as a chart."""
        from socr.figures.extractor import has_chart_marks

        pdf = _make_prose_only_pdf(tmp_path)
        doc, page = self._open_page(pdf)
        try:
            assert has_chart_marks(page) is False
        finally:
            doc.close()

    def test_monochrome_lineplot_is_false_negative(self, tmp_path: Path) -> None:
        """Monochrome academic line-plot: documented false-negative.

        A B&W line-plot with only thin neutral strokes cannot be distinguished
        from ruling by the deterministic mark logic (_has_vector_data_marks returns
        False).  has_chart_marks() returns False.

        This is an accepted limitation: detecting monochrome plots requires either
        a VLM pass or a raster rendering heuristic (both out of scope for PP-7).
        The test documents the false-negative, not a bug.
        """
        from socr.figures.extractor import has_chart_marks

        pdf = _make_monochrome_lineplot_pdf(tmp_path)
        doc, page = self._open_page(pdf)
        try:
            result = has_chart_marks(page)
            # Documented false-negative: monochrome plots without colour are
            # not detected.  If this assertion FAILS in the future (i.e. result
            # is True), the detector improved — update the doc comment above.
            assert result is False, (
                "Monochrome line-plot was detected as a chart.  If this is "
                "intentional (detector improved), update the test comment."
            )
        finally:
            doc.close()

    def test_framed_monochrome_spike_plot_detected(self, tmp_path: Path) -> None:
        """GH-150 A1: a framed monochrome spike plot IS detected.

        All strokes are thin and neutral (``_has_vector_data_marks`` False),
        but the axis-aligned frame covering the cluster with >= 5 interior
        marks passes ``_has_framed_data_cluster``.
        """
        from socr.figures.extractor import has_chart_marks

        pdf = _make_framed_spike_pdf(tmp_path, frame="rect")
        doc, page = self._open_page(pdf)
        try:
            assert has_chart_marks(page) is True, (
                "Framed monochrome spike plot was not detected — "
                "_has_framed_data_cluster should admit this cluster"
            )
        finally:
            doc.close()

    def test_diagonal_stroke_is_not_a_frame(self, tmp_path: Path) -> None:
        """A diagonal stroke whose bounding box spans the cluster is no frame.

        Guards the structural frame test: frame candidates must be built of
        axis-aligned strokes, so a large diagonal/curved decoration cannot
        masquerade as a plot frame via its bounding box.
        """
        from socr.figures.extractor import has_chart_marks

        pdf = _make_framed_spike_pdf(tmp_path, frame="diagonal")
        doc, page = self._open_page(pdf)
        try:
            assert has_chart_marks(page) is False, (
                "A diagonal stroke's bounding box was accepted as a plot frame"
            )
        finally:
            doc.close()

    def test_duplicate_frame_does_not_count_itself(self, tmp_path: Path) -> None:
        """A frame drawn twice must not count its twin as an interior mark.

        With only 4 real interior spikes, the duplicate frame drawing (which
        is itself frame-sized) must not supply the 5th mark.
        """
        from socr.figures.extractor import MIN_DRAWINGS_FOR_VECTOR, has_chart_marks

        pdf = _make_framed_spike_pdf(tmp_path, frame="double", n_spikes=MIN_DRAWINGS_FOR_VECTOR - 1)
        doc, page = self._open_page(pdf)
        try:
            assert has_chart_marks(page) is False, (
                "A duplicate frame drawing counted itself toward the interior-mark threshold"
            )
        finally:
            doc.close()

    def test_has_chart_marks_logs_debug(self, tmp_path: Path, caplog) -> None:
        """has_chart_marks logs mark counts / rejection reason at DEBUG level."""
        import logging

        from socr.figures.extractor import has_chart_marks

        pdf = _make_decorated_prose_pdf(tmp_path)
        doc, page = self._open_page(pdf)
        try:
            with caplog.at_level(logging.DEBUG, logger="socr.figures.extractor"):
                has_chart_marks(page)
            # Should have logged something about drawings / clusters.
            assert any("has_chart_marks" in r.message for r in caplog.records), (
                "Expected has_chart_marks to emit at least one DEBUG log"
            )
        finally:
            doc.close()


# ---------------------------------------------------------------------------
# _is_chart_asset_page() unit tests
# ---------------------------------------------------------------------------


class TestIsChartAssetPage:
    """Unit tests for UnifiedPipeline._is_chart_asset_page()."""

    def _make_pipeline(self, native_first: bool = True, **config_overrides):
        from socr.core.config import EngineType, PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        cfg = PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            enabled_engines=list(EngineType),
            agentic=False,
            quiet=True,
            native_first=native_first,
            **config_overrides,
        )
        return UnifiedPipeline(cfg)

    def _make_native_ps(self, page_num: int = 1, has_tables: bool = False):
        from socr.core.state import PageState

        return PageState(
            page_num=page_num,
            is_born_digital=True,
            native_text="GDP Growth Forecast 2025",
            needs_ocr_enhancement=False,
            has_tables=has_tables,
        )

    def test_vector_chart_page_fires(self, tmp_path: Path) -> None:
        """_is_chart_asset_page returns True for a genuine vector chart page."""
        pipeline = self._make_pipeline()
        ps = self._make_native_ps()
        pdf = _make_vector_chart_pdf(tmp_path)
        assert pipeline._is_chart_asset_page(1, ps, pdf) is True

    def test_table_page_with_chart_marks_also_fires(self, tmp_path: Path) -> None:
        """GH-150 TICKET-B1: a table signal no longer suppresses chart ELIGIBILITY.

        Scope: this pins the predicate only. `_is_chart_asset_page` answers
        eligibility, not the final route — before B1 a table signal made it
        return False outright, so a mixed page could never be arbitrated at all.

        It does NOT assert that such a page reaches the page-level chart-asset
        lane; it does not. Mixed pages are deliberately held out of that lane and
        keep their normal route, where the chart is emitted inline at its own
        y-position. That routing behaviour is pinned by
        ``TestAgenticChartLaneRouting::
        test_mixed_chart_table_page_keeps_its_normal_route``.
        """
        pipeline = self._make_pipeline()
        ps = self._make_native_ps(has_tables=True)
        pdf = _make_vector_chart_pdf(tmp_path)
        assert pipeline._is_chart_asset_page(1, ps, pdf) is True

    def test_trusted_native_still_false_for_table_page(self, tmp_path: Path) -> None:
        """_is_trusted_native_without_ocr keeps excluding table pages (byte-identical).

        Pins the predicate split (GH-150 TICKET-B1): the new shared
        _is_native_eligible_without_ocr drops the table exclusion, but
        _is_trusted_native_without_ocr must still apply it — its second caller
        (the non-agentic native-bypass branch) must not change behavior.
        """
        pipeline = self._make_pipeline()
        ps = self._make_native_ps(has_tables=True)
        assert pipeline._is_trusted_native_without_ocr(1, ps) is False

    def test_prose_only_page_does_not_fire(self, tmp_path: Path) -> None:
        """_is_chart_asset_page returns False for a clean prose page."""
        pipeline = self._make_pipeline()
        ps = self._make_native_ps()
        pdf = _make_prose_only_pdf(tmp_path)
        assert pipeline._is_chart_asset_page(1, ps, pdf) is False

    def test_not_native_first_does_not_fire(self, tmp_path: Path) -> None:
        """When native_first=False, _is_chart_asset_page returns False (no native lane)."""
        pipeline = self._make_pipeline(native_first=False)
        ps = self._make_native_ps()
        pdf = _make_vector_chart_pdf(tmp_path)
        # When native_first=False, _is_trusted_native_without_ocr returns False,
        # so _is_chart_asset_page also returns False.
        assert pipeline._is_chart_asset_page(1, ps, pdf) is False

    def test_raster_chart_fires(self, tmp_path: Path) -> None:
        """_is_chart_asset_page fires for a page with an embedded raster image."""
        pipeline = self._make_pipeline()
        ps = self._make_native_ps()
        pdf = _make_raster_chart_pdf(tmp_path)
        assert pipeline._is_chart_asset_page(1, ps, pdf) is True


# ---------------------------------------------------------------------------
# _render_chart_page_png() unit tests
# ---------------------------------------------------------------------------


class TestRenderChartPagePng:
    """Unit tests for _render_chart_page_png()."""

    def _make_pipeline(self):
        from socr.core.config import EngineType, PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        cfg = PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            enabled_engines=list(EngineType),
            agentic=False,
            quiet=True,
        )
        return UnifiedPipeline(cfg)

    def test_png_saved_to_figures_dir(self, tmp_path: Path) -> None:
        """_render_chart_page_png saves a PNG file in the given figures_dir."""
        pipeline = self._make_pipeline()
        pdf = _make_vector_chart_pdf(tmp_path)
        figures_dir = tmp_path / "figures"
        saved = pipeline._render_chart_page_png(pdf, 1, figures_dir)
        assert Path(saved).exists(), "PNG file must exist after render"
        assert saved.endswith(".png"), "Output must be a PNG"
        assert "chart_page_1" in saved

    def test_png_saved_even_without_save_figures_flag(self, tmp_path: Path) -> None:
        """PNG is produced by _render_chart_page_png regardless of --save-figures.

        The flag contract narrowing: chart-lane captures are mandatory preservation
        artifacts — they bypass the global save_figures gate.
        """
        from socr.core.config import EngineType, PipelineConfig
        from socr.pipeline.orchestrator import UnifiedPipeline

        # Explicitly OFF.
        cfg = PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            enabled_engines=list(EngineType),
            agentic=False,
            quiet=True,
            save_figures=False,
        )
        pipeline = UnifiedPipeline(cfg)
        pdf = _make_vector_chart_pdf(tmp_path)
        figures_dir = tmp_path / "figures"
        # Should NOT raise, should produce a file.
        saved = pipeline._render_chart_page_png(pdf, 1, figures_dir)
        assert Path(saved).exists()

    def test_render_failure_raises_runtime_error(self, tmp_path: Path) -> None:
        """A render failure raises RuntimeError (fail-closed, never silent)."""
        pipeline = self._make_pipeline()
        non_existent_pdf = tmp_path / "does_not_exist.pdf"
        figures_dir = tmp_path / "figures"
        with pytest.raises(RuntimeError, match="chart-lane PNG render failed"):
            pipeline._render_chart_page_png(non_existent_pdf, 1, figures_dir)


# ---------------------------------------------------------------------------
# Agentic-loop routing integration tests
# ---------------------------------------------------------------------------


def _make_agentic_pipeline(save_figures: bool = False, table_judge_ladder: bool = True):
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    cfg = PipelineConfig(
        primary_engine=EngineType.DEEPSEEK,
        enabled_engines=list(EngineType),
        agentic=True,
        quiet=True,
        native_first=True,
        save_figures=save_figures,
        table_judge_ladder=table_judge_ladder,
    )
    return UnifiedPipeline(cfg)


def _make_state_with_page(
    pdf_path: Path,
    page_count: int = 1,
    native_text: str = "GDP Growth Forecast 2025",
    has_tables: bool = False,
):
    """Build a DocumentState with one born-digital page (for agentic routing)."""
    from unittest.mock import patch

    from socr.core.born_digital import DocumentAssessment, PageAssessment
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf_path, page_count=page_count)
    state = DocumentState(handle=handle)

    for pn in range(1, page_count + 1):
        ps = PageState(
            page_num=pn,
            is_born_digital=True,
            native_text=native_text,
            needs_ocr_enhancement=False,
            has_tables=has_tables,
        )
        state.pages[pn] = ps

    # Build a minimal assessment.
    pages_assess = [
        PageAssessment(
            page_num=pn,
            is_born_digital=True,
            native_text=native_text,
            confidence=0.9,
            needs_ocr_enhancement=False,
            has_tables=has_tables,
        )
        for pn in range(1, page_count + 1)
    ]
    state._last_assessment = DocumentAssessment(path=pdf_path, pages=pages_assess)

    return state


def test_chart_eligibility_recursion_failure_preserves_table_route_and_audit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """GH-181: chart eligibility failure is visible without changing table routing."""
    from socr.core.audit_log import build_run_audit
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision

    pdf = _make_vector_chart_pdf(tmp_path)
    pipeline = _make_agentic_pipeline()
    state = _make_state_with_page(pdf, has_tables=True, native_text="Table 1 data")
    pipeline._last_assessment = state._last_assessment
    routed_output = PageOutput(
        page_num=1,
        text="| Col A | Col B |\n|---|---|\n| 1.0 | 2.0 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    routed_decision = PageDecision(page_num=1, final_output=routed_output, accepted=True)

    with (
        caplog.at_level(logging.WARNING, logger="socr.pipeline.orchestrator"),
        patch(
            "socr.pipeline.orchestrator.has_chart_marks",
            side_effect=RecursionError("chart detector recursion exhausted"),
        ),
        patch("socr.pipeline.orchestrator.route_page", return_value=routed_decision) as mock_route,
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
    ):
        pipeline._phase_agentic(state, tmp_path)

    mock_route.assert_called_once()
    assert state.pages[1].best_output is routed_output
    assert state.pages[1].best_output.text == routed_output.text
    assert state.pages[1].best_output.engine == routed_output.engine

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "chart eligibility" in warnings[0].getMessage().lower()

    failures = [e for e in state.events if e.kind == "chart_asset_detection_failed"]
    assert len(failures) == 1
    failure = failures[0]
    assert failure.page_num == 1
    assert failure.data["error_type"] == "RecursionError"

    audit = build_run_audit(state)
    payload = json.loads(json.dumps(audit.to_dict()))
    saved_failures = [e for e in payload["events"] if e["kind"] == "chart_asset_detection_failed"]
    assert len(saved_failures) == 1
    assert saved_failures[0]["page_num"] == 1
    assert saved_failures[0]["data"]["error_type"] == "RecursionError"


def _run_chart_detection_case(tmp_path: Path, *, fail_detector: bool):
    """GH-318 helper: one agentic+assemble run, differing only in detector outcome."""
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import PageDecision

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = _make_vector_chart_pdf(tmp_path)
    # P1 golden audit: pin the ladder off; this test is about chart detection
    pipeline = _make_agentic_pipeline(table_judge_ladder=False)
    state = _make_state_with_page(pdf, has_tables=True, native_text="Table 1 data")
    pipeline._last_assessment = state._last_assessment
    routed_output = PageOutput(
        page_num=1,
        text="| Col A | Col B |\n|---|---|\n| 1.0 | 2.0 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    routed_decision = PageDecision(page_num=1, final_output=routed_output, accepted=True)

    marks = (
        patch(
            "socr.pipeline.orchestrator.has_chart_marks",
            side_effect=RecursionError("chart detector recursion exhausted"),
        )
        if fail_detector
        else patch("socr.pipeline.orchestrator.has_chart_marks", return_value=False)
    )

    with (
        marks,
        patch("socr.pipeline.orchestrator.route_page", return_value=routed_decision),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
    ):
        pipeline._phase_agentic(state, tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = pipeline._phase_assemble(state, out_dir)
    return state, result


def test_gh318_chart_detection_failure_is_visible_at_page_and_document_status(
    tmp_path: Path,
) -> None:
    """GH-318: a swallowed chart-detection failure must not ship a clean SUCCESS.

    Pins a DIFFERENCE, never an absolute tuple: the same pipeline runs twice and
    only the detector's behaviour changes. Absolute statuses here are
    provider-dependent (CLAUDE.md), so asserting a measured tuple would pass
    locally and fail in CI on a correctly-behaving page.
    """
    from socr.core.result import DocumentStatus, PageStatus

    state_bad, res_bad = _run_chart_detection_case(tmp_path / "failed", fail_detector=True)
    state_ok, res_ok = _run_chart_detection_case(tmp_path / "clean", fail_detector=False)

    # 1. The content still wins -- the fail-soft route of #297 is preserved and
    #    audit_passed is NOT used as a quality flag (flipping it would discard
    #    the page's text: the #252 round-1 defect).
    assert "Col A" in res_bad.markdown
    assert state_bad.pages[1].best_output.text == state_ok.pages[1].best_output.text
    assert state_bad.pages[1].best_output.audit_passed is True

    # 2. Page status differs: the failed run is no longer a clean SUCCESS.
    assert state_ok.pages[1].best_output.status == PageStatus.SUCCESS
    assert state_bad.pages[1].best_output.status != state_ok.pages[1].best_output.status
    assert state_bad.pages[1].best_output.status == PageStatus.WARNING

    # 3. Document status differs -- AUDIT_FAILED is the "completed with
    #    warnings, output written" path the CLI already reports.
    assert res_ok.status == DocumentStatus.SUCCESS
    assert res_bad.status != res_ok.status
    assert res_bad.status == DocumentStatus.AUDIT_FAILED

    # 4. The signal reaches metadata, not just audit_log.json.
    assert res_bad.error and "chart eligibility detection failed" in res_bad.error
    assert not (res_ok.error and "chart eligibility" in res_ok.error)

    # 5. The flag is on the page, so it survives into the sidecar and resume.
    assert state_bad.pages[1].chart_asset_detection_failed is True
    assert state_ok.pages[1].chart_asset_detection_failed is False


def test_gh318_detection_flag_survives_resume_restore(tmp_path: Path) -> None:
    """GH-318: a clean sidecar must never erase a failure recorded this run.

    Exercises the real ``_restore_terminal_page_state``. OCR pages can resume
    BEFORE chart eligibility runs, so restoring with a plain assignment would let
    a stale clean sidecar clear a live failure. The restore ORs instead; this
    test fails if that is ever weakened back to assignment.
    """
    import json as _json

    from ocr_output_contract import doc_dir_for, relative_key

    from socr.core.result import PageOutput, PageStatus

    pdf = _make_vector_chart_pdf(tmp_path)
    pipeline = _make_agentic_pipeline()
    state = _make_state_with_page(pdf)
    out_dir = tmp_path / "out"

    # A terminal sidecar from an EARLIER run, in which detection was clean.
    doc_dir = doc_dir_for(out_dir, relative_key(pdf, pdf.parent))
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "00001.json").write_text(
        _json.dumps({"chart_asset_detection_failed": False}), encoding="utf-8"
    )

    # THIS run has already recorded a detection failure for the same page.
    state.pages[1].chart_asset_detection_failed = True
    pipeline._scan_root = pdf.parent

    resumed = PageOutput(
        page_num=1, text="body", status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    pipeline._restore_terminal_page_state(state, 1, resumed, out_dir)

    assert state.pages[1].chart_asset_detection_failed is True, (
        "a clean sidecar from an earlier run must not clear a failure recorded this run"
    )


class TestAgenticChartLaneRouting:
    """Integration tests for chart lane in the PP-2 fused agentic loop."""

    def test_chart_page_routes_to_chart_asset_lane(self, tmp_path: Path) -> None:
        """A genuine vector chart page routes to the chart-asset lane (not the OCR ladder).

        The output engine must be 'chart_asset'; the text must contain the native prose
        AND an embedded chart PNG ref; the audit log must carry a 'chart_asset_page' event.
        """
        pdf = _make_vector_chart_pdf(tmp_path)
        pipeline = _make_agentic_pipeline()
        state = _make_state_with_page(pdf)
        pipeline._last_assessment = state._last_assessment

        # Run only the agentic phase (we patch route_page to ensure no VLM is called).
        with patch("socr.pipeline.orchestrator.route_page") as mock_route:
            pipeline._phase_agentic(state, tmp_path)
            # route_page must NOT have been called for this page (chart lane bypasses OCR).
            mock_route.assert_not_called()

        ps = state.pages[1]
        assert ps.best_output is not None, "Chart page must have a best_output"
        assert ps.best_output.engine == "chart_asset", (
            f"Expected engine='chart_asset', got '{ps.best_output.engine}'"
        )
        # B1: native prose retained.
        assert "GDP Growth Forecast 2025" in (ps.best_output.text or "")
        # B1: chart PNG ref embedded.
        assert "![Chart page 1]" in (ps.best_output.text or ""), (
            "Expected an embedded chart PNG ref in the page output text"
        )
        # Audit event.
        chart_events = [e for e in state.events if e.kind == "chart_asset_page"]
        assert chart_events, "Expected a 'chart_asset_page' AuditEvent"
        assert chart_events[0].page_num == 1
        # GH-150 TICKET-B1: single-signal (chart-only) page must NOT emit an
        # arbitration event — that event is reserved for both-signals pages.
        arb_events = [e for e in state.events if e.kind == "chart_table_arbitration"]
        assert not arb_events, "Chart-only page must not emit a chart_table_arbitration event"

    def test_axis_tick_scale_is_fenced_not_shipped_as_body_prose(self, tmp_path: Path) -> None:
        """GH-369: the lane declares "data values not transcribed", so a column
        of bare numbers must not ship beside the image as body prose.

        This pins the WIRE, not the helper. ``fence_chart_axis_residue`` has its
        own unit coverage in ``test_gh369_chart_axis_residue.py``; a green helper
        suite is not a gate on the page that actually ships (GH-359 ruling 6).
        Revert the call in ``_agentic_chart_asset_page`` and this test fails.
        """
        pdf = _make_vector_chart_pdf(tmp_path)
        pipeline = _make_agentic_pipeline()
        state = _make_state_with_page(
            pdf,
            native_text=(
                "Figure 4. Uncertainty and risks in economic projections\n"
                "Number of participants\n"
                "2\n"
                "4\n"
                "6\n"
                "Lower\n"
                "March projections"
            ),
        )
        pipeline._last_assessment = state._last_assessment

        with patch("socr.pipeline.orchestrator.route_page"):
            pipeline._phase_agentic(state, tmp_path)

        text = state.pages[1].best_output.text or ""

        # Nothing is dropped -- a dropped number is worse than a missing one.
        for line in ("Figure 4. Uncertainty and risks in economic projections", "2", "4", "6"):
            assert line in text, f"chart lane lost {line!r}"

        # The tick scale is inside the fence; the caption is not.
        assert "socr:chart-axis-residue" in text
        head, _, tail = text.partition("socr:chart-axis-residue")
        for number in ("2", "4", "6"):
            assert f"\n{number}\n" in tail, f"tick label {number!r} is not inside the fence"
            # GH-387: presence in the fence is only half of it. A writer that
            # COPIES the residue into the comment and leaves ps.native_text
            # beside it satisfies the assertion above while the page still
            # ships a column of bare numbers as body prose -- the whole defect.
            assert number not in head.splitlines(), (
                f"tick label {number!r} is still body prose above the fence"
            )
        assert "Figure 4. Uncertainty and risks in economic projections" in head
        assert "March projections" in head

    def test_a_lone_year_ships_in_the_body_and_gains_no_fence(self, tmp_path: Path) -> None:
        """GH-387: the lone-number rule pinned at the WIRE, not just the helper.

        ``TestALoneNumberIsContentNotFurniture`` pins this on
        ``split_chart_axis_residue``. Revert the helper to fence every bare
        number and the existing wire test stays green, because its fixture is a
        run of 2/4/6. A chart page carrying a single year must keep it visible:
        it is content, and the fence is not where a reader looks.
        """
        pdf = _make_vector_chart_pdf(tmp_path)
        pipeline = _make_agentic_pipeline()
        state = _make_state_with_page(
            pdf,
            native_text=(
                "Figure 2. Outlook for the coming year\n"
                "Published 2025\n"
                "2025\n"
                "See the notes on page 4"
            ),
        )
        pipeline._last_assessment = state._last_assessment

        with patch("socr.pipeline.orchestrator.route_page"):
            pipeline._phase_agentic(state, tmp_path)

        text = state.pages[1].best_output.text or ""

        assert "socr:chart-axis-residue" not in text, (
            "a single number is not an axis scale; fencing it hides content from the rendered page"
        )
        assert "2025" in text.splitlines()
        assert "Figure 2. Outlook for the coming year" in text

    def test_chart_page_without_tick_labels_gains_no_fence(self, tmp_path: Path) -> None:
        """Difference control: the SAME lane, the SAME fixture, differing only in
        whether the native layer carries bare-numeric lines. A chart page whose
        text has none must be byte-identical to its pre-GH-369 output."""
        pdf = _make_vector_chart_pdf(tmp_path)
        pipeline = _make_agentic_pipeline()
        state = _make_state_with_page(pdf, native_text="GDP Growth Forecast 2025")
        pipeline._last_assessment = state._last_assessment

        with patch("socr.pipeline.orchestrator.route_page"):
            pipeline._phase_agentic(state, tmp_path)

        text = state.pages[1].best_output.text or ""
        assert "GDP Growth Forecast 2025" in text
        assert "socr:chart-axis-residue" not in text

    def test_chart_png_produced_without_save_figures_flag(self, tmp_path: Path) -> None:
        """PNG is saved by the chart lane even when save_figures=False."""
        pdf = _make_vector_chart_pdf(tmp_path)
        pipeline = _make_agentic_pipeline(save_figures=False)
        state = _make_state_with_page(pdf)
        pipeline._last_assessment = state._last_assessment

        with patch("socr.pipeline.orchestrator.route_page"):
            pipeline._phase_agentic(state, tmp_path)

        # A PNG must have been created in the figures dir.
        png_files = list(tmp_path.rglob("chart_page_*.png"))
        assert png_files, (
            "chart-lane must save a PNG even when --save-figures is off; "
            f"found no chart_page_*.png under {tmp_path}"
        )

    def test_clean_prose_page_ships_native(self, tmp_path: Path) -> None:
        """A clean prose page ships as native text (chart lane does not fire)."""
        pdf = _make_prose_only_pdf(tmp_path)
        pipeline = _make_agentic_pipeline()
        state = _make_state_with_page(pdf, native_text="Lorem ipsum " * 20)
        pipeline._last_assessment = state._last_assessment

        with patch("socr.pipeline.orchestrator.route_page"):
            pipeline._phase_agentic(state, tmp_path)

        ps = state.pages[1]
        assert ps.best_output is not None
        # Must be native, not chart_asset.
        assert ps.best_output.engine == "native", (
            f"Expected engine='native' for prose page, got '{ps.best_output.engine}'"
        )
        chart_events = [e for e in state.events if e.kind == "chart_asset_page"]
        assert not chart_events, "Clean prose page must not emit a chart_asset_page event"

    def test_table_page_routes_to_ocr_ladder(self, tmp_path: Path) -> None:
        """A table page does NOT go to the chart lane — it still routes to the OCR ladder."""
        from socr.core.result import PageOutput, PageStatus

        pdf = _make_dense_table_pdf(tmp_path)
        pipeline = _make_agentic_pipeline()
        state = _make_state_with_page(pdf, has_tables=True, native_text="Table 1 data")
        pipeline._last_assessment = state._last_assessment

        # Mock route_page to return a successful OCR result for the table page.
        mock_decision = MagicMock()
        mock_decision.accepted = True
        mock_decision.attempts = []
        mock_decision.final_output = PageOutput(
            page_num=1,
            text="| Col A | Col B |\n|---|---|\n| 1.0 | 2.0 |",
            status=PageStatus.SUCCESS,
            engine="deepseek",
            audit_passed=True,
        )
        mock_decision.winning_engine = "deepseek"
        mock_decision.total_cost_usd = 0.002

        # Hermetic: patch _available_engines_for_agentic so the provider ladder is
        # non-empty regardless of whether ollama/qwen is installed — otherwise the
        # loop bails before the patched route_page (no providers) and the table page
        # is never routed (best_output stays None) in a no-provider env like CI.
        from socr.core.providers import PROFILE_QWEN_LOCAL

        with (
            patch("socr.pipeline.orchestrator.route_page", return_value=mock_decision),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
        ):
            pipeline._phase_agentic(state, tmp_path)

        ps = state.pages[1]
        assert ps.best_output is not None
        # Table pages must go to OCR, not chart_asset.
        assert ps.best_output.engine != "chart_asset", (
            "Table page must NOT route to the chart-asset lane"
        )
        chart_events = [e for e in state.events if e.kind == "chart_asset_page"]
        assert not chart_events, "Table page must not emit a chart_asset_page event"

    def test_mixed_chart_table_page_keeps_its_normal_route(self, tmp_path: Path) -> None:
        """GH-150 TICKET-B1: a page with BOTH chart marks and a table signal does NOT
        take the page-level chart-asset lane.

        The page-level lane appends its PNG ref after the whole of ``ps.native_text``,
        which is correct only when the chart IS the page. On a mixed page the chart
        sits between other regions, so appending sinks it below every table that
        followed it in the source — the reading-order break caught by
        ``test_table_repair_parity.py::TestEndToEndParity::
        test_agentic_parity_on_ce_like_fixture``.

        Such pages therefore keep their normal route, where
        ``rowize_from_words_chart_aware`` emits an inline placeholder at the chart's
        own y-position. This test pins that routing decision and the audit event that
        records it, including the audit_log.json round-trip.
        """
        import json

        from socr.core.audit_log import build_run_audit
        from socr.core.providers import PROFILE_QWEN_LOCAL
        from socr.core.result import PageOutput, PageStatus

        pdf = _make_vector_chart_pdf(tmp_path)
        pipeline = _make_agentic_pipeline()
        # Both signals: chart marks (from the fixture) AND has_tables=True.
        state = _make_state_with_page(pdf, has_tables=True)
        pipeline._last_assessment = state._last_assessment

        # route_page must return a REAL PageOutput, not a bare MagicMock. With a
        # MagicMock, ``best_output.engine != "chart_asset"`` and
        # ``"![Chart page 1](" not in best_output.text`` are both vacuously True
        # (MagicMock compares unequal to every string and its ``__contains__``
        # returns False), so the two content assertions below would pass even if
        # the page HAD been claimed by the page-level chart lane — the exact
        # regression this test exists to catch.
        mock_decision = MagicMock()
        mock_decision.accepted = True
        mock_decision.attempts = []
        mock_decision.final_output = PageOutput(
            page_num=1,
            text="| Col A | Col B |\n|---|---|\n| 1.0 | 2.0 |",
            status=PageStatus.SUCCESS,
            engine="deepseek",
            audit_passed=True,
        )
        mock_decision.winning_engine = "deepseek"
        mock_decision.total_cost_usd = 0.002

        with (
            patch(
                "socr.pipeline.orchestrator.route_page", return_value=mock_decision
            ) as mock_route,
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
        ):
            pipeline._phase_agentic(state, tmp_path)
            # The page keeps its normal route rather than being intercepted.
            mock_route.assert_called()

        ps = state.pages[1]
        assert ps.best_output is not None
        # Not claimed by the page-level chart lane — it carries the ROUTED output.
        assert ps.best_output.engine == "deepseek"
        assert ps.best_output.engine != "chart_asset"

        # The page-level lane's whole-page PNG ref must NOT be appended: on a mixed
        # page the chart is represented inline by the region placeholder instead.
        assert "![Chart page 1](" not in (ps.best_output.text or "")
        # Positive control: the routed text really did survive to best_output, so
        # the "not in" assertion above is inspecting real content, not a stub.
        assert "| Col A | Col B |" in (ps.best_output.text or "")

        chart_events = [e for e in state.events if e.kind == "chart_asset_page"]
        assert not chart_events, "Mixed page must not emit a chart_asset_page event"

        # The arbitration is still recorded — no silent routing.
        arb_events = [e for e in state.events if e.kind == "chart_table_arbitration"]
        assert len(arb_events) == 1, "Expected exactly one chart_table_arbitration event"
        arb = arb_events[0]
        assert arb.page_num == 1
        assert arb.engine == "chart_region"
        assert arb.data["winner"] == "chart_region"
        assert arb.data["loser"] == "chart_asset_page"
        assert arb.data["chart_marks"] is True
        assert arb.data["table_signal"] is True

        # Round-trip through the audit log to prove the event survives serialization.
        audit = build_run_audit(state)
        audit.save(tmp_path / "audit_log.json")
        saved = json.loads((tmp_path / "audit_log.json").read_text())
        saved_events = saved.get("events", saved) if isinstance(saved, dict) else saved
        # Locate the arbitration event in whatever shape build_run_audit produced.
        found = None
        if isinstance(saved_events, list):
            found = next(
                (e for e in saved_events if e.get("kind") == "chart_table_arbitration"), None
            )
        assert found is not None, "chart_table_arbitration event must survive the audit round-trip"
        assert found["page_num"] == 1
        assert found["data"]["winner"] == "chart_region"

    def test_render_failure_produces_hard_audit_error(self, tmp_path: Path) -> None:
        """A chart-lane PNG render failure emits a 'chart_asset_render_failed' audit event.

        Fail-closed: the page output has status=WARNING and audit_passed=False;
        the render failure is never silent.
        """
        pdf = _make_vector_chart_pdf(tmp_path)
        pipeline = _make_agentic_pipeline()
        state = _make_state_with_page(pdf)
        pipeline._last_assessment = state._last_assessment

        # Patch _render_chart_page_png to raise RuntimeError.
        with (
            patch.object(
                pipeline,
                "_render_chart_page_png",
                side_effect=RuntimeError("simulated PNG render failure"),
            ),
            patch("socr.pipeline.orchestrator.route_page"),
        ):
            pipeline._phase_agentic(state, tmp_path)

        ps = state.pages[1]
        assert ps.best_output is not None
        assert ps.best_output.engine == "chart_asset"
        # Fail-closed: audit_passed must be False on render failure.
        assert ps.best_output.audit_passed is False, (
            "PNG render failure must produce audit_passed=False (fail-closed)"
        )
        # Hard audit event must be present.
        error_events = [e for e in state.events if e.kind == "chart_asset_render_failed"]
        assert error_events, "Expected a 'chart_asset_render_failed' AuditEvent"


# ---------------------------------------------------------------------------
# CHART_MIN_CLUSTER_AREA constant is documented (not a magic literal)
# ---------------------------------------------------------------------------


def test_chart_min_cluster_area_is_named_constant() -> None:
    """CHART_MIN_CLUSTER_AREA must be exported and documented (no magic literals)."""
    from socr.figures.extractor import CHART_MIN_CLUSTER_AREA

    assert isinstance(CHART_MIN_CLUSTER_AREA, float)
    assert CHART_MIN_CLUSTER_AREA > 0
    # Sanity: must be larger than MIN_AREA (80×80) to suppress tiny decorative marks.
    from socr.figures.extractor import MIN_AREA

    assert CHART_MIN_CLUSTER_AREA > MIN_AREA, (
        "CHART_MIN_CLUSTER_AREA must exceed MIN_AREA to suppress tiny decorative marks"
    )


# ---------------------------------------------------------------------------
# PP-7-R1: Frozen-output fail-closed test
#
# Regression guard: _winning_page_output (manifest.py) must NOT re-stamp a
# chart-render-failed page as engine=native / audit_passed=True / status=SUCCESS.
# ---------------------------------------------------------------------------


def test_frozen_output_is_audit_failed_on_render_failure(tmp_path: Path) -> None:
    """PP-7-R1: frozen output for a chart-render-failed page must stay WARNING/audit-failed.

    Before the fix, _winning_page_output fell through from the non-passing best_output
    to the native-text branch and stamped engine=native, audit_passed=True, status=SUCCESS
    — scrubbing the fail-closed intent at manifest-freeze time.

    The fix: ps.chart_asset_render_failed=True is included in the native_is_fallback
    predicate in manifest.py, so the frozen output stays WARNING/audit_passed=False.
    """
    from socr.core.manifest import _winning_page_output
    from socr.core.result import PageOutput, PageStatus

    pdf = _make_vector_chart_pdf(tmp_path)
    state = _make_state_with_page(pdf)

    ps = state.pages[1]

    # Simulate what the chart lane sets on a render failure.
    ps.chart_asset_render_failed = True
    failed_chart_out = PageOutput(
        page_num=1,
        text="GDP Growth Forecast 2025",  # native prose retained
        status=PageStatus.WARNING,
        engine="chart_asset",
        audit_passed=False,
        cost_usd=0.0,
    )
    ps.attempts.append(failed_chart_out)
    ps.best_output = failed_chart_out

    # Call _winning_page_output — this is what the manifest freezes.
    frozen = _winning_page_output(state, 1)

    assert frozen.audit_passed is False, (
        "Frozen output for a chart-render-failed page must have audit_passed=False; "
        f"got audit_passed={frozen.audit_passed!r} engine={frozen.engine!r} "
        f"status={frozen.status!r}"
        " — _winning_page_output fell through to native SUCCESS (fail-closed scrubbed)"
    )
    assert frozen.status != PageStatus.SUCCESS, (
        "Frozen output for a chart-render-failed page must NOT have status=SUCCESS; "
        f"got status={frozen.status!r} — manifest-freeze scrubbed the failure"
    )
    # The native_is_fallback path correctly sets engine='native' (content comes from native
    # text), but it must be WARNING/audit_passed=False — not a silent success stamp.
    # The two assertions above are the load-bearing fail-closed checks.


# ---------------------------------------------------------------------------
# PP-7-R2: Document/run-summary level fail-closed test
#
# Regression guard: _phase_assemble() must surface a chart-render-failed page
# at the DOCUMENT level (final_result.status != SUCCESS, metadata WARNING/partial,
# native_fallback_pages entry) — not just at the page/manifest level.
# ---------------------------------------------------------------------------


def test_assemble_reports_non_success_for_render_failed_chart_page(tmp_path: Path) -> None:
    """PP-7-R2: a doc with a render-failed chart page must NOT report SUCCESS.

    Before the fix, native_fallback_pages in _phase_assemble only checked
    needs_ocr_enhancement and native_table_structure_failed, so a chart-render-
    failed page slipped through and the document reported SUCCESS/COMPLETED.

    The fix: chart_asset_render_failed is OR-d into the native_fallback_pages
    predicate, so final_result.status is AUDIT_FAILED (not SUCCESS) and the
    run summary/metadata reflects the failure.
    """
    from unittest.mock import patch

    from socr.core.config import EngineType, PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.core.result import DocumentStatus, PageOutput, PageStatus
    from socr.core.state import DocumentState
    from socr.pipeline.orchestrator import UnifiedPipeline

    cfg = PipelineConfig(
        primary_engine=EngineType.DEEPSEEK,
        enabled_engines=list(EngineType),
        agentic=False,
        quiet=True,
        native_first=True,
        # P1 (owner ruling Q3, 2026-09-03): the table-judge ladder is ON by
        # default and fail-closed, so a table page on a machine with no
        # reachable rung ships UNVERIFIED and the document is no longer
        # SUCCESS. This test is about a different lane, so the flag is
        # PINNED off rather than the assertion weakened
        # (docs/log/2026-09-03_p1-ladder-flip.md, "Test audit").
        table_judge_ladder=False,
    )
    pipeline = UnifiedPipeline(cfg)

    pdf = _make_vector_chart_pdf(tmp_path)

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=1)
    state = DocumentState(handle=handle)

    ps = state.pages[1]
    ps.is_born_digital = True
    ps.native_text = "GDP Growth Forecast 2025"
    ps.chart_asset_render_failed = True  # render failed

    # Simulate what the chart lane sets: attempt with audit_passed=False.
    failed_out = PageOutput(
        page_num=1,
        text="GDP Growth Forecast 2025",
        status=PageStatus.WARNING,
        engine="chart_asset",
        audit_passed=False,
        cost_usd=0.0,
    )
    ps.attempts.append(failed_out)
    ps.best_output = failed_out

    # Run _phase_assemble — this is what computes final_result and document status.
    final_result = pipeline._phase_assemble(state, tmp_path)

    assert final_result.status != DocumentStatus.SUCCESS, (
        "A document with a render-failed chart page must NOT report DocumentStatus.SUCCESS; "
        f"got status={final_result.status!r}. "
        "chart_asset_render_failed must be OR-d into native_fallback_pages in _phase_assemble."
    )
    # Must be AUDIT_FAILED (has text but not all pages passing).
    assert final_result.status == DocumentStatus.AUDIT_FAILED, (
        f"Expected AUDIT_FAILED for a render-failed chart doc, got {final_result.status!r}"
    )
