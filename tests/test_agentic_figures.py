"""GH-86 tests: agentic placeholder cleanup + assemble-time figure embedding."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ocr_output_contract import assemble_pages, doc_dir_for, figures_dir_for, relative_key

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.normalizer import OutputNormalizer
from socr.core.result import DocumentStatus, EngineResult, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState
from socr.figures.extractor import ExtractedFigure, ExtractionResult
from socr.pipeline.orchestrator import UnifiedPipeline


def _make_pipeline(**overrides) -> UnifiedPipeline:
    defaults = dict(
        primary_engine=EngineType.DEEPSEEK,
        quiet=True,
        audit_enabled=False,
        agentic=True,
        save_figures=True,
        describe_figures=False,
        write_manifest=False,
        dual_pass_tables=False,
        judge_hard_pages=False,
    )
    defaults.update(overrides)
    return UnifiedPipeline(PipelineConfig(**defaults))


def _make_state(pdf_path: Path, page_count: int = 1) -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf_path, page_count=page_count)
    state = DocumentState(handle=handle)
    for pn in range(1, page_count + 1):
        state.pages[pn] = PageState(page_num=pn, has_figures=True)
    return state


def _doc_layout(tmp_path: Path, pdf_name: str = "fig_test.pdf") -> tuple[Path, Path, Path]:
    pdf_path = tmp_path / pdf_name
    pdf_path.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "ocr"
    doc_dir = doc_dir_for(output_dir, relative_key(pdf_path, tmp_path))
    figures_dir = figures_dir_for(doc_dir)
    return pdf_path, doc_dir, figures_dir


def _make_result() -> EngineResult:
    return EngineResult(
        document_path=Path("/tmp/fig_test.pdf"),
        engine="deepseek",
        status=DocumentStatus.SUCCESS,
        pages=[
            PageOutput(
                page_num=0,
                text="",
                status=PageStatus.SUCCESS,
                engine="deepseek",
                audit_passed=True,
            )
        ],
        processing_time=1.0,
        audit_passed=True,
    )


class TestVlmPlaceholderStripping:
    """VLM sentinel paths are always stripped (GH-86 defect #3)."""

    def test_strip_image_png_and_image_url(self) -> None:
        norm = OutputNormalizer()
        text = "Body\n\n![](image.png)\n\n![Caption](image-url)\n\nTail"
        out = norm.strip_phantom_images(text, output_dir=Path("/tmp/nowhere"))
        assert "image.png" not in out
        assert "image-url" not in out
        assert "Body" in out
        assert "Tail" in out

    def test_sentinel_stripped_even_when_file_exists(self, tmp_path: Path) -> None:
        """A namesake image.png on disk must not resurrect a VLM placeholder."""
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        norm = OutputNormalizer()
        text = "See ![x](image.png) here"
        out = norm.strip_phantom_images(text, output_dir=tmp_path)
        assert "![x]" not in out


class TestAgenticPageImageSanitize:
    """Agentic pre-flush hook strips placeholders; embed deferred to assemble."""

    def test_unresolvable_placeholder_stripped_without_save_figures(self, tmp_path: Path) -> None:
        pdf_path, doc_dir, _figures_dir = _doc_layout(tmp_path)
        pipeline = _make_pipeline(save_figures=False)
        state = _make_state(pdf_path)

        page_out = PageOutput(
            page_num=1,
            text="Intro\n\n![Fig](image-url)\n\nOutro",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )

        pipeline._sanitize_agentic_page_image_refs(state, 1, page_out, doc_dir)

        assert "image-url" not in page_out.text
        assert "![Fig]" not in page_out.text
        assert "Intro" in page_out.text
        assert "Outro" in page_out.text
        unresolved = [e for e in state.events if e.kind == "figure_placeholder_unresolved"]
        assert unresolved, "expected audit event when placeholder stripped without extraction"

    def test_save_figures_does_not_audit_placeholder_at_sanitize(self, tmp_path: Path) -> None:
        """With save_figures on, placeholders are stripped pre-flush; embed comes at assemble."""
        pdf_path, doc_dir, _ = _doc_layout(tmp_path)
        pipeline = _make_pipeline(save_figures=True)
        state = _make_state(pdf_path)

        page_out = PageOutput(
            page_num=1,
            text="Body\n\n![](image.png)",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )

        pipeline._sanitize_agentic_page_image_refs(state, 1, page_out, doc_dir)

        assert "image.png" not in page_out.text
        assert not [e for e in state.events if e.kind == "figure_placeholder_unresolved"]


class TestDescribeAndEmbedFigurePaths:
    """Assemble-time figure phase embeds real figures/ paths (GH-86 defects #1/#2)."""

    def test_extracted_figure_yields_relative_figures_path(self, tmp_path: Path) -> None:
        pdf_path, doc_dir, figures_dir = _doc_layout(tmp_path)
        pipeline = _make_pipeline(agentic=False)
        state = _make_state(pdf_path)
        result = _make_result()

        figures_dir.mkdir(parents=True, exist_ok=True)
        png_path = figures_dir / "figure_1_page1.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_fig = ExtractedFigure(
            figure_num=1,
            page_num=1,
            image=MagicMock(),
            saved_path=str(png_path),
            bbox=(0.0, 0.0, 100.0, 100.0),
        )
        input_text = assemble_pages(["Page body with chart discussion."])

        with patch("socr.pipeline.orchestrator.FigureExtractor") as mock_extractor:
            mock_extractor.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path / "ocr", input_text)

        assert "![Figure 1](figures/figure_1_page1.png)" in out
        assert png_path.is_file()

    def test_saved_png_path_matches_emitted_ref(self, tmp_path: Path) -> None:
        pdf_path, doc_dir, figures_dir = _doc_layout(tmp_path)
        pipeline = _make_pipeline(agentic=False)
        state = _make_state(pdf_path)
        result = _make_result()

        figures_dir.mkdir(parents=True, exist_ok=True)
        png_path = figures_dir / "figure_2_page3.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_fig = ExtractedFigure(
            figure_num=2,
            page_num=3,
            image=MagicMock(),
            saved_path=str(png_path),
        )
        bodies = ["p1", "p2", "Page three body."]
        input_text = assemble_pages(bodies, page_numbers=[1, 2, 3])

        with patch("socr.pipeline.orchestrator.FigureExtractor") as mock_extractor:
            mock_extractor.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path / "ocr", input_text)

        assert "figures/figure_2_page3.png" in out
        assert (doc_dir / "figures" / "figure_2_page3.png").is_file()


class TestAgenticAssembleRunsFigurePhase:
    """Agentic must use the existing _describe_and_embed_figures phase (PP-5)."""

    def test_agentic_runs_describe_and_embed_when_save_figures(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path, _, _ = _doc_layout(tmp_path)
        state = _make_state(pdf_path)

        page_out = PageOutput(
            page_num=1,
            text="Page one.",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        state.pages[1].best_output = page_out
        state.pages[1].attempts.append(page_out)

        with patch.object(
            pipeline, "_describe_and_embed_figures", return_value="Page one."
        ) as mock:
            pipeline._phase_assemble(state, tmp_path / "ocr")
            mock.assert_called_once()
