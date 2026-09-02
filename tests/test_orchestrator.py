"""Tests for UnifiedPipeline (5-phase orchestrator)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ocr_output_contract import run_fingerprint

from socr.audit.heuristics import HeuristicsChecker
from socr.audit.scorer import FailureModeScorer, ScoringResult
from socr.core.born_digital import BornDigitalDetector, DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    FigureInfo,
    PageOutput,
    PageStatus,
)
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.state import DocumentState, PageState
from socr.figures.extractor import ExtractionResult, ExtractedFigure
from socr.math.recover import CORRUPT_MATH_PROMPT, CorruptMathRegion
from socr.pipeline.orchestrator import UnifiedPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handle(page_count: int = 3) -> DocumentHandle:
    """DocumentHandle without touching the filesystem."""
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        h = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=page_count)
    return h


def _make_config(**overrides) -> PipelineConfig:
    defaults = dict(
        primary_engine=EngineType.DEEPSEEK,
        fallback_chain=[EngineType.GEMINI],
        enabled_engines=list(EngineType),
        save_figures=False,
        quiet=True,
        tiered=False,  # Disable tiered routing in tests (avoids fitz.open on fake PDFs)
        # R174b: agentic is now the only path. This used to be agentic=False so the
        # deterministic backbone/score/repair tests could mock at the backbone level;
        # those tests and that lane are gone.
        agentic=True,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_engine_result(
    text: str = "This is a good OCR result with enough words to pass the audit heuristics check easily.",
    engine: str = "deepseek",
    status: DocumentStatus = DocumentStatus.SUCCESS,
    page_num: int = 0,
    failure_mode: FailureMode = FailureMode.NONE,
    audit_passed: bool = True,
) -> EngineResult:
    """Build an EngineResult with a single PageOutput."""
    return EngineResult(
        document_path=Path("/tmp/fake.pdf"),
        engine=engine,
        status=status,
        failure_mode=failure_mode,
        pages=[
            PageOutput(
                page_num=page_num,
                text=text,
                status=PageStatus.SUCCESS if status == DocumentStatus.SUCCESS else PageStatus.ERROR,
                engine=engine,
                audit_passed=audit_passed,
            )
        ],
        processing_time=1.0,
        audit_passed=audit_passed,
    )


def _setup_mock_engine(
    mock_engine: MagicMock,
    result: EngineResult | None = None,
    text: str = "",
    name: str = "deepseek",
) -> None:
    """Configure a mock engine to support both process_document and process_pages."""
    if result is None:
        result = _make_engine_result(text=text or _good_text(), engine=name)
    mock_engine.name = name
    mock_engine.is_available.return_value = True
    mock_engine.model_version = ""
    mock_engine.process_document.return_value = result

    page_text = result.markdown if result.success else ""
    page_status = PageStatus.SUCCESS if result.success else PageStatus.ERROR
    audit = result.audit_passed if result.success else False

    def _mock_process_pages(pdf_path, page_nums, config, dpi=200):
        return [
            PageOutput(
                page_num=pn,
                text=page_text,
                status=page_status,
                engine=name,
                audit_passed=audit,
            )
            for pn in page_nums
        ]

    mock_engine.process_pages.side_effect = _mock_process_pages


def _make_bd_assessment(
    page_count: int,
    born_digital_pages: set[int] | None = None,
    complex_pages: set[int] | None = None,
    table_pages: set[int] | None = None,
) -> DocumentAssessment:
    """Build a DocumentAssessment with specified born-digital pages.

    Args:
        page_count: Total number of pages.
        born_digital_pages: Set of 1-indexed page numbers that are born-digital.
        complex_pages: Subset of born_digital_pages that have complex content
            (tables/figures/equations) and need OCR enhancement.
        table_pages: Subset of born_digital_pages with table-like structure but
            otherwise clean native text.
    """
    bd = born_digital_pages or set()
    cx = complex_pages or set()
    tables = table_pages or set()
    pages = []
    for i in range(1, page_count + 1):
        is_bd = i in bd
        needs_enhancement = is_bd and i in cx
        has_tables = needs_enhancement or (is_bd and i in tables)
        pages.append(
            PageAssessment(
                page_num=i,
                is_born_digital=is_bd,
                native_text=f"Native text for page {i}" if is_bd else "",
                confidence=0.9,
                needs_ocr_enhancement=needs_enhancement,
                has_tables=has_tables,
            )
        )
    return DocumentAssessment(path=Path("/tmp/fake.pdf"), pages=pages)


def _good_text() -> str:
    """Text that passes heuristics (>50 words, no garbage)."""
    return (
        "This document presents an analysis of market dynamics across "
        "several European economies during the post-pandemic recovery "
        "period. We examine monetary policy transmission mechanisms and "
        "their effects on inflation expectations, output gaps, and "
        "financial stability indicators. Our empirical framework builds "
        "on vector autoregressive models with sign restrictions, "
        "estimated using Bayesian methods on quarterly macroeconomic "
        "data spanning the period from 2019 to 2024. The results "
        "suggest that unconventional monetary policy tools had "
        "asymmetric effects across core and peripheral economies."
    )


def _bad_text() -> str:
    """Text that fails heuristics (too few words)."""
    return "short"


# ---------------------------------------------------------------------------
# Phase 1: Analyze
# ---------------------------------------------------------------------------


class TestPhaseAnalyze:
    def test_born_digital_pages_marked_in_state(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(3))

        assessment = _make_bd_assessment(3, born_digital_pages={1, 3})
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = assessment

        pipeline._phase_analyze(state)

        assert state.pages[1].is_born_digital is True
        assert state.pages[1].native_text == "Native text for page 1"
        assert state.pages[2].is_born_digital is False
        assert state.pages[3].is_born_digital is True

    def test_born_digital_pages_skip_repair(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        assessment = _make_bd_assessment(2, born_digital_pages={1})
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = assessment

        pipeline._phase_analyze(state)

        # Page 1 is born-digital with text -> does not need repair
        assert not state.pages[1].needs_repair
        # Page 2 is scanned, no output yet -> needs repair
        assert state.pages[2].needs_repair

    def test_no_born_digital_pages(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        assessment = _make_bd_assessment(2, born_digital_pages=set())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = assessment

        pipeline._phase_analyze(state)

        assert not state.pages[1].is_born_digital
        assert not state.pages[2].is_born_digital


# ---------------------------------------------------------------------------
# Phase 2: Backbone OCR
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 3: Score
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 4: Selective Repair
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase 5: Assemble
# ---------------------------------------------------------------------------


class TestPhaseAssemble:
    def test_assemble_success(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        # Set up passing pages
        for i in range(1, 3):
            state.pages[i].best_output = PageOutput(
                page_num=i,
                text=f"Content for page {i}",
                status=PageStatus.SUCCESS,
                engine="deepseek",
                audit_passed=True,
            )

        result = pipeline._phase_assemble(state, Path("/tmp/out"))

        assert result.status == DocumentStatus.SUCCESS
        assert result.success
        assert "Content for page 1" in result.markdown
        assert "Content for page 2" in result.markdown

    def test_assemble_with_whole_doc(self, tmp_path: Path) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        # Only whole-doc output, no per-page
        whole_out = PageOutput(
            page_num=0,
            text="Full document text here",
            status=PageStatus.SUCCESS,
            engine="deepseek",
            audit_passed=True,
        )
        result = EngineResult(
            document_path=Path("/tmp/fake.pdf"),
            engine="deepseek",
            status=DocumentStatus.SUCCESS,
            pages=[whole_out],
        )
        state.apply_result(result)

        final = pipeline._phase_assemble(state, tmp_path)

        # A markerless whole-doc blob on a 2-page document covers only page 1
        # after splitting; page 2 ships an explicit failure marker and the
        # run is demoted from SUCCESS instead of silently recording a clean
        # pass with an empty page.
        assert final.status == DocumentStatus.AUDIT_FAILED
        assert "Full document text here" in final.markdown
        assert "[page 2 failed: no usable OCR output]" in final.markdown
        assert final.error and "page(s) 2" in final.error

    def test_assemble_empty_doc(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        result = pipeline._phase_assemble(state, Path("/tmp/out"))

        assert result.status == DocumentStatus.ERROR
        assert not result.success

    def test_assemble_saves_markdown(self, tmp_path: Path) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))

        state.pages[1].best_output = PageOutput(
            page_num=1,
            text="Hello world",
            status=PageStatus.SUCCESS,
            engine="deepseek",
            audit_passed=True,
        )

        result = pipeline._phase_assemble(state, tmp_path)

        stem = "fake"
        md_path = tmp_path / stem / f"{stem}.md"
        assert md_path.exists()
        # Canon body: '## Page N' header per page, splittable back to the text.
        from ocr_output_contract import assemble_pages, split_native_pages

        assert md_path.read_text() == assemble_pages(["Hello world"])
        assert split_native_pages(md_path.read_text()) == ["Hello world"]

    def test_assemble_writes_canonical_metadata(self, tmp_path: Path) -> None:
        """_phase_assemble writes per-doc + root metadata.json keyed by the
        input-relative path, validated against the contract conformance harness."""
        import json

        fitz = pytest.importorskip("fitz")
        from ocr_output_contract.conformance import ExpectedDoc, assert_conforms

        # A real PDF on disk so the checksum can be computed.
        pdf = tmp_path / "src" / "paper.pdf"
        pdf.parent.mkdir(parents=True)
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf))
        doc.close()

        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=DocumentHandle.from_path(pdf))
        state.pages[1].best_output = PageOutput(
            page_num=1,
            text="## Page 1\n\nHello canonical world",
            status=PageStatus.SUCCESS,
            engine="deepseek",
            audit_passed=True,
        )

        out_root = tmp_path / "out"
        pipeline._phase_assemble(state, out_root)

        # Per-doc sidecar exists beside the .md, keyed by the relative path.
        doc_meta = out_root / "paper" / "metadata.json"
        root_meta = out_root / "metadata.json"
        assert doc_meta.exists()
        assert root_meta.exists()

        root = json.loads(root_meta.read_text())
        assert "paper.pdf" in root["files"]  # input-relative key, not basename-mangled
        entry = root["files"]["paper.pdf"]
        assert entry["status"] == "completed"
        assert entry["checksum"].startswith("sha256:")
        assert entry["backend"] == "socr"

        # The output conforms to the canonical contract.
        assert_conforms(out_root, [ExpectedDoc(rel_key="paper.pdf", pages=1)])

    def test_assemble_born_digital_text_used(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        # Page 1 is born-digital
        state.pages[1].is_born_digital = True
        state.pages[1].native_text = "Native born-digital content"
        # Page 2 has OCR output
        state.pages[2].best_output = PageOutput(
            page_num=2,
            text="OCR content",
            status=PageStatus.SUCCESS,
            engine="deepseek",
            audit_passed=True,
        )

        result = pipeline._phase_assemble(state, Path("/tmp/out"))

        assert "Native born-digital content" in result.markdown
        assert "OCR content" in result.markdown

    def test_assemble_partial_failure(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(3))

        # Only page 1 succeeded
        state.pages[1].best_output = PageOutput(
            page_num=1,
            text="Only this page worked",
            status=PageStatus.SUCCESS,
            engine="deepseek",
            audit_passed=True,
        )
        # Pages 2 and 3 still need repair

        result = pipeline._phase_assemble(state, Path("/tmp/out"))

        # Has text but pages need repair -> AUDIT_FAILED
        assert result.status == DocumentStatus.AUDIT_FAILED
        assert "Only this page worked" in result.markdown


# ---------------------------------------------------------------------------
# Full pipeline (end-to-end with mocks)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_full_loop_success(self, tmp_path: Path) -> None:
        """Mock all externals and run the full agentic loop end to end."""
        config = _make_config(quiet=True, judge_backend="heuristic")
        pipeline = UnifiedPipeline(config)
        # R174b: agentic is the only lane now, so the provider ladder must be
        # patched or this passes locally (ollama present) and fails in CI.
        pipeline._available_engines_for_agentic = MagicMock(return_value=[PROFILE_QWEN_LOCAL])

        # Mock born-digital detection
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(3, born_digital_pages=set())

        # Mock engine
        good_result = _make_engine_result(text=_good_text())
        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result
        _setup_mock_engine(mock_engine, result=good_result, name=mock_engine.name)

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            with patch.object(DocumentHandle, "from_path") as mock_from_path:
                mock_from_path.return_value = _make_handle(3)
                result = pipeline.process(Path("/tmp/fake.pdf"), tmp_path)

        assert result.success
        assert result.pages_processed == 3

    def test_full_loop_born_digital_skip(self, tmp_path: Path) -> None:
        """Born-digital pages should skip OCR entirely."""
        config = _make_config(quiet=True)
        pipeline = UnifiedPipeline(config)

        # All pages are born-digital
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(2, born_digital_pages={1, 2})

        # Engine still runs (backbone processes whole doc)
        good_result = _make_engine_result(text=_good_text())
        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result
        _setup_mock_engine(mock_engine, result=good_result, name=mock_engine.name)

        # GH-318: this test's PDF path is a stand-in -- DocumentHandle is mocked,
        # but chart-eligibility detection opens the real path and raises
        # FileNotFoundError. That crash now flags the document (a detector that
        # cannot read the source never decided the page's routing), which is the
        # intended new behaviour and unrelated to what this test covers. Patch the
        # detector off so the test exercises born-digital skipping, not file IO.
        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            with patch.object(DocumentHandle, "from_path") as mock_from_path:
                with patch.object(pipeline, "_is_chart_asset_page", return_value=False):
                    mock_from_path.return_value = _make_handle(2)
                    result = pipeline.process(Path("/tmp/fake.pdf"), tmp_path)

        assert result.success
        # The born-digital pages don't need repair
        # Markdown should contain native text from born-digital pages


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    def test_batch_processes_all_pdfs(self, tmp_path: Path) -> None:
        # Create fake PDFs
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        for name in ["doc1.pdf", "doc2.pdf"]:
            (pdf_dir / name).write_bytes(b"%PDF-fake")

        config = _make_config(quiet=True)
        pipeline = UnifiedPipeline(config)

        # Mock everything
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(1, born_digital_pages=set())

        good_result = _make_engine_result(text=_good_text())
        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result
        _setup_mock_engine(mock_engine, result=good_result, name=mock_engine.name)

        out_dir = tmp_path / "output"

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            with patch.object(DocumentHandle, "from_path") as mock_from_path:
                mock_from_path.return_value = _make_handle(1)
                results = pipeline.process_batch(pdf_dir, out_dir)

        assert len(results) == 2

    def test_batch_empty_directory(self, tmp_path: Path) -> None:
        pdf_dir = tmp_path / "empty"
        pdf_dir.mkdir()

        config = _make_config(quiet=True)
        pipeline = UnifiedPipeline(config)

        results = pipeline.process_batch(pdf_dir, tmp_path / "output")
        assert results == []

    def test_batch_dry_run(self, tmp_path: Path) -> None:
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "doc.pdf").write_bytes(b"%PDF-fake")

        config = _make_config(quiet=True, dry_run=True)
        pipeline = UnifiedPipeline(config)

        results = pipeline.process_batch(pdf_dir, tmp_path / "output")
        assert results == []

    def test_batch_skips_already_processed(self, tmp_path: Path) -> None:
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "doc.pdf").write_bytes(b"%PDF-fake")

        out_dir = tmp_path / "output"

        config = _make_config(quiet=True)
        pipeline = UnifiedPipeline(config)

        # Mark as already processed via the canonical RootIndex resume gate
        # (MetadataManager no longer participates — RootIndex is the sole index).
        # process_batch imports RootIndex lazily from the contract package, so we
        # patch it at its source module.
        with patch("ocr_output_contract.RootIndex") as MockIndex:
            MockIndex.return_value.is_completed.return_value = True
            results = pipeline.process_batch(pdf_dir, out_dir)

        assert results == []


# ---------------------------------------------------------------------------
# Max retries limiting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Figure extraction
# ---------------------------------------------------------------------------


class TestFigures:
    def test_figures_extracted_when_enabled(self, tmp_path: Path) -> None:
        config = _make_config(save_figures=True)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))

        state.pages[1].best_output = PageOutput(
            page_num=1,
            text="Content",
            status=PageStatus.SUCCESS,
            engine="deepseek",
            audit_passed=True,
        )

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:
            mock_extractor = MockExtractor.return_value
            mock_extractor.extract.return_value = ExtractionResult(figures=[], cap_reached=False)

            result = pipeline._phase_assemble(state, tmp_path)

        MockExtractor.assert_called_once()

    def test_scanned_pages_excluded_from_figure_extraction(self, tmp_path: Path) -> None:
        """Issue #42: the assessment's scanned pages are passed as skips."""
        config = _make_config(save_figures=True)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        for n in (1, 2):
            state.pages[n].best_output = PageOutput(
                page_num=n,
                text="Content",
                status=PageStatus.SUCCESS,
                engine="deepseek",
                audit_passed=True,
            )
        pipeline._last_assessment = DocumentAssessment(
            path=state.handle.path,
            pages=[
                PageAssessment(page_num=1, is_born_digital=True, native_text="x", confidence=1.0),
                PageAssessment(page_num=2, is_born_digital=False, native_text="", confidence=1.0),
            ],
        )

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:
            mock_extractor = MockExtractor.return_value
            mock_extractor.extract.return_value = ExtractionResult(figures=[], cap_reached=False)

            pipeline._phase_assemble(state, tmp_path)

        mock_extractor.extract.assert_called_once_with(state.handle.path, skip_pages={2})

    def test_figures_skipped_when_disabled(self, tmp_path: Path) -> None:
        config = _make_config(save_figures=False)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))

        state.pages[1].best_output = PageOutput(
            page_num=1,
            text="Content",
            status=PageStatus.SUCCESS,
            engine="deepseek",
            audit_passed=True,
        )

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:
            result = pipeline._phase_assemble(state, tmp_path)

        MockExtractor.assert_not_called()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Truncation retry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-page backbone (replaced chunked backbone)
# Chunking is gone — all pages go through process_pages.
# These tests are intentionally minimal.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-engine mode
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Figure description and embedding
# ---------------------------------------------------------------------------


class TestDescribeAndEmbedFigures:
    """Tests for _describe_and_embed_figures (Problem 2)."""

    def _make_pipeline(self, **overrides) -> UnifiedPipeline:
        return UnifiedPipeline(_make_config(save_figures=True, **overrides))

    def test_no_figures_returns_text_unchanged(self, tmp_path: Path) -> None:
        pipeline = self._make_pipeline()
        state = DocumentState(handle=_make_handle(1))
        result = _make_engine_result(text=_good_text())

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path, "Some text")

        assert out == "Some text"
        assert result.figures == []

    def test_figures_without_vision_engine(self, tmp_path: Path) -> None:
        """Without GEMINI_API_KEY, figures are saved but not described."""
        pipeline = self._make_pipeline()
        state = DocumentState(handle=_make_handle(1))
        result = _make_engine_result(text=_good_text())

        fig_path = tmp_path / "test_doc" / "figures" / "figure_1_page1.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig_path.write_bytes(b"\x89PNG")

        mock_fig = ExtractedFigure(
            figure_num=1,
            page_num=1,
            image=None,
            saved_path=str(fig_path),
        )

        with (
            patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor,
            patch.dict("os.environ", {}, clear=False),
            patch.object(pipeline, "_get_vision_engine", return_value=None),
        ):
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(
                state,
                result,
                tmp_path,
                "OCR text",
            )

        assert len(result.figures) == 1
        assert result.figures[0].description == ""
        # Figure block appended with image ref
        assert "**Figure 1** (page 1)" in out
        assert "![Figure 1]" in out

    def test_figures_with_vision_engine(self, tmp_path: Path) -> None:
        """With a vision engine and --describe-figures, figures are described."""
        pipeline = self._make_pipeline(describe_figures=True)
        state = DocumentState(handle=_make_handle(1))
        result = _make_engine_result(text=_good_text())

        fig_dir = tmp_path / "test_doc" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / "figure_1_page1.png"
        fig_path.write_bytes(b"\x89PNG")

        mock_image = MagicMock()
        mock_fig = ExtractedFigure(
            figure_num=1,
            page_num=1,
            image=mock_image,
            saved_path=str(fig_path),
        )

        mock_engine = MagicMock()
        mock_engine.name = "gemini-api"
        mock_engine.describe_figure.return_value = FigureInfo(
            figure_num=0,
            page_num=0,
            figure_type="chart",
            description="A bar chart showing quarterly revenue.",
            engine="gemini-api",
        )

        with (
            patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor,
            patch.object(pipeline, "_get_vision_engine", return_value=mock_engine),
        ):
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(
                state,
                result,
                tmp_path,
                "OCR text",
            )

        assert len(result.figures) == 1
        assert result.figures[0].description == "A bar chart showing quarterly revenue."
        assert result.figures[0].figure_type == "chart"
        assert "**Figure 1** (page 1): A bar chart showing quarterly revenue." in out
        assert "![Figure 1]" in out
        mock_engine.close.assert_called_once()

    # GH-47A: cap-reached produces a durable audit event
    def test_cap_reached_emits_audit_event_and_console_warning(self, tmp_path: Path) -> None:
        """GH-47A AC1: cap_reached=True must append a figure_cap_reached AuditEvent
        to state.events (durable audit) AND emit a yellow console warning.

        Uses the normal (non-quiet) path so both side-effects are observable.
        """
        from socr.core.audit_log import AuditEvent

        pipeline = self._make_pipeline(quiet=False)  # non-quiet: console warning must fire
        state = DocumentState(handle=_make_handle(1))
        result = _make_engine_result(text=_good_text())

        mock_console = MagicMock()
        with (
            patch("socr.pipeline.orchestrator.FigureExtractor") as mock_extractor_cls,
            patch("socr.pipeline.orchestrator.console", mock_console),
        ):
            mock_extractor_cls.return_value.extract.return_value = ExtractionResult(
                figures=[], cap_reached=True
            )
            pipeline._describe_and_embed_figures(state, result, tmp_path, "Some text")

        # Durable audit: at least one AuditEvent with kind="figure_cap_reached"
        cap_events = [
            e for e in state.events if isinstance(e, AuditEvent) and e.kind == "figure_cap_reached"
        ]
        assert cap_events, (
            "Expected a figure_cap_reached AuditEvent in state.events when "
            "cap_reached=True, but none was found."
        )
        assert cap_events[0].data.get("figures_max_total") is not None

        # Console warning: a yellow warning must have been printed
        printed_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("yellow" in c or "cap" in c.lower() for c in printed_calls), (
            "Expected a yellow console warning about the figure cap, but none was emitted."
        )

    def test_cap_reached_quiet_suppresses_console_but_keeps_audit_event(
        self, tmp_path: Path
    ) -> None:
        """GH-47A AC1 (quiet path): AuditEvent is still appended even when quiet=True,
        but the console.print call is suppressed.
        """
        from socr.core.audit_log import AuditEvent

        pipeline = self._make_pipeline(quiet=True)
        state = DocumentState(handle=_make_handle(1))
        result = _make_engine_result(text=_good_text())

        mock_console = MagicMock()
        with (
            patch("socr.pipeline.orchestrator.FigureExtractor") as mock_extractor_cls,
            patch("socr.pipeline.orchestrator.console", mock_console),
        ):
            mock_extractor_cls.return_value.extract.return_value = ExtractionResult(
                figures=[], cap_reached=True
            )
            pipeline._describe_and_embed_figures(state, result, tmp_path, "Some text")

        # Durable audit event must still be present even in quiet mode
        cap_events = [
            e for e in state.events if isinstance(e, AuditEvent) and e.kind == "figure_cap_reached"
        ]
        assert cap_events, "figure_cap_reached AuditEvent must be appended even when quiet=True."

        # Console warning must be suppressed
        printed_calls = [str(c) for c in mock_console.print.call_args_list]
        assert not any("yellow" in c for c in printed_calls), (
            "Console yellow warning must be suppressed when quiet=True."
        )


# ---------------------------------------------------------------------------
# GH-50: Save-figures / describe-figures split (acceptance criteria)
# ---------------------------------------------------------------------------


def _make_mock_figure(tmp_path: Path, fig_num: int = 1, page_num: int = 1) -> "ExtractedFigure":
    """Helper: a synthetic ExtractedFigure with a saved PNG."""
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"figure_{fig_num}_page{page_num}.png"
    fig_path.write_bytes(b"\x89PNG")
    return ExtractedFigure(
        figure_num=fig_num,
        page_num=page_num,
        image=MagicMock(),
        saved_path=str(fig_path),
    )


class TestSaveFiguresNoVLM:
    """GH-50 AC1: ``--save-figures`` writes PNGs + image refs but NO caption prose."""

    def test_save_figures_alone_produces_no_description(self, tmp_path: Path) -> None:
        """``save_figures=True, describe_figures=False`` must not call the VLM."""
        config = _make_config(save_figures=True, describe_figures=False)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))
        result = _make_engine_result(text=_good_text())
        mock_fig = _make_mock_figure(tmp_path)

        with (
            patch("socr.pipeline.orchestrator.FigureExtractor") as mock_extractor_cls,
            patch.object(pipeline, "_get_vision_engine") as mock_get_vision,
        ):
            mock_extractor_cls.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path, "OCR text")

        # VLM engine must never have been asked for
        mock_get_vision.assert_not_called()
        # Image reference block is present
        assert "![Figure 1]" in out
        # No description prose attached to the block
        assert result.figures[0].description == ""

    def test_save_figures_alone_produces_image_ref_block(self, tmp_path: Path) -> None:
        """Image-ref markdown is still appended even without captions."""
        config = _make_config(save_figures=True, describe_figures=False)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))
        result = _make_engine_result(text=_good_text())
        mock_fig = _make_mock_figure(tmp_path)

        with patch("socr.pipeline.orchestrator.FigureExtractor") as mock_extractor_cls:
            mock_extractor_cls.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path, "Body text")

        assert "**Figure 1** (page 1)" in out
        assert "![Figure 1]" in out
        # Ensure the OCR body text is preserved
        assert "Body text" in out


class TestDescribeFiguresVLM:
    """GH-50 AC2: ``--describe-figures`` calls the VLM caption engine."""

    def test_describe_figures_calls_vision_engine(self, tmp_path: Path) -> None:
        """``describe_figures=True`` must call ``_get_vision_engine()``."""
        config = _make_config(save_figures=True, describe_figures=True)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))
        result = _make_engine_result(text=_good_text())
        mock_fig = _make_mock_figure(tmp_path)

        mock_engine = MagicMock()
        mock_engine.name = "gemini-api"
        mock_engine.describe_figure.return_value = FigureInfo(
            figure_num=1,
            page_num=1,
            figure_type="chart",
            description="GDP growth over five years.",
            engine="gemini-api",
        )

        with (
            patch("socr.pipeline.orchestrator.FigureExtractor") as mock_extractor_cls,
            patch.object(pipeline, "_get_vision_engine", return_value=mock_engine),
        ):
            mock_extractor_cls.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path, "OCR text")

        mock_engine.describe_figure.assert_called_once()
        assert result.figures[0].description == "GDP growth over five years."
        assert "GDP growth over five years." in out

    def test_describe_figures_false_skips_vision_engine(self, tmp_path: Path) -> None:
        """``describe_figures=False`` must not instantiate the VLM at all."""
        config = _make_config(save_figures=True, describe_figures=False)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))
        result = _make_engine_result(text=_good_text())
        mock_fig = _make_mock_figure(tmp_path)

        called = []
        with patch("socr.pipeline.orchestrator.FigureExtractor") as mock_extractor_cls:
            mock_extractor_cls.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            with patch.object(
                pipeline,
                "_get_vision_engine",
                side_effect=lambda: called.append(1) or None,
            ):
                pipeline._describe_and_embed_figures(state, result, tmp_path, "OCR text")

        assert not called, "_get_vision_engine must not be called when describe_figures=False"


class TestFigurePhasePreservesOCRText:
    """GH-50 AC3: caption-phase failure must not destroy already-written OCR text."""

    def test_caption_failure_preserves_ocr_markdown(self, tmp_path: Path) -> None:
        """A crash in the caption sub-loop must not clobber the saved .md."""
        config = _make_config(save_figures=True, describe_figures=True)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))
        good = PageOutput(
            page_num=1,
            text="Precious OCR content.",
            status=PageStatus.SUCCESS,
            engine="deepseek",
            audit_passed=True,
        )
        state.pages[1].attempts.append(good)
        state.pages[1].best_output = good

        # _describe_and_embed_figures raises mid-caption
        with patch.object(
            pipeline,
            "_describe_and_embed_figures",
            side_effect=RuntimeError("VLM API timed out"),
        ):
            result = pipeline._phase_assemble(state, tmp_path)

        # The OCR text must be on disk
        md_files = list(tmp_path.rglob("*.md"))
        assert md_files, "markdown file must exist after caption-phase crash"
        assert "Precious OCR content." in md_files[0].read_text()
        # And returned in the result
        assert "Precious OCR content." in result.markdown


class TestDescribeFiguresResumeFingerprint:
    """GH-50 AC4: resume fingerprint invalidates when describe_figures changes."""

    def _config_fingerprint(self, **kwargs) -> str:
        cfg = _make_config(**kwargs)
        pipeline = UnifiedPipeline(cfg)
        return pipeline._run_fingerprint()

    def test_describe_figures_changes_fingerprint(self) -> None:
        """Toggling describe_figures must produce a different fingerprint."""
        fp_off = self._config_fingerprint(save_figures=True, describe_figures=False)
        fp_on = self._config_fingerprint(save_figures=True, describe_figures=True)
        assert fp_off != fp_on, (
            "fingerprint must differ when describe_figures changes "
            "so the resume gate reprocesses the doc"
        )

    def test_save_figures_still_changes_fingerprint(self) -> None:
        """``save_figures`` alone still invalidates the fingerprint (regression guard)."""
        fp_off = self._config_fingerprint(save_figures=False, describe_figures=False)
        fp_on = self._config_fingerprint(save_figures=True, describe_figures=False)
        assert fp_off != fp_on


class TestCLIDescribeFiguresFlag:
    """GH-50: CLI wiring — --describe-figures implies --save-figures."""

    def test_describe_figures_implies_save_figures(self) -> None:
        from socr.cli import build_config

        cfg = build_config(describe_figures=True, save_figures=False)
        assert cfg.describe_figures is True
        assert cfg.save_figures is True, "--describe-figures must imply --save-figures"

    def test_save_figures_alone_does_not_enable_describe(self) -> None:
        from socr.cli import build_config

        cfg = build_config(save_figures=True, describe_figures=False)
        assert cfg.save_figures is True
        assert cfg.describe_figures is False

    def test_neither_flag_leaves_both_false(self) -> None:
        from socr.cli import build_config

        cfg = build_config(save_figures=False, describe_figures=False)
        assert cfg.save_figures is False
        assert cfg.describe_figures is False


class TestBuildFigureBlocks:
    """Tests for _build_figure_blocks static method."""

    def test_single_figure_with_description(self, tmp_path: Path) -> None:
        fig_path = tmp_path / "figures" / "figure_1_page2.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig_path.write_bytes(b"\x89PNG")

        figures = [
            FigureInfo(
                figure_num=1,
                page_num=2,
                figure_type="chart",
                description="A line chart.",
                image_path=str(fig_path),
            )
        ]
        result = UnifiedPipeline._build_figure_blocks(figures, tmp_path)
        assert "**Figure 1** (page 2): A line chart." in result
        assert "![Figure 1](figures/figure_1_page2.png)" in result

    def test_figure_without_description(self, tmp_path: Path) -> None:
        fig_path = tmp_path / "figures" / "figure_1_page1.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig_path.write_bytes(b"\x89PNG")

        figures = [
            FigureInfo(
                figure_num=1,
                page_num=1,
                figure_type="extracted",
                description="",
                image_path=str(fig_path),
            )
        ]
        result = UnifiedPipeline._build_figure_blocks(figures, tmp_path)
        assert "**Figure 1** (page 1)" in result
        assert ": " not in result.split("\n")[0]  # no description suffix

    def test_no_image_path_skipped(self, tmp_path: Path) -> None:
        figures = [
            FigureInfo(
                figure_num=1,
                page_num=1,
                figure_type="extracted",
                description="desc",
                image_path=None,
            )
        ]
        result = UnifiedPipeline._build_figure_blocks(figures, tmp_path)
        assert result == ""


class TestPhantomImageStrippingInAssemble:
    """Verify phantom images are stripped during _phase_assemble."""

    def test_phantom_refs_stripped_in_assemble(self, tmp_path: Path) -> None:
        config = _make_config(save_figures=False)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))

        # Simulate a whole-doc attempt with phantom image refs
        text_with_phantoms = (
            "Some OCR text\n\n"
            "![img](img-0.jpeg)\n\n"
            "More text\n\n"
            "![Page 1](./extracted_images/page1.png)"
        )
        page_out = PageOutput(
            page_num=0,
            text=text_with_phantoms,
            status=PageStatus.SUCCESS,
            engine="gemini",
            audit_passed=True,
        )
        result = _make_engine_result(text=text_with_phantoms)
        state.apply_result(result)

        final = pipeline._phase_assemble(state, tmp_path)
        # Phantom refs should be gone
        assert "![img]" not in final.markdown
        assert "![Page 1]" not in final.markdown
        assert "Some OCR text" in final.markdown
        assert "More text" in final.markdown


# ---------------------------------------------------------------------------
# Native-first pipeline
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tier-2 per-page escalation
# ---------------------------------------------------------------------------


class TestTieredEscalation:
    """Per-page audit scoring inside _backbone_native_first.

    When local (Tier 2) outputs fail the per-page heuristic audit, those
    pages should be removed from local results and re-routed to the cloud
    engine (Tier 3).
    """

    def _setup(
        self,
        page_count: int = 5,
        scanned_pages: set[int] | None = None,
    ) -> tuple["UnifiedPipeline", "DocumentState"]:
        """Tiered pipeline with all-scanned or custom pages."""
        from socr.core.config import EngineType

        config = _make_config(
            quiet=True,
            native_first=True,
            tiered=True,
            primary_engine=EngineType.GEMINI,
            local_engine=EngineType.GLM,
        )
        pipeline = UnifiedPipeline(config)
        # All pages are scanned (not born-digital) so they all go to OCR
        scanned = scanned_pages or set(range(1, page_count + 1))
        assessment = _make_bd_assessment(page_count, born_digital_pages=set())
        state = DocumentState(handle=_make_handle(page_count))
        state.apply_born_digital(assessment)
        pipeline._last_assessment = assessment
        return pipeline, state

    def _mock_classify_pages(self, easy: set[int], hard: set[int]):
        """Return a side_effect for classify_pages that splits pages."""
        from socr.core.difficulty import DifficultyAssessment, PageDifficulty

        def _classify(pdf_path, page_nums, page_hints=None):
            result = {}
            for pn in page_nums:
                diff = PageDifficulty.EASY if pn in easy else PageDifficulty.HARD
                result[pn] = DifficultyAssessment(page_num=pn, difficulty=diff, reasons=[])
            return result

        return _classify


# ---------------------------------------------------------------------------
# Default-path parity (TICKET-12): characterize UnifiedPipeline as the CLI
# default before StandardPipeline is deleted. Pins the behaviors codex flagged:
# scanned docs OCR + write output, prose-only docs do ZERO OCR, and an
# unavailable engine yields an ERROR result (not a crash / not silent success).
# ---------------------------------------------------------------------------


class TestDefaultPathParity:
    """Full process() runs over a real tiny PDF with a mocked engine."""

    @staticmethod
    def _real_pdf(tmp_path, n_pages=2):
        fitz = pytest.importorskip("fitz")
        path = tmp_path / "paper.pdf"
        doc = fitz.open()
        for i in range(n_pages):
            doc.new_page().insert_text((72, 72), f"page {i + 1}")
        doc.save(str(path))
        doc.close()
        return path

    def test_default_scanned_doc_ocrs_and_writes_output(self, tmp_path):
        pdf = self._real_pdf(tmp_path, n_pages=2)
        config = _make_config(primary_engine=EngineType.DEEPSEEK, judge_backend="heuristic")
        pipeline = UnifiedPipeline(config)
        # R174b: agentic is the only lane now, so the provider ladder must be
        # patched or this passes locally (ollama present) and fails in CI.
        pipeline._available_engines_for_agentic = MagicMock(return_value=[PROFILE_QWEN_LOCAL])
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            2,
            born_digital_pages=set(),  # all scanned -> must OCR
        )
        mock_engine = MagicMock()
        _setup_mock_engine(mock_engine, name="deepseek")

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            result = pipeline.process(pdf, tmp_path)

        assert result.status == DocumentStatus.SUCCESS
        md = tmp_path / "paper" / "paper.md"
        assert md.exists() and md.read_text(encoding="utf-8").strip()
        mock_engine.process_pages.assert_called()  # OCR actually ran

    def test_default_prose_only_skips_ocr(self, tmp_path):
        """The fast-path invariant codex required: prose-only born-digital docs
        do zero OCR work and never call an engine."""
        pdf = self._real_pdf(tmp_path, n_pages=2)
        config = _make_config(primary_engine=EngineType.DEEPSEEK)
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            2,
            born_digital_pages={1, 2},
            complex_pages=set(),  # prose only
        )
        mock_engine = MagicMock()
        _setup_mock_engine(mock_engine, name="deepseek")

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            result = pipeline.process(pdf, tmp_path)

        assert result.status == DocumentStatus.SUCCESS
        mock_engine.process_pages.assert_not_called()
        mock_engine.process_document.assert_not_called()
        assert "Native text for page 1" in (tmp_path / "paper" / "paper.md").read_text()

    def test_default_engine_unavailable_returns_error(self, tmp_path):
        pdf = self._real_pdf(tmp_path, n_pages=1)
        config = _make_config(primary_engine=EngineType.DEEPSEEK)
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            1,
            born_digital_pages=set(),  # scanned -> needs the (unavailable) engine
        )
        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = False

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            result = pipeline.process(pdf, tmp_path)

        assert result.status == DocumentStatus.ERROR


# ---------------------------------------------------------------------------
# Agentic cost-aware routing integration (config.agentic=True). The judge is
# forced to "heuristic" so tests are deterministic and never touch Ollama.
# ---------------------------------------------------------------------------


def _mock_engine_named(name: str, text: str, ok: bool = True) -> MagicMock:
    m = MagicMock()
    m.name = name
    m.is_available.return_value = True
    m.model_version = ""

    def _pp(pdf_path, page_nums, config, dpi=200):
        status = PageStatus.SUCCESS if ok else PageStatus.ERROR
        return [
            PageOutput(page_num=pn, text=text, status=status, engine=name, audit_passed=ok)
            for pn in page_nums
        ]

    m.process_pages.side_effect = _pp
    return m


class TestAgenticIntegration:
    @staticmethod
    def _real_pdf(tmp_path, n_pages=1):
        fitz = pytest.importorskip("fitz")
        path = tmp_path / "paper.pdf"
        doc = fitz.open()
        for i in range(n_pages):
            doc.new_page().insert_text((72, 72), f"page {i + 1}")
        doc.save(str(path))
        doc.close()
        return path

    def test_agentic_records_cost_and_writes_manifest(self, tmp_path):
        pdf = self._real_pdf(tmp_path, 1)
        config = _make_config(
            agentic=True, judge_backend="heuristic", enabled_engines=[EngineType.GEMINI]
        )
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(1, born_digital_pages=set())
        eng = _mock_engine_named("gemini", _good_text())

        with patch("socr.pipeline.orchestrator.get_engine", return_value=eng):
            result = pipeline.process(pdf, tmp_path)

        assert result.status == DocumentStatus.SUCCESS
        # Gemini is $0.0002/page and one page was routed -> cost recorded.
        assert result.cost == pytest.approx(0.0002)
        # agentic mode writes a replayable manifest by default.
        doc_dir = tmp_path / "paper"
        assert (doc_dir / "manifest.json").exists()
        assert (doc_dir / "cache").is_dir()

    def test_agentic_escalates_from_cheap_to_cloud(self, tmp_path):
        pdf = self._real_pdf(tmp_path, 1)
        config = _make_config(
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GLM, EngineType.GEMINI],
        )
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(1, born_digital_pages=set())

        glm = _mock_engine_named("glm", "too short")  # fails heuristic min words
        gemini = _mock_engine_named("gemini", _good_text())  # passes

        def _get(engine_type):
            return glm if engine_type == EngineType.GLM else gemini

        with patch("socr.pipeline.orchestrator.get_engine", side_effect=_get):
            result = pipeline.process(pdf, tmp_path)

        assert result.status == DocumentStatus.SUCCESS
        # GLM (free) was tried and rejected, escalated to Gemini ($0.0002).
        assert result.cost == pytest.approx(0.0002)
        assert _good_text()[:25] in result.markdown  # gemini's output won

    def test_agentic_corrupt_math_flag_uses_region_recovery_not_whole_page(self, tmp_path):
        """GH-271: the opt-in crop guardrail must be live in agentic mode.

        The paired runs differ only in ``recover_corrupt_math``.  With the flag
        off, the known-deficient native page follows the existing whole-page
        ladder and falls back to native when the judge rejects it.  With the flag
        on, only the corrupt equation region is re-read; surrounding native prose
        remains in the shipped hybrid output.
        """
        from socr.core.providers import PROFILE_QWEN_LOCAL
        from socr.pipeline.agentic import AcceptDecision

        pdf = self._real_pdf(tmp_path, 1)
        native = "Clean prose before.\nPðA or BÞ ¼ PðAÞ þ PðBÞ\nClean prose after."
        assessment = DocumentAssessment(
            path=pdf,
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text=native,
                    confidence=0.9,
                    needs_ocr_enhancement=True,
                    has_equations=True,
                    has_corrupt_math=True,
                    word_count=100,
                )
            ],
        )

        class _RejectJudge:
            def assess(self, output, provider):
                return AcceptDecision(accept=False, reason="paired fixture rejection")

        outcomes = {}
        for enabled in (False, True):
            run_dir = tmp_path / ("on" if enabled else "off")
            crop_path = run_dir / pdf.stem / "equations" / "corrupt_math_p00001_r001.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop_path.write_bytes(b"retained crop fixture")
            config = _make_config(
                agentic=True,
                native_first=True,
                recover_corrupt_math=enabled,
                ollama_host="http://math-host.test:11434",
                judge_backend="heuristic",
                enabled_engines=[EngineType.QWEN],
            )
            pipeline = UnifiedPipeline(config)
            pipeline.bd_detector = MagicMock()
            pipeline.bd_detector.detect.return_value = assessment
            pipeline._available_engines_for_agentic = MagicMock(return_value=[PROFILE_QWEN_LOCAL])
            pipeline._build_page_judge = MagicMock(return_value=_RejectJudge())
            whole_page = _mock_engine_named("qwen", "rejected whole-page candidate")

            with (
                patch("socr.pipeline.orchestrator.get_engine", return_value=whole_page),
                patch(
                    "socr.math.recover.recover_math_regions",
                    return_value=[
                        CorruptMathRegion(
                            rect=object(),
                            source_text="PðA or BÞ ¼ PðAÞ þ PðBÞ",
                            crop_path="equations/corrupt_math_p00001_r001.png",
                            raw_latex=r"P(A \text{ or } B) = P(A) + P(B)",
                            validation_ok=True,
                            validation_reason="ok",
                            model_id=config.math_model,
                            attempts=1,
                        )
                    ],
                ) as recover_regions,
                patch.object(pipeline, "_resolve_crop_vlm_model", return_value=None),
                patch.object(pipeline, "_resolve_judge_model", return_value=""),
            ):
                result = pipeline.process(pdf, run_dir)

            outcomes[enabled] = (result, whole_page, recover_regions, run_dir)

        off_result, off_engine, off_recover, _off_dir = outcomes[False]
        on_result, on_engine, on_recover, on_dir = outcomes[True]

        off_engine.process_pages.assert_called_once()
        off_recover.assert_not_called()
        assert native in off_result.markdown

        on_engine.process_pages.assert_not_called()
        on_recover.assert_called_once()
        assert on_recover.call_args.kwargs["host"] == "http://math-host.test:11434"
        assert on_result.status == DocumentStatus.AUDIT_FAILED
        assert on_result.audit_passed is False
        assert on_result.engine == "native+math"
        assert on_result.cost is None
        assert on_result.error == "corrupt equation candidate unverified on page(s) 1"
        assert len(on_result.pages) == 1
        assert on_result.pages[0].page_num == 0
        assert "Clean prose before." in on_result.markdown
        assert "Clean prose after." in on_result.markdown
        assert r"P(A \text{ or } B) = P(A) + P(B)" in on_result.markdown
        assert "syntax only, non-authoritative" in on_result.markdown
        assert "rejected whole-page candidate" not in on_result.markdown
        assert "¼" not in on_result.markdown and "ð" not in on_result.markdown

        import json

        doc_dir = on_dir / pdf.stem
        sidecar = json.loads((doc_dir / "pages" / "00001.json").read_text())
        winner = sidecar["winning_output"]
        fragment = (doc_dir / "pages" / "00001.md").read_text()
        assert sidecar["terminal"] is True
        assert sidecar["status"] == PageStatus.WARNING.value
        assert winner["text"] == fragment
        assert winner["status"] == PageStatus.WARNING.value
        assert winner["audit_passed"] is False
        assert winner["engine"] == "native+math"
        assert winner["provider_id"] == "corrupt-math-region"
        assert winner["provider_model"] == config.math_model
        assert winner["cost_usd"] is None
        # Cold review round 6: this lane journals a page EngineResult with an
        # UNKNOWN cost, so the page's recorded spend must be unknown too. The
        # default 0.0 would persist a KNOWN zero and let a resumed run treat
        # unmetered spend as no spend at all.
        assert on_result.cost is None, "control: the live document total is unknown"
        assert sidecar["page_cost_usd"] is None
        manifest = json.loads((doc_dir / "manifest.json").read_text())
        fingerprint = manifest["entries"]["1"]["fingerprint"]

        assert fingerprint["engine"] == "native+math"
        assert fingerprint["model_version"] == config.math_model
        assert fingerprint["prompt_hash"] == run_fingerprint(
            config.math_model,
            "ollama-compatible",
            None,
            CORRUPT_MATH_PROMPT,
        )
        assert {event["kind"] for event in sidecar["audit_events"]} >= {
            "corrupt_math_region_recovery",
            "corrupt_math_hybrid_shipped",
        }
        shipped_event = next(
            event
            for event in sidecar["audit_events"]
            if event["kind"] == "corrupt_math_hybrid_shipped"
        )
        assert shipped_event["data"] == {
            "provider_id": "corrupt-math-region",
            "provider_model": config.math_model,
            "provider_backend": "ollama-compatible",
            "cost_usd": None,
            "crop_paths": ["equations/corrupt_math_p00001_r001.png"],
            "audit_passed": False,
        }
        assert "rejected whole-page candidate" not in fragment

        resume_state = DocumentState(handle=DocumentHandle.from_path(pdf))
        resume_state.apply_born_digital(assessment)
        assert pipeline._load_terminal_page(resume_state, 1, on_dir) is None

    @pytest.mark.parametrize("render_fails", [False, True], ids=["chart-saved", "chart-failed"])
    def test_agentic_corrupt_math_chart_overlap_preserves_or_surfaces_chart(
        self,
        tmp_path,
        render_fails,
    ):
        import json

        from socr.core.providers import PROFILE_QWEN_LOCAL

        pdf = self._real_pdf(tmp_path, 1)
        native = "Clean prose.\nPðA or BÞ ¼ PðAÞ þ PðBÞ"
        assessment = DocumentAssessment(
            path=pdf,
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text=native,
                    confidence=0.9,
                    needs_ocr_enhancement=True,
                    has_equations=True,
                    has_corrupt_math=True,
                    word_count=100,
                )
            ],
        )
        config = _make_config(
            agentic=True,
            native_first=True,
            recover_corrupt_math=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.QWEN],
        )
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = assessment
        pipeline._available_engines_for_agentic = MagicMock(return_value=[PROFILE_QWEN_LOCAL])
        chart_path = tmp_path / pdf.stem / "figures" / "chart-page.png"
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        chart_path.write_bytes(b"retained chart fixture")
        render_chart = (
            MagicMock(side_effect=OSError("chart directory unavailable"))
            if render_fails
            else MagicMock(return_value=chart_path)
        )
        region = CorruptMathRegion(
            rect=object(),
            source_text="PðA or BÞ ¼ PðAÞ þ PðBÞ",
            crop_path="equations/corrupt_math_p00001_r001.png",
            raw_latex=r"P(A \text{ or } B) = P(A) + P(B)",
            validation_ok=True,
            validation_reason="ok",
            model_id=config.math_model,
            attempts=1,
        )

        with (
            patch("socr.math.recover.recover_math_regions", return_value=[region]),
            patch.object(pipeline, "_is_chart_asset_page", return_value=True),
            patch.object(pipeline, "_render_chart_page_png", render_chart),
            patch.object(pipeline, "_resolve_crop_vlm_model", return_value=None),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            result = pipeline.process(pdf, tmp_path)

        sidecar = json.loads((tmp_path / pdf.stem / "pages" / "00001.json").read_text())
        arbitration = next(
            event for event in sidecar["audit_events"] if event["kind"] == "chart_math_arbitration"
        )
        expected_ref = "" if render_fails else "![Chart page 1](figures/chart-page.png)"
        assert arbitration["data"] == {
            "winner": "native+math" if render_fails else "native+math+chart_asset",
            "chart_png_rendered": not render_fails,
            "chart_png_path": expected_ref,
        }
        if render_fails:
            assert "![Chart page 1]" not in result.markdown
            assert any(
                event["kind"] == "chart_asset_render_failed" for event in sidecar["audit_events"]
            )
        else:
            assert expected_ref in result.markdown

    def test_combined_legacy_engine_keeps_corrupt_math_fingerprint(self, tmp_path):
        pdf = self._real_pdf(tmp_path, 1)
        state = DocumentState(handle=DocumentHandle.from_path(pdf))
        state.record_engine_run(
            EngineResult(
                document_path=pdf,
                engine="native+math+qwen",
                status=DocumentStatus.AUDIT_FAILED,
            )
        )
        pipeline = UnifiedPipeline(_make_config(math_model="fixture-math-model"))

        inputs = pipeline._fingerprint_inputs(state)

        assert inputs["native+math"] == (
            "fixture-math-model",
            "ollama-compatible",
            None,
            CORRUPT_MATH_PROMPT,
        )

    @pytest.mark.parametrize(
        ("config_overrides", "reason_fragment"),
        [
            ({"strict_local": True}, "strict-local forbids remote model"),
            ({"max_cost_per_page": 0.01}, "has no configured price"),
            ({"cost_budget": 0.01}, "has no configured price"),
        ],
        ids=["strict-local", "page-cost-cap", "document-cost-cap"],
    )
    def test_agentic_corrupt_math_remote_model_policy_is_visible(
        self,
        tmp_path,
        config_overrides,
        reason_fragment,
    ):
        """GH-271: direct equation calls obey local-only and dollar policies."""
        import json

        pdf = self._real_pdf(tmp_path, 1)
        native = "Native prose.\nPðA or BÞ ¼ PðAÞ þ PðBÞ"
        assessment = DocumentAssessment(
            path=pdf,
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text=native,
                    confidence=0.9,
                    needs_ocr_enhancement=True,
                    has_equations=True,
                    has_corrupt_math=True,
                    word_count=100,
                )
            ],
        )
        config_values = {
            "agentic": True,
            "native_first": True,
            "recover_corrupt_math": True,
            "judge_backend": "heuristic",
            "enabled_engines": [],
        }
        config_values.update(config_overrides)
        config = _make_config(**config_values)
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = assessment
        pipeline._available_engines_for_agentic = MagicMock(return_value=[])
        region = CorruptMathRegion(
            rect=object(),
            source_text="PðA or BÞ ¼ PðAÞ þ PðBÞ",
            crop_path="equations/corrupt_math_p00001_r001.png",
            raw_latex="",
            validation_ok=False,
            validation_reason="fixture overwritten below",
            model_id=config.math_model,
            attempts=0,
        )

        def _policy_result(*args, model_disabled_reason="", **kwargs):
            region.validation_reason = model_disabled_reason
            return [region]

        with (
            patch(
                "socr.math.recover.recover_math_regions",
                side_effect=_policy_result,
            ) as recover_regions,
            patch.object(pipeline, "_resolve_crop_vlm_model", return_value=None),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            result = pipeline.process(pdf, tmp_path)

        reason = recover_regions.call_args.kwargs["model_disabled_reason"]
        assert reason_fragment in reason
        assert result.status == DocumentStatus.AUDIT_FAILED
        sidecar = json.loads((tmp_path / pdf.stem / "pages" / "00001.json").read_text())
        recovery_event = next(
            event
            for event in sidecar["audit_events"]
            if event["kind"] == "corrupt_math_region_recovery"
        )
        assert recovery_event["data"]["model_call_skipped"] is True
        assert recovery_event["data"]["model_disabled_reason"] == reason
        assert recovery_event["data"]["regions"][0]["attempts"] == 0
        assert sidecar["winning_output"]["cost_usd"] == 0.0

    def test_agentic_corrupt_math_runs_without_whole_page_provider_and_fails_closed(self, tmp_path):
        """GH-271: an empty whole-page ladder cannot suppress region evidence."""
        pdf = self._real_pdf(tmp_path, 1)
        native = "Native prose survives.\nPðA or BÞ ¼ PðAÞ þ PðBÞ\n \t\n"
        assessment = DocumentAssessment(
            path=pdf,
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text=native,
                    confidence=0.9,
                    needs_ocr_enhancement=True,
                    has_equations=True,
                    has_corrupt_math=True,
                    word_count=100,
                )
            ],
        )
        config = _make_config(
            agentic=True,
            native_first=True,
            recover_corrupt_math=True,
            judge_backend="heuristic",
            enabled_engines=[],
        )
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = assessment
        pipeline._available_engines_for_agentic = MagicMock(return_value=[])
        whole_page = _mock_engine_named("qwen", "must not run")

        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=whole_page),
            patch("socr.math.recover.recover_math_regions", return_value=[]) as recover_regions,
            patch.object(pipeline, "_resolve_crop_vlm_model", return_value=None),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            result = pipeline.process(pdf, tmp_path)

        recover_regions.assert_called_once()
        whole_page.process_pages.assert_not_called()
        assert result.status == DocumentStatus.AUDIT_FAILED
        assert "Native prose survives." in result.markdown
        assert "PðA or BÞ ¼ PðAÞ þ PðBÞ" in result.markdown
        assert native in result.markdown
        assert "corrupt equation unresolved" in result.markdown
        assert result.error == "corrupt equation candidate unverified on page(s) 1"

    @pytest.mark.parametrize(
        ("config_overrides", "has_tables", "whole_page_runs"),
        [
            ({"native_only": True}, False, False),
            ({"native_first": False}, False, True),
            ({}, True, True),
        ],
        ids=["native-only", "no-native-first", "table-page"],
    )
    def test_agentic_corrupt_math_region_lane_respects_routing_safeguards(
        self,
        tmp_path,
        config_overrides,
        has_tables,
        whole_page_runs,
    ):
        """GH-271: unsafe or explicitly disabled pages never use line splicing."""
        from socr.core.providers import PROFILE_QWEN_LOCAL

        pdf = self._real_pdf(tmp_path, 1)
        native = "Native prose.\nPðA or BÞ ¼ PðAÞ þ PðBÞ"
        assessment = DocumentAssessment(
            path=pdf,
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text=native,
                    confidence=0.9,
                    needs_ocr_enhancement=True,
                    has_tables=has_tables,
                    has_equations=True,
                    has_corrupt_math=True,
                    word_count=100,
                )
            ],
        )
        config_values = {
            "agentic": True,
            "native_first": True,
            "recover_corrupt_math": True,
            "native_only": False,
            "judge_backend": "heuristic",
            "enabled_engines": [EngineType.QWEN],
        }
        config_values.update(config_overrides)
        config = _make_config(**config_values)
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = assessment
        pipeline._available_engines_for_agentic = MagicMock(return_value=[PROFILE_QWEN_LOCAL])
        whole_page = _mock_engine_named("qwen", _good_text())

        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=whole_page),
            patch("socr.math.recover.recover_math_regions") as recover_regions,
            patch.object(pipeline, "_resolve_crop_vlm_model", return_value=None),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            result = pipeline.process(pdf, tmp_path)

        recover_regions.assert_not_called()
        if whole_page_runs:
            whole_page.process_pages.assert_called_once()
        else:
            whole_page.process_pages.assert_not_called()
            assert native in result.markdown

    def test_default_mode_writes_no_manifest(self, tmp_path):
        pdf = self._real_pdf(tmp_path, 1)
        config = _make_config(agentic=False, enabled_engines=[EngineType.DEEPSEEK])
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(1, born_digital_pages=set())
        eng = _mock_engine_named("deepseek", _good_text())

        with patch("socr.pipeline.orchestrator.get_engine", return_value=eng):
            pipeline.process(pdf, tmp_path)

        # Manifest is opt-in: not written unless agentic or write_manifest.
        assert not (tmp_path / "paper" / "manifest.json").exists()

    def test_no_native_first_forces_ocr_on_prose(self, tmp_path):
        """--no-native-first routes born-digital prose through the cost ladder
        instead of taking free native text."""
        pdf = self._real_pdf(tmp_path, 1)
        config = _make_config(
            agentic=True,
            native_first=False,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GEMINI],
        )
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        # Page IS born-digital prose -> would normally take native text for free.
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            1, born_digital_pages={1}, complex_pages=set()
        )
        eng = _mock_engine_named("gemini", _good_text())

        with patch("socr.pipeline.orchestrator.get_engine", return_value=eng):
            result = pipeline.process(pdf, tmp_path)

        # Because native-first is off, the prose page was OCR'd via the ladder
        # (Gemini, $0.0002) rather than served free from native text.
        assert result.cost == pytest.approx(0.0002)
        assert _good_text()[:25] in result.markdown


# ---------------------------------------------------------------------------
# GH-37: --native-only CLI control tests
# ---------------------------------------------------------------------------


class TestNativeOnlyConfig:
    """GH-37: PipelineConfig default and CLI wiring for native_only."""

    def test_default_is_false(self) -> None:
        """native_only defaults to False (no behaviour change by default)."""
        cfg = PipelineConfig()
        assert cfg.native_only is False

    def test_build_config_native_only_flag(self) -> None:
        """--native-only sets native_only=True on PipelineConfig."""
        from socr.cli import build_config

        cfg = build_config(native_only=True)
        assert cfg.native_only is True

    def test_build_config_default_native_only_false(self) -> None:
        """Omitting --native-only leaves native_only=False."""
        from socr.cli import build_config

        cfg = build_config()
        assert cfg.native_only is False

    def test_incompatible_flags_no_native_first_wins(self, capsys) -> None:
        """--native-only + --no-native-first: --no-native-first wins, warning emitted."""
        from socr.cli import build_config

        cfg = build_config(native_only=True, no_native_first=True)
        # --no-native-first wins: native_first is off, so native_only is irrelevant.
        assert cfg.native_first is False
        # native_only is NOT set because --no-native-first makes it incoherent.
        assert cfg.native_only is False

    def test_native_only_does_not_disable_native_first(self) -> None:
        """--native-only does not affect native_first; it remains True."""
        from socr.cli import build_config

        cfg = build_config(native_only=True)
        assert cfg.native_first is True
        assert cfg.native_only is True


class TestNativeOnlyFingerprint:
    """GH-37: native_only must invalidate the resume fingerprint."""

    def _fp(self, **kwargs) -> str:
        cfg = _make_config(**kwargs)
        return UnifiedPipeline(cfg)._run_fingerprint()

    def test_native_only_changes_fingerprint(self) -> None:
        """Toggling native_only must produce a different fingerprint."""
        fp_off = self._fp(native_only=False)
        fp_on = self._fp(native_only=True)
        assert fp_off != fp_on, (
            "fingerprint must differ when native_only changes "
            "so the resume gate reprocesses the doc under the new policy"
        )


class TestNativeOnlyRouting:
    """GH-37: native_only routing behaviour in backbone and agentic paths."""

    @staticmethod
    def _real_pdf(tmp_path, n_pages=2):
        fitz = pytest.importorskip("fitz")
        path = tmp_path / "paper.pdf"
        doc = fitz.open()
        for i in range(n_pages):
            page = doc.new_page()
            # Add enough text so born-digital check passes on the real detector
            for j in range(12):
                page.insert_text(
                    (72, 72 + j * 14),
                    f"Page {i + 1} sentence {j}: economic analysis of monetary policy.",
                    fontsize=11,
                    fontname="helv",
                )
        doc.save(str(path))
        doc.close()
        return path

    def test_native_only_agentic_suppresses_enhancement(self, tmp_path) -> None:
        """Agentic path: native_only=True keeps BD enhancement pages native (cost=0)."""
        pdf = self._real_pdf(tmp_path, 2)
        config = _make_config(
            agentic=True,
            native_first=True,
            native_only=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GEMINI],
        )
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        # Both pages born-digital; page 2 has needs_ocr_enhancement=True.
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            2, born_digital_pages={1, 2}, complex_pages={2}
        )
        eng = _mock_engine_named("gemini", _good_text())

        with patch("socr.pipeline.orchestrator.get_engine", return_value=eng):
            result = pipeline.process(pdf, tmp_path)

        # All pages served from native text (cost=0, no OCR calls).
        assert result.cost == pytest.approx(0.0), (
            "native_only should keep enhancement pages native — no OCR cost"
        )

    def test_agentic_routes_clean_table_pages_to_ocr(self, tmp_path) -> None:
        """Agentic path: clean born-digital table pages use the OCR ladder."""
        pdf = self._real_pdf(tmp_path, 2)
        config = _make_config(
            agentic=True,
            native_first=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GEMINI],
        )
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            2, born_digital_pages={1, 2}, table_pages={2}
        )
        eng = _mock_engine_named("gemini", _good_text())

        with patch("socr.pipeline.orchestrator.get_engine", return_value=eng):
            result = pipeline.process(pdf, tmp_path)

        assert result.cost == pytest.approx(0.0002)
        assert "Native text for page 1" in result.markdown
        assert _good_text() in result.markdown
        assert eng.process_pages.call_args.kwargs["page_nums"] == [2]

    def test_agentic_no_provider_table_fallback_is_audit_failed(self, tmp_path) -> None:
        """No-provider table fallback is visible, not a clean native success."""
        pdf = self._real_pdf(tmp_path, 1)
        config = _make_config(
            agentic=True,
            native_first=True,
            judge_backend="heuristic",
            enabled_engines=[],
        )
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            1, born_digital_pages={1}, table_pages={1}
        )

        result = pipeline.process(pdf, tmp_path)

        assert result.status == DocumentStatus.AUDIT_FAILED
        assert "Native text for page 1" in result.markdown

    def test_agentic_table_judge_reject_all_rungs_is_fail_closed(self, tmp_path) -> None:
        """Provenance-masking guard: provider available, OCR returns SUCCESS+non-empty
        content, but judge rejects all rungs — result must not be a clean success.

        Regression for the bug where native_table_structure_failed was never set
        in the agentic path when the decision was not accepted, so _assemble_result
        silently treated the native-text fallback as a passing entry.
        """
        pdf = self._real_pdf(tmp_path, 1)
        config = _make_config(
            agentic=True,
            native_first=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GEMINI],
        )
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        # Build an assessment where page 1 has word_count above the audit minimum
        # (50 words) so that _sparse_page_ok returns False and the heuristic
        # judge applies the full word-count gate.  _make_bd_assessment leaves
        # word_count=0, which makes every page sparse-OK and silently accepts
        # any non-empty output — defeating the rejection scenario we want to test.
        page1_assessment = PageAssessment(
            page_num=1,
            is_born_digital=True,
            native_text="Native text for page 1",
            confidence=0.9,
            needs_ocr_enhancement=False,
            has_tables=True,
            word_count=100,  # above audit_min_words=50 → not sparse → judge applies full gate
        )
        pipeline.bd_detector.detect.return_value = DocumentAssessment(
            path=pdf, pages=[page1_assessment]
        )
        # Engine returns PageStatus.SUCCESS with non-empty content, but text is
        # too short (1 word) to pass the full heuristic gate → judge rejects all rungs.
        eng = _mock_engine_named("gemini", _bad_text())

        # CI has no ollama and no provider. Both seams are patched even though
        # this test selects the heuristic judge: the provider ladder must not be
        # empty (the loop bails before routing) and `_phase_judge_hard_pages`
        # builds an OllamaVisionJudge and POSTs to it regardless of
        # `judge_backend` unless the judge model resolves empty.
        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=eng),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
        ):
            result = pipeline.process(pdf, tmp_path)

        # (a) the structure-class floor is a page-level ERROR, not a warning.
        assert result.status == DocumentStatus.ERROR, (
            f"expected ERROR but got {result.status}; "
            "native table page with all-rejected OCR must fail closed"
        )
        # (b) page must carry the floor, not be stamped audit_passed=True.
        #
        # GH-508: this was read from `manifest.json`, guarded by
        # `if manifest_path.exists()`, and NONE of it ever ran. `_write_manifest`
        # is gated on `has_text`, and `_assemble_result` only reaches ERROR when
        # `has_text` is False -- so on this exact path the manifest is never
        # written and the guard was always False. (Doubly dead: the lookup used
        # `manifest["pages"]`, and the key is `entries`.)
        #
        # The per-page sidecar IS written on this path, carries the same three
        # facts, and needs no guard -- its absence is itself a failure worth
        # reporting.
        import json

        sidecar_path = tmp_path / "paper" / "pages" / "00001.json"
        assert sidecar_path.exists(), (
            "no page sidecar was written for a fail-closed page, so nothing "
            "records WHY the document errored"
        )
        sidecar = json.loads(sidecar_path.read_text())
        winner = sidecar.get("winning_output", {})

        assert winner.get("audit_passed") is not True, (
            "the shipped page must NOT be audit_passed=True when every OCR rung was rejected"
        )
        assert sidecar.get("status") == PageStatus.ERROR.value, (
            f"the sidecar does not carry the floor: {sidecar.get('status')!r}"
        )
        assert "Native text for page 1" not in winner.get("text", ""), (
            "the rejected page fell back to native text under a failure status; "
            "a fail-closed page must not ship unverified content as its body"
        )

    def test_native_only_agentic_suppresses_clean_table_pages(self, tmp_path) -> None:
        """Agentic native_only=True keeps clean table pages native."""
        pdf = self._real_pdf(tmp_path, 2)
        config = _make_config(
            agentic=True,
            native_first=True,
            native_only=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GEMINI],
        )
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            2, born_digital_pages={1, 2}, table_pages={2}
        )
        eng = _mock_engine_named("gemini", _good_text())

        with patch("socr.pipeline.orchestrator.get_engine", return_value=eng):
            result = pipeline.process(pdf, tmp_path)

        assert result.cost == pytest.approx(0.0)
        eng.process_pages.assert_not_called()

    def test_native_only_agentic_still_ocrs_scans(self, tmp_path) -> None:
        """Agentic path: native_only=True still routes scans through the OCR ladder."""
        pdf = self._real_pdf(tmp_path, 2)
        config = _make_config(
            agentic=True,
            native_first=True,
            native_only=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.GEMINI],
        )
        pipeline = UnifiedPipeline(config)
        pipeline.bd_detector = MagicMock()
        # Page 1 born-digital, page 2 is a scan.
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            2, born_digital_pages={1}, complex_pages=set()
        )
        eng = _mock_engine_named("gemini", _good_text())

        with patch("socr.pipeline.orchestrator.get_engine", return_value=eng):
            result = pipeline.process(pdf, tmp_path)

        # Page 2 (scan) must be OCR'd via Gemini ($0.0002).
        assert result.cost == pytest.approx(0.0002), (
            "scans must still be OCR'd even under native_only"
        )


# ---------------------------------------------------------------------------
# GH-47C: figure_recoverable_labels audit event
# ---------------------------------------------------------------------------


class TestRecoverableLabelAudit:
    """GH-47C (Option C — log-only): _record_figure_recoverable_labels records
    native word tokens from the figure bbox in the audit log without comparing
    them against the caption and without emitting any warning.
    """

    def _make_pipeline(self, **overrides) -> UnifiedPipeline:
        return UnifiedPipeline(_make_config(save_figures=True, describe_figures=True, **overrides))

    def _make_pdf_with_text_inside_rect(self, tmp_path: Path) -> Path:
        """Synthetic born-digital PDF with native text tokens inside a figure rect."""
        fitz = pytest.importorskip("fitz")
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        # Place native text tokens inside a known rect.
        page.insert_text(fitz.Point(110, 200), "AxisLabel", fontsize=9)
        page.insert_text(fitz.Point(200, 300), "SeriesA", fontsize=9)
        pdf_path = tmp_path / "native_labels.pdf"
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    def _make_pdf_raster_only(self, tmp_path: Path) -> Path:
        """Synthetic PDF with a rasterized image and no native text in the figure rect."""
        import io

        fitz = pytest.importorskip("fitz")
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (300, 300), color="blue").save(buf, format="PNG")
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_image(fitz.Rect(100, 100, 400, 400), stream=buf.getvalue())
        pdf_path = tmp_path / "raster_only.pdf"
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    def test_recoverable_labels_recorded_for_described_vector_figure(self, tmp_path: Path) -> None:
        """GH-47C AC2: described figures with native tokens inside bbox get a
        figure_recoverable_labels AuditEvent whose data.recoverable_labels is non-empty
        and whose inapplicable flag is False.
        Caption is not modified.
        """
        from socr.core.audit_log import AuditEvent

        pdf_path = self._make_pdf_with_text_inside_rect(tmp_path)
        pipeline = self._make_pipeline()
        state = DocumentState(handle=_make_handle(1))
        with patch.object(DocumentHandle, "__post_init__", lambda self: None):
            state.handle = DocumentHandle(path=pdf_path, page_count=1)

        # bbox covers the text tokens inserted at (110,200) and (200,300).
        bbox = (90.0, 180.0, 400.0, 350.0)
        fig_info = FigureInfo(
            figure_num=1,
            page_num=1,
            figure_type="chart",
            description=(
                "[model-generated, non-authoritative gist] A chart showing AxisLabel trends."
            ),
            bbox=bbox,
        )
        original_description = fig_info.description

        pipeline._record_figure_recoverable_labels(state, fig_info)

        label_events = [
            e
            for e in state.events
            if isinstance(e, AuditEvent) and e.kind == "figure_recoverable_labels"
        ]
        assert label_events, (
            "Expected a figure_recoverable_labels AuditEvent for a described vector figure."
        )
        ev = label_events[0]
        assert ev.data["figure_num"] == 1
        assert ev.data["inapplicable"] is False, (
            "inapplicable must be False when native words are found in the region"
        )
        assert len(ev.data["recoverable_labels"]) > 0, (
            "recoverable_labels must contain the native text tokens inside the figure bbox"
        )
        # Caption must be unchanged (AC: never rewrite captions).
        assert fig_info.description == original_description, (
            "Caption must not be modified by the label recovery path"
        )

    def test_recoverable_labels_inapplicable_for_raster_figure(self, tmp_path: Path) -> None:
        """GH-47C: rasterized figures yield zero recoverable tokens; inapplicable=True.
        No warning is emitted, no failure is recorded — an empty set is correct.
        Caption is unchanged.
        """
        from socr.core.audit_log import AuditEvent

        pdf_path = self._make_pdf_raster_only(tmp_path)
        pipeline = self._make_pipeline()
        state = DocumentState(handle=_make_handle(1))
        with patch.object(DocumentHandle, "__post_init__", lambda self: None):
            state.handle = DocumentHandle(path=pdf_path, page_count=1)

        bbox = (100.0, 100.0, 400.0, 400.0)
        fig_info = FigureInfo(
            figure_num=1,
            page_num=1,
            figure_type="image",
            description="[model-generated, non-authoritative gist] A blue chart.",
            bbox=bbox,
        )
        original_description = fig_info.description

        pipeline._record_figure_recoverable_labels(state, fig_info)

        label_events = [
            e
            for e in state.events
            if isinstance(e, AuditEvent) and e.kind == "figure_recoverable_labels"
        ]
        assert label_events, (
            "Expected a figure_recoverable_labels AuditEvent even for raster figures."
        )
        ev = label_events[0]
        assert ev.data["recoverable_labels"] == [], (
            "recoverable_labels must be empty for a rasterized figure"
        )
        assert ev.data["inapplicable"] is True, (
            "inapplicable must be True when zero native words are found in the region"
        )
        # No failure / warning kind must be emitted.
        assert all(e.kind == "figure_recoverable_labels" for e in label_events), (
            "No warning or failure event must be emitted for inapplicable raster figures"
        )
        # Caption must be unchanged.
        assert fig_info.description == original_description

    def test_no_label_event_when_figure_not_described(self, tmp_path: Path) -> None:
        """GH-47C: figures that were not described (no caption) must not trigger
        the recoverable-label path.  This tests the was_described gate in the
        _describe_and_embed_figures loop.
        """
        fig_path = tmp_path / "test_doc" / "figures" / "figure_1_page1.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig_path.write_bytes(b"\x89PNG")

        mock_fig = ExtractedFigure(
            figure_num=1,
            page_num=1,
            image=MagicMock(),
            saved_path=str(fig_path),
            bbox=(50.0, 50.0, 300.0, 400.0),
        )

        # describe_figures=False → vision engine is None → was_described=False
        pipeline = UnifiedPipeline(_make_config(save_figures=True, describe_figures=False))
        state = DocumentState(handle=_make_handle(1))
        result = _make_engine_result(text=_good_text())

        with (
            patch("socr.pipeline.orchestrator.FigureExtractor") as mock_extractor_cls,
            patch.object(pipeline, "_get_vision_engine", return_value=None),
            patch.object(pipeline, "_record_figure_recoverable_labels") as mock_record,
        ):
            mock_extractor_cls.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            pipeline._describe_and_embed_figures(state, result, tmp_path, "OCR text")

        (
            mock_record.assert_not_called(),
            (
                "_record_figure_recoverable_labels must not be called when figures were not described"
            ),
        )

    def test_no_label_event_when_bbox_is_none(self, tmp_path: Path) -> None:
        """GH-47C: _record_figure_recoverable_labels is a no-op when bbox is None
        (e.g. xref images without a placement rect).
        """
        from socr.core.audit_log import AuditEvent

        pipeline = self._make_pipeline()
        state = DocumentState(handle=_make_handle(1))
        with patch.object(DocumentHandle, "__post_init__", lambda self: None):
            state.handle = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=1)

        fig_info = FigureInfo(
            figure_num=1,
            page_num=1,
            figure_type="image",
            description="some caption",
            bbox=None,  # no bbox → inapplicable by precondition
        )
        pipeline._record_figure_recoverable_labels(state, fig_info)

        label_events = [
            e
            for e in state.events
            if isinstance(e, AuditEvent) and e.kind == "figure_recoverable_labels"
        ]
        assert label_events == [], (
            "_record_figure_recoverable_labels must emit no event when bbox is None"
        )
