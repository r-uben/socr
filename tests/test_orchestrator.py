"""Tests for UnifiedPipeline (5-phase orchestrator)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from socr.audit.heuristics import HeuristicsChecker
from socr.audit.scorer import FailureModeScorer, ScoringResult
from socr.core.born_digital import BornDigitalDetector, DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState, PageState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.pipeline.repair import RepairPlan, RepairRouter


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
        audit_enabled=True,
        max_retries=2,
        save_figures=False,
        quiet=True,
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


def _make_bd_assessment(
    page_count: int,
    born_digital_pages: set[int] | None = None,
) -> DocumentAssessment:
    """Build a DocumentAssessment with specified born-digital pages."""
    bd = born_digital_pages or set()
    pages = []
    for i in range(1, page_count + 1):
        is_bd = i in bd
        pages.append(
            PageAssessment(
                page_num=i,
                is_born_digital=is_bd,
                native_text=f"Native text for page {i}" if is_bd else "",
                confidence=0.9,
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

class TestPhaseBackbone:
    def test_backbone_applies_result_to_state(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(3))

        good_result = _make_engine_result(text=_good_text())

        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            result = pipeline._phase_backbone(state, Path("/tmp/out"))

        assert result is good_result
        assert len(state.engine_runs) == 1
        assert len(state.whole_doc_attempts) == 1

    def test_backbone_unavailable_engine(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = False

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            result = pipeline._phase_backbone(state, Path("/tmp/out"))

        assert result.status == DocumentStatus.ERROR
        assert len(state.engine_runs) == 1


# ---------------------------------------------------------------------------
# Phase 3: Score
# ---------------------------------------------------------------------------

class TestPhaseScore:
    def test_score_whole_doc_passing(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        result = _make_engine_result(text=_good_text())
        state.apply_result(result)

        pipeline._phase_score(state, result)

        whole_page = result.pages[0]
        assert whole_page.audit_passed is True
        assert whole_page.failure_mode == FailureMode.NONE

    def test_score_whole_doc_failing(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        result = _make_engine_result(text=_bad_text())
        state.apply_result(result)

        pipeline._phase_score(state, result)

        whole_page = result.pages[0]
        assert whole_page.audit_passed is False
        assert whole_page.failure_mode != FailureMode.NONE
        assert result.status == DocumentStatus.AUDIT_FAILED

    def test_score_per_page(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        # Create per-page outputs (page_num > 0)
        good_page = PageOutput(
            page_num=1, text=_good_text(),
            status=PageStatus.SUCCESS, engine="deepseek",
        )
        bad_page = PageOutput(
            page_num=2, text=_bad_text(),
            status=PageStatus.SUCCESS, engine="deepseek",
        )
        result = EngineResult(
            document_path=Path("/tmp/fake.pdf"),
            engine="deepseek",
            status=DocumentStatus.SUCCESS,
            pages=[good_page, bad_page],
            processing_time=1.0,
        )
        state.apply_result(result)

        pipeline._phase_score(state, result)

        assert good_page.audit_passed is True
        assert bad_page.audit_passed is False


# ---------------------------------------------------------------------------
# Phase 4: Selective Repair
# ---------------------------------------------------------------------------

class TestPhaseRepair:
    def test_no_repair_when_all_pages_pass(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        # Mark all pages as having good outputs
        for i in range(1, 3):
            state.pages[i].best_output = PageOutput(
                page_num=i, text=_good_text(),
                status=PageStatus.SUCCESS, engine="deepseek",
                audit_passed=True,
            )

        pipeline._phase_repair(state, Path("/tmp/out"))

        # No engine calls should have been made
        assert len(state.engine_runs) == 0

    def test_repair_attempts_fallback_engine(self) -> None:
        config = _make_config(
            fallback_chain=[EngineType.GEMINI],
            max_retries=1,
        )
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))

        # Page 1 has a failing attempt from deepseek
        bad_attempt = PageOutput(
            page_num=0, text=_bad_text(),
            status=PageStatus.SUCCESS, engine="deepseek",
            audit_passed=False, failure_mode=FailureMode.LOW_WORD_COUNT,
        )
        bad_result = EngineResult(
            document_path=Path("/tmp/fake.pdf"),
            engine="deepseek",
            status=DocumentStatus.AUDIT_FAILED,
            pages=[bad_attempt],
        )
        state.apply_result(bad_result)

        # Mock the fallback engine to return a good result
        good_result = _make_engine_result(
            text=_good_text(), engine="gemini",
        )

        mock_engine = MagicMock()
        mock_engine.name = "gemini"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            pipeline._phase_repair(state, Path("/tmp/out"))

        # Gemini result should have been applied
        assert len(state.engine_runs) == 2  # deepseek + gemini

    def test_repair_respects_max_retries(self) -> None:
        config = _make_config(max_retries=2)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))

        call_count = 0

        def mock_get_engine(engine_type):
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.name = engine_type.value
            mock.is_available.return_value = True
            # Always return a bad result to force retries
            mock.process_document.return_value = _make_engine_result(
                text=_bad_text(), engine=engine_type.value,
                audit_passed=False,
            )
            return mock

        # Ensure the repair router always has a fresh engine to suggest
        pipeline.repair_router = RepairRouter(
            _make_config(
                fallback_chain=[
                    EngineType.GEMINI, EngineType.MISTRAL,
                    EngineType.NOUGAT, EngineType.MARKER,
                ],
            )
        )

        with patch("socr.pipeline.orchestrator.get_engine", side_effect=mock_get_engine):
            pipeline._phase_repair(state, Path("/tmp/out"))

        # Should have made at most max_retries attempts
        assert call_count <= config.max_retries

    def test_repair_stops_when_all_pages_fixed(self) -> None:
        config = _make_config(max_retries=3)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))

        # The repair engine returns a good whole-doc result
        good_result = _make_engine_result(
            text=_good_text(), engine="gemini",
        )
        mock_engine = MagicMock()
        mock_engine.name = "gemini"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            pipeline._phase_repair(state, Path("/tmp/out"))

        # With whole-doc CLI engines, pages don't get per-page best_output
        # from the repair. But the whole_doc_attempts list should have an entry.
        assert len(state.whole_doc_attempts) >= 1


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
                page_num=i, text=f"Content for page {i}",
                status=PageStatus.SUCCESS, engine="deepseek",
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
            page_num=0, text="Full document text here",
            status=PageStatus.SUCCESS, engine="deepseek",
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

        assert final.status == DocumentStatus.SUCCESS
        assert "Full document text here" in final.markdown

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
            page_num=1, text="Hello world",
            status=PageStatus.SUCCESS, engine="deepseek",
            audit_passed=True,
        )

        result = pipeline._phase_assemble(state, tmp_path)

        stem = "fake"
        md_path = tmp_path / stem / f"{stem}.md"
        assert md_path.exists()
        assert md_path.read_text() == "Hello world"

    def test_assemble_born_digital_text_used(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(2))

        # Page 1 is born-digital
        state.pages[1].is_born_digital = True
        state.pages[1].native_text = "Native born-digital content"
        # Page 2 has OCR output
        state.pages[2].best_output = PageOutput(
            page_num=2, text="OCR content",
            status=PageStatus.SUCCESS, engine="deepseek",
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
            page_num=1, text="Only this page worked",
            status=PageStatus.SUCCESS, engine="deepseek",
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
        """Mock all externals and run the full 5-phase loop."""
        config = _make_config(quiet=True)
        pipeline = UnifiedPipeline(config)

        # Mock born-digital detection
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            3, born_digital_pages=set()
        )

        # Mock engine
        good_result = _make_engine_result(text=_good_text())
        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

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
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            2, born_digital_pages={1, 2}
        )

        # Engine still runs (backbone processes whole doc)
        good_result = _make_engine_result(text=_good_text())
        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            with patch.object(DocumentHandle, "from_path") as mock_from_path:
                mock_from_path.return_value = _make_handle(2)
                result = pipeline.process(Path("/tmp/fake.pdf"), tmp_path)

        assert result.success
        # The born-digital pages don't need repair
        # Markdown should contain native text from born-digital pages

    def test_full_loop_with_fallback(self, tmp_path: Path) -> None:
        """Primary fails audit, fallback succeeds."""
        config = _make_config(quiet=True, max_retries=1)
        pipeline = UnifiedPipeline(config)

        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            1, born_digital_pages=set()
        )

        bad_result = _make_engine_result(
            text=_bad_text(), engine="deepseek",
            audit_passed=False,
        )
        good_result = _make_engine_result(
            text=_good_text(), engine="gemini",
        )

        call_count = [0]
        def mock_get(engine_type):
            call_count[0] += 1
            mock = MagicMock()
            mock.name = engine_type.value
            mock.is_available.return_value = True
            if call_count[0] == 1:
                # First call: primary engine (deepseek) - bad result
                mock.process_document.return_value = bad_result
            else:
                # Subsequent calls: fallback engine - good result
                mock.process_document.return_value = good_result
            return mock

        with patch("socr.pipeline.orchestrator.get_engine", side_effect=mock_get):
            with patch.object(DocumentHandle, "from_path") as mock_from_path:
                mock_from_path.return_value = _make_handle(1)
                result = pipeline.process(Path("/tmp/fake.pdf"), tmp_path)

        # Should succeed thanks to fallback
        assert result.status in (DocumentStatus.SUCCESS, DocumentStatus.AUDIT_FAILED)

    def test_audit_disabled_skips_score_and_repair(self, tmp_path: Path) -> None:
        """When audit is disabled, phases 3 and 4 are skipped."""
        config = _make_config(quiet=True, audit_enabled=False)
        pipeline = UnifiedPipeline(config)

        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            1, born_digital_pages=set()
        )

        good_result = _make_engine_result(text=_good_text())
        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            with patch.object(DocumentHandle, "from_path") as mock_from_path:
                mock_from_path.return_value = _make_handle(1)
                result = pipeline.process(Path("/tmp/fake.pdf"), tmp_path)

        # Engine should have been called only once (no repair)
        assert mock_engine.process_document.call_count == 1


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
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(
            1, born_digital_pages=set()
        )

        good_result = _make_engine_result(text=_good_text())
        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

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

        # Mark as already processed via metadata
        with patch("socr.pipeline.orchestrator.MetadataManager") as MockMeta:
            mock_meta = MockMeta.return_value
            mock_meta.is_processed.return_value = True

            results = pipeline.process_batch(pdf_dir, out_dir)

        assert results == []


# ---------------------------------------------------------------------------
# Max retries limiting
# ---------------------------------------------------------------------------

class TestMaxRetries:
    def test_zero_max_retries_skips_repair(self) -> None:
        config = _make_config(max_retries=0)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))
        # Page needs repair but max_retries=0

        pipeline._phase_repair(state, Path("/tmp/out"))

        # No engine calls
        assert len(state.engine_runs) == 0

    def test_one_max_retry_runs_once(self) -> None:
        config = _make_config(max_retries=1)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))

        good_result = _make_engine_result(text=_good_text(), engine="gemini")
        mock_engine = MagicMock()
        mock_engine.name = "gemini"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            pipeline._phase_repair(state, Path("/tmp/out"))

        assert mock_engine.process_document.call_count == 1


# ---------------------------------------------------------------------------
# Figure extraction
# ---------------------------------------------------------------------------

class TestFigures:
    def test_figures_extracted_when_enabled(self, tmp_path: Path) -> None:
        config = _make_config(save_figures=True)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))

        state.pages[1].best_output = PageOutput(
            page_num=1, text="Content",
            status=PageStatus.SUCCESS, engine="deepseek",
            audit_passed=True,
        )

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:
            mock_extractor = MockExtractor.return_value
            mock_extractor.extract.return_value = []

            result = pipeline._phase_assemble(state, tmp_path)

        MockExtractor.assert_called_once()

    def test_figures_skipped_when_disabled(self, tmp_path: Path) -> None:
        config = _make_config(save_figures=False)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))

        state.pages[1].best_output = PageOutput(
            page_num=1, text="Content",
            status=PageStatus.SUCCESS, engine="deepseek",
            audit_passed=True,
        )

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:
            result = pipeline._phase_assemble(state, tmp_path)

        MockExtractor.assert_not_called()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_pipeline_importable_from_package(self) -> None:
        from socr.pipeline import UnifiedPipeline as UP
        assert UP is UnifiedPipeline

    def test_constructor_creates_all_components(self) -> None:
        config = _make_config()
        pipeline = UnifiedPipeline(config)

        assert isinstance(pipeline.heuristics, HeuristicsChecker)
        assert isinstance(pipeline.scorer, FailureModeScorer)
        assert isinstance(pipeline.repair_router, RepairRouter)
        assert isinstance(pipeline.bd_detector, BornDigitalDetector)

    def test_repair_with_unavailable_engine(self) -> None:
        config = _make_config(max_retries=1)
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(1))

        mock_engine = MagicMock()
        mock_engine.name = "gemini"
        mock_engine.is_available.return_value = False

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            pipeline._phase_repair(state, Path("/tmp/out"))

        # Engine was not called because it's unavailable
        mock_engine.process_document.assert_not_called()


# ---------------------------------------------------------------------------
# Truncation retry
# ---------------------------------------------------------------------------

class TestTruncationRetry:
    """Tests for retry-on-truncation before fallback (L1B-06)."""

    def _make_truncated_state(
        self, engine: str = "gemini", page_count: int = 10,
    ) -> DocumentState:
        """Build a state with a truncated whole-doc attempt."""
        state = DocumentState(handle=_make_handle(page_count))
        truncated_attempt = PageOutput(
            page_num=0,
            text="Truncated output with only a handful of words",
            status=PageStatus.SUCCESS,
            engine=engine,
            audit_passed=False,
            failure_mode=FailureMode.TRUNCATED,
        )
        truncated_result = EngineResult(
            document_path=Path("/tmp/fake.pdf"),
            engine=engine,
            status=DocumentStatus.AUDIT_FAILED,
            failure_mode=FailureMode.TRUNCATED,
            pages=[truncated_attempt],
            processing_time=2.0,
        )
        state.apply_result(truncated_result)
        return state

    def test_retry_succeeds_on_second_attempt(self) -> None:
        """Truncated first attempt, retry with same engine succeeds."""
        config = _make_config(
            primary_engine=EngineType.GEMINI,
            truncation_retries=1,
            max_retries=2,
        )
        pipeline = UnifiedPipeline(config)
        # Use page_count=1 so _good_text() won't trip the truncation
        # heuristic (which only fires for expected_pages > 5).
        state = self._make_truncated_state("gemini", page_count=1)

        good_result = _make_engine_result(
            text=_good_text(), engine="gemini",
        )
        mock_engine = MagicMock()
        mock_engine.name = "gemini"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            pipeline._phase_repair(state, Path("/tmp/out"))

        # The retry should have called the same engine once
        assert mock_engine.process_document.call_count == 1
        # Should now have a passing whole-doc attempt
        assert any(w.audit_passed for w in state.whole_doc_attempts)
        # Only 2 engine runs total: original truncated + retry
        assert len(state.engine_runs) == 2

    def test_retry_also_truncates_falls_through_to_fallback(self) -> None:
        """Both attempts truncate, then fallback chain is tried."""
        config = _make_config(
            primary_engine=EngineType.GEMINI,
            fallback_chain=[EngineType.DEEPSEEK],
            truncation_retries=1,
            max_retries=2,
        )
        pipeline = UnifiedPipeline(config)
        state = self._make_truncated_state("gemini")

        # Retry also returns truncated
        still_truncated = _make_engine_result(
            text="Still truncated short output",
            engine="gemini",
            audit_passed=False,
        )
        still_truncated.pages[0].failure_mode = FailureMode.TRUNCATED
        still_truncated.pages[0].audit_passed = False

        # Fallback returns good result
        good_result = _make_engine_result(
            text=_good_text(), engine="deepseek",
        )

        call_count = [0]

        def mock_get(engine_type):
            call_count[0] += 1
            mock = MagicMock()
            mock.name = engine_type.value
            mock.is_available.return_value = True
            if engine_type == EngineType.GEMINI:
                mock.process_document.return_value = still_truncated
            else:
                mock.process_document.return_value = good_result
            return mock

        with patch("socr.pipeline.orchestrator.get_engine", side_effect=mock_get):
            pipeline._phase_repair(state, Path("/tmp/out"))

        # Should have tried gemini (truncation retry) + deepseek (fallback)
        assert call_count[0] >= 2
        # Should have engine runs: original + truncation retry + fallback
        assert len(state.engine_runs) >= 3

    def test_non_truncation_failure_skips_retry(self) -> None:
        """Hallucination failure should NOT trigger truncation retry."""
        config = _make_config(
            primary_engine=EngineType.GEMINI,
            fallback_chain=[EngineType.DEEPSEEK],
            truncation_retries=1,
            max_retries=1,
        )
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(10))

        # Failing with HALLUCINATION, not TRUNCATED
        halluc_attempt = PageOutput(
            page_num=0,
            text="Use a standard font. Include all figures. Proofread your work.",
            status=PageStatus.SUCCESS,
            engine="gemini",
            audit_passed=False,
            failure_mode=FailureMode.HALLUCINATION,
        )
        halluc_result = EngineResult(
            document_path=Path("/tmp/fake.pdf"),
            engine="gemini",
            status=DocumentStatus.AUDIT_FAILED,
            failure_mode=FailureMode.HALLUCINATION,
            pages=[halluc_attempt],
            processing_time=2.0,
        )
        state.apply_result(halluc_result)

        good_result = _make_engine_result(
            text=_good_text(), engine="deepseek",
        )

        engines_called = []

        def mock_get(engine_type):
            engines_called.append(engine_type.value)
            mock = MagicMock()
            mock.name = engine_type.value
            mock.is_available.return_value = True
            mock.process_document.return_value = good_result
            return mock

        with patch("socr.pipeline.orchestrator.get_engine", side_effect=mock_get):
            pipeline._phase_repair(state, Path("/tmp/out"))

        # Should NOT have retried gemini for truncation;
        # should have gone straight to fallback chain.
        # The first engine resolved via the fallback should be deepseek
        # (gemini was already tried).
        assert "gemini" not in engines_called or engines_called[0] == "deepseek"

    def test_truncation_retries_zero_disables_retry(self) -> None:
        """Setting truncation_retries=0 skips the retry entirely."""
        config = _make_config(
            primary_engine=EngineType.GEMINI,
            fallback_chain=[EngineType.DEEPSEEK],
            truncation_retries=0,
            max_retries=2,
        )
        pipeline = UnifiedPipeline(config)
        state = self._make_truncated_state("gemini")

        good_result = _make_engine_result(
            text=_good_text(), engine="deepseek",
        )

        engines_called = []

        def mock_get(engine_type):
            engines_called.append(engine_type.value)
            mock = MagicMock()
            mock.name = engine_type.value
            mock.is_available.return_value = True
            mock.process_document.return_value = good_result
            return mock

        with patch("socr.pipeline.orchestrator.get_engine", side_effect=mock_get):
            pipeline._phase_repair(state, Path("/tmp/out"))

        # Should NOT have retried gemini; should go straight to fallback
        # (deepseek is the first untried engine in fallback_chain)
        assert engines_called[0] == "deepseek"


# ---------------------------------------------------------------------------
# Chunked backbone (L1B-07)
# ---------------------------------------------------------------------------

class TestChunkedBackbone:
    """Tests for _backbone_chunked: splitting long PDFs before OCR."""

    def test_short_doc_skips_chunking(self) -> None:
        """Documents below chunk_threshold use the normal backbone path."""
        config = _make_config(
            quiet=True,
            chunk_threshold=30,
            chunk_size=20,
        )
        pipeline = UnifiedPipeline(config)
        # 10 pages -- well below threshold
        state = DocumentState(handle=_make_handle(10))

        good_result = _make_engine_result(text=_good_text())
        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            result = pipeline._phase_backbone(state, Path("/tmp/out"))

        # Should have called process_document once on the original path
        mock_engine.process_document.assert_called_once()
        call_path = mock_engine.process_document.call_args[0][0]
        assert call_path == state.handle.path
        assert result.success

    def test_long_doc_triggers_chunking(self) -> None:
        """Documents above chunk_threshold use the chunked backbone path."""
        config = _make_config(
            quiet=True,
            chunk_threshold=5,
            chunk_size=3,
        )
        pipeline = UnifiedPipeline(config)
        # 10 pages > threshold of 5
        state = DocumentState(handle=_make_handle(10))

        chunk_texts = [
            f"Chunk {i} text with sufficient words to be meaningful"
            for i in range(1, 5)
        ]
        call_idx = [0]

        def mock_process_document(pdf_path, output_dir, cfg):
            idx = call_idx[0]
            call_idx[0] += 1
            text = chunk_texts[idx] if idx < len(chunk_texts) else "extra"
            return _make_engine_result(text=text, engine="deepseek")

        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.side_effect = mock_process_document

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            with patch("socr.pipeline.orchestrator.PDFChunker") as MockChunker:
                from socr.core.chunker import PDFChunk

                mock_chunker_instance = MockChunker.return_value
                mock_chunker_instance.chunk.return_value = [
                    PDFChunk(
                        chunk_num=i,
                        start_page=(i - 1) * 3 + 1,
                        end_page=min(i * 3, 10),
                        path=Path(f"/tmp/chunk{i}.pdf"),
                        page_count=min(3, 10 - (i - 1) * 3),
                    )
                    for i in range(1, 5)
                ]

                result = pipeline._phase_backbone(state, Path("/tmp/out"))

        # Should have called process_document once per chunk (4 chunks)
        assert mock_engine.process_document.call_count == 4
        assert result.success
        # The combined text should contain all chunk texts
        combined = result.pages[0].text
        for ct in chunk_texts:
            assert ct in combined
        # Result should be a whole-doc output (page_num=0)
        assert result.pages[0].page_num == 0
        # Should be applied to state
        assert len(state.whole_doc_attempts) == 1

    def test_chunked_backbone_handles_chunk_failure(self) -> None:
        """When some chunks fail, the result still contains successful chunks."""
        config = _make_config(
            quiet=True,
            chunk_threshold=5,
            chunk_size=3,
        )
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(10))

        call_idx = [0]

        def mock_process_document(pdf_path, output_dir, cfg):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 1:
                # Second chunk fails
                return EngineResult(
                    document_path=pdf_path,
                    engine="deepseek",
                    status=DocumentStatus.ERROR,
                    error="Engine crashed",
                )
            return _make_engine_result(
                text=f"Chunk {idx + 1} good text",
                engine="deepseek",
            )

        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.side_effect = mock_process_document

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            with patch("socr.pipeline.orchestrator.PDFChunker") as MockChunker:
                from socr.core.chunker import PDFChunk

                mock_chunker_instance = MockChunker.return_value
                mock_chunker_instance.chunk.return_value = [
                    PDFChunk(
                        chunk_num=i,
                        start_page=(i - 1) * 3 + 1,
                        end_page=min(i * 3, 10),
                        path=Path(f"/tmp/chunk{i}.pdf"),
                        page_count=min(3, 10 - (i - 1) * 3),
                    )
                    for i in range(1, 5)
                ]

                result = pipeline._phase_backbone(state, Path("/tmp/out"))

        # 3 out of 4 chunks succeeded, so overall result should succeed
        assert result.success
        combined = result.pages[0].text
        assert "Chunk 1 good text" in combined
        # Chunk 2 failed, so its text should NOT be in the combined output
        assert "Chunk 2 good text" not in combined
        assert "Chunk 3 good text" in combined

    def test_chunked_backbone_all_chunks_fail(self) -> None:
        """When all chunks fail, the result is an error."""
        config = _make_config(
            quiet=True,
            chunk_threshold=5,
            chunk_size=3,
        )
        pipeline = UnifiedPipeline(config)
        state = DocumentState(handle=_make_handle(10))

        def mock_process_document(pdf_path, output_dir, cfg):
            return EngineResult(
                document_path=pdf_path,
                engine="deepseek",
                status=DocumentStatus.ERROR,
                error="Engine crashed",
            )

        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.side_effect = mock_process_document

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            with patch("socr.pipeline.orchestrator.PDFChunker") as MockChunker:
                from socr.core.chunker import PDFChunk

                mock_chunker_instance = MockChunker.return_value
                mock_chunker_instance.chunk.return_value = [
                    PDFChunk(
                        chunk_num=i,
                        start_page=(i - 1) * 3 + 1,
                        end_page=min(i * 3, 10),
                        path=Path(f"/tmp/chunk{i}.pdf"),
                        page_count=min(3, 10 - (i - 1) * 3),
                    )
                    for i in range(1, 5)
                ]

                result = pipeline._phase_backbone(state, Path("/tmp/out"))

        assert result.status == DocumentStatus.ERROR
        assert not result.success

    def test_chunk_threshold_boundary(self) -> None:
        """Document with exactly chunk_threshold pages should NOT be chunked."""
        config = _make_config(
            quiet=True,
            chunk_threshold=10,
            chunk_size=5,
        )
        pipeline = UnifiedPipeline(config)
        # Exactly 10 pages == threshold
        state = DocumentState(handle=_make_handle(10))

        good_result = _make_engine_result(text=_good_text())
        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            result = pipeline._phase_backbone(state, Path("/tmp/out"))

        # Should NOT have chunked -- just one call with the original path
        mock_engine.process_document.assert_called_once()
        call_path = mock_engine.process_document.call_args[0][0]
        assert call_path == state.handle.path

    def test_chunk_threshold_plus_one(self) -> None:
        """Document with chunk_threshold+1 pages SHOULD be chunked."""
        config = _make_config(
            quiet=True,
            chunk_threshold=10,
            chunk_size=5,
        )
        pipeline = UnifiedPipeline(config)
        # 11 pages > threshold of 10
        state = DocumentState(handle=_make_handle(11))

        good_result = _make_engine_result(text=_good_text())
        mock_engine = MagicMock()
        mock_engine.name = "deepseek"
        mock_engine.is_available.return_value = True
        mock_engine.process_document.return_value = good_result

        with patch("socr.pipeline.orchestrator.get_engine", return_value=mock_engine):
            with patch("socr.pipeline.orchestrator.PDFChunker") as MockChunker:
                from socr.core.chunker import PDFChunk

                mock_chunker_instance = MockChunker.return_value
                mock_chunker_instance.chunk.return_value = [
                    PDFChunk(
                        chunk_num=1,
                        start_page=1,
                        end_page=5,
                        path=Path("/tmp/chunk1.pdf"),
                        page_count=5,
                    ),
                    PDFChunk(
                        chunk_num=2,
                        start_page=6,
                        end_page=10,
                        path=Path("/tmp/chunk2.pdf"),
                        page_count=5,
                    ),
                    PDFChunk(
                        chunk_num=3,
                        start_page=11,
                        end_page=11,
                        path=Path("/tmp/chunk3.pdf"),
                        page_count=1,
                    ),
                ]

                result = pipeline._phase_backbone(state, Path("/tmp/out"))

        # Should have chunked -- 3 calls
        assert mock_engine.process_document.call_count == 3
        assert result.success
