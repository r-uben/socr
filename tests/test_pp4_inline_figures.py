"""PP-4 tests: per-page figure extraction + inline embedding.

Acceptance criteria verified here:

1. Figures appear INLINE within their ``## Page N`` section in the final .md
   (not a trailing doc-tail appendix).
2. ``figure_<N>_page<P>.png`` filenames are doc-global + monotonic across pages
   (no renumbering regression) — tested with a multi-page multi-figure doc.
3. ``max_total`` cap fires with a durable ``figure_cap_reached`` AuditEvent
   attached to the crossing page (not page_num=0).
4. Vision engine is constructed ONCE per document and closed ONCE.
5. ``--save-figures`` only (no ``--describe-figures``) → no VLM call; blocks
   have empty descriptions (GH-50 parity).
6. A figure-free document produces a byte-identical .md (PP-1 byte-identity).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ocr_output_contract import assemble_pages, split_native_pages

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FigureInfo,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState
from socr.figures.extractor import ExtractedFigure, ExtractionResult
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_handle(page_count: int = 3) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        h = DocumentHandle(path=Path("/tmp/pp4_fake.pdf"), page_count=page_count)
    return h


def _make_pipeline(**overrides) -> UnifiedPipeline:
    defaults = dict(
        primary_engine=EngineType.DEEPSEEK,
        quiet=True,
        save_figures=True,
        describe_figures=False,
        write_manifest=False,
        agentic=False,
        dual_pass_tables=False,
        judge_hard_pages=False,
    )
    defaults.update(overrides)
    return UnifiedPipeline(PipelineConfig(**defaults))


def _make_result() -> EngineResult:
    return EngineResult(
        document_path=Path("/tmp/pp4_fake.pdf"),
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


def _make_mock_fig(tmp_path: Path, fig_num: int, page_num: int) -> ExtractedFigure:
    """Synthetic ExtractedFigure with a saved PNG on disk."""
    figs_dir = tmp_path / "pp4_fake" / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figs_dir / f"figure_{fig_num}_page{page_num}.png"
    fig_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return ExtractedFigure(
        figure_num=fig_num,
        page_num=page_num,
        image=None,
        saved_path=str(fig_path),
    )


# ---------------------------------------------------------------------------
# 1. Inline placement: figure blocks appear within ## Page N
# ---------------------------------------------------------------------------


class TestInlinePlacement:
    """PP-4 AC1: figures are embedded inside the page section, not at doc tail."""

    def test_figure_embedded_inside_page_section(self, tmp_path: Path) -> None:
        """A single-page doc: figure block is inside ## Page 1, not appended after."""
        pipeline = _make_pipeline()
        state = DocumentState(handle=_make_handle(page_count=1))
        result = _make_result()

        mock_fig = _make_mock_fig(tmp_path, fig_num=1, page_num=1)
        # Assembled body for a single page.
        input_text = assemble_pages(["Page one body text."])

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:  # noqa: N806
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path, input_text)

        # Output must have ## Page 1 header.
        assert "## Page 1" in out
        # Find the page section and verify the figure block is inside it.
        pages = split_native_pages(out)
        assert len(pages) == 1
        page_body = pages[0]
        assert "**Figure 1** (page 1)" in page_body, (
            "Figure block must be inside the ## Page 1 body, not appended after the section."
        )
        assert "![Figure 1]" in page_body

    def test_figure_on_correct_page_multi_page_doc(self, tmp_path: Path) -> None:
        """Three-page doc, figure on page 2: block appears in ## Page 2, not page 1 or 3."""
        pipeline = _make_pipeline()
        state = DocumentState(handle=_make_handle(page_count=3))
        result = _make_result()

        # Figure on page 2.
        mock_fig = _make_mock_fig(tmp_path, fig_num=1, page_num=2)
        input_text = assemble_pages(["Page one.", "Page two.", "Page three."])

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:  # noqa: N806
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path, input_text)

        pages = split_native_pages(out)
        assert len(pages) == 3
        # Page 2 (index 1) must contain the figure block.
        assert "**Figure 1** (page 2)" in pages[1], "Figure block must appear in ## Page 2."
        # Page 1 and 3 must NOT contain figure blocks.
        assert "**Figure 1**" not in pages[0], "Figure must NOT appear in page 1 body."
        assert "**Figure 1**" not in pages[2], "Figure must NOT appear in page 3 body."

    def test_figures_on_different_pages_embedded_correctly(self, tmp_path: Path) -> None:
        """Two figures on two different pages: each is embedded in the correct section."""
        pipeline = _make_pipeline()
        state = DocumentState(handle=_make_handle(page_count=3))
        result = _make_result()

        fig1 = _make_mock_fig(tmp_path, fig_num=1, page_num=1)
        fig3 = _make_mock_fig(tmp_path, fig_num=2, page_num=3)
        input_text = assemble_pages(["First page.", "Second page.", "Third page."])

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:  # noqa: N806
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[fig1, fig3], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path, input_text)

        pages = split_native_pages(out)
        assert len(pages) == 3
        assert "**Figure 1** (page 1)" in pages[0], "Figure 1 must be in ## Page 1."
        assert "**Figure 2** (page 3)" in pages[2], "Figure 2 must be in ## Page 3."
        # Page 2 is untouched.
        assert "**Figure" not in pages[1], "Page 2 must have no figure blocks."


# ---------------------------------------------------------------------------
# 2. Doc-global monotonic figure numbering across pages
# ---------------------------------------------------------------------------


class TestGlobalMonotonicNumbering:
    """PP-4 AC2: figure counter is doc-global and monotonic regardless of page order."""

    def test_figure_numbering_is_monotonic_across_pages(self, tmp_path: Path) -> None:
        """Multi-page doc with two figures: numbers are 1 and 2, not reset per page.

        The extractor assigns numbers; this test verifies the orchestrator
        preserves them and embeds them in the correct pages.
        """
        pipeline = _make_pipeline()
        state = DocumentState(handle=_make_handle(page_count=4))
        result = _make_result()

        # Extractor assigns doc-global numbers: figure 1 on page 1, figure 2 on page 3.
        fig1 = _make_mock_fig(tmp_path, fig_num=1, page_num=1)
        fig2 = _make_mock_fig(tmp_path, fig_num=2, page_num=3)
        input_text = assemble_pages(["P1 body.", "P2 body.", "P3 body.", "P4 body."])

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:  # noqa: N806
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[fig1, fig2], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path, input_text)

        pages = split_native_pages(out)
        assert len(pages) == 4
        # Figure 1 on page 1, figure 2 on page 3 — no renumbering.
        assert "**Figure 1** (page 1)" in pages[0]
        assert "**Figure 2** (page 3)" in pages[2]
        # Global figure refs present in full output.
        assert "figure_1_page1.png" in out or "figure_1" in out
        assert "figure_2_page3.png" in out or "figure_2" in out

    def test_figure_filenames_encode_source_page(self, tmp_path: Path) -> None:
        """PNG filenames follow ``figure_<N>_page<P>`` convention.

        The extractor's ``_save`` method encodes both the global figure number
        and the source page number.  This test confirms the orchestrator embeds
        the paths as returned by the extractor (no re-encoding).
        """
        pipeline = _make_pipeline()
        state = DocumentState(handle=_make_handle(page_count=2))
        result = _make_result()

        fig1 = _make_mock_fig(tmp_path, fig_num=1, page_num=2)
        input_text = assemble_pages(["P1 body.", "P2 body."])

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:  # noqa: N806
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[fig1], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path, input_text)

        # The PNG path must encode page 2.
        assert "page2" in out, (
            f"Expected 'page2' in figure filename embedded in output; got:\n{out}"
        )


# ---------------------------------------------------------------------------
# 3. Cap AuditEvent at the crossing page
# ---------------------------------------------------------------------------


class TestCapEventPage:
    """PP-4 AC3: figure_cap_reached AuditEvent is attached to the stopping page."""

    def test_cap_event_carries_cap_page_when_provided(self, tmp_path: Path) -> None:
        """cap_page from ExtractionResult is used as page_num on the AuditEvent."""
        from socr.core.audit_log import AuditEvent

        pipeline = _make_pipeline(quiet=False)
        state = DocumentState(handle=_make_handle(page_count=5))
        result = _make_result()

        mock_console = MagicMock()
        with (
            patch("socr.pipeline.orchestrator.FigureExtractor") as mock_cls,
            patch("socr.pipeline.orchestrator.console", mock_console),
        ):
            mock_cls.return_value.extract.return_value = ExtractionResult(
                figures=[],
                cap_reached=True,
                cap_page=3,  # extraction stopped at page 3
            )
            pipeline._describe_and_embed_figures(state, result, tmp_path, "Some text")

        cap_events = [
            e for e in state.events if isinstance(e, AuditEvent) and e.kind == "figure_cap_reached"
        ]
        assert cap_events, "Expected figure_cap_reached AuditEvent."
        assert cap_events[0].page_num == 3, (
            f"AuditEvent page_num must be the cap page (3), got {cap_events[0].page_num}."
        )
        assert cap_events[0].data.get("figures_max_total") is not None

    def test_cap_event_falls_back_to_zero_when_cap_page_none(self, tmp_path: Path) -> None:
        """When cap_page is None (legacy), page_num falls back to 0."""
        from socr.core.audit_log import AuditEvent

        pipeline = _make_pipeline()
        state = DocumentState(handle=_make_handle())
        result = _make_result()

        with patch("socr.pipeline.orchestrator.FigureExtractor") as mock_cls:
            mock_cls.return_value.extract.return_value = ExtractionResult(
                figures=[], cap_reached=True, cap_page=None
            )
            pipeline._describe_and_embed_figures(state, result, tmp_path, "text")

        cap_events = [
            e for e in state.events if isinstance(e, AuditEvent) and e.kind == "figure_cap_reached"
        ]
        assert cap_events
        assert cap_events[0].page_num == 0


# ---------------------------------------------------------------------------
# 4. Vision engine lifecycle: constructed once, closed once
# ---------------------------------------------------------------------------


class TestVisionEngineLifecycle:
    """PP-4 AC4: vision engine is built once per doc and closed once."""

    def test_vision_engine_constructed_once_and_closed_once(self, tmp_path: Path) -> None:
        """With describe_figures=True and multiple figures, _get_vision_engine is
        called exactly once and close() is called exactly once."""
        pipeline = _make_pipeline(describe_figures=True)
        state = DocumentState(handle=_make_handle(page_count=3))
        result = _make_result()

        fig1 = _make_mock_fig(tmp_path, fig_num=1, page_num=1)
        fig2 = _make_mock_fig(tmp_path, fig_num=2, page_num=2)
        fig3 = _make_mock_fig(tmp_path, fig_num=3, page_num=3)
        # Give each mock figure a PIL image so the VLM branch is entered.
        for f in (fig1, fig2, fig3):
            f.image = MagicMock()

        input_text = assemble_pages(["P1.", "P2.", "P3."])

        mock_engine = MagicMock()
        mock_engine.name = "test-engine"
        mock_engine.describe_figure.return_value = FigureInfo(
            figure_num=0,
            page_num=0,
            figure_type="chart",
            description="A chart.",
            engine="test-engine",
        )

        construct_count = []

        def _fake_get_vision():
            construct_count.append(1)
            return mock_engine

        with (
            patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor,  # noqa: N806
            patch.object(pipeline, "_get_vision_engine", side_effect=_fake_get_vision),
        ):
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[fig1, fig2, fig3], cap_reached=False
            )
            pipeline._describe_and_embed_figures(state, result, tmp_path, input_text)

        assert len(construct_count) == 1, (
            f"Vision engine must be constructed exactly once; "
            f"was called {len(construct_count)} time(s)."
        )
        mock_engine.close.assert_called_once_with()

    def test_vision_engine_not_constructed_when_save_only(self, tmp_path: Path) -> None:
        """With describe_figures=False, _get_vision_engine is never called."""
        pipeline = _make_pipeline(describe_figures=False)
        state = DocumentState(handle=_make_handle(page_count=1))
        result = _make_result()

        fig1 = _make_mock_fig(tmp_path, fig_num=1, page_num=1)
        fig1.image = MagicMock()
        input_text = assemble_pages(["Page body."])

        with (
            patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor,  # noqa: N806
            patch.object(pipeline, "_get_vision_engine") as mock_get_vision,
        ):
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[fig1], cap_reached=False
            )
            pipeline._describe_and_embed_figures(state, result, tmp_path, input_text)

        mock_get_vision.assert_not_called()


# ---------------------------------------------------------------------------
# 5. --save-figures only: no VLM, empty descriptions (GH-50 parity, inline)
# ---------------------------------------------------------------------------


class TestSaveFiguresOnlyInline:
    """PP-4 AC5: save_figures=True, describe_figures=False → no VLM, refs inline."""

    def test_save_only_produces_image_ref_inline_no_caption(self, tmp_path: Path) -> None:
        """Image-ref block appears inline in the page section; description is empty."""
        pipeline = _make_pipeline(save_figures=True, describe_figures=False)
        state = DocumentState(handle=_make_handle(page_count=1))
        result = _make_result()

        fig1 = _make_mock_fig(tmp_path, fig_num=1, page_num=1)
        input_text = assemble_pages(["Body text on page one."])

        with (
            patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor,  # noqa: N806
            patch.object(pipeline, "_get_vision_engine") as mock_get_vision,
        ):
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[fig1], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path, input_text)

        # VLM never invoked.
        mock_get_vision.assert_not_called()
        # Description is empty: no colon after the figure label.
        assert "**Figure 1** (page 1):" not in out
        assert "**Figure 1** (page 1)" in out
        # Image ref is present.
        assert "![Figure 1]" in out
        # Block is inline (inside ## Page 1 section).
        pages = split_native_pages(out)
        assert len(pages) == 1
        assert "**Figure 1** (page 1)" in pages[0]
        # Result has empty description.
        assert result.figures[0].description == ""


# ---------------------------------------------------------------------------
# 6. Figure-free doc: byte-identical .md (PP-1 byte-identity preserved)
# ---------------------------------------------------------------------------


class TestFigureFreeByteIdentity:
    """PP-4 AC6: a document with no figures returns text unchanged."""

    def test_no_figures_returns_text_unchanged(self, tmp_path: Path) -> None:
        """When no figures are extracted, _describe_and_embed_figures returns
        the original text without modification (preserving PP-1 byte-identity)."""
        pipeline = _make_pipeline()
        state = DocumentState(handle=_make_handle(page_count=3))
        result = _make_result()

        input_text = assemble_pages(["First page.", "Second page.", "Third page."])

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:  # noqa: N806
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[], cap_reached=False
            )
            out = pipeline._describe_and_embed_figures(state, result, tmp_path, input_text)

        assert out == input_text, (
            "A figure-free document must return text byte-identically; "
            f"got {len(out)} bytes, expected {len(input_text)} bytes."
        )

    def test_no_figures_leaves_fragment_files_untouched(self, tmp_path: Path) -> None:
        """_describe_and_embed_figures never touches fragment files.

        Fragment writes are the sole responsibility of _rewrite_all_fragments,
        called unconditionally from _phase_assemble.  When no figures are
        extracted, _describe_and_embed_figures returns early and must not write
        or modify any fragment file.
        """
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path
        state = DocumentState(handle=_make_handle(page_count=2))
        result = _make_result()

        # Pre-existing fragment for page 1.
        frag_dir = tmp_path / "pp4_fake" / "pages"
        frag_dir.mkdir(parents=True, exist_ok=True)
        frag1 = frag_dir / "00001.md"
        frag1.write_text("Original fragment body.", encoding="utf-8")
        mtime_before = frag1.stat().st_mtime

        input_text = assemble_pages(["Original fragment body.", "Page 2 body."])

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:  # noqa: N806
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[], cap_reached=False
            )
            pipeline._describe_and_embed_figures(state, result, tmp_path, input_text)

        mtime_after = frag1.stat().st_mtime
        assert mtime_before == mtime_after, (
            "Fragment file must not be touched when no figures are extracted."
        )


# ---------------------------------------------------------------------------
# 7. Fragment == final .md byte-identity for figured docs with phantom refs
# ---------------------------------------------------------------------------


class TestFragmentFinalByteIdentity:
    """PP-4: stitch(fragments) == final .md for every doc.

    _rewrite_all_fragments is the single authoritative fragment rewrite, called
    unconditionally from _phase_assemble after strip_phantom_images and the figure
    phase (if any).  These unit tests mirror _phase_assemble's call sequence:
    call _describe_and_embed_figures, then call _rewrite_all_fragments on the
    result, then assert stitch(fragments) == final .md.
    """

    def test_stitch_fragments_equals_final_md_with_phantom_ref(self, tmp_path: Path) -> None:
        """Doc with a figure AND a phantom image ref on the same page:

        1. Strip phantom ref first (as _phase_assemble does before calling the
           figure phase).
        2. Write raw (pre-strip) fragments (as PP-1 flush does).
        3. Run _describe_and_embed_figures on the stripped body → figure inline.
        4. Run _rewrite_all_fragments on the result (as _phase_assemble does).
        5. _stitch_fragments must equal the final .md byte-for-byte and the
           fragment for page 1 must not contain the phantom ref.
        """
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path
        state = DocumentState(handle=_make_handle(page_count=2))
        result = _make_result()

        phantom_ref = "![phantom](figures/ghost.png)"
        raw_page1 = f"Page one body with a phantom ref. {phantom_ref}"
        raw_page2 = "Page two body, no figures."

        stripped_page1 = "Page one body with a phantom ref. "
        stripped_body = assemble_pages([stripped_page1, raw_page2])

        # Write the raw (pre-strip) fragments — what PP-1 flush produces.
        frag_dir = tmp_path / "pp4_fake" / "pages"
        frag_dir.mkdir(parents=True, exist_ok=True)
        (frag_dir / "00001.md").write_text(raw_page1, encoding="utf-8")
        (frag_dir / "00002.md").write_text(raw_page2, encoding="utf-8")

        mock_fig = _make_mock_fig(tmp_path, fig_num=1, page_num=1)

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:  # noqa: N806
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            final_md = pipeline._describe_and_embed_figures(state, result, tmp_path, stripped_body)

        # Mimic _phase_assemble's unconditional rewrite that follows _describe_and_embed_figures.
        pipeline._rewrite_all_fragments(state, tmp_path, final_md)

        # Figure must be inline in page 1.
        pages = split_native_pages(final_md)
        assert len(pages) == 2
        assert "**Figure 1** (page 1)" in pages[0], "Figure must be inline in page 1."
        assert "ghost.png" not in final_md, "Phantom ref must not appear in the final .md."

        stitched = pipeline._stitch_fragments(state, tmp_path)
        assert stitched == final_md, (
            "stitch(fragments) must equal the final .md byte-for-byte.\n"
            f"stitched ({len(stitched)} bytes):\n{stitched!r}\n\n"
            f"final_md ({len(final_md)} bytes):\n{final_md!r}"
        )

        frag1_text = (frag_dir / "00001.md").read_text(encoding="utf-8")
        assert "ghost.png" not in frag1_text, (
            "Fragment page 1 must contain the stripped body, not the raw OCR body."
        )

    def test_non_figure_phantom_page_fragment_matches_final_md(self, tmp_path: Path) -> None:
        """Phantom ref on a NON-figure page: fragment must not contain the phantom.

        Repro: 2 pages, phantom ghost.png on page 2 (no figure), figure on page 1.
        PP-1 writes raw fragments. strip_phantom_images removes ghost.png. Figure
        phase embeds figure on page 1. _rewrite_all_fragments (called from
        _phase_assemble) rewrites BOTH fragments.
        """
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path
        state = DocumentState(handle=_make_handle(page_count=2))
        result = _make_result()

        raw_page1 = "Page one body, no phantom refs."
        phantom_ref = "![ghost](figures/ghost.png)"
        raw_page2 = f"Page two body with a phantom ref. {phantom_ref}"

        stripped_page2 = "Page two body with a phantom ref. "
        stripped_body = assemble_pages([raw_page1, stripped_page2])

        frag_dir = tmp_path / "pp4_fake" / "pages"
        frag_dir.mkdir(parents=True, exist_ok=True)
        (frag_dir / "00001.md").write_text(raw_page1, encoding="utf-8")
        (frag_dir / "00002.md").write_text(raw_page2, encoding="utf-8")

        mock_fig = _make_mock_fig(tmp_path, fig_num=1, page_num=1)

        with patch("socr.pipeline.orchestrator.FigureExtractor") as MockExtractor:  # noqa: N806
            MockExtractor.return_value.extract.return_value = ExtractionResult(
                figures=[mock_fig], cap_reached=False
            )
            final_md = pipeline._describe_and_embed_figures(state, result, tmp_path, stripped_body)

        # Mimic _phase_assemble's unconditional rewrite.
        pipeline._rewrite_all_fragments(state, tmp_path, final_md)

        assert "ghost.png" not in final_md, "Phantom ref must be absent from the final .md."

        frag2_text = (frag_dir / "00002.md").read_text(encoding="utf-8")
        assert "ghost.png" not in frag2_text, (
            "Fragment for non-figure phantom page must not contain ghost.png after rewrite."
        )

        stitched = pipeline._stitch_fragments(state, tmp_path)
        assert stitched == final_md, (
            "stitch(fragments) must equal the final .md byte-for-byte.\n"
            f"stitched ({len(stitched)} bytes):\n{stitched!r}\n\n"
            f"final_md ({len(final_md)} bytes):\n{final_md!r}"
        )

    def test_figure_free_doc_with_phantom_rewrites_fragments(self, tmp_path: Path) -> None:
        """Figure-free doc with a phantom ref: _rewrite_all_fragments fixes the fragment.

        This is the case missed by REVISE round 2, where _rewrite_all_fragments was
        only called from inside _describe_and_embed_figures (which early-returns when
        no figures are extracted). With the fix, _phase_assemble calls
        _rewrite_all_fragments unconditionally on the stripped final_text, covering
        figure-free docs too.

        Simulates _phase_assemble's figure-free flow directly:
        1. Write raw (pre-strip) fragment with phantom ref.
        2. Compute stripped final text (no figures, phantom removed).
        3. Call _rewrite_all_fragments(state, output_dir, stripped_final).
        4. Assert stitch(fragments) == stripped_final AND phantom is absent from both.
        """
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path
        state = DocumentState(handle=_make_handle(page_count=2))

        phantom_ref = "![ghost](figures/ghost.png)"
        raw_page1 = "Page one clean body."
        raw_page2 = f"Page two body with phantom. {phantom_ref}"

        stripped_page2 = "Page two body with phantom. "
        stripped_final = assemble_pages([raw_page1, stripped_page2])

        # Write raw pre-strip fragments (what PP-1 flush emits).
        frag_dir = tmp_path / "pp4_fake" / "pages"
        frag_dir.mkdir(parents=True, exist_ok=True)
        (frag_dir / "00001.md").write_text(raw_page1, encoding="utf-8")
        (frag_dir / "00002.md").write_text(raw_page2, encoding="utf-8")

        # _phase_assemble calls _rewrite_all_fragments unconditionally after strip.
        pipeline._rewrite_all_fragments(state, tmp_path, stripped_final)

        # ghost.png must be absent from the fragment.
        frag2_text = (frag_dir / "00002.md").read_text(encoding="utf-8")
        assert "ghost.png" not in frag2_text, (
            "Fragment page 2 must be rewritten to the stripped body (no ghost.png)."
        )

        # stitch(fragments) must equal the stripped final text.
        stitched = pipeline._stitch_fragments(state, tmp_path)
        assert stitched == stripped_final, (
            "stitch(fragments) must equal the stripped final .md byte-for-byte.\n"
            f"stitched ({len(stitched)} bytes):\n{stitched!r}\n\n"
            f"stripped_final ({len(stripped_final)} bytes):\n{stripped_final!r}"
        )


# ---------------------------------------------------------------------------
# 8. ExtractionResult.cap_page field
# ---------------------------------------------------------------------------


class TestExtractionResultCapPage:
    """PP-4: ExtractionResult carries cap_page for localising the cap AuditEvent."""

    def test_cap_page_defaults_to_none(self) -> None:
        r = ExtractionResult()
        assert r.cap_page is None

    def test_cap_page_set_when_provided(self) -> None:
        r = ExtractionResult(figures=[], cap_reached=True, cap_page=4)
        assert r.cap_page == 4

    def test_extractor_sets_cap_page_when_cap_hit(self, tmp_path: Path) -> None:
        """FigureExtractor.extract sets cap_page to the first unprocessed page."""
        pytest.importorskip("fitz")
        pytest.importorskip("PIL")

        import io

        import fitz
        from PIL import Image

        from socr.figures.extractor import FigureExtractor

        # 3-page PDF with one big embedded image per page.
        buf = io.BytesIO()
        Image.new("RGB", (400, 300), color=(200, 100, 50)).save(buf, format="JPEG", quality=95)
        jpg = buf.getvalue()

        doc = fitz.open()
        for _ in range(3):
            p = doc.new_page(width=612, height=792)
            p.insert_image(fitz.Rect(50, 200, 400, 500), stream=jpg)
        pdf_path = tmp_path / "cap_test.pdf"
        doc.save(pdf_path)
        doc.close()

        result = FigureExtractor(max_total=1, max_per_page=1).extract(pdf_path)
        assert result.cap_reached
        # cap_page must be 2 (page 2 is the first page that was not processed).
        assert result.cap_page == 2, (
            f"Expected cap_page=2 (first unprocessed page) but got {result.cap_page}."
        )
