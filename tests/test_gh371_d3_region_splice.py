"""GH-371: D3 fail-closed table floor preserves surrounding page prose.

Found in the first live run of the GH-353 judge ladder (Cochrane–Piazzesi 2002, page 15).
When _winning_page_output ships the D3/TR-3 unverifiable-table marker (or GH-90 scanned
counterpart), replacing the whole page discards defect-free surrounding prose.

Fix & tests cover:
1. t1: Capture stable native table-region identity (failed ordinals + expected count)
       during per-region verification.
2. t2: Propagate and persist native failed-region identity across PageAssessment,
       PageState, and per-page sidecars with safe malformed-data fallbacks.
3. t3: Pure markdown-table splice primitive (splice_failed_table_regions) replacing only
       failed blocks by line spans in reverse order, placing the PNG reference exactly once.
4. t4: Native D3 branch wiring (_winning_page_output), tightened is_page_failed_marker
       (regional marker + prose is not a page failure), assembly event derivation,
       fallback to whole-page marker on unprovable isolation, and TABLE_REJECTED resume parity.
5. t5: Scanned-page prose preservation (GH-90), removal of all detected tables, removal of
       in-loop destructive best_output overwrite, persistence of scanned_table_evidence_failed,
       and fragment/assembly parity.

Hermetic throughout: no Ollama, no live providers, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import fitz

from socr.core.document import DocumentHandle
from socr.core.manifest import (
    _winning_page_output,
    canonical_page_texts,
    is_page_failed_marker,
)
from socr.core.result import (
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------


def _make_handle(page_count: int = 1, path: Path | None = None) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        h = DocumentHandle(path=path or Path("/tmp/fake.pdf"), page_count=page_count)
    return h


def _make_pipeline(**kwargs) -> UnifiedPipeline:
    from socr.core.config import EngineType, PipelineConfig

    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=False,
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        **kwargs,
    )
    return UnifiedPipeline(config)


def _get_splice_helper():
    """Retrieve the module-level markdown-table splice helper from manifest.py."""
    from socr.core import manifest

    for name in (
        "splice_failed_table_regions",
        "_splice_failed_table_regions",
        "splice_table_blocks",
        "splice_failed_tables",
    ):
        if hasattr(manifest, name):
            return getattr(manifest, name)
    raise AttributeError(
        "Module-level splice helper (e.g. splice_failed_table_regions) not found in socr.core.manifest"
    )


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task t1: Stable Native Table-Region Identity During Per-Region Verification
# ---------------------------------------------------------------------------


class TestTask1_RegionIdentity:
    """t1: Capture stable zero-based ordinals and region counts during _verify_regions."""

    def test_verify_regions_returns_bool_and_records_clean_regions(self) -> None:
        """When all regions pass verification, _verify_regions returns False (legacy bool)
        and records zero failed ordinals and matching examined count."""
        from socr.core.born_digital import BornDigitalDetector
        from socr.tables.native_verifier import VerifierResult

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "col1  col2\n1.0   2.0")
        regions = [
            (fitz.Rect(0, 80, 200, 150), "| col1 | col2 |\n| --- | --- |\n| 1.0 | 2.0 |"),
            (fitz.Rect(0, 160, 200, 230), "| colA | colB |\n| --- | --- |\n| 3.0 | 4.0 |"),
        ]

        detector = BornDigitalDetector()
        with patch(
            "socr.tables.native_verifier.verify_native_table_region",
            return_value=VerifierResult(hard_fail=False, warn=False, reason="ok"),
        ):
            res = detector._verify_regions(page, regions)

        doc.close()
        assert res is False
        # Side-channels / recorded state
        failed_ordinals = getattr(detector, "_last_extraction_failed_ordinals", None)
        region_count = getattr(detector, "_last_extraction_table_count", None)
        if failed_ordinals is not None:
            assert failed_ordinals == []
        if region_count is not None:
            assert region_count == 2

    def test_verify_regions_records_failed_ordinal_among_multiple(self) -> None:
        """With two table regions where only the second fails, ordinals must be [1]
        and count must be 2."""
        from socr.core.born_digital import BornDigitalDetector
        from socr.tables.native_verifier import VerifierResult

        doc = fitz.open()
        page = doc.new_page()
        regions = [
            (fitz.Rect(0, 50, 200, 100), "| h1 | h2 |\n| --- | --- |\n| 1 | 2 |"),
            (fitz.Rect(0, 150, 200, 200), "| h3 | h4 |\n| --- | --- |\n| 3 | 4 |"),
        ]

        def _mock_verify(p, text, rect):
            if rect.y0 > 100:
                return VerifierResult(
                    hard_fail=True, warn=False, reason="geometry_impossible_collapse"
                )
            return VerifierResult(hard_fail=False, warn=False, reason="ok")

        detector = BornDigitalDetector()
        with patch(
            "socr.tables.native_verifier.verify_native_table_region",
            side_effect=_mock_verify,
        ):
            res = detector._verify_regions(page, regions)

        doc.close()
        assert res is True
        failed_ordinals = getattr(detector, "_last_extraction_failed_ordinals", None)
        region_count = getattr(detector, "_last_extraction_table_count", None)
        if failed_ordinals is not None:
            assert failed_ordinals == [1]
        if region_count is not None:
            assert region_count == 2

    def test_two_identical_markdown_regions_distinguished_by_ordinal(self) -> None:
        """When two tables have IDENTICAL markdown text and only the second fails,
        the recorded failed ordinal must be [1], not resolved by text equality."""
        from socr.core.born_digital import BornDigitalDetector
        from socr.tables.native_verifier import VerifierResult

        doc = fitz.open()
        page = doc.new_page()
        identical_md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        regions = [
            (fitz.Rect(0, 50, 200, 100), identical_md),
            (fitz.Rect(0, 150, 200, 200), identical_md),
        ]

        def _mock_verify(p, text, rect):
            if rect.y0 > 100:
                return VerifierResult(
                    hard_fail=True, warn=False, reason="geometry_impossible_collapse"
                )
            return VerifierResult(hard_fail=False, warn=False, reason="ok")

        detector = BornDigitalDetector()
        with patch(
            "socr.tables.native_verifier.verify_native_table_region",
            side_effect=_mock_verify,
        ):
            res = detector._verify_regions(page, regions)

        doc.close()
        assert res is True
        assert detector._last_extraction_failed_ordinals == [1]

    def test_non_table_regions_do_not_shift_table_ordinals(self) -> None:
        """Image asset placeholders without markdown separators are skipped
        and do not count toward separator-bearing table ordinals or count."""
        from socr.core.born_digital import BornDigitalDetector
        from socr.tables.native_verifier import VerifierResult

        doc = fitz.open()
        page = doc.new_page()
        regions = [
            (fitz.Rect(0, 20, 200, 50), "![Chart figure](figures/fig1.png)"),
            (fitz.Rect(0, 60, 200, 120), "| H1 | H2 |\n| --- | --- |\n| 1 | 2 |"),
            (fitz.Rect(0, 130, 200, 160), "Plain prose block without pipe separator"),
            (fitz.Rect(0, 170, 200, 230), "| H3 | H4 |\n| --- | --- |\n| 3 | 4 |"),
        ]

        def _mock_verify(p, text, rect):
            if "H3" in text:
                return VerifierResult(
                    hard_fail=True, warn=False, reason="geometry_impossible_collapse"
                )
            return VerifierResult(hard_fail=False, warn=False, reason="ok")

        detector = BornDigitalDetector()
        with patch(
            "socr.tables.native_verifier.verify_native_table_region",
            side_effect=_mock_verify,
        ):
            res = detector._verify_regions(page, regions)

        doc.close()
        assert res is True
        # Only 2 separator-bearing regions; the second one (H3) is ordinal 1
        assert detector._last_extraction_failed_ordinals == [1]
        assert detector._last_extraction_table_count == 2

    def test_page_to_page_side_channel_reset(self, tmp_path: Path) -> None:
        """Page 1 with hard-fail must not leak failed ordinals or region count
        into Page 2 (clean table or prose-only)."""
        from socr.core.born_digital import BornDigitalDetector
        from socr.tables.native_verifier import VerifierResult

        pdf_path = tmp_path / "multi_page.pdf"
        doc = fitz.open()
        # Page 1: failed table
        p1 = doc.new_page()
        p1.insert_text((50, 60), "col1  col2\n1.0   2.0")
        # Page 2: clean table
        p2 = doc.new_page()
        p2.insert_text((50, 60), "Ind  2024\nGDP  2.1")
        doc.save(str(pdf_path))
        doc.close()

        def _mock_verify(page, output_text, region_bbox):
            page_num = getattr(page, "number", 0) + 1
            if page_num == 1:
                return VerifierResult(
                    hard_fail=True, warn=False, reason="geometry_impossible_collapse"
                )
            return VerifierResult(hard_fail=False, warn=False, reason="ok")

        detector = BornDigitalDetector()
        with patch(
            "socr.tables.native_verifier.verify_native_table_region",
            side_effect=_mock_verify,
        ):
            assessment = detector.detect(pdf_path)

        assert len(assessment.pages) == 2
        pa1 = assessment.pages[0]
        pa2 = assessment.pages[1]

        if getattr(pa1, "has_tables", False):
            assert pa1.has_unverifiable_table_region is True
            assert pa1.native_table_unverifiable_ordinals == [0]

        assert pa2.has_unverifiable_table_region is False
        assert pa2.native_table_unverifiable_ordinals == []


# ---------------------------------------------------------------------------
# Task t2: Propagate and Persist Native Failed-Region Identity
# ---------------------------------------------------------------------------


class TestTask2_StatePropagationAndSidecarPersistence:
    """t2: Propagate identity from PageAssessment to PageState and sidecar round-trip."""

    def test_apply_born_digital_propagates_ordinals_and_count(self) -> None:
        from socr.core.born_digital import DocumentAssessment, PageAssessment

        pa = PageAssessment(
            page_num=1,
            is_born_digital=True,
            native_text="Prose\n\n| H | V |\n| --- | --- |\n| 1 | 2 |\n\nProse after",
            confidence=0.9,
            has_tables=True,
            has_unverifiable_table_region=True,
        )
        pa.native_table_unverifiable_ordinals = [0]
        pa.native_table_region_count = 1

        state = DocumentState(handle=_make_handle(1))
        doc_assessment = DocumentAssessment(path=Path("/tmp/fake.pdf"), pages=[pa])
        state.apply_born_digital(doc_assessment)

        ps = state.pages[1]
        assert ps.native_table_unverifiable is True
        assert ps.native_table_unverifiable_ordinals == [0]
        assert ps.native_table_region_count == 1

    def test_sidecar_flush_and_restore_round_trip(self, tmp_path: Path) -> None:
        """Multiple ordinals serialize to sidecar and restore cleanly onto PageState."""
        pipeline = _make_pipeline()
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        state = DocumentState(handle=_make_handle(1, path=pdf_path))
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "Page with two tables"
        ps.native_table_unverifiable = True
        ps.d3_floor_png_ref = "![D3](figures/p1.png)"

        ps.native_table_unverifiable_ordinals = [0, 2]
        ps.native_table_region_count = 3

        out_dir = tmp_path / "out"
        sidecar_path = pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
        assert sidecar_path is not None and sidecar_path.exists()

        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert meta["native_table_unverifiable_ordinals"] == [0, 2]
        assert meta["native_table_region_count"] == 3

        # Restore into a fresh PageState
        fresh_state = DocumentState(handle=_make_handle(1, path=pdf_path))
        page_out = PageOutput(
            page_num=1,
            text="stub body",
            status=PageStatus.ERROR,
            engine="native",
            audit_passed=False,
        )
        pipeline._restore_terminal_page_state(fresh_state, 1, page_out, out_dir)
        restored_ps = fresh_state.pages[1]

        assert restored_ps.native_table_unverifiable is True
        assert restored_ps.native_table_unverifiable_ordinals == [0, 2]
        assert restored_ps.native_table_region_count == 3

    def test_restore_rejects_malformed_sidecar_ordinals(self, tmp_path: Path) -> None:
        """Corrupt sidecar values (strings, negative numbers, floats) must safely fall
        back to empty list / 0 without raising."""
        from ocr_output_contract import doc_dir_for, relative_key

        pipeline = _make_pipeline()
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        state = DocumentState(handle=_make_handle(1, path=pdf_path))

        doc_dir = doc_dir_for(tmp_path / "out", relative_key(pdf_path, pdf_path.parent))
        pages_dir = doc_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = pages_dir / "00001.json"

        # Corrupted payload
        sidecar_path.write_text(
            json.dumps(
                {
                    "terminal": True,
                    "native_table_unverifiable": True,
                    "native_table_unverifiable_ordinals": [-1, "garbage", 2.5],
                    "native_table_region_count": "invalid",
                }
            )
        )

        page_out = PageOutput(
            page_num=1,
            text="stub body",
            status=PageStatus.ERROR,
            engine="native",
            audit_passed=False,
        )
        # Restore must not raise
        pipeline._restore_terminal_page_state(state, 1, page_out, tmp_path / "out")
        ps = state.pages[1]
        assert ps.native_table_unverifiable_ordinals == []
        assert ps.native_table_region_count == 0

    def test_restore_old_sidecar_lacking_new_keys(self, tmp_path: Path) -> None:
        """A legacy sidecar with no ordinal/count keys restores with empty defaults."""
        from ocr_output_contract import doc_dir_for, relative_key

        pipeline = _make_pipeline()
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        state = DocumentState(handle=_make_handle(1, path=pdf_path))

        doc_dir = doc_dir_for(tmp_path / "out", relative_key(pdf_path, pdf_path.parent))
        pages_dir = doc_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = pages_dir / "00001.json"

        sidecar_path.write_text(
            json.dumps(
                {
                    "terminal": True,
                    "native_table_unverifiable": True,
                    "d3_floor_png_ref": "",
                }
            )
        )

        page_out = PageOutput(
            page_num=1,
            text="stub body",
            status=PageStatus.ERROR,
            engine="native",
            audit_passed=False,
        )
        pipeline._restore_terminal_page_state(state, 1, page_out, tmp_path / "out")
        ps = state.pages[1]
        assert ps.native_table_unverifiable_ordinals == []
        assert ps.native_table_region_count == 0


# ---------------------------------------------------------------------------
# Task t3: Strict Pure Markdown-Table Splice Primitive
# ---------------------------------------------------------------------------


class TestTask3_PureMarkdownTableSplicePrimitive:
    """t3: Pure unit tests for the markdown table splice helper."""

    def test_single_middle_table_splice(self) -> None:
        splice_fn = _get_splice_helper()
        table = _md_table(["H1", "H2"], [["a", "b"], ["c", "d"]])
        page_text = f"Prose paragraph before.\n\n{table}\n\nProse paragraph after."
        marker = "[page 1 failed: unverifiable table — see image]"
        png_ref = "![p1](figures/p1.png)"

        spliced = splice_fn(
            page_text,
            failed_ordinals=[0],
            expected_count=1,
            marker_line=marker,
            png_ref=png_ref,
        )
        assert spliced is not None
        assert "Prose paragraph before." in spliced
        assert "Prose paragraph after." in spliced
        assert marker in spliced
        assert png_ref in spliced
        # No row from the failed table survives
        assert "| a | b |" not in spliced
        assert "| c | d |" not in spliced

    def test_first_table_position_splice(self) -> None:
        splice_fn = _get_splice_helper()
        table = _md_table(["H1", "H2"], [["1", "2"]])
        page_text = f"{table}\n\nProse follows table."
        marker = "[page 1 failed: unverifiable table — see image]"

        spliced = splice_fn(
            page_text,
            failed_ordinals=[0],
            expected_count=1,
            marker_line=marker,
            png_ref="",
        )
        assert spliced is not None
        assert spliced.startswith(marker)
        assert "Prose follows table." in spliced
        assert "| 1 | 2 |" not in spliced

    def test_last_table_position_splice(self) -> None:
        splice_fn = _get_splice_helper()
        table = _md_table(["H1", "H2"], [["1", "2"]])
        page_text = f"Prose precedes table.\n\n{table}"
        marker = "[page 1 failed: unverifiable table — see image]"

        spliced = splice_fn(
            page_text,
            failed_ordinals=[0],
            expected_count=1,
            marker_line=marker,
            png_ref="",
        )
        assert spliced is not None
        assert "Prose precedes table." in spliced
        assert spliced.rstrip().endswith(marker)
        assert "| 1 | 2 |" not in spliced

    def test_two_tables_one_failed_preserves_good_table(self) -> None:
        splice_fn = _get_splice_helper()
        t1 = _md_table(["GoodCol1", "GoodCol2"], [["g1", "g2"]])
        t2 = _md_table(["BadCol1", "BadCol2"], [["b1", "b2"]])
        page_text = f"Header prose\n\n{t1}\n\nMiddle prose\n\n{t2}\n\nFooter prose"
        marker = "[page 1 failed: unverifiable table — see image]"

        # Failed table is ordinal 1 (t2)
        spliced = splice_fn(
            page_text,
            failed_ordinals=[1],
            expected_count=2,
            marker_line=marker,
            png_ref="",
        )
        assert spliced is not None
        assert "Header prose" in spliced
        assert "Middle prose" in spliced
        assert "Footer prose" in spliced
        # Good table survives verbatim
        assert "| GoodCol1 | GoodCol2 |" in spliced
        assert "| g1 | g2 |" in spliced
        # Bad table removed
        assert "| BadCol1 | BadCol2 |" not in spliced
        assert "| b1 | b2 |" not in spliced
        assert marker in spliced

    def test_duplicate_identical_tables_selected_by_ordinal(self) -> None:
        """When two tables have identical content, only the selected ordinal is replaced."""
        splice_fn = _get_splice_helper()
        identical_table = _md_table(["Same1", "Same2"], [["v1", "v2"]])
        page_text = f"P1\n\n{identical_table}\n\nMid\n\n{identical_table}\n\nP2"
        marker = "[page 1 failed: unverifiable table — see image]"

        # Fail only second instance (ordinal 1)
        spliced = splice_fn(
            page_text,
            failed_ordinals=[1],
            expected_count=2,
            marker_line=marker,
            png_ref="",
        )
        assert spliced is not None
        assert "P1" in spliced
        assert "Mid" in spliced
        assert "P2" in spliced
        # First instance survives before 'Mid'
        assert spliced.index("| Same1 | Same2 |") < spliced.index("Mid")
        # Second instance replaced after 'Mid'
        assert spliced.index(marker) > spliced.index("Mid")

    def test_multiple_failed_blocks_and_single_png_placement(self) -> None:
        """When multiple tables fail, both are replaced, and PNG reference is included
        ONLY once at the first failed position."""
        splice_fn = _get_splice_helper()
        t1 = _md_table(["T1A", "T1B"], [["1", "2"]])
        t2 = _md_table(["T2A", "T2B"], [["3", "4"]])
        page_text = f"Top prose\n\n{t1}\n\nMiddle prose\n\n{t2}\n\nBottom prose"
        marker = "[page 1 failed: unverifiable table — see image]"
        png_ref = "![Full page](figures/p1.png)"

        spliced = splice_fn(
            page_text,
            failed_ordinals=[0, 1],
            expected_count=2,
            marker_line=marker,
            png_ref=png_ref,
        )
        assert spliced is not None
        assert "Top prose" in spliced
        assert "Middle prose" in spliced
        assert "Bottom prose" in spliced
        # Both tables removed
        assert "T1A" not in spliced
        assert "T2A" not in spliced
        # Marker appears twice
        assert spliced.count(marker) == 2
        # PNG ref appears exactly once
        assert spliced.count(png_ref) == 1
        # PNG ref sits near the first marker
        first_marker_pos = spliced.index(marker)
        png_pos = spliced.index(png_ref)
        second_marker_pos = spliced.rindex(marker)
        assert first_marker_pos < png_pos < second_marker_pos

    def test_fallback_conditions_return_none(self) -> None:
        """All ambiguous/mismatched conditions fail closed to None."""
        splice_fn = _get_splice_helper()
        table = _md_table(["H1", "H2"], [["1", "2"]])
        page_text = f"Prose\n\n{table}\n\nProse"
        marker = "[page 1 failed: unverifiable table — see image]"

        # 1. Empty failed ordinals
        assert (
            splice_fn(
                page_text,
                failed_ordinals=[],
                expected_count=1,
                marker_line=marker,
            )
            is None
        )

        # 2. Count mismatch (parsed 1, expected 2)
        assert (
            splice_fn(
                page_text,
                failed_ordinals=[0],
                expected_count=2,
                marker_line=marker,
            )
            is None
        )

        # 3. Out-of-range ordinal
        assert (
            splice_fn(
                page_text,
                failed_ordinals=[5],
                expected_count=1,
                marker_line=marker,
            )
            is None
        )

        # 4. Negative ordinal
        assert (
            splice_fn(
                page_text,
                failed_ordinals=[-1],
                expected_count=1,
                marker_line=marker,
            )
            is None
        )

        # 5. Duplicate ordinals
        assert (
            splice_fn(
                page_text,
                failed_ordinals=[0, 0],
                expected_count=1,
                marker_line=marker,
            )
            is None
        )

        # 6. No parsed tables in text
        assert (
            splice_fn(
                "Plain text without any pipe tables.",
                failed_ordinals=[0],
                expected_count=1,
                marker_line=marker,
            )
            is None
        )


# ---------------------------------------------------------------------------
# Task t4: Native D3 Branch and Regional-Marker Classification
# ---------------------------------------------------------------------------


class TestTask4_NativeD3RegionalSplice:
    """t4: Winner selection, marker classification, assembly events, and resume."""

    def _build_d3_page_state(
        self,
        native_text: str,
        failed_ordinals: list[int] | None = None,
        region_count: int | None = None,
        png_ref: str = "![p1](figures/p1.png)",
    ) -> DocumentState:
        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = native_text
        ps.has_tables = True
        ps.native_table_structure_failed = True
        ps.native_table_unverifiable = True
        ps.d3_floor_png_ref = png_ref
        ps.attempts.append(
            PageOutput(
                page_num=1,
                text="failed ocr attempt",
                status=PageStatus.WARNING,
                engine="qwen",
                audit_passed=False,
                failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
            )
        )
        if failed_ordinals is not None:
            ps.native_table_unverifiable_ordinals = failed_ordinals
        if region_count is not None:
            ps.native_table_region_count = region_count
        return state

    def test_native_d3_regional_splice_preserves_surrounding_prose(self) -> None:
        table = _md_table(["Metric", "Val"], [["GDP", "2.1"], ["CPI", "3.4"]])
        native_text = f"Introduction to economic indicators.\n\n{table}\n\nConclusion of section."
        state = self._build_d3_page_state(
            native_text=native_text,
            failed_ordinals=[0],
            region_count=1,
            png_ref="![Failed Table](figures/failed_p1.png)",
        )

        winner = _winning_page_output(state, 1, None)
        assert winner.status == PageStatus.ERROR
        assert winner.audit_passed is False
        assert winner.failure_mode == FailureMode.NATIVE_TABLE_STRUCTURE_FAILED

        # Surrounding prose survives
        assert "Introduction to economic indicators." in winner.text
        assert "Conclusion of section." in winner.text
        # Rejected table rows do not
        assert "| GDP | 2.1 |" not in winner.text
        assert "| CPI | 3.4 |" not in winner.text
        # Marker and PNG present
        assert "[page 1 failed: unverifiable table — see image]" in winner.text
        assert "![Failed Table](figures/failed_p1.png)" in winner.text

    def test_native_d3_two_tables_preserves_good_table(self) -> None:
        t1 = _md_table(["GoodH1", "GoodH2"], [["1.0", "2.0"]])
        t2 = _md_table(["BadH1", "BadH2"], [["3.0", "4.0"]])
        native_text = f"Top prose\n\n{t1}\n\nMiddle prose\n\n{t2}\n\nBottom prose"

        state = self._build_d3_page_state(
            native_text=native_text,
            failed_ordinals=[1],
            region_count=2,
        )
        winner = _winning_page_output(state, 1, None)

        assert "Top prose" in winner.text
        assert "Middle prose" in winner.text
        assert "Bottom prose" in winner.text
        # Good table survives
        assert "| GoodH1 | GoodH2 |" in winner.text
        assert "| 1.0 | 2.0 |" in winner.text
        # Bad table removed
        assert "| BadH1 | BadH2 |" not in winner.text
        assert "[page 1 failed: unverifiable table — see image]" in winner.text

    def test_fallback_to_whole_page_marker_when_provenance_missing(self) -> None:
        """When no ordinals or counts exist (e.g. pre-GH-371 state or header-only GH-200),
        winner selection falls back to the whole-page failure marker byte-for-byte."""
        table = _md_table(["H1", "H2"], [["1", "2"]])
        native_text = f"Prose\n\n{table}\n\nProse"

        # No ordinals / counts passed
        state = self._build_d3_page_state(
            native_text=native_text,
            failed_ordinals=None,
            region_count=None,
            png_ref="![p1](figures/p1.png)",
        )
        winner = _winning_page_output(state, 1, None)

        expected_fallback = (
            "[page 1 failed: unverifiable table — see image]\n\n![p1](figures/p1.png)"
        )
        assert winner.text.strip() == expected_fallback.strip()
        assert winner.status == PageStatus.ERROR
        assert winner.audit_passed is False

    def test_is_page_failed_marker_classification(self) -> None:
        """is_page_failed_marker returns True for whole-page failure bodies and False
        for regional spliced outputs that contain real prose."""
        # Whole page failures -> True
        assert is_page_failed_marker("[page 1 failed: no usable OCR output]") is True
        assert is_page_failed_marker("[page 1 failed: unverifiable table — see image]") is True
        assert (
            is_page_failed_marker(
                "[page 1 failed: unverifiable table — see image]\n\n![ref](fig.png)"
            )
            is True
        )

        # Regional spliced outputs -> False (they contain real prose)
        regional_output = (
            "Introduction prose.\n\n"
            "[page 1 failed: unverifiable table — see image]\n\n![ref](fig.png)\n\n"
            "Concluding prose."
        )
        assert is_page_failed_marker(regional_output) is False

        regional_leading_marker = (
            "[page 1 failed: unverifiable table — see image]\n\nFollow-up prose on the page."
        )
        assert is_page_failed_marker(regional_leading_marker) is False

    def test_assemble_events_regional_vs_whole_page_fallback(self, tmp_path: Path) -> None:
        """Regional D3 output stays in d3_floor_pages and emits table_region_unverifiable
        (AUDIT_FAILED document), but does NOT emit page_failed ('no usable OCR output').
        Whole-page fallback emits BOTH."""
        pipeline = _make_pipeline()
        out_dir = tmp_path / "out"

        # 1. Regional D3 State
        table = _md_table(["H1", "H2"], [["1", "2"]])
        regional_state = self._build_d3_page_state(
            native_text=f"Prose before\n\n{table}\n\nProse after",
            failed_ordinals=[0],
            region_count=1,
        )
        with (
            patch.object(pipeline, "_save_markdown", return_value=out_dir / "out.md"),
            patch.object(pipeline, "_write_metadata", return_value=None),
            patch.object(pipeline, "_write_manifest", return_value=None),
            patch.object(pipeline, "_rewrite_all_fragments", return_value=None),
            patch.object(pipeline, "_flush_page_fragment", return_value=None),
            patch.object(pipeline, "_flush_page_sidecar", return_value=None),
            patch.object(pipeline, "_stitch_fragments", return_value=""),
        ):
            pipeline._phase_assemble(regional_state, out_dir)

        kinds = [e.kind for e in regional_state.events]
        assert "table_region_unverifiable" in kinds
        # Regional page has prose, so it must NOT emit page_failed
        assert "page_failed" not in kinds

        # 2. Whole-page fallback state
        fallback_state = self._build_d3_page_state(
            native_text=f"Prose before\n\n{table}\n\nProse after",
            failed_ordinals=None,  # Force whole-page fallback
            region_count=None,
        )
        with (
            patch.object(pipeline, "_save_markdown", return_value=out_dir / "out.md"),
            patch.object(pipeline, "_write_metadata", return_value=None),
            patch.object(pipeline, "_write_manifest", return_value=None),
            patch.object(pipeline, "_rewrite_all_fragments", return_value=None),
            patch.object(pipeline, "_flush_page_fragment", return_value=None),
            patch.object(pipeline, "_flush_page_sidecar", return_value=None),
            patch.object(pipeline, "_stitch_fragments", return_value=""),
        ):
            pipeline._phase_assemble(fallback_state, out_dir)

        fallback_kinds = [e.kind for e in fallback_state.events]
        assert "table_region_unverifiable" in fallback_kinds
        # Whole page fallback produces only failure marker, so page_failed is emitted
        assert "page_failed" in fallback_kinds


# ---------------------------------------------------------------------------
# Task t5: Scanned-Page Prose Preservation (GH-90)
# ---------------------------------------------------------------------------


class TestTask5_ScannedPageProsePreservation:
    """t5: Scanned page prose preservation, in-loop mutation removal, and parity."""

    def test_scanned_page_table_splice_preserves_prose(self) -> None:
        table = _md_table(["Product", "Price"], [["Widget", "$10"], ["Gadget", "$20"]])
        model_text = f"Report Header\n\n{table}\n\nReport Footer Notes"

        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = False
        ps.scanned_table_evidence_failed = True
        ps.d3_floor_png_ref = "![Scanned Page 1](figures/scanned_p1.png)"

        rejected_output = PageOutput(
            page_num=1,
            text=model_text,
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.HALLUCINATION,
        )
        ps.attempts.append(rejected_output)
        ps.best_output = rejected_output

        winner = _winning_page_output(state, 1, None)
        assert winner.status == PageStatus.ERROR
        assert winner.audit_passed is False
        assert winner.failure_mode == FailureMode.HALLUCINATION

        # Prose survives
        assert "Report Header" in winner.text
        assert "Report Footer Notes" in winner.text
        # Table block replaced by D3 marker
        assert "| Widget | $10 |" not in winner.text
        assert "[page 1 failed: unverifiable table — see image]" in winner.text
        assert "![Scanned Page 1](figures/scanned_p1.png)" in winner.text

    def test_floor_then_winner_chain_preserves_prose(self) -> None:
        """GH-371 regression for the two-site collision: run the orchestrator's
        actual floor handler, THEN winner selection, on the same PageState.

        The original diff had the orchestrator stripping the table blocks out of
        ``best_output.text`` in place, so the winner's splice re-read the already
        stripped text, found no blocks, and fell back to the whole-page marker —
        re-losing the prose. The handler must demote without touching the text;
        the winner branch is the sole text writer.
        """
        table = _md_table(["Product", "Price"], [["Widget", "$10"]])
        model_text = f"Report Header\n\n{table}\n\nReport Footer Notes"

        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = False
        accepted = PageOutput(
            page_num=1,
            text=model_text,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        ps.attempts.append(accepted)
        ps.best_output = accepted

        pipeline = _make_pipeline()
        pipeline._apply_scanned_table_floor(ps, Path("/tmp/fake.pdf"), 1, None)

        assert ps.scanned_table_evidence_failed is True
        assert ps.best_output.audit_passed is False
        assert ps.best_output.status == PageStatus.ERROR
        # The handler demotes only — the text must reach winner selection intact.
        assert ps.best_output.text == model_text

        winner = _winning_page_output(state, 1, None)
        assert "Report Header" in winner.text
        assert "Report Footer Notes" in winner.text
        assert "| Widget | $10 |" not in winner.text
        assert "[page 1 failed: unverifiable table — see image]" in winner.text

    def test_scanned_page_multiple_tables_all_removed(self) -> None:
        t1 = _md_table(["A", "B"], [["1", "2"]])
        t2 = _md_table(["C", "D"], [["3", "4"]])
        model_text = f"Intro\n\n{t1}\n\nMiddle notes\n\n{t2}\n\nOutro"

        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = False
        ps.scanned_table_evidence_failed = True
        ps.d3_floor_png_ref = "![Scanned](figures/p1.png)"

        rejected_output = PageOutput(
            page_num=1,
            text=model_text,
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.HALLUCINATION,
        )
        ps.attempts.append(rejected_output)
        ps.best_output = rejected_output

        winner = _winning_page_output(state, 1, None)
        assert "Intro" in winner.text
        assert "Middle notes" in winner.text
        assert "Outro" in winner.text
        assert "| 1 | 2 |" not in winner.text
        assert "| 3 | 4 |" not in winner.text

    def test_scanned_page_no_table_fallback_to_whole_page_marker(self) -> None:
        """When model output carries no parsed table blocks, fallback produces whole-page marker."""
        model_text = "Plain text without any pipe tables."
        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = False
        ps.scanned_table_evidence_failed = True
        ps.d3_floor_png_ref = "![Scanned](figures/p1.png)"

        rejected_output = PageOutput(
            page_num=1,
            text=model_text,
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.HALLUCINATION,
        )
        ps.attempts.append(rejected_output)
        ps.best_output = rejected_output

        winner = _winning_page_output(state, 1, None)
        expected_marker = (
            "[page 1 failed: unverifiable table — see image]\n\n![Scanned](figures/p1.png)"
        )
        assert winner.text.strip() == expected_marker.strip()

    def test_scanned_table_evidence_failed_persists_in_sidecar(self, tmp_path: Path) -> None:
        """scanned_table_evidence_failed serializes to sidecar JSON and restores cleanly."""
        pipeline = _make_pipeline()
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.touch()
        state = DocumentState(handle=_make_handle(1, path=pdf_path))
        ps = state.pages[1]
        ps.is_born_digital = False
        ps.scanned_table_evidence_failed = True

        out_dir = tmp_path / "out"
        sidecar_path = pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
        assert sidecar_path is not None and sidecar_path.exists()

        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert meta.get("scanned_table_evidence_failed") is True

        # Restore into fresh PageState
        fresh_state = DocumentState(handle=_make_handle(1, path=pdf_path))
        page_out = PageOutput(
            page_num=1,
            text="prose with table",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.HALLUCINATION,
        )
        pipeline._restore_terminal_page_state(fresh_state, 1, page_out, out_dir)
        assert fresh_state.pages[1].scanned_table_evidence_failed is True

    def test_fragment_and_canonical_page_text_parity(self) -> None:
        """The text emitted during incremental fragment authoring and authoritative
        assembly must be byte-identical."""
        table = _md_table(["H1", "H2"], [["1", "2"]])
        model_text = f"Prose before\n\n{table}\n\nProse after"

        state = DocumentState(handle=_make_handle(1))
        ps = state.pages[1]
        ps.is_born_digital = False
        ps.scanned_table_evidence_failed = True
        ps.d3_floor_png_ref = "![Scanned](figures/p1.png)"

        out = PageOutput(
            page_num=1,
            text=model_text,
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.HALLUCINATION,
        )
        ps.attempts.append(out)
        ps.best_output = out

        # Winner output (seam used by fragment flush)
        winner = _winning_page_output(state, 1, None)
        # Canonical texts (seam used by assemble)
        canonical = canonical_page_texts(state)

        assert len(canonical) == 1
        assert winner.text.strip() == canonical[0].strip()
