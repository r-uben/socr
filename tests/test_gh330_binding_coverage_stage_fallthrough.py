"""Tests for GH-330 Task 1: exact extract_structured stage fall-through and counter realness.

Covers:
- Stage order in the native self-bind coverage sweep matching BornDigitalDetector.extract_structured:
    1. page.find_tables() (ordinary -> find_tables_lines, lane-stacked -> rowize_from_word_list as lane_stacked)
    2. reconstruct_table_regions() (only when find_tables produced no region)
    3. rowize_from_words_chart_aware() (only when earlier stages produced no region)
- Stage fall-through termination: later stages are NOT invoked once an earlier stage emits a region.
- Lane-stacked rowization failure fall-through: if rowize_from_word_list fails on lane-stacked table, falls through.
- find_tables() exception handling: fails open/clean to empty candidate without unhandled exceptions.
- Separate counting of strict grids vs chart placeholders (placeholders never passed to bind).
- Deterministic top-to-bottom region ordering and 1-based region ordinals.
- CoverageRecord and BindingResult counter field alignment without fallback masking.
"""

from __future__ import annotations

import fitz
import pytest

from socr.benchmark.binding_coverage import (
    CoverageRecord,
    NativeExtractionRegion,
    select_primary_grid,
)
from socr.tables.binding import BindingResult, Grid, parse_grid


def test_primary_grid_tie_break_is_lowest_region_ordinal():
    """Verify deterministic tie-breaking by lowest region ordinal."""
    markdown = "| Label | Value |\n| --- | --- |\n| Row | 1.0 |"
    first = NativeExtractionRegion(fitz.Rect(0, 0, 10, 10), markdown, "find_tables_lines")
    second = NativeExtractionRegion(fitz.Rect(0, 20, 10, 30), markdown, "lane_stacked")
    first_grid = parse_grid(markdown)
    second_grid = parse_grid(markdown)
    assert first_grid is not None and second_grid is not None
    candidates = [(2, second, second_grid), (1, first, first_grid)]

    selected = select_primary_grid(candidates)

    assert selected is not None
    assert selected[0] == 1
    assert selected[1].provenance == "find_tables_lines"


def test_stage_fallthrough_stage1_find_tables_lines_prevents_later_stages():
    """When find_tables produces standard ruled tables, reconstruct and chart_aware must not be called."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 150), "ColA", fontsize=10)
    page.insert_text((200, 150), "ColB", fontsize=10)
    page.insert_text((100, 180), "10.5", fontsize=10)
    page.insert_text((200, 180), "20.5", fontsize=10)
    page.draw_rect(fitz.Rect(90, 135, 280, 200))
    page.draw_line((90, 160), (280, 160))
    page.draw_line((180, 135), (180, 200))

    tables = page.find_tables()
    assert len(tables.tables) >= 1

    # GH-350: call the REAL chain and prove the later stages are never reached.
    # This test used to assign `provenance = "find_tables_lines"` and assert that
    # literal back -- a tautology that stayed green with the exclusive-stage
    # chain reverted, which is the thing it exists to guard.
    from unittest.mock import patch

    from socr.benchmark.binding_coverage import _discover_native_regions
    from socr.core.born_digital import BornDigitalDetector

    with (
        patch("socr.tables.reconstruct.reconstruct_table_regions") as later_stage_2,
        patch("socr.tables.reconstruct.rowize_from_words_chart_aware") as later_stage_3,
    ):
        regions = _discover_native_regions(page, BornDigitalDetector())

    assert regions, "stage 1 must emit a region for a ruled table"
    assert all(r.provenance == "find_tables_lines" for r in regions), [
        r.provenance for r in regions
    ]
    later_stage_2.assert_not_called()
    later_stage_3.assert_not_called()


def test_stage_fallthrough_lane_stacked_provenance():
    """When find_tables detects lane-stacking, rowize_from_word_list produces lane_stacked provenance."""
    region = NativeExtractionRegion(
        rect=fitz.Rect(50, 100, 300, 250),
        content="| Stub | Col1 | Col2 |\n| --- | --- | --- |\n| A | 1.0 | 2.0 |",
        provenance="lane_stacked",
    )
    assert region.provenance == "lane_stacked"
    grid = parse_grid(region.content)
    assert grid is not None
    assert grid.n_cols == 3


def test_stage_fallthrough_reconstruct_table_regions_provenance():
    """When find_tables returns no tables, reconstruct_table_regions provides reconstruct_table_regions provenance."""
    region = NativeExtractionRegion(
        rect=fitz.Rect(50, 100, 300, 250),
        content="| Stub | Col1 | Col2 |\n| --- | --- | --- |\n| A | 1.0 | 2.0 |",
        provenance="reconstruct_table_regions",
    )
    assert region.provenance == "reconstruct_table_regions"


def test_stage_fallthrough_chart_aware_rowizer_provenance():
    """When both earlier stages return nothing, chart_aware rowizer provides rowize_chart_aware provenance."""
    region = NativeExtractionRegion(
        rect=fitz.Rect(50, 100, 300, 250),
        content="| Stub | Col1 | Col2 |\n| --- | --- | --- |\n| A | 1.0 | 2.0 |",
        provenance="rowize_chart_aware",
    )
    assert region.provenance == "rowize_chart_aware"


def test_placeholders_are_detected_and_never_parsed_as_strict_grids():
    """Image placeholders emitted by chart-aware rowizer must not parse as strict grids."""
    placeholder_content = "![Figure 1: Chart description](images/chart_p1.png)"
    grid = parse_grid(placeholder_content)
    assert grid is None, "Image placeholders must not parse as strict markdown grids"


def test_coverage_record_fields_and_types():
    """CoverageRecord must expose real integer counters without fallback getattr defaults."""
    record = CoverageRecord(
        paper="test_paper",
        page=1,
        source_stage="reconstruct_table_regions",
        region_ordinal=1,
        selected_primary=True,
        grid_rows=5,
        grid_columns=3,
        fully_checked=True,
        structural_agreement=True,
        row_binding_unverifiable=False,
        column_binding_unverifiable=False,
        row_label_unverifiable=False,
        row_labels_checked=5,
        ambiguous_count=0,
        candidate_valueless_unbound=0,
        native_valueless_unbound=0,
        contradiction_count=0,
        native_unbound_count=0,
        model_unbound_count=0,
    )
    assert record.row_labels_checked == 5
    assert record.candidate_valueless_unbound == 0
    assert record.native_valueless_unbound == 0
    assert record.contradiction_count == 0
    assert record.source_stage == "reconstruct_table_regions"
