"""Deterministic primary-grid selection for GH-330 coverage."""

from __future__ import annotations

import fitz

from socr.benchmark.binding_coverage import select_primary_grid
from socr.benchmark.binding_coverage import NativeExtractionRegion
from socr.tables.binding import parse_grid


def test_primary_grid_tie_break_is_lowest_region_ordinal():
    markdown = "| Label | Value |\n| --- | --- |\n| Row | 1.0 |"
    first = NativeExtractionRegion(fitz.Rect(0, 0, 10, 10), markdown, "first")
    second = NativeExtractionRegion(fitz.Rect(0, 20, 10, 30), markdown, "second")
    first_grid = parse_grid(markdown)
    second_grid = parse_grid(markdown)
    assert first_grid is not None and second_grid is not None
    candidates = [(2, second, second_grid), (1, first, first_grid)]

    selected = select_primary_grid(candidates)

    assert selected is not None
    assert selected[0] == 1
    assert selected[1].provenance == "first"
