"""GH-352 leftovers: one column-projection helper and scoreboard model_unbound."""

from __future__ import annotations

from socr.benchmark.binding_coverage import (
    CoverageRecord,
    CoverageReport,
    _aggregate,
    summary_text,
)
from socr.tables.binding import (
    Grid,
    _candidate_data_column_indices,
    _project_candidate_data_columns,
    bind,
    parse_grid,
)


def _w(x0, y0, x1, y1, text):
    return (x0, y0, x1, y1, text, 0, 0, 0)


def _two_lane_page():
    return [
        _w(50, 100, 90, 110, "Coef"),
        _w(150, 100, 180, 110, "1.10"),
        _w(250, 100, 280, 110, "1.11"),
        _w(50, 130, 90, 140, "SE"),
        _w(150, 130, 180, 140, "0.05"),
        _w(250, 130, 280, 140, "0.06"),
    ]


_WIDER_THAN_THE_PAGE = "\n".join(
    [
        "|      | OLS  | note | IV   | note2 |",
        "| --- | --- | --- | --- | --- |",
        "| Coef | 1.10 |      | 1.11 |       |",
        "| SE   | 0.05 |      | 0.06 |       |",
    ]
)


class TestCandidateDataColumnHelperIsSingleSourceOfTruth:
    def test_projected_grid_columns_match_index_helper(self) -> None:
        grid = parse_grid(_WIDER_THAN_THE_PAGE)
        assert grid is not None

        indices = _candidate_data_column_indices(grid)
        projected = _project_candidate_data_columns(grid)

        assert indices == (1, 3)
        assert projected.n_cols - 1 == len(indices)
        for projected_col, original_col in enumerate(indices, start=1):
            for row_idx, row in enumerate(grid.rows):
                projected_row = projected.rows[row_idx]
                if original_col < len(row):
                    assert projected_row[projected_col] == row[original_col]

    def test_widest_row_union_keeps_columns_from_the_widest_numeric_rows(self) -> None:
        """A narrower row must not pull columns into the remap that projection drops."""
        markdown = "\n".join(
            [
                "|      | A | B | C |",
                "| --- | --- | --- | --- |",
                "| wide | 1.0 | 2.0 | 3.0 |",
                "| narrow | 4.0 |     |     |",
            ]
        )
        grid = parse_grid(markdown)
        assert grid is not None

        assert _candidate_data_column_indices(grid) == (1, 2, 3)
        assert _project_candidate_data_columns(grid).n_cols == 4

    def test_bind_remap_uses_the_same_indices_as_projection(self) -> None:
        """Regression: remap drift would surface invented model_unbound on self-bind."""
        result = bind(_two_lane_page(), _WIDER_THAN_THE_PAGE)

        assert result.column_binding_unverifiable is True
        assert {(cell.row_path[-1], cell.token) for cell in result.model_unbound} == set()


class TestModelUnboundNonemptyOnScoreboard:
    def test_coverage_record_carries_model_unbound_nonempty(self) -> None:
        record = CoverageRecord(
            paper="test_paper",
            page=1,
            source_stage="find_tables_lines",
            region_ordinal=1,
            selected_primary=True,
            grid_rows=2,
            grid_columns=3,
            fully_checked=False,
            structural_agreement=False,
            row_binding_unverifiable=True,
            column_binding_unverifiable=True,
            row_label_unverifiable=False,
            row_labels_checked=2,
            ambiguous_count=0,
            candidate_valueless_unbound=0,
            native_valueless_unbound=0,
            contradiction_count=2,
            native_unbound_count=0,
            model_unbound_count=2,
            model_unbound_nonempty=True,
        )

        assert record.model_unbound_nonempty is True
        assert record.model_unbound_count == 2

    def test_aggregate_counts_model_unbound_nonempty_from_record_flag(self) -> None:
        empty = CoverageRecord(
            paper="a",
            page=1,
            source_stage="find_tables_lines",
            region_ordinal=1,
            selected_primary=True,
            grid_rows=1,
            grid_columns=2,
            fully_checked=False,
            structural_agreement=False,
            row_binding_unverifiable=True,
            column_binding_unverifiable=True,
            row_label_unverifiable=False,
            row_labels_checked=0,
            ambiguous_count=0,
            candidate_valueless_unbound=0,
            native_valueless_unbound=0,
            contradiction_count=0,
            native_unbound_count=0,
            model_unbound_count=0,
            model_unbound_nonempty=False,
        )
        nonempty = CoverageRecord(
            paper="b",
            page=1,
            source_stage="find_tables_lines",
            region_ordinal=1,
            selected_primary=True,
            grid_rows=1,
            grid_columns=2,
            fully_checked=False,
            structural_agreement=False,
            row_binding_unverifiable=True,
            column_binding_unverifiable=True,
            row_label_unverifiable=False,
            row_labels_checked=0,
            ambiguous_count=0,
            candidate_valueless_unbound=0,
            native_valueless_unbound=0,
            contradiction_count=0,
            native_unbound_count=0,
            model_unbound_count=3,
            model_unbound_nonempty=True,
        )

        totals = _aggregate((empty, nonempty), denominator=2)

        assert totals["model_unbound_nonempty"] == 1

    def test_whole_page_summary_includes_model_unbound_nonempty(self) -> None:
        report = CoverageReport(
            summary={
                "total_pages": 15,
                "bindable_pages": 13,
                "strict_grids": 13,
                "placeholder_regions": 0,
                "no_grid_pages": 2,
                "bindable_pages_by_stage": {"find_tables_lines": 13},
                "region_scoped": {
                    "denominator": 13,
                    "fully_checked": 0,
                    "structural_agreement": 0,
                    "row_binding_unverifiable": 13,
                    "column_binding_unverifiable": 1,
                    "row_label_unverifiable": 0,
                    "row_labels_checked_positive": 0,
                    "ambiguity_nonempty": 0,
                    "candidate_valueless_unbound": 0,
                    "native_valueless_unbound": 0,
                    "cell_contradiction_nonempty": 0,
                    "row_label_contradiction_nonempty": 0,
                    "contradiction_nonempty": 0,
                    "native_unbound_nonempty": 0,
                    "model_unbound_nonempty": 4,
                },
                "whole_page": {
                    "denominator": 15,
                    "fully_checked": 0,
                    "structural_agreement": 0,
                    "row_binding_unverifiable": 15,
                    "column_binding_unverifiable": 15,
                    "row_label_unverifiable": 0,
                    "row_labels_checked_positive": 0,
                    "ambiguity_nonempty": 0,
                    "candidate_valueless_unbound": 0,
                    "native_valueless_unbound": 0,
                    "cell_contradiction_nonempty": 0,
                    "row_label_contradiction_nonempty": 0,
                    "contradiction_nonempty": 0,
                    "native_unbound_nonempty": 0,
                    "model_unbound_nonempty": 6,
                },
            },
            regions=(),
            pages=(),
        )

        text = summary_text(report)

        assert "model unbound non-empty: 6/15" in text
        assert "model unbound non-empty: 4" in text
