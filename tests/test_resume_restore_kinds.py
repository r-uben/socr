"""Pin the exact set of audit-event kinds a resumed page replays.

``resume_restore_kinds`` is assembled from the table-ladder terminals, the equation lane
and ``_RESUME_REPLAYED``. The expected set below is hard-coded from ``main`` at 3cf5f87
(before the notes moved into ``_RESUME_REPLAYED``), so dropping or adding a kind in a
refactor fails here instead of silently changing what a resumed run reports.
"""

from socr.pipeline.orchestrator import UnifiedPipeline

EXPECTED_RESUME_KINDS = frozenset(
    {
        "chart_counts_derived",
        "chart_counts_not_derived",
        "chart_grid_contradicted",
        "chart_grid_not_reconciled",
        "chart_grid_reconciled",
        "chart_table_skeleton_suppressed",
        "chart_table_skeleton_unbound",
        "equation_lane_detection_failed",
        "equation_lane_no_region",
        "equation_region_reading_attached",
        "equation_region_reading_rejected",
        "equation_region_reading_unaligned",
        "equation_region_reading_unsafe_markup",
        "equation_region_reading_unvalidated",
        "equation_region_reading_unverifiable",
        "equation_sidecar_refused",
        "equation_sidecar_skipped_no_page_output",
        "judge_wedged_circuit_open",
        "native_encoding_hygiene_suspect",
        "native_math_font_unrecovered",
        "native_math_unrecovered",
        "native_ship_gate_deferred",
        "native_unrecovered_symbol_glyphs",
        "possible_table_structure_not_reconstructed",
        "rotated_native_table_quarantined",
        "scanned_figure_asset",
        "source_evidence_no_witness_backend",
        "source_evidence_table_label_unverified",
        "table_binding_adjudicated",
        "table_binding_boundary_resolved",
        "table_binding_boundary_unresolved",
        "table_ditto_unresolved",
        "table_escalation_timeout",
        "table_escalation_withheld",
        "table_ladder_accepted",
        "table_ladder_budget_exhausted",
        "table_ladder_rejected",
        "table_ladder_unverified",
        "table_ladder_withheld",
        "table_spacer_rows_dropped",
        "table_wrapped_label_merged",
        "rejudge_accepted",
        "rejudge_error",
        "rejudge_rejected",
        "rejudge_timeout",
        "visual_values_not_transcribed",
    }
)


def test_resume_restore_kinds_is_exactly_the_pre_refactor_union() -> None:
    assert UnifiedPipeline.resume_restore_kinds() == EXPECTED_RESUME_KINDS


def test_the_hard_coded_set_has_the_expected_size() -> None:
    assert len(EXPECTED_RESUME_KINDS) == 46
