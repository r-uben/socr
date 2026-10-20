"""Pin the exact set of audit-event kinds a resumed page replays.

``resume_restore_kinds`` is assembled from the table-ladder terminals, the equation lane
and ``_RESUME_REPLAYED``. The expected set below is hard-coded from ``main`` at 3cf5f87
(before the notes moved into ``_RESUME_REPLAYED``), so dropping or adding a kind in a
refactor fails here instead of silently changing what a resumed run reports.

Every later addition carries its reason at the line, so the list stays a ledger of decisions
rather than a snapshot that is updated whenever it goes red.
"""

from socr.pipeline.orchestrator import UnifiedPipeline

EXPECTED_RESUME_KINDS = frozenset(
    {
        # #1205 round 2 (Argon P1-2 of the round-2 review): emitted at the crop
        # reread, which a resumed page skips, and in both TABLE_DISTRUST_KINDS and
        # CREDENTIAL_BLOCKING_EVENT_KINDS -- so without replay the page comes back
        # absent from tables_trust.json with its credential restored. #1210 made that
        # reachable by admitting accepted prior-OCR pages to the ledger.
        "table_crop_repair_unverifiable",
        # #1187: emitted by the page loop's unshipped-figure step, which a terminal
        # resumed page skips. The page's own picture or mark is restored from the sidecar
        # fields, so losing the event would lose only the REPORT -- the document note and
        # the CLI line read these, which is the #252 shape this allowlist exists for.
        "unshipped_figure_rendered",
        "unshipped_figure_render_failed",
        "unshipped_figure_unlocated",
        "unshipped_figure_not_found",
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
        # #1063 added this kind: a detected crop the clean-equation reader was refused on.
        "equation_sidecar_skipped_model_refused",
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
        "table_furniture_removed",
        # #1144 added this kind. The reason first given here -- "a credentialed page
        # can now SHIP carrying it" -- was WRONG: that credential path was reverted and
        # table_escalation_refused is still in CREDENTIAL_BLOCKING_EVENT_KINDS. It is
        # replayed for the same reason as its GH-851 sibling below: _escalate_table_page
        # is skipped by a terminal resumed page while tables_trust still reads the kind,
        # so a resumed run that dropped it would report the page as more trusted than
        # the first run did.
        "table_escalation_refused",
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
        # #1181: the deterministic EXACT_PASS accept. An ADDITION, declared here rather
        # than left to drift. Emitted by the audit gate inside the page loop, which a
        # terminal resumed page skips, and the page_judge=UNKNOWN clause keys on it -- so a
        # resumed run that dropped it would un-mark a page the first run marked.
        "native_table_verifier_exact_pass",
        # default-on native per-cell binding: emitted by _run_native_cell_binding_check,
        # which a terminal resumed page skips (same #252 / GH-353 replay contract).
        "native_cell_check_ambiguous",
        "native_cell_check_contradicted",
        "native_cell_check_skipped",
        "native_cell_check_verified",
        "blank_page",
        # PR #1160: counted by the run summary, emitted by steps a resumed page skips.
        "provider_empty_response",
        "page_retry",
        # #1145 slice 3: the LEGACY equation seam's two records
        # (``_attach_equation_latex_sidecars``, under --detect-equations
        # --recover-clean-equations). An ADDITION, deliberately, and this pin is doing its
        # job by making it explicit rather than letting the set drift.
        #
        # They were absent, and by ``EQUATION_LANE_EVENT_KINDS``'s own docstring a missing
        # kind "disappears from audit_log.json the moment the page resumes" -- already the
        # #252 shape for that seam and independent of #1145. The accept record is now
        # load-bearing for #1145 too, because the unverified-equation clause keys on it and
        # a clause whose evidence is dropped on resume silently un-demotes the page. The
        # refusal record follows the GH-522 argument already made for the region lane's:
        # refusing a reading is only acceptable because the crop survives for a human to
        # check, and a record dropped on resume holds that guarantee only until the next run.
        #
        # Neither was visible to ``tests/test_gh522_lane_events_survive_resume.py``, whose
        # scanner read only ``kind="equation_..."`` keyword arguments while that seam assigns
        # its kind to a local first. The scanner now sees both shapes, which is what
        # surfaced these.
        "equation_latex_accepted",
        "equation_latex_rejected_kept_crop",
    }
)


def test_resume_restore_kinds_is_exactly_the_pre_refactor_union() -> None:
    assert UnifiedPipeline.resume_restore_kinds() == EXPECTED_RESUME_KINDS


def test_the_hard_coded_set_has_the_expected_size() -> None:
    # 48 after #1100 (blank_page); 49 with #1063's equation_sidecar_skipped_model_refused;
    # 51 with PR #1160's provider_empty_response and page_retry;
    # 52 with #1144's table_escalation_refused;
    # 54 with #1145 slice 3's two legacy-equation-seam kinds (equation_latex_accepted and
    # equation_latex_rejected_kept_crop), which were never replayed and are explained at
    # their lines above;
    # 55 with #1181's native_table_verifier_exact_pass;
    # 59 with default-on native per-cell binding's four terminal kinds;
    # 60 with #1205's table_crop_repair_unverifiable;
    # 65 with #1187's five unshipped-figure kinds;
    # 64 after #1187 round 2 dropped ``unshipped_figure_partially_shipped``: the outcome it
    # named -- a page naming more figures than it shipped crops -- stopped being a distinct
    # behaviour when every page the lane acts on became MARKED (Argon P0-1).
    assert len(EXPECTED_RESUME_KINDS) == 64
