"""P6 stage C: exact-difference oracle and corpus behavior tests.

Design: `docs/log/2026-09-02_p6-selector-collapse-design.md` and
`docs/log/2026-09-02_p6-stage-ab-disposition-contract.md`.

Stage C migrates three assemble buckets (structure_class_model_pages,
structure_class_floor_pages, corrupt_math_hybrid_pages) from private
SelectionProvenance membership to exact public PageDisposition equality.
When the post-selection emission guard rewrites a candidate to a fail-closed marker:
- It leaves the migrated bucket (final disposition != migrated pair).
- It no longer emits the stale shipped audit kinds (structure_class_model_table_kept,
  corrupt_math_hybrid_shipped).
- It drops from the respective CLI bucket line and audit log count.
- It updates the corrupt-math error note.
- Derivatives: removing structure_class_model_table_kept on page 9 updates
  tables_trust.json (flags count, page 9 reasons/details, kind counts), the
  table-trust CLI summary line, and the table-trust error note in result_error
  and metadata.json.

All unaffected surfaces (document status, result status, result audit_passed,
all winning output bytes, markdown fragments, final .md, manifest entries,
failed pages, flag-derived buckets, and orthogonal buckets) remain byte-identical
or match the exact enumerated oracle patch.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

fitz = pytest.importorskip("fitz")

from conftest import (  # noqa: E402
    FLAG_DERIVED_BUCKET_NAMES,
    old_disposition_buckets,
    old_orthogonal_assemble_buckets,
)
from p6_corpus_fixture import (  # noqa: E402
    HYBRID_CLEAN_PAGE,
    HYBRID_REWRITTEN_PAGE,
    STRUCT_FLOOR_PAGE,
    STRUCT_MODEL_PASSING_PAGE,
    STRUCT_MODEL_REWRITTEN_PAGE,
    build_corpus_state,
    capture,
    make_pdf,
)
from p6_stage_c_oracle import (  # noqa: E402
    DISPOSITION_SURFACES,
    EXPECTED_PAGE_DISPOSITIONS,
    EXPECTED_STAGE_C_DIFFERENCE_ENTRIES,
    EXPECTED_STAGE_C_DIFFERENCES,
    GUARD_REWRITTEN_PAGES,
    VOLATILE_KEYS,
    apply_stage_c_patch,
    assert_capture_diff_matches_oracle,
    collect_disposition_leaves,
    compute_leaf_diff,
    normalize_capture,
    page_dispositions_by_surface,
)

from socr.core.manifest import (  # noqa: E402
    PageEnding,
    PagePrimaryReason,
    is_page_failed_marker,
)
from socr.pipeline.orchestrator import _derive_orthogonal_assemble_buckets  # noqa: E402

PRECHANGE_PATH = Path(__file__).parent / "fixtures" / "p6" / "prechange_assemble.json"


@pytest.fixture(scope="class")
def prechange() -> dict[str, Any]:
    return json.loads(PRECHANGE_PATH.read_text())


@pytest.fixture
def current(tmp_path: Path) -> dict[str, Any]:
    return capture(tmp_path)


@pytest.fixture(scope="class")
def patched_prechange(prechange: dict[str, Any]) -> dict[str, Any]:
    return apply_stage_c_patch(normalize_capture(prechange))


class TestStageCCompleteExactDelta:
    """The complete exact delta between stage-A/B capture and current capture matches oracle."""

    def test_complete_exact_delta_matches_shared_oracle(
        self, prechange: dict[str, Any], current: dict[str, Any]
    ) -> None:
        """Applying the expected leaf differences leaves zero residual diff."""
        assert_capture_diff_matches_oracle(prechange, current)

    def test_granular_differences_match_enumerated_surfaces(
        self, prechange: dict[str, Any], current: dict[str, Any]
    ) -> None:
        """Every individual enumerated difference entry is verified at its leaf path."""
        norm_old = normalize_capture(prechange)
        norm_new = normalize_capture(current)

        for entry in EXPECTED_STAGE_C_DIFFERENCE_ENTRIES:
            curr_old = norm_old
            for step in entry.path:
                curr_old = curr_old[step]

            curr_new = norm_new
            for step in entry.path:
                curr_new = curr_new[step]

            assert curr_old == entry.old_value, (
                f"Old value mismatch at {entry.path}: "
                f"expected {entry.old_value!r}, got {curr_old!r}"
            )
            assert curr_new == entry.new_value, (
                f"New value mismatch at {entry.path}: "
                f"expected {entry.new_value!r}, got {curr_new!r}"
            )

    def test_no_entire_surface_is_permitted_to_differ(
        self, prechange: dict[str, Any], current: dict[str, Any]
    ) -> None:
        """Surfaces differ ONLY at enumerated leaf paths, never wholesale."""
        norm_old = normalize_capture(prechange)
        norm_new = normalize_capture(current)
        measured_diff = compute_leaf_diff(norm_old, norm_new)

        assert set(measured_diff.keys()) == set(EXPECTED_STAGE_C_DIFFERENCES.keys())


class TestRewrittenNamedCases:
    """The two guard-rewritten named cases: page 9 (structure-class) and page 12 (corrupt-math)."""

    def test_rewritten_cases_former_migrated_bucket_is_absent(
        self, current: dict[str, Any]
    ) -> None:
        """Only the three migrated buckets are checked; rewritten pages are absent."""
        buckets = current["buckets"]
        assert STRUCT_MODEL_REWRITTEN_PAGE not in buckets["structure_class_model_pages"]
        assert STRUCT_MODEL_REWRITTEN_PAGE not in buckets["structure_class_floor_pages"]
        assert STRUCT_MODEL_REWRITTEN_PAGE not in buckets["corrupt_math_hybrid_pages"]

        assert HYBRID_REWRITTEN_PAGE not in buckets["structure_class_model_pages"]
        assert HYBRID_REWRITTEN_PAGE not in buckets["structure_class_floor_pages"]
        assert HYBRID_REWRITTEN_PAGE not in buckets["corrupt_math_hybrid_pages"]

    def test_stale_shipped_kind_absent_from_state_events(self, current: dict[str, Any]) -> None:
        """State events do not emit the stale shipped audit kind for rewritten pages."""
        events = current["events"]
        for p_num, kind, eng, detail in events:
            if p_num == STRUCT_MODEL_REWRITTEN_PAGE:
                assert kind != "structure_class_model_table_kept"
            if p_num == HYBRID_REWRITTEN_PAGE:
                assert kind != "corrupt_math_hybrid_shipped"

    def test_stale_shipped_kind_absent_from_audit_log(self, current: dict[str, Any]) -> None:
        """audit_log.json events omit stale shipped kinds for rewritten pages."""
        log_events = current["audit_log"][0]["events"]
        for ev in log_events:
            p_num = ev.get("page_num")
            kind = ev.get("kind")
            if p_num == STRUCT_MODEL_REWRITTEN_PAGE:
                assert kind != "structure_class_model_table_kept"
            if p_num == HYBRID_REWRITTEN_PAGE:
                assert kind != "corrupt_math_hybrid_shipped"

    def test_stale_shipped_kind_absent_from_page_sidecars(self, current: dict[str, Any]) -> None:
        """Page sidecars omit stale shipped kinds for rewritten pages."""
        p9_sidecar = current["sidecars"][f"p6_corpus/pages/{STRUCT_MODEL_REWRITTEN_PAGE:05d}.json"]
        p9_kinds = [ev["kind"] for ev in p9_sidecar["audit_events"]]
        assert "structure_class_model_table_kept" not in p9_kinds
        assert "page_failed" in p9_kinds
        assert "table_structure_failed" in p9_kinds

        p12_sidecar = current["sidecars"][f"p6_corpus/pages/{HYBRID_REWRITTEN_PAGE:05d}.json"]
        p12_kinds = [ev["kind"] for ev in p12_sidecar["audit_events"]]
        assert "corrupt_math_hybrid_shipped" not in p12_kinds
        assert "page_failed" in p12_kinds
        assert "table_structure_failed" in p12_kinds

    def test_rewritten_cases_failure_disposition_and_markers(self, current: dict[str, Any]) -> None:
        """Failure disposition, marker text, and winning_output remain fail-closed."""
        p9_rec = next(
            r for r in current["page_contract"] if r["page_num"] == STRUCT_MODEL_REWRITTEN_PAGE
        )
        assert p9_rec["disposition"] == [
            PageEnding.FAIL_CLOSED_MARKER.value,
            PagePrimaryReason.INVALID_TABLE_EMISSION.value,
        ]
        assert p9_rec["is_failure_marker"] is True
        assert p9_rec["winning_output"]["status"] == "error"
        assert p9_rec["winning_output"]["audit_passed"] is False
        assert p9_rec["winning_output"]["failure_mode"] == "table_emission_invalid"
        assert is_page_failed_marker(p9_rec["winning_output"]["text"])

        p12_rec = next(
            r for r in current["page_contract"] if r["page_num"] == HYBRID_REWRITTEN_PAGE
        )
        assert p12_rec["disposition"] == [
            PageEnding.FAIL_CLOSED_MARKER.value,
            PagePrimaryReason.INVALID_TABLE_EMISSION.value,
        ]
        assert p12_rec["is_failure_marker"] is True
        assert p12_rec["winning_output"]["status"] == "error"
        assert p12_rec["winning_output"]["audit_passed"] is False
        assert p12_rec["winning_output"]["failure_mode"] == "table_emission_invalid"
        assert is_page_failed_marker(p12_rec["winning_output"]["text"])

    def test_rewritten_cases_page_failed_event_and_cli_entry(self, current: dict[str, Any]) -> None:
        """page_failed event and CLI failed-page entry remain intact."""
        events = current["events"]
        failed_event_pages = [p_num for p_num, kind, eng, d in events if kind == "page_failed"]
        assert STRUCT_MODEL_REWRITTEN_PAGE in failed_event_pages
        assert HYBRID_REWRITTEN_PAGE in failed_event_pages

        cli = current["cli"]
        assert "5 page(s) produced no usable output: [2, 6, 9, 10, 12]" in cli

    def test_rewritten_cases_final_markdown_and_manifest_blobs(
        self, current: dict[str, Any], prechange: dict[str, Any]
    ) -> None:
        """Markdown fragments, final markdown, and manifest blobs remain unchanged."""
        p9_frag_key = f"p6_corpus/pages/{STRUCT_MODEL_REWRITTEN_PAGE:05d}.md"
        p12_frag_key = f"p6_corpus/pages/{HYBRID_REWRITTEN_PAGE:05d}.md"

        assert current["markdown"][p9_frag_key] == prechange["markdown"][p9_frag_key]
        assert current["markdown"][p12_frag_key] == prechange["markdown"][p12_frag_key]
        assert (
            current["markdown"]["p6_corpus/p6_corpus.md"]
            == prechange["markdown"]["p6_corpus/p6_corpus.md"]
        )

        manifest_entries = current["manifest"][0]["entries"]
        p9_manifest = manifest_entries[str(STRUCT_MODEL_REWRITTEN_PAGE)]
        assert p9_manifest["disposition"] == {
            "ending": "fail_closed_marker",
            "primary_reason": "invalid_table_emission",
        }
        assert (
            p9_manifest["blob_ref"]
            == prechange["manifest"][0]["entries"][str(STRUCT_MODEL_REWRITTEN_PAGE)]["blob_ref"]
        )

        p12_manifest = manifest_entries[str(HYBRID_REWRITTEN_PAGE)]
        assert p12_manifest["disposition"] == {
            "ending": "fail_closed_marker",
            "primary_reason": "invalid_table_emission",
        }
        assert (
            p12_manifest["blob_ref"]
            == prechange["manifest"][0]["entries"][str(HYBRID_REWRITTEN_PAGE)]["blob_ref"]
        )


class TestCleanAndFloorControls:
    """Passing structure-class (8), genuine structure-floor (10), and clean hybrid (11) controls."""

    def test_controls_bucket_membership(
        self, current: dict[str, Any], prechange: dict[str, Any]
    ) -> None:
        """Clean and floor controls retain their bucket memberships."""
        buckets = current["buckets"]
        assert STRUCT_MODEL_PASSING_PAGE in buckets["structure_class_model_pages"]
        assert STRUCT_FLOOR_PAGE in buckets["structure_class_floor_pages"]
        assert HYBRID_CLEAN_PAGE in buckets["corrupt_math_hybrid_pages"]

        assert STRUCT_MODEL_PASSING_PAGE in prechange["buckets"]["structure_class_model_pages"]
        assert STRUCT_FLOOR_PAGE in prechange["buckets"]["structure_class_floor_pages"]
        assert HYBRID_CLEAN_PAGE in prechange["buckets"]["corrupt_math_hybrid_pages"]

    def test_controls_bucket_audit_kind_and_events(
        self, current: dict[str, Any], prechange: dict[str, Any]
    ) -> None:
        """Audit kinds for controls remain in events, sidecars, and audit_log."""
        events = current["events"]
        assert any(
            p == STRUCT_MODEL_PASSING_PAGE and k == "structure_class_model_table_kept"
            for p, k, eng, d in events
        )
        assert any(
            p == STRUCT_FLOOR_PAGE and k == "structure_class_ladder_exhausted_floor"
            for p, k, eng, d in events
        )
        assert any(
            p == HYBRID_CLEAN_PAGE and k == "corrupt_math_hybrid_shipped" for p, k, eng, d in events
        )

        p8_sidecar = current["sidecars"][f"p6_corpus/pages/{STRUCT_MODEL_PASSING_PAGE:05d}.json"]
        assert any(
            ev["kind"] == "structure_class_model_table_kept" for ev in p8_sidecar["audit_events"]
        )

        p10_sidecar = current["sidecars"][f"p6_corpus/pages/{STRUCT_FLOOR_PAGE:05d}.json"]
        assert any(
            ev["kind"] == "structure_class_ladder_exhausted_floor"
            for ev in p10_sidecar["audit_events"]
        )

        p11_sidecar = current["sidecars"][f"p6_corpus/pages/{HYBRID_CLEAN_PAGE:05d}.json"]
        assert any(
            ev["kind"] == "corrupt_math_hybrid_shipped" for ev in p11_sidecar["audit_events"]
        )

    def test_controls_cli_lines(self, current: dict[str, Any]) -> None:
        """CLI output contains expected lines for controls."""
        cli = current["cli"]
        assert (
            "1 structure-class page(s) shipped the model's grid reading over native "
            "(native may not author a grid): [8]" in cli
        )
        assert (
            "1 structure-class page(s) hit the fail-closed floor "
            "(usable grid candidates refused/absent; marker plus page image selected; "
            "native geometry grid withheld): [10]" in cli
        )
        assert (
            "1 page(s) shipped crop-backed equation candidate(s); "
            "mathematical fidelity remains unverified: [11]" in cli
        )

    def test_controls_sidecar_audit_log_trust_manifest_and_markdown(
        self, current: dict[str, Any], prechange: dict[str, Any]
    ) -> None:
        """All control surfaces equal prechange after normalization."""
        for p in (STRUCT_MODEL_PASSING_PAGE, STRUCT_FLOOR_PAGE, HYBRID_CLEAN_PAGE):
            sidecar_key = f"p6_corpus/pages/{p:05d}.json"
            frag_key = f"p6_corpus/pages/{p:05d}.md"

            assert normalize_capture(current["sidecars"][sidecar_key]) == normalize_capture(
                prechange["sidecars"][sidecar_key]
            )
            assert current["markdown"][frag_key] == prechange["markdown"][frag_key]

            manifest_p_curr = current["manifest"][0]["entries"][str(p)]
            manifest_p_old = prechange["manifest"][0]["entries"][str(p)]
            assert normalize_capture(manifest_p_curr) == normalize_capture(manifest_p_old)

        # Tables trust for pages 8 and 10
        trust_curr = current["tables_trust"][0]["pages"]
        trust_old = prechange["tables_trust"][0]["pages"]
        assert (
            trust_curr[str(STRUCT_MODEL_PASSING_PAGE)] == trust_old[str(STRUCT_MODEL_PASSING_PAGE)]
        )
        assert trust_curr[str(STRUCT_FLOOR_PAGE)] == trust_old[str(STRUCT_FLOOR_PAGE)]


class TestUnaffectedSurfaces:
    """Document status, result status, winning outputs, and orthogonal groups remain identical."""

    def test_document_status_and_result_status(
        self, current: dict[str, Any], prechange: dict[str, Any]
    ) -> None:
        """Top-level document and result status are unchanged."""
        assert current["doc_status"] == prechange["doc_status"]
        assert current["result_status"] == prechange["result_status"]
        assert current["result_audit_passed"] == prechange["result_audit_passed"]

    def test_all_winning_output_bytes(
        self, current: dict[str, Any], prechange: dict[str, Any]
    ) -> None:
        """All 12 pages retain byte-identical winning output text and attributes."""
        for p_curr, p_old in zip(current["page_contract"], prechange["page_contract"]):
            assert p_curr["page_num"] == p_old["page_num"]
            assert p_curr["winning_output"]["text"] == p_old["winning_output"]["text"]
            assert p_curr["winning_output"]["engine"] == p_old["winning_output"]["engine"]
            assert p_curr["winning_output"]["status"] == p_old["winning_output"]["status"]
            assert (
                p_curr["winning_output"]["audit_passed"] == p_old["winning_output"]["audit_passed"]
            )
            assert (
                p_curr["winning_output"]["failure_mode"] == p_old["winning_output"]["failure_mode"]
            )

    def test_manifest_replay_data(self, current: dict[str, Any], prechange: dict[str, Any]) -> None:
        """Replay data in manifest entries is unchanged except at enumerated diffs."""
        norm_curr_entries = normalize_capture(current["manifest"][0]["entries"])
        norm_old_entries = normalize_capture(prechange["manifest"][0]["entries"])
        assert norm_curr_entries == norm_old_entries

    def test_failed_pages_and_native_fallback_pages(
        self, current: dict[str, Any], prechange: dict[str, Any]
    ) -> None:
        """Failed page list [2, 6, 9, 10, 12] is unchanged."""
        curr_failed = [r["page_num"] for r in current["page_contract"] if r["is_failure_marker"]]
        old_failed = [r["page_num"] for r in prechange["page_contract"] if r["is_failure_marker"]]
        assert curr_failed == [2, 6, 9, 10, 12]
        assert curr_failed == old_failed

    def test_flag_derived_buckets_unchanged(
        self, current: dict[str, Any], prechange: dict[str, Any]
    ) -> None:
        """The three flag-derived buckets match prechange capture exactly."""
        for name in FLAG_DERIVED_BUCKET_NAMES:
            assert current["buckets"][name] == prechange["buckets"][name]

    def test_every_orthogonal_group_unchanged(self, tmp_path: Path) -> None:
        """Every orthogonal assemble bucket matches pre-extraction predicate."""
        state = build_corpus_state(make_pdf(tmp_path))
        orth_new = _derive_orthogonal_assemble_buckets(state)
        orth_old = old_orthogonal_assemble_buckets(state)
        assert orth_new == orth_old


class TestTheDispositionSurfaceIsCompared:
    """Cold review round 1, finding 1: the public disposition is captured and pinned.

    The oracle used to strip ``disposition`` as if it were volatile, and the
    baseline capture was written stripped, so the one field this stage is about
    was never compared. It is compared now. The expected disposition delta is
    EMPTY -- stage C changes which bucket reads a disposition, not the
    disposition itself -- and these tests pin that emptiness against the actual
    values rather than against an absence of data.
    """

    def test_disposition_is_not_treated_as_volatile(self) -> None:
        """The normalizer must not excise the field under test."""
        assert "disposition" not in VOLATILE_KEYS

    def test_the_baseline_capture_actually_carries_dispositions(
        self, prechange: dict[str, Any]
    ) -> None:
        """A re-stripped baseline would make every assertion below vacuous."""
        leaves = collect_disposition_leaves(prechange)
        assert leaves, "the stage-A/B baseline capture carries no disposition at all"
        surfaces = page_dispositions_by_surface(prechange)
        for surface in DISPOSITION_SURFACES:
            assert set(surfaces[surface]) == set(EXPECTED_PAGE_DISPOSITIONS), surface

    def test_baseline_and_current_dispositions_match_the_pinned_table(
        self, prechange: dict[str, Any], current: dict[str, Any]
    ) -> None:
        """All 12 pages, all three persisted surfaces, both sides of stage C."""
        for capture, side in ((prechange, "stage-A/B"), (current, "stage-C")):
            surfaces = page_dispositions_by_surface(capture)
            for surface in DISPOSITION_SURFACES:
                assert surfaces[surface] == EXPECTED_PAGE_DISPOSITIONS, (
                    f"{side} {surface} dispositions differ from the pinned table"
                )

    def test_guard_rewritten_pages_kept_their_disposition_across_the_stage(
        self, prechange: dict[str, Any], current: dict[str, Any]
    ) -> None:
        """Pages 9 and 12 were already fail-closed BEFORE stage C.

        That is why the migration drops them from the three buckets: the public
        disposition already named an invalid table emission. Stage C changes
        their bucket membership and nothing about their disposition.
        """
        old_surfaces = page_dispositions_by_surface(prechange)
        new_surfaces = page_dispositions_by_surface(current)
        for page in GUARD_REWRITTEN_PAGES:
            for surface in DISPOSITION_SURFACES:
                assert old_surfaces[surface][page] == (
                    "fail_closed_marker",
                    "invalid_table_emission",
                ), (surface, page)
                assert new_surfaces[surface][page] == old_surfaces[surface][page], (surface, page)

        old_buckets = prechange["buckets"]
        new_buckets = current["buckets"]
        assert STRUCT_MODEL_REWRITTEN_PAGE in old_buckets["structure_class_model_pages"]
        assert STRUCT_MODEL_REWRITTEN_PAGE not in new_buckets["structure_class_model_pages"]
        assert HYBRID_REWRITTEN_PAGE in old_buckets["corrupt_math_hybrid_pages"]
        assert HYBRID_REWRITTEN_PAGE not in new_buckets["corrupt_math_hybrid_pages"]

    def test_no_disposition_leaf_appears_in_the_expected_difference_set(self) -> None:
        """The enumerated oracle claims zero disposition differences."""
        offenders = [path for path in EXPECTED_STAGE_C_DIFFERENCES if "disposition" in path]
        assert offenders == [], offenders

    def test_every_disposition_leaf_is_identical_across_the_stage(
        self, prechange: dict[str, Any], current: dict[str, Any]
    ) -> None:
        """Leaf-by-leaf, at full path, with no reliance on the diff walker."""
        old_leaves = collect_disposition_leaves(normalize_capture(prechange))
        new_leaves = collect_disposition_leaves(normalize_capture(current))
        assert set(old_leaves) == set(new_leaves)
        differing = {
            path: (old_leaves[path], new_leaves[path])
            for path in old_leaves
            if old_leaves[path] != new_leaves[path]
        }
        assert differing == {}, differing


class TestNegativeControl:
    """Negative control: replacing migrated membership with stage-A/B rule fails oracle."""

    def test_negative_control_fails_on_stage_ab_provenance_rule(
        self, prechange: dict[str, Any], current: dict[str, Any], tmp_path: Path
    ) -> None:
        """A capture with stage-A/B provenance rule raises AssertionError at migrated bucket."""
        perturbed_capture = copy.deepcopy(current)
        state = build_corpus_state(make_pdf(tmp_path))
        stage_ab_buckets = old_disposition_buckets(state)
        perturbed_capture["buckets"] = {k: sorted(list(v)) for k, v in stage_ab_buckets.items()}

        with pytest.raises(AssertionError) as exc_info:
            assert_capture_diff_matches_oracle(prechange, perturbed_capture)

        err_msg = str(exc_info.value)
        assert (
            "buckets.structure_class_model_pages" in err_msg
            or "buckets.corrupt_math_hybrid_pages" in err_msg
        ), f"Negative control did not fail at a named migrated-bucket path:\n{err_msg}"
