"""P6 Stage C: how the six mandated assemble buckets are derived, and why.

Design: §2 and §8 of ``docs/log/2026-09-02_p6-selector-collapse-design.md``, as amended
by cold review rounds 1 and 2.

Three buckets are derived from the exact shipped ``PageDisposition`` pair:
``structure_class_model_pages``, ``structure_class_floor_pages``, and
``corrupt_math_hybrid_pages``. ``SelectionProvenance`` is internal selection
provenance, not the public contract: a post-selection guard may rewrite a selected
page's bytes and disposition, so bucket membership follows what actually ships.

Three are flag-derived and read ``PageState`` directly: ``d3_model_table_pages``,
``d3_floor_pages``, ``flagged_model_pages``. Those are native-lane verdicts a page can
carry while selection ends on a different branch, so no tag or disposition can express
them; see ``tests/test_p6_stage_ab_difference.py`` and ``tests/conftest.py``.

Hermetic: constructs ``DocumentState`` and ``FinalizedPageRecord`` fixtures directly,
no provider, no pipeline run.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    FinalizedPageRecord,
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
    finalized_page_records,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import _derive_disposition_buckets  # noqa: E402


def _pdf(tmp_path, page_count: int = 6):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    for _ in range(page_count):
        doc.new_page().insert_text(
            (54, 72), "born-digital prose long enough to count as a real text layer here."
        )
    doc.save(path)
    doc.close()
    return path


def _new_state(tmp_path, page_count: int = 6) -> DocumentState:
    return DocumentState(handle=DocumentHandle.from_path(_pdf(tmp_path, page_count=page_count)))


def _bd(p, text="native prose") -> None:
    p.is_born_digital = True
    p.native_text = text


def _rejected_attempt(page_num: int, text: str) -> PageOutput:
    return PageOutput(
        page_num=page_num,
        text=text,
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )


# ---------------------------------------------------------------------------
# Pinned six bucket contracts
# ---------------------------------------------------------------------------

DISPOSITION_BUCKET_PAIRS = {
    "structure_class_model_pages": PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.STRUCTURE_CLASS
    ),
    "structure_class_floor_pages": PageDisposition(
        PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.STRUCTURE_CLASS
    ),
    "corrupt_math_hybrid_pages": PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.CORRUPT_MATH_HYBRID
    ),
}

#: The three that are NOT tag-derivable. Each is keyed on a native-lane verdict a page
#: can carry while selection ends on an entirely different branch -- measured with a
#: page holding the full D3 conjunction AND a passing non-native ``best_output``. They
#: keep exactly their pre-change flag-derived membership; see
#: ``tests/test_p6_stage_ab_difference.py`` and ``tests/conftest.py``.
FLAG_DERIVED_BUCKETS = ("d3_model_table_pages", "d3_floor_pages", "flagged_model_pages")


@pytest.mark.parametrize("bucket_name, disposition", sorted(DISPOSITION_BUCKET_PAIRS.items()))
def test_each_migrated_bucket_matches_its_exact_disposition_pair(
    tmp_path, bucket_name, disposition
) -> None:
    """The exact shipped pair claims the page for every provenance value."""
    state = _new_state(tmp_path, page_count=1)
    for provenance in SelectionProvenance:
        matching_rec = FinalizedPageRecord(
            output=PageOutput(page_num=1, text="text"),
            disposition=disposition,
            selection_provenance=provenance,
        )

        buckets = _derive_disposition_buckets(state, [matching_rec])
        assert buckets[bucket_name] == {1}, (bucket_name, provenance)
        for other_name, pages in buckets.items():
            if other_name not in DISPOSITION_BUCKET_PAIRS:
                continue
            if other_name != bucket_name:
                assert 1 not in pages, (bucket_name, provenance, other_name)


def test_a_former_provenance_with_a_different_disposition_claims_no_migrated_bucket(
    tmp_path,
) -> None:
    """A former selection tag cannot override the final shipped disposition."""
    state = _new_state(tmp_path, page_count=1)
    rewritten = PageDisposition(
        ending=PageEnding.FAIL_CLOSED_MARKER,
        primary_reason=PagePrimaryReason.INVALID_TABLE_EMISSION,
    )
    former_provenance = {
        "structure_class_model_pages": (
            SelectionProvenance.STRUCTURE_CLASS_GRID_PASSING,
            SelectionProvenance.STRUCTURE_CLASS_GRID_FLAGGED,
        ),
        "structure_class_floor_pages": (SelectionProvenance.STRUCTURE_CLASS_FLOOR,),
        "corrupt_math_hybrid_pages": (SelectionProvenance.CORRUPT_MATH_HYBRID,),
    }

    for bucket_name, provenances in former_provenance.items():
        for provenance in provenances:
            rec = FinalizedPageRecord(
                output=PageOutput(page_num=1, text="text"),
                disposition=rewritten,
                selection_provenance=provenance,
            )
            buckets = _derive_disposition_buckets(state, [rec])
            for migrated_name in DISPOSITION_BUCKET_PAIRS:
                assert 1 not in buckets[migrated_name], (
                    bucket_name,
                    provenance,
                    migrated_name,
                )


def test_a_guard_rewritten_hybrid_is_absent_from_migrated_buckets_but_keeps_flag_membership(
    tmp_path,
) -> None:
    """A guard-rewritten hybrid loses stale shipped-bucket membership only.

    Its independent native-lane flag remains eligible for the flag-derived bucket;
    those PageState facts are orthogonal to the shipped disposition.
    """
    state = _new_state(tmp_path, page_count=1)
    page = state.pages[1]
    _bd(page)
    page.native_table_structure_defective = True
    flagged = PageOutput(
        page_num=1,
        text="| x | y |\n|---|---|\n| 3 | 4 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
    )
    setattr(flagged, "rejection_class", "ambiguous_deferred")
    page.attempts.append(flagged)
    page.best_output = flagged
    rec = FinalizedPageRecord(
        output=PageOutput(page_num=1, text="text"),
        disposition=PageDisposition(
            ending=PageEnding.FAIL_CLOSED_MARKER,
            primary_reason=PagePrimaryReason.INVALID_TABLE_EMISSION,
        ),
        selection_provenance=SelectionProvenance.CORRUPT_MATH_HYBRID,
    )
    buckets = _derive_disposition_buckets(state, [rec])
    for name in DISPOSITION_BUCKET_PAIRS:
        assert 1 not in buckets[name], name
    assert 1 in buckets["flagged_model_pages"]


@pytest.mark.parametrize("bucket_name", FLAG_DERIVED_BUCKETS)
def test_the_flag_derived_buckets_ignore_the_record_entirely(tmp_path, bucket_name) -> None:
    """A record tagged for one of these buckets, on a page with no such flag, claims nothing.

    The inverse of the test above, and the reason the two families are separated:
    these buckets read ``PageState``, so neither a hand-built disposition nor a
    hand-built tag can put a page into them or take one out.
    """
    state = _new_state(tmp_path, page_count=1)
    for tag in (
        SelectionProvenance.UNVERIFIABLE_TABLE_MODEL_KEPT,
        SelectionProvenance.UNVERIFIABLE_TABLE_NATIVE,
        SelectionProvenance.FLAGGED_MODEL_KEPT,
    ):
        rec = FinalizedPageRecord(
            output=PageOutput(page_num=1, text="text"),
            disposition=PageDisposition(
                ending=PageEnding.MODEL_OUTPUT,
                primary_reason=PagePrimaryReason.NATIVE_TABLE_UNVERIFIABLE,
            ),
            selection_provenance=tag,
        )
        assert _derive_disposition_buckets(state, [rec])[bucket_name] == set(), tag


# ---------------------------------------------------------------------------
# Synthetic Precedence Matrix covering all 16 provenance rows + both structure statuses
# ---------------------------------------------------------------------------


def test_synthetic_precedence_matrix_across_all_provenance_branches(tmp_path) -> None:
    """Cover every selector provenance row, both structure-grid statuses, and
    assert exact expected bucket membership."""
    state = _new_state(tmp_path, page_count=16)

    # 1. CORRUPT_MATH_HYBRID -> corrupt_math_hybrid_pages
    p1 = state.pages[1]
    _bd(p1)
    hybrid = PageOutput(
        page_num=1,
        text="native prose plus math crop",
        status=PageStatus.WARNING,
        engine="native+math",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    p1.attempts.append(hybrid)
    p1.corrupt_math_hybrid = hybrid

    # 2. PASSING_BEST_OUTPUT -> None of 6
    p2 = state.pages[2]
    _bd(p2)
    best2 = PageOutput(
        page_num=2,
        text="passing model prose",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    p2.attempts.append(best2)
    p2.best_output = best2

    # 3. UNVERIFIABLE_TABLE_SCANNED -> None of 6
    p3 = state.pages[3]
    p3.is_born_digital = False
    p3.native_text = ""
    bad3 = PageOutput(
        page_num=3,
        text="scanned table hallucination",
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.HALLUCINATION,
    )
    p3.attempts.append(bad3)
    p3.best_output = bad3

    # 4. UNVERIFIABLE_TABLE_MODEL_KEPT -> d3_model_table_pages
    p4 = state.pages[4]
    _bd(p4)
    p4.native_table_structure_failed = True
    p4.native_table_unverifiable = True
    grid4 = PageOutput(
        page_num=4,
        text="| a | b |\n|---|---|\n| 1 | 2 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
    )
    setattr(grid4, "rejection_class", "ambiguous_deferred")
    p4.attempts.append(grid4)

    # 5. UNVERIFIABLE_TABLE_NATIVE -> d3_floor_pages
    p5 = state.pages[5]
    _bd(p5)
    p5.native_table_structure_failed = True
    p5.native_table_unverifiable = True
    failed5 = _rejected_attempt(5, "no grid attempt")
    p5.attempts.append(failed5)
    p5.best_output = failed5

    # 6. ROTATED_TEXT_SHREDDED -> None of 6
    p6 = state.pages[6]
    _bd(p6)
    p6.native_rotated_text_shredded = True

    # 7. FLAGGED_MODEL_KEPT -> flagged_model_pages
    p7 = state.pages[7]
    _bd(p7)
    p7.native_table_structure_defective = True
    table7 = PageOutput(
        page_num=7,
        text="| x | y |\n|---|---|\n| 3 | 4 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
    )
    setattr(table7, "rejection_class", "ambiguous_deferred")
    p7.attempts.append(table7)
    p7.best_output = table7

    # 8. STRUCTURE_CLASS_GRID_PASSING (status=SUCCESS) -> structure_class_model_pages
    p8 = state.pages[8]
    _bd(p8, text="0.03 0.91 0.44\nn slope R2\n")
    p8.has_tables = True
    nat8 = PageOutput(
        page_num=8,
        text=p8.native_text,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    grid8 = PageOutput(
        page_num=8,
        text="| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |",
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
    )
    setattr(grid8, "rejection_class", "ambiguous_deferred")
    p8.attempts.extend([nat8, grid8])
    p8.best_output = nat8

    # 9. STRUCTURE_CLASS_GRID_FLAGGED (status=WARNING) -> structure_class_model_pages
    p9 = state.pages[9]
    _bd(p9, text="0.03 0.91 0.44\nn slope R2\n")
    p9.has_tables = True
    nat9 = PageOutput(
        page_num=9,
        text=p9.native_text,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    grid9 = PageOutput(
        page_num=9,
        text="| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |",
        status=PageStatus.WARNING,
        engine="gemini",
        audit_passed=False,
    )
    setattr(grid9, "rejection_class", "ambiguous_deferred")
    p9.attempts.extend([nat9, grid9])
    p9.best_output = nat9

    # 10. STRUCTURE_CLASS_FLOOR -> structure_class_floor_pages
    p10 = state.pages[10]
    _bd(p10, text="0.03 0.91 0.44\nn slope R2\n")
    p10.has_tables = True
    nat10 = PageOutput(
        page_num=10,
        text=p10.native_text,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    model10 = _rejected_attempt(10, "prose with no markdown table")
    p10.attempts.extend([nat10, model10])
    p10.best_output = nat10

    # 11. NATIVE_FALLBACK -> None of 6
    p11 = state.pages[11]
    _bd(p11)
    p11.needs_ocr_enhancement = True
    failed11 = _rejected_attempt(11, "failed OCR attempt")
    p11.attempts.append(failed11)
    p11.best_output = failed11

    # 12. NATIVE_CLEAN -> None of 6
    p12 = state.pages[12]
    _bd(p12)

    # 13. WHOLE_DOC_SECTION -> None of 6
    p13 = state.pages[13]
    p13.is_born_digital = False
    p13.native_text = ""

    # 14. BEST_OUTPUT_UNVERIFIED -> None of 6
    p14 = state.pages[14]
    p14.is_born_digital = False
    p14.native_text = ""
    unver14 = PageOutput(
        page_num=14,
        text="unverified text",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
    )
    p14.attempts.append(unver14)
    p14.best_output = unver14

    # 15. BEST_ATTEMPT_FLAGGED -> None of 6
    p15 = state.pages[15]
    p15.is_born_digital = False
    p15.native_text = ""
    att15 = PageOutput(
        page_num=15,
        text="flagged attempt",
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.MODEL_OUTPUT_FLAGGED,
    )
    p15.attempts.append(att15)
    p15.best_output = None

    # 16. NO_TEXT_MARKER -> None of 6
    p16 = state.pages[16]
    p16.is_born_digital = False
    p16.native_text = ""

    records = finalized_page_records(state)
    buckets = _derive_disposition_buckets(state, records)

    assert buckets["corrupt_math_hybrid_pages"] == {1}
    assert buckets["d3_model_table_pages"] == {4}
    assert buckets["d3_floor_pages"] == {5}
    assert buckets["flagged_model_pages"] == {7}
    assert buckets["structure_class_model_pages"] == {
        8,
        9,
    }  # Both passing and flagged grid statuses
    assert buckets["structure_class_floor_pages"] == {10}

    # Pages outside the six buckets
    claimed = set().union(*buckets.values())
    unclaimed = {2, 3, 6, 11, 12, 13, 14, 15, 16}
    assert unclaimed.isdisjoint(claimed)


# ---------------------------------------------------------------------------
# Precedence preemption branches
# ---------------------------------------------------------------------------


def test_higher_precedence_branches_preempt_lower_branches(tmp_path) -> None:
    """Prove that higher-precedence branches preempt lower-precedence helper predicates."""
    state = _new_state(tmp_path, page_count=4)

    # Page 1: Corrupt math hybrid preempts native fallback
    p1 = state.pages[1]
    _bd(p1)
    p1.needs_ocr_enhancement = True
    hybrid = PageOutput(
        page_num=1,
        text="native prose + math candidate",
        status=PageStatus.WARNING,
        engine="native+math",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    failed_model1 = _rejected_attempt(1, "failed model attempt")
    p1.attempts.extend([hybrid, failed_model1])
    p1.best_output = failed_model1
    p1.corrupt_math_hybrid = hybrid

    # Page 2: D3 model preempts flagged model and structure class
    p2 = state.pages[2]
    _bd(p2)
    p2.native_table_structure_failed = True
    p2.native_table_unverifiable = True
    p2.native_table_structure_defective = True
    p2.has_tables = True
    grid2 = PageOutput(
        page_num=2,
        text="| a | b |\n|---|---|\n| 1 | 2 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
    )
    setattr(grid2, "rejection_class", "ambiguous_deferred")
    p2.attempts.append(grid2)

    # Page 3: Flagged model preempts structure class and native fallback
    p3 = state.pages[3]
    _bd(p3)
    p3.native_table_structure_defective = True
    p3.has_tables = True
    p3.needs_ocr_enhancement = True
    table3 = PageOutput(
        page_num=3,
        text="| x | y |\n|---|---|\n| 3 | 4 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
    )
    setattr(table3, "rejection_class", "ambiguous_deferred")
    p3.attempts.append(table3)
    p3.best_output = table3

    # Page 4: Structure class grid winner preempts native fallback
    p4 = state.pages[4]
    _bd(p4, text="0.03 0.91 0.44\nn slope R2\n")
    p4.has_tables = True
    p4.needs_ocr_enhancement = True
    nat4 = PageOutput(
        page_num=4,
        text=p4.native_text,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    grid4 = PageOutput(
        page_num=4,
        text="| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |",
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
    )
    setattr(grid4, "rejection_class", "ambiguous_deferred")
    p4.attempts.extend([nat4, grid4])
    p4.best_output = nat4

    records = finalized_page_records(state)
    buckets = _derive_disposition_buckets(state, records)

    assert buckets["corrupt_math_hybrid_pages"] == {1}
    assert 1 not in buckets["d3_model_table_pages"]
    assert 1 not in buckets["structure_class_model_pages"]

    assert buckets["d3_model_table_pages"] == {2}
    assert 2 not in buckets["flagged_model_pages"]
    assert 2 not in buckets["structure_class_model_pages"]

    assert buckets["flagged_model_pages"] == {3}
    assert 3 not in buckets["structure_class_model_pages"]

    assert buckets["structure_class_model_pages"] == {4}


# ---------------------------------------------------------------------------
# Emission guard interactions
# ---------------------------------------------------------------------------


def test_emission_guard_rewrites_and_demotions_in_bucket_derivation(tmp_path) -> None:
    """Test that emission-marker rewrites and content-only demotions interact
    correctly with bucket derivation:
      - A marker rewrite sets (FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION) -> 0 buckets
      - A content-only demotion keeps (MODEL_OUTPUT, ACCEPTED_OUTPUT) -> 0 buckets
    """
    state = _new_state(tmp_path, page_count=2)

    # Page 1: Malformed table emission -> marker rewrite
    p1 = state.pages[1]
    _bd(p1)
    bad_table = PageOutput(
        page_num=1,
        text="| a | b |\n| --- |\n| 1 | 2 |",  # Column count mismatch
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    p1.attempts.append(bad_table)
    p1.best_output = bad_table

    # Page 2: Empty cells table emission -> content-only demotion
    p2 = state.pages[2]
    _bd(p2)
    empty_cells = PageOutput(
        page_num=2,
        text="| H1 | H2 |\n| --- | --- |\n| - | - |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    p2.attempts.append(empty_cells)
    p2.best_output = empty_cells

    records = finalized_page_records(state)
    assert records[0].disposition == PageDisposition(
        PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.INVALID_TABLE_EMISSION
    )
    assert records[1].disposition == PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.ACCEPTED_OUTPUT
    )

    buckets = _derive_disposition_buckets(state, records)
    for name, pages in buckets.items():
        assert 1 not in pages, f"Page 1 wrongly claimed by {name}"
        assert 2 not in pages, f"Page 2 wrongly claimed by {name}"


# ---------------------------------------------------------------------------
# Mutual exclusivity & control tests
# ---------------------------------------------------------------------------


def test_every_page_is_claimed_by_at_most_one_of_the_six_buckets(tmp_path) -> None:
    state = _new_state(tmp_path, page_count=16)
    records = finalized_page_records(state)
    buckets = _derive_disposition_buckets(state, records)
    seen: dict[int, str] = {}
    for name, pages in buckets.items():
        for n in pages:
            assert n not in seen, f"page {n} claimed by both {seen[n]!r} and {name!r}"
            seen[n] = name


def test_the_ordinary_clean_page_is_claimed_by_none_of_the_six(tmp_path) -> None:
    state = _new_state(tmp_path, page_count=1)
    _bd(state.pages[1])
    records = finalized_page_records(state)
    buckets = _derive_disposition_buckets(state, records)
    for name, pages in buckets.items():
        assert 1 not in pages, f"clean native page wrongly claimed by {name!r}"


# ---------------------------------------------------------------------------
# Contract tests: native_fallback_pages & orthogonal buckets
# ---------------------------------------------------------------------------


def test_native_fallback_pages_is_not_derived_from_the_demoted_native_ending() -> None:
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"
    tree = ast.parse((src / "pipeline" / "orchestrator.py").read_text())

    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "native_fallback_pages" for t in node.targets)
    ]
    buckets = [
        a
        for a in assigns
        if isinstance(a.value, ast.ListComp)
        and any(
            isinstance(node, ast.Attribute) and node.attr == "pages"
            for gen in a.value.generators
            for node in ast.walk(gen.iter)
        )
    ]
    assert len(buckets) == 1
    bucket_src = ast.unparse(buckets[0])

    assert "DEMOTED_NATIVE" not in bucket_src, (
        "native_fallback_pages must not be rewritten as an ending comparison -- "
        f"the design doc rules this out explicitly: {bucket_src}"
    )


def test_the_six_orthogonal_buckets_are_untouched_by_the_disposition_derivation(tmp_path) -> None:
    state = _new_state(tmp_path, page_count=1)
    records = finalized_page_records(state)
    buckets = _derive_disposition_buckets(state, records)
    orthogonal = {
        "native_only_distrust_pages",
        "value_drift_pages",
        "fabricated_ref_pages",
        "text_grid_rejected_pages",
        "chart_detection_failed_pages",
        "table_rejected_pages",
        "table_unverified_pages",
        "failed_pages",
        "native_fallback_pages",
    }
    assert set(buckets.keys()).isdisjoint(orthogonal), (
        f"_derive_disposition_buckets must return exactly the six mandated "
        f"buckets, not {set(buckets.keys()) & orthogonal}"
    )
    assert set(buckets.keys()) == set(DISPOSITION_BUCKET_PAIRS) | set(FLAG_DERIVED_BUCKETS)
