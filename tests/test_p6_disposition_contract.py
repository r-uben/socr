"""P6 Stage A: the finalization-aware page disposition contract.

Design: ``docs/log/2026-09-02_p6-selector-collapse-design.md`` §8 (panel
synthesis). Plan tasks t3-t5. This is the VOCABULARY layer: the public
``PageEnding`` / ``PagePrimaryReason`` / ``PageDisposition`` types, the
renamed-but-unchanged 16-row selector kept as private ``SelectionProvenance``,
and the one explicit total mapping between them. It does not touch persistence
(``test_p6_disposition_persistence.py``) or the post-guard finalization seam
(``test_p6_disposition_finalization.py``).

Hermetic: pure AST + in-process enum/dataclass checks, no provider, no
pipeline run -- the same posture as ``test_r7_winner_kind_tags.py``, which
this contract must not weaken (see ``test_p6_selection_provenance_tags.py``).
"""

from __future__ import annotations

import ast
import dataclasses
import inspect

import pytest

from socr.core import manifest


def test_page_ending_has_exactly_the_ruled_four_members() -> None:
    """§8 Q1: kept as a fourth ending (DEMOTED_NATIVE), not three, not five."""
    PageEnding = manifest.PageEnding
    names = {m.name for m in PageEnding}
    assert names == {"NATIVE_PROSE", "MODEL_OUTPUT", "FAIL_CLOSED_MARKER", "DEMOTED_NATIVE"}


def test_demoted_native_documents_its_exit_criterion() -> None:
    """§8 ruling: DEMOTED_NATIVE is a measured, temporary deviation from the
    three-ending ruling and must carry the exit criterion in its own docs --
    not a design note that can drift away from the code it describes.
    """
    doc = (manifest.PageEnding.__doc__ or "") + (
        inspect.getdoc(manifest.PageEnding.DEMOTED_NATIVE) or ""
    )
    for trigger in (
        "needs_ocr_enhancement",
        "chart_asset_render_failed",
        "text_grid_rejected",
    ):
        assert trigger in doc, (
            f"DEMOTED_NATIVE's docs must name {trigger!r} as one of the triggers "
            "to be independently hand-checked and assigned to N or F later"
        )


def test_page_primary_reason_is_a_normalized_cause_not_a_row_rename() -> None:
    """It must NOT have 16 (or 15) members -- that would just rename
    ``SelectionProvenance``, which is exactly the "renaming rather than
    removing" failure §8 names and rules against.
    """
    PagePrimaryReason = manifest.PagePrimaryReason
    # The invariant is SEMANTIC, not a magic count: the normalized cause
    # vocabulary must stay strictly smaller than the selector it normalizes.
    # The literal ``< 15`` this replaced was the same statement with the
    # selector's size frozen into it, so a legitimate new cause (P1's
    # TABLE_JUDGE_WITHHELD) failed a guard that was not about it. Stated
    # against SelectionProvenance itself, the guard cannot go stale and is
    # not weakened -- it still forbids one member per selector row.
    n = len(list(PagePrimaryReason))
    rows = len(list(manifest.SelectionProvenance))
    assert n < rows, (
        f"PagePrimaryReason has {n} members against {rows} selector rows -- "
        "normalization must merge causes that answer the same question, not "
        "carry one member per selector row"
    )
    names = {m.name for m in PagePrimaryReason}
    for required in (
        "CORRUPT_MATH_HYBRID",
        "NATIVE_TABLE_UNVERIFIABLE",
        "NATIVE_TABLE_DISTRUST",
        "STRUCTURE_CLASS",
        "CLEAN_NATIVE_PROSE",
        "WHOLE_DOCUMENT_SECTION",
        "UNACCEPTED_OUTPUT_KEPT",
        "NO_USABLE_OUTPUT",
        "INVALID_TABLE_EMISSION",
        # P1 (owner ruling Q2): a withheld table is a cause socr can name
        # exactly, so it must not fall back to SHIPPED_FAILURE_MARKER.
        "TABLE_JUDGE_WITHHELD",
    ):
        assert required in names, f"PagePrimaryReason is missing {required}"


def test_page_disposition_is_frozen_and_serializes_only_ending_and_reason() -> None:
    PageDisposition = manifest.PageDisposition
    assert dataclasses.is_dataclass(PageDisposition)
    assert PageDisposition.__dataclass_params__.frozen is True

    d = PageDisposition(
        ending=manifest.PageEnding.MODEL_OUTPUT,
        primary_reason=manifest.PagePrimaryReason.CLEAN_NATIVE_PROSE,
    )
    payload = d.to_dict()
    assert set(payload.keys()) == {"ending", "primary_reason"}, (
        "the serialized disposition must never leak provenance or alerts "
        f"onto the wire: {sorted(payload.keys())}"
    )


def test_page_disposition_round_trips() -> None:
    d = manifest.PageDisposition(
        ending=manifest.PageEnding.FAIL_CLOSED_MARKER,
        primary_reason=manifest.PagePrimaryReason.NO_USABLE_OUTPUT,
    )
    assert manifest.PageDisposition.from_dict(d.to_dict()) == d


def test_from_dict_ignores_unknown_sibling_keys() -> None:
    """A sidecar/manifest entry carries many other fields alongside
    ``disposition``; the object's own ``from_dict`` must not choke if it is
    ever handed the enclosing dict by accident, and must ignore any NEW key a
    later ticket adds beside ``ending``/``primary_reason``.
    """
    payload = {
        "ending": "model_output",
        "primary_reason": "clean_native_prose",
        "some_future_field": "ignored",
    }
    got = manifest.PageDisposition.from_dict(payload)
    assert got.ending is manifest.PageEnding.MODEL_OUTPUT
    assert got.primary_reason is manifest.PagePrimaryReason.CLEAN_NATIVE_PROSE


# ---------------------------------------------------------------------------
# SelectionProvenance: the renamed WinnerKind, unchanged in every structural
# guarantee R7 pinned. These mirror test_r7_winner_kind_tags.py's checks
# because that file's OWN bijection pin is retargeted to this name (see
# test_p6_selection_provenance_tags.py) rather than relaxed here.
# ---------------------------------------------------------------------------


def test_selection_provenance_keeps_all_sixteen_members() -> None:
    SelectionProvenance = manifest.SelectionProvenance
    assert len(list(SelectionProvenance)) == 16, (
        "the selector's 16 endings must not be merged in this task -- Stage A/B "
        "is behaviour-preserving; merging is S3/S4 of the design doc, out of scope"
    )


def test_selection_provenance_is_private_not_the_public_disposition() -> None:
    """§8 Q3 ruling: WinnerKind becomes selection PROVENANCE, kept as a
    separate internal field, never the public disposition.
    """
    assert not hasattr(manifest, "WinnerKind"), (
        "WinnerKind must be renamed to SelectionProvenance, not kept as an "
        "additional public alias -- that would leave two names for one concept"
    )
    # PageDisposition must not be constructible FROM provenance directly in a
    # way that lets a provenance member leak onto the wire unmapped.
    fields = {f.name for f in dataclasses.fields(manifest.PageDisposition)}
    assert "selection_provenance" not in fields
    assert "provenance" not in fields


PROVENANCE_TO_DISPOSITION_PAIRS = {
    "UNVERIFIABLE_TABLE_MODEL_KEPT": ("MODEL_OUTPUT", "NATIVE_TABLE_UNVERIFIABLE"),
    "UNVERIFIABLE_TABLE_NATIVE": ("FAIL_CLOSED_MARKER", "NATIVE_TABLE_UNVERIFIABLE"),
    "FLAGGED_MODEL_KEPT": ("MODEL_OUTPUT", "NATIVE_TABLE_DISTRUST"),
    "STRUCTURE_CLASS_GRID_PASSING": ("MODEL_OUTPUT", "STRUCTURE_CLASS"),
    "STRUCTURE_CLASS_FLOOR": ("FAIL_CLOSED_MARKER", "STRUCTURE_CLASS"),
    "CORRUPT_MATH_HYBRID": ("MODEL_OUTPUT", "CORRUPT_MATH_HYBRID"),
}


@pytest.mark.parametrize(
    ("provenance_name", "expected"),
    sorted(PROVENANCE_TO_DISPOSITION_PAIRS.items()),
)
def test_the_six_mandated_bucket_pairs_are_pinned_exactly(provenance_name, expected) -> None:
    """§8's merge criterion, as data: for each of the six buckets Stage B
    must derive from the disposition, the (ending, reason) pair the design
    doc pins in §3.
    """
    provenance = manifest.SelectionProvenance[provenance_name]
    disposition = manifest.provenance_to_disposition(provenance)
    exp_ending, exp_reason = expected
    assert disposition.ending is manifest.PageEnding[exp_ending]
    assert disposition.primary_reason is manifest.PagePrimaryReason[exp_reason]


def test_the_mapping_is_total_over_every_provenance_member() -> None:
    """t5: a totality assertion, not a bijection -- normalized reasons
    deliberately let multiple selector rows share one cause.
    """
    for member in manifest.SelectionProvenance:
        disposition = manifest.provenance_to_disposition(member)
        assert isinstance(disposition, manifest.PageDisposition), (
            f"{member.name} has no mapped disposition -- an unmapped provenance "
            "member would crash finalization for whatever page reaches it"
        )


def test_both_structure_grid_rows_and_the_floor_share_structure_class() -> None:
    """§3: both structure-grid rows plus the floor normalize to STRUCTURE_CLASS,
    differing only by ending -- this is the allowed collapse, pinned explicitly
    rather than left to an accidental pass of the totality check above.
    """
    passing = manifest.provenance_to_disposition(
        manifest.SelectionProvenance.STRUCTURE_CLASS_GRID_PASSING
    )
    flagged = manifest.provenance_to_disposition(
        manifest.SelectionProvenance.STRUCTURE_CLASS_GRID_FLAGGED
    )
    floor = manifest.provenance_to_disposition(manifest.SelectionProvenance.STRUCTURE_CLASS_FLOOR)

    assert {passing.primary_reason, flagged.primary_reason, floor.primary_reason} == {
        manifest.PagePrimaryReason.STRUCTURE_CLASS
    }
    assert passing.ending is manifest.PageEnding.MODEL_OUTPUT
    assert flagged.ending is manifest.PageEnding.MODEL_OUTPUT
    assert floor.ending is manifest.PageEnding.FAIL_CLOSED_MARKER


def test_both_d3_native_rows_share_native_table_unverifiable() -> None:
    """§3: the D3 model-kept and D3 floor rows both answer "native table
    unverifiable", differing only by whether a model reading was kept.
    """
    kept = manifest.provenance_to_disposition(
        manifest.SelectionProvenance.UNVERIFIABLE_TABLE_MODEL_KEPT
    )
    floor = manifest.provenance_to_disposition(
        manifest.SelectionProvenance.UNVERIFIABLE_TABLE_NATIVE
    )
    assert kept.primary_reason is manifest.PagePrimaryReason.NATIVE_TABLE_UNVERIFIABLE
    assert floor.primary_reason is manifest.PagePrimaryReason.NATIVE_TABLE_UNVERIFIABLE
    assert kept.ending is manifest.PageEnding.MODEL_OUTPUT
    assert floor.ending is manifest.PageEnding.FAIL_CLOSED_MARKER


def test_native_clean_maps_to_native_prose_ending() -> None:
    d = manifest.provenance_to_disposition(manifest.SelectionProvenance.NATIVE_CLEAN)
    assert d.ending is manifest.PageEnding.NATIVE_PROSE


def test_native_fallback_maps_to_demoted_native_ending() -> None:
    """§8 Q1 ruling: NATIVE_FALLBACK is the fourth ending, not floored and not
    silently promoted to clean prose.
    """
    d = manifest.provenance_to_disposition(manifest.SelectionProvenance.NATIVE_FALLBACK)
    assert d.ending is manifest.PageEnding.DEMOTED_NATIVE


def test_no_text_marker_and_the_fail_closed_family_map_to_fail_closed() -> None:
    for name in (
        "NO_TEXT_MARKER",
        "UNVERIFIABLE_TABLE_SCANNED",
        "ROTATED_TEXT_SHREDDED",
    ):
        d = manifest.provenance_to_disposition(manifest.SelectionProvenance[name])
        assert d.ending is manifest.PageEnding.FAIL_CLOSED_MARKER, name


def test_mapping_totality_does_not_collapse_unrelated_provenance_members() -> None:
    """Guard against the totality check above being satisfied by mapping
    EVERYTHING to one reason. Distinct causes must stay distinct primary
    reasons even though the enum shrank.
    """
    corrupt = manifest.provenance_to_disposition(manifest.SelectionProvenance.CORRUPT_MATH_HYBRID)
    clean = manifest.provenance_to_disposition(manifest.SelectionProvenance.NATIVE_CLEAN)
    whole_doc = manifest.provenance_to_disposition(manifest.SelectionProvenance.WHOLE_DOC_SECTION)
    assert len({corrupt.primary_reason, clean.primary_reason, whole_doc.primary_reason}) == 3
