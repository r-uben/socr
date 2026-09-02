"""R7 part one: the disposition tag on ``_select_page_output_tagged``.

The tag exists so callers stop re-deriving "which branch shipped this page?"
with mirror predicates. That only holds if the tag is TOTAL (every ending
carries one) and EXCLUSIVE (exactly one ending runs). Exclusivity is structural
-- the cascade is loop-free and single-return -- so these tests pin the shape
itself, not a sampled outcome.

Retargeted to the renamed private provenance enum (SelectionProvenance) while
keeping the full structural proof intact.

Hermetic: pure AST + a wrapper identity check. No provider, no pipeline run.
"""

from __future__ import annotations

import ast
import inspect
from collections import defaultdict
from unittest.mock import MagicMock

from socr.core import manifest
from socr.core.manifest import (
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
)


def _cascade() -> ast.FunctionDef:
    src = inspect.getsource(manifest)
    return next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_select_page_output_tagged"
    )


def _returns(fn: ast.FunctionDef) -> list[ast.Return]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Return)]


def _tag_names(ret: ast.Return) -> list[str]:
    """The SelectionProvenance names an ending can yield, in source order."""
    v = ret.value
    assert isinstance(v, ast.Tuple) and len(v.elts) == 2, f"untagged return @{ret.lineno}"
    tag = v.elts[1]
    parts = [tag.body, tag.orelse] if isinstance(tag, ast.IfExp) else [tag]
    names = []
    for node in parts:
        assert isinstance(node, ast.Attribute), f"non-SelectionProvenance tag @{ret.lineno}"
        assert isinstance(node.value, ast.Name) and node.value.id == "SelectionProvenance"
        names.append(node.attr)
    return names


def test_cascade_is_loop_free_so_exactly_one_ending_runs() -> None:
    """Exclusivity is a property of the code's SHAPE, not a convention.

    A loop (or a comprehension containing a return, which cannot occur) would
    break the "exactly one ending per page" guarantee the tag depends on.
    """
    fn = _cascade()
    loops = [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While))]
    assert loops == [], "a loop in the cascade breaks one-ending-per-page"
    assert len(_returns(fn)) == 15


def test_every_ending_carries_a_tag() -> None:
    """Totality: an untagged return would ship a page no caller can classify."""
    for r in _returns(_cascade()):
        assert _tag_names(r), f"untagged return at line {r.lineno}"


def test_tags_and_endings_are_in_bijection() -> None:
    """No dead member, and no two endings sharing a tag.

    A member declared but never returned is a disposition callers will branch on
    and never see. Two endings sharing a tag silently merges two dispositions --
    the exact "counted under a disposition it does not have" bug class R7 exists
    to kill.
    """
    used = [n for r in _returns(_cascade()) for n in _tag_names(r)]
    assert len(used) == len(set(used)) == 16, "two endings share a tag or tag count != 16"
    assert set(used) == {k.name for k in SelectionProvenance}
    assert len({k.value for k in SelectionProvenance}) == len(list(SelectionProvenance)) == 16


def test_tag_order_matches_enum_declaration_order() -> None:
    """Precedence lives in the cascade's order, so the enum must mirror it."""
    fn = _cascade()
    in_source = [n for r in sorted(_returns(fn), key=lambda r: r.lineno) for n in _tag_names(r)]
    assert in_source == [k.name for k in SelectionProvenance]


def test_public_wrapper_returns_the_output_not_the_tuple() -> None:
    """The tag is INTERNAL: the public function must still yield a bare PageOutput.

    The output-only wrapper drops private provenance while page_disposition is
    constructed only after final guards.
    """
    sentinel_output = manifest.PageOutput(
        page_num=1,
        text="| a | b |\n| --- |\n| 1 | 2 |",  # invalid table emission triggering guard
        status=manifest.PageStatus.SUCCESS,
        audit_passed=True,
    )
    calls: list[tuple] = []

    def _fake(state, page_num, whole_doc=None):
        calls.append((state, page_num, whole_doc))
        return sentinel_output, SelectionProvenance.PASSING_BEST_OUTPUT

    original_tagged = manifest._select_page_output_tagged
    original_with_prov = manifest._select_page_output_with_provenance
    manifest._select_page_output_tagged = _fake
    manifest._select_page_output_with_provenance = _fake
    try:
        # 1. Output-only wrapper drops private provenance tag and yields bare PageOutput
        got = manifest._select_page_output("STATE", 7, "WHOLE")
        assert got is sentinel_output, "wrapper did not delegate, or did not drop the tag"
        assert calls == [("STATE", 7, "WHOLE")], "wrapper dropped or reordered arguments"

        # 2. page_disposition is constructed only after final guards
        mock_state = MagicMock()
        mock_state.pages.get.return_value = None
        disp = manifest.page_disposition(mock_state, 1)
        assert disp.ending is PageEnding.FAIL_CLOSED_MARKER
        assert disp.primary_reason is PagePrimaryReason.INVALID_TABLE_EMISSION
    finally:
        manifest._select_page_output_tagged = original_tagged
        manifest._select_page_output_with_provenance = original_with_prov

    ann = inspect.signature(manifest._select_page_output).return_annotation
    assert ann in ("PageOutput", manifest.PageOutput)


def test_every_provenance_member_maps_to_a_disposition() -> None:
    """Mapping totality: every SelectionProvenance member maps to a valid PageDisposition."""
    for member in SelectionProvenance:
        d = manifest.provenance_to_disposition(member)
        assert isinstance(d, PageDisposition), f"{member.name} has no mapped disposition"
        assert isinstance(d.ending, PageEnding)
        assert isinstance(d.primary_reason, PagePrimaryReason)


def test_provenance_to_disposition_pins_allowed_equivalence_groups() -> None:
    """Pin the explicit allowed equivalence groups from SelectionProvenance to PageDisposition.

    Do not assert a bijection from provenance to PageDisposition because normalized primary
    reasons deliberately allow multiple selector rows to share one cause. Instead, pin the
    explicit allowed equivalence partitions so no accidental collapse is accepted:

    Under PageDisposition (ending, primary_reason):
      - STRUCTURE_CLASS_GRID_PASSING and STRUCTURE_CLASS_GRID_FLAGGED collapse to
        (MODEL_OUTPUT, STRUCTURE_CLASS)
      - BEST_OUTPUT_UNVERIFIED and BEST_ATTEMPT_FLAGGED collapse to
        (MODEL_OUTPUT, UNACCEPTED_OUTPUT_KEPT)
      - All other 12 provenance members map to 12 distinct dispositions (14 total groups).

    Under primary_reason alone:
      - {STRUCTURE_CLASS_GRID_PASSING, STRUCTURE_CLASS_GRID_FLAGGED, STRUCTURE_CLASS_FLOOR} -> STRUCTURE_CLASS
      - {UNVERIFIABLE_TABLE_MODEL_KEPT, UNVERIFIABLE_TABLE_NATIVE} -> NATIVE_TABLE_UNVERIFIABLE
      - {BEST_OUTPUT_UNVERIFIED, BEST_ATTEMPT_FLAGGED} -> UNACCEPTED_OUTPUT_KEPT
      - The remaining 9 members map to 9 distinct primary reasons (12 total reasons).
    """
    by_disposition: dict[PageDisposition, set[SelectionProvenance]] = defaultdict(set)
    by_reason: dict[PagePrimaryReason, set[SelectionProvenance]] = defaultdict(set)

    for member in SelectionProvenance:
        d = manifest.provenance_to_disposition(member)
        by_disposition[d].add(member)
        by_reason[d.primary_reason].add(member)

    # 1. Total count of mapped provenance members must be exactly 16
    assert len(list(SelectionProvenance)) == 16

    # 2. Check full disposition equivalence groups (exactly 14 distinct disposition pairs)
    assert len(by_disposition) == 14

    expected_multi_dispositions = {
        PageDisposition(PageEnding.MODEL_OUTPUT, PagePrimaryReason.STRUCTURE_CLASS): {
            SelectionProvenance.STRUCTURE_CLASS_GRID_PASSING,
            SelectionProvenance.STRUCTURE_CLASS_GRID_FLAGGED,
        },
        PageDisposition(PageEnding.MODEL_OUTPUT, PagePrimaryReason.UNACCEPTED_OUTPUT_KEPT): {
            SelectionProvenance.BEST_OUTPUT_UNVERIFIED,
            SelectionProvenance.BEST_ATTEMPT_FLAGGED,
        },
    }

    for disp, members in expected_multi_dispositions.items():
        assert by_disposition[disp] == members, f"mismatch for multi-member disposition {disp}"

    single_disposition_count = sum(1 for members in by_disposition.values() if len(members) == 1)
    assert single_disposition_count == 12

    # 3. Check primary reason equivalence groups (exactly 12 distinct primary reasons)
    assert len(by_reason) == 12

    expected_multi_reasons = {
        PagePrimaryReason.STRUCTURE_CLASS: {
            SelectionProvenance.STRUCTURE_CLASS_GRID_PASSING,
            SelectionProvenance.STRUCTURE_CLASS_GRID_FLAGGED,
            SelectionProvenance.STRUCTURE_CLASS_FLOOR,
        },
        PagePrimaryReason.NATIVE_TABLE_UNVERIFIABLE: {
            SelectionProvenance.UNVERIFIABLE_TABLE_MODEL_KEPT,
            SelectionProvenance.UNVERIFIABLE_TABLE_NATIVE,
        },
        PagePrimaryReason.UNACCEPTED_OUTPUT_KEPT: {
            SelectionProvenance.BEST_OUTPUT_UNVERIFIED,
            SelectionProvenance.BEST_ATTEMPT_FLAGGED,
        },
    }

    for reason, members in expected_multi_reasons.items():
        assert by_reason[reason] == members, f"mismatch for multi-member reason {reason}"

    single_reason_count = sum(1 for members in by_reason.values() if len(members) == 1)
    assert single_reason_count == 9
