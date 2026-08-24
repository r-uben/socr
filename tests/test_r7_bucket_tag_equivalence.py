"""R7 part two: each bucket predicate must equal the disposition it claims.

The orchestrator historically re-derived "which branch shipped this page?" with
its own predicate, one per bucket, each carrying a comment insisting it must
match ``_winning_page_output`` EXACTLY. Nothing checked that. This module does,
mechanically, over the flag combinations that reach the cascade.

It is the safety net for swapping those predicates to read ``WinnerKind``: a
bucket that agrees with its tag everywhere here can be replaced without a
behaviour change. One that disagrees cannot -- see #292, pinned below as the
one known divergence, so the swap reproduces it rather than silently fixing it.

Hermetic: synthetic PageState only. No provider, no pipeline run, no I/O.
"""

from __future__ import annotations

import itertools

import pytest

from socr.core.manifest import (
    WinnerKind,
    _select_page_output_tagged,
    d3_floor_kept_model_output,
    flagged_model_page_output,
    structure_class_grid_winner,
    structure_class_native_fallback_applies,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState

_FLAGS = (
    "is_born_digital",
    "needs_ocr_enhancement",
    "native_table_structure_failed",
    "native_table_unverifiable",
    "native_table_header_unattributed",
    "native_table_structure_defective",
    "native_rotated_text_shredded",
    "chart_asset_render_failed",
    "text_grid_rejected",
)

#: bucket name -> (current orchestrator predicate, the tags it claims to describe)
_EQUIVALENT = {
    "flagged_model_pages": (
        lambda p: flagged_model_page_output(p) is not None,
        {WinnerKind.FLAGGED_MODEL_KEPT},
    ),
    "structure_class_model_pages": (
        lambda p: structure_class_grid_winner(p) is not None,
        {WinnerKind.STRUCTURE_CLASS_GRID_PASSING, WinnerKind.STRUCTURE_CLASS_GRID_FLAGGED},
    ),
    "structure_class_native_fallback_pages": (
        structure_class_native_fallback_applies,
        {WinnerKind.STRUCTURE_CLASS_NO_GRID},
    ),
    "d3_model_table_pages": (
        lambda p: d3_floor_kept_model_output(p) is not None,
        {WinnerKind.UNVERIFIABLE_TABLE_MODEL_KEPT},
    ),
}


def _output(engine: str = "qwen", *, passed: bool = True) -> PageOutput:
    return PageOutput(
        page_num=1,
        text="body",
        status=PageStatus.SUCCESS if passed else PageStatus.WARNING,
        engine=engine,
        audit_passed=passed,
        failure_mode=FailureMode.NONE,
    )


def _page(combo, *, attempts: bool, passes: bool, hybrid: bool) -> PageState:
    ps = PageState(page_num=1)
    for name, val in zip(_FLAGS, combo):
        setattr(ps, name, val)
    ps.native_text = "native body" if ps.is_born_digital else None
    if attempts:
        ps.attempts = [_output(passed=passes)]
        ps.best_output = _output(passed=passes)
    if hybrid:
        hyb = _output("native+math")
        ps.attempts.append(hyb)
        ps.corrupt_math_hybrid = hyb
    return ps


def _tag_of(ps: PageState) -> WinnerKind:
    state = DocumentState.__new__(DocumentState)
    state.pages = {1: ps}
    state.events = []
    return _select_page_output_tagged(state, 1, None)[1]


def _states():
    for combo in itertools.product((False, True), repeat=len(_FLAGS)):
        for attempts, passes, hybrid in itertools.product(
            (False, True), (False, True), (False, True)
        ):
            yield _page(combo, attempts=attempts, passes=passes, hybrid=hybrid)


@pytest.mark.parametrize("bucket", sorted(_EQUIVALENT))
def test_bucket_predicate_equals_its_disposition(bucket: str) -> None:
    """These four may be swapped to read the tag without changing behaviour."""
    predicate, claimed = _EQUIVALENT[bucket]
    disagreements = []
    for ps in _states():
        tag = _tag_of(ps)
        if bool(predicate(ps)) != (tag in claimed):
            disagreements.append((tag.name, {f: getattr(ps, f) for f in _FLAGS if getattr(ps, f)}))
    assert not disagreements, f"{bucket} disagrees with its tag: {disagreements[:3]}"


def test_corrupt_math_hybrid_bucket_is_broader_than_its_disposition() -> None:
    """#292, pinned so R7 part two REPRODUCES it rather than silently fixing it.

    ``corrupt_math_hybrid_pages`` tests only that the flag is set; the cascade
    additionally requires the hybrid to be un-shredded, un-blocked, present in
    ``attempts`` and engine ``native+math``. The bucket is therefore strictly
    broader -- it claims pages the manifest ships as something else.

    Asserted as a STRICT superset, in both directions: if the bucket ever stops
    over-claiming, #292 has been fixed and this test should be deleted along
    with the pin. If it ever under-claims, that is a new and different bug.
    """
    over = 0
    for ps in _states():
        in_bucket = getattr(ps, "corrupt_math_hybrid", None) is not None
        in_tag = _tag_of(ps) is WinnerKind.CORRUPT_MATH_HYBRID
        assert not (in_tag and not in_bucket), "tag without the flag is a new bug"
        over += in_bucket and not in_tag
    assert over > 0, "#292 appears fixed -- delete this pin and the divergence note"
