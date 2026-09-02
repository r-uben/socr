"""P4-R t5: the `_is_equation_region_lane_page` predicate and its interaction
with `_is_agentic_trusted_native`, modeled on `_is_corrupt_math_recovery_page`.

Acceptance pinned here:
  - `_is_equation_region_lane_page` true only when `equation_region_lane` +
    agentic + native-eligible + not `native_only` + born digital + native text
    + `has_equations`, and false with tables, corrupt math, or shredded
    rotated native text.
  - No count/character threshold anywhere in the predicate.
  - `_is_trusted_native_without_ocr` (shared/non-agentic) stays unchanged;
    `_is_agentic_trusted_native` subtracts P4-R pages.
  - flag off -> trusted native; flag on -> not agentic-trusted;
    `native_only` -> trusted native regardless of P4-R.

Written against `UnifiedPipeline` + `PageState` directly (no PDF I/O needed
for the predicate itself). Requires t4's `equation_region_lane` field and t5's
predicate; skips cleanly until both exist.
"""

from __future__ import annotations

import dataclasses

import pytest

from socr.core.config import PipelineConfig
from socr.core.state import PageState
from socr.pipeline.orchestrator import UnifiedPipeline


def _pipeline(**overrides) -> UnifiedPipeline:
    cfg = PipelineConfig(**overrides)
    return UnifiedPipeline(cfg)


def _equation_page(**overrides) -> PageState:
    defaults = dict(
        page_num=1,
        is_born_digital=True,
        native_text="x^2 + y^2 = z^2",
        has_tables=False,
        has_equations=True,
        has_corrupt_math=False,
        native_rotated_text_shredded=False,
    )
    defaults.update(overrides)
    return PageState(**defaults)


def test_the_predicate_and_flag_exist_at_all():
    """Cold review round 1, finding 6: no self-skip on a landed acceptance
    test -- a missing predicate must fail, not vanish."""
    assert any(f.name == "equation_region_lane" for f in dataclasses.fields(PipelineConfig))
    assert callable(UnifiedPipeline._is_equation_region_lane_page)


class TestPredicateExistsAndGates:
    def _predicate(self, pipeline):
        return pipeline._is_equation_region_lane_page

    def test_eligible_equation_page_is_true_when_flag_on(self):
        pipeline = _pipeline(equation_region_lane=True, native_first=True)
        predicate = self._predicate(pipeline)
        ps = _equation_page()
        assert predicate(1, ps) is True

    def test_false_when_flag_off(self):
        pipeline = _pipeline(equation_region_lane=False, native_first=True)
        predicate = self._predicate(pipeline)
        ps = _equation_page()
        assert predicate(1, ps) is False

    def test_false_when_has_tables(self):
        pipeline = _pipeline(equation_region_lane=True, native_first=True)
        predicate = self._predicate(pipeline)
        ps = _equation_page(has_tables=True)
        assert predicate(1, ps) is False

    def test_false_when_corrupt_math(self):
        pipeline = _pipeline(equation_region_lane=True, native_first=True)
        predicate = self._predicate(pipeline)
        ps = _equation_page(has_corrupt_math=True)
        assert predicate(1, ps) is False

    def test_false_when_native_rotated_text_shredded(self):
        pipeline = _pipeline(equation_region_lane=True, native_first=True)
        predicate = self._predicate(pipeline)
        ps = _equation_page(native_rotated_text_shredded=True)
        assert predicate(1, ps) is False

    def test_false_when_no_native_text(self):
        pipeline = _pipeline(equation_region_lane=True, native_first=True)
        predicate = self._predicate(pipeline)
        ps = _equation_page(native_text=None)
        assert predicate(1, ps) is False

    def test_false_when_not_born_digital(self):
        pipeline = _pipeline(equation_region_lane=True, native_first=True)
        predicate = self._predicate(pipeline)
        ps = _equation_page(is_born_digital=False)
        assert predicate(1, ps) is False

    def test_false_under_native_only(self):
        pipeline = _pipeline(equation_region_lane=True, native_first=True, native_only=True)
        predicate = self._predicate(pipeline)
        ps = _equation_page()
        assert predicate(1, ps) is False

    def test_no_length_threshold_single_char_equation_text_still_eligible(self):
        """No count/character threshold anywhere in the predicate."""
        pipeline = _pipeline(equation_region_lane=True, native_first=True)
        predicate = self._predicate(pipeline)
        ps = _equation_page(native_text="x")
        assert predicate(1, ps) is True


class TestTrustedNativeInteraction:
    def test_shared_predicate_unchanged_flag_off(self):
        pipeline = _pipeline(equation_region_lane=False, native_first=True)
        ps = _equation_page()
        assert pipeline._is_trusted_native_without_ocr(1, ps) is True

    def test_agentic_trusted_native_excludes_p4r_page_when_flag_on(self):
        pipeline = _pipeline(equation_region_lane=True, native_first=True)
        ps = _equation_page()
        assert pipeline._is_agentic_trusted_native(1, ps) is False

    def test_agentic_trusted_native_true_when_flag_off(self):
        pipeline = _pipeline(equation_region_lane=False, native_first=True)
        ps = _equation_page()
        assert pipeline._is_agentic_trusted_native(1, ps) is True

    def test_native_only_wins_regardless_of_p4r_flag(self):
        pipeline = _pipeline(equation_region_lane=True, native_first=True, native_only=True)
        ps = _equation_page()
        assert pipeline._is_agentic_trusted_native(1, ps) is True

    def test_table_and_equation_page_unaffected_by_p4r_either_way(self):
        """Mixed table+equation pages keep going through the existing table
        route regardless of the P4-R flag."""
        ps = _equation_page(has_tables=True)
        for flag in (True, False):
            pipeline = _pipeline(equation_region_lane=flag, native_first=True)
            assert pipeline._is_trusted_native_without_ocr(1, ps) is False
            assert pipeline._is_agentic_trusted_native(1, ps) is False
