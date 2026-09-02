"""P4-R t9: pin that PageState.is_structure_class() cannot fire on an
equation-only (table-free) page, before and after the P4-R lane exists.

This is the regression guard for BLOCKING 1 on #269: the previous attempt (R3)
widened `is_structure_class()` to accept equation pages, which then either
falsely accepted a pipe-shape hallucination as a grid or falsely demoted a
page that had shipped a free native SUCCESS. P4-R must never repeat this --
`has_equations` is deliberately absent from `is_structure_class()`.

These tests use only current, already-implemented surface (`PageState`,
`is_structure_class`) so they are expected to PASS immediately and continue to
pass once P4-R (t1-t8) lands; they exist to catch a future regression, not to
pin unimplemented behavior.
"""

from __future__ import annotations

from socr.core.result import PageStatus
from socr.core.state import PageState


class TestStructureClassStaysTableOnly:
    def test_equation_only_page_is_not_structure_class(self):
        ps = PageState(page_num=1, has_tables=False, has_equations=True)
        assert ps.is_structure_class() is False

    def test_table_only_page_is_structure_class(self):
        ps = PageState(page_num=1, has_tables=True, has_equations=False)
        assert ps.is_structure_class() is True

    def test_table_and_equation_page_remains_structure_class(self):
        """A mixed page stays structure-class solely because has_tables=True."""
        ps = PageState(page_num=1, has_tables=True, has_equations=True)
        assert ps.is_structure_class() is True

    def test_neither_table_nor_equation_page_is_not_structure_class(self):
        ps = PageState(page_num=1, has_tables=False, has_equations=False)
        assert ps.is_structure_class() is False

    def test_is_structure_class_ignores_native_equations_style_output(self):
        """A passing page whose text carries equation markup must not flip the
        table-only predicate -- it is driven purely by `has_tables`."""
        from socr.core.result import FailureMode, PageOutput

        ps = PageState(page_num=1, has_tables=False, has_equations=True)
        ps.best_output = PageOutput(
            page_num=1,
            text=(
                "x^2 + y^2 = z^2\n\n<!-- socr-equation: structurally-validated LaTeX candidate -->"
            ),
            status=PageStatus.SUCCESS,
            engine="native+equations",
            audit_passed=True,
            failure_mode=FailureMode.NONE,
        )
        assert ps.is_structure_class() is False


class TestReachesStructureClassBranch:
    """Behavioral proof that the C2 branch is unreachable for equation-only
    pages, via the real gate rather than string-inspecting source.

    NOTE: `_reaches_structure_class_branch` is a module-level function in
    `socr.core.manifest`, not a method on the pipeline. The tests stage looked
    for it on `UnifiedPipeline` and therefore SKIPPED both arms, which made
    this pin vacuous -- the one thing ruling 2 most needs asserted. Corrected
    to call the function where it actually lives.
    """

    def test_branch_unreachable_for_ordinary_native_equation_page(self):
        from socr.core.manifest import _reaches_structure_class_branch
        from socr.core.result import FailureMode, PageOutput

        ps = PageState(page_num=1, has_tables=False, has_equations=True)
        ps.best_output = PageOutput(
            page_num=1,
            text="x^2 + y^2 = z^2",
            status=PageStatus.SUCCESS,
            engine="native",
            audit_passed=True,
            failure_mode=FailureMode.NONE,
        )
        assert _reaches_structure_class_branch(ps) is False

    def test_branch_unreachable_for_passing_native_plus_equations_attempt(self):
        """Same page shape, but with the P4-R engine label -- still no branch
        access, since is_structure_class() never consults the engine name."""
        from socr.core.manifest import _reaches_structure_class_branch
        from socr.core.result import FailureMode, PageOutput

        ps = PageState(page_num=1, has_tables=False, has_equations=True)
        ps.best_output = PageOutput(
            page_num=1,
            text="x^2 + y^2 = z^2\n\n<!-- socr-equation -->\n```latex\nx^2+y^2=z^2\n```",
            status=PageStatus.SUCCESS,
            engine="native+equations",
            audit_passed=True,
            failure_mode=FailureMode.NONE,
        )
        assert _reaches_structure_class_branch(ps) is False

    def test_branch_unreachable_whatever_the_lane_flag_is(self):
        """The floor's reachability is a property of the PAGE, not of P4-R's
        flag: neither state of `equation_region_lane` can open the branch on a
        table-free page."""
        from socr.core.config import PipelineConfig
        from socr.core.manifest import _reaches_structure_class_branch
        from socr.core.result import FailureMode, PageOutput

        for flag in (True, False):
            cfg = PipelineConfig(equation_region_lane=flag)
            assert cfg.equation_region_lane is flag
            for engine in ("native", "native+equations"):
                ps = PageState(page_num=1, has_tables=False, has_equations=True)
                ps.best_output = PageOutput(
                    page_num=1,
                    text="y = 2x + 1",
                    status=PageStatus.SUCCESS,
                    engine=engine,
                    audit_passed=True,
                    failure_mode=FailureMode.NONE,
                )
                assert ps.is_structure_class() is False
                assert _reaches_structure_class_branch(ps) is False

    def test_a_table_page_does_reach_the_branch_positive_control(self):
        """Without this the pins above could pass because the gate is broken.

        The shape that DOES reach the branch: a born-digital structure-class
        page whose winner is the native lane (so native is distrusted) with a
        non-native attempt on record. Change `has_tables` to False -- the only
        difference -- and the branch closes, which is precisely ruling 2.
        """
        from socr.core.manifest import _reaches_structure_class_branch
        from socr.core.result import FailureMode, PageOutput

        def _page(*, has_tables: bool) -> PageState:
            ps = PageState(
                page_num=1,
                has_tables=has_tables,
                has_equations=True,
                is_born_digital=True,
                native_text="| a | b |",
            )
            native_win = PageOutput(
                page_num=1,
                text="| a | b |",
                status=PageStatus.SUCCESS,
                engine="native",
                audit_passed=True,
                failure_mode=FailureMode.NONE,
            )
            model_attempt = PageOutput(
                page_num=1,
                text="| a | b |\n| --- | --- |\n| 1 | 2 |",
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=False,
                failure_mode=FailureMode.NONE,
            )
            ps.attempts = [model_attempt, native_win]
            ps.best_output = native_win
            return ps

        assert _reaches_structure_class_branch(_page(has_tables=True)) is True
        assert _reaches_structure_class_branch(_page(has_tables=False)) is False
