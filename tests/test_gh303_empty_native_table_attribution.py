"""GH-303: an empty native table is named for what it is, not as GH-151.

A native table with a valid header and delimiter but no body content is demoted via
the ``native_table_structure_defective`` aggregate, which includes the GH-190 content
term. The native audit loop then mapped *(aggregate AND NOT emission)* to
``grid_shape`` -- GH-151's defect, whose ``defect_detail`` text describes ragged
widths and a detached label row, neither of which happened.

Right disposition, wrong ticket: anyone counting GH-151 against GH-190 mis-attributed
the page. This is an attribution fix only. The page is still demoted, never restamped
SUCCESS, and ``--native-only`` is not overridden.

Testing note (CLAUDE.md): nothing here drives a provider. Each test pins a DIFFERENCE
between two pages that vary only in which defect fired.
"""

from __future__ import annotations

from socr.core.config import PipelineConfig
from socr.pipeline.orchestrator import UnifiedPipeline

EMPTY_BODY = "| Year | Coefficient |\n| --- | --- |\n|  |  |\n|  |  |\n"
RAGGED = "| Year | Coefficient |\n| --- | --- |\n| 2019 | 0.31 | 0.44 |\n"


class _Assessment:
    def __init__(self, pages):
        self.pages = pages


class _Page:
    """Only the flags the native audit loop reads."""

    def __init__(self, *, content_defect="", emission_defect="", structure_defective=False):
        self.page_num = 1
        self.native_table_content_defect = content_defect
        self.native_table_emission_defect = emission_defect
        self.native_table_structure_defective = structure_defective
        self.native_table_header_unattributed = False
        self.has_unverifiable_table_region = False


class _State:
    def __init__(self):
        self.events = []


def _defects_for(page) -> list[str]:
    """Run the native audit loop over one page and return its recorded defects."""
    pipe = UnifiedPipeline(PipelineConfig(quiet=True))
    state = _State()
    pipe._emit_native_table_structure_events(state, _Assessment([page]))
    if not state.events:
        return []
    return list(state.events[0].data.get("defects", []))


def test_an_empty_native_table_is_not_reported_as_grid_shape():
    """The regression: the GH-190 term arrived under GH-151's name."""
    empty = _defects_for(_Page(content_defect="table_content_empty", structure_defective=True))

    assert "table_content_empty" in empty
    assert "grid_shape" not in empty, (
        "an empty body is GH-190, not GH-151's ragged-widths / detached-label defect"
    )


def test_a_genuinely_ragged_grid_is_still_grid_shape():
    """The other side of the difference -- the fix must not steal GH-151's cases."""
    ragged = _defects_for(_Page(structure_defective=True))

    assert ragged == ["grid_shape"]


def test_the_two_pages_are_reported_differently():
    """Pinned as a DIFFERENCE: before the fix both produced `grid_shape`."""
    empty = _defects_for(_Page(content_defect="table_content_empty", structure_defective=True))
    ragged = _defects_for(_Page(structure_defective=True))

    assert empty != ragged


def test_an_emission_defect_still_wins_over_the_content_term():
    """Ordering is unchanged: raw-emission provenance keeps its precedence."""
    both = _defects_for(
        _Page(
            emission_defect="table_latex_leak",
            content_defect="table_content_empty",
            structure_defective=True,
        )
    )

    assert both == ["table_latex_leak"]


def test_the_page_is_still_demoted_not_restamped():
    """Attribution fix only -- the disposition must not move."""
    pipe = UnifiedPipeline(PipelineConfig(quiet=True))
    state = _State()
    pipe._emit_native_table_structure_events(
        state, _Assessment([_Page(content_defect="table_content_empty", structure_defective=True)])
    )

    assert len(state.events) == 1
    event = state.events[0]
    assert event.kind == "table_structure_failed"
    assert event.engine == "native"
    assert "demoted to flagged WARNING" in event.detail
    # The named cause reaches the human-readable detail, not just the data blob.
    assert "no body content" in event.detail


def test_a_clean_page_raises_nothing():
    """No defect, no event -- the loop must not manufacture one."""
    assert _defects_for(_Page()) == []


def _real_state():
    """A real DocumentState, built the way GH-226's guard builds one."""
    from pathlib import Path
    from unittest.mock import patch

    from socr.core.state import DocumentHandle, DocumentState

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=Path("/tmp/gh345.pdf"), page_count=1)
    return DocumentState(handle=handle)


class TestTheProductionStampNotAStubbedFlag:
    """GH-345. The tests above stub ``native_table_content_defect`` on a fake
    page and call the emitter directly, so reverting the ``PageAssessment``
    kwarg or the ``apply_born_digital`` copy left them green: the mapper was
    pinned, the production stamp was not.

    This follows GH-226's pattern -- build a real ``PageAssessment`` and
    propagate it through ``apply_born_digital`` -- and takes the defect from
    ``structure_check.table_content_defect`` rather than a literal, so the
    fixture cannot drift from what production computes.
    """

    def _assessment(self, native_text: str):
        from pathlib import Path

        from socr.core.born_digital import DocumentAssessment, PageAssessment
        from socr.tables import structure_check

        defect = structure_check.table_content_defect(native_text)
        return (
            DocumentAssessment(
                path=Path("/tmp/gh345.pdf"),
                pages=[
                    PageAssessment(
                        page_num=1,
                        is_born_digital=True,
                        native_text=native_text,
                        confidence=1.0,
                        has_tables=True,
                        native_table_structure_defective=True,
                        native_table_content_defect=defect,
                    )
                ],
            ),
            defect,
        )

    def test_the_empty_body_defect_is_what_structure_check_produces(self) -> None:
        """Precondition: if the fixture stopped being an empty-body table this
        class would pin nothing, so assert the source of truth first."""
        from socr.tables import structure_check

        assert structure_check.table_content_defect(EMPTY_BODY) == "table_content_empty"
        assert structure_check.table_content_defect(RAGGED) == ""

    def test_apply_born_digital_copies_the_content_defect_onto_page_state(self) -> None:
        assessment, defect = self._assessment(EMPTY_BODY)
        state = _real_state()
        state.apply_born_digital(assessment)

        assert state.pages[1].native_table_content_defect == defect

    def test_empty_and_ragged_still_differ_after_propagation(self) -> None:
        """The difference pin the ticket asks to keep, now through production."""
        empty_state = _real_state()
        empty_state.apply_born_digital(self._assessment(EMPTY_BODY)[0])

        ragged_state = _real_state()
        ragged_state.apply_born_digital(self._assessment(RAGGED)[0])

        assert (
            empty_state.pages[1].native_table_content_defect
            != ragged_state.pages[1].native_table_content_defect
        )
