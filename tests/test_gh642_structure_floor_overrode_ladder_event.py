"""GH-642: pin the ``structure_floor_overrode_ladder`` audit event.

TICKET-A1b (#634) added ``_derive_structure_floor_overrides`` to
``audit_log.py``: the judge ladder ACCEPTED some table on a page
(``TABLE_LADDER_ACCEPTED_KIND`` in ``state.events``), but S1's structure-class
floor (``manifest.structure_class_floor_applies``) still discarded every
candidate and shipped the fail-closed marker instead. Despite being wired
into ``build_run_audit`` and the ``TABLE_DISTRUST_KINDS``/rank tables, this
event had zero test references anywhere in the suite before this file --
nothing pinned that it fires exactly once when the floor overrides an
accepted ladder verdict, or that it stays silent when the floor does not
apply.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from test_s1_structure_class_winner_corroboration import (
    BAD_MD,
    GOOD_MD,
    _floored_structure_class_page,
    _grid_reading_output,
)

from socr.core.audit_log import AuditEvent, build_run_audit
from socr.core.document import DocumentHandle
from socr.core.manifest import structure_class_floor_applies, structure_class_grid_winner
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge.table_verdict import TABLE_LADDER_ACCEPTED_KIND

NATIVE_PROSE = "Table 1 below reports quarterly balances for 2018-2020."


def _handle(pages: int = 1) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        return DocumentHandle(path=Path("/tmp/gh642.pdf"), page_count=pages)


def _non_grid_attempt(engine: str = "qwen") -> PageOutput:
    """A model attempt that reaches the S1 branch (R3: at least one non-
    native rung ran) but authors no usable grid, so
    ``structure_class_grid_winner`` returns ``None`` and the floor applies.
    """
    return PageOutput(
        page_num=1,
        text="no table here, just prose that never forms a grid",
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=False,
        confidence=0.5,
        failure_mode=FailureMode.NONE,
    )


def _floored_structure_class_state() -> DocumentState:
    """A born-digital, structure-class page whose S1 branch is reached and
    whose floor applies (``structure_class_floor_applies`` is True), plus a
    ``TABLE_LADDER_ACCEPTED_KIND`` event recorded for that same page -- the
    exact shape ``_derive_structure_floor_overrides`` looks for.
    """
    state = DocumentState(handle=_handle(1))
    p = state.pages[1]
    p.is_born_digital = True
    p.native_text = NATIVE_PROSE
    p.has_tables = True
    p.attempts = [_non_grid_attempt()]
    p.best_output = p.attempts[-1]
    state.events.append(
        AuditEvent(page_num=1, kind=TABLE_LADDER_ACCEPTED_KIND, detail="ladder accepted")
    )
    return state


def test_floor_override_fires_when_ladder_accepted_but_floor_applies() -> None:
    state = _floored_structure_class_state()
    assert structure_class_floor_applies(state.pages[1]) is True  # precondition

    audit = build_run_audit(state)
    override_events = [e for e in audit.events if e.kind == "structure_floor_overrode_ladder"]
    assert len(override_events) == 1
    assert override_events[0].page_num == 1


def test_no_floor_override_when_corroboration_wins_the_real_selection_path() -> None:
    """Astra review (#642): the no-event control must exercise the actual
    corroboration/winner-selection path -- not a page that exits winner
    selection before candidate scoring even runs (``has_tables=False``,
    which ``_reaches_structure_class_branch`` rejects for an unrelated
    reason). Reusing ``test_s1_structure_class_winner_corroboration``'s own
    fixture: a ragged candidate (``GOOD_MD``) whose rows measurably
    reproduce the cached native words wins the A1b row-corroboration
    fallback over a non-corroborating sibling (``BAD_MD``), so
    ``structure_class_floor_applies`` is False via the REAL predicate chain
    (``_reaches_structure_class_branch`` -> ``structure_class_grid_winner``
    -> ``_row_corroborated_grid_winner``), nothing mocked.
    """
    good = _grid_reading_output("qwen", GOOD_MD)
    bad = _grid_reading_output("gemini", BAD_MD)
    p = _floored_structure_class_page(with_native_words=True, attempts=[bad, good])

    winner = structure_class_grid_winner(p)
    assert winner is not None and winner.engine == "qwen"  # precondition: a real winner exists
    assert structure_class_floor_applies(p) is False  # precondition: the floor does not apply

    state = DocumentState(handle=_handle(1))
    state.pages[1] = p
    state.events.append(
        AuditEvent(page_num=1, kind=TABLE_LADDER_ACCEPTED_KIND, detail="ladder accepted")
    )

    audit = build_run_audit(state)
    override_events = [e for e in audit.events if e.kind == "structure_floor_overrode_ladder"]
    assert override_events == []
