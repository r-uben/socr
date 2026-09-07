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

from socr.core.audit_log import AuditEvent, build_run_audit
from socr.core.document import DocumentHandle
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
    from socr.core.manifest import structure_class_floor_applies

    state = _floored_structure_class_state()
    assert structure_class_floor_applies(state.pages[1]) is True  # precondition

    audit = build_run_audit(state)
    override_events = [e for e in audit.events if e.kind == "structure_floor_overrode_ladder"]
    assert len(override_events) == 1
    assert override_events[0].page_num == 1


def test_no_floor_override_when_floor_does_not_apply() -> None:
    """Same ladder-accepted event, but the page is NOT structure-class (no
    tables detected) -- the floor never applies, so the override event must
    not fire.
    """
    state = DocumentState(handle=_handle(1))
    p = state.pages[1]
    p.is_born_digital = True
    p.native_text = "ordinary prose, no tables at all"
    p.has_tables = False
    p.attempts = [_non_grid_attempt()]
    p.best_output = p.attempts[-1]
    state.events.append(
        AuditEvent(page_num=1, kind=TABLE_LADDER_ACCEPTED_KIND, detail="ladder accepted")
    )

    from socr.core.manifest import structure_class_floor_applies

    assert structure_class_floor_applies(state.pages[1]) is False  # precondition

    audit = build_run_audit(state)
    override_events = [e for e in audit.events if e.kind == "structure_floor_overrode_ladder"]
    assert override_events == []
