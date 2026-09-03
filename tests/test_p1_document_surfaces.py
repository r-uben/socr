"""P1 (task t10): TABLE_WITHHELD surfaces at every document/consumer
boundary -- assemble buckets, document status, the ladder note, and the
trust registries -- not just the page's own ``failure_mode``.

Pure/hermetic: exercises ``_derive_orthogonal_assemble_buckets`` and
``_table_ladder_terminal`` directly against a bare ``DocumentState``, no
pipeline run, no I/O.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.document import DocumentHandle
from socr.core.result import FailureMode
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import (
    _ORTHOGONAL_ASSEMBLE_BUCKET_NAMES,
    _derive_orthogonal_assemble_buckets,
    _table_ladder_terminal,
)


def _state(page_count: int = 2) -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=Path("/tmp/doc.pdf"), page_count=page_count)
    return DocumentState(handle=handle)


class TestTableLadderTerminalRecognizesWithheld:
    def test_withheld_disposition_is_recognized(self) -> None:
        state = _state(1)
        state.pages[1].table_ladder_disposition = FailureMode.TABLE_WITHHELD
        assert _table_ladder_terminal(state.pages[1]) is FailureMode.TABLE_WITHHELD

    def test_withheld_is_distinct_from_rejected_and_unverified(self) -> None:
        state = _state(1)
        state.pages[1].table_ladder_disposition = FailureMode.TABLE_WITHHELD
        terminal = _table_ladder_terminal(state.pages[1])
        assert terminal is not FailureMode.TABLE_REJECTED
        assert terminal is not FailureMode.TABLE_UNVERIFIED


class TestOrthogonalAssembleBucket:
    def test_table_withheld_pages_is_a_registered_bucket_name(self) -> None:
        assert "table_withheld_pages" in _ORTHOGONAL_ASSEMBLE_BUCKET_NAMES

    def test_withheld_page_lands_in_its_own_bucket(self) -> None:
        state = _state(3)
        state.pages[1].table_ladder_disposition = FailureMode.TABLE_WITHHELD
        state.pages[2].table_ladder_disposition = FailureMode.TABLE_REJECTED
        # page 3 stays clean

        buckets = _derive_orthogonal_assemble_buckets(state)

        assert buckets["table_withheld_pages"] == [1]
        assert buckets["table_rejected_pages"] == [2]
        assert buckets["table_unverified_pages"] == []

    def test_the_three_table_buckets_are_mutually_exclusive(self) -> None:
        state = _state(3)
        state.pages[1].table_ladder_disposition = FailureMode.TABLE_WITHHELD
        state.pages[2].table_ladder_disposition = FailureMode.TABLE_REJECTED
        state.pages[3].table_ladder_disposition = FailureMode.TABLE_UNVERIFIED

        buckets = _derive_orthogonal_assemble_buckets(state)

        withheld = set(buckets["table_withheld_pages"])
        rejected = set(buckets["table_rejected_pages"])
        unverified = set(buckets["table_unverified_pages"])
        assert withheld.isdisjoint(rejected)
        assert withheld.isdisjoint(unverified)
        assert rejected.isdisjoint(unverified)

    def test_no_withheld_pages_yields_empty_bucket(self) -> None:
        state = _state(1)
        buckets = _derive_orthogonal_assemble_buckets(state)
        assert buckets["table_withheld_pages"] == []


class TestTableLadderWithheldRegisteredInTrustModule:
    def test_table_ladder_withheld_is_a_distrust_kind(self) -> None:
        from socr.core.tables_trust import TABLE_DISTRUST_KINDS

        assert "table_ladder_withheld" in TABLE_DISTRUST_KINDS

    def test_table_ladder_withheld_has_a_terminal_note(self) -> None:
        from socr.core.tables_trust import LADDER_TERMINAL_NOTES

        assert "table_ladder_withheld" in LADDER_TERMINAL_NOTES

    def test_withheld_table_appears_in_the_built_trust_index(self) -> None:
        from socr.core.tables_trust import build_tables_trust

        events = [_fake_event("table_ladder_withheld", page_num=1, table_id="t0")]
        trust = build_tables_trust("doc.pdf", events)

        assert 1 in trust.pages
        assert "table_ladder_withheld" in trust.pages[1].reasons

    def test_accepted_event_does_not_clear_a_different_withheld_table_on_same_page(self) -> None:
        """A table_ladder_accepted event is table-scoped resolution -- it
        must not blanket-clear a sibling table's WITHHELD verdict on the
        same page's trust index."""
        from socr.core.tables_trust import build_tables_trust

        events = [
            _fake_event("table_ladder_withheld", page_num=1, table_id="t0"),
            _fake_event("table_ladder_accepted", page_num=1, table_id="t1"),
        ]
        trust = build_tables_trust("doc.pdf", events)

        assert 1 in trust.pages, "the withheld table's distrust must survive a sibling's acceptance"
        assert "table_ladder_withheld" in trust.pages[1].reasons


def _fake_event(kind: str, *, page_num: int, table_id: str):
    from socr.core.audit_log import AuditEvent

    return AuditEvent(page_num=page_num, kind=kind, data={"table_id": table_id})


class TestTheLatchNamesTheRungItActuallyRecorded:
    """Cold review round 1, finding 8.

    The document note and the CLI line used to read only the page-level
    boolean ``table_judge_retry_pending`` and then tell the user an
    ADJUDICATOR rung was unavailable. The latch may belong to a reader
    instead: table A on a page is withheld after a blind mismatch while table
    B on the same page cannot reach ``gemini``, and the page reducer yields
    WITHHELD plus a ``gemini`` latch. Naming the adjudicator there sends the
    reader to fix the wrong provider, even though ``table_judge_retry_rungs``
    holds the answer.
    """

    def _withheld_state_latched_on(self, kinds):
        state = _state(1)
        ps = state.pages[1]
        ps.table_ladder_disposition = FailureMode.TABLE_WITHHELD
        ps.table_judge_retry_pending = bool(kinds)
        ps.table_judge_retry_rungs = list(kinds)
        return state

    def test_the_helper_returns_the_recorded_kinds(self):
        from socr.pipeline.orchestrator import _latched_rung_kinds

        state = self._withheld_state_latched_on(["gemini"])
        assert _latched_rung_kinds(state, [1]) == ["gemini"]

    def test_an_unlatched_page_contributes_no_kind(self):
        from socr.pipeline.orchestrator import _latched_rung_kinds

        state = self._withheld_state_latched_on([])
        assert _latched_rung_kinds(state, [1]) == []

    def test_the_document_note_names_a_reader_latch_as_the_reader(self):
        from socr.pipeline.orchestrator import UnifiedPipeline

        gemini_note = UnifiedPipeline._table_judge_ladder_note(
            self._withheld_state_latched_on(["gemini"])
        )
        adjudicator_note = UnifiedPipeline._table_judge_ladder_note(
            self._withheld_state_latched_on(["adjudicator"])
        )

        assert "gemini" in gemini_note
        assert "adjudicator" not in gemini_note
        assert "adjudicator" in adjudicator_note
        # The two latches must not read identically -- that was the bug.
        assert gemini_note != adjudicator_note

    def test_an_unlatched_withhold_promises_no_retry(self):
        from socr.pipeline.orchestrator import UnifiedPipeline

        note = UnifiedPipeline._table_judge_ladder_note(self._withheld_state_latched_on([]))
        assert "retryable on resume" not in note
