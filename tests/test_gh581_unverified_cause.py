"""GH-581: an UNVERIFIED terminal must name its real cause and must not

promise a retry the P1 latch will not perform.

Two defects, one branch of ``_run_table_judge_gate``'s per-table message
loop each:

* the binding-contradiction and default UNVERIFIED branches printed a fixed
  "retryable on resume" clause regardless of whether the page's own latch
  (``table_judge_retry_pending``) actually fired for it;
* the audit trail dropped the cause -- ``RungResult.error``/``unavailable``/
  ``refusal`` never reached ``rung_trail``, and the guard chain's own
  decision (``guard_detail_by_table``) was read only by the ACCEPTED/
  WITHHELD messages, never by UNVERIFIED.

Each test below pins a DIFFERENCE (CLAUDE.md's own rule): the same table,
varying exactly one fact -- whether a rung was unavailable, what a rung's
¬S1 answer said, whether native geometry contradicted the extraction, or
whether the guard chain cleared the table -- and asserts the wording and the
``data`` block track that fact, never a fixed default.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from _adjudicator_doubles import mismatching_adjudicator, unavailable_adjudicator

from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge.table_verdict import (
    CAUSE_BINDING_CONTRADICTION,
    CAUSE_GUARD_ERROR,
    CAUSE_GUARD_NOT_CLEARED,
    CAUSE_INSUFFICIENT_CORROBORATION,
    CAUSE_MISSING_TABLE_TERMINAL,
    CAUSE_NO_RUNGS_CONFIGURED,
    CAUSE_NO_WITNESS,
    CAUSE_RUNG_NOT_ACCEPTED,
    CAUSE_RUNG_UNAVAILABLE,
    CAUSE_UNKNOWN,
    CAUSE_WITNESS_PREPARATION_ERROR,
    TABLE_LADDER_UNVERIFIED_CAUSES,
    TABLE_LADDER_UNVERIFIED_KIND,
    RungResult,
    TableJudgeVerdict,
)
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.binding import BindingEvidence
from socr.tables.witness import TableWitness, WitnessScope, WitnessStatus

_TABLE_MD = (
    "| c0 | c1 | c2 | c3 |\n"
    "| --- | --- | --- | --- |\n"
    "| 10 | 11 | 12 | 13 |\n"
    "| 20 | 21 | 22 | 23 |\n"
    "| 30 | 31 | 32 | 33 |\n"
)


def _ruled_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """One page, one ruled (fully boxed) table -> exactly 1 located witness."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 220, 300, 380]
    rows = [100 + i * 22 for i in range(4)]
    for r, y in enumerate(rows):
        for c, x in enumerate(cols):
            page.insert_text((x + 4, y + 12), f"{r}{c}", fontsize=9)
    for yy in rows:
        page.draw_line((100, yy), (460, yy))
    for xx in cols + [460]:
        page.draw_line((xx, rows[0]), (xx, rows[-1]))
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_config(**overrides) -> PipelineConfig:
    kwargs = dict(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        table_judge_ladder=True,
    )
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def _make_pipeline(*, evidence=BindingEvidence.ABSTAIN, adjudicator=None) -> UnifiedPipeline:
    pipeline = UnifiedPipeline(_make_config())
    pipeline._binding_evidence_for_witness = lambda *a, **kw: (None, evidence)
    pipeline._build_table_cell_adjudicator = lambda: adjudicator
    return pipeline


def _make_state(pdf_path: Path) -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf_path, page_count=1)
    return DocumentState(handle=handle)


def _bo(text: str = _TABLE_MD) -> PageOutput:
    return PageOutput(
        page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )


class _QueueRung:
    def __init__(self, results: list[RungResult], rung_id: str = "fake") -> None:
        self._results = list(results)
        self.rung_id = rung_id
        self.calls: list[tuple] = []

    def __call__(self, crop_path, markdown, prior_findings):
        self.calls.append((crop_path, markdown, prior_findings))
        return self._results.pop(0)


def _unverified_event(state: DocumentState) -> AuditEvent:
    events = [e for e in state.events if e.kind == TABLE_LADDER_UNVERIFIED_KIND]
    assert len(events) == 1, f"expected exactly one unverified terminal, got {events}"
    return events[0]


def _events_of_kind(state: DocumentState, kind: str) -> list[AuditEvent]:
    return [e for e in state.events if e.kind == kind]


class _StubWitnessCtx:
    def __init__(self, witnesses: list[TableWitness]) -> None:
        self._witnesses = witnesses

    def __enter__(self):
        return self._witnesses

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# 1. Same table: latch absent vs latch present.
# ---------------------------------------------------------------------------


class TestRetryableTracksTheLatch:
    def test_latch_absent_vs_present_on_the_same_table(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()

        # -- latch absent: the rung answered but was not accepted (¬S1,
        #    no availability signal) --
        state_absent = _make_state(_ruled_pdf(tmp_path / "absent"))
        rung_absent = _QueueRung(
            [RungResult(rung="fake1", ok=False, error="no JSON object found in rung output")]
        )
        pipeline._run_table_judge_gate(state_absent, 1, state_absent.pages[1], _bo(), [rung_absent])
        event_absent = _unverified_event(state_absent)

        assert event_absent.data["latched"] is False
        assert state_absent.pages[1].table_judge_retry_pending is False
        assert "retryable" not in event_absent.detail

        # -- latch present: the SAME table, the SAME rung, differing only in
        #    ``unavailable=True`` (a transport outage) --
        state_present = _make_state(_ruled_pdf(tmp_path / "present"))
        rung_present = _QueueRung(
            [
                RungResult(
                    rung="fake1",
                    ok=False,
                    error="httpx.ConnectError: connection refused",
                    unavailable=True,
                )
            ]
        )
        pipeline._run_table_judge_gate(
            state_present, 1, state_present.pages[1], _bo(), [rung_present]
        )
        event_present = _unverified_event(state_present)

        assert event_present.data["latched"] is True
        assert state_present.pages[1].table_judge_retry_pending is True
        assert "retryable on resume" in event_present.detail


# ---------------------------------------------------------------------------
# 2. A rung that answered but was not accepted: the trail carries its
#    reason, and the cause names it as an unaccepted answer, not an outage.
# ---------------------------------------------------------------------------


class TestRungNotAcceptedCarriesItsReason:
    def test_ok_false_with_a_reason_is_named_in_the_trail_and_the_cause(
        self, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline()
        state = _make_state(_ruled_pdf(tmp_path))
        rung = _QueueRung(
            [RungResult(rung="fake1", ok=False, error="malformed verdict: missing 'confidence'")]
        )
        pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [rung])

        event = _unverified_event(state)
        trail = event.data["rung_trail"]
        assert len(trail) == 1
        assert trail[0]["error"] == "malformed verdict: missing 'confidence'"
        assert trail[0]["unavailable"] is False
        assert trail[0]["refusal"] is False
        assert event.data["cause"] == CAUSE_RUNG_NOT_ACCEPTED
        assert event.data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES
        assert event.data["latched"] is False


# ---------------------------------------------------------------------------
# 3. A binding CONTRADICT with no rung unavailable: names the cause,
#    promises no retry.
# ---------------------------------------------------------------------------


class TestBindingContradictionNamesItsCause:
    def test_contradiction_with_no_unavailable_rung_is_not_retryable(self, tmp_path: Path) -> None:
        # An adjudicator that WOULD clear the table if it were ever asked --
        # GH-575 says it must not be, because a contradiction is terminal on
        # its own. Present here only to prove that.
        adj = mismatching_adjudicator()
        pipeline = _make_pipeline(evidence=BindingEvidence.CONTRADICT, adjudicator=adj)
        state = _make_state(_ruled_pdf(tmp_path))
        rung = _QueueRung(
            [
                RungResult(
                    rung="fake1",
                    ok=True,
                    verdict=TableJudgeVerdict(verdict="FAIL", confidence="high", findings=[]),
                )
            ]
        )
        pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [rung])

        event = _unverified_event(state)
        assert event.data["cause"] == CAUSE_BINDING_CONTRADICTION
        assert event.data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES
        assert event.data["latched"] is False
        assert "retryable" not in event.detail
        assert adj.calls == [], "the adjudicator must never be asked after a contradiction"


# ---------------------------------------------------------------------------
# 4. Two-low-pass with the guard chain unable to clear it: the event's
#    guard_detail matches the guard's own decision detail, cause names it.
# ---------------------------------------------------------------------------


class TestGuardNotClearedNamesTheGuardDetail:
    def test_two_low_pass_no_adjudicator_is_guard_not_cleared(self, tmp_path: Path) -> None:
        # No adjudicator configured -- the guard chain reaches its NOT_CLEARED
        # disposition (nobody established anything against or for the table).
        pipeline = _make_pipeline(evidence=BindingEvidence.ABSTAIN, adjudicator=None)
        state = _make_state(_ruled_pdf(tmp_path))
        two_low = [
            _QueueRung(
                [
                    RungResult(
                        rung="r1",
                        ok=True,
                        verdict=TableJudgeVerdict(
                            verdict="PASS", confidence="low", findings=[], doubts=["R1C2"]
                        ),
                    )
                ],
                "r1",
            ),
            _QueueRung(
                [
                    RungResult(
                        rung="r2",
                        ok=True,
                        verdict=TableJudgeVerdict(
                            verdict="PASS", confidence="low", findings=[], doubts=["R2C3"]
                        ),
                    )
                ],
                "r2",
            ),
        ]
        pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), two_low)

        event = _unverified_event(state)
        assert event.data["cause"] == CAUSE_GUARD_NOT_CLEARED
        assert event.data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES
        assert event.data["guard_detail"], "the guard chain's own detail must be recorded"
        # GH-581 cold review round 4, finding 1: the raw guard detail lives
        # ONLY in structured data now, never interpolated into ``detail``.
        assert event.data["guard_detail"] not in event.detail
        assert event.data["latched"] is False
        assert "retryable" not in event.detail


# ---------------------------------------------------------------------------
# 5. A mixed page: an unwitnessed table beside a table with an unavailable
#    rung. The page latch is true, but the unwitnessed table's OWN
#    "not retryable" fact must not be overwritten -- the wording must carry
#    both the table-scoped fact and the page-scoped one, never silently
#    disagree (GH-581 cold review round 1, finding 1).
# ---------------------------------------------------------------------------


def _mixed_page_witnesses(tmp_path: Path) -> list[TableWitness]:
    crop_path = tmp_path / "crop.png"
    crop_path.write_bytes(b"fake-png")
    return [
        TableWitness(
            table_id="p1-t0",
            page_num=1,
            block_index=0,
            markdown=_TABLE_MD,
            status=WitnessStatus.LOCATED,
            crop_path=crop_path,
            scope=WitnessScope.LOCATED,
        ),
        TableWitness(
            table_id="p1-t1",
            page_num=1,
            block_index=1,
            markdown=_TABLE_MD,
            status=WitnessStatus.MISSING,
            scope=WitnessScope.NONE,
        ),
    ]


class TestMixedPageLatchIsPageScoped:
    def test_positive_clause_present_only_when_the_page_latch_fires(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()

        # -- page latch ABSENT: the located table's rung answered but was
        #    not accepted (no availability signal) --
        state_absent = _make_state(_ruled_pdf(tmp_path / "absent"))
        rung_absent = _QueueRung(
            [RungResult(rung="fake1", ok=False, error="no JSON object found in rung output")]
        )
        with patch(
            "socr.tables.witness.prepare_table_witnesses",
            return_value=_StubWitnessCtx(_mixed_page_witnesses(tmp_path / "absent")),
        ):
            pipeline._run_table_judge_gate(
                state_absent, 1, state_absent.pages[1], _bo(), [rung_absent]
            )

        unwitnessed_absent = next(
            e
            for e in _events_of_kind(state_absent, TABLE_LADDER_UNVERIFIED_KIND)
            if e.data.get("table_id") == "p1-t1"
        )
        assert state_absent.pages[1].table_judge_retry_pending is False
        assert unwitnessed_absent.data["latched"] is False
        assert unwitnessed_absent.data.get("retryable") is False
        assert "the page itself is retryable" not in unwitnessed_absent.detail
        assert "will not be retried" in unwitnessed_absent.detail

        # -- page latch PRESENT: the SAME shape, but the located table's
        #    rung is a transport outage (unavailable=True) --
        state_present = _make_state(_ruled_pdf(tmp_path / "present"))
        rung_present = _QueueRung(
            [
                RungResult(
                    rung="fake1",
                    ok=False,
                    error="httpx.ConnectError: connection refused",
                    unavailable=True,
                )
            ]
        )
        with patch(
            "socr.tables.witness.prepare_table_witnesses",
            return_value=_StubWitnessCtx(_mixed_page_witnesses(tmp_path / "present")),
        ):
            pipeline._run_table_judge_gate(
                state_present, 1, state_present.pages[1], _bo(), [rung_present]
            )

        unwitnessed_present = next(
            e
            for e in _events_of_kind(state_present, TABLE_LADDER_UNVERIFIED_KIND)
            if e.data.get("table_id") == "p1-t1"
        )
        assert state_present.pages[1].table_judge_retry_pending is True
        assert unwitnessed_present.data["latched"] is True
        # The table's OWN fact is unchanged: p1-t1 itself is still never
        # going to be retried, only the PAGE is (because of its sibling).
        assert unwitnessed_present.data.get("retryable") is False
        assert "the page itself is retryable" in unwitnessed_present.detail
        assert "will not be retried" in unwitnessed_present.detail


# ---------------------------------------------------------------------------
# 6. No rung configured for a located witness: a known configuration fact,
#    not the generic unknown fallback.
# ---------------------------------------------------------------------------


class TestNoRungsConfiguredIsItsOwnCause:
    def test_empty_rung_list_on_a_located_witness_is_no_rungs_configured(
        self, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline()
        state = _make_state(_ruled_pdf(tmp_path))
        pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [])

        event = _unverified_event(state)
        assert event.data["cause"] == CAUSE_NO_RUNGS_CONFIGURED
        assert event.data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES
        assert event.data["latched"] is False
        assert "retryable" not in event.detail


# ---------------------------------------------------------------------------
# 7. A lone low-confidence PASS with no corroboration: its own cause, not
#    the generic unknown fallback and not a guard-chain concern.
# ---------------------------------------------------------------------------


class TestInsufficientCorroborationIsItsOwnCause:
    def test_a_lone_low_confidence_pass_is_insufficient_corroboration(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        state = _make_state(_ruled_pdf(tmp_path))
        rung = _QueueRung(
            [
                RungResult(
                    rung="fake1",
                    ok=True,
                    verdict=TableJudgeVerdict(
                        verdict="PASS", confidence="low", findings=[], doubts=[]
                    ),
                )
            ]
        )
        pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [rung])

        event = _unverified_event(state)
        assert event.data["cause"] == CAUSE_INSUFFICIENT_CORROBORATION
        assert event.data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES
        assert event.data["latched"] is False
        assert "retryable" not in event.detail


# ---------------------------------------------------------------------------
# Cold review round 2
# ---------------------------------------------------------------------------


def _two_low_pass_rungs():
    return [
        _QueueRung(
            [
                RungResult(
                    rung="r1",
                    ok=True,
                    verdict=TableJudgeVerdict(
                        verdict="PASS", confidence="low", findings=[], doubts=["R1C2"]
                    ),
                )
            ],
            "r1",
        ),
        _QueueRung(
            [
                RungResult(
                    rung="r2",
                    ok=True,
                    verdict=TableJudgeVerdict(
                        verdict="PASS", confidence="low", findings=[], doubts=["R2C3"]
                    ),
                )
            ],
            "r2",
        ),
    ]


# ---------------------------------------------------------------------------
# 8. The assemble-time completeness backfill: a second emit site for
#    table_ladder_unverified, outside the per-table message loop, must not
#    repeat the false "retryable on resume" promise (finding 1).
# ---------------------------------------------------------------------------


class TestAssembleBackfillNamesItsOwnCause:
    def test_backfill_terminal_is_missing_table_terminal_not_retryable(
        self, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline()
        state = _make_state(_ruled_pdf(tmp_path))
        ps = state.pages[1]
        ps.best_output = _bo()

        pipeline._backfill_missing_table_ladder_terminals(state, [_TABLE_MD])

        event = _unverified_event(state)
        assert ps.table_judge_retry_pending is False
        assert ps.table_ladder_incomplete is True
        assert "retryable on resume" not in event.detail
        assert event.data["cause"] == CAUSE_MISSING_TABLE_TERMINAL
        assert event.data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES
        assert event.data["latched"] is False
        assert event.data["guard_detail"] is None
        assert event.data["rung_trail"] == []

    def test_document_note_names_it_incomplete_not_retryable_or_not(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        state = _make_state(_ruled_pdf(tmp_path))
        ps = state.pages[1]
        ps.best_output = _bo()
        pipeline._backfill_missing_table_ladder_terminals(state, [_TABLE_MD])
        ps.table_ladder_disposition = FailureMode.TABLE_UNVERIFIED

        note = pipeline._table_judge_ladder_note(state)
        assert note is not None
        assert "incomplete" in note
        # Must not be claimed by either of the other two sentences.
        assert "ladder exhausted without an answer" not in note
        assert "ladder found no accepted verdict" not in note


# ---------------------------------------------------------------------------
# 9. Cause classification keys on the TYPED fields, not on whether ``error``
#    happens to be non-empty (finding 2).
# ---------------------------------------------------------------------------


class TestCauseClassificationIsTyped:
    def test_empty_error_with_unavailable_is_still_rung_unavailable(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()

        state_empty = _make_state(_ruled_pdf(tmp_path / "empty"))
        rung_empty = _QueueRung([RungResult(rung="fake1", ok=False, error="", unavailable=True)])
        pipeline._run_table_judge_gate(state_empty, 1, state_empty.pages[1], _bo(), [rung_empty])
        event_empty = _unverified_event(state_empty)

        state_worded = _make_state(_ruled_pdf(tmp_path / "worded"))
        rung_worded = _QueueRung(
            [RungResult(rung="fake1", ok=False, error="boom", unavailable=True)]
        )
        pipeline._run_table_judge_gate(state_worded, 1, state_worded.pages[1], _bo(), [rung_worded])
        event_worded = _unverified_event(state_worded)

        assert event_empty.data["cause"] == CAUSE_RUNG_UNAVAILABLE
        assert event_empty.data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES
        assert event_worded.data["cause"] == CAUSE_RUNG_UNAVAILABLE
        assert event_worded.data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES
        assert event_empty.data["latched"] is True
        assert event_worded.data["latched"] is True
        # The text differs (nothing to join vs the real reason); the
        # classification does not.
        assert event_empty.data["rung_trail"][0]["error"] == ""
        assert event_worded.data["rung_trail"][0]["error"] == "boom"


# ---------------------------------------------------------------------------
# 10. The guard chain's TYPED decision reaches the UNVERIFIED event's data,
#     not just its free-text detail (finding 3).
# ---------------------------------------------------------------------------


class TestGuardDecisionTypedFields:
    def test_unavailable_vs_refusal_are_distinguishable(self, tmp_path: Path) -> None:
        outage = unavailable_adjudicator(refusal=False)
        pipeline_outage = _make_pipeline(adjudicator=outage)
        state_outage = _make_state(_ruled_pdf(tmp_path / "outage"))
        pipeline_outage._run_table_judge_gate(
            state_outage, 1, state_outage.pages[1], _bo(), _two_low_pass_rungs()
        )
        event_outage = _unverified_event(state_outage)

        refusal = unavailable_adjudicator(refusal=True)
        pipeline_refusal = _make_pipeline(adjudicator=refusal)
        state_refusal = _make_state(_ruled_pdf(tmp_path / "refusal"))
        pipeline_refusal._run_table_judge_gate(
            state_refusal, 1, state_refusal.pages[1], _bo(), _two_low_pass_rungs()
        )
        event_refusal = _unverified_event(state_refusal)

        assert event_outage.data["adjudicator_ran"] is True
        assert event_outage.data["guard_unavailable"] is True
        assert event_outage.data["guard_refusal"] is False

        assert event_refusal.data["adjudicator_ran"] is True
        assert event_refusal.data["guard_unavailable"] is True
        assert event_refusal.data["guard_refusal"] is True

        for ev in (event_outage, event_refusal):
            assert ev.data["guard_disposition"] == "not_cleared"
            assert ev.data["adjudicator_suppressed"] is False
            assert ev.data["requested_refs"]

    def test_configured_absent_vs_suppressed_are_distinguishable(self, tmp_path: Path) -> None:
        # -- never configured at all --
        pipeline_absent = _make_pipeline(adjudicator=None)
        state_absent = _make_state(_ruled_pdf(tmp_path / "absent"))
        pipeline_absent._run_table_judge_gate(
            state_absent, 1, state_absent.pages[1], _bo(), _two_low_pass_rungs()
        )
        event_absent = _unverified_event(state_absent)

        # -- configured, but refused earlier THIS run, so the gate suppresses it --
        adj = mismatching_adjudicator()
        pipeline_suppressed = _make_pipeline(adjudicator=adj)
        pipeline_suppressed._table_rung_refused_this_run.add("adjudicator")
        state_suppressed = _make_state(_ruled_pdf(tmp_path / "suppressed"))
        pipeline_suppressed._run_table_judge_gate(
            state_suppressed, 1, state_suppressed.pages[1], _bo(), _two_low_pass_rungs()
        )
        event_suppressed = _unverified_event(state_suppressed)

        assert event_absent.data["adjudicator_ran"] is False
        assert event_absent.data["adjudicator_suppressed"] is False
        assert event_absent.data["guard_unavailable"] is False

        assert event_suppressed.data["adjudicator_ran"] is False
        assert event_suppressed.data["adjudicator_suppressed"] is True
        assert event_suppressed.data["guard_unavailable"] is True
        assert adj.calls == [], "a suppressed adjudicator must never actually be called"


# ---------------------------------------------------------------------------
# Cold review round 3
# ---------------------------------------------------------------------------


class TestGuardCauseSelectedByMembership:
    def test_empty_guard_detail_still_keeps_guard_not_cleared(self, tmp_path: Path) -> None:
        """GH-581 cold review round 3, finding 1. A guard decision with an
        EMPTY ``detail`` (a valid ``BlindCellResult(unavailable=True,
        error="")``) must not fall through to the generic unknown cause --
        selection is by table MEMBERSHIP in ``guard_detail_by_table``, never
        by truthiness of the detail string itself."""
        empty = unavailable_adjudicator(error="")
        pipeline_empty = _make_pipeline(adjudicator=empty)
        state_empty = _make_state(_ruled_pdf(tmp_path / "empty"))
        pipeline_empty._run_table_judge_gate(
            state_empty, 1, state_empty.pages[1], _bo(), _two_low_pass_rungs()
        )
        event_empty = _unverified_event(state_empty)

        worded = unavailable_adjudicator(error="simulated outage")
        pipeline_worded = _make_pipeline(adjudicator=worded)
        state_worded = _make_state(_ruled_pdf(tmp_path / "worded"))
        pipeline_worded._run_table_judge_gate(
            state_worded, 1, state_worded.pages[1], _bo(), _two_low_pass_rungs()
        )
        event_worded = _unverified_event(state_worded)

        for ev in (event_empty, event_worded):
            assert ev.data["cause"] == CAUSE_GUARD_NOT_CLEARED
            assert ev.data["guard_unavailable"] is True
            assert ev.data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES


class TestMixedPageBucketPrecedence:
    def test_backfill_latch_matches_the_pages_own_retry_pending(self, tmp_path: Path) -> None:
        """GH-581 cold review round 3, finding 2. A mixed page: one table's
        rung is genuinely unavailable (sets the page latch through the
        normal per-table gate), a SECOND table on the same page reaches
        assemble with no ladder terminal at all (the completeness backfill).
        The backfill's own ``latched`` must equal the page's real
        ``table_judge_retry_pending`` -- not a hard-coded False -- and the
        page must appear in exactly ONE wording bucket (retryable wins over
        incomplete), never listed twice for the same underlying fact."""
        pipeline = _make_pipeline()
        state = _make_state(_ruled_pdf(tmp_path))
        ps = state.pages[1]
        ps.best_output = _bo()

        # First table: genuinely unavailable rung -> latches the page.
        rung = _QueueRung([RungResult(rung="fake1", ok=False, error="outage", unavailable=True)])
        pipeline._run_table_judge_gate(state, 1, ps, _bo(), [rung])
        assert ps.table_judge_retry_pending is True

        # Second table: reaches assemble with no terminal of its own.
        second_table_md = _TABLE_MD + "\nprose\n\n" + _TABLE_MD
        pipeline._backfill_missing_table_ladder_terminals(state, [second_table_md])

        backfill_event = next(
            e
            for e in _events_of_kind(state, TABLE_LADDER_UNVERIFIED_KIND)
            if e.data.get("cause") == CAUSE_MISSING_TABLE_TERMINAL
        )
        assert backfill_event.data["latched"] == ps.table_judge_retry_pending
        assert backfill_event.data["latched"] is True
        assert "the page itself is retryable on resume" in backfill_event.detail

        note = pipeline._table_judge_ladder_note(state)
        assert note is not None
        assert note.count("page(s) 1") == 1, (
            f"page 1 is latched AND incomplete for the same underlying fact -- it must "
            f"appear in exactly one sentence, not both: {note}"
        )
        assert "retryable on resume" in note
        assert "incomplete" not in note


# ---------------------------------------------------------------------------
# The literal acceptance invariant: for EVERY UNVERIFIED path,
# ("retryable" in detail) == data["latched"]. A negative claim must avoid
# the substring entirely, not merely negate it.
# ---------------------------------------------------------------------------


def _event_no_witness(tmp_path: Path) -> AuditEvent:
    pipeline = _make_pipeline()
    state = _make_state(_ruled_pdf(tmp_path))
    witnesses = [
        TableWitness(
            table_id="p1-t0",
            page_num=1,
            block_index=0,
            markdown=_TABLE_MD,
            status=WitnessStatus.MISSING,
            scope=WitnessScope.NONE,
        )
    ]
    with patch(
        "socr.tables.witness.prepare_table_witnesses",
        return_value=_StubWitnessCtx(witnesses),
    ):
        pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [_QueueRung([])])
    return _unverified_event(state)


def _event_rung_unavailable(tmp_path: Path) -> AuditEvent:
    pipeline = _make_pipeline()
    state = _make_state(_ruled_pdf(tmp_path))
    rung = _QueueRung([RungResult(rung="fake1", ok=False, error="down", unavailable=True)])
    pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [rung])
    return _unverified_event(state)


def _event_rung_not_accepted(tmp_path: Path) -> AuditEvent:
    pipeline = _make_pipeline()
    state = _make_state(_ruled_pdf(tmp_path))
    rung = _QueueRung([RungResult(rung="fake1", ok=False, error="malformed")])
    pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [rung])
    return _unverified_event(state)


def _event_binding_contradiction(tmp_path: Path) -> AuditEvent:
    adj = mismatching_adjudicator()
    pipeline = _make_pipeline(evidence=BindingEvidence.CONTRADICT, adjudicator=adj)
    state = _make_state(_ruled_pdf(tmp_path))
    rung = _QueueRung(
        [
            RungResult(
                rung="fake1",
                ok=True,
                verdict=TableJudgeVerdict(verdict="FAIL", confidence="high", findings=[]),
            )
        ]
    )
    pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [rung])
    return _unverified_event(state)


def _event_guard_not_cleared(tmp_path: Path) -> AuditEvent:
    pipeline = _make_pipeline(adjudicator=None)
    state = _make_state(_ruled_pdf(tmp_path))
    pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), _two_low_pass_rungs())
    return _unverified_event(state)


def _event_guard_error(tmp_path: Path) -> AuditEvent:
    pipeline = _make_pipeline(adjudicator=mismatching_adjudicator())
    state = _make_state(_ruled_pdf(tmp_path))
    with patch(
        "socr.pipeline.orchestrator.evaluate_cell_guard",
        side_effect=RuntimeError("guard chain blew up"),
    ):
        pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), _two_low_pass_rungs())
    return _unverified_event(state)


def _event_witness_preparation_error(tmp_path: Path) -> AuditEvent:
    pipeline = _make_pipeline()
    state = _make_state(_ruled_pdf(tmp_path))
    with patch(
        "socr.tables.witness.prepare_table_witnesses",
        side_effect=RuntimeError("boom"),
    ):
        pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [_QueueRung([])])
    return _unverified_event(state)


def _event_no_rungs_configured(tmp_path: Path) -> AuditEvent:
    pipeline = _make_pipeline()
    state = _make_state(_ruled_pdf(tmp_path))
    pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [])
    return _unverified_event(state)


def _event_insufficient_corroboration(tmp_path: Path) -> AuditEvent:
    pipeline = _make_pipeline()
    state = _make_state(_ruled_pdf(tmp_path))
    rung = _QueueRung(
        [
            RungResult(
                rung="fake1",
                ok=True,
                verdict=TableJudgeVerdict(verdict="PASS", confidence="low", findings=[], doubts=[]),
            )
        ]
    )
    pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [rung])
    return _unverified_event(state)


def _event_missing_table_terminal(tmp_path: Path) -> AuditEvent:
    pipeline = _make_pipeline()
    state = _make_state(_ruled_pdf(tmp_path))
    ps = state.pages[1]
    ps.best_output = _bo()
    pipeline._backfill_missing_table_ladder_terminals(state, [_TABLE_MD])
    return _unverified_event(state)


_UNVERIFIED_PATH_BUILDERS = {
    CAUSE_NO_WITNESS: _event_no_witness,
    CAUSE_RUNG_UNAVAILABLE: _event_rung_unavailable,
    CAUSE_RUNG_NOT_ACCEPTED: _event_rung_not_accepted,
    CAUSE_BINDING_CONTRADICTION: _event_binding_contradiction,
    CAUSE_GUARD_NOT_CLEARED: _event_guard_not_cleared,
    CAUSE_GUARD_ERROR: _event_guard_error,
    CAUSE_WITNESS_PREPARATION_ERROR: _event_witness_preparation_error,
    CAUSE_NO_RUNGS_CONFIGURED: _event_no_rungs_configured,
    CAUSE_INSUFFICIENT_CORROBORATION: _event_insufficient_corroboration,
    CAUSE_MISSING_TABLE_TERMINAL: _event_missing_table_terminal,
}


class TestRetryableSubstringMatchesLatchedForEveryPath:
    @pytest.mark.parametrize("expected_cause", sorted(_UNVERIFIED_PATH_BUILDERS))
    def test_retryable_substring_iff_latched(self, tmp_path: Path, expected_cause: str) -> None:
        event = _UNVERIFIED_PATH_BUILDERS[expected_cause](tmp_path)
        assert event.data["cause"] == expected_cause
        assert ("retryable" in event.detail) == event.data["latched"], (
            f"cause={expected_cause}: detail={event.detail!r} latched={event.data['latched']!r}"
        )


# ---------------------------------------------------------------------------
# Cold review round 4
# ---------------------------------------------------------------------------


class TestDetailNeverInterpolatesRawText:
    def test_identical_details_regardless_of_reserved_word_in_the_raw_error(
        self, tmp_path: Path
    ) -> None:
        """GH-581 cold review round 4, finding 1. Two otherwise identical
        fixtures differ ONLY in whether the raw rung error text happens to
        contain the reserved substring "retryable" -- an untrusted
        diagnostic must never be able to trip the literal invariant. Both
        events must produce the SAME fixed detail, and the raw text must
        show up only in structured data (``rung_trail[].error``), never in
        ``detail`` itself."""
        pipeline = _make_pipeline()

        state_benign = _make_state(_ruled_pdf(tmp_path / "benign"))
        rung_benign = _QueueRung(
            [RungResult(rung="fake1", ok=False, error="malformed verdict json")]
        )
        pipeline._run_table_judge_gate(state_benign, 1, state_benign.pages[1], _bo(), [rung_benign])
        event_benign = _unverified_event(state_benign)

        state_tricky = _make_state(_ruled_pdf(tmp_path / "tricky"))
        rung_tricky = _QueueRung(
            [RungResult(rung="fake1", ok=False, error="deterministic parser says retryable token")]
        )
        pipeline._run_table_judge_gate(state_tricky, 1, state_tricky.pages[1], _bo(), [rung_tricky])
        event_tricky = _unverified_event(state_tricky)

        assert event_benign.detail == event_tricky.detail, (
            "the raw error text leaked into the fixed detail phrase"
        )
        assert event_benign.data["latched"] is False
        assert event_tricky.data["latched"] is False
        assert ("retryable" in event_benign.detail) == event_benign.data["latched"]
        assert ("retryable" in event_tricky.detail) == event_tricky.data["latched"]
        # The raw text is preserved, but only in structured data.
        assert "retryable" not in event_tricky.detail
        assert event_tricky.data["rung_trail"][0]["error"] == (
            "deterministic parser says retryable token"
        )

    def test_witness_preparation_error_text_is_structured_only(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        state = _make_state(_ruled_pdf(tmp_path))
        with patch(
            "socr.tables.witness.prepare_table_witnesses",
            side_effect=RuntimeError("boom says retryable too"),
        ):
            pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), [_QueueRung([])])
        event = _unverified_event(state)
        assert "retryable" not in event.detail
        assert event.data["latched"] is False
        assert event.data["witness_error"] == "RuntimeError: boom says retryable too"


class TestGuardPrepFailureStaysInsideTheGuardChain:
    def test_ref_resolution_failure_is_guard_error_and_keeps_the_real_trail(
        self, tmp_path: Path
    ) -> None:
        """GH-581 cold review round 4, finding 2. A raise from
        ``resolve_cell_refs`` -- guard PREPARATION, not the adjudicator call
        itself -- must be caught by the guard chain's OWN fail-closed
        handler, not escape to the outer per-witness exception in
        ``_run_table_judge_gate`` (which fabricates a synthetic "unknown"
        rung and discards the real reader trail)."""
        pipeline = _make_pipeline(adjudicator=mismatching_adjudicator())
        state = _make_state(_ruled_pdf(tmp_path))
        with patch(
            "socr.judge.table_verdict.resolve_cell_refs",
            side_effect=RuntimeError("guard prep boom"),
        ):
            pipeline._run_table_judge_gate(state, 1, state.pages[1], _bo(), _two_low_pass_rungs())
        event = _unverified_event(state)
        assert event.data["cause"] == CAUSE_GUARD_ERROR
        assert event.data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES
        assert event.data["guard_detail"] == "guard chain error: RuntimeError"
        trail_rungs = {row["rung"] for row in event.data["rung_trail"]}
        assert trail_rungs == {"r1", "r2"}, (
            f"the real reader trail was discarded in favour of a fabricated rung: "
            f"{event.data['rung_trail']}"
        )
        assert "unknown" not in trail_rungs
        assert "retryable" not in event.detail
        assert event.data["latched"] is False


# ---------------------------------------------------------------------------
# Cold review round 5
# ---------------------------------------------------------------------------


class TestGuardPrepAvailabilityFailureStillLatches:
    def test_connection_error_latches_runtime_error_does_not(self, tmp_path: Path) -> None:
        """GH-581 cold review round 5, finding 1. Moving ref extraction/
        resolution inside the guard chain's own try (round 4, finding 2)
        must not silently change latch behaviour for an exception that IS
        an availability failure -- before that move, such an exception
        escaped to the outer per-witness handler, which classifies
        ``is_availability_exception`` and, when true, synthesizes an
        unavailable rung that latches the page. A ``ConnectionError`` from
        ``resolve_cell_refs`` must still latch; a plain ``RuntimeError``
        (a defect in this code, not a transport failure) must still not --
        the SAME paired difference GH-575 draws everywhere else."""
        pipeline = _make_pipeline(adjudicator=mismatching_adjudicator())

        state_conn = _make_state(_ruled_pdf(tmp_path / "conn"))
        with patch(
            "socr.judge.table_verdict.resolve_cell_refs",
            side_effect=ConnectionError("guard preparation transport-shaped failure"),
        ):
            pipeline._run_table_judge_gate(
                state_conn, 1, state_conn.pages[1], _bo(), _two_low_pass_rungs()
            )
        event_conn = _unverified_event(state_conn)

        state_runtime = _make_state(_ruled_pdf(tmp_path / "runtime"))
        with patch(
            "socr.judge.table_verdict.resolve_cell_refs",
            side_effect=RuntimeError("a defect, not an outage"),
        ):
            pipeline._run_table_judge_gate(
                state_runtime, 1, state_runtime.pages[1], _bo(), _two_low_pass_rungs()
            )
        event_runtime = _unverified_event(state_runtime)

        assert state_conn.pages[1].table_judge_retry_pending is True
        assert event_conn.data["latched"] is True
        assert event_conn.data["cause"] == CAUSE_GUARD_ERROR
        assert {row["rung"] for row in event_conn.data["rung_trail"]} == {"r1", "r2"}
        assert "retryable on resume" in event_conn.detail

        assert state_runtime.pages[1].table_judge_retry_pending is False
        assert event_runtime.data["latched"] is False
        assert event_runtime.data["cause"] == CAUSE_GUARD_ERROR
        assert {row["rung"] for row in event_runtime.data["rung_trail"]} == {"r1", "r2"}
        assert "retryable" not in event_runtime.detail


# ---------------------------------------------------------------------------
# Cold review round 6
# ---------------------------------------------------------------------------


def _legacy_unverified_event_dict(meta: dict) -> dict:
    return next(e for e in meta["audit_events"] if e["kind"] == TABLE_LADDER_UNVERIFIED_KIND)


class TestLegacySidecarEventsAreNormalizedOnRestore:
    def test_pre_581_sidecar_no_longer_replays_the_false_retry_promise(
        self, tmp_path: Path
    ) -> None:
        """GH-581 cold review round 6, finding 1. A sidecar written by a
        pre-GH-581 build has no ``latched``/``cause``/``guard_detail`` at
        all, and its raw ``detail`` may still say "retryable on resume"
        unconditionally. A mixed page -- a WITHHELD content-terminal sibling
        plus this deterministic UNVERIFIED table -- is intentionally
        resumable, so this legacy shape must be normalized on restore, not
        replayed verbatim."""
        import json

        pdf = _ruled_pdf(tmp_path / "src")
        pipeline = _make_pipeline()
        state = _make_state(pdf)
        output = _bo()
        state.pages[1].attempts.append(output)
        state.pages[1].best_output = output
        rung = _QueueRung([RungResult(rung="fake1", ok=False, error="malformed")])
        pipeline._run_table_judge_gate(state, 1, state.pages[1], output, [rung])
        # A mixed page can have a content-terminal sibling (WITHHELD) and
        # this deterministic UNVERIFIED table; such pages are intentionally
        # resumable.
        state.pages[1].table_ladder_disposition = FailureMode.TABLE_WITHHELD

        out_dir = tmp_path / "out"
        pipeline._flush_page_fragment(state, 1, output.text, out_dir)
        sidecar = pipeline._flush_page_sidecar(state, 1, out_dir)
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        legacy = _legacy_unverified_event_dict(meta)
        legacy["detail"] = (
            "table p1-t0 unverified by the judge ladder (infra problem, retryable on resume)"
        )
        legacy["data"].pop("latched", None)
        legacy["data"].pop("cause", None)
        legacy["data"].pop("guard_detail", None)
        sidecar.write_text(json.dumps(meta), encoding="utf-8")

        resumed = _make_state(pdf)
        resumed.pages[1].attempts.clear()
        resumed.pages[1].best_output = None
        restored_output = pipeline._load_terminal_page(resumed, 1, out_dir)
        assert restored_output is not None, "the normal resume gate did not admit the mixed page"
        pipeline._restore_terminal_page_state(resumed, 1, restored_output, out_dir)
        event = next(e for e in resumed.events if e.kind == TABLE_LADDER_UNVERIFIED_KIND)

        page_latch = resumed.pages[1].table_judge_retry_pending
        assert page_latch is False, "this table's rung answered but was not accepted -- no latch"
        assert ("retryable" in event.detail) == page_latch
        assert event.data["latched"] is page_latch
        assert event.data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES
        # GH-581 cold review round 7, finding 2: this row is NOT a true
        # pre-581 shape -- ``_run_table_judge_gate`` already writes explicit
        # ``unavailable: False``/``refusal: False`` on every trail row
        # (round 1's fix), so popping only ``latched``/``cause``/
        # ``guard_detail`` from the EVENT still leaves the ROW fully typed.
        # Both bits explicitly present and both False is itself a typed
        # fact -- it proves the rung answered but was not accepted.
        assert event.data["cause"] == CAUSE_RUNG_NOT_ACCEPTED, (
            "an ok=False rung with unavailable/refusal EXPLICITLY False proves "
            "rung_not_accepted, not the generic unknown fallback"
        )
        assert "guard_detail" in event.data
        assert "rung_trail" in event.data
        assert "witness_scope" in event.data
        # The gate's own admission facts are untouched by the normalizer.
        assert resumed.pages[1].table_ladder_disposition == FailureMode.TABLE_WITHHELD

    def test_true_pre_581_row_shape_with_no_typed_bits_is_unknown(self) -> None:
        """GH-581 cold review round 7, finding 2's control: a row shape that
        genuinely predates even the ``unavailable``/``refusal`` keys (only
        ``rung``/``ok``/``executing``, as ``_run_table_judge_gate`` wrote
        before round 1) proves nothing and must stay ``CAUSE_UNKNOWN``."""
        from socr.pipeline.orchestrator import UnifiedPipeline

        legacy_data = {
            "table_id": "p1-t0",
            "rung_trail": [{"rung": "fake1", "ok": False, "executing": "fake1"}],
        }
        detail, data = UnifiedPipeline._normalize_legacy_unverified_event(
            "table p1-t0 unverified by the judge ladder (infra problem, retryable on resume)",
            legacy_data,
            False,
        )
        assert data["cause"] == CAUSE_UNKNOWN
        assert "retryable" not in detail

    def test_legacy_trail_with_unavailable_rung_proves_rung_unavailable(self) -> None:
        """A legacy trail entry that DOES carry a typed ``unavailable: true``
        proves a specific cause -- unlike the ambiguous default-fallback
        case above. Unit-level: a latched page is correctly NOT admitted by
        the resume gate at all (retry latches exist precisely so resume
        skips them), so this exercises the normalizer directly rather than
        the full gate-admission path, which the first test in this class
        already covers for the (resumable, unlatched) case."""
        from socr.pipeline.orchestrator import UnifiedPipeline

        legacy_data = {
            "table_id": "p1-t0",
            "rung_trail": [
                {
                    "rung": "fake1",
                    "ok": False,
                    "executing": "fake1",
                    "error": "down",
                    "unavailable": True,
                }
            ],
        }
        detail, data = UnifiedPipeline._normalize_legacy_unverified_event(
            "table p1-t0 unverified by the judge ladder (infra problem, retryable on resume)",
            legacy_data,
            True,
        )
        assert "retryable" in detail
        assert data["latched"] is True
        assert data["cause"] == CAUSE_RUNG_UNAVAILABLE
        assert data["cause"] in TABLE_LADDER_UNVERIFIED_CAUSES
        assert "guard_detail" in data
        assert "witness_scope" in data

        # Paired difference: the SAME trail, but the caller's restored latch
        # is False (a sibling table's rung recovered by resume time, or this
        # was never actually the source of the page's own latch) -- the
        # detail must track that, never the legacy raw text.
        detail_unlatched, data_unlatched = UnifiedPipeline._normalize_legacy_unverified_event(
            "table p1-t0 unverified by the judge ladder (infra problem, retryable on resume)",
            legacy_data,
            False,
        )
        assert "retryable" not in detail_unlatched
        assert data_unlatched["latched"] is False

    def test_legacy_refusal_bit_also_proves_rung_unavailable(self) -> None:
        """GH-581 cold review round 7, finding 2: ``refusal: true`` is exactly
        as retry-latching as ``unavailable: true`` (the same rule the
        fresh-emission classifier uses) and must prove the same cause."""
        from socr.pipeline.orchestrator import UnifiedPipeline

        legacy_data = {
            "table_id": "p1-t0",
            "rung_trail": [
                {
                    "rung": "fake1",
                    "ok": False,
                    "executing": "fake1",
                    "error": "quota exceeded",
                    "unavailable": False,
                    "refusal": True,
                }
            ],
        }
        _detail, data = UnifiedPipeline._normalize_legacy_unverified_event(
            "table p1-t0 unverified by the judge ladder (infra problem, retryable on resume)",
            legacy_data,
            True,
        )
        assert data["cause"] == CAUSE_RUNG_UNAVAILABLE
