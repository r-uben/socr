"""P1 (task t6): the blind-cell guard service -- the single place Q1's
two-low tiebreak and Q2's withhold path both call.

Order (owner ruling, "Consequence for the flip build"):

1. Run geometry (``classify_binding_evidence``, t5). PASS overrules
   immediately, no Kimi call, no cost. CONTRADICT short-circuits ONLY for the
   Q1 (two-low) caller -- Q2's "otherwise adjudicator" wording means a
   REJECTED-path caller continues to blind transcription even on CONTRADICT.
   ABSTAIN always continues to blind transcription for both callers.
2. Kimi runs ONLY when every requested cell ref resolves (t1's resolver).
   All transcribed tokens agreeing with extraction (via
   ``socr.tables.adjudication.tokens_agree``) clears; anything else does not.
3. Metering: exactly one ``EngineResult`` per executed Kimi call, charged
   once through ``DocumentState.record_engine_run``, gated by the page cap
   and remaining document budget BEFORE the call is made. A budget refusal
   makes no call and does not set the availability latch.

This module has no PDF/CLI dependency of its own -- geometry evidence and the
Kimi adjudicator are both injected, so every test here is pure/hermetic.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from socr.core.config import PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.state import DocumentState
from socr.judge.table_cell_guard import (
    CellGuardDecision,
    GuardDisposition,
    evaluate_cell_guard,
)
from socr.judge.table_rung_ollama import BlindCellResult, adjudicator_rung_id
from socr.tables.binding import BindingEvidence

REFS = ["R1C2", "R2C3"]
EXTRACTION_TOKENS = {"R1C2": "100", "R2C3": "400"}


def _state(page_num: int = 1) -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=__import__("pathlib").Path("/tmp/doc.pdf"), page_count=1)
    state = DocumentState(handle=handle)
    return state


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig(**overrides)


def _matching_kimi(tokens: dict[str, str] | None = None):
    payload = tokens or EXTRACTION_TOKENS

    def _adjudicator(crop_path, refs):
        return BlindCellResult(
            rung=adjudicator_rung_id("test-adjudicator:cloud"), ok=True, tokens=dict(payload)
        )

    return _adjudicator


def _unavailable_kimi(refusal: bool = False):
    def _adjudicator(crop_path, refs):
        return BlindCellResult(
            rung=adjudicator_rung_id("test-adjudicator:cloud"),
            ok=False,
            error="simulated",
            unavailable=True,
            refusal=refusal,
        )

    return _adjudicator


def _defective_kimi():
    def _adjudicator(crop_path, refs):
        return BlindCellResult(
            rung=adjudicator_rung_id("test-adjudicator:cloud"), ok=False, error="malformed json"
        )

    return _adjudicator


def _never_called_kimi():
    def _adjudicator(crop_path, refs):
        raise AssertionError("kimi must not be called")

    return _adjudicator


# --------------------------------------------------------------------------
# Geometry short-circuit
# --------------------------------------------------------------------------


class TestGeometryShortCircuit:
    def test_geometry_pass_returns_immediately_no_kimi_call(self):
        state = _state()
        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.PASS,
            adjudicator=_never_called_kimi(),
            config=_config(),
        )
        assert isinstance(decision, CellGuardDecision)
        assert decision.disposition is GuardDisposition.VERIFIED_BY_GEOMETRY
        assert decision.adjudicator_ran is False
        assert state.total_cost in (0.0, None)

    def test_geometry_contradict_short_circuits_for_two_low_caller(self):
        """Q1's caller: geometry CONTRADICT means E1's own rule -- end UNVERIFIED
        without ever asking Kimi."""
        state = _state()
        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.CONTRADICT,
            adjudicator=_never_called_kimi(),
            config=_config(),
        )
        assert decision.disposition is GuardDisposition.CONTRADICTED
        assert decision.adjudicator_ran is False

    def test_geometry_contradict_never_reaches_the_adjudicator_for_either_caller(self):
        """GH-575 (cold review round 1, finding 1): an ACTIVE contradiction is
        terminal on BOTH paths and the adjudicator is not asked.

        The guard has no per-caller contradiction switch any more, so there is
        one arm to pin, and this asserts the strongest available fact: an
        adjudicator that would have CLEARED the table is never called, so a
        contradicted table cannot be published on a lucky blind token."""
        would_clear = _matching_kimi()
        decision = evaluate_cell_guard(
            state=_state(),
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.CONTRADICT,
            adjudicator=would_clear,
            config=_config(),
        )
        assert decision.adjudicator_ran is False
        assert decision.disposition is GuardDisposition.CONTRADICTED
        assert "continue_on_contradict" not in inspect.signature(evaluate_cell_guard).parameters

    def test_geometry_abstain_always_continues_to_kimi(self):
        state = _state()
        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_matching_kimi(),
            config=_config(),
        )
        assert decision.adjudicator_ran is True
        assert decision.disposition is GuardDisposition.VERIFIED_BY_BLIND_CELL_TRANSCRIPTION


# --------------------------------------------------------------------------
# Blind-cell clearance
# --------------------------------------------------------------------------


class TestBlindClearance:
    def test_all_tokens_match_clears(self):
        state = _state()
        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_matching_kimi(),
            config=_config(),
        )
        assert decision.disposition is GuardDisposition.VERIFIED_BY_BLIND_CELL_TRANSCRIPTION

    def test_one_mismatched_token_does_not_clear(self):
        state = _state()
        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_matching_kimi({"R1C2": "100", "R2C3": "999"}),
            config=_config(),
        )
        # GH-575: an ACTIVE blind mismatch is its own disposition, because it
        # is the only non-clearing outcome that is evidence AGAINST the table
        # and the only one that may withhold bytes.
        assert decision.disposition is GuardDisposition.MISMATCHED
        assert decision.cleared is False

    def test_empty_refs_does_not_clear(self):
        state = _state()
        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=[],
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_never_called_kimi(),
            config=_config(),
        )
        assert decision.disposition is GuardDisposition.NOT_CLEARED
        assert decision.adjudicator_ran is False

    def test_unresolvable_ref_does_not_clear(self):
        """A ref absent from ``extraction_tokens`` means the resolver (t1)
        failed and the whole set is unresolved."""
        state = _state()
        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens={"R1C2": "100"},  # R2C3 missing
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_never_called_kimi(),
            config=_config(),
        )
        assert decision.disposition is GuardDisposition.NOT_CLEARED
        assert decision.adjudicator_ran is False

    def test_kimi_defect_does_not_clear_and_does_not_latch(self):
        state = _state()
        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_defective_kimi(),
            config=_config(),
        )
        assert decision.disposition is GuardDisposition.NOT_CLEARED
        assert decision.unavailable is False

    def test_kimi_unavailable_does_not_clear_and_latches(self):
        state = _state()
        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_unavailable_kimi(),
            config=_config(),
        )
        assert decision.disposition is GuardDisposition.NOT_CLEARED
        assert decision.unavailable is True

    def test_kimi_refusal_latches_the_same_as_outage(self):
        state = _state()
        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_unavailable_kimi(refusal=True),
            config=_config(),
        )
        assert decision.unavailable is True
        assert decision.refusal is True


# --------------------------------------------------------------------------
# Cost / budget accounting
# --------------------------------------------------------------------------


class TestCostAndMetering:
    def test_executed_call_charges_the_page_exactly_once(self):
        state = _state()
        config = _config(table_judge_adjudicator_cost_per_call_usd=0.02)

        evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_matching_kimi(),
            config=config,
        )

        assert len(state.engine_runs) == 1
        assert state.total_cost == pytest.approx(0.02)

    def test_geometry_pass_never_charges_anything(self):
        state = _state()
        config = _config(table_judge_adjudicator_cost_per_call_usd=0.02)

        evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.PASS,
            adjudicator=_never_called_kimi(),
            config=config,
        )

        assert len(state.engine_runs) == 0

    def test_per_page_cap_covering_current_spend_plus_call_blocks_the_call(self):
        state = _state()
        state.pages[1].page_cost_usd = 0.09
        config = _config(table_judge_adjudicator_cost_per_call_usd=0.05, max_cost_per_page=0.10)

        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_never_called_kimi(),
            config=config,
        )

        assert decision.adjudicator_ran is False
        assert decision.disposition is GuardDisposition.NOT_CLEARED
        assert decision.unavailable is False  # budget refusal is not an outage/latch

    def test_per_page_cap_with_room_allows_the_call(self):
        state = _state()
        state.pages[1].page_cost_usd = 0.02
        config = _config(table_judge_adjudicator_cost_per_call_usd=0.05, max_cost_per_page=0.10)

        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_matching_kimi(),
            config=config,
        )

        assert decision.adjudicator_ran is True

    def test_document_budget_treats_unknown_prior_total_as_zero_remaining(self):
        """``DocumentState.total_cost`` is None whenever any prior run is
        unmetered -- the guard must not treat that as unlimited remaining
        budget, or an unmetered lane silently buys free adjudicator calls."""
        state = _state()
        from socr.core.result import DocumentStatus, EngineResult

        state.record_engine_run(
            EngineResult(document_path=state.handle.path, engine="qwen", cost=None),
            page_nums=[1],
        )
        config = _config(table_judge_adjudicator_cost_per_call_usd=0.01, cost_budget=1.0)

        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_never_called_kimi(),
            config=config,
        )

        assert decision.adjudicator_ran is False

    def test_budget_refusal_does_not_set_the_latch(self):
        state = _state()
        state.pages[1].page_cost_usd = 1.0
        config = _config(table_judge_adjudicator_cost_per_call_usd=0.05, max_cost_per_page=0.10)

        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_never_called_kimi(),
            config=config,
        )
        assert decision.unavailable is False

    def test_disabled_cap_zero_means_unlimited(self):
        state = _state()
        state.pages[1].page_cost_usd = 1000.0
        config = _config(table_judge_adjudicator_cost_per_call_usd=0.05, max_cost_per_page=0.0)

        decision = evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_matching_kimi(),
            config=config,
        )
        assert decision.adjudicator_ran is True

    def test_known_zero_cost_still_records_exactly_one_run(self):
        state = _state()
        config = _config(table_judge_adjudicator_cost_per_call_usd=0.0)

        evaluate_cell_guard(
            state=state,
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=_matching_kimi(),
            config=config,
        )
        assert len(state.engine_runs) == 1
        assert state.total_cost == 0.0


class TestUnreadableIsNeitherClearanceNorDisagreement:
    """Cold review round 2, N2.

    A cell the blind reader could not read is a NON-reading. It is not
    evidence in either direction, so it can neither clear the table nor
    condemn it -- and the empty token, which IS a reading, must keep doing
    both.
    """

    def _adjudicator(self, tokens, unreadable=()):
        def _adj(crop_path, refs):
            return BlindCellResult(
                rung=adjudicator_rung_id("test-adjudicator:cloud"),
                ok=True,
                tokens=dict(tokens),
                unreadable=tuple(unreadable),
            )

        return _adj

    def _decide(self, adjudicator, extraction):
        return evaluate_cell_guard(
            state=_state(),
            page_num=1,
            crop_path=None,
            extraction_tokens=extraction,
            requested_refs=list(extraction),
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=adjudicator,
            config=_config(),
        )

    def test_an_unreadable_cell_is_not_a_mismatch(self):
        """Nobody read a different value, so nothing may be withheld. This was
        MISMATCHED before, which withholds a rejected table's bytes on the
        strength of a reading that never happened."""
        decision = self._decide(self._adjudicator({}, unreadable=["R1C2"]), {"R1C2": "100"})
        assert decision.disposition is GuardDisposition.NOT_CLEARED
        assert decision.cleared is False
        assert "unreadable" in decision.detail

    def test_an_unreadable_cell_does_not_clear_an_empty_extraction(self):
        """The other half. An empty extraction cell plus 'I could not read it'
        used to normalize equal and CLEAR a rejected MISSING_VALUE case, with
        no visual evidence at all behind the clearance."""
        decision = self._decide(self._adjudicator({}, unreadable=["R1C2"]), {"R1C2": ""})
        assert decision.disposition is GuardDisposition.NOT_CLEARED

    def test_one_unreadable_cell_spoils_an_otherwise_agreeing_set(self):
        """Fail-closed, as everywhere else in this chain: a table is never
        cleared on the subset of cells that happened to be legible."""
        decision = self._decide(
            self._adjudicator({"R1C2": "100"}, unreadable=["R2C3"]), EXTRACTION_TOKENS
        )
        assert decision.disposition is GuardDisposition.NOT_CLEARED

    def test_an_unreadable_answer_never_latches(self):
        """It is a deterministic answer from a reachable adjudicator, not an
        outage: the call happened and would happen again identically."""
        decision = self._decide(self._adjudicator({}, unreadable=["R1C2"]), {"R1C2": "100"})
        assert decision.unavailable is False
        assert decision.refusal is False

    def test_a_visibly_empty_cell_still_clears_an_empty_extraction(self):
        """The positive control. ``""`` IS a reading -- the model looked and
        the cell was blank -- so it still corroborates an empty extraction."""
        decision = self._decide(self._adjudicator({"R1C2": ""}), {"R1C2": ""})
        assert decision.disposition is GuardDisposition.VERIFIED_BY_BLIND_CELL_TRANSCRIPTION

    def test_a_visibly_empty_cell_still_mismatches_a_non_empty_extraction(self):
        """The other positive control, and the reason this is not simply
        'treat empty as unknown': a blank cell read against a printed value is
        a real disagreement and must still be able to withhold."""
        decision = self._decide(self._adjudicator({"R1C2": ""}), {"R1C2": "North"})
        assert decision.disposition is GuardDisposition.MISMATCHED


class TestSuppressedAdjudicatorLatchesOnlyWhereItWasNeeded:
    """Cold review round 2, N3.

    When the per-run breaker has already tripped, the caller passes no
    adjudicator and says so. That is an outage and must latch -- but only for
    a table that would actually have called it. Geometry that settles the
    table, and a doubt set with nothing askable in it, both return before the
    adjudicator step and must leave the page unlatched, or every table on the
    page inherits a retry promise for a call it was never going to make.
    """

    def _decide(self, evidence, refs, extraction):
        return evaluate_cell_guard(
            state=_state(),
            page_num=1,
            crop_path=None,
            extraction_tokens=extraction,
            requested_refs=refs,
            geometry_evidence=evidence,
            adjudicator=None,
            config=_config(),
            adjudicator_suppressed=True,
        )

    def test_it_latches_when_the_chain_really_reached_the_adjudicator(self):
        decision = self._decide(BindingEvidence.ABSTAIN, REFS, EXTRACTION_TOKENS)
        assert decision.disposition is GuardDisposition.NOT_CLEARED
        assert decision.unavailable is True

    @pytest.mark.parametrize("evidence", [BindingEvidence.PASS, BindingEvidence.CONTRADICT])
    def test_geometry_that_settles_the_table_never_latches(self, evidence):
        decision = self._decide(evidence, REFS, EXTRACTION_TOKENS)
        assert decision.unavailable is False

    def test_an_empty_doubt_set_never_latches(self):
        decision = self._decide(BindingEvidence.ABSTAIN, [], EXTRACTION_TOKENS)
        assert decision.disposition is GuardDisposition.NOT_CLEARED
        assert decision.unavailable is False

    def test_an_unresolvable_doubt_set_never_latches(self):
        decision = self._decide(BindingEvidence.ABSTAIN, ["R9C9"], EXTRACTION_TOKENS)
        assert decision.unavailable is False

    def test_no_adjudicator_configured_at_all_still_never_latches(self):
        """The control that keeps the two causes apart: nothing was
        suppressed, so there is nothing transient to wait for."""
        decision = evaluate_cell_guard(
            state=_state(),
            page_num=1,
            crop_path=None,
            extraction_tokens=EXTRACTION_TOKENS,
            requested_refs=REFS,
            geometry_evidence=BindingEvidence.ABSTAIN,
            adjudicator=None,
            config=_config(),
        )
        assert decision.unavailable is False
