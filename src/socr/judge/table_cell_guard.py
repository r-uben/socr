"""P1 (owner rulings Q1/Q2): the two-guard chain, as ONE service.

Both ruled paths call this and nothing else:

* **Q1** — the ladder ended with two low-confidence PASSes. Formerly a
  quorum, now a pending decision.
* **Q2** — the ladder ended REJECTED. Before a table is withheld, the same
  two guards run in the same order, because both readers can be wrong about
  an unusual-but-correct table.

Order, cheapest first, exactly as ruled:

1. **Native geometry** (free, local, no model). ``BindingEvidence.PASS``
   means rows AND columns were fully checked and nothing disagreed, so the
   readers are overruled and the table is verified. A matching numeric
   multiset alone never reaches PASS -- matching numbers prove "not
   invented", never "correctly placed", which is the whole GH-273 shape.

   ``CONTRADICT`` is TERMINAL on BOTH paths (GH-575; cold review round 1,
   finding 1). An ACTIVE binding contradiction ends the table UNVERIFIED and
   the adjudicator does not run. The earlier build let the REJECTED path
   continue past a contradiction, which produced the exact inversion GH-575
   forbids: native geometry actively disagrees, the adjudicator happens to
   guess the same token, and a table nothing corroborated ships as
   "verified by blind cell transcription". The adjudicator therefore
   receives only binding-ABSTAIN cases.

2. **Blind cell transcription** (a third vendor, one call). It is handed the
   crop and the doubted cell references, nothing else, and the caller
   compares its tokens to the extraction with ``tokens_agree`` -- this
   repo's single equality rule, not a second one invented here. EVERY
   requested cell must resolve and EVERY resolved cell must agree. Anything
   less does not clear: a partial agreement is not evidence about the cells
   nobody could check.

   The answer has THREE states, not two (cold review round 2, N2). A token
   is a reading; the EMPTY token is also a reading ("I looked, the cell is
   blank"); and a cell the model reports as unreadable is a NON-reading. A
   non-reading can neither clear the table nor condemn it, so it ends the
   chain NOT_CLEARED before any comparison happens. Collapsing the last two
   into one empty string -- which the prompt used to ask for -- let a table
   nobody looked at be withheld against a non-empty extraction, or cleared
   against an empty one.

Cost, in one place and before the call: an enabled per-page cap must cover
the page's CURRENT spend plus this call's rate, and an enabled document
budget treats an unknown prior total as ZERO remaining (an unmetered lane
must never silently buy free adjudicator calls). A budget refusal makes no
call and NEVER sets the availability latch -- it reproduces identically on
every rerun, so latching it would make the document permanently unskippable
and change nothing.

Metering: exactly one ``EngineResult`` per executed call, handed once to
``DocumentState.record_engine_run(..., page_nums=[page_num])``, which is
itself the one place that both journals the run and charges the page. It is
NOT also passed to ``_add_page_cost``: that would double the page's spend
and make a resumed run's arithmetic disagree with a live one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from socr.core.config import PipelineConfig
from socr.core.state import DocumentState
from socr.judge.table_rung_ollama import BlindCellResult, adjudicator_rung_id
from socr.tables.adjudication import tokens_agree
from socr.tables.binding import BindingEvidence

logger = logging.getLogger(__name__)

#: The engine name the adjudicator's metered run is journalled under. Named
#: for the ROLE, not the vendor, so a later vendor swap does not rewrite
#: every historical journal entry's meaning.
ENGINE_NAME = "table_blind_cell_adjudicator"


class GuardDisposition(str, Enum):
    """What the chain established. Closed set; every value is fail-closed
    except the two that explicitly clear the table."""

    #: Geometry proved rows AND columns. The readers are overruled.
    VERIFIED_BY_GEOMETRY = "verified_by_geometry"
    #: Every doubted cell was transcribed blind and agreed with the
    #: extraction. The readers are overruled.
    VERIFIED_BY_BLIND_CELL_TRANSCRIPTION = "verified_by_blind_cell_transcription"
    #: Native geometry ACTIVELY disagreed with the emitted table. Terminal on
    #: both paths, and the adjudicator never ran (GH-575).
    CONTRADICTED = "contradicted"
    #: The adjudicator DID look and its blind reading of at least one doubted
    #: cell disagreed with the extraction. This is the only non-clearing
    #: disposition that is positive evidence AGAINST the table, and the only
    #: one that may withhold its bytes on the Q2 path (GH-575).
    MISMATCHED = "blind_cell_mismatch"
    #: Everything else: no refs, an unresolved ref, no adjudicator, a
    #: deterministic defect, an outage, or a budget refusal. Nobody
    #: established anything, so the table is merely unverified.
    NOT_CLEARED = "not_cleared"


@dataclass(frozen=True)
class CellGuardDecision:
    """The chain's answer plus the evidence it rests on.

    ``unavailable`` / ``refusal`` are the LATCH bits and mean exactly what
    they mean on a rung result: the adjudicator could not be reached or was
    externally refused, so the page is worth retrying on a later run. A
    deterministic defect and a budget refusal both leave them False.
    """

    geometry_evidence: BindingEvidence
    adjudicator_ran: bool
    disposition: GuardDisposition
    unavailable: bool = False
    refusal: bool = False
    #: Free-text reason for the audit trail (budget refusal, mismatch,
    #: unresolved refs). Never load-bearing for a decision.
    detail: str = ""
    #: The adjudicator's raw result, kept for the audit trail when it ran.
    blind_result: BlindCellResult | None = field(default=None, compare=False)

    @property
    def cleared(self) -> bool:
        return self.disposition in (
            GuardDisposition.VERIFIED_BY_GEOMETRY,
            GuardDisposition.VERIFIED_BY_BLIND_CELL_TRANSCRIPTION,
        )


#: The adjudicator's call shape: crop + canonical refs -> tokens.
CellAdjudicator = Callable[[Path, Sequence[str]], BlindCellResult]


def _remaining_document_budget(state: DocumentState, config: PipelineConfig) -> float | None:
    """Budget still available to this document, or None when uncapped.

    The same fail-closed rule the equation lane applies: an unmetered earlier
    call makes the remaining budget unknowable, and an unknown subtotal is
    treated as no budget, never as zero spend.
    """
    if config.cost_budget <= 0:
        return None
    total_cost = state.total_cost
    if total_cost is None:
        return 0.0
    return max(config.cost_budget - total_cost, 0.0)


def _budget_refusal(
    state: DocumentState, page_num: int, config: PipelineConfig, call_cost: float
) -> str:
    """Why this call must not be made, or "" when it may be. Checked FIRST."""
    cap = config.max_cost_per_page
    if cap > 0:
        page_spend = getattr(state.pages.get(page_num), "page_cost_usd", 0.0)
        if page_spend is None:
            # Unmetered page spend is the same unknown as an unmetered total:
            # it cannot be shown to fit under the cap, so it does not.
            return "page spend is unmetered; the per-page cap cannot be shown to cover this call"
        if page_spend + call_cost > cap:
            return (
                f"blind-cell adjudication skipped: page spend {page_spend} plus "
                f"{call_cost} exceeds --max-cost-per-page ({cap})"
            )
    remaining = _remaining_document_budget(state, config)
    if remaining is not None and call_cost > remaining:
        return (
            f"blind-cell adjudication skipped: {call_cost} exceeds the remaining "
            f"document budget ({remaining})"
        )
    return ""


def _record_call(
    state: DocumentState,
    page_num: int,
    config: PipelineConfig,
    call_cost: float,
    adjudicator: "CellAdjudicator",
) -> None:
    """Journal one executed adjudicator call, exactly once.

    ``model_version`` is the EXECUTING identity the callable advertises, not a
    module default (cold review round 1, finding 5). A configured non-default
    model changes the run fingerprint either way; recording the default
    alongside it would leave a journal that is precisely and confidently wrong
    about which model produced the clearance.

    ``record_engine_run`` is the sole journal author AND the sole page-spend
    charger, so this is one call and not two. The cost is always a KNOWN
    float, including a known zero -- ``None`` would mean "unmetered" and
    would poison every later budget decision for this document.
    """
    from socr.core.result import EngineResult

    state.record_engine_run(
        EngineResult(
            document_path=state.handle.path,
            engine=ENGINE_NAME,
            model_version=getattr(
                adjudicator,
                "rung_id",
                adjudicator_rung_id(config.table_judge_adjudicator_model),
            ),
            cost=float(call_cost),
        ),
        page_nums=[page_num],
    )


def evaluate_cell_guard(
    *,
    state: DocumentState,
    page_num: int,
    crop_path: Path | None,
    extraction_tokens: dict[str, str],
    requested_refs: Sequence[str],
    geometry_evidence: BindingEvidence,
    adjudicator: CellAdjudicator | None,
    config: PipelineConfig,
    adjudicator_suppressed: bool = False,
) -> CellGuardDecision:
    """Run the ruled two-guard chain for ONE table. Never raises.

    ``extraction_tokens`` maps each canonical cell reference to the token the
    EXTRACTION holds for it -- already resolved by the caller through
    ``resolve_cell_refs``, which fails the whole set closed if any reference
    is malformed, missing, out of range or non-unique. A reference absent
    from this mapping is therefore an unresolved reference, and the chain
    does not clear a table on the cells it merely happened to resolve.

    There is no per-caller contradiction switch (GH-575, cold review round 1,
    finding 1): an ACTIVE contradiction is terminal for both callers and the
    adjudicator is never asked. The Q1/Q2 difference lives entirely in what
    the CALLER does with a non-clearing disposition, which is the right place
    for it -- only the caller knows whether the readers rejected the table.
    """
    if geometry_evidence is BindingEvidence.PASS:
        return CellGuardDecision(
            geometry_evidence=geometry_evidence,
            adjudicator_ran=False,
            disposition=GuardDisposition.VERIFIED_BY_GEOMETRY,
            detail="native geometry checked rows and columns and found no disagreement",
        )

    if geometry_evidence is BindingEvidence.CONTRADICT:
        return CellGuardDecision(
            geometry_evidence=geometry_evidence,
            adjudicator_ran=False,
            disposition=GuardDisposition.CONTRADICTED,
            detail="native geometry contradicts the emitted table",
        )

    refs = [str(ref) for ref in requested_refs]
    if not refs:
        return CellGuardDecision(
            geometry_evidence=geometry_evidence,
            adjudicator_ran=False,
            disposition=GuardDisposition.NOT_CLEARED,
            detail="no cell-localizable doubt to check",
        )
    unresolved = [ref for ref in refs if ref not in extraction_tokens]
    if unresolved:
        return CellGuardDecision(
            geometry_evidence=geometry_evidence,
            adjudicator_ran=False,
            disposition=GuardDisposition.NOT_CLEARED,
            detail=f"cell references did not resolve against the emitted table: {unresolved}",
        )
    if adjudicator is None:
        # Cold review round 2, N3. ``adjudicator_suppressed`` means the caller
        # WOULD have had an adjudicator but withheld it because it already
        # refused us or was unreachable this run. That is an outage, and it
        # latches -- but only HERE, at the one point where the chain has
        # established that this table actually needed the adjudicator: an
        # ABSTAIN geometry verdict with a resolvable, non-empty doubt set. A
        # table cleared or condemned by geometry, or one with nothing to ask
        # about, has already returned above and never reaches this line, so it
        # cannot be latched for a call it was never going to make.
        return CellGuardDecision(
            geometry_evidence=geometry_evidence,
            adjudicator_ran=False,
            disposition=GuardDisposition.NOT_CLEARED,
            unavailable=bool(adjudicator_suppressed),
            detail=(
                "the blind-cell adjudicator was unavailable earlier in this run"
                if adjudicator_suppressed
                else "no blind-cell adjudicator configured"
            ),
        )

    call_cost = float(config.table_judge_adjudicator_cost_per_call_usd)
    refusal_reason = _budget_refusal(state, page_num, config, call_cost)
    if refusal_reason:
        # Deliberately NOT an outage: a cap or budget is settled by
        # configuration and reproduces identically on every rerun.
        return CellGuardDecision(
            geometry_evidence=geometry_evidence,
            adjudicator_ran=False,
            disposition=GuardDisposition.NOT_CLEARED,
            detail=refusal_reason,
        )

    try:
        result = adjudicator(crop_path if crop_path is None else Path(crop_path), refs)
    except Exception as exc:
        from socr.judge.table_verdict import is_availability_exception

        logger.warning(
            "blind-cell adjudicator raised %s: %s", type(exc).__name__, exc, exc_info=True
        )
        result = BlindCellResult(
            rung=getattr(
                adjudicator,
                "rung_id",
                adjudicator_rung_id(config.table_judge_adjudicator_model),
            ),
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            unavailable=is_availability_exception(exc),
        )
    # The call was ATTEMPTED, so it is metered whatever it answered -- an
    # outage still spends wall clock and, on a subscription, a request.
    _record_call(state, page_num, config, call_cost, adjudicator)

    if not result.ok:
        return CellGuardDecision(
            geometry_evidence=geometry_evidence,
            adjudicator_ran=True,
            disposition=GuardDisposition.NOT_CLEARED,
            unavailable=bool(result.unavailable),
            refusal=bool(result.refusal),
            detail=result.error,
            blind_result=result,
        )

    # Cold review round 2, N2. A cell the blind reader reported it could not
    # read is a NON-reading, and a non-reading is not evidence in either
    # direction: it cannot clear the table (nobody looked) and it cannot
    # condemn it (nobody read anything different). It is checked FIRST, so an
    # unreadable cell can never be silently compared against the extraction as
    # if the model had answered "blank".
    unreadable = [ref for ref in refs if ref in set(result.unreadable)]
    if unreadable:
        return CellGuardDecision(
            geometry_evidence=geometry_evidence,
            adjudicator_ran=True,
            disposition=GuardDisposition.NOT_CLEARED,
            detail=f"blind transcription reported these cells unreadable: {unreadable}",
            blind_result=result,
        )
    missing = [ref for ref in refs if ref not in result.tokens]
    if missing:
        # Belt and braces: a reading was neither given nor declared missing.
        # Treated as a non-reading for the same reason.
        return CellGuardDecision(
            geometry_evidence=geometry_evidence,
            adjudicator_ran=True,
            disposition=GuardDisposition.NOT_CLEARED,
            detail=f"blind transcription returned no reading for: {missing}",
            blind_result=result,
        )
    disagreed = [
        ref
        for ref in refs
        if not tokens_agree(result.tokens[ref], extraction_tokens[ref], kind="cell")
    ]
    if disagreed:
        # The one non-clearing outcome that is EVIDENCE AGAINST the table: a
        # blind reader looked at the crop and read something else. Only this
        # may withhold bytes on the Q2 path.
        return CellGuardDecision(
            geometry_evidence=geometry_evidence,
            adjudicator_ran=True,
            disposition=GuardDisposition.MISMATCHED,
            detail=f"blind transcription disagreed on {disagreed}",
            blind_result=result,
        )
    return CellGuardDecision(
        geometry_evidence=geometry_evidence,
        adjudicator_ran=True,
        disposition=GuardDisposition.VERIFIED_BY_BLIND_CELL_TRANSCRIPTION,
        detail=f"blind transcription agreed on every doubted cell ({len(refs)})",
        blind_result=result,
    )
