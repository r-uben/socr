"""Durable per-run audit log of *interesting* OCR events.

The pipeline takes several quality interventions that, until now, surfaced only to
the console and (opt-in) the replay manifest: a Gemini RECITATION refusal escalated
to an open model, a hard-page judge rejecting a page, a dual-pass table patch. A
batch run over a real corpus is uninspectable without a persistent record of these
- you cannot see *what* was patched, or whether the judge fired noisily.

This module collects those events into ``audit_log.json`` next to the output, every
run. Two sources, merged:

1. **Appended at the source** (``DocumentState.events``) - the semantic verdicts
   that carry rich detail the state can't reconstruct: judge rejections (the
   issues the model named) and dual-pass reconciliations (the exact changed cells).
2. **Derived from attempts** - escalations are unambiguous ``FailureMode`` enums on
   the per-page attempt list, so they are recomputed here rather than threaded
   through every phase. RECITATION recovery is the headline case.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from socr.core.result import FailureMode

# Failure modes that mean "this attempt was refused/garbled and a later engine had
# to take over" - the escalations worth recording. RECITATION (Gemini copyright
# filter) is the one the owner most needs visibility into; the rest round out the
# picture of where the cheap engines fell down.
_ESCALATION_MODES = {
    FailureMode.RECITATION,
    FailureMode.HALLUCINATION,
    FailureMode.TRUNCATED,
    FailureMode.GARBAGE,
    FailureMode.REFUSAL,
    FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
}


#: GH-519: the chart-asset lane's debt, said outright and addressable.
#:
#: The lane preserves a figure page's native prose and its page PNG, and the
#: existing ``chart_asset_page`` event says in prose that data values are not
#: transcribed. A figure with in-image text -- axis labels, a legend, an
#: embedded table -- ships that text nowhere in the markdown, and nothing
#: counted how often that happened. ``docs/log/2026-09-02_p4-structure-lane-
#: design.md`` section 7 (Q3) ruled that figures are done for preservation, not
#: for machine-readable extraction, and that the debt must be VISIBLE rather
#: than buried.
#:
#: A kind of its own, rather than more prose inside ``chart_asset_page``,
#: because a consumer counting the debt should not have to parse a sentence.
VISUAL_VALUES_NOT_TRANSCRIBED_KIND = "visual_values_not_transcribed"


@dataclass
class AuditEvent:
    """One notable event during a run.

    ``kind`` is a stable machine token (``recitation_escalation``,
    ``judge_reject``, ``dualpass_patch``, ``dualpass_flag``, ``escalation``).
    ``detail`` is a human one-liner; ``data`` holds structured extras (changed
    cells, named issues) for batch analysis.
    """

    page_num: int
    kind: str
    engine: str = ""
    detail: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunAudit:
    pdf_filename: str
    events: list[AuditEvent] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.events:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "pdf_filename": self.pdf_filename,
            "event_count": len(self.events),
            "counts": self.counts(),
            "events": [e.to_dict() for e in self.events],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def summary_line(self) -> str:
        """One-line console summary, or '' when nothing notable happened."""
        if not self.events:
            return ""
        parts = [f"{n} {kind}" for kind, n in sorted(self.counts().items())]
        return ", ".join(parts)


def build_run_audit(state) -> RunAudit:
    """Derive a ``RunAudit`` from a completed ``DocumentState``.

    Merges events appended at the source (``state.events``) with escalations
    recomputed from each page's attempt list, then orders by page then phase.
    """
    events: list[AuditEvent] = list(getattr(state, "events", []))
    events.extend(_derive_escalations(state))
    events.extend(_derive_structure_floor_overrides(state))

    # Stable order: by page, then a coarse phase rank so a page's story reads
    # top-to-bottom (engine escalation -> judge -> dual-pass -> TR-3 detection
    # -> D3 floor).
    #
    # The two TR-3 kinds are deliberately separate, and the distinction is what
    # a consumer of ``tables_trust.json`` reads to tell detection from
    # disposition:
    #   - ``table_region_geometry_hard_fail`` (GH-205) is emitted at ANALYZE
    #     time on every page whose native table region hard-failed per-region
    #     geometry verification. It records the detection and nothing else: no
    #     page status, no document status and no routing is keyed on it, so the
    #     page's native text may still ship as the winner.
    #   - ``table_region_unverifiable`` is the D3 fail-closed DISPOSITION: the
    #     same geometry hard-fail, but the OCR ladder also failed, so the region
    #     was routed to the image-asset lane (failed-table marker + PNG ref)
    #     rather than shipping a plausible-but-wrong collapsed or ragged table.
    # A D3 page therefore carries both, in that order, and the pair reads as
    # "detected here, acted on there" rather than as one kind counted twice.
    rank = {
        "recitation_escalation": 0,
        "escalation": 1,
        "judge_reject": 2,
        "dualpass_patch": 3,
        "dualpass_flag": 3,
        "table_value_drift_unadjudicated": 4,
        "table_region_geometry_hard_fail": 4,
        "chart_math_arbitration": 4,
        # GH-271: region evidence/candidate first, final shipped-warning
        # disposition second.
        "corrupt_math_region_recovery": 4,
        # #263: analyze-time detection AND disposition in one -- unlike the
        # TR-3 pair above there is no second kind, because this flag always
        # acts (the page is routed to OCR and refuses to ship its fragments).
        "rotated_text_shredded": 4,
        "table_region_unverifiable": 5,
        # #262: the D3 disposition's sibling -- same phase, opposite outcome
        # (a model grid superseded the floor). Ranked explicitly rather than
        # falling into the default 9, so a page's story still reads in order.
        "d3_floor_model_table_kept": 5,
        "flagged_model_table_kept": 6,
        "native_fallback": 6,
        "corrupt_math_hybrid_shipped": 6,
        # S1/P2 (#269/#317): same disposition rank as the two above -- a
        # structure-class page's winner is a kept-but-unverified model grid or
        # a fail-closed floor, either way the last word on its table content.
        "structure_class_model_table_kept": 6,
        "structure_class_ladder_exhausted_floor": 6,
        # TICKET-A1b (#634): same phase/rank as the two lines above -- one
        # more shape of "the structure-class winner-selection story for this
        # page", either the row-corroborated candidate that shipped or the
        # ladder-accepted candidate the floor overrode anyway.
        "structure_class_row_corroborated": 6,
        "structure_floor_overrode_ladder": 6,
        "page_failed": 7,
    }
    events.sort(key=lambda e: (e.page_num, rank.get(e.kind, 9)))
    return RunAudit(pdf_filename=state.handle.filename, events=events)


def _derive_escalations(state) -> list[AuditEvent]:
    """One event per page attempt that failed in a way that forced a handoff.

    An attempt counts as an escalation only if a *later* attempt exists for the
    same page (something took over). RECITATION is called out by its own kind so
    it is trivially greppable across a batch.
    """
    out: list[AuditEvent] = []
    for page_num in sorted(state.pages):
        attempts = state.pages[page_num].attempts
        for i, a in enumerate(attempts):
            if a.failure_mode not in _ESCALATION_MODES:
                continue
            # GH-34: only count a successor as a real recovery if it produced
            # usable text.  An empty successor is not a recovery — it should
            # never generate a misleading recovered_by event.
            next_attempt = attempts[i + 1] if i + 1 < len(attempts) else None
            if not next_attempt or not (next_attempt.text and next_attempt.text.strip()):
                continue  # nothing usable recovered it; not an escalation
            took_over = next_attempt.engine
            kind = (
                "recitation_escalation"
                if a.failure_mode == FailureMode.RECITATION
                else "escalation"
            )
            detail = f"{a.engine} {a.failure_mode.value} -> {took_over}"
            out.append(
                AuditEvent(
                    page_num=page_num,
                    kind=kind,
                    engine=a.engine,
                    detail=detail,
                    data={"failure_mode": a.failure_mode.value, "recovered_by": took_over},
                )
            )
    return out


def _derive_structure_floor_overrides(state) -> list[AuditEvent]:
    """TICKET-A1b (#634): generalises #589 (option c).

    A per-table judge-ladder ACCEPT (``TABLE_LADDER_ACCEPTED_KIND``, appended
    at the source in ``state.events`` from ``_phase_agentic``) is a verdict
    about that table's own content, entirely independent of S1's
    grid-authorship/corroboration selection. A page can therefore end up
    with an accepted table AND still have its structure-class floor apply
    (``manifest.structure_class_floor_applies`` -- true only when
    ``structure_class_grid_winner`` itself, corroboration fallback included,
    found nothing to ship): the accepted table was never in the grid-shaped
    candidate pool at all (an OCR rung the ladder scored on its OWN text
    reconstruction, not on the grid-authored markdown attempts S1 chooses
    between), or corroboration's stricter ordered-row check still rejected
    it. Derived here (not appended at the selection site in ``manifest.py``)
    for the same reason ``_derive_escalations`` is derived rather than
    threaded through every phase: the two facts it compares -- "ladder
    accepted a table here" and "the floor still applies here" -- are each
    already unambiguous and available on ``state`` without a new mutation
    point, so recomputing them here cannot drift from either source of
    truth.
    """
    from socr.core.manifest import structure_class_floor_applies
    from socr.judge.table_verdict import TABLE_LADDER_ACCEPTED_KIND

    accepted_pages: set[int] = {
        e.page_num
        for e in getattr(state, "events", []) or []
        if getattr(e, "kind", "") == TABLE_LADDER_ACCEPTED_KIND
    }
    out: list[AuditEvent] = []
    for page_num in sorted(accepted_pages):
        p = state.pages.get(page_num)
        if p is None or not structure_class_floor_applies(p):
            continue
        out.append(
            AuditEvent(
                page_num=page_num,
                kind="structure_floor_overrode_ladder",
                detail=(
                    "judge ladder accepted a table on this page, but the "
                    "structure-class floor still discarded every candidate "
                    "and shipped the fail-closed marker instead"
                ),
            )
        )
    return out
