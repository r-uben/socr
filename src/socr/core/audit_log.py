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
        # S1 (#269): same disposition rank as the two above -- a structure-class
        # page's winner is a kept-but-unverified model grid or a native-prose
        # fallback, either way the last word on this page's table content.
        "structure_class_model_table_kept": 6,
        "structure_class_native_fallback": 6,
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
