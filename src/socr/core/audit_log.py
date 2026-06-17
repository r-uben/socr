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
    # top-to-bottom (engine escalation -> judge -> dual-pass -> D3 floor).
    # ``table_region_unverifiable`` is TR-3's distinct event: a born-digital
    # table region that hard-failed per-region geometry verification and was
    # routed to the image-asset lane (failed-table marker + PNG ref) rather
    # than shipping as a plausible-but-wrong collapsed or ragged table.
    rank = {
        "recitation_escalation": 0,
        "escalation": 1,
        "judge_reject": 2,
        "dualpass_patch": 3,
        "dualpass_flag": 3,
        "table_region_unverifiable": 4,
        "native_fallback": 5,
        "page_failed": 6,
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
