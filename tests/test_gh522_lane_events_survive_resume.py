"""GH-522 / cubic P1 on #537: a lane event nobody replays is a record with a
shelf life.

`_restore_terminal_page_state` replays a page's audit events from its terminal
sidecar, filtered through `EQUATION_LANE_EVENT_KINDS`. A kind missing from that
allowlist is silently dropped on resume -- so the record exists on the run that
produced it and nowhere afterwards.

That is not cosmetic for a REFUSAL record. GH-522 withholds an unverifiable
LaTeX reading and justifies it by keeping the crop: the event carries the
`crop_path` a human would follow to check the reading by hand. Drop the event on
resume and the evidence pointer goes with it, so the guarantee holds only until
the next run. `equation_region_reading_unverifiable` was missing exactly that
way.

Fixing the one entry is not the interesting part. This file requires every
`equation_*` kind the orchestrator EMITS to be either replayed or explicitly
listed as not replayed, with a reason -- so the next new kind forces the
decision instead of inheriting whichever answer nobody chose.
"""

from __future__ import annotations

import ast
from pathlib import Path

from socr.pipeline.orchestrator import UnifiedPipeline

_ORCHESTRATOR = (
    Path(__file__).resolve().parents[1] / "src" / "socr" / "pipeline" / "orchestrator.py"
)

#: Emitted, deliberately NOT replayed, with the reason. An entry here is a
#: decision; the absence of one is an oversight, which is the difference this
#: file exists to make visible.
_NOT_REPLAYED: dict[str, str] = {
    # Informational and per-region: it records that a region was found, which the
    # restored page's own outcome already implies. Not evidence of a decision.
    "equation_region_detected": "informational; the page's restored outcome implies it",
    # A refusal record, and therefore the same class as the GH-522 one this file
    # was written for -- filed rather than fixed here, because it belongs to the
    # sidecar guard's design and not to this ticket.
    "equation_sidecar_refused": "refusal record; replay status is GH-540, open",
}


def _emitted_equation_kinds() -> set[str]:
    """Every `kind="equation_*"` literal the orchestrator constructs.

    Read from the parse tree rather than by grep, so a mention in a comment,
    docstring or test name cannot be mistaken for an emit.
    """
    tree = ast.parse(_ORCHESTRATOR.read_text())
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "kind"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
                and keyword.value.value.startswith("equation_")
            ):
                kinds.add(keyword.value.value)
    return kinds


def test_every_emitted_lane_event_is_replayed_or_explicitly_not() -> None:
    """The guard. A new kind must be decided about, not defaulted."""
    emitted = _emitted_equation_kinds()
    assert emitted, "no equation events found; the AST scan is measuring nothing"

    undecided = sorted(emitted - UnifiedPipeline.EQUATION_LANE_EVENT_KINDS - set(_NOT_REPLAYED))
    assert not undecided, (
        f"equation event kind(s) neither replayed nor listed as deliberately "
        f"not replayed: {undecided}. A kind missing from "
        "EQUATION_LANE_EVENT_KINDS is dropped by _restore_terminal_page_state, "
        "so the record exists on the run that produced it and nowhere "
        "afterwards. Add it to the allowlist, or to _NOT_REPLAYED with the "
        "reason."
    )


def test_the_unverifiable_refusal_is_replayed() -> None:
    """The specific entry, named, because it carries the evidence pointer.

    Without it, GH-522's "the crop stays on disk as evidence" survives one run.
    """
    assert "equation_region_reading_unverifiable" in UnifiedPipeline.EQUATION_LANE_EVENT_KINDS, (
        "the GH-522 refusal record is dropped on resume, and the crop_path "
        "pointing at the withheld reading goes with it"
    )


def test_the_allowlist_has_no_entries_nothing_emits() -> None:
    """The other direction: a stale entry is a decision about nothing, and makes
    the list look more complete than it is."""
    emitted = _emitted_equation_kinds()
    stale = sorted(
        k for k in UnifiedPipeline.EQUATION_LANE_EVENT_KINDS if k.startswith("equation_")
    )
    unmatched = [k for k in stale if k not in emitted]
    assert not unmatched, f"EQUATION_LANE_EVENT_KINDS replays kinds nothing emits: {unmatched}"


def test_not_replayed_entries_carry_a_reason() -> None:
    """An empty reason is an oversight wearing a decision's clothes."""
    empty = [k for k, why in _NOT_REPLAYED.items() if not why.strip()]
    assert not empty, f"listed as not replayed with no reason: {empty}"
