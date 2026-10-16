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
}


def _emitted_equation_kinds() -> set[str]:
    """Every `equation_*` event kind the orchestrator constructs, however it gets there.

    Read from the parse tree rather than by grep, so a mention in a comment,
    docstring or test name cannot be mistaken for an emit.

    TWO shapes, and the second was a blind spot this guard had until #1145 slice 3. A
    keyword argument (`kind="equation_region_reading_attached"`) is the region lane's. The
    LEGACY seam -- `_attach_equation_latex_sidecars`, under `--detect-equations
    --recover-clean-equations` -- picks its kind in a branch and assigns it to a local
    (`kind = "equation_latex_accepted"`), then passes `kind=kind`. The scanner saw no
    literal, so that whole family was invisible to this guard in BOTH directions: the
    allowlist had no entry for it and "every emitted kind is replayed or explicitly not"
    passed anyway. That is how `equation_latex_accepted` came to be dropped on every resume
    (the #252 shape) with a guard in place that exists to catch exactly that.
    """
    tree = ast.parse(_ORCHESTRATOR.read_text())
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if (
                    keyword.arg == "kind"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                    and keyword.value.value.startswith("equation_")
                ):
                    kinds.add(keyword.value.value)
        # ``kind = "equation_..."`` -- a kind chosen in a branch and passed on as a variable.
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            value = node.value.value
            if not (isinstance(value, str) and value.startswith("equation_")):
                continue
            if any(isinstance(target, ast.Name) and target.id == "kind" for target in node.targets):
                kinds.add(value)
    return kinds


def test_the_scanner_sees_a_kind_assigned_to_a_variable() -> None:
    """#1145: the blind spot above, pinned, so narrowing the scanner fails here.

    Without this the scanner could go back to keyword-only and every guard in this file
    would stay green while a whole family of events was once again invisible to them --
    which is what happened, and is why `equation_latex_accepted` was dropped on resume.
    The kind is NAMED rather than read from the scanner's own output, which would be a
    tautology.
    """
    emitted = _emitted_equation_kinds()
    assert "equation_latex_accepted" in emitted, (
        "the legacy seam assigns its kind to a local before passing it, and the scanner "
        "cannot see that shape -- so this file's guards do not cover that seam"
    )
    assert "equation_region_reading_attached" in emitted, emitted


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


def test_the_legacy_sidecar_refusal_is_replayed() -> None:
    """GH-540: the legacy seam's refusal, decided rather than left inherited.

    The shipped markdown does say a reading was withheld -- an HTML comment
    beside the crop carries the validation reason -- so the output is not
    silent. But the WITHHELD READING ITSELF (`raw_latex`) lives only in this
    event, and dropping it on resume leaves nobody able to say WHAT was refused,
    only that something was.

    That, plus the asymmetry: the region lane's refusal survived a resume and
    this one did not, for no principle.
    """
    assert "equation_sidecar_refused" in UnifiedPipeline.EQUATION_LANE_EVENT_KINDS


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


def test_a_refusal_actually_survives_the_replay_filter() -> None:
    """The allowlist is a means; surviving the filter is the end.

    Membership tests above assert a name is in a frozenset. This calls the
    assembly `_restore_terminal_page_state` itself uses, so dropping a TERM from
    the union -- not just an entry from the set -- is caught.

    The first version rebuilt that union from the same three sources (cubic P2
    on #551). Editing the real one would have left this green while the filter
    silently stopped replaying a whole family: the failure this guard exists to
    catch, reproduced inside the guard.
    """
    restore_kinds = UnifiedPipeline.resume_restore_kinds()

    for kind in ("equation_sidecar_refused", "equation_region_reading_unverifiable"):
        assert kind in restore_kinds, (
            f"{kind} does not survive the resume filter, so the record of a "
            "withheld reading exists only on the run that produced it"
        )

    # And the control: a kind nobody replays must NOT slip through, or the
    # assertion above would hold for any string at all.
    assert "equation_region_detected" not in restore_kinds
