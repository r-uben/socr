"""GH-169: the judge's verdict must survive for EVERY attempt.

`_phase_agentic` copied `att.reason` onto the output only when the rejected
output was EMPTY. `build_manifest` then journals `skip_reason or failure_mode`,
so an ordinary semantic rejection -- a provider that produced a table the judge
refused -- journaled as `"none"`. The one question the manifest exists to
answer, *why did the ladder escalate past this rung*, had no answer.

Measured before the fix:

    rejected, non-empty output -> manifest reason: 'none'
    rejected, EMPTY output     -> manifest reason: 'provider returned nothing'
    accepted                   -> manifest reason: 'none'

`judge_reason` is a NEW field rather than a wider use of `skip_reason`, which
documents why a rung was never TRIED. Conflating them would make "budget
exceeded" and "the judge rejected this reading" indistinguishable.
"""

from __future__ import annotations

import pytest

from socr.core.result import FailureMode, PageOutput, PageStatus

REASON = "native_table_verifier: ambiguous_lane_count_mismatch"


def _journal_reason(out: PageOutput) -> str:
    """The expression `build_manifest` uses for an attempt's journal entry."""
    return (
        getattr(out, "skip_reason", "")
        or getattr(out, "judge_reason", "")
        or out.failure_mode.value
    )


def _attempt(*, accepted: bool, text: str, judge_reason: str, skip_reason: str = "") -> PageOutput:
    out = PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=accepted,
    )
    out.judge_reason = judge_reason
    out.skip_reason = skip_reason
    return out


def test_a_rejected_non_empty_attempt_keeps_its_reason() -> None:
    """The defect: this journaled "none"."""
    out = _attempt(accepted=False, text="| a | 1 |", judge_reason=REASON)
    assert _journal_reason(out) == REASON, (
        "a provider whose reading the judge refused still journals no reason, so "
        "the escalation cannot be audited"
    )


def test_an_accepted_attempt_keeps_its_reason_too() -> None:
    """Acceptance asks for every attempted provider, accepted included."""
    out = _attempt(accepted=True, text="| a | 1 |", judge_reason="judge accepted: high confidence")
    assert _journal_reason(out) == "judge accepted: high confidence"


def test_a_skip_reason_still_wins() -> None:
    """Precedence: a rung never TRIED must not be reported as a judge verdict.

    This is why `judge_reason` is a separate field. If the new one took
    priority, "budget exceeded" would be overwritten by whatever verdict text
    happened to be lying around.
    """
    out = _attempt(accepted=False, text="", judge_reason=REASON, skip_reason="budget exceeded")
    assert _journal_reason(out) == "budget exceeded"


def test_the_failure_mode_is_still_the_last_resort() -> None:
    """An attempt with neither reason falls back as before -- no behaviour change."""
    out = _attempt(accepted=False, text="", judge_reason="")
    out.failure_mode = FailureMode.TIMEOUT
    assert _journal_reason(out) == FailureMode.TIMEOUT.value


@pytest.mark.parametrize(
    ("accepted", "text", "judge_reason", "skip_reason"),
    [
        (True, "| a | 1 |", "judge accepted: high confidence", ""),
        (False, "| a | 1 |", REASON, ""),
        (False, "", "judge failed to parse a verdict", ""),
        (False, "", "", "budget exceeded"),
    ],
    ids=["accepted", "semantic-rejection", "judge-failure", "budget-skip"],
)
def test_it_survives_a_serialisation_round_trip(
    accepted: bool, text: str, judge_reason: str, skip_reason: str
) -> None:
    """A field the manifest never persisted would be lost on replay.

    The acceptance names acceptance, semantic rejection, judge failure and
    budget skip; each is a row here.
    """
    original = _attempt(
        accepted=accepted, text=text, judge_reason=judge_reason, skip_reason=skip_reason
    )
    restored = PageOutput.from_dict(original.to_dict())

    assert restored.judge_reason == judge_reason
    assert restored.skip_reason == skip_reason
    assert _journal_reason(restored) == _journal_reason(original)


def test_the_production_sites_are_the_ones_being_described() -> None:
    """`_journal_reason` above is a replica; this pins the real lines.

    Without it the whole file is a test of its own helper: reverting either
    production site would leave every assertion above green. That is the exact
    failure this backlog keeps turning up, so it is guarded explicitly.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"

    # 1. the orchestrator sets judge_reason for EVERY attempt -- no condition.
    orch = (src / "pipeline" / "orchestrator.py").read_text()
    tree = ast.parse(orch)
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "judge_reason" for t in node.targets)
    ]
    assert len(assigns) == 1, f"expected exactly one judge_reason assignment, got {len(assigns)}"
    rhs = ast.unparse(assigns[0].value)
    assert "att.reason" in rhs, f"judge_reason is not taken from the attempt: {rhs}"
    assert "if" not in rhs, (
        f"judge_reason is set conditionally, which is the GH-169 defect itself: {rhs}"
    )

    # 2. the manifest journal consults it.
    manifest = (src / "core" / "manifest.py").read_text()
    mtree = ast.parse(manifest)
    reason_values = [
        ast.unparse(node.value)
        for node in ast.walk(mtree)
        if isinstance(node, ast.Dict)
        for key, node_value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "reason"
        for node in [type("_", (), {"value": node_value})()]
    ]
    assert any("judge_reason" in v for v in reason_values), (
        f"no manifest journal entry consults judge_reason: {reason_values}"
    )
