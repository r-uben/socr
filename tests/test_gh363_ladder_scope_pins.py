"""GH-363: the two scope claims in the table-judge-ladder plan, made checkable.

#362 shipped the plan; these pins were called out on the PR and never landed.
By the time they were written every ticket was DONE, so they are not
forward-looking instructions any more -- they are statements about what shipped,
and a statement about shipped code should be verified rather than asserted.

Both were verified against main before being written into `TICKETS.md`. These
tests keep them true.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"
PLAN = (
    pathlib.Path(__file__).resolve().parents[1]
    / "docs"
    / "plans"
    / "table-judge-ladder"
    / "TICKETS.md"
)


def _enclosing_functions(tree: ast.AST, predicate) -> set[str]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    found: set[str] = set()
    for node in ast.walk(tree):
        if not predicate(node):
            continue
        fn: ast.AST = node
        while fn in parents and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = parents[fn]
        found.add(getattr(fn, "name", "<module>"))
    return found


def test_the_table_judge_gate_is_agentic_only() -> None:
    """B1's scope claim: one caller, inside `_phase_agentic`.

    If the gate gains a caller on a non-agentic path, the plan's scope sentence
    becomes false and #317's native-unwitnessed hole changes shape -- either way
    the record must not keep asserting the old scope.
    """
    tree = ast.parse((SRC / "pipeline" / "orchestrator.py").read_text())

    callers = _enclosing_functions(
        tree,
        lambda n: (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_run_table_judge_gate"
        ),
    )
    assert callers == {"_phase_agentic"}, (
        f"the table judge gate is no longer agentic-only; called from {sorted(callers)}"
    )


def test_the_gate_skips_chart_asset_pages() -> None:
    """B1's skip rule, as a DIFFERENCE against the same page on another engine.

    Asserting only that a chart_asset page does nothing is vacuous: with mocked
    arguments the helper can return early for unrelated reasons, and the first
    version of this test passed with the skip removed. The non-chart page is the
    control -- it must get further.
    """
    from unittest.mock import MagicMock

    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    def _calls(engine: str) -> int:
        output = PageOutput(
            page_num=1,
            text="| A | B |\n| --- | --- |\n| 1 | 2 |",
            status=PageStatus.SUCCESS,
            engine=engine,
            audit_passed=True,
        )
        state, ps, rungs = MagicMock(), MagicMock(), MagicMock()
        try:
            UnifiedPipeline._run_table_judge_gate(MagicMock(), state, 1, ps, output, rungs)
        except Exception:
            # A mocked run may not complete; what matters is how far it got
            # before failing, which the call counts below record.
            pass
        return len(state.mock_calls) + len(ps.mock_calls) + len(rungs.mock_calls)

    chart_calls = _calls("chart_asset")
    other_calls = _calls("qwen")

    assert chart_calls == 0, f"a chart_asset page was not skipped: {chart_calls} calls"
    assert other_calls > 0, (
        "the control page was skipped too, so this test cannot tell the skip "
        "rule from the helper failing early for another reason"
    )


def test_s2_keys_on_verdict_not_on_finding_code() -> None:
    """A1/A4's S2 contract: `code` is evidence, not a second accept gate.

    Bake-off finding 3 (GH-356): wrong-code-right-verdict is real on hard
    fabrication, so a ladder that branched on `code` would reject tables it had
    correctly judged.
    """
    tree = ast.parse((SRC / "judge" / "table_ladder.py").read_text())

    code_reads = [
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "code"
    ]
    assert not code_reads, (
        "the ladder reducer now reads a finding `code`; S2 must key on `verdict` "
        "and `confidence` only (A1 contract)"
    )


def test_the_plan_records_both_pins() -> None:
    """The record itself -- these tests exist to keep TICKETS.md honest."""
    plan = PLAN.read_text()
    assert "agentic-only" in plan, "B1's scope pin is missing from the plan"
    assert "Non-agentic emit paths are out of scope" in plan
    assert "keys on **`verdict` only**" in plan, "A1's S2 pin is missing from the plan"
