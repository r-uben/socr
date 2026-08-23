"""Regression reproductions for GH-226's final table-emission defects."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.document import DocumentHandle
from socr.core.manifest import _winning_page_output, canonical_page_texts
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.tables.reconcile import (
    TABLE_EMISSION_NONE,
    TABLE_EMISSION_WIDTH_MISMATCH,
    table_emission_defect,
)
from socr.tables.structure_check import (
    DEFECT_GRID_SHAPE,
    DEFECT_TABLE_LATEX_LEAK,
    DEFECT_TABLE_WIDTH_MISMATCH,
    table_output_defect,
)


def test_residual_latex_table_command_is_a_shipping_defect() -> None:
    emitted = "\n".join(
        [
            r"|  | Outturn | \multicolumn{2}{c}{Forecast} |",
            "| --- | --- | --- |",
            "| Revenue | 10 | 11 |",
        ]
    )

    assert table_output_defect(emitted, None, None) == DEFECT_TABLE_LATEX_LEAK


def test_delimiter_width_mismatch_is_a_shipping_defect() -> None:
    emitted = "\n".join(
        [
            "| A | B | C |",
            "| --- | --- |",
            "| 1 | 2 | 3 |",
        ]
    )

    assert table_output_defect(emitted, None, None) == DEFECT_TABLE_WIDTH_MISMATCH


@pytest.mark.parametrize(
    "command",
    [
        r"\multicolumn{2}{c}{Forecast}",
        r"\multirow{2}{*}{Forecast}",
        r"\cline{1-2}",
        r"\hline",
    ],
)
def test_each_table_only_latex_command_has_the_same_defect(command: str) -> None:
    emitted = f"| A | {command} |\n| --- | --- |\n| 1 | 2 |"

    assert table_output_defect(emitted, None, None) == DEFECT_TABLE_LATEX_LEAK


@pytest.mark.parametrize(
    "emitted",
    [
        "| A | B |\n| --- | --- | --- |\n| 1 | 2 |",
        "A | B | C\n--- | ---\n1 | 2 | 3",
    ],
)
def test_delimiter_mismatch_covers_wider_and_borderless_forms(emitted: str) -> None:
    assert table_emission_defect(emitted) == TABLE_EMISSION_WIDTH_MISMATCH


def test_generic_ragged_body_keeps_its_existing_grid_shape_policy() -> None:
    emitted = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 | 5 |"

    assert table_emission_defect(emitted) == TABLE_EMISSION_NONE
    assert table_output_defect(emitted, None, None) == DEFECT_GRID_SHAPE


@pytest.mark.parametrize(
    "emitted",
    [
        "```markdown\n| A | \\multicolumn{2}{c}{B} |\n| --- | --- |\n| 1 | 2 |\n```",
        "<!--\n| A | \\multicolumn{2}{c}{B} |\n| --- | --- |\n| 1 | 2 |\n-->",
        "    | A | \\multicolumn{2}{c}{B} |\n    | --- | --- |\n    | 1 | 2 |",
        r"Prose explaining \multicolumn{2}{c}{Forecast} without a table.",
    ],
)
def test_non_emitted_examples_are_ignored(emitted: str) -> None:
    assert table_output_defect(emitted, None, None) == ""


def test_inline_math_and_escaped_pipes_are_valid_cells() -> None:
    emitted = "\n".join(
        [
            r"| Variable \| definition | $\alpha$ | $\frac{\Delta y}{y}$ |",
            "| --- | --- | --- |",
            r"| Revenue \| net | 0.12*** | (0.03) |",
        ]
    )

    assert table_emission_defect(emitted) == TABLE_EMISSION_NONE


def test_later_invalid_table_is_found_after_an_earlier_clean_table() -> None:
    emitted = "\n\n".join(
        [
            "| A | B |\n| --- | --- |\n| 1 | 2 |",
            r"| C | \hline |" + "\n| --- | --- |\n| 3 | 4 |",
        ]
    )

    assert table_output_defect(emitted, None, None) == DEFECT_TABLE_LATEX_LEAK


def _state_with_passing_page(text: str) -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=Path("/tmp/gh226.pdf"), page_count=1)
    state = DocumentState(handle=handle)
    output = PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    state.pages[1].attempts.append(output)
    state.pages[1].best_output = output
    return state


def test_final_winner_fails_closed_if_an_earlier_path_missed_the_guard() -> None:
    state = _state_with_passing_page("| A | B | C |\n| --- | --- |\n| 1 | 2 | 3 |")

    winner = _winning_page_output(state, 1)

    assert winner.status is PageStatus.ERROR
    assert winner.audit_passed is False
    assert winner.failure_mode is FailureMode.TABLE_EMISSION_INVALID
    assert DEFECT_TABLE_WIDTH_MISMATCH in winner.text
    assert "| 1 | 2 | 3 |" not in winner.text


def test_preflagged_output_does_not_bypass_the_hard_emission_failure() -> None:
    """GH-226 syntax defects are hard failures, not GH-259 ragged warnings."""
    state = _state_with_passing_page(r"| A | \hline |" + "\n| --- | --- |\n| 1 | 2 |")
    state.pages[1].best_output.status = PageStatus.WARNING
    state.pages[1].best_output.audit_passed = False

    winner = _winning_page_output(state, 1)

    assert winner.status is PageStatus.ERROR
    assert winner.failure_mode is FailureMode.TABLE_EMISSION_INVALID
    assert DEFECT_TABLE_LATEX_LEAK in winner.text


def test_whole_document_cli_output_uses_the_same_final_guard() -> None:
    state = _state_with_passing_page("clean per-page placeholder")
    state.pages[1].best_output = None
    state.pages[1].attempts.clear()
    state.whole_doc_attempts.append(
        PageOutput(
            page_num=0,
            text=(
                "## Page 1\n\n"
                r"| A | \multicolumn{2}{c}{B} |"
                "\n| --- | --- |\n| 1 | 2 |"
            ),
            status=PageStatus.SUCCESS,
            engine="gemini-cli",
            audit_passed=True,
        )
    )

    page_text = canonical_page_texts(state)[0]

    assert page_text.startswith("[page 1 failed: invalid table emission")
    assert DEFECT_TABLE_LATEX_LEAK in page_text
