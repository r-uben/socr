"""Regression reproductions for GH-226's final table-emission defects."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from ocr_output_contract import assemble_pages

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.cache import BlobStore
from socr.core.config import PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import (
    Manifest,
    _winning_page_output,
    build_manifest,
    canonical_page_texts,
    replay,
)
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
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


@pytest.mark.parametrize(
    "literal",
    [
        r"`\hline`",
        r"\\hline",
    ],
)
def test_literal_latex_command_names_inside_table_cells_are_ignored(literal: str) -> None:
    emitted = f"| Command | Meaning |\n| --- | --- |\n| {literal} | literal text |"

    assert table_output_defect(emitted, None, None) == ""


def test_raw_html_pre_block_is_not_emitted_table_content() -> None:
    emitted = "\n".join(
        [
            "<pre>",
            r"| Command | \hline |",
            "| --- | --- |",
            "| 1 | 2 |",
            "</pre>",
        ]
    )

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


def test_missing_page_text_cannot_crash_the_final_guard() -> None:
    state = _state_with_passing_page("temporary")
    state.pages[1].best_output.text = None  # type: ignore[assignment]

    winner = _winning_page_output(state, 1)

    assert winner.text == ""


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


def test_manifest_revalidates_the_exact_saved_body_before_replay(tmp_path: Path) -> None:
    state = _state_with_passing_page("Clean selected page text.")
    state.pages[1].best_output.engine = "native"
    saved_body = assemble_pages([r"| A | \multicolumn{2}{c}{B} |" + "\n| --- | --- |\n| 1 | 2 |"])
    store = BlobStore(tmp_path / "cache")

    manifest = build_manifest(state, store, saved_body=saved_body)
    frozen = store.get_page(manifest.entries[1].blob_ref)

    assert frozen.status is PageStatus.ERROR
    assert frozen.failure_mode is FailureMode.TABLE_EMISSION_INVALID
    assert DEFECT_TABLE_LATEX_LEAK in frozen.text
    assert DEFECT_TABLE_LATEX_LEAK in replay(manifest, store)


def test_post_figure_body_guard_updates_markdown_sidecar_manifest_and_status(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((60, 60), "Clean native page text for final-body testing.")
    doc.save(pdf_path)
    doc.close()

    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    output = PageOutput(
        page_num=1,
        text="Clean selected page text.",
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    state.pages[1].attempts.append(output)
    state.pages[1].best_output = output
    invalid_final = assemble_pages(
        [r"| A | \multicolumn{2}{c}{B} |" + "\n| --- | --- |\n| 1 | 2 |"]
    )
    pipeline = UnifiedPipeline(PipelineConfig(save_figures=True, write_manifest=True, quiet=True))
    out_dir = tmp_path / "out"

    with patch.object(
        pipeline,
        "_describe_and_embed_figures",
        return_value=invalid_final,
    ):
        result = pipeline._phase_assemble(state, out_dir)

    assert result.status is DocumentStatus.AUDIT_FAILED
    assert result.audit_passed is False
    markdown = next(out_dir.rglob("paper.md")).read_text()
    fragment = next(out_dir.rglob("pages/00001.md")).read_text()
    sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    assert DEFECT_TABLE_LATEX_LEAK in markdown
    assert DEFECT_TABLE_LATEX_LEAK in fragment
    assert sidecar["winning_output"]["status"] == PageStatus.ERROR.value
    assert sidecar["winning_output"]["failure_mode"] == FailureMode.TABLE_EMISSION_INVALID.value
    assert any(
        event["data"].get("site") == "final_body"
        for event in sidecar["audit_events"]
        if event["kind"] == "table_structure_failed"
    )

    manifest_path = next(out_dir.rglob("manifest.json"))
    manifest = Manifest.load(manifest_path)
    manifest_store = BlobStore(manifest_path.parent / "cache")
    frozen = manifest_store.get_page(manifest.entries[1].blob_ref)
    assert frozen.status is PageStatus.ERROR
    assert frozen.failure_mode is FailureMode.TABLE_EMISSION_INVALID
    assert DEFECT_TABLE_LATEX_LEAK in replay(manifest, manifest_store)


def test_native_emission_defect_reason_survives_assessment_propagation() -> None:
    assessment = DocumentAssessment(
        path=Path("/tmp/gh226.pdf"),
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text="table",
                confidence=1.0,
                has_tables=True,
                native_table_structure_defective=True,
                native_table_emission_defect=DEFECT_TABLE_WIDTH_MISMATCH,
            )
        ],
    )
    state = _state_with_passing_page("temporary")

    state.apply_born_digital(assessment)

    assert state.pages[1].native_table_emission_defect == DEFECT_TABLE_WIDTH_MISMATCH

    audit_state = _state_with_passing_page("temporary")
    pipeline = UnifiedPipeline(PipelineConfig(native_only=True, quiet=True))
    with patch.object(pipeline.bd_detector, "detect", return_value=assessment):
        pipeline._phase_analyze(audit_state)

    events = [event for event in audit_state.events if event.kind == "table_structure_failed"]
    assert len(events) == 1
    assert events[0].data["defects"] == [DEFECT_TABLE_WIDTH_MISMATCH]
    assert "grid_shape" not in events[0].data["defects"]
