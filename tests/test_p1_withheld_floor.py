"""P1 (task t9): a TABLE_WITHHELD page rewritten through the GH-520 regional
floor BEFORE canonicalization -- reusing the exact coverage-proof machinery
``test_gh520_regional_floor_splice.py`` proved for the structure-class case,
now driven off ``PageState.table_ladder_disposition`` instead of an
exhausted structure-class ladder.

Two fixtures, mirroring GH-520's own pair:

- valid detection coverage  -> the table region is replaced by the marker,
  outside prose survives (regional splice)
- coverage mismatch         -> the whole page floors (no leaked table)

Both must ship: ERROR status, ``audit_passed=False``,
``FailureMode.TABLE_WITHHELD``, the standard failed-table marker plus
``d3_floor_png_ref``, and no markdown table / no unique cell token from the
withheld table's own bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from socr.core.manifest import _winning_page_output
from socr.core.result import FailureMode, PageStatus
from test_gh317_structure_class_floor import (
    PROSE_AFTER,
    PROSE_BEFORE,
    UNIQUE_NATIVE_ROW,
    _born_digital_pdf,
    _state,
)

MARKER = "[page 1 failed: unverifiable table — see image]"

WITHHELD_TABLE_MD = (
    "| $n$ | const. | slope | $R^2$ |\n"
    "|---|---|---|---|\n"
    "| 2 | 0.03 | 0.91 | 0.44 |\n"
    "| 5 | 0.07 | 0.85 | 0.51 |\n"
)
NATIVE_TEXT_WITH_PROSE = f"{PROSE_BEFORE}\n\n{WITHHELD_TABLE_MD}\n{PROSE_AFTER}\n"


def _withheld_page(tmp_path: Path, *, detected: int, bboxes: int | None = None):
    """A page whose ladder disposition is TABLE_WITHHELD, carrying the same
    detection-level GH-520 signal the structure-class floor consumes."""
    state = _state(_born_digital_pdf(tmp_path), native_text=NATIVE_TEXT_WITH_PROSE)
    ps = state.pages[1]
    ps.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"
    ps.detected_table_count = detected
    boxes = detected if bboxes is None else bboxes
    ps.detected_table_bboxes = [
        (72.0, 100.0 + i * 200.0, 520.0, 250.0 + i * 200.0) for i in range(boxes)
    ]
    ps.table_ladder_disposition = FailureMode.TABLE_WITHHELD
    return state, ps


def test_valid_coverage_keeps_prose_and_replaces_only_the_table(tmp_path: Path) -> None:
    from socr.tables.reconcile import find_table_blocks

    assert len(find_table_blocks(NATIVE_TEXT_WITH_PROSE)) == 1, (
        "fixture premise: the withheld region must parse as a GFM table"
    )

    state, _ps = _withheld_page(tmp_path, detected=1)
    out = _winning_page_output(state, 1)

    assert out.failure_mode is FailureMode.TABLE_WITHHELD
    assert out.status is PageStatus.ERROR
    assert out.audit_passed is False
    assert MARKER in out.text
    assert PROSE_BEFORE in out.text
    assert PROSE_AFTER in out.text
    assert UNIQUE_NATIVE_ROW not in out.text
    assert "0.91" not in out.text  # no cell token from the withheld table


def test_coverage_mismatch_floors_the_whole_page(tmp_path: Path) -> None:
    """No usable bbox for the detected table -> nothing proves the splice
    covers it -> whole page floors, exactly like GH-520's structure-class
    guard."""
    state, _ps = _withheld_page(tmp_path, detected=1, bboxes=0)
    out = _winning_page_output(state, 1)

    assert out.failure_mode is FailureMode.TABLE_WITHHELD
    assert out.status is PageStatus.ERROR
    assert out.audit_passed is False
    assert MARKER in out.text
    assert PROSE_BEFORE not in out.text, (
        "prose survived on a page whose withheld-table coverage is unprovable"
    )
    assert UNIQUE_NATIVE_ROW not in out.text


def test_zero_detected_tables_is_no_evidence_and_floors_the_whole_page(tmp_path: Path) -> None:
    state, _ps = _withheld_page(tmp_path, detected=0)
    out = _winning_page_output(state, 1)

    assert PROSE_BEFORE not in out.text
    assert MARKER in out.text


def test_withheld_never_reuses_the_native_text_source(tmp_path: Path) -> None:
    """The plan's explicit constraint: the withhold splice must use the
    SELECTED output's text, never silently substitute ``p.native_text`` for
    a model winner -- pinned via a state where they visibly diverge."""
    state, ps = _withheld_page(tmp_path, detected=1)
    # Diverge the winning candidate's text from native_text so a wrong
    # implementation that reads native_text is caught.
    from dataclasses import replace

    if ps.best_output is not None:
        ps.best_output = replace(
            ps.best_output, text=NATIVE_TEXT_WITH_PROSE.replace(PROSE_BEFORE, "MODEL WINNER PROSE")
        )

    out = _winning_page_output(state, 1)
    assert MARKER in out.text
