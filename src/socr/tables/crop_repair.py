"""Value-guarded LOCAL-VLM crop fallback for structurally broken tables (GH-56).

When deterministic header repair declines (or the native verifier hard-fails),
the bounded table-crop VLM re-read may patch the page — even with
``auto_patch_tables=False`` — but ONLY when the patch measurably improves
structural verification.  Partial crop timeouts still block patching.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from socr.tables.header_repair import (
    _MIN_HEADER_DATA_COL_GAP,
    detect_header_column_collapse,
)
from socr.tables.native_verifier import VerifierState, verify_native_table
from socr.tables.reconcile import find_table_blocks


@dataclass(frozen=True)
class StructuralDefectSnapshot:
    """Structural defect state derived from markdown + optional native geometry."""

    header_collapsed: bool
    header_col_gap: int
    verifier_hard_fail: bool
    verifier_state: str
    label_binding_failure: bool
    output_col_count: int
    native_lane_count: int


def detect_general_column_collapse(grid: list[list[str]]) -> tuple[bool, int, int]:
    """Format-agnostic header collapse: modal data width vs header width.

    Unlike ``detect_header_column_collapse`` (numeric data rows only), counts
    every populated body row so categorical / text tables are included.
    """
    if len(grid) < 2:
        return False, 0, 0

    header_cols = len(grid[0])
    data_widths: list[int] = []
    for row in grid[1:]:
        if any(cell.strip() for cell in row):
            data_widths.append(len(row))

    if not data_widths:
        return False, header_cols, header_cols

    expected_cols = Counter(data_widths).most_common(1)[0][0]
    gap = expected_cols - header_cols
    collapsed = gap >= _MIN_HEADER_DATA_COL_GAP
    return collapsed, header_cols, expected_cols


def _max_header_col_gap(text: str) -> tuple[bool, int]:
    """Return (any_collapsed, max_gap) across all markdown table blocks."""
    blocks = find_table_blocks(text)
    if not blocks:
        return False, 0

    any_collapsed = False
    max_gap = 0
    for block in blocks:
        numeric_collapsed, hdr, expected = detect_header_column_collapse(block.grid)
        general_collapsed, g_hdr, g_expected = detect_general_column_collapse(block.grid)
        collapsed = numeric_collapsed or general_collapsed
        if collapsed:
            any_collapsed = True
            gap = max(expected - hdr, g_expected - g_hdr)
            max_gap = max(max_gap, gap)
    return any_collapsed, max_gap


def snapshot_structural_defects(
    text: str,
    *,
    fitz_page=None,
) -> StructuralDefectSnapshot:
    """Capture structural defect signals for *text* (optionally vs native geometry)."""
    header_collapsed, header_col_gap = _max_header_col_gap(text)

    verifier_hard_fail = False
    verifier_state = VerifierState.AMBIGUOUS
    label_binding_failure = False
    output_col_count = 0
    native_lane_count = 0

    if fitz_page is not None and text.strip():
        vr = verify_native_table(fitz_page, text)
        verifier_hard_fail = vr.hard_fail
        verifier_state = vr.state
        output_col_count = vr.output_col_count
        native_lane_count = vr.native_lane_count
        label_binding_failure = any(
            row.get("predicate") == "label_binding_failure" for row in vr.drifted_rows
        )

    return StructuralDefectSnapshot(
        header_collapsed=header_collapsed,
        header_col_gap=header_col_gap,
        verifier_hard_fail=verifier_hard_fail,
        verifier_state=verifier_state,
        label_binding_failure=label_binding_failure,
        output_col_count=output_col_count,
        native_lane_count=native_lane_count,
    )


def defect_severity(snapshot: StructuralDefectSnapshot) -> tuple[int, int, int, int, int]:
    """Lexicographic severity tuple — lower is better (fewer / milder defects)."""
    lane_gap = (
        abs(snapshot.output_col_count - snapshot.native_lane_count)
        if snapshot.native_lane_count >= 2
        else 0
    )
    return (
        1 if snapshot.verifier_hard_fail else 0,
        1 if snapshot.label_binding_failure else 0,
        1 if snapshot.header_collapsed else 0,
        snapshot.header_col_gap,
        lane_gap,
    )


def page_needs_crop_repair_fallback(
    page_text: str,
    *,
    native_table_unverifiable: bool = False,
    fitz_page=None,
) -> bool:
    """True when a structurally broken table should attempt crop-VLM repair."""
    if native_table_unverifiable:
        return True

    collapsed, _gap = _max_header_col_gap(page_text)
    if collapsed:
        return True

    if fitz_page is not None and page_text.strip():
        vr = verify_native_table(fitz_page, page_text)
        if vr.hard_fail:
            return True

    return False


def crop_patch_improves_verification(
    before_text: str,
    after_text: str,
    *,
    fitz_page=None,
) -> bool:
    """True when *after_text* strictly reduces structural defects vs *before_text*."""
    before = snapshot_structural_defects(before_text, fitz_page=fitz_page)
    after = snapshot_structural_defects(after_text, fitz_page=fitz_page)

    if defect_severity(after) >= defect_severity(before):
        return False

    # Faithfulness: a patch that leaves hard-fail in place is not an improvement.
    if before.verifier_hard_fail and after.verifier_hard_fail:
        return False

    return True
