"""Adversarial tests for default-on native per-cell binding (Fable/Astra review)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from socr.judge.table_verdict import (
    NATIVE_CELL_CHECK_AMBIGUOUS_KIND,
    NATIVE_CELL_CHECK_SKIPPED_KIND,
    NATIVE_CELL_CHECK_VERIFIED_KIND,
)
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.binding import BindingResult, UnboundCell, parse_grid
from socr.tables.native_cell_check import (
    NativeCellOutcome,
    check_table_cell_binding,
    classify_binding_for_cell_check,
)
from socr.tables.native_verifier import (
    binding_numeric_tokens_equivalent,
    _canonical_binding_numeric,
)


def _minimal_binding(**kwargs) -> BindingResult:
    defaults = dict(
        matched_cells=[],
        contradicted_cells=[],
        row_label_contradictions=[],
        native_unbound=[],
        model_unbound=[],
        ambiguous_count=0,
        row_binding={},
        native_rows=[],
        column_binding_unverifiable=False,
        row_binding_unverifiable=False,
    )
    defaults.update(kwargs)
    return BindingResult(**defaults)


def test_unbound_empty_col_path_is_not_verified() -> None:
    """Native unbound with numeric token and empty col_path must not verify."""
    md = "| A | B |\n| --- | --- |\n| x | 1 |\n"
    result = _minimal_binding(
        matched_cells=[object()],  # type: ignore[list-item]
        native_unbound=[UnboundCell(row_path=("x",), col_path=(), token="99")],
    )
    assert classify_binding_for_cell_check(result, md) is not NativeCellOutcome.VERIFIED


def test_single_matched_cell_without_full_coverage_is_ambiguous() -> None:
    md = "| A | B |\n| --- | --- |\n| r1 | 1 |\n| r2 | 2 |\n"
    result = _minimal_binding(
        matched_cells=[object()],  # type: ignore[list-item]
        row_binding={0: 0},
        row_binding_unverifiable=True,
    )
    assert classify_binding_for_cell_check(result, md) is not NativeCellOutcome.VERIFIED


def test_row_binding_without_numeric_comparisons_is_ambiguous() -> None:
    md = "| A | B |\n| --- | --- |\n| r1 | 1 |\n"
    result = _minimal_binding(row_binding={0: 0}, row_binding_unverifiable=True)
    assert classify_binding_for_cell_check(result, md) is not NativeCellOutcome.VERIFIED


def test_percent_is_not_equal_to_bare_number() -> None:
    assert binding_numeric_tokens_equivalent("45%", "45") is False


def test_comma_thousands_triplet_is_ambiguous_not_verified() -> None:
    assert _canonical_binding_numeric("1,250")[1] is True
    assert binding_numeric_tokens_equivalent("1,250", "1250") is None


def test_parenthesised_negative_binding_normalization() -> None:
    assert _canonical_binding_numeric("(0,4)")[0] == "-0.4"


def test_interior_thousand_space_is_stripped_for_binding() -> None:
    left, amb = _canonical_binding_numeric("1\u202f234,5")
    assert amb is False
    assert left == "1234.5"


def test_unit_suffix_md_euro_is_parsed() -> None:
    assert _canonical_binding_numeric("1,5 M€")[0] == "1.5"


def test_swapped_row_labels_same_values_not_verified_when_columns_unverifiable() -> None:
    """Astra 3: permuted row labels with agreeing numerics must not verify."""
    md = "| L | a | b |\n| --- | --- | --- |\n| Alpha | 1 | 2 |\n| Beta | 3 | 4 |\n"
    swapped = "| L | a | b |\n| --- | --- | --- |\n| Beta | 1 | 2 |\n| Alpha | 3 | 4 |\n"
    native_rows = [
        SimpleNamespace(
            is_parent=False,
            row_path=("Alpha",),
            lane_tokens={0: ("1", False), 1: ("2", False)},
        ),
        SimpleNamespace(
            is_parent=False,
            row_path=("Beta",),
            lane_tokens={0: ("3", False), 1: ("4", False)},
        ),
    ]
    binding = _minimal_binding(
        column_binding_unverifiable=True,
        row_binding={0: 0, 1: 1},
        native_rows=native_rows,
    )
    assert classify_binding_for_cell_check(binding, md) is NativeCellOutcome.VERIFIED
    assert classify_binding_for_cell_check(binding, swapped) is not NativeCellOutcome.VERIFIED


def test_model_unbound_numeric_blocks_when_columns_unverifiable() -> None:
    md = "| A | B |\n| --- | --- |\n| r | 1 |\n"
    result = _minimal_binding(
        column_binding_unverifiable=True,
        row_binding={0: 0},
        native_rows=[
            SimpleNamespace(
                is_parent=False,
                row_path=("r",),
                lane_tokens={0: ("1", False)},
            )
        ],
        model_unbound=[UnboundCell(row_path=("r",), col_path=("B",), token="9")],
    )
    assert classify_binding_for_cell_check(result, md) is not NativeCellOutcome.VERIFIED


def test_ambiguous_comma_thousands_in_row_is_ambiguous_not_verified() -> None:
    md = "| A | B |\n| --- | --- |\n| r | 1,250 |\n"
    binding = _minimal_binding(
        column_binding_unverifiable=True,
        row_binding={0: 0},
        native_rows=[
            SimpleNamespace(
                is_parent=False,
                row_path=("r",),
                lane_tokens={0: ("1250", False)},
            )
        ],
    )
    assert classify_binding_for_cell_check(binding, md) is NativeCellOutcome.AMBIGUOUS


def test_lane_width_mismatch_without_slot_credit_never_verifies() -> None:
    """Lane mismatch on an unverifiable table must not reach VERIFIED on numeric zip alone."""
    md = "| A | B | C |\n| --- | --- | --- |\n| r | 1 | 2 | 3 |\n"
    binding = _minimal_binding(
        column_binding_unverifiable=True,
        row_binding={0: 0},
        native_rows=[
            SimpleNamespace(
                is_parent=False,
                row_path=("r",),
                lane_tokens={0: ("1", False)},
            )
        ],
    )
    assert classify_binding_for_cell_check(binding, md) is not NativeCellOutcome.VERIFIED


def test_dash_position_swap_check_table_cell_binding_e2e(tmp_path: Path) -> None:
    """'7 | -' vs '- | 7' through bind(), not only slot helper."""
    pdf = tmp_path / "dash_swap.pdf"  # type: ignore[operator]
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, 280, 160)
    page.draw_rect(rect, width=0.8)
    for y, vals in [(85, ("R1", "7", "-")), (105, ("R2", "8", "3"))]:
        x = 60
        for v in vals:
            page.insert_text((x, y), v, fontsize=10)
            x += 55
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    words = doc[0].get_text("words")
    good = "| L | v1 | v2 |\n| --- | --- | --- |\n| R1 | 7 | - |\n| R2 | 8 | 3 |\n"
    bad = "| L | v1 | v2 |\n| --- | --- | --- |\n| R1 | - | 7 |\n| R2 | 8 | 3 |\n"
    ok = check_table_cell_binding(
        table_id="t0", words=words, markdown=good, region_bbox=rect, spans=[]
    )
    no = check_table_cell_binding(
        table_id="t0", words=words, markdown=bad, region_bbox=rect, spans=[]
    )
    doc.close()
    assert ok.outcome is NativeCellOutcome.VERIFIED
    assert no.outcome is not NativeCellOutcome.VERIFIED


def test_empty_native_words_is_ambiguous_not_contradicted() -> None:
    r = check_table_cell_binding(
        table_id="t0",
        words=[],
        markdown="| A | B |\n| --- | --- |\n| x | 1 |\n",
        region_bbox=(0, 0, 100, 100),
        spans=[],
    )
    assert r.outcome is NativeCellOutcome.AMBIGUOUS


def test_dash_position_swap_is_contradicted() -> None:
    """'7 | -' vs '- | 7' must disagree when native lanes preserve column slots."""
    from socr.tables.native_cell_check import _compare_bound_row

    native_row = SimpleNamespace(
        is_parent=False,
        lane_tokens={0: ("7", False), 1: ("-", False)},
    )
    good_row = ["R", "7", "-"]
    bad_row = ["R", "-", "7"]
    assert _compare_bound_row(native_row, good_row) is True
    assert _compare_bound_row(native_row, bad_row) is False


def test_data_row_of_years_is_not_skipped_as_header() -> None:
    md = "| L | y1 | y2 |\n| --- | --- | --- |\n| Row | 2022 | 2023 |\n"
    from socr.tables.native_cell_check import _is_year_header_row

    grid = parse_grid(md)
    assert grid is not None
    assert len(grid.rows) == 1
    assert _is_year_header_row(grid.rows[0]) is False


def test_resume_restore_includes_native_cell_event_kinds() -> None:
    kinds = UnifiedPipeline.resume_restore_kinds()
    for kind in (
        NATIVE_CELL_CHECK_VERIFIED_KIND,
        NATIVE_CELL_CHECK_SKIPPED_KIND,
        NATIVE_CELL_CHECK_AMBIGUOUS_KIND,
        "native_cell_check_contradicted",
    ):
        assert kind in kinds


def test_native_cell_check_error_path_is_atomic(tmp_path: Path) -> None:
    """Mid-loop failure must roll back prior native-cell events before skip records."""
    from socr.core.audit_log import AuditEvent
    from socr.core.state import DocumentState, PageState
    from socr.core.document import DocumentHandle

    pdf = tmp_path / "rollback.pdf"  # type: ignore[operator]
    pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = PageState(page_num=1, is_born_digital=True, has_tables=True)
    state.pages[1] = ps
    ps.native_cell_binding["p1-t0"] = {"outcome": "verified"}
    ps.native_cell_binding["p1-t1"] = {"outcome": "verified"}
    state.events.extend(
        [
            AuditEvent(
                page_num=1,
                kind=NATIVE_CELL_CHECK_VERIFIED_KIND,
                engine="qwen",
                detail="table p1-t0",
                data={},
            ),
            AuditEvent(
                page_num=1,
                kind=NATIVE_CELL_CHECK_VERIFIED_KIND,
                engine="qwen",
                detail="table p1-t1",
                data={},
            ),
            AuditEvent(
                page_num=2,
                kind=NATIVE_CELL_CHECK_VERIFIED_KIND,
                engine="qwen",
                detail="other page",
                data={},
            ),
        ]
    )

    UnifiedPipeline.rollback_native_cell_page_state(state, 1, ps, block_count=2)

    kinds = [e.kind for e in state.events if e.page_num == 1]
    assert NATIVE_CELL_CHECK_VERIFIED_KIND not in kinds
    assert ps.native_cell_binding == {}
    assert sum(1 for e in state.events if e.page_num == 2) == 1
