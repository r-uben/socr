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
    _prepare_numeric_cell,
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


def _draw_table_pdf(
    tmp_path: Path,
    *,
    rows: list[tuple[str, ...]],
    rect: fitz.Rect | None = None,
    margin_number: tuple[float, float, str] | None = None,
) -> tuple[Path, fitz.Rect, list]:
    pdf = tmp_path / "table.pdf"  # type: ignore[operator]
    doc = fitz.open()
    page = doc.new_page()
    region = rect or fitz.Rect(50, 80, 320, 80 + 22 * (len(rows) + 2))
    page.draw_rect(region, width=0.5)
    y = region.y0 + 18
    x_label = region.x0 + 8
    x_cols = [region.x0 + 110, region.x0 + 170, region.x0 + 230]
    for row in rows:
        page.insert_text((x_label, y), row[0], fontsize=10)
        for col_i, val in enumerate(row[1:]):
            page.insert_text((x_cols[col_i], y), val, fontsize=10)
        y += 20
    if margin_number is not None:
        mx, my, num = margin_number
        page.insert_text((mx, my), num, fontsize=10)
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    words = doc[0].get_text("words")
    doc.close()
    return pdf, region, words


def test_unbound_empty_col_path_is_not_verified(tmp_path: Path) -> None:
    """Dropped data-row numerics through real bind() must not reach VERIFIED."""
    full_md = "| L | a | b |\n| --- | --- | --- |\n| Taux | 7 | 9 |\n| Data | 1 | 2 |\n"
    dropped_md = "| L | a | b |\n| --- | --- | --- |\n| Data | 1 | 2 |\n"
    _, region, words = _draw_table_pdf(
        tmp_path,
        rows=[("Taux", "7", "9"), ("Data", "1", "2")],
    )
    r = check_table_cell_binding(
        table_id="t0", words=words, markdown=dropped_md, region_bbox=region, spans=[]
    )
    assert r.outcome is not NativeCellOutcome.VERIFIED


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


def test_parenthesised_thousands_ambiguous_not_equal_to_bare() -> None:
    assert binding_numeric_tokens_equivalent("(1,250)", "1,250") is False


def test_interior_thousand_space_is_stripped_for_binding() -> None:
    left, amb = _canonical_binding_numeric("1\u202f234,5")
    assert amb is False
    assert left == "1234.5"


def test_unit_suffix_md_euro_is_parsed() -> None:
    assert _canonical_binding_numeric("1,5 M€")[0] == "1.5@M€"
    assert binding_numeric_tokens_equivalent("1,5 M€", "1,5 Md€") is False


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


def test_native_unbound_year_header_in_header_band_is_excludable() -> None:
    from socr.tables.native_cell_check import _native_unbound_cell_is_excludable
    from socr.tables.binding import BindingResult

    binding = BindingResult(
        row_binding={1: 1},
        native_rows=[
            SimpleNamespace(row_path=("Hdr",), is_parent=False, label_bbox=(0, 0, 0, 10)),
            SimpleNamespace(row_path=("Data",), is_parent=False, label_bbox=(0, 20, 0, 30)),
        ],
        native_header_band_y_max=12.0,
        native_data_y_min=20.0,
    )
    cell = UnboundCell(
        row_path=("Hdr",),
        col_path=(),
        token="2024",
        token_bbox=(100, 5, 120, 12),
    )
    assert _native_unbound_cell_is_excludable(cell, binding, (50, 0, 200, 40))


def test_native_unbound_page_number_outside_region_is_excludable() -> None:
    from socr.tables.native_cell_check import _native_unbound_cell_is_excludable
    from socr.tables.binding import BindingResult

    binding = BindingResult(row_binding={0: 0})
    cell = UnboundCell(
        row_path=(),
        col_path=(),
        token="999",
        token_bbox=(400, 750, 420, 762),
    )
    table_region = (50, 80, 320, 140)
    assert _native_unbound_cell_is_excludable(cell, binding, table_region)


def test_native_unbound_footnote_below_table_is_excludable() -> None:
    from socr.tables.native_cell_check import _native_unbound_cell_is_excludable
    from socr.tables.binding import BindingResult

    binding = BindingResult(
        row_binding={0: 0},
        native_rows=[
            SimpleNamespace(
                row_path=("Data",),
                is_parent=False,
                label_bbox=(0, 20, 0, 35),
            ),
        ],
    )
    cell = UnboundCell(
        row_path=("Data",),
        col_path=("note",),
        token="8,8",
        token_bbox=(100, 70, 120, 82),
    )
    assert _native_unbound_cell_is_excludable(cell, binding, (0, 0, 200, 50))


def test_native_unbound_data_cell_blocks_when_columns_unverifiable() -> None:
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
        native_unbound=[
            UnboundCell(row_path=("Dropped row",), col_path=("B",), token="9"),
        ],
    )
    assert classify_binding_for_cell_check(result, md) is not NativeCellOutcome.VERIFIED


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


@pytest.mark.parametrize(
    "full_rows,dropped_md,label",
    [
        (
            [("Taux", "7", "9"), ("Data", "1", "2")],
            "| L | a | b |\n| --- | --- | --- |\n| Data | 1 | 2 |\n",
            "dropped_taux_row",
        ),
        (
            [("Année de base", "2019", "2020"), ("Data", "1", "2")],
            "| L | a | b |\n| --- | --- | --- |\n| Data | 1 | 2 |\n",
            "dropped_annee_de_base",
        ),
        (
            [("Data", "1", "2"), ("** dont Etat", "3", "4")],
            "| L | a | b |\n| --- | --- | --- |\n| Data | 1 | 2 |\n",
            "dropped_footnote_row",
        ),
    ],
)
def test_dropped_native_rows_not_verified_via_real_bind(
    tmp_path: Path, full_rows, dropped_md, label
) -> None:
    _, region, words = _draw_table_pdf(tmp_path, rows=full_rows)
    r = check_table_cell_binding(
        table_id="t0", words=words, markdown=dropped_md, region_bbox=region, spans=[]
    )
    assert r.outcome is not NativeCellOutcome.VERIFIED, label


def test_spacer_row_table_still_verified_via_real_bind(tmp_path: Path) -> None:
    md = "| L | v |\n| --- | --- |\n| | |\n| Data | 1 |\n"
    _, region, words = _draw_table_pdf(
        tmp_path,
        rows=[("Data", "1")],
    )
    r = check_table_cell_binding(
        table_id="t0", words=words, markdown=md, region_bbox=region, spans=[]
    )
    assert r.outcome is NativeCellOutcome.VERIFIED


def test_text_header_first_data_row_dropped_not_verified(tmp_path: Path) -> None:
    """Fable H1: text header band must not hide dropped first data row."""
    hdr = ("Libellé", "Montant1", "Montant2")
    rows = [("Effectifs", "12", "15"), ("Budget", "340", "360"), ("Taux", "7", "9")]
    doc = fitz.open()
    page = doc.new_page()
    y = 70
    for x, v in zip((60, 200, 260), hdr, strict=True):
        if v:
            page.insert_text((x, y), v, fontsize=10)
    y += 20
    for row in rows:
        page.insert_text((60, y), row[0], fontsize=10)
        page.insert_text((200, y), row[1], fontsize=10)
        page.insert_text((260, y), row[2], fontsize=10)
        y += 20
    region = fitz.Rect(50, 40, 330, y)
    page.draw_rect(region, width=0.8)
    pdf = tmp_path / "hdr.pdf"  # type: ignore[operator]
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    words = doc[0].get_text("words")
    doc.close()
    md = "| " + " | ".join(hdr) + " |\n| --- | --- | --- |\n"
    for row in rows[1:]:
        md += "| " + " | ".join(row) + " |\n"
    r = check_table_cell_binding(
        table_id="t0", words=words, markdown=md, region_bbox=region, spans=[]
    )
    assert r.outcome is not NativeCellOutcome.VERIFIED


def test_gapped_total_row_dropped_not_verified(tmp_path: Path) -> None:
    natt = [
        ("Recettes", "120", "150"),
        ("Dépenses", "340", "360"),
        ("", "", ""),
        ("Total", "460", "510"),
    ]
    doc = fitz.open()
    page = doc.new_page()
    y = 70
    for row in natt:
        if row[0]:
            page.insert_text((60, y), row[0], fontsize=10)
        if row[1]:
            page.insert_text((200, y), row[1], fontsize=10)
        if row[2]:
            page.insert_text((260, y), row[2], fontsize=10)
        y += 20
    region = fitz.Rect(50, 40, 330, y)
    page.draw_rect(region, width=0.8)
    pdf = tmp_path / "total_gap.pdf"  # type: ignore[operator]
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    words = doc[0].get_text("words")
    doc.close()
    md = "| Libellé | c1 | c2 |\n| --- | --- | --- |\n"
    md += "| Recettes | 120 | 150 |\n| Dépenses | 340 | 360 |\n"
    r = check_table_cell_binding(
        table_id="t0", words=words, markdown=md, region_bbox=region, spans=[]
    )
    assert r.outcome is not NativeCellOutcome.VERIFIED


def test_orchestrator_pair_regions_uses_bbox_tuples(tmp_path: Path) -> None:
    """Orchestrator passes ``[tuple(b.bbox) for b in located_boxes]`` into pair_regions."""
    from socr.tables.locate import locate_tables
    from socr.tables.native_contradiction import pair_regions

    doc = fitz.open()
    page = doc.new_page()
    top_rows = [("Alpha", "10", "20"), ("Beta", "30", "40")]
    low_rows = [("Alpha", "10", "20"), ("Beta", "30", "41")]
    for top, rows in ((40, top_rows), (200, low_rows)):
        y = top + 30
        for row in rows:
            page.insert_text((60, y), row[0], fontsize=10)
            page.insert_text((200, y), row[1], fontsize=10)
            page.insert_text((260, y), row[2], fontsize=10)
            y += 20
        page.draw_rect(fitz.Rect(50, top, 330, y), width=0.8)
    pdf = tmp_path / "two.pdf"  # type: ignore[operator]
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    page = doc[0]
    top = fitz.Rect(50, 40, 330, 110)
    low = fitz.Rect(50, 200, 330, 270)
    top_md = "| L | a | b |\n| --- | --- | --- |\n| Alpha | 10 | 20 |\n| Beta | 30 | 40 |\n"
    low_md = "| L | a | b |\n| --- | --- | --- |\n| Alpha | 10 | 20 |\n| Beta | 30 | 41 |\n"
    paired = pair_regions(page, [top_md, low_md], [tuple(top), tuple(low)])
    assert paired[0] == tuple(top) and paired[1] == tuple(low)
    doc.close()


def test_parse_failure_is_ambiguous_not_verified(tmp_path: Path) -> None:
    _, region, words = _draw_table_pdf(tmp_path, rows=[("Data", "1", "2")])
    for bad_md in ("not a table", "| L | v |\n| --- | --- |\n"):
        r = check_table_cell_binding(
            table_id="t0", words=words, markdown=bad_md, region_bbox=region, spans=[]
        )
        assert r.outcome is NativeCellOutcome.AMBIGUOUS


def test_two_tables_pairs_first_box_when_markdown_matches_first(tmp_path: Path) -> None:
    from socr.tables.native_contradiction import pair_regions

    doc = fitz.open()
    page = doc.new_page()
    top = fitz.Rect(40, 40, 280, 120)
    low = fitz.Rect(40, 200, 280, 280)
    page.draw_rect(top, width=0.5)
    page.draw_rect(low, width=0.5)
    for y, vals in [(70, ("Alpha", "10", "20")), (95, ("Beta", "30", "40"))]:
        x = 50
        for v in vals:
            page.insert_text((x, y), v, fontsize=10)
            x += 60
    for y, vals in [(230, ("Gamma", "99", "88")), (255, ("Delta", "77", "66"))]:
        x = 50
        for v in vals:
            page.insert_text((x, y), v, fontsize=10)
            x += 60
    pdf = tmp_path / "two_tables.pdf"  # type: ignore[operator]
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    page = doc[0]
    words = page.get_text("words")
    top_md = "| L | a | b |\n| --- | --- | --- |\n| Alpha | 10 | 20 |\n| Beta | 30 | 40 |\n"
    low_md = "| L | a | b |\n| --- | --- | --- |\n| Gamma | 99 | 88 |\n| Delta | 77 | 66 |\n"
    boxes = [top, low]
    paired = pair_regions(page, [top_md, low_md], boxes)
    assert paired[0] is top
    assert paired[1] is low
    r = check_table_cell_binding(
        table_id="t0", words=words, markdown=top_md, region_bbox=top, spans=[]
    )
    doc.close()
    assert r.outcome is NativeCellOutcome.VERIFIED


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
