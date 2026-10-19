"""Shepherd round-4 adversarial probes (Fable/Astra PR #1206 @ bdb7e427+).

Blocker-focused cases: duplicate model tables, nested label swaps, gapped totals,
orchestrator pairing/skip path.
"""

from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path

import fitz
import pytest

from socr.core.manifest import UNVERIFIED_TABLE_MARK
from socr.judge.table_verdict import NATIVE_CELL_CHECK_SKIPPED_KIND
from socr.tables.locate import locate_tables
from socr.tables.native_cell_check import (
    NativeCellOutcome,
    check_table_cell_binding,
)
from socr.tables.native_contradiction import pair_regions

from test_shepherd_adv_r3 import (
    A,
    NOT_V,
    V,
    _AdvFixtures,
    _two_tables_pdf,
    TOP,
    LOW,
)

pytest.importorskip("fitz")


@pytest.fixture
def adv(tmp_path: Path) -> _AdvFixtures:
    return _AdvFixtures(tmp_path)


def _md(rows, header=("Libellé", "c1", "c2")) -> str:
    n = len(header)
    s = "| " + " | ".join(header) + " |\n|" + " --- |" * n + "\n"
    for r in rows:
        s += "| " + " | ".join(r) + " |\n"
    return s


def test_r4_T2_total_dropped_far_below_not_verified(adv: _AdvFixtures) -> None:
    """Total row 130pt below body must not verify when dropped (T2-total)."""
    tb = [
        ("Recettes A", "120", "150"),
        ("Recettes B", "340", "360"),
        ("Sous-total", "460", "510"),
        ("Dépenses C", "200", "210"),
        ("Dépenses D", "100", "110"),
        ("Total", "760", "830"),
    ]
    ys = [70, 85, 100, 115, 130, 260]
    w, r = adv.mkpdf("T2", tb, ys=ys)
    adv.run(w, r, _md(tb[:-1]), NOT_V)


def test_r4_solde_row_far_below_not_verified(adv: _AdvFixtures) -> None:
    rows = [("Budget", "340", "360"), ("Solde", "4", "5")]
    ys = [70, 130]
    w, r = adv.mkpdf("solde_gap", rows, ys=ys)
    adv.run(w, r, _md([rows[0]]), NOT_V)


def _mk_nested_pdf(tmp_path: Path):
    rows = [
        ("Dépenses", "", ""),
        ("  dont Etat", "100", "200"),
        ("  dont Collectivités", "60", "80"),
    ]
    doc = fitz.open()
    page = doc.new_page()
    y = 70
    for lab, a, b in rows:
        x = 75 if lab.startswith("  ") else 60
        page.insert_text((x, y), lab.strip(), fontsize=10)
        if a:
            page.insert_text((200, y), a, fontsize=10)
        if b:
            page.insert_text((260, y), b, fontsize=10)
        y += 20
    rect = fitz.Rect(50, 40, 330, y)
    page.draw_rect(rect, width=0.8)
    pdf = tmp_path / "nested.pdf"
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    words = doc[0].get_text("words")
    doc.close()
    return words, rect


def test_r4_nested_child_label_swap_not_verified(tmp_path: Path) -> None:
    words, rect = _mk_nested_pdf(tmp_path)
    swapped = _md(
        [
            ("Dépenses", "", ""),
            ("dont Collectivités", "100", "200"),
            ("dont Etat", "60", "80"),
        ]
    )
    r = check_table_cell_binding(
        table_id="t0", words=words, markdown=swapped, region_bbox=tuple(rect), spans=[]
    )
    assert r.outcome is not NativeCellOutcome.VERIFIED


def test_r4_nested_swap_extra_column_not_verified(tmp_path: Path) -> None:
    words, rect = _mk_nested_pdf(tmp_path)
    swapped = _md(
        [
            ("Dépenses", "", "", ""),
            ("dont Collectivités", "100", "200", ""),
            ("dont Etat", "60", "80", ""),
        ],
        header=("L", "a", "b", "c"),
    )
    r = check_table_cell_binding(
        table_id="t0", words=words, markdown=swapped, region_bbox=tuple(rect), spans=[]
    )
    assert r.outcome is not NativeCellOutcome.VERIFIED


def test_r4_duplicate_model_blocks_do_not_both_verify(tmp_path: Path) -> None:
    d, pg = _two_tables_pdf(tmp_path, TOP, LOW)
    try:
        mds = [_md(TOP), _md(TOP)]
        boxes = locate_tables(pg)
        paired = pair_regions(pg, mds, [tuple(b.bbox) for b in boxes])
        assert paired[0] is None or paired[1] is None
        verified = 0
        for pick, m in zip(paired, mds, strict=True):
            if pick is None:
                continue
            res = check_table_cell_binding(
                table_id="t0",
                words=pg.get_text("words"),
                markdown=m,
                region_bbox=pick,
                spans=[],
            )
            if res.outcome is NativeCellOutcome.VERIFIED:
                verified += 1
        assert verified <= 1
    finally:
        d.close()


def test_r4_M_dupTOP_orch_not_verified(tmp_path: Path) -> None:
    d, pg = _two_tables_pdf(tmp_path, TOP, LOW)
    try:
        outs: list[NativeCellOutcome] = []
        boxes = locate_tables(pg)
        words = pg.get_text("words")
        paired = pair_regions(pg, [_md(TOP), _md(TOP)], [tuple(b.bbox) for b in boxes])
        for pick, m in zip(paired, [_md(TOP), _md(TOP)], strict=True):
            if pick is None:
                outs.append(A)
                continue
            outs.append(
                check_table_cell_binding(
                    table_id="t0",
                    words=words,
                    markdown=m,
                    region_bbox=pick,
                    spans=[],
                ).outcome
            )
        assert all(o is not V for o in outs)
    finally:
        d.close()


def test_r4_orchestrator_skip_pairing_inconclusive_e2e(tmp_path: Path) -> None:
    from unittest.mock import patch

    import socr.tables.witness as W
    import test_resf_native_cell_pipeline as pipe

    md = (pipe.FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")

    def _boxless(real):
        @contextlib.contextmanager
        def cm(*args, **kwargs):
            with real(*args, **kwargs) as ws:
                yield [dataclasses.replace(w, box=None) for w in ws]

        return cm

    with (
        patch.object(W, "prepare_table_witnesses", _boxless(W.prepare_table_witnesses)),
        patch(
            "socr.tables.native_contradiction.pair_regions",
            side_effect=lambda _page, mds, _boxes: [None] * len(mds),
        ),
    ):
        _pipe, _result, state, out_dir = pipe._run_page(
            tmp_path, "resf_p59.pdf", md, native_cell_gating=True
        )

    sidecar = pipe._page_sidecar(out_dir, 1)
    win = sidecar["winning_output"]
    assert win["status"] == "warning"
    assert win["failure_mode"] == "table_unverified"
    assert UNVERIFIED_TABLE_MARK in win["text"]
    kinds = [e["kind"] for e in pipe._audit_events(out_dir) if e.get("page_num") == 1]
    assert NATIVE_CELL_CHECK_SKIPPED_KIND in kinds
    assert all(v.get("outcome") != "verified" for v in state.pages[1].native_cell_binding.values())


def test_r4_orchestrator_duplicate_tables_e2e(tmp_path: Path) -> None:
    import test_resf_native_cell_pipeline as pipe

    top = [("Alpha", "10", "20"), ("Beta", "30", "40")]
    low = [("Gamma", "50", "60"), ("Delta", "70", "80")]

    def _synth(name: str, tables: list) -> Path:
        pdf = tmp_path / name
        doc = fitz.open()
        page = doc.new_page()
        for i, rows in enumerate(tables):
            top_y = 40 + 160 * i
            y = top_y + 30
            for row in rows:
                for x, v in zip((60, 200, 260), row, strict=True):
                    page.insert_text((x, y), v, fontsize=10)
                y += 20
            page.draw_rect(fitz.Rect(50, top_y, 330, y), width=0.8)
            for yy in range(top_y + 15, int(y), 20):
                page.draw_line((50, yy), (330, yy), width=0.6)
        doc.save(str(pdf))
        doc.close()
        return pdf

    pdf = _synth("dup_top_x2.pdf", [top, low])
    md = "Intro\n\n" + _md(top) + "\n\n" + _md(top)
    _pipe, _result, state, out_dir = pipe._run_page(tmp_path, str(pdf), md)
    outcomes = {v.get("outcome") for v in state.pages[1].native_cell_binding.values()}
    assert "verified" not in outcomes or len(outcomes) > 1
    verified_count = sum(
        1 for v in state.pages[1].native_cell_binding.values() if v.get("outcome") == "verified"
    )
    assert verified_count == 0
