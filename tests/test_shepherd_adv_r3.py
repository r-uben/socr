"""Shepherd round-3 adversarial probes (Fable/Astra PR #1206 review).

One pytest per case from ``adv_r3.py`` so the next review round can converge on
the same 50 scenarios.
"""

from __future__ import annotations

import os
from pathlib import Path

import fitz
import pytest

from socr.tables.binding import BindingResult
from socr.tables.locate import locate_tables
from socr.tables.native_cell_check import (
    NativeCellOutcome,
    check_table_cell_binding,
    classify_binding_for_cell_check,
)
from socr.tables.native_contradiction import pair_regions

V = NativeCellOutcome.VERIFIED
C = NativeCellOutcome.CONTRADICTED
A = NativeCellOutcome.AMBIGUOUS
NOT_V = (C, A)


class _AdvFixtures:
    """Lazy-built PDF fixtures shared across Shepherd cases."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self._cache: dict[str, tuple[list, fitz.Rect]] = {}

    def draw(
        self,
        page,
        rows,
        *,
        top: float = 40,
        step: float = 20,
        colx=(60, 200, 260),
        header=None,
        border: bool = True,
        grid: bool = False,
    ) -> fitz.Rect:
        y = top + 30
        if header:
            for x, v in zip(colx, header, strict=True):
                if v:
                    page.insert_text((x, y), v, fontsize=10)
            y += step
        for vals in rows:
            for x, v in zip(colx, vals, strict=True):
                if v:
                    page.insert_text((x, y), v, fontsize=10)
            y += step
        rect = fitz.Rect(50, top, 330, y)
        if border:
            page.draw_rect(rect, width=0.8)
        if grid:
            yy = top
            while yy <= y:
                page.draw_line((50, yy), (330, yy), width=0.6)
                yy += step
            for gx in (50, 190, 250, 330):
                page.draw_line((gx, top), (gx, y), width=0.6)
        return rect

    def mkpdf(
        self,
        key: str,
        rows,
        **kw,
    ) -> tuple[list, fitz.Rect]:
        if key in self._cache:
            return self._cache[key]
        p = self.tmp / f"{key}.pdf"
        doc = fitz.open()
        page = doc.new_page()
        rect = self.draw(page, rows, **kw)
        doc.save(str(p))
        doc.close()
        d = fitz.open(str(p))
        words = d[0].get_text("words")
        d.close()
        self._cache[key] = (words, rect)
        return words, rect

    @staticmethod
    def md(rows, header=("Libellé", "c1", "c2")) -> str:
        n = len(header)
        s = "| " + " | ".join(header) + " |\n|" + " --- |" * n + "\n"
        for r in rows:
            s += "| " + " | ".join(r) + " |\n"
        return s

    def run(
        self,
        words,
        rect,
        markdown: str,
        ok: tuple[NativeCellOutcome, ...],
        *,
        region: bool = True,
    ) -> None:
        r = check_table_cell_binding(
            table_id="t0",
            words=words,
            markdown=markdown,
            region_bbox=(tuple(rect) if region else None),
            spans=[],
        )
        assert r.outcome in ok, (
            f"outcome={r.outcome.value} want {[o.value for o in ok]} "
            f"fully_checked={getattr(r.binding, 'fully_checked', None)} "
            f"structural={getattr(r.binding, 'structural_agreement', None)}"
        )


@pytest.fixture
def adv(tmp_path: Path) -> _AdvFixtures:
    return _AdvFixtures(tmp_path)


NAT = [("Effectifs", "12", "15"), ("Budget", "340", "360"), ("Taux", "7", "9")]
TOP = [("Alpha", "10", "20"), ("Beta", "30", "40")]
LOW = [("Alpha", "10", "20"), ("Beta", "30", "41")]


# --- B1 dropped rows ---


def test_shepherd_B1_ctl(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md(NAT), (V,))


def test_shepherd_B1a_drop_last(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md(NAT[:2]), NOT_V)


def test_shepherd_B1a_mid(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md([NAT[0], NAT[2]]), NOT_V)


def test_shepherd_B1a_first(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md(NAT[1:]), NOT_V)


def test_shepherd_B1a_page(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md(NAT[:2]), NOT_V, region=False)


def test_shepherd_B1a_30pt(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small_30", NAT, step=30)
    adv.run(w, r, adv.md(NAT[:2]), NOT_V)


def test_shepherd_B1a_total_ctl(adv: _AdvFixtures) -> None:
    natt = [
        ("Recettes", "120", "150"),
        ("Dépenses", "340", "360"),
        ("", "", ""),
        ("Total", "460", "510"),
    ]
    wt, rt = adv.mkpdf("p_total_gap", natt)
    adv.run(wt, rt, adv.md(natt[:2] + [natt[3]]), (V,))


def test_shepherd_B1a_total(adv: _AdvFixtures) -> None:
    natt = [
        ("Recettes", "120", "150"),
        ("Dépenses", "340", "360"),
        ("", "", ""),
        ("Total", "460", "510"),
    ]
    wt, rt = adv.mkpdf("p_total_gap", natt)
    adv.run(wt, rt, adv.md(natt[:2]), NOT_V)


def test_shepherd_B1b(adv: _AdvFixtures) -> None:
    naty = [
        ("Effectifs", "1200", "1500"),
        ("Année de base", "2019", "2020"),
        ("Budget", "3400", "3600"),
    ]
    wy, ry = adv.mkpdf("p_year", naty)
    adv.run(wy, ry, adv.md([naty[0], naty[2]]), NOT_V)


def test_shepherd_B1b_first(adv: _AdvFixtures) -> None:
    naty1 = [
        ("Année de base", "2019", "2020"),
        ("Effectifs", "1200", "1500"),
        ("Budget", "3400", "3600"),
    ]
    wy1, ry1 = adv.mkpdf("p_year_first", naty1)
    adv.run(wy1, ry1, adv.md(naty1[1:]), NOT_V)


def test_shepherd_B1c(adv: _AdvFixtures) -> None:
    natf = [
        ("Effectifs", "1200", "1500"),
        ("** dont Etat", "800", "900"),
        ("Budget", "3400", "3600"),
    ]
    wf, rf = adv.mkpdf("p_foot", natf)
    adv.run(wf, rf, adv.md([natf[0], natf[2]]), NOT_V)


def test_shepherd_B1c_last(adv: _AdvFixtures) -> None:
    natf2 = [
        ("Effectifs", "1200", "1500"),
        ("Budget", "3400", "3600"),
        ("** dont Etat", "800", "900"),
    ]
    wf2, rf2 = adv.mkpdf("p_foot_last", natf2)
    adv.run(wf2, rf2, adv.md(natf2[:2]), NOT_V)


def test_shepherd_ctl_wrong(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md([NAT[0], NAT[1], ("Taux", "7", "8")]), (C,))


def test_shepherd_ctl_invent(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md(NAT + [("Solde", "4", "5")]), NOT_V)


def test_shepherd_ctl_wrongyear(adv: _AdvFixtures) -> None:
    naty = [
        ("Effectifs", "1200", "1500"),
        ("Année de base", "2019", "2020"),
        ("Budget", "3400", "3600"),
    ]
    wy, ry = adv.mkpdf("p_year", naty)
    adv.run(wy, ry, adv.md([naty[0], ("Année de base", "2018", "2020"), naty[2]]), (C,))


# --- B2 paren negatives ---


def test_shepherd_B2_ctl(adv: _AdvFixtures) -> None:
    natn = [("Résultat", "(1,250)", "300"), ("Budget", "3400", "3600")]
    wn, rn = adv.mkpdf("p_paren", natn)
    adv.run(wn, rn, adv.md(natn), (V, A))


def test_shepherd_B2(adv: _AdvFixtures) -> None:
    natn = [("Résultat", "(1,250)", "300"), ("Budget", "3400", "3600")]
    wn, rn = adv.mkpdf("p_paren", natn)
    adv.run(wn, rn, adv.md([("Résultat", "1,250", "300"), natn[1]]), NOT_V)


def test_shepherd_B2b(adv: _AdvFixtures) -> None:
    natn = [("Résultat", "(1,250)", "300"), ("Budget", "3400", "3600")]
    wn, rn = adv.mkpdf("p_paren", natn)
    adv.run(wn, rn, adv.md([("Résultat", "-1,250", "300"), natn[1]]), (V, A))


def test_shepherd_B2c(adv: _AdvFixtures) -> None:
    natn = [("Résultat", "(0,4)", "300"), ("Budget", "3400", "3600")]
    w, r = adv.mkpdf("p_paren_s", natn)
    adv.run(w, r, adv.md([("Résultat", "0,4", "300"), natn[1]]), NOT_V)


# --- B3 spacer ---


def test_shepherd_B3(adv: _AdvFixtures) -> None:
    nats = [("Recettes", "1200", "1500"), ("Dépenses", "3400", "3600"), ("Solde", "-2200", "-2100")]
    ws, rs = adv.mkpdf("p_spacer", nats)
    adv.run(ws, rs, adv.md([nats[0], ("", "", ""), nats[1], nats[2]]), (V, A))


def test_shepherd_B3_wrong(adv: _AdvFixtures) -> None:
    nats = [("Recettes", "1200", "1500"), ("Dépenses", "3400", "3600"), ("Solde", "-2200", "-2100")]
    ws, rs = adv.mkpdf("p_spacer", nats)
    adv.run(
        ws,
        rs,
        adv.md([nats[0], nats[1], ("", "", ""), ("Solde", "-2100", "-2200")]),
        NOT_V,
    )


def test_shepherd_B3_wrong2(adv: _AdvFixtures) -> None:
    nats = [("Recettes", "1200", "1500"), ("Dépenses", "3400", "3600"), ("Solde", "-2200", "-2100")]
    ws, rs = adv.mkpdf("p_spacer", nats)
    adv.run(
        ws,
        rs,
        adv.md([nats[0], ("", "", ""), ("Dépenses", "3400", "3700"), nats[2]]),
        NOT_V,
    )


# --- B4 ---


def test_shepherd_a1(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md([NAT[0], NAT[1], ("Taux", "", "")]), NOT_V)


def test_shepherd_a2(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md([NAT[0], NAT[1], ("Taux", "n.d.", "n.d.")]), NOT_V)


def test_shepherd_a3(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md([NAT[0], NAT[1], ("Taux", "-", "-")]), NOT_V)


def test_shepherd_a4(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md([NAT[0], NAT[1], ("Taux", "7", "")]), NOT_V)


def test_shepherd_a5(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md([(a, b) for a, b, _ in NAT], header=("Libellé", "c1")), NOT_V)


def test_shepherd_a6(adv: _AdvFixtures) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    adv.run(w, r, adv.md([NAT[0], NAT[1], ("Total", "7", "9")]), NOT_V)


# --- B5 multi-table (orchestrator-like) ---


def _two_tables_pdf(tmp_path: Path, top_rows, low_rows):
    p = tmp_path / "p_two.pdf"
    doc = fitz.open()
    page = doc.new_page()
    _AdvFixtures(tmp_path).draw(page, top_rows, top=40, grid=True)
    _AdvFixtures(tmp_path).draw(page, low_rows, top=200, grid=True)
    doc.save(str(p))
    doc.close()
    d = fitz.open(str(p))
    return d, d[0]


def _orch_like(page, mds, ok_outcomes):
    boxes = locate_tables(page)
    words = page.get_text("words")
    paired = pair_regions(page, mds, [tuple(b.bbox) for b in boxes])
    outs = []
    for i, m in enumerate(mds):
        pick = paired[i] if i < len(paired) else None
        if pick is None:
            outs.append(A)
            continue
        res = check_table_cell_binding(
            table_id=f"t{i}",
            words=words,
            markdown=m,
            region_bbox=pick,
            spans=[],
        )
        outs.append(res.outcome)
    for o in outs:
        assert o in ok_outcomes


@pytest.mark.parametrize(
    "tag,mds,ok",
    [
        ("b-ctl", [None], (V, A)),  # filled in test
        ("b-tie1", [None], NOT_V),
        ("b-tie2", [None, None], NOT_V),
        ("b-sub", [None], NOT_V),
        ("b-low-wrong", [None], (V, A)),
    ],
)
def test_shepherd_B5_orch(adv: _AdvFixtures, tmp_path: Path, tag: str, mds, ok) -> None:
    d, pg = _two_tables_pdf(tmp_path, TOP, LOW)
    try:
        if tag == "b-ctl":
            mds = [adv.md(TOP)]
        elif tag == "b-tie1":
            mds = [adv.md([TOP[0], ("Beta", "30", "99")])]
        elif tag == "b-tie2":
            mds = [adv.md([TOP[0], ("Beta", "30", "99")]), adv.md([TOP[0], ("Beta", "30", "98")])]
        elif tag == "b-sub":
            mds = [adv.md([TOP[0]])]
        else:
            mds = [adv.md(TOP)]
        _orch_like(pg, mds, ok)
    finally:
        d.close()


# --- parse failure ---


@pytest.mark.parametrize(
    "markdown",
    [
        "| a | b |\n| x | 1 |\n",
        "not a table at all",
        "| L | v |\n| --- | --- |\n",
    ],
)
def test_shepherd_parse_failure_check(adv: _AdvFixtures, markdown: str) -> None:
    w, r = adv.mkpdf("p_small", NAT)
    rr = check_table_cell_binding(
        table_id="t0", words=w, markdown=markdown, region_bbox=tuple(r), spans=[]
    )
    assert rr.outcome is A


def test_shepherd_c4_classify() -> None:
    br = BindingResult(
        matched_cells=[],
        contradicted_cells=[],
        row_label_contradictions=[],
        native_unbound=[],
        model_unbound=[],
        ambiguous_count=0,
        row_binding={0: 0},
        native_rows=[],
        column_binding_unverifiable=True,
        row_binding_unverifiable=True,
    )
    assert classify_binding_for_cell_check(br, "garbage") is A


# --- b2 pair_regions with bbox tuples ---


def _b2_boxes(pg):
    _lb = locate_tables(pg)
    boxes = sorted(
        [b for b in _lb if (b.bbox[3] - b.bbox[1]) < 150],
        key=lambda b: b.bbox[1],
    )
    return [tuple(b.bbox) for b in boxes]


def test_shepherd_b2_unique(adv: _AdvFixtures, tmp_path: Path) -> None:
    d, pg = _two_tables_pdf(tmp_path, TOP, LOW)
    try:
        boxes_t = _b2_boxes(pg)
        paired = pair_regions(pg, [adv.md(TOP), adv.md(LOW)], boxes_t)
        got = [None if p_ is None else ("top" if p_ is boxes_t[0] else "low") for p_ in paired]
        assert got == ["top", "low"]
    finally:
        d.close()


def test_shepherd_b2_tie(adv: _AdvFixtures, tmp_path: Path) -> None:
    d, pg = _two_tables_pdf(tmp_path, TOP, LOW)
    try:
        boxes_t = _b2_boxes(pg)
        mds = [adv.md([TOP[0], ("Beta", "30", "99")]), adv.md(LOW)]
        paired = pair_regions(pg, mds, boxes_t)
        got = [None if p_ is None else ("top" if p_ is boxes_t[0] else "low") for p_ in paired]
        assert got == [None, "low"]
    finally:
        d.close()


def test_shepherd_b2_both_tie(adv: _AdvFixtures, tmp_path: Path) -> None:
    d, pg = _two_tables_pdf(tmp_path, TOP, LOW)
    try:
        boxes_t = _b2_boxes(pg)
        mds = [
            adv.md([TOP[0], ("Beta", "30", "99")]),
            adv.md([TOP[0], ("Beta", "30", "98")]),
        ]
        paired = pair_regions(pg, mds, boxes_t)
        got = [None if p_ is None else ("top" if p_ is boxes_t[0] else "low") for p_ in paired]
        assert got == [None, None]
    finally:
        d.close()


def test_shepherd_b2_claim(adv: _AdvFixtures, tmp_path: Path) -> None:
    d, pg = _two_tables_pdf(tmp_path, TOP, LOW)
    try:
        boxes_t = _b2_boxes(pg)
        paired = pair_regions(pg, [adv.md(TOP), adv.md(TOP)], boxes_t)
        got = [None if p_ is None else ("top" if p_ is boxes_t[0] else "low") for p_ in paired]
        assert got == [None, None]
    finally:
        d.close()


# --- H text header ---


HDR = ("Libellé", "Montant1", "Montant2")


def test_shepherd_H_ctl(adv: _AdvFixtures) -> None:
    wh, rh = adv.mkpdf("p_hdr", NAT, header=HDR)
    adv.run(wh, rh, adv.md(NAT, header=HDR), (V,))


def test_shepherd_H1(adv: _AdvFixtures) -> None:
    wh, rh = adv.mkpdf("p_hdr", NAT, header=HDR)
    adv.run(wh, rh, adv.md(NAT[1:], header=HDR), NOT_V)


def test_shepherd_H2(adv: _AdvFixtures) -> None:
    wh, rh = adv.mkpdf("p_hdr", NAT, header=HDR)
    adv.run(wh, rh, adv.md(NAT[:2], header=HDR), NOT_V)


def test_shepherd_H3(adv: _AdvFixtures) -> None:
    why, rhy = adv.mkpdf("p_hdry", NAT, header=("", "2024", "2025"))
    adv.run(why, rhy, adv.md(NAT[1:], header=("Libellé", "2024", "2025")), NOT_V)


# --- D / Y blank stub ---


COLX4 = (60, 170, 230, 290)
N4 = [("R1", "10", "", "20"), ("R2", "30", "40", "50"), ("R3", "60", "70", "80")]
H4 = ("L", "a", "b", "c")
NY = [("r1", "1", "2"), ("", "2020", "2021"), ("r2", "3", "4")]


def test_shepherd_D_ctl(adv: _AdvFixtures) -> None:
    w4, r4 = adv.mkpdf("p_blank", N4, colx=COLX4)
    adv.run(
        w4,
        r4,
        adv.md([("R1", "10", "-", "20"), N4[1], N4[2]], header=H4),
        (V, A),
    )


def test_shepherd_D1(adv: _AdvFixtures) -> None:
    w4, r4 = adv.mkpdf("p_blank", N4, colx=COLX4)
    adv.run(
        w4,
        r4,
        adv.md([("R1", "-", "10", "20"), N4[1], N4[2]], header=H4),
        NOT_V,
    )


def test_shepherd_Y_ctl(adv: _AdvFixtures) -> None:
    wyb, ryb = adv.mkpdf("p_blankstub", NY)
    adv.run(wyb, ryb, adv.md(NY), (V, A))


def test_shepherd_Y4(adv: _AdvFixtures) -> None:
    wyb, ryb = adv.mkpdf("p_blankstub", NY)
    adv.run(wyb, ryb, adv.md([NY[0], ("", "2021", "2020"), NY[2]]), NOT_V)


def test_shepherd_Y5(adv: _AdvFixtures) -> None:
    wyb, ryb = adv.mkpdf("p_blankstub", NY)
    adv.run(
        wyb,
        ryb,
        adv.md(
            [("r1", "1", "2", ""), ("", "2021", "2020", ""), ("r2", "3", "4", "")],
            header=("L", "c1", "c2", "c3"),
        ),
        NOT_V,
    )
