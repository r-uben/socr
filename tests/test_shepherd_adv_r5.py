"""Shepherd round-5 adversarial probes (PR #1206 fail-closed pass).

Ports adv_r5.py, adv_r5b/c, p59both, and orchestrator e2e cases from the review scripts.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from socr.core.manifest import UNVERIFIED_TABLE_MARK

FIXTURES = Path(__file__).parent / "fixtures" / "resf_native_cell"
from socr.judge.table_verdict import NATIVE_CELL_CHECK_SKIPPED_KIND
from socr.tables.locate import locate_tables
from socr.tables.native_cell_check import NativeCellOutcome, check_table_cell_binding
from socr.tables.native_contradiction import (
    locator_boxes_for_pairing,
    pair_regions,
    unclaimed_locator_box_indices,
)

from test_shepherd_adv_r3 import A, C, NOT_V, V, _AdvFixtures, TOP

pytest.importorskip("fitz")

H3 = ("Libellé", "c1", "c2")
# adv_r5.py naming: LOW = distinct second table; LOWs = near-duplicate of TOP (Beta 41).
LOWS = [("Alpha", "10", "20"), ("Beta", "30", "41")]
LOW = [("Gamma", "50", "60"), ("Delta", "70", "80")]


def _mdx(rows, extra: bool) -> str:
    header = ("Libellé", "c1", "c2", "c3") if extra else H3
    body = [r + ("",) for r in rows] if extra else rows
    n = len(header)
    s = "| " + " | ".join(header) + " |\n|" + " --- |" * n + "\n"
    for r in body:
        s += "| " + " | ".join(r) + " |\n"
    return s


def _mk_nested_pdf(tmp_path: Path, name: str, rows):
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
    pdf = tmp_path / name
    doc.save(str(pdf))
    doc.close()
    doc = fitz.open(str(pdf))
    words = doc[0].get_text("words")
    doc.close()
    return words, rect


def _mdn(rows, extra: bool) -> str:
    flat = [(lab.strip(), a, b) for lab, a, b in rows]
    return _mdx(flat, extra)


def _orch_outcomes(pg, mds: list[str]):
    raw = [tuple(b.bbox) for b in locate_tables(pg)]
    kept = locator_boxes_for_pairing(pg, raw)
    words = pg.get_text("words")
    pr = pair_regions(pg, mds, raw)
    un = unclaimed_locator_box_indices(pg, mds, raw, pr)
    outs: list[tuple[str, NativeCellOutcome]] = []
    for m, pick in zip(mds, pr, strict=True):
        if pick is None:
            outs.append(("skip", A))
            continue
        res = check_table_cell_binding(
            table_id="t",
            words=words,
            markdown=m,
            region_bbox=pick,
            spans=[],
            locator_boxes=raw,
            page=pg,
        )
        outs.append((f"box{kept.index(pick)}", res.outcome))
    return raw, kept, outs, un


def _drawfull(page, rows, top, x0=50, x1=330):
    y = top + 30
    for vals in rows:
        for x, v in zip((x0 + 10, x0 + 150, x0 + 210), vals, strict=True):
            page.insert_text((x, y), v, fontsize=10)
        y += 20
    page.draw_rect(fitz.Rect(x0, top, x1, y), width=0.8)
    for yy in range(top + 15, int(y), 20):
        page.draw_line((x0, yy), (x1, yy), width=0.6)
    for gx in (x0 + 140, x0 + 200):
        page.draw_line((gx, top), (gx, y), width=0.6)
    return fitz.Rect(x0, top, x1, y)


def _page_of(tmp_path: Path, name: str, tables: list, extra=None):
    pdf = tmp_path / name
    doc = fitz.open()
    pg = doc.new_page()
    for i, t in enumerate(tables):
        _drawfull(pg, t, 40 + 160 * i)
    if extra is not None:
        extra(pg)
    doc.save(str(pdf))
    doc.close()
    d = fitz.open(str(pdf))
    return d, d[0]


def _build_adv_r5_cell_cases() -> list[tuple[str, tuple[NativeCellOutcome, ...]]]:
    """Case ids and expected outcome sets (adv_r5 cell-check section)."""
    cases: list[tuple[str, tuple[NativeCellOutcome, ...]]] = []
    pairs = [
        ("P1", "Etat", "Etat et opérateurs"),
        ("P2", "Dépenses", "Dépenses totales"),
        ("P3", "dont Etat", "dont Etat et ODAC"),
        ("P4", "Recettes", "Recettes nettes"),
        ("P5", "Prestations", "Prestations sociales"),
        ("P6", "Total", "Total général"),
        ("P7", "Solde", "Solde public"),
        ("P8", "A", "Autres"),
        ("P9", "Etat", "Etats-Unis"),
    ]
    for tag, a, b in pairs:
        cases.append((f"{tag}-ctl", (V, A)))
        cases.append((f"{tag}-sw", NOT_V))
        cases.append((f"{tag}-swx", NOT_V))
        cases.append((f"{tag}-renx", NOT_V))
    bl = [("Etat", "100", "200"), ("Collectivités", "60", "80"), ("Sécurité sociale", "30", "40")]
    cases.extend(
        [
            ("BL-ctl", (V, A)),
            ("BL-blank", NOT_V),
            ("BL-swapblank", NOT_V),
        ]
    )
    frows = [
        ("F1", "(1) Prestations sociales"),
        ("F2", "(1) Dépenses"),
        ("F3", "* dont Etat"),
        ("F4", "**Total**"),
        ("F5", "** Cotisations"),
        ("F6", "Note : total"),
        ("F7", "Source : DB"),
        ("F8", "Champ : régime général"),
        ("F9", "Lecture : solde"),
        ("F10", "Etat (1)"),
        ("F11", "*Collectivités"),
        ("F12", "(2) Etat et ODAC"),
        ("F13", "**Solde**"),
        ("F14", "** dont Etat"),
    ]
    for tag, _lab in frows:
        ok_ctl: tuple[NativeCellOutcome, ...] = (V, A)
        cases.append((f"{tag}-ctl", ok_ctl))
        cases.append((f"{tag}-drop", NOT_V))
        cases.append((f"{tag}-droplast", NOT_V))
    for lab in (
        "Total",
        "Solde",
        "Note",
        "Source : DB",
        "Ensemble",
        "Total APU",
        "(1) Total",
        "Prestations",
    ):
        for gap in (30, 60, 130):
            cases.append((f"G-{lab}-{gap}", NOT_V))
    np1 = [
        ("Dépenses", "", ""),
        ("  Etat", "100", "200"),
        ("  Etat et opérateurs", "60", "80"),
        ("Recettes", "", ""),
        ("  dont TVA", "300", "400"),
    ]
    cases.extend(
        [
            ("PN1-ctl", (V, A)),
            ("PN1-sw", NOT_V),
            ("PN1-swx", NOT_V),
            ("PN1-parx", NOT_V),
            ("PN2-swx", NOT_V),
            ("PN2-sw", NOT_V),
        ]
    )
    _ = bl, np1, pairs  # referenced in runners below via ids
    return cases


ADV_R5_CELL_CASES = _build_adv_r5_cell_cases()


@pytest.fixture
def adv(tmp_path: Path) -> _AdvFixtures:
    return _AdvFixtures(tmp_path)


def _run_adv_r5_cell(adv: _AdvFixtures, case_id: str) -> None:
    pairs = [
        ("P1", "Etat", "Etat et opérateurs"),
        ("P2", "Dépenses", "Dépenses totales"),
        ("P3", "dont Etat", "dont Etat et ODAC"),
        ("P4", "Recettes", "Recettes nettes"),
        ("P5", "Prestations", "Prestations sociales"),
        ("P6", "Total", "Total général"),
        ("P7", "Solde", "Solde public"),
        ("P8", "A", "Autres"),
        ("P9", "Etat", "Etats-Unis"),
    ]
    for tag, a, b in pairs:
        rows = [(a, "100", "200"), (b, "60", "80"), ("Divers", "5", "6")]
        w, r = adv.mkpdf(f"{tag}.pdf", rows)
        sw = [(b, "100", "200"), (a, "60", "80"), ("Divers", "5", "6")]
        ren = [(a, "100", "200"), (a, "60", "80"), ("Divers", "5", "6")]
        mapping = {
            f"{tag}-ctl": _mdx(rows, False),
            f"{tag}-sw": _mdx(sw, False),
            f"{tag}-swx": _mdx(sw, True),
            f"{tag}-renx": _mdx(ren, True),
        }
        if case_id in mapping:
            ok: tuple[NativeCellOutcome, ...] = (
                NOT_V if ("sw" in case_id or "ren" in case_id) else (V, A)
            )
            adv.run(w, r, mapping[case_id], ok)
            return
    bl = [("Etat", "100", "200"), ("Collectivités", "60", "80"), ("Sécurité sociale", "30", "40")]
    if case_id == "BL-ctl":
        w, r = adv.mkpdf("BL.pdf", bl)
        adv.run(w, r, _mdx(bl, True), (V, A))
        return
    if case_id == "BL-blank":
        w, r = adv.mkpdf("BL.pdf", bl)
        adv.run(w, r, _mdx([("", "100", "200"), ("", "60", "80"), bl[2]], True), NOT_V)
        return
    if case_id == "BL-swapblank":
        w, r = adv.mkpdf("BL.pdf", bl)
        adv.run(w, r, _mdx([("", "100", "200"), ("Etat", "60", "80"), bl[2]], True), NOT_V)
        return
    frows = [
        ("F1", "(1) Prestations sociales"),
        ("F2", "(1) Dépenses"),
        ("F3", "* dont Etat"),
        ("F4", "**Total**"),
        ("F5", "** Cotisations"),
        ("F6", "Note : total"),
        ("F7", "Source : DB"),
        ("F8", "Champ : régime général"),
        ("F9", "Lecture : solde"),
        ("F10", "Etat (1)"),
        ("F11", "*Collectivités"),
        ("F12", "(2) Etat et ODAC"),
        ("F13", "**Solde**"),
        ("F14", "** dont Etat"),
    ]
    for tag, lab in frows:
        rows = [("Etat", "100", "200"), (lab, "60", "80"), ("Divers", "5", "6")]
        w, r = adv.mkpdf(f"{tag}.pdf", rows)
        if case_id == f"{tag}-ctl":
            adv.run(w, r, _mdx(rows, False), (V, A))
            return
        if case_id == f"{tag}-drop":
            adv.run(w, r, _mdx([rows[0], rows[2]], False), NOT_V)
            return
        if case_id == f"{tag}-droplast":
            w2, r2 = adv.mkpdf(f"{tag}L.pdf", [rows[0], rows[2], rows[1]])
            adv.run(w2, r2, _mdx([rows[0], rows[2]], False), NOT_V)
            return
    for lab in (
        "Total",
        "Solde",
        "Note",
        "Source : DB",
        "Ensemble",
        "Total APU",
        "(1) Total",
        "Prestations",
    ):
        for gap in (30, 60, 130):
            if case_id != f"G-{lab}-{gap}":
                continue
            rows = [
                ("Etat", "11", "21"),
                ("ODAC", "12", "22"),
                ("APUL", "13", "23"),
                (lab, "36", "66"),
            ]
            ys = [70, 85, 100, 100 + gap]
            w, r = adv.mkpdf(f"G{gap}.pdf", rows, ys=ys, step=15)
            adv.run(w, r, adv.md(rows[:3]), NOT_V)
            return
    np1 = [
        ("Dépenses", "", ""),
        ("  Etat", "100", "200"),
        ("  Etat et opérateurs", "60", "80"),
        ("Recettes", "", ""),
        ("  dont TVA", "300", "400"),
    ]
    w, r = _mk_nested_pdf(adv.tmp, "PN1.pdf", np1)
    sw = [np1[0], ("  Etat et opérateurs", "100", "200"), ("  Etat", "60", "80")] + np1[3:]
    par = [np1[0], ("  Dépenses", "100", "200")] + np1[2:]
    np2 = [("Dépenses", "", ""), ("  dont Etat", "100", "200"), ("  dont Etat et ODAC", "60", "80")]
    w2, r2 = _mk_nested_pdf(adv.tmp, "PN2.pdf", np2)
    sw2 = [np2[0], ("  dont Etat et ODAC", "100", "200"), ("  dont Etat", "60", "80")]
    nested_map = {
        "PN1-ctl": (w, r, _mdn(np1, False), (V, A)),
        "PN1-sw": (w, r, _mdn(sw, False), NOT_V),
        "PN1-swx": (w, r, _mdn(sw, True), NOT_V),
        "PN1-parx": (w, r, _mdn(par, True), NOT_V),
        "PN2-swx": (w2, r2, _mdn(sw2, True), NOT_V),
        "PN2-sw": (w2, r2, _mdn(sw2, False), NOT_V),
    }
    if case_id in nested_map:
        ww, rr, md, ok = nested_map[case_id]
        adv.run(ww, rr, md, ok)
        return
    raise AssertionError(f"unknown adv_r5 cell case {case_id}")


@pytest.mark.parametrize(
    ("case_id", "expected"),
    ADV_R5_CELL_CASES,
    ids=[c[0] for c in ADV_R5_CELL_CASES],
)
def test_adv_r5_cell_case(
    adv: _AdvFixtures, case_id: str, expected: tuple[NativeCellOutcome, ...]
) -> None:
    _run_adv_r5_cell(adv, case_id)
    # _run_adv_r5_cell asserts internally; expected is documentation for ids


D_PAIR_CASES = [
    ("D-ctl1", [TOP], [TOP], [(V,)], False),
    ("D-ctl2", [TOP, LOW], [TOP, LOW], [(V,), (V,)], False),
    ("D-ctl2s", [TOP, LOWS], [TOP, LOWS], [(V, A), (V, A)], False),
    ("D-ctl3", [TOP, LOW, LOWS], [TOP, LOW, LOWS], [(V, A), (V, A), (V, A)], False),
    ("D-dup1", [TOP], [TOP, TOP], [NOT_V, NOT_V], True),
    ("D-dup2", [TOP, LOW], [TOP, TOP], [NOT_V, NOT_V], True),
    ("D-only", [TOP, LOW], [TOP], [(V, A)], True),
    ("D-only3", [TOP, LOW, LOWS], [TOP, LOW], [(V, A), (V, A)], True),
    (
        "D-dup2L",
        [TOP, LOW],
        [TOP, TOP, LOW],
        [NOT_V, NOT_V, (V, A)],
        True,
    ),
    ("D-tie", [TOP, TOP], [TOP], [NOT_V], False),
]

D_SPECIAL_CASE_IDS = ("D-inner", "D-inner-drop")


@pytest.mark.parametrize(
    ("tag", "tabs", "blocks", "wants", "want_flag"),
    D_PAIR_CASES,
    ids=[c[0] for c in D_PAIR_CASES],
)
def test_adv_r5_pairing_case(
    tmp_path: Path,
    tag: str,
    tabs: list,
    blocks: list,
    wants: list,
    want_flag: bool,
) -> None:
    d, pg = _page_of(tmp_path, f"{tag}.pdf", tabs)
    try:
        mds = [adv_md(b) for b in blocks]
        _raw, _kept, outs, un = _orch_outcomes(pg, mds)
        if tag == "D-tie":
            assert outs[0][0] == "skip"
        assert all(o in w_ for (_, o), w_ in zip(outs, wants, strict=True))
        verified_boxes = [box_id for box_id, o in outs if o is V]
        assert len(verified_boxes) == len(set(verified_boxes))
        flagged = bool(un) or any(o is not V for _, o in outs)
        if want_flag:
            assert flagged
    finally:
        d.close()


def _inner_cell_frame(pg) -> None:
    pg.draw_rect(fitz.Rect(195, 75, 245, 95), width=0.8)


def _textbox_frame(pg) -> None:
    pg.draw_rect(fitz.Rect(50, 400, 330, 470), width=0.8)
    for i, t in enumerate(
        [
            "Encadré : la loi de finances 2025",
            "prévoit 3 mesures et 12 dispositifs",
            "applicables au 1er janvier.",
        ]
    ):
        pg.insert_text((60, 420 + i * 15), t, fontsize=10)


@pytest.mark.parametrize("case_id", D_SPECIAL_CASE_IDS)
def test_adv_r5_d_special_case(tmp_path: Path, case_id: str) -> None:
    """adv_r5 duplicate/sub-panel geometry cases (D-inner*, D-textbox)."""
    full = TOP + [("Gamma", "50", "60")]
    if case_id == "D-inner":
        d, pg = _page_of(tmp_path, "D-inner.pdf", [full], extra=_inner_cell_frame)
        try:
            mds = [adv_md(full)]
            raw, kept, outs, un = _orch_outcomes(pg, mds)
            o = outs[0][1]
            tall = len(kept) == 1 and kept[0][3] - kept[0][1] > 60
            assert o is not V or kept == raw or tall, (o, kept, raw, un)
        finally:
            d.close()
        return
    if case_id == "D-inner-drop":
        d, pg = _page_of(tmp_path, "D-inner2.pdf", [full], extra=_inner_cell_frame)
        try:
            _raw, _kept, outs, _un = _orch_outcomes(pg, [adv_md(TOP)])
            assert outs[0][1] is not V
        finally:
            d.close()
        return
    raise AssertionError(case_id)


def test_adv_r5_d_textbox_no_false_lost_table(tmp_path: Path) -> None:
    """adv_r5 D-textbox: framed prose beside one table must not create unclaimed regions."""
    d, pg = _page_of(tmp_path, "D-textbox.pdf", [TOP], extra=_textbox_frame)
    try:
        _raw, kept, outs, un = _orch_outcomes(pg, [adv_md(TOP)])
        assert outs[0][1] in (V, A)
        assert len(kept) >= 1
        assert not un
    finally:
        d.close()


def test_adv_r5_script_ports_one_hundred_twenty_three_ids() -> None:
    """adv_r5.py: 111 cell cases + 12 duplicate/geometry D cases."""
    assert len(ADV_R5_CELL_CASES) == 111
    assert len(D_PAIR_CASES) == 10
    assert len(D_SPECIAL_CASE_IDS) == 2


def adv_md(rows, header=("Libellé", "c1", "c2")) -> str:
    n = len(header)
    s = "| " + " | ".join(header) + " |\n|" + " --- |" * n + "\n"
    for r in rows:
        s += "| " + " | ".join(r) + " |\n"
    return s


def _nest_subpanel_pdf(tmp_path: Path) -> tuple[fitz.Document, fitz.Page, list]:
    """adv_r5c NEST: outer table + ruled inner panel over ~4 rows."""
    rows = [
        (n, str(10 + i), str(20 + i))
        for i, n in enumerate(
            ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota"], 1
        )
    ]
    doc = fitz.open()
    pg = doc.new_page()
    y = 70
    for r in rows:
        for x, v in zip((60, 200, 260), r, strict=True):
            pg.insert_text((x, y), v, fontsize=10)
        y += 20
    bottom = y - 10
    pg.draw_rect(fitz.Rect(50, 40, 330, bottom), width=0.8)
    ib = 40 + 85
    pg.draw_rect(fitz.Rect(50, 40, 330, ib), width=0.8)
    for yy in range(55, int(ib), 20):
        pg.draw_line((50, yy), (330, yy), width=0.5)
    for gx in (190, 250):
        pg.draw_line((gx, 40), (gx, ib), width=0.5)
    pdf = tmp_path / "nest.pdf"
    doc.save(str(pdf))
    doc.close()
    d = fitz.open(str(pdf))
    return d, d[0], rows


def test_adv_r5c_subpanel_only_not_verified(tmp_path: Path) -> None:
    """Outer table with rows outside ruled inner panel: inner-only markdown must not verify."""
    d, pg, rows = _nest_subpanel_pdf(tmp_path)
    try:
        _raw, _kept, outs, _un = _orch_outcomes(pg, [adv_md(rows[:3])])
        assert outs[0][1] is not V
    finally:
        d.close()


def test_adv_r5c_full_table_not_contradicted_by_inner_box(tmp_path: Path) -> None:
    """Full-table markdown must not be contradicted because a nested ruled box exists."""
    d, pg, rows = _nest_subpanel_pdf(tmp_path)
    try:
        _raw, _kept, outs, _un = _orch_outcomes(pg, [adv_md(rows)])
        assert outs[0][1] is not C
        assert outs[0][1] in (V, A)
    finally:
        d.close()


def test_r5_orchestrator_unpaired_native_region_e2e(tmp_path: Path) -> None:
    """``process`` on born-digital RESF p.59 with one table still runs native-cell pairing."""
    import test_resf_native_cell_pipeline as pipe

    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    _pipe, _result, state, out_dir = pipe._run_page(tmp_path, "resf_p59.pdf", md)
    ps = state.pages[1]
    assert ps.is_born_digital is True
    assert ps.native_cell_binding
    kinds = [e["kind"] for e in pipe._audit_events(out_dir) if e.get("page_num") == 1]
    assert "native_cell_check_verified" in kinds


def test_r5_native_cell_binding_unclaimed_region_recorded(tmp_path: Path) -> None:
    """Lost native region skip is emitted from the real orchestrator helper (unpaired geometry)."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import DocumentState, PageState
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = tmp_path / "two_tables_one_block.pdf"
    d, _pg = _page_of(tmp_path, "two_tables_one_block.pdf", [TOP, LOW])
    d.close()

    cfg = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        strict_local=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        table_judge_ladder=False,
        quiet=True,
    )
    pipeline = UnifiedPipeline(cfg)
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = PageState(page_num=1, is_born_digital=True, has_tables=True)
    state.pages[1] = ps
    bo = PageOutput(
        page_num=1,
        text="Intro\n\n" + adv_md(TOP),
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    pipeline._run_native_cell_binding_check(state, 1, ps, bo)
    assert any(k.startswith("p1-native-region-") for k in ps.native_cell_binding)
    assert any(v.get("skip_reason") for v in ps.native_cell_binding.values())


def test_r5_resf_p59_both_tableaux_cell_check() -> None:
    """RESF p.59 Tableau 1 + Tableau 2: pair, check cells, no unclaimed locator boxes."""
    doc = fitz.open(FIXTURES / "resf_p59.pdf")
    pg = doc[0]
    raw = [tuple(b.bbox) for b in locate_tables(pg)]
    words = pg.get_text("words")
    md1 = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    md2 = (FIXTURES / "p59_tableau2.md").read_text(encoding="utf-8")
    pr = pair_regions(pg, [md1, md2], raw)
    un = unclaimed_locator_box_indices(pg, [md1, md2], raw, pr)
    assert not un
    assert pr[0] is not None and pr[1] is not None
    peers0 = [tuple(pr[1])]
    peers1 = [tuple(pr[0])]
    o1 = check_table_cell_binding(
        table_id="t1",
        words=words,
        markdown=md1,
        region_bbox=pr[0],
        spans=[],
        peer_region_bboxes=peers0,
        locator_boxes=raw,
        page=pg,
    ).outcome
    o2 = check_table_cell_binding(
        table_id="t2",
        words=words,
        markdown=md2,
        region_bbox=pr[1],
        spans=[],
        peer_region_bboxes=peers1,
        locator_boxes=raw,
        page=pg,
    ).outcome
    assert o1 is V
    assert o2 in (V, A)
    doc.close()


def test_r5_resf_p59_both_tableaux_pipeline_no_lost_table(tmp_path: Path) -> None:
    import test_resf_native_cell_pipeline as pipe

    md1 = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    md2 = (FIXTURES / "p59_tableau2.md").read_text(encoding="utf-8")
    md = md1.strip() + "\n\n" + md2.strip()
    _pipe, _result, state, out_dir = pipe._run_page(tmp_path, "resf_p59.pdf", md)
    ps = state.pages[1]
    assert not any(k.startswith("p1-native-region-") for k in ps.native_cell_binding)
    outcomes = {k: v.get("outcome") for k, v in ps.native_cell_binding.items()}
    assert any(o == "verified" for o in outcomes.values())
    assert not any(v.get("skip_reason") for v in ps.native_cell_binding.values())
    kinds = [e["kind"] for e in pipe._audit_events(out_dir) if e.get("page_num") == 1]
    assert "native_cell_check_skipped" not in kinds


def test_r5_resf_p59_emitted_table_verified_not_over_flagged(tmp_path: Path) -> None:
    """RESF p.59 Tableau 1 through the pipeline stays verified (no lost-table over-flag)."""
    import test_resf_native_cell_pipeline as pipe

    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    _pipe, _result, state, out_dir = pipe._run_page(tmp_path, "resf_p59.pdf", md)
    verified = [
        k for k, v in state.pages[1].native_cell_binding.items() if v.get("outcome") == "verified"
    ]
    assert verified
    win = pipe._page_sidecar(out_dir, 1)["winning_output"]
    assert UNVERIFIED_TABLE_MARK not in win["text"]


def test_r5_p67_total_prefix_swap_not_verified(tmp_path: Path) -> None:
    from test_resf_native_cell_check import FIXTURES

    doc = fitz.open(FIXTURES / "resf_p67.pdf")
    page = doc[0]
    region = tuple(locate_tables(page)[0].bbox)
    md_lines = (FIXTURES / "p67_tableau8.md").read_text(encoding="utf-8").splitlines()
    rows = [line for line in md_lines if line.startswith("|") and "---" not in line]
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header = cells[0]
    body = cells[1:]
    idx = {r[0]: i for i, r in enumerate(body)}
    tot = idx["Total"]
    totm = next(i for i, r in enumerate(body) if r[0].startswith("Total des moindres"))
    body_sw = [list(r) for r in body]
    body_sw[tot][0], body_sw[totm][0] = body[totm][0], body[tot][0]
    swapped_md = "| " + " | ".join(header) + " |\n|" + " --- |" * len(header) + "\n"
    for r in body_sw:
        swapped_md += "| " + " | ".join(r) + " |\n"
    res = check_table_cell_binding(
        table_id="t0",
        words=page.get_text("words"),
        markdown=swapped_md,
        region_bbox=region,
        spans=[],
    )
    assert res.outcome is not NativeCellOutcome.VERIFIED
    doc.close()
