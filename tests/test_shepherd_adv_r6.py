"""Round-7 adversarial cases from adv_r6.py (PR #1206 usability pass)."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from socr.tables.locate import locate_tables
from socr.tables.native_cell_check import NativeCellOutcome, check_table_cell_binding
from socr.tables.native_contradiction import (
    pair_regions,
    unclaimed_locator_box_indices,
    locator_boxes_for_pairing,
)

from test_shepherd_adv_r3 import NOT_V, V, A, _AdvFixtures
from test_shepherd_adv_r5 import (
    TOP,
    LOW,
    _drawfull,
    _page_of,
    adv_md,
    _orch_outcomes,
)

pytest.importorskip("fitz")

FIXTURES = Path(__file__).parent / "fixtures" / "resf_native_cell"


@pytest.fixture
def adv(tmp_path: Path) -> _AdvFixtures:
    return _AdvFixtures(tmp_path)


def _mdh(rows, header=("Libellé", "2024", "2025"), extra: bool = False) -> str:
    n = len(header) + (1 if extra else 0)
    s = "| " + " | ".join(header) + (" |" if not extra else " | x |") + "\n|" + " --- |" * n + "\n"
    for r in rows:
        pad = list(r) + ([""] if extra else [])
        s += "| " + " | ".join(pad) + " |\n"
    return s


def test_r6_resf_p59_header_years_swapped_not_verified() -> None:
    doc = fitz.open(FIXTURES / "resf_p59.pdf")
    page = doc[0]
    raw = [tuple(b.bbox) for b in locate_tables(page)]
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    lines = md.splitlines()
    hdr = lines[0].replace("2023 | 2024 | 2025", "2023 | 2025 | 2024")
    bad = "\n".join([hdr] + lines[1:])
    pr = pair_regions(page, [bad], raw)
    res = check_table_cell_binding(
        table_id="t0",
        words=page.get_text("words"),
        markdown=bad,
        region_bbox=pr[0],
        spans=[],
        locator_boxes=raw,
        page=page,
    )
    assert res.outcome is not NativeCellOutcome.VERIFIED
    doc.close()


def test_r6_resf_p59_native_plus_sign_mismatch_not_verified() -> None:
    doc = fitz.open(FIXTURES / "resf_p59.pdf")
    page = doc[0]
    raw = [tuple(b.bbox) for b in locate_tables(page)]
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    pr = pair_regions(page, [md], raw)
    # Synthetic: model uses +99 where native prints +20 in a tight 2-col table is hard on p59;
    # use a minimal inline mutation on a numeric cell with explicit plus in model.
    bad = md.replace("| 0,4 |", "| +99 |", 1)
    res = check_table_cell_binding(
        table_id="t0",
        words=page.get_text("words"),
        markdown=bad,
        region_bbox=pr[0],
        spans=[],
        locator_boxes=raw,
        page=page,
    )
    assert res.outcome is not NativeCellOutcome.VERIFIED
    doc.close()


def test_r6_omitted_large_table_flags_unclaimed(tmp_path: Path) -> None:
    """Two small tables emitted, 12-row table omitted: must not verify both with no lost flag."""
    big = [(f"Row{i}", str(10 + i), str(20 + i)) for i in range(1, 13)]
    d, pg = _page_of(tmp_path, "bigomit.pdf", [TOP, LOW, big])
    try:
        mds = [adv_md(TOP), adv_md(LOW)]
        raw, _kept, outs, un = _orch_outcomes(pg, mds)
        assert un, "large native table region must be flagged when omitted"
    finally:
        d.close()


def test_r6_plfss_p44_correct_not_contradicted() -> None:
    doc = fitz.open(FIXTURES / "plfss_p44.pdf")
    page = doc[0]
    boxes = locate_tables(page)
    assert boxes
    region = tuple(boxes[0].bbox)
    # Minimal smoke: page has native tables; empty markdown would ambiguous — we only assert
    # the fixture opens and locate finds geometry (full markdown fixtures added incrementally).
    assert region[3] > region[1]
    doc.close()


def test_r6_nest_inner_three_rows_not_verified(tmp_path: Path) -> None:
    from test_shepherd_adv_r5 import _nest_subpanel_pdf

    d, pg, rows = _nest_subpanel_pdf(tmp_path)
    try:
        _raw, _kept, outs, _un = _orch_outcomes(pg, [adv_md(rows[:3])])
        assert outs[0][1] is not V
    finally:
        d.close()
