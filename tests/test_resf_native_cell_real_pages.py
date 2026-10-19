"""Golden transcriptions for attached RESF / PLFSS real-document table pages."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from socr.core.born_digital import flatten_page_spans
from socr.tables.locate import locate_tables
from socr.tables.native_cell_check import NativeCellOutcome, check_table_cell_binding
from socr.tables.native_contradiction import pair_regions

FIXTURES = Path(__file__).parent / "fixtures" / "resf_native_cell"

_OK = frozenset({NativeCellOutcome.VERIFIED, NativeCellOutcome.AMBIGUOUS})


def _run_page(
    pdf_name: str,
    md_name: str,
    *,
    table_idx: int = 0,
    peer_md: str | None = None,
    peer_table_idx: int = 0,
) -> None:
    doc = fitz.open(FIXTURES / pdf_name)
    page = doc[0]
    raw = [tuple(b.bbox) for b in locate_tables(page)]
    md = (FIXTURES / md_name).read_text(encoding="utf-8")
    mds = [md]
    if peer_md:
        mds.insert(peer_table_idx, (FIXTURES / peer_md).read_text(encoding="utf-8"))
    pr = pair_regions(page, mds, raw)
    pick = pr[-1] if peer_md is None else pr[1]
    assert pick is not None, f"pairing failed for {pdf_name} / {md_name}"
    peers = [pr[0]] if peer_md and pr[0] is not None else None
    words = page.get_text("words")
    spans = flatten_page_spans(page)
    r = check_table_cell_binding(
        table_id="real",
        words=words,
        markdown=md,
        region_bbox=pick,
        spans=spans,
        peer_region_bboxes=peers,
        locator_boxes=raw,
        page=page,
    )
    doc.close()
    assert r.outcome in _OK, f"{pdf_name} {md_name}: {r.outcome.value} — {r.detail}"


class TestRealDocumentGoldenPages:
    def test_resf_p59_tableau1(self) -> None:
        _run_page("resf_p59.pdf", "p59_tableau1.md", table_idx=1)

    def test_resf_p59_tableau2(self) -> None:
        _run_page("resf_p59.pdf", "p59_tableau2.md", peer_md="p59_tableau1.md")

    def test_resf_p67_tableau8(self) -> None:
        _run_page("resf_p67.pdf", "p67_tableau8.md")

    def test_plfss_p38_tableau18(self) -> None:
        _run_page("plfss_p38.pdf", "p38_tableau18.md", table_idx=1)

    def test_plfss_p44_compensation_slice(self) -> None:
        _run_page("plfss_p44.pdf", "p44_compensation_slice.md")

    def test_plfss_p65_pharma_slice(self) -> None:
        _run_page("plfss_p65.pdf", "p65_pharma_slice.md")
