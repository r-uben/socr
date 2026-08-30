"""TICKET-B0 — table witness preparation (GH-353)."""

from __future__ import annotations

import fitz
import pytest

from socr.tables.locate import locate_tables
from socr.tables.witness import WitnessStatus, prepare_table_witnesses

_MD_ONE_TABLE = "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
_MD_TWO_TABLES = (
    "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    "\n"
    "prose between the two tables\n"
    "\n"
    "| c | d |\n| --- | --- |\n| 3 | 4 |\n"
)


def _draw_booktabs_table(page, x0: float, x1: float, top: float, rows: int, gap: float = 22.0):
    """A booktabs (rule-anchored) table: top/mid/bottom rules, no verticals."""
    ys = [top + i * gap for i in range(rows)]
    for i, y in enumerate(ys):
        page.insert_text((x0 + 4, y + 12), f"row{i}", fontsize=9)
    page.draw_line((x0, ys[0] - 4), (x1, ys[0] - 4))
    page.draw_line((x0, ys[1] - 2), (x1, ys[1] - 2))
    page.draw_line((x0, ys[-1] + 12), (x1, ys[-1] + 12))
    return ys


def _build_pdf(tmp_path, style: str):
    """Build a one-page PDF matching a locator-relevant shape; save to disk."""
    doc = fitz.open()
    page = doc.new_page()
    if style == "ruled":
        cols = [100, 220, 300, 380]
        rows = [100 + i * 22 for i in range(4)]
        for r, y in enumerate(rows):
            for c, x in enumerate(cols):
                page.insert_text((x + 4, y + 12), f"{r}{c}", fontsize=9)
        for yy in rows:
            page.draw_line((100, yy), (460, yy))
        for xx in cols + [460]:
            page.draw_line((xx, rows[0]), (xx, rows[-1]))
    elif style == "borderless":
        cols = [100, 220, 300, 380]
        rows = [100 + i * 22 for i in range(4)]
        for r, y in enumerate(rows):
            for c, x in enumerate(cols):
                page.insert_text((x + 4, y + 12), f"{r}{c}", fontsize=9)
    elif style == "stacked_booktabs":
        # Two booktabs tables sharing the same x-span. The locator's rule
        # grouping is x-overlap only (no vertical-gap check, documented
        # limitation at tables/locate.py:132), so these merge into ONE band
        # regardless of the vertical gap between them.
        _draw_booktabs_table(page, 100, 460, top=100, rows=4)
        _draw_booktabs_table(page, 100, 460, top=260, rows=4)
    else:  # pragma: no cover - test misuse
        raise ValueError(style)
    pdf_path = tmp_path / "doc.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_located_single_block_single_box(tmp_path):
    pdf_path = _build_pdf(tmp_path, "ruled")
    with prepare_table_witnesses(pdf_path, page_num=1, markdown=_MD_ONE_TABLE) as witnesses:
        assert len(witnesses) == 1
        w = witnesses[0]
        assert w.status is WitnessStatus.LOCATED
        assert w.box is not None
        assert w.crop_path is not None
        assert w.crop_path.exists()
        assert w.table_id == "p1-t0"
        assert w.markdown.strip() == _MD_ONE_TABLE.strip()
    # Guaranteed cleanup: the crop file is gone once the context exits.
    assert not w.crop_path.exists()


def test_ambiguous_two_blocks_one_merged_box(tmp_path):
    pdf_path = _build_pdf(tmp_path, "stacked_booktabs")
    # Sanity: the locator really does over-merge these into one band.
    doc = fitz.open(str(pdf_path))
    boxes = locate_tables(doc[0])
    doc.close()
    assert len(boxes) == 1

    with prepare_table_witnesses(pdf_path, page_num=1, markdown=_MD_TWO_TABLES) as witnesses:
        assert len(witnesses) == 2
        for w in witnesses:
            assert w.status is WitnessStatus.AMBIGUOUS
            assert w.box is None
            assert w.crop_path is None
            assert w.boxes_found_on_page == 1
            assert w.note
        assert witnesses[0].table_id == "p1-t0"
        assert witnesses[1].table_id == "p1-t1"


def test_missing_witness_no_located_box(tmp_path):
    pdf_path = _build_pdf(tmp_path, "borderless")
    with prepare_table_witnesses(pdf_path, page_num=1, markdown=_MD_ONE_TABLE) as witnesses:
        assert len(witnesses) == 1
        w = witnesses[0]
        assert w.status is WitnessStatus.MISSING
        assert w.box is None
        assert w.crop_path is None
        assert w.boxes_found_on_page == 0
        assert w.note


def test_crop_cleanup_on_exception(tmp_path):
    pdf_path = _build_pdf(tmp_path, "ruled")
    crop_path = None

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with prepare_table_witnesses(pdf_path, page_num=1, markdown=_MD_ONE_TABLE) as witnesses:
            crop_path = witnesses[0].crop_path
            assert crop_path.exists()
            raise _Boom

    assert crop_path is not None
    assert not crop_path.exists()


def test_no_table_blocks_yields_empty_and_never_opens_pdf(tmp_path):
    nonexistent = tmp_path / "does-not-exist.pdf"
    with prepare_table_witnesses(nonexistent, page_num=1, markdown="just prose\n") as witnesses:
        assert witnesses == []


def test_missing_pdf_degrades_to_missing_witness_never_raises(tmp_path):
    nonexistent = tmp_path / "does-not-exist.pdf"
    with prepare_table_witnesses(nonexistent, page_num=1, markdown=_MD_ONE_TABLE) as witnesses:
        assert len(witnesses) == 1
        w = witnesses[0]
        assert w.status is WitnessStatus.MISSING
        assert w.box is None
        assert w.crop_path is None
        assert "cannot open pdf" in w.note
