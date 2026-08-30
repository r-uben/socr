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
    elif style == "two_ruled":
        # Two separate ruled tables, vertically non-overlapping AND with
        # deliberately non-overlapping x-spans, each with its own verticals so
        # find_tables() reports two distinct boxes. The x-offset matters:
        # _booktabs_tables independently scans ALL horizontal rules on the
        # page (including a ruled table's own row lines) and groups them by
        # x-overlap (>= 0.6 shared width, tables/locate.py _RULE_X_OVERLAP) --
        # same-column ruled tables would have their rules merged into one
        # spurious extra booktabs band spanning both tables.
        top_cols = [100, 220, 300, 380]
        bottom_cols = [340, 460, 540, 620]
        top_rows = [100 + i * 22 for i in range(4)]
        bottom_rows = [320 + i * 22 for i in range(4)]
        for cols, rows in ((top_cols, top_rows), (bottom_cols, bottom_rows)):
            x0, x1 = cols[0], cols[-1] + 80
            for r, y in enumerate(rows):
                for c, x in enumerate(cols):
                    page.insert_text((x + 4, y + 12), f"{r}{c}", fontsize=9)
            for yy in rows:
                page.draw_line((x0, yy), (x1, yy))
            for xx in cols + [x1]:
                page.draw_line((xx, rows[0]), (xx, rows[-1]))
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


def test_located_two_blocks_two_boxes_pair_in_order(tmp_path):
    """The most common real multi-table page shape: two emitted blocks, two
    distinct non-overlapping boxes -> both LOCATED, paired top-to-bottom in
    the SAME order as they were emitted (index pairing after each source's
    own reading-order sort -- see witness.py's module docstring for the
    residual assumption this rests on)."""
    pdf_path = _build_pdf(tmp_path, "two_ruled")
    doc = fitz.open(str(pdf_path))
    boxes = locate_tables(doc[0])
    doc.close()
    assert len(boxes) == 2
    assert boxes[0].bbox[1] < boxes[1].bbox[1]  # sanity: top box first

    with prepare_table_witnesses(pdf_path, page_num=1, markdown=_MD_TWO_TABLES) as witnesses:
        assert len(witnesses) == 2
        top, bottom = witnesses
        assert top.status is WitnessStatus.LOCATED
        assert bottom.status is WitnessStatus.LOCATED
        assert top.box is not None and bottom.box is not None
        # Correct pairing: the FIRST emitted block (top.markdown has "a|b")
        # gets the geometrically TOP box; the second gets the bottom box.
        assert top.box.bbox[1] < bottom.box.bbox[1]
        assert "a" in top.markdown and "b" in top.markdown
        assert "c" in bottom.markdown and "d" in bottom.markdown
        assert top.table_id == "p1-t0"
        assert bottom.table_id == "p1-t1"
        assert top.crop_path is not None and top.crop_path.exists()
        assert bottom.crop_path is not None and bottom.crop_path.exists()
        assert top.crop_path != bottom.crop_path


def _two_ruled_boxes_pdf(tmp_path, top_values: list[str] | None, bottom_values: list[str] | None):
    """Two ruled tables (non-overlapping x-spans, see ``two_ruled`` above),
    with each cell's text controlled explicitly (``None`` -> blank interior,
    rules only, so the box has zero native words for corroboration tests)."""
    doc = fitz.open()
    page = doc.new_page()
    top_cols = [100, 220]
    bottom_cols = [340, 460]
    top_rows = [100 + i * 22 for i in range(2)]
    bottom_rows = [320 + i * 22 for i in range(2)]
    for cols, rows, values in (
        (top_cols, top_rows, top_values),
        (bottom_cols, bottom_rows, bottom_values),
    ):
        x0, x1 = cols[0], cols[-1] + 80
        if values is not None:
            it = iter(values)
            for y in rows:
                for x in cols:
                    page.insert_text((x + 4, y + 12), next(it), fontsize=9)
        for yy in rows:
            page.draw_line((x0, yy), (x1, yy))
        for xx in cols + [x1]:
            page.draw_line((xx, rows[0]), (xx, rows[-1]))
    pdf_path = tmp_path / "doc.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_corroboration_neutral_ships_normal_order(tmp_path):
    """Corroborated case: native content matches the identity pairing ->
    ships LOCATED with the correct (unchanged) pairing."""
    top_values = ["111", "222", "333", "444"]
    bottom_values = ["555", "666", "777", "888"]
    pdf_path = _two_ruled_boxes_pdf(tmp_path, top_values, bottom_values)
    md = (
        "| a | b |\n| --- | --- |\n| 111 | 222 |\n| 333 | 444 |\n"
        "\n"
        "prose between the two tables\n"
        "\n"
        "| c | d |\n| --- | --- |\n| 555 | 666 |\n| 777 | 888 |\n"
    )
    with prepare_table_witnesses(pdf_path, page_num=1, markdown=md) as witnesses:
        assert len(witnesses) == 2
        top, bottom = witnesses
        assert top.status is WitnessStatus.LOCATED
        assert bottom.status is WitnessStatus.LOCATED
        assert top.box.bbox[1] < bottom.box.bbox[1]
        assert "111" in top.markdown
        assert "555" in bottom.markdown


def test_corroboration_contradicted_swapped_content_demotes_to_ambiguous(tmp_path):
    """The swap case: markdown emitted in reverse content order of geometry --
    block 0's numbers actually belong to the geometrically BOTTOM box and
    block 1's numbers to the geometrically TOP box. Both members' evidence
    strictly favors the alternate assignment -> positive contradiction ->
    both demote to AMBIGUOUS (never a silent auto-swap)."""
    top_values = ["111", "222", "333", "444"]
    bottom_values = ["555", "666", "777", "888"]
    pdf_path = _two_ruled_boxes_pdf(tmp_path, top_values, bottom_values)
    # First emitted block (index 0, would pair with the TOP box by identity)
    # actually carries the BOTTOM table's numbers, and vice versa.
    md = (
        "| c | d |\n| --- | --- |\n| 555 | 666 |\n| 777 | 888 |\n"
        "\n"
        "prose between the two tables\n"
        "\n"
        "| a | b |\n| --- | --- |\n| 111 | 222 |\n| 333 | 444 |\n"
    )
    with prepare_table_witnesses(pdf_path, page_num=1, markdown=md) as witnesses:
        assert len(witnesses) == 2
        for w in witnesses:
            assert w.status is WitnessStatus.AMBIGUOUS
            assert w.box is None
            assert w.crop_path is None
            assert "corroboration" in w.note


def test_corroboration_neutral_no_native_words_ships_index_pairing(tmp_path):
    """No mass-demotion: boxes exist (rules detected) but carry zero native
    words (a scanned-looking / sparse table). Absent evidence is NEUTRAL, not
    contradiction, so the plain index pairing still ships LOCATED."""
    pdf_path = _two_ruled_boxes_pdf(tmp_path, None, None)
    with prepare_table_witnesses(pdf_path, page_num=1, markdown=_MD_TWO_TABLES) as witnesses:
        assert len(witnesses) == 2
        for w in witnesses:
            assert w.status is WitnessStatus.LOCATED
            assert w.box is not None
            assert w.crop_path is not None


def test_corroboration_tie_identical_tables_ships_index_pairing(tmp_path):
    """Identical-content tables: evidence ties for every pairing (a swap
    would score identically) -> neutral, not a strict majority -> the
    identity pairing ships (harmless either way)."""
    values = ["111", "222", "333", "444"]
    pdf_path = _two_ruled_boxes_pdf(tmp_path, values, list(values))
    md = (
        "| a | b |\n| --- | --- |\n| 111 | 222 |\n| 333 | 444 |\n"
        "\n"
        "prose between the two tables\n"
        "\n"
        "| c | d |\n| --- | --- |\n| 111 | 222 |\n| 333 | 444 |\n"
    )
    with prepare_table_witnesses(pdf_path, page_num=1, markdown=md) as witnesses:
        assert len(witnesses) == 2
        for w in witnesses:
            assert w.status is WitnessStatus.LOCATED
            assert w.box is not None
            assert w.crop_path is not None


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
