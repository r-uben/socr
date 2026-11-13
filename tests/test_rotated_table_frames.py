"""Rotated table pages: the row check, the recorded boxes and the crops share one frame.

Convention (``socr.core.born_digital``): every table box socr records or passes is in the
UNROTATED text-layer frame, where ``get_text`` words and ``get_drawings`` live; pixels are
drawn after ``pixmap_clip``. Before it, on a /Rotate 90 Greenbook page (2002-03-19 p27) the
row check got a displayed-frame box, found no text line inside it, called every row
UNVERIFIABLE, and a table whose labels had slid one row shipped (marked) instead of being
withheld as CONTRADICTED.

Every assertion is a DIFFERENCE against the /Rotate 0 control drawn from the same
instructions (``test_tb54_rotated_table_box``), over /Rotate 0, 90, 180 and 270 and over
both table styles, plus the transposed page (text that reads upright only after /Rotate 90,
the p27 shape). No model, no provider.
"""

from __future__ import annotations

import fitz
import numpy as np
import pytest
from PIL import Image
from test_tb54_rotated_table_box import (
    COL_RIGHT_EDGES,
    CONTENT_H,
    CONTENT_W,
    FONT_SIZE,
    ROTATIONS,
    ROW_STEP,
    ROW_Y0,
    ROWS,
    RULE_X0,
    RULE_X1,
    RULE_YS,
    _contains,
    _value_words,
)

from socr.tables.locate import locate_tables
from socr.tables.row_binding_check import RowVerdict, TableOutcome, check_page_row_binding

STYLES = ("booktabs", "ruled")
HEADER = "|  |  |  |  |  |\n| --- | --- | --- | --- | --- |\n"


def _correct_markdown() -> str:
    return HEADER + "\n".join(f"| {label} | " + " | ".join(v) + " |" for label, v in ROWS)


def _slid_markdown() -> str:
    """The p27 shape: the first label ships empty and every row's values slide one label down."""
    labels = [label for label, _ in ROWS]
    values = [v for _, v in ROWS]
    lines = [f"| {labels[0]} |  |  |  |  |"]
    lines += [f"| {labels[i + 1]} | " + " | ".join(values[i]) + " |" for i in range(len(ROWS) - 1)]
    return HEADER + "\n".join(lines)


MARKER = fitz.Rect(RULE_X0 + 2, RULE_YS[1] + 6, RULE_X0 + 8, RULE_YS[2] - 6)  # in the label column
RED = (1, 0, 0)
GRID_XS = (RULE_X0, 160.0, 340.0, 490.0, 640.0, RULE_X1)
FAMILIES = ("stored", "displayed")


def _build(path, rotation: int, style: str, family: str):
    """The table of ``test_tb54_rotated_table_box`` with a red bar in its label column.

    ``stored``: drawn horizontally in the page's own (unrotated) frame, then /Rotate is
    set -- the Tealbook /Rotate 270 shape, which DISPLAYS sideways. ``displayed``: drawn
    so that it reads upright AFTER /Rotate (every point mapped through
    ``derotation_matrix``, glyphs turned by the same angle) -- the 2002-03 Greenbook p27
    shape. Same drawing instructions, in display coordinates, for both.
    """
    doc = fitz.open()
    if family == "stored" or rotation in (0, 180):
        page = doc.new_page(width=CONTENT_W, height=CONTENT_H)
    else:
        page = doc.new_page(width=CONTENT_H, height=CONTENT_W)
    if family == "displayed":
        page.set_rotation(rotation)
    to_raw = page.derotation_matrix if family == "displayed" else fitz.Identity
    glyph_turn = rotation if family == "displayed" else 0

    def at(x, y):
        return fitz.Point(x, y) * to_raw

    def line(a, b):
        page.draw_line(at(*a), at(*b), width=0.5)

    if style == "booktabs":
        for y in RULE_YS:
            line((RULE_X0, y), (RULE_X1, y))
    else:
        ys = [RULE_YS[0] + k * (RULE_YS[2] - RULE_YS[0]) / 6 for k in range(7)]
        for y in ys:
            line((RULE_X0, y), (RULE_X1, y))
        for x in GRID_XS:
            line((x, ys[0]), (x, ys[-1]))
    for i, (label, values) in enumerate(ROWS):
        y = ROW_Y0 + i * ROW_STEP
        page.insert_text(at(RULE_X0 + 10, y), label, fontsize=FONT_SIZE, rotate=glyph_turn)
        for right, v in zip(COL_RIGHT_EDGES, values):
            x = right - fitz.get_text_length(v, fontsize=FONT_SIZE)
            page.insert_text(at(x, y), v, fontsize=FONT_SIZE, rotate=glyph_turn)
    marker = fitz.Rect(at(MARKER.x0, MARKER.y0), at(MARKER.x1, MARKER.y1)).normalize()
    page.draw_rect(marker, color=RED, fill=RED)
    if family == "stored":
        page.set_rotation(rotation)
    doc.save(path)
    doc.close()
    return fitz.open(path)


def _pages(tmp_path):
    out = {}
    for family in FAMILIES:
        for style in STYLES:
            for rotation in ROTATIONS:
                if family == "displayed" and rotation == 0:
                    continue  # identical to stored at /Rotate 0
                name = f"{family}-{style}-{rotation}.pdf"
                out[(family, style, rotation)] = _build(
                    str(tmp_path / name), rotation, style, family
                )
    return out


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    docs = _pages(tmp_path_factory.mktemp("frames"))
    yield {k: d[0] for k, d in docs.items()}
    for d in docs.values():
        d.close()


def _outcome(page, markdown):
    """The row check exactly as the pipeline calls it: regions from ``locate_tables``."""
    regions = [tuple(b.bbox) for b in locate_tables(page)] or None
    results, _ = check_page_row_binding(page, markdown, regions)
    assert len(results) == 1
    return results[0]


# ---------------------------------------------------------------------------
# The row check
# ---------------------------------------------------------------------------


def test_the_located_box_holds_every_value_word_in_the_words_frame(pages) -> None:
    """The box the row check receives is in the frame its words are in, on every page."""
    for key, page in pages.items():
        boxes = locate_tables(page)
        assert len(boxes) == 1, (key, [b.bbox for b in boxes])
        outside = [w[4] for w in _value_words(page) if not _contains(boxes[0].bbox, w)]
        assert outside == [], (key, outside)


def test_a_correct_table_is_bound_on_every_rotation(pages) -> None:
    for key, page in pages.items():
        result = _outcome(page, _correct_markdown())
        assert result.outcome is TableOutcome.BOUND, (key, [r.verdict for r in result.rows])


def test_labels_slid_one_row_are_contradicted_on_every_rotation(pages) -> None:
    """The p27 shape: the /Rotate 0 control convicts it, and so must every rotated page."""
    control = _outcome(pages[("stored", "booktabs", 0)], _slid_markdown())
    assert control.outcome is TableOutcome.CONTRADICTED
    convicted = {RowVerdict.DISPLACED, RowVerdict.EMPTIED, RowVerdict.LABEL_LOST}
    for key, page in pages.items():
        result = _outcome(page, _slid_markdown())
        assert result.outcome is TableOutcome.CONTRADICTED, (key, [r.verdict for r in result.rows])
        assert sum(result.count(v) for v in convicted) == sum(
            control.count(v) for v in convicted
        ), key


# ---------------------------------------------------------------------------
# The boxes the run records (find_tables) are stored in the words' frame too
# ---------------------------------------------------------------------------


def test_recorded_detector_boxes_are_in_the_words_frame(pages) -> None:
    from socr.core.born_digital import BornDigitalDetector

    # find_tables needs a drawn grid; the booktabs page records no box at all.
    _, control = BornDigitalDetector._detect_table_regions(pages[("stored", "ruled", 0)])
    assert control
    control_words = len(
        [w for w in _value_words(pages[("stored", "ruled", 0)]) if _contains(control[0], w)]
    )
    assert control_words == sum(len(v) for _, v in ROWS)
    for key, page in pages.items():
        if key[1] != "ruled":
            continue
        _, boxes = BornDigitalDetector._detect_table_regions(page)
        assert len(boxes) == 1, (key, boxes)
        inside = [w for w in _value_words(page) if _contains(boxes[0], w)]
        assert len(inside) == control_words, (key, boxes[0], len(inside))


# ---------------------------------------------------------------------------
# Crops: complete (the row-label column is inside) and upright
# ---------------------------------------------------------------------------


def _red_centroid(img: Image.Image):
    arr = np.asarray(img.convert("RGB")).astype(int)
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
    if not red.any():
        return None
    ys, xs = np.nonzero(red)
    return xs.mean() / red.shape[1], ys.mean() / red.shape[0]


def test_table_crop_holds_the_label_column_and_reads_upright(pages) -> None:
    from socr.tables.extract import TableCropExtractor

    bad = []
    for key, page in pages.items():
        boxes = locate_tables(page)
        assert len(boxes) == 1, key
        out = TableCropExtractor(reader=None, crop_dpi=72)._render_crop(page, boxes[0], page.rect)
        assert out is not None, key
        img = Image.open(out).copy()
        out.unlink()
        centroid = _red_centroid(img)
        # Complete: the label column (the red bar) is inside. Upright: the table is wider
        # than tall, as drawn, and the label column sits at the crop's left.
        if centroid is None or not img.width > img.height or not centroid[0] < 0.5:
            bad.append((key, img.size, centroid))
    assert CONTENT_W > CONTENT_H
    assert not bad, bad
