"""TB-54: table boxes on /Rotate pages share the frame of words and drawings.

PyMuPDF reports ``get_text`` words and ``get_drawings`` in the UNROTATED frame and
``find_tables`` / ``page.rect`` / ``get_pixmap(clip=)`` in the ROTATED frame. Before
the fix a /Rotate 270 page clipped every rule at ``page.rect`` (x <= 612 on a 792-wide
content frame), ``find_tables`` boxes came back in the other frame, and the two
disagreed on /Rotate 90.

Every assertion pins a DIFFERENCE against the /Rotate 0 control built from the same
drawing instructions, never a value measured on one machine. Fixtures are synthetic
(fitz, ``tmp_path``); no corpus, no model, no provider.
"""

from __future__ import annotations

import re

import fitz
import pytest

from socr.tables import locate
from socr.tables.locate import locate_tables, pixmap_clip
from socr.tables.row_binding_check import RowVerdict, TableOutcome, check_page_row_binding

ROTATIONS = (0, 90, 180, 270)

# Content frame (unrotated, landscape): the printed table spans 30..760 on a
# 792-wide page, so a clamp at the 612-wide rotated rect cuts it.
CONTENT_W, CONTENT_H = 792.0, 612.0
RULE_X0, RULE_X1 = 30.0, 760.0
RULE_YS = (100.0, 130.0, 330.0)  # top, under header, bottom (booktabs)
COL_RIGHT_EDGES = (330.0, 480.0, 630.0, 750.0)  # numbers end here (right aligned)
LABELS = ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")
ROWS = (
    ("Alpha", ("11.1", "12.2", "13.3", "14.4")),
    ("Beta", ("21.1", "22.2", "23.3", "24.4")),
    ("Gamma", ("31.1", "32.2", "33.3", "34.4")),
    ("Delta", ("41.1", "42.2", "43.3", "44.4")),
    ("Epsilon", ("51.1", "52.2", "53.3", "54.4")),
)
ROW_Y0, ROW_STEP = 155.0, 35.0
FONT_SIZE = 10.0
NUMERIC = re.compile(r"^\d+\.\d$")
# Tolerance between a rotated page's box and the /Rotate 0 control: the rule
# flatness the detector itself uses to call a stroke straight (geometry, not tuned).
TOL = locate._RULE_FLATNESS_PT


def _draw_rows(page, text_rotate: int = 0) -> None:
    page.insert_text((RULE_X0 + 10, 118), "Table 1", fontsize=FONT_SIZE)
    for i, (label, values) in enumerate(ROWS):
        y = ROW_Y0 + i * ROW_STEP
        page.insert_text((RULE_X0 + 10, y), label, fontsize=FONT_SIZE)
        for right, v in zip(COL_RIGHT_EDGES, values):
            width = fitz.get_text_length(v, fontsize=FONT_SIZE)
            page.insert_text((right - width, y), v, fontsize=FONT_SIZE)


def _build(path, rotation: int, style: str, bar: bool = False):
    """One landscape-content page; same drawing instructions for every rotation."""
    doc = fitz.open()
    page = doc.new_page(width=CONTENT_W, height=CONTENT_H)
    _draw_rows(page)
    if style == "booktabs":
        for y in RULE_YS:
            page.draw_line((RULE_X0, y), (RULE_X1, y), width=0.5)
    else:  # fully gridded
        ys = [RULE_YS[0] + k * (RULE_YS[2] - RULE_YS[0]) / 6 for k in range(7)]
        xs = [RULE_X0, 160.0, 340.0, 490.0, 640.0, RULE_X1]
        for y in ys:
            page.draw_line((RULE_X0, y), (RULE_X1, y), width=0.5)
        for x in xs:
            page.draw_line((x, ys[0]), (x, ys[-1]), width=0.5)
    if bar:  # edge-to-edge banner at the page edge, drawn as four line items (as the corpus bar is)
        for y in (567.0, CONTENT_H):
            page.draw_line((0, y), (CONTENT_W, y), color=(0, 0.6, 0), width=0.5)
    page.set_rotation(rotation)
    doc.save(path)
    doc.close()
    return fitz.open(path)


def _value_words(page):
    return [w for w in page.get_text("words") if NUMERIC.match(w[4])]


def _contains(box, w) -> bool:
    return box[0] <= w[0] and box[1] <= w[1] and w[2] <= box[2] and w[3] <= box[3]


@pytest.fixture
def make(tmp_path):
    opened = []

    def _make(rotation: int, style: str, bar: bool = False):
        doc = _build(str(tmp_path / f"{style}-{rotation}-{int(bar)}.pdf"), rotation, style, bar)
        opened.append(doc)
        return doc[0]

    yield _make
    for d in opened:
        d.close()


def _close(a, b, tol=TOL) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


@pytest.mark.parametrize("style", ["booktabs", "ruled"])
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_box_covers_every_value_word_like_the_rotate_0_control(make, style, rotation):
    control = make(0, style)
    control_boxes = locate_tables(control)
    assert len(control_boxes) == 1
    page = make(rotation, style)
    boxes = locate_tables(page)
    # Exactly one box per table: no wrong-frame duplicate survives dedup.
    assert len(boxes) == 1, [(b.bbox, b.source) for b in boxes]
    box = boxes[0].bbox
    words = _value_words(page)
    assert len(words) == len(_value_words(control)) == 20
    outside = [w[4] for w in words if not _contains(box, w)]
    assert outside == []
    # Same table extent as the control, in the words frame.
    assert _close(box, control_boxes[0].bbox), (box, control_boxes[0].bbox)


@pytest.mark.parametrize("rotation", (90, 180, 270))
def test_rotated_box_is_not_clipped_at_the_rotated_page_width(make, rotation):
    page = make(rotation, "booktabs")
    box = locate_tables(page)[0].bbox
    assert box[2] == pytest.approx(RULE_X1, abs=TOL)


@pytest.mark.parametrize("style", ["booktabs", "ruled"])
def test_rotate_0_output_is_the_old_behaviour(make, style):
    page = make(0, style)
    boxes = locate_tables(page)
    assert len(boxes) == 1
    if style == "booktabs":
        # Old behaviour: (min x, top rule, max x, bottom rule) of the drawn rules.
        assert boxes[0].bbox == (RULE_X0, RULE_YS[0], RULE_X1, RULE_YS[2])
    else:
        raw = page.find_tables().tables
        assert len(raw) == 1
        assert boxes[0].bbox == tuple(float(v) for v in raw[0].bbox)


@pytest.mark.parametrize("rotation", (90, 180, 270))
def test_banner_bar_rule_pair_does_not_replace_the_table_box(make, rotation):
    page = make(rotation, "booktabs", bar=True)
    boxes = locate_tables(page)
    assert len(boxes) == 1, [(b.bbox, b.source) for b in boxes]
    assert _close(boxes[0].bbox, (RULE_X0, RULE_YS[0], RULE_X1, RULE_YS[2]))


def test_rotate_0_keeps_a_full_width_rule_pair(make):
    """The full-span exclusion is gated on /Rotate pages: /Rotate 0 is unchanged."""
    page = make(0, "booktabs", bar=True)
    boxes = locate_tables(page)
    # Old behaviour kept (a separate fix): the bar rules join the table's band.
    assert [b.bbox for b in boxes] == [(0.0, RULE_YS[0], CONTENT_W, CONTENT_H)]


def _build_sideways(path, rotation: int = 90):
    """Portrait mediabox, /Rotate 90: the table reads upright only after rotation, so the
    printed horizontal rules are vertical strokes in the unrotated frame."""
    w, h = 612.0, 792.0
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)

    # Printed (displayed) frame is 792 wide x 612 tall; draw the displayed table, then map
    # each displayed point (X, Y) to the unrotated frame for rotation 90: (x, y) = (Y, h - X)
    # is not needed: text is inserted with rotate=90 so it reads upright when displayed.
    def to_raw(X, Y):
        return (w - Y, X)  # displayed (X right, Y down) -> unrotated for /Rotate 90

    for Y in RULE_YS:
        page.draw_line(to_raw(RULE_X0, Y), to_raw(RULE_X1, Y), width=0.5)
    texts = [((RULE_X0 + 10, 118), "Table 1")]
    for i, (label, values) in enumerate(ROWS):
        y = ROW_Y0 + i * ROW_STEP
        texts.append(((RULE_X0 + 10, y), label))
        for right, v in zip(COL_RIGHT_EDGES, values):
            texts.append(((right - fitz.get_text_length(v, fontsize=FONT_SIZE), y), v))
    for (X, Y), s in texts:
        page.insert_text(to_raw(X, Y), s, fontsize=FONT_SIZE, rotate=90)
    page.set_rotation(rotation)
    doc.save(path)
    doc.close()
    return fitz.open(path)


def test_rotate_90_sideways_booktabs_box_is_in_the_words_frame(tmp_path):
    doc = _build_sideways(str(tmp_path / "sideways.pdf"))
    try:
        page = doc[0]
        words = _value_words(page)
        assert len(words) == 20
        boxes = locate_tables(page)
        assert len(boxes) == 1, [(b.bbox, b.source) for b in boxes]
        outside = [w[4] for w in words if not _contains(boxes[0].bbox, w)]
        assert outside == []
    finally:
        doc.close()


@pytest.mark.parametrize("style", ["booktabs", "ruled"])
@pytest.mark.parametrize("rotation", (90, 180, 270))
def test_pixmap_clip_area_equals_the_control(make, style, rotation):
    control = make(0, style)
    page = make(rotation, style)
    control_box = locate_tables(control)[0].bbox
    box = locate_tables(page)[0].bbox
    c = fitz.Rect(pixmap_clip(control, control_box))
    clip = fitz.Rect(pixmap_clip(page, box))
    # The displayed crop is the same table: same area, and it is the rotated page's
    # own displayed region (inside page.rect).
    assert clip.get_area() == pytest.approx(c.get_area(), abs=TOL * (c.width + c.height))
    assert page.rect.contains(clip + (-TOL, -TOL, TOL, TOL))
    pix = page.get_pixmap(clip=clip)
    cpix = control.get_pixmap(clip=c)
    assert sorted((pix.width, pix.height)) == sorted((cpix.width, cpix.height))


def test_pixmap_clip_is_identity_at_rotate_0(make):
    page = make(0, "booktabs")
    bbox = (1.0, 2.0, 3.0, 4.0)
    assert pixmap_clip(page, bbox) is bbox


@pytest.mark.parametrize("style", ["booktabs", "ruled"])
@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
def test_row_binding_is_bound_on_a_rotated_page(make, style, rotation):
    page = make(rotation, style)
    # Every markdown row with content is judged, the header included; the synthetic page
    # prints no heading line, so the heading row is empty (nothing to judge).
    header = "|  |  |  |  |  |\n| --- | --- | --- | --- | --- |\n"
    markdown = header + "\n".join(f"| {label} | " + " | ".join(v) + " |" for label, v in ROWS)
    boxes = [b.bbox for b in locate_tables(page)]
    results, _ = check_page_row_binding(page, markdown, boxes)
    assert len(results) == 1
    assert results[0].outcome is TableOutcome.BOUND, [r.verdict for r in results[0].rows]
    assert results[0].count(RowVerdict.BOUND) == len(ROWS)


def test_image_fallback_boxes_are_converted_to_the_words_frame(make, monkeypatch):
    """locate_tables_image reads a rotated-frame raster; its boxes must be unrotated."""
    page = make(90, "booktabs")
    # Blank the vector detectors so the scanned-page fallback fires.
    monkeypatch.setattr(locate, "_ruled_tables", lambda p: [])
    monkeypatch.setattr(locate, "_booktabs_tables", lambda p: [])
    monkeypatch.setattr(locate, "_is_scanned", lambda p: True)
    rotated_frame_box = (50.0, 60.0, 300.0, 200.0)
    monkeypatch.setattr(
        "socr.tables.image_locate.locate_tables_image",
        lambda p: [locate.TableBox(bbox=rotated_frame_box, source="image")],
    )
    got = locate_tables(page)[0].bbox
    expected = fitz.Rect(rotated_frame_box) * page.derotation_matrix
    expected.normalize()
    assert got == pytest.approx(tuple(expected))


# The three pixmap consumers: each must render the same table area as the /Rotate 0
# control (image dimensions are compared as an unordered pair because the renderer
# may turn the crop upright).

CROP_DPI = 72


def _dims_from_png(path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _site_extract(page, box):
    from socr.tables.extract import TableCropExtractor

    ex = TableCropExtractor(reader=None, crop_dpi=CROP_DPI)
    path = ex._render_crop(page, box, page.rect)
    assert path is not None
    return _dims_from_png(path)


def _site_witness(page, box):
    from socr.tables.witness import _render_crop_safe

    path = _render_crop_safe(fitz_path(page), 1, box, CROP_DPI)
    assert path is not None
    return _dims_from_png(path)


def _site_source_evidence(page, box):
    from socr.tables.source_evidence import _render_crop_pixmap

    pix = _render_crop_pixmap(page, box.bbox, CROP_DPI)
    return (pix.width, pix.height)


def fitz_path(page) -> str:
    return page.parent.name


def _site_page_witness(page, box):
    """Full-page render through the witness: the page-sized box must cover the page."""
    from socr.tables.witness import _render_page_safe

    path = _render_page_safe(fitz_path(page), 1, CROP_DPI)
    assert path is not None
    return _dims_from_png(path)


@pytest.mark.parametrize("site", [_site_extract, _site_witness, _site_source_evidence])
@pytest.mark.parametrize("rotation", (90, 180, 270))
def test_pixmap_sites_crop_the_same_table_area_as_the_control(make, site, rotation):
    control = make(0, "booktabs")
    page = make(rotation, "booktabs")
    c = site(control, locate_tables(control)[0])
    got = site(page, locate_tables(page)[0])
    assert sorted(got) == sorted(c)


@pytest.mark.parametrize("rotation", (90, 270))
def test_witness_full_page_render_covers_the_whole_rotated_page(make, rotation):
    page = make(rotation, "booktabs")
    w, h = _site_page_witness(page, None)
    # 72 dpi: the whole page, not a corner of it (padding clamps to page.rect). An
    # unordered pair, as above: the renderer turns the page upright
    # (test_upright_rotated_pages pins the orientation).
    assert sorted((w, h)) == sorted((round(page.rect.width), round(page.rect.height)))
