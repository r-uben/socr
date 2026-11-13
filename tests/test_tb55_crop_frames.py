"""TB-55 part 1: crops of text-layer boxes on /Rotate pages use the right frame.

PyMuPDF reports ``get_text`` words, ``get_text(clip=)`` and ``get_drawings`` in the
UNROTATED frame; ``page.rect`` and ``get_pixmap(clip=)`` use the ROTATED frame.
TB-54 converted the table-box crop sites. This file pins the remaining table-lane
sites:

* ``UnifiedPipeline._render_adjudication_crop`` (bbox AND clip_limits are text-layer
  coordinates);
* the ``render_rotation_for(clip=)`` argument at the table crop sites (it forwards to
  ``get_text(clip=)``, so it needs the unrotated frame).

TB-55 also converted ``source_evidence._glyph_bbox_has_ink``. That change is NOT on
this branch (it flipped 9 of 70 rotated #1099 verdicts toward pass, TB-56), so its two
tests are not carried here.

The oracle never calls ``pixmap_clip``: word boxes are mapped to the displayed frame
with ``fitz.Rect(word) * page.rotation_matrix`` here. Every assertion compares a
/Rotate 90/270 page against the /Rotate 0 control of the same drawing instructions.
No model, no provider, no corpus.
"""

from __future__ import annotations

import fitz
import pytest
from PIL import Image
from test_tb54_rotated_table_box import (
    _site_extract,
    _site_source_evidence,
    _site_witness,
    _value_words,
    fitz_path,
    make,  # noqa: F401  (pytest fixture)
)

from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.extract import CROP_PADDING_PT
from socr.tables.locate import locate_tables

ROTATED = (90, 270)


def _pipe() -> UnifiedPipeline:
    # _render_adjudication_crop uses only its arguments.
    return UnifiedPipeline.__new__(UnifiedPipeline)


class _ClipSpy:
    """Record every ``get_pixmap(clip=)`` the code under test issues."""

    def __init__(self, monkeypatch):
        self.clips: list[fitz.Rect] = []
        original = fitz.Page.get_pixmap
        spy = self

        def wrapped(page, *args, **kwargs):
            clip = kwargs.get("clip")
            if clip is not None:
                spy.clips.append(fitz.Rect(clip))
            return original(page, *args, **kwargs)

        monkeypatch.setattr(fitz.Page, "get_pixmap", wrapped)


def _render(page, bbox, clip_limits=None):
    path = _pipe()._render_adjudication_crop(
        fitz_path(page), 1, tuple(bbox), clip_limits=clip_limits
    )
    assert path is not None
    return path


def _ink(path) -> int:
    with Image.open(path) as im:
        return sum(1 for px in im.convert("L").getdata() if px < 255)


def _size(path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def _target_word(page):
    """A value word near the middle of the table (text-layer, unrotated frame)."""
    words = sorted(_value_words(page), key=lambda w: (w[1], w[0]))
    return words[len(words) // 2]


def _word_in_same_place(control, page, word):
    """The word of ``page`` with the same text at the same unrotated position."""
    hits = [w for w in page.get_text("words") if w[4] == word[4]]
    assert hits
    return min(hits, key=lambda w: abs(w[0] - word[0]) + abs(w[1] - word[1]))


@pytest.mark.parametrize("rotation", ROTATED)
def test_adjudication_crop_of_a_word_matches_the_rotate_0_control(make, monkeypatch, rotation):
    control = make(0, "booktabs")
    page = make(rotation, "booktabs")
    word = _target_word(control)
    assert _word_in_same_place(control, page, word)[:4] == pytest.approx(word[:4], abs=0.01)

    spy = _ClipSpy(monkeypatch)
    ctrl_path = _render(control, word[:4])
    got_path = _render(page, word[:4])
    ctrl_clip, got_clip = spy.clips

    # Independent oracle: the word in the DISPLAYED frame lies inside the crop clip.
    displayed = fitz.Rect(word[:4]) * page.rotation_matrix
    displayed.normalize()
    assert got_clip.contains(displayed), (got_clip, displayed)
    # Same crop area as the control, up to the quarter turn.
    assert sorted(_size(got_path)) == sorted(_size(ctrl_path))
    # The crop shows the glyphs: ink equals the control's (a wrong-frame crop is blank
    # or shows other cells), within 5% for antialiasing at the quarter turn.
    assert _ink(ctrl_path) > 0
    assert _ink(got_path) == pytest.approx(_ink(ctrl_path), rel=0.05)


@pytest.mark.parametrize("rotation", ROTATED)
def test_adjudication_clip_limits_are_converted_too(make, monkeypatch, rotation):
    control = make(0, "booktabs")
    page = make(rotation, "booktabs")
    word = _target_word(control)
    x0, y0, x1, y1 = word[:4]
    # A cell band around the word, in text-layer coordinates, clearly tighter than
    # the padded word box in x so the limit is what decides the clip.
    pad = CROP_PADDING_PT / 2
    limits = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    spy = _ClipSpy(monkeypatch)
    ctrl_path = _render(control, word[:4], clip_limits=limits)
    got_path = _render(page, word[:4], clip_limits=limits)
    _, got_clip = spy.clips

    displayed_limits = fitz.Rect(limits) * page.rotation_matrix
    displayed_limits.normalize()
    displayed_word = fitz.Rect(word[:4]) * page.rotation_matrix
    displayed_word.normalize()
    assert got_clip.contains(displayed_word)
    assert displayed_limits.contains(got_clip), (got_clip, displayed_limits)
    assert sorted(_size(got_path)) == sorted(_size(ctrl_path))
    assert _ink(got_path) == pytest.approx(_ink(ctrl_path), rel=0.05)


def _legacy_adjudication_png(page, bbox, clip_limits):
    """The pre-change rendering: pad and clamp the raw bbox, render, no conversion."""
    from socr.core.born_digital import upright_rotation_for
    from socr.tables.extract import DEFAULT_CROP_DPI

    r = page.rect
    x0, y0, x1, y1 = bbox
    clip = fitz.Rect(
        max(r.x0, x0 - CROP_PADDING_PT),
        max(r.y0, y0 - CROP_PADDING_PT),
        min(r.x1, x1 + CROP_PADDING_PT),
        min(r.y1, y1 + CROP_PADDING_PT),
    )
    if clip_limits is not None:
        clip &= fitz.Rect(clip_limits)
    mat = fitz.Matrix(DEFAULT_CROP_DPI / 72, DEFAULT_CROP_DPI / 72)
    rotation = upright_rotation_for(page, clip=clip)
    if rotation != 0:
        mat.prerotate(rotation)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    return pix.width, pix.height, pix.samples


@pytest.mark.parametrize("with_limits", (False, True))
def test_rotate_0_adjudication_crop_is_byte_identical_to_the_old_path(make, with_limits):
    page = make(0, "booktabs")
    word = _target_word(page)
    x0, y0, x1, y1 = word[:4]
    limits = (x0 - 3, y0 - 3, x1 + 3, y1 + 3) if with_limits else None
    path = _render(page, word[:4], clip_limits=limits)
    w, h, samples = _legacy_adjudication_png(page, word[:4], limits)
    with Image.open(path) as im:
        assert im.size == (w, h)
        assert im.convert("RGB").tobytes() == bytes(samples)


def _spy_rotation_clips(monkeypatch):
    """Record the clip each table crop site hands to ``render_rotation_for``."""
    import socr.core.born_digital as bd

    recorded: list = []
    original = bd.render_rotation_for

    def spy(page, clip=None):
        if clip is not None:
            recorded.append(fitz.Rect(clip))
        return original(page, clip=clip)

    monkeypatch.setattr(bd, "render_rotation_for", spy)
    return recorded


def _site_adjudication(page, box):
    path = _render(page, box.bbox)
    return _size(path)


@pytest.mark.parametrize(
    "site", [_site_extract, _site_witness, _site_source_evidence, _site_adjudication]
)
@pytest.mark.parametrize("rotation", ROTATED)
def test_rotation_probe_clip_selects_the_tables_words(make, monkeypatch, site, rotation):
    """``get_text(clip=)`` filters in the unrotated frame: the clip the sites pass to
    ``render_rotation_for`` must select the table's words there, as at /Rotate 0."""
    control = make(0, "booktabs")
    page = make(rotation, "booktabs")
    box = locate_tables(page)[0]

    recorded = _spy_rotation_clips(monkeypatch)
    site(page, box)
    assert recorded, "the site never probed the clip rotation"
    probe = recorded[-1]
    table_words = [w for w in page.get_text("words") if fitz.Rect(w[:4]).intersects(box.bbox)]
    selected = page.get_text("words", clip=probe)
    # Every table word is selected (padding may add neighbours; the control proves
    # the table has none outside its box).
    assert {w[:5] for w in table_words} <= {w[:5] for w in selected}

    recorded.clear()
    site(control, locate_tables(control)[0])
    ctrl_selected = control.get_text("words", clip=recorded[-1])
    assert len(selected) == len(ctrl_selected)
