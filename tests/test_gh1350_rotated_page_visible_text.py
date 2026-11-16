"""GH-1350: the visible-ink filter must keep printed characters on rotated pages.

``rawdict`` char boxes are in the page's UNROTATED space; ``get_pixmap`` / ``clip`` work
in the ROTATED space. Before the fix a ``/Rotate`` 90 or 270 page lost most printed
numbers while an upright copy of the same content kept them all. The tests pin
DIFFERENCES across rotations of identical content, not machine-specific values.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.tables.source_evidence import (  # noqa: E402
    _tokens_from_page_text_layer,
    build_born_digital_evidence,
    collect_table_tokens,
    verify_table_tokens,
)

ROTATIONS = (0, 90, 180, 270)
VISIBLE = ("4821", "7733", "1957", "6104")
HIDDEN_WHITE = "9911"
HIDDEN_ALPHA = "8822"
HIDDEN_UNDER_IMAGE = "5533"


def _page(rotation: int):
    """Same content for every rotation: visible numbers plus three kinds of hidden ones."""
    doc = fitz.open()
    blank = fitz.open()
    page = doc.new_page(width=500, height=700)
    blank_page = blank.new_page(width=200, height=60)
    fs = {"fontsize": 14}
    for i, value in enumerate(VISIBLE):
        page.insert_text((60, 80 + 30 * i), f"item {value}", **fs)
    page.insert_text((60, 240), HIDDEN_WHITE, color=(1, 1, 1), **fs)
    page.insert_text((60, 270), HIDDEN_ALPHA, render_mode=3, **fs)
    page.insert_text((60, 300), HIDDEN_UNDER_IMAGE, **fs)
    page.insert_image(
        fitz.Rect(50, 280, 250, 310),
        pixmap=blank_page.get_pixmap(),
        keep_proportion=False,
    )
    page.set_rotation(rotation)
    blank.close()
    return doc, page


def _numbers(page):
    return _tokens_from_page_text_layer(page)[0]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_visible_numbers_survive_and_hidden_ones_are_dropped(rotation: int) -> None:
    doc, page = _page(rotation)
    try:
        assert page.rotation == rotation
        surviving = _numbers(page)
        for value in VISIBLE:
            assert surviving[value] == 1, (rotation, value)
        for hidden in (HIDDEN_WHITE, HIDDEN_ALPHA, HIDDEN_UNDER_IMAGE):
            assert surviving[hidden] == 0, (rotation, hidden)
    finally:
        doc.close()


def test_surviving_counter_is_identical_across_rotations() -> None:
    counters = {}
    for rotation in ROTATIONS:
        doc, page = _page(rotation)
        try:
            counters[rotation] = _numbers(page)
        finally:
            doc.close()
    for rotation in ROTATIONS:
        assert counters[rotation] == counters[0], rotation


def _evidence_and_verdict(rotation: int, values: tuple[str, ...]):
    doc, page = _page(rotation)
    try:
        table = "| label | value |\n| --- | --- |\n" + "\n".join(f"| item | {v} |" for v in values)
        tokens = collect_table_tokens(table)
        with patch("socr.tables.locate.locate_tables", return_value=[]):
            bundle = build_born_digital_evidence(page, ocr_image_fn=lambda _pix: "")
        return verify_table_tokens(bundle, tokens)
    finally:
        doc.close()


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_printed_table_passes_on_rotated_page(rotation: int) -> None:
    assert _evidence_and_verdict(rotation, VISIBLE).passed


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_unprinted_and_hidden_numbers_still_fail_on_rotated_page(rotation: int) -> None:
    for bogus in ("3141", HIDDEN_WHITE, HIDDEN_ALPHA, HIDDEN_UNDER_IMAGE):
        result = _evidence_and_verdict(rotation, (*VISIBLE[:2], bogus))
        assert not result.passed, (rotation, bogus)
