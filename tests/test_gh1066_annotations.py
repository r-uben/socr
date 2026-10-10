"""GH-1066: a reader's annotation is not chart data.

PyMuPDF's ``get_drawings()`` includes annotation appearances, so a Highlight (a
filled coloured rectangle) passed ``_has_vector_data_marks`` and a highlighted
prose page was routed to the chart lane. Detection now runs on the page content
without annotations, but never drops drawings by annotation rectangle: a Square
a reader draws around a real chart must not hide it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.figures.extractor import has_chart_marks  # noqa: E402

HIGHLIGHT = (1, 1, 0)


def _prose_page(doc: fitz.Document) -> fitz.Page:
    page = doc.new_page(width=612, height=792)
    for i in range(30):
        page.insert_text(
            (72, 72 + i * 14), f"ordinary prose line {i} of a normal page", fontsize=10
        )
    return page


def _chart(page: fitz.Page) -> None:
    cols = [(0.9, 0.1, 0.1), (0.1, 0.1, 0.9), (0.1, 0.8, 0.1), (0.9, 0.1, 0.1), (0.1, 0.1, 0.9)]
    for i, (col, x) in enumerate(zip(cols, [100, 180, 260, 340, 420])):
        page.draw_rect(fitz.Rect(x, 500 - i * 40, x + 60, 680), color=col, fill=col, width=1)


def _highlight(page: fitz.Page) -> None:
    """Three stacked line highlights over prose.

    One highlight is a single drawing and too small to be a chart cluster. The
    detector needs MIN_DATA_MARKS coloured fills in a cluster of at least
    CHART_MIN_CLUSTER_AREA, which a reader's multi-line highlight reaches (the
    #1055 reproducer carried three fills).
    """
    for k in range(3):
        page.add_highlight_annot(fitz.Rect(72, 100 + k * 40, 540, 114 + k * 40)).update()


def _reopen(doc: fitz.Document, tmp_path: Path) -> fitz.Document:
    pdf = tmp_path / "p.pdf"
    doc.save(pdf)
    doc.close()
    return fitz.open(pdf)


def _has_highlight_fill(page: fitz.Page) -> bool:
    return any(d.get("fill") == HIGHLIGHT for d in page.get_drawings())


def test_highlighted_prose_page_is_not_a_chart(tmp_path: Path) -> None:
    doc = fitz.open()
    _highlight(_prose_page(doc))
    doc = _reopen(doc, tmp_path)
    page = doc[0]
    assert page.first_annot is not None
    assert _has_highlight_fill(page), "premise: raw get_drawings() sees the highlight fill"
    assert has_chart_marks(page) is False


def test_prose_page_without_annotation_is_not_a_chart(tmp_path: Path) -> None:
    doc = fitz.open()
    _prose_page(doc)
    doc = _reopen(doc, tmp_path)
    page = doc[0]
    assert page.first_annot is None
    assert not _has_highlight_fill(page)
    assert has_chart_marks(page) is False


def test_real_chart_with_highlight_is_still_a_chart(tmp_path: Path) -> None:
    doc = fitz.open()
    page = _prose_page(doc)
    _chart(page)
    _highlight(page)
    doc = _reopen(doc, tmp_path)
    page = doc[0]
    assert page.first_annot is not None
    assert _has_highlight_fill(page)
    assert has_chart_marks(page) is True


def test_real_chart_inside_reader_square_is_still_a_chart(tmp_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 40), "GDP Growth Forecast 2025", fontsize=14)
    _chart(page)
    sq = page.add_rect_annot(fitz.Rect(90, 380, 500, 690))
    sq.set_colors(stroke=(1, 0, 0))
    sq.update()
    doc = _reopen(doc, tmp_path)
    page = doc[0]
    assert page.first_annot is not None
    assert page.first_annot.type[1] == "Square"
    assert has_chart_marks(page) is True


def test_a_hidden_layer_stays_hidden_on_an_annotated_page() -> None:
    """#1066 (Astra review): ``insert_pdf`` drops optional-content settings.

    A page whose chart-like marks sit on a layer that is OFF, plus a highlight:
    the original page draws only the highlight, so it is not a chart page. An
    annotation-free copy would lose the layer configuration and draw the hidden
    bars, turning an irrelevant annotation into a routing change.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(20):
        page.insert_text(
            (72, 72 + i * 14), f"ordinary prose line {i} of a normal page", fontsize=10
        )
    hidden = doc.add_ocg("hidden layer", on=False)
    for k in range(6):
        shape = page.new_shape()
        shape.draw_rect(fitz.Rect(100 + k * 50, 500, 130 + k * 50, 700 - k * 20))
        shape.finish(color=(0, 0, 0), fill=(0.9, 0.2, 0.2), oc=hidden)
        shape.commit()
    page.add_highlight_annot(fitz.Rect(72, 100, 540, 112))
    reopened = fitz.open("pdf", doc.tobytes())
    try:
        target = reopened[0]
        fills = [d.get("fill") for d in target.get_drawings()]
        assert fills == [(1.0, 1.0, 0.0)], f"fixture: only the highlight should draw: {fills}"
        assert target.parent.get_ocgs(), "fixture: the document must carry optional content"
        assert not has_chart_marks(target), (
            "an annotation-free copy redrew a hidden layer and made this a chart page"
        )
    finally:
        reopened.close()
