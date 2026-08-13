"""GH-150 A2 — hermetic, detector-only pins of wave 1's A1 framed-cluster behaviour.

These tests exercise only ``socr.figures.extractor.has_chart_marks`` (and its private
``_has_vector_data_marks`` companion, for two fixture self-checks) against in-memory,
serialize-and-reopen PDF fixtures built with PyMuPDF. There is no pipeline, no provider,
no network — nothing here calls ``_phase_agentic`` or ``process()``.

GH-150 B2 (wave 3) will append pipeline-level tests to this same file and, when it does,
MUST add the ``_available_engines_for_agentic`` patch for CI hermeticity (CI has no
ollama / no provider). Nothing in A2's tests needs that patch.
"""

import math

import pytest

fitz = pytest.importorskip("fitz")

from socr.figures.extractor import (
    CHART_FRAME_MIN_CLUSTER_COVERAGE,
    CHART_MIN_CLUSTER_AREA,
    CLUSTER_GAP,
    DATA_STROKE_MIN_WIDTH,
    MIN_DATA_MARKS,
    MIN_DRAWINGS_FOR_VECTOR,
    _has_vector_data_marks,
    has_chart_marks,
)

# US Letter, in points — page canvas size only, not a detector threshold.
PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0

# Every stroke in these fixtures is neutral and strictly below the data-mark stroke
# gate, so `_has_vector_data_marks` stays False and the framed-cluster path (or its
# absence) is what each test actually exercises.
NEUTRAL_GRAY = (0.35, 0.35, 0.35)
STROKE_WIDTH = DATA_STROKE_MIN_WIDTH * 0.6

# Side length whose square equals the cluster-area gate; fixtures scale off this so a
# passing cluster clears CHART_MIN_CLUSTER_AREA by construction, not by luck.
_GATE_SIDE = math.sqrt(CHART_MIN_CLUSTER_AREA)
FRAME_WIDTH = _GATE_SIDE
FRAME_HEIGHT = _GATE_SIDE * 1.25


def _reopen(doc):
    """Serialize and reopen a fixture so assertions run against real PDF bytes."""
    data = doc.tobytes()
    doc.close()
    return fitz.open(stream=data, filetype="pdf")


def _rect_area(rect) -> float:
    return (rect.x1 - rect.x0) * (rect.y1 - rect.y0)


def _union_bbox(drawings):
    rects = [d["rect"] for d in drawings if d.get("rect") is not None]
    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)
    return fitz.Rect(x0, y0, x1, y1)


def _largest_rect_drawing(drawings):
    return max(drawings, key=lambda d: _rect_area(d["rect"]))


# --- Fixture builders -------------------------------------------------------------


def _make_framed_spike_doc():
    """(a) A thin-stroke spike plot: enclosing axes rect + interior spike marks.

    Fixture constraint from the A1 ruling (TICKETS.md:30-36): the frame alone is not
    enough — it must also carry >= MIN_DRAWINGS_FOR_VECTOR separately-drawn interior
    marks, or the framed-cluster gate correctly returns False and this fixture would
    be mistaken for an A1 regression instead of its own bug.
    """
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    x0, y0 = 100.0, 100.0
    frame = fitz.Rect(x0, y0, x0 + FRAME_WIDTH, y0 + FRAME_HEIGHT)
    page.draw_rect(frame, color=NEUTRAL_GRAY, width=STROKE_WIDTH)

    n_spikes = MIN_DRAWINGS_FOR_VECTOR + 2
    inset = 6.0
    usable_width = FRAME_WIDTH - 2 * inset
    step = usable_width / (n_spikes - 1)
    base_y = frame.y1 - inset
    top_y_bound = frame.y0 + inset
    for i in range(n_spikes):
        x = frame.x0 + inset + i * step
        height = (i % 4 + 1) * (FRAME_HEIGHT - 2 * inset) / 5
        top_y = max(top_y_bound, base_y - height)
        page.draw_line(
            fitz.Point(x, base_y), fitz.Point(x, top_y), color=NEUTRAL_GRAY, width=STROKE_WIDTH
        )
    return doc


def _make_filled_bar_doc():
    """(b) A filled bar chart: MIN_DATA_MARKS separately-drawn, non-neutral bars."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    bar_width = _GATE_SIDE / 3
    bar_gap = CLUSTER_GAP / 3
    bar_height = _GATE_SIDE
    colors = [(0.8, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.1, 0.8)]
    assert len(colors) == MIN_DATA_MARKS
    x = 100.0
    y_bottom = 400.0
    for color in colors:
        rect = fitz.Rect(x, y_bottom - bar_height, x + bar_width, y_bottom)
        page.draw_rect(rect, color=color, fill=color, width=STROKE_WIDTH)
        x += bar_width + bar_gap
    return doc


def _make_prose_with_rules_doc():
    """(c) Dense prose plus exactly three thin neutral rules spaced past CLUSTER_GAP."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    prose = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5
    page.insert_textbox(fitz.Rect(50, 50, 550, 700), prose, fontsize=9)

    rule_spacing = CLUSTER_GAP * 2
    rule_width = 200.0
    x = 100.0
    for i in range(3):
        y = 150.0 + i * rule_spacing
        page.draw_line(
            fitz.Point(x, y), fitz.Point(x + rule_width, y), color=NEUTRAL_GRAY, width=STROKE_WIDTH
        )
    return doc


def _make_ruling_table_doc():
    """(d) A ruling-only grid, no outer rect, one cluster whose area beats the gate."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    x0, y0 = 100.0, 100.0
    n_lines = MIN_DRAWINGS_FOR_VECTOR + 1
    spacing = CLUSTER_GAP - 5
    span = (n_lines - 1) * spacing
    for i in range(n_lines):
        y = y0 + i * spacing
        page.draw_line(
            fitz.Point(x0, y), fitz.Point(x0 + span, y), color=NEUTRAL_GRAY, width=STROKE_WIDTH
        )
    for i in range(n_lines):
        x = x0 + i * spacing
        page.draw_line(
            fitz.Point(x, y0), fitz.Point(x, y0 + span), color=NEUTRAL_GRAY, width=STROKE_WIDTH
        )
    return doc


def _make_bordered_table_doc():
    """(e) Characterization only: outer rect + >= MIN_DRAWINGS_FOR_VECTOR interior rulings."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    x0, y0 = 100.0, 100.0
    outer = fitz.Rect(x0, y0, x0 + FRAME_WIDTH, y0 + FRAME_HEIGHT)
    page.draw_rect(outer, color=NEUTRAL_GRAY, width=STROKE_WIDTH)

    n_rules = MIN_DRAWINGS_FOR_VECTOR + 2
    inset = 6.0
    usable_height = FRAME_HEIGHT - 2 * inset
    step = usable_height / (n_rules - 1)
    for i in range(n_rules):
        y = outer.y0 + inset + i * step
        page.draw_line(
            fitz.Point(outer.x0 + inset, y),
            fitz.Point(outer.x1 - inset, y),
            color=NEUTRAL_GRAY,
            width=STROKE_WIDTH,
        )
    return doc


# --- Classification tests -----------------------------------------------------------


def test_framed_spike_plot_has_chart_marks():
    reopened = _reopen(_make_framed_spike_doc())
    assert has_chart_marks(reopened[0]) is True


def test_filled_bar_chart_has_chart_marks():
    reopened = _reopen(_make_filled_bar_doc())
    assert has_chart_marks(reopened[0]) is True


def test_prose_with_scattered_rules_has_no_chart_marks():
    reopened = _reopen(_make_prose_with_rules_doc())
    assert has_chart_marks(reopened[0]) is False


def test_ruling_only_table_has_no_chart_marks():
    reopened = _reopen(_make_ruling_table_doc())
    assert has_chart_marks(reopened[0]) is False


# --- Fixture-validity self-checks ----------------------------------------------------


def test_framed_spike_fixture_has_no_raster_images():
    """Closes the raster fast path (extractor.py:1003) so this pins the vector path."""
    reopened = _reopen(_make_framed_spike_doc())
    assert reopened[0].get_images() == []


def test_framed_spike_fixture_is_framed_cluster_not_data_marks():
    """Closes extractor.py:1058 — proves the TICKETS.md:30-36 fixture constraint."""
    reopened = _reopen(_make_framed_spike_doc())
    drawings = reopened[0].get_drawings()
    assert _has_vector_data_marks(drawings) is False

    frame = _largest_rect_drawing(drawings)
    frame_rect = frame["rect"]
    union_area = _rect_area(_union_bbox(drawings))
    assert _rect_area(frame_rect) / union_area >= CHART_FRAME_MIN_CLUSTER_COVERAGE

    interior = [
        d
        for d in drawings
        if d is not frame
        and d.get("rect") is not None
        and d["rect"].x0 >= frame_rect.x0
        and d["rect"].y0 >= frame_rect.y0
        and d["rect"].x1 <= frame_rect.x1
        and d["rect"].y1 <= frame_rect.y1
    ]
    assert len(interior) >= MIN_DRAWINGS_FOR_VECTOR


def test_filled_bar_fixture_is_data_marks_cluster():
    """Same area gate as extractor.py:1044, computed from the fixture's own rects."""
    reopened = _reopen(_make_filled_bar_doc())
    drawings = reopened[0].get_drawings()
    assert _has_vector_data_marks(drawings) is True

    rects = sorted((d["rect"] for d in drawings if d.get("rect") is not None), key=lambda r: r.x0)
    gaps = [rects[i + 1].x0 - rects[i].x1 for i in range(len(rects) - 1)]
    assert all(gap <= CLUSTER_GAP for gap in gaps)
    assert _rect_area(_union_bbox(drawings)) >= CHART_MIN_CLUSTER_AREA


# --- Characterization (documented defect, not a desired-behaviour pin) --------------


def test_bordered_table_currently_overreaches_chart_detection():
    # This documents a detector defect expected to flip False in a future extractor.py
    # ticket: post-B1, a bordered table whose native table detector does not fire is
    # chart-only and takes the whole-page PNG lane — the silent-content-loss shape —
    # and settling it needs an extractor.py change outside this write set, so it needs
    # a wave-3+ ticket decision.
    reopened = _reopen(_make_bordered_table_doc())
    assert has_chart_marks(reopened[0]) is True
