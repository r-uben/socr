"""#735 / #734: the Stage 1 reader must not fabricate on the Fed SEP pages.

Every chart here is DRAWN by the test, so what the reader ought to say is what
the drawing was made from, not an oracle read off a document. Each test pins a
DIFFERENCE -- the same page twice with one thing changed -- so nothing here can
pin an outcome that happens to hold on one machine or one provider state.

The four defects, each with its own difference:

* an invisible wrapper rectangle bridged two panels into one region (#734);
* the tick ladders drawn at the two ends of one axis disagreed by less than
  their own stroke width, and exact equality discarded the axis;
* the bins were "the densest row below the axis", which a page's FOOTNOTE wins;
* the calibration gated measured heights but not the zeros read off absence;
* a legend swatch set above a bin label was mistaken for a data mark;
* a staircase emitted as one compound path was dropped, and every bin of that
  series then fell to the axis and published a hard zero.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from socr.figures.chart_reader import INTEGER, PRESENT, read_chart_page
from socr.tables.reconstruct import chart_region_bboxes

X0, X1 = 100.0, 400.0
BASE = 400.0
UNIT = 10.0  # points per participant
TICK_VALUES = (2, 4, 6, 8, 10)
BINS = ("B1", "B2", "B3", "B4")
SOLID = "September projections"
DASHED = "June projections"


def _bin_geometry(n: int) -> tuple[float, list[float]]:
    width = (X1 - X0 - 20.0) / n
    return width, [X0 + 10.0 + width * (i + 0.5) for i in range(n)]


def draw_panel(
    path: Path,
    *,
    bars: dict[str, int] | None = None,
    tick_labels: tuple[str, ...] | None = None,
    right_ladder_offset: float = 0.0,
    tick_width: float = 0.4,
    draw_bin_labels: bool = True,
    footnote: str | None = None,
    legend_over_bin_centre: bool = False,
    dashed_levels: tuple[int, ...] | None = None,
    dashed_compound: bool = False,
) -> tuple[fitz.Document, list]:
    """One panel: an axis with a ladder at each end, bins, a legend, bars.

    The knobs are exactly the things the four fixes turn on, and nothing else,
    so a test can change one of them and hold the rest of the drawing fixed.
    """
    labels = list(BINS)
    bin_w, centres = _bin_geometry(len(labels))
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    page.draw_line(fitz.Point(X0, BASE), fitz.Point(X1, BASE), width=tick_width)
    texts = tick_labels if tick_labels is not None else tuple(str(v) for v in TICK_VALUES)
    for value, text in zip(TICK_VALUES, texts, strict=True):
        y = BASE - value * UNIT
        page.draw_line(fitz.Point(X0, y), fitz.Point(X0 + 10.0, y), width=tick_width)
        page.draw_line(
            fitz.Point(X1 - 10.0, y + right_ladder_offset),
            fitz.Point(X1, y + right_ladder_offset),
            width=tick_width,
        )
        # Tick labels go OUTSIDE the plot's span, which is where calibrate_y
        # looks for them and where no bar or bin label can be.
        page.insert_text(fitz.Point(X1 + 5.0, y + 2.0), text, fontsize=5)

    if draw_bin_labels:
        for centre, label in zip(centres, labels, strict=True):
            page.insert_text(fitz.Point(centre - len(label) * 1.4, BASE + 10.0), label, fontsize=5)
    if footnote is not None:
        # A footnote is set across the page, so it reaches outside the axis'
        # own span -- which is the whole of what tells it from a label row.
        page.insert_text(fitz.Point(X0 - 40.0, BASE + 40.0), footnote, fontsize=5)

    swatch_cx = centres[1] if legend_over_bin_centre else (centres[0] + centres[1]) / 2.0
    sx0, sx1 = swatch_cx - 4.0, swatch_cx + 4.0
    sy = BASE - 100.0
    page.draw_rect(
        fitz.Rect(sx0, sy - 2.0, sx1, sy + 2.0),
        color=(0, 0, 0),
        fill=(0.4, 0.6, 0.8),
        width=0.3,
    )
    page.insert_text(fitz.Point(sx1 + 4.0, sy + 1.5), SOLID, fontsize=5)
    if dashed_levels is not None:
        dy = BASE - 90.0
        page.draw_line(
            fitz.Point(sx0, dy),
            fitz.Point(sx1, dy),
            width=1.5,
            dashes="[2 2] 0",
            color=(0, 0.4, 0.7),
        )
        page.insert_text(fitz.Point(sx1 + 4.0, dy + 1.5), DASHED, fontsize=5)

    for i, label in enumerate(labels):
        count = (bars or {}).get(label, 0)
        if count <= 0:
            continue
        page.draw_rect(
            fitz.Rect(
                centres[i] - bin_w * 0.4, BASE - count * UNIT, centres[i] + bin_w * 0.4, BASE
            ),
            color=(0, 0, 0),
            fill=(0.4, 0.6, 0.8),
            width=0.3,
        )

    if dashed_levels is not None:
        edges = [X0 + 10.0 + bin_w * i for i in range(len(labels) + 1)]
        ys = [BASE - c * UNIT for c in dashed_levels]
        segments: list[tuple[fitz.Point, fitz.Point]] = []
        prev = BASE
        for i, y in enumerate(ys):
            if y != prev:
                segments.append(
                    (fitz.Point(edges[i], min(y, prev)), fitz.Point(edges[i], max(y, prev)))
                )
            if y != BASE:
                segments.append((fitz.Point(edges[i], y), fitz.Point(edges[i + 1], y)))
            prev = y
        if prev != BASE:
            segments.append(
                (fitz.Point(edges[-1], min(prev, BASE)), fitz.Point(edges[-1], max(prev, BASE)))
            )
        style = {"width": 1.5, "dashes": "[2 2] 0", "color": (0, 0.4, 0.7)}
        if dashed_compound:
            # ONE path carrying the whole staircase, which is what the Fed's
            # generator emits: nine items under a single bounding rectangle.
            shape = page.new_shape()
            for a, b in segments:
                shape.draw_line(a, b)
            shape.finish(**style)
            shape.commit()
        else:
            for a, b in segments:
                page.draw_line(a, b, **style)

    doc.save(str(path))
    reopened = fitz.open(str(path))
    return reopened, [reopened[0].rect]


def read_panel(tmp_path: Path, name: str, **kw):
    """``(panel or None, refusal or None)`` for a one-panel page."""
    doc, bboxes = draw_panel(tmp_path / f"{name}.pdf", **kw)
    reading = read_chart_page(doc[0], bboxes, page_num=1)
    return reading.panels.get(1), reading.refusals.get(1)


def series_counts(panel, name: str) -> list:
    for s in panel.series:
        if s.name == name:
            return [c.count if c.status == INTEGER else None for c in s.cells]
    raise AssertionError(f"no series {name!r} in {[s.name for s in panel.series]}")


# ---------------------------------------------------------------------------
# #734 -- a drawing that paints nothing is not ink
# ---------------------------------------------------------------------------


def _two_clusters(path: Path, wrapper_fill, wrapper_opacity: float) -> fitz.Document:
    """Two ink blocks a clear gap apart, inside one wrapper rectangle."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for top in (100.0, 400.0):
        for row in range(5):
            page.draw_rect(
                fitz.Rect(100.0, top + row * 30.0, 260.0, top + row * 30.0 + 20.0),
                color=(0, 0, 0),
                fill=(0.2, 0.4, 0.8),
                width=0.5,
            )
    page.draw_rect(
        fitz.Rect(90.0, 90.0, 270.0, 560.0),
        color=None,
        fill=wrapper_fill,
        fill_opacity=wrapper_opacity,
    )
    doc.save(str(path))
    return fitz.open(str(path))


def test_an_invisible_wrapper_no_longer_bridges_two_panels(tmp_path: Path) -> None:
    """The wrapper's COLOUR is the only difference, and it decides the count.

    White on white paints nothing, so it can neither anchor a cluster nor
    bridge two; the same rectangle in a colour the page shows is ink, and the
    two blocks are then genuinely one region. On the SEP pages the invisible
    wrapper is what collapsed five panels into one (#734).
    """
    invisible = chart_region_bboxes(_two_clusters(tmp_path / "white.pdf", (1, 1, 1), 1.0)[0])
    visible = chart_region_bboxes(_two_clusters(tmp_path / "grey.pdf", (0.8, 0.8, 0.8), 1.0)[0])
    assert len(invisible) == 2
    assert len(visible) == 1


def test_a_fully_transparent_wrapper_does_not_bridge_either(tmp_path: Path) -> None:
    """Opacity is read the same way colour is: zero alpha shows nothing."""
    clear = chart_region_bboxes(_two_clusters(tmp_path / "clear.pdf", (0.8, 0.8, 0.8), 0.0)[0])
    opaque = chart_region_bboxes(_two_clusters(tmp_path / "opaque.pdf", (0.8, 0.8, 0.8), 1.0)[0])
    assert len(clear) == 2
    assert len(opaque) == 1


# ---------------------------------------------------------------------------
# The tick ladders at the two ends of one axis
# ---------------------------------------------------------------------------


def test_ladders_that_disagree_within_their_own_stroke_still_find_the_axis(
    tmp_path: Path,
) -> None:
    """Only the right ladder's offset changes, and the stroke width bounds it.

    The Fed's generator rounds the two copies of one ladder apart by up to
    0.053pt against a 0.336pt tick stroke. Half a stroke width is the finest
    an edge can be located at, so a disagreement inside it is one ladder drawn
    twice; a disagreement many strokes wide is two different ladders and the
    axis is refused.
    """
    inside, refusal_inside = read_panel(
        tmp_path, "inside", bars={"B1": 3}, right_ladder_offset=0.15, tick_width=0.4
    )
    outside, refusal_outside = read_panel(
        tmp_path, "outside", bars={"B1": 3}, right_ladder_offset=3.0, tick_width=0.4
    )
    assert inside is not None, refusal_inside
    assert outside is None
    assert (
        "no axis" in (refusal_outside or "").lower() or "frame" in (refusal_outside or "").lower()
    )


def test_the_agreement_window_is_the_stroke_width_not_a_constant(tmp_path: Path) -> None:
    """One offset, two stroke widths: the thicker stroke admits it."""
    offset = 0.6
    thin, _ = read_panel(
        tmp_path, "thin", bars={"B1": 3}, right_ladder_offset=offset, tick_width=0.4
    )
    thick, _ = read_panel(
        tmp_path, "thick", bars={"B1": 3}, right_ladder_offset=offset, tick_width=2.0
    )
    assert thin is None
    assert thick is not None


# ---------------------------------------------------------------------------
# The P1: the bins are the axis' own label row, never the densest row
# ---------------------------------------------------------------------------

FOOTNOTE = "Definitions of variables and other information are available in the note"


def test_a_footnote_longer_than_the_label_row_changes_nothing(tmp_path: Path) -> None:
    """The footnote wins a popularity contest and loses a geometric one.

    It is set across the page, so it reaches outside the axis' own span, and
    it is not the first row below the axis. The reading is identical with and
    without it -- bins, counts and all.
    """
    plain, _ = read_panel(tmp_path, "plain", bars={"B1": 3, "B3": 5})
    noted, _ = read_panel(tmp_path, "noted", bars={"B1": 3, "B3": 5}, footnote=FOOTNOTE)
    assert plain is not None and noted is not None
    assert [b.label for b in plain.bins] == list(BINS)
    assert [b.label for b in noted.bins] == list(BINS)
    assert series_counts(plain, SOLID) == series_counts(noted, SOLID)


def test_without_a_label_row_the_footnote_is_not_promoted_to_bins(tmp_path: Path) -> None:
    """Drawing the label row is the difference between a panel and a refusal.

    This is the #735 P1 in one assertion: with the labels drawn the panel
    reads; with them removed the page still carries the footnote, and the
    reader must publish NOTHING rather than a table of footnote words with
    hard zeros beneath them.
    """
    labelled, _ = read_panel(tmp_path, "labelled", bars={"B1": 3}, footnote=FOOTNOTE)
    unlabelled, refusal = read_panel(
        tmp_path, "unlabelled", bars={"B1": 3}, footnote=FOOTNOTE, draw_bin_labels=False
    )
    assert labelled is not None
    assert [b.label for b in labelled.bins] == list(BINS)
    assert unlabelled is None
    assert refusal
    assert "Definitions" not in refusal


# ---------------------------------------------------------------------------
# The calibration gates the zeros too
# ---------------------------------------------------------------------------


def test_a_scale_that_does_not_close_publishes_no_cell_not_even_a_zero(
    tmp_path: Path,
) -> None:
    """Only the top tick's printed VALUE changes; the drawing is identical.

    Mislabelling it makes the least-squares fit miss every tick by many
    points, far more than the half a participant a count occupies. The bars
    are then refused by ``_resolve`` as before -- but so are the empty bins,
    which used to read a confident 0 off a scale the reader had just failed to
    fit. That zero path is the one the SEP pages published through.
    """
    good, _ = read_panel(tmp_path, "good", bars={"B1": 3})
    bad, refusal = read_panel(
        tmp_path, "bad", bars={"B1": 3}, tick_labels=("2", "4", "6", "8", "30")
    )
    assert good is not None
    assert series_counts(good, SOLID) == [3, 0, 0, 0]
    assert bad is None
    assert refusal and "half a count" in refusal


# ---------------------------------------------------------------------------
# The legend set inside the plot, above a bin label
# ---------------------------------------------------------------------------


def test_a_swatch_above_a_bin_label_is_still_a_legend(tmp_path: Path) -> None:
    """The swatch moves over a bin label centre and nothing else changes.

    The SEP legends from Dec 2020 to Dec 2023 are set inside the plot directly
    above a bin label, so every swatch covered a label centre; treating that
    alone as the mark of a data bar discarded the legend and with it both
    series names. A data mark also spans its whole bin, and a swatch does not.
    """
    clear, _ = read_panel(tmp_path, "clear", bars={"B1": 3, "B3": 5})
    over, _ = read_panel(tmp_path, "over", bars={"B1": 3, "B3": 5}, legend_over_bin_centre=True)
    assert clear is not None and over is not None
    assert [s.name for s in clear.series] == [SOLID]
    assert [s.name for s in over.series] == [SOLID]
    assert series_counts(clear, SOLID) == series_counts(over, SOLID)


def test_the_series_name_stops_at_the_plots_edge(tmp_path: Path) -> None:
    """The y tick labels share the legend's rows and must not join its name."""
    panel, _ = read_panel(tmp_path, "named", bars={"B1": 3})
    assert panel is not None
    assert [s.name for s in panel.series] == [SOLID]


# ---------------------------------------------------------------------------
# A staircase drawn as one compound path
# ---------------------------------------------------------------------------


def test_a_compound_staircase_is_unresolved_not_a_column_of_zeros(
    tmp_path: Path,
) -> None:
    """The SAME staircase, emitted as separate lines and as one path.

    Drawn as separate operators the reader recovers the levels. Drawn as one
    compound path it recovers nothing -- and the failure has to surface as
    UNRESOLVED, because with no run and no riser left every bin falls to the
    axis and a fabricated zero looks exactly like an observed one. That is how
    the SEP pages published a hard-zero column for the prior meeting.
    """
    levels = (4, 4, 2, 2)
    separate, _ = read_panel(tmp_path, "sep", bars={"B1": 3}, dashed_levels=levels)
    compound, _ = read_panel(
        tmp_path, "cmp", bars={"B1": 3}, dashed_levels=levels, dashed_compound=True
    )
    assert separate is not None and compound is not None
    assert series_counts(separate, DASHED) == list(levels)
    got = series_counts(compound, DASHED)
    assert got == [None] * len(levels)
    detail = next(s.detail for s in compound.series if s.name == DASHED)
    assert "cannot decompose" in detail


# ---------------------------------------------------------------------------
# The reference page, which must not move
# ---------------------------------------------------------------------------

DOTPLOT_PDF = Path.home() / "Data/socr/fixtures/dotplot/dotplot-p20.pdf"


@pytest.mark.skipif(not DOTPLOT_PDF.exists(), reason="corpus fixture not present")
def test_every_published_reference_cell_sits_on_a_numeric_range_bin() -> None:
    """Provenance, stated as a property rather than as a count.

    The reference is a dot plot whose bins are printed numeric ranges. No cell
    it publishes may be keyed by anything else -- which is the property the
    footnote columns of #735 violated, and the one the per-page measurement in
    docs/log/2026-09-12_735-sep-reader.md reports as zero across all 23 pages.
    """
    doc = fitz.open(str(DOTPLOT_PDF))
    page = doc[0]
    reading = read_chart_page(page, chart_region_bboxes(page), page_num=20)
    assert reading.refusals == {}, reading.refusals
    assert len(reading.panels) == 5
    published = 0
    for panel in reading.panels.values():
        keyed = {b.label for b in panel.bins}
        for s in panel.series:
            if s.presence != PRESENT:
                continue
            for cell in s.cells:
                if cell.status != INTEGER:
                    continue
                published += 1
                assert cell.bin_label in keyed
                for atom in cell.bin_label.replace("−", "-").split("-"):
                    assert atom.strip().replace(".", "").isdigit(), cell.bin_label
    assert published > 0
