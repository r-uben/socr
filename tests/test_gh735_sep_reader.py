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


# ---------------------------------------------------------------------------
# Round 2 -- the bin row is the row the BARS corroborate
# ---------------------------------------------------------------------------


def draw_captioned(
    path: Path,
    *,
    caption: str | None,
    bars: dict[str, int],
    extra_bars: tuple[tuple[float, float, int], ...] = (),
) -> tuple[fitz.Document, list]:
    """A panel with an optional text row set BETWEEN the axis and its labels.

    This is the chart's own x-unit annotation, which every rule keyed on text
    position gets wrong in one direction or the other: the densest row is the
    footnote, the first row is the caption.
    """
    labels = list(BINS)
    bin_w, centres = _bin_geometry(len(labels))
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_line(fitz.Point(X0, BASE), fitz.Point(X1, BASE), width=0.4)
    for value in TICK_VALUES:
        y = BASE - value * UNIT
        page.draw_line(fitz.Point(X0, y), fitz.Point(X0 + 10.0, y), width=0.4)
        page.draw_line(fitz.Point(X1 - 10.0, y), fitz.Point(X1, y), width=0.4)
        page.insert_text(fitz.Point(X1 + 5.0, y + 2.0), str(value), fontsize=5)
    if caption is not None:
        mid = (centres[0] + centres[-1]) / 2.0
        page.insert_text(fitz.Point(mid - 20.0, BASE + 8.0), caption, fontsize=5)
    for centre, label in zip(centres, labels, strict=True):
        page.insert_text(fitz.Point(centre - len(label) * 1.4, BASE + 22.0), label, fontsize=5)
    swatch_cx = (centres[0] + centres[1]) / 2.0
    sx0, sx1 = swatch_cx - 4.0, swatch_cx + 4.0
    sy = BASE - 100.0
    page.draw_rect(
        fitz.Rect(sx0, sy - 2.0, sx1, sy + 2.0),
        color=(0, 0, 0),
        fill=(0.4, 0.6, 0.8),
        width=0.3,
    )
    page.insert_text(fitz.Point(sx1 + 4.0, sy + 1.5), SOLID, fontsize=5)
    for i, label in enumerate(labels):
        count = bars.get(label, 0)
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
    for bx0, bx1, count in extra_bars:
        page.draw_rect(
            fitz.Rect(bx0, BASE - count * UNIT, bx1, BASE),
            color=(0, 0, 0),
            fill=(0.4, 0.6, 0.8),
            width=0.3,
        )
    doc.save(str(path))
    reopened = fitz.open(str(path))
    return reopened, [reopened[0].rect]


def read_captioned(tmp_path: Path, name: str, caption: str | None, bars, **kw):
    doc, bboxes = draw_captioned(tmp_path / f"{name}.pdf", caption=caption, bars=bars, **kw)
    reading = read_chart_page(doc[0], bboxes, page_num=1)
    return reading.panels.get(1), reading.refusals.get(1)


BARS = {"B1": 3, "B2": 5, "B3": 4, "B4": 2}


def test_a_unit_row_between_the_axis_and_the_labels_is_not_the_bins(tmp_path: Path) -> None:
    """The caption is the only difference, and it must change nothing.

    Taking the first row below the axis put a two-word ``Percent range`` in the
    bins, absorbed the real labels into it as a second atom line, and published
    one count under a column the page never labelled. The bars decide instead:
    each covers exactly one printed label centre and none covers exactly one
    caption word, so the labels win on the drawing's own evidence.
    """
    plain, plain_refusal = read_captioned(tmp_path, "plain", None, BARS)
    capt, capt_refusal = read_captioned(tmp_path, "capt", "Percent range", BARS)
    assert plain is not None, plain_refusal
    assert capt is not None, capt_refusal
    assert [b.label for b in plain.bins] == list(BINS)
    assert [b.label for b in capt.bins] == list(BINS)
    assert series_counts(plain, SOLID) == [3, 5, 4, 2]
    assert series_counts(capt, SOLID) == series_counts(plain, SOLID)


def test_a_five_word_caption_publishes_nothing_of_its_own(tmp_path: Path) -> None:
    """A longer caption would have won the old density contest as well.

    Whatever the reader ends up publishing, no cell of it may be keyed by a
    word of the caption -- that is the whole of what #735 is about.
    """
    caption = "Percent range of the projection"
    panel, refusal = read_captioned(tmp_path, "capt5", caption, BARS)
    if panel is None:
        assert refusal
        return
    words = set(caption.split())
    assert not (words & {b.label for b in panel.bins}), [b.label for b in panel.bins]
    assert [b.label for b in panel.bins] == list(BINS)


def test_with_one_candidate_row_the_bars_are_not_asked(tmp_path: Path) -> None:
    """Corroboration decides WHICH row, never whether a bar is good.

    The bar here straddles a bin boundary, so it covers no label centre and
    corroborates nothing. With a single candidate row there is nothing for it
    to pick between, so the panel still reads and the bar refuses only the bins
    it casts doubt over. Add the caption and a choice appears that no bar can
    settle, so the panel is refused rather than read against either row. The
    caption is the only difference between the two drawings.
    """
    _bin_w, centres = _bin_geometry(len(BINS))
    straddle = ((centres[0] + centres[1]) / 2.0 - 5.0, (centres[0] + centres[1]) / 2.0 + 5.0, 5)
    alone, alone_refusal = read_captioned(tmp_path, "alone", None, {}, extra_bars=(straddle,))
    contested, contested_refusal = read_captioned(
        tmp_path, "contested", "Percent range", {}, extra_bars=(straddle,)
    )
    assert alone is not None, alone_refusal
    assert [b.label for b in alone.bins] == list(BINS)
    assert contested is None
    assert contested_refusal and "not corroborated" in contested_refusal


# ---------------------------------------------------------------------------
# Round 2 -- "paints nothing" is measured against the page's own ground
# ---------------------------------------------------------------------------


def _ground_page(path: Path, ground, wrapper) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    if ground is not None:
        page.draw_rect(fitz.Rect(0, 0, 612, 792), color=None, fill=ground)
    for top in (100.0, 400.0):
        for row in range(5):
            page.draw_rect(
                fitz.Rect(100.0, top + row * 30.0, 260.0, top + row * 30.0 + 20.0),
                color=(0, 0, 0),
                fill=(0.2, 0.4, 0.8),
                width=0.5,
            )
    page.draw_rect(fitz.Rect(90.0, 90.0, 270.0, 560.0), color=None, fill=wrapper)
    doc.save(str(path))
    return fitz.open(str(path))


def test_white_is_not_privileged_the_page_ground_is(tmp_path: Path) -> None:
    """Four pages, one wrapper rectangle, two grounds. The colour decides.

    On a white page a white wrapper is invisible and a grey one is ink. On a
    page painted dark the roles swap exactly: the wrapper in the page's own
    dark colour is invisible, and a WHITE wrapper is ink. A test against the
    literal white would get the dark page backwards in both directions.
    """
    dark = (0.1, 0.1, 0.1)
    on_white = {
        "white": len(chart_region_bboxes(_ground_page(tmp_path / "ww.pdf", None, (1, 1, 1))[0])),
        "grey": len(
            chart_region_bboxes(_ground_page(tmp_path / "wg.pdf", None, (0.8, 0.8, 0.8))[0])
        ),
    }
    on_dark = {
        "ground": len(chart_region_bboxes(_ground_page(tmp_path / "dg.pdf", dark, dark)[0])),
        "white": len(chart_region_bboxes(_ground_page(tmp_path / "dw.pdf", dark, (1, 1, 1))[0])),
    }
    assert on_white == {"white": 2, "grey": 1}
    assert on_dark == {"ground": 2, "white": 1}


def test_a_white_wrapper_declared_in_cmyk_is_still_invisible(tmp_path: Path) -> None:
    """CMYK white is (0,0,0,0); the comparison converts before it compares."""
    doc = _ground_page(tmp_path / "cmyk.pdf", None, (0.0, 0.0, 0.0, 0.0))
    assert len(chart_region_bboxes(doc[0])) == 2


# ---------------------------------------------------------------------------
# Round 2 -- the derived bin label is a well-formed Stage 0 key
# ---------------------------------------------------------------------------


def test_a_two_line_range_label_joins_with_exactly_one_dash() -> None:
    """The page sets the connector itself; the join must not add a second.

    The Fed prints `2.13-` over `2.37` with a U+2212, so joining the atoms as
    drawn gave `2.13--2.37`, which Stage 0's `_key_atoms` rejects outright as a
    key with an empty part -- 2 072 of 7 077 labels over the corpora measured.
    A derived grid that cannot be parsed can never be matched to a withheld
    one. A trailing dash on the LAST atom is the page's own text and stays.
    """
    from socr.figures.chart_data import _key_atoms
    from socr.figures.chart_reader import _join_atoms

    assert _join_atoms(["2.13−", "2.37"]) == "2.13-2.37"
    assert _join_atoms(["0.00-", "0.37"]) == "0.00-0.37"
    assert _join_atoms(["1.88", "2.12"]) == "1.88-2.12"
    assert _join_atoms(["B1"]) == "B1"
    assert _join_atoms(["2013-"]) == "2013-"
    for drawn, joined in (
        (["2.13−", "2.37"], "2.13-2.37"),
        (["0.00-", "0.37"], "0.00-0.37"),
    ):
        assert _key_atoms(_join_atoms(drawn)) is not None, drawn
        assert _key_atoms("-".join(drawn)) is None, "the un-stripped form is the defect"
        assert joined == _join_atoms(drawn)


@pytest.mark.skipif(not DOTPLOT_PDF.exists(), reason="corpus fixture not present")
def test_every_reference_bin_label_parses_as_a_stage_0_key() -> None:
    """The property on a real page, not on a hand-written atom list."""
    from socr.figures.chart_data import _key_atoms

    doc = fitz.open(str(DOTPLOT_PDF))
    page = doc[0]
    reading = read_chart_page(page, chart_region_bboxes(page), page_num=20)
    labels = [b.label for panel in reading.panels.values() for b in panel.bins]
    assert labels
    assert [lbl for lbl in labels if _key_atoms(lbl) is None] == []


# ---------------------------------------------------------------------------
# Round 2 -- recorded, not charged: the ladder slack is the ink, not the scale
# ---------------------------------------------------------------------------


def test_a_ladder_disagreement_the_stroke_admits_is_charged_as_residual(
    tmp_path: Path,
) -> None:
    """The admission window is the stroke width and is not scaled to the pitch.

    A 4pt disagreement against a 20pt tick pitch is refused at a thin stroke
    and admitted at a thick one, which is recorded as a residual in
    docs/plans/chart-data/STATUS.md rather than charged: where it is admitted,
    the displaced ladder moves the fitted zero, the panel's residual carries
    it, and the panel gate and `_resolve` both charge that. The difference
    pinned is that admission never turns into a published number the
    calibration does not support.
    """
    thin, thin_refusal = read_panel(
        tmp_path, "lad_thin", bars={"B1": 3}, right_ladder_offset=4.0, tick_width=0.4
    )
    thick, _ = read_panel(
        tmp_path, "lad_thick", bars={"B1": 3}, right_ladder_offset=4.0, tick_width=9.0
    )
    assert thin is None and thin_refusal
    assert thick is not None
    assert thick.calibration.residual < thick.calibration.half_count_points


# ---------------------------------------------------------------------------
# Round 2 addendum -- two plots in one region, with no white wrapper involved
# ---------------------------------------------------------------------------


def _two_charts(path: Path, *, lower_bar: int, separate_grounds: bool):
    """Two complete charts, sharing one visible background or on two.

    Nothing here is invisible: the background is light grey and is real ink.
    The only difference between the two drawings is whether that ink is one
    rectangle spanning both charts (which the detector bridges into one
    region) or one rectangle per chart.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    if separate_grounds:
        page.draw_rect(fitz.Rect(60, 240, 560, 440), color=None, fill=(0.95, 0.95, 0.95))
        page.draw_rect(fitz.Rect(60, 540, 560, 740), color=None, fill=(0.95, 0.95, 0.95))
    else:
        page.draw_rect(fitz.Rect(60, 240, 560, 740), color=None, fill=(0.95, 0.95, 0.95))
    for x, base, title, count in (
        (100, 400, "FIRST PANEL", 5),
        (120, 700, "SECOND PANEL", lower_bar),
    ):
        page.draw_line((x, base), (x + 300, base), width=0.4)
        for n in (2, 4, 6, 8, 10):
            y = base - n * 10
            page.draw_line((x, y), (x + 10, y), width=0.4)
            page.draw_line((x + 290, y), (x + 300, y), width=0.4)
            page.insert_text((x + 305, y + 2), str(n), fontsize=5)
        for i in range(4):
            page.insert_text((x + 42 + i * 70, base + 10), f"B{i + 1}", fontsize=5)
        page.insert_text((x + 20, base - 120), title, fontsize=8)
        page.draw_rect(
            fitz.Rect(x + 20, base - count * 10, x + 75, base),
            color=(0, 0, 0),
            fill=(0.4, 0.6, 0.8),
            width=0.3,
        )
    page.draw_rect(fitz.Rect(210, 598, 218, 602), color=(0, 0, 0), fill=(0.4, 0.6, 0.8), width=0.3)
    page.insert_text((222, 601.5), "June projections", fontsize=5)
    doc.save(str(path))
    doc.close()
    doc = fitz.open(str(path))
    page = doc[0]
    return page, chart_region_bboxes(page)


def _titled_counts(reading) -> dict[str, list[int]]:
    return {
        panel.label or "": [c.count for s in panel.series for c in s.cells if c.count is not None]
        for panel in reading.panels.values()
    }


def test_one_region_over_two_frames_reads_neither(tmp_path: Path) -> None:
    """The same two charts, bridged into one region and drawn on two.

    Split apart, each chart is read under its own title. Bridged, the reader
    refuses the region rather than publishing the lower chart's bar under the
    upper chart's title -- which is what it did before, at a calibration
    residual of exactly 0.0, because a residual tests one ladder against one
    scale and says nothing about which of two plots the ink belongs to.
    """
    apart_page, apart_boxes = _two_charts(
        tmp_path / "apart.pdf", lower_bar=7, separate_grounds=True
    )
    apart = read_chart_page(apart_page, apart_boxes, page_num=1)
    together_page, together_boxes = _two_charts(
        tmp_path / "together.pdf", lower_bar=7, separate_grounds=False
    )
    together = read_chart_page(together_page, together_boxes, page_num=1)

    assert len(apart_boxes) == 2 and len(together_boxes) == 1
    apart_counts = _titled_counts(apart)
    assert apart_counts["FIRST PANEL"][0] == 5
    assert apart_counts["SECOND PANEL"][0] == 7
    assert together.panels == {}
    assert "2 plot frames" in together.refusals[1]


def test_the_lower_frames_bar_never_reaches_the_upper_frames_title(tmp_path: Path) -> None:
    """Changing only the LOWER chart's bar must not change the UPPER panel.

    Pinned as a difference so it cannot pass by the region happening to refuse:
    whatever the bridged region publishes under `FIRST PANEL`, it is the same
    for a lower bar of 7 and a lower bar of 9, because the upper chart is
    identical in both drawings.
    """
    seven_page, seven_boxes = _two_charts(
        tmp_path / "seven.pdf", lower_bar=7, separate_grounds=False
    )
    nine_page, nine_boxes = _two_charts(tmp_path / "nine.pdf", lower_bar=9, separate_grounds=False)
    seven = _titled_counts(read_chart_page(seven_page, seven_boxes, page_num=1))
    nine = _titled_counts(read_chart_page(nine_page, nine_boxes, page_num=1))
    assert seven.get("FIRST PANEL") == nine.get("FIRST PANEL")


def test_one_plot_drawn_with_a_top_rule_is_still_one_frame(tmp_path: Path) -> None:
    """The control: a plot box is not two plots.

    A panel's top rule encloses the same tick ladder as its axis, so a rule
    that counted candidate axes would refuse every boxed chart. Frames are
    counted by the LADDER they read, and both rules read one.
    """
    from socr.figures.chart_reader import find_frames, page_marks

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    base, x = 400, 100
    page.draw_line((x, base), (x + 300, base), width=0.4)
    page.draw_line((x, base - 120), (x + 300, base - 120), width=0.4)
    for n in (2, 4, 6, 8, 10):
        y = base - n * 10
        page.draw_line((x, y), (x + 10, y), width=0.4)
        page.draw_line((x + 290, y), (x + 300, y), width=0.4)
    found = find_frames(page_marks(page))
    assert len(found) == 1
    assert found[0].baseline == base
