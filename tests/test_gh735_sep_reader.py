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
BINS = ("1.0", "2.0", "3.0", "4.0")
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
        tmp_path, "inside", bars={"1.0": 3}, right_ladder_offset=0.15, tick_width=0.4
    )
    outside, refusal_outside = read_panel(
        tmp_path, "outside", bars={"1.0": 3}, right_ladder_offset=3.0, tick_width=0.4
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
        tmp_path, "thin", bars={"1.0": 3}, right_ladder_offset=offset, tick_width=0.4
    )
    thick, _ = read_panel(
        tmp_path, "thick", bars={"1.0": 3}, right_ladder_offset=offset, tick_width=2.0
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
    plain, _ = read_panel(tmp_path, "plain", bars={"1.0": 3, "3.0": 5})
    noted, _ = read_panel(tmp_path, "noted", bars={"1.0": 3, "3.0": 5}, footnote=FOOTNOTE)
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
    labelled, _ = read_panel(tmp_path, "labelled", bars={"1.0": 3}, footnote=FOOTNOTE)
    unlabelled, refusal = read_panel(
        tmp_path, "unlabelled", bars={"1.0": 3}, footnote=FOOTNOTE, draw_bin_labels=False
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
    good, _ = read_panel(tmp_path, "good", bars={"1.0": 3})
    bad, refusal = read_panel(
        tmp_path, "bad", bars={"1.0": 3}, tick_labels=("2", "4", "6", "8", "30")
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
    clear, _ = read_panel(tmp_path, "clear", bars={"1.0": 3, "3.0": 5})
    over, _ = read_panel(tmp_path, "over", bars={"1.0": 3, "3.0": 5}, legend_over_bin_centre=True)
    assert clear is not None and over is not None
    assert [s.name for s in clear.series] == [SOLID]
    assert [s.name for s in over.series] == [SOLID]
    assert series_counts(clear, SOLID) == series_counts(over, SOLID)


def test_the_series_name_stops_at_the_plots_edge(tmp_path: Path) -> None:
    """The y tick labels share the legend's rows and must not join its name."""
    panel, _ = read_panel(tmp_path, "named", bars={"1.0": 3})
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
    separate, _ = read_panel(tmp_path, "sep", bars={"1.0": 3}, dashed_levels=levels)
    compound, _ = read_panel(
        tmp_path, "cmp", bars={"1.0": 3}, dashed_levels=levels, dashed_compound=True
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
    stray_unit_token: bool = False,
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
    if stray_unit_token:
        # The unit annotation set on the LABEL row's own baseline, past the end
        # of the axis: one token outside the span disqualifies the whole row.
        page.insert_text(fitz.Point(X1 + 4.0, BASE + 22.0), "Percent", fontsize=5)
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


BARS = {"1.0": 3, "2.0": 5, "3.0": 4, "4.0": 2}


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


def test_with_nothing_attesting_a_row_the_caption_makes_no_difference(
    tmp_path: Path,
) -> None:
    """The layout question is no longer asked, so its answer cannot matter.

    The bar here straddles a bin boundary, so it covers no label centre and
    attests nothing. Round 3 read this page against the only row inside the
    axis' span and round 4 against the first such row; both were defeated by a
    prose row printed where the rule looked (#735 review). The caption is the
    only difference between the two drawings, and with no corroborated row
    there are no bins either way.
    """
    _bin_w, centres = _bin_geometry(len(BINS))
    straddle = ((centres[0] + centres[1]) / 2.0 - 5.0, (centres[0] + centres[1]) / 2.0 + 5.0, 5)
    alone, alone_refusal = read_captioned(tmp_path, "alone", None, {}, extra_bars=(straddle,))
    contested, contested_refusal = read_captioned(
        tmp_path, "contested", "Percent range", {}, extra_bars=(straddle,)
    )
    assert alone is None
    assert contested is None
    assert alone_refusal == contested_refusal, (alone_refusal, contested_refusal)
    assert alone_refusal and "not corroborated" in alone_refusal


def test_one_token_past_the_axis_end_cannot_hand_the_bins_to_a_caption(
    tmp_path: Path,
) -> None:
    """The reviewer's route to the round-1 defect, through the lone-candidate rule.

    A unit annotation set on the label row's OWN baseline but past the end of
    the axis disqualifies that row by the span test, leaving the caption below
    it as the only candidate. The only difference between the two drawings is
    that one token. Neither may publish a count under a word of prose.
    """
    intact, _ = read_captioned(tmp_path, "intact", "Effective federal funds rate", BARS)
    assert intact is not None
    assert [b.label for b in intact.bins] == list(BINS)

    lone, lone_refusal = read_captioned(
        tmp_path, "lone", "Effective federal funds rate", BARS, stray_unit_token=True
    )
    published = (
        [] if lone is None else [c for s in lone.series for c in s.cells if c.count is not None]
    )
    assert not published, [b.label for b in lone.bins]
    assert lone is None and lone_refusal and "not corroborated" in lone_refusal


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
        assert _key_atoms("-".join(drawn)) != _key_atoms(_join_atoms(drawn)), (
            "the un-stripped form is the defect: it parses, to a different key"
        )
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
        tmp_path, "lad_thin", bars={"1.0": 3}, right_ladder_offset=4.0, tick_width=0.4
    )
    thick, _ = read_panel(
        tmp_path, "lad_thick", bars={"1.0": 3}, right_ladder_offset=4.0, tick_width=9.0
    )
    assert thin is None and thin_refusal
    assert thick is not None
    assert thick.calibration.residual < thick.calibration.half_count_points


# ---------------------------------------------------------------------------
# Round 2 addendum -- two plots in one region, with no white wrapper involved
# ---------------------------------------------------------------------------


def _two_charts(path: Path, *, lower_bar: int, separate_grounds: bool, lower_dx: float = 0.0):
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
        (120 + lower_dx, 700, "SECOND PANEL", lower_bar),
    ):
        page.draw_line((x, base), (x + 300, base), width=0.4)
        for n in (2, 4, 6, 8, 10):
            y = base - n * 10
            page.draw_line((x, y), (x + 10, y), width=0.4)
            page.draw_line((x + 290, y), (x + 300, y), width=0.4)
            page.insert_text((x + 305, y + 2), str(n), fontsize=5)
        for i in range(4):
            page.insert_text((x + 42 + i * 70, base + 10), f"{i + 1}.0", fontsize=5)
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


def test_the_two_frame_refusal_needs_the_tick_COLUMNS_to_differ(tmp_path: Path) -> None:
    """The documented limit of the frame count: it is keyed on ladder identity.

    Two stacked plots drawn in the SAME column put their tick marks at one x
    span, so both ladders land in one span group, every candidate axis reads
    that one merged ladder, and the region holds one frame by this count. The
    refusal therefore does not fire; what catches the drawing instead is the
    residual gate, since a ladder of doubled length cannot fit one scale. Both
    outcomes are refusals, which is why this is recorded as a limit rather
    than charged -- but the two are reached by different routes, and STATUS
    item 12 says so. Found by the #735 reviewer (`test_rev735f.py`).
    """
    offset_page, offset_boxes = _two_charts(
        tmp_path / "offset.pdf", lower_bar=7, separate_grounds=False
    )
    offset = read_chart_page(offset_page, offset_boxes, page_num=1)
    same_page, same_boxes = _two_charts(
        tmp_path / "same.pdf", lower_bar=7, separate_grounds=False, lower_dx=-20.0
    )
    same = read_chart_page(same_page, same_boxes, page_num=1)

    assert offset.panels == {} and same.panels == {}
    assert "2 plot frames" in offset.refusals[1]
    assert "2 plot frames" not in same.refusals[1]
    assert "closes only to" in same.refusals[1]


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


# ---------------------------------------------------------------------------
# Round 5 -- with no bar on the axis there are no bins, and no fallback
# ---------------------------------------------------------------------------


def draw_dashed(
    path: Path,
    *,
    caption: str | None = None,
    caption_first: bool = False,
    labels_out_of_span: bool = False,
    stray_unit_token: bool = False,
    solid_bars: bool = False,
) -> tuple[fitz.Document, list]:
    """A staircase drawn as dashed STROKES, so no bar rests on the axis.

    ``_resting_bars`` is empty here, which is the page shape the deleted
    fallback used to decide by layout. The knobs are the four ingredients the
    #735 review used against it, in every combination it found: a prose caption
    above or below the label row, the label row pushed a few points outside the
    axis' span, and one unit word printed on the label row's own baseline past
    the end of the axis. Pass ``solid_bars`` to draw the same counts as filled
    bars instead, which is the only thing that puts a bar on the axis.
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
    label_y = BASE + 26.0 if caption_first else BASE + 10.0
    shift = -(centres[0] - X0) - 8.0 if labels_out_of_span else 0.0
    if caption is not None:
        page.insert_text(
            fitz.Point(X0 + 40.0, BASE + 10.0 if caption_first else BASE + 26.0),
            caption,
            fontsize=5,
        )
    for centre, label in zip(centres, labels, strict=True):
        page.insert_text(fitz.Point(centre + shift - len(label) * 1.4, label_y), label, fontsize=5)
    if stray_unit_token:
        page.insert_text(fitz.Point(X1 + 4.0, label_y), "Percent", fontsize=5)
    style: dict = {} if solid_bars else {"dashes": "[2 2] 0"}
    for i, label in enumerate(labels):
        count = BARS[label]
        y = BASE - count * UNIT
        if solid_bars:
            page.draw_rect(
                fitz.Rect(centres[i] - bin_w * 0.4, y, centres[i] + bin_w * 0.4, BASE),
                color=(0, 0, 0),
                fill=(0.4, 0.6, 0.8),
                width=0.3,
            )
        else:
            page.draw_line(
                fitz.Point(centres[i] - bin_w * 0.4, y),
                fitz.Point(centres[i] + bin_w * 0.4, y),
                width=1.0,
                **style,
            )
    if solid_bars:
        swatch_cx = (centres[0] + centres[1]) / 2.0
        page.draw_rect(
            fitz.Rect(swatch_cx - 4.0, BASE - 102.0, swatch_cx + 4.0, BASE - 98.0),
            color=(0, 0, 0),
            fill=(0.4, 0.6, 0.8),
            width=0.3,
        )
        page.insert_text(fitz.Point(swatch_cx + 8.0, BASE - 98.5), SOLID, fontsize=5)
    else:
        page.draw_line(
            fitz.Point(X0 + 20.0, BASE - 130.0),
            fitz.Point(X0 + 36.0, BASE - 130.0),
            width=1.0,
            **style,
        )
        page.insert_text(fitz.Point(X0 + 40.0, BASE - 128.5), DASHED, fontsize=5)
    doc.save(str(path))
    reopened = fitz.open(str(path))
    return reopened, [reopened[0].rect]


def read_dashed(tmp_path: Path, name: str, **kw):
    doc, bboxes = draw_dashed(tmp_path / f"{name}.pdf", **kw)
    reading = read_chart_page(doc[0], bboxes, page_num=1)
    return reading.panels.get(1), reading.refusals.get(1)


def test_the_bars_are_the_only_thing_that_can_say_which_row_carries_the_bins(
    tmp_path: Path,
) -> None:
    """The whole rule, as one difference: dashed strokes against filled bars.

    Same counts, same labels, same caption, same page. Drawn as filled bars,
    each covers exactly one printed label centre, the labels are corroborated
    and the panel reads what it was drawn from. Drawn as dashed levels, nothing
    stands on the axis, no row is corroborated by anything, and the panel is
    refused. There is no layout fallback: four of them were tried and each
    published a prose row as the bins (#735 review rounds 1-4).
    """
    dashed, dashed_refusal = read_dashed(tmp_path, "dashed", caption="Effective funds rate here")
    assert dashed is None
    assert dashed_refusal and "not corroborated" in dashed_refusal

    solid, solid_refusal = read_dashed(
        tmp_path, "solid", caption="Effective funds rate here", solid_bars=True
    )
    assert solid is not None, solid_refusal
    assert [b.label for b in solid.bins] == list(BINS)
    counts = {c.bin_label: c.count for s in solid.series for c in s.cells}
    assert counts == {b: BARS[b] for b in BINS}, counts


def test_no_arrangement_of_prose_under_a_dashed_axis_can_publish_a_count(
    tmp_path: Path,
) -> None:
    """Every page the four layout rules were defeated by, pinned together.

    Each of these satisfied one of the rules the review tried, and each
    published a fabricated label with a fabricated count at a perfect residual:
    the caption alone (uniqueness), the caption first (nearest), and the two
    together with one unit word past the axis end, which satisfied both halves
    of the conjunction at once. None of them may publish anything.
    """
    shapes = {
        "bare": {},
        "caption_below": {"caption": "Effective funds rate here"},
        "caption_first": {"caption": "Effective funds rate here", "caption_first": True},
        "stray_token": {"caption": "Effective funds rate here", "stray_unit_token": True},
        "first_and_alone": {
            "caption": "Effective funds rate here",
            "caption_first": True,
            "labels_out_of_span": True,
        },
        "corpus_shape": {
            "caption": "Percent range",
            "caption_first": True,
            "stray_unit_token": True,
        },
    }
    for name, kw in shapes.items():
        panel, refusal = read_dashed(tmp_path, name, **kw)
        assert panel is None, (name, [b.label for b in panel.bins])
        assert refusal and "not corroborated" in refusal, (name, refusal)


def test_a_bar_that_covers_no_label_centre_attests_nothing(tmp_path: Path) -> None:
    """A straddling bar is on the axis but corroborates no row.

    The bars are asked whether each covers exactly ONE token of a row, which is
    what a histogram bar does to its own label and what a caption is never
    subject to. A bar straddling a bin boundary covers two label centres or
    none, so it answers nothing, and the panel has no more evidence than a
    dashed one. The bar's width is the only difference between the drawings.
    """
    _bin_w, centres = _bin_geometry(len(BINS))
    mid = (centres[0] + centres[1]) / 2.0
    straddling, refusal = read_captioned(
        tmp_path, "straddling", None, {}, extra_bars=((mid - 5.0, mid + 5.0, 5),)
    )
    assert straddling is None
    assert refusal and "not corroborated" in refusal

    owning, owning_refusal = read_captioned(
        tmp_path, "owning", None, {}, extra_bars=((centres[0] - 5.0, centres[0] + 5.0, 5),)
    )
    assert owning is not None, owning_refusal
    assert [b.label for b in owning.bins] == list(BINS)


# ---------------------------------------------------------------------------
# Round 6: a bar standing somewhere does not make the row under it the labels
# ---------------------------------------------------------------------------


def _narrow_bins(path: Path, *, bars_on_labels: bool) -> tuple[fitz.Document, list]:
    """Four narrow bins mid-axis, a four-word caption spaced more widely below.

    The ONLY knob is where the bars stand. Every other mark is identical, the
    bin labels are ordinary text inside the axis' span in both, and the caption
    is drawn in both. Standing on the label centres, the bars attest the labels;
    standing on the caption's words, they attest the caption -- which is the
    whole of the round-5 rule's remaining hole.
    """
    centres = [160.0, 180.0, 200.0, 220.0]
    caption_centres = [130.0, 190.0, 250.0, 310.0]
    caption = ["Effective", "federal", "funds", "rate"]
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_line(fitz.Point(X0, BASE), fitz.Point(X1, BASE), width=0.4)
    for value in TICK_VALUES:
        y = BASE - value * UNIT
        page.draw_line(fitz.Point(X0, y), fitz.Point(X0 + 10.0, y), width=0.4)
        page.draw_line(fitz.Point(X1 - 10.0, y), fitz.Point(X1, y), width=0.4)
        page.insert_text(fitz.Point(X1 + 5.0, y + 2.0), str(value), fontsize=5)
    for centre, label in zip(centres, BINS, strict=True):
        page.insert_text(fitz.Point(centre - len(label) * 1.4, BASE + 10.0), label, fontsize=5)
    for centre, word in zip(caption_centres, caption, strict=True):
        page.insert_text(fitz.Point(centre - len(word) * 1.2, BASE + 26.0), word, fontsize=5)
    swatch_cx = 300.0
    page.draw_rect(
        fitz.Rect(swatch_cx - 4.0, BASE - 102.0, swatch_cx + 4.0, BASE - 98.0),
        color=(0, 0, 0),
        fill=(0.4, 0.6, 0.8),
        width=0.3,
    )
    page.insert_text(fitz.Point(swatch_cx + 8.0, BASE - 98.5), SOLID, fontsize=5)
    stand = centres if bars_on_labels else caption_centres
    for centre, count in zip(stand, (3, 5, 4, 2), strict=True):
        page.draw_rect(
            fitz.Rect(centre - 4.0, BASE - count * UNIT, centre + 4.0, BASE),
            color=(0, 0, 0),
            fill=(0.4, 0.6, 0.8),
            width=0.3,
        )
    doc.save(str(path))
    reopened = fitz.open(str(path))
    return reopened, [reopened[0].rect]


def _published(panel) -> list[tuple[str, int | None]]:
    if panel is None:
        return []
    return [(c.bin_label, c.count) for s in panel.series for c in s.cells if c.status == INTEGER]


def test_bars_stood_on_a_caption_publish_nothing_under_its_words(tmp_path: Path) -> None:
    """The bars moved, and nothing else. Only one of the two may be read.

    Standing on the printed label centres they attest the labels and the panel
    reads what it was drawn from. Standing on the caption's words they attest
    the caption -- and the printed label row is then sitting below the axis,
    inside the span, with four columns the drawing says nothing about. Two rows
    are equally good homes for the same four marks, so the panel is refused
    rather than published under a row of prose (#735 round 6).
    """
    doc, bboxes = _narrow_bins(tmp_path / "labels.pdf", bars_on_labels=True)
    on_labels = read_chart_page(doc[0], bboxes, page_num=1)
    panel = on_labels.panels.get(1)
    assert panel is not None, on_labels.refusals
    assert [b.label for b in panel.bins] == list(BINS)
    assert _published(panel) == list(zip(BINS, (3, 5, 4, 2), strict=True))

    doc, bboxes = _narrow_bins(tmp_path / "caption.pdf", bars_on_labels=False)
    moved = read_chart_page(doc[0], bboxes, page_num=1)
    assert moved.panels.get(1) is None, [b.label for b in moved.panels[1].bins]
    assert "not corroborated" in moved.refusals.get(1, ""), moved.refusals


def test_a_tie_on_one_bar_is_not_broken_by_the_row_being_higher(tmp_path: Path) -> None:
    """A sparse chart, with and without the caption: the caption changes nothing.

    One bar over ``B2`` covers exactly one printed label centre AND exactly one
    caption word, because the caption's two words sit under the right-hand end
    of its sweep. Scored on coverage alone both rows tie at 1 and the upper --
    the caption -- took the bins, publishing ``Percent-B2 | range-B3``. The bar
    is 56pt wide and the interval the caption's own two words derive is 16pt, so
    it is not a bar of that row at all, and the labels win on the drawing.
    """
    plain, plain_refusal = read_captioned(tmp_path, "plain", None, {"2.0": 5})
    capt, capt_refusal = read_captioned(tmp_path, "capt", "Percent range", {"2.0": 5})
    assert plain is not None, plain_refusal
    assert capt is not None, capt_refusal
    assert [b.label for b in plain.bins] == list(BINS)
    assert [b.label for b in capt.bins] == [b.label for b in plain.bins]
    assert _published(capt) == _published(plain)
    assert ("2.0", 5) in _published(capt), _published(capt)


def test_prose_over_a_stray_cannot_turn_it_into_a_bin(tmp_path: Path) -> None:
    """A rectangle on the axis that no printed bin can claim, and a caption over it.

    With no caption the panel is refused: the stray covers no label centre, so
    nothing attests a row. Adding the caption used to rescue it -- the stray
    covers one caption word, so the caption scored 1 while the complete, printed
    label row scored 0 -- which made the mark no bin could claim into evidence
    for the prose that happened to cover it. The four printed labels are still
    drawn below the axis with nothing said about them, so the caption cannot
    take the bins from them, and the caption must change nothing.
    """
    bare, bare_refusal = read_captioned(tmp_path, "bare", None, {}, extra_bars=((235.0, 242.0, 5),))
    capt, capt_refusal = read_captioned(
        tmp_path, "capt_stray", "Percent range", {}, extra_bars=((235.0, 242.0, 5),)
    )
    assert bare is None, [b.label for b in bare.bins]
    assert capt is None, [b.label for b in capt.bins]
    assert bare_refusal and "not corroborated" in bare_refusal
    assert capt_refusal == bare_refusal


# ---------------------------------------------------------------------------
# Round 7: the bins are a row of numbers, and nothing else is asked
# ---------------------------------------------------------------------------

_R7_CENTRES = [160.0, 180.0, 200.0, 220.0]
_R7_COUNTS = (3, 5, 4, 2)


def _numeric_panel(
    path: Path,
    *,
    labels: tuple[str, ...] = BINS,
    label_y: float = BASE + 10.0,
    extra_row: tuple[str, ...] | None = None,
    extra_centres: list[float] | None = None,
    extra_y: float = BASE + 26.0,
    bars_on_extra: bool = False,
) -> tuple[fitz.Document, list]:
    """One panel, its bin labels, and optionally one more row of text.

    The knobs are only what the round-7 rule turns on: what the label row says,
    what the extra row says, where it is drawn, and which of the two the bars
    stand on.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_line(fitz.Point(X0, BASE), fitz.Point(X1, BASE), width=0.4)
    for value in TICK_VALUES:
        y = BASE - value * UNIT
        page.draw_line(fitz.Point(X0, y), fitz.Point(X0 + 10.0, y), width=0.4)
        page.draw_line(fitz.Point(X1 - 10.0, y), fitz.Point(X1, y), width=0.4)
        page.insert_text(fitz.Point(X1 + 5.0, y + 2.0), str(value), fontsize=5)
    for centre, label in zip(_R7_CENTRES, labels, strict=True):
        page.insert_text(fitz.Point(centre - len(label) * 1.4, label_y), label, fontsize=5)
    centres = extra_centres or [130.0, 190.0, 250.0, 310.0]
    if extra_row is not None:
        for centre, word in zip(centres, extra_row, strict=True):
            page.insert_text(fitz.Point(centre - len(word) * 1.2, extra_y), word, fontsize=5)
    page.draw_rect(
        fitz.Rect(296.0, BASE - 102.0, 304.0, BASE - 98.0),
        color=(0, 0, 0),
        fill=(0.4, 0.6, 0.8),
        width=0.3,
    )
    page.insert_text(fitz.Point(308.0, BASE - 98.5), SOLID, fontsize=5)
    for centre, count in zip(centres if bars_on_extra else _R7_CENTRES, _R7_COUNTS, strict=True):
        page.draw_rect(
            fitz.Rect(centre - 4.0, BASE - count * UNIT, centre + 4.0, BASE),
            color=(0, 0, 0),
            fill=(0.4, 0.6, 0.8),
            width=0.3,
        )
    doc.save(str(path))
    reopened = fitz.open(str(path))
    return reopened, [reopened[0].rect]


def _read_numeric(path: Path, **kw):
    doc, bboxes = _numeric_panel(path, **kw)
    reading = read_chart_page(doc[0], bboxes, page_num=1)
    panel = reading.panels.get(1)
    if panel is None:
        return None, reading.refusals.get(1)
    return (
        [b.label for b in panel.bins],
        [(c.bin_label, c.count) for s in panel.series for c in s.cells if c.status == INTEGER],
    )


def test_a_footnote_below_a_read_chart_no_longer_refuses_it(tmp_path: Path) -> None:
    """The round-6 rival rule refused a correct chart because of a footnote.

    The bars attest the printed numeric labels and nothing is in doubt. Round 6
    then asked whether any silent row below the axis was an equally good home
    for the marks, and a five-word footnote — drawn over nothing, covered by
    nothing, saying nothing about the bins — answered yes and refused the whole
    panel. That rule is deleted, and the footnote must now change nothing.
    """
    plain_bins, plain_cells = _read_numeric(tmp_path / "plain.pdf")
    footed_bins, footed_cells = _read_numeric(
        tmp_path / "footed.pdf",
        extra_row=("Note", "excludes", "one", "absent", "participant"),
        extra_centres=[120.0, 170.0, 230.0, 290.0, 350.0],
        extra_y=BASE + 30.0,
    )
    assert plain_bins == list(BINS), plain_cells
    assert footed_bins == plain_bins
    assert footed_cells == plain_cells
    assert plain_cells == list(zip(BINS, _R7_COUNTS, strict=True))


def test_a_row_of_words_cannot_be_the_bins_wherever_the_bars_stand(tmp_path: Path) -> None:
    """The whole of round 7, as one difference: what the attested row SAYS.

    Standing on the printed numbers the bars attest them and the panel reads.
    Moved onto the caption's words — the only change — the row they attest
    carries words, so it cannot be a row of bins and the panel is refused
    instead of publishing counts under `Effective | federal | funds | rate`.
    """
    caption = ("Effective", "federal", "funds", "rate")
    on_labels = _read_numeric(tmp_path / "labels.pdf", extra_row=caption)
    on_caption = _read_numeric(tmp_path / "caption.pdf", extra_row=caption, bars_on_extra=True)
    assert on_labels[0] == list(BINS), on_labels
    assert on_caption[0] is None, on_caption
    assert "not corroborated" in on_caption[1]


def test_a_chart_whose_bins_are_words_is_out_of_scope_and_refuses(tmp_path: Path) -> None:
    """The measured cost of the gate, pinned as a difference.

    The same drawing twice, the bars on the label row in both. Numeric labels
    read; the same bins labelled by category refuse, because the owner's ruling
    is best-effort extraction of NUMERIC charts and a categorical histogram is
    outside it. No corpus page is affected: all 7 142 published bin labels of
    both corpora and the reference parse as numbers.
    """
    numeric = _read_numeric(tmp_path / "numeric.pdf")
    words = _read_numeric(tmp_path / "words.pdf", labels=("Asia", "Europe", "Africa", "Oceania"))
    assert numeric[0] == list(BINS), numeric
    assert words[0] is None, words
    assert "not corroborated" in words[1]


def test_a_numeric_annotation_row_still_takes_the_bins(tmp_path: Path) -> None:
    """OPEN FABRICATION ROUTE — this pins what the reader DOES, not what it should.

    The numeric gate asks whether a row is label-SHAPED. It cannot ask whether
    the row IS the labels, and a numeric annotation printed above the bin labels
    satisfies everything the reader can check: every token parses, every bar lies
    wholly inside its annotation-derived bin, and no bar covers a real label
    centre. The annotation therefore scores 4 to 0 and the counts are published
    under `0 | 5 | 10 | 15` while the page's own `1.0 2.0 3.0 4.0` is ignored.

    No layout test separates the two rows: the fabricating row is the topmost
    in-span numeric row, which is exactly the arrangement all 840 corpus calls
    show for genuine labels. Round 7 is the last change to the selection logic
    (STATUS item 18), so this is pinned rather than closed — if a later change
    alters it, this test says so instead of letting it drift.
    """
    bins, cells = _read_numeric(
        tmp_path / "annotation.pdf",
        extra_row=("0", "5", "10", "15"),
        extra_y=BASE + 10.0,
        label_y=BASE + 26.0,
        bars_on_extra=True,
    )
    assert bins == ["0", "5", "10", "15"], bins
    assert cells == [("0", 3), ("5", 5), ("10", 4), ("15", 2)], cells


def test_a_two_line_range_label_with_a_trailing_dash_is_still_the_bins(tmp_path: Path) -> None:
    """The form both corpora actually draw, pinned against the numeric gate.

    A Fed dot-plot prints its bins over two lines, ``0.13-`` above ``0.37``, so
    the row the gate judges carries tokens ending in a range dash. Handed to the
    key grammar unstripped, such a token splits into ``0.13`` and an empty part
    and is malformed -- and the first version of the numeric gate therefore
    rejected the real label row of every SEP page, fell through to the second
    line, and published ``0.37`` where the page says ``0.13-0.37``. Only the
    corpus dumps caught that. This pins it without them: the trailing-dash row
    must be the bins, and its label must be the two lines joined.
    """
    ranged_bins, ranged_cells = _read_numeric(
        tmp_path / "ranged.pdf",
        labels=("0.13-", "0.38-", "0.63-", "0.88-"),
        extra_row=("0.37", "0.62", "0.87", "1.12"),
        extra_centres=_R7_CENTRES,
        extra_y=BASE + 20.0,
    )
    assert ranged_bins == ["0.13-0.37", "0.38-0.62", "0.63-0.87", "0.88-1.12"], ranged_cells
    assert [count for _label, count in ranged_cells] == list(_R7_COUNTS), ranged_cells

    # The same drawing with single-line labels reads the same counts, so what
    # the pin above tests is the LABEL, not the geometry.
    plain_bins, plain_cells = _read_numeric(tmp_path / "plain.pdf")
    assert plain_bins == list(BINS)
    assert [count for _label, count in plain_cells] == [count for _label, count in ranged_cells]


def test_a_row_of_infinities_is_not_a_row_of_bins() -> None:
    """``float`` parses more words than a bin label can be.

    ``nan``, ``inf``, ``Infinity`` and their signed forms all parse, so a row of
    them passed the numeric gate on the strength of the parser alone. A bin is a
    position on a printed axis, so a label that is not a FINITE number is not a
    bin. Pinned as a difference against the finite row of the same shape.
    """
    from socr.figures.chart_reader import WordRow, _numeric_row

    def row(*words: str) -> WordRow:
        tokens = tuple((100.0 + 20.0 * i, 8.0, w) for i, w in enumerate(words))
        return WordRow(y0=410.0, y1=416.0, x0=95.0, x1=200.0, text=" ".join(words), tokens=tokens)

    assert _numeric_row(row("1.0", "2.0", "3.0"))
    assert _numeric_row(row("-0.5", "0.5"))
    for words in (("nan", "nan"), ("inf", "inf"), ("Infinity", "1.0"), ("-inf", "2.0")):
        assert not _numeric_row(row(*words)), words


def test_the_measurement_tool_is_runnable_as_a_module() -> None:
    """``python -m socr.figures.measure_chart_bins`` must not exit 0 in silence.

    Without a ``__main__`` guard the module imports, defines ``main``, never
    calls it, and exits 0 having printed nothing — which reads exactly like a
    corpus with no chart pages. The log cites this tool as the reason its 840-call
    table can be re-derived from the tree, so it has to actually run. A silent
    success is how the first numeric gate got through (#735 round 7 review).
    """
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-m", "socr.figures.measure_chart_bins", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "corpus" in done.stdout, done.stdout


def test_words_outside_the_axis_are_not_joined_into_the_bin_labels(tmp_path: Path) -> None:
    """A second line owns only what is drawn inside the bins, and nothing else.

    Nearest-centre assignment has to put every token somewhere, so the outer
    columns extend without limit unless bounded: two words set on the second
    line's baseline but OUTSIDE the axis, one at each end, were joined into the
    outer bins as ``0.13-Additional 0.37`` and ``0.88-1.12 footnote``. That
    corrupts two already-correct numeric labels with page prose, and both keys
    are rejected by Stage 0's own grammar (#735 round 7 review).

    The difference is only those two words. Every bar, every endpoint and the
    attested label row are identical in both drawings, so the labels must be too.
    """
    from socr.figures.chart_data import _key_atoms

    ranged = dict(
        labels=("0.13-", "0.38-", "0.63-", "0.88-"),
        extra_centres=list(_R7_CENTRES),
        extra_y=BASE + 20.0,
    )
    clean_bins, clean_cells = _read_numeric(
        tmp_path / "clean.pdf", extra_row=("0.37", "0.62", "0.87", "1.12"), **ranged
    )
    ranged["extra_centres"] = [40.0, *_R7_CENTRES, 460.0]
    fringed_bins, fringed_cells = _read_numeric(
        tmp_path / "fringed.pdf",
        extra_row=("Additional", "0.37", "0.62", "0.87", "1.12", "footnote"),
        **ranged,
    )
    assert clean_bins == ["0.13-0.37", "0.38-0.62", "0.63-0.87", "0.88-1.12"], clean_bins
    assert fringed_bins == clean_bins, fringed_bins
    assert fringed_cells == clean_cells
    # Every published key still parses as a Stage 0 column key.
    assert all(_key_atoms(label) is not None for label in fringed_bins), fringed_bins


def test_a_five_word_prose_row_below_the_labels_joins_none_of_itself(tmp_path: Path) -> None:
    """Prose under the chart is not a second line of its labels.

    Five words set below four numeric columns published
    ``1.0-Note excludes | 2.0-one | 3.0-absent | 4.0-participant``, whose first
    key Stage 0's grammar rejects outright (#735 round 7 review). A token joins
    a column only if it is drawn inside the axis and inside that column's own
    interval, so the words that fall outside are dropped and the row cannot
    account for every column.
    """
    plain_bins, plain_cells = _read_numeric(tmp_path / "plain.pdf")
    prose_bins, prose_cells = _read_numeric(
        tmp_path / "prose.pdf",
        extra_row=("Note", "excludes", "one", "absent", "participant"),
        extra_centres=[150.0, 163.0, 180.0, 200.0, 220.0],
    )
    assert plain_bins == list(BINS)
    assert prose_bins == plain_bins, prose_bins
    assert prose_cells == plain_cells


def test_every_published_bin_label_parses_as_a_stage_0_key(tmp_path: Path) -> None:
    """The branch's own invariant, on synthetic pages rather than on the corpus.

    Two defects reached a commit on this branch that this one property would
    have caught, and in both cases the only thing that noticed was a corpus
    dump: the numeric gate's first version relabelled every SEP page from
    ``0.13-0.37`` to ``0.37``, and the partition rule joined out-of-axis prose
    into genuine ranges as ``0.13-Additional 0.37``, whose key is malformed.
    Neither needs a corpus to detect -- every label this reader publishes is
    supposed to parse through the grammar Stage 0 uses for a column key, and
    that is asserted here directly, hermetically, over the shapes the reviews
    produced.

    Note what this does NOT assert: that the label is the RIGHT one. A prose
    caption drawn at the bin centres is joined in as ``1.0-Effective``, which
    parses perfectly well and is the wrong header (STATUS item 15). Parsing is
    a floor, not a proof.
    """
    from socr.figures.chart_data import _key_atoms

    ranged = dict(labels=("0.13-", "0.38-", "0.63-", "0.88-"), extra_y=BASE + 20.0)
    drawings = {
        "plain": {},
        "two_line_range": dict(
            extra_row=("0.37", "0.62", "0.87", "1.12"), extra_centres=list(_R7_CENTRES), **ranged
        ),
        "off_axis_prose": dict(
            extra_row=("Additional", "0.37", "0.62", "0.87", "1.12", "footnote"),
            extra_centres=[40.0, *_R7_CENTRES, 460.0],
            **ranged,
        ),
        "prose_row": dict(
            extra_row=("Note", "excludes", "one", "absent", "participant"),
            extra_centres=[150.0, 163.0, 180.0, 200.0, 220.0],
        ),
        "caption_at_centres": dict(
            extra_row=("Effective", "federal", "funds", "rate"),
            extra_centres=list(_R7_CENTRES),
            extra_y=BASE + 20.0,
        ),
        "numeric_annotation": dict(
            extra_row=("0", "5", "10", "15"),
            extra_y=BASE + 10.0,
            label_y=BASE + 26.0,
            bars_on_extra=True,
        ),
    }
    published = 0
    for name, options in drawings.items():
        bins, _cells = _read_numeric(tmp_path / f"{name}.pdf", **options)
        if bins is None:
            continue
        published += 1
        malformed = [label for label in bins if _key_atoms(label) is None]
        assert not malformed, (name, bins, malformed)
    assert published == len(drawings), published
