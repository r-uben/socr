"""#635 Stage 1: read a vector bar chart's counts off the page's own drawings.

Stage 0 (``figures.chart_data``) proved that the empty markdown grid a model
emits for a chart panel is a derivation of that panel, and withheld it. What
shipped in its place said, truthfully, that the counts were not extracted.
This module extracts them -- from the PDF's vector drawing operators, never
from a raster and never from a model.

The reading is geometric end to end:

* the **panel** is one of ``chart_region_bboxes``'s regions, named by the
  unique interior label Stage 0 already isolates;
* the **y scale** is fitted to the tick MARKS the page stroked, each paired
  with the numeric word drawn beside it, and validated against every other
  tick and against the axis line;
* the **bins** are the printed two-line tick labels, and their intervals are
  the midpoints between consecutive label centres -- no uniform-spacing
  assumption, no tolerance to choose;
* a **series** is a drawing STYLE (a filled rectangle, or a dashed stroked
  path) bound to a name by the figure's own legend -- the swatch geometry
  beside the legend text. Never a colour name;
* a **count** is an interval, not a number: the measured height, widened by
  the calibration's own residual and by half the mark's stroke width, divided
  by the fitted points-per-participant. An integer is emitted only when the
  interval admits exactly one non-negative integer. Otherwise the cell is
  ``UNRESOLVED``, and it stays ``UNRESOLVED`` -- no residual is ever allocated
  to make a column sum.

Scope for this round: **vector** charts only. A scanned or rasterised chart
has no drawing operators, ``read_chart_page`` finds no frame, and the page
keeps Stage 0's "counts not extracted" note. A raster fallback (pixel column
profiling) is deliberately NOT built here.

Two things this module refuses, because they are how a chart reader comes to
publish a confident wrong number:

* a fixed pixel, colour-distance or model-confidence cutoff. Every tolerance
  below is derived from the drawing being measured -- the stroke width the
  page chose, the tick spacing it printed, the residual the fit actually left;
* unconditional rounding. ``3.6`` participants is not four participants; it is
  a calibration that did not close, and the cell says so.

A zero is a reading, not a default. A bin is zero only when the series is
PRESENT in the panel, the bin lies inside the drawn plot, and the panel's
geometry positively shows nothing there -- for a bar series, a histogram whose
bars rest on the axis so a zero bar is invisible by construction; for a
staircase, an outline the page draws descending to the axis where it meets the
bin from a neighbouring level. Where a neighbouring level simply stops instead,
the bin is ``UNRESOLVED``: an outline that ended and an outline that fell to
zero are the same picture. A series with no marks at all is ``unresolved`` presence, never a
column of zeros: the 2021 panel of the corpus fixture draws no June outline,
and "the series is absent" and "every June bin is zero" are not distinguishable
from that panel alone.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Bumped whenever a change could move a published count. Persisted per cell.
READER_VERSION = "635-stage1/1"

#: Audit event kinds.
CHART_DERIVATION = "chart_counts_derived"
CHART_DERIVATION_REFUSED = "chart_counts_not_derived"

#: Cell statuses.
INTEGER = "integer"
UNRESOLVED = "unresolved"
#: The literal the published table carries for an unresolved cell.
UNRESOLVED_MARKER = "UNRESOLVED"

#: Series presence in a panel.
PRESENT = "present"
ABSENT = "absent"
PRESENCE_UNRESOLVED = "unresolved"

#: Drawing styles a series can be drawn in.
SOLID_FILL = "solid_fill"
DASHED_STROKE = "dashed_stroke"

#: Acceptance-hook verdicts.
ACCEPT = "accept"
REJECT = "reject"
NO_OPINION = "no_opinion"

#: Verification outcome of one panel's derivation.
VERIFIED = "verified"
UNVERIFIED = "unverified"
REJECTED = "rejected"

_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")
#: A key's atoms are split on the range dash, exactly as Stage 0 splits a
#: column key, so a derived bin label and a withheld one are the same string.
_ATOM_JOIN = "-"


# ---------------------------------------------------------------------------
# Marks: the page's drawing operators, normalised
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mark:
    """One drawing the page painted, reduced to what a chart reader needs.

    ``width`` is the stroke width the page chose. It is this module's ONLY
    source of edge-localisation uncertainty: a stroked path's coordinates are
    its centreline, and the painted edge lies half a stroke width either side
    of it. Nothing here is a tuned constant -- a page that strokes thinner
    measures more precisely, and says so.
    """

    filled: bool
    dashed: bool
    width: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def horizontal(self) -> bool:
        return abs(self.y1 - self.y0) <= self.tolerance and self.x1 > self.x0

    @property
    def vertical(self) -> bool:
        return abs(self.x1 - self.x0) <= self.tolerance and self.y1 > self.y0

    @property
    def tolerance(self) -> float:
        """Half the stroke width: the page's own coordinate resolution here."""
        return max(self.width, 0.0) / 2.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


def _dashed(pattern) -> bool:
    """A dash pattern with at least one non-zero entry means a dashed stroke.

    PyMuPDF reports the PDF dash array verbatim (``"[] 0"`` for solid). The
    dash LENGTHS are never compared between two marks: the corpus fixture
    draws its legend swatch with a different pattern from the staircase it
    stands for, and requiring equality there would unbind the legend.
    """
    text = str(pattern or "")
    inner = text[text.find("[") + 1 : text.find("]")] if "[" in text and "]" in text else ""
    for token in inner.replace(",", " ").split():
        try:
            if float(token) > 0:
                return True
        except ValueError:
            return False
    return False


def page_marks(page) -> list[Mark]:
    """Every drawing on *page*, as :class:`Mark`. Never raises."""
    try:
        drawings = page.get_drawings() or []
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("chart_reader: get_drawings failed: %s", exc)
        return []
    out: list[Mark] = []
    for d in drawings:
        try:
            rect = d["rect"]
            out.append(
                Mark(
                    filled=d.get("fill") is not None,
                    dashed=_dashed(d.get("dashes")),
                    width=float(d.get("width") or 0.0),
                    x0=float(rect.x0),
                    y0=float(rect.y0),
                    x1=float(rect.x1),
                    y1=float(rect.y1),
                )
            )
        except Exception:  # pragma: no cover - defensive
            continue
    return out


def _in_box(mark: Mark, box) -> bool:
    return (
        float(box.x0) <= mark.x0
        and mark.x1 <= float(box.x1)
        and float(box.y0) <= mark.y0
        and mark.y1 <= float(box.y1)
    )


# ---------------------------------------------------------------------------
# Frame and y calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """The panel's plot frame: the axis line the bars rest on, and its span."""

    baseline: float
    x0: float
    x1: float
    tick_ys: tuple[float, ...]


def _stroked_horizontals(marks: list[Mark]) -> list[Mark]:
    return [m for m in marks if not m.filled and not m.dashed and m.horizontal]


def _span_groups(marks: list[Mark]) -> dict[tuple[float, float], list[float]]:
    groups: dict[tuple[float, float], list[float]] = {}
    for m in marks:
        groups.setdefault((round(m.x0, 3), round(m.x1, 3)), []).append(m.cy)
    return {span: sorted({round(y, 3) for y in ys}) for span, ys in groups.items()}


def find_frame(marks: list[Mark]) -> Frame | None:
    """The panel's axis line and its tick values, or ``None``.

    An axis is not "a long line"; it is *the line the ticks attach to*. So the
    two are found together: a candidate axis is a stroked horizontal, and its
    ladders are the groups of stroked horizontals that repeat one shorter span
    at two or more distinct heights INSIDE the candidate and sharing one of its
    endpoints -- which is where a tick is drawn and nowhere else. The plot's own
    top rule spans exactly the axis and shares both endpoints, so it is excluded
    by being the candidate's own span; the page's header rule attaches to
    nothing and is never a candidate at all.

    Among the candidates that have ladders, the axis is the LOWEST -- the plot's
    bottom rule, the one a bar rests on. Ladders that disagree about which
    heights are ticked (a left and a right ladder must agree) abstain, because a
    scale fitted to one of two contradictory ladders is a guess.
    """
    horizontals = _stroked_horizontals(marks)
    if not horizontals:
        return None
    groups = _span_groups(horizontals)
    best: tuple[Mark, tuple[float, ...]] | None = None
    for axis in horizontals:
        span = (round(axis.x0, 3), round(axis.x1, 3))
        ladders = [
            ys
            for other, ys in groups.items()
            if other != span
            and len(ys) >= 2
            and span[0] <= other[0]
            and other[1] <= span[1]
            and (other[0] == span[0] or other[1] == span[1])
        ]
        if not ladders:
            continue
        if len({tuple(ys) for ys in ladders}) != 1:
            logger.debug("chart_reader: tick ladders disagree at y=%.2f; skipping", axis.cy)
            continue
        ticks = tuple(ladders[0])
        if any(abs(axis.cy - t) <= max(axis.tolerance, 1e-6) for t in ticks):
            continue
        if best is None or axis.cy > best[0].cy:
            best = (axis, ticks)
    if best is None:
        return None
    axis, ticks = best
    return Frame(baseline=axis.cy, x0=axis.x0, x1=axis.x1, tick_ys=ticks)


@dataclass(frozen=True)
class YCalibration:
    """Points per unit, fitted to the labelled ticks and checked against the rest.

    ``residual`` is the largest absolute disagreement, in points, between the
    fit and any observation used to check it -- every labelled tick plus the
    axis line, which the fit must place at zero. It is carried into every
    count interval, so a calibration that barely closed produces wide
    intervals and therefore ``UNRESOLVED`` cells, rather than confident wrong
    integers.
    """

    points_per_unit: float
    zero_y: float
    pairs: tuple[tuple[float, float], ...]
    residual: float
    checked_ticks: int

    @property
    def half_count_points(self) -> float:
        """Δy/(2Δn): the distance on the page that half a count occupies."""
        return abs(self.points_per_unit) / 2.0

    def value(self, y: float) -> float:
        return (self.zero_y - y) / abs(self.points_per_unit)

    def to_dict(self) -> dict:
        return {
            "points_per_unit": self.points_per_unit,
            "zero_y": self.zero_y,
            "tick_pairs": [list(p) for p in self.pairs],
            "residual_points": self.residual,
            "ticks_checked": self.checked_ticks,
            "half_count_points": self.half_count_points,
        }


def _numeric(text: str) -> float | None:
    stripped = text.strip().replace(",", "")
    return float(stripped) if _NUM_RE.match(stripped) else None


@dataclass(frozen=True)
class WordRow:
    """One drawn word-row inside a region: its extent and its text."""

    y0: float
    y1: float
    x0: float
    x1: float
    text: str
    tokens: tuple[tuple[float, float, str], ...]

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


def region_word_rows(page, bboxes, index: int, page_bottom: float) -> list[WordRow]:
    """Word-rows the region at *index* owns, top to bottom. Never raises.

    The region OWNS a vertical slab: its own box, extended down to the top of
    the next region (or the foot of the page). The extension is not generosity
    -- ``chart_region_bboxes`` clips a panel at its plot, so a chart's second
    line of two-line x tick labels and its x axis title are drawn a point or
    two BELOW the box, and a reader confined to the box could never see the
    upper endpoint of a single printed bin. Stage 0 takes the same strip for
    the same reason (``region_axis_rows``).

    Ownership inside the box follows Stage 0's convention exactly: a word
    belongs to the box its drawing ORIGIN lies in, and a word also inside
    another region's box belongs to neither. In the strip below, the whole word
    must fall within the region's horizontal span, so the strip cannot pull in
    a marginal note or a neighbouring column.
    """
    from socr.figures.chart_data import _contested, _inside

    try:
        words = page.get_text("words") or []
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("chart_reader: get_text('words') failed: %s", exc)
        return []
    box = bboxes[index - 1]
    below = [float(o.y0) for o in bboxes if float(o.y0) > float(box.y0)]
    limit = min(below) if below else page_bottom
    buckets: dict[int, list] = {}
    for w in words:
        x0, y0, x1, y1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
        text = str(w[4])
        if not text.strip():
            continue
        if _contested(bboxes, index, x0, y0):
            continue
        in_box = _inside(box, x0, y0)
        in_strip = float(box.y1) < y0 < limit and float(box.x0) <= x0 and x1 <= float(box.x1)
        if not (in_box or in_strip):
            continue
        buckets.setdefault(round(y0, 1), []).append((x0, y0, x1, y1, text))
    rows: list[WordRow] = []
    for key in sorted(buckets):
        items = sorted(buckets[key])
        rows.append(
            WordRow(
                y0=min(i[1] for i in items),
                y1=max(i[3] for i in items),
                x0=min(i[0] for i in items),
                x1=max(i[2] for i in items),
                text=" ".join(i[4] for i in items),
                tokens=tuple(((i[0] + i[2]) / 2.0, i[2] - i[0], i[4]) for i in items),
            )
        )
    return rows


def calibrate_y(frame: Frame, rows: list[WordRow]) -> YCalibration | None:
    """Fit value -> y from the numeric words drawn beside the stroked ticks.

    A tick's label is the numeric word whose vertical centre is nearest that
    tick and which is drawn OUTSIDE the plot's horizontal span -- where an
    axis label goes and where no bar, outline or bin label can be. The pairing
    window is half the smallest gap between two ticks, so a label can never be
    claimed by the wrong tick, and there is no tolerance to pick.

    Fitted by least squares over at least two labelled ticks, then CHECKED:
    against every labelled tick, and against the axis line, which a scale read
    off a chart whose bars rest on the axis must place at zero. The largest
    disagreement becomes ``residual``.
    """
    ticks = list(frame.tick_ys)
    if len(ticks) < 2:
        return None
    gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
    window = min(abs(g) for g in gaps) / 2.0
    if window <= 0:
        return None
    candidates = [
        r for r in rows if (r.x1 < frame.x0 or r.x0 > frame.x1) and _numeric(r.text) is not None
    ]
    pairs: list[tuple[float, float]] = []
    for tick in ticks:
        near = [r for r in candidates if abs(r.cy - tick) <= window]
        if len(near) != 1:
            continue
        value = _numeric(near[0].text)
        if value is None:  # pragma: no cover - filtered above
            continue
        pairs.append((value, tick))
    if len(pairs) < 2:
        logger.debug("chart_reader: fewer than two labelled ticks; no calibration")
        return None
    n = len(pairs)
    mean_v = sum(v for v, _y in pairs) / n
    mean_y = sum(y for _v, y in pairs) / n
    denom = sum((v - mean_v) ** 2 for v, _y in pairs)
    if denom <= 0:
        return None
    slope = sum((v - mean_v) * (y - mean_y) for v, y in pairs) / denom
    if slope >= 0:
        # y grows downward on a PDF page, so a larger value must sit HIGHER.
        logger.debug("chart_reader: y scale is not inverted; refusing")
        return None
    zero_y = mean_y - slope * mean_v
    residual = max(abs(y - (zero_y + slope * v)) for v, y in pairs)
    residual = max(residual, abs(frame.baseline - zero_y))
    return YCalibration(
        points_per_unit=abs(slope),
        zero_y=zero_y,
        pairs=tuple(pairs),
        residual=residual,
        checked_ticks=n,
    )


# ---------------------------------------------------------------------------
# X bins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bin:
    """One printed x bin: its label as drawn, its centre and its interval."""

    label: str
    centre: float
    lo: float
    hi: float

    def to_dict(self) -> dict:
        return {"label": self.label, "centre": self.centre, "x0": self.lo, "x1": self.hi}


def _alnum_tokens(row: WordRow) -> list[tuple[float, str]]:
    return [(cx, text) for cx, _w, text in row.tokens if any(ch.isalnum() for ch in text)]


def _aligned(row: WordRow, centres: list[float]) -> list[str] | None:
    """The row's token nearest each of *centres*, or ``None`` if it does not align.

    This is how a two-line tick label is printed: the upper endpoint above the
    lower one, in the same column. A row that reuses a token for two columns,
    or whose picks run backwards, is not a second line of these labels.
    """
    tokens = _alnum_tokens(row)
    if len(tokens) < len(centres):
        return None
    picked = [min(range(len(tokens)), key=lambda i: abs(tokens[i][0] - c)) for c in centres]
    if len(set(picked)) != len(picked) or picked != sorted(picked):
        return None
    return [tokens[i][1] for i in picked]


def read_bins(frame: Frame, rows: list[WordRow]) -> list[Bin]:
    """The printed bin labels below the axis, with their intervals.

    The interval is the midpoints between consecutive label centres, with the
    outer edges mirrored from the neighbouring half-gap. It needs no
    uniform-spacing assumption and no tolerance. It is deliberately NOT what a
    mark is assigned by: a text bbox carries side bearings, so a printed label's
    centre sits a fraction of a point off the bin centre the chart was drawn on,
    and an interval derived from it cuts a bar's own edge. Assignment goes by
    the label CENTRE lying inside the mark -- see ``_owned_bins``. The interval
    is used only to say which bins an UNASSIGNABLE mark casts doubt over.
    """
    below = [r for r in rows if r.y0 > frame.baseline]
    if not below:
        return []
    best: WordRow | None = None
    for row in below:
        count = len(_alnum_tokens(row))
        if count >= 2 and (best is None or count > len(_alnum_tokens(best))):
            best = row
    if best is None:
        return []
    primaries = _alnum_tokens(best)
    centres = [cx for cx, _t in primaries]
    atoms: list[list[str]] = [[t] for _cx, t in primaries]
    cursor = below.index(best)
    for row in below[cursor + 1 :]:
        aligned = _aligned(row, centres)
        if aligned is None:
            break
        for i, token in enumerate(aligned):
            atoms[i].append(token)
    if len(centres) < 2:
        return []
    edges: list[float] = []
    for i, c in enumerate(centres):
        left = (centres[i - 1] + c) / 2.0 if i else c - (centres[1] - centres[0]) / 2.0
        right = (
            (c + centres[i + 1]) / 2.0
            if i + 1 < len(centres)
            else c + (centres[-1] - centres[-2]) / 2.0
        )
        edges.append(left)
        if i + 1 == len(centres):
            edges.append(right)
    out: list[Bin] = []
    for i, c in enumerate(centres):
        out.append(Bin(label=_ATOM_JOIN.join(atoms[i]), centre=c, lo=edges[i], hi=edges[i + 1]))
    return out


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegendEntry:
    """A series name bound to the drawing STYLE of the swatch beside it."""

    name: str
    style: str
    swatch: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {"name": self.name, "style": self.style, "swatch": list(self.swatch)}


def read_legend(frame: Frame, marks: list[Mark], rows: list[WordRow], bins: list[Bin]):
    """Legend entries in this panel: ``(entries, swatch marks)``.

    A legend swatch is a mark that does NOT rest on the axis, is narrower than
    a bin, covers no bin's printed label centre (a data mark spans at least its
    own bin, so it does both), and has a word-row beginning to its right whose
    vertical centre lies within the swatch's own painted extent. The name is that row's text, as drawn. The
    style comes from the swatch's geometry -- a filled rectangle, or a dashed
    stroke -- never from its colour, which no part of this module reads.
    """
    if not bins:
        return [], []
    entries: list[LegendEntry] = []
    swatches: list[Mark] = []
    narrowest = min(b.hi - b.lo for b in bins)
    for m in marks:
        if abs(m.y1 - frame.baseline) <= max(m.tolerance, 1e-6):
            continue
        # A swatch is a horizontal sample of the series' ink: a filled rectangle
        # or a stroked run. The vertical risers of a staircase are neither, and
        # excluding them here is what stops a tick label standing to their right
        # from being read as a series name.
        if not (m.filled or m.horizontal):
            continue
        # A data mark of a histogram spans at least one whole bin, so it is at
        # least a bin wide AND it covers that bin's printed label centre. A
        # swatch does neither. Both tests are kept: the width alone would sit on
        # a floating-point knife edge, since an outline segment one bin wide is
        # exactly as wide as the bin it draws.
        if (m.x1 - m.x0) >= narrowest or any(m.x0 <= b.centre <= m.x1 for b in bins):
            continue
        style = SOLID_FILL if m.filled else (DASHED_STROKE if m.dashed else "")
        if not style:
            continue
        reach = max(m.tolerance, 1e-6)
        named = [r for r in rows if r.x0 > m.x1 and (m.y0 - reach) <= r.cy <= (m.y1 + reach)]
        if not named:
            continue
        # The entry's name is the row drawn NEXT to the swatch. A y tick label
        # standing far to the right on the same line is also "to the right",
        # and taking the nearest is what keeps it from being read as a series
        # name. Legend text is set beside its swatch; nothing else is.
        label = min(named, key=lambda r: r.x0)
        entries.append(
            LegendEntry(name=label.text.strip(), style=style, swatch=(m.x0, m.y0, m.x1, m.y1))
        )
        swatches.append(m)
    # A style named twice in one legend names nothing.
    styles = [e.style for e in entries]
    if len(set(styles)) != len(styles):
        logger.debug("chart_reader: legend binds one style to two names; refusing")
        return [], swatches
    return entries, swatches


# ---------------------------------------------------------------------------
# Series geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One (series, bin) reading, with everything needed to check it."""

    bin_label: str
    status: str
    count: int | None
    interval: tuple[float, float]
    detail: str
    bar_bbox: tuple[float, float, float, float] | None = None
    top_y: float | None = None
    baseline_y: float | None = None
    empty_bin_observed: bool = False

    def to_dict(self) -> dict:
        return {
            "bin_label": self.bin_label,
            "status": self.status,
            "count": self.count,
            "interval": list(self.interval),
            "detail": self.detail,
            "bar_bbox": list(self.bar_bbox) if self.bar_bbox else None,
            "top_y": self.top_y,
            "baseline_y": self.baseline_y,
            "empty_bin_observed": self.empty_bin_observed,
        }

    @property
    def rendered(self) -> str:
        return UNRESOLVED_MARKER if self.status != INTEGER else str(self.count)


@dataclass(frozen=True)
class SeriesReading:
    """One series in one panel."""

    name: str
    style: str
    presence: str
    cells: tuple[Cell, ...] = ()
    detail: str = ""

    @property
    def counts(self) -> dict[str, int]:
        return {c.bin_label: c.count for c in self.cells if c.status == INTEGER}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "style": self.style,
            "presence": self.presence,
            "detail": self.detail,
            "cells": [c.to_dict() for c in self.cells],
        }


def _resolve(
    height: float, uncertainty: float, cal: YCalibration
) -> tuple[int | None, tuple[float, float], str]:
    """Turn a measured height into an integer count, or refuse.

    The interval is the height widened by *uncertainty* on each side and
    divided by the fitted points per participant. An integer is emitted only
    when exactly one non-negative integer lies inside -- which is what the
    calibration-derived bound guarantees when the uncertainty is below half a
    count, and what it correctly stops guaranteeing when it is not.
    """
    unit = abs(cal.points_per_unit)
    if unit <= 0:  # pragma: no cover - calibrate_y refuses a zero slope
        return None, (0.0, 0.0), "the calibration has no scale"
    if uncertainty >= cal.half_count_points:
        lo, hi = (height - uncertainty) / unit, (height + uncertainty) / unit
        return (
            None,
            (lo, hi),
            (
                f"measured uncertainty {uncertainty:.3f}pt is not below half a count "
                f"({cal.half_count_points:.3f}pt), so no integer is uniquely supported"
            ),
        )
    lo = (height - uncertainty) / unit
    hi = (height + uncertainty) / unit
    candidates = [n for n in range(max(0, math.floor(lo)), math.floor(hi) + 1) if lo <= n <= hi]
    if len(candidates) != 1:
        return (
            None,
            (lo, hi),
            (
                f"the count interval [{lo:.3f}, {hi:.3f}] supports "
                f"{len(candidates)} non-negative integers, not one"
            ),
        )
    return candidates[0], (lo, hi), ""


def _owned_bins(mark: Mark, bins: list[Bin]) -> list[int]:
    """The bins whose printed label centre lies inside *mark*'s footprint.

    This is the whole of bin assignment, and it is deliberately not a
    containment test against a derived interval. A histogram's mark spans its
    bin, so the bin's label centre is inside it and the neighbouring centres --
    a full bin away on either side -- are not. A mark that spans several bins,
    as a staircase's level does, owns exactly those. A mark that owns NOTHING
    is a mark this reader cannot place, and every bin it touches is unresolved
    rather than quietly read as empty.
    """
    return [i for i, b in enumerate(bins) if mark.x0 <= b.centre <= mark.x1]


def _doubted_by(strays: list[Mark], b: Bin) -> bool:
    """A mark this reader could not place overlaps this bin's interval."""
    return any(m.x1 > b.lo and m.x0 < b.hi for m in strays)


def read_solid_series(
    name: str,
    frame: Frame,
    cal: YCalibration,
    bins: list[Bin],
    marks: list[Mark],
) -> SeriesReading:
    """A bar series: filled rectangles standing on the axis."""
    bars = [
        m
        for m in marks
        if m.filled
        and abs(m.y1 - frame.baseline) <= max(m.tolerance, cal.residual)
        and m.y0 < frame.baseline
        and frame.x0 <= m.x0
        and m.x1 <= frame.x1
    ]
    if not bars:
        return SeriesReading(
            name=name,
            style=SOLID_FILL,
            presence=PRESENCE_UNRESOLVED,
            detail=(
                "no filled bar of this series is drawn anywhere in the panel, so the "
                "series being absent and every bin being zero are not distinguishable "
                "from this panel's geometry"
            ),
        )
    owned: dict[int, list[Mark]] = {}
    strays: list[Mark] = []
    for bar in bars:
        mine = _owned_bins(bar, bins)
        # A histogram bar spans exactly one bin by construction, so owning
        # none and owning several are the same evidence failure: the drawing
        # does not establish which bin this bar counts. Publishing it in every
        # bin it covers would state its height once per bin and invent the
        # difference. (The dashed reader does NOT share this rule -- a
        # staircase level legitimately runs across several bins.)
        if len(mine) != 1:
            strays.append(bar)
            continue
        owned.setdefault(mine[0], []).append(bar)
    cells: list[Cell] = []
    for i, b in enumerate(bins):
        here = owned.get(i, [])
        if _doubted_by(strays, b):
            cells.append(
                Cell(
                    bin_label=b.label,
                    status=UNRESOLVED,
                    count=None,
                    interval=(0.0, 0.0),
                    detail=(
                        "a bar overlaps this bin but does not cover exactly one bin's "
                        "printed label centre -- it covers none, or it spans several -- "
                        "so which bin it belongs to is not established"
                    ),
                    baseline_y=frame.baseline,
                )
            )
            continue
        if len(here) > 1:
            cells.append(
                Cell(
                    bin_label=b.label,
                    status=UNRESOLVED,
                    count=None,
                    interval=(0.0, 0.0),
                    detail=f"{len(here)} bars of this series cover this bin's label centre",
                    baseline_y=frame.baseline,
                )
            )
            continue
        if not here:
            inside = frame.x0 <= b.lo and b.hi <= frame.x1
            cells.append(
                Cell(
                    bin_label=b.label,
                    status=INTEGER if inside else UNRESOLVED,
                    count=0 if inside else None,
                    interval=(0.0, 0.0),
                    detail=(
                        "the bin lies inside the drawn plot and carries no bar; this "
                        "series' bars rest on the axis, so a zero bar is invisible by "
                        "construction and the empty bin is the observation"
                        if inside
                        else "the bin is not wholly inside the drawn plot, so it was not observed"
                    ),
                    baseline_y=frame.baseline,
                    empty_bin_observed=inside,
                )
            )
            continue
        bar = here[0]
        height = frame.baseline - bar.y0
        uncertainty = cal.residual + bar.tolerance
        count, interval, why = _resolve(height, uncertainty, cal)
        cells.append(
            Cell(
                bin_label=b.label,
                status=INTEGER if count is not None else UNRESOLVED,
                count=count,
                interval=interval,
                detail=why,
                bar_bbox=(bar.x0, bar.y0, bar.x1, bar.y1),
                top_y=bar.y0,
                baseline_y=frame.baseline,
            )
        )
    return SeriesReading(name=name, style=SOLID_FILL, presence=PRESENT, cells=tuple(cells))


def read_dashed_series(
    name: str,
    frame: Frame,
    cal: YCalibration,
    bins: list[Bin],
    marks: list[Mark],
) -> SeriesReading:
    """A staircase series: a dashed outline over the bins.

    The outline is a level function of x. Its horizontal runs give the level
    where they run; everywhere else inside the drawn plot the level is the
    axis, because the axis is already stroked there and an outline lying on it
    adds no ink. That reading is not assumed -- it is CHECKED against every
    vertical riser: a riser must join the level on its left to the level on its
    right. One riser that does not means the outline was not reconstructed, and
    the whole series goes ``UNRESOLVED`` rather than publishing a staircase the
    page does not draw.

    That check is about the risers that exist. A bin no run covers needs the
    opposite: the riser that brings the outline DOWN to the axis beside it must
    have been drawn. Where the neighbouring level is there and its descent is
    not, the bin is ``UNRESOLVED`` -- see the empty-bin branch below.
    """
    inside = [
        m
        for m in marks
        if m.dashed and frame.x0 <= m.x0 and m.x1 <= frame.x1 and m.y1 <= frame.baseline + 1e-6
    ]
    if not inside:
        return SeriesReading(
            name=name,
            style=DASHED_STROKE,
            presence=PRESENCE_UNRESOLVED,
            detail=(
                "no dashed outline of this series is drawn anywhere in the panel, so "
                "the series being absent and every bin being zero are not "
                "distinguishable from this panel's geometry"
            ),
        )
    runs = [m for m in inside if m.horizontal]
    risers = [m for m in inside if m.vertical and not m.horizontal]

    def level_at(x: float) -> float:
        covering = [m for m in runs if m.x0 - m.tolerance <= x <= m.x1 + m.tolerance]
        if not covering:
            return frame.baseline
        return min(covering, key=lambda m: abs(m.cx - x)).cy

    # A riser is probed half a bin to either side, not a hair's breadth: the
    # runs it joins START at it, so a probe inside their own stroke tolerance
    # lands on both of them at once and the step disappears. Half a bin is the
    # nearest distance at which a histogram outline's level is unambiguous --
    # its levels change only at bin edges -- and it is derived from the bins
    # the page printed, not chosen.
    probe = min(b.hi - b.lo for b in bins) / 2.0 if bins else 0.0
    broken = ""
    for v in risers:
        step = max(v.tolerance, cal.residual)
        left, right = level_at(v.x0 - probe), level_at(v.x0 + probe)
        ends = sorted((v.y0, v.y1))
        if not (
            abs(min(left, right) - ends[0]) <= step and abs(max(left, right) - ends[1]) <= step
        ):
            broken = (
                f"a riser of the outline at x={v.x0:.2f} spans "
                f"[{v.y0:.2f}, {v.y1:.2f}] but joins levels {left:.2f} and {right:.2f}"
            )
            break
    if broken:
        return SeriesReading(
            name=name,
            style=DASHED_STROKE,
            presence=PRESENT,
            cells=tuple(
                Cell(
                    bin_label=b.label,
                    status=UNRESOLVED,
                    count=None,
                    interval=(0.0, 0.0),
                    detail=broken,
                    baseline_y=frame.baseline,
                )
                for b in bins
            ),
            detail=broken,
        )

    strays = [m for m in runs if not _owned_bins(m, bins)]
    cover = {i: [m for m in runs if i in _owned_bins(m, bins)] for i in range(len(bins))}

    def descent_at(x: float, seg: Mark) -> str:
        """``""`` when the outline is drawn coming down to the axis at *x*.

        Otherwise the reason it is not established, in the page's own terms.
        Every bound here is measured off the drawing -- the riser's own half
        stroke width, the run's, and the calibration residual -- and it is held
        to the SAME half-count rule ``_resolve`` holds every height to. Without
        that, a stroke thicker than the per-participant pitch would let a
        descent stopping most of a participant above the axis count as reaching
        it: the one number such a panel published would be the one number its
        own geometry could not support.
        """
        slack = max(seg.tolerance, cal.residual)
        coarse = False
        stopped = False
        for v in risers:
            bound = max(v.tolerance, slack)
            if abs(v.cx - x) > bound:
                continue
            if abs(max(v.y0, v.y1) - frame.baseline) > bound:
                stopped = True
                continue
            if bound >= cal.half_count_points:
                coarse = True
                continue
            return ""
        if coarse:
            return (
                "a riser is drawn there, but its own edge uncertainty is not below half "
                "a count, so it does not establish that the outline reached the axis"
            )
        if stopped:
            return "the riser drawn there stops short of the axis"
        return "no riser of the outline is drawn descending to the axis there"

    cells: list[Cell] = []
    for i, b in enumerate(bins):
        covering = cover[i]
        if _doubted_by(strays, b):
            cells.append(
                Cell(
                    bin_label=b.label,
                    status=UNRESOLVED,
                    count=None,
                    interval=(0.0, 0.0),
                    detail=(
                        "a run of the outline overlaps this bin but covers no bin's "
                        "printed label centre, so its bin is not established"
                    ),
                    baseline_y=frame.baseline,
                )
            )
            continue
        if len(covering) > 1:
            cells.append(
                Cell(
                    bin_label=b.label,
                    status=UNRESOLVED,
                    count=None,
                    interval=(0.0, 0.0),
                    detail=(
                        "the outline draws more than one level over this bin, so the "
                        "bin's own level is not a single measurement"
                    ),
                    baseline_y=frame.baseline,
                )
            )
            continue
        if not covering:
            observed = frame.x0 <= b.lo and b.hi <= frame.x1
            # An uncovered bin is the axis only where the page SHOWS the
            # outline coming down to it, and the question is about the whole
            # undrawn stretch, not about two array indices: the evidence for a
            # bin in the MIDDLE of a gap is the descent at the far end of that
            # gap, which is one bin away or five. So walk out in each direction
            # to the nearest bin the outline says anything about, and ask that
            # one. A bin whose own level this reader could not resolve is
            # evidence MISSING, never evidence not required -- it is exactly
            # the case where the drawing is least trustworthy. The detail
            # describes what was drawn, because these details are the audit
            # trail behind every published zero.
            missing: list[str] = []
            supported: list[str] = []
            for step in (-1, 1):
                j = i + step
                while 0 <= j < len(bins) and not cover[j] and not _doubted_by(strays, bins[j]):
                    j += step
                if not 0 <= j < len(bins):
                    continue
                near = cover[j]
                if len(near) != 1 or _doubted_by(strays, bins[j]):
                    missing.append(
                        f"the level over {bins[j].label} is not itself established, so "
                        "it cannot witness a descent to the axis"
                    )
                    continue
                seg = near[0]
                if abs(seg.cy - frame.baseline) <= max(seg.tolerance, cal.residual):
                    supported.append(f"the outline runs on the axis over {bins[j].label}")
                    continue
                why = descent_at(seg.x1 if j < i else seg.x0, seg)
                if why:
                    missing.append(
                        f"where the outline leaves the level over {bins[j].label}, {why}"
                    )
                else:
                    supported.append(
                        f"the outline is drawn descending to the axis beside {bins[j].label}"
                    )
            if observed and missing:
                cells.append(
                    Cell(
                        bin_label=b.label,
                        status=UNRESOLVED,
                        count=None,
                        interval=(0.0, 0.0),
                        detail=(
                            "the outline draws no level over this bin, and "
                            + "; ".join(missing)
                            + ", so the outline stopping short of this bin and the "
                            "outline falling to zero over it are the same picture"
                        ),
                        baseline_y=frame.baseline,
                    )
                )
                continue
            cells.append(
                Cell(
                    bin_label=b.label,
                    status=INTEGER if observed else UNRESOLVED,
                    count=0 if observed else None,
                    interval=(0.0, 0.0),
                    detail=(
                        (
                            "the outline draws no level over this bin while the axis is "
                            "stroked across it, and "
                            + (
                                "; ".join(supported)
                                if supported
                                else "the outline draws no level anywhere on either side "
                                "of it for it to descend from"
                            )
                            + ", so the level here is the axis"
                        )
                        if observed
                        else "the bin is not wholly inside the drawn plot, so it was not observed"
                    ),
                    baseline_y=frame.baseline,
                    empty_bin_observed=observed,
                )
            )
            continue
        seg = covering[0]
        height = frame.baseline - seg.cy
        uncertainty = cal.residual + seg.tolerance
        count, interval, why = _resolve(height, uncertainty, cal)
        cells.append(
            Cell(
                bin_label=b.label,
                status=INTEGER if count is not None else UNRESOLVED,
                count=count,
                interval=interval,
                detail=why,
                bar_bbox=(max(seg.x0, b.lo), seg.cy, min(seg.x1, b.hi), frame.baseline),
                top_y=seg.cy,
                baseline_y=frame.baseline,
            )
        )
    return SeriesReading(name=name, style=DASHED_STROKE, presence=PRESENT, cells=tuple(cells))


# ---------------------------------------------------------------------------
# Panels and the page reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelReading:
    """One chart region, read."""

    page_num: int
    region_index: int
    label: str
    bins: tuple[Bin, ...]
    series: tuple[SeriesReading, ...]
    calibration: YCalibration
    frame: Frame
    crop_filename: str = ""
    crop_sha256: str = ""
    crop_dpi: int = 0
    crop_clip: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    source_checksum: str = ""
    count_unit: str = ""
    bin_unit: str = ""
    legend_from_region: int = 0
    verification: str = UNVERIFIED
    verification_detail: str = ""
    internal_issues: tuple[str, ...] = ()

    @property
    def readable(self) -> bool:
        return any(s.presence == PRESENT for s in self.series)

    @property
    def unresolved_cells(self) -> int:
        return sum(1 for s in self.series for c in s.cells if c.status != INTEGER)

    @property
    def resolved_cells(self) -> int:
        return sum(1 for s in self.series for c in s.cells if c.status == INTEGER)

    def to_dict(self) -> dict:
        return {
            "reader_version": READER_VERSION,
            "page_num": self.page_num,
            "region_index": self.region_index,
            "panel": self.label,
            "source_checksum": self.source_checksum,
            "crop": self.crop_filename,
            "crop_sha256": self.crop_sha256,
            "crop_dpi": self.crop_dpi,
            "crop_clip": list(self.crop_clip),
            "bin_unit": self.bin_unit,
            "count_unit": self.count_unit,
            "legend_from_region": self.legend_from_region,
            "bins": [b.to_dict() for b in self.bins],
            "calibration": self.calibration.to_dict(),
            "frame": {
                "baseline": self.frame.baseline,
                "x0": self.frame.x0,
                "x1": self.frame.x1,
                "tick_ys": list(self.frame.tick_ys),
            },
            "series": [s.to_dict() for s in self.series],
            "verification": self.verification,
            "verification_detail": self.verification_detail,
            "internal_issues": list(self.internal_issues),
        }


@dataclass
class PageReading:
    """Every chart region on one page, read or refused."""

    page_num: int
    panels: dict[int, PanelReading] = field(default_factory=dict)
    refusals: dict[int, str] = field(default_factory=dict)


def _panel_label(frame: Frame, rows: list[WordRow], shared: set[str]) -> str:
    """The panel's own heading.

    The topmost row that (a) no other panel draws, (b) is drawn wholly inside
    the plot's horizontal span, and (c) sits above the highest tick. That is
    where a panel heading goes, and the three conditions between them exclude
    the figure's axis titles (drawn in every panel, and overhanging the plot
    where the tick labels are), the tick labels themselves, and the legend
    (below the heading).
    """
    top_tick = min(frame.tick_ys) if frame.tick_ys else frame.baseline
    for row in rows:
        text = row.text.strip()
        if not text or text in shared:
            continue
        if row.y1 > top_tick:
            continue
        if not (frame.x0 <= row.x0 and row.x1 <= frame.x1):
            continue
        return text
    return ""


def _titles(frame: Frame, rows: list[WordRow], bins: list[Bin]) -> tuple[str, str]:
    """``(count unit, bin unit)`` as the page draws them, or empty strings.

    The count unit is the topmost multi-token row above the highest tick that
    is NOT drawn wholly inside the plot -- an axis annotation overhangs the
    plot on the side its tick labels are set, which is exactly what tells it
    apart from a panel heading set inside the frame. The bin unit is the first
    row below the printed bin labels that is narrower than the bin row itself.
    Both are read off the page; neither is named anywhere in this module.
    """
    top_tick = min(frame.tick_ys) if frame.tick_ys else frame.baseline
    count_unit = ""
    for row in rows:
        if row.y1 > top_tick or len(_alnum_tokens(row)) < 2:
            continue
        if frame.x0 <= row.x0 and row.x1 <= frame.x1:
            continue
        count_unit = row.text.strip()
        break
    bin_unit = ""
    if bins:
        rightmost = max(b.centre for b in bins)
        for row in [r for r in rows if r.y0 > frame.baseline]:
            picks = _alnum_tokens(row)
            if picks and len(picks) < len(bins) and row.x0 < rightmost:
                bin_unit = row.text.strip()
                break
    return count_unit, bin_unit


def read_chart_page(
    page,
    bboxes,
    *,
    page_num: int,
    source_checksum: str = "",
    crop_names: dict[int, str] | None = None,
    crop_digests: dict[int, tuple[str, int, tuple[float, float, float, float]]] | None = None,
) -> PageReading:
    """Read every chart region on *page*. Never raises.

    A region that cannot be read -- no vector frame, no calibration, no bins,
    no legend that names its styles -- is recorded as a refusal with the reason,
    and its crop and Stage 0 note stand unchanged. Refusing is the safe outcome
    and is not an error.
    """
    reading = PageReading(page_num=page_num)
    if not bboxes:
        return reading
    marks = page_marks(page)
    if not marks:
        for idx in range(1, len(bboxes) + 1):
            reading.refusals[idx] = (
                "the page draws no vector operators (a raster chart is out of "
                "scope for this reader)"
            )
        return reading

    try:
        page_bottom = float(page.rect.y1)
    except Exception:  # pragma: no cover - defensive
        page_bottom = max((float(b.y1) for b in bboxes), default=0.0)
    rows_by_region = {
        idx: region_word_rows(page, bboxes, idx, page_bottom) for idx in range(1, len(bboxes) + 1)
    }
    counts: dict[str, int] = {}
    for rows in rows_by_region.values():
        for text in {r.text.strip() for r in rows}:
            counts[text] = counts.get(text, 0) + 1
    shared = {t for t, n in counts.items() if n > 1}

    frames: dict[int, Frame] = {}
    cals: dict[int, YCalibration] = {}
    bins_by_region: dict[int, list[Bin]] = {}
    marks_by_region: dict[int, list[Mark]] = {}
    legends: dict[int, list[LegendEntry]] = {}
    swatches: dict[int, list[Mark]] = {}

    for idx, box in enumerate(bboxes, start=1):
        own = [m for m in marks if _in_box(m, box)]
        marks_by_region[idx] = own
        frame = find_frame(own)
        if frame is None:
            reading.refusals[idx] = (
                "no vector plot frame (a stroked axis with a tick ladder) is drawn in "
                "this region; a raster chart is out of scope for this reader"
            )
            continue
        rows = rows_by_region[idx]
        cal = calibrate_y(frame, rows)
        if cal is None:
            reading.refusals[idx] = (
                "fewer than two of the region's ticks carry a numeric label beside them, "
                "so no y scale could be fitted"
            )
            continue
        bins = read_bins(frame, rows)
        if len(bins) < 2:
            reading.refusals[idx] = "the region prints fewer than two x bin labels below its axis"
            continue
        frames[idx] = frame
        cals[idx] = cal
        bins_by_region[idx] = bins
        entries, marks_used = read_legend(frame, own, rows, bins)
        legends[idx] = entries
        swatches[idx] = marks_used

    # Figure-level legend binding. Two regions belong to one FIGURE when they
    # print the same bin labels in the same order and calibrate against the
    # same tick values -- the page's own evidence that they are panels of one
    # chart. A legend found in one member then names the styles for the whole
    # group, and the region it came from is recorded per panel. A group whose
    # members carry two DIFFERENT legends binds nothing: which one governs is
    # not established, and guessing it is how one panel's series names end up
    # on another panel's bars.
    groups: dict[tuple, list[int]] = {}
    for idx in frames:
        key = (
            tuple(b.label for b in bins_by_region[idx]),
            tuple(sorted(v for v, _y in cals[idx].pairs)),
        )
        groups.setdefault(key, []).append(idx)
    group_legend: dict[int, tuple[list[LegendEntry], int]] = {}
    group_units: dict[int, tuple[str, str]] = {}
    for members in groups.values():
        found = [(idx, legends[idx]) for idx in members if legends[idx]]
        if found:
            shapes = {tuple(sorted((e.style, e.name) for e in entries)) for _idx, entries in found}
            if len(shapes) == 1:
                source_idx, entries = found[0]
                for idx in members:
                    group_legend[idx] = (entries, source_idx)
        # The axis titles are drawn once per panel, but a panel's bbox is cut
        # at its plot, so on a stacked figure a panel's own title can fall in
        # the slab of the panel above it. The titles are a property of the
        # FIGURE, so they are resolved over the group by the same evidence that
        # binds its legend -- and only when the members that do print one all
        # print the SAME one.
        seen_counts = {
            _titles(frames[idx], rows_by_region[idx], bins_by_region[idx])[0] for idx in members
        } - {""}
        seen_bins = {
            _titles(frames[idx], rows_by_region[idx], bins_by_region[idx])[1] for idx in members
        } - {""}
        units = (
            next(iter(seen_counts)) if len(seen_counts) == 1 else "",
            next(iter(seen_bins)) if len(seen_bins) == 1 else "",
        )
        for idx in members:
            group_units[idx] = units

    for idx in sorted(frames):
        bound = group_legend.get(idx)
        if bound is None:
            reading.refusals[idx] = (
                "no legend on this figure binds a series name to a drawing style, so the "
                "panel's marks cannot be attributed to a named series"
            )
            continue
        entries, legend_region = bound
        frame, cal, bins = frames[idx], cals[idx], bins_by_region[idx]
        data_marks = [m for m in marks_by_region[idx] if m not in swatches[idx]]
        series: list[SeriesReading] = []
        for entry in entries:
            if entry.style == SOLID_FILL:
                series.append(read_solid_series(entry.name, frame, cal, bins, data_marks))
            elif entry.style == DASHED_STROKE:
                series.append(read_dashed_series(entry.name, frame, cal, bins, data_marks))
        if not any(s.presence == PRESENT for s in series):
            reading.refusals[idx] = (
                "the legend names series this panel draws no marks for, so nothing was read"
            )
            continue
        rows = rows_by_region[idx]
        count_unit, bin_unit = group_units.get(idx, ("", ""))
        digest = (crop_digests or {}).get(idx, ("", 0, (0.0, 0.0, 0.0, 0.0)))
        panel = PanelReading(
            page_num=page_num,
            region_index=idx,
            label=_panel_label(frame, rows, shared),
            bins=tuple(bins),
            series=tuple(series),
            calibration=cal,
            frame=frame,
            crop_filename=(crop_names or {}).get(idx, ""),
            crop_sha256=digest[0],
            crop_dpi=digest[1],
            crop_clip=digest[2],
            source_checksum=source_checksum,
            count_unit=count_unit,
            bin_unit=bin_unit,
            legend_from_region=legend_region,
        )
        reading.panels[idx] = _internally_checked(panel)
    return reading


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

#: The caller hook: ``(survey_key, horizon, {series name: {bin label: count}})``
#: -> ``ACCEPT`` / ``REJECT`` / ``NO_OPINION``. Totals live in the CALLER. This
#: module never holds an expected total, for any survey, series or horizon.
ConstraintHook = Callable[[str, str, dict[str, dict[str, int]]], str]


def _internally_checked(panel: PanelReading) -> PanelReading:
    """The checks this module can make without any caller knowledge.

    Every emitted count is a non-negative integer, and every bin of a PRESENT
    series has a status -- complete observed-bin accounting, so a panel cannot
    quietly omit a bin it failed to look at. A failure here does not discard
    the reading; it is recorded and the panel stays unverified, because the
    evidence a reader needs to judge it is the reading itself.
    """
    issues: list[str] = []
    for s in panel.series:
        if s.presence != PRESENT:
            continue
        if len(s.cells) != len(panel.bins):
            issues.append(
                f"series “{s.name}” accounts for {len(s.cells)} of {len(panel.bins)} bins"
            )
        for c in s.cells:
            if c.status == INTEGER and (c.count is None or c.count < 0):
                issues.append(f"series “{s.name}” bin {c.bin_label} emitted a non-integer count")
    if not issues:
        return panel
    return PanelReading(**{**panel.__dict__, "internal_issues": tuple(issues)})


def verify_panel(
    panel: PanelReading,
    survey_key: str,
    hook: ConstraintHook | None,
) -> PanelReading:
    """Apply the caller's acceptance hook to one panel.

    Without a hook the derivation is labelled ``unverified`` -- published,
    marked, and never presented as checked. A hook that REJECTS rejects the
    DERIVATION: the counts are withheld from the published table and the crop
    stands. It never rejects the image, and it never causes a number to be
    adjusted to satisfy it.
    """
    if panel.internal_issues:
        return PanelReading(
            **{
                **panel.__dict__,
                "verification": REJECTED,
                "verification_detail": "; ".join(panel.internal_issues),
            }
        )
    if hook is None:
        return PanelReading(
            **{
                **panel.__dict__,
                "verification": UNVERIFIED,
                "verification_detail": "no acceptance hook was supplied",
            }
        )
    payload = {s.name: s.counts for s in panel.series if s.presence == PRESENT}
    try:
        verdict = str(hook(survey_key, panel.label, payload) or NO_OPINION)
    except Exception as exc:
        logger.warning("#635: acceptance hook raised on p%d: %s", panel.page_num, exc)
        return PanelReading(
            **{
                **panel.__dict__,
                "verification": UNVERIFIED,
                "verification_detail": f"the acceptance hook raised {type(exc).__name__}",
            }
        )
    if verdict == ACCEPT:
        detail = "the caller's acceptance hook accepted this derivation"
        return PanelReading(
            **{**panel.__dict__, "verification": VERIFIED, "verification_detail": detail}
        )
    if verdict == REJECT:
        return PanelReading(
            **{
                **panel.__dict__,
                "verification": REJECTED,
                "verification_detail": "the caller's acceptance hook rejected this derivation",
            }
        )
    return PanelReading(
        **{
            **panel.__dict__,
            "verification": UNVERIFIED,
            "verification_detail": "the caller's acceptance hook had no opinion",
        }
    )


# ---------------------------------------------------------------------------
# Published artifact
# ---------------------------------------------------------------------------


def derivation_prefix(page_num: int, region_index: int) -> str:
    return f"> **Chart region {region_index} on page {page_num} — counts read from the source**"


def _escape(text: str) -> str:
    return text.replace("|", "\\|").strip()


def panel_block(panel: PanelReading) -> str:
    """The markdown this panel publishes where its empty grid stood.

    A rejected derivation publishes no table: the note says the counts were
    derived and refused, and names the constraint that refused them. The crop
    is referenced either way -- rejecting a derivation never rejects the image.
    """
    present = [s for s in panel.series if s.presence == PRESENT]
    unknown = [s for s in panel.series if s.presence != PRESENT]
    head = derivation_prefix(panel.page_num, panel.region_index)
    if panel.label:
        head += f" ({_escape(panel.label)})"
    if panel.verification == REJECTED:
        return (
            f"{head} — the counts derived from this chart's vector geometry were "
            f"REJECTED ({panel.verification_detail}) and are not published. The chart "
            f"image `{panel.crop_filename}` is preserved."
        )
    flag = {
        VERIFIED: "verified against the caller's acceptance constraint",
        UNVERIFIED: f"UNVERIFIED ({panel.verification_detail})",
    }[panel.verification]
    units = ""
    if panel.count_unit or panel.bin_unit:
        units = (
            f" Columns are the chart's printed bins"
            f"{f' ({_escape(panel.bin_unit)})' if panel.bin_unit else ''}; "
            f"cells are counts"
            f"{f' ({_escape(panel.count_unit)})' if panel.count_unit else ''}."
        )
    absent = ""
    if unknown:
        absent = " " + " ".join(
            f"Series “{_escape(s.name)}”: {s.presence} — {s.detail}." for s in unknown
        )
    lines = [
        f"{head} — read from this page's own vector drawings by socr chart reader "
        f"{READER_VERSION}; {flag}.{units}{absent} The chart image "
        f"`{panel.crop_filename}` is preserved.",
        "",
        "| Series | " + " | ".join(_escape(b.label) for b in panel.bins) + " |",
        "| :--- | " + " | ".join("---:" for _b in panel.bins) + " |",
    ]
    for s in present:
        by_label = {c.bin_label: c for c in s.cells}
        cells = [
            by_label[b.label].rendered if b.label in by_label else UNRESOLVED_MARKER
            for b in panel.bins
        ]
        lines.append(f"| {_escape(s.name)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Crop provenance
# ---------------------------------------------------------------------------


def crop_digest(page, bbox, dpi: int) -> tuple[str, int, tuple[float, float, float, float]]:
    """``(sha256, dpi, clip)`` for the crop the pipeline will write for *bbox*.

    Rendered with the SAME matrix, rotation and clip ``_render_chart_region_crops``
    uses, so the digest names the file that ships. Returns an empty digest on
    any failure -- provenance that could not be computed says so rather than
    naming a file it did not see.
    """
    clip = (float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1))
    try:
        import fitz

        from socr.core.born_digital import upright_rotation_for

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        rotation = upright_rotation_for(page, clip=bbox)
        if rotation:
            mat = mat.prerotate(rotation)
        pix = page.get_pixmap(matrix=mat, clip=bbox)
        return hashlib.sha256(pix.tobytes("png")).hexdigest(), dpi, clip
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("chart_reader: crop digest failed: %s", exc)
        return "", dpi, clip
