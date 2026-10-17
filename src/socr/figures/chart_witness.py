"""#1155: the document's own projection table as an independent witness for a
chart reading.

A deterministic chart reading needs something to be checked against before its
numbers ship, and the alternatives available on the page are weak. The
participant-count sum is blind to a one-bin shift, to a bin swap, to offsetting
errors, and to binding the previous meeting's series as the current one's --
both the September and June 2015 FOMC meetings had 17 participants, so June's
bars bound as September's sum to 17 just as happily. The figure's own
``count_unit`` ("Number of participants") declares a unit, not a number.

The Fed SEP release prints, on another page of the SAME document, a table that
publishes each variable's median, central tendency and RANGE for each year, as
exact PDF characters. That range is independent of the drawing, so it can
witness the drawing. Measured for 2015-09-17 against the Fed's own published
HTML (``fomcprojtabl20150917.htm``): all 210 published numeric values appear on
the PDF page exactly, recall 1.0000.

What this module provides:

* ``parse_table1_witness`` -- the published ranges, keyed by variable, series
  row and year, for rows that pass their OWN self-check;
* ``gate_series`` -- whether a series' bins are consistent with one such range.

Two rules this module exists to enforce, both of them the owner's:

* **a wrong number is worse than a missing one.** Every failure path here
  returns a refusal with a named reason, never a guess and never a silent pass.
  No witness, an unrecognised layout, a row that fails its own self-check, or a
  reading that contradicts the range: all withhold.
* **the witness must itself be proven.** A parsed row is only a witness when
  its own arithmetic holds -- the median and the central tendency each inside
  the range, the median inside the central tendency, every value bound to one
  column of the table's own header, and no year printing a range while a group
  the table declares is absent. A table whose parse is unchecked would launder
  a parsing bug into a published number.

How it reads a table (#1184). Geometrically. The columns are the labels the
table's own HEADER prints, never a grid clustered out of the data: every data
row seeds such a grid, so each cell then falls inside its own cluster's spread
by identity and a stray figure is admitted by the tolerance it widened. Each
value binds to one header column -- within a quarter of a cell's own lo-to-hi
span, which is the distance a dropped figure would move it -- or the row is
refused by name. Groups are placed by where their headings sit: beside the
columns when the header's year sequence repeats (2010-04-28, 2015-09-17), and
as stacked blocks over one shared grid when it does not (2008-10-29). The
reading frame comes from the direction the page records, so a table drawn
rotated (2015-09-17) is read like any other.

What it does NOT do. The count rule this replaced -- a row's value total
against the header's column total -- rejected a column the release never
prints (longer-run core PCE, uncollected) and every row of a stacked table.
Binding tells those apart; it does not make the layout set closed. The Fed
minutes corpus runs to 1,021 documents over more decades than the three
measured here, and pre-2007 fractional projections, horizon headers that are
not years, and content drawn at other angles are all REFUSED with a named
reason rather than guessed at.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass, field

#: A range separator as the releases print it: en dash, em dash, hyphen, or the
#: word "to". NOT a minus sign -- a bin label's "1.8-1.9" is two positive
#: numbers joined by a hyphen, and reading that hyphen as a sign yields
#: ``(1.8, -1.9)``, which then fails every comparison silently. This cost one
#: wrong measurement run during #1155 before it was caught.
_MINUS = "-−"
#: A figure, which MAY be negative. The Fed prints negative projections
#: ("-0.3 to 0.5", 2008-10-29 Table 1) and negative federal-funds bins
#: ("−0.37-−0.13", with a Unicode MINUS, not a hyphen). An unsigned pattern
#: dropped those signs silently: Argon measured
#: ``bin_interval("−0.37-−0.13")`` returning ``(0.37, 0.13)`` -- an interval
#: whose low exceeds its high -- and ``-0.3 to 0.5`` parsed as ``(0.3, 0.5)``.
_NUM = rf"[{_MINUS}]?\d+\.\d+"
#: The separator between two figures. A hyphen is BOTH the separator and a
#: possible sign, so the two are told apart by what follows: a separating
#: hyphen always has a figure after it, and that figure may carry its own
#: sign. Matching the sign first would eat the separator; requiring a digit or
#: a sign after the separator keeps "−0.37-−0.13" splitting on the bare "-".
_SEP = rf"(?:\s*(?:to|–|—)\s*|\s*[-−]\s*(?=[{_MINUS}]?\d))"
_RANGE_RE = re.compile(rf"({_NUM}){_SEP}({_NUM})")
_YEAR_RE = re.compile(r"^(?:19|20)\d\d$")
#: One printed figure, as a WHOLE token. The table prints every projection with
#: a decimal point; requiring one is what keeps a year label ("2008") out of the
#: value stream, since the header band's own tokens would otherwise read as four
#: integer cells of a data row.
_VALUE_RE = re.compile(rf"^{_NUM}$")
#: The separator between a cell's two figures, as its OWN token. Pairing is done
#: on three CONSECUTIVE tokens rather than by scanning a band with a regex: a
#: band-wide scan pairs the last figure of one column with the first figure of
#: the column two to its right whenever the cell between them is empty, and
#: publishes that as a range (Argon, #1184 design review P0-2.3).
_SEP_TOKENS = frozenset({"to", "–", "—", "-", "−"})
#: A cell the table prints as present-but-empty. 2008-10-29 spells it "n/a" and
#: 2015-09-17 "n.a."; matching only one of them loses a column's shape.
_HOLE_RE = re.compile(r"^n[./]a\.?$")
#: A sub-row of a variable, naming the meeting whose projections it repeats.
_PRIOR_ROW_RE = re.compile(r"^([A-Z][a-z]+)\s+projection")
#: Footnote markers and dot leaders a row label carries in print but not in its
#: identity ("Core PCE inflation3. . . . . .").
_LABEL_NOISE_RE = re.compile(r"[.\s]+$|\d+$")

#: The column groups the projection table prints. This is the ONE place a
#: printed word is matched, and it is unavoidable: the witness wanted is
#: specifically the published RANGE, and the self-check "median inside central
#: tendency inside range" cannot be stated without knowing which group is
#: which. Variable names, years and column counts stay derived from the page.
GROUP_MEDIAN = "median"
GROUP_CENTRAL = "central tendency"
GROUP_RANGE = "range"
_GROUP_WORDS = {
    "median": GROUP_MEDIAN,
    "central": GROUP_CENTRAL,
    "tendency": GROUP_CENTRAL,
    "range": GROUP_RANGE,
}

#: ``layout`` values. Named so a caller's mark can quote them.
LAYOUT_SIDE_BY_SIDE_GROUPS = "side_by_side_groups"
LAYOUT_STACKED_BLOCKS = "stacked_blocks"

#: ``refusal`` values: why a whole table yielded no witness at all.
REFUSED_NO_TEXT = "the page carries no text layer to read a table from"
REFUSED_UNSUPPORTED_ROTATION = "the page's text is drawn at an angle this parser does not read"
REFUSED_NO_HEADER = "no year header row found in this table"
REFUSED_NO_SCALE = (
    "no two-figure cell on the page, so the column tolerance cannot be derived from it"
)
REFUSED_COLUMNS_UNRESOLVED = (
    "the header's column labels do not resolve to one repeated sequence of columns"
)
REFUSED_GROUPS_UNRESOLVED = "the table's column groups could not be placed against its columns"
REFUSED_NO_RANGE_GROUP = "the table prints no range group, so it can witness nothing"
REFUSED_NO_DATA_ROWS = "no row of this table binds to the columns its own header declares"

#: Floating-point slack on a containment test. The releases print one decimal,
#: so this is far below anything printed and exists only so 2.0 <= 2.0 holds
#: after a float round-trip.
_EPSILON = 1e-9

#: Row-level rejection reasons. A row carrying one of these yields NO witness,
#: while the rest of the table still does.
REJECTED_UNBOUND_CELL = "a value is printed in no column of this table"
REJECTED_TWO_CELLS_IN_A_COLUMN = "two values are printed in one column"
REJECTED_NO_LABEL = "the row carries no label beside its values"
REJECTED_CENTRAL_OUTSIDE_RANGE = "central tendency is not inside the range"
REJECTED_MEDIAN_OUTSIDE_RANGE = "the median is not inside the range"
REJECTED_MISSING_GROUP = (
    "a year prints a range but not every group this table declares, so a cell of that "
    "row was lost rather than never printed"
)
REJECTED_ORPHAN_PRIOR_ROW = (
    "a previous-meeting sub-row whose own variable row did not survive its checks"
)
REJECTED_MEDIAN_OUTSIDE_CENTRAL = "median is not inside the central tendency"

#: ``gate_series`` refusal reasons.
GATE_PASSED = ""
GATE_NO_FILLED_BINS = "the series has no filled bin to check"
GATE_UNPARSEABLE_BIN = "a bin label could not be read as a numeric interval"
GATE_OUT_OF_RANGE = "the filled bins do not span the published range"

#: How much of a cell's own lo-to-hi span a cell may sit away from its column's
#: centre and still be that column's cell.
#:
#: The lo-to-hi centre distance is exactly how far a cell's centre moves if one
#: of its two figures is dropped, so HALF of it is the decision boundary between
#: "printed centred in this column" and "shifted by a whole figure", and a
#: QUARTER sits inside that boundary with margin on each side. Measured on the
#: three releases #1184 names (``leads/1184/probes/feasibility2.txt``): the
#: separation is 20.20-22.20pt, so the tolerance is 5.05-5.55pt; a real cell
#: sits at most 1.10pt from its column, and a dropped figure moves one 10.10pt.
#: Five times inside, two times outside. This is the whole reason a lost figure
#: is REFUSED rather than quietly becoming an empty column.
_TOLERANCE_FRACTION_OF_CELL_SPAN = 0.25


@dataclass(frozen=True)
class _Column:
    """One printed column of the table: where it is, and what the header calls it."""

    centre: float
    label: str
    group: str = ""


@dataclass(frozen=True)
class _Cell:
    """One value printed in a band: a range, a lone figure, or a declared hole."""

    centre: float
    u0: float
    u1: float
    lo: float | None
    hi: float | None
    span: float | None
    first_token: int

    @property
    def is_hole(self) -> bool:
        return self.lo is None


@dataclass(frozen=True)
class WitnessTable:
    """Published ranges that passed their own self-check, plus what was refused.

    ``rows`` maps ``(variable, series_row, year)`` to ``(lo, hi)``.
    ``series_row`` is ``"current"`` for the variable's own row and ``"prior"``
    for its ``"<Month> projection"`` sub-row -- the previous meeting's figures,
    which a series must be matched to by ITS OWN month, never by position.

    ``variable`` is the row label AS THE BAND PRINTS IT, normalised. 2015-09-17
    wraps "Core PCE inflation4" so that its data band reads only "Core PCE";
    nothing here reassembles it, and :meth:`variable_for` is how a caller with
    the full name finds the key.
    """

    rows: dict[tuple[str, str, str], tuple[float, float]] = field(default_factory=dict)
    rejected: tuple[tuple[str, str, str], ...] = ()
    prior_months: frozenset[str] = frozenset()
    layout: str = ""
    refusal: str = ""

    def range_for(self, variable: str, series_row: str, year: str) -> tuple[float, float] | None:
        key = self.variable_for(variable)
        if key is None:
            return None
        return self.rows.get((key, series_row, year))

    def variable_for(self, name: str) -> str | None:
        """The table's own key for *name*, or None when it is not unambiguous.

        The caller holds a full variable name ("Core PCE inflation") and the
        table may hold a truncated one ("core pce"), so the caller's name is
        tested for STARTING WITH the key, not the reverse. Two keys matching is
        an ambiguity, and an ambiguous variable gets no witness rather than the
        first of them.
        """
        wanted = _normalise_label(name)
        if not wanted:
            return None
        keys = {k for k, _, _ in self.rows}
        if wanted in keys:
            return wanted
        hits = sorted(k for k in keys if k and wanted.startswith(k))
        return hits[0] if len(hits) == 1 else None

    def series_row_for(self, series_name: str) -> str:
        """Which row of the table a series belongs to, by its OWN month.

        Every series on these figures is month-prefixed ("September
        projections", "April projections"), so testing merely for a month name
        classes all of them as the prior meeting's -- a bug that checked every
        current-meeting series against the previous meeting's range during
        #1155. A series is the prior meeting's only when its month is one the
        table's own sub-rows name.
        """
        head = re.match(r"([A-Z][a-z]+)\b", series_name or "")
        return "prior" if (head and head.group(1) in self.prior_months) else "current"


def _normalise_label(text: str) -> str:
    """A row label reduced to its identity.

    NFKD so 2015-09-17's "inﬂation" ligature folds to "inflation"; trailing dot
    leaders and footnote digits dropped; case folded. What survives is what two
    releases printing the same variable differently still agree on.
    """
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = re.sub(r"\s+", " ", folded).strip()
    while True:
        trimmed = _LABEL_NOISE_RE.sub("", folded).strip()
        if trimmed == folded:
            return folded.casefold()
        folded = trimmed


def _bands(page) -> tuple[int, dict[float, list[tuple[float, float, str]]]]:
    """The page's words in the table's own reading frame, banded by row.

    ``u`` runs along a row and ``v`` across rows. The rotation comes from
    ``born_digital.upright_rotation_for``, which reads the direction the page
    itself records: ``page.rotation`` is 0 on 2015-09-17 because the release
    draws rotated CONTENT inside an unrotated page, and "a word box taller than
    it is wide" fails on exactly this table, because a rotated ONE-CHARACTER
    token -- most of a numeric table -- is wider than it is tall.
    """
    try:
        from socr.core.born_digital import upright_rotation_for

        rotation = int(upright_rotation_for(page)) % 360
    except Exception:
        rotation = 0
    try:
        words = page.get_text("words")
    except Exception:
        return rotation, {}
    bands: dict[float, list[tuple[float, float, str]]] = {}
    for x0, y0, x1, y1, text, *_ in words:
        if rotation == 0:
            v, u0, u1 = y0, x0, x1
        elif rotation == 90:
            v, u0, u1 = x0, -y1, -y0
        elif rotation == 180:
            v, u0, u1 = -y1, -x1, -x0
        elif rotation == 270:
            v, u0, u1 = -x1, y0, y1
        else:
            return rotation, {}
        bands.setdefault(round(v, 1), []).append((u0, u1, text))
    return rotation, {v: sorted(toks) for v, toks in bands.items()}


def _cells_of(tokens: list[tuple[float, float, str]]) -> list[_Cell]:
    """The value cells of one band, left to right.

    A cell is three consecutive tokens ``figure separator figure``, a lone
    figure read as the degenerate range ``[x, x]`` (the releases print a
    collapsed range as one figure, and dropping those loses whole variables), or
    a declared hole. Consecutiveness is what stops a range being paired across
    an empty column.
    """
    out: list[_Cell] = []
    index = 0
    while index < len(tokens):
        u0, u1, text = tokens[index]
        if _VALUE_RE.match(text):
            paired = (
                index + 2 < len(tokens)
                and tokens[index + 1][2] in _SEP_TOKENS
                and _VALUE_RE.match(tokens[index + 2][2])
            )
            if paired:
                b0, b1, btext = tokens[index + 2]
                lo_centre, hi_centre = (u0 + u1) / 2, (b0 + b1) / 2
                out.append(
                    _Cell(
                        centre=(lo_centre + hi_centre) / 2,
                        u0=u0,
                        u1=b1,
                        lo=_as_float(text),
                        hi=_as_float(btext),
                        span=hi_centre - lo_centre,
                        first_token=index,
                    )
                )
                index += 3
                continue
            value = _as_float(text)
            out.append(
                _Cell(
                    centre=(u0 + u1) / 2,
                    u0=u0,
                    u1=u1,
                    lo=value,
                    hi=value,
                    span=None,
                    first_token=index,
                )
            )
        elif _HOLE_RE.match(text):
            out.append(
                _Cell(
                    centre=(u0 + u1) / 2,
                    u0=u0,
                    u1=u1,
                    lo=None,
                    hi=None,
                    span=None,
                    first_token=index,
                )
            )
        index += 1
    return out


def _group_words_of(tokens: list[tuple[float, float, str]]):
    """``(centre, group)`` per token, or None when a token is not a group word.

    A band qualifies only when EVERY token is a group word. The prose and the
    footnotes of these releases also say "central tendency" and "range" --
    2010-04-28 p11 prints "had a central tendency of 3.2 to 3.7 percent" and
    "1. The central tendency excludes ..." -- and admitting those would put a
    group boundary in the middle of the page's text.
    """
    if not tokens:
        return None
    out: list[tuple[float, str]] = []
    for u0, u1, text in tokens:
        word = _normalise_label(text)
        if word not in _GROUP_WORDS:
            return None
        out.append(((u0 + u1) / 2, _GROUP_WORDS[word]))
    return out


def _period_of(names: list[str]) -> tuple[int, int]:
    """``(period length, repetition count)`` of a repeating label sequence."""
    total = len(names)
    for size in range(1, total + 1):
        if total % size == 0 and all(names[i] == names[i % size] for i in range(total)):
            return size, total // size
    return total, 1


def _resolve_columns(
    bands: dict[float, list[tuple[float, float, str]]],
    header_v: float,
    first_data_v: float,
    group_band_vs: frozenset[float],
) -> tuple[list[_Column], int, str]:
    """The table's printed columns, read from the table's own header.

    Columns come from the HEADER, never from clustering the data. A grid built
    from the data is circular -- every data row seeds it, so every cell lies
    inside its own cluster's spread by identity and a stray figure is admitted
    by the tolerance it widened (Argon, #1184 design review P0-1). Rebuilt
    unsupervised on the real releases, a data-clustered grid gives a
    within-column spread of 38.01pt against a 36.49pt column gap on 2008-10-29
    -- no grid at all, because that page draws a prose column beside the table.

    Returns ``(columns, repetitions, refusal)``.
    """
    years = sorted((u0 + u1) / 2 for u0, u1, text in bands[header_v] if _YEAR_RE.match(text))
    named = sorted(
        ((u0 + u1) / 2, text) for u0, u1, text in bands[header_v] if _YEAR_RE.match(text)
    )
    period, repetitions = _period_of([name for _, name in named])
    gaps = [years[i + 1] - years[i] for i in range(len(years) - 1)]
    if not gaps:
        return [], 0, REFUSED_COLUMNS_UNRESOLVED
    # The pitch of the LAST repetition, not of the first. The groups are not
    # printed at one width: on 2015-09-17 the median group's columns hold a
    # single figure and sit 26.4pt apart, while the range group's hold a pair
    # and sit 41.4pt apart. Bounding the trailing column with the first group's
    # pitch cuts the last "Longer run" off the table.
    tail = gaps[(repetitions - 1) * period :][: period - 1]
    pitch = statistics.median(tail or gaps)

    # A column the header names with something other than a year ("Longer run").
    # Its tokens come from the header band and the bands between it and the
    # first data band, minus any band that names a group. Bounded on the left by
    # the first year column -- what sits left of the first value column is the
    # row-label heading ("Variable"), not a column -- and on the right by one
    # pitch for the column itself plus half a pitch for a token's own width,
    # which is what keeps the "Federal Open Market Committee" printed down the
    # edge of 2015-09-17 p13 out of the trailing "Longer run" column. A figure
    # is never a column's name either, so a stray value printed above the first
    # data row cannot invent a column.
    right_bound = years[-1] + 1.5 * pitch
    extra = [
        (u0, u1, text)
        for v, tokens in bands.items()
        if v not in group_band_vs and header_v <= v < first_data_v
        for u0, u1, text in tokens
        if not _YEAR_RE.match(text)
        and not _VALUE_RE.match(text)
        and years[0] < (u0 + u1) / 2 <= right_bound
    ]

    columns = [_Column(centre=centre, label=name) for centre, name in named]
    edges = [*years, float("inf")]
    for index in range(len(edges) - 1):
        lo, hi = edges[index], edges[index + 1]
        here = [t for t in extra if lo < (t[0] + t[1]) / 2 <= hi]
        if not here:
            continue
        ordered: dict[str, None] = {}
        for _, _, text in sorted(here):
            ordered.setdefault(text, None)
        columns.append(
            _Column(
                centre=(min(t[0] for t in here) + max(t[1] for t in here)) / 2,
                label=" ".join(ordered),
            )
        )
    columns.sort(key=lambda column: column.centre)

    size, found = _period_of([column.label for column in columns])
    if found != repetitions or size * found != len(columns):
        return [], 0, REFUSED_COLUMNS_UNRESOLVED
    return columns, repetitions, ""


def _place_groups(
    columns: list[_Column],
    repetitions: int,
    group_tokens: list[tuple[float, str]],
) -> tuple[list[_Column], str]:
    """Attach each column to its group, by where the group headers are printed.

    One mechanism, two placements. When the year sequence repeats, the groups
    are printed SIDE BY SIDE and every group word falls inside the u-span of the
    repetition it heads (2010-04-28, 2015-09-17). When it does not repeat, the
    groups are STACKED BLOCKS over one shared grid, so the group belongs to the
    band rather than to the column (2008-10-29) and is attached there.

    Revision 1 of the design gave each column to the NEAREST group header along
    u. Argon measured that wrong on 2015-09-17: the central tendency's 2015
    column sits 89.65pt from "Median1" and 95.55pt from "tendency2", so
    nearest-header hands it to the median group and the whole table refuses.

    Returns ``(columns, refusal)``; an empty group on a column means the band
    carries it.
    """
    if repetitions <= 1:
        return columns, ""
    size = len(columns) // repetitions
    placed: list[_Column] = []
    for index in range(repetitions):
        block = columns[index * size : (index + 1) * size]
        lo = block[0].centre
        hi = block[-1].centre
        # The block owns everything from its own first column to its last; a
        # group word printed centred over the block falls inside that span.
        names = {name for centre, name in group_tokens if lo <= centre <= hi}
        if len(names) != 1:
            return columns, REFUSED_GROUPS_UNRESOLVED
        group = names.pop()
        placed.extend(
            _Column(centre=column.centre, label=column.label, group=group) for column in block
        )
    return placed, ""


def _row_label(tokens: list[tuple[float, float, str]], first_cell_token: int, cut: float) -> str:
    """The label printed beside this band's values.

    Walked LEFT from the first value, stopping at the first gap wider than
    *cut*. A row's label is not the start of its band: 2008-10-29 p12 draws
    Table 1 beside a prose column, so the band holding "June projection. . . 1.0
    to 1.6 ..." BEGINS "unemployment, and inflation in 2008, 2009, 2010, and"
    (``probes/dump_layout.txt`` line 797). Taking the band's leading tokens
    there would bind the previous meeting's figures under a label read out of a
    sentence (Argon, #1184 design review P0-3.1).
    """
    if first_cell_token <= 0:
        return ""
    run = [tokens[first_cell_token - 1][2]]
    for index in range(first_cell_token - 2, -1, -1):
        if tokens[index + 1][0] - tokens[index][1] > cut:
            break
        run.append(tokens[index][2])
    return " ".join(reversed(run))


def parse_table1_witness(page) -> WitnessTable:
    """Parse the projection table on *page* into self-checked witness ranges.

    Returns a ``WitnessTable`` whose ``rows`` are only those that passed the
    self-check. A layout this function cannot resolve returns an EMPTY table
    carrying a named ``refusal``; the caller must withhold rather than proceed.
    Never raises.

    The self-check is per-COLUMN, not per-count (#1184, orchestrator ruling):
    every value a row prints must bind to exactly one of the table's own
    columns, and no column may receive two. The count rule it replaces -- the
    row's value total equalling the header's column total -- cannot tell a cell
    the table never printed from one this parser lost, so it rejected both, and
    with them 2010-04-28's core PCE rows (longer-run core PCE is not collected,
    the release's own footnote 3) and every row of 2008-10-29 (its groups are
    stacked, so a row legitimately carries half the columns).

    Binding is stronger, not weaker, on exactly the case the count rule caught:
    a cell that lost one of its two figures is no longer printed centred in its
    column, it is a whole figure off, so it binds to nothing and the row is
    REJECTED. Measured on the three releases: a real cell sits at most 1.10pt
    from its column against a 5.05-5.55pt tolerance, and a dropped figure moves
    one 10.10pt.
    """
    rotation, bands = _bands(page)
    if rotation not in (0, 90, 180, 270):
        return WitnessTable(refusal=REFUSED_UNSUPPORTED_ROTATION)
    if not bands:
        return WitnessTable(refusal=REFUSED_NO_TEXT)

    cells_by_band = {v: _cells_of(tokens) for v, tokens in bands.items()}
    spans = [c.span for cells in cells_by_band.values() for c in cells if c.span is not None]
    if not spans:
        return WitnessTable(refusal=REFUSED_NO_SCALE)
    tolerance = statistics.median(spans) * _TOLERANCE_FRACTION_OF_CELL_SPAN

    group_tokens: list[tuple[float, str]] = []
    group_band_vs: set[float] = set()
    block_names: dict[float, str] = {}
    for v, tokens in bands.items():
        words = _group_words_of(tokens)
        if words is None:
            continue
        group_band_vs.add(v)
        group_tokens.extend(words)
        names = {name for _, name in words}
        if len(names) == 1:
            block_names[v] = names.pop()

    header_v = max(
        bands,
        key=lambda v: sum(1 for _, _, text in bands[v] if _YEAR_RE.match(text)),
    )
    year_centres = [(u0 + u1) / 2 for u0, u1, text in bands[header_v] if _YEAR_RE.match(text)]
    if len(year_centres) < 2:
        return WitnessTable(refusal=REFUSED_NO_HEADER)

    first_data_v = None
    for v in sorted(x for x in bands if x > header_v):
        if v in group_band_vs:
            continue
        values = [c for c in cells_by_band[v] if not c.is_hole]
        on_grid = sum(
            1 for c in values if min(abs(c.centre - y) for y in year_centres) <= tolerance
        )
        if len(values) >= 2 and on_grid >= 2:
            first_data_v = v
            break
    if first_data_v is None:
        return WitnessTable(refusal=REFUSED_NO_DATA_ROWS)

    columns, repetitions, refusal = _resolve_columns(
        bands, header_v, first_data_v, frozenset(group_band_vs)
    )
    if refusal:
        return WitnessTable(refusal=refusal)
    columns, refusal = _place_groups(columns, repetitions, group_tokens)
    if refusal:
        return WitnessTable(refusal=refusal)

    layout = LAYOUT_SIDE_BY_SIDE_GROUPS if repetitions > 1 else LAYOUT_STACKED_BLOCKS
    declared_groups = (
        {column.group for column in columns} if repetitions > 1 else set(block_names.values())
    )
    if GROUP_RANGE not in declared_groups:
        return WitnessTable(refusal=REFUSED_NO_RANGE_GROUP)

    centres = [column.centre for column in columns]
    collected: dict[tuple[str, str], dict[str, dict[str, tuple[float, float]]]] = {}
    rejected: list[tuple[str, str, str]] = []
    prior_months: set[str] = set()
    current_variable: str | None = None

    for v in sorted(x for x in bands if x >= first_data_v):
        if v in group_band_vs:
            continue
        values = [c for c in cells_by_band[v] if not c.is_hole]
        if len(values) < 2:
            continue
        tokens = bands[v]
        printed = _row_label(tokens, min(c.first_token for c in cells_by_band[v]), tolerance)
        prior = _PRIOR_ROW_RE.match(printed.strip())
        if prior is not None:
            # A sub-row names a MEETING, not a variable, so it only means
            # anything under the variable row above it. With that row rejected
            # there is nothing to attach it to: publishing it under the last
            # variable that did survive puts one variable's figures under
            # another's key (the reviewer measured unemployment's prior row
            # returning GDP's range), and publishing it under its own printed
            # label invents a variable called "january projection".
            if current_variable is None:
                rejected.append((_normalise_label(printed), "prior", REJECTED_ORPHAN_PRIOR_ROW))
                continue
            variable, series_row = current_variable, "prior"
        else:
            variable, series_row = _normalise_label(printed), "current"

        if not variable:
            rejected.append((_normalise_label(printed), series_row, REJECTED_NO_LABEL))
            current_variable = None
            continue

        bound: dict[int, _Cell] = {}
        failure = ""
        # Holes are bound along with the figures. They carry no value, but they
        # occupy a column, so an "n/a" printed where no column is -- or in a
        # column that already holds a figure -- is refused rather than dropped.
        # Before, both callers filtered holes out before binding and the branch
        # that reads them was dead code (PR #1195 review).
        for cell in cells_by_band[v]:
            index = min(range(len(columns)), key=lambda i: abs(cell.centre - centres[i]))
            if abs(cell.centre - centres[index]) > tolerance or not (
                cell.u0 <= centres[index] <= cell.u1
            ):
                failure = REJECTED_UNBOUND_CELL
                break
            if index in bound:
                failure = REJECTED_TWO_CELLS_IN_A_COLUMN
                break
            bound[index] = cell
        if failure:
            rejected.append((variable, series_row, failure))
            # A rejected variable row must not lend its name to the sub-row
            # below it, or the previous meeting's figures bind under it.
            current_variable = None
            continue

        if series_row == "prior":
            prior_months.add(prior.group(1))
        else:
            current_variable = variable

        band_group = ""
        if repetitions <= 1:
            above = [bv for bv in block_names if bv < v]
            band_group = block_names[max(above)] if above else ""
        by_year = collected.setdefault((variable, series_row), {})
        for index, cell in bound.items():
            group = columns[index].group or band_group
            if not group or cell.is_hole:
                continue
            by_year.setdefault(columns[index].label, {})[group] = (cell.lo, cell.hi)

    found: dict[tuple[str, str, str], tuple[float, float]] = {}
    for (variable, series_row), by_year in collected.items():
        broken = ""
        for groups in by_year.values():
            full = groups.get(GROUP_RANGE)
            central = groups.get(GROUP_CENTRAL)
            median = groups.get(GROUP_MEDIAN)
            # A year that prints a range but not every group the table declares
            # has LOST a cell. A column the release never prints is blank in
            # every group -- longer-run core PCE, which these releases do not
            # collect, prints no median, no central tendency and no range -- so
            # a gap in one group only cannot be that. Without this the central
            # cell could go missing and the range ship with nothing above it to
            # check against (PR #1195 review, blocker 1).
            if full and declared_groups - set(groups):
                broken = REJECTED_MISSING_GROUP
                break
            if (
                full
                and central
                and not (full[0] - _EPSILON <= central[0] and central[1] <= full[1] + _EPSILON)
            ):
                broken = REJECTED_CENTRAL_OUTSIDE_RANGE
                break
            # Checked against the median DIRECTLY, not only through the central
            # tendency: chaining the two leaves the range unchecked whenever the
            # middle link is absent, which is how "4.1 to 4.6" shipped on a row
            # whose own median was 3.5.
            if (
                full
                and median
                and not (full[0] - _EPSILON <= median[0] and median[1] <= full[1] + _EPSILON)
            ):
                broken = REJECTED_MEDIAN_OUTSIDE_RANGE
                break
            if (
                central
                and median
                and not (central[0] - _EPSILON <= median[0] <= central[1] + _EPSILON)
            ):
                broken = REJECTED_MEDIAN_OUTSIDE_CENTRAL
                break
        if broken:
            rejected.append((variable, series_row, broken))
            continue
        for year, groups in by_year.items():
            if GROUP_RANGE in groups:
                found[(variable, series_row, year)] = groups[GROUP_RANGE]

    return WitnessTable(
        rows=found,
        rejected=tuple(rejected),
        prior_months=frozenset(prior_months),
        layout=layout,
    )


def _as_float(token: str) -> float:
    """A figure's value, with the Unicode MINUS (U+2212) normalised to ASCII."""
    return float(token.replace("\u2212", "-"))


def bin_interval(bin_label: str) -> tuple[float, float] | None:
    """A bin label's own numeric interval, or None when it carries no figures.

    A hyphen in these labels is BOTH the separator and a possible sign, so the
    label is matched as a RANGE rather than scanned for figures: scanning
    "1.8-1.9" with a sign-bearing pattern finds ``1.8`` and ``-1.9``, which is
    the same silent corruption in the opposite direction from the unsigned
    pattern that preceded it (that one read "\u22120.37-\u22120.13" as
    ``(0.37, 0.13)``, low above high). The range pattern knows where the
    separator is and keeps each figure's own sign.
    """
    text = bin_label or ""
    match = _RANGE_RE.search(text)
    if match:
        return (_as_float(match.group(1)), _as_float(match.group(2)))
    single = re.search(_NUM, text)
    if not single:
        return None
    value = _as_float(single.group(0))
    return (value, value)


def gate_series(filled_bin_labels, published_range) -> tuple[bool, str]:
    """Whether a series' filled bins are consistent with its published range.

    The check: the LOWEST filled bin must contain the published minimum and the
    HIGHEST filled bin must contain the published maximum. That is the sharpest
    form the witness supports -- the range is by definition the lowest and
    highest projection any participant made, so the bins holding them are
    determined, and shifting the reading by one bin in either direction moves
    at least one of them off its endpoint.

    Its limit, stated because it is real and was measured rather than argued:
    a shift that leaves both extremes inside their own bins still passes. This
    gate narrows what can ship; it does not prove a reading correct.

    ``filled_bin_labels`` is the labels of the bins the reader resolved to a
    count above zero, in bin order.
    """
    if not filled_bin_labels:
        return False, GATE_NO_FILLED_BINS
    low = bin_interval(filled_bin_labels[0])
    high = bin_interval(filled_bin_labels[-1])
    if low is None or high is None:
        return False, GATE_UNPARSEABLE_BIN
    published_low, published_high = published_range
    if not (low[0] - 1e-9 <= published_low <= low[1] + 1e-9):
        return False, GATE_OUT_OF_RANGE
    if not (high[0] - 1e-9 <= published_high <= high[1] + 1e-9):
        return False, GATE_OUT_OF_RANGE
    return True, GATE_PASSED


#: What a panel line says when the numbers DO ship. The owner's standing rule is that a
#: model reading ships "(OCR, not verified)" and a mark never claims a value was verified
#: (#1181). A gated chart reading is in the same position: the gate NARROWS what can ship,
#: it does not prove a reading correct, and only the two extreme bars were compared at
#: all. Saying so where the reader is looking is the point -- the limitation is in this
#: module's docstring too, but a docstring is not read by whoever cites the number.
SHIPPED_REASON = (
    "counts read from the drawing, not verified; only the lowest and highest filled bars "
    "were checked against this document's own projection-table range"
)


def panel_mark_line(
    page_num: int,
    region_index: int,
    region_count: int,
    reason: str,
    *,
    shipped: bool,
    decided: bool = True,
) -> str:
    """One canonical #1145 line for one chart region of a page.

    ``region_index`` is 1-based in the order ``chart_region_bboxes`` returns, which is
    the page's DRAWING order, not a geometric one. That is the same index the region's
    crop is named with (``chart_region_p{page}_{index}.png``), so a reader can always
    match this line to the picture it is about. No claim is made here about where the
    region sits on the page, because nothing sorts that list.

    ``decided`` is False when the check could not run or could not decide (no witness,
    nothing resolved); it selects UNKNOWN over FAIL, the distinction #1145 draws.
    """
    from socr.core.marks import Check, CheckResult, page_unchecked_clause_mark, region_scope

    return page_unchecked_clause_mark(
        page_num,
        Check.CHART_REGION,
        CheckResult.UNKNOWN if (shipped or not decided) else CheckResult.FAIL,
        region_scope("chart", region_index),
        f"region {region_index} of {region_count}: {reason}",
    )


#: What a region's line says when its picture could not be drawn. The page still ships its
#: prose and its other regions; a render failure must never cost the page its text, and
#: must never pass silently either.
PICTURE_NOT_RENDERED = "picture not rendered ({error}); values not shipped"
#: What a region's line says in this version, where the gate is built but no count is
#: wired to an output path yet. Distinct from a gate REFUSAL on purpose: saying the
#: numbers were judged and rejected, when nothing judged them, would be a false record.
COUNTS_NOT_WIRED = (
    "the reader resolved this panel, but counts are not shipped in this version; "
    "the witness gate is not yet wired to an output path"
)


def region_clause_records(
    page_num: int, *, region_pictures: dict, reading, region_count: int
) -> tuple[dict, ...]:
    """#1155 round 3: one MACHINE-READABLE clause per chart region, from the results.

    The first version of this lane wrote a per-region line into the page and recorded
    nothing: all 12 finalized verdicts carried zero ``chart_region`` clauses while the
    Markdown showed a mark for every region (#1155 PR 1 re-review, Astra and Argon both).
    A rendered mark string is not a record -- nothing downstream parses the page back --
    so the per-region outcome is built HERE, as the clause dicts
    ``core.verdict.VerdictClause.from_dict`` reads, and the page's lines are rendered
    FROM those same records by :func:`region_clause_mark`. One source, so the line a
    reader sees and the clause a consumer reads cannot drift apart.

    ``result`` is UNKNOWN when the check could not run or could not decide (no picture,
    nothing resolved, counts not wired) and FAIL only where a region's own outcome was
    established -- the distinction #1145 draws, applied per region rather than per page.
    """
    from socr.core.marks import Check, CheckResult, region_scope

    records: list[dict] = []
    for index in range(1, region_count + 1):
        picture = region_pictures.get(index)
        panel = reading.panels.get(index) if getattr(reading, "panels", None) else None
        if not picture:
            reason = PICTURE_NOT_RENDERED.format(
                error=region_pictures.get(f"error:{index}", "no picture was produced")
            )
            decided = False
        elif panel is None:
            reason = (
                "the reader resolved no panel here"
                f" ({reading.refusals.get(index, 'no reason recorded')}); values not shipped"
            )
            decided = False
        elif not panel.label:
            reason = (
                "the panel carries no printed label, so its values cannot be bound to a "
                "year; values not shipped"
            )
            decided = True
        else:
            reason = f"panel {panel.label}: {COUNTS_NOT_WIRED}"
            decided = False
        records.append(
            {
                "check": Check.CHART_REGION.value,
                "result": (CheckResult.FAIL if decided else CheckResult.UNKNOWN).value,
                "scope": region_scope("chart", index),
                "reason": f"region {index} of {region_count}: {reason}",
            }
        )
    return tuple(records)


def clause_mark(page_num: int, record: dict) -> str:
    """The canonical line one clause record of this lane states.

    Round 4: every mark this lane writes is rendered through here, from the record that
    is also handed to the verdict -- the per-region records above and the two
    page-scoped ones below. A mark rendered from anything else could say something the
    record does not, which is the defect round 3 fixed for regions and round 4 for the
    page-scoped lines (Astra P2).
    """
    from socr.core.marks import page_unchecked_clause_mark

    return page_unchecked_clause_mark(
        page_num,
        record["check"],
        record["result"],
        record["scope"],
        record["reason"],
    )


#: What one line says when the reading-order gate refused this page's recovered prose.
#: #1144 round 5 holds back text whose bands are not in the page's own reading order, and
#: that refusal used to be SILENT here: the builder set ``recovered = ""`` and the page
#: shipped crops with no statement that its text had been withheld and -- because the
#: full-page image was embedded in the string being discarded -- no way to read it either
#: (#1155 PR 1 re-review, Astra).
ORDER_REFUSED_REASON = (
    "this page's text was withheld whole: the bands recovered from it are not in the "
    "page's own reading order (#1144), and words in the wrong order rank below missing "
    "ones; read the page's text from the page image"
)


def _page_coverage_record(reason: str) -> dict:
    """One page-scoped ``native_coverage=FAIL`` record: text of the page did not ship."""
    from socr.core.marks import SCOPE_PAGE, Check, CheckResult

    return {
        "check": Check.NATIVE_COVERAGE.value,
        "result": CheckResult.FAIL.value,
        "scope": SCOPE_PAGE,
        "reason": reason,
    }


def order_refused_record() -> dict:
    """The record for a reading-order refusal of the whole page's prose.

    Round 4 (Astra P2): round 3 produced only a STRING here, so the page said
    ``native_coverage=FAIL`` in its Markdown while its terminal JSON carried no such
    clause -- a machine-readable audit could not tell that the page's whole prose had
    been withheld for ordering. The record is now what exists, and the line is rendered
    from it (:func:`clause_mark`).
    """
    return _page_coverage_record(ORDER_REFUSED_REASON)


def floor_replacement_body(
    page,
    page_num: int,
    recovered_prose: str,
    *,
    region_pictures: dict,
    reading,
    region_count: int,
    extra_marks: list | None = None,
    page_image: str | None = None,
    region_records: tuple[dict, ...] | list[dict] | None = None,
) -> str | None:
    """#1155: what a chart page ships INSTEAD of a bare whole-page failure marker.

    The structure-class floor withholds a page whole when it finds no usable table
    box to splice prose around. On a page that is a CHART, that withheld everything
    -- prose, pictures and the chart reading alike -- and 9 Fed SEP pages shipped
    103 characters of marker and nothing else.

    ``recovered_prose`` is what #1144's own lane positively identified as prose
    OUTSIDE every table box AND every chart region; this function never sees the
    page's raw text. The first version of it did, and shipped the chart's own
    y-axis ticks and shredded bin labels as loose numbers beside a mark saying
    values were not shipped -- 149 numeric lines on one page (#1155 PR 1 review,
    Astra 2 / Argon 2). Nothing inside a chart region can reach the page now,
    because the geometry withheld it before this function was called.

    ``region_pictures`` maps a region index to a picture reference ALREADY
    rendered, and only references not already present in ``recovered_prose`` are
    appended: the native layer embeds its own inline region placeholders, and
    appending unconditionally emitted ten links for five regions.

    ``region_records`` are the per-region clause records the caller has already built
    and RECORDED on the page (``region_clause_records``); the lines below are rendered
    from them, so the page's text and the page's verdict state the same thing by
    construction. Omitted, they are computed here, which is what the probes and the
    isolated tests do -- the bytes are identical either way.

    Returns ``None`` when the reading resolves nothing, leaving the floor's
    existing behaviour untouched: this may only ever ADD to a page that was
    going to ship nothing.
    """
    if not getattr(reading, "panels", None):
        return None

    records = (
        tuple(region_records)
        if region_records is not None
        else region_clause_records(
            page_num,
            region_pictures=region_pictures,
            reading=reading,
            region_count=region_count,
        )
    )
    lines = [clause_mark(page_num, record) for record in records]
    pictures: list[str] = []
    for index in range(1, region_count + 1):
        picture = region_pictures.get(index)
        if picture and picture not in (recovered_prose or ""):
            pictures.append(picture)

    parts = ["\n".join([*lines, *(extra_marks or [])])]
    prose = (recovered_prose or "").strip()
    if prose:
        parts.append(prose)
    parts.extend(pictures)
    # The page image ONCE, after the crops, when the lane withheld page text.
    if page_image and page_image not in (recovered_prose or ""):
        parts.append(page_image)
    return "\n\n".join(parts)


#: What one line says in place of the per-run failure markers the prose lane interleaves.
#: The lane marks EVERY withheld run, which on a Fed SEP page is 3-6 identical
#: "[page N failed: no usable OCR output]" blocks plus the page image each time, because
#: the digit-band rule withholds every figure title and note (they carry "2.A", "table 1").
#: That caption behaviour is #1154's and is NOT changed here; what is changed is saying it
#: once, in the canonical grammar, with the picture once.
WITHHELD_RUNS_REASON = (
    "{runs} run(s) of this page's text were withheld: the prose rule holds back any line "
    "carrying a figure, which on this page includes the figure titles and notes (#1154); "
    "read them from the page image"
)


def collapse_withheld_runs(recovered: str, marker_line: str, png_ref: str, page_num: int):
    """Replace the lane's repeated per-run markers with ONE canonical line.

    Returns ``(prose_without_markers, clause_record_or_None, image_or_None)``. The
    caller renders the record with :func:`clause_mark`, places that line with the
    per-region marks and carries the image once, so a page carries one statement of what
    was withheld instead of one per run (#1155 PR 1 re-review, Argon 4). Round 4 changed
    the middle element from the rendered LINE to the record it is rendered from, so the
    same statement reaches the page and the verdict.
    """
    text = recovered or ""
    runs = text.count(marker_line) if marker_line else 0
    if marker_line:
        text = text.replace(marker_line, "")
    had_image = bool(png_ref) and png_ref in text
    if png_ref:
        text = text.replace(png_ref, "")
    # The lane writes its own "[page image](...)" beside each withheld run, which is a
    # different spelling from the caller's ``png_ref``. Both have to go, or the page
    # carries one picture link per withheld run (4 on 2010-04-28 p15) on top of the
    # single image appended at the end.
    lane_images = re.findall(r"\[page image\]\([^)]*\)", text)
    for ref in set(lane_images):
        text = text.replace(ref, "")
    had_image = had_image or bool(lane_images)
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if not line.strip() and (not cleaned or not cleaned[-1].strip()):
            continue
        cleaned.append(line)
    # PAGE scope and the native-coverage check: this is not about a chart region, it
    # is about text of the page that did not ship. A region-scoped chart_region clause
    # here would impugn a region that is in fact represented.
    #
    # Round 4: a RECORD, not a rendered line. Astra's P2 was that the order-refusal line
    # existed only as a string; this line had the same gap and fires on all 12 real Fed
    # pages where that one fires on none, so fixing only the one that was named would
    # leave the commoner case unrecorded. The caller renders it with ``clause_mark`` and
    # hands the same record to the verdict.
    record = _page_coverage_record(WITHHELD_RUNS_REASON.format(runs=runs)) if runs else None
    return (
        "\n".join(cleaned).strip(),
        record,
        (png_ref if (had_image or runs) and png_ref else None),
    )
