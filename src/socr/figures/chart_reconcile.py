"""#734 Stage A: reconcile a model's FILLED chart grid against read geometry.

#635 Stage 0 withholds a chart region's EMPTY derivation, and Stage 1 reads the
region's vector drawings and publishes the counts it can prove. Between the two
sits the case this module answers: the model emitted a grid for the chart region
with NUMBERS in it. Nothing in the grid says whether those numbers were read off
the page or invented, and a filled grid is exactly what an extraction that
succeeded looks like -- so the page ships them unchecked, which is how
``| 0.13-0.37 | 17 | 17 |`` reached the corpus beside a panel whose geometry
derives ``17, 0, 0, ...``.

The rule is the design's (``docs/plans/chart-data/DESIGN.md``, Stage 2):

* reconcile **by cell identity** -- a grid cell and a reader cell are the same
  cell when the labels naming them are the same label, never when they merely
  occupy the same position. Position is the model's layout, and the model's
  layout is what is in question;
* **a contradicted model number is never published.** Geometry holding a
  different integer for that identity is a contradiction, and the cell publishes
  nothing at all;
* **competing counts are never averaged and totals never force agreement.**
  There is no arithmetic here: the only value a cell can support is one both
  sides independently arrived at, and it is the value itself, not a
  reconstruction of it;
* **geometry with no opinion is not a contradiction.** A reader cell that is
  UNRESOLVED, or absent because the reader found no such cell, says nothing
  about the model's number. The number survives as unverified -- neither
  corroborated nor impeached -- because refusing it would delete a reading on
  the strength of a refusal to read.

**Two things here are NOT the design's, and are called out so no reader takes
them for ratified:**

* **The orientation rule is this module's own.** DESIGN.md says to reconcile by
  cell identity and says nothing about which axis of a grid carries the bins.
  Deciding it by counting bin matches per axis is a choice made here, and it
  rests on a precondition the corpus happens to satisfy and no rule enforces:
  that a grid's OTHER axis does not also carry the panel's bin labels. Where it
  fails -- single-value bins that could head either axis -- the tie refuses the
  grid rather than transposing it, which is the safe direction, but it is a
  refusal caused by this rule and not by the page.
* **``published`` does not check what Stage 2 requires.** The design says
  agreement must satisfy calibration and constraints; ``published`` checks only
  that the two sides hold the same integer. The calibration behind the reader's
  number was already enforced when the reader emitted it (an unresolvable cell
  is UNRESOLVED, and geometry with no opinion never agrees), but the
  constraint half -- the caller's acceptance hook, ``chart_reader.verify_panel``
  -- is not consulted here at all. A cell can therefore be ``agreed`` inside a
  panel whose derivation a caller would reject.

Pure: no I/O, no state, no page. It takes a grid the caller already parsed and
the ``PanelReading`` for the region the caller already bound it to, and returns
a verdict per cell. Binding a grid to a region, and deciding what the page then
ships, are the caller's (Stage B); this module has no opinion on either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from socr.figures.chart_data import (
    _EMPH_RE,
    FilledGrid,
    _fold,
    _is_separator,
    _key_atoms,
    _split_row,
)
from socr.figures.chart_reader import INTEGER, Cell, PanelReading
from socr.tables.label_canonical import decode_label_cell

#: Cell outcomes.
#: The model's number and geometry's are the same number.
AGREED = "agreed"
#: Geometry holds a DIFFERENT integer for this identity.
CONTRADICTED = "contradicted"
#: Geometry has no number for this identity -- UNRESOLVED, or no such cell.
UNKNOWN_TO_GEOMETRY = "unknown_to_geometry"
#: The grid cell carries no integer, so there is no model number to check.
NOT_A_COUNT = "not_a_count"

#: WHY a cell is unknown to geometry. Four different failures wear the same
#: status, and only the cause tells them apart: geometry read the cell and
#: could not resolve it; geometry read this bin under other series but nothing
#: it read is named the way this column is; geometry read this series but no
#: bin of it is named the way this row is; or neither label meets anything.
#: The middle two are cells that a matching label WOULD have judged -- the
#: laundering surface -- and folding them into "geometry was silent" states
#: something false about the page.
CAUSE_READER_UNRESOLVED = "reader_unresolved"
CAUSE_CELL_ABSENT = "cell_absent"
CAUSE_SERIES_UNMATCHED = "series_unmatched"
CAUSE_BIN_UNMATCHED = "bin_unmatched"
CAUSE_NEITHER_MATCHED = "neither_matched"

#: Which axis of the grid carries the chart's bins. Decided by identity: the
#: axis whose labels name the reader's bins, never by the grid's shape.
BINS_IN_HEADER = "bins_in_header"
BINS_IN_COLUMN = "bins_in_column"

#: A count is a whole number of things counted. A cell carrying anything else
#: -- a percentage, a range, a socr marker, prose -- publishes no count, so
#: there is nothing for geometry to agree with or contradict.
_INT_RE = re.compile(r"^[+-]?\d+$")
#: Stage 0's keys and Stage 1's joined labels both set a range with no space
#: around its dash, because that is how the page prints a tick label. A model
#: writing the same bin as "0.13 – 0.37" has spaced it; closing the space is
#: what makes the two the SAME bin rather than an unmatched one, and it is the
#: page's own convention being applied, not a tolerance.
_SPACED_DASH_RE = re.compile(r"\s*[-\u2013\u2014\u2212]\s*")


@dataclass(frozen=True)
class CellVerdict:
    """One (bin, series) identity, as the grid has it and as geometry has it."""

    bin_label: str
    series_name: str
    status: str
    model_text: str
    model_count: int | None
    reader_count: int | None
    detail: str = ""
    #: Set only when ``status`` is UNKNOWN_TO_GEOMETRY; one of the CAUSE_*
    #: constants. Empty for every other status.
    cause: str = ""

    @property
    def published(self) -> int | None:
        """The count this verdict supports publishing, if any.

        Only agreement supports a number, and the number is the one BOTH sides
        hold. A contradiction publishes nothing -- not the model's value, not
        geometry's, and above all not a value derived from the two.
        """
        return self.model_count if self.status == AGREED else None

    def to_dict(self) -> dict:
        return {
            "bin_label": self.bin_label,
            "series_name": self.series_name,
            "status": self.status,
            "model_text": self.model_text,
            "model_count": self.model_count,
            "reader_count": self.reader_count,
            "cause": self.cause,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class UncoveredReading:
    """One identity GEOMETRY read that no cell of the grid addressed.

    The reader→model direction, which every other coverage field lacks. A grid
    can drop a whole series, a whole bin row, or part of one series across some
    bins, and the first two are the only ones an axis-level field can express
    at all -- so this is recorded per identity and any axis summary is derived
    from it, never computed beside it.

    ``reader_count`` is ``None`` when geometry read the cell and resolved no
    count. That distinction is the point: an unaddressed UNRESOLVED cell is
    nothing lost, while an unaddressed cell carrying a number is a reading that
    existed, was proven, and went nowhere.
    """

    series_name: str
    bin_label: str
    reader_count: int | None

    def to_dict(self) -> dict:
        return {
            "series_name": self.series_name,
            "bin_label": self.bin_label,
            "reader_count": self.reader_count,
        }


@dataclass(frozen=True)
class GridReconciliation:
    """One filled grid, checked cell by cell against one panel's geometry."""

    page_num: int
    region_index: int
    table_index: int
    orientation: str
    cells: tuple[CellVerdict, ...] = ()
    #: Labels the GRID names that geometry never read.
    unmatched_bins: tuple[str, ...] = ()
    unmatched_series: tuple[str, ...] = ()
    #: The inverse direction, at CELL granularity: identities geometry read
    #: that no grid cell addressed. A series-level field cannot express a
    #: dropped bin row or a partial drop, and a grid that drops half a chart
    #: while keeping real labels is otherwise a clean sheet -- every other
    #: field here runs model→reader and has nothing to say about it.
    uncovered: tuple[UncoveredReading, ...] = ()
    refusal: str = ""

    def _count(self, status: str) -> int:
        return sum(1 for c in self.cells if c.status == status)

    @property
    def agreed(self) -> int:
        return self._count(AGREED)

    @property
    def contradicted(self) -> int:
        return self._count(CONTRADICTED)

    @property
    def unknown(self) -> int:
        return self._count(UNKNOWN_TO_GEOMETRY)

    @property
    def not_a_count(self) -> int:
        return self._count(NOT_A_COUNT)

    @property
    def uncovered_count(self) -> int:
        """Reader identities no grid cell addressed."""
        return len(self.uncovered)

    @property
    def uncovered_with_count(self) -> int:
        """Those of them geometry actually resolved a number for."""
        return sum(1 for u in self.uncovered if u.reader_count is not None)

    @property
    def published_cells(self) -> int:
        """Cells whose verdict supports publishing a number."""
        return sum(1 for c in self.cells if c.published is not None)

    @property
    def uncovered_beside_published(self) -> int:
        """Uncovered readings WITH a number on a grid that publishes cells.

        The severity split, and it runs opposite to the volume. A grid that
        published nothing -- every column caption-headed, nothing matched --
        can leave a whole chart uncovered and still ship no number, so its
        uncovered readings are recall loss and nothing is fabricated. The
        dangerous shape is the small one: a grid whose cells DID agree and
        publish, beside geometry that was never consulted. That page ships a
        table which reads as fully checked and is half a chart.

        Zero when the grid published nothing, whatever its uncovered count.
        """
        return self.uncovered_with_count if self.published_cells else 0

    @property
    def unpublished_series(self) -> tuple[str, ...]:
        """Reader series NO cell of the grid addressed, derived from ``uncovered``.

        Derived rather than computed beside it, so the two can never disagree:
        a series is unpublished exactly when none of its identities was
        addressed, and an addressed identity always leaves a cell carrying that
        series' label. A series the grid drops only partly is deliberately
        absent here -- it is not unpublished, and the loss is in ``uncovered``
        where it can be counted per cell.
        """
        addressed = {_series_key(c.series_name) for c in self.cells}
        out: list[str] = []
        for entry in self.uncovered:
            if _series_key(entry.series_name) not in addressed and entry.series_name not in out:
                out.append(entry.series_name)
        return tuple(out)

    @property
    def coverage_complete(self) -> bool:
        """Every cell of both sides met a counterpart.

        False whenever a grid label named nothing geometry read, or geometry
        read a cell the grid never addressed. It says nothing about whether
        the numbers agree -- only about whether the comparison was able to see
        all of them.
        """
        return not (self.unmatched_bins or self.unmatched_series or self.uncovered)

    @property
    def verified(self) -> bool:
        """This grid, as a whole, was compared against geometry and survived.

        Requires three things, and incomplete coverage withholds it: verdicts
        were reached at all, no cell was contradicted, and every reading on
        both sides was actually COMPARED -- not merely named. A grid whose
        cells are all unknown corroborated nothing and is not verified, which
        is what stops "unknown" from ever being reported as checked. Without
        the third, a model earns a clean bill by renaming precisely the series
        that would have contradicted it -- the rename removes the cells rather
        than the disagreement.

        Withholding only ever removes agreement. It cannot manufacture a
        contradiction, and it changes no cell verdict: ``CellVerdict.published``
        is unaffected, so an agreed cell in a partly-covered grid is still an
        agreed cell. What is withheld is the claim about the GRID.
        """
        return (
            bool(self.cells)
            and not self.refusal
            and self.contradicted == 0
            and self.coverage_complete
        )

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "region_index": self.region_index,
            "table_index": self.table_index,
            "orientation": self.orientation,
            "refusal": self.refusal,
            "agreed": self.agreed,
            "contradicted": self.contradicted,
            "unknown_to_geometry": self.unknown,
            "not_a_count": self.not_a_count,
            "coverage_complete": self.coverage_complete,
            "verified": self.verified,
            "unmatched_bins": list(self.unmatched_bins),
            "unmatched_series": list(self.unmatched_series),
            "unpublished_series": list(self.unpublished_series),
            "uncovered_count": self.uncovered_count,
            "uncovered_with_count": self.uncovered_with_count,
            "uncovered_beside_published": self.uncovered_beside_published,
            "uncovered": [u.to_dict() for u in self.uncovered],
            "cells": [c.to_dict() for c in self.cells],
        }


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _label_text(cell: str) -> str:
    """A label cell as the repo already reads one, for comparison only.

    ``decode_label_cell`` is ``binding``'s own view of a row label -- entities
    decoded, the leading indentation run gone -- so a label the binder treats as
    ``Swiss francs`` is compared as ``Swiss francs`` here too. One layer of
    markdown emphasis comes off after it, because ``**Participants**`` and
    ``Participants`` are the same label written twice.
    """
    return _EMPH_RE.sub("", decode_label_cell(cell))


def _series_key(text: str) -> str:
    """Compare-ready form of a series name. Empty when the cell names nothing."""
    return _fold(_label_text(text))


def _bin_key(text: str) -> tuple[str, ...]:
    """Compare-ready form of a bin label: its ATOMS, in order.

    Stage 0 splits a column key into atoms on the range dash and Stage 1 joins a
    printed two-line tick label back with exactly one dash (``_join_atoms``), so
    the atoms are the form in which ``1.88-2.12`` drawn over two lines,
    ``1.88 – 2.12`` written by a model, and the reader's own ``1.88-2.12`` are
    one bin. A label that is not a well-formed key (a categorical bin, a word)
    is its own single atom -- comparable, and comparable only to another label
    that reduces the same way.
    """
    atoms = _key_atoms(_SPACED_DASH_RE.sub("-", _label_text(text)))
    if atoms is not None:
        return tuple(atoms)
    folded = _fold(_label_text(text))
    return (folded,) if folded else ()


def _model_count(cell: str) -> int | None:
    stripped = _EMPH_RE.sub("", decode_label_cell(cell)).strip()
    return int(stripped) if _INT_RE.match(stripped) else None


def _reader_index(panel: PanelReading) -> dict[tuple[str, tuple[str, ...]], tuple[str, Cell]]:
    """``{(series key, bin key): (series name, cell)}`` for every cell read.

    The name is carried because an identity the grid never addresses has to be
    reported in the reader's OWN words -- there is no grid label for it, which
    is precisely what makes it uncovered. Insertion order is the panel's, so
    anything derived from this iterates deterministically.
    """
    out: dict[tuple[str, tuple[str, ...]], tuple[str, Cell]] = {}
    for series in panel.series:
        key = _series_key(series.name)
        if not key:
            continue
        for cell in series.cells:
            bin_key = _bin_key(cell.bin_label)
            if bin_key:
                out[(key, bin_key)] = (series.name, cell)
    return out


def _duplicated(keys: list) -> list:
    seen: set = set()
    dupes: list = []
    for key in keys:
        if key and key in seen and key not in dupes:
            dupes.append(key)
        seen.add(key)
    return dupes


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def reconcile_grid(grid: FilledGrid, panel: PanelReading) -> GridReconciliation:
    """Check every cell of *grid* against *panel*'s geometry, by identity.

    The caller has already decided that this grid is this region's; that
    binding is not re-litigated here and no cell verdict is evidence about it.

    Which axis of the grid carries the bins is read off the identities, not off
    the grid's shape: the reader publishes series as rows and bins as columns
    (``chart_reader.panel_block``) while a model commonly writes the transpose,
    and both are reconciled the same way because both name the same cells. The
    axis matching MORE of the panel's bins carries them; a tie -- including no
    match at all on either axis -- is a refusal, because the grid and the panel
    then share no cell identity and every verdict would be a guess about which
    label meant what.

    The orientation rule is this module's own and not the design's, and it has
    a precondition: that the grid's other axis does not ALSO carry the panel's
    bin labels. Where a chart prints single-value bins that could plausibly
    head either axis, both axes match equally and the tie refuses the grid.
    That is the safe direction -- a refusal, never a silent transposition --
    but it is caused by this rule rather than by the page.

    A label repeated on any of the FOUR identity axes is a refusal: the grid's
    bins, the grid's series, the panel's series, and the panel's bins within
    one series. Two cells with one identity cannot both be the cell a verdict
    is about, and the panel's own axes are no more exempt than the grid's --
    an unguarded one does not refuse, it silently keeps whichever cell was read
    last and judges the model against it.
    """
    empty = GridReconciliation(
        page_num=panel.page_num,
        region_index=panel.region_index,
        table_index=grid.table_index,
        orientation="",
    )

    index = _reader_index(panel)
    if not index:
        return _refuse(empty, "the panel holds no cell to reconcile against")

    # The panel's own identity axes, checked BEFORE the grid's: an index built
    # over a repeated label has already lost a cell by overwriting it, so no
    # later guard can see what went missing.
    if _duplicated([_series_key(s.name) for s in panel.series]):
        return _refuse(empty, "the panel repeats a series name, so its cells have no identity")
    for series in panel.series:
        if _duplicated([_bin_key(cell.bin_label) for cell in series.cells]):
            return _refuse(
                empty,
                f"the panel repeats a bin label within series “{series.name}”, "
                "so one of its cells has no single identity",
            )

    reader_bins = {bin_key for _series, bin_key in index}
    header_keys = [_bin_key(cell) for cell in grid.data_headers]
    row_keys = [_bin_key(label) for label, _cells in grid.rows]
    header_hits = sum(1 for key in header_keys if key in reader_bins)
    row_hits = sum(1 for key in row_keys if key in reader_bins)
    if header_hits == row_hits:
        return _refuse(
            empty,
            "neither axis of the grid names this panel's bins more than the other; "
            f"{header_hits} header key(s) and {row_hits} row label(s) match",
        )
    orientation = BINS_IN_HEADER if header_hits > row_hits else BINS_IN_COLUMN
    empty = GridReconciliation(
        page_num=panel.page_num,
        region_index=panel.region_index,
        table_index=grid.table_index,
        orientation=orientation,
    )

    # One list of (bin label, series name, cell text), in the grid's own
    # reading order, so the two layouts differ here and nowhere after.
    if orientation == BINS_IN_HEADER:
        bin_labels = list(grid.data_headers)
        series_labels = [label for label, _cells in grid.rows]
        entries = [
            (bin_labels[c], series_labels[r], row[c] if c < len(row) else "")
            for r, (_label, row) in enumerate(grid.rows)
            for c in range(len(bin_labels))
        ]
    else:
        bin_labels = [label for label, _cells in grid.rows]
        series_labels = list(grid.data_headers)
        entries = [
            (bin_labels[r], series_labels[c], row[c] if c < len(row) else "")
            for r, (_label, row) in enumerate(grid.rows)
            for c in range(len(series_labels))
        ]

    for axis, labels, key in (
        ("bin", bin_labels, _bin_key),
        ("series", series_labels, _series_key),
    ):
        dupes = _duplicated([key(label) for label in labels])
        if dupes:
            return _refuse(
                empty,
                f"the grid repeats a {axis} label, so a cell of it has no single identity",
            )
    reader_series_keys = {series for series, _bin in index}

    cells: list[CellVerdict] = []
    unmatched_bins: list[str] = []
    unmatched_series: list[str] = []
    # Every identity a grid cell actually COMPARED. Naming an identity is not
    # examining it: a cell left blank, a cell carrying prose, and a cell whose
    # reader counterpart resolved nothing all name a reading and check none of
    # it. Recording the name instead of the comparison let a grid earn coverage
    # for a column it declined to fill -- the same clean bill the rename route
    # buys, one move cheaper, since the identity is addressed and the reading
    # therefore never shows up as uncovered.
    addressed: set[tuple[str, tuple[str, ...]]] = set()
    for bin_label, series_name, raw in entries:
        bin_key, series_key = _bin_key(bin_label), _series_key(series_name)
        found = index.get((series_key, bin_key))
        reader_cell = found[1] if found is not None else None
        if reader_cell is None:
            if bin_key not in reader_bins and bin_label not in unmatched_bins:
                unmatched_bins.append(bin_label)
            if series_key not in reader_series_keys and series_name not in unmatched_series:
                unmatched_series.append(series_name)
        reader_count = (
            reader_cell.count if reader_cell is not None and reader_cell.status == INTEGER else None
        )
        model_count = _model_count(raw)
        cause = ""
        if model_count is None:
            status, detail = NOT_A_COUNT, "the grid cell carries no count"
        elif reader_count is None:
            status = UNKNOWN_TO_GEOMETRY
            cause, detail = _unknown_cause(
                reader_cell is not None,
                bin_key in reader_bins,
                series_key in reader_series_keys,
                bin_label,
                series_name,
            )
        elif reader_count == model_count:
            status, detail = AGREED, ""
        else:
            status, detail = (
                CONTRADICTED,
                f"geometry reads {reader_count} where the grid says {model_count}",
            )
        # Only a verdict that put two numbers side by side is coverage. This
        # is the whole of the rule, and it keeps the invariant: withholding
        # still only ever removes agreement and can never manufacture a
        # contradiction, because nothing here touches a cell's own verdict.
        if status in (AGREED, CONTRADICTED):
            addressed.add((series_key, bin_key))
        cells.append(
            CellVerdict(
                bin_label=bin_label,
                series_name=series_name,
                status=status,
                model_text=raw,
                model_count=model_count,
                reader_count=reader_count,
                detail=detail,
                cause=cause,
            )
        )

    # The reader→model direction. Everything above runs model→reader and is
    # therefore silent about a grid that simply publishes less than the panel
    # holds: drop a column keeping a real series name, or drop one bin row,
    # and every field above reports a clean sheet while half the geometry the
    # reader proved is never consulted. This is that half, per identity.
    uncovered = tuple(
        UncoveredReading(
            series_name=name,
            bin_label=cell.bin_label,
            reader_count=cell.count if cell.status == INTEGER else None,
        )
        for key, (name, cell) in index.items()
        if key not in addressed
    )

    return GridReconciliation(
        page_num=panel.page_num,
        region_index=panel.region_index,
        table_index=grid.table_index,
        orientation=orientation,
        cells=tuple(cells),
        unmatched_bins=tuple(unmatched_bins),
        unmatched_series=tuple(unmatched_series),
        uncovered=uncovered,
    )


def _unknown_cause(
    cell_exists: bool, bin_matched: bool, series_matched: bool, bin_label: str, series_name: str
) -> tuple[str, str]:
    """Why geometry has no number here, and a detail that is true of THIS cell.

    The distinction the caller cannot reconstruct afterwards: a cell geometry
    read and could not resolve is geometry's own limit, while a cell whose bin
    matched and whose series label did not is a reading that EXISTS and was
    never compared. Reporting the second as the first says the page was
    checked where it was not.
    """
    if cell_exists:
        return (
            CAUSE_READER_UNRESOLVED,
            "geometry read this cell and resolved no count; the model's value is unverified",
        )
    if bin_matched and series_matched:
        return (
            CAUSE_CELL_ABSENT,
            "geometry read this bin and this series but holds no cell for the pair; "
            "the model's value is unverified",
        )
    if bin_matched:
        return (
            CAUSE_SERIES_UNMATCHED,
            f"geometry holds a reading for bin “{bin_label}”, but no series it read is named "
            f"“{series_name}”; this cell was never compared",
        )
    if series_matched:
        return (
            CAUSE_BIN_UNMATCHED,
            f"geometry read series “{series_name}”, but no bin it read is "
            f"named “{bin_label}”; "
            "this cell was never compared",
        )
    return (
        CAUSE_NEITHER_MATCHED,
        f"neither bin “{bin_label}” nor series “{series_name}” names "
        "anything geometry read; "
        "this cell was never compared",
    )


def _refuse(base: GridReconciliation, reason: str) -> GridReconciliation:
    """A reconciliation that happened and reached no cell verdict at all.

    Not an error and not a verdict: the grid and the panel could not be shown
    to be about the same cells, so nothing here corroborates or impeaches any
    number in the grid. A caller treating a refusal as agreement, or as
    contradiction, would be inventing the result this module declined to reach.
    """
    return GridReconciliation(
        page_num=base.page_num,
        region_index=base.region_index,
        table_index=base.table_index,
        orientation=base.orientation,
        refusal=reason,
    )


# ---------------------------------------------------------------------------
# #734 Stage B: what a reconciled page records and ships
# ---------------------------------------------------------------------------

#: A filled grid was bound to a region and compared, cell by cell.
GRID_RECONCILED = "chart_grid_reconciled"
#: At least one cell of a compared grid disagreed with geometry. The cell's
#: value is withheld from the body; BOTH numbers are kept on the event.
GRID_CONTRADICTED = "chart_grid_contradicted"
#: A filled grid on a chart page reached no verdict -- nothing bound it, or the
#: reconciler refused it. Its numbers ship unchecked, and that is the finding.
GRID_RECONCILE_REFUSED = "chart_grid_not_reconciled"

#: What stands in the body where a contradicted number was. The sibling of
#: ``chart_reader.UNRESOLVED_MARKER`` and it means a different thing: that one
#: says geometry could not read the cell, this one says two readings of the
#: cell disagreed and neither is published. A reader who sees it must go to the
#: source -- which is why it is a word and not a blank.
CONTRADICTED_MARKER = "WITHHELD"


def withhold_contradicted(grid: FilledGrid, result: GridReconciliation) -> str | None:
    """*grid*'s markdown with every CONTRADICTED cell replaced by the marker.

    ``None`` when the grid has no contradicted cell, so a caller can leave the
    page byte-identical in the overwhelmingly common case.

    **Only contradicted cells are touched, and this is the whole policy.** A
    cell geometry could not resolve, or never met, keeps the model's number:
    refusing it would delete a reading on the strength of a refusal to read,
    and on the SEP corpus that would silently drop 424 numbers on the strength
    of a documented reader limit (#739). What a contradiction removes is one
    number that two independent readings disagree about -- and it removes BOTH,
    the model's and geometry's, because nothing here adjudicates between them.

    The row is rebuilt from its own cells, so only the contradicted row's
    spacing changes; every other line of the grid is left as the model wrote
    it. Position is recovered from the labels the verdict carries, which came
    verbatim off this grid, so the cell rewritten is the cell judged.
    """
    targets = {(c.bin_label, c.series_name) for c in result.cells if c.status == CONTRADICTED}
    if not targets or not result.orientation:
        return None

    lines = grid.text.split("\n")
    sep = next(
        (i for i, line in enumerate(lines) if _is_separator(_split_row(line))),
        None,
    )
    if sep is None:  # pragma: no cover - the grid parsed with a separator to exist
        return None

    headers = list(grid.data_headers)
    touched = False
    for row_index, (row_label, _cells) in enumerate(grid.rows):
        line_index = sep + 1 + row_index
        if line_index >= len(lines):  # pragma: no cover - rows come from these lines
            break
        cells = _split_row(lines[line_index])
        for column, header in enumerate(headers):
            identity = (
                (header, row_label) if result.orientation == BINS_IN_HEADER else (row_label, header)
            )
            if identity not in targets or column + 1 >= len(cells):
                continue
            cells[column + 1] = CONTRADICTED_MARKER
            touched = True
        lines[line_index] = "| " + " | ".join(cells) + " |"
    return "\n".join(lines) if touched else None


def grid_digest(text: str) -> str:
    """SHA-256 of a grid's markdown, on the same rule ``FilledGrid.sha256`` uses.

    What socr writes for a grid it withheld into, so a later crossing can
    recognise its OWN output by identity rather than by searching the text for a
    marker a model is equally entitled to write.
    """
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _grid_key(page_num: int, data: dict) -> tuple:
    """The identity of one grid on one page: (page, table, region)."""
    return (page_num, data.get("table_index"), data.get("region_index"))


def latest_grid_reconciliations(events) -> dict[tuple, dict]:
    """``{(page, table, region): data}`` keeping the LATEST reading of each grid.

    *events* is an iterable of ``(page_num, kind, data)``. A page crossed at
    several rungs files one reconciliation per DISTINCT candidate, so the last
    one is what the body now holds; earlier ones are history.
    """
    out: dict[tuple, dict] = {}
    for page_num, kind, data in events:
        if kind == GRID_RECONCILED:
            out[_grid_key(page_num, data)] = data
    return out


def _materialise(events) -> list[tuple]:
    """The event stream as a list, because it is read TWICE below.

    Callers naturally pass a generator -- the orchestrator filters state.events
    by page inline -- and a generator read twice is empty on the second pass.
    That silently produced ZERO withheld cells everywhere: measured on the SEP
    corpus, 10 markers in the bodies and 0 pages demoted, which is the very
    "body lost a number and a surface says it did not" shape this lane exists
    to stop, reintroduced by the fix for it.
    """
    return list(events)


def withheld_cell_identities(events) -> set[tuple]:
    """Cells the CURRENT reading of each grid withholds -- not every one ever.

    The recency rule the grid count already uses, applied to the cells, because
    a contradiction is a property of one READING of a grid rather than of the
    page. A later rung that gets the same cell right files a fresh
    reconciliation carrying a new ``sha256``; unioning every contradiction event
    ever filed would keep counting the retired one, so a page whose body holds
    four agreed cells and nothing withheld would report four agreed PLUS one
    withheld -- five cells on a four-cell grid, and a number the body does not
    contain.

    A contradiction is kept only when its grid digest is still the digest of
    that grid's latest reading. That also keeps the withheld cells of a grid
    socr rewrote: the rewritten bytes are skipped rather than re-reconciled, so
    the contradicting reading remains the latest one.
    """
    events = _materialise(events)
    latest = latest_grid_reconciliations(events)
    out: set[tuple] = set()
    for page_num, kind, data in events:
        if kind != GRID_CONTRADICTED:
            continue
        key = _grid_key(page_num, data)
        current = latest.get(key)
        if current is None or current.get("sha256") != data.get("sha256"):
            continue
        cell = data.get("cell") or {}
        out.add(key + (cell.get("bin_label"), cell.get("series_name")))
    return out


def unreconciled_grid_identities(events) -> set[tuple]:
    """Grids that reached no verdict, by IDENTITY rather than by event.

    One grid re-emitted with different bytes files a refusal per candidate, so a
    raw event count reported three unchecked grids where the page carries one.
    """
    return {
        _grid_key(page_num, data)
        for page_num, kind, data in events
        if kind == GRID_RECONCILE_REFUSED
    }


def _series_coverage(result: GridReconciliation) -> list[tuple[str, int, int]]:
    """``(series, uncovered identities, of those carrying a number)``, per series.

    Per SERIES and not only per page, because "geometry never saw the December
    column" is what a reader citing this page needs, while "an opinion on a
    fifth of this page" hides which half was checked. Order is the reader's own.
    """
    out: list[tuple[str, int, int]] = []
    index: dict[str, int] = {}
    for entry in result.uncovered:
        if entry.series_name not in index:
            index[entry.series_name] = len(out)
            out.append((entry.series_name, 0, 0))
        position = index[entry.series_name]
        name, total, with_count = out[position]
        out[position] = (name, total + 1, with_count + (entry.reader_count is not None))
    return out


def reconciliation_note(page_num: int, results: list[GridReconciliation]) -> str:
    """The disclosure a reconciled page carries in its own body.

    Four things it must never do, each learned rather than assumed:

    * **never report an unknown or a refused cell as checked.** The unchecked
      count is stated outright, beside the checked one;
    * **never add "unknown" to "uncovered".** A cell geometry could not resolve
      is BOTH -- the model wrote a number nobody could check, AND a reading
      existed that nothing compared -- so summing them double-counts one cell.
      They answer different questions and are reported on separate lines;
    * **never rank the coverage findings by volume.** A grid that published
      nothing can leave a whole chart uncovered while shipping no number: that
      is recall loss. The dangerous shape is the small one -- geometry never
      consulted BESIDE cells that did agree and publish, a table that reads as
      checked and is half a chart. That is reported first whatever its size;
    * **say geometry ABSTAINED, never that it failed or disagreed.** On the SEP
      corpus all 90 zero-resolving series refuse explicitly, every one of them
      ``dashed_stroke`` naming the segment it cannot decompose (#739). A
      refusal with a stated reason is not a disagreement and not a silence.
    """
    compared = [r for r in results if not r.refusal]
    refused = [r for r in results if r.refusal]
    agreed = sum(r.agreed for r in compared)
    contradicted = sum(r.contradicted for r in compared)
    unknown = sum(r.unknown for r in compared)

    lines = [
        f"> **Chart grids on page {page_num} — checked against the page's own geometry** — "
        f"{len(compared)} filled grid(s) were compared cell by cell with the counts read "
        f"from this page's vector drawings. {agreed} cell(s) agreed. "
        f"{contradicted} cell(s) CONTRADICTED geometry and are withheld "
        f"(shown as `{CONTRADICTED_MARKER}`; neither reading is published). "
        f"{unknown} cell(s) carry a number geometry did not check, so they ship UNVERIFIED — "
        "neither corroborated nor impeached."
    ]
    if refused:
        lines.append(
            f"> A further {len(refused)} grid(s) reached no verdict at all "
            f"({'; '.join(sorted({r.refusal for r in refused}))}), so none of their "
            "numbers was compared against anything."
        )

    for result in compared:
        for cell in result.cells:
            if cell.status != CONTRADICTED:
                continue
            lines.append(
                f"> Region {result.region_index}, series “{cell.series_name}”, bin "
                f"“{cell.bin_label}”: the grid says {cell.model_count}, geometry reads "
                f"{cell.reader_count}. Both are withheld; nothing here adjudicates between them."
            )

    beside = [r for r in compared if r.uncovered_beside_published]
    for result in beside:
        for name, total, with_count in _series_coverage(result):
            if not with_count:
                continue
            lines.append(
                f"> Region {result.region_index}, series “{name}”: geometry read "
                f"{with_count} count(s) across {total} bin(s) that NO cell of this grid "
                "addressed — beside cells of the same grid that did agree and publish. "
                "This table reads as checked and is only partly checked."
            )
    recall = [r for r in compared if r.uncovered and not r.uncovered_beside_published]
    for result in recall:
        for name, total, with_count in _series_coverage(result):
            lines.append(
                f"> Region {result.region_index}, series “{name}”: {total} bin(s) geometry "
                f"read were not addressed by this grid, {with_count} of them carrying a "
                "count. This grid published no agreed cell, so nothing here is a check "
                "that passed — the readings are simply absent from the table."
            )
    if any(c.cause == CAUSE_READER_UNRESOLVED for r in compared for c in r.cells):
        lines.append(
            "> Where geometry has no count, it ABSTAINED with a stated reason rather than "
            "failing silently or disagreeing; the reason is on this page's audit events, "
            "per cell. An abstention is not evidence against the model's number."
        )
    return "\n".join(lines)
