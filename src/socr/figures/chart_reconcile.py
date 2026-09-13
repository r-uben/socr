"""#734 Stage A: reconcile a model's FILLED chart grid against read geometry.

#635 Stage 0 withholds a chart region's EMPTY derivation, and Stage 1 reads the
region's vector drawings and publishes the counts it can prove. Between the two
sits the case this module answers: the model emitted a grid for the chart region
with NUMBERS in it. Nothing in the grid says whether those numbers were read off
the page or invented, and a filled grid is exactly what an extraction that
succeeded looks like -- so the page ships them unchecked, which is how
``| 0.13-0.37 | 17 | 17 |`` reached the corpus beside a panel whose geometry
derives ``17, 0, 0, ...``.

The rule is the design's (``docs/plans/chart-data/DESIGN.md``, Stage 2) and this
module implements that and nothing more:

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

Pure: no I/O, no state, no page. It takes a grid the caller already parsed and
the ``PanelReading`` for the region the caller already bound it to, and returns
a verdict per cell. Binding a grid to a region, and deciding what the page then
ships, are the caller's (Stage B); this module has no opinion on either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from socr.figures.chart_data import _EMPH_RE, FilledGrid, _fold, _key_atoms
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
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GridReconciliation:
    """One filled grid, checked cell by cell against one panel's geometry."""

    page_num: int
    region_index: int
    table_index: int
    orientation: str
    cells: tuple[CellVerdict, ...] = ()
    unmatched_bins: tuple[str, ...] = ()
    unmatched_series: tuple[str, ...] = ()
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
            "unmatched_bins": list(self.unmatched_bins),
            "unmatched_series": list(self.unmatched_series),
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


def _reader_index(panel: PanelReading) -> dict[tuple[str, tuple[str, ...]], Cell]:
    """``{(series key, bin key): cell}`` for every cell the reader holds."""
    out: dict[tuple[str, tuple[str, ...]], Cell] = {}
    for series in panel.series:
        key = _series_key(series.name)
        if not key:
            continue
        for cell in series.cells:
            bin_key = _bin_key(cell.bin_label)
            if bin_key:
                out[(key, bin_key)] = cell
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

    A label repeated on either axis of either side is a refusal too: two cells
    with one identity cannot both be the cell a verdict is about.
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
    if _duplicated([_series_key(s.name) for s in panel.series]):
        return _refuse(empty, "the panel repeats a series name, so its cells have no identity")

    cells: list[CellVerdict] = []
    unmatched_bins: list[str] = []
    unmatched_series: list[str] = []
    for bin_label, series_name, raw in entries:
        bin_key, series_key = _bin_key(bin_label), _series_key(series_name)
        reader_cell = index.get((series_key, bin_key))
        if reader_cell is None:
            if bin_key not in reader_bins and bin_label not in unmatched_bins:
                unmatched_bins.append(bin_label)
            if series_key not in reader_series_keys and series_name not in unmatched_series:
                unmatched_series.append(series_name)
        reader_count = (
            reader_cell.count if reader_cell is not None and reader_cell.status == INTEGER else None
        )
        model_count = _model_count(raw)
        if model_count is None:
            status, detail = NOT_A_COUNT, "the grid cell carries no count"
        elif reader_count is None:
            status, detail = (
                UNKNOWN_TO_GEOMETRY,
                "geometry holds no count for this cell; the model's value is unverified"
                if reader_cell is not None
                else "geometry has no such cell; the model's value is unverified",
            )
        elif reader_count == model_count:
            status, detail = AGREED, ""
        else:
            status, detail = (
                CONTRADICTED,
                f"geometry reads {reader_count} where the grid says {model_count}",
            )
        cells.append(
            CellVerdict(
                bin_label=bin_label,
                series_name=series_name,
                status=status,
                model_text=raw,
                model_count=model_count,
                reader_count=reader_count,
                detail=detail,
            )
        )

    return GridReconciliation(
        page_num=panel.page_num,
        region_index=panel.region_index,
        table_index=grid.table_index,
        orientation=orientation,
        cells=tuple(cells),
        unmatched_bins=tuple(unmatched_bins),
        unmatched_series=tuple(unmatched_series),
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
