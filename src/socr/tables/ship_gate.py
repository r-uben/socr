"""GH-916/GH-917: order-, sign-, coverage- and direction-aware gate in front of native-first SHIP.

``plan_native_table`` ships on ``EXACT_PASS``, which pairs rows by numeric
multiset and ignores standalone sign glyphs. A grid can therefore pass with a
detached minus (``| - | 0.23 |`` where the PDF prints ``-0.23``), with rows or
panels missing, with rows and cells in the wrong order, or with text written in
another direction folded into its cells. This module checks the shipped markdown
against the PDF's own words for exactly those faults.

Every predicate DEFERs and none REFUSEs. A REFUSE sends an upright page to the
D3 image floor with no model attempt; a DEFER sends it to ``route_page`` + judges
+ the table ladder, which is the only outcome that can still recover the table.
The caller records the faults as an audit event so the replacement's provenance
shows why the native grid was rejected.

Predicates (each has its own function; ``direction_unavailable`` is reported by
``foreign_direction_faults``):

``sign_detached``   an output cell ends in a sign glyph (bare cell, end of a
                    populated cell, or attached tail) and the next cell starts a
                    number, AND the source prints that sign in contact with that
                    number's digits (``detached_sign_pairs``, the #887 criterion
                    shared with the ``find_tables`` merge). Bound by row AND column:
                    every source line with the row's numeric sequence must carry
                    the contact on the number's own position, else abstain.
``row_order``       rows that pair to exactly one source row by numeric multiset
                    must appear in increasing source y.
``cell_order``      on such a pair the output's left-to-right numeric cells must
                    equal the source's x-sorted numeric words.
``data_row_missing``  a source row inside the table's vertical span that occupies
                    >= 2 of the table's lanes and has no output row left. The span
                    (``table_spans``) is the core paired rows extended OUTWARD to
                    full-width rows no further than ``_PANEL_GAP_ROWS`` row pitches
                    away (``extended_span``), plus the whole interior between two
                    blocks of the same table; both ends are inclusive. Output rows
                    are counted, not merely found.
``label_row_missing`` a numeric-free source row strictly between the first and last
                    CORE paired row, within the table's x-extent, whose text is not
                    found (counted, per output cell, dehyphenated) in the grid. Nothing
                    inside the span is exempt as prose or a note: a Notes or Source
                    heading between panels is a table label. Rows below the last core
                    row (a swallowed Notes paragraph) are never scanned.
``header_band_missing``  (GH-917) a numeric-free source row above the table's first core
                    row, within the same ``_PANEL_GAP_ROWS`` reach, whose in-lane words each
                    sit over a distinct table lane (>= ``_MIN_LANES_PER_ROW`` of them) and
                    are not all in the grid: a column-header band the rowizer dropped.
``text_in_numeric_column``  (GH-917) a shipped grid row below the table's first data row that
                    is not itself a data row and carries alphabetic text in a column the
                    data rows establish as numeric (a caption, footnote paragraph, equation
                    fragment or "(Continued)" marker emitted as a row). Output-side, per
                    block; header rows above the first data row, panel labels that start in
                    the label columns, number-with-marker cells, placeholders and
                    parenthesised numbers are exempt (see ``text_in_numeric_column_faults``).
``prose_in_header``  (GH-936) a source row above the table's first core row that the grid
                    absorbed WHOLE into its header rows and that is ONE run of two or more
                    words, with no gap wider than ``ALIGNED_RUN_GAP_MAX_WORD_SPACES`` page word
                    spaces: a caption or notes sentence emitted as column headings.
                    ``text_in_numeric_column`` exempts the header band by design and cannot see it.
``foreign_direction``  (GH-917) the grid carries a source word whose text-line direction
                    differs from another carried word's, per output table block. Needs the
                    page's line directions (``LineDirections``); two directions are the
                    same when the angle between them is below
                    ``_SAME_TEXT_DIRECTION_TOL_RAD``.
``direction_unavailable``  (GH-917) line directions were supplied and cannot be trusted
                    (extraction failed, empty map, or a carried word has no entry), or were
                    not supplied at all. Always DEFER and recorded: a production caller never
                    has the direction check silently disabled; only
                    ``LineDirections.unchecked_for_tests()`` skips it.

Thresholds. Two are MEASURED, not derived: ``_PANEL_GAP_ROWS`` (the outward reach, in row
pitches; ``socr-measure-ship-gate-gaps`` reproduces the evidence) and
``_SAME_TEXT_DIRECTION_TOL_RAD`` (the direction tolerance, from the corpus's observed
spread). The lane tolerances are the rowizer's own named quantities (``_LANE_X_TOL_PT`` x
``_LANE_SNAP_MULT``, ``_MIN_LANES_PER_ROW``). Order checks abstain when a row's multiset is
not unique on either side. Coverage is derived from source geometry anchored on unique
pairs, never from the output-derived y-band the value guard uses.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, TypedDict

from socr.core.born_digital import ALIGNED_RUN_GAP_MAX_WORD_SPACES
from socr.tables.native_verifier import (
    _MD_SEP_RE,
    _cluster_x_positions,
    _normalize_numeric_token,
    _numeric_tokens_from_text,
    _parse_output_row_cells,
    is_numeric_token,
)
from socr.tables.reconstruct import (
    _LANE_SNAP_MULT,
    _LANE_X_TOL_PT,
    _MIN_LANES_PER_ROW,
    _NUM_TOKEN_RE,
    _NUMERIC_RE,
    _SIGN_GLYPHS,
    detached_sign_pairs,
)

logger = logging.getLogger(__name__)

#: A PyMuPDF ``get_text("words")`` tuple: ``(x0, y0, x1, y1, text, block, line, word)``.
Word = Sequence
#: One markdown table as rows of cells (header first, separator removed).
Block = list[list[str]]
#: Source words grouped by rounded y0, x-sorted within a row (``_source_rows``).
SourceRows = dict[int, list[Word]]
#: ``(output row index, source y)`` for one block, in output order (``_unique_pairs``).
BlockPairs = list[tuple[int, int]]


class GateFault(TypedDict):
    predicate: str
    detail: str


def _fault(predicate: str, detail: str) -> GateFault:
    return {"predicate": predicate, "detail": detail}


#: Audit-event kind recorded when the gate defers a native grid that exact-passed.
SHIP_GATE_KIND = "native_ship_gate_deferred"
#: Prefix of ``NativeTablePlan.reason`` for a gate DEFER.
SHIP_GATE_REASON_PREFIX = "ship_gate"

SIGN_DETACHED = "sign_detached"
ROW_ORDER = "row_order"
CELL_ORDER = "cell_order"
DATA_ROW_MISSING = "data_row_missing"
LABEL_ROW_MISSING = "label_row_missing"
FOREIGN_DIRECTION = "foreign_direction"
DIRECTION_UNAVAILABLE = "direction_unavailable"
HEADER_BAND_MISSING = "header_band_missing"
TEXT_IN_NUMERIC_COLUMN = "text_in_numeric_column"
PROSE_IN_HEADER = "prose_in_header"
GATE_ERROR = "gate_error"

_LEADING_NUMBER_RE = re.compile(r"^\(?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
#: Distinct lanes a paired row needs to be CORE (``_table_geometry``), and so the
#: fewest lanes a block can have and still be judged the same table as another.
_MIN_CORE_LANES = 2
#: Largest vertical gap, in row pitches, between a table's edge row and the next
#: full-width row that extends its span OUTWARD. It is the whole outward reach (no
#: floors), so 0 means no outward extension. Rows between consecutive blocks of the same
#: table are covered regardless (``table_spans``). MEASURED: 5 is the smallest bound that
#: reaches every known omitted row; the first false extension appears at 8. Evidence and
#: the full sweep: ``socr-measure-ship-gate-gaps`` and
#: ``docs/log/2026-10-01_916-native-ship-gate.md``.
_PANEL_GAP_ROWS = 5
#: Separator inserted where a matched label was removed, so two neighbouring
#: removals can never concatenate into a new match.
_CONSUMED_MARK = "\x00"
#: Snap radius, in points, within which an x position belongs to a lane.
_SNAP_PT = _LANE_X_TOL_PT * _LANE_SNAP_MULT


def _compact(text: str) -> str:
    # NFKC folds typographic ligatures (U+FB00 "ff") so the PDF's "Staff" and
    # the grid's "Staff" are the same word.
    return unicodedata.normalize("NFKC", text).replace(" ", "")


def _is_source_number(text: str) -> bool:
    """A source word the verifier's own native rows treat as a number, plus ``.23``.

    The verifier's source side (``_NUM_TOKEN_RE``) misses a leading decimal that its
    output side reads as a number; including it keeps ``-`` + ``.23`` pairable.
    """
    if _NUM_TOKEN_RE.match(text) and _NUMERIC_RE.search(text):
        return True
    return text[:1] == "." and is_numeric_token(text)


def _leading_number(token: str) -> str:
    """The number a token starts with, normalised (``0.230,`` and ``0.230`` agree)."""
    m = _LEADING_NUMBER_RE.match(token)
    return _normalize_numeric_token(m.group(0)) if m else _normalize_numeric_token(token)


def _key(tokens) -> tuple[str, ...]:
    return tuple(sorted(_normalize_numeric_token(t) for t in tokens))


def _output_blocks(markdown: str) -> list[Block]:
    """Markdown tables as lists of cell lists (header first, separator removed)."""
    blocks: list[Block] = []
    current: list[str] = []

    def flush() -> None:
        if current and any(_MD_SEP_RE.match(line) for line in current):
            blocks.append(
                [_parse_output_row_cells(line) for line in current if not _MD_SEP_RE.match(line)]
            )
        current.clear()

    for raw in (markdown or "").splitlines():
        line = raw.strip()
        if line.startswith("|"):
            current.append(line)
        else:
            flush()
    flush()
    return blocks


def _row_tokens(cells: list[str]) -> list[str]:
    return _numeric_tokens_from_text(" | ".join(cells))


def _out_seq(cells: list[str]) -> list[str]:
    """An output row's numeric tokens, normalised, in left-to-right order."""
    return [_normalize_numeric_token(t) for t in _row_tokens(cells)]


def _src_seq(row_words: list[Word]) -> list[str]:
    """A source row's words, normalised, in the order given (x-sorted by ``_source_rows``)."""
    return [_normalize_numeric_token(w[4]) for w in row_words]


def _source_rows(words: list[Word]) -> SourceRows:
    """All words grouped by rounded y0 (the rowizer's own grouping), x-sorted."""
    rows: dict[int, list[Word]] = defaultdict(list)
    for w in words:
        rows[round(w[1])].append(w)
    for ws in rows.values():
        ws.sort(key=lambda w: w[0])
    return rows


def _numeric_words(row_words: list[Word]) -> list[Word]:
    return [w for w in row_words if _is_source_number(w[4])]


def _unique_pairs(blocks: list[Block], src_rows: SourceRows) -> list[BlockPairs]:
    """Per block, the output rows that pair to exactly one source row, and vice versa.

    A pair is ``(row index, source y)``: the row's numeric multiset occurs once in the
    output and once in the source.
    """
    by_key: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for y, ws in src_rows.items():
        nums = _numeric_words(ws)
        if nums:
            by_key[_key(w[4] for w in nums)].append(y)
    keys = [[_key(_row_tokens(cells)) for cells in block] for block in blocks]
    out_count: Counter = Counter(k for block_keys in keys for k in block_keys if k)
    pairs: list[BlockPairs] = []
    for block_keys in keys:
        found: BlockPairs = []
        for idx, k in enumerate(block_keys):
            if k and out_count[k] == 1 and len(by_key.get(k, ())) == 1:
                found.append((idx, by_key[k][0]))
        pairs.append(found)
    return pairs


def sign_detached_faults(
    words: list[Word], blocks: list[Block], src_rows: SourceRows
) -> list[GateFault]:
    """A sign cell before a number whose source sign is in contact, bound by row AND column.

    The output row's numeric sequence must equal the x-sorted numeric words of
    every candidate source line, and the contact must be on the word at the
    number's own position. If a candidate line (a legitimate placeholder row
    with identical numbers) lacks that contact, or no line has the same
    sequence, the gate abstains.
    """
    sign_pairs = detached_sign_pairs(words)
    if not sign_pairs:
        return []
    contact_ids = {id(d) for _s, d in sign_pairs}
    lines = {y: nums for y, ws in src_rows.items() if (nums := _numeric_words(ws))}
    faults: list[GateFault] = []
    for block in blocks:
        for cells in block:
            seq = _out_seq(cells)
            if not seq:
                continue
            bound = [y for y, ws in lines.items() if _src_seq(ws) == seq]
            if not bound:
                continue  # no source line carries this row in this order: abstain
            for i in range(len(cells) - 1):
                if cells[i].rstrip()[-1:] not in _SIGN_GLYPHS:
                    continue  # bare cell, end of a populated cell, or attached tail
                m = _LEADING_NUMBER_RE.match(cells[i + 1].strip())
                if not m:
                    continue
                pos = len(_row_tokens(cells[: i + 1]))
                if pos >= len(seq) or _leading_number(seq[pos]) != _leading_number(m.group(0)):
                    continue
                if all(id(lines[y][pos]) in contact_ids for y in bound):
                    faults.append(
                        _fault(
                            SIGN_DETACHED,
                            f"cell {i} is a sign before {seq[pos]!r}; the PDF prints "
                            "the sign in contact with that number",
                        )
                    )
    return faults


def row_order_faults(pairs: list[BlockPairs]) -> list[GateFault]:
    """Blocks whose uniquely paired rows are not in increasing source y."""
    faults: list[GateFault] = []
    for found in pairs:
        ys = [y for _idx, y in found]
        if any(b <= a for a, b in zip(ys, ys[1:])):
            faults.append(
                _fault(ROW_ORDER, "rows appear in a different vertical order than the source")
            )
    return faults


def cell_order_faults(
    blocks: list[Block], pairs: list[BlockPairs], src_rows: SourceRows
) -> list[GateFault]:
    """Paired rows whose numeric cells are not in the source's left-to-right order."""
    faults: list[GateFault] = []
    for block, found in zip(blocks, pairs):
        for idx, y in found:
            if _out_seq(block[idx]) != _src_seq(_numeric_words(src_rows[y])):
                faults.append(
                    _fault(
                        CELL_ORDER,
                        f"row {idx}: numeric cells are not in the source's left-to-right order",
                    )
                )
    return faults


def _lane_of(x: float, lanes: list[float]) -> int | None:
    if not lanes:
        return None
    best = min(range(len(lanes)), key=lambda i: abs(lanes[i] - x))
    return best if abs(lanes[best] - x) <= _SNAP_PT else None


def _lane_count(words: list[Word], lanes: list[float]) -> int:
    """How many distinct *lanes* the words' x positions fall in."""
    return len({_lane_of(w[0], lanes) for w in words} - {None})


def _table_geometry(found: BlockPairs, src_rows: SourceRows):
    """``(lanes, core_ys)`` of one block, or None when table membership is unclear.

    Lanes are the x-clusters of numeric words in paired rows that carry >= 2
    numeric words, kept only when >= 2 paired rows use them. A paired row is
    CORE when it occupies >= 2 of those lanes (``_MIN_CORE_LANES``); nothing else
    changes core membership. A false DEFER costs one model call, a missed fault can
    ship a wrong number. A prose line that paired by one stray number (``p<0.01``)
    has one numeric word and is not core.
    """
    multi = [(y, _numeric_words(src_rows[y])) for _i, y in found]
    multi = [(y, ws) for y, ws in multi if len(ws) >= 2]
    if len(multi) < 2:
        return None
    centres = _cluster_x_positions([w[0] for _y, ws in multi for w in ws])
    support: Counter = Counter()
    for _y, ws in multi:
        for lane in {_lane_of(w[0], centres) for w in ws} - {None}:
            support[lane] += 1
    lanes = [c for i, c in enumerate(centres) if support[i] >= 2]
    if not lanes:
        return None
    core = [y for y, ws in multi if _lane_count(ws, lanes) >= _MIN_CORE_LANES]
    if len(core) < 2:
        return None
    return lanes, core


def _block_geometries(pairs: list[BlockPairs], src_rows: SourceRows) -> list:
    """``_table_geometry`` of every block, computed once per gate run."""
    return [_table_geometry(found, src_rows) for found in pairs]


def extended_span(
    core: list[int],
    lanes: list[float],
    src_rows: SourceRows,
    strong_k: int,
    panel_gap_rows: float | None = _PANEL_GAP_ROWS,
) -> tuple[int, int]:
    """The table's vertical span: its core rows, extended outward to full-width rows.

    A dropped first/last data row or a dropped panel is bracketed by no paired row,
    so the span must reach past the first/last core row. Walk outward; only a
    FULL-WIDTH row (>= ``strong_k`` lanes, the table's own modal width) extends the
    span, and only if it is within ``reach`` of the current edge, where ``reach`` is
    ``panel_gap_rows`` row pitches (measured, see ``_PANEL_GAP_ROWS``). Prose or labels
    in between neither extend the span nor bridge to a numeric line further away.
    ``panel_gap_rows=None`` removes the bound (the benchmark's structural variant).
    """
    y_lo, y_hi = min(core), max(core)
    ys_sorted = sorted(core)
    pitch = statistics.median([b - a for a, b in zip(ys_sorted, ys_sorted[1:])] or [0])
    # The bound is the whole reach: no floor, so bound 0 extends nothing and the
    # benchmark's sweep measures exactly what the constant does.
    reach = float("inf") if panel_gap_rows is None else panel_gap_rows * pitch

    def walk(edge: int, outward_ys) -> int:
        """Move *edge* across each full-width row, nearest first, until a gap exceeds reach."""
        for y in outward_ys:
            if _lane_count(_numeric_words(src_rows[y]), lanes) >= strong_k:
                if abs(y - edge) > reach:
                    break
                edge = y
        return edge

    y_lo = walk(y_lo, sorted((y for y in src_rows if y < y_lo), reverse=True))
    y_hi = walk(y_hi, sorted(y for y in src_rows if y > y_hi))
    return y_lo, y_hi


def _same_table_lanes(a: list[float], b: list[float]) -> bool:
    """Whether two blocks' lane sets are the same table's columns, by lanes alone.

    Every lane of the narrower block lies within the snap radius of a lane of the
    wider one, and the narrower block has at least the lanes a CORE row needs
    (``_MIN_CORE_LANES``, the same minimum ``_table_geometry`` uses). So a two-column
    panel whose columns line up with two of a four-column table's is the same table;
    two unrelated tables whose columns do not line up are not. Proximity on the page
    plays no part.
    """
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return len(small) >= _MIN_CORE_LANES and all(_lane_of(x, big) is not None for x in small)


class _Span(NamedTuple):
    """One table's lanes, core rows and vertical extent (a tuple, so callers can unpack)."""

    lanes: list[float]
    core: list[int]
    y_lo: int
    y_hi: int


def table_spans(
    blocks: list[Block],
    pairs: list[BlockPairs],
    src_rows: SourceRows,
    panel_gap_rows=_PANEL_GAP_ROWS,
    geos: list | None = None,
) -> list[_Span]:
    """One ``_Span(lanes, core, y_lo, y_hi)`` per block whose table membership is established.

    Two different reaches, kept separate:

    * OUTWARD, beyond a table's first and last block: governed by ``panel_gap_rows``
      (``extended_span``). Bound 0 means no outward extension.
    * BETWEEN consecutive blocks of the SAME table (lane sets consistent, see
      ``_same_table_lanes``): the interior is inside the table by construction and is
      always covered, whatever the bound. A row omitted from the gap between two blocks
      has the table on both sides.

    ``strong_k`` (the width a row needs to extend the span outward) is the table's
    modal DISTINCT-LANE count over its core rows, with the rowizer's own minimum.
    ``geos`` is ``_block_geometries(pairs, src_rows)`` when the caller already has it.
    """
    if geos is None:
        geos = _block_geometries(pairs, src_rows)
    spans: list[_Span] = []
    for geo in geos:
        if geo is None:
            continue
        lanes, core = geo
        widths = Counter(_lane_count(_numeric_words(src_rows[y]), lanes) for y in core)
        strong_k = max(widths.most_common(1)[0][0], _MIN_LANES_PER_ROW)
        y_lo, y_hi = extended_span(core, lanes, src_rows, strong_k, panel_gap_rows)
        spans.append(_Span(lanes, core, y_lo, y_hi))
    order = sorted(range(len(spans)), key=lambda i: min(spans[i].core))
    for pos, i in enumerate(order):
        for j in order[pos + 1 :]:
            if _same_table_lanes(spans[i].lanes, spans[j].lanes):
                # consecutive blocks of one table: cover everything between their cores
                spans[i] = spans[i]._replace(y_hi=max(spans[i].y_hi, min(spans[j].core)))
                spans[j] = spans[j]._replace(y_lo=min(spans[j].y_lo, max(spans[i].core)))
                break
    return spans


def data_row_missing_faults(
    blocks: list[Block],
    pairs: list[BlockPairs],
    src_rows: SourceRows,
    panel_gap_rows: float | None = _PANEL_GAP_ROWS,
    geos: list | None = None,
) -> list[GateFault]:
    """A source row inside the table's own vertical span with no output row left.

    Membership is spatial: inside the span ``table_spans`` gives (core rows extended
    outward, plus the interior between blocks of the same table; both ends inclusive)
    and in >= 2 of the table's lanes. The grid's numbers are counted, not merely
    found, so a dropped copy of a repeated row is a fault.
    """
    faults: list[GateFault] = []
    # Numeric tokens the grid carries and no paired row has claimed. A candidate
    # source row is present only if ALL its numbers are still available here, and
    # claiming them uses them up, so a dropped copy of a repeated row is missing
    # and a row whose numbers sit merged inside another row's cells is present.
    pool: Counter = Counter()
    for block in blocks:
        for cells in block:
            pool.update(_out_seq(cells))
    for block, found in zip(blocks, pairs):
        for idx, _y in found:
            pool.subtract(_out_seq(block[idx]))
    anchor_ys = {y for found in pairs for _i, y in found}
    spans = table_spans(blocks, pairs, src_rows, panel_gap_rows, geos)
    for y, ws in sorted(src_rows.items()):
        if y in anchor_ys:
            continue
        # Judge the row ONCE, against the UNION of the lanes of every block whose span
        # covers it. A row between a three-lane and a four-lane block that keeps its
        # first three values and loses the fourth must not pass the narrow check and
        # then be shielded from the wider one.
        applicable = [lanes for lanes, _core, y_lo, y_hi in spans if y_lo <= y <= y_hi]
        if not applicable:
            continue
        numeric = _numeric_words(ws)
        in_lane = [w for w in numeric if any(_lane_of(w[0], L) is not None for L in applicable)]
        k = max(_lane_count(numeric, L) for L in applicable)
        if k < _MIN_CORE_LANES:
            continue
        need = Counter(_normalize_numeric_token(w[4]) for w in in_lane)
        if all(pool[t] >= n for t, n in need.items()):
            pool.subtract(need)
            continue
        faults.append(
            _fault(
                DATA_ROW_MISSING,
                f"source row at y={y} with {k} numeric lane(s) "
                f"({', '.join(w[4] for w in in_lane[:6])}) has no output row",
            )
        )
    return faults


class _CellText:
    """Per-cell normalised text of one output block, with occurrence counting."""

    def __init__(self, block: Block) -> None:
        self.original = [_compact(c) for cells in block for c in cells if c]
        self.avail = list(self.original)

    def take(self, text: str) -> bool | None:
        """Consume one occurrence of *text* from a single cell.

        True: consumed. False: the grid has it but every occurrence is used up.
        None: no single cell contains it as a whole.
        """
        for i, cell in enumerate(self.avail):
            j = cell.find(text)
            if j >= 0:
                self.avail[i] = cell[:j] + _CONSUMED_MARK + cell[j + len(text) :]
                return True
        return False if any(text in cell for cell in self.original) else None

    def has_word(self, word: str) -> bool:
        w = _compact(word)
        return any(w in cell for cell in self.original)


def label_row_missing_faults(
    blocks: list[Block],
    pairs: list[BlockPairs],
    src_rows: SourceRows,
    geos: list | None = None,
) -> list[GateFault]:
    """A numeric-free source row strictly inside a table whose words the grid lacks.

    Scans the rows strictly between the first and last CORE paired row, within the
    x-extent of the core words the grid carries. Text is matched per output cell,
    counted (a repeated label needs a repeated cell) and dehyphenated across a
    line break. ``geos`` is ``_block_geometries(pairs, src_rows)`` when already known.
    """
    if geos is None:
        geos = _block_geometries(pairs, src_rows)
    faults: list[GateFault] = []
    for block, found, geo in zip(blocks, pairs, geos):
        if geo is None:
            continue
        lanes, core = geo
        out_tokens = {tok for cells in block for cell in cells for tok in cell.split()}
        carried = [w for y in core for w in src_rows[y] if w[4] in out_tokens]
        if not carried:
            continue
        # The table's x-extent is the bounding box of the core rows' words
        # that the shipped grid carries. Words of another text column that
        # share a y-row never appear in the grid, so they do not widen it.
        x_lo = min(w[0] for w in carried)
        x_hi = max(w[2] for w in carried)
        y_lo, y_hi = min(core), max(core)
        anchor_ys = {y for _i, y in found}
        label_rows: list[tuple[int, list]] = []
        for y, ws in sorted(src_rows.items()):
            if not (y_lo < y < y_hi) or y in anchor_ys:
                continue
            inside = [w for w in ws if w[0] >= x_lo and w[2] <= x_hi]
            if not inside or any(_is_source_number(w[4]) for w in inside):
                continue
            label_rows.append((y, inside))
        text = _CellText(block)
        carry_ok = False  # the previous row ended in a hyphen whose joined word is present
        for n, (y, inside) in enumerate(label_rows):
            toks = [w[4] for w in inside]
            if carry_ok:
                # Its first word is the tail of the previous row's hyphenated word,
                # already matched as the joined word.
                toks = toks[1:]
                carry_ok = False
                if not toks:
                    continue
            nxt = label_rows[n + 1][1] if n + 1 < len(label_rows) else None
            joined = None
            variants = [toks]
            if nxt and len(toks[-1]) > 1 and toks[-1].endswith("-"):
                # A hyphenated line break ("Evalu-" / "ation"): also accept the joined word.
                joined = toks[-1][:-1] + nxt[0][4]
                variants.append(toks[:-1] + [joined])
            carry_ok = joined is not None and text.has_word(joined)
            missing: list[str] | None = None
            for variant in variants:
                got = text.take(_compact("".join(variant)))  # True / False / None, see take()
                if got:
                    missing = []
                    break
                if got is False:
                    missing = variant  # a repeated label with no occurrence left
            if missing is None:
                # Not a whole-row match in any one cell: fall back to word presence.
                missing = [
                    t
                    for k, t in enumerate(toks)
                    if not text.has_word(t) and not (k == len(toks) - 1 and carry_ok)
                ]
            if missing:
                faults.append(
                    _fault(
                        LABEL_ROW_MISSING,
                        f"source label row at y={y} inside the table has word(s) "
                        f"missing from the grid: {', '.join(missing[:6])}",
                    )
                )
    return faults


@dataclass(frozen=True)
class LineDirections:
    """GH-917: the text-line direction of every ``(block, line)`` of one page.

    ``dirs`` maps the ``(block_no, line_no)`` pair a ``get_text("words")`` tuple
    carries (``word[5]``, ``word[6]``) to the line's direction vector as PyMuPDF
    reports it. ``fault`` is non-empty when extraction failed; the gate then DEFERs
    (``direction_unavailable``) instead of skipping P7. Build it with
    ``line_directions_for_page``.
    """

    dirs: dict = field(default_factory=dict)
    fault: str = ""
    #: True only for ``unchecked_for_tests()``: the gate skips P7. Production code
    #: never constructs it; ``grep unchecked_for_tests`` finds every use.
    unchecked: bool = False

    @classmethod
    def unchecked_for_tests(cls) -> "LineDirections":
        """A grep-able sentinel for tests that do not exercise P7. Never use in src."""
        return cls(unchecked=True)


def line_directions_for_page(page) -> LineDirections:
    """Line directions of *page*, aligned to ``page.get_text("words")`` indices.

    ``TEXTFLAGS_WORDS`` is load-bearing: without it ``get_text("dict")`` splits blocks
    around image blocks, so a dict ``(block, line)`` index lands on a different line
    from the word's (measured: 19882 of 19997 words misaligned over 60 image pages,
    0 of 48593 with the flag). Never raises: a failure comes back as
    ``LineDirections(fault=...)`` so a caller cannot mistake it for "no directions
    needed".
    """
    try:
        import pymupdf

        dirs: dict = {}
        for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_WORDS)["blocks"]:
            if block.get("type") != 0:
                continue
            for line_no, line in enumerate(block.get("lines", [])):
                dirs[(block["number"], line_no)] = tuple(line["dir"])
        return LineDirections(dirs=dirs)
    except Exception as exc:
        logger.warning("line direction extraction failed (%s); the gate will defer", exc)
        return LineDirections(fault=f"{type(exc).__name__}: {exc}")


def _unit_direction(value) -> tuple[float, float] | None:
    """A finite, non-zero 2-vector, or None."""
    try:
        x, y = (float(c) for c in value)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)) or (x == 0.0 and y == 0.0):
        return None
    return x, y


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Angle in radians, 0..pi, between two direction vectors of the SAME page frame.

    A direction is a vector, not an axis: text upside down (pi) is a different
    direction from text upright (0). Rotating the page rotates both vectors alike, so
    the angle between two lines of one page is frame-independent.
    """
    return math.atan2(abs(a[0] * b[1] - a[1] * b[0]), a[0] * b[0] + a[1] * b[1])


#: Largest angle, in radians, between two text lines that are the same text for the
#: purposes of P7. MEASURED, not derived from table width: on the 35 rotated and 92
#: upright SHIP pages (160 table blocks on 124 pages, 14913 lines carrying core-row
#: words; ``docs/log/2026-10-01_917-gate-direction-header.md`` round 2) the lines of one table's
#: core rows have exactly one direction vector in 159 of 160 blocks, i.e. an observed
#: maximum deviation of 0 (the 160th, Martens p41, has a genuine 90 degree running head
#: in a core row's y-band). PyMuPDF reports ``dir`` from float32-precision text matrices,
#: so the floor is about 1.2e-7; 1e-5 rad (0.0006 degrees) is ~100x that and about 17x
#: below the smallest deliberate rotation worth the name (0.01 degrees = 1.7e-4 rad).
#: Width plays no part: a slightly rotated stamp can contaminate a cell without
#: crossing a lane, so lane displacement is not evidence that two lines are one text.
_SAME_TEXT_DIRECTION_TOL_RAD = 1e-5


def _direction_unavailable(why: str) -> GateFault:
    return _fault(
        DIRECTION_UNAVAILABLE,
        f"text-line directions could not be used ({why}); deferred, never shipped",
    )


def foreign_direction_faults(words: list[Word], blocks: list[Block], line_dirs) -> list[GateFault]:
    """GH-917 P7: a table block carrying words written in more than one direction.

    ``line_dirs`` is required. ``LineDirections.unchecked_for_tests()`` is the only
    way to skip the check. ``None``, a non-``LineDirections``, a failed or empty map,
    or a carried word without a usable entry DEFERs with ``direction_unavailable`` and
    the reason recorded: a plumbing fault never turns this predicate off.

    Membership is per output block. A source word is carried when its text occurs in
    some cell of that block (``_CellText.has_word``: substring presence, NOT occurrence
    attribution). So a short foreign word (a vertical page number, a one-letter word)
    whose text is a substring of any cell counts as carried: a deliberately
    conservative over-DEFER, since a false DEFER costs one model read and a missed
    foreign word can ship a wrong cell.

    Two carried words have the same direction when the angle between them is below
    ``_SAME_TEXT_DIRECTION_TOL_RAD`` (measured, independent of table width). A fault is
    any PAIR of carried words that does not (a tie between two directions included).
    Nothing chains: a ~ b and b ~ c do not make a ~ c.
    """
    if isinstance(line_dirs, LineDirections) and line_dirs.unchecked:
        return []
    if line_dirs is None:
        return [_direction_unavailable("line_dirs was not supplied")]
    if not isinstance(line_dirs, LineDirections):
        return [_direction_unavailable(f"line_dirs is a {type(line_dirs).__name__}")]
    if line_dirs.fault:
        return [_direction_unavailable(f"extraction failed: {line_dirs.fault}")]
    if not line_dirs.dirs:
        return [_direction_unavailable("the page's line-direction map is empty")]
    faults: list[GateFault] = []
    for block in blocks:
        text = _CellText(block)
        in_grid: dict[str, bool] = {}
        carried = [w for w in words if in_grid.setdefault(w[4], text.has_word(w[4]))]
        if not carried:
            continue
        resolved = []
        missing = []
        for w in carried:
            key = (w[5], w[6]) if len(w) > 6 else None
            vec = _unit_direction(line_dirs.dirs[key]) if key in line_dirs.dirs else None
            if vec is None:
                missing.append(w[4])
            else:
                resolved.append((w, vec))
        if missing:
            faults.append(
                _direction_unavailable(
                    f"{len(missing)} carried word(s) have no usable line direction "
                    f"({', '.join(missing[:6])})"
                )
            )
            continue
        tol = _SAME_TEXT_DIRECTION_TOL_RAD
        vectors = list(dict.fromkeys(vec for _w, vec in resolved))
        if not any(
            _angle_between(a, b) >= tol for i, a in enumerate(vectors) for b in vectors[i + 1 :]
        ):
            continue
        counts = Counter(vec for _w, vec in resolved)
        body = max(vectors, key=lambda v: counts[v])  # first-listed wins a tie
        foreign = [w[4] for w, vec in resolved if _angle_between(vec, body) >= tol]
        faults.append(
            _fault(
                FOREIGN_DIRECTION,
                f"the grid carries word(s) written in a different direction from "
                f"the rest of the table's text ({', '.join(foreign[:6])})",
            )
        )
    return faults


def _run_count(inside: list[Word], unit: float | None) -> int:
    """GH-942: how many runs *inside* (x-sorted) split into at gaps wider than a run gap.

    The run split is the aligned-run one (``ALIGNED_RUN_GAP_MAX_WORD_SPACES`` word spaces,
    in the page's own ``_median_word_gap`` unit): words closer than that belong to one
    heading, wider gaps separate headings. ``0`` when the page has no measurable word gap.
    """
    from socr.core.born_digital import ALIGNED_RUN_GAP_MAX_WORD_SPACES

    if unit is None or not inside:
        return 0
    return 1 + sum(
        1
        for a, b in zip(inside, inside[1:])
        if b[0] - a[2] > ALIGNED_RUN_GAP_MAX_WORD_SPACES * unit
    )


def header_band_missing_faults(
    blocks: list[Block],
    pairs: list[BlockPairs],
    src_rows: SourceRows,
    geos: list | None = None,
) -> list[GateFault]:
    """GH-917: a column-header band above the first core row that the grid omits.

    The rowizer can drop a header band whose first word is the row stub (Fama p561: 12
    header words over 12 lanes, none in the grid). ``label_row_missing`` cannot see it:
    it scans only strictly between the first and last core row.

    A candidate is a source row ``y`` above a block's first core row, no further than
    ``_PANEL_GAP_ROWS`` row pitches (the outward reach ``data_row_missing`` uses), not
    itself a paired row, whose words inside the table's x-extent contain no number, with
    at least one word absent from the block's cells. It fires when its lane-region words
    (those at or right of the first lane less one snap; the stub column is outside it)
    number at least ``_MIN_LANES_PER_ROW`` (the rowizer's own minimum for a row of lanes,
    so a two-word title is not a header) and each has its x-centre within the lane snap
    of a table lane, no two over the same lane (a header over the columns, not prose that
    happens to cross them). A page with no such source row cannot fire.

    GH-942: the lane clause misses a real header in two shapes, a multi-word heading (two
    words over one lane) and a right-aligned or centred one (no x-centre on a lane), so it
    ships grids that lost their column labels. The row also fires when its in-extent words
    split into at least ``_MIN_LANES_PER_ROW`` runs (``_run_count``); the absence test, the
    reach and the numeric-free test are shared, and the absence stays per block. The run
    clause is OR-ed with the lane clause, never a replacement: the lane clause alone catches
    rows the run clause does not.
    """
    if geos is None:
        geos = _block_geometries(pairs, src_rows)
    faults: list[GateFault] = []
    # The gap unit needs each word's (block, line) indices; a bare 5-tuple word has none
    # and is reported by ``foreign_direction`` (``direction_unavailable``), not here.
    unit = _median_word_gap([w for ws in src_rows.values() for w in ws if len(w) > 6])
    for block, found, geo in zip(blocks, pairs, geos):
        if geo is None:
            continue
        lanes, core = geo
        out_tokens = {tok for cells in block for cell in cells for tok in cell.split()}
        carried = [w for y in core for w in src_rows[y] if w[4] in out_tokens]
        if not carried:
            continue
        x_lo = min(w[0] for w in carried)
        x_hi = max(w[2] for w in carried)
        ys = sorted(core)
        first = ys[0]
        reach = _PANEL_GAP_ROWS * statistics.median([b - a for a, b in zip(ys, ys[1:])])
        anchor_ys = {y for _i, y in found}
        text = _CellText(block)
        for y, ws in sorted(src_rows.items()):
            if y >= first or first - y > reach or y in anchor_ys:
                continue
            inside = [w for w in ws if w[0] >= x_lo and w[2] <= x_hi]
            if not inside or any(_is_source_number(w[4]) for w in inside):
                continue
            absent = [w[4] for w in inside if not text.has_word(w[4])]
            if not absent:
                continue
            region = [w for w in inside if w[0] >= lanes[0] - _SNAP_PT]
            hits = [_lane_of((w[0] + w[2]) / 2, lanes) for w in region]
            on_lanes = (
                len(region) >= _MIN_LANES_PER_ROW
                and None not in hits
                and len(set(hits)) == len(region)
            )
            runs = _run_count(inside, unit)
            if on_lanes or runs >= _MIN_LANES_PER_ROW:
                shape = (
                    f"{len(region)} word(s) over {len(set(hits))} distinct table lanes"
                    if on_lanes
                    else f"{len(inside)} word(s) in {runs} runs"
                )
                faults.append(
                    _fault(
                        HEADER_BAND_MISSING,
                        f"source row at y={y} above the first data row has {shape}; "
                        f"{len(absent)} missing from the grid: {', '.join(absent[:6])}",
                    )
                )
    return faults


#: A cell that is one number: optional sign (glyph, then optional space: the PDF prints
#: ``- 0.28`` with the glyph as its own word), optional parenthesis or bracket, the digits,
#: then footnote dressing (``*``, a dagger, ``%``, a closing bracket) and at most
#: ``_MARKER_MAX_LETTERS`` trailing letters (``0.23a``). ``1/53-`` and ``1927-`` are not
#: numbers: the remainder after the digits may not hold a digit or a slash.
_MARKER_MAX_LETTERS = 1
_NUMBER_CELL_RE = re.compile(
    r"^[*\u2020\u2021\u00a7#]*[(\[]?[" + "".join(sorted(_SIGN_GLYPHS)) + r"+]?"
    r"(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"[)\]]?[%*\u2020\u2021\u00a7#]*"
    r"(?:[(]?[^\W\d_]{1," + str(_MARKER_MAX_LETTERS) + r"}[)]?)?$"
)

#: Rows in which one column must hold the same text for it to be that column's placeholder
#: (``n.a.``) rather than prose: a repeat is the definition, one occurrence is not evidence.
_PLACEHOLDER_MIN_ROWS = 2

_EMPTY, _NUMBER, _TEXT, _OTHER = "empty", "number", "text", "other"
#: A text its own column repeats (see ``text_in_numeric_column_faults``).
_PLACEHOLDER = "placeholder"
#: Kinds that fill a cell of a data row without being prose: a value, a bare dash or
#: punctuation, a column's own placeholder.
_FILLS_A_VALUE_CELL = frozenset({_NUMBER, _OTHER, _PLACEHOLDER})


def _cell_kind(cell: str, *, canonical: bool) -> str:
    """``number`` (a value with its dressing), ``text`` (carries a letter), ``empty``, or
    ``other`` (punctuation, a bare dash, a range fragment ``1927-``: no letter, no value).

    ``canonical=False`` is the original GH-917 classifier (the one-letter-marker regex only).
    ``canonical=True`` asks the verifier's numeric contract first (currency prefixes,
    ``∗``/``✱``/dagger dressing, markdown emphasis, entities: GH-932). Both are run, see
    ``text_in_numeric_column_faults``: a wider classifier can change which row is the last data
    row, so it is only ever used in addition to the original."""
    compact = _compact(cell)
    if not compact:
        return _EMPTY
    if _NUMBER_CELL_RE.match(compact) or (canonical and is_numeric_token(compact)):
        return _NUMBER
    return _TEXT if any(c.isalpha() for c in compact) else _OTHER


def _data_rows(kinds: list[list[str]], paired: set[int], *, panels: bool) -> list[int]:
    """Indices of the block's data rows (see ``text_in_numeric_column_faults``).

    ``panels=False`` is the original rule (GH-917); ``panels=True`` is the disjoint-panel
    rule (GH-932), which is only ever run IN ADDITION to the original."""
    candidates = [
        i for i, ks in enumerate(kinds) if i in paired and ks.count(_NUMBER) >= _MIN_CORE_LANES
    ]
    cols = _numeric_columns(kinds, candidates, panels=panels)
    # A row carrying a number or two (a panel label with a year range, a "p < 0.05" note)
    # is a candidate but not a data row: its values must reach most numeric columns (a
    # dash or a placeholder in a value cell counts: a row can print one where it has no number).
    covered = {
        i
        for i in candidates
        if 2 * sum(1 for c in cols if c < len(kinds[i]) and kinds[i][c] in _FILLS_A_VALUE_CELL)
        > len(cols)
    }
    if not panels:
        return sorted(covered)

    # A table of disjoint panels (each row fills only its own panel's columns) never reaches
    # more than half of them, so a row also counts when its exact set of number columns is
    # shared by another candidate row: a repeat is evidence, a one-off label is not
    # (``_PLACEHOLDER_MIN_ROWS``).
    def support(i: int) -> frozenset[int]:
        return frozenset(c for c, k in enumerate(kinds[i]) if k == _NUMBER)

    supports = Counter(support(i) for i in candidates)
    return [i for i in candidates if i in covered or supports[support(i)] >= _PLACEHOLDER_MIN_ROWS]


def _numeric_columns(kinds: list[list[str]], rows: list[int], *, panels: bool) -> list[int]:
    """Columns where numbers fill MORE THAN HALF of the *rows*.

    ``panels=True`` counts only the rows that have something in the column (a row with an
    empty cell there says nothing about it: disjoint panels would otherwise leave every
    column at exactly half) and needs at least ``_PLACEHOLDER_MIN_ROWS`` such rows: one
    filled cell is not a column."""
    width = max((len(ks) for ks in kinds), default=0)
    cols = []
    for c in range(width):
        cells = [kinds[i][c] if c < len(kinds[i]) else _EMPTY for i in rows]
        if panels:
            cells = [k for k in cells if k != _EMPTY]
            if len(cells) < _PLACEHOLDER_MIN_ROWS:
                continue
        if 2 * cells.count(_NUMBER) > len(cells):
            cols.append(c)
    return cols


def _rule_faults(
    block: Block, kinds: list[list[str]], found: BlockPairs, *, panels: bool
) -> dict[int, dict[int, str]]:
    """Evidence of one block under one data-row rule: row index -> {column: cell text}."""
    out: dict[int, dict[int, str]] = {}
    data = _data_rows(kinds, {idx for idx, _y in found}, panels=panels)
    if len(data) < 2:
        return out
    numeric_cols = _numeric_columns(kinds, data, panels=panels)
    if not numeric_cols:
        return out
    first, last = data[0], data[-1]
    data_set = set(data)
    for i in range(first + 1, len(block)):
        if i in data_set:
            continue
        ks = kinds[i]
        lead = next((c for c, k in enumerate(ks) if k != _EMPTY), None)
        if lead is not None and lead < numeric_cols[0] and i < last:
            continue  # a panel label starting in the label columns
        hit = [c for c in numeric_cols if c < len(ks) and ks[c] == _TEXT]
        if hit:
            out[i] = {c: block[i][c] for c in hit}
    return out


#: The (canonical classifier, disjoint-panel rule) members whose faults are unioned. The first
#: is the GH-917 predicate unchanged.
_TNC_MEMBERS = ((False, False), (True, False), (True, True))


def _block_kinds(block: Block, *, canonical: bool) -> list[list[str]]:
    """Cell kinds of *block*, with a text its own column repeats turned into a placeholder."""
    kinds = [[_cell_kind(c, canonical=canonical) for c in row] for row in block]
    repeats: Counter = Counter()
    for row, ks in zip(block, kinds):
        repeats.update({(c, _compact(row[c]).casefold()) for c, k in enumerate(ks) if k == _TEXT})
    for row, ks in zip(block, kinds):
        for c, k in enumerate(ks):
            if k == _TEXT and repeats[(c, _compact(row[c]).casefold())] >= _PLACEHOLDER_MIN_ROWS:
                ks[c] = _PLACEHOLDER
    return kinds


def text_in_numeric_column_faults(blocks: list[Block], pairs: list[BlockPairs]) -> list[GateFault]:
    """GH-917: a non-data row carrying letters in a column the data rows establish as numeric.

    Per output block, all derived from the block's own rows:

    * PLACEHOLDERS: a text that the SAME column holds in two or more rows (``n.a.`` in a
      column of a table that prints it) is a placeholder, not prose; it counts as neither
      text nor number. A bare dash has no letter and is never text. No vocabulary: prose
      rarely repeats verbatim down one column (two identical footnote lines in one
      column would read as a placeholder: a missed DEFER, not a false one).
    * DATA ROWS: rows that pair uniquely to a source row (so the PDF confirms them), carry at
      least ``_MIN_CORE_LANES`` number cells (the rowizer's own minimum for a row of values),
      and whose values (numbers, dashes, placeholders) reach more than half of the columns
      that the paired rows establish as numeric. Fewer than two data rows: abstain.
    * NUMERIC COLUMNS: a column where MORE THAN HALF of the data rows hold a number.
    * EXEMPT: every row above the first data row (the header band, whatever it holds), a
      number with its marker or standard-error parentheses, and a row BEFORE the last data
      row whose first non-empty cell is left of the first numeric column (a panel label
      that occupies the label columns).
    * FAULT: any other row, below the first data row, with a text cell in a numeric column.
      So a footnote or "(Continued)" under the data, or a sub-header floating over numeric
      columns between data rows, defers.

    GH-932: each block is judged by three members and the faults are UNIONED (by row; evidence
    from members faulting the same row is merged), so the predicate can only add DEFERs relative to
    GH-917. ``_TNC_MEMBERS`` is (original classifier, rule above) = EXACTLY the GH-917 predicate,
    (verifier's numeric classifier, rule above), and (verifier's classifier, ``panels=True``). A
    wider classifier can change which row is the last data row, so it is never the only member.
    ``panels=True`` counts a column as numeric against only the data rows that fill it and admits
    rows sharing another data row's numeric support; it catches disjoint panels (panel A in some
    columns, panel B in others), where the rule above finds no majority and abstains.

    Limit (measured, ``docs/log/2026-10-01_917-text-in-numeric-column.md``): text emitted
    ABOVE the first data row (a caption or equation fragments between the title and the
    column headings) is indistinguishable from a column heading by the grid alone, so it is
    not reported here.
    """
    faults: list[GateFault] = []
    for block, found in zip(blocks, pairs):
        if not any(block):
            # A separator-only or empty block has no cells to judge; skipping it keeps
            # it from raising into ``native_ship_gate``'s handler, which would replace
            # the other blocks' faults with ``gate_error`` (Astra, PR #931).
            continue
        # MONOTONE by construction (GH-932): the first member is EXACTLY the GH-917 predicate
        # (original classifier, original rule), so every fault it found is still found; the
        # others only add rows. The gate only DEFERs, so a union can add a DEFER and never lose
        # one. Evidence from members that fault the same row is merged.
        evidence: dict[int, dict[int, str]] = {}
        for canonical, panels in _TNC_MEMBERS:
            kinds = _block_kinds(block, canonical=canonical)
            for i, cols in _rule_faults(block, kinds, found, panels=panels).items():
                evidence.setdefault(i, {}).update(cols)
        for i in sorted(evidence):
            hit = sorted(evidence[i])
            faults.append(
                _fault(
                    TEXT_IN_NUMERIC_COLUMN,
                    f"row {i} carries text in numeric column(s) "
                    f"{', '.join(str(c) for c in hit[:4])}: "
                    f"{' | '.join(evidence[i][c] for c in hit[:3])!r}",
                )
            )
    return faults


def _is_one_run(row_words: list[Word], word_space: float) -> bool:
    """True when no gap between x-sorted neighbours exceeds ``ALIGNED_RUN_GAP_MAX_WORD_SPACES``
    word spaces (the #934 R1 / ``_detect_column_gutter`` yardstick, reused)."""
    limit = ALIGNED_RUN_GAP_MAX_WORD_SPACES * word_space
    return all(b[0] - a[2] <= limit for a, b in zip(row_words, row_words[1:]))


#: Text lines outside every table's vertical extent that must contribute a gap before the page's
#: word space is trusted. A repeat is evidence, one line is not (the ``_PLACEHOLDER_MIN_ROWS`` rule).
_MIN_SPACING_LINES = _PLACEHOLDER_MIN_ROWS


def _page_word_space(words: list[Word], extents: list[tuple[float, float]]) -> float | None:
    """The page's median same-line word gap, measured ONLY on lines outside every table extent.

    Inside a table the only gaps are column gutters (per-cell or whole-row PDF lines alike), which
    would make a real multi-column header read as one run. Body prose, captions and notes outside the
    extents print the ordinary word space. ``None`` (the caller abstains) when fewer than
    ``_MIN_SPACING_LINES`` such lines have a gap, or when the words carry no block/line indices.
    """
    by_line: dict[tuple, list[Word]] = defaultdict(list)
    for w in words:
        if len(w) <= 6:
            continue
        if any(lo <= w[1] <= hi for lo, hi in extents):
            continue
        by_line[(w[5], w[6])].append(w)
    gaps: list[float] = []
    lines = 0
    for line_words in by_line.values():
        ordered = sorted(line_words, key=lambda w: w[0])
        line_gaps = [b[0] - a[2] for a, b in zip(ordered, ordered[1:]) if b[0] - a[2] > 0]
        if line_gaps:
            lines += 1
            gaps.extend(line_gaps)
    if lines < _MIN_SPACING_LINES:
        return None
    return statistics.median(gaps)


def prose_in_header_faults(
    words: list[Word],
    blocks: list[Block],
    pairs: list[BlockPairs],
    src_rows: SourceRows,
    geos: list | None = None,
) -> list[GateFault]:
    """GH-936: a caption or notes sentence absorbed whole into a grid's header rows.

    ``text_in_numeric_column`` exempts every row above the first data row, and
    ``header_band_missing`` only looks for header words the grid dropped. Neither sees a
    source row the grid KEPT but that is prose, not a column heading.

    Per block with geometry: the header rows are the output rows above the first CORE paired
    row. A source row above the first core row is *carried* when every one of its words
    occurs (counted, NFKC) as a token of those header rows. The predicate fires on a carried
    row that is ONE run of at least two words (``_is_one_run``): column headings sit over
    separate lanes and so split into runs at the page's lane gutter, a sentence does not.
    Measured on the 127-page census (GH-936 design): +7 DEFERs against GH-935, of which 3 are
    real, 1 is a broken page, 3 are false (a genuine one-run spanning heading, a panel title,
    an in-table panel label); a false DEFER costs one model read.

    The yardstick is the page's word space measured on lines OUTSIDE every table's extent (core
    rows less/plus the outward reach of ``_PANEL_GAP_ROWS`` pitches), see ``_page_word_space``; with
    too little such text the predicate abstains rather than guess from column gutters.

    Known holes, by construction: a page with no text outside its tables (it abstains); a one-word caption (indistinguishable from a one-word
    heading) and a caption with a gap wider than the bound (reads as lane-shaped). No font
    size clause: the design measured zero census gain and an exact float comparison is brittle.
    No panel-label exemption: it would save one false DEFER and add code.
    """
    if geos is None:
        geos = _block_geometries(pairs, src_rows)
    extents = []
    for geo in geos:
        if geo is None:
            continue
        ys = sorted(geo[1])
        reach = _PANEL_GAP_ROWS * statistics.median([b - a for a, b in zip(ys, ys[1:])])
        extents.append((ys[0] - reach, ys[-1] + reach))
    word_space = _page_word_space(words, extents) if extents else None
    if not word_space:
        return []
    faults: list[GateFault] = []
    for block, found, geo in zip(blocks, pairs, geos):
        if geo is None:
            continue
        _lanes, core = geo
        core_set = set(core)
        first_idx = min(i for i, y in found if y in core_set)
        header = Counter(
            unicodedata.normalize("NFKC", tok).strip()
            for row in block[:first_idx]
            for cell in row
            for tok in cell.split()
        )
        if not header:
            continue
        for y in sorted(src_rows):
            if y >= min(core):
                break
            row_words = src_rows[y]
            if len(row_words) < 2 or not _is_one_run(row_words, word_space):
                continue
            need = Counter(unicodedata.normalize("NFKC", w[4]).strip() for w in row_words)
            if all(header[tok] >= n for tok, n in need.items()):
                faults.append(
                    _fault(
                        PROSE_IN_HEADER,
                        f"source row at y={y} above the first data row is one run of "
                        f"{len(row_words)} words (no gap over {ALIGNED_RUN_GAP_MAX_WORD_SPACES:g} "
                        "word spaces) and the grid carries all of it in its header rows",
                    )
                )
    return faults


def native_ship_gate(words: list[Word], markdown: str, *, line_dirs) -> tuple[GateFault, ...]:
    """Faults found by the ship gate, or ``()`` when the grid may ship.

    A gate that raises must not ship the grid: it reports ``gate_error`` so the
    caller defers to normal routing. ``line_dirs`` (GH-917) is required: a
    ``LineDirections`` enables ``foreign_direction`` and an unusable one (or ``None``)
    DEFERs with ``direction_unavailable``; only ``LineDirections.unchecked_for_tests()``
    skips it.
    """
    try:
        blocks = _output_blocks(markdown)
        if not blocks or not words:
            return ()
        src_rows = _source_rows(words)
        pairs = _unique_pairs(blocks, src_rows)
        geos = _block_geometries(pairs, src_rows)
        faults: list[GateFault] = []
        faults += sign_detached_faults(words, blocks, src_rows)
        faults += row_order_faults(pairs)
        faults += cell_order_faults(blocks, pairs, src_rows)
        faults += data_row_missing_faults(blocks, pairs, src_rows, geos=geos)
        faults += label_row_missing_faults(blocks, pairs, src_rows, geos)
        faults += header_band_missing_faults(blocks, pairs, src_rows, geos)
        faults += text_in_numeric_column_faults(blocks, pairs)
        faults += prose_in_header_faults(words, blocks, pairs, src_rows, geos)
        faults += foreign_direction_faults(words, blocks, line_dirs)
        return tuple(faults)
    except Exception as exc:
        logger.warning("native ship gate failed (%s); deferring", exc)
        return (_fault(GATE_ERROR, f"{type(exc).__name__}: {exc}"),)
