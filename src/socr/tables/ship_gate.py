"""GH-916: order-, sign- and coverage-aware gate in front of native-first SHIP.

``plan_native_table`` ships on ``EXACT_PASS``, which pairs rows by numeric
multiset and ignores standalone sign glyphs. A grid can therefore pass with a
detached minus (``| - | 0.23 |`` where the PDF prints ``-0.23``), with rows or
panels missing, or with rows and cells in the wrong order. This module checks
the shipped markdown against the PDF's own words for exactly those faults.

Every predicate DEFERs and none REFUSEs. A REFUSE sends an upright page to the
D3 image floor with no model attempt; a DEFER sends it to ``route_page`` + judges
+ the table ladder, which is the only outcome that can still recover the table.
The caller records the faults as an audit event so the replacement's provenance
shows why the native grid was rejected.

Predicates (each a separate function, each independently disabled in its test):

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
``data_row_missing``  a source row strictly inside the table's own vertical span
                    (first to last CORE paired row) that occupies >= 2 of the
                    table's lanes and has no output row left. Output rows are
                    counted, not merely found.
``label_row_missing`` a numeric-free source row strictly between the first and last
                    CORE paired row, within the table's x-extent, whose text is not
                    found (counted, per output cell, dehyphenated) in the grid. Nothing
                    inside the span is exempt as prose or a note: a Notes or Source
                    heading between panels is a table label. Rows below the last core
                    row (a swallowed Notes paragraph) are never scanned.

Every tolerance is an existing named rowizer quantity (``_LANE_X_TOL_PT`` x
``_LANE_SNAP_MULT``, ``_MIN_LANES_PER_ROW``); nothing here is a new threshold.
Order checks abstain when a row's multiset is not unique on either side. Coverage
is derived from source geometry anchored on unique pairs, never from the
output-derived y-band the value guard uses.
"""

from __future__ import annotations

import logging
import re
import statistics
import unicodedata
from collections import Counter, defaultdict

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

#: Audit-event kind recorded when the gate defers a native grid that exact-passed.
SHIP_GATE_KIND = "native_ship_gate_deferred"
#: Prefix of ``NativeTablePlan.reason`` for a gate DEFER.
SHIP_GATE_REASON_PREFIX = "ship_gate"

SIGN_DETACHED = "sign_detached"
ROW_ORDER = "row_order"
CELL_ORDER = "cell_order"
DATA_ROW_MISSING = "data_row_missing"
LABEL_ROW_MISSING = "label_row_missing"
GATE_ERROR = "gate_error"

_LEADING_NUMBER_RE = re.compile(r"^\(?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
#: Distinct lanes a paired row needs to be CORE (``_table_geometry``), and so the
#: fewest lanes a block can have and still be judged the same table as another.
_MIN_CORE_LANES = 2
#: Largest vertical gap, in row pitches, allowed between the table's edge row and
#: the next full-width row that extends its span OUTWARD; the whole outward reach (no
#: floors), so bound 0 means no outward extension. Rows between consecutive blocks of
#: the same table are covered regardless (see ``table_spans``), so this bound governs
#: only reach beyond a table's first and last block. Chosen from what each bound does on the corpus, measured
#: by ``socr-measure-ship-gate-gaps`` (35 rotated + 92 upright native-first SHIP
#: pages; gaps between consecutive core paired rows: median 1.0, p90 2.0, p95 2.25,
#: p99 4.83). Row level, against 14 source rows known to be omitted from their grids
#: (Fama p398, lopez-lira p32, bugel p11, brochet p21, segal p66): reached 0 of 14 at
#: bounds 0-1, 4 at 2, 10 at 3, 12 at 4, 14 at 5, and 14 at every larger bound.
#: Newly FIRING rows beyond the known ones: at 5 only woodford index rows, ramey p104
#: text-table rows and three more omitted bugel rows (all real or non-table); 6 adds
#: nothing that fires; 8 first adds ljungvist p7, a table of contents. So 5 is the
#: smallest bound that reaches every known omitted row, and the first false extension
#: appears at 8. The unbounded structural rule fires on the same rows as 8.
_PANEL_GAP_ROWS = 5
#: Separator inserted where a matched label was removed, so two neighbouring
#: removals can never concatenate into a new match.
_USED = "\x00"


def _compact(text: str) -> str:
    # NFKC folds typographic ligatures (U+FB00 "ff") so the PDF's "Staff" and
    # the grid's "Staff" are the same word.
    return unicodedata.normalize("NFKC", text).replace(" ", "")


def _is_num(text: str) -> bool:
    """A source word the verifier's own native rows treat as a number, plus ``.23``.

    The verifier's source side (``_NUM_TOKEN_RE``) misses a leading decimal that its
    output side reads as a number; including it keeps ``-`` + ``.23`` pairable.
    """
    if _NUM_TOKEN_RE.match(text) and _NUMERIC_RE.search(text):
        return True
    return text[:1] == "." and is_numeric_token(text)


def _lead(token: str) -> str:
    """The number a token starts with, normalised (``0.230,`` and ``0.230`` agree)."""
    m = _LEADING_NUMBER_RE.match(token)
    return _normalize_numeric_token(m.group(0)) if m else _normalize_numeric_token(token)


def _key(tokens) -> tuple[str, ...]:
    return tuple(sorted(_normalize_numeric_token(t) for t in tokens))


def _output_blocks(markdown: str) -> list[list[list[str]]]:
    """Markdown tables as lists of cell lists (header first, separator removed)."""
    blocks: list[list[list[str]]] = []
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


def _source_rows(words: list) -> dict[int, list]:
    """All words grouped by rounded y0 (the rowizer's own grouping), x-sorted."""
    rows: dict[int, list] = defaultdict(list)
    for w in words:
        rows[round(w[1])].append(w)
    for ws in rows.values():
        ws.sort(key=lambda w: w[0])
    return rows


def _numeric_words(row_words: list) -> list:
    return [w for w in row_words if _is_num(w[4])]


class _Anchors:
    """Output rows that pair to exactly one source row, and vice versa."""

    def __init__(self, blocks, src_rows) -> None:
        by_key: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for y, ws in src_rows.items():
            nums = _numeric_words(ws)
            if nums:
                by_key[_key(w[4] for w in nums)].append(y)
        out_count: Counter = Counter()
        for block in blocks:
            for cells in block:
                k = _key(_row_tokens(cells))
                if k:
                    out_count[k] += 1
        self.out_count = out_count
        #: per block: [(row index, source y)] in output order.
        self.per_block: list[list[tuple[int, int]]] = []
        for block in blocks:
            found: list[tuple[int, int]] = []
            for idx, cells in enumerate(block):
                k = _key(_row_tokens(cells))
                if k and out_count[k] == 1 and len(by_key.get(k, ())) == 1:
                    found.append((idx, by_key[k][0]))
            self.per_block.append(found)


def sign_detached_faults(words: list, blocks, src_rows) -> list[dict]:
    """A sign cell before a number whose source sign is in contact, bound by row AND column.

    The output row's numeric sequence must equal the x-sorted numeric words of
    every candidate source line, and the contact must be on the word at the
    number's own position. If a candidate line (a legitimate placeholder row
    with identical numbers) lacks that contact, or no line has the same
    sequence, the gate abstains.
    """
    pairs = detached_sign_pairs(words)
    if not pairs:
        return []
    contact_ids = {id(d) for _s, d in pairs}
    lines = {y: _numeric_words(ws) for y, ws in src_rows.items()}
    lines = {y: ws for y, ws in lines.items() if ws}
    faults: list[dict] = []
    for block in blocks:
        for cells in block:
            seq = [_normalize_numeric_token(t) for t in _row_tokens(cells)]
            if not seq:
                continue
            bound = [
                y for y, ws in lines.items() if [_normalize_numeric_token(w[4]) for w in ws] == seq
            ]
            if not bound:
                continue  # no source line carries this row in this order: abstain
            for i in range(len(cells) - 1):
                if cells[i].rstrip()[-1:] not in _SIGN_GLYPHS:
                    continue  # bare cell, end of a populated cell, or attached tail
                m = _LEADING_NUMBER_RE.match(cells[i + 1].strip())
                if not m:
                    continue
                pos = len(_row_tokens(cells[: i + 1]))
                if pos >= len(seq) or _lead(seq[pos]) != _lead(m.group(0)):
                    continue
                if all(id(lines[y][pos]) in contact_ids for y in bound):
                    faults.append(
                        {
                            "predicate": SIGN_DETACHED,
                            "detail": (
                                f"cell {i} is a sign before {seq[pos]!r}; the PDF prints "
                                "the sign in contact with that number"
                            ),
                        }
                    )
    return faults


def order_faults(blocks, anchors: _Anchors, src_rows, *, cells: bool) -> list[dict]:
    """``cells=False``: row_order; ``cells=True``: cell_order."""
    faults: list[dict] = []
    for block, found in zip(blocks, anchors.per_block):
        if not cells:
            ys = [y for _idx, y in found]
            if any(b <= a for a, b in zip(ys, ys[1:])):
                faults.append(
                    {
                        "predicate": ROW_ORDER,
                        "detail": "rows appear in a different vertical order than the source",
                    }
                )
            continue
        for idx, y in found:
            out_seq = [_normalize_numeric_token(t) for t in _row_tokens(block[idx])]
            src_seq = [_normalize_numeric_token(w[4]) for w in _numeric_words(src_rows[y])]
            if out_seq != src_seq:
                faults.append(
                    {
                        "predicate": CELL_ORDER,
                        "detail": f"row {idx}: numeric cells are not in the source's left-to-right order",
                    }
                )
    return faults


def _snap() -> float:
    return _LANE_X_TOL_PT * _LANE_SNAP_MULT


def _lanes(anchor_numeric_words: list) -> list[float]:
    return _cluster_x_positions([w[0] for w in anchor_numeric_words])


def _lane_of(x: float, lanes: list[float]) -> int | None:
    if not lanes:
        return None
    best = min(range(len(lanes)), key=lambda i: abs(lanes[i] - x))
    return best if abs(lanes[best] - x) <= _snap() else None


def _table_geometry(found, src_rows):
    """``(lanes, core_ys)`` of one block, or None when table membership is unclear.

    Lanes are the x-clusters of numeric words in paired rows that carry >= 2
    numeric words, kept only when >= 2 paired rows use them. A paired row is
    CORE when it occupies >= 2 of those lanes. Nothing else (word count, width, a
    Notes/Source opener) changes core membership: a false DEFER costs one model call,
    a missed fault can ship a wrong number. A prose line that paired by one stray
    number (``p<0.01``) has one numeric word and is not core.
    """
    multi = [(y, _numeric_words(src_rows[y])) for _i, y in found]
    multi = [(y, ws) for y, ws in multi if len(ws) >= 2]
    if len(multi) < 2:
        return None
    centres = _lanes([w for _y, ws in multi for w in ws])
    support: Counter = Counter()
    for _y, ws in multi:
        for lane in {_lane_of(w[0], centres) for w in ws} - {None}:
            support[lane] += 1
    lanes = [c for i, c in enumerate(centres) if support[i] >= 2]
    if not lanes:
        return None

    def lane_count(ws: list) -> int:
        return len({_lane_of(w[0], lanes) for w in ws} - {None})

    core = [y for y, ws in multi if lane_count(ws) >= _MIN_CORE_LANES]
    if len(core) < 2:
        return None
    return lanes, core


def extended_span(
    core: list[int],
    lanes: list[float],
    src_rows: dict,
    strong_k: int,
    panel_gap_rows: float | None = _PANEL_GAP_ROWS,
) -> tuple[int, int]:
    """The table's vertical span: its core rows, extended outward to full-width rows.

    A dropped first/last data row or a dropped panel is bracketed by no paired row,
    so the span must reach past the first/last core row. Walk outward; only a
    FULL-WIDTH row (>= ``strong_k`` lanes, the table's own modal width) extends the
    span, and only if it is within ``reach`` of the current edge, where ``reach`` is
    ``panel_gap_rows`` row pitches (measured, see ``_PANEL_GAP_ROWS``). Prose or labels in between
    neither extend the span nor bridge to a numeric line further away.
    ``panel_gap_rows=None`` removes the bound (the benchmark's structural variant).
    """
    y_lo, y_hi = min(core), max(core)

    def lane_count(y: int) -> int:
        words = _numeric_words(src_rows[y])
        return len({_lane_of(w[0], lanes) for w in words} - {None})

    ys_sorted = sorted(core)
    pitch = statistics.median([b - a for a, b in zip(ys_sorted, ys_sorted[1:])] or [0])
    # The bound is the whole reach: no floor, so bound 0 extends nothing and the
    # benchmark's sweep measures exactly what the constant does.
    reach = float("inf") if panel_gap_rows is None else panel_gap_rows * pitch
    for y in sorted((y for y in src_rows if y < y_lo), reverse=True):
        if lane_count(y) >= strong_k:
            if y_lo - y > reach:
                break
            y_lo = y
    for y in sorted(y for y in src_rows if y > y_hi):
        if lane_count(y) >= strong_k:
            if y - y_hi > reach:
                break
            y_hi = y
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


def table_spans(blocks, anchors: _Anchors, src_rows, panel_gap_rows=_PANEL_GAP_ROWS):
    """``[(lanes, core, y_lo, y_hi)]`` for each block whose table membership is established.

    Two different reaches, kept separate:

    * OUTWARD, beyond a table's first and last block: governed by ``panel_gap_rows``
      (``extended_span``). Bound 0 means no outward extension.
    * BETWEEN consecutive blocks of the SAME table (lane sets consistent, see
      ``_same_table_lanes``): the interior is inside the table by construction and is
      always covered, whatever the bound. A row omitted from the gap between two blocks
      has the table on both sides.

    ``strong_k`` (the width a row needs to extend the span outward) is the table's
    modal DISTINCT-LANE count over its core rows, with the rowizer's own minimum.
    """
    spans = []
    for found in anchors.per_block:
        geo = _table_geometry(found, src_rows)
        if geo is None:
            continue
        lanes, core = geo

        def lane_count(y: int, lanes=lanes) -> int:
            return len({_lane_of(w[0], lanes) for w in _numeric_words(src_rows[y])} - {None})

        modal = Counter(lane_count(y) for y in core).most_common(1)[0][0]
        strong_k = max(modal, _MIN_LANES_PER_ROW)
        y_lo, y_hi = extended_span(core, lanes, src_rows, strong_k, panel_gap_rows)
        spans.append([lanes, core, y_lo, y_hi])
    order = sorted(range(len(spans)), key=lambda i: min(spans[i][1]))
    for pos, i in enumerate(order):
        for j in order[pos + 1 :]:
            if _same_table_lanes(spans[i][0], spans[j][0]):
                # consecutive blocks of one table: cover everything between their cores
                spans[i][3] = max(spans[i][3], min(spans[j][1]))
                spans[j][2] = min(spans[j][2], max(spans[i][1]))
                break
    return [tuple(sp) for sp in spans]


def data_row_missing_faults(
    blocks, anchors: _Anchors, src_rows, panel_gap_rows: float | None = _PANEL_GAP_ROWS
) -> list[dict]:
    """A source row inside the table's own vertical span with no output row left.

    Membership is spatial: strictly between the first and last CORE paired row
    and in >= 2 of the table's lanes. The grid's numbers are counted, not merely
    found, so a dropped copy of a repeated row is a fault.
    """
    faults: list[dict] = []
    # Numeric tokens the grid carries and no paired row has claimed. A candidate
    # source row is present only if ALL its numbers are still available here, and
    # claiming them uses them up, so a dropped copy of a repeated row is missing
    # and a row whose numbers sit merged inside another row's cells is present.
    pool: Counter = Counter()
    for block in blocks:
        for cells in block:
            pool.update(_normalize_numeric_token(t) for t in _row_tokens(cells))
    for block, found in zip(blocks, anchors.per_block):
        for idx, _y in found:
            pool.subtract(_normalize_numeric_token(t) for t in _row_tokens(block[idx]))
    anchor_ys = {y for found in anchors.per_block for _i, y in found}
    spans = table_spans(blocks, anchors, src_rows, panel_gap_rows)
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
        k = max(len({_lane_of(w[0], L) for w in numeric} - {None}) for L in applicable)
        if k < _MIN_CORE_LANES:
            continue
        need = Counter(_normalize_numeric_token(w[4]) for w in in_lane)
        if all(pool[t] >= n for t, n in need.items()):
            pool.subtract(need)
            continue
        faults.append(
            {
                "predicate": DATA_ROW_MISSING,
                "detail": (
                    f"source row at y={y} with {k} numeric lane(s) "
                    f"({', '.join(w[4] for w in in_lane[:6])}) has no output row"
                ),
            }
        )
    return faults


class _CellText:
    """Per-cell normalised text of one output block, with occurrence counting."""

    def __init__(self, block) -> None:
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
                self.avail[i] = cell[:j] + _USED + cell[j + len(text) :]
                return True
        return False if any(text in cell for cell in self.original) else None

    def has_word(self, word: str) -> bool:
        w = _compact(word)
        return any(w in cell for cell in self.original)


def label_row_missing_faults(blocks, anchors: _Anchors, src_rows) -> list[dict]:
    faults: list[dict] = []
    for block, found in zip(blocks, anchors.per_block):
        geo = _table_geometry(found, src_rows)
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
            if not inside or any(_is_num(w[4]) for w in inside):
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
                got = text.take(_compact("".join(variant)))
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
                    {
                        "predicate": LABEL_ROW_MISSING,
                        "detail": (
                            f"source label row at y={y} inside the table has word(s) "
                            f"missing from the grid: {', '.join(missing[:6])}"
                        ),
                    }
                )
    return faults


def native_ship_gate(words: list, markdown: str) -> tuple[dict, ...]:
    """Faults found by the ship gate, or ``()`` when the grid may ship.

    A gate that raises must not ship the grid: it reports ``gate_error`` so the
    caller defers to normal routing.
    """
    try:
        blocks = _output_blocks(markdown)
        if not blocks or not words:
            return ()
        src_rows = _source_rows(words)
        anchors = _Anchors(blocks, src_rows)
        faults: list[dict] = []
        faults += sign_detached_faults(words, blocks, src_rows)
        faults += order_faults(blocks, anchors, src_rows, cells=False)
        faults += order_faults(blocks, anchors, src_rows, cells=True)
        faults += data_row_missing_faults(blocks, anchors, src_rows)
        faults += label_row_missing_faults(blocks, anchors, src_rows)
        return tuple(faults)
    except Exception as exc:
        logger.warning("native ship gate failed (%s); deferring", exc)
        return ({"predicate": GATE_ERROR, "detail": f"{type(exc).__name__}: {exc}"},)
