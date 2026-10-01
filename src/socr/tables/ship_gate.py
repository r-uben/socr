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
    _SPLIT_GAP_MIN_PT,
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
#: A number that ends in one of these is punctuation inside a sentence ("for 7, 5,"),
#: not a table value; a source word like that does not count as numeric for the gate.
_SENTENCE_PUNCT = frozenset(",.;:")
#: Largest vertical gap, in row pitches, allowed between the table's edge row and
#: the next full-width row that extends its span. Derived from data, not picked:
#: on the 35 rotated + 92 upright native-first SHIP pages (1822 gaps between
#: consecutive core paired rows, each divided by its block's median gap) the
#: distribution is median 1.0, p90 2.0, p95 2.25, p99 4.83, with the tail beyond
#: it made of index-page "tables" (max 28.45). The p99 rounded up is 5.
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
    if text[-1:] in _SENTENCE_PUNCT:
        return False  # "7," / "2025." inside a sentence, not a table value
    if _NUM_TOKEN_RE.match(text) and _NUMERIC_RE.search(text):
        return True
    return text[:1] == "." and is_numeric_token(text)


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
                if pos >= len(seq) or seq[pos] != _normalize_numeric_token(m.group(0)):
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
    CORE when it occupies >= 2 of those lanes and is not prose-with-numbers (see
    ``off_lane`` below). A prose line that paired by one stray number (``p<0.01``)
    or carries a few numerals is not core, so a Notes paragraph swallowed into
    the grid does not stretch the table's span.
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
    # A table row has about one cell per lane in the lane region; a prose line that
    # happens to carry numbers ("for 7, 5, 3 and ...") has many more words there.
    # "Many more" is relative to this table's own paired rows: the median count of
    # non-numeric words in the lane region, plus one word per lane.
    lo, hi = min(lanes) - _snap(), max(lanes) + _snap()

    def off_lane(y: int) -> int:
        return sum(1 for w in src_rows[y] if lo <= w[0] <= hi and not _is_num(w[4]))

    allowed = statistics.median(off_lane(y) for y, _ws in multi) + len(lanes)
    core = [
        y
        for y, ws in multi
        if len({_lane_of(w[0], lanes) for w in ws} - {None}) >= 2 and off_lane(y) <= allowed
    ]
    if len(core) < 2:
        return None
    return lanes, core


def data_row_missing_faults(blocks, anchors: _Anchors, src_rows) -> list[dict]:
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
    geos = [_table_geometry(found, src_rows) for found in anchors.per_block]
    for geo in geos:
        if geo is None:
            continue  # table membership not established: abstain
        lanes, core = geo
        y_lo, y_hi = min(core), max(core)
        modal = Counter(len(_numeric_words(src_rows[y])) for y in core).most_common(1)[0][0]
        strong_k = max(modal, _MIN_LANES_PER_ROW)

        def lane_hits(y: int) -> list:
            return [w for w in _numeric_words(src_rows[y]) if _lane_of(w[0], lanes) is not None]

        def lane_count(y: int) -> int:
            return len({_lane_of(w[0], lanes) for w in lane_hits(y)})

        # A dropped first/last data row or a dropped panel is bracketed by no paired
        # row, so the span must reach past the first/last core row. How far is
        # bounded by _PANEL_GAP_ROWS row pitches (measured, see its comment) or by
        # the gap between its own blocks.
        ys_sorted = sorted(core)
        pitch = statistics.median([b - a for a, b in zip(ys_sorted, ys_sorted[1:])] or [0])
        # Blocks of the same table (same lane count) show how far apart its own
        # blocks sit; that gap is allowed too.
        peers = sorted(y for g in geos if g is not None and len(g[0]) == len(lanes) for y in g[1])
        own_gap = max([b - a for a, b in zip(peers, peers[1:])] or [0])
        reach = max(_PANEL_GAP_ROWS * pitch, _SPLIT_GAP_MIN_PT, own_gap)
        # Walk outward; only a FULL-WIDTH row (as many lanes as the table's own
        # modal row) extends the span, and only if it is within ``reach`` of the
        # current edge. Prose or labels in between neither extend the span nor
        # bridge to a numeric line further away.
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
        for y, ws in sorted(src_rows.items()):
            if y in anchor_ys or not (y_lo <= y <= y_hi):
                continue
            in_lane = lane_hits(y)
            k = lane_count(y)
            if k < 2:
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
