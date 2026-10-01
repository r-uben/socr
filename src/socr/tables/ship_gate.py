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

``sign_detached``   an output cell is a lone sign (or a populated cell ends in
                    one) followed by a digit-leading cell, AND the source prints
                    that sign in contact with that number's digits
                    (``detached_sign_pairs``, the #887 criterion shared with the
                    ``find_tables`` merge) on a source line bound to THIS output row.
``row_order``       rows that pair to exactly one source row by numeric multiset
                    must appear in increasing source y.
``cell_order``      on such a pair the output's left-to-right numeric cells must
                    equal the source's x-sorted numeric words.
``data_row_missing``  a source row that belongs to the table by its OWN lanes (the
                    lanes of rows that pair uniquely) and has no output row.
``label_row_missing`` a numeric-free source row between the first and last paired
                    row, inside the table's x-extent, with a word the output lacks.

Every tolerance is an existing named rowizer quantity (``_LANE_X_TOL_PT`` x
``_LANE_SNAP_MULT``, ``_MIN_LANES_PER_ROW``); nothing here is a new threshold.
Order checks abstain when a row's multiset is not unique on either side. Coverage
is derived from source geometry anchored on unique pairs, never from the
output-derived y-band the value guard uses.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict

from socr.tables.native_verifier import (
    _MD_SEP_RE,
    _cluster_x_positions,
    _normalize_numeric_token,
    _numeric_tokens_from_text,
    _parse_output_row_cells,
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

_LEADING_NUMBER_RE = re.compile(r"^\(?\d[\d,]*(?:\.\d+)?")


def _compact(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace(" ", "")


def _is_num(text: str) -> bool:
    return bool(_NUM_TOKEN_RE.match(text) and _NUMERIC_RE.search(text))


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
    pairs = detached_sign_pairs(words)
    if not pairs:
        return []
    src_counters: dict[int, Counter] = {}
    for y, ws in src_rows.items():
        nums = _numeric_words(ws)
        if nums:
            src_counters[y] = Counter(_key(w[4] for w in nums))
    faults: list[dict] = []
    for block in blocks:
        for cells in block:
            row_counter = Counter(_key(_row_tokens(cells)))
            bound = [y for y, c in src_counters.items() if not (row_counter - c)]
            if not bound:
                continue  # no source line carries this row: abstain
            for i in range(len(cells) - 1):
                tail = cells[i].split()
                if not tail or tail[-1] not in _SIGN_GLYPHS:
                    continue
                m = _LEADING_NUMBER_RE.match(cells[i + 1].strip())
                if not m:
                    continue
                num = _normalize_numeric_token(m.group(0))
                for s, d in pairs:
                    dm = _LEADING_NUMBER_RE.match(d[4])
                    if round(d[1]) in bound and dm and _normalize_numeric_token(dm.group(0)) == num:
                        faults.append(
                            {
                                "predicate": SIGN_DETACHED,
                                "detail": (
                                    f"cell {i} is a bare sign before {num!r}; the PDF prints "
                                    "the sign in contact with that number"
                                ),
                            }
                        )
                        break
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
    best = min(range(len(lanes)), key=lambda i: abs(lanes[i] - x))
    return best if abs(lanes[best] - x) <= _snap() else None


def data_row_missing_faults(blocks, anchors: _Anchors, src_rows) -> list[dict]:
    faults: list[dict] = []
    out_pool: Counter = Counter()
    for block in blocks:
        for cells in block:
            k = _key(_row_tokens(cells))
            if k:
                out_pool[k] += 1
    anchor_ys = {y for found in anchors.per_block for _i, y in found}
    for block, found in zip(blocks, anchors.per_block):
        if len(found) < 2:
            continue  # table membership not established: abstain
        anchor_words = [w for _i, y in found for w in _numeric_words(src_rows[y])]
        lanes = _lanes(anchor_words)
        widths = Counter(len(_numeric_words(src_rows[y])) for _i, y in found)
        modal = widths.most_common(1)[0][0]
        y_lo = min(y for _i, y in found)
        y_hi = max(y for _i, y in found)
        header_tokens = Counter(_key(_row_tokens(block[0])))
        for y, ws in sorted(src_rows.items()):
            if y in anchor_ys:
                continue
            in_lane = [w for w in _numeric_words(ws) if _lane_of(w[0], lanes) is not None]
            k = len({_lane_of(w[0], lanes) for w in in_lane})
            member = (k >= 2 and y_lo < y < y_hi) or k >= max(modal, _MIN_LANES_PER_ROW)
            if not member:
                continue
            row_key = _key(w[4] for w in in_lane)
            if out_pool.get(row_key):
                continue
            if not (Counter(row_key) - header_tokens):
                continue  # absorbed into the column header
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


def label_row_missing_faults(blocks, anchors: _Anchors, src_rows) -> list[dict]:
    faults: list[dict] = []
    for block, found in zip(blocks, anchors.per_block):
        if len(found) < 2:
            continue
        out_tokens = {tok for cells in block for cell in cells for tok in cell.split()}
        # NFKC folds typographic ligatures (U+FB00 "ff") so the PDF's "Staff"
        # and the grid's "Staff" are the same word.
        compact = _compact("".join(c for cells in block for c in cells))
        anchor_ys = {y for _i, y in found}
        # The table's x-extent is the bounding box of the anchored rows' words
        # that the shipped grid carries. Words of another text column that
        # share a y-row never appear in the grid, so they do not widen it.
        carried = [w for y in anchor_ys for w in src_rows[y] if w[4] in out_tokens]
        if not carried:
            continue
        x_lo = min(w[0] for w in carried)
        x_hi = max(w[2] for w in carried)
        y_lo, y_hi = min(anchor_ys), max(anchor_ys)
        for y, ws in sorted(src_rows.items()):
            if not (y_lo < y < y_hi) or y in anchor_ys:
                continue
            inside = [w for w in ws if w[0] >= x_lo and w[2] <= x_hi]
            if not inside or any(_is_num(w[4]) for w in inside):
                continue
            absent = [w[4] for w in inside if _compact(w[4]) not in compact]
            if absent:
                faults.append(
                    {
                        "predicate": LABEL_ROW_MISSING,
                        "detail": (
                            f"source label row at y={y} inside the table has word(s) "
                            f"missing from the grid: {', '.join(absent[:6])}"
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
