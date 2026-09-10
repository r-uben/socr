"""Deterministic repair for malformed multi-band table headers (GH-56/GH-276).

When a VLM (or a spanning-header merge) collapses probability-bin / range
headers into one markdown cell while data rows retain the full column count,
header-to-value binding breaks and the native verifier flags the table.

This module rebuilds the header row from born-digital word geometry: data
column lanes are derived from native numeric-token x-positions in rows that
match the OCR output's values; header words above those rows are snapped into
per-lane cells.  Multi-line headers are then merged with the same
``_collapse_header_prefix`` logic used by the rowizer.

For a narrower spanning-header prefix above a unanimous wider body (GH-276),
the module instead widens the prefix arithmetically.  Moving a nonblank group
label requires native geometry; otherwise the missing-cell placement is
ambiguous and the repair abstains.

No model call; both paths use only the markdown and, where needed,
``page.get_text("words")`` (or a pre-fetched list).
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from socr.tables.native_verifier import (
    _numeric_multiset_from_tokens,
)
from socr.tables.reconcile import find_table_blocks
from socr.tables.reconstruct import (
    _LANE_SNAP_MULT,
    _LANE_X_TOL_PT,
    _NUM_TOKEN_RE,
    _NUMERIC_RE,
    _SPLIT_GAP_MIN_PT,
    _SPLIT_GAP_MULT,
    _grid_to_markdown,
)

logger = logging.getLogger(__name__)

# Range / bin markers in probability-band and binned numeric table headers.
_RANGE_MARKER_RE = re.compile(r"%|\+/-|[<>]")
_ORDINAL_CELL_RE = re.compile(r"^(?:\((\d+)\)|(\d+))$")

# Minimum gap between header column count and data column count before repair
# is attempted.  Set to 2: a single-column stub label (Forecaster + data cols)
# is a legitimate 1-col difference and must not trigger repair.
_MIN_HEADER_DATA_COL_GAP = 2

# Minimum numeric cells in a data row for lane derivation and anchor matching.
_MIN_DATA_NUMERIC_CELLS = 3


def detect_header_column_collapse(grid: list[list[str]]) -> tuple[bool, int, int]:
    """Return whether the grid's header has fewer columns than its data body.

    Returns ``(collapsed, header_cols, expected_cols)``.  ``expected_cols`` is
    the modal width of "data" rows (rows with >= ``_MIN_DATA_NUMERIC_CELLS``
    numeric tokens anywhere in the row).
    """
    if len(grid) < 2:
        return False, 0, 0

    header_cols = len(grid[0])
    data_widths: list[int] = []
    for row in grid[1:]:
        numeric_count = sum(
            1
            for cell in row
            if cell.strip() and _NUM_TOKEN_RE.match(cell.strip()) and _NUMERIC_RE.search(cell)
        )
        if numeric_count >= _MIN_DATA_NUMERIC_CELLS:
            data_widths.append(len(row))

    if not data_widths:
        return False, header_cols, header_cols

    expected_cols = Counter(data_widths).most_common(1)[0][0]
    gap = expected_cols - header_cols
    collapsed = gap >= _MIN_HEADER_DATA_COL_GAP
    return collapsed, header_cols, expected_cols


def _repair_too_narrow_spanning_header(
    grid: list[list[str]],
    words: list,
) -> list[list[str]] | None:
    """Widen an unambiguous spanning-header prefix to the body's width.

    This is the narrow spanning-band GH-276 case, distinct from GH-56's
    collapsed header reconstruction.  The wider width must be the unique modal
    width below the markdown header, and every row from its first occurrence
    onward must already have that width.  That makes a single anomalously wide
    body row an abstention rather than a reason to widen the table.

    Exactly one secondary header band, a single primary banner, and a
    full-width sequential ordinal row are required.  The secondary band must
    expose two group labels separated by a blank, with the second label at the
    source row's right edge.  Its native x position identifies the target data
    lane; all other placements abstain.
    """
    if len(grid) < 3:
        return None

    header_cols = len(grid[0])
    widths = Counter(len(row) for row in grid[1:])
    ranked_widths = widths.most_common()
    if not ranked_widths:
        return None

    expected_cols, support = ranked_widths[0]
    if expected_cols <= header_cols:
        return None
    if len(ranked_widths) > 1 and ranked_widths[1][1] == support:
        return None

    first_full_width = next(
        (idx for idx, row in enumerate(grid[1:], start=1) if len(row) == expected_cols),
        None,
    )
    # Exactly one secondary narrow band distinguishes this case from a single
    # collapsed header and keeps body rows from being reclassified as headers.
    if first_full_width != 2:
        return None
    if any(len(row) != expected_cols for row in grid[first_full_width:]):
        return None
    if any(len(row) != header_cols for row in grid[:first_full_width]):
        return None
    primary_nonblank = [cell for cell in grid[0][1:] if cell.strip()]
    secondary = grid[1]
    secondary_nonblank = [idx for idx, cell in enumerate(secondary) if cell.strip()]
    if (
        grid[0][0].strip()
        or len(primary_nonblank) != 1
        or secondary[0].strip()
        or len(secondary_nonblank) != 2
        or secondary_nonblank[-1] != len(secondary) - 1
        or not any(not cell.strip() for cell in secondary[1:-1])
        or not _is_sequential_ordinal_row(grid[first_full_width])
    ):
        return None
    if any(
        _NUM_TOKEN_RE.match(cell.strip()) and _NUMERIC_RE.search(cell)
        for row in grid[1:first_full_width]
        for cell in row
        if cell.strip()
    ):
        # A short numeric row is body data, not evidence of a spanning header.
        return None

    deficit = expected_cols - header_cols
    repaired: list[list[str]] = [list(grid[0]) + [""] * deficit]
    for row in grid[1:first_full_width]:
        widened = list(row)
        lane_idx = _native_label_lane(grid, widened[-1], words)
        if lane_idx is None:
            return None
        current_cell = len(widened) - 1
        target_cell = lane_idx + 1  # account for the blank stub column
        insert_before = target_cell - current_cell
        insert_after = deficit - insert_before
        if insert_before < 0 or insert_after < 0:
            return None
        widened = widened[:-1] + [""] * insert_before + [widened[-1]] + [""] * insert_after
        repaired.append(widened)

    repaired.extend(list(row) for row in grid[first_full_width:])
    return repaired


def _is_sequential_ordinal_row(row: list[str]) -> bool:
    """Whether *row* is a blank stub followed by 1..N column ordinals."""
    if len(row) < 2 or row[0].strip():
        return False
    ordinals: list[int] = []
    for cell in row[1:]:
        match = _ORDINAL_CELL_RE.fullmatch(cell.strip())
        if match is None:
            return False
        ordinals.append(int(match.group(1) or match.group(2)))
    return ordinals == list(range(1, len(row)))


def _all_rows_by_y(words: list) -> dict[int, list]:
    """Group all words by rounded y0 (not just numeric tokens)."""
    row_map: dict[int, list] = defaultdict(list)
    for w in words:
        row_map[round(w[1])].append(w)
    for y in row_map:
        row_map[y].sort(key=lambda w: w[0])
    return row_map


def _row_numeric_multiset(row_words: list) -> Counter:
    return _numeric_multiset_from_tokens([w[4] for w in row_words])


def _best_anchor_y(
    rows_by_y: dict[int, list],
    grid: list[list[str]],
) -> float | None:
    """Find the native y-row whose numeric multiset exactly matches a data row."""
    for row in grid[1:]:
        out_ms = _numeric_multiset_from_tokens(row)
        if len(out_ms) < _MIN_DATA_NUMERIC_CELLS:
            continue
        for y in sorted(rows_by_y.keys()):
            nat_ms = _row_numeric_multiset(rows_by_y[y])
            if out_ms == nat_ms:
                return float(y)
    return None


def _median_row_gap(ys: list[int]) -> float:
    if len(ys) < 2:
        return _SPLIT_GAP_MIN_PT
    gaps = [b - a for a, b in zip(ys, ys[1:]) if b > a]
    return statistics.median(gaps) if gaps else _SPLIT_GAP_MIN_PT


def _local_table_ys(rows_by_y: dict[int, list], anchor_y: int) -> list[int]:
    """Y-groups in the table neighbourhood around *anchor_y* (not the whole page).

    Dense CE pages have sub-point name/value offsets that drive the page-wide
    median gap down to ~1 pt, which makes a page-wide split threshold too small
    to bridge real header-to-data spacing (~25 pt).  Restrict gap estimation to
    rows whose words overlap the anchor row's x-extent.
    """
    anchor_words = rows_by_y.get(anchor_y, [])
    if not anchor_words:
        return [anchor_y]
    x0 = min(w[0] for w in anchor_words) - 20.0
    x1 = max(w[2] for w in anchor_words) + 20.0
    local: list[int] = []
    for y, row_words in rows_by_y.items():
        if any(x0 <= w[0] <= x1 or x0 <= w[2] <= x1 for w in row_words):
            local.append(y)
    return sorted(local) if local else [anchor_y]


def _is_table_header_row(
    row_words: list,
    lane_centers: list[float],
    data_start_x: float,
) -> bool:
    """True when *row_words* looks like a column-metadata header (not a title row).

    Requires at least two lane-aligned tokens AND at least one range/bin marker
    (``%``, ``+/-``, ``<``, ``>``) or connector word (``to``, ``or``, ``more``,
    ``less``) among the lane-aligned tokens.  Section titles like "Foreign
    Exchange Rates" lack these markers and are excluded.
    """
    if _lane_aligned_word_count(row_words, lane_centers, data_start_x) < 2:
        return False
    snap_margin = _LANE_X_TOL_PT * _LANE_SNAP_MULT
    connectors = frozenset({"to", "or", "more", "less"})
    for w in row_words:
        if w[0] < data_start_x - snap_margin:
            continue
        if not any(abs(lane_centers[i] - w[0]) <= snap_margin for i in range(len(lane_centers))):
            continue
        text = w[4]
        if _RANGE_MARKER_RE.search(text) or text in connectors:
            return True
    return False


def _lane_aligned_word_count(
    row_words: list,
    lane_centers: list[float],
    data_start_x: float,
) -> int:
    """Count words in *row_words* that snap to a data column lane."""
    snap_margin = _LANE_X_TOL_PT * _LANE_SNAP_MULT
    count = 0
    for w in row_words:
        if w[0] < data_start_x - snap_margin:
            continue
        if any(abs(lane_centers[i] - w[0]) <= snap_margin for i in range(len(lane_centers))):
            count += 1
    return count


def _header_bridge_gap(local_ys: list[int], anchor_y: int, split_threshold: float) -> float:
    """Maximum y-gap allowed between the anchor data row and the first header row.

    Label-only rows (e.g. ``Euro1`` at x≈40) sit between the header band and the
    data values at x≈160+ but are excluded from ``local_ys`` because they fall
    outside the anchor row's x-extent.  The bridge gap is derived from the
    table-local y distribution: the distance from the anchor to the nearest local
    row above it, capped below by ``split_threshold``.
    """
    below = sorted(y for y in local_ys if y < anchor_y)
    if not below:
        return split_threshold
    anchor_to_nearest = anchor_y - below[-1]
    consecutive = [below[i] - below[i - 1] for i in range(1, len(below))]
    return max(split_threshold, anchor_to_nearest, max(consecutive) if consecutive else 0.0)


def _header_ys(
    rows_by_y: dict[int, list],
    local_ys: list[int],
    anchor_y: int,
    split_threshold: float,
    lane_centers: list[float],
    data_start_x: float,
    bands: list[tuple[float, float]] | None = None,
) -> list[int]:
    """Collect multi-line header y-groups directly above the data anchor.

    Only rows with at least two lane-aligned words qualify as table-header
    metadata (probability-bin labels, year rows, etc.).  Prose lines above the
    table — which lack lane-aligned tokens — terminate the upward scan even when
    the vertical gap is small.
    """
    below = sorted(y for y in local_ys if y < anchor_y)
    if not below:
        return []

    bridge_gap = _header_bridge_gap(local_ys, anchor_y, split_threshold)

    # #696: the marker/connector requirement in ``_is_table_header_row`` was
    # written for probability-bin bands ("0-2%", "5 or more") and does not
    # recognise a period band ("Apr 18 | Jul 18 | ..."), which is the standard
    # layout of every central-bank survey annex in the corpus — so the whole
    # chain abstained on all nine pages issue #696 measured. ``_is_lane_band_row``
    # is the disjunctive second term: runs that live inside the data columns
    # and carry text. Both terms still reject prose, which straddles the left
    # edge of the columns, and data rows, whose lane runs are all numeric.
    #
    # ``bands=None`` keeps the original predicate alone. The caller asks for the
    # widened one ONLY to feed the flattening fold, which abstains on anything
    # that is not a clean spanning band — so the legacy repair path never sees a
    # row this term admitted and stays byte-identical.

    def _is_header(y: int, min_runs: int) -> bool:
        row_words = rows_by_y.get(y, [])
        if _is_table_header_row(row_words, lane_centers, data_start_x):
            return True
        return bands is not None and _is_lane_band_row(row_words, bands, min_runs)

    # Start at the header row nearest to the anchor (label-only rows between
    # header and data are absent from local_ys).
    nearest: int | None = None
    for y in reversed(below):
        if anchor_y - y > bridge_gap:
            break
        if _is_header(y, 2):
            nearest = y
            break
    if nearest is None:
        return []

    # Collect contiguous header lines upward from the nearest band. One run is
    # enough here: a group heading that wraps ("Loans to small" over "and
    # medium-sized enterprises") puts a single run on its topmost line.
    header: list[int] = [nearest]
    prev = nearest
    for y in reversed([y for y in below if y < nearest]):
        if prev - y > split_threshold:
            break
        if _is_header(y, 1):
            header.append(y)
            prev = y
        else:
            break
    return sorted(header)


def _derive_lane_groups(
    rows_by_y: dict[int, list], data_ys: list[int]
) -> list[list[tuple[float, float]]]:
    """Cluster numeric tokens from native data rows into column lanes.

    Each lane is a list of ``(x0, x1)`` at DISTINCT x0 — the same distinct-x0
    clustering ``_derive_lane_centers`` has always done, with the token's right
    edge carried along so a caller can see how wide the printed column is.
    """
    per_row: list[list[tuple[float, float]]] = []
    for y in data_ys:
        row_words = rows_by_y.get(y, [])
        # Skip rows with fewer than _MIN_DATA_NUMERIC_CELLS — these are header
        # annotations or chart ticks, not probability-bin data values.
        if len(_row_numeric_multiset(row_words)) < _MIN_DATA_NUMERIC_CELLS:
            continue
        per_row.append(
            [
                (w[0], w[2])
                for w in row_words
                if _NUM_TOKEN_RE.match(w[4]) and _NUMERIC_RE.search(w[4])
            ]
        )
    if not per_row:
        return []

    # #696: ``_data_row_ys`` walks downward from the anchor and bridges rows it
    # does not accept, so on a page whose footnotes sit within one row gap of
    # the last data row it hands us a prose line as well. A footnote such as
    # "(score of 1) ... weights from 1 to 5" carries enough numerals to clear
    # _MIN_DATA_NUMERIC_CELLS and its numerals sit at x positions no data
    # column occupies, inventing lanes on both sides of the table (measured on
    # the 2018 BLS survey: 13 lanes derived for a 10-column table). Lanes are a
    # property of the table's OWN rows, so derive them only from rows at least
    # as wide as the modal accepted row. Ties resolve to the wider count, which
    # can only ever keep more lanes, never fewer.
    tally = Counter(len(xs) for xs in per_row)
    modal_width = max(tally.items(), key=lambda kv: (kv[1], kv[0]))[0]
    right_of: dict[float, float] = {}
    for row in per_row:
        if len(row) < modal_width:
            continue
        for x0, x1 in row:
            right_of[x0] = max(right_of.get(x0, x1), x1)
    if not right_of:
        return []

    xs_sorted = sorted(right_of)
    lanes: list[list[float]] = []
    for x in xs_sorted:
        if lanes and x - lanes[-1][-1] <= _LANE_X_TOL_PT:
            lanes[-1].append(x)
        else:
            lanes.append([x])
    return [[(x, right_of[x]) for x in group] for group in lanes]


def _derive_lane_centers(rows_by_y: dict[int, list], data_ys: list[int]) -> list[float]:
    """Cluster numeric-token x-positions from native data rows into column lanes."""
    return [
        sum(x0 for x0, _x1 in group) / len(group)
        for group in _derive_lane_groups(rows_by_y, data_ys)
    ]


def _native_label_lane(grid: list[list[str]], label: str, words: list) -> int | None:
    """Return the native data-lane index occupied by *label*, or abstain.

    The label must appear as one contiguous native word sequence above an
    exact numeric anchor row.  Its horizontal centre is assigned only when it
    lies inside the data lanes' dynamically derived outer half-gaps.
    """
    if not words:
        return None
    rows_by_y = _all_rows_by_y(words)
    anchor_y = _best_anchor_y(rows_by_y, grid)
    if anchor_y is None:
        return None

    anchor_y_int = round(anchor_y)
    local_ys = _local_table_ys(rows_by_y, anchor_y_int)
    split_threshold = max(_SPLIT_GAP_MULT * _median_row_gap(local_ys), _SPLIT_GAP_MIN_PT)
    expected_lanes = max(len(row) for row in grid) - 1
    if expected_lanes < 2:
        return None
    data_ys = _data_row_ys(rows_by_y, anchor_y_int, split_threshold, local_ys)
    full_width_ys = [
        y
        for y in data_ys
        if sum(
            1
            for word in rows_by_y.get(y, [])
            if _NUM_TOKEN_RE.match(word[4]) and _NUMERIC_RE.search(word[4])
        )
        == expected_lanes
    ]
    if not full_width_ys:
        return None
    lane_centers = _derive_lane_centers(rows_by_y, full_width_ys)
    if len(lane_centers) != expected_lanes:
        return None

    label_tokens = re.findall(r"[\w&]+", label.casefold())
    if not label_tokens:
        return None

    candidate_centres: list[float] = []
    for y, row_words in rows_by_y.items():
        if y >= anchor_y_int:
            continue
        native_tokens = [w[4].casefold() for w in row_words]
        span = len(label_tokens)
        for start in range(len(native_tokens) - span + 1):
            if native_tokens[start : start + span] != label_tokens:
                continue
            matched = row_words[start : start + span]
            candidate_centres.append((matched[0][0] + matched[-1][2]) / 2)

    if not candidate_centres:
        return None

    left_gap = lane_centers[1] - lane_centers[0]
    right_gap = lane_centers[-1] - lane_centers[-2]
    lower_bound = lane_centers[0] - left_gap / 2
    upper_bound = lane_centers[-1] + right_gap / 2
    lane_indices: set[int] = set()
    for centre in candidate_centres:
        if not lower_bound <= centre <= upper_bound:
            continue
        distances = [abs(centre - lane) for lane in lane_centers]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        if distances.count(distances[nearest]) != 1:
            return None
        lane_indices.add(nearest)

    if len(lane_indices) != 1:
        return None
    return lane_indices.pop()


def _assign_words_to_lanes(
    row_words: list,
    lane_centers: list[float],
    data_start_x: float,
) -> list[str]:
    """Map one native row's words into label + per-lane header cells."""
    snap_margin = _LANE_X_TOL_PT * _LANE_SNAP_MULT
    label_words = [w[4] for w in row_words if w[0] < data_start_x - snap_margin]
    label = " ".join(label_words).strip()

    row_cells = [""] * len(lane_centers)
    for w in row_words:
        if w[0] < data_start_x - snap_margin:
            continue
        best = min(range(len(lane_centers)), key=lambda i: abs(lane_centers[i] - w[0]))
        if abs(lane_centers[best] - w[0]) <= snap_margin:
            existing = row_cells[best]
            row_cells[best] = (existing + " " + w[4]).strip() if existing else w[4]

    return [label] + row_cells


# --------------------------------------------------------------------------
# #696 — spanning group headings, flattened into the columns beneath them
#
# Owner ruling (2026-09-10, issue #696): a group heading that spans several
# data columns folds into each of them, so a two-level header ships as ONE
# header row of per-column names ("Overall Apr 18", "Overall Jul 18", ...).
# Markdown has no spanning cell, so the alternative — the model's own two-band
# emission — is a header row narrower than its body, which is precisely the
# `grid_shape` / `header_unattributed` refusal that lost 9 of 30 pages in the
# 2026-09-06 ECB census.
#
# The span of a heading is read off geometry that is already here: the numeric
# tokens this module already clusters into lanes, widened into contiguous
# column BANDS at the middle of each printed gutter. Every column then goes to
# the heading covering most of it — a comparison, not a tolerance, so no tuned
# constant is involved and none is added.
#
# Segmentation into headings uses the (block, line) identity PyMuPDF already
# attaches to every word, not an x-gap threshold: on the census pages each
# group heading and each leaf label is its own (block, line), including the
# ones that wrap over three typeset lines. A PDF whose producer emits every
# word as its own line degrades to per-word spans rather than failing.
# --------------------------------------------------------------------------


def _lane_bands(lane_groups: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Widen lane token clusters into contiguous column bands.

    An interior boundary is the middle of the GUTTER actually printed between
    two columns — halfway from the right edge of the widest value on the left
    to the left edge of the leftmost value on the right — not the midpoint
    between lane centres.  Centres are computed from token x0 alone, so on a
    table of right-aligned values of unequal width they sit left of the visual
    column and a midpoint boundary lands inside the printed text of the column
    to its right.  Measured on the 2018 BLS survey, that put the boundary 1.8pt
    left of the end of the group heading "and medium-sized" and folded it into
    a column it does not cover.

    The outer edges extend by the same half-gutter, mirroring the outer
    half-gap bound ``_native_label_lane`` already uses.  Returns ``[]`` for
    fewer than two lanes, where "which columns does this heading span" has no
    content.
    """
    if len(lane_groups) < 2:
        return []
    extents = [
        (min(x0 for x0, _x1 in group), max(x1 for _x0, x1 in group)) for group in lane_groups
    ]
    interior = [(a[1] + b[0]) / 2 for a, b in zip(extents, extents[1:])]
    left = extents[0][0] - (interior[0] - extents[0][1])
    right = extents[-1][1] + (extents[-1][0] - interior[-1])
    edges = [left] + interior + [right]
    return list(zip(edges, edges[1:]))


def _row_runs(row_words: list) -> list[tuple[float, float, str]]:
    """Split one native row into headings, using PyMuPDF's own word grouping.

    Returns ``(x0, x1, text)`` per run, left to right.  Words carry
    ``(block_no, line_no)`` at indices 5 and 6; a word tuple short enough to
    lack them (a hand-built fixture) is treated as its own run.
    """
    grouped: dict[tuple, list] = defaultdict(list)
    order: list[tuple] = []
    for i, w in enumerate(row_words):
        key = (w[5], w[6]) if len(w) > 6 else ("_w", i)
        if key not in grouped:
            order.append(key)
        grouped[key].append(w)

    runs: list[tuple[float, float, str]] = []
    for key in order:
        ws = sorted(grouped[key], key=lambda w: w[0])
        text = " ".join(w[4] for w in ws).strip()
        if not text:
            continue
        runs.append((min(w[0] for w in ws), max(w[2] for w in ws), text))
    return sorted(runs, key=lambda r: r[0])


def _run_lane_span(run: tuple[float, float, str], bands: list[tuple[float, float]]) -> list[int]:
    """Lane indices whose band overlaps *run*'s x-extent.

    A coarse admission test, used to decide whether a row belongs to the header
    band at all, and to drop a run reaching across every column (a table title,
    which heads none of them).  ``_claim_lanes`` is what actually places a run,
    and it resolves the slivers this returns.
    """
    x0, x1, _text = run
    span = [i for i, (a, b) in enumerate(bands) if x0 < b and x1 > a]
    if len(span) == len(bands) and len(bands) > 1:
        return []
    return span


def _split_row_runs(
    row_words: list,
    bands: list[tuple[float, float]],
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]] | None:
    """Partition one row's runs into ``(label_runs, lane_runs)``.

    Returns ``None`` when a run STRADDLES the left edge of the data columns.
    That is the signature of prose — a sentence set in the label column that
    runs on under the table's columns — and never of a header band, whose stub
    head stops before the first column and whose headings start inside them.
    """
    if not bands:
        return None
    data_left = bands[0][0]
    label_runs: list[tuple[float, float, str]] = []
    lane_runs: list[tuple[float, float, str]] = []
    for run in _row_runs(row_words):
        x0, x1, _text = run
        if x1 <= data_left:
            label_runs.append(run)
        elif x0 >= data_left:
            lane_runs.append(run)
        else:
            return None
    return label_runs, lane_runs


def _is_lane_band_row(row_words: list, bands: list[tuple[float, float]], min_runs: int) -> bool:
    """True when *row_words* is a header band lying inside the data columns.

    Requires at least *min_runs* runs that span at least one lane, and at
    least one non-numeric token among them.  An all-numeric lane row is a data
    row (or a bare year band, which ``_is_table_header_row`` does not reach
    either — this predicate is additive and widens nothing there).
    """
    split = _split_row_runs(row_words, bands)
    if split is None:
        return False
    _label_runs, lane_runs = split
    spanning = [run for run in lane_runs if _run_lane_span(run, bands)]
    if len(spanning) < min_runs:
        return False
    return any(
        not (_NUM_TOKEN_RE.match(tok) and _NUMERIC_RE.search(tok))
        for _x0, _x1, text in spanning
        for tok in text.split()
    )


def _is_lane_data_row(row_words: list, bands: list[tuple[float, float]]) -> bool:
    """True when every run inside the data columns is a bare numeric value."""
    split = _split_row_runs(row_words, bands)
    if split is None:
        return False
    _label_runs, lane_runs = split
    if len(lane_runs) < 2:
        return False
    return all(
        _NUM_TOKEN_RE.match(tok) and _NUMERIC_RE.search(tok)
        for _x0, _x1, text in lane_runs
        for tok in text.split()
    )


def _top_data_y(
    rows_by_y: dict[int, list],
    anchor_y: int,
    local_ys: list[int],
    split_threshold: float,
    bands: list[tuple[float, float]],
) -> int:
    """First data row of the table the anchor belongs to.

    ``_best_anchor_y`` matches on a numeric multiset of at least
    ``_MIN_DATA_NUMERIC_CELLS`` DISTINCT values, so on a table whose opening
    rows are uniform (the ECB survey tables open with a row of ten zeroes) the
    anchor is not the first data row and the header band sits a whole row
    further up than the bridge gap allows. Walk up over rows that are nothing
    but numbers in the data columns; the header band is the first row that is
    not, and it stays where the caller can find it.
    """
    if not bands:
        return anchor_y
    top = anchor_y
    for y in reversed([y for y in local_ys if y < anchor_y]):
        if top - y > split_threshold:
            break
        if not _is_lane_data_row(rows_by_y.get(y, []), bands):
            break
        top = y
    return top


def _claim_lanes(
    lane_runs: list[tuple[float, float, str]],
    bands: list[tuple[float, float]],
    unclaimed: set[int],
) -> list[list[int]] | None:
    """Give each still-unclaimed lane to the run of *lane_runs* covering most of it.

    Returns one lane list per run, in the order the runs were given.  ``None``
    when two runs cover a lane equally, or when a run wins a discontiguous set
    of lanes — either way the row does not describe a column block and there is
    nothing safe to fold.
    """
    claims: list[list[int]] = [[] for _ in lane_runs]
    if not lane_runs:
        return claims

    for lane in sorted(unclaimed):
        band_lo, band_hi = bands[lane]
        overlaps = [max(0.0, min(x1, band_hi) - max(x0, band_lo)) for x0, x1, _t in lane_runs]
        widest = max(overlaps)
        if widest <= 0.0:
            continue
        if overlaps.count(widest) != 1:
            return None
        claims[overlaps.index(widest)].append(lane)

    for claim in claims:
        if claim and claim != list(range(claim[0], claim[-1] + 1)):
            return None
    return claims


def _fold_header_bands_into_lanes(
    header_rows_words: list[list],
    bands: list[tuple[float, float]],
) -> list[str] | None:
    """Flatten a multi-band header into one cell per data lane.

    *header_rows_words* is the native header band, top row first.  Each run is
    appended to every lane it heads, so a group heading reaches every column
    beneath it and a leaf label reaches only its own.  Returns
    ``[label] + per-lane cells``, or ``None`` on any doubt.

    Two stages, both read from the data upward.

    First the LEAF row, the band nearest the data. It must resolve to exactly
    one run per lane, covering every lane. Without that the leaf structure is
    unknown and there is nothing to fold into — which is also what keeps a WIDE
    LEAF label from being taken for a group heading: a probability-bin band
    ("-14% to -22%") is typeset wider than its own column and reaches its
    neighbours', so coverage alone would fold it sideways and corrupt a header
    that was never spanning (measured on the GH-56 exchange-rate fixture).

    Then the column BLOCKS, walking the bands above the leaf from the nearest
    upward and letting each row claim only lanes no lower row has claimed. A
    group heading is set on whichever typeset line its wrapping happens to end
    on, so the blocks are not all declared by one row: on the 2013 survey the
    five groups resolve over five different lines, one of which also carries a
    continuation fragment of a group already blocked out below it. Once the
    blocks are known, EVERY run — those that defined a block and those that
    wrapped — is attached to the single block it covers most. That is what
    stops a continuation line set a shade wider than the rest of its own
    heading from claiming the neighbouring column outright, which it otherwise
    does because nothing else on its line competes for it (measured at 2.4pt on
    the 2018 survey's "and medium-sized").
    """
    if not bands or len(header_rows_words) < 2:
        return None

    parsed: list[tuple[list[str], list[tuple[float, float, str]]]] = []
    for row_words in header_rows_words:
        split = _split_row_runs(row_words, bands)
        if split is None:
            return None
        label_runs, lane_runs = split
        # A run reaching across every column is a table title, not a group
        # heading: it carries no per-column information (``_run_lane_span``).
        lane_runs = [run for run in lane_runs if _run_lane_span(run, bands)]
        parsed.append(([text for _x0, _x1, text in label_runs], lane_runs))

    leaf_claims = _claim_lanes(parsed[-1][1], bands, set(range(len(bands))))
    if leaf_claims is None or any(len(claim) != 1 for claim in leaf_claims):
        return None
    if sorted(claim[0] for claim in leaf_claims) != list(range(len(bands))):
        return None

    blocks: list[list[int]] = []
    unclaimed = set(range(len(bands)))
    for _label_texts, lane_runs in reversed(parsed[:-1]):
        if not unclaimed:
            break
        claims = _claim_lanes(lane_runs, bands, unclaimed)
        if claims is None:
            return None
        for claim in claims:
            if claim:
                blocks.append(claim)
                unclaimed -= set(claim)
    # Columns no heading spans stand alone; a run over one attaches to it.
    blocks.extend([lane] for lane in sorted(unclaimed))
    blocks.sort()
    if not blocks:
        return None
    extents = [(bands[block[0]][0], bands[block[-1]][1]) for block in blocks]

    label_parts: list[str] = []
    lane_parts: list[list[str]] = [[] for _ in bands]
    saw_spanning = False
    for index, (label_texts, lane_runs) in enumerate(parsed):
        label_parts.extend(label_texts)
        for run_index, (x0, x1, text) in enumerate(lane_runs):
            if index == len(parsed) - 1:
                targets = [leaf_claims[run_index]]
            else:
                covered = [max(0.0, min(x1, hi) - max(x0, lo)) for lo, hi in extents]
                # A run that covers MOST of more than one block is a parent
                # heading over all of them -- a third level above the group
                # row, which reaches every column in every block it covers.
                # Attaching it to one child by greatest overlap, as this did,
                # silently drops it from the other children whenever the two
                # overlaps differ by a hair. A run that covers most of exactly
                # one block, or of none, is that block's own heading or a
                # wrapped fragment of it, and goes to the block it covers most
                # -- which is what keeps a continuation line set a shade wider
                # than its siblings from claiming the neighbouring column.
                majority = [
                    i
                    for i, overlap in enumerate(covered)
                    if 2.0 * overlap > (extents[i][1] - extents[i][0])
                ]
                if len(majority) > 1:
                    targets = [blocks[i] for i in majority]
                else:
                    widest = max(covered)
                    if widest <= 0.0 or covered.count(widest) != 1:
                        return None
                    targets = [blocks[covered.index(widest)]]
            if len(targets) > 1 or any(len(target) > 1 for target in targets):
                saw_spanning = True
            for target in targets:
                for lane in target:
                    lane_parts[lane].append(text)

    if not saw_spanning:
        return None
    return [" ".join(label_parts).strip()] + [" ".join(parts).strip() for parts in lane_parts]


def _merge_multiline_header_rows(header_rows: list[list[str]]) -> list[str]:
    """Merge geometry-derived header lines into one row, column by column."""
    if not header_rows:
        return []
    if len(header_rows) == 1:
        return list(header_rows[0])

    ncol = max(len(r) for r in header_rows)
    padded = [r + [""] * (ncol - len(r)) for r in header_rows]
    merged: list[str] = []
    for ci in range(ncol):
        parts = [r[ci] for r in padded if r[ci].strip()]
        merged.append(" ".join(parts))
    return merged


def _first_data_row_idx(grid: list[list[str]], expected_cols: int) -> int:
    """Index of the first body row that carries the modal data column count."""
    for i, row in enumerate(grid[1:], start=1):
        numeric_count = sum(
            1
            for cell in row
            if cell.strip() and _NUM_TOKEN_RE.match(cell.strip()) and _NUMERIC_RE.search(cell)
        )
        if numeric_count >= _MIN_DATA_NUMERIC_CELLS and len(row) >= expected_cols - 1:
            return i
    return len(grid)


def _header_is_faithful(header_row: list[str], expected_cols: int) -> bool:
    """True when every data-lane header cell (cols 1..expected_cols-1) is non-empty."""
    if len(header_row) < expected_cols:
        return False
    return all(header_row[i].strip() for i in range(1, expected_cols))


def _data_row_ys(
    rows_by_y: dict[int, list],
    anchor_y: int,
    split_threshold: float,
    local_ys: list[int],
) -> list[int]:
    """Collect y-groups at and below *anchor_y* that belong to the same table."""
    data_ys = [anchor_y]
    prev = anchor_y
    for y in local_ys:
        if y <= anchor_y:
            continue
        if y - prev <= split_threshold:
            # Only count rows that look like data (>=2 numeric tokens)
            if len(_row_numeric_multiset(rows_by_y[y])) >= 2:
                data_ys.append(y)
            prev = y
        else:
            break
    return data_ys


@dataclass(frozen=True)
class _TableGeometry:
    """The anchor -> lane -> band chain, computed once for a (grid, words) pair."""

    rows_by_y: dict[int, list]
    local_ys: list[int]
    anchor_y: int
    split_threshold: float
    lane_centers: list[float]
    data_start_x: float
    bands: list[tuple[float, float]]


def _table_geometry(grid: list[list[str]], words: list) -> _TableGeometry | None:
    """Locate *grid* on the page and derive its data lanes. ``None`` on any abstain."""
    if not words:
        return None
    rows_by_y = _all_rows_by_y(words)
    if not rows_by_y:
        return None

    anchor_y = _best_anchor_y(rows_by_y, grid)
    if anchor_y is None:
        logger.debug("header_repair: no anchor y-row with exact multiset match")
        return None

    anchor_y_int = round(anchor_y)
    local_ys = _local_table_ys(rows_by_y, anchor_y_int)
    split_threshold = max(_SPLIT_GAP_MULT * _median_row_gap(local_ys), _SPLIT_GAP_MIN_PT)

    data_ys = _data_row_ys(rows_by_y, anchor_y_int, split_threshold, local_ys)
    lane_groups = _derive_lane_groups(rows_by_y, data_ys)
    lane_centers = [sum(x0 for x0, _x1 in g) / len(g) for g in lane_groups]
    if len(lane_centers) < 2:
        logger.debug("header_repair: fewer than 2 data lanes derived")
        return None

    return _TableGeometry(
        rows_by_y=rows_by_y,
        local_ys=local_ys,
        anchor_y=anchor_y_int,
        split_threshold=split_threshold,
        lane_centers=lane_centers,
        data_start_x=lane_centers[0],
        bands=_lane_bands(lane_groups),
    )


def _spanning_header_bands(geom: _TableGeometry) -> tuple[list[str], list[list]] | None:
    """Flatten the page's spanning header band (#696).

    Returns ``(flattened_header, band_rows)`` -- the one-cell-per-lane header
    and the native word rows it was folded from, top row first. The second half
    is what tells a caller which rows of the MODEL's grid are header material:
    the page prints those words above its data, and nothing else in the
    candidate's header prefix is accounted for by them.
    """
    band_ys = _header_ys(
        geom.rows_by_y,
        geom.local_ys,
        _top_data_y(geom.rows_by_y, geom.anchor_y, geom.local_ys, geom.split_threshold, geom.bands),
        geom.split_threshold,
        geom.lane_centers,
        geom.data_start_x,
        geom.bands,
    )
    if not band_ys:
        return None
    band_rows = [geom.rows_by_y[y] for y in band_ys]
    flattened = _fold_header_bands_into_lanes(band_rows, geom.bands)
    if flattened is None:
        return None
    return flattened, band_rows


def _header_band_tokens(band_rows: list[list]) -> set[str]:
    """Casefolded word texts the page prints inside its header band."""
    return {w[4].strip().casefold() for row in band_rows for w in row if w[4].strip()}


def _candidate_header_depth(grid: list[list[str]], band_rows: list[list]) -> int:
    """How many leading rows of *grid* the page's header band accounts for.

    Row 0 is the markdown header and is header material by construction. Every
    further row is header material only while every token it carries is one the
    page prints in the band -- and never when the row reads as data. The first
    row that fails ends the header; everything from there down is BODY and must
    survive the rewrite verbatim, printed panel row or not. Deciding the
    boundary by "the first sufficiently numeric row" instead deleted any
    label-only row a table sets between its leaf headings and its first value.
    """
    band_tokens = _header_band_tokens(band_rows)
    depth = 1
    for row in grid[1:]:
        numeric_cells = sum(
            1
            for cell in row
            if cell.strip() and _NUM_TOKEN_RE.match(cell.strip()) and _NUMERIC_RE.search(cell)
        )
        if numeric_cells >= _MIN_DATA_NUMERIC_CELLS:
            break
        tokens = [tok.strip().casefold() for cell in row for tok in cell.split() if tok.strip()]
        if not tokens or any(tok not in band_tokens for tok in tokens):
            break
        depth += 1
    return depth


def native_header_row(
    grid: list[list[str]], words: list, *, require_spanning: bool = False
) -> list[str] | None:
    """Derive the header row implied by native word geometry.

    Runs the SAME anchor -> lane -> header-band chain as
    ``repair_collapsed_header``, but WITHOUT that function's
    ``detect_header_column_collapse`` gate: header attribution (GH-200) must
    also check tables whose header/data column counts already agree — that is
    exactly the "destroyed but not collapsed" case (a header band replaced by
    blanks, or shifted, while the column count stays put). Returns ``None`` on
    any abstain in the chain: no anchor row with an exact numeric-multiset
    match, fewer than 2 derived data lanes, or no lane-aligned header band
    above the anchor. Callers must treat ``None`` as UNVERIFIABLE, never as a
    pass or a fail.

    Result[0] is the label cell (words left of the first data lane);
    result[1:] are the per-lane header cells, one per derived data column.

    ``require_spanning`` (#696) narrows the result to the flattened form: with
    it set, only a header the page really does typeset as a spanning band is
    returned, and a single-level header abstains. A caller that rewrites a
    header which is not otherwise broken needs that distinction; a caller
    checking attribution does not.
    """
    geom = _table_geometry(grid, words)
    if geom is None:
        return None

    # #696: a spanning group heading folds into every column beneath it. This
    # runs first, on its own band scan, and abstains on anything that is not a
    # clean spanning band; the legacy scan and repair below are untouched by it.
    spanning = _spanning_header_bands(geom)
    if spanning is not None:
        return spanning[0]
    if require_spanning:
        return None

    hdr_ys = _header_ys(
        geom.rows_by_y,
        geom.local_ys,
        geom.anchor_y,
        geom.split_threshold,
        geom.lane_centers,
        geom.data_start_x,
    )
    if not hdr_ys:
        logger.debug("header_repair: no header y-rows above anchor y=%d", geom.anchor_y)
        return None

    header_grid: list[list[str]] = []
    for y in hdr_ys:
        row_cells = _assign_words_to_lanes(geom.rows_by_y[y], geom.lane_centers, geom.data_start_x)
        if any(c.strip() for c in row_cells):
            header_grid.append(row_cells)

    if not header_grid:
        return None

    return _merge_multiline_header_rows(header_grid)


def repair_collapsed_header(
    grid: list[list[str]],
    words: list,
) -> list[list[str]] | None:
    """Rebuild a collapsed header using native word geometry.

    Returns a new grid (header + original data rows, width-normalised) when
    repair succeeds, else ``None``.  Data cell VALUES are taken from the input
    *grid* — only the header structure is reconstructed.
    """
    collapsed, _header_cols, expected_cols = detect_header_column_collapse(grid)
    if not collapsed or not words:
        return None

    header_row = native_header_row(grid, words)
    if header_row is None:
        return None

    if len(header_row) != expected_cols:
        # Pad or trim to match the modal data width (never drop data columns).
        if len(header_row) < expected_cols:
            header_row = header_row + [""] * (expected_cols - len(header_row))
        else:
            header_row = header_row[:expected_cols]

    if not _header_is_faithful(header_row, expected_cols):
        logger.debug(
            "header_repair: declined — empty data-lane cell(s) in %r",
            header_row,
        )
        return None

    data_start = _first_data_row_idx(grid, expected_cols)
    body_rows: list[list[str]] = []
    for row in grid[data_start:]:
        if len(row) < expected_cols:
            row = row + [""] * (expected_cols - len(row))
        elif len(row) > expected_cols:
            row = row[:expected_cols]
        body_rows.append(row)

    if not body_rows:
        return None

    repaired = [header_row] + body_rows
    logger.debug(
        "header_repair: rebuilt header %d→%d cols",
        _header_cols,
        expected_cols,
    )
    return repaired


def flatten_multiband_header(
    grid: list[list[str]],
    words: list,
) -> list[list[str]] | None:
    """Fold a full-width two-level header into one row of per-column names (#696).

    The sibling of ``repair_collapsed_header`` for the case where the model did
    NOT lose columns: it kept the body's width and expressed the spanning band
    with padding cells, so ``detect_header_column_collapse`` sees nothing wrong
    and the table still ships with a group heading over a blank and its leaf
    labels one row down. Markdown cannot carry that, and the census pages where
    it happened arrived with the leaf band shifted (2018 Q8: two leaf labels
    missing, the remaining eight slid left), which is a silent header/value
    rebinding, not a cosmetic one.

    Deliberately narrow. It requires BOTH an emitted header of more than one
    band -- the first data row is not ``grid[1]`` -- AND native geometry that
    really does typeset a spanning band (``require_spanning``). A table with a
    single header row is byte-identical to before, whatever its content.
    """
    if len(grid) < 3 or not words:
        return None

    collapsed, _header_cols, expected_cols = detect_header_column_collapse(grid)
    if collapsed or expected_cols < 2 or len(grid[0]) != expected_cols:
        return None

    geom = _table_geometry(grid, words)
    if geom is None:
        return None
    spanning = _spanning_header_bands(geom)
    if spanning is None:
        return None
    header_row, band_rows = spanning
    if len(header_row) != expected_cols:
        return None
    if not _header_is_faithful(header_row, expected_cols):
        logger.debug("header_repair: declined flatten — empty lane in %r", header_row)
        return None

    # The header/body boundary comes from the page, not from "the first row
    # with enough numbers in it": a row the candidate prints and the header
    # band does not account for is BODY, and folding it away is content loss.
    data_start = _candidate_header_depth(grid, band_rows)
    if data_start < 2 or data_start >= len(grid):
        return None

    body_rows: list[list[str]] = []
    for row in grid[data_start:]:
        if len(row) < expected_cols:
            row = row + [""] * (expected_cols - len(row))
        elif len(row) > expected_cols:
            row = row[:expected_cols]
        body_rows.append(row)

    logger.debug("header_repair: flattened %d header band(s)", data_start)
    return [header_row] + body_rows


def repair_table_headers_in_text(
    words: list,
    markdown: str,
) -> tuple[str, int]:
    """Repair collapsed headers in every markdown table block in *markdown*.

    Returns ``(new_markdown, repair_count)``.
    """
    if not markdown.strip():
        return markdown, 0

    blocks = find_table_blocks(markdown)
    if not blocks:
        return markdown, 0

    lines = markdown.splitlines()
    repair_count = 0
    # Process blocks bottom-up so line indices stay valid after splices.
    for block in reversed(blocks):
        repaired = _repair_too_narrow_spanning_header(block.grid, words)
        if repaired is None:
            collapsed, _, _ = detect_header_column_collapse(block.grid)
            if collapsed:
                repaired = repair_collapsed_header(block.grid, words)
            else:
                repaired = flatten_multiband_header(block.grid, words)
        if repaired is None:
            continue
        # assume_header: `repaired`'s row 0 is a header this module just rebuilt
        # from word geometry and gated on `_header_is_faithful`. Letting
        # `_grid_to_markdown` re-infer it would demote a numeric-shaped header
        # band and discard the repair (GH-146).
        new_md = _grid_to_markdown(repaired, assume_header=True)
        lines[block.start : block.end + 1] = new_md.splitlines()
        repair_count += 1

    if repair_count == 0:
        return markdown, 0
    return "\n".join(lines), repair_count


def repair_table_headers_on_page(page, markdown: str) -> tuple[str, int]:
    """Convenience wrapper: fetch words from *page* and repair *markdown*."""
    try:
        words = page.get_text("words")
    except Exception:
        words = []
    return repair_table_headers_in_text(words, markdown)
