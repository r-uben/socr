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

    # Start at the header row nearest to the anchor (label-only rows between
    # header and data are absent from local_ys).
    nearest: int | None = None
    for y in reversed(below):
        if anchor_y - y > bridge_gap:
            break
        if _is_table_header_row(rows_by_y.get(y, []), lane_centers, data_start_x):
            nearest = y
            break
    if nearest is None:
        return []

    # Collect contiguous header lines upward from the nearest band.
    header: list[int] = [nearest]
    prev = nearest
    for y in reversed([y for y in below if y < nearest]):
        if prev - y > split_threshold:
            break
        row_words = rows_by_y.get(y, [])
        if _is_table_header_row(row_words, lane_centers, data_start_x):
            header.append(y)
            prev = y
        else:
            break
    return sorted(header)


def _derive_lane_centers(rows_by_y: dict[int, list], data_ys: list[int]) -> list[float]:
    """Cluster numeric-token x-positions from native data rows into column lanes."""
    num_xs: list[float] = []
    for y in data_ys:
        row_words = rows_by_y.get(y, [])
        # Skip rows with fewer than _MIN_DATA_NUMERIC_CELLS — these are header
        # annotations or chart ticks, not probability-bin data values.
        if len(_row_numeric_multiset(row_words)) < _MIN_DATA_NUMERIC_CELLS:
            continue
        for w in row_words:
            if _NUM_TOKEN_RE.match(w[4]) and _NUMERIC_RE.search(w[4]):
                num_xs.append(w[0])
    if not num_xs:
        return []

    xs_sorted = sorted(set(num_xs))
    lanes: list[list[float]] = []
    for x in xs_sorted:
        if lanes and x - lanes[-1][-1] <= _LANE_X_TOL_PT:
            lanes[-1].append(x)
        else:
            lanes.append([x])
    return [sum(g) / len(g) for g in lanes]


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


def native_header_row(grid: list[list[str]], words: list) -> list[str] | None:
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
    """
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
    lane_centers = _derive_lane_centers(rows_by_y, data_ys)
    if len(lane_centers) < 2:
        logger.debug("header_repair: fewer than 2 data lanes derived")
        return None

    data_start_x = lane_centers[0]

    hdr_ys = _header_ys(
        rows_by_y, local_ys, anchor_y_int, split_threshold, lane_centers, data_start_x
    )
    if not hdr_ys:
        logger.debug("header_repair: no header y-rows above anchor y=%d", anchor_y_int)
        return None

    header_grid: list[list[str]] = []
    for y in hdr_ys:
        row_cells = _assign_words_to_lanes(rows_by_y[y], lane_centers, data_start_x)
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
            if not collapsed:
                continue
            repaired = repair_collapsed_header(block.grid, words)
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
