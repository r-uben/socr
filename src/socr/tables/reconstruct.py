"""Reconstruct markdown grids for born-digital tables that have no ruling lines.

The defect this fixes (found on the owner's real corpus): booktabs-style tables
(top/mid/bottom rules only, the econ-paper norm) make ``page.find_tables()`` (the
default "lines" strategy) return zero tables, so ``extract_structured`` falls back
to ``get_text("text")`` — which dumps the table as a flat 1-D token stream
(``Industry / b / b / s / h / FabPr / 0.253 / ...``). The numbers are char-exact
but the row x column grid is gone.

PyMuPDF's ``find_tables(strategy="text")`` infers columns from text alignment and
DOES recover the grid (validated on Fama-French 1997: correct values in the right
cells). Its bbox is too over-inclusive to use for *localization* (it swallows
surrounding prose), but its *cell extraction* is exactly what we want here. So we
run it only on table-dominant pages, clean the grid (drop empty rows/columns,
strip page running-heads), and keep it only if it still looks like a table.

Word-geometry rowizer (TR-1): when the text-strategy find_tables also over-merges
(whitespace-gutter pages with multiple regions — charts, prose — on the same page),
``rowize_from_words`` segments the page by vertical gaps derived from the page's own
row-height distribution (1.5 × median inter-row gap) and builds a column grid
directly from PyMuPDF word coordinates. This is the correct fallback for CE-style
whitespace-gutter tables that have no ruling lines AND live on a multi-region page.
Crucially, a missing token in a lane produces a blank cell — never a skipped column.

No model, no Ollama — pure PyMuPDF on the page's own text layer, so every value is
the char-exact native glyph.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)

# Performance guard, not a data threshold: PyMuPDF's text-strategy find_tables
# cost grows ~quadratically with the token count, and on dense non-table pages
# that slip through the columnar gate (reference lists, full-page equation pages)
# it can run for minutes. A genuine data table is a few hundred words (Fama-French
# p8 = 213). Skip pages far above that — they are both slow and unlikely to be a
# clean table. Raise if a legitimately huge table is ever missed.
_MAX_PAGE_WORDS = 1500

# A reconstructed grid is kept only if it still looks like a table after cleaning:
# enough rows, at least two columns, and a real share of numeric cells (econ
# tables are mostly numbers). These guard against text-strategy firing on prose.
_MIN_ROWS = 3
_MIN_COLS = 2
_MIN_NUMERIC_FRAC = 0.20
# Majority of non-empty rows must be real data rows (>=2 numeric cells) for a
# text-strategy grid to be trusted. Rejects whole-page over-capture on
# prose/references pages that merely contain a small embedded table.
_MIN_DATA_ROW_FRAC = 0.5

_NUMERIC_RE = re.compile(r"-?\d")
# A token that is essentially a number (table value): 0.253, (0.014), 1,204, 45%.
_NUM_TOKEN_RE = re.compile(r"^[\(\[]?-?\d[\d.,]*[\)\]%]?$")

# The structural table gate. A real data table puts numbers in vertical lanes and
# each data ROW populates several lanes at once (FabPr | 0.253 | 0.124 | 0.179 |
# 0.211). A reference list scatters numbers one-per-line, so no row co-occupies
# multiple numeric lanes. These distinguish a grid from prose/references without
# the cost (or truncation risk) of running text-strategy first. Validated on real
# pages: a table page yields ~10-50 multi-column rows; a references page yields 0.
_LANE_X_TOL_PT = 6.0  # numeric tokens within this x distance share a lane
_MIN_LANES_PER_ROW = 3  # a data row must populate this many numeric lanes
_MIN_TABLE_ROWS = 3  # and there must be this many such rows
# A running-head row swept in from the page margin reads like a journal/volume
# line. Matched with OCR tolerance because older PDFs carry corrupted text layers
# (observed "Joumal" for "Journal", "(/997)" for "(1997)"): journal-name tokens
# ("jou[rm]nal", "econom", "review", "quarterly") or a year/volume in parens.
# Best-effort: a stray runhead row is cosmetic, never a correctness risk.
_RUNHEAD_RE = re.compile(
    r"jou[rm]nal|econom|review|quarterly|econometrica|\([\d/]{3,5}\)",
    re.IGNORECASE,
)


def reconstruct_table_regions(page) -> list[tuple[object, str]]:
    """Return ``(rect, markdown)`` pairs for text-aligned tables on ``page``.

    Same shape as the lines-strategy path in ``extract_structured`` so the caller
    can interleave them with prose identically. Never raises.
    """
    try:
        import fitz

        # Structural gate: only reconstruct where numbers actually form a grid
        # (multiple numeric lanes co-occupied per row). This skips references /
        # prose pages that the cheap columnar heuristic false-fires on — and which
        # make text-strategy run for minutes — without the truncation risk of
        # clipping to a rule band. The word cap stays as a final cost backstop.
        if not has_numeric_columns(page):
            return []
        if len(page.get_text("words")) > _MAX_PAGE_WORDS:
            logger.debug("skipping text-strategy reconstruct: page too dense")
            return []
        result = page.find_tables(vertical_strategy="text", horizontal_strategy="text")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("text-strategy find_tables failed: %s", exc)
        return []

    out: list[tuple[object, str]] = []
    for table in getattr(result, "tables", []):
        try:
            grid = table.extract()
        except Exception:
            continue
        cleaned = _clean_grid(grid)
        if not _looks_tabular(cleaned):
            continue
        md = _grid_to_markdown(cleaned)
        if md:
            out.append((fitz.Rect(table.bbox), md))
    return out


def has_numeric_columns(page) -> bool:
    """True if the page's numeric tokens form a real grid (not a reference list).

    Clusters numeric token x-positions into lanes, then counts rows that populate
    at least ``_MIN_LANES_PER_ROW`` lanes at once. Cheap (text only, no table
    inference) and truncation-free, so it is the gate before the expensive
    text-strategy call. Never raises.
    """
    try:
        words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    except Exception:  # pragma: no cover - defensive
        return False
    nums = [
        (w[0], round(w[1])) for w in words if _NUM_TOKEN_RE.match(w[4]) and _NUMERIC_RE.search(w[4])
    ]
    if len(nums) < _MIN_LANES_PER_ROW * _MIN_TABLE_ROWS:
        return False

    xs = sorted({x for x, _ in nums})
    lanes: list[list[float]] = []
    for x in xs:
        if lanes and x - lanes[-1][-1] <= _LANE_X_TOL_PT:
            lanes[-1].append(x)
        else:
            lanes.append([x])
    lane_of = {x: i for i, lane in enumerate(lanes) for x in lane}

    row_lanes: dict[float, set] = {}
    for x, y in nums:
        row_lanes.setdefault(y, set()).add(lane_of[x])
    grid_rows = sum(1 for ls in row_lanes.values() if len(ls) >= _MIN_LANES_PER_ROW)
    return grid_rows >= _MIN_TABLE_ROWS


def _clean_grid(grid) -> list[list[str]]:
    """Normalise cells, drop empty rows, a leading running-head row, empty columns."""
    g = [[("" if c is None else str(c)).replace("\n", " ").strip() for c in row] for row in grid]
    g = [r for r in g if any(r)]  # drop fully-empty rows
    # Drop a leading row that is a page running-head (one populated cell, journal-ish).
    while g and _is_runhead(g[0]):
        g = g[1:]
    if not g:
        return []
    ncol = max(len(r) for r in g)
    g = [r + [""] * (ncol - len(r)) for r in g]
    keep = [c for c in range(ncol) if any(r[c] for r in g)]  # drop all-empty columns
    return [[r[c] for c in keep] for r in g]


def _is_runhead(row: list[str]) -> bool:
    """A page running-head swept into the grid: its joined text reads like a
    journal/volume/page line, not table data.

    Matched on the whole row because text-strategy may split the head across
    several cells. A real data row ("FabPr 0.253 0.124 ...") never matches the
    journal/(year)/page-range pattern.
    """
    joined = " ".join(c for c in row if c)
    return bool(joined) and bool(_RUNHEAD_RE.search(joined))


def _looks_tabular(grid: list[list[str]]) -> bool:
    if len(grid) < _MIN_ROWS or not grid or len(grid[0]) < _MIN_COLS:
        return False
    cells = [c for row in grid for c in row]
    nonempty = [c for c in cells if c]
    if not nonempty:
        return False
    numeric = sum(1 for c in nonempty if _NUMERIC_RE.search(c))
    if numeric / len(nonempty) < _MIN_NUMERIC_FRAC:
        return False

    # Localization guard against whole-page over-capture: PyMuPDF's text-strategy
    # grids EVERYTHING on a page by whitespace alignment, so a page that is mostly
    # prose/references with one small embedded data table comes back as a single
    # page-spanning "table" whose prose lines are shredded into cells. A genuine
    # data table has most of its rows populate multiple numeric lanes; a prose page
    # with a small table does not. Require a MAJORITY of non-empty rows to be real
    # data rows (>=2 numeric cells). Empirically separates a clean booktabs grid
    # (~0.83) from a prose+references+small-table page (~0.40). Rejected pages fall
    # back to plain get_text() (clean linearized prose), never a shredded grid.
    data_rows = sum(1 for row in grid if sum(1 for c in row if c and _NUMERIC_RE.search(c)) >= 2)
    nonempty_rows = sum(1 for row in grid if any(row))
    if not nonempty_rows:
        return False
    return data_rows / nonempty_rows >= _MIN_DATA_ROW_FRAC


def _grid_to_markdown(grid: list[list[str]]) -> str:
    """Render a cleaned grid as a GitHub-markdown table (header = first row)."""
    if not grid:
        return ""
    ncol = len(grid[0])
    header = grid[0]
    body = grid[1:]
    lines = [
        "| " + " | ".join(_esc(c) for c in header) + " |",
        "| " + " | ".join(["---"] * ncol) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(_esc(c) for c in row) + " |")
    return "\n".join(lines)


def _esc(cell: str) -> str:
    return cell.replace("|", r"\|")


# ---------------------------------------------------------------------------
# Word-geometry rowizer (TR-1)
# ---------------------------------------------------------------------------
# Gap threshold multiplier: an inter-row gap > (SPLIT_GAP_MULT × median row gap)
# is treated as a region boundary. 1.5 is calibrated to separate table sections
# (13-16 pt row height → threshold ~20 pt) from surrounding regions without
# splitting multi-line headers (5-6 pt y-jitter within a header). This is a
# structural constant derived from the observed row-height distribution, not a
# magic pixel value — any table with consistent row heights will produce a clean
# split at natural region boundaries.
_SPLIT_GAP_MULT = 1.5

# Minimum absolute gap (pt) for a region split. Prevents splitting on sub-pixel
# y-jitter in densely-set headers where word baselines may vary by 1-2 pt.
_SPLIT_GAP_MIN_PT = 10.0

# Words within this factor × _LANE_X_TOL_PT of the nearest data-column x are
# assigned to that column; words further left go into the label cell.
# Using 3× the lane tolerance gives a comfortable snap radius for slight PDF
# text-positioning variation while keeping labels out of the first data column.
_LANE_SNAP_MULT = 3


def rowize_from_words(page) -> list[tuple[object, str]]:
    """Return ``(rect, markdown)`` pairs built from word geometry on ``page``.

    This is the TR-1 fallback for pages where ``find_tables(strategy="text")``
    over-merges multiple regions (chart + table + prose) into one spanning grid
    that fails the ``_looks_tabular`` gate. It segments the page by detecting
    vertical gaps larger than 1.5 × the page's median inter-row gap (derived
    from the page's own word positions — no fixed pixel threshold), then tries
    to rowize each segment independently.

    A missing token in a column lane produces a blank cell — the caller's
    normalisation layer maps ``""`` → ``"na"`` for parity checks.

    Guard: the same ``has_numeric_columns`` structural gate as
    ``reconstruct_table_regions`` is applied first. This prevents the fallback
    from firing on reference/bibliography pages where 2-column aligned rows
    (author + year, or year + page) happen to pass ``_looks_tabular`` but do
    not form a real numeric grid — which would produce spurious markdown tables
    (silent content corruption on the citation corpus). A genuine table page
    has ≥ ``_MIN_LANES_PER_ROW`` numeric lanes co-occupied per row; bibliography
    pages do not.

    Never raises. Returns ``[]`` if no valid table segment is found.
    """
    # Structural gate: only rowize where numeric tokens form a real grid.
    # Mirrors the identical guard in reconstruct_table_regions (reconstruct.py
    # line ~85) and ensures the same class of pages (references, equation dumps)
    # that are already safely skipped by the text-strategy path are also skipped
    # here. The word cap (_MAX_PAGE_WORDS) is NOT applied: unlike find_tables
    # (strategy="text") which scales quadratically, the word-geometry rowizer is
    # O(n) so the only skip needed is the structural gate, not a cost backstop.
    if not has_numeric_columns(page):
        return []

    try:
        words = page.get_text("words")  # (x0,y0,x1,y1,text,block,line,word)
    except Exception:
        return []

    return rowize_from_word_list(words)


def rowize_from_word_list(
    words: list,
) -> list[tuple[object, str]]:
    """Build ``(rect, markdown)`` pairs from a flat list of PyMuPDF word tuples.

    The word list has the PyMuPDF ``get_text("words")`` shape:
    ``(x0, y0, x1, y1, text, block_no, line_no, word_no)``.

    Segmentation: rows are grouped by rounded ``y0``; consecutive y-groups
    separated by a gap > max(_SPLIT_GAP_MULT × median_gap, _SPLIT_GAP_MIN_PT)
    are treated as distinct regions. Each segment passes through the same
    ``_looks_tabular`` gate as ``reconstruct_table_regions``.

    Column detection: numeric token x-positions are clustered into lanes (same
    ``_LANE_X_TOL_PT`` as ``has_numeric_columns``); words to the left of the
    leftmost lane minus a snap margin are concatenated into the label cell for
    that row. A lane with no token in a given row becomes a blank (``""``)
    cell — never dropped.

    Never raises. Returns ``[]`` if no valid table segment is found.
    """
    try:
        import fitz
    except ImportError:  # pragma: no cover
        return []

    if not words:
        return []

    # ------------------------------------------------------------------
    # 1. Group words into y-rows (round y0 to nearest point to merge
    #    words on the same baseline that differ by sub-point jitter).
    # ------------------------------------------------------------------
    rows_by_y: dict[int, list] = defaultdict(list)
    for w in words:
        rows_by_y[round(w[1])].append(w)

    ys = sorted(rows_by_y.keys())
    if len(ys) < _MIN_TABLE_ROWS:
        return []

    # ------------------------------------------------------------------
    # 2. Derive split threshold from the page's own row-height distribution.
    #    The median gap between consecutive y-groups reflects the typical
    #    row spacing for this document — no fixed pt constant.
    # ------------------------------------------------------------------
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    median_gap = statistics.median(gaps)
    split_threshold = max(_SPLIT_GAP_MULT * median_gap, _SPLIT_GAP_MIN_PT)

    # ------------------------------------------------------------------
    # 3. Split into y-segments at gaps above the threshold.
    # ------------------------------------------------------------------
    segments: list[list[int]] = [[ys[0]]]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] > split_threshold:
            segments.append([ys[i]])
        else:
            segments[-1].append(ys[i])

    # ------------------------------------------------------------------
    # 4. For each segment, attempt to build a column-consistent grid.
    # ------------------------------------------------------------------
    out: list[tuple[object, str]] = []
    for seg_ys in segments:
        seg_words: list = []
        for y in seg_ys:
            seg_words.extend(rows_by_y[y])

        grid_and_rect = _rowize_segment(seg_words, seg_ys, rows_by_y)
        if grid_and_rect is None:
            continue
        grid, x0, y0, x1, y1 = grid_and_rect

        cleaned = _clean_grid(grid)
        if not _looks_tabular(cleaned):
            logger.debug(
                "rowizer: segment y=%d..%d rejected by _looks_tabular", seg_ys[0], seg_ys[-1]
            )
            continue
        md = _grid_to_markdown(cleaned)
        if md:
            out.append((fitz.Rect(x0, y0, x1, y1), md))

    return out


def _rowize_segment(
    seg_words: list,
    seg_ys: list[int],
    rows_by_y: dict[int, list],
) -> tuple[list[list[str]], float, float, float, float] | None:
    """Build a raw grid from the words in one y-segment.

    Returns ``(grid, x0, y0, x1, y1)`` on success, ``None`` if the segment
    has too few numeric tokens to form a plausible table.

    Column lanes are identified from numeric-token x-positions.  Words to the
    left of the first lane (minus a snap margin) are collected into a single
    label cell per row.  A lane absent from a row produces ``""`` — the blank
    cell that maps to ``"na"`` in parity checks.
    """
    # Find numeric tokens to detect column lanes
    num_words = [w for w in seg_words if _NUM_TOKEN_RE.match(w[4]) and _NUMERIC_RE.search(w[4])]
    if len(num_words) < _MIN_LANES_PER_ROW * _MIN_TABLE_ROWS:
        return None

    # Cluster numeric x-positions into lanes (same algorithm as has_numeric_columns)
    num_xs = sorted(set(w[0] for w in num_words))
    lanes_x: list[list[float]] = []
    for x in num_xs:
        if lanes_x and x - lanes_x[-1][-1] <= _LANE_X_TOL_PT:
            lanes_x[-1].append(x)
        else:
            lanes_x.append([x])

    if len(lanes_x) < _MIN_COLS:
        return None

    # Lane centres (mean x of all xs in the cluster)
    lane_centers = [sum(xs) / len(xs) for xs in lanes_x]
    data_start_x = lane_centers[0]

    # Words further left than (data_start_x − snap_margin) form the label cell
    snap_margin = _LANE_X_TOL_PT * _LANE_SNAP_MULT

    # ------------------------------------------------------------------
    # Build one grid row per y-group in the segment.
    # ------------------------------------------------------------------
    grid: list[list[str]] = []
    # Use the actual word bounding-box coordinates for the region rect so that
    # extract_structured's overlap check correctly suppresses the corresponding
    # prose text blocks (which start at the same y as the words, not at the
    # rounded y-group centre).
    bbox_x0 = min(w[0] for w in seg_words)
    bbox_x1 = max(w[2] for w in seg_words)
    bbox_y0 = min(w[1] for w in seg_words)
    bbox_y1 = max(w[3] for w in seg_words)

    for y in seg_ys:
        row_ws = sorted(rows_by_y[y], key=lambda w: w[0])

        # Label: concatenate all words to the left of the data boundary
        label_words = [w[4] for w in row_ws if w[0] < data_start_x - snap_margin]
        label = " ".join(label_words) if label_words else ""

        # Data cells: assign each word to the nearest lane by x-distance.
        # A lane with no word assigned stays "" (blank / na).
        row_cells = [""] * len(lane_centers)
        for w in row_ws:
            if w[0] < data_start_x - snap_margin:
                continue  # already in the label
            best = min(range(len(lane_centers)), key=lambda i: abs(lane_centers[i] - w[0]))
            if abs(lane_centers[best] - w[0]) <= _LANE_X_TOL_PT * _LANE_SNAP_MULT:
                existing = row_cells[best]
                row_cells[best] = (existing + " " + w[4]).strip() if existing else w[4]

        # Always emit the label as a first cell so all rows share the same
        # column layout.  An empty label yields "" (empty first cell).
        grid_row = [label] + row_cells
        if any(c.strip() for c in grid_row):
            grid.append(grid_row)

    if not grid:
        return None

    # Pad to uniform width and drop all-empty columns (standard cleanup)
    ncols = max(len(r) for r in grid)
    grid = [r + [""] * (ncols - len(r)) for r in grid]
    keep = [c for c in range(ncols) if any(r[c] for r in grid)]
    grid = [[r[c] for c in keep] for r in grid]

    return grid, bbox_x0, bbox_y0, bbox_x1, bbox_y1
