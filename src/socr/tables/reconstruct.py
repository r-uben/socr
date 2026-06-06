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

No model, no Ollama — pure PyMuPDF on the page's own text layer, so every value is
the char-exact native glyph.
"""

from __future__ import annotations

import logging
import re

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
    return numeric / len(nonempty) >= _MIN_NUMERIC_FRAC


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
