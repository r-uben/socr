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

Chart-clip (TR-2): ``chart_region_bboxes`` detects vector-chart clusters on the page
using the same drawing-cluster logic as ``has_chart_marks`` (figures/extractor.py),
then expands each cluster's bbox by the page's own median inter-row word gap so that
nearby text labels (axis tick values, year labels below bars) are included in the
chart region and excluded from the table rowizer.  This fixes the CE p.4 failure mode
where chart tick-label rows at x=54 diluted the historical table's data_row_frac
below the ``_looks_tabular`` threshold.  The chart region is returned as a
placeholder image-ref entry (``![chart region ...](...)``) by
``rowize_from_words_chart_aware`` so reading-order reassembly in
``extract_structured`` can emit it in the correct position.

No model, no Ollama — pure PyMuPDF on the page's own text layer, so every value is
the char-exact native glyph.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)

# GH-144 review finding 4: rejecting a text-strategy grid for destroying a
# numeric token is only logged here, not surfaced at page/document/CLI level
# (`born_digital.py` / `orchestrator.py` are out of scope for this file's
# write set — see GH-147 A2, running concurrently). Tracked as a follow-up
# rather than implemented here; referenced in the warning below so the log
# line points at the open gap instead of looking resolved.
_SILENT_LOSS_FOLLOWUP_ISSUE = 195

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

    GH-144: the text-strategy grid from ``page.find_tables(strategy="text")``
    infers lane boundaries from whitespace alignment, and on booktabs-style
    label+value rows (single internal space in the label, a space RUN before
    the first value) a lane boundary can land strictly inside a numeric
    token's own bbox — splitting ``0.67`` into ``0`` + ``.67`` at the grid-
    construction call itself (see
    ``docs/plans/gh144-rowizer-destroys-values/logs/2026-08-12_A1-boundary-diagnosis.md``).
    ``_destroyed_numeric_tokens`` detects this per table by re-locating each
    native numeric token's raw extract() cell and checking the token string
    survives verbatim; on detection the text-strategy grid is rejected and
    the already-proven-lossless word-geometry rowizer (``rowize_from_word_list``)
    is used instead, scoped to a TIGHT bbox built from ``_numeric_row_bbox``
    (the union of this table's rows that actually carry a numeric cell) — not
    the destroyed table's own ``bbox`` and not the page's full word list. Both
    were tried first: the table's own whitespace-inferred ``bbox`` routinely
    overruns into whatever text sits just beneath the last data row (that is
    how the notes paragraph ended up inside the destroyed grid at all), and
    the unclipped full page merges the table with that same trailing text
    because the table-to-notes gap can be smaller than the rowizer's own
    y-gap split threshold. Scoping to the numeric rows' own union avoids both.
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
        words = page.get_text("words")
        if len(words) > _MAX_PAGE_WORDS:
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

        # Scope both the destruction check and the fallback to this table's
        # own numeric-bearing rows (GH-144 review finding 2) — not
        # `table.bbox`, which is whitespace-inferred by find_tables and
        # routinely overruns past the last numeric row into whatever text
        # sits just beneath it (that is how the notes paragraph ended up
        # inside this very grid). A numeral caught in that overrun with no
        # containing cell (an axis tick, a page number) would otherwise be
        # counted "destroyed" and reject an otherwise-good grid.
        numeric_scope = _numeric_row_bbox(table, grid, words)
        destroyed = _destroyed_numeric_tokens(
            words,
            table,
            grid,
            numeric_scope if numeric_scope is not None else fitz.Rect(table.bbox),
        )
        if destroyed:
            values = [rec["value"] for rec in destroyed]
            logger.warning(
                "reconstruct_table_regions: text-strategy grid destroyed %d numeric "
                "token(s) %s in region %s — rejecting the text-strategy grid, falling "
                "back to the word-geometry rowizer scoped to the table's numeric rows "
                "(follow-up: GH-%s tracks surfacing this at page/document/CLI level)",
                len(destroyed),
                values,
                table.bbox,
                _SILENT_LOSS_FOLLOWUP_ISSUE,
            )
            # Fall back on the word-geometry rowizer, but scope it to a TIGHT
            # bbox derived from this table's own numeric rows — not the raw,
            # unclipped page (the trailing-notes gap is smaller than the
            # rowizer's own split threshold, so an unclipped call merges the
            # notes paragraph into the same segment and fragments it) and not
            # the destroyed table's own `bbox` (see above). The tight bbox is
            # the union of only the rows that contain at least one numeric
            # cell: a data-driven boundary, not a tuned constant, and it is
            # exactly the rows a numeric table owns.
            if numeric_scope is None:
                # No numeric-bearing row at all — `_looks_tabular`'s numeric-
                # fraction gate makes this unlikely to reach here, but if it
                # does there is no tight scope to compute; broadening to the
                # full page is exactly the notes-merge risk the tight scope
                # exists to avoid, so it is never done silently.
                logger.warning(
                    "reconstruct_table_regions: no numeric-bearing row found for "
                    "region %s — broadening the fallback rowizer to the full page "
                    "word list instead of a tight scope",
                    table.bbox,
                )
                scoped_words = words
            else:
                # GH-144 A2b: a header band carries no numeric cell, so the
                # tight numeric-row scope by construction excludes it —
                # extend it to also cover a preceding lane-snapping header
                # band, or `_prepend_header_band` downstream has nothing to
                # prepend (review finding 3).
                tight = _extend_scope_for_header(numeric_scope, words)
                scoped_words = [w for w in words if tight.contains(fitz.Point(w[0], w[1]))]
            rowized = rowize_from_word_list(scoped_words)
            if rowized:
                out.extend(rowized)
            # Whether or not the scoped fallback found anything usable, this
            # table's own text-strategy grid is rejected — move on to the
            # next table in `result.tables` rather than abandoning the ones
            # already collected in `out` (GH-144 review finding 1: a
            # multi-table page must not lose its other, undamaged tables).
            continue

        md = _grid_to_markdown(cleaned)
        if md:
            out.append((fitz.Rect(table.bbox), md))
    return out


def _cell_has_numeric_token(cell) -> bool:
    """True if a raw cell is entirely numeric-token-shaped, tolerant of the
    noise ``table.extract()`` leaves in a RAW cell (embedded newlines,
    padding, or more than one space-separated value merged into a single
    cell).

    ``_NUM_TOKEN_RE`` is anchored (``^...$``) and was written to match a
    single already-split token; matching it directly against a raw
    ``extract()`` cell (GH-144 review finding 3) misses ``"0.67\\n"``, a
    padded cell, or a multi-value cell, silently shrinking the numeric-row
    scope this predicate feeds. Splitting on whitespace and requiring EVERY
    piece to match keeps the exact same anchored regex — no new numeric
    grammar — while tolerating the raw cell's own formatting noise.

    Deliberately ``all()``, not ``any()``: a prose cell that merely contains
    a numeral (``"is 106 obser"`` — a notes sentence split mid-word by the
    text-strategy grid) must NOT count as a numeric cell, or a stray digit
    like a sample size or an axis tick pulls the whole prose row into the
    numeric-row scope this predicate feeds (finding 2's own example, "106",
    reproduced verbatim by an ``any()`` version of this check against
    ``tests/test_region_overlap_gh145.py``'s own fixture).
    """
    if cell is None:
        return False
    toks = str(cell).split()
    return bool(toks) and all(_NUM_TOKEN_RE.match(tok) for tok in toks)


def _numeric_row_bbox(table, grid: list[list], words: list):
    """Return the ``fitz.Rect`` union of ``table``'s rows that carry a numeric
    cell, or ``None`` if none do.

    Used to re-scope both the destruction check and the word-geometry
    rowizer fallback when the text-strategy grid is rejected for destroying
    a numeric token: `table.bbox` itself is whitespace-inferred and can
    overrun past the table's real content into whatever text sits just
    beneath it (see ``reconstruct_table_regions``). A row that contains no
    numeric cell (``_cell_has_numeric_token``) is not part of the numeric
    table this rejection concerns — data-driven, not a tuned threshold.

    Unions each qualifying row's ``table.rows[n].bbox`` with the native
    bbox of every WORD whose centre falls inside that row rect. PyMuPDF's
    row bbox is a layout cell rect, not a glyph-tight one, and can sit a
    point or two inside a word's own native bbox (e.g. a row bbox
    ``y0=96.96`` while its own label word's native bbox starts at
    ``y0=94.33``); unioning the row bbox alone would then silently exclude
    that word from the scope built here — the same class of silent loss
    this whole rejection path exists to prevent.
    """
    try:
        import fitz
    except ImportError:
        return None

    rect = None
    for n, row in enumerate(getattr(table, "rows", [])):
        grid_row = grid[n] if n < len(grid) else None
        if grid_row is None:
            continue
        if not any(_cell_has_numeric_token(cell) for cell in grid_row):
            continue
        row_bbox = getattr(row, "bbox", None)
        if row_bbox is None:
            continue
        orig = fitz.Rect(row_bbox)
        r = fitz.Rect(row_bbox)
        # Test containment against `orig` (frozen), never the growing `r`:
        # testing against `r` while also expanding it would let one
        # absorbed word's bbox grow the rect enough to catch the NEXT word
        # in iteration order, cascading arbitrarily far down the page
        # instead of staying local to this row.
        for w in words:
            wcx, wcy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
            if orig.x0 <= wcx <= orig.x1 and orig.y0 <= wcy <= orig.y1:
                r = r | fitz.Rect(w[0], w[1], w[2], w[3])
        rect = r if rect is None else rect | r
    return rect


def _cell_contains_centre(cell, cx: float, cy: float) -> bool:
    """True if ``cell`` (an ``(x0, y0, x1, y1)`` tuple) contains point ``(cx, cy)``.

    Half-open membership matching PyMuPDF ``Table.extract()``'s own cell-fill
    convention (``x0 <= cx < x1 and y0 <= cy < y1``) so a point sitting exactly
    on a shared boundary belongs to the cell whose LOW edge it touches, not
    both neighbours. Returns ``False`` when ``cell`` is ``None`` (an unfilled
    / merged-away cell position has no rect to test against).
    """
    if cell is None:
        return False
    x0, y0, x1, y1 = cell
    return x0 <= cx < x1 and y0 <= cy < y1


def _destroyed_numeric_tokens(words: list, table, grid: list[list], scope_bbox) -> list[dict]:
    """Return one record per native numeric token the text-strategy grid destroyed.

    For every numeric native token on the page whose centre point falls
    inside ``scope_bbox`` (the caller-supplied scope for this table — tokens
    elsewhere are not this table's concern), locate the raw
    ``table.rows[n].cells[c]`` rect containing that centre (rows/cells
    index-align with ``table.extract()``'s nested loops), then check the
    token's exact string is a literal substring of the corresponding RAW
    ``grid[n][c]`` cell (never the ``_clean_grid`` output — cleaning
    drops/reindexes columns and would mis-map).

    ``scope_bbox`` is expected to be the union of this table's own
    numeric-bearing rows (``_numeric_row_bbox``), not ``table.bbox``:
    ``table.bbox`` is whitespace-inferred by ``find_tables`` and routinely
    overruns into notes/prose text below the table, and a numeral caught in
    that overrun with no containing cell (an axis tick, a page number) would
    otherwise be counted "destroyed" and reject an otherwise-good grid
    (GH-144 review finding 2). The caller decides the fallback scope when
    ``_numeric_row_bbox`` returns ``None``; this function only ever tests
    against whatever rect it is given.

    Conservative by construction: any ambiguity is counted as destroyed, never
    silently skipped —

    - no cell rect contains the token's centre ("no-cell"),
    - the containing table cell is itself ``None`` ("cell=None", the merged-
      cell case — inferred from the table containing at least one ``None``
      cell elsewhere, since a ``None`` cell can never match
      ``_cell_contains_centre``'s membership test directly),
    - the aligned ``grid[n][c]`` position is out of range or ``None``
      ("grid-missing" / "grid-none"),
    - or the token string is not a literal substring of that raw cell text
      ("boundary-split", carrying the containing cell rect and, as a
      diagnostic only (not part of the reject decision), any cell x-edge that
      sits strictly inside the token's own bbox — the mechanism A1 measured).

    No page-level ``Counter``, no boundary-intersection count, no fuzzy
    normalisation, no new threshold — exact string survival, scoped per table.
    """
    try:
        import fitz
    except ImportError:  # pragma: no cover
        return []

    bbox = fitz.Rect(scope_bbox)
    rows = getattr(table, "rows", [])
    has_none_cell = any(
        cell is None for row in rows for cell in (getattr(row, "cells", None) or [])
    )

    records: list[dict] = []
    for w in words:
        text = w[4]
        if not (_NUM_TOKEN_RE.match(text) and _NUMERIC_RE.search(text)):
            continue
        x0, y0, x1, y1 = w[0], w[1], w[2], w[3]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if cx < bbox.x0 or cx >= bbox.x1 or cy < bbox.y0 or cy >= bbox.y1:
            continue  # outside this table's scope — not this table's concern

        containing = None
        row_idx = col_idx = None
        for n, row in enumerate(rows):
            cells = getattr(row, "cells", None) or []
            for c, cell in enumerate(cells):
                if _cell_contains_centre(cell, cx, cy):
                    containing = cell
                    row_idx, col_idx = n, c
                    break
            if containing is not None:
                break

        if containing is None:
            reason = "cell=None" if has_none_cell else "no-cell"
            records.append({"value": text, "bbox": (x0, y0, x1, y1), "boundary": reason})
            continue

        grid_row = grid[row_idx] if row_idx is not None and row_idx < len(grid) else None
        raw_cell = grid_row[col_idx] if grid_row is not None and col_idx < len(grid_row) else None
        if grid_row is None or col_idx >= len(grid_row):
            records.append({"value": text, "bbox": (x0, y0, x1, y1), "boundary": "grid-missing"})
            continue
        if raw_cell is None:
            records.append({"value": text, "bbox": (x0, y0, x1, y1), "boundary": "grid-none"})
            continue

        if text not in raw_cell:
            cell_x0, _cell_y0, cell_x1, _cell_y1 = containing
            edge_inside = None
            if cell_x0 > x0 and cell_x0 < x1:
                edge_inside = cell_x0
            elif cell_x1 > x0 and cell_x1 < x1:
                edge_inside = cell_x1
            records.append(
                {
                    "value": text,
                    "bbox": (x0, y0, x1, y1),
                    "boundary": {
                        "reason": "boundary-split",
                        "cell": containing,
                        "raw_cell_text": raw_cell,
                        "edge_inside_token_bbox": edge_inside,
                    },
                }
            )
    return records


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


def _is_data_row(row: list[str]) -> bool:
    """True when *row* carries observations and must not become a header.

    A row is data when it names an entity in column 0 **and** at least one data
    cell holds a value-shaped token (``0.67``, ``(0.14)``, ``1,204``, ``45%``).
    That is the shape of a table body line: label + values.

    Value-shapedness is delegated to ``native_verifier.is_numeric_token``, which
    strips presentation before matching (GH-103): significance markers
    (``0.67***``), markdown emphasis (``**23,126**``), unicode minus
    (``−0.253``) and currency prefixes (``£43.2``). The bare anchored
    ``_NUM_TOKEN_RE`` rejects all of those, which would let a starred coefficient
    row — the common case in an econometrics table — become the header again.

    Deliberately narrower than the negation of ``_is_header_row``. A header may
    well name its label column (``['Firm', 'Nominal', 'Real']``) — that row is
    not data because its cells are words, not values. The residual ambiguity is
    a single-line header whose cells are themselves numeric, such as
    ``['Firm', '2024', '2025']`` or ``['Variable', '(1)', '(2)']``; those are
    indistinguishable from data by shape alone and are treated as data, trading
    a lost column name for a guaranteed-present observation. Multi-line headers,
    where the year or column-number band sits on its own row with an empty
    column 0, are unaffected — ``_is_header_row`` accepts those and
    ``_collapse_header_prefix`` merges them before this check runs.
    """
    # Imported lazily: native_verifier imports _NUM_TOKEN_RE from this module.
    from socr.tables.native_verifier import is_numeric_token

    if _is_header_row(row):
        return False
    return any(is_numeric_token(c) for c in row[1:] if c.strip())


def _grid_to_markdown(grid: list[list[str]], *, assume_header: bool = False) -> str:
    """Render a cleaned grid as a GitHub-markdown table.

    Row 0 becomes the header row *unless* it is demonstrably a data row (see
    ``_is_data_row``), in which case an empty header row is emitted and row 0
    stays in the body. Promoting a data row invents a schema and removes an
    observation from the table without any numeric-multiset check noticing
    (GH-146); an empty header is ugly but lossless.

    ``assume_header`` opts out of that inference for callers that already know
    row 0 is a header. ``header_repair`` is the one such caller: it rebuilds the
    header from native word geometry and gates it on ``_header_is_faithful``, so
    re-guessing there would discard a verified repair and emit an empty header
    for any numeric-shaped header band (``['Firm', '2024', '2025']``). Evidence
    beats inference — but only where the evidence actually exists.
    """
    if not grid:
        return ""
    ncol = len(grid[0])
    if not assume_header and _is_data_row(grid[0]):
        header = [""] * ncol
        body = grid
    else:
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


def _is_header_row(row: list[str]) -> bool:
    """True when the row is safe to merge into the multi-line header prefix.

    Two kinds of rows safely belong to a multi-line header:

    1. **Empty-col-0 rows** — pure column-metadata rows like
       ``['', 'GDP', 'GDP']`` or ``['', '2024', '2025']``.
       No entity label; only header cell content.

    2. **All-empty-data rows with non-empty col 0** — the label-column name
       row like ``['Forecaster', '', '', '']``.  Provides the name for the
       label column but carries no values in any data cell.

    Everything else — col 0 non-empty AND at least one non-empty data cell —
    is a data row that must NEVER be swallowed into the header.  This covers:

    * All-na forecaster rows (``['EarlyFirm', '', '', '']``) — they look
      identical to the label-column-name row structurally, but the scan stops
      *before* reaching them because a real data row (or the end of the
      empty/label-only prefix) intervenes first.
    * Integer-only rows (``['Car Sales', '16', '17']``) and decimal-value
      rows (``['Ashford Capital', '2.1', '1.9']``).
    """
    col0 = row[0].strip() if row else ""
    if not col0:
        return True  # empty col 0 — pure column-metadata row
    # Non-empty col 0: safe only when ALL data cells are empty
    # (label-column-name row like ['Forecaster', '', '', '']).
    return not any(c.strip() for c in row[1:])


def _collapse_header_prefix(grid: list[list[str]]) -> list[list[str]]:
    """Merge a multi-row header prefix into a single combined header row.

    Some tables (e.g. CE-style indicator-year grids) use two or three header
    lines:

        Row 0:  |             | GDP  | GDP  | CPI  | CPI  |  (indicator names — col 0 empty)
        Row 1:  | Forecaster  |      |      |      |      |  (label name; data cells empty)
        Row 2:  |             | 2024 | 2025 | 2024 | 2025 |  (year names — col 0 empty)
        Row 3+: data rows (col 0 non-empty AND at least one non-empty data cell)

    Header-prefix detection uses ``_is_header_row``: a row belongs to the header
    prefix iff its col 0 is empty OR (col 0 is non-empty AND all data cells are
    empty — the label-column-name row).  The scan stops at the FIRST row where
    col 0 is non-empty AND any data cell is non-empty — that is a data row.

    Column cells are merged with a space separator, giving a single header::

        | Forecaster | GDP 2024 | GDP 2025 | CPI 2024 | CPI 2025 |

    This lets the parity checker (after normalising underscores → spaces) find
    ``GDP_2024`` → ``GDP 2024`` as a unique header cell.

    Returns the original grid unchanged when:
      - Fewer than two header-like rows exist at the top (no collapsing needed).
      - All rows are header-like (degenerate — no data body to keep).

    Safety guarantee: only rows satisfying ``_is_header_row`` are merged; a data
    row (including all-na rows like ``['EarlyFirm', '', '']`` that appear *after*
    a non-header row would have stopped the scan) is NEVER swallowed.
    """
    if len(grid) < 2:
        return grid

    # Find the boundary: first non-header row (a row with at least one
    # decimal value in data columns).
    hdr_end = 0
    while hdr_end < len(grid) and _is_header_row(grid[hdr_end]):
        hdr_end += 1

    # No multi-line header (0 or 1 header rows), or every row is header-like
    # (degenerate — no data body to keep).
    if hdr_end < 2 or hdr_end >= len(grid):
        return grid

    # Normalise widths so column-wise zipping works across all header rows.
    ncol = max(len(r) for r in grid)
    header_rows = [r + [""] * (ncol - len(r)) for r in grid[:hdr_end]]

    # Merge column-by-column: concatenate non-empty cells with a single space.
    merged: list[str] = []
    for ci in range(ncol):
        parts = [r[ci] for r in header_rows if r[ci].strip()]
        merged.append(" ".join(parts))

    return [merged] + list(grid[hdr_end:])


# ---------------------------------------------------------------------------
# Chart-clip helpers (TR-2)
# ---------------------------------------------------------------------------
# Minimum area (pt²) for a vector drawing cluster to be considered a chart.
# Reuses the same constant as figures/extractor.py CHART_MIN_CLUSTER_AREA.
# Defined here independently to avoid a cross-module import cycle.
_CHART_MIN_CLUSTER_AREA_PT2: float = 120.0 * 120.0  # 14 400 pt²

# Gap (pt) used when clustering drawing bboxes into chart regions.
# Reuses the same value as figures/extractor.py CLUSTER_GAP = 30.
_CHART_CLUSTER_GAP_PT: float = 30.0


def _drawing_bboxes(page) -> list[tuple[float, float, float, float]]:
    """Return (x0, y0, x1, y1) for every non-degenerate drawing on *page*.

    Returns [] on error or when there are no drawings.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
        if x1 > x0 or y1 > y0:  # skip zero-area point/line markers
            boxes.append((x0, y0, x1, y1))
    return boxes


def _union_find_clusters(
    boxes: list[tuple[float, float, float, float]],
    gap: float,
) -> list[tuple[float, float, float, float]]:
    """Cluster *boxes* by proximity (boxes within *gap* pt of each other merge).

    Returns the merged bbox for each cluster as (x0, y0, x1, y1).
    """
    if not boxes:
        return []
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, n):
            bx0, by0, bx1, by1 = boxes[j]
            h_gap = max(0.0, bx0 - ax1) if ax1 < bx0 else max(0.0, ax0 - bx1)
            v_gap = max(0.0, by0 - ay1) if ay1 < by0 else max(0.0, ay0 - by1)
            if h_gap <= gap and v_gap <= gap:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged: list[tuple[float, float, float, float]] = []
    for indices in clusters.values():
        x0 = min(boxes[i][0] for i in indices)
        y0 = min(boxes[i][1] for i in indices)
        x1 = max(boxes[i][2] for i in indices)
        y1 = max(boxes[i][3] for i in indices)
        merged.append((x0, y0, x1, y1))
    return merged


def _has_filled_rects_or_thick_strokes(page, bbox: tuple[float, float, float, float]) -> bool:
    """Return True if the drawing cluster in *bbox* contains chart-like marks.

    Chart-like = filled rectangles (bars, area fills) or strokes wider than
    hairline. Self-contained approximation of the extractor's data-marks
    signal (kept import-free to avoid cycles from figures/extractor.py
    ``_has_vector_data_marks``) — NOT a mirror: the semantics already differ
    (any ``'f'``/``'fs'`` fill, any non-black colour, width > 1.0, vs. the
    extractor's coloured-fill-or-thick-stroke check with a neutral-colour
    carve-out). As of GH-150 A1, ``has_chart_marks`` additionally accepts
    framed thin-stroke clusters via ``_has_framed_data_cluster``, which this
    function does not implement.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return False
    x0, y0, x1, y1 = bbox
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        # Check the drawing overlaps the cluster bbox
        if rect.x0 > x1 or rect.x1 < x0 or rect.y0 > y1 or rect.y1 < y0:
            continue
        d_type = d.get("type", "")
        # Filled rect (bars, area fills)
        if d_type in ("f", "fs"):
            return True
        # Coloured stroke (non-black = chart line)
        color = d.get("color") or d.get("stroke_color")
        if color and color != (0, 0, 0) and color != (0.0, 0.0, 0.0):
            return True
        # Thick stroke (> 1pt width)
        width = d.get("width", 0.0) or 0.0
        if width > 1.0:
            return True
    return False


def chart_region_bboxes(page) -> list[object]:
    """Return ``fitz.Rect`` bboxes covering chart-drawing clusters on *page*.

    Uses the same union-find clustering as ``figures/extractor.py`` but is
    self-contained to avoid import cycles.  Each qualifying cluster
    (area >= ``_CHART_MIN_CLUSTER_AREA_PT2`` AND has chart-like marks) is
    expanded by the page's own median inter-word-row gap so that nearby text
    labels (axis tick values, year labels below bars) fall inside the returned
    bbox and are excluded from the table rowizer.

    The margin is derived from the page's word geometry — no fixed pt constant.
    Returns ``[]`` when there are no qualifying chart clusters or on error.

    Never raises.
    """
    try:
        import fitz
    except ImportError:  # pragma: no cover
        return []

    boxes = _drawing_bboxes(page)
    if not boxes:
        return []

    clusters = _union_find_clusters(boxes, _CHART_CLUSTER_GAP_PT)

    # Compute the page's median inter-word-row gap for the expansion margin.
    # Falls back to _SPLIT_GAP_MIN_PT when the page has too few word rows.
    try:
        words = page.get_text("words")
    except Exception:
        words = []
    ys_all = sorted({round(w[1]) for w in words}) if words else []
    if len(ys_all) >= 2:
        gaps = [ys_all[i + 1] - ys_all[i] for i in range(len(ys_all) - 1)]
        row_margin = statistics.median(gaps)
    else:
        row_margin = _SPLIT_GAP_MIN_PT

    result: list[object] = []
    for cx0, cy0, cx1, cy1 in clusters:
        area = (cx1 - cx0) * (cy1 - cy0)
        if area < _CHART_MIN_CLUSTER_AREA_PT2:
            continue
        if not _has_filled_rects_or_thick_strokes(page, (cx0, cy0, cx1, cy1)):
            continue
        # Expand the bbox to capture nearby text labels.
        #
        # Vertical expansion: one row-height margin above and below to capture
        # year labels (below the bars) and chart title (above).
        #
        # Horizontal expansion: axis tick labels appear to the LEFT of the
        # chart's vertical axis line (the leftmost drawing element).  Tick
        # labels are typically 1-3 characters wide (~10-25pt at small fonts)
        # and positioned a few pt to the left of the axis.  A single
        # row_margin may not reach them (e.g. tick labels at x=54 with
        # chart x0=73 and row_margin=13 → expanded x0=60 still misses x=54).
        # To reliably capture them, extend x0 to 0 (page left edge): tick
        # labels are the only content to the left of the axis in the chart's
        # y-band, so this is safe.  On the right side, row_margin is enough.
        pw = page.rect.width
        ph = page.rect.height
        expanded = fitz.Rect(
            0.0,  # left: extend to page left edge to capture axis tick labels
            max(0.0, cy0 - row_margin),
            min(pw, cx1 + row_margin),
            min(ph, cy1 + row_margin),
        )
        result.append(expanded)
    return result


def _word_in_any_bbox(w: tuple, bboxes: list) -> bool:
    """Return True if word *w* falls inside any of the given ``fitz.Rect`` bboxes.

    Only the word's top-left corner (x0, y0) is checked — the standard
    convention for PyMuPDF word-in-region tests.
    """
    wx, wy = w[0], w[1]
    for b in bboxes:
        if b.x0 <= wx <= b.x1 and b.y0 <= wy <= b.y1:
            return True
    return False


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


def rowize_from_words_chart_aware(
    page,
    page_num: int = 1,
) -> list[tuple[object, str]]:
    """Return ``(rect, content)`` pairs for table AND chart regions on *page*.

    This is the TR-2 chart-aware variant of ``rowize_from_words``.  It:

    1. Calls ``chart_region_bboxes(page)`` to get expanded chart-drawing bboxes.
    2. Excludes words inside those bboxes from the table rowizer, so chart tick
       labels and year labels do not dilute the historical table's
       ``data_row_frac`` and cause ``_looks_tabular`` to reject it.
    3. Returns a placeholder ``(rect, image_ref)`` entry for each chart cluster
       so ``extract_structured`` can emit it in reading order.

    The returned list contains BOTH table entries (``(rect, markdown_table)``)
    and chart entries (``(rect, "![chart region N](chart_region_pN.png)")``)
    sorted by ``rect.y0``.  The caller is responsible for building the final
    reading-order output.

    If no chart clusters exist on the page (or the page has no drawings), this
    falls back to plain ``rowize_from_words(page)`` and returns no chart entries
    — so the caller behaves identically to the TR-1 path for non-chart pages.

    Never raises.  Returns ``[]`` if no valid table segment is found and no chart
    regions exist.
    """
    chart_bboxes = chart_region_bboxes(page)
    if not chart_bboxes:
        # No chart regions — fall back to the plain rowizer (TR-1 path).
        return rowize_from_words(page)

    # Structural gate on the full word set (chart words included): if the page
    # doesn't form a numeric grid at all, skip expensive clustering below.
    if not has_numeric_columns(page):
        # But still return chart placeholders so the chart region is represented.
        result: list[tuple[object, str]] = []
        for idx, cbbox in enumerate(chart_bboxes):
            label = f"chart region {idx + 1}"
            ref = f"![{label}](chart_region_p{page_num}_{idx + 1}.png)"
            result.append((cbbox, ref))
        return result

    try:
        all_words = page.get_text("words")
    except Exception:
        return rowize_from_words(page)

    # Split words into chart words and non-chart (table/prose) words.
    non_chart_words = [w for w in all_words if not _word_in_any_bbox(w, chart_bboxes)]

    # Rowize the non-chart words.
    table_regions = rowize_from_word_list(non_chart_words)

    # Build placeholder entries for each chart cluster.
    chart_regions: list[tuple[object, str]] = []
    for idx, cbbox in enumerate(chart_bboxes):
        label = f"chart region {idx + 1}"
        ref = f"![{label}](chart_region_p{page_num}_{idx + 1}.png)"
        chart_regions.append((cbbox, ref))

    # Merge and sort by y0 (reading order, top-to-bottom).
    all_regions = table_regions + chart_regions
    all_regions.sort(key=lambda r: r[0].y0)
    return all_regions


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
    # Segments that already became their own table (index -> True). A
    # segment's words are spoken for once it is consumed; absorbing its
    # trailing rows into the NEXT segment's header band would then corrupt
    # the table it already built, not repair the next one's header
    # (CodeRabbit nit on PR #192).
    consumed: set[int] = set()
    for i, seg_ys in enumerate(segments):
        seg_words: list = []
        for y in seg_ys:
            seg_words.extend(rows_by_y[y])

        grid_and_rect = _rowize_segment(seg_words, seg_ys, rows_by_y)
        if grid_and_rect is None:
            continue
        grid, x0, y0, x1, y1 = grid_and_rect

        # GH-144 A2b (#146 residual): a column-header band stranded in the
        # preceding segment (see _prepend_header_band) belongs to this
        # table -- but only when that segment was NOT already consumed by
        # its own table (see `consumed` above).
        if i > 0 and (i - 1) not in consumed:
            grid, x0, y0, x1, y1 = _prepend_header_band(
                grid, x0, y0, x1, y1, seg_words, segments[i - 1], rows_by_y
            )

        cleaned = _clean_grid(grid)
        # TR-2: collapse multi-line headers (e.g. indicator row + year row)
        # into a single combined header row so the parity checker can match
        # "GDP 2024" against the ground-truth column name "GDP_2024".
        cleaned = _collapse_header_prefix(cleaned)
        if not _looks_tabular(cleaned):
            logger.debug(
                "rowizer: segment y=%d..%d rejected by _looks_tabular", seg_ys[0], seg_ys[-1]
            )
            continue
        md = _grid_to_markdown(cleaned)
        if md:
            out.append((fitz.Rect(x0, y0, x1, y1), md))
            consumed.add(i)

    return out


def _numeric_lane_centers(num_words: list) -> list[float] | None:
    """Cluster numeric-token x-positions into lanes; return each lane's centre.

    Byte-faithful extraction of ``_rowize_segment``'s own lane-cluster block
    (same algorithm as ``has_numeric_columns``), so a caller that needs a
    segment's data-lane geometry without running the rest of
    ``_rowize_segment`` — GH-144 A2b's header-prepend step — uses the
    identical placement logic ``_rowize_segment`` uses for itself. Returns
    ``None`` when fewer than ``_MIN_COLS`` lanes are found.
    """
    num_xs = sorted(set(w[0] for w in num_words))
    lanes_x: list[list[float]] = []
    for x in num_xs:
        if lanes_x and x - lanes_x[-1][-1] <= _LANE_X_TOL_PT:
            lanes_x[-1].append(x)
        else:
            lanes_x.append([x])

    if len(lanes_x) < _MIN_COLS:
        return None

    return [sum(xs) / len(xs) for xs in lanes_x]


def _is_numeric_word(word: tuple) -> bool:
    """True if a PyMuPDF word tuple's text is a numeric token (``0.67``,
    ``(0.14)``, ``45%``)."""
    return bool(_NUM_TOKEN_RE.match(word[4]) and _NUMERIC_RE.search(word[4]))


def _extend_scope_for_header(tight, words: list):
    """Extend ``tight`` upward to include a preceding lane-snapping header band.

    GH-144 review finding 3: ``tight`` (``_numeric_row_bbox``) is, by
    construction, the union of rows that carry a numeric cell — a header
    band carries none, so scoping the fallback rowizer's word list to
    ``tight`` alone silently excludes the header from ``scoped_words``
    before ``rowize_from_word_list`` ever runs, making
    ``_prepend_header_band`` (A2b) a no-op exactly when A2 fires: it cannot
    prepend a header word that was never in scope to begin with.

    Mirrors ``_prepend_header_band``'s own absorption rule so the two stay
    consistent: walk upward from ``tight``'s top edge, row by row, absorbing
    a row only while every word in it snaps to one of ``tight``'s own
    numeric-lane x-centres and the row itself carries no numeric token (a
    header names columns, it does not contain their values). Stops at the
    first row that fails either test — an unrelated title/subtitle/running
    -head line above the header, or a second data row from a distinct table
    stacked directly above this one. Returns ``tight`` unchanged when there
    is no lane geometry to snap to or nothing above it qualifies.
    """
    try:
        import fitz
    except ImportError:
        return tight

    in_scope = [w for w in words if tight.contains(fitz.Point(w[0], w[1]))]
    num_words = [w for w in in_scope if _is_numeric_word(w)]
    lane_centers = _numeric_lane_centers(num_words)
    if lane_centers is None:
        return tight

    snap_radius = _LANE_X_TOL_PT * _LANE_SNAP_MULT

    def _snaps(word: tuple) -> bool:
        return min(abs(c - word[0]) for c in lane_centers) <= snap_radius

    rows_by_y: dict[int, list] = defaultdict(list)
    for w in words:
        rows_by_y[round(w[1])].append(w)

    # Compare against the rounded boundary, not the raw float `tight.y0`:
    # `rows_by_y`'s keys are `round(w[1])`, so a boundary row's own rounded
    # key can be numerically LESS than `tight.y0` itself (e.g. round(94.325)
    # == 94 < 94.325) and get treated as "above" its own row, breaking the
    # walk immediately on the row that defines the scope instead of ever
    # reaching the real header above it.
    boundary_y = round(tight.y0)
    above_ys = sorted((y for y in rows_by_y if y < boundary_y), reverse=True)
    new_y0 = tight.y0
    for y in above_ys:
        row_ws = rows_by_y[y]
        if (
            not row_ws
            or any(_is_numeric_word(w) for w in row_ws)
            or not all(_snaps(w) for w in row_ws)
        ):
            break
        new_y0 = min(new_y0, min(w[1] for w in row_ws))

    if new_y0 == tight.y0:
        return tight
    return fitz.Rect(tight.x0, new_y0, tight.x1, tight.y1)


def _prepend_header_band(
    grid: list[list[str]],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    seg_words: list,
    prev_seg_ys: list[int],
    rows_by_y: dict[int, list],
) -> tuple[list[list[str]], float, float, float, float]:
    """Prepend a header band stranded in the PRECEDING y-segment (GH-144 A2b).

    ``rowize_from_word_list``'s y-gap segmenter can isolate a column-header
    band (e.g. "Nominal  Real  Inflation") into its own segment when the gap
    above the data rows exceeds the split threshold; that header-only segment
    then has zero numeric tokens, so ``_rowize_segment`` discards it before it
    is ever a region candidate (A1 log §5-6 — the #146 residual).

    Walks ``prev_seg_ys`` bottom-up (the segment immediately above this one)
    and absorbs the maximal TRAILING run of y-rows whose every word snaps to
    one of THIS segment's data-lane x-centres, within the same
    ``_LANE_X_TOL_PT * _LANE_SNAP_MULT`` tolerance ``_rowize_segment`` uses
    for its own cell placement. Stops at the first row with any non-snapping
    word (an unrelated title/subtitle/running-head line above the header),
    so it never absorbs a prefix past a gap. Returns the grid and rect
    unchanged when there is no previous segment, this segment has no numeric
    lanes, or nothing in the previous segment snaps.
    """
    num_words = [w for w in seg_words if _NUM_TOKEN_RE.match(w[4]) and _NUMERIC_RE.search(w[4])]
    lane_centers = _numeric_lane_centers(num_words)
    if lane_centers is None or not prev_seg_ys:
        return grid, x0, y0, x1, y1

    snap_radius = _LANE_X_TOL_PT * _LANE_SNAP_MULT

    def _snaps(word: tuple) -> bool:
        return min(abs(c - word[0]) for c in lane_centers) <= snap_radius

    eligible_ys: list[int] = []
    for y in reversed(prev_seg_ys):
        row_ws = rows_by_y.get(y, [])
        # A row that itself carries a numeric token is a data row, not a
        # header — never absorb it, even if its words happen to snap to
        # this segment's lanes (e.g. a second table's trailing data row
        # stacked directly above this one).
        if (
            not row_ws
            or any(_is_numeric_word(w) for w in row_ws)
            or not all(_snaps(w) for w in row_ws)
        ):
            break
        eligible_ys.append(y)

    if not eligible_ys:
        return grid, x0, y0, x1, y1

    eligible_ys.reverse()  # restore top-to-bottom reading order

    header_rows: list[list[str]] = []
    band_words: list = []
    for y in eligible_ys:
        row_ws = sorted(rows_by_y[y], key=lambda w: w[0])
        band_words.extend(row_ws)
        row_cells = [""] * len(lane_centers)
        for w in row_ws:
            best = min(range(len(lane_centers)), key=lambda i: abs(lane_centers[i] - w[0]))
            existing = row_cells[best]
            row_cells[best] = (existing + " " + w[4]).strip() if existing else w[4]
        header_rows.append([""] + row_cells)  # empty label cell -> _is_header_row(True)

    new_x0 = min([x0] + [w[0] for w in band_words])
    new_y0 = min([y0] + [w[1] for w in band_words])
    new_x1 = max([x1] + [w[2] for w in band_words])
    new_y1 = max([y1] + [w[3] for w in band_words])

    return header_rows + grid, new_x0, new_y0, new_x1, new_y1


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

    lane_centers = _numeric_lane_centers(num_words)
    if lane_centers is None:
        return None
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

    # No padding or column-drop here. Every row is exactly `[label] +
    # row_cells` with `row_cells` always `len(lane_centers)` wide (it is
    # initialised as such above and never appended to), so `grid` is already
    # uniform width by construction — the drop step this replaced was dead
    # code. Leaving the all-empty-column drop to `_clean_grid` (called
    # AFTER `_prepend_header_band` prepends the header band, in
    # `rowize_from_word_list`) keeps the header rows' column positions
    # aligned with these data rows': both are built against the same
    # `lane_centers`, so dropping columns here — before the header band is
    # even known — would reindex the data grid's columns first and misalign
    # them against the header rows `_prepend_header_band` builds afterwards
    # (GH-144 review finding 3). It also means a lane with no data in this
    # segment but a name in the header band is no longer deleted before the
    # header gets a chance to populate it.
    return grid, bbox_x0, bbox_y0, bbox_x1, bbox_y1
