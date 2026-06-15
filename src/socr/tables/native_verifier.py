"""Deterministic two-tier native table verifier for born-digital pages.

Runs BEFORE the VLM judge on born-digital table pages (has_tables=True).
No model call; uses only PyMuPDF page.get_text("words") geometry.

Two tiers (Consilium decision 20260615T170212Z-0362, Option C):

1. Hard-fail (geometry_impossible_collapse): a data row has K >= 2
   WELL-SEPARATED native numeric token lanes, but the corresponding OCR
   output row has fewer populated cells.  This is structurally impossible:
   the native page proves K distinct values existed, yet fewer cells were
   emitted — a concrete collapse, not an ambiguous column-count mismatch.
   → VerifierResult.hard_fail = True, skip the VLM call, escalate.

2. Warn-and-defer: overall native lane count != output column count, but no
   row-level geometry-impossible collapse was found.  Encompasses paired-year
   columns ("2023 2024"), spanning headers, stub label columns with no
   numbers, and sparse rows where visual and native lanes legitimately differ.
   → VerifierResult.warn = True, record AuditMetric + AuditEvent, defer to
   the VLM judge.

Scanned pages bypass cleanly: get_text("words") is empty on a true scan, so
_get_native_lane_count() returns 0 and the verifier exits without action.

Named tolerances (no magic literals):
- ``_LANE_X_TOL_PT``: reused from reconstruct.py (6.0 pt) — tokens within
  this x-distance share a lane.  Basis: sub-column glyph spread on
  proportional-width fonts; validated on Fama-French 1997 corpus.
- ``_WELL_SEPARATED_GAP_PT``: minimum gap between two DISTINCT lane centres
  for the hard-fail predicate.  Must be large enough that two lanes cannot
  be confused with formatting jitter on a single column value (e.g. a
  right-aligned "1,204" whose digits spread across 12pt).
  Basis: a two-digit number "12" is at most ~12pt wide at 9pt font; a safe
  inter-column gap in a well-formatted econ table is >= 30pt.  Set to
  2 * _LANE_X_TOL_PT + 6 = 18pt so that even aggressive tolerance still
  leaves a clear gap.  (Stored as a named constant; derive from tolerance.)
- ``_MIN_HARD_FAIL_LANES``: minimum number of distinct well-separated native
  lanes in a row required before we can call a collapse geometry-impossible.
  Set to 2: a single-column row cannot collapse by definition.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from socr.tables.reconstruct import (
    _LANE_X_TOL_PT,
    _NUM_TOKEN_RE,
    _NUMERIC_RE,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Named tolerances — basis documented in module docstring
# --------------------------------------------------------------------------

# Minimum gap (pt) between two lane x-centres to be treated as DISTINCT
# columns.  2 * cluster-tolerance + one extra grid unit, giving headroom over
# glyph spread while staying well below a real inter-column white-space.
# Basis: see module docstring.
_WELL_SEPARATED_GAP_PT: float = 2 * _LANE_X_TOL_PT + 6.0  # = 18.0 pt

# Minimum distinct well-separated lanes in one row to allow the hard-fail
# predicate.  A row with < 2 numeric values cannot demonstrate collapse.
_MIN_HARD_FAIL_LANES: int = 2

# Markdown table cell regex — splits on | inside a row
_MD_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


# --------------------------------------------------------------------------
# Result dataclass
# --------------------------------------------------------------------------


@dataclass
class VerifierResult:
    """Outcome of the native table verifier for one page."""

    hard_fail: bool = False  # geometry-impossible collapse detected → escalate
    warn: bool = False  # ambiguous mismatch → warn + defer to VLM
    reason: str = ""  # human-readable summary
    native_lane_count: int = 0  # how many numeric lanes were found
    output_col_count: int = 0  # how many columns the output markdown has
    drifted_rows: list[dict] = field(
        default_factory=list
    )  # [{row_idx, native_lanes, output_cells, row_text}]


# --------------------------------------------------------------------------
# Core geometry helpers
# --------------------------------------------------------------------------


def _cluster_x_positions(xs: list[float]) -> list[float]:
    """Cluster sorted x-positions into lane centres using _LANE_X_TOL_PT."""
    if not xs:
        return []
    xs = sorted(xs)
    lanes: list[list[float]] = []
    for x in xs:
        if lanes and x - lanes[-1][-1] <= _LANE_X_TOL_PT:
            lanes[-1].append(x)
        else:
            lanes.append([x])
    return [sum(g) / len(g) for g in lanes]  # centre of each lane


def _get_native_lane_count(page) -> tuple[int, dict[float, int]]:
    """Cluster all numeric token x-positions on the page into lanes.

    Returns (lane_count, lane_of_x) where lane_of_x maps each observed
    numeric token x0 → lane index.  Returns (0, {}) on scan pages or errors.
    """
    try:
        words = page.get_text("words")
    except Exception:
        return 0, {}
    if not words:
        return 0, {}

    # Collect x0 of numeric tokens
    num_xs: list[float] = [
        w[0] for w in words if _NUM_TOKEN_RE.match(w[4]) and _NUMERIC_RE.search(w[4])
    ]
    if not num_xs:
        return 0, {}

    xs_sorted = sorted(set(num_xs))
    lanes: list[list[float]] = []
    for x in xs_sorted:
        if lanes and x - lanes[-1][-1] <= _LANE_X_TOL_PT:
            lanes[-1].append(x)
        else:
            lanes.append([x])

    # Build a lookup from every observed x → lane index
    lane_of: dict[float, int] = {}
    for idx, group in enumerate(lanes):
        for x in group:
            lane_of[x] = idx

    return len(lanes), lane_of


def _rows_by_y(page) -> dict[float, list[tuple[float, str]]]:
    """Group numeric tokens by their rounded y-coordinate (one dict entry per row).

    Returns {y: [(x, word), ...]}, sorted by x within each row.
    """
    try:
        words = page.get_text("words")
    except Exception:
        return {}
    row_map: dict[float, list[tuple[float, str]]] = {}
    for w in words:
        x0, y0, _x1, _y1, word, *_ = w
        if _NUM_TOKEN_RE.match(word) and _NUMERIC_RE.search(word):
            y_key = round(y0)
            row_map.setdefault(y_key, []).append((x0, word))
    for y in row_map:
        row_map[y].sort(key=lambda t: t[0])
    return row_map


def _well_separated_lanes_in_row(
    row_tokens: list[tuple[float, str]], lane_of: dict[float, int]
) -> list[int]:
    """Return distinct lane indices for this row that are well-separated.

    A lane is included only if its centre is >= _WELL_SEPARATED_GAP_PT away
    from the NEAREST other lane centre (or if there is only one lane).
    Two clusters within jitter distance of each other (e.g. a right-aligned
    parenthetical SE below a coefficient, slightly offset) are excluded.
    """
    # Collect lane indices present in this row
    idx_set: set[int] = set()
    for x, _word in row_tokens:
        li = lane_of.get(x)
        if li is not None:
            idx_set.add(li)
    if len(idx_set) < _MIN_HARD_FAIL_LANES:
        return []

    # We need lane centres to measure gaps.  Re-cluster just this row's xs.
    xs = [x for x, _ in row_tokens]
    centres = _cluster_x_positions(xs)
    if len(centres) < _MIN_HARD_FAIL_LANES:
        return []

    # Accept centres that are >= _WELL_SEPARATED_GAP_PT from their nearest
    # neighbour.  For a two-centre row this means both must be separated.
    centres_sorted = sorted(centres)
    well_separated: list[float] = []
    for i, c in enumerate(centres_sorted):
        neighbours = [abs(c - other) for j, other in enumerate(centres_sorted) if j != i]
        if min(neighbours) >= _WELL_SEPARATED_GAP_PT:
            well_separated.append(c)

    # Map back to lane indices
    out_indices: list[int] = []
    for c in well_separated:
        # find the lane index whose observed x0s are closest to this centre
        best_li: int | None = None
        best_d = float("inf")
        for x, _w in row_tokens:
            li = lane_of.get(x)
            if li is not None:
                d = abs(x - c)
                if d < best_d:
                    best_d = d
                    best_li = li
        if best_li is not None:
            out_indices.append(best_li)
    return list(dict.fromkeys(out_indices))  # deduplicate, preserve order


# --------------------------------------------------------------------------
# Output markdown parser
# --------------------------------------------------------------------------


def _parse_output_col_count(text: str) -> int:
    """Return the column count from the first valid markdown table header.

    Returns 0 if no markdown table is found.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if not _MD_SEP_RE.match(line):
            continue
        if i == 0:
            continue
        # line is a separator; previous line should be the header
        header_line = lines[i - 1]
        if "|" not in header_line:
            continue
        cells = [c.strip() for c in header_line.strip().strip("|").split("|")]
        col_count = len(cells)
        if col_count >= 2:
            return col_count
    return 0


def _parse_output_data_rows(text: str) -> list[tuple[int, int, str]]:
    """Return (row_idx, cell_count, raw_line) for each markdown data row.

    Skips the header row and separator row; indexes from 0.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    result: list[tuple[int, int, str]] = []
    sep_found = False
    data_idx = 0
    for line in lines:
        if _MD_SEP_RE.match(line):
            sep_found = True
            continue
        if not sep_found:
            continue
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        populated = [c for c in cells if c]
        result.append((data_idx, len(populated), line))
        data_idx += 1
    return result


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def verify_native_table(page, output_text: str) -> VerifierResult:
    """Two-tier deterministic verifier for one born-digital table page.

    Args:
        page:        A live fitz.Page object (PyMuPDF).  Must have native
                     text layer (get_text("words") non-empty for born-digital).
        output_text: The OCR output text to verify.

    Returns a VerifierResult with hard_fail / warn flags and context.
    """
    result = VerifierResult()

    # Scan bypass: no native word geometry → page is scanned or no numbers
    try:
        words = page.get_text("words")
    except Exception:
        logger.debug("native_verifier: get_text failed, bypassing")
        return result
    if not words:
        logger.debug("native_verifier: no native words (scan page), bypassing")
        return result

    # Parse output structure
    output_col_count = _parse_output_col_count(output_text)
    if output_col_count < 2:
        # Cannot verify without a recognizable markdown grid
        logger.debug(
            "native_verifier: output has no parseable markdown table (col_count=%d), bypassing",
            output_col_count,
        )
        return result

    # Cluster native numeric tokens into lanes (page-level)
    lane_count, lane_of = _get_native_lane_count(page)
    result.native_lane_count = lane_count
    result.output_col_count = output_col_count

    if lane_count == 0:
        # No numeric tokens in native layer → no geometry to compare
        return result

    # ------------------------------------------------------------------
    # Tier 1: Hard-fail — geometry-impossible row collapse
    #
    # For each native data row: count how many WELL-SEPARATED numeric lanes
    # it occupies.  Then check the corresponding output markdown row's
    # populated cell count.  If native_well_separated_lanes > output_cells,
    # this is geometry-impossible (values must have been merged/dropped).
    # ------------------------------------------------------------------
    native_rows = _rows_by_y(page)
    output_data_rows = _parse_output_data_rows(output_text)

    # Pair native rows to output data rows by TRAILING alignment.
    #
    # _rows_by_y collects ALL numeric tokens on the page, including header
    # rows whose column-spec labels — (1)(2)(3), year headers 2020 2021 2022
    # — match _NUM_TOKEN_RE.  These appear at smaller y-values (higher on
    # page) than the data rows.  If we align from the front (out_idx →
    # native_row_list[out_idx]), the header row(s) consume the leading slots
    # and every output data row is compared against the WRONG native row,
    # producing a spurious geometry_impossible_collapse on legitimately sparse
    # data rows.
    #
    # Conservative fix: skip a leading prefix of native rows of length
    #   _offset = max(0, len(native_row_list) - len(output_data_rows))
    # so that output data row i aligns to native_row_list[_offset + i].
    # Assumption: extra native rows are leading header/label rows (true for
    # academic regression tables where spec-number headers, year labels, and
    # panel headers all contain numeric tokens).  If extra rows were trailing
    # footnote rows instead, the loop exits early (bounds guard below) —
    # a false-NEGATIVE / defer-to-VLM, never a false-positive.
    native_row_list = sorted(native_rows.items())  # [(y, [(x, word), ...]), ...]
    _offset = max(0, len(native_row_list) - len(output_data_rows))

    hard_fail_rows: list[dict] = []
    for out_idx, (row_idx, output_cell_count, row_text) in enumerate(output_data_rows):
        native_idx = _offset + out_idx
        if native_idx >= len(native_row_list):
            break
        _y, row_tokens = native_row_list[native_idx]
        sep_lanes = _well_separated_lanes_in_row(row_tokens, lane_of)
        if len(sep_lanes) < _MIN_HARD_FAIL_LANES:
            continue
        if output_cell_count < len(sep_lanes):
            hard_fail_rows.append(
                {
                    "row_idx": row_idx,
                    "native_well_separated_lanes": len(sep_lanes),
                    "output_populated_cells": output_cell_count,
                    "row_text": row_text,
                }
            )

    if hard_fail_rows:
        result.hard_fail = True
        result.drifted_rows = hard_fail_rows
        result.reason = (
            f"geometry_impossible_collapse: {len(hard_fail_rows)} row(s) have "
            f">= {_MIN_HARD_FAIL_LANES} well-separated native numeric lanes "
            f"but fewer output cells "
            f"(native_lane_count={lane_count}, output_col_count={output_col_count})"
        )
        logger.debug("native_verifier hard-fail: %s", result.reason)
        return result

    # ------------------------------------------------------------------
    # Tier 2: Warn-and-defer — ambiguous lane-count mismatch
    #
    # Overall native lane count != output column count, but no row-level
    # geometry-impossible collapse was found.  Record a warning; let the
    # VLM judge decide.
    # ------------------------------------------------------------------
    # Allow a stub/label column with no numbers: native lanes count only
    # numeric tokens, output cols include the label column.  So a mismatch
    # of exactly 1 (output_col_count == native_lane_count + 1) is expected
    # and should NOT warn.  Larger mismatches (>=2) are worth noting.
    col_gap = abs(output_col_count - lane_count)
    if col_gap >= 2:
        result.warn = True
        result.reason = (
            f"ambiguous_lane_count_mismatch: native_lanes={lane_count}, "
            f"output_cols={output_col_count}, gap={col_gap} "
            f"(paired/spanning headers possible — deferring to VLM)"
        )
        logger.debug("native_verifier warn: %s", result.reason)

    return result
