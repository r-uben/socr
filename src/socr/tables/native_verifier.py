"""Deterministic two-tier native table verifier for born-digital pages.

Runs BEFORE the VLM judge on born-digital table pages (has_tables=True).
No model call; uses only PyMuPDF page.get_text("words") geometry.

Two tiers (Consilium decision 20260615T170212Z-0362, Option C; updated TR-4):

1. Hard-fail (value_guard): the output table's numeric-token content does not
   match the native page's numeric tokens on a per-row basis.  Specifically,
   BOTH of the following must hold for a PASS; failure of either is a hard-fail:

   a. NUMBER-COMPLETENESS (row-aware multiset): the count of output rows that
      contain >= 1 numeric token must equal the count of native numeric rows;
      AND for each paired row (overlap-based match) the N2-normalized
      numeric-token multiset must be identical.  This ACCEPTS year-paired
      tables ("2.3 1.9" in one cell ↔ two native tokens in one row) and
      REJECTS dropped values, invented values, dropped forecaster rows
      (row_count_mismatch), and merged rows that alter the row count.

   b. LABEL-BINDING (adjacency-dominance interleaved detection): fires ONLY
      when the SYSTEMATIC name/value-offset pattern dominates the table —
      i.e., the count of ADJACENT (unlabeled-numeric, label-only) or
      (label-only, unlabeled-numeric) consecutive-row pairs is >= the count
      of clean labeled-numeric rows.  A few isolated section headers, sub-
      table year headers, or two-line labels (vastly outnumbered by clean
      labeled rows) do NOT trigger this predicate.

   Token normalization N2 (deterministic string equality, no float parsing):
   strip thousands-separator commas (1,204 → 1204), strip parenthesis wrappers
   (econ-table negative-sign convention: (0.014) → 0.014), strip trailing %
   sign.  Stated decimal precision is preserved (2.10 ≠ 2.1 after
   normalization).  Both native and output tokens are normalized identically
   before comparison.

   → VerifierResult.hard_fail = True, skip the VLM call, escalate.

2. Warn-and-defer: overall native lane count != output column count, but the
   value guard passed.  Encompasses paired-year columns ("2023 2024"), spanning
   headers, stub label columns with no numbers, and sparse rows where visual
   and native lanes legitimately differ.
   → VerifierResult.warn = True, record AuditMetric + AuditEvent, defer to
   the VLM judge.

Scanned pages bypass cleanly: get_text("words") is empty on a true scan, so
_get_native_lane_count() returns 0 and the verifier exits without action.

Named tolerances (no magic literals):
- ``_LANE_X_TOL_PT``: reused from reconstruct.py (6.0 pt) — tokens within
  this x-distance share a lane.  Basis: sub-column glyph spread on
  proportional-width fonts; validated on Fama-French 1997 corpus.
- ``_WELL_SEPARATED_GAP_PT``: minimum gap between two DISTINCT lane centres
  (retained for Tier 2 warn context; no longer used in Tier 1 hard-fail).
  Set to 2 * _LANE_X_TOL_PT + 6 = 18pt.
- ``_MIN_HARD_FAIL_LANES``: minimum number of distinct well-separated native
  lanes in a row required before we can call a collapse geometry-impossible.
  Set to 2: a single-column row cannot collapse by definition.

GH-151 A2: column-binding check (additive, model-free, report-only)
---------------------------------------------------------------------
This section adds a THIRD, independent report — ``BindingReport``, produced
by ``verify_column_binding`` / ``verify_column_binding_region`` below — that
checks whether each output value token is bound to the correct NATIVE
COLUMN LANE.  This is a different question from the value-guard above,
which reasons about per-row numeric-token multisets and never looks at
column identity.

- No model call: pure PyMuPDF geometry (``page.get_text("words")``) plus
  markdown parsing, exactly like the value-guard above.
- Report-only: ``BindingReport`` carries no ``hard_fail`` / ``warn`` /
  ``state`` field and is never merged into ``VerifierResult``.  Nothing in
  this module, and no pipeline phase, reads it yet.
- NO call site in this ticket. Consumption is TICKET-B1 (global wave 3,
  extra depends-on GH-147 A2 — see docs/plans/extraction-defects/STATUS.md).
- The companion evidence fields (``lane_shared_across_columns``,
  ``column_order_violation``) are detail-only and carry NO severity of
  their own; ``BindingReport.failed`` is derived purely from ``misbound``.

Plan reference: c87af1a:docs/plans/gh151-structural-gate/TICKETS.md
(TICKET-A2).  ``docs/plans/gh151-structural-gate/`` is not present on the
main working tree — cite the commit path above, never a live path.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
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

# Fallback row-height estimate (pt) for the y-band margin when fewer than two
# native rows are matched (no inter-row gap to measure). ~1.5x a 10pt type line:
# provides a small margin that excludes inter-row non-table content without
# reaching an adjacent table row (which sits >= a full row pitch away).
_YBAND_SINGLE_ROW_HEIGHT_PT: float = 15.0

# Cap (pt) on the y-band row-height estimate. Even on a double-spaced layout the
# margin stays bounded so the y-band can never engulf an adjacent table's rows
# on the same page (the residual "bad localization" failure mode).
_YBAND_ROW_HEIGHT_CAP_PT: float = 30.0

# Minimum distinct well-separated lanes in one row to allow the hard-fail
# predicate.  A row with < 2 numeric values cannot demonstrate collapse.
_MIN_HARD_FAIL_LANES: int = 2

# Markdown table cell regex — splits on | inside a row
_MD_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


# --------------------------------------------------------------------------
# Result dataclass
# --------------------------------------------------------------------------


class VerifierState:
    """Tri-state verdict constants (TR-6).

    EXACT_PASS   — scoped: row counts match, no label-binding, no multiset mismatch.
                   The table is verified deterministically; no model needed.
    CERTAIN_FAIL — systematic label-binding OR multiset mismatch under clean row
                   alignment. Indicates SILENT CORRUPTION of content → hard-reject.
    AMBIGUOUS    — row-count / lane discrepancy after pairing-derived y-band scoping,
                   OR no trusted native layer, OR scanned page. Table ships FLAGGED;
                   VLM judge (TR-7) may refine.
    """

    EXACT_PASS: str = "EXACT_PASS"
    CERTAIN_FAIL: str = "CERTAIN_FAIL"
    AMBIGUOUS: str = "AMBIGUOUS"


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
    row_count_warn: bool = False  # row-count mismatch soft-flag (table ships, status→WARNING)
    row_count_warn_reason: str = ""  # detail for the row_count_warn event
    # TR-6 tri-state verdict (derived from the fields above; set by _verify_from_words)
    state: str = VerifierState.AMBIGUOUS  # default before verification: AMBIGUOUS


@dataclass
class BindingReport:
    """GH-151 A2: column-binding check for one table region.

    Additive, model-free, report-only companion to ``VerifierResult`` — see
    the module docstring's "GH-151 A2" section.  Checks whether each output
    value token resolves to the NATIVE COLUMN LANE its output column
    implies, independent of the value-guard's row-level multiset/label
    checks above.

    NO call site in this ticket; consumption is TICKET-B1 (global wave 3,
    extra depends-on GH-147 A2).  Plan reference:
    c87af1a:docs/plans/gh151-structural-gate/TICKETS.md (TICKET-A2).

    Tri-state contract (for TICKET-B1 to assert on — a consumer keying only
    on ``failed`` cannot tell "nothing checkable" from "clean" without the
    counters below):

    - **bypass**            — ``checked == 0 and unresolved == 0``
      (scan page, no parseable markdown table, or no pairable rows).
    - **nothing-checkable** — ``checked == 0 and unresolved > 0``
      (value tokens were seen but none resolved into a checked
      (row, value-column) signature).
    - **clean**             — ``checked > 0 and not misbound``.
    - **misbound**          — ``failed`` is True (``bool(misbound)``); this
      is the ONLY thing that sets ``failed``.

    ``lane_shared_across_columns`` / ``column_order_violation`` are
    companion evidence for TICKET-B1's diagnosis, not verdicts — they NEVER
    set ``failed``.
    """

    misbound: list[dict] = field(default_factory=list)
    lane_shared_across_columns: bool = False
    lane_shared_detail: list[dict] = field(default_factory=list)
    column_order_violation: bool = False
    column_order_detail: list[dict] = field(default_factory=list)
    checked: int = 0  # (row, value-column) signatures that participated in comparison
    certain: int = 0  # value tokens uniquely resolved to a lane
    unresolved: int = 0  # value tokens that could not uniquely resolve to a lane

    @property
    def failed(self) -> bool:
        """True iff ``misbound`` is non-empty. Companion flags never set this."""
        return bool(self.misbound)


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
    return _lane_count_from_words(words)


def _lane_count_from_words(words: list) -> tuple[int, dict[float, int]]:
    """Cluster numeric x-positions from *words* into lanes.

    Factored out of ``_get_native_lane_count`` so the same logic can be used
    on a bbox-clipped word list (per-region scoping for TR-2).

    Returns ``(lane_count, lane_of_x)`` where ``lane_of_x`` maps each
    observed numeric token x0 → lane index.  Returns ``(0, {})`` on empty or
    all-non-numeric input.
    """
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
    return _rows_by_y_from_words(words)


def _rows_by_y_from_words(words: list) -> dict[float, list[tuple[float, str]]]:
    """Group numeric tokens from *words* by their rounded y-coordinate.

    Factored out of ``_rows_by_y`` so the same logic can be used on a
    bbox-clipped word list (per-region scoping for TR-2).

    Returns ``{y: [(x, word), ...]}``, sorted by x within each row.
    """
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


def _output_header_numeric_tokens(text: str) -> frozenset[str]:
    """Return N2-normalized numeric tokens from the markdown table HEADER row.

    Only the header row (the line immediately before the separator) is
    included.  This set is used by the row-count check to identify LEADING
    native numeric rows that are "header-like" — their tokens were absorbed
    into the output header rather than a data row, so they should not be
    counted as expected output data rows.

    For example, a spec-number header ``| | (1) | (2) | (3) |`` in the output
    corresponds to native tokens ``(1) (2) (3)`` at a distinct y-position above
    the data rows.  A native row whose N2-normalized tokens are ALL present in
    the output header token set AND which appears above the first data-like row
    is treated as a leading header row and excluded from the native data-row
    count.  (The positional constraint is applied by ``_value_guard``; this
    function only returns the token set.)

    Returns an empty frozenset if no markdown table header is found.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if not _MD_SEP_RE.match(line):
            continue
        if i == 0:
            continue
        header_line = lines[i - 1]
        if "|" not in header_line:
            continue
        # Collect numeric tokens from the header line
        header_tokens: set[str] = set()
        for cell in header_line.strip().strip("|").split("|"):
            for tok in re.split(r"\s+", cell.strip()):
                if tok and _NUM_TOKEN_RE.match(tok) and _NUMERIC_RE.search(tok):
                    header_tokens.add(_normalize_numeric_token(tok))
        return frozenset(header_tokens)
    return frozenset()


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


def _numeric_tokens_from_text(text: str) -> list[str]:
    """Return table-value-like numeric tokens from a Markdown row."""
    candidates = re.split(r"[\s|]+", text.strip())
    return [
        token
        for token in (c.strip() for c in candidates)
        if token and _NUM_TOKEN_RE.match(token) and _NUMERIC_RE.search(token)
    ]


def _numeric_tokens_from_native_row(row_tokens: list[tuple[float, str]]) -> list[str]:
    """Return numeric token text from one native row."""
    return [word for _x, word in row_tokens]


def _token_overlap_count(output_tokens: list[str], native_tokens: list[str]) -> int:
    """Count multiset overlap so repeated values still carry information."""
    return sum((Counter(output_tokens) & Counter(native_tokens)).values())


def _parse_output_row_cells(line: str) -> list[str]:
    """Split a markdown table row into cell strings (stripped, no outer pipes).

    Returns a list of cell content strings in column order.  The first cell
    (index 0) is the row label column.
    """
    return [c.strip() for c in line.strip().strip("|").split("|")]


# Currency symbols that prefix a value in fiscal tables.
_CURRENCY_PREFIXES = frozenset("£$€¥")


def strip_presentation(tok: str) -> str:
    """Remove presentation-only decoration from a would-be numeric token.

    GH-103. Three things are styling or encoding, not value, and each caused a
    token to be dropped entirely rather than compared:

    1. **Markdown emphasis.** ``**23,126**`` never matched the anchored
       ``_NUM_TOKEN_RE``, so every bolded numeric cell vanished from the multiset.
       On a table whose section-total rows are bold — which is how VLMs emit them —
       the source-evidence gate then had nothing left to check and passed
       vacuously.
    2. **Unicode minus and dashes.** Native fiscal PDFs carry U+2212 while a VLM
       emits ASCII ``-``. Both sides were dropped, so a value that actually agreed
       looked like it was missing from both. (``reconcile._norm`` already folded
       these; this path did not.)
    3. **Currency prefixes.** ``£43.2`` failed on both sides, and asymmetrically
       when the native layer glues the symbol and the markdown cell drops it
       because the unit lives in the caption.

    Deliberately NOT stripped: parentheses, trailing ``%`` and thousands commas,
    which ``_normalize_numeric_token`` already handles and which carry meaning
    (the econ negative convention).
    """
    tok = tok.strip().replace(" ", " ").strip()
    for mark in ("**", "__", "*", "_"):
        while tok.startswith(mark):
            tok = tok[len(mark) :]
        while tok.endswith(mark):
            tok = tok[: -len(mark)]
    for dash in ("−", "–", "—"):
        tok = tok.replace(dash, "-")
    while tok[:1] in _CURRENCY_PREFIXES:
        tok = tok[1:]
    return tok.strip()


def is_numeric_token(tok: str) -> bool:
    """True when *tok* is a table value once presentation is stripped."""
    cleaned = strip_presentation(tok)
    return bool(_NUM_TOKEN_RE.match(cleaned) and _NUMERIC_RE.search(cleaned))


def _normalize_numeric_token(tok: str) -> str:
    """N2 canonical form for numeric-string comparison (no float parsing).

    Rules (applied in order):
    1. Strip outer whitespace.
    2. Strip parenthesis wrappers (econ negative-sign convention):
       ``(0.014)`` → ``0.014``.
    3. Strip trailing ``%`` sign: ``45%`` → ``45``.
    4. Remove thousands-separator commas between digit groups:
       ``1,204`` → ``1204``, ``1,234,567`` → ``1234567``.
    5. Return the resulting decimal string verbatim — stated precision is
       preserved (``2.10`` ≠ ``2.1``); no float/Decimal cast is performed.

    Both native tokens and output tokens are normalized with this function
    before multiset comparison, so the comparison is always N2↔N2.
    """
    tok = strip_presentation(tok)
    # Strip parenthesis wrappers (econ-table negative-sign: (X) means -X)
    if tok.startswith("(") and tok.endswith(")") and len(tok) > 2:
        tok = tok[1:-1]
    # Strip trailing percent
    if tok.endswith("%"):
        tok = tok[:-1]
    # Remove thousands-separator commas (between digit groups only)
    # Repeat three times to handle up to 1,234,567,890
    for _ in range(3):
        tok = re.sub(r"(\d),(\d)", r"\1\2", tok)
    return tok


def _numeric_multiset_from_tokens(
    raw_tokens: list[str],
) -> Counter:
    """Return a Counter of normalized numeric tokens from a list of raw strings.

    Only tokens that match ``_NUM_TOKEN_RE`` AND ``_NUMERIC_RE`` are included;
    non-numeric strings (labels, punctuation) are discarded.  Each included
    token is normalized with ``_normalize_numeric_token`` (N2) before counting.
    """
    result: Counter = Counter()
    for tok in raw_tokens:
        if is_numeric_token(tok):
            result[_normalize_numeric_token(tok)] += 1
    return result


def _parse_all_data_rows(
    text: str,
) -> tuple[list[tuple[int, str, Counter]], list[tuple[int, str]]]:
    """Parse output markdown table into data rows classified by content.

    Returns a pair:
      - ``numeric_rows``: list of ``(row_idx, label, numeric_multiset)`` for
        every row after the separator that contains >= 1 numeric token.
        ``label`` is the content of column 0 (stripped).
        ``numeric_multiset`` is the N2-normalized Counter of all numeric tokens
        found anywhere in the row (across all cells).
      - ``label_only_rows``: list of ``(row_idx, label)`` for every row after
        the separator that has a NON-EMPTY label (col 0) and ZERO numeric
        tokens.  These are "label-only rows" — section headers, forecaster-
        name rows in the interleaved failure mode, panel headings, etc.

    This classification is used by the label-binding predicate: the interleaved
    failure pattern produces BOTH numeric rows with empty labels AND label-only
    rows.  A regression table with SE rows (empty label + a value) produces
    numeric rows with empty labels but NO label-only rows — the two are
    distinguished without over-firing on valid SE rows.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    numeric_rows: list[tuple[int, str, Counter]] = []
    label_only_rows: list[tuple[int, str]] = []
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
        cells = _parse_output_row_cells(line)
        # Collect all tokens across all cells
        all_tokens: list[str] = []
        for cell in cells:
            all_tokens.extend(re.split(r"\s+", cell))
        multiset = _numeric_multiset_from_tokens(all_tokens)
        label = cells[0] if cells else ""
        if multiset:
            numeric_rows.append((data_idx, label, multiset))
        elif label:
            # Non-empty label, zero numeric tokens → label-only row
            label_only_rows.append((data_idx, label))
        data_idx += 1
    return numeric_rows, label_only_rows


def _count_adjacent_interleaved_pairs(
    numeric_rows: list[tuple[int, str, Counter]],
    label_only_rows: list[tuple[int, str]],
) -> tuple[int, int]:
    """Count adjacent (unlabeled-numeric, label-only) pairs and clean labeled rows.

    Scans all data rows in row-index order.  An ADJACENT INTERLEAVED PAIR is
    defined as a consecutive pair of rows where one is an unlabeled-numeric row
    (has >= 1 numeric token, empty col-0 label) and the other is a label-only
    row (non-empty label, zero numeric tokens) — in either order.

    A CLEAN LABELED ROW is a numeric row with a non-empty label.

    Returns ``(adjacent_pairs, clean_labeled_count)``.  The label-binding
    predicate fires only when
    ``adjacent_pairs > 0 and adjacent_pairs >= clean_labeled_count``.
    This ensures that a few isolated section headers, sub-table year headers,
    or multi-line labels (vastly outnumbered by 40+ clean labeled rows) do NOT
    trigger the predicate — they produce at most 1-2 adjacent pairs against
    a backdrop of many clean rows.
    """
    # Build a unified timeline of all data rows, tagging each by kind:
    #   "unlabeled_numeric", "labeled_numeric", "label_only"
    row_kinds: dict[int, str] = {}
    for row_idx, label, _ms in numeric_rows:
        if label:
            row_kinds[row_idx] = "labeled_numeric"
        else:
            row_kinds[row_idx] = "unlabeled_numeric"
    for row_idx, _label in label_only_rows:
        # numeric_rows takes precedence (shouldn't overlap, but be safe)
        if row_idx not in row_kinds:
            row_kinds[row_idx] = "label_only"

    sorted_rows = sorted(row_kinds.items())  # [(row_idx, kind), ...]

    adjacent_pairs = 0
    clean_labeled = 0

    for i, (_idx, kind) in enumerate(sorted_rows):
        if kind == "labeled_numeric":
            clean_labeled += 1
        if i + 1 < len(sorted_rows):
            _next_idx, next_kind = sorted_rows[i + 1]
            # Adjacent pair: one unlabeled-numeric + one label-only, consecutive
            if (kind == "unlabeled_numeric" and next_kind == "label_only") or (
                kind == "label_only" and next_kind == "unlabeled_numeric"
            ):
                adjacent_pairs += 1

    return adjacent_pairs, clean_labeled


def _value_guard(
    words: list,
    output_text: str,
    scope_label: str = "page",
) -> tuple[bool, str, list[dict], dict | None]:
    """TR-4 value-guard: row-count soft-flag + row-aware multiset + label-binding.

    Returns ``(hard_fail, reason, drifted_rows, row_count_warn_info)``.

    ``hard_fail`` is True only for the two predicates that indicate SILENT
    CORRUPTION of table content (wrong/shifted numbers or systematic label
    mis-binding).  Row-count gaps are a soft completeness warning — the table
    ships with a WARNING flag rather than being routed to the D3 image floor.

    ROW-COUNT CHECK → SOFT WARNING (not hard-fail):
        The count of output rows containing >= 1 numeric token is compared to
        the count of native numeric rows (after excluding LEADING header-like
        rows whose tokens all appear in the output's markdown header).  A
        mismatch means the output either has more or fewer numeric rows than
        the native PDF.  This is a COMPLETENESS concern (a potentially dropped
        or invented row, or chart/prose numbers counted in the native that are
        absent from the output because they were correctly rendered as images).
        The table SHIPS but the page/document status is demoted to WARNING and
        a ``value_guard_row_count_warning`` audit event is recorded.  When
        ``row_count_warn_info`` is not None, the caller sets
        ``VerifierResult.row_count_warn = True``.

        Rationale (codex + gemini panel): a missing/flagged row is better than
        a wrong number.  Row-count gaps are confounded by chart/prose numerals
        on complex pages (e.g. CE p.4 where the bar chart's tick labels and the
        main forecaster grid share the same native word list), so silently
        discarding the table would be more harmful than shipping it flagged.

    LABEL-BINDING → HARD-FAIL (unchanged):
        Fires when the SYSTEMATIC name/value-offset pattern dominates the table
        (adjacent interleaved pairs >= clean labeled rows).  This catches the
        TR-4a failure mode (values in rows with empty labels, names in rows
        with no data) which produces SILENTLY WRONG output — every number would
        be mis-attributed.

    NUMBER-COMPLETENESS (row-aware multiset) → HARD-FAIL (unchanged):
        For each output data row paired to a native numeric row, the
        N2-normalized numeric-token multisets must be identical.  Mismatches
        mean a number was dropped, invented, or shifted within a row — silent
        corruption at the cell level.

    Returns:
        hard_fail:           True if label-binding or multiset fires.
        reason:              Human-readable summary of the hard-fail predicate.
        drifted_rows:        List of per-row detail dicts for the hard-fail.
        row_count_warn_info: Dict with row-count details if a mismatch was
                             detected (table ships, status → WARNING); else None.
    """
    native_rows_map = _rows_by_y_from_words(words)
    native_row_list = sorted(native_rows_map.items())
    output_data_rows = _parse_output_data_rows(output_text)
    numeric_output_rows, label_only_rows = _parse_all_data_rows(output_text)

    drifted: list[dict] = []
    row_count_warn_info: dict | None = None

    # --- Header-like row exclusion ---
    # Exclude LEADING header-like native rows (spec-number headers, year-label
    # rows) whose tokens were absorbed into the output's markdown column header.
    # POSITIONAL CONSTRAINT: only exclude rows above the first data-like native
    # row so footer year-annotations are not incorrectly excluded.
    output_header_tokens = _output_header_numeric_tokens(output_text)

    def _is_header_like(row_tokens: list[tuple[float, str]]) -> bool:
        return bool(
            output_header_tokens
            and all(_normalize_numeric_token(w) in output_header_tokens for _x, w in row_tokens)
        )

    first_data_y: float | None = None
    for y, row_tokens in native_row_list:
        if not _is_header_like(row_tokens):
            first_data_y = y
            break

    effective_native_rows = [
        (y, row_tokens)
        for y, row_tokens in native_row_list
        if not (_is_header_like(row_tokens) and (first_data_y is None or y < first_data_y))
    ]

    # --- TR-6 Pairing-derived y-band scoping ---
    # Run a preliminary overlap-based pairing to discover which y-range of native
    # rows actually aligns with the output table.  Restrict effective_native_rows
    # to this matched y-band + a half-row margin before counting.  This excludes
    # chart tick labels, prose figures, and historical-table rows that share the
    # native word list (whole-page scope) but are absent from the output table
    # being verified.  "Table region" is derived from the pairing itself, not a
    # magic constant or an unreliable bbox cluster.
    if effective_native_rows and output_data_rows:
        prelim_pairs = _pair_output_to_native_rows(effective_native_rows, output_data_rows)
        if prelim_pairs:
            # Collect y-values of matched native rows directly from the pairing.
            native_tok_to_y: dict[int, float] = {id(rt): y for y, rt in effective_native_rows}
            matched_ys_direct: list[float] = []
            for _out_row, matched_toks in prelim_pairs:
                tok_id = id(matched_toks)
                if tok_id in native_tok_to_y:
                    matched_ys_direct.append(native_tok_to_y[tok_id])
            if matched_ys_direct:
                # Estimate row-height as median gap between consecutive matched ys,
                # capped at 30pt.  Use half this as the margin around the y-band.
                sorted_matched = sorted(matched_ys_direct)
                if len(sorted_matched) >= 2:
                    gaps = [b - a for a, b in zip(sorted_matched, sorted_matched[1:]) if b > a]
                    row_height_est = (
                        sorted(gaps)[len(gaps) // 2] if gaps else _YBAND_SINGLE_ROW_HEIGHT_PT
                    )
                else:
                    row_height_est = _YBAND_SINGLE_ROW_HEIGHT_PT
                row_height_est = min(row_height_est, _YBAND_ROW_HEIGHT_CAP_PT)
                margin = row_height_est * 0.5
                band_y0 = sorted_matched[0] - margin
                band_y1 = sorted_matched[-1] + margin
                scoped_native_rows = [
                    (y, rt) for y, rt in effective_native_rows if band_y0 <= y <= band_y1
                ]
                logger.debug(
                    "native_verifier [%s] TR-6 pairing y-band: y=%.0f..%.0f "
                    "(%d → %d effective native rows; margin=%.1fpt)",
                    scope_label,
                    band_y0,
                    band_y1,
                    len(effective_native_rows),
                    len(scoped_native_rows),
                    margin,
                )
                effective_native_rows = scoped_native_rows

    # --- Row-count check: SOFT WARNING → AMBIGUOUS state (table ships; status → WARNING) ---
    native_numeric_count = len(effective_native_rows)
    output_numeric_count = len(numeric_output_rows)

    if native_numeric_count > 0 and output_numeric_count != native_numeric_count:
        row_count_warn_reason = (
            f"value_guard_row_count_warning: output has {output_numeric_count} "
            f"numeric row(s) vs {native_numeric_count} effective native numeric row(s) "
            f"(after pairing-derived y-band scoping) — "
            f"possible dropped/invented row or alignment gap; "
            f"table ships with WARNING status"
        )
        row_count_warn_info = {
            "predicate": "row_count_mismatch",
            "native_numeric_rows": native_numeric_count,
            "output_numeric_rows": output_numeric_count,
            "detail": (
                f"output has {output_numeric_count} numeric row(s) but "
                f"native has {native_numeric_count} effective numeric row(s) "
                f"(scoped)"
            ),
        }
        logger.debug(
            "native_verifier [%s] row-count warning (scoped): %s",
            scope_label,
            row_count_warn_reason,
        )
        # Continue to check label-binding and multiset — a count gap does not
        # suppress hard-fail for genuinely corrupted tables.

    # --- Label-binding check (adjacency-dominance interleaved-pattern detection) ---
    # HARD-FAIL → CERTAIN_FAIL: fires only when the SYSTEMATIC offset pattern dominates.
    adjacent_pairs, clean_labeled = _count_adjacent_interleaved_pairs(
        numeric_output_rows, label_only_rows
    )
    if adjacent_pairs > 0 and adjacent_pairs >= clean_labeled:
        unlabeled_numeric = [
            (row_idx, label, ms) for (row_idx, label, ms) in numeric_output_rows if not label
        ]
        for row_idx, _label, _ms in unlabeled_numeric:
            drifted.append(
                {
                    "row_idx": row_idx,
                    "predicate": "label_binding_failure",
                    "detail": (
                        "output data row has numeric tokens but empty label (col 0); "
                        f"adjacent interleaved pairs ({adjacent_pairs}) dominate "
                        f"clean labeled rows ({clean_labeled}) — "
                        "systematic name/value offset pattern detected"
                    ),
                }
            )
        reason = (
            f"value_guard_label_binding: {adjacent_pairs} adjacent interleaved pair(s) "
            f">= {clean_labeled} clean labeled row(s) — "
            f"name/value binding is broken (systematic interleaved row pattern)"
        )
        logger.debug("native_verifier [%s] value-guard label-binding: %s", scope_label, reason)
        return True, reason, drifted, row_count_warn_info

    # --- Number-completeness: per-paired-row multiset equality → HARD-FAIL ---
    # Use the existing overlap-based pairing to match output rows to native rows.
    # (Uses the y-band-scoped effective_native_rows from above.)
    if not effective_native_rows:
        return False, "", [], row_count_warn_info

    paired_rows = _pair_output_to_native_rows(effective_native_rows, output_data_rows)

    for (row_idx, _cell_count, row_text), native_row_tokens in paired_rows:
        raw_output_tokens: list[str] = re.split(r"[\s|]+", row_text.strip())
        out_ms = _numeric_multiset_from_tokens(raw_output_tokens)
        nat_ms = _numeric_multiset_from_tokens([word for _x, word in native_row_tokens])

        if out_ms != nat_ms:
            drifted.append(
                {
                    "row_idx": row_idx,
                    "predicate": "multiset_mismatch",
                    "output_multiset": dict(out_ms),
                    "native_multiset": dict(nat_ms),
                    "detail": (f"output numeric tokens {dict(out_ms)} != native {dict(nat_ms)}"),
                }
            )

    if drifted:
        # CERTAIN_FAIL only under CLEAN alignment (row counts matched → no
        # row_count_warn). When a row-count discrepancy is present, the per-row
        # pairing is UNRELIABLE: on a multi-element page (e.g. CE — two tables
        # with a chart/prose block between them) the mismatch is almost always
        # MIS-PAIRING caused by non-table native rows (chart axis ticks, prose
        # figures) contaminating the row set, NOT a genuinely wrong number. Per
        # the harness design (docs/log/2026-06-17_verification-harness-design.md),
        # a multiset mismatch UNDER a count discrepancy is AMBIGUOUS: the table
        # SHIPS flagged (the row-count warning already surfaces the uncertainty)
        # rather than being discarded. Hard-fail is reserved for clean-alignment
        # pages, where a multiset mismatch is a certain dropped/shifted value.
        if row_count_warn_info is not None:
            logger.debug(
                "native_verifier [%s] multiset mismatch UNDER row-count discrepancy "
                "→ AMBIGUOUS (ship flagged, not hard-fail): %d row(s)",
                scope_label,
                len(drifted),
            )
            return False, "", drifted, row_count_warn_info
        reason = (
            f"value_guard_multiset_mismatch: {len(drifted)} paired row(s) have "
            f"numeric-token multiset mismatch (dropped/invented/shifted values)"
        )
        logger.debug("native_verifier [%s] value-guard multiset: %s", scope_label, reason)
        return True, reason, drifted, row_count_warn_info

    return False, "", [], row_count_warn_info


def _pair_output_to_native_rows(
    native_row_list: list[tuple[float, list[tuple[float, str]]]],
    output_data_rows: list[tuple[int, int, str]],
) -> list[tuple[tuple[int, int, str], list[tuple[float, str]]]]:
    """Pair output data rows to native numeric rows conservatively.

    If native/output row counts match exactly, order alignment is the fallback
    because there is no evidence of extra native numeric rows. If the counts
    differ, extra native rows may be leading headers or trailing footnotes. In
    that case, only a unique positive numeric-token overlap is strong enough
    for the hard-fail predicate. Ambiguous or unpaired rows are skipped here and
    left to the VLM judge.
    """
    if not native_row_list or not output_data_rows:
        return []

    same_row_count = len(native_row_list) == len(output_data_rows)
    used_native: set[int] = set()
    pairs: list[tuple[tuple[int, int, str], list[tuple[float, str]]]] = []

    for out_idx, output_row in enumerate(output_data_rows):
        _row_idx, _output_cell_count, row_text = output_row
        output_tokens = _numeric_tokens_from_text(row_text)
        matched_native_idx: int | None = None

        if output_tokens:
            scored: list[tuple[int, int]] = []
            for native_idx, (_y, row_tokens) in enumerate(native_row_list):
                if native_idx in used_native:
                    continue
                score = _token_overlap_count(
                    output_tokens, _numeric_tokens_from_native_row(row_tokens)
                )
                if score > 0:
                    scored.append((score, native_idx))

            if scored:
                best_score = max(score for score, _idx in scored)
                best_native_idxs = [
                    native_idx for score, native_idx in scored if score == best_score
                ]
                if len(best_native_idxs) == 1:
                    matched_native_idx = best_native_idxs[0]

        # Ordinal fallback applies ONLY to output rows that carry numeric tokens.
        # A label-only row (panel/section header like "Panel A.", zero numeric
        # tokens) cannot demonstrate a geometry-impossible collapse — pairing it
        # by ordinal position with a native spec-number header row ((1)(2)(3))
        # would compare its single label cell against the header's lanes and
        # false-fail. Such rows are skipped (deferred), never hard-failed.
        if (
            matched_native_idx is None
            and same_row_count
            and output_tokens
            and out_idx not in used_native
        ):
            matched_native_idx = out_idx

        if matched_native_idx is None:
            continue

        used_native.add(matched_native_idx)
        _y, row_tokens = native_row_list[matched_native_idx]
        pairs.append((output_row, row_tokens))

    return pairs


# --------------------------------------------------------------------------
# GH-151 A2: column-binding check — private helpers (additive, report-only)
# --------------------------------------------------------------------------


def _pairing_scoped_native_rows(
    native_row_list: list[tuple[float, list[tuple[float, str]]]],
    output_data_rows: list[tuple[int, int, str]],
    output_text: str,
    scope_label: str = "page",
) -> list[tuple[float, list[tuple[float, str]]]]:
    """Inline mirror of ``_value_guard``'s row-scoping (GH-151 A2).

    Column-binding needs the SAME scoped row set the value-guard checks
    multisets against, so a value token is only ever compared against
    genuinely-paired native rows — not chart tick labels or prose numerals
    that share the page's word list.  This function mirrors, in the same
    order, the two blocks inside ``_value_guard`` (see lines ~670-741 of
    this module): (1) the leading header-like exclusion, then (2) the TR-6
    pairing-derived y-band.  Extracting a single shared helper is deferred
    (barred by this ticket — see module docstring); any future change to
    ``_value_guard``'s scoping must be mirrored here by hand.

    Reuses only the existing named constants (``_YBAND_SINGLE_ROW_HEIGHT_PT``,
    ``_YBAND_ROW_HEIGHT_CAP_PT``) — introduces no new tolerance.
    """
    # --- (1) Leading header-like row exclusion ---
    output_header_tokens = _output_header_numeric_tokens(output_text)

    def _is_header_like(row_tokens: list[tuple[float, str]]) -> bool:
        return bool(
            output_header_tokens
            and all(_normalize_numeric_token(w) in output_header_tokens for _x, w in row_tokens)
        )

    first_data_y: float | None = None
    for y, row_tokens in native_row_list:
        if not _is_header_like(row_tokens):
            first_data_y = y
            break

    effective_native_rows = [
        (y, row_tokens)
        for y, row_tokens in native_row_list
        if not (_is_header_like(row_tokens) and (first_data_y is None or y < first_data_y))
    ]

    # --- (2) TR-6 pairing-derived y-band ---
    if effective_native_rows and output_data_rows:
        prelim_pairs = _pair_output_to_native_rows(effective_native_rows, output_data_rows)
        if prelim_pairs:
            native_tok_to_y: dict[int, float] = {id(rt): y for y, rt in effective_native_rows}
            matched_ys_direct: list[float] = []
            for _out_row, matched_toks in prelim_pairs:
                tok_id = id(matched_toks)
                if tok_id in native_tok_to_y:
                    matched_ys_direct.append(native_tok_to_y[tok_id])
            if matched_ys_direct:
                sorted_matched = sorted(matched_ys_direct)
                if len(sorted_matched) >= 2:
                    gaps = [b - a for a, b in zip(sorted_matched, sorted_matched[1:]) if b > a]
                    row_height_est = (
                        sorted(gaps)[len(gaps) // 2] if gaps else _YBAND_SINGLE_ROW_HEIGHT_PT
                    )
                else:
                    row_height_est = _YBAND_SINGLE_ROW_HEIGHT_PT
                row_height_est = min(row_height_est, _YBAND_ROW_HEIGHT_CAP_PT)
                margin = row_height_est * 0.5
                band_y0 = sorted_matched[0] - margin
                band_y1 = sorted_matched[-1] + margin
                scoped = [(y, rt) for y, rt in effective_native_rows if band_y0 <= y <= band_y1]
                logger.debug(
                    "native_verifier [%s] GH-151 A2 binding y-band: y=%.0f..%.0f "
                    "(%d -> %d effective native rows; margin=%.1fpt)",
                    scope_label,
                    band_y0,
                    band_y1,
                    len(effective_native_rows),
                    len(scoped),
                    margin,
                )
                return scoped

    return effective_native_rows


def _derive_lane_centres(
    scoped_rows: list[tuple[float, list[tuple[float, str]]]],
) -> list[float]:
    """Ordered native column-lane centres across the scoped rows (GH-151 A2).

    ``_rows_by_y_from_words`` already filters to numeric tokens only, so
    every x-position collected here is a genuine value token.  Clusters
    them with the existing ``_cluster_x_positions`` (uses ``_LANE_X_TOL_PT``;
    no new tolerance is introduced).  There is deliberately NO rejection
    branch in lane resolution: every x that contributed to the scoped rows
    also contributed to this clustering, so nearest-centre assignment
    (see ``_nearest_lane_index``) is always deterministic and needs no
    additional tolerance check.
    """
    xs: list[float] = [x for _y, row_tokens in scoped_rows for x, _w in row_tokens]
    return _cluster_x_positions(xs)


def _nearest_lane_index(x: float, centres: list[float]) -> int:
    """Index of the centre nearest *x*; ties broken by the lower index."""
    best_idx = 0
    best_d = float("inf")
    for idx, c in enumerate(centres):
        d = abs(x - c)
        if d < best_d:
            best_d = d
            best_idx = idx
    return best_idx


def _resolve_value_lane(
    n2_value: str,
    native_row_tokens: list[tuple[float, str]],
    centres: list[float],
) -> int | tuple | None:
    """Resolve *n2_value* to a lane index within one paired native row (GH-151 A2).

    Resolves ONLY when the N2-normalized value occurs EXACTLY once among
    this row's native tokens (also N2-normalized).  A value repeated within
    the row has no unique lane, so it is never greedily matched to the
    leftmost occurrence — that would silently fabricate a binding.  Zero or
    multiple occurrences return ``None``; the caller counts these toward
    ``unresolved``, never toward ``misbound``.
    """
    matches = [(x, w) for x, w in native_row_tokens if _normalize_numeric_token(w) == n2_value]
    if len(matches) != 1:
        return None
    x, _w = matches[0]
    return _nearest_lane_index(x, centres)


def _binding_from_words(
    words: list,
    output_text: str,
    scope_label: str = "page",
) -> BindingReport:
    """Shared core of the GH-151 A2 column-binding check.

    See ``BindingReport``'s docstring for the tri-state contract this
    produces.  Additive / report-only: never raises, never touches
    ``VerifierResult``.  NO call site in this ticket; consumer is
    TICKET-B1 (global wave 3).  Plan reference:
    c87af1a:docs/plans/gh151-structural-gate/TICKETS.md (TICKET-A2).
    """
    if not words:
        return BindingReport()

    output_col_count = _parse_output_col_count(output_text)
    output_data_rows = _parse_output_data_rows(output_text)
    if output_col_count < 2 or not output_data_rows:
        return BindingReport()

    native_row_list = sorted(_rows_by_y_from_words(words).items())
    scoped = _pairing_scoped_native_rows(
        native_row_list, output_data_rows, output_text, scope_label
    )
    pairs = _pair_output_to_native_rows(scoped, output_data_rows)
    if not pairs:
        # No row could be paired at all — nothing was checkable.  Counters
        # stay at 0 (bypass); see BindingReport's tri-state docstring.
        return BindingReport()

    centres = _derive_lane_centres(scoped)
    value_col_count = max(output_col_count - 1, 0)

    # column_signatures[output_column_ordinal] = {row_idx: signature_tuple}
    column_signatures: dict[int, dict[int, tuple[int, ...]]] = {}
    certain = 0
    unresolved = 0

    for (row_idx, _cell_count, row_text), native_row_tokens in pairs:
        cells = _parse_output_row_cells(row_text)  # cells[0] is the label stub
        for col_idx, cell in enumerate(cells[1:]):
            lanes: list[int] = []
            for tok in cell.split():
                if not is_numeric_token(tok):
                    continue
                n2 = _normalize_numeric_token(tok)
                lane = _resolve_value_lane(n2, native_row_tokens, centres)
                if lane is None:
                    unresolved += 1
                else:
                    certain += 1
                    lanes.append(lane)
            if not lanes:
                continue  # zero tokens resolved for this cell -> not checked
            column_signatures.setdefault(col_idx, {})[row_idx] = tuple(lanes)

    checked = sum(len(row_sigs) for row_sigs in column_signatures.values())

    misbound: list[dict] = []
    modal_single_lane: dict[int, int] = {}

    for col_idx, row_sigs in sorted(column_signatures.items()):
        counts = Counter(row_sigs.values())
        best_count = max(counts.values())
        # Deterministic tie-break: among tied counts, lexicographically
        # smallest signature tuple.
        modal_sig = min(sig for sig, c in counts.items() if c == best_count)
        if len(modal_sig) == 1:
            modal_single_lane[col_idx] = modal_sig[0]
        for row_idx, sig in sorted(row_sigs.items()):
            if sig != modal_sig:
                misbound.append(
                    {
                        "row_idx": row_idx,
                        "output_column": col_idx,
                        "expected_signature": modal_sig,
                        "observed_signature": sig,
                        "predicate": "modal_disagreement",
                    }
                )

    # Subsidiary ordinal predicate (ruling amendment 1): only meaningful when
    # the value-column count exactly equals the ordered lane count — this
    # structurally excludes packed paired-year layouts (V < L), so it can
    # never reopen the geometry_impossible_collapse false-positive class
    # (see module docstring lines 12-18 and 37-40).
    if value_col_count == len(centres) and value_col_count > 0:
        for col_idx, lane in sorted(modal_single_lane.items()):
            if lane == col_idx:
                continue
            for row_idx, sig in sorted(column_signatures[col_idx].items()):
                if len(sig) != 1:
                    continue  # only single-lane row signatures participate
                misbound.append(
                    {
                        "row_idx": row_idx,
                        "output_column": col_idx,
                        "expected_signature": (col_idx,),
                        "observed_lane": lane,
                        "predicate": "ordinal_mismatch",
                    }
                )

    # Companion evidence — detail-only, NEVER sets `failed`.
    lane_to_cols: dict[int, list[int]] = {}
    for col_idx, lane in modal_single_lane.items():
        lane_to_cols.setdefault(lane, []).append(col_idx)
    lane_shared_detail = [
        {"lane": lane, "columns": sorted(cols)}
        for lane, cols in lane_to_cols.items()
        if len(cols) >= 2
    ]
    lane_shared_across_columns = bool(lane_shared_detail)

    ordered_cols = sorted(modal_single_lane.keys())
    lanes_in_col_order = [modal_single_lane[c] for c in ordered_cols]
    column_order_violation = any(b <= a for a, b in zip(lanes_in_col_order, lanes_in_col_order[1:]))
    column_order_detail = (
        [{"columns": ordered_cols, "lanes": lanes_in_col_order}] if column_order_violation else []
    )

    return BindingReport(
        misbound=misbound,
        lane_shared_across_columns=lane_shared_across_columns,
        lane_shared_detail=lane_shared_detail,
        column_order_violation=column_order_violation,
        column_order_detail=column_order_detail,
        checked=checked,
        certain=certain,
        unresolved=unresolved,
    )


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def _verify_from_words(
    words: list,
    output_text: str,
    scope_label: str = "page",
) -> VerifierResult:
    """Core verifier logic on a pre-fetched / pre-clipped word list.

    Factored out of ``verify_native_table`` so the same two-tier check can be
    run on a bbox-scoped word list (per-region verification, TR-2) without
    repeating the logic.

    Args:
        words:       PyMuPDF ``get_text("words")`` tuples for the scope being
                     verified (whole page or a bbox-clipped sub-set).
        output_text: The OCR output text to verify.
        scope_label: Human-readable label for debug messages (e.g. "page" or
                     "region y=60..140").

    Returns a VerifierResult with hard_fail / warn flags and context.
    """
    result = VerifierResult()

    # Parse output structure
    output_col_count = _parse_output_col_count(output_text)
    if output_col_count < 2:
        logger.debug(
            "native_verifier [%s]: no parseable markdown table (col_count=%d), bypassing",
            scope_label,
            output_col_count,
        )
        return result

    # Cluster numeric tokens into lanes (scoped to the supplied word list).
    lane_count, lane_of = _lane_count_from_words(words)
    result.native_lane_count = lane_count
    result.output_col_count = output_col_count

    if lane_count == 0:
        return result

    # ------------------------------------------------------------------
    # Tier 1: Hard-fail — value-guard (TR-4 row-count + multiset + label-binding)
    #
    # Replaces the old geometry_impossible_collapse predicate, which compared
    # native lane count against output cell count and produced false positives
    # for year-paired tables (all values present per row, just in fewer cells).
    #
    # Three predicates in order:
    #   (a) ROW-COUNT: soft warning (table ships, status→WARNING) when output
    #       numeric-row count ≠ effective native numeric-row count.
    #   (b) LABEL-BINDING: hard-fail when adjacent interleaved pairs dominate
    #       clean labeled rows — the systematic TR-4a name/value offset pattern.
    #   (c) MULTISET: hard-fail when per-paired-row numeric-token multisets
    #       differ (dropped/invented/shifted values within a matched row).
    # ------------------------------------------------------------------
    vg_hard_fail, vg_reason, vg_drifted, vg_row_count_warn = _value_guard(
        words, output_text, scope_label
    )

    # Propagate soft row-count warning regardless of hard-fail outcome.
    if vg_row_count_warn is not None:
        result.row_count_warn = True
        result.row_count_warn_reason = vg_row_count_warn.get("detail", "")

    if vg_hard_fail:
        result.hard_fail = True
        result.drifted_rows = vg_drifted
        result.reason = vg_reason
        result.state = VerifierState.CERTAIN_FAIL
        logger.debug("native_verifier [%s] hard-fail: %s", scope_label, result.reason)
        return result

    # ------------------------------------------------------------------
    # Tier 2: Warn-and-defer — ambiguous lane-count mismatch
    # ------------------------------------------------------------------
    col_gap = abs(output_col_count - lane_count)
    if col_gap >= 2:
        result.warn = True
        result.reason = (
            f"ambiguous_lane_count_mismatch: native_lanes={lane_count}, "
            f"output_cols={output_col_count}, gap={col_gap} "
            f"(paired/spanning headers possible — deferring to VLM)"
        )
        result.state = VerifierState.AMBIGUOUS
        logger.debug("native_verifier [%s] warn: %s", scope_label, result.reason)

    # TR-6 tri-state: if no hard-fail, no Tier-2 warn, and no row-count warn
    # → EXACT_PASS (clean 1:1 alignment, matching multisets, no label-binding).
    # Row-count warn alone → AMBIGUOUS (count gap after scoping, but multiset clean).
    if not result.hard_fail and not result.warn:
        if result.row_count_warn:
            result.state = VerifierState.AMBIGUOUS
        else:
            result.state = VerifierState.EXACT_PASS

    return result


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

    return _verify_from_words(words, output_text, scope_label="page")


def verify_native_table_region(
    page,
    output_text: str,
    region_bbox,
) -> VerifierResult:
    """Per-region variant of ``verify_native_table``.

    Scopes the native geometry check to words WITHIN *region_bbox* rather than
    the whole page.  This is the TR-2 fix for the false
    ``geometry_impossible_collapse`` that fires when a page contains multiple
    tables: whole-page lane counting combines all schemas into a single
    (too-high) lane count and compares it to a single table's column count.

    Per-region scoping means each table region is verified against its own
    numeric lanes, so a 4-column historical table with 3 data-lanes passes
    cleanly even when the main forecaster grid contributes 4 more lanes on
    the same page.

    Args:
        page:         A live fitz.Page object (PyMuPDF).
        output_text:  The OCR output text to verify (should be the markdown for
                      this region only, not the whole-page output).
        region_bbox:  A ``fitz.Rect`` (or anything with ``.x0 .y0 .x1 .y1``)
                      bounding the table region to scope verification to.

    Returns a VerifierResult scoped to the region.
    """
    try:
        all_words = page.get_text("words")
    except Exception:
        logger.debug("native_verifier_region: get_text failed, bypassing")
        return VerifierResult()
    if not all_words:
        return VerifierResult()

    # Clip words to the region bbox (top-left corner of each word must be inside).
    rx0, ry0, rx1, ry1 = region_bbox.x0, region_bbox.y0, region_bbox.x1, region_bbox.y1
    region_words = [w for w in all_words if rx0 <= w[0] <= rx1 and ry0 <= w[1] <= ry1]
    if not region_words:
        return VerifierResult()

    scope_label = f"region y={ry0:.0f}..{ry1:.0f}"
    return _verify_from_words(region_words, output_text, scope_label=scope_label)


def verify_column_binding(page, output_text: str) -> BindingReport:
    """Whole-page GH-151 A2 column-binding check (additive, report-only).

    Mirrors ``verify_native_table``'s scan-bypass idiom.  NO call site in
    this ticket; consumer is TICKET-B1 (global wave 3).  Plan reference:
    c87af1a:docs/plans/gh151-structural-gate/TICKETS.md (TICKET-A2).

    Args:
        page:        A live fitz.Page object (PyMuPDF).
        output_text: The OCR output text to check.

    Returns a BindingReport (see its docstring for the tri-state contract).
    """
    try:
        words = page.get_text("words")
    except Exception:
        logger.debug("native_verifier: get_text failed, bypassing (binding check)")
        return BindingReport()
    if not words:
        logger.debug("native_verifier: no native words (scan page), bypassing (binding check)")
        return BindingReport()
    return _binding_from_words(words, output_text, scope_label="page")


def verify_column_binding_region(
    page,
    output_text: str,
    region_bbox,
) -> BindingReport:
    """Per-region GH-151 A2 column-binding check (additive, report-only).

    Mirrors ``verify_native_table_region``'s bbox-clip idiom.  NO call site
    in this ticket; consumer is TICKET-B1 (global wave 3).  Plan reference:
    c87af1a:docs/plans/gh151-structural-gate/TICKETS.md (TICKET-A2).

    Args:
        page:         A live fitz.Page object (PyMuPDF).
        output_text:  The OCR output text to check (should be the markdown
                      for this region only, not the whole-page output).
        region_bbox:  A ``fitz.Rect`` (or anything with ``.x0 .y0 .x1 .y1``)
                      bounding the table region to scope the check to.

    Returns a BindingReport scoped to the region.
    """
    try:
        all_words = page.get_text("words")
    except Exception:
        logger.debug("native_verifier_region: get_text failed, bypassing (binding check)")
        return BindingReport()
    if not all_words:
        return BindingReport()

    rx0, ry0, rx1, ry1 = region_bbox.x0, region_bbox.y0, region_bbox.x1, region_bbox.y1
    region_words = [w for w in all_words if rx0 <= w[0] <= rx1 and ry0 <= w[1] <= ry1]
    if not region_words:
        return BindingReport()

    scope_label = f"region y={ry0:.0f}..{ry1:.0f}"
    return _binding_from_words(region_words, output_text, scope_label=scope_label)
